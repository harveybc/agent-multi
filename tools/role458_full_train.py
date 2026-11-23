"""Full-history SELL factorial. Each seed owns one device and a resumable sequential queue."""
import copy
import fcntl
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from app.structural_refit import load_bundle
from role448_worker import Sell,bids,heads,severity,attach_engine,file_sha,snapshot,emit
from role405_plateau import POLICY,observe

ROOT=Path('/var/tmp/role448-temporal-4plus1')
OUT=Path('/var/tmp/role458-full-train')
read=lambda p:json.loads(p.read_text())

def save(p,x):
    tmp=p.with_suffix('.writing');torch.save(x,tmp);tmp.replace(p)

def better(a,b):
    return b is None or (a[0] if a[0]!=b[0] else a[1]>b[1]+1e-8)

def lower(st,rate):
    st=copy.deepcopy(st);st['opt']['param_groups'][0]['lr']=rate
    return st

def prepare():
    bundle=Path('/var/tmp/sac399-series/model/61/bundle')
    recipe,arch=read(ROOT/'recipe.json'),read(ROOT/'arch.json')
    assert recipe==read(bundle/'RECIPE.json') and arch==read(bundle/'MODEL_CONFIG.json')['architecture']
    src,_=load_bundle(bundle,'cpu')
    net=Sell(recipe,dict(arch,core_dropout=.3)).eval()
    assert file_sha(ROOT/'7.npz')==read(ROOT/'7.json')['inputs']
    with src.loss.actor_network_params.to_module(src.actor),torch.inference_mode(),np.load(ROOT/'7.npz') as z:
        old=src.actor_net.encoder.eval();net.encoder.load_state_dict(old.state_dict(),strict=True)
        n=0
        for s in range(8):
            for x in torch.from_numpy(z[f'x{s}']).split(256):
                assert torch.equal(old({'features':x}),net.encoder({'features':x}))
                n+=len(x)
        assert n==18899
        save(OUT/'encoder.pt',snapshot(net.encoder))
    emit(OUT/'TRANSFER.json',dict(source=file_sha(bundle/'model.pt'),encoder=file_sha(OUT/'encoder.pt'),
         rows=n,width=208,max_error=0,recipe=file_sha(ROOT/'recipe.json'),arch=file_sha(ROOT/'arch.json')))

def worker(seed):
    assert seed in (7,19,43)
    root=OUT/f'seed-{seed}';root.mkdir(exist_ok=True)
    lock=(root/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available()
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    spec=read(ROOT/f'{seed}.json');transfer=read(OUT/'TRANSFER.json')
    assert transfer['encoder']==file_sha(OUT/'encoder.pt') and transfer['rows']==18899
    for n,k in [(f'{seed}.npz','inputs'),(f'{seed}.sim.gz','sim'),('recipe.json','recipe'),('arch.json','arch')]:
        assert file_sha(ROOT/n)==spec[k]
    assert transfer['recipe']==spec['recipe'] and transfer['arch']==spec['arch']
    recipe,arch=read(ROOT/'recipe.json'),read(ROOT/'arch.json')
    attach_engine(ROOT/'engine')
    with gzip.open(ROOT/f'{seed}.sim.gz','rb') as f:panels=cloudpickle.load(f)
    for panel,ref in panels.values():assert panel.account(panel.raw)==ref
    with np.load(ROOT/f'{seed}.npz') as z:
        xx=[z[f'x{s}'] for s in range(9)];yy=np.concatenate([z[f'y{s}'] for s in range(8)])
        ix=[z[f'i{s}'] for s in range(9)]
        for s in range(9):
            t=z[f't{s}'];block=spec['blocks'][str(s)]
            assert (t[1:]>t[:-1]).all() and block['start']<=t.min()<=t.max()<block['end']
    train=torch.from_numpy(np.concatenate(xx[:8])).cuda();val=torch.from_numpy(xx[8]).cuda()
    assert train.shape==(18899,13,61) and val.shape==(4899,13,61) and yy.shape==(18899,4)
    assert torch.isfinite(train).all() and torch.isfinite(val).all() and np.isfinite(yy).all()
    scale=float(yy.std(dtype=np.float64));assert scale>0
    labels=torch.from_numpy(yy/scale).cuda()
    for init,mode in [('pretrained','restore'),('random','restore'),('pretrained','fork'),('random','fork')]:
        dest=root/f'{init}-{mode}';dest.mkdir(exist_ok=True)
        config=dict(seed=seed,init=init,mode=mode,device=torch.cuda.get_device_name(),
            source=transfer,inputs=spec['inputs'],sim=spec['sim'],script=file_sha(__file__),policy=POLICY,
            train_years=[2020,2021,2022,2023],val_year=2024,rows=len(train),val_rows=len(val),features=61,
            window=13,batch=32,steps_per_pass=591,lr=.0003,wd=.01,core_dropout=.3,
            epochs=2000,fork_epochs=5,label_scale=scale)
        if (dest/'CONFIG.json').exists():assert read(dest/'CONFIG.json')==config
        emit(dest/'CONFIG.json',config)
        if (dest/'DONE.json').exists():continue
        torch.manual_seed(seed);net=Sell(recipe,dict(arch,core_dropout=.3))
        digest=lambda m:hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in m.state_dict().values())).hexdigest()
        headsha=digest(net.head)
        if init=='pretrained':net.encoder.load_state_dict(torch.load(OUT/'encoder.pt',weights_only=True),strict=True)
        assert headsha==digest(net.head)
        dps=[(n,m.p) for n,m in net.named_modules() if isinstance(m,torch.nn.Dropout) and m.p]
        assert len(dps)==4 and all(n.startswith('encoder.core.') and p==.3 for n,p in dps)
        emit(dest/'INIT.json',dict(head=headsha,encoder=digest(net.encoder),active_dropout=dps))
        net.cuda();opt=torch.optim.AdamW(net.parameters(),lr=config['lr'],weight_decay=config['wd'])
        def state():return dict(w=snapshot(net),opt=copy.deepcopy(opt.state_dict()))
        def load(st):
            net.load_state_dict(st['w']);opt.load_state_dict(copy.deepcopy(st['opt']))
        @torch.inference_mode()
        def score():
            net.eval()
            pred=lambda x:torch.cat([heads(net(b))[-1] for b in x.split(512)]).cpu().numpy()
            ref=pred(train);v=pred(val)
            accounts=[]
            for year in range(2020,2025):
                pn=panels[year][0];sc=np.zeros_like(pn.raw)
                if year==2024:sc[ix[8],1,:]=bids(v,ref)
                else:
                    for s in (2*(year-2020),2*(year-2020)+1):
                        a=sum(map(len,xx[:s]));sc[ix[s],1,:]=bids(ref[a:a+len(xx[s])],ref)
                accounts.append(pn.account(sc))
            a=accounts[-1];return (a['trades']>2*a['weeks'],a['ret']),accounts,np.sort(ref,axis=0)
        path=dest/'resume.pt'
        if path.exists():
            st=torch.load(path,weights_only=False,map_location='cpu')
            ep,champ,best,bad,miss,branches,until,roundbest,history,passes=st
        else:
            best,ac,ref=score();champ=state()|dict(reference=ref,epoch=0)
            ep=bad=miss=until=passes=0;branches={'A':champ};roundbest=None
            history=[dict(epoch=0,accounts=ac,key=best,champion=0)]
            save(dest/'champ.pt',champ)
        while ep<2000 and bad<80:
            ep+=1;start=time.monotonic();improved=False;reports={}
            order=np.random.default_rng([458,seed,ep]).permutation(len(train))
            assert len(order)==len(np.unique(order))==len(train)
            for tag,st in list(branches.items()):
                load(st);torch.manual_seed(45800000+seed*2001+ep);net.train()
                for k in range(0,len(order),32):
                    ids=order[k:k+32];opt.zero_grad(set_to_none=True)
                    loss=severity(net(train[ids]),labels[ids]);assert torch.isfinite(loss)
                    loss.backward();opt.step()
                passes+=1;key,ac,ref=score();current=state();branches[tag]=current
                reports[tag]=dict(key=key,accounts=ac,lr=opt.param_groups[0]['lr'])
                if better(key,best):
                    best=key;champ=current|dict(reference=ref,epoch=ep)
                    improved=True
                if until and better(key,None if roundbest is None else roundbest[0]):
                    roundbest=(key,tag,ep,current)
            bad=0 if improved else bad+int(ep>=100)
            event=None;rate=opt.param_groups[0]['lr']
            if until:
                if ep==until or ep==2000 or bad>=80:
                    _,winner,elected,ch=roundbest;branches={'A':ch};roundbest=None;until=0
                    event=dict(selected=winner,from_epoch=elected)
                miss=0 if improved else miss+1
            else:
                miss,new=observe(ep,improved,miss,rate,bad>=80 or ep==2000)
                if new is not None:
                    tail=branches['A'];branches={'A':lower(champ,new)}
                    event=dict(old_lr=rate,new_lr=new,restore_epoch=champ['epoch'])
                    if mode=='fork':branches['B']=lower(tail,new);until=min(ep+5,2000)
            history.append(dict(epoch=ep,branches=reports,champion=champ['epoch'],patience=bad,
                                lr_misses=miss,event=event,passes=passes,seconds=time.monotonic()-start))
            if improved:save(dest/'champ.pt',champ)
            save(path,(ep,champ,best,bad,miss,branches,until,roundbest,history,passes))
            emit(dest/'PROGRESS.json',history[-1])
        load(champ);key,ac,ref=score()
        assert key==best and np.array_equal(ref,champ['reference'])
        save(dest/'champ.pt',champ)
        emit(dest/'HISTORY.json',history)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        graph,ax=plt.subplots()
        for i,title in enumerate(['2020','2021','2022','2023','VAL 2024']):
            ax.plot([row['branches']['A']['accounts'][i]['ret'] for row in history[1:]],label=title)
        ax.set(xlabel='Epoch (A)',ylabel='Net return (%)');ax.legend();graph.tight_layout()
        graph.savefig(dest/'returns.png');plt.close(graph)
        emit(dest/'DONE.json',dict(champ_epoch=champ['epoch'],epochs=ep,passes=passes,key=key,
                                  accounts=ac,stop='patience' if bad>=80 else 'cap',promoted=False))
    emit(root/'DONE.json',dict(seed=seed,arms=4,promoted=False))

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    if sys.argv[1]=='prepare':prepare()
    else:worker(int(sys.argv[1]))
