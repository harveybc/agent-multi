"""End-to-end temporal OPEN_SELL specialist. Only the current semester is optimized."""
import copy
import gzip
import json
import os
import sys
import time
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from torch import nn
from agent_plugins.three_signal_sac import ThreeLogits
from app.frozen_recipe_inputs import file_sha
from app.recipe_runtime import attach_engine
from role429_worker import model,heads,severity,snapshot,save
import role405_plateau as plateau


def emit(path,data):
    tmp=path.with_suffix('.writing')
    tmp.write_text(json.dumps(data,allow_nan=False));tmp.replace(path)


class Sell(nn.Module):
    def __init__(self,recipe,arch):
        super().__init__()
        self.encoder=ThreeLogits(recipe,dict(arch)).encoder
        self.head=model()
        assert self.encoder.features_dim==208 and len(self.encoder.branches)==22

    def forward(self,x):
        return self.head(self.encoder({'features':x}))


def bids(p,reference):
    ordered=np.sort(reference,axis=0)
    eligible=(p>0)&(p>np.quantile(reference,.9,axis=0))
    ranks=np.column_stack([np.searchsorted(ordered[:,h],p[:,h],'right')/len(reference) for h in range(4)])
    return np.where(eligible,ranks,0)


def run(root,seed,policy_output=None,activity_trial=False):
    if activity_trial:
        assert seed==7 and policy_output is not None
    def eligible(v):
        return v['trades']>2*v['weeks'] if activity_trial else v['trades']>0
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    attach_engine(root/'engine')
    manifest=root/f'{seed}.json'
    sig=file_sha(manifest)
    spec=json.loads(manifest.read_text())
    for file,k in [(f'{seed}.npz','inputs'),(f'{seed}.sim.gz','sim'),('recipe.json','recipe'),('arch.json','arch')]:
        assert file_sha(root/file)==spec[k]
    src=root/f'run-{seed}'
    dest=(policy_output if policy_output else root)/f'run-{seed}';dest.mkdir(exist_ok=True)
    recipe,arch=[json.loads((root/k).read_text()) for k in ('recipe.json','arch.json')]
    with np.load(root/f'{seed}.npz') as d:
        x=[torch.from_numpy(d[f'x{s}']).cuda() for s in range(9)]
        y=[torch.from_numpy(d[f'y{s}']).cuda() for s in range(8)]
        indices=[d[f'i{s}'] for s in range(9)]
    with gzip.open(root/f'{seed}.sim.gz','rb') as f:panels=cloudpickle.load(f)
    for pn,ref in panels.values():assert pn.account(pn.raw)==ref
    torch.manual_seed(seed)
    net=Sell(recipe,arch).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.001)
    def state():return dict(weights=snapshot(net),optimizer=plateau.snapshot(opt))
    def restore(st):
        net.load_state_dict(st['weights']);opt.load_state_dict(copy.deepcopy(st['optimizer']))
        assert all(torch.equal(p.cpu(),st['weights'][k]) for k,p in net.state_dict().items())
    initial=state()
    blocks={**{f'branch_{j}':m for j,m in enumerate(net.encoder.branches)},
            'fusion':net.encoder.fusion,'attention':net.encoder.post_fusion,'core':net.encoder.core,'head':net.head}
    scale=float(y[0][torch.isfinite(y[0]).all(1)].std(unbiased=False));assert scale>0
    # A disposable real update must reach every architectural component.
    ix=torch.where(torch.isfinite(y[0]).all(1))[0][:64]
    loss=severity(net(x[0][ix]),y[0][ix]/scale);loss.backward()
    witness={}
    for k,m in blocks.items():
        ps=list(m.parameters())
        assert ps and all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in ps)
        witness[k]=dict(params=sum(p.numel() for p in ps),grad=sum(float(p.grad.norm()) for p in ps))
        assert witness[k]['grad']>0
    bef={k:[p.detach().clone() for p in m.parameters()] for k,m in blocks.items()}
    opt.step()
    for k,m in blocks.items():
        witness[k]['change']=sum(float((p-b).detach().norm()) for p,b in zip(m.parameters(),bef[k]))
        assert witness[k]['change']>0
    restore(initial);opt.zero_grad(set_to_none=True)
    emit(dest/'PREFLIGHT.json',dict(manifest=sig,device=torch.cuda.get_device_name(),blocks=witness,
                                    total=sum(p.numel() for p in net.parameters()),native_parity=5))

    @torch.inference_mode()
    def measure(s,table=None):
        net.eval()
        def pred(b):
            return torch.cat([heads(net(a))[-1] for a in x[b].split(512)]).cpu().numpy()
        reference=pred(s) if table is None else table
        def acct(b):
            pn=panels[2020+b//2][0]
            scores=np.zeros_like(pn.raw)
            scores[indices[b],1,:]=bids(pred(b),reference)
            a=pn.account(scores)
            assert a['buy']==0 and np.isfinite(a['ret'])
            return a
        return acct(8),[acct(b) for b in range(s)],np.quantile(reference,.9,axis=0).tolist(),reference

    load=lambda p:torch.load(p,weights_only=False,map_location='cpu')
    if policy_output and not activity_trial:
        original=src/'s0-common.pt'
        d=load(original)
        assert d['sig']==sig
        restore(d['champ'])
        with np.load(Path('/var/tmp/role449-cross')/f'policies-{seed}'/'s0-common.npz') as f:
            assert str(f['weight_file_sha'])==file_sha(original) and str(f['source_manifest'])==sig
            reference=f['reference'].copy()
        v,_,q,_=measure(0,reference)
        previous=json.loads((src/'s0-common.json').read_text())
        assert v==previous['val'] and q==previous['cut']
        d['champ']['reference']=reference
        save(d,dest/'s0-common.pt')
        emit(dest/'s0-common.json',dict(**previous,source='448_reused_exact'))

    for s in range(5 if activity_trial else 0,8):
        # Keep only labelled rows of this one block in the optimizer view.
        valid=torch.isfinite(y[s]).all(1)
        tx,ty=x[s][valid],y[s][valid]/scale
        assert len(tx)==spec['blocks'][str(s)]['labelled']
        for arm in (('validation',) if activity_trial else ('common',) if s==0 else ('validation','retention')):
            name=f's{s}-{arm}'
            dst=dest/f'{name}.pt';resume=dest/f'{name}.resume'
            if dst.exists():
                assert load(dst)['sig']==sig
                continue
            parent=initial if not s else load(dest/f's{s-1}-{"common" if s==1 else arm}.pt')['champ']
            restore(parent)
            def metric(v,p):
                return float((v['ret']+np.mean([a['ret'] for a in p]))/2) if arm=='retention' and p else v['ret']
            if resume.exists():
                r=load(resume);assert r['sig']==sig
                restore(r['current']);ch=r['champ'];hist=r['hist'];best=r['best'];ep=r['ep'];bad=r['bad'];miss=r['miss'];active=r['active']
            else:
                v,p,q,ref=measure(s,parent['reference'] if policy_output else None)
                best=metric(v,p);active=eligible(v)
                if activity_trial:assert active and best>0
                ep=bad=miss=0;ch=state()
                if policy_output:
                    previous=json.loads((dest/f's{s-1}-{"common" if s==1 else arm}.json').read_text())
                    assert v==previous['val'] and p[:-1]==previous['prior'] and q==previous['cut']
                    ch['reference']=ref
                    v1,p1,q1,r1=measure(s)
                    recal=metric(v1,p1)
                    won=eligible(v1) and (not active or recal>best+1e-8)
                    if activity_trial and s==5:
                        witness=json.loads((policy_output/'control-entry.json').read_text())
                        assert v==witness['parent'] and v1==witness['recal'] and not won and witness['recal_adopted']
                    emit(dest/f'{name}-entry.json',dict(parent=v,parent_prior=p,parent_metric=best,
                         recal=v1,recal_prior=p1,recal_metric=recal,recal_adopted=won))
                    if won:
                        v,p,q,best,active=v1,p1,q1,recal,True
                        ch['reference']=np.sort(r1,axis=0)
                hist=[dict(epoch=0,val=v,prior=p,metric=best,cut=q,lr=opt.param_groups[0]['lr'])]
            for epoch in range(len(hist),2001):
                start=time.monotonic()
                net.train()
                order=np.random.default_rng([448,seed,s,epoch]).permutation(len(tx))
                for k in np.array_split(order,(len(order)+511)//512):
                    opt.zero_grad(set_to_none=True);out=severity(net(tx[k]),ty[k])
                    assert torch.isfinite(out)
                    out.backward();opt.step()
                v,p,q,ref=measure(s);value=metric(v,p)
                improved=eligible(v) and (not active or value>best+1e-8)
                if improved:
                    ch=state();best=value;ep=epoch;bad=0;active=True
                    if policy_output:ch['reference']=np.sort(ref,axis=0)
                elif epoch>=100:bad+=1
                miss,rate=plateau.observe(epoch,improved,miss,opt.param_groups[0]['lr'],bad>=80)
                if rate is not None:plateau.restore(net,opt,ch['weights'],ch['optimizer'],rate)
                if rate is not None and policy_output:
                    restored_v,restored_p,_,_=measure(s,ch['reference'])
                    assert restored_v==hist[ep]['val'] and restored_p==hist[ep]['prior']
                    if activity_trial:assert eligible(restored_v)
                row=dict(epoch=epoch,val=v,prior=p,metric=value,cut=q,best=ep,patience=bad,
                         lr=opt.param_groups[0]['lr'],rollback=rate is not None,seconds=time.monotonic()-start)
                hist.append(row)
                emit(dest/'PROGRESS.json',dict(stage=s,arm=arm,**row))
                if bad>=80 or epoch==2000:break
                if epoch%10==0:
                    save(dict(sig=sig,current=state(),champ=ch,hist=hist,best=best,ep=ep,bad=bad,miss=miss,active=active),resume)
            restore(ch)
            v,p,q,_=measure(s,ch.get('reference'))
            assert metric(v,p)==best
            if activity_trial:assert eligible(v)
            if policy_output:assert v==hist[ep]['val'] and p==hist[ep]['prior']
            save(dict(sig=sig,champ=ch,best=ep,history=hist,active=active),dst)
            emit(dest/f'{name}.json',dict(epochs=len(hist)-1,best=ep,val=v,prior=p,metric=best,cut=q,active=active,
                                         stop='patience' if bad>=80 else 'cap',rows=len(tx)))
            resume.unlink(missing_ok=True)
    emit(dest/'DONE.json',dict(seed=seed,stages=3 if activity_trial else 8,promoted=False))


if __name__=='__main__':run(Path(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3]) if len(sys.argv)>3 else None,
                           sys.argv[4:]==['--activity-trial'])
