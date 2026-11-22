"""Independent role heads on frozen market embeddings. No SAC actor actions."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tools import role404_data as data
from tools.role404_data import DEST, HS, ROOT, ROLES, HOUR, file_sha, save
from app.structural_refit import load_bundle

def checkpoint(path, value):
    tmp=path.with_suffix('.tmp')
    torch.save(value,tmp);tmp.replace(path)

def cache():
    DEST.mkdir(exist_ok=True,parents=True)
    with (DEST/'cache.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (DEST/'CACHE.json').exists():
            rec=json.loads((DEST/'CACHE.json').read_text())
            if rec['code']!=file_sha(__file__) or rec['sha']!=file_sha(DEST/'market.npz'):raise ValueError('cache identity')
            return
        data.setup()
        bundle=ROOT/'model/61/bundle'
        m,_=load_bundle(bundle,'cuda');a=m.actor_net
        assert not a.core_first_batchnorm and isinstance(a.regularizer,nn.Identity)
        x=data.MARKET.inputs; ids=np.flatnonzero(x.valid)
        with m.loss.actor_network_params.to_module(m.actor), torch.inference_mode():
            a.encoder.eval();parts=[]
            for k in range(0,len(ids),256):
                v=torch.as_tensor(x.windows[ids[k:k+256]],device='cuda')
                parts.append(a.encoder({'features':v}).cpu().numpy())
            w=np.concatenate(parts);assert np.isfinite(w).all()
        features=np.zeros((len(x.valid),w.shape[1]),dtype='float32');features[ids]=w
        np.savez(DEST/'market.tmp.npz',x=features,t=x.row_ts_ns,valid=x.valid)
        (DEST/'market.tmp.npz').replace(DEST/'market.npz')
        save(DEST/'CACHE.json',dict(code=file_sha(__file__),sha=file_sha(DEST/'market.npz'),
             weights=file_sha(bundle/'model.pt'),width=w.shape[1],n=len(ids),training=False))

def basis(g):
    # External trilinear lookup; no account values go through a learned layer.
    v=g/(g+g.new_tensor([1.,1.,24.]))
    b=torch.ones_like(v[:,:1])
    for c in v.split(1,dim=1): b=torch.cat((b*(1-c),b*c),1)
    return b

class Expert(nn.Module):
    def __init__(self,width,close,method):
        super().__init__();self.n=8 if close else 1;self.m=3 if method=='adv' else 1
        self.net=nn.Sequential(nn.Linear(width,8),nn.SiLU(),nn.Linear(8,len(HS)*self.n*self.m))
    def forward(self,x,h,g):
        a=self.net(x).reshape(-1,len(HS),self.n,self.m)[torch.arange(len(h),device=h.device),h]
        raw=(a*(basis(g)[:,:,None] if self.n==8 else 1.)).sum(1)
        if self.m==1:return raw
        return torch.stack((raw[:,0],raw[:,0]-F.softplus(raw[:,1]),raw[:,0]+F.softplus(raw[:,2])),1)

def arbitration(buy,sell):
    if not np.isfinite([buy,sell]).all():return 0
    if buy>max(0.,sell):return 1
    if sell>max(0.,buy):return 2
    return 0

def checks():
    for a,b,expected in [(1,2,2),(2,1,1),(1,1,0),(-1,-2,0),(float('nan'),1,0)]:
        assert arbitration(a,b)==expected
    torch.manual_seed(19)
    nets=[Expert(4,r>=2,'adv') for r in range(4)]
    x=torch.ones(3,4);h=torch.tensor([0,1,3]);g=torch.ones(3,3)
    for i,m in enumerate(nets):
        before=[n.net[-1].weight.detach().clone() for n in nets]
        opt=torch.optim.SGD(m.parameters(),lr=.1);y=m(x,h,g)
        assert torch.all(y[:,1]<=y[:,0]) and torch.all(y[:,0]<=y[:,2])
        opt.zero_grad();y.sum().backward();opt.step()
        assert [not torch.equal(w,n.net[-1].weight) for w,n in zip(before,nets)]==[j==i for j in range(4)]
    b=basis(torch.tensor([[0.,0.,0.],[1.,2.,36.]]));torch.testing.assert_close(b.sum(1),torch.ones(2))

def fit(role,years,seed,method,paired_root=None,fold_end=None):
    checks()
    paired=paired_root is not None
    device='cpu' if paired else 'cuda'
    rollback=paired or os.environ.get('ROLE_RUN')=='405'
    if rollback:
        from tools import role405_plateau as plateau
        cr=json.loads((DEST/'CACHE.json').read_text())
        if (cr['code']!='03dba8150d2ef61d5ed75edbca4cd7a3e04be181543bdd3bbfdcb50bd3d0ecaa'
            or cr['weights']!=file_sha(ROOT/'model/61/bundle/model.pt') or cr['sha']!=file_sha(DEST/'market.npz')):
            raise ValueError('unqualified 404 cache')
    else:cache()
    root=Path('/var/tmp/role405') if rollback else DEST
    if paired:
        from tools import role406_cycle as cycle
        root=paired_root/'champion'
        root.mkdir(parents=True,exist_ok=True)
    year_end=fold_end or 2024
    if fold_end is not None and (not paired or fold_end not in (2020,2021,2022,2023)):raise ValueError('fold contract')
    year0=year_end-years;out=root/f'{method}-{role}-{years}-{seed}'
    out.mkdir(exist_ok=True);started=time.monotonic()
    torch.manual_seed(seed);torch.set_float32_matmul_precision('highest')
    bank=np.load(DEST/'market.npz');ts=bank['t'];valid=bank['valid']
    x=torch.from_numpy(bank['x']).to(device);parts=[];hashes={}
    for y in range(year0,year_end):
        for hi,h in enumerate(HS):
            d=DEST/f'labels-{y}-{h}';rec=json.loads((d/'DONE.json').read_text())
            if rec['year']!=y or rec['horizon']!=h or rec['code']!=file_sha(data.__file__) or rec['sha']!=file_sha(d/'rows.npz'):
                raise ValueError('label identity')
            hashes[d.name]=rec['sha']
            with np.load(d/'rows.npz') as z: ids=z[role+'_id'];v=z[role]
            assert np.all(valid[ids[:,0]]) and np.all(ids[:,1]<=ts[ids[:,0]])
            parts.append((ids,v,np.full(len(ids),hi,'int64')))
    ids,v,hi=[np.concatenate(t) for t in zip(*parts)];del parts
    monitor_cut=f'{year_end-1}-07-01'
    cut=int(np.datetime64(monitor_cut,'ns').astype('int64'))
    fitix=np.flatnonzero((ts[ids[:,0]]<cut)&(ids[:,2]<cut))
    valix=np.flatnonzero(ids[:,1]>=cut)
    if fold_end:
        bound=int(np.datetime64(f'{year_end}-01-01','ns').astype('int64'))
        valix=valix[(ids[valix,2]<bound)&(ts[ids[valix,0]]<bound)]
        assert ids[fitix,2].max()<cut and ids[valix,2].max()<bound
    assert set(hi[fitix])==set(hi[valix])==set(range(4))
    scale=max(1.e-6,float(v[fitix,3].std()))
    t=lambda a:torch.as_tensor(a,device=device)
    ix=t(ids[:,0].copy());g=t(v[:,:3].copy());hi=t(hi);y=t(v[:,3].copy()/scale)
    fi=t(fitix);vi=t(valix)
    model=Expert(x.shape[1],role.startswith('CLOSE'),method).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    contract=dict(role=role,years=years,seed=seed,method=method,heads=1,
        features='frozen market-only encoding; external position lookup',code=file_sha(__file__),
        labels=hashes,encoder=json.loads((DEST/'CACHE.json').read_text())['weights'],
        gpu='cpu' if paired else torch.cuda.get_device_name(),fit=len(fi),monitor=len(vi),scale=scale,
        monitor_cut=monitor_cut,epochs=2000,patience=80,min_epoch=100,batch=512)
    if rollback:
        contract.update(plateau=plateau.POLICY,scheduler_code=file_sha(plateau.__file__),
                        cache_sha=cr['sha'],monitor_metric='internal validation loss, minimize')
    if paired:contract.update(cycle_code=file_sha(cycle.__file__),paired_epochs=cycle.CYCLE)
    if fold_end:contract.update(score_year=fold_end,first_fit_year=year0,monitor_label_end_max=int(ids[valix,2].max()))
    if (out/'CONTRACT.json').exists() and json.loads((out/'CONTRACT.json').read_text())!=contract:raise ValueError('resume contract')
    save(out/'CONTRACT.json',contract)
    def loss(p,target):
        if method=='cls':return F.binary_cross_entropy_with_logits(p[:,0],(target>0).float())
        e=target[:,None]-p[:,1:];q=e.new_tensor([.1,.9])
        return F.mse_loss(p[:,0],target)+torch.maximum(q*e,(q-1)*e).mean()
    @torch.no_grad()
    def evaluate(rows,net=model):
        sums=torch.zeros(3,device=device)
        for a in rows.split(8192):
            p=net(x[ix[a]],hi[a],g[a]);v=y[a];dec=p[:,0]>0
            sums+=torch.stack((loss(p,v)*len(a),(v.abs()*(dec!=(v>0))).sum()*scale,dec.sum().float()))
        return (sums/len(rows)).tolist()
    epoch=bad=best_ep=0;best=float('inf');best_weights=None
    lr_bad=reductions=0;best_moments=None
    fork_epoch=None;explorer=expl_opt=None;paired_done=False
    probe=fi[torch.linspace(0,len(fi)-1,min(1024,len(fi)),device=device).long()]
    if (out/'resume.pt').exists():
        rec=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        epoch,bad,best,best_ep,best_weights=rec['state']
        model.load_state_dict(rec['model']);optim.load_state_dict(rec['optim'])
        if rollback:lr_bad,reductions,best_moments=rec['plateau']
        if paired:
            fork_epoch,paired_done,branch=rec['fork']
            if branch is not None:
                explorer,expl_opt=cycle.clone_pair(model,optim)
                explorer.load_state_dict(branch[0]);expl_opt.load_state_dict(branch[1])
            torch.set_rng_state(rec['rng'])
        else:torch.cuda.set_rng_state(rec['rng'])
    while epoch<2000 and (epoch<100 or bad<80):
        epoch+=1;model.train()
        if explorer is not None:
            explorer.train()
            explore_lr=cycle.rate(epoch-fork_epoch)
            expl_opt.param_groups[0]['lr']=explore_lr
            assert abs(optim.param_groups[0]['lr']-1.2e-5)<1e-14
        for a in fi[torch.randperm(len(fi),device=device)].split(512):
            optim.zero_grad(set_to_none=True);p=model(x[ix[a]],hi[a],g[a])
            l=loss(p,y[a]);l.backward();optim.step()
            if explorer is not None:
                expl_opt.zero_grad(set_to_none=True)
                loss(explorer(x[ix[a]],hi[a],g[a]),y[a]).backward();expl_opt.step()
        model.eval()
        m=evaluate(vi);tr=evaluate(fi)
        if not np.isfinite(tr+m).all():raise ValueError('nonfinite')
        improved=m[0]<best-1e-8
        if improved:
            best,best_ep,bad=m[0],epoch,0
            best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if rollback:best_moments=plateau.snapshot(optim)
        elif epoch>=100:bad+=1
        record=dict(epoch=epoch,patience=bad,best_epoch=best_ep,train=tr,monitor=m,seconds=time.monotonic()-started)
        if fold_end and fork_epoch is not None and epoch-fork_epoch==cycle.CYCLE:
            assert abs(optim.param_groups[0]['lr']-1.2e-5)<1e-14
            checkpoint(out/'expert.pt',model.state_dict())
            save(out/f'epoch-{epoch:04}.json',record)
            save(out/'RESULT.json',dict(weights=file_sha(out/'expert.pt'),metrics=m,fork=fork_epoch,
                epoch=epoch,score_year=fold_end,reductions=reductions,stop='fixed_fine_endpoint',promoted=False))
            return
        if explorer is not None:
            explorer.eval();em=evaluate(vi,explorer)
            if not np.isfinite(em).all():raise ValueError('explorer nonfinite')
            record.update(explorer_monitor=em,explorer_lr=explore_lr)
            if epoch-fork_epoch==cycle.CYCLE:
                corrected=introduced=0
                with torch.no_grad():
                    for a in vi.split(8192):
                        ok=(model(x[ix[a]],hi[a],g[a])[:,0]>0)==(y[a]>0)
                        exok=(explorer(x[ix[a]],hi[a],g[a])[:,0]>0)==(y[a]>0)
                        corrected+=int((~ok & exok).sum());introduced+=int((ok & ~exok).sum())
                for arm,net,metrics in [('fine',model,m),('explore',explorer,em)]:
                    d=paired_root/arm/out.name;d.mkdir(parents=True,exist_ok=True)
                    save(d/'CONTRACT.json',contract)
                    checkpoint(d/'expert.pt',net.state_dict())
                    save(d/'RESULT.json',dict(weights=file_sha(d/'expert.pt'),metrics=metrics,
                        fork=fork_epoch,epoch=epoch,relative_epoch=cycle.CYCLE,anchor=file_sha(out/'anchor.pt'),
                        explorer_corrected=corrected,explorer_introduced=introduced,monitor_n=len(vi),
                        promoted=False))
                paired_done=True;explorer=expl_opt=None
        if rollback:
            lr=optim.param_groups[0]['lr']
            lr_bad,rate=plateau.observe(epoch,improved,lr_bad,lr,bad>=80 or epoch>=2000)
            if rate is not None:
                with torch.no_grad():before=model(x[ix[probe]],hi[probe],g[probe])[:,0]
                plateau.restore(model,optim,best_weights,best_moments,rate);reductions+=1
                if paired and reductions==2:
                    if not fold_end:explorer,expl_opt=cycle.clone_pair(model,optim)
                    fork_epoch=epoch
                    checkpoint(out/'anchor.pt',dict(model=model.state_dict(),optim=optim.state_dict(),
                        rng=torch.get_rng_state(),champion=best_ep,epoch=epoch))
                with torch.no_grad():after=model(x[ix[probe]],hi[probe],g[probe])[:,0]
                record['rollback']=dict(champion=best_ep,global_es_bad=bad,
                    pre_loss=m[0],champion_loss=best,probe_n=len(probe),
                    sign_changes=int(((before>0)!=(after>0)).sum().item()),
                    mean_abs_score_change=float((before-after).abs().mean().item()))
            record.update(lr_used=lr,lr_next=optim.param_groups[0]['lr'],lr_bad=lr_bad,reductions=reductions)
        save(out/f'epoch-{epoch:04}.json',record)
        payload=dict(state=(epoch,bad,best,best_ep,best_weights),model=model.state_dict(),
                     optim=optim.state_dict(),rng=torch.get_rng_state() if paired else torch.cuda.get_rng_state())
        if paired:payload['fork']=(fork_epoch,paired_done,None if explorer is None else
                                    (explorer.state_dict(),expl_opt.state_dict()))
        if rollback:payload['plateau']=(lr_bad,reductions,best_moments)
        checkpoint(out/'resume.pt',payload)
    if fold_end:raise ValueError('no complete fine endpoint before budget')
    assert best_weights is not None
    checkpoint(out/'expert.pt',best_weights)
    model.load_state_dict(torch.load(out/'expert.pt',weights_only=True))
    restored=evaluate(vi);assert abs(restored[0]-best)<1e-5
    save(out/'RESULT.json',dict(epochs=epoch,restored=best_ep,metrics=restored,
        stop='patience' if bad>=80 else 'cap',weights=file_sha(out/'expert.pt'),promoted=False,
        reductions=reductions,final_lr=optim.param_groups[0]['lr'],
        **(dict(pair_status='COMPLETE' if paired_done else 'NO_FULL_PAIR',fork=fork_epoch) if paired else {})))

if __name__=='__main__':
    if len(os.sys.argv)==1:checks();cache()
    else:
        r,yr,s,m=os.sys.argv[1:];assert r in ROLES and m in ('adv','cls') and yr in ('4','8')
        fit(r,int(yr),int(s),m)
