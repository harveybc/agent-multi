"""Standalone sequential GPU worker: frozen market latents, Bayesian final layers."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def save_json(p,v):
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,allow_nan=False,indent=2));tmp.replace(p)

def weights(q,bounds):
    span=bounds[1]-bounds[0]
    z=np.where(span>0,(q-bounds[0])/np.where(span>0,span,1),.5).clip(0,1)
    bits=np.array([[i>>2&1,i>>1&1,i&1] for i in range(8)],bool)
    return np.where(bits,z[:,None,:],1-z[:,None,:]).prod(-1)

class Head(nn.Module):
    def __init__(self,d,k):
        super().__init__()
        self.hidden=nn.Sequential(nn.Linear(d,8),nn.SiLU())
        self.mu=nn.Parameter(torch.zeros(2,k,9))
        self.rho=nn.Parameter(torch.full_like(self.mu,-4.))

    def forward(self,x,w):
        h=self.hidden(x)
        phi=w[:,:,None]*torch.cat([h,torch.ones_like(h[:,:1])],1)[:,None,:]
        std=F.softplus(self.rho)
        m=phi.flatten(1)@self.mu.flatten(1).T
        v=phi.flatten(1).square()@std.flatten(1).square().T
        kl=.5*(std.square()+self.mu.square()-1-2*std.log()).sum()
        return m,v.clamp_min(1e-12).sqrt(),kl

def fit_role(path,role,a,source):
    folder=a.out/role;folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(19)
    with np.load(path) as f:x,q,y,bp,fold=[f[k] for k in ('x','q','y','bp','fold')]
    if not all(np.isfinite(z).all() for z in (x,q,y,bp)) or set(fold)!={0,1,2,3}:raise ValueError('input support')
    fit=fold==0
    xmu=x[fit].mean(0);xsd=x[fit].std(0);xsd=np.where(xsd>1e-8,xsd,1.)
    bmu,bsd=float(bp[fit].mean()),float(bp[fit].std())
    if not bsd>0 or not 0<y[fit].mean()<1:raise ValueError('degenerate target')
    bounds=np.quantile(q[fit],[.1,.9],axis=0)
    w=weights(q,bounds) if role.startswith('close') else np.ones((len(x),1))
    dev=torch.device('cuda')
    def tensor(v):return torch.as_tensor(v,dtype=torch.float32,device=dev)
    X,W,Y,B=map(tensor,((x-xmu)/xsd,w,y,(bp-bmu)/bsd))
    ids=[torch.as_tensor(np.flatnonzero(fold==i),device=dev) for i in range(4)]
    head=Head(x.shape[1],w.shape[1]).to(dev)
    opt=torch.optim.Adam(head.parameters(),lr=a.lr)
    nodes,ww=np.polynomial.hermite.hermgauss(16)
    nodes,ww=tensor(nodes*2**.5),tensor(ww/np.sqrt(np.pi))
    prior=float(y[fit].mean());median=float(np.median(bp[fit]))
    @torch.no_grad()
    def metrics(i,temp=1.,dump=False):
        head.eval();idx=ids[i]
        m,s,_=head(X[idx],W[idx]);draw=torch.sigmoid((m[:,:1]+s[:,:1]*nodes)/temp)
        prob=(draw@ww).clamp(1e-6,1-1e-6);label=Y[idx]
        mean_bp=m[:,1]*bsd+bmu;truth=B[idx]*bsd+bmu
        res=dict(n=len(idx),positive=int(label.sum().item()),log_loss=float(F.binary_cross_entropy(prob,label)),
            brier=float((prob-label).square().mean()),base_brier=float((prior-label).square().mean()),
            base_log_loss=float(F.binary_cross_entropy(torch.full_like(label,prior),label)),
            mae_bp=float((mean_bp-truth).abs().mean()),base_mae_bp=float((median-truth).abs().mean()),
            mean_epistemic_sd=float((((draw-prob[:,None]).square()@ww).clamp_min(0)).sqrt().mean()))
        if dump:
            arr=prob.cpu().numpy();lbl=label.cpu().numpy()
            groups=np.minimum((arr*10).astype(int),9)
            res['reliability']=[dict(bin=j,n=int(sum(groups==j)),p=float(arr[groups==j].mean()),
                frequency=float(lbl[groups==j].mean())) for j in sorted(map(int,np.unique(groups)))]
        return res
    trace=[];best=float('inf');es=plateau=0;lr=a.lr;start=time.monotonic()
    for ep in range(1,a.epochs+1):
        head.train()
        for idx in ids[0][torch.randperm(len(ids[0]),device=dev)].split(32):
            m,s,kl=head(X[idx],W[idx])
            logits=m[:,0]+s[:,0]*torch.randn_like(m[:,0])
            loss=F.binary_cross_entropy_with_logits(logits,Y[idx])+((m[:,1]-B[idx]).square()+s[:,1].square()).mean()+kl/len(ids[0])
            opt.zero_grad();loss.backward();opt.step()
        tr,va=metrics(0),metrics(1)
        score=va['log_loss']
        if not np.isfinite(score):raise ValueError('nonfinite loss')
        if score<best-1e-8:
            best=score;best_ep=ep;best_h=copy.deepcopy(head.state_dict());best_o=copy.deepcopy(opt.state_dict())
            es=plateau=0
            torch.save(best_h,folder/'best.pt.tmp');(folder/'best.pt.tmp').replace(folder/'best.pt')
        else:es+=1;plateau+=1
        trace.append(dict(epoch=ep,train=tr,monitor=va,lr=lr,patience=es,best_epoch=best_ep))
        if ep==1 or ep%5==0 or es==a.patience:
            save_json(folder/'progress.json',dict(**trace[-1],elapsed=time.monotonic()-start))
        if es>=a.patience:break
        if plateau>=a.plateau:
            lr/=5;head.load_state_dict(best_h);opt.load_state_dict(best_o)
            for g in opt.param_groups:g['lr']=lr
            plateau=0
    head.load_state_dict(torch.load(folder/'best.pt',map_location=dev,weights_only=True))
    # Calibration uses its own dates, never the monitored or assessed split.
    temp=min((.5,.75,1.,1.5,2.,3.,5.),key=lambda t:metrics(2,t)['log_loss'])
    save_json(folder/'trace.json',trace)
    np.savez(folder/'apply.npz',xmu=xmu,xsd=xsd,bmu=bmu,bsd=bsd,bounds=bounds,T=temp)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,sharex=True)
    for ax,metric in zip(axes,('log_loss','mae_bp')):
        for split in ('train','monitor'):ax.plot([r['epoch'] for r in trace],[r[split][metric] for r in trace],label=split)
        ax.axvline(best_ep,linestyle=':',color='black');ax.set_ylabel(metric);ax.legend()
    axes[-1].set_xlabel('epoch');fig.tight_layout();fig.savefig(folder/'curve.png');plt.close(fig)
    save_json(folder/'result.json',dict(role=role,source=source,epochs=ep,best_epoch=best_ep,seconds=time.monotonic()-start,
        stop='patience' if es>=a.patience else 'budget',temperature=temp,
        raw=[metrics(i,dump=True) for i in range(4)],calibrated_2023=metrics(3,temp,True)))
    print(role,ep,best_ep,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=2000);p.add_argument('--patience',type=int,default=80)
    p.add_argument('--plateau',type=int,default=20);p.add_argument('--lr',type=float,default=3e-4)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.data/'DATA.json').read_text())
    ident=hashlib.sha256((a.data/'DATA.json').read_bytes()+Path(__file__).read_bytes()+repr((a.epochs,a.patience,a.plateau,a.lr)).encode()).hexdigest()
    save_json(a.out/'RUN.json',dict(identity=ident,source=meta['source'],device=torch.cuda.get_device_name(),
             epochs=a.epochs,patience=a.patience,plateau=a.plateau,lr=a.lr,seed=19))
    for role in ('close_sell','close_buy','open_buy','open_sell'):
        name=role+'.npz'
        if hashlib.sha256((a.data/name).read_bytes()).hexdigest()!=meta['files'][name]:raise ValueError('data hash')
        finished=a.out/role/'result.json'
        if finished.exists():
            if json.loads(finished.read_text())['source']!=ident:raise ValueError('wrong run')
            continue
        fit_role(a.data/name,role,a,ident)
    save_json(a.out/'DONE.json',dict(identity=ident,roles=4))
