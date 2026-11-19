"""Local closed-set classifiers. No provider calls. Worker needs numpy/torch/matplotlib."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_role384 import save_json

def head(d,k):
    return nn.Sequential(nn.Linear(d,8),nn.SiLU(),nn.Linear(8,k))

def run(args):
    torch.set_num_threads(1)
    a,b=args.data,args.out
    meta=json.loads((a/'DATA.json').read_text())
    for n,h in meta['files'].items():
        if hashlib.sha256((a/n).read_bytes()).hexdigest()!=h:raise ValueError('data')
    b.mkdir(exist_ok=False,parents=True)
    with np.load(a/'market.npz') as f:market=f['x']
    for role,k in [('flat',3),('long',2),('short',2)]:
        d=b/role;d.mkdir();torch.manual_seed(19)
        with np.load(a/(role+'.npz')) as f:
            idx,y,N,split=[f[s] for s in ('idx','y','bp','split')]
        N=N[:,:k]
        x=market[idx];mu=x[split==0].mean(0);sd=x[split==0].std(0)
        sd=np.where(sd>1e-8,sd,1)
        prior=np.bincount(y[split==0],minlength=k)/sum(split==0)
        if min(prior)==0:raise ValueError(('missing training class',role,prior))
        dev=torch.device(args.device)
        X=torch.tensor((x-mu)/sd,device=dev);Y=torch.tensor(y,device=dev)
        blocks=[torch.as_tensor(np.flatnonzero(split==i),device=dev) for i in range(4)]
        net=head(x.shape[1],k).to(dev);opt=torch.optim.Adam(net.parameters(),lr=.0003)
        def nll(i):
            with torch.inference_mode():return float(F.cross_entropy(net(X[blocks[i]]),Y[blocks[i]]))
        best=float('inf');es=pat=0;lr=.0003;hist=[]
        for ep in range(1,2001):
            net.train()
            for j in blocks[0][torch.randperm(len(blocks[0]),device=dev)].split(32):
                loss=F.cross_entropy(net(X[j]),Y[j]);opt.zero_grad();loss.backward();opt.step()
            net.eval();train,val=nll(0),nll(1)
            if not np.isfinite([train,val]).all():raise ValueError('diverged')
            if val<best-1e-8:
                best,be=val,ep
                bh=copy.deepcopy(net.state_dict());bo=copy.deepcopy(opt.state_dict());es=pat=0
            else:es+=1;pat+=1
            hist.append(dict(epoch=ep,train=train,monitor=val,patience=es,lr=lr,best_epoch=be))
            if ep==1 or ep%5==0:save_json(d/'PROGRESS.json',hist[-1])
            if es>=80:break
            if pat>=20:
                net.load_state_dict(bh);opt.load_state_dict(bo);lr/=5
                for g in opt.param_groups:g['lr']=lr
                pat=0
        net.load_state_dict(bh);net.eval()
        with torch.inference_mode():
            cal=net(X[blocks[2]])
            T=min((.5,.75,1.,1.5,2.,3.,5.),key=lambda t:float(F.cross_entropy(cal/t,Y[blocks[2]])))
            raw=net(X[blocks[3]])
            p=(raw/T).softmax(1).cpu().numpy()
        gold=y[split==3];v=N[split==3];dec=p.argmax(1)
        def regret(choice):return float((v.max(1)-v[np.arange(len(v)),choice]).mean())
        res=dict(role=role,epochs=ep,best_epoch=be,lr=lr,patience=es,T=T,fit_counts=np.bincount(y[split==0],minlength=k).tolist(),
                 gold_counts=np.bincount(gold,minlength=k).tolist(),decisions=np.bincount(dec,minlength=k).tolist(),
                 accuracy=float(np.mean(dec==gold)),prior_accuracy=float(np.mean(prior.argmax()==gold)),
                 nll=float(F.cross_entropy(raw/T,Y[blocks[3]])),prior_nll=float(-np.log(prior[gold]).mean()),
                 brier=float(((p-np.eye(k)[gold])**2).sum(1).mean()),
                 prior_brier=float(((prior-np.eye(k)[gold])**2).sum(1).mean()),
                 regret_bp=regret(dec),prior_regret_bp=regret(prior.argmax()))
        torch.save(net.cpu().state_dict(),d/'head.pt');np.savez(d/'apply.npz',mu=mu,sd=sd,T=T)
        save_json(d/'TRACE.json',hist)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for s in ('train','monitor'):ax.plot([r['epoch'] for r in hist],[r[s] for r in hist],label=s)
        ax.axvline(be,ls=':',color='black');ax.set(xlabel='Epoch',ylabel='Log loss');ax.legend()
        fig.savefig(d/'curve.png');plt.close(fig)
        save_json(d/'RESULT.json',res);print(role,ep,be,flush=True)
    save_json(b/'DONE.json',dict(data_sha=hashlib.sha256((a/'DATA.json').read_bytes()).hexdigest(),
                                  torch=str(torch.__version__),device=str(dev),seed=19))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');run(p.parse_args())
