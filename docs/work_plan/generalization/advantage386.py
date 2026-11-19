"""Fit the two expected account advantages on authenticated 385 examples."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_direct385 import head,save_json

DATA='623f19664da3775df54fadbc94285b20f958b65163ecc8c0d3181df98ece22c0'

def choose(v):
    if v.ndim!=2 or v.shape[1]!=2 or not np.isfinite(v).all():raise ValueError('advantage')
    b,s=v.T
    return np.where((b>s)&(b>0),1,np.where((s>b)&(s>0),2,0))

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def fit(a,b,seed):
    if digest(a/'DATA.json')!=DATA:raise ValueError('data manifest')
    files=json.loads((a/'DATA.json').read_text())['files']
    for n in ['flat.npz','market.npz']:
        if digest(a/n)!=files[n]:raise ValueError(n)
    with np.load(a/'market.npz') as d:market=d['x']
    with np.load(a/'flat.npz') as d:
        x=market[d['idx']];v=d['bp'][:,1:];split=d['split']
    b.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed)
    tr=split==0;mu=x[tr].mean(0);sd=x[tr].std(0);sd=np.where(sd>1e-8,sd,1)
    offset=v[tr].mean(0);scale=float(np.sqrt(np.mean((v[tr]-offset)**2)))
    if scale<=0:raise ValueError('constant targets')
    X=torch.tensor((x-mu)/sd,device='cuda');Y=torch.tensor((v-offset)/scale,device='cuda',dtype=X.dtype)
    ix=[torch.as_tensor(np.flatnonzero(split==i),device='cuda') for i in range(4)]
    net=head(x.shape[1],2).cuda();lr=.0003;opt=torch.optim.Adam(net.parameters(),lr=lr)
    best=float('inf');es=pat=0;trace=[]
    for epoch in range(1,2001):
        net.train()
        for j in ix[0][torch.randperm(len(ix[0]),device='cuda')].split(32):
            loss=F.mse_loss(net(X[j]),Y[j]);opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.inference_mode():train,val=[float(F.mse_loss(net(X[j]),Y[j])) for j in ix[:2]]
        if not np.isfinite([train,val]).all():raise ValueError('loss')
        if val<best-1e-8:
            best=val;be=epoch;bh=copy.deepcopy(net.state_dict());bo=copy.deepcopy(opt.state_dict());es=pat=0
        else:es+=1;pat+=1
        trace.append(dict(epoch=epoch,train=train,monitor=val,lr=lr,patience=es,best_epoch=be))
        if epoch==1 or epoch%5==0:save_json(b/'PROGRESS.json',trace[-1])
        if es==80:break
        if pat==20:
            net.load_state_dict(bh);opt.load_state_dict(bo);lr/=5
            for g in opt.param_groups:g['lr']=lr
            pat=0
    net.load_state_dict(bh);j=ix[3]
    with torch.inference_mode():pred=net(X[j]).cpu().numpy()*scale+offset
    decision=choose(pred);target=v[split==3];values=np.column_stack((np.zeros(len(target)),target))
    np.savez(b/'apply.npz',mu=mu,sd=sd,offset=offset,scale=scale,held_actions=decision)
    torch.save(net.cpu().state_dict(),b/'head.pt')
    save_json(b/'TRACE.json',trace)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for k in ('train','monitor'):ax.plot([r[k] for r in trace],label=k)
    ax.axvline(be-1,ls=':');ax.legend();ax.set(xlabel='Epoch (zero-based)',ylabel='Standardized MSE')
    fig.savefig(b/'curve.png');plt.close(fig)
    save_json(b/'RESULT.json',dict(seed=seed,data=DATA,head=digest(b/'head.pt'),apply=digest(b/'apply.npz'),
        epochs=epoch,best_epoch=be,decisions=np.bincount(decision,minlength=3).tolist(),regret_bp=float(
        np.mean(values.max(1)-values[np.arange(len(values)),decision])),nop_regret_bp=float(values.max(1).mean()),
        mse_bp2=float(((pred-target)**2).mean()),mean_baseline_mse_bp2=float(((offset-target)**2).mean())))
    print(seed,epoch,be,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);p.add_argument('seeds',nargs='+',type=int)
    a=p.parse_args()
    for seed in a.seeds:fit(a.data,a.out/str(seed),seed)
