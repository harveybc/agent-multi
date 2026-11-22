"""One deterministic control/intervention fit; receives TRAIN and monitor only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import role405_plateau as plateau


def model():
    return nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,12))


def heads(z):
    z=z.reshape(-1,4,3)
    p=z[:,:,0].sigmoid()
    w,l=F.softplus(z[:,:,1]),F.softplus(z[:,:,2])
    return z[:,:,0],w,l,p*w-(1-p)*l


def severity(z,y):
    s,w,l,_=heads(z)
    return F.binary_cross_entropy_with_logits(s,(y>0).float())+F.mse_loss(torch.where(y>0,w,l),y.abs())


def pairloss(a,b,ya,yb):
    sign=(ya-yb).sign()
    return (F.softplus(-sign*(a-b))*(sign!=0)).sum()/(sign!=0).sum().clamp(min=1)


def partners(groups,rng):
    p=np.arange(sum(map(len,groups)))
    for g in groups:
        order=rng.permutation(g)
        p[order]=np.roll(order,1)
    return p


def snapshot(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def save(obj,path):
    tmp=path.with_suffix('.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)


def train(path,sha):
    assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==sha
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    dev=torch.device('cuda:0')
    with np.load(path) as z:
        X,Y,V,T=[torch.from_numpy(z[k]).to(dev) for k in ('x','y','v','t')]
        wk=z['week']; seed=int(z['seed'])
    stem=path.with_suffix('')
    ck=stem.with_suffix('.resume')
    final=stem.with_suffix('.pt')
    if final.exists():
        assert torch.load(final,map_location='cpu',weights_only=False)['sha']==sha
        return
    groups=[np.flatnonzero(wk==w) for w in np.unique(wk)]
    torch.manual_seed(seed)
    a=model().to(dev); nets=[a,copy.deepcopy(a)]
    opts=[torch.optim.AdamW(n.parameters(),lr=3e-4,weight_decay=.001) for n in nets]
    initial=snapshot(a)
    best=[float('inf')]*2; epochs=[0,0]; bad=[0,0]; miss=[0,0]; champs=[None,None]; moms=[None,None]
    hist=[]; start=1
    g=torch.Generator().manual_seed(seed)
    if ck.exists():
        d=torch.load(ck,map_location='cpu',weights_only=False)
        assert d['sha']==sha
        for n,o,w,m in zip(nets,opts,d['nets'],d['opts']):
            n.load_state_dict(w);o.load_state_dict(m)
        best,epochs,bad,miss,champs,moms,hist=[d[k] for k in ('best','epochs','bad','miss','champs','moms','hist')]
        g.set_state(d['rng']);start=len(hist)+1
    def record(final_epoch):
        data=dict(sha=sha,champs=champs,initial=initial,epochs=epochs,hist=hist,
                  device=torch.cuda.get_device_name(),stop='patience' if min(bad)>=80 else 'cap')
        if final_epoch:
            save(data,final)
        else:
            save(dict(**data,best=best,bad=bad,miss=miss,moms=moms,rng=g.get_state(),
                      nets=[snapshot(n) for n in nets],opts=[o.state_dict() for o in opts]),ck)
    clock=time.monotonic()
    for epoch in range(start,2001):
        other=partners(groups,np.random.default_rng(seed*10000+epoch))
        assert np.all(wk==wk[other])
        ix=torch.randperm(len(X),generator=g)
        for k in ix.split(512):
            j=torch.from_numpy(other[k.numpy()]).to(dev);k=k.to(dev)
            inp,y=X[k],Y[k]
            for arm,(net,o) in enumerate(zip(nets,opts)):
                o.zero_grad(set_to_none=True);z=net(inp);loss=severity(z,y)
                if arm:
                    loss=loss+pairloss(heads(z)[-1],heads(net(X[j]))[-1],y,Y[j])
                loss.backward();o.step()
        vals=[]
        for j,(net,o) in enumerate(zip(nets,opts)):
            with torch.no_grad():v=F.mse_loss(heads(net(V))[-1],T).item()
            assert np.isfinite(v)
            vals.append(v)
            improved=v<best[j]-1e-8
            if improved:
                best[j]=v;epochs[j]=epoch;bad[j]=0;champs[j]=snapshot(net);moms[j]=plateau.snapshot(o)
            elif epoch>=100:bad[j]+=1
            miss[j],rate=plateau.observe(epoch,improved,miss[j],o.param_groups[0]['lr'],False)
            if rate is not None:plateau.restore(net,o,champs[j],moms[j],rate)
        with torch.no_grad():
            fit=[float(F.mse_loss(heads(n(X))[-1],Y)) for n in nets]
        hist.append(dict(epoch=epoch,fit=fit,monitor=vals,best_epoch=epochs.copy(),patience=bad.copy(),
                         lr=[o.param_groups[0]['lr'] for o in opts]))
        if epoch%10==0 or epoch==1:
            print(path.stem,hist[-1],round(time.monotonic()-clock,1),flush=True)
        if min(bad)>=80 or epoch==2000:
            record(True);break
        if epoch%20==0:record(False)


if __name__=='__main__':
    train(Path(sys.argv[1]),sys.argv[2])
