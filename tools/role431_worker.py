"""Paired growing-history stages on one GPU. No H2 data on the worker."""
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import role429_worker as w
import role405_plateau as plateau


def inherit(net,parent):
    net.load_state_dict(parent)
    assert all(torch.equal(t.cpu(),parent[k]) for k,t in net.state_dict().items())


def train(path,sha,last):
    assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==sha
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    with np.load(path) as z:
        x,y,v,t=[torch.from_numpy(z[k]).cuda() for k in ('x','y','v','t')]
        sem=z['sem']; seed=int(z['seed'])
    torch.manual_seed(seed);a=w.model().cuda();nets=[a,copy.deepcopy(a)]
    parents=[w.snapshot(n) for n in nets]
    steps=(len(x)+511)//512
    assert 0<=last<=sem.max()
    for s in range(last+1):
        dest=path.with_suffix(f'.stage{s}.pt')
        if dest.exists():
            saved=torch.load(dest,map_location='cpu',weights_only=False)
            assert saved['sha']==sha
            parents=saved['champs'];continue
        subsets=[np.arange(len(x)),np.flatnonzero(sem<=s)]
        rngs=[np.random.default_rng(seed+100*s) for _ in nets]
        for n,p in zip(nets,parents):inherit(n,p)
        opts=[torch.optim.AdamW(n.parameters(),lr=.0003,weight_decay=.001) for n in nets]
        def measure(net):
            with torch.inference_mode():return float((w.heads(net(v))[-1]-t).square().mean())
        best=[measure(n) for n in nets];init=best.copy();champs=copy.deepcopy(parents)
        moms=[plateau.snapshot(o) for o in opts];ep=[0,0];bad=[0,0];miss=[0,0]
        hist=[];rollback=[]; exposures=np.zeros((2,len(np.unique(sem))),dtype=int)
        for epoch in range(1,2001):
            vals=[]
            for j,(net,o,pool,rng) in enumerate(zip(nets,opts,subsets,rngs)):
                order=np.concatenate([rng.permutation(pool) for _ in range((len(x)+len(pool)-1)//len(pool))])[:len(x)]
                exposures[j]+=np.bincount(sem[order],minlength=exposures.shape[1])
                for k in np.array_split(order,steps):
                    k=torch.from_numpy(k).cuda()
                    o.zero_grad(set_to_none=True);w.severity(net(x[k]),y[k]).backward();o.step()
                val=measure(net);assert np.isfinite(val);vals.append(val)
                improved=val<best[j]-1e-8
                if improved:
                    best[j]=val;ep[j]=epoch;bad[j]=0;champs[j]=w.snapshot(net);moms[j]=plateau.snapshot(o)
                elif epoch>=100:bad[j]+=1
                miss[j],lr=plateau.observe(epoch,improved,miss[j],o.param_groups[0]['lr'],False)
                if lr is not None:
                    rollback.append((j,epoch,w.snapshot(net)))
                    plateau.restore(net,o,champs[j],moms[j],lr)
            hist.append((epoch,vals,bad.copy(),[o.param_groups[0]['lr'] for o in opts]))
            if epoch==1 or epoch%50==0:print(path.stem,s,epoch,ep,bad,flush=True)
            if max(bad)>=80:break
        assert exposures[0].sum()==exposures[1].sum()
        for j,n in enumerate(nets):
            inherit(n,champs[j])
            assert abs(measure(n)-best[j])<1e-7 and best[j]<=init[j]
        w.save(dict(sha=sha,parent=parents,champs=champs,epochs=ep,history=hist,rollback=rollback,
                    steps_per_arm=epoch*steps,exposures=exposures.tolist(),parent_mse=init,best=best,
                    stopped='paired_patience' if max(bad)>=80 else 'cap'),dest)
        parents=champs


if __name__=='__main__':train(Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]))
