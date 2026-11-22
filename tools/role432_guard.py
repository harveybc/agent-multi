"""Paired stagewise weight repair vs guard only; no account evaluation here."""
import copy
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from role431_worker import inherit,w,plateau


def legal(score,floor):
    return bool(np.all(np.asarray(score)>=np.asarray(floor)-1e-9))


def run(path,sha):
    assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==sha
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    with np.load(path) as z:
        x,y,v,t=[torch.from_numpy(z[k]).cuda() for k in ('x','y','v','t')]
        sem,week,raw,seed=z['sem'],z['week'],z['raw'],int(z['seed'])
        n=w.model()
        first={k:torch.from_numpy(z[k].copy()) for k in n.state_dict()}
    assert not legal([3.,1.99],[2.,2.]) and legal([2.,2.],[2.,2.])
    nets=[n.cuda(),copy.deepcopy(n).cuda()]
    parents=[first,first];steps=(len(x)+511)//512
    groups=[[np.flatnonzero((sem==h)&(week==q)) for q in np.unique(week[sem==h])] for h in np.unique(sem)]
    @torch.inference_mode()
    def scores(n,s):
        # Evaluate only previously available rows. Fixed per-semester call shape
        # keeps numerical tie handling identical at later checkpoints.
        total=0.;cnt=0;out=[]
        for h in range(s):
            ix=np.flatnonzero(sem==h);p=w.heads(n(x[ix]))[-1].cpu().numpy()
            for g in groups[h]:
                pick=g[np.argsort(-p[np.searchsorted(ix,g)],axis=0,kind='stable')[:max(1,len(g)//10)]]
                total+=raw[pick,np.arange(4)].sum(dtype=np.float64);cnt+=pick.size
            out.append(total/cnt)
        return out
    @torch.inference_mode()
    def mse(n):return float((w.heads(n(v))[-1]-t).square().mean())
    previous=[[],[]]
    for s in range(1,int(sem.max())+1):
        dest=path.with_suffix(f'.s{s}.pt')
        if dest.exists():
            old=torch.load(dest,map_location='cpu',weights_only=False);assert old['sha']==sha
            parents=old['champs'];previous=old['exit'];continue
        for n,p in zip(nets,parents):inherit(n,p)
        floors=[scores(n,s) for n in nets]
        for f,p in zip(floors,previous):assert legal(f[:len(p)],p)
        opts=[torch.optim.AdamW(n.parameters(),lr=.0003,weight_decay=.001) for n in nets]
        best=[mse(n) for n in nets];champs=copy.deepcopy(parents)
        moms=[plateau.snapshot(o) for o in opts];ep=[0,0];bad=[0,0];miss=[0,0]
        rejected=[None,None];hist=[];log=[];rewinds=[]
        pool=np.flatnonzero(sem<=s);rng=np.random.default_rng(seed+100*s)
        for epoch in range(1,2001):
            order=np.concatenate([rng.permutation(pool) for _ in range((len(x)+len(pool)-1)//len(pool))])[:len(x)]
            for ids in np.array_split(order,steps):
                k=torch.from_numpy(ids).cuda()
                for n,o in zip(nets,opts):
                    o.zero_grad(set_to_none=True);w.severity(n(x[k]),y[k]).backward();o.step()
            vals=[]
            for j,(n,o) in enumerate(zip(nets,opts)):
                val=mse(n);vals.append(val);assert np.isfinite(val)
                improved=val<best[j]-1e-8
                if improved:
                    sc=scores(n,s);improved=legal(sc,floors[j])
                    if not improved and (rejected[j] is None or rejected[j][0]>val):
                        rejected[j]=(val,w.snapshot(n),sc,epoch)
                if improved:
                    best[j]=val;champs[j]=w.snapshot(n);moms[j]=plateau.snapshot(o);ep[j]=epoch;bad[j]=0
                elif epoch>=100:bad[j]+=1
                miss[j],rate=plateau.observe(epoch,improved,miss[j],o.param_groups[0]['lr'],False)
                if rate is not None:
                    rewinds.append((epoch,j,w.snapshot(n)))
                    plateau.restore(n,o,champs[j],moms[j],rate)
            hist.append([epoch,vals,best.copy(),bad.copy(),[o.param_groups[0]['lr'] for o in opts]])
            if epoch==1 or epoch%100==0:print(path.stem,s,epoch,bad,flush=True)
            if max(bad)>=80:break
        repair=None
        # Diagnose both arms identically, adopt only in treatment at the boundary.
        for j,r in enumerate(rejected):
            if r is None:continue
            for layer in ('0.','2.'):
                for fraction in (.5,1.):
                    hybrid={k:(torch.lerp(a,parents[j][k],fraction) if k.startswith(layer) else a)
                            for k,a in r[1].items()}
                    inherit(nets[j],hybrid);val=mse(nets[j]);sc=scores(nets[j],s)
                    log.append([j,r[3],r[0],r[2],layer,fraction,val,sc,legal(sc,floors[j])])
                    if j==1 and legal(sc,floors[j]) and val<best[j]-1e-8:
                        champs[j]=hybrid;best[j]=val;repair=[layer,fraction,r[3]]
        for n,p in zip(nets,champs):inherit(n,p)
        previous=[scores(n,s+1) for n in nets]
        for f,p in zip(floors,previous):assert legal(p[:-1],f)
        w.save(dict(sha=sha,parent=parents,champs=champs,entry=floors,exit=previous,
                    repairs=log,adopted=repair,history=hist,epochs=ep,updates=steps*epoch,
                    rollback=rewinds),dest)
        parents=champs


if __name__=='__main__':run(Path(sys.argv[1]),sys.argv[2])
