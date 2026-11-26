"""Conditional weekly calendar-placebo schedules; standalone NumPy CPU worker."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HOUR=3_600_000_000_000
DRAWS=8192


def schedule(t,ex,accept,start,end,cutoff,strict,nop_only):
    """Count feasible suffixes; weighted branches sample uniform complete schedules."""
    n=len(t); m=len(start)
    trade=np.where(accept,np.searchsorted(t,ex,'right'),np.arange(n)+1)
    assert np.all(trade>np.arange(n)) and np.all(np.diff(t)>0)
    allow=np.zeros((m,n),bool); jump=np.zeros((m,n),int)
    for j,(s,e) in enumerate(zip(start,end)):
        span=e-s
        goal=np.searchsorted(t,e,'right')-np.searchsorted(t,s)
        jump[j]=np.searchsorted(t,t+span,'right')
        allow[j]=((t//HOUR%24//6==s//HOUR%24//6)&(t+span<=cutoff)
                  &(jump[j]-np.arange(n)==goal))
        if strict:allow[j]&=(t//(24*HOUR)==s//(24*HOUR))
        if nop_only:allow[j]&=~accept
    ways=np.zeros((m+1,n+1))
    ways[m,:]=1.
    for j in reversed(range(m)):
        for i in reversed(range(n)):
            ways[j,i]=ways[j,trade[i]]
            if allow[j,i]:ways[j,i]+=ways[j+1,jump[j,i]]
    assert ways[0,0]>=1 and np.isfinite(ways[0,0])
    p=np.zeros((m+1,n)); following=np.zeros((m+1,n),int)
    for j in range(m):
        np.divide(ways[j+1,jump[j]],ways[j,:n],out=p[j],where=allow[j]&(ways[j,:n]>0))
        following[j]=jump[j]
    return trade,p,following,ways[0,0]


def simulate(path,strict,nop_only,n_draws=DRAWS):
    with np.load(path) as a:
        t,w,score,ex,pnl,starts,until=[a[k] for k in ('ts','week','score','exit','pnl','starts','until')]
    y=int(path.stem)
    rng=np.random.default_rng([486,y,int(strict),int(nop_only)])
    # Last path forces the archived waits; other paths do not consult BUY or realized returns to sample.
    size=n_draws+1
    bal=np.ones(size);peak=bal.copy();dd=np.zeros(size); hours=np.zeros(size)
    orders=np.zeros(size,int);moved=np.zeros(size,int)
    entropy=0.
    for cutoff in np.unique(w):
        take=w==cutoff
        tt=t[take];ee=ex[take];pos=score[take]>0;g=pnl[take]/10000
        sel=(starts>=tt[0])&(starts<=tt[-1])
        s,e=starts[sel],until[sel]
        trade,p,jump,ways=schedule(tt,ee,pos,s,e,cutoff,strict,nop_only)
        entropy+=np.log10(ways)
        at=np.zeros(size,int); token=np.zeros(size,int); start_set=set(s.tolist())
        for i,now in enumerate(tt):
            active=np.flatnonzero(at==i)
            if len(active)==0:continue
            yes=rng.random(len(active))<p[token[active],i]
            if active[-1]==n_draws:yes[-1]=now in start_set
            held=active[yes];step=active[~yes]
            if len(held):
                j=token[held]
                moved[held]+=now!=s[j]
                at[held]=jump[j,i];token[held]+=1
            at[step]=trade[i]
            if pos[i]:
                bal[step]*=1+g[i];peak[step]=np.maximum(peak[step],bal[step])
                dd[step]=np.maximum(dd[step],1-bal[step]/peak[step])
                hours[step]+=(ee[i]-now)/HOUR;orders[step]+=1
        np.testing.assert_array_equal(token,np.full(size,len(s)))
    assert moved[-1]==0
    return dict(ret=100*(bal-1),dd=100*dd,orders=orders,hours=hours,moved=moved,
                log10_schedules=entropy,pauses=len(starts))


def job(task):
    src,out,y,strict,nop=task
    data=simulate(src/f'{y}.npz',strict,nop)
    np.savez_compressed(out/f'{y}-{int(strict)}-{int(nop)}.npz',**data)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('src',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--strict',type=int,choices=[0,1],required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(job,[(a.src,a.out,y,bool(a.strict),nop) for y in range(2020,2024) for nop in (False,True)]))
