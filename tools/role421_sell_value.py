"""Automatic SELL-only sign, economic mean, and conditional severity contrasts."""
import copy
import fcntl
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from role420_sell_refit import common, atomic, base, panel, plateau

OUT=Path('/var/tmp/role421-sell-value')
ARMS=('sign','value','severity')


def output(arm,z,average):
    if arm=='sign':
        p=z.sigmoid()
        return p*average[0]+(1-p)*average[1]
    if arm=='value':return z
    z=z.reshape(-1,4,3)
    p=z[:,:,0].sigmoid()
    return p*F.softplus(z[:,:,1])-(1-p)*F.softplus(z[:,:,2])


def loss(arm,z,y):
    win=(y>0).float()
    if arm=='sign':return F.binary_cross_entropy_with_logits(z,win)
    if arm=='value':return F.mse_loss(z,y)
    z=z.reshape(-1,4,3)
    amount=torch.where(y>0,F.softplus(z[:,:,1]),F.softplus(z[:,:,2]))
    return F.binary_cross_entropy_with_logits(z[:,:,0],win)+F.mse_loss(amount,y.abs())


def score(p,pn,dates,ix,managed,jul,all_ts,positive_only):
    h1=all_ts<jul;enabled=~h1
    ranks=np.column_stack([np.searchsorted(np.sort(col[h1]),col,side='right')/h1.sum() for col in p.T])
    eligible=enabled[:,None]&(p>np.quantile(p[h1],.9,axis=0))
    if positive_only:eligible &= p>0
    bids=np.zeros((len(p),2,4));bids[:,1]=np.where(eligible,ranks,0)
    ac=pn.account(bids);assert ac['buy']==0
    pick=[];est=[]
    for h in range(4):
        wk=(dates//86400000000000+3)//7
        for w in np.unique(wk[dates>=jul]):
            k=np.flatnonzero((wk==w)&(dates>=jul));n=max(1,len(k)//10)
            chosen=k[np.argsort(-p[ix[k],h],kind='stable')[:n]]
            pick.extend(managed[chosen,h]);est.extend(p[ix[chosen],h])
    v=np.array(pick)
    return dict(account=ac,matched_n=len(v),bp=float(v.mean()),pred=float(np.mean(est)),
                gains=float(v.clip(0).mean()),losses=float((-v).clip(0).mean()),win=float((v>0).mean()))


def worker(task):
    year,seed=task;tag=f'{year}-{seed}'
    if (OUT/f'{tag}.json').is_file():return tag
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    x,t,valid=bank
    rows=[common(y,seed,t) for y in range(2020,year)]
    ids,end,net=[np.concatenate([r[j] for r in rows]) for j in (0,1,3)]
    cut=base.ns(f'{year-1}-07-01');train=(end<cut)&(t[ids]<cut);monitor=(t[ids]>=cut)&(end<base.ns(f'{year}-01-01'))
    assert valid[ids].all() and end[train].max()<t[ids[monitor]].min()
    X=torch.from_numpy(x[ids[train]]);V=torch.from_numpy(x[ids[monitor]])
    scale=float(net[train].std());assert scale>0
    Y=torch.tensor(net[train]/scale,dtype=torch.float32);T=torch.tensor(net[monitor]/scale,dtype=torch.float32)
    means=[torch.stack([c[c>0].mean() for c in Y.T]),torch.stack([c[c<=0].mean() for c in Y.T])]
    assert all(torch.isfinite(a).all() for a in means)
    torch.manual_seed(seed);s=base.Expert(208,False,'cls').net
    n=[s,copy.deepcopy(s),nn.Sequential(copy.deepcopy(s[0]),nn.SiLU(),nn.Linear(8,12))]
    with torch.no_grad():
        n[2][-1].weight[::3].copy_(s[-1].weight);n[2][-1].bias[::3].copy_(s[-1].bias)
    opts=[torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.001) for m in n]
    best=[np.inf]*3;ep=[0]*3;bad=[0]*3;miss=[0]*3;champ=[None]*3;mom=[None]*3
    history=[];gen=torch.Generator().manual_seed(seed)
    for epoch in range(1,2001):
        for k in torch.randperm(len(X),generator=gen).split(512):
            for arm,m,o in zip(ARMS,n,opts):
                o.zero_grad(set_to_none=True);l=loss(arm,m(X[k]),Y[k]);l.backward();o.step()
        values=[]
        for j,(arm,m,o) in enumerate(zip(ARMS,n,opts)):
            with torch.no_grad():v=F.mse_loss(output(arm,m(V),means),T).item()
            assert np.isfinite(v)
            values.append(v);improved=v<best[j]-1e-8
            if improved:
                best[j]=v;ep[j]=epoch;bad[j]=0
                champ[j]=copy.deepcopy(m.state_dict());mom[j]=plateau.snapshot(o)
            elif epoch>=100:bad[j]+=1
            miss[j],rate=plateau.observe(epoch,improved,miss[j],o.param_groups[0]['lr'],False)
            if rate is not None:plateau.restore(m,o,champ[j],mom[j],rate)
        history.append([epoch,values,bad.copy(),[o.param_groups[0]['lr'] for o in opts]])
        if epoch%20==0:atomic(OUT/f'progress-{tag}.json',dict(epoch=epoch,best=ep,patience=bad))
        if min(bad)>=80:break
    ids,_,_,managed,z=common(year,seed,t);dates=t[ids];all_ts=z['origin_ns']
    pn=panel(year,seed,*bank);ii=np.searchsorted(t,all_ts);ix=np.searchsorted(all_ts,dates)
    assert np.array_equal(t[ii],all_ts) and np.array_equal(all_ts[ix],dates) and np.array_equal(z['p'],pn.p)
    jul=base.ns(f'{year}-07-01');weeks=len(np.unique(pn.e.seg_of[pn.bars[all_ts>=jul]]))
    arms={}
    for j,(arm,m) in enumerate(zip(ARMS,n)):
        m.load_state_dict(champ[j]);m.eval()
        torch.save(dict(weights=m.state_dict(),scale=scale,means=means),OUT/f'{tag}-{arm}.pt')
        with torch.no_grad():p=(output(arm,m(torch.from_numpy(x[ii])),means)*scale).numpy()
        assert np.isfinite(p).all()
        arms[arm]={gate:score(p,pn,dates,ix,managed,jul,all_ts,gate=='positive') for gate in ('rank','positive')}
    atomic(OUT/f'{tag}.json',dict(year=year,seed=seed,epochs=epoch,best=ep,history=history,arms=arms,weeks=weeks))
    return tag


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    assert not base.read(Path('/var/tmp/role420-sell-paired/DONE.json'))['promoted']
    OUT.mkdir(exist_ok=True)
    with (OUT/'lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sig={p.name:base.file_sha(p) for p in [Path(__file__),Path(__file__).with_name('role420_sell_refit.py')]}
        if (OUT/'CONFIG.json').exists():assert sig==base.read(OUT/'CONFIG.json')['sources']
        atomic(OUT/'CONFIG.json',dict(sources=sig,monitor='common managed MSE, train-only scale',policy=plateau.POLICY,
            schedule='same permutations/epochs, joint patience 80, max 2000',replication='all six: above sign and zero in account and matched bp; >2 trades/week'))
        tasks=[(y,s) for y in (2022,2023) for s in (7,19,43)]
        with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
            for fut in as_completed([pool.submit(worker,t) for t in tasks]):print(fut.result(),flush=True)
        rows=[base.read(OUT/f'{y}-{s}.json') for y,s in tasks]
        passed=[]
        for arm in ARMS[1:]:
            for gate in ('rank','positive'):
                def good(r):
                    a=r['arms'][arm][gate];b=r['arms']['sign'][gate]
                    return a['bp']>max(0,b['bp']) and a['account']['ret']>max(0,b['account']['ret']) and a['account']['trades']>2*r['weeks']
                if all(map(good,rows)):passed.append([arm,gate])
        lines=['# 421: SELL magnitude experiment','','|Year|Seed|Epochs|Head|Matched bp|Win %|Rank return % / trades|Positive EV return % / trades|','|---|---|---|---|---|---|---|---|']
        for r in rows:
            for arm,a in r['arms'].items():
                v=a['rank'];ac=v['account'];pc=a['positive']['account']
                lines.append(f"|{r['year']}|{r['seed']}|{r['epochs']}|{arm}|{v['bp']:.2f}|{100*v['win']:.1f}|{ac['ret']:.2f} / {ac['trades']}|{pc['ret']:.2f} / {pc['trades']}|")
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        atomic(OUT/'DONE.json',dict(candidates=passed,promoted=False))


if __name__=='__main__':main()
