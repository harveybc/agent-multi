"""Test horizon sharing vs capacity on 421's conditional SELL payoff task."""
import copy
import fcntl
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import numpy as np
import torch
from torch import nn
from role421_sell_value import OUT as PREV, common, atomic, base, panel, score, output, loss, plateau

OUT=Path('/var/tmp/role423-sell-horizons')
NAMES=('shared8','shared32','split8')


class Split(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,3)) for _ in range(4)])

    def forward(self,x):
        return torch.cat([head(x) for head in self.heads],dim=1)


def worker(job):
    year,seed=job;tag=f'{year}-{seed}'
    if (OUT/f'{tag}.json').exists():return tag
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    x,t,valid=bank
    past=[common(v,seed,t) for v in range(2020,year)]
    ids,ends,y=[np.concatenate([r[j] for r in past]) for j in (0,1,3)]
    cut=base.ns(f'{year-1}-07-01')
    tr=(t[ids]<cut)&(ends<cut);va=(t[ids]>=cut)&(ends<base.ns(f'{year}-01-01'))
    assert valid[ids].all() and ends[tr].max()<t[ids[va]].min()
    scale=float(y[tr].std())
    X=torch.from_numpy(x[ids[tr]]);V=torch.from_numpy(x[ids[va]])
    Y=torch.tensor(y[tr]/scale,dtype=torch.float32);T=torch.tensor(y[va]/scale,dtype=torch.float32)
    torch.manual_seed(seed);split=Split()
    wide=nn.Sequential(nn.Linear(208,32),nn.SiLU(),nn.Linear(32,12))
    # All 32 units and output biases start identical. The wide control may
    # subsequently learn connections across horizons; split cannot.
    with torch.no_grad():
        wide[-1].weight.zero_()
        for j,h in enumerate(split.heads):
            a,b=slice(8*j,8*j+8),slice(3*j,3*j+3)
            wide[0].weight[a].copy_(h[0].weight);wide[0].bias[a].copy_(h[0].bias)
            wide[-1].weight[b,a].copy_(h[-1].weight);wide[-1].bias[b].copy_(h[-1].bias)
        torch.testing.assert_close(wide(X[:32]),split(X[:32]))
    torch.manual_seed(seed)
    small=nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,12))
    nets=[small,wide,split]
    gen=torch.Generator().manual_seed(seed);opts=[torch.optim.AdamW(n.parameters(),lr=.0003,weight_decay=.001) for n in nets]
    best=[np.inf]*3;ep=[0]*3;bad=[0]*3;miss=[0]*3;champ=[None]*3;mom=[None]*3;history=[]
    before=time.time()
    for epoch in range(1,2001):
        for k in torch.randperm(len(X),generator=gen).split(512):
            for net,o in zip(nets,opts):
                o.zero_grad(set_to_none=True);loss('severity',net(X[k]),Y[k]).backward();o.step()
        values=[]
        for j,(net,o) in enumerate(zip(nets,opts)):
            with torch.no_grad():v=(output('severity',net(V),None)-T).square().mean().item()
            assert np.isfinite(v);values.append(v)
            improved=v<best[j]-1e-8
            if improved:
                best[j]=v;ep[j]=epoch;bad[j]=0;champ[j]=copy.deepcopy(net.state_dict());mom[j]=plateau.snapshot(o)
            elif epoch>=100:bad[j]+=1
            miss[j],rate=plateau.observe(epoch,improved,miss[j],o.param_groups[0]['lr'],False)
            if rate is not None:plateau.restore(net,o,champ[j],mom[j],rate)
        history.append([epoch,values,bad.copy(),[o.param_groups[0]['lr'] for o in opts]])
        if epoch%10==0:atomic(OUT/f'progress-{tag}.json',dict(epoch=epoch,seconds=time.time()-before,best=ep,patience=bad))
        if min(bad)>=80:break
    idx,_,_,managed,z=common(year,seed,t);dates=t[idx];ts=z['origin_ns'];ii=np.searchsorted(t,ts);ix=np.searchsorted(ts,dates)
    pn=panel(year,seed,*bank);assert np.array_equal(pn.p,z['p']) and np.array_equal(t[ii],ts) and np.array_equal(ts[ix],dates)
    jul=base.ns(f'{year}-07-01');results={}
    for j,(name,net) in enumerate(zip(NAMES,nets)):
        net.load_state_dict(champ[j]);net.eval()
        torch.save(dict(weights=net.state_dict(),scale=scale),OUT/f'{tag}-{name}.pt')
        with torch.no_grad():p=(output('severity',net(torch.from_numpy(x[ii])),None)*scale).numpy()
        results[name]={g:score(p,pn,dates,ix,managed,jul,ts,g=='positive') for g in ('rank','positive')}
    atomic(OUT/f'{tag}.json',dict(year=year,seed=seed,epochs=epoch,best=ep,history=history,arms=results,
        params=[sum(p.numel() for p in n.parameters()) for n in nets],weeks=int(np.unique(pn.e.seg_of[pn.bars[ts>=jul]]).size)))
    return tag


def main():
    assert (Path('/var/tmp/role422-sell-cross')/'DONE.json').is_file()
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    OUT.mkdir(exist_ok=True)
    with (OUT/'lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sig={p.name:base.file_sha(p) for p in [Path(__file__),Path(__file__).with_name('role421_sell_value.py')]}
        if (OUT/'CONFIG.json').exists():assert base.read(OUT/'CONFIG.json')['sources']==sig
        atomic(OUT/'CONFIG.json',dict(sources=sig,arms=NAMES,rule='both held-forward semesters and all seeds; split must also beat wide control, not just small',promoted=False))
        tasks=[(y,s) for y in (2022,2023) for s in (7,19,43)]
        with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
            for f in as_completed([pool.submit(worker,t) for t in tasks]):print(f.result(),flush=True)
        rows=[base.read(OUT/f'{y}-{s}.json') for y,s in tasks];passed=[]
        for a in NAMES[1:]:
            for gate in ('rank','positive'):
                def qualifies(r):
                    v=r['arms'][a][gate];refs=[r['arms'][k][gate] for k in NAMES[:NAMES.index(a)]]
                    return (v['bp']>max([0.]+[q['bp'] for q in refs]) and
                            v['account']['ret']>max([0.]+[q['account']['ret'] for q in refs]) and v['account']['trades']>2*r['weeks'])
                if all(map(qualifies,rows)):passed.append([a,gate])
        lines=['# 423: SELL horizon specialists','','|Year|Seed|Epochs|Head|Matched bp|Rank return % / trades|Positive EV return % / trades|','|---|---|---|---|---|---|---|']
        for r in rows:
            for k,d in r['arms'].items():
                a=d['rank']['account'];b=d['positive']['account']
                lines.append(f"|{r['year']}|{r['seed']}|{r['epochs']}|{k}|{d['rank']['bp']:.2f}|{a['ret']:.2f} / {a['trades']}|{b['ret']:.2f} / {b['trades']}|")
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
        atomic(OUT/'DONE.json',dict(replication_candidates=passed,promoted=False))


if __name__=='__main__':main()
