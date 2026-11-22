"""Resume six independent probes; overlap device inference with CPU native replay."""
import fcntl
import os
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from role429_campaign import TOOLS,PY,base,run,atomic,common,panel,score
from role426_sell_rank_control import contrasts,weighted

OUT=Path('/var/tmp/role435-within-week')
REMOTE=Path('/var/tmp/role435-worker')
HOST=os.environ['ROLE431_SSH_TARGET']


def replay(tag,bank):
    year,seed=map(int,tag.split('-'));x,t,valid=bank
    ids,end,_,y,z=common(year,seed,t);dt=t[ids];ts=z['origin_ns']
    ix=np.searchsorted(ts,dt);assert np.array_equal(ts[ix],dt)
    jul=base.ns(f'{year}-07-01');cal=(dt<jul)&(end<jul);held=dt>=jul
    assert end[cal].max()<dt[held].min() and dt[held].max()<base.ns(f'{year+1}-01-01')
    pn=panel(year,seed,*bank);assert np.array_equal(pn.p,z['p'])
    arms={};weeks=int(np.unique(pn.e.seg_of[pn.bars[ts>=jul]]).size)
    with np.load(OUT/f'{tag}.scores.npz') as d:
        assert str(d['sha'])==base.file_sha(OUT/f'{tag}.npz')
        for key in ('linear','linear_shift','nonlinear','nonlinear_shift'):
            p=d[key];assert p.shape==(len(ts),4) and np.isfinite(p).all()
            # One intercept per horizon fitted only on completed H1 outcomes.
            pred=p+(y[cal]-p[ix[cal]]).mean(0)
            r=score(pred,pn,dt,ix,y,jul,ts,True)
            rows=contrasts(pred[ix[held]],y[held],dt[held])
            assert abs(r['bp']-weighted(rows,'model'))<.001
            arms[key]=dict(**r,random_bp=weighted(rows,'control'),weeks=weeks,
                           per_horizon={str(h):weighted([v for v in rows if v['horizon']==h],'model') for h in base.HS})
    atomic(OUT/f'{tag}.json',arms);return arms


def lane(jobs,device,remote,bank,pool):
    tasks=[]
    for tag in jobs:
        year,seed=map(int,tag.split('-'))
        path=OUT/f'{tag}.npz';arc=Path('/var/tmp/role432-prefix-repair')/path.name
        source=torch.load(arc.with_suffix('.s1.pt'),map_location='cpu',weights_only=False)
        assert source['sha']==base.file_sha(arc)
        _,_,_,_,z=common(year,seed,bank[1])
        ii=np.searchsorted(bank[1],z['origin_ns'])
        assert np.array_equal(bank[1][ii],z['origin_ns']) and bank[2][ii].all()
        if not path.exists():
            with np.load(arc) as d:np.savez(path,x=d['x'],y=d['raw'],week=d['week'],u=bank[0][ii])
        sha=base.file_sha(path);dest=path.with_suffix('.scores.npz')
        if not dest.exists():
            root=REMOTE if remote else OUT
            if remote:run(['rsync','-a',str(path),f'{HOST}:{root}/'])
            cmd=['env',f'CUDA_VISIBLE_DEVICES={device}','CUBLAS_WORKSPACE_CONFIG=:4096:8',PY,
                 str((root if remote else TOOLS)/'role435_worker.py'),str(root/path.name),sha]
            run(['ssh','-o','ServerAliveInterval=15',HOST,shlex.join(cmd)] if remote else cmd)
            if remote:run(['rsync','-a',f'{HOST}:{REMOTE}/{dest.name}',str(OUT)+'/'])
        tasks.append(pool.submit(replay,tag,bank))
    return tasks


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    assert base.read(Path('/var/tmp/role434-monitor/DONE.json'))['qualified']==0
    market=base.file_sha(base.DEST/'market.npz');assert market==base.read(base.DEST/'CACHE.json')['sha']
    cfg=dict(source={k:base.file_sha(TOOLS/k) for k in ('role435_worker.py','role435_campaign.py')},market=market,
             scope='original 432 fit support; no 2024; H1 purged intercept only',
             gate='each of six: above zero/random/shift on matched bp, above zero/shift in account, >2 trades/week')
    if (OUT/'PROTOCOL.json').exists():assert base.read(OUT/'PROTOCOL.json')==cfg
    atomic(OUT/'PROTOCOL.json',cfg)
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    run(['ssh',HOST,f'mkdir -p {REMOTE}'])
    run(['rsync','-a',str(TOOLS/'role435_worker.py'),f'{HOST}:{REMOTE}/'])
    plans=[(['2023-7','2023-43'],1,True),(['2023-19','2022-43'],0,True),(['2022-7','2022-19'],0,False)]
    with ThreadPoolExecutor(3) as cpu,ThreadPoolExecutor(3) as gpu:
        fs=[gpu.submit(lane,*p,bank,cpu) for p in plans]
        cells=[f.result() for g in fs for f in g.result()]
    passed={}
    for k in ('linear','nonlinear'):
        hits=0
        for c in cells:
            a,b=c[k],c[k+'_shift'];ac=a['account']
            hits+=int(a['bp']>max(0,b['bp'],a['random_bp']) and ac['ret']>max(0,b['account']['ret'])
                      and ac['trades']>2*a['weeks'])
        passed[k]=hits
    atomic(OUT/'DONE.json',dict(cells=len(cells),passed=passed,replication=[k for k,v in passed.items() if v==6],promoted=False))


if __name__=='__main__':main()
