"""Parallel devices, sequential prefix fits, CPU retention and final replay."""
import fcntl
import os
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from role429_campaign import TOOLS,PY,run,base,common,atomic,outcomes,w
from role426_sell_rank_control import contrasts,weighted
from role431_worker import inherit

OUT=Path('/var/tmp/role431-growing')
REMOTE='/var/tmp/role431-worker'
HOST=os.environ['ROLE431_SSH_TARGET']
DAY=86400000000000


def prepare(year,seed,bank):
    x,t,valid=bank
    past=[common(yr,seed,t) for yr in range(2020,year)]
    ids,end,y=[np.concatenate([q[k] for q in past]) for k in (0,1,3)]
    dt=t[ids];assert valid[ids].all()
    blocks=[];audit=[]
    for yr in range(2020,year):
        for m,start,finish in [(1,f'{yr}-01-01',f'{yr}-07-01'),(7,f'{yr}-07-01',f'{yr+1}-01-01')]:
            if yr==year-1 and m==7:break
            lo,hi=base.ns(start),base.ns(finish);cut=hi-28*DAY
            a=np.flatnonzero((dt>=lo+3*DAY)&(dt<cut)&(end<cut))
            b=np.flatnonzero((dt>=cut+3*DAY)&(dt<hi)&(end<hi))
            assert len(a)>512 and len(b)>20 and end[a].max()<dt[b].min()
            blocks.append(a);audit.append(b)
    mon=np.flatnonzero((dt>=base.ns(f'{year-1}-07-01')+3*DAY)&(end<base.ns(f'{year}-01-01')))
    tr=np.concatenate(blocks);ai=np.concatenate(audit)
    assert not np.intersect1d(tr,ai).size and end[ai].max()<dt[mon].min()
    scale=y[blocks[0]].std();assert scale>0
    path=OUT/f'{year}-{seed}.npz'
    if not path.exists():
        np.savez(path,x=x[ids[tr]],y=(y[tr]/scale).astype('f'),v=x[ids[mon]],t=(y[mon]/scale).astype('f'),
                 sem=np.repeat(np.arange(len(blocks)),list(map(len,blocks))),seed=seed)
    return path,scale,[(x[ids[b]],y[b],dt[b]) for b in audit]


def evaluate(path,s,scale,aud,year,seed,bank,sha):
    d=torch.load(path.with_suffix(f'.stage{s}.pt'),map_location='cpu',weights_only=False)
    assert d['sha']==sha
    rows=[];accounts=[]
    for state in d['champs']:
        n=w.model();inherit(n,state);n.eval()
        blocks=[]
        for x,y,t in aud[:s+1]:
            with torch.no_grad():p=w.heads(n(torch.from_numpy(x)))[-1].numpy()*scale
            r=contrasts(p,y,t)
            blocks.append(dict(bp=weighted(r,'model'),random=weighted(r,'control'),weeks=r))
        rows.append(blocks)
        if s==len(aud)-1:accounts.append(outcomes(n,scale,bank,year,seed)[0])
    result=dict(stage=s,updates=d['steps_per_arm'],exposures=d['exposures'],epochs=len(d['history']),
                best_epoch=d['epochs'],stop=d['stopped'],retention=rows,accounts=accounts)
    atomic(path.with_suffix(f'.stage{s}.json'),result)
    return result


def lane(plan,bank):
    jobs,device,remote=plan;results=[]
    for year,seed in jobs:
        path,scale,aud=prepare(year,seed,bank);sha=base.file_sha(path)
        if remote:run(['rsync','-a',str(path),f'{HOST}:{REMOTE}/'])
        with ThreadPoolExecutor(1) as pool:
            pending=[]
            for s in range(len(aud)):
                root=Path(REMOTE) if remote else OUT
                cmd=['env',f'CUDA_VISIBLE_DEVICES={device}','CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=1',
                     PY,str((root if remote else TOOLS)/'role431_worker.py'),str(root/path.name),sha,str(s)]
                with path.with_suffix('.log').open('a') as log:
                    run(['ssh','-o','ServerAliveInterval=15',HOST,shlex.join(cmd)] if remote else cmd,
                        stdout=log,stderr=log)
                if remote:run(['rsync','-a',f'{HOST}:{REMOTE}/{path.stem}.stage{s}.pt',str(OUT)+'/'])
                pending.append(pool.submit(evaluate,path,s,scale,aud,year,seed,bank,sha))
            last=[f.result() for f in pending][-1];a,b=last['accounts']
            results.append(b['bp']>max(0,a['bp'],b['random_bp']) and b['account']['ret']>max(0,a['account']['ret'])
                           and b['account']['trades']>2*b['scoring_weeks'] and all(
                           child['bp']>=parent['bp'] for parent,child in zip(*last['retention'])))
    return results


def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    n=w.model();parent=w.snapshot(n)
    with torch.no_grad():next(n.parameters()).add_(1)
    inherit(n,parent)
    market=base.file_sha(base.DEST/'market.npz');assert market==base.read(base.DEST/'CACHE.json')['sha']
    config=dict(market=market,files={p:base.file_sha(TOOLS/p) for p in
                ('role431_worker.py','role431_campaign.py','role429_worker.py','role405_plateau.py')},
                blocks='last 28 calendar days per semester; 72h start gap, purge by outcome end',
                preprocessing='fixed retrospective upstream encoder; label scale from first fit block',
                policy='reset AdamW 3e-4 at each stage for both; common epoch size full fit; stop at first ES or 2000',
                retention='screen zero bp decline vs paired full-history arm on each audit block at final stage',
                gate='all 6 positive active and above paired control in return, above zero/control/random in matched bp; retention screen')
    if (OUT/'CONFIG.json').exists():assert config==base.read(OUT/'CONFIG.json')
    atomic(OUT/'CONFIG.json',config)
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    run(['ssh',HOST,f'mkdir -p {REMOTE}'])
    run(['rsync','-a',*[str(TOOLS/f) for f in ('role431_worker.py','role429_worker.py','role405_plateau.py')],f'{HOST}:{REMOTE}/'])
    plans=[([(2023,7),(2023,43)],1,True), ([(2023,19),(2022,43)],0,True), ([(2022,7),(2022,19)],0,False)]
    with ThreadPoolExecutor(3) as pool:
        fs=[pool.submit(lane,p,bank) for p in plans]
        passed=[v for f in fs for v in f.result()]
    atomic(OUT/'DONE.json',dict(pairs=len(passed),passed=sum(passed),next='exposure-matched shuffle' if all(passed)
                               else 'do not advance',promoted=False))


if __name__=='__main__':main()
