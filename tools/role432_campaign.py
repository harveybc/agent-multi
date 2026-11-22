"""Distribute guarded pairs and audit each final state outside the guard."""
import os
import fcntl
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from role431_campaign import prepare,OUT as PREV,DAY,HOST,TOOLS,PY,run,base,common,atomic,outcomes,w
from role426_sell_rank_control import contrasts,weighted

OUT=Path('/var/tmp/role432-prefix-repair')
REMOTE='/var/tmp/role432-worker'


def lane(plan,bank):
    jobs,dev,remote=plan; result=[]
    for year,seed in jobs:
        tag=f'{year}-{seed}';path=OUT/f'{tag}.npz'
        src,scale,aud=prepare(year,seed,bank)
        first=torch.load(src.with_suffix('.stage0.pt'),map_location='cpu',weights_only=False)
        assert first['sha']==base.file_sha(src)
        rows=[common(yr,seed,bank[1]) for yr in range(2020,year)]
        ids,end,y=[np.concatenate([r[k] for r in rows]) for k in (0,1,3)]
        dt=bank[1][ids];keep=[]
        for s in range(len(aud)):
            yr=2020+s//2;lo=base.ns(f'{yr}-01-01' if s%2==0 else f'{yr}-07-01')
            hi=base.ns(f'{yr}-07-01' if s%2==0 else f'{yr+1}-01-01')
            keep.extend(np.flatnonzero((dt>=lo+3*DAY)&(dt<hi-28*DAY)&(end<hi-28*DAY)))
        with np.load(src) as z:
            assert np.array_equal(z['x'],bank[0][ids[keep]]) and np.array_equal(z['y'],(y[keep]/scale).astype('f'))
            if not path.exists():np.savez(path,**dict(z),week=(dt[keep]//DAY+3)//7,raw=y[keep],
                                           **{k:v.numpy() for k,v in first['champs'][1].items()})
        sha=base.file_sha(path)
        root=Path(REMOTE) if remote else OUT
        if remote:run(['rsync','-a',str(path),f'{HOST}:{REMOTE}/'])
        args=['env',f'CUDA_VISIBLE_DEVICES={dev}','CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=1',
              PY,str((root if remote else TOOLS)/'role432_guard.py'),str(root/path.name),sha]
        with path.with_suffix('.log').open('a') as log:
            run(['ssh',HOST,shlex.join(args)] if remote else args,stdout=log,stderr=log)
        if remote:run(['rsync','-a',f'{HOST}:{REMOTE}/{tag}.s*.pt',str(OUT)+'/'])
        detail=[]
        for s in range(1,len(aud)):
            d=torch.load(path.with_suffix(f'.s{s}.pt'),map_location='cpu',weights_only=False)
            assert d['sha']==sha
            arms=[]
            for state in d['champs']:
                n=w.model();n.load_state_dict(state);n.eval()
                ret=[]
                for x,y,t in aud[:s+1]:
                    with torch.no_grad():p=w.heads(n(torch.from_numpy(x)))[-1].numpy()*scale
                    ret.append(weighted(contrasts(p,y,t),'model'))
                arms.append(dict(audit=ret,held=outcomes(n,scale,bank,year,seed)[0] if s==len(aud)-1 else None))
            detail.append(dict(stage=s,entry=d['entry'],exit=d['exit'],adopted=d['adopted'],
                               swaps=d['repairs'],updates=d['updates'],arms=arms))
        atomic(path.with_suffix('.json'),detail);result.append(tag)
    return result


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    lock=(OUT/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sig={p:base.file_sha(TOOLS/p) for p in ('role432_campaign.py','role432_guard.py','role431_worker.py','role429_worker.py','role405_plateau.py')}
    if (OUT/'CONFIG.json').exists():assert base.read(OUT/'CONFIG.json')==sig
    atomic(OUT/'CONFIG.json',sig)
    assert base.file_sha(base.DEST/'market.npz')==base.read(PREV/'CONFIG.json')['market']
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    run(['ssh',HOST,f'mkdir -p {REMOTE}'])
    run(['rsync','-a',*[str(TOOLS/p) for p in sig if 'campaign' not in p],f'{HOST}:{REMOTE}/'])
    plans=[([(2023,7),(2023,43)],1,True), ([(2023,19),(2022,43)],0,True), ([(2022,7),(2022,19)],0,False)]
    with ThreadPoolExecutor(3) as pool:
        fs=[pool.submit(lane,p,bank) for p in plans]; tags=sum([f.result() for f in fs],[])
    rows=[base.read(OUT/f'{tag}.json') for tag in tags]
    atomic(OUT/'DONE.json',dict(cells=tags,adopted=sum(s['adopted'] is not None for r in rows for s in r),
                               promoted=False))


if __name__=='__main__':main()
