"""Three independent device lanes; collect segment checkpoints, audit and replay on CPU."""
import fcntl
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from role431_campaign import DAY,HOST,PY,TOOLS,atomic,base,common,outcomes,prepare,run,w
from role434_score import weekly

OUT=Path('/var/tmp/role447-segment-only')
REMOTE=Path('/var/tmp/role447-worker')


def lane(tags,device,remote,bank):
    x,t,_=bank
    results=[]
    for year,seed in tags:
        tag=f'{year}-{seed}'
        path=OUT/f'{tag}.npz'
        orig,scale,aud=prepare(year,seed,bank)
        with np.load(orig) as pack:
            ids,end,_,y,_=common(year-1,seed,t)
            dt=t[ids];mask=(dt>=base.ns(f'{year-1}-07-01')+3*DAY)&(end<base.ns(f'{year}-01-01'))
            assert np.array_equal(pack['v'],x[ids[mask]]) and np.array_equal(pack['t'],(y[mask]/scale).astype('f'))
            with np.load(Path('/var/tmp/role432-prefix-repair')/path.name) as src:
                assert all(np.array_equal(src[k],pack[k]) for k in pack.files)
                assert np.array_equal((src['raw']/scale).astype('f'),pack['y'])
                if not path.exists():
                    np.savez(path,**dict(pack),week=src['week'],raw=src['raw'],vw=(dt[mask]//DAY+3)//7,vy=y[mask])
        sha=base.file_sha(path)
        root=REMOTE if remote else OUT
        if remote:run(['rsync','-a',str(path),f'{HOST}:{REMOTE}/'])
        cmd=['env',f'CUDA_VISIBLE_DEVICES={device}','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8',
             PY,str((root if remote else TOOLS)/'role447_worker.py'),str(root/path.name),sha]
        with path.with_suffix('.log').open('a') as log:
            run(['ssh','-o','ServerAliveInterval=15',HOST,shlex.join(cmd)] if remote else cmd,stdout=log,stderr=log)
        if remote:run(['rsync','-a',f'{HOST}:{REMOTE}/{tag}.s*.pt',str(OUT)+'/'])
        rows=[]
        for s in range(len(aud)):
            for arm in (('common',) if s==0 else ('validation','retention')):
                d=torch.load(path.with_suffix(f'.s{s}.{arm}.pt'),map_location='cpu',weights_only=False)
                assert d['sha']==sha and d['prior_rows_optimized']==0
                net=w.model();net.load_state_dict(d['weights']);net.eval()
                audit=[]
                with torch.inference_mode():
                    for xx,yy,tt in aud[:s+1]:
                        pred=w.heads(net(torch.from_numpy(xx)))[-1].numpy()
                        audit.append(float(weekly(pred,yy,(tt//DAY+3)//7).mean()))
                rows.append(dict(stage=s,arm=arm,epochs=d['epochs'],best=d['best_epoch'],stop=d['stop'],
                                 audit_bp=audit,champ=d['history'][d['best_epoch']],rollbacks=sum(r[-1] for r in d['history']),
                                 held=outcomes(net,scale,bank,year,seed)[0] if s==len(aud)-1 else None))
        atomic(OUT/f'{tag}.json',rows);results.append(tag)
    return results


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    files=('role447_worker.py','role429_worker.py','role405_plateau.py')
    protocol=dict(files={k:base.file_sha(TOOLS/k) for k in (*files,'role447_campaign.py')},
                  market=base.file_sha(base.DEST/'market.npz'),scope='SELL head only; same fixed encoder; no 2024')
    assert protocol['market']==base.read(base.DEST/'CACHE.json')['sha']
    if (OUT/'PROTOCOL.json').exists(): assert base.read(OUT/'PROTOCOL.json')==protocol
    atomic(OUT/'PROTOCOL.json',protocol)
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    run(['ssh',HOST,f'mkdir -p {REMOTE}'])
    run(['rsync','-a',*[str(TOOLS/k) for k in files],f'{HOST}:{REMOTE}/'])
    plans=[([(2023,7),(2023,43)],1,True), ([(2023,19),(2022,43)],0,True), ([(2022,7),(2022,19)],0,False)]
    with ThreadPoolExecutor(3) as pool:
        fs=[pool.submit(lane,*p,bank) for p in plans]
        tags=[t for f in fs for t in f.result()]
    atomic(OUT/'DONE.json',dict(cells=tags,promoted=False))
