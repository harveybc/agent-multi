"""Six cells on three device lanes. Seal selectors before CPU audit and H2 replay."""
import fcntl
import json
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from role431_campaign import HOST,TOOLS,PY,prepare,base,common,atomic,outcomes,w,run,DAY
from role434_score import candidates,weekly

SRC=Path('/var/tmp/role432-prefix-repair')
OUT=Path('/var/tmp/role434-monitor')
REMOTE=Path('/var/tmp/role434-monitor-worker')


def scorecell(tag,dev,remote,bank):
    year,seed=map(int,tag.split('-'));path=OUT/(tag+'.monitor.npz')
    ids,end,_,y,_=common(year-1,seed,bank[1])
    dt=bank[1][ids];mask=(dt>=base.ns(f'{year-1}-07-01')+3*DAY)&(end<base.ns(f'{year}-01-01'))
    prev,scale,aud=prepare(year,seed,bank)
    with np.load(SRC/f'{tag}.npz') as z:
        assert np.array_equal(z['v'],bank[0][ids[mask]]) and np.array_equal(z['t'],(y[mask]/scale).astype('f'))
    np.savez(path,week=(dt[mask]//DAY+3)//7,raw=y[mask])
    root=REMOTE if remote else OUT
    dest=OUT/f'{tag}-sealed.json'
    if remote:run(['rsync','-a',str(path),f'{HOST}:{REMOTE}/'])
    cmd=['env',f'CUDA_VISIBLE_DEVICES={dev}','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8',PY,
         str((root if remote else TOOLS)/'role434_score.py'),'/var/tmp/role432-worker' if remote else str(SRC),
         str(root/path.name),str(root/dest.name),tag]
    run(['ssh',HOST,shlex.join(cmd)] if remote else cmd)
    if remote:run(['rsync','-a',f'{HOST}:{REMOTE}/{dest.name}',str(OUT)+'/'])
    return tag,base.read(dest),scale,aud


def evaluate(bundle,bank):
    tag,stages,scale,aud=bundle;year,seed=map(int,tag.split('-'))
    for s in stages:
        k=s['stage']
        d=torch.load(SRC/f'{tag}.s{k}.pt',map_location='cpu',weights_only=False)
        states=dict(candidates(d));result={}
        for key in dict.fromkeys((s['A'],s['B'])):
            net=w.model();net.load_state_dict(states[key]);net.eval()
            values=[]
            with torch.inference_mode():
                for x,y,t in aud[:k+1]:
                    p=w.heads(net(torch.from_numpy(x)))[-1].numpy()
                    values.append(weekly(p,y,(t//DAY+3)//7))
            result[key]=dict(audit_bp=float(np.concatenate(values).mean()),blocks=[float(a.mean()) for a in values],
                             held=outcomes(net,scale,bank,year,seed)[0] if k==len(aud)-1 else None)
        s['eval']=result
    atomic(OUT/f'{tag}.json',stages)
    return stages


def lane(jobs,dev,remote,bank,pool):
    return [pool.submit(evaluate,scorecell(tag,dev,remote,bank),bank) for tag in jobs]


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    lock=(OUT/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC.parent/'role431-growing/CONFIG.json')['market']
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    rule=dict(candidates='432 guard-only parent, champion and pre-rollback; unique within stage',
              A='min monitor MSE tolerance 1e-8',B='max weekly equal-quota mean managed advantage; tie min MSE',
              gate='require B>A final held matched bp and net return in each of six cells, active >2 trades/week and positive net; otherwise no feedback retraining')
    atomic(OUT/'PROTOCOL.json',rule)
    run(['ssh',HOST,f'mkdir -p {REMOTE}'])
    run(['rsync','-a',*[str(TOOLS/k) for k in ('role434_score.py','role429_worker.py','role405_plateau.py')],f'{HOST}:{REMOTE}/'])
    plans=[(['2023-7','2023-43'],1,True),(['2023-19','2022-43'],0,True),(['2022-7','2022-19'],0,False)]
    with ThreadPoolExecutor(3) as cpu, ThreadPoolExecutor(3) as gpu:
        fs=[gpu.submit(lane,*p,bank,cpu) for p in plans]
        cells=[f.result() for lanejobs in fs for f in lanejobs.result()]
    final=[];different=0
    for c in cells:
        different+=sum(s['A']!=s['B'] for s in c)
        s=c[-1];a,b=[s['eval'][s[k]]['held'] for k in ('A','B')]
        final.append(b['bp']>a['bp'] and b['account']['ret']>max(a['account']['ret'],0)
                     and b['account']['trades']>2*b['scoring_weeks'])
    atomic(OUT/'DONE.json',dict(stages=sum(map(len,cells)),different=different,qualified=sum(final),
                                decision='eligible for prospective training contrast' if all(final) else 'stop; no scheduler change'))


if __name__=='__main__':main()
