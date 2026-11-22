"""Audit every floor-eligible 432 repair on reserved opportunities, without H2."""
import csv
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from role431_campaign import HOST,TOOLS,PY,prepare,base,w,run,atomic
from role424_sell_bias import selected

SRC=Path('/var/tmp/role432-prefix-repair')
OUT=Path('/var/tmp/role433-repair-audit')
REMOTE=Path('/var/tmp/role433-audit-worker')


def measure(st,blocks):
    net=w.model();net.load_state_dict(st);net.eval()
    with torch.inference_mode():
        return [selected(w.heads(net(torch.from_numpy(x)))[-1].numpy(),t) for x,y,t in blocks]


def audit(tag,bank):
    year,seed=map(int,tag.split('-'))
    _,_,audit_blocks=prepare(year,seed,bank)
    details=[]
    with (OUT/f'{tag}-swaps.csv').open('w') as fd:
        writer=csv.writer(fd);writer.writerow(['stage','layer','fraction','reference','block','origin_ns','horizon','change','net_advantage_bp'])
        for path in sorted(OUT.glob(tag+'.s*.pt')):
            s=int(path.suffixes[-2][2:]);d=torch.load(path,map_location='cpu',weights_only=False)
            blocks=audit_blocks[:s+1]
            refs={key:measure(d[key],blocks) for key in ('raw','accepted','parent')}
            for _,ep,_,_,layer,frac,_,_,ok in d['wanted']:
                assert ok
                hybrid={k:(torch.lerp(v,d['parent'][k],frac) if k.startswith(layer) else v) for k,v in d['raw'].items()}
                cand=measure(hybrid,blocks)
                row=dict(tag=tag,stage=s,layer=layer,fraction=frac,adopted=d['adopted']==[layer,frac,ep])
                for name,ref in refs.items():
                    vals=[]
                    for b,((_,y,t),a,c) in enumerate(zip(blocks,ref,cand)):
                        enter=c&~a;leave=a&~c
                        assert enter.sum()==leave.sum() and c.sum()>0
                        for typ,mask in [('in',enter),('out',leave)]:
                            for i,h in zip(*mask.nonzero()):
                                writer.writerow([s,layer,frac,name,b,int(t[i]),base.HS[h],typ,y[i,h]])
                        delta=float(y[c].mean()-y[a].mean())
                        assert np.isclose(delta,(y[enter].sum()-y[leave].sum())/c.sum())
                        vals.append(dict(k=int(c.sum()),swaps=int(enter.sum()),delta=delta,
                                         in_sum=float(y[enter].sum()),out_sum=float(y[leave].sum())))
                    row[name]=dict(blocks=vals,delta=float(np.average([v['delta'] for v in vals],weights=[v['k'] for v in vals])))
                details.append(row)
    return details


def lane(jobs,dev,remote,bank):
    result=[]
    for tag in jobs:
        want=sum(r[0]==1 and r[-1] for s in base.read(SRC/f'{tag}.json') for r in s['swaps'])
        if not want:continue
        script=(REMOTE if remote else TOOLS)/'role433_recover.py'
        args=['env',f'CUDA_VISIBLE_DEVICES={dev}','CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=1',PY,
              str(script),'/var/tmp/role432-worker' if remote else str(SRC),str(REMOTE if remote else OUT),tag]
        with (OUT/f'{tag}.log').open('w') as fd:
            run(['ssh',HOST,shlex.join(args)] if remote else args,stdout=fd,stderr=fd)
        if remote:run(['rsync','-a',f'{HOST}:{REMOTE}/{tag}.s*.pt',str(OUT)+'/'])
        rows=audit(tag,bank);assert len(rows)==want
        result+=rows
    return result


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    assert base.read(SRC/'DONE.json')['adopted']==2
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC.parent/'role431-growing/CONFIG.json')['market']
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    run(['ssh',HOST,f'mkdir -p {REMOTE}'])
    run(['rsync','-a',*[str(TOOLS/p) for p in ('role433_recover.py','role432_guard.py','role431_worker.py','role429_worker.py','role405_plateau.py')],f'{HOST}:{REMOTE}/'])
    with ThreadPoolExecutor(3) as pool:
        futures=[pool.submit(lane,j,d,r,bank) for j,d,r in [(['2023-7','2023-43'],1,True),
                 (['2023-19','2022-43'],0,True),(['2022-7','2022-19'],0,False)]]
        rows=sum([f.result() for f in futures],[])
    assert len(rows)==12 and sum(r['adopted'] for r in rows)==2
    atomic(OUT/'RESULTS.json',rows)


if __name__=='__main__':main()
