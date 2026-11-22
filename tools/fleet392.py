"""Stage only one four-year path per width. Reuse the unmodified real controls."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from tools.fleet391 import call,write


def run(a):
    r=a.root;p=a.pilot
    workers=[(a.secondary,61,0,6,'price7'),(a.dual,71,1,4,'context7')]
    for h,w,g,cap,path in workers:
        available=int(next(x.split()[1] for x in call(h,['cat','/proc/meminfo']).splitlines() if x.startswith('MemAvailable:')))/1048576
        free=int(call(h,['nvidia-smi','-i',str(g),'--query-gpu=memory.free','--format=csv,noheader,nounits']))
        if available<cap+2 or free<3072:raise ValueError(('admission',w,available,free))
        call(h,['mkdir','-p',str(r/'runtime'),str(r/'data')])
        for src,dst in [(str(a.runtime)+'/',r/'runtime'),(str(a.banks/'COHORTS.json'),r/'data'),(str(a.banks/path),r/'data')]:
            subprocess.run(['rsync','-a',src,h+':'+str(dst)+'/'],check=True)
        uid=call(h,['nvidia-smi','-i',str(g),'--query-gpu=uuid','--format=csv,noheader'])
        state=call(h,['systemctl','--user','show',f'full392-{w}.service','-p','ActiveState','--value'])
        if state=='active':continue
        call(h,['systemd-run','--user',f'--unit=full392-{w}',f'--property=MemoryMax={cap}G',
            '--property=WorkingDirectory='+str(r/'runtime'),'/usr/bin/env','CUDA_VISIBLE_DEVICES='+uid,
            'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONUNBUFFERED=1',
            'CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH=.:/var/tmp/selector-024-20261104',
            'STRATEGY_ENGINE_ROOT='+str(a.parent/'engine'),str(a.python),'-m','tools.full392','--width',
            str(w),'--root',str(r),'--data',str(r/'data'),'--pilot',str(p),'--parent',str(a.parent)])
    jobs=[('local',p/'61/real','pilot391-61-real'),(a.dual,p/'71/real','pilot391-71-real'),
          *[(h,r/str(w),f'full392-{w}') for h,w,*_ in workers]]
    read=lambda f:json.loads(f.read_text())
    while True:
        summary={}
        for h,d,u in jobs:
            if h!='local':
                d.mkdir(parents=True,exist_ok=True)
                subprocess.run(['rsync','-a','--include=/*.json','--exclude=*',h+':'+str(d)+'/',str(d)+'/'],check=False,stderr=subprocess.DEVNULL)
            state=call(h,['systemctl','--user','show',u+'.service','-p','ActiveState','--value'])
            summary[u]=dict(state=state,complete=(d/'RESULT.json').exists())
            if (d/'PROGRESS.json').exists():
                progress=read(d/'PROGRESS.json')
                summary[u]['progress']={k:progress[k] for k in ('stage','epoch','steps','updates','collection') if k in progress}
        write(r/'FLEET.json',summary)
        if all(s['complete'] for s in summary.values()):break
        if any(s['state'] in ('failed','inactive') and not s['complete'] for s in summary.values()):
            raise RuntimeError('worker stopped without result')
        time.sleep(30)
    results={}
    for w in ('61','71'):
        ctrl,aug=p/w/'real',r/w
        c,s=[read(d/'CONTRACT.json') for d in (ctrl,aug)]
        if any(c[k]!=s[k] for k in ('start','support','seed','epochs')):raise ValueError('unpaired')
        records=[]
        for x,y in zip(read(ctrl/'HISTORY.json'),read(aug/'HISTORY.json'),strict=True):
            cv=y['collection'];rp=y['replay_mixture']
            if (x['epoch']!=y['epoch'] or cv['real']!=cv['expected_real'] or cv['synthetic']!=cv['expected_synthetic']
                or rp['real_samples']!=rp['synthetic_samples'] or not rp['real_samples']):raise ValueError('coverage')
            records.append(dict(epoch=y['epoch'],real=x['metrics'],four_plus_four=y['metrics'],
                real_updates=x['updates'],aug_updates=y['updates'],coverage=cv,replay=rp))
        if len(records)!=4:raise ValueError('epochs')
        results[w]=records
    write(r/'COMPARISON.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','pilot','runtime','banks','parent','python'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('secondary','dual'):p.add_argument('--'+k,required=True)
    run(p.parse_args())
