"""Stage immutable pilot inputs, launch capped workers, then collect paired curves."""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def call(host,args):
    cmd=(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,shlex.join(args)]
         if host!='local' else args)
    return subprocess.check_output(cmd,text=True).strip()


def write(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)


def run(a):
    root=a.root
    slots=[('local',0,61,0,6),(a.secondary,0,61,10,6),
           (a.dual,0,71,0,4),(a.dual,1,71,10,4)]
    # Use UUIDs: CUDA enumeration need not match the ordering in nvidia-smi.
    for host in set(s[0] for s in slots):
        subset=[s for s in slots if s[0]==host]
        free=int(next(line.split()[1] for line in call(host,['cat','/proc/meminfo']).splitlines()
                      if line.startswith('MemAvailable:')))/1048576
        if free < sum(s[-1] for s in subset)+2:raise ValueError(('RAM admission',host,free))
        for _,gpu,*_ in subset:
            v=int(call(host,['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']))
            if v<3072:raise ValueError(('VRAM admission',host,gpu,v))
        if host!='local':
            call(host,['mkdir','-p',str(root/'runtime')])
            subprocess.run(['rsync','-a',str(root/'runtime')+'/',host+':'+str(root/'runtime')+'/'],check=True)
            for w in set(s[2] for s in subset):
                subprocess.run(['rsync','-a','--exclude=real/','--exclude=mix/',str(root/str(w)),host+':'+str(root)+'/'],check=True)
    jobs=[]
    for host,gpu,width,dose,cap in slots:
        tag='mix' if dose else 'real'
        folder=root/str(width)/tag
        uid=call(host,['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'])
        unit=f'pilot391-{width}-{tag}'
        call(host,['systemd-run','--user','--unit='+unit,'--property=MemoryMax='+str(cap)+'G',
             '--property=WorkingDirectory='+str(root/'runtime'),'/usr/bin/env','CUDA_VISIBLE_DEVICES='+uid,
             'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONUNBUFFERED=1',
             'PYTHONPATH=.:/var/tmp/selector-024-20261104','CUBLAS_WORKSPACE_CONFIG=:4096:8',
             'STRATEGY_ENGINE_ROOT='+str(a.parent/'engine'),str(a.python),'-m','tools.pilot391','train',
             '--width',str(width),'--dose',str(dose),'--root',str(root),'--parent',str(a.parent)])
        jobs.append((host,unit,folder))
    while True:
        closed=True;state={}
        for host,unit,folder in jobs:
            status=call(host,['systemctl','--user','show',unit+'.service','-p','ActiveState','--value'])
            if host!='local':
                folder.mkdir(exist_ok=True)
                subprocess.run(['rsync','-a','--include=/*.json','--exclude=*',
                    host+':'+str(folder)+'/',str(folder)+'/'],check=False,stderr=subprocess.DEVNULL)
            result=folder/'RESULT.json'
            state[unit]=dict(status=status,completed=result.exists())
            if status=='failed' or (status=='inactive' and not result.exists()):
                state[unit]['error']=call(host,['journalctl','--user','-u',unit,'-n','8','-o','cat'])
            elif not result.exists():closed=False
        write(root/'FLEET.json',state)
        if closed:break
        time.sleep(30)
    if not all(v['completed'] for v in state.values()):raise RuntimeError('worker failed, see FLEET')
    comparison={}
    for w in (61,71):
        d=root/str(w)
        contracts=[json.loads((d/t/'CONTRACT.json').read_text()) for t in ('real','mix')]
        if any(contracts[0][k]!=contracts[1][k] for k in ('start','support','seed','epochs')):raise ValueError('unpaired')
        histories=[json.loads((d/t/'HISTORY.json').read_text()) for t in ('real','mix')]
        if not all([r['epoch'] for r in hist]==[1,2,3,4] for hist in histories):raise ValueError('epochs')
        comparison[str(w)]=[dict(epoch=a['epoch'],real=a['metrics'],mix=b['metrics'],
            real_updates=a['updates'],mix_updates=b['updates'],
            validation_delta_pp=b['metrics']['VALIDATION']['return_pct']-a['metrics']['VALIDATION']['return_pct'])
            for a,b in zip(*histories,strict=True)]
    write(root/'COMPARISON.json',comparison)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','python','parent'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('secondary','dual'):p.add_argument('--'+k,required=True)
    run(p.parse_args())
