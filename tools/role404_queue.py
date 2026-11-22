"""Durable single coordinator. No duplicate dispatch on SSH outages."""
from itertools import product
import os
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path('/var/tmp/role404')
RUNTIME='/var/tmp/sac399-series/runtime'
ROLLBACK=os.environ.get('ROLE_RUN')=='405'
if ROLLBACK:
    ROOT=Path('/var/tmp/role405')
    RUNTIME='/var/tmp/role405-runtime'
PY='/var/tmp/discrete-sac-probe-venv/bin/python'
SLOTS=[('gamma-lan',1),('dragon-lan',0),('gamma-lan',0),('omega',0)]
ROLES=['CLOSE_BUY','CLOSE_SELL','OPEN_BUY','OPEN_SELL']
ENV=['--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1',
     '--setenv=PYTHONUNBUFFERED=1','--setenv=PYTHONPATH=.:/var/tmp/selector-024-20261104',
     '--setenv=STRATEGY_ENGINE_ROOT=/var/tmp/sac70-warm-v3-20261108/engine']
if ROLLBACK:
    ENV=[v for v in ENV if not v.startswith('--setenv=PYTHONPATH=')]+[
        '--setenv=ROLE_RUN=405','--setenv=PYTHONPATH=.:/var/tmp/sac399-series/runtime:/var/tmp/selector-024-20261104']
def call(h,args):
    return subprocess.run(args if h=='omega' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',h,shlex.join(args)],
                          text=True,capture_output=True,timeout=90)
def persist(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=1));tmp.replace(p)
def collect(h,key):
    if h!='omega':
        d=ROOT/key;d.mkdir(exist_ok=True)
        subprocess.run(['rsync','-a','--exclude=resume.pt',f'{h}:{ROOT}/{key}/',str(d)+'/'],check=True,timeout=90)
    r=json.loads((ROOT/key/'RESULT.json').read_text())
    if not (ROOT/key/'expert.pt').exists() or not r['epochs']>=100:raise ValueError(key)
def run():
    jobs=[(m,r,y,s) for m in ('adv','cls') for s,y in product((7,19,43),(4,8)) for r in ROLES]
    path=ROOT/'QUEUE.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    while True:
        for host,index in SLOTS:
            try:
                if ROLLBACK:
                    control=json.loads(Path('/var/tmp/role404/QUEUE.json').read_text())
                    busy=False
                    for old in control.values():
                        if old['status']=='assigned' and (old['host'],old['index'])==(host,index):
                            active=call(host,['systemctl','--user','is-active',old['unit']])
                            if active.returncode not in (3,4):
                                busy=True;break
                    if busy:continue
                items=[(key,v) for key,v in state.items() if v['host']==host and v['index']==index and v['status']=='assigned']
                if items:
                    key,rec=items[0]
                    result=call(host,['test','-s',str(ROOT/key/'RESULT.json')])
                    if result.returncode==0:
                        collect(host,key);rec['status']='done';persist(path,state)
                    elif result.returncode!=1:continue
                    else:
                        check=call(host,['systemctl','--user','show',rec['unit'],'-p','ActiveState','--value'])
                        if check.stdout.strip() in ('active','activating','deactivating'):continue
                        if check.stdout.strip()=='failed':
                            rec['status']='failed';persist(path,state)
                            print('FAILED',key,flush=True);continue
                        if check.returncode not in (0,4) or check.stdout.strip() not in ('inactive',''):continue
                if items and items[0][1]['status']=='assigned':
                    key,rec=items[0];m,r,y,s=rec['args']
                else:
                    job=next((j for j in jobs if '-'.join(map(str,j)) not in state),None)
                    if job is None:continue
                    m,r,y,s=job;key='-'.join(map(str,job))
                    rec=dict(host=host,index=index,status='assigned',unit=ROOT.name+'-'+key.lower(),args=job)
                    state[key]=rec;persist(path,state)
                gpu=call(host,['nvidia-smi','-i',str(index),'--query-gpu=uuid','--format=csv,noheader'])
                if gpu.returncode:continue
                a=['systemd-run','--user','--unit='+rec['unit'],'--property=MemoryMax=3500M',
                   '--property=Nice=5','--working-directory='+RUNTIME,'--setenv=CUDA_VISIBLE_DEVICES='+gpu.stdout.strip(),
                   *ENV,PY,'-m','tools.role404_fit',r,str(y),str(s),m]
                p=call(host,a)
                print(key,host,index,p.returncode,p.stderr.strip(),flush=True)
            except (subprocess.SubprocessError,OSError) as ex:print(host,index,str(ex),flush=True)
        if len(state)==len(jobs) and all(v['status'] in ('done','failed') for v in state.values()):break
        time.sleep(10)
    if ROLLBACK:
        subprocess.run(['systemctl','--user','enable','--now','role404-coordinator.service'],check=True)
    if any(v['status']=='failed' for v in state.values()):raise RuntimeError('failed fits: see QUEUE.json')

if __name__=='__main__':
    with (ROOT/'coordinator.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
