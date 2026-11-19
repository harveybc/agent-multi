"""Coordinate admitted native SAC workers; journal the owner before dispatch."""
import argparse
import fcntl
import json
import shlex
import subprocess
import time
from pathlib import Path

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def command(host,line):
    args=['bash','-c',line] if host=='local' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,line]
    return subprocess.check_output(args,text=True,timeout=35).strip()

def inspect(host,unit):
    lines=command(host,f'systemctl --user show {unit} -p ActiveState -p Result -p MemoryCurrent -p LoadState')
    return dict(r.split('=',1) for r in lines.splitlines())

def run(c):
    root=Path(c['root']);state=root/'OWNERS.json';out=root/'QUEUE.json'
    lock=(root/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owners=json.loads(state.read_text()) if state.exists() else {}
    jobs=[(y,s) for s in (7,19) for y in (0,4,8,16)]
    key=lambda y,s:f'{y}-{s}'
    while True:
        done={};active=[];notes=[];stats={};progress={}
        for name,slot in c['slots'].items():
            host=slot['host']
            if host in stats:continue
            try:
                info=command(host,"awk '/MemAvailable:/{print $2}' /proc/meminfo; cat /proc/sys/kernel/random/boot_id; df -Pk "+str(root)+" | tail -1 | awk '{print $4}'").splitlines()
                stats[host]=dict(mib=int(info[0])//1024,boot=info[1],disk=int(info[2])//1024)
            except (OSError,subprocess.SubprocessError) as e:notes.append([host,'unreachable',str(e)])
        for y,s in jobs:
            k=key(y,s)
            if k not in owners:continue
            rec=owners[k];name=rec['slot'];host=c['slots'][name]['host']
            if host not in stats:
                active.append((name,0));continue
            result=f'{root}/results/{y}/{s}/RESULT.json'
            if command(host,'test -s '+result+' && echo yes || echo no')=='yes':
                done[k]=json.loads(command(host,'cat '+result));continue
            unit=f'aug389-{k}.service';u=inspect(host,unit)
            if u['ActiveState'] in ('active','activating','deactivating'):
                cur=u['MemoryCurrent'];active.append((name,int(cur)//2**20 if cur.isdigit() else 0))
                path=f'{root}/results/{y}/{s}/PROGRESS.json'
                raw=command(host,'if test -s '+path+'; then cat '+path+'; fi')
                if raw:
                    snap=json.loads(raw)
                    progress[k]={f:snap.get(f) for f in ('stage','epoch','steps','updates','epoch_seconds','bad','selected_epoch')}
            elif u['LoadState']=='not-found' and (rec['boot']!=stats[host]['boot'] or not rec['launched']):
                rec['launched']=False
            else:notes.append([k,'halted on worker',u])
        for y,s in jobs:
            k=key(y,s)
            if k in done or (k in owners and owners[k]['launched']):continue
            if s==19 and any(key(yy,7) not in owners for yy in (0,4,8,16)):continue
            order=c['order'][str(y)] if k not in owners else [owners[k]['slot']]
            for name in order:
                slot=c['slots'][name];host=slot['host'];cap=slot['cap']
                if host not in stats or any(n==name for n,_ in active):continue
                # Reserve the unused portion of other 389 workers' hard limits.
                reserved=sum(max(0,c['slots'][n]['cap']*1024-used) for n,used in active if c['slots'][n]['host']==host)
                if stats[host]['mib']-reserved<cap*1024+2048 or stats[host]['disk']<8192:continue
                unit=f'aug389-{k}'
                owners[k]=dict(slot=name,boot=stats[host]['boot'],launched=False);save(state,owners)
                cmd=['systemd-run','--user','--unit='+unit,'--working-directory='+str(root/'runtime'),
                    '-p','MemoryAccounting=yes','-p',f'MemoryHigh={cap-1}G','-p',f'MemoryMax={cap}G',
                    '-p','MemorySwapMax=256M','-p','OOMPolicy=stop','-p','Nice=5','/usr/bin/env',
                    'CUDA_VISIBLE_DEVICES='+str(slot['gpu']),*c['env'],c['python'],'-u','-m','tools.augmentation389','train',
                    *c['args'],'--years',str(y),'--seed',str(s)]
                command(host,shlex.join(cmd))
                owners[k]['launched']=True;save(state,owners);active.append((name,0));break
        save(out,dict(time=time.time(),owners=owners,completed=list(done),active_slots=active,progress=progress,
                      pending=[key(y,s) for y,s in jobs if key(y,s) not in owners],hosts=stats,notes=notes))
        if len(done)==len(jobs):
            budgets={(v['updates'],v['epochs']) for v in done.values()}
            if len(budgets)!=1:raise ValueError('unmatched training budgets')
            save(root/'COMPARISON.json',done);return
        time.sleep(45)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config',type=Path)
    run(json.loads(p.parse_args().config.read_text()))
