"""Durable, memory-admitted column dispatch over owner-configured CPU hosts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_receipt(folder,identity):
    folder=Path(folder).resolve();receipt=read(folder/'DONE.json')
    if receipt['identity']!=identity:raise ValueError('CPU_POOL_RECEIPT_IDENTITY')
    for name,digest in receipt['files'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder) or sha(p)!=digest:raise ValueError('CPU_POOL_RECEIPT_BYTES:'+name)


def discover_owners(observations,completed,assignments):
    result=dict(assignments)
    for name,observation in observations.items():
        for j in observation['started']:
            if j in completed:continue
            if str(j) in result and result[str(j)]!=name:raise ValueError('CPU_POOL_AMBIGUOUS_OWNER')
            result[str(j)]=name
    return result


def admit(total,completed,assignments,hosts):
    assigned=dict(assignments);new={}
    for j in range(total):
        if j in completed or str(j) in assigned:continue
        eligible=[]
        for name,h in hosts.items():
            if h.get('unavailable',False):continue
            active=[k for k,v in assigned.items() if v==name and int(k) not in completed]
            reserved=sum(max(0,h['reservation']-h['memory'].get(k,0)) for k in active)
            if len(active)<h['cap'] and h['available']-h['reserve']-reserved>=h['reservation']:
                eligible.append((len(active)/h['cap'],name))
        if not eligible:break
        name=min(eligible)[1];assigned[str(j)]=name;new[j]=name
    return new


def unit(j):return f'am-stl69-column@{j}.service'


def inspect(root,total,identity,old_units):
    root=Path(root);done=[];started=[];busy=[];progress={}
    for j in range(total):
        folder=root/'columns'/f'{j:03d}'
        if (folder/'DONE.json').exists():verify_receipt(folder,identity);done.append(j);continue
        if (folder/'PARTIAL.npz').exists() or (folder/'worker.lock').exists():started.append(j)
        if (folder/'PROGRESS.json').exists():progress[str(j)]=read(folder/'PROGRESS.json')
        if (folder/'worker.lock').exists():
            with (folder/'worker.lock').open('r') as f:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:busy.append(j)
    properties='Id,ActiveState,SubState,MemoryCurrent,NRestarts,Result'
    output=subprocess.run(['systemctl','--user','show',*map(unit,range(total)),*old_units,
        '-p',properties],capture_output=True,text=True,check=True).stdout
    services={}
    for block in output.strip().split('\n\n'):
        row=dict(line.split('=',1) for line in block.splitlines() if '=' in line)
        if 'Id' in row:services[row['Id']]=row
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return dict(done=done,started=started,busy=busy,progress=progress,services=services,
        available=int(mem['MemAvailable'].split()[0])*1024,code_sha256=sha(__file__),
        old_running=[n for n in old_units if services.get(n,{}).get('SubState') in ('start','running','auto-restart')])


def command(host,args):
    cmd=args if host.get('ssh') is None else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host['ssh'],shlex.join(args)]
    return subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=90).stdout


def run(config):
    cfg=read(config);root=Path(cfg['root']);operations=root/'operations';operations.mkdir(exist_ok=True)
    with (operations/'DYNAMIC_POOL.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=operations/'DYNAMIC_ASSIGNMENTS.json'
        state=read(ledger) if ledger.exists() else dict(identity=cfg['identity'],assignments={},created=time.time())
        if state['identity']!=cfg['identity']:raise ValueError('CPU_POOL_IDENTITY')
        while True:
            observations={};errors={}
            for name,h in cfg['hosts'].items():
                try:
                    value=command(h,[cfg['python'],cfg['tool'],'inspect','--root',cfg['root'],
                        '--total',str(cfg['total']),'--identity',cfg['identity'],'--old-units',*h['old_units']])
                    observations[name]=json.loads(value)
                    if observations[name]['code_sha256']!=sha(__file__):raise ValueError('CPU_POOL_RUNTIME')
                    if observations[name]['old_running']:raise ValueError('CPU_POOL_OLD_QUEUE_ACTIVE')
                except (subprocess.SubprocessError,OSError) as exc:errors[name]=repr(exc)
            completed=set().union(*(set(v['done']) for v in observations.values())) if observations else set()
            # Never expire an owner during a network outage: its process may still be running.
            state['assignments']=discover_owners(observations,completed,state['assignments'])
            resources={}
            for name,v in observations.items():
                h=cfg['hosts'][name];memory={}
                for j in range(cfg['total']):
                    s=v['services'].get(unit(j),{});m=s.get('MemoryCurrent','0')
                    memory[str(j)]=int(m) if m.isdigit() else 0
                resources[name]=dict(cap=h['cap'],reserve=h['reserve_bytes'],reservation=cfg['slot_bytes'],
                    available=v['available'],memory=memory)
            if not errors:
                additions=admit(cfg['total'],completed,state['assignments'],resources)
                state['assignments'].update({str(j):name for j,name in additions.items()})
            write(ledger,state)
            failures={}
            for key,name in state['assignments'].items():
                j=int(key)
                if j in completed or name not in observations:continue
                v=observations[name];s=v['services'].get(unit(j),{})
                if s.get('ActiveState')=='failed':failures[key]=s;continue
                if j in v['busy'] or s.get('ActiveState') in ('active','activating'):continue
                try:command(cfg['hosts'][name],['systemctl','--user','enable','--now','--no-block',unit(j)])
                except (subprocess.SubprocessError,OSError) as exc:errors[name]=repr(exc)
            write(operations/'DYNAMIC_STATUS.json',dict(at=time.time(),identity=cfg['identity'],
                completed=sorted(completed),assignments=state['assignments'],hosts=observations,
                errors=errors,failed_columns=failures))
            if len(completed)==cfg['total']:
                write(operations/'DYNAMIC_DONE.json',dict(at=time.time(),identity=cfg['identity'],columns=len(completed)));return
            time.sleep(cfg.get('poll_seconds',20))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','inspect'))
    p.add_argument('--config');p.add_argument('--root');p.add_argument('--total',type=int)
    p.add_argument('--identity');p.add_argument('--old-units',nargs='*',default=[]);a=p.parse_args()
    if a.mode=='run':run(a.config)
    else:print(json.dumps(inspect(a.root,a.total,a.identity,a.old_units)))


if __name__=='__main__':main()
