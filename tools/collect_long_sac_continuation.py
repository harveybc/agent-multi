"""Read-only online continuation status, completion authentication and paired reports."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from tools.run_discrete_curve_probe import read, write, now


REMOTE = '''import json,pathlib,hashlib,statistics
root=pathlib.Path(ROOT)
def load(p):return json.loads(p.read_text()) if p.exists() else None
spec=load(root/'SPEC.json');result=load(root/'RESULT.json');arms=[];verified=False
for name in ('control','actor_lr01','all_lr01'):
 p=root/name;r=load(p/'RESULT.json');history=load(p/'HISTORY.json') or []
 best=load(p/'BEST.json') or {};zero=load(p/'EPOCH_ZERO.json')
 last=history[-1] if history else None
 durations=[h['epoch_seconds'] for h in history[-5:] if 'epoch_seconds' in h]
 seconds=statistics.median(durations) if durations else None
 epochs=last['epoch'] if last else 0
 remaining=None if seconds is None else max(0,100-epochs)*seconds
 ok=bool(r and spec and r['parent_identity']==spec['identity'] and r['cold_exact'] and r['seed']==spec['seed'] and r['arm']==name)
 if ok:
  for n,digest in r['files'].items():
   f=(p/n).resolve()
   ok=ok and f.is_relative_to(p.resolve()) and f.is_file() and hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==digest
 selected=(best.get('best') or {}).get('path')
 arms.append(dict(arm=name,result=r,verified=ok,progress=load(p/'PROGRESS.json'),epochs=epochs,
  last_metrics=None if last is None else last['metrics'],monitor=None if last is None else last['monitor'],
  selected_epoch=selected,baseline=None if zero is None else zero['metrics'],
  seconds_per_epoch=seconds,seconds_to_minimum_budget=remaining))
if result and spec:verified=result['identity']==spec['identity'] and all(a['verified'] for a in arms) and {a['arm']:a for a in result['arms']}=={a['arm']:a['result'] for a in arms}
print(json.dumps(dict(spec=spec,result=result,verified=verified,arms=arms,progress=load(root/'PROGRESS.json'),failure=load(root/'FAILURE.json'))))
'''


def fetch(entry):
    command=['python3','-c',REMOTE.replace('ROOT',repr(entry['root']))]
    if entry.get('host'):
        command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',entry['host'],shlex.join(command)]
    p=subprocess.run(command,capture_output=True,text=True,timeout=90,check=True)
    r=json.loads(p.stdout)
    if r['spec'] and r['spec']['seed']!=entry['seed']:
        raise ValueError('LONG_FLEET_SEED')
    return r


def collect(fleet,output):
    if len(fleet['workers'])!=3 or {e['seed'] for e in fleet['workers']}!={7,19,43}:
        raise ValueError('LONG_FLEET_MEMBERS')
    output.mkdir(parents=True,exist_ok=True)
    workers=[];specs=[];results=[]
    for entry in fleet['workers']:
        try:r=fetch(entry)
        except Exception as exc:r=dict(verified=False,error=str(exc))
        write(output/(str(entry['seed'])+'.json'),r)
        workers.append(dict(seed=entry['seed'],**r))
        if r.get('spec'):specs.append(r['spec'])
        if r.get('verified'):results.extend(r['result']['arms'])
    for key in ('source','manifest','parent','code','design','effective','arms'):
        if any(s[key]!=specs[0][key] for s in specs[1:]):
            raise ValueError('LONG_FLEET_CONTRACT:'+key)
    complete=all(w['verified'] for w in workers)
    write(output/'STATUS.json',dict(at=now(),complete=complete,workers=workers,promoted=False))
    lines=['# Long Online SAC Continuation','',
        '| Seed | Arm | Completed epochs | Selected epoch | Last TRAIN % | Last VAL % | Bad / limit | Minutes/epoch |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for worker in workers:
        for a in worker.get('arms',[]):
            m=a['last_metrics'];bad=(a['monitor'] or {}).get('bad',0)
            values=('n/a','n/a') if not m else tuple(f"{m[p]['return_pct']:.6f}" for p in ('TRAIN','VALIDATION'))
            minutes='unmeasured' if a['seconds_per_epoch'] is None else f"{a['seconds_per_epoch']/60:.2f}"
            lines.append(f"|{worker['seed']}|{a['arm']}|{a['epochs']}/2000|{a['selected_epoch']}|{values[0]}|{values[1]}|{bad}/20|{minutes}|")
    lines+=['','Patience is observed throughout but cannot stop before epoch100. The selected checkpoint may remain epoch0. '
        'ETA to minimum is not ETA to convergence or queue completion. Within-seed contrasts are primary; seeds run on different devices.',
        'No promotion, phase2 closure, live-trading deployment or DEAP release is automated.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if complete:
        write(output/'RESULT.json',dict(status='NINE_LONG_RUNS_COMPLETE',arms=results,promoted=False,
            phase2_closed=False,deap_released=False))
    return complete


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fleet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    while True:
        if collect(read(a.fleet),a.output) or not a.watch:break
        time.sleep(180)
