"""Read-only fleet observer for sealed paired plateau continuations."""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import time

from tools.run_discrete_curve_probe import read, write, now


REMOTE = '''import json,pathlib
r=pathlib.Path(ROOT)
def load(p):return json.loads(p.read_text()) if p.exists() else None
s=load(r/'SPEC.json');arms=[]
for n in ('plateau','fixed'):
 p=r/n;h=load(p/'HISTORY.json') or [];src=0 if not s else s['source_epoch']
 recent=[x for x in h if x['epoch']>src]
 arms.append(dict(arm=n,last=h[-1] if h else None,additional_epochs=len(recent),
   seconds=[x['epoch_seconds'] for x in recent[-5:] if 'epoch_seconds' in x],
   progress=load(p/'PROGRESS.json'),result=load(p/'RESULT.json')))
q=load(r/'QUALIFICATION.json')
print(json.dumps(dict(spec=s,arms=arms,qualification=None if q is None else
 {k:q[k] for k in ('native_exact','next_four_sampled_updates_exact','source_seal_sha256')},
 progress=load(r/'PROGRESS.json'),result=load(r/'RESULT.json'),failure=load(r/'FAILURE.json'))))
'''


def collect(fleet, output):
    output.mkdir(parents=True,exist_ok=True)
    rows=[];workers=[]
    for entry in fleet['workers']:
        command=['python3','-c',REMOTE.replace('ROOT',repr(entry['root']))]
        if entry.get('host'):
            command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',entry['host'],shlex.join(command)]
        try:
            proc=subprocess.run(command,check=True,capture_output=True,text=True,timeout=60)
            value=json.loads(proc.stdout)
            if value['spec'] and value['spec']['seed']!=entry['seed']:raise ValueError('FLEET_SEED')
        except Exception as exc:value=dict(error=str(exc))
        workers.append(dict(seed=entry['seed'],**value))
        for arm in value.get('arms',[]):
            last=arm['last'];metrics=None if last is None else last['metrics']
            rates=None if last is None else last.get('actor_plateau',{}).get('after')
            rate='n/a' if rates is None else f'{rates[0]:.3g}'
            duration=statistics.median(arm['seconds'])/60 if arm['seconds'] else None
            numbers=['n/a','n/a'] if metrics is None else [f'{metrics[p]["return_pct"]:.6f}' for p in ('TRAIN','VALIDATION')]
            rows.append(f'|{entry["seed"]}|{arm["arm"]}|{arm["additional_epochs"]}|{None if last is None else last["epoch"]}|{numbers[0]}|{numbers[1]}|{rate}|{duration}|')
    complete=all(w.get('result',{}).get('status')=='MATCHED_FORK_COMPLETE' if w.get('result') else False for w in workers)
    write(output/'STATUS.json',dict(at=now(),complete=complete,workers=workers,promoted=False))
    (output/'REPORT.md').write_text('\n'.join(['# Sealed Plateau Continuation','',
        '| Seed | Arm | New epochs | Total epoch | Last TRAIN % | Last VAL % | Actor LR | Minutes/new epoch |',
        '|---|---|---:|---:|---:|---:|---:|---:|',*rows,'',
        'Source metrics shown at zero new epochs are not new results. Scheduler first, then equal-budget fixed control. No automatic promotion.'])+'\n')
    return complete


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fleet',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        if collect(read(a.fleet),a.output) or not a.watch:break
        time.sleep(180)
