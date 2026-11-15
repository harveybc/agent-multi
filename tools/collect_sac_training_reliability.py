"""Read-only fleet collection and paired endpoint report; never promote a model."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from tools.run_discrete_curve_probe import read, write, now


def fetch(entry):
    script = """import json,pathlib,hashlib
p=pathlib.Path(ROOT)
def load(n):
 f=p/n
 return json.loads(f.read_text()) if f.exists() else None
r=load('RESULT.json');s=load('SPEC.json');q=load('QUALIFICATION.json')
ok=False
if r and s and q:
 ok=r['identity']==s['identity']==q['identity'] and q['passed']
 for a in r['arms']:
  f=p/a['name']/'LATEST.pt'
  ok=ok and f.exists() and hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==a['checkpoint_sha256']
print(json.dumps(dict(result=r,spec=s,qualification=q,verified=ok,progress=load('PROGRESS.json'),failure=load('FAILURE.json'))))
""".replace('ROOT',repr(entry['root']))
    command=['python3','-c',script]
    if entry.get('host'):
        command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',entry['host'],shlex.join(command)]
    p=subprocess.run(command,capture_output=True,text=True,timeout=90,check=True)
    result=json.loads(p.stdout)
    if result['spec'] and result['spec']['seed']!=entry['seed']:
        raise ValueError('FLEET_SEED_IDENTITY')
    return result


def collect(fleet,output):
    if len(fleet['workers'])!=3 or {e['seed'] for e in fleet['workers']}!={7,19,43}:
        raise ValueError('FLEET_REQUIRES_THREE_DISTINCT_SEEDS')
    output.mkdir(parents=True,exist_ok=True);entries=[];pairs=[]
    for entry in fleet['workers']:
        try:r=fetch(entry)
        except Exception as exc:r=dict(verified=False,error=str(exc))
        write(output/(str(entry['seed'])+'.json'),r)
        entries.append(dict(seed=entry['seed'],verified=r.get('verified',False),
            progress=r.get('progress'),failure=r.get('failure'),error=r.get('error')))
        if not r.get('verified'):continue
        arms=r['result']['arms'];control=next(a for a in arms if a['name']=='control')
        for arm in arms:
            row=dict(seed=entry['seed'],arm=arm['name'],cold_exact=arm['cold_exact'])
            for period in ('TRAIN','VALIDATION'):
                m=arm['final_metrics'][period];c=control['final_metrics'][period]
                row[period]=dict(return_pct=m['return_pct'],delta_return_pct=m['return_pct']-c['return_pct'],
                    sharpe=m['sharpe'],trades=m['trades'],orders_per_week=m['orders_per_week'])
            pairs.append(row)
    complete=len(entries)==3 and all(e['verified'] for e in entries)
    write(output/'STATUS.json',dict(at=now(),complete=False,workers=entries,promoted=False,deap_released=False))
    if complete:
        specs=[read(output/(str(e['seed'])+'.json'))['spec'] for e in entries]
        for key in ('source','manifest','parent','code','design','effective','arms','updates_per_arm','trainability'):
            if any(s[key]!=specs[0][key] for s in specs[1:]):raise ValueError('FLEET_CONTRACT_MISMATCH:'+key)
        write(output/'RESULT.json',dict(status='THREE_SEEDS_COMPLETE',pairs=pairs,promoted=False,
            phase2_closed=False,deap_released=False,
            scope='paired fixed-replay endpoints, not proof of full online convergence'))
        lines=['# Full SAC Reliability: Paired Endpoints','',
            '| Seed | Arm | TRAIN return % | Validation return % | Validation delta vs control | Validation orders |',
            '|---|---|---:|---:|---:|---:|']
        for p in pairs:
            v=p['VALIDATION'];lines.append(f"|{p['seed']}|{p['arm']}|{p['TRAIN']['return_pct']:.6f}|{v['return_pct']:.6f}|{v['delta_return_pct']:.6f}|{v['trades']}|")
        lines+=['','No model promotion, phase2 closure or DEAP release. Compare changes to the preserved baseline as well as control; merely reducing drift is not learning.']
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
        write(output/'STATUS.json',dict(at=now(),complete=True,workers=entries,promoted=False,deap_released=False))
    return complete


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fleet',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        done=collect(read(a.fleet),a.output)
        if done or not a.watch:break
        time.sleep(180)
