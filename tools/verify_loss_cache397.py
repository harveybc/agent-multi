"""Independent fail-closed receipt for an epoch of the cache experiment."""
import argparse
import json
import time
from pathlib import Path


FIELDS = ('epoch','steps','decisions','updates','epoch_updates','mean_update','last_update','metrics')


def compare(before, after, gate, timing):
    mismatches=[]
    def walk(path,a,b):
        if isinstance(a,dict) and isinstance(b,dict):
            for k in sorted(a.keys() | b.keys()):
                if k not in a or k not in b: mismatches.append(path+'.'+k)
                else: walk(path+'.'+k,a[k],b[k])
        elif a!=b: mismatches.append(path)
    for k in FIELDS: walk(k,before[k],after[k])
    first_epoch_seconds=after['epoch_seconds']+gate['setup_seconds']
    return dict(exact_metrics_losses_and_budget=not mismatches,differences=mismatches,
                control_seconds=before['epoch_seconds'],epoch_seconds=after['epoch_seconds'],
                cache_setup_seconds=gate['setup_seconds'],epoch_and_cache_setup_seconds=first_epoch_seconds,
                speedup_with_setup=before['epoch_seconds']/first_epoch_seconds,
                speedup_epoch_only=timing['speedup'],paired_updates=gate['paired_updates'],
                pass_gate=not mismatches and not any(gate['max_deltas'].values()) and
                          first_epoch_seconds < before['epoch_seconds'],promote_model=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('control',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--wait',type=int,default=0)
    a=parser.parse_args()
    deadline=time.monotonic()+a.wait
    end=a.candidate/'COMPARISON.json'
    while not end.exists() and time.monotonic()<deadline: time.sleep(5)
    read=lambda p:json.loads(p.read_text())
    result=compare(read(a.control/'HISTORY.json')[0],read(a.candidate/'HISTORY.json')[0],
                   read(a.candidate/'GATE.json'),read(end))
    temp=a.candidate/'EPOCH_PARITY.tmp'
    temp.write_text(json.dumps(result,indent=2))
    temp.replace(a.candidate/'EPOCH_PARITY.json')
    return 0 if result['pass_gate'] else 1


if __name__=='__main__':
    raise SystemExit(main())
