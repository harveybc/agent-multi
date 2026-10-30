#!/usr/bin/env python3
"""Read-only fleet polling and authenticated collection; never starts experiments."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.zigzag_opportunities import sha
from tools.optimize_dynamic_account import digest,Receipts
from tools.run_crossed_account_queue import load
from tools.verify_absolute_early_close import write


def authenticate(root):
    manifest=json.loads((root/'MANIFEST.json').read_text());identity=manifest.pop('identity')
    if digest(manifest)!=identity: raise ValueError('COLLECTION_MANIFEST')
    result=load(root/'COMPLETE.json',identity)
    if result is None: raise ValueError('COLLECTION_INCOMPLETE')
    for name,expected in result['artifacts'].items():
        if Path(name).name!=name or sha(root/name)!=expected:
            raise ValueError('COLLECTION_ARTIFACT')
    chain=Receipts(root/'CANDIDATES.jsonl',identity)
    if chain.chain!=result['receipt_tail']: raise ValueError('COLLECTION_CHAIN')
    tests=load(root/'TESTS.json',identity)
    if tests is None or tests['skipped'] or tests['passed']<200: raise ValueError('COLLECTION_TESTS')
    return dict(identity=identity,mode=result['mode'],status=result['status'],
        candidates=result['candidates'],search_gate=result['search_gate'],
        confirmation=result['confirmation'],tests=tests,complete_sha256=sha(root/'COMPLETE.json'))


def collect(lane,out):
    root=Path(lane['root']);host=lane.get('ssh')
    if host:
        if host.startswith('-') or ':' in host: raise ValueError('COLLECTOR_HOST')
        process=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,
            'cat '+shlex.quote(str(root/'STATE.json'))],capture_output=True,text=True,timeout=20,check=True)
        state=json.loads(process.stdout)
    else:
        state=json.loads((root/'STATE.json').read_text())
    if state['status']!='COMPLETE': return state
    if host:
        dest=out/lane['name'];temp=out/(lane['name']+'.incoming')
        if not dest.exists():
            temp.mkdir(exist_ok=True)
            subprocess.run(['rsync','-a','--protect-args','-e','ssh -o BatchMode=yes -o ConnectTimeout=8',
                host+':'+str(root)+'/',str(temp)+'/'],check=True,timeout=300,capture_output=True)
            authenticate(temp);temp.replace(dest)
        root=dest
    return dict(status='COMPLETE',result=authenticate(root))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());out=a.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    names=[x['name'] for x in cfg['lanes']]
    if len(names)!=len(set(names)) or any(not n or Path(n).name!=n or n in ('.','..') for n in names):
        raise ValueError('COLLECTOR_NAMES')
    config_sha=sha(a.config)
    while True:
        if sha(a.config)!=config_sha: raise ValueError('COLLECTOR_CONFIG_CHANGED')
        states={}
        for lane in cfg['lanes']:
            try: states[lane['name']]=collect(lane,out)
            except (OSError,subprocess.SubprocessError,json.JSONDecodeError) as exc:
                states[lane['name']]=dict(status='WAIT_CONNECTION_OR_FILE',error_type=type(exc).__name__)
        write(out/'STATE.json',dict(heartbeat_utc=time.time(),lanes=states))
        if all(r['status'] in ('COMPLETE','FAILED') for r in states.values()):
            lines=['# Causal history: collected CPU lanes','',
                'Synthetic diagnostic evidence, not financial validation or final learned strategy qualification.','']
            for name,state in states.items():
                lines.extend(['## '+name,'',f"Status: {state['status']}.",''])
                if state['status']!='COMPLETE': continue
                r=state['result'];lines.extend([f"Identity: `{r['identity']}`.",
                    f"Tests: {r['tests']['passed']}; candidates: {r['candidates']}; search gate: {r['search_gate']['eligible']}.",'',
                    '| Policy | Return % | Worst DD % | Orders/week | Early closes | Harmful closes |',
                    '| --- | ---: | ---: | ---: | ---: | ---: |'])
                for policy,row in r['confirmation'].items():
                    lines.append(f"| {policy} | {row['objectives'][0]:.4f} | {-row['objectives'][1]:.4f} | "
                        f"{row['mean_orders']:.2f} | {row['early_closes']} | {row['close_metrics']['destructive']} |")
                lines.append('')
            (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
            write(out/'COMPLETE.json',dict(status='COLLECTED',config_sha256=config_sha,lanes=states,
                result_sha256=sha(out/'RESULTS.md'),automatic_promotion=False))
            return
        time.sleep(60)


if __name__=='__main__': main()
