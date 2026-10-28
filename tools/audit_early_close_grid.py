"""Read frozen synthetic batches and retain per-cell early-close capture evidence."""
import argparse
from collections import Counter
import fcntl
import gzip
import json
from pathlib import Path
import time

from app.early_close_audit import audit_episode
from app.frozen_recipe_inputs import file_sha, require
from app.rl_subset import digest
from tools.recipe_sac_queue import read, write, now

ROOT=Path(__file__).resolve().parents[1]


def reference_scope(identity):
    modes=set(identity.get('design',{}).get('axes',{}).get('mode',[]))
    full={'position_optimal_long','position_optimal_short'}
    if modes and modes<=full:
        return 'CAPTURE_UNDER_FULL_REMAINING_EXECUTABLE_POSITION_REFERENCE_NOT_LEARNING'
    if modes & full:
        return 'MIXED_POSITION_AND_FINITE_HORIZON_REFERENCES_NOT_LEARNING'
    return 'CAPTURE_UNDER_RETAINED_FINITE_HORIZON_REFERENCE_NOT_GLOBAL_OPTIMALITY'


def tick(source, output, max_batches=256):
    identity=read(source/'IDENTITY.json')
    pin=dict(source_identity=digest(identity),code={n:file_sha(ROOT/n) for n in
        ('app/early_close_audit.py','tools/audit_early_close_grid.py')})
    output.mkdir(parents=True,exist_ok=True)
    with (output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'IDENTITY.json').exists():
            require(read(output/'IDENTITY.json')==pin,'EARLY_AUDIT_IDENTITY')
        else:write(output/'IDENTITY.json',pin)
        counts=Counter();failures=[];audits={};new=0
        terminal=read(source/'CLOSE.json') if (source/'CLOSE.json').exists() else None
        for receipt in sorted((source/'batches').glob('*.json')):
            target=output/'batches'/receipt.name
            if target.exists():
                r=read(target)
                require(r['source_receipt_sha']==file_sha(receipt),'EARLY_AUDIT_SOURCE_CHANGED')
                require(r['payload_sha']==digest(r['payload']),'EARLY_AUDIT_CACHE_CHANGED')
                if terminal:
                    require(file_sha(receipt.with_suffix('.json.gz'))==read(receipt)['sha256'],
                            'EARLY_AUDIT_TERMINAL_DATA')
            else:
                if new>=max_batches:continue
                original=read(receipt);data=receipt.with_suffix('.json.gz')
                require(original['identity']==digest(identity) and file_sha(data)==original['sha256'],
                        'EARLY_AUDIT_SOURCE_HASH')
                batch=json.loads(gzip.decompress(data.read_bytes()))
                cases=[];summary=Counter()
                for row in batch['rows']:
                    require(row['state']=='PASS','EARLY_AUDIT_SOURCE_FAILURE')
                    audit=audit_episode(row['result']);summary.update(audit['counts'])
                    cases.append(dict(case_id=row['case_id'],**audit))
                require(len(cases)==original['count'],'EARLY_AUDIT_BATCH_COUNT')
                payload=dict(cases=cases,counts=dict(summary))
                r=dict(source_receipt_sha=file_sha(receipt),payload_sha=digest(payload),payload=payload)
                write(target,r);new+=1
            audits[receipt.name]=file_sha(target)
            counts.update(r['payload']['counts'])
            for case in r['payload']['cases']:
                if case['failures']:failures.append(dict(case_id=case['case_id'],failures=case['failures']))
        complete=bool(terminal and counts['cells']==terminal['completed'])
        state='FAILED' if failures else 'COMPLETE' if complete else 'FOLLOWING_SOURCE'
        report=dict(state=state,counts=dict(counts),failure_count=len(failures),failures=failures[:20],
            capture_rate=counts['captured']/counts['opportunities'] if counts['opportunities'] else None,
            early_close_trade_share=counts['early_closes']/counts['trades'] if counts['trades'] else None,
            scope=reference_scope(identity),
            target_early_close_share=None,financial_admission=False,learning_qualified=False,at=now())
        write(output/'STATUS.json',report)
        require(not failures,'EARLY_CLOSE_CAPTURE_FAILED_SEE_STATUS')
        if complete:
            require(terminal['identity']==digest(identity) and terminal['failures']==0,'EARLY_AUDIT_SOURCE_CLOSE')
            require(set(terminal['receipts'])==set(audits),'EARLY_AUDIT_RECEIPT_SET')
            for name,sha in terminal['receipts'].items():
                require(file_sha(source/'batches'/name)==sha,'EARLY_AUDIT_CLOSE_HASH')
            if not (output/'CLOSE.json').exists():
                write(output/'CLOSE.json',dict(**report,audit_receipts=audits,
                    two_sided_close_witness=all(counts[s+'.captured']>0 for s in ('long','short'))))
        return complete


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        if a.watch and not (a.source/'IDENTITY.json').exists():
            time.sleep(30)
            continue
        done=tick(a.source,a.output)
        if done or not a.watch:break
        time.sleep(30)
