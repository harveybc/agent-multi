#!/usr/bin/env python3
"""Sequential five-arm native controller verification, no training or DEAP."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.absolute_exit_corpus import specs
from app.close_hold_verification import arms, run_case, prefix_future_witness
from app.early_close_metrics import summarize_exits
from app.zigzag_opportunities import engine, sha
from tools.verify_absolute_early_close import source_pins, write


def pins(engine_root):
    p=source_pins(engine_root)
    for name in ('app/close_hold_controller.py','app/close_hold_reports.py',
                 'app/close_hold_verification.py','tools/verify_close_hold_controller.py'):
        p['agent_multi'][name]=sha(ROOT/name)
    return p


def summarize(records):
    groups=defaultdict(list)
    for r in records:
        s=r['spec']
        groups[(r['arm'],'ALL','ALL','ALL')].append(r['metrics'])
        groups[(r['arm'],s['horizon'],s['direction'],s['mode'])].append(r['metrics'])
    return [dict(arm=k[0],horizon=k[1],direction=k[2],mode=k[3],**summarize_exits(v))
            for k,v in groups.items()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('NEW_OUTPUT_REQUIRED')
    root=ROOT/'docs/audits/evidence/absolute_early_close_20261029'
    previous=json.loads((root/'REPORT.json').read_text())
    manifest=json.loads((root/'MANIFEST.json').read_text())
    config_path=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    if (sha(config_path)!=manifest['config_sha256']
            or sha(root/'CASES.json.gz')!=previous['cases_sha256']):
        raise ValueError('REFERENCE_IDENTITY')
    config=json.loads(config_path.read_text())
    refs=json.loads(gzip.decompress((root/'CASES.json.gz').read_bytes()))
    if [r['spec'] for r in refs]!=list(specs()): raise ValueError('EXACT_PAIRED_SUPPORT')
    engine_root=a.engine_root.resolve()
    engine(engine_root)
    from app.short_horizon.expert_guided import absolute
    if Path(absolute.__file__).resolve()!=engine_root/'app/short_horizon/expert_guided/absolute.py':
        raise ValueError('ENGINE_IMPORT')
    code=pins(engine_root)
    for repo,files in manifest['sources'].items():
        if any(code[repo].get(k)!=v for k,v in files.items()):
            raise ValueError('REFERENCE_EXECUTION_CHANGED')
    identity=dict(schema='close_hold_controller_verification.v1',sources=code,arms=arms(),
                  reference_identity=previous['identity'],reference_cases_sha256=previous['cases_sha256'],
                  noisy_ideal_innovation_std_price=.0001,config_sha256=sha(config_path))
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    a.output.mkdir(parents=True)
    write(a.output/'MANIFEST.json',identity|dict(identity=digest))
    records=[]; witnesses=[]; start=time.monotonic()
    try:
        for i,r in enumerate(refs):
            if i%72==0:
                witnesses.append(prefix_future_witness(r['spec'],config))
            for arm in arms():
                records.append(run_case(r['spec'],config,r,arm))
            if (i+1)%64==0:
                state=dict(status='RUNNING',positions=i+1,total=len(refs),rollouts=len(records),
                           elapsed_seconds=time.monotonic()-start,identity=digest)
                write(a.output/'STATE.json',state); print(json.dumps(state),flush=True)
        if code!=pins(engine_root) or sha(config_path)!=identity['config_sha256']:
            raise ValueError('SOURCE_MUTATION')
        raw=(json.dumps(records,sort_keys=True,allow_nan=False)+'\n').encode()
        (a.output/'CASES.json.gz').write_bytes(gzip.compress(raw,mtime=0))
        report=dict(status='PASS_IDEAL_COMPONENT_AND_COMPLETED_DIAGNOSTICS',identity=digest,
            positions=len(refs),native_rollouts=len(records),prefix_invariance_witnesses=witnesses,
            rows=summarize(records),cases_sha256=sha(a.output/'CASES.json.gz'),
            learned_reporter_tested=False,parameters_optimized=False,
            scope='NATIVE_REPORT_DRIVEN_CONTROLLER; IDEAL_AND_NOISY_FUTURE_ARE_TEST_ONLY; PREFIX_IS_DIAGNOSTIC')
        write(a.output/'REPORT.json',report)
        write(a.output/'STATE.json',dict(status='COMPLETE',identity=digest,rollouts=len(records),
                                       elapsed_seconds=time.monotonic()-start))
        print(json.dumps([r for r in report['rows'] if r['horizon']=='ALL']),flush=True)
    except Exception as exc:
        write(a.output/'STATE.json',dict(status='FAILED',identity=digest,rollouts=len(records),
                                       error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':
    main()
