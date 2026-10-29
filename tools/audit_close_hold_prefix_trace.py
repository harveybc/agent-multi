#!/usr/bin/env python3
"""Read-only ablation of full HOLD traces; does not claim a new backtest."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def audit(cases,reference):
    rows=[r for r in cases if r['arm']=='prefix_guarded']
    refs={r['spec']['id']:r for r in reference}
    if len(rows)!=len(refs) or {r['spec']['id'] for r in rows}!=set(refs):
        raise ValueError('PAIRED_SUPPORT')
    if any(r['metrics']['early_close'] for r in rows):
        raise ValueError('NEED_COMPLETE_PASSIVE_TRACES')
    out={}
    for mode in ('without_persistence','without_uncertainty_or_persistence'):
        signals=executable=0
        for r in rows:
            candidates=[t for t in r['trace'] if
                (t['advantage_risk']>.02 if mode=='without_persistence'
                 else t['expected_close_net']-t['best_expected_wait_net']>1e-7)]
            first=candidates[0] if candidates else None
            signals+=first is not None
            executable+=first is not None and first['decision_bar']+1 in refs[r['spec']['id']]['fill_bars']
        out[mode]=dict(first_positive_requests=signals,executable_early_requests=executable,
                       no_positive_request=len(rows)-signals)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args()
    raw=a.cases.read_bytes(); ref=a.reference.read_bytes()
    print(json.dumps(dict(schema='prefix_close_trace_audit.v1',
        cases_sha256=hashlib.sha256(raw).hexdigest(),reference_sha256=hashlib.sha256(ref).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        result=audit(json.loads(gzip.decompress(raw)),json.loads(gzip.decompress(ref)))),indent=2))


if __name__=='__main__':
    main()
