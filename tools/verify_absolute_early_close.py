#!/usr/bin/env python3
"""Sequential synthetic native exit verification, with independent cash oracle."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.absolute_exit_corpus import specs, evaluate_case
from app.early_close_metrics import summarize_exits
from app.zigzag_opportunities import engine, sha


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def source_pins(engine_root):
    own = ['app/absolute_exit_corpus.py', 'app/early_close_metrics.py',
           'app/optimal_exit.py', 'app/horizon_risk_rules.py', 'app/zigzag_risk_profiles.py',
           'app/protection_reference.py', 'app/zigzag_opportunities.py',
           'tools/verify_absolute_early_close.py']
    return dict(agent_multi={p: sha(ROOT / p) for p in own},
                heuristic_strategy={str(p.relative_to(engine_root)): sha(p)
                    for p in sorted((engine_root / 'app/short_horizon').rglob('*.py'))})


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        s = row['spec']
        for policy, metrics in row['policies'].items():
            groups[('ALL', 'ALL', 'ALL', policy)].append(metrics)
            groups[(s['horizon'], s['direction'], s['mode'], policy)].append(metrics)
    return [dict(horizon=k[0], direction=k[1], mode=k[2], policy=k[3],
                 **summarize_exits(v)) for k, v in groups.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('NEW_OUTPUT_REQUIRED')
    if sha(a.config) != a.config_sha:
        raise ValueError('CONFIG_IDENTITY')
    config = json.loads(a.config.read_text())
    if (config['rel_volume'] != .1 or config['leverage'] != 50 or config['quality'] != 1
            or config['schema'] != 'shared_horizon_risk_controls.v1'):
        raise ValueError('BUSINESS_CONTRACT')
    engine_root = a.engine_root.resolve()
    engine(engine_root)
    from app.short_horizon.expert_guided import absolute
    if Path(absolute.__file__).resolve() != engine_root / 'app/short_horizon/expert_guided/absolute.py':
        raise ValueError('IMPORTED_ENGINE_IDENTITY')
    pins = source_pins(engine_root)
    cases = list(specs())
    if len(cases) != 1744 or len({s['id'] for s in cases}) != len(cases):
        raise ValueError('CORPUS_COVERAGE')
    identity = dict(schema='absolute_exit_corpus.v1', sources=pins,
                    config_sha256=a.config_sha, scenarios=cases)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    a.output.mkdir(parents=True)
    write(a.output / 'MANIFEST.json', identity | dict(identity=digest))
    start = time.monotonic()
    records = []
    try:
        for spec in cases:
            records.append(evaluate_case(spec, config))
            if len(records) % 64 == 0:
                progress = dict(status='RUNNING', identity=digest, completed=len(records),
                                total=len(cases), elapsed_seconds=time.monotonic()-start)
                write(a.output / 'STATE.json', progress)
                print(json.dumps(progress), flush=True)
        if pins != source_pins(engine_root) or sha(a.config) != a.config_sha:
            raise ValueError('SOURCES_CHANGED_DURING_REPLAY')
        raw = (json.dumps(records, sort_keys=True, allow_nan=False) + '\n').encode()
        (a.output / 'CASES.json.gz').write_bytes(gzip.compress(raw, mtime=0))
        import numpy as np
        import pandas as pd
        report = dict(status='PASS', identity=digest, positions=len(records),
            native_close_checks=sum(r['native_close_checks'] for r in records),
            native_passive_checks=len(records),
            max_cash_error_usd=max(r['max_cash_error'] for r in records),
            cases_sha256=sha(a.output / 'CASES.json.gz'),
            environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
            scope='SYNTHETIC_FIXED_POSITION_EXECUTION_AND_METRICS_NOT_LEARNED_POLICY',
            rows=aggregate(records))
        write(a.output / 'REPORT.json', report)
        write(a.output / 'STATE.json', dict(status='COMPLETE', identity=digest,
            completed=len(records), total=len(cases), elapsed_seconds=time.monotonic()-start))
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)
    except Exception as exc:
        write(a.output / 'STATE.json', dict(status='FAILED', identity=digest,
            completed=len(records), total=len(cases), error_type=type(exc).__name__,
            error=str(exc), elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
