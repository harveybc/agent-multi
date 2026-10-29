#!/usr/bin/env python3
"""Authenticate and describe immutable TRAIN oracle paths; never place orders."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.zigzag_opportunities import load_tape, sha
from app.zigzag_risk_profiles import profile, verify_paths
from tools.zigzag_opportunity_deap import authenticated, identity, write


def build(root):
    spec = json.loads((root/'SPEC.private.json').read_text())
    ident = identity(spec)
    report = authenticated(root/'RESULTS.json', ident)
    if spec['future_pivots'] != 3 or spec['horizons'] != [3, 6, 12, 24]:
        raise ValueError('PROFILE_SOURCE_CONTRACT')
    if [r['horizon'] for r in report['rows']] != spec['horizons']:
        raise ValueError('HORIZON_SUPPORT')
    tape = load_tape(spec)
    consumed = [root/'SPEC.private.json', root/'RESULTS.json']
    for row in report['rows']:
        path = root/row['records_file']
        if path.resolve().parent != root.resolve() or sha(path) != row['records_sha256']:
            raise ValueError('RECORD_HASH_OR_LOCATION')
        consumed.append(path)
    before = {p.name: sha(p) for p in consumed}
    rows = []
    for row in report['rows']:
        with np.load(root/row['records_file'], allow_pickle=False) as archive:
            # NPZ indexing decompresses on every access; materialize once per horizon.
            records = {key: archive[key] for key in archive.files}
        audit = verify_paths(tape, records)
        result = profile(tape, records)
        rows.append(dict(horizon=row['horizon'], params=row['params'],
                         audit=audit, **result))
    if before != {p.name: sha(p) for p in consumed} or identity(spec) != ident:
        raise ValueError('SOURCE_CHANGED_DURING_PROFILE')
    return dict(schema='zigzag_train_risk_profiles.v1', source_identity=ident,
                source_sha256=before,
                code_sha256={p: sha(ROOT/p) for p in
                             ('app/zigzag_risk_profiles.py', 'tools/zigzag_risk_profiles.py')},
                scope='DESCRIPTIVE_ORACLE_REFERENCES_NOT_EXECUTABLE_SL_TP_OR_ACCOUNT_RESULTS',
                train=spec['train'], future_pivots=3, pip_size=1e-4,
                costs_status=tape.metadata['costs_status'],
                calendar_subset='selected release <= conservative last observed flat boundary; no endpoint substitution',
                uncertainty='Whole endpoint-bar OHLC; unknown intrabar extreme order; no confidence intervals from overlapping origins',
                sl_semantics='MAE from entry, no buffer; equality may trigger a stop',
                tp_semantics='signed ideal pivot distance; oracle_all includes losing opportunities, not valid TP levels',
                volume=dict(formula='q * rel_volume * balance', rel_volume=.1,
                            q_range=[0, 1], risk_budget_selected=False,
                            horizon_size_premium=False),
                source_unchanged=True, rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.root.resolve()):
        raise ValueError('NEW_OUTPUT_OUTSIDE_SOURCE_REQUIRED')
    result = build(a.root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write(a.output, result)
    print(json.dumps(dict(status='COMPLETE', source_unchanged=True,
                          paths=sum(r['audit']['paths'] for r in result['rows']),
                          report_sha256=sha(a.output))))
