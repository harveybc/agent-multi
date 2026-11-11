"""Replay qualified source changes through frozen STL, without training or promotion."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels

from app.stl_source_repair import repair_endpoints
from app.causal_stl_inputs import CausalSTLColumn
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from app.structural_refit import verify_files
from tools.run_bayesian_head_contrast import verify_done
from tools.rebuild_recovered_fx_features import COLS


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def authenticated(root, runtime):
    ready = json.loads((root/'PREPARED.json').read_text())
    verify_files(root, ready['files'])
    cfg = json.loads((root/'CONFIG.json').read_text())
    if cfg['identity'] != digest({k: v for k, v in cfg.items() if k != 'identity'}):
        raise ValueError('STL_PARENT_IDENTITY')
    if cfg['statsmodels'] != statsmodels.__version__:
        raise ValueError('STL_PARENT_LIBRARY_VERSION')
    verify_files(runtime, cfg['sources'])
    return cfg


def repair_column(task):
    name, old, new, seasonal, parameters = task
    result, affected, changed = repair_endpoints(old, new, seasonal, parameters)
    recovered = ~np.isfinite(seasonal) & np.isfinite(result)
    # Recompute a prefix independently: a suffix must not affect emitted values.
    rows = np.flatnonzero(affected)
    checks = []
    if len(rows):
        for cut in np.unique(rows[np.linspace(0, len(rows)-1, 3, dtype=int)]+1):
            prefix = new[:cut]
            stream = CausalSTLColumn(parameters, prefix[max(0, cut-parameters['history']):cut-1])
            value = stream.update(prefix[-1])
            if not (value == result[cut-1] or (np.isnan(value) and np.isnan(result[cut-1]))):
                raise ValueError('STL_REPAIR_PREFIX')
            checks.append(int(cut))
    return name, result, dict(changed_input_rows=int(changed.sum()),
                             recomputed_rows=int(affected.sum()),
                             recovered_endpoints=int(recovered.sum()),
                             prefix_checks=checks, unchanged_outside_support=True)


def window_mask(values, observations, stride):
    valid = np.ones(len(values), dtype=bool)
    finite = np.isfinite(values).all(axis=1)
    for offset in range(0, observations*stride, stride):
        if offset:
            valid[:offset] = False
            valid[offset:] &= finite[:-offset]
        else:
            valid &= finite
    return valid


def _run(root, repair, output, workers, source_runtime):
    if not 1 <= workers <= 4:
        raise ValueError('STL_REPAIR_WORKERS')
    for source in (root, repair):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError('STL_REPAIR_OUTPUT_OVERLAPS_SOURCE')
    cfg = authenticated(root, source_runtime)
    receipt = json.loads((repair/'DONE.json').read_text())
    verify_files(repair, receipt['files'])
    column_pins = {}
    for j, name in enumerate(cfg['columns']):
        folder = root/'columns'/f'{j:03d}'
        verify_done(folder, cfg['identity'])
        column_pins[name] = file_sha(folder/'DONE.json')
    pins = dict(parent_prepared=file_sha(root/'PREPARED.json'),
                repair_done=file_sha(repair/'DONE.json'),
                runtime=file_sha(Path(__file__)),
                adapter=file_sha(Path(__file__).parents[1]/'app/stl_source_repair.py'),
                column_receipts=column_pins)
    identity = digest(pins)
    if (output/'DONE.json').exists():
        done = json.loads((output/'DONE.json').read_text())
        if done['identity'] != identity:
            raise ValueError('STL_REPAIR_RESUME_IDENTITY')
        verify_files(output, done['files'])
        return
    output.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((root/'SOURCE_RECIPE.json').read_text())
    with np.load(root/'RAW.npz') as data:
        raw = data['values'].copy(); times = data['times'].copy()
        indices = data['parent_indices'].copy()
    frame = pd.read_parquet(repair/'STORE_CANDIDATE.parquet', columns=['timestamp']+cfg['columns'],
                            filters=[('timestamp', '<', pd.Timestamp('2025-01-01', tz='UTC'))])
    stamps = pd.DatetimeIndex(frame.timestamp).as_unit('ns').asi8 + 3600*10**9
    if not np.array_equal(stamps, times):
        raise ValueError('STL_REPAIR_GRID')
    new = frame[cfg['columns']].to_numpy(float)
    for j, column in enumerate(recipe['tensor']['columns']):
        if column['consumer_ops'] == ['DIFF']:
            new[:, j] = np.r_[np.nan, np.diff(new[:, j])]
        elif column['consumer_ops']:
            raise ValueError('STL_REPAIR_UNKNOWN_CONSUMER_OP')
    untouched = [j for j, c in enumerate(cfg['columns']) if c not in COLS]
    if not np.array_equal(raw[:, untouched], new[:, untouched], equal_nan=True):
        raise ValueError('STL_REPAIR_UNDECLARED_COLUMN_CHANGE')
    seasonal = []; tasks = []
    for j, name in enumerate(cfg['columns']):
        folder = root/'columns'/f'{j:03d}'
        verify_done(folder, cfg['identity'])
        with np.load(folder/'SEASONAL.npz') as data:
            values = data['values'].copy()
        seasonal.append(values)
        if name in COLS:
            params = json.loads((folder/'RESULT.json').read_text())['parameters']
            tasks.append((name, raw[:, j], new[:, j], values, params))
    before = np.column_stack(seasonal)
    after = before.copy(); reports = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, values, report in pool.map(repair_column, tasks):
            after[:, cfg['columns'].index(name)] = values
            reports[name] = report
            write(output/'PROGRESS.json', dict(stage='RECOMPUTING', columns=reports))
            print(json.dumps(dict(column=name, **report)), flush=True)
    w = cfg['original_window']
    raw_valid = window_mask(raw, w['observations'], w['stride_rows'])[indices]
    old_valid = raw_valid & window_mask(before, w['observations'], w['stride_rows'])[indices]
    new_valid = raw_valid & window_mask(after, w['observations'], w['stride_rows'])[indices]
    if (old_valid & ~new_valid).any():
        raise ValueError('STL_REPAIR_LOST_COVERAGE')
    months = pd.DatetimeIndex(pd.to_datetime(times[indices], utc=True)).strftime('%Y-%m')
    coverage = {month: dict(raw=int((raw_valid & (months==month)).sum()),
                           before=int((old_valid & (months==month)).sum()),
                           after=int((new_valid & (months==month)).sum())) for month in sorted(set(months))}
    missing = [m for m, v in coverage.items() if v['raw'] and not v['after']]
    np.savez_compressed(output/'REPAIRED_SEASONAL.npz', values=after, times=times,
                        parent_indices=indices)
    report = dict(schema='stl_source_propagation.v1', identity=identity, pins=pins,
                  column_receipts=column_pins, columns=reports, monthly_coverage=coverage,
                  recovered_original_windows=int((new_valid & ~old_valid).sum()),
                  remaining_empty_months=missing, no_imputation=True,
                  frozen_parameters=True, parameters_retuned=False,
                  diagnostic_only=True, training_released=False, model_promoted=False)
    write(output/'RESULT.json', report)
    # Authenticate inputs again before publishing a terminal receipt.
    authenticated(root, source_runtime); verify_files(repair, receipt['files'])
    if (file_sha(Path(__file__)) != pins['runtime']
            or file_sha(Path(__file__).parents[1]/'app/stl_source_repair.py') != pins['adapter']):
        raise ValueError('STL_REPAIR_RUNTIME_CHANGED')
    for j, name in enumerate(cfg['columns']):
        verify_done(root/'columns'/f'{j:03d}', cfg['identity'])
        if file_sha(root/'columns'/f'{j:03d}'/'DONE.json') != column_pins[name]:
            raise ValueError('STL_REPAIR_INPUT_CHANGED')
    write(output/'DONE.json', dict(identity=identity, files={name: file_sha(output/name) for name in
          ('RESULT.json', 'REPAIRED_SEASONAL.npz')}))
    print(json.dumps(dict(recovered_windows=report['recovered_original_windows'], empty_months=missing)), flush=True)


def run(root, repair, output, workers, source_runtime):
    for source in (root, repair, source_runtime):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError('STL_REPAIR_OUTPUT_OVERLAPS_SOURCE')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(root, repair, output, workers, source_runtime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'repair', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--source-runtime', type=Path, default=Path(__file__).resolve().parents[1])
    run(**vars(parser.parse_args()))
