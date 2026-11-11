"""Recover the consumer's six activity estimators, never the retired global fit."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_reference(times, predicted, reference):
    other = pd.DatetimeIndex(pd.to_datetime(reference['timestamp'], utc=True))
    if not times.is_unique or not times.equals(other):
        raise ValueError('CONSUMER_SUPPORT_MISMATCH')
    if not np.isfinite(predicted).all() or not np.array_equal(
            predicted, reference['act.p_hit'].to_numpy(float)):
        raise ValueError('CONSUMER_PROBABILITY_MISMATCH')


def replay(bundle, X):
    if list(X.columns) != bundle['features']:
        raise ValueError('FEATURE_ORDER')
    if not np.array_equal(X.index.asi8, bundle['times_ns']):
        raise ValueError('REPLAY_GRID')
    values = X.to_numpy(float)
    p = bundle['estimators'][0].predict_proba(values)[:, 1]
    for model, rows in zip(bundle['estimators'][1:], bundle['prediction_rows'], strict=True):
        p[rows] = model.predict_proba(values[rows])[:, 1]
    return p


def recover(spec_path, output):
    spec = json.loads(Path(spec_path).read_text())
    files = {k: Path(v['path']) for k, v in spec['files'].items()}
    for name, path in files.items():
        if sha(path) != spec['files'][name]['sha256']:
            raise ValueError('INPUT_DIGEST:' + name)
    root = Path(spec['predictor_root'])
    sys.path.insert(0, str(root))
    AP = importlib.import_module('tools.activity_producer')
    F = importlib.import_module('tools.fold_local_producers')
    if Path(AP.__file__).resolve() != files['producer'].resolve():
        raise ValueError('PRODUCER_IMPORT_IDENTITY')
    if Path(F.__file__).resolve() != files['scope_producer'].resolve():
        raise ValueError('SCOPE_IMPORT_IDENTITY')
    manifest = json.loads(files['manifest'].read_text())
    if manifest['consumer_scope']['report_sha256'] != sha(files['scope']):
        raise ValueError('SCOPE_NOT_BOUND_TO_CONSUMER')
    if manifest['file_sha256'] != sha(files['store']):
        raise ValueError('STORE_NOT_BOUND_TO_CONSUMER')
    report = json.loads(files['scope'].read_text())
    bars, X, hit, resolution, features = AP.load_inputs(
        files['snapshot'].parent, files['bars'].parent, files['zigzag'].parent,
        files['pool'], files['decision'])
    if features != report['activity']['features']:
        raise ValueError('HISTORICAL_FEATURES')
    allowed = F.allowed_mask(bars.index, report['scope'])
    fitted, fit_rows = [], []
    original_fit = AP._fit

    def capture(values, labels, rows):
        model = original_fit(values, labels, rows)
        fitted.append(model)
        fit_rows.append(rows.copy())
        return model

    AP._fit = capture
    k = report['activity']['oof_folds']
    try:
        probabilities, metadata = AP.scope_predictions(
            X, hit, resolution, allowed, k=k,
            embargo=report['activity']['embargo_bars'],
            lookback=report['activity']['input_lookback_bars'], mode='purged_kfold')
    finally:
        AP._fit = original_fit
    for key in ('fit_rows_full', 'oof_folds', 'embargo_bars', 'input_lookback_bars', 'allowed_bars'):
        if metadata[key] != report['activity'][key]:
            raise ValueError('SCOPE_METADATA:' + key)
    reference = pd.read_parquet(files['store'], columns=['timestamp', 'act.p_hit'])
    check_reference(bars.index, probabilities, reference)
    indices = np.flatnonzero(allowed)
    edges = np.linspace(0, len(indices), k + 1).astype(int)
    prediction_rows = [indices[edges[i]:edges[i+1]] for i in range(k)]
    if len(fitted) != k + 1:
        raise ValueError('ESTIMATOR_COUNT')
    bundle = dict(features=features, estimators=fitted, fit_rows=fit_rows,
                  prediction_rows=prediction_rows, times_ns=bars.index.asi8,
                  scope=report['scope'], metadata=metadata)
    check_reference(bars.index, replay(bundle, X), reference)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    target = out / 'CONSUMER_ACTIVITY.joblib'
    joblib.dump(bundle, target)
    check_reference(bars.index, replay(joblib.load(target), X), reference)
    for name, path in files.items():
        if sha(path) != spec['files'][name]['sha256']:
            raise ValueError('INPUT_CHANGED:' + name)
    receipt = dict(schema='consumer_activity_recovery.v1',
                   status='EXACT_CONSUMER_REPLAY', sklearn_version=sklearn.__version__,
                   spec_sha256=sha(spec_path), source_sha256={k: sha(v) for k, v in files.items()},
                   rows=len(X), estimators=len(fitted), features=features, metadata=metadata,
                   weights_sha256=sha(target), maximum_error=0., reload_exact=True,
                   live_ready=False, source_changes=0, model_promoted=False)
    (out / 'RESULT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    recover(args.spec, args.output)
