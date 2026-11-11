"""Replay the RETIRED global activity estimator; NOT the current consumer producer."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier


def assert_parity(times, probabilities, reference):
    expected_times = pd.DatetimeIndex(pd.to_datetime(reference['timestamp'], utc=True))
    if not times.is_unique or not expected_times.is_unique or not times.equals(expected_times):
        raise ValueError('TIMESTAMP_SUPPORT_MISMATCH')
    expected = reference['p_hit'].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.array_equal(probabilities, expected):
        raise ValueError('PROBABILITY_MISMATCH')


def recover(screen, decision, reference, output):
    inputs = {name: Path(path) for name, path in
              [('screen', screen), ('decision', decision), ('reference', reference)]}
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest()
               for name, path in inputs.items()}
    inp = pd.read_parquet(screen)
    spec = json.loads(Path(decision).read_text())
    features = sorted({(d['feature'], int(d['lag'])) for d in spec['activity_confirmed']})
    if any(lag < 0 for _, lag in features):
        raise ValueError('NEGATIVE_LAG')
    times = pd.DatetimeIndex(pd.to_datetime(inp['timestamp'], utc=True))
    if not times.is_monotonic_increasing or not times.is_unique:
        raise ValueError('INVALID_TIME_ORDER')
    X = pd.DataFrame({f'{f}@{lag}': inp[f].shift(lag).to_numpy(float)
                      for f, lag in features}, index=times)
    y = inp['label_hit'].to_numpy(float)
    train = (inp['period'].to_numpy() == 'TRAIN') & np.isfinite(y)
    if not train.any() or not set(np.unique(y[train])) == {0., 1.}:
        raise ValueError('INVALID_TRAIN_LABELS')
    if times[train].max() >= pd.Timestamp('2023-01-01', tz='UTC'):
        raise ValueError('HISTORICAL_TRAIN_BOUNDARY')
    model = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=200,
                                          max_iter=300, learning_rate=.05, random_state=0)
    model.fit(X[train], y[train].astype(int))
    probabilities = model.predict_proba(X)[:, 1]
    sealed = pd.read_parquet(reference)
    assert_parity(times, probabilities, sealed)
    # Keep failed or concurrent recovery artifacts separate from a prior export.
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    weights = out / 'ACTIVITY.joblib'
    joblib.dump(model, weights)
    assert_parity(times, joblib.load(weights).predict_proba(X)[:, 1], sealed)
    for name, path in inputs.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digests[name]:
            raise ValueError('SOURCE_CHANGED_DURING_RECOVERY')
    report = dict(schema='activity_recovery.v1', status='RETIRED_GLOBAL_REPLAY_ONLY',
                  consumer_compatible=False,
                  sklearn_version=sklearn.__version__, inputs_sha256=digests,
                  features=list(X.columns), train_rows=int(train.sum()),
                  train_start=times[train].min().isoformat(),
                  train_end=times[train].max().isoformat(),
                  replay_rows=len(X), maximum_error=0., reload_exact=True,
                  parameters=model.get_params(), fitted_iterations=int(model.n_iter_),
                  weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  live_ready=False, training_store_changed=False)
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('screen', 'decision', 'reference', 'output'):
        parser.add_argument('--' + name, required=True)
    recover(**vars(parser.parse_args()))
