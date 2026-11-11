"""Attribute STL support loss without modifying stores, weights or missing data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite_windows(values, endpoints, offsets):
    indices = np.asarray(endpoints)[:, None] + np.asarray(offsets)[None, :]
    inside = (indices >= 0) & (indices < len(values))
    return inside.all(1) & np.isfinite(values[np.clip(indices, 0, len(values)-1)]).all(1)


def source_window(frame, stamp, column):
    """Presence is not permission to splice sources; retain neighboring differences."""
    if frame.index.has_duplicates:
        raise ValueError('DUPLICATE_SOURCE_TIMESTAMP')
    rows = frame[column].reindex(pd.date_range(stamp-pd.Timedelta(hours=1),
                                             stamp+pd.Timedelta(hours=1), freq='h'))
    return {str(t): float(v) if np.isfinite(v) else None for t, v in rows.items()}


def audit(root, cross_asset, recovered, lake):
    cfg = read(root/'CONFIG.json')
    receipt = read(root/'PREPARED.json')
    for name, digest in receipt['files'].items():
        if sha(root/name) != digest:
            raise ValueError('SOURCE_DIGEST:'+name)
    with np.load(root/'RAW.npz') as f:
        times = pd.to_datetime(f['times'], utc=True)
        parent = f['parent_indices'].copy()
    months = times[parent].strftime('%Y-%m')
    support = read(root/'operations/SUPPORT_AND_ENCODING.json')
    empty = [m for m, row in support['months'].items() if row['common'] == 0]
    per_column = {}
    for j, name in enumerate(cfg['columns']):
        folder = root/'columns'/f'{j:03d}'
        done = read(folder/'DONE.json')
        if done['identity'] != cfg['identity']:
            raise ValueError('COLUMN_IDENTITY')
        for file, digest in done['files'].items():
            if sha(folder/file) != digest:
                raise ValueError('COLUMN_DIGEST:'+file)
        with np.load(folder/'SEASONAL.npz') as f:
            values = f['values']
        valid = finite_windows(values, parent, cfg['original_window']['slot_offsets_rows'])
        per_column[name] = {m: int((~valid[months == m]).sum()) for m in empty}
    cutoff = pd.Timestamp('2025-01-01', tz='UTC')
    cross = pd.read_parquet(cross_asset, filters=[('timestamp', '<', cutoff)])
    pairs = ['gbpusd', 'usdchf', 'usdjpy', 'eurgbp', 'eurjpy', 'audusd']
    cols = [f'fx.{p}.ret_1' for p in pairs]
    bad = cross.loc[cross[cols].isna().any(axis=1), ['timestamp']+cols]
    incidents = [{str(k): (str(v) if k == 'timestamp' else
                           float(v) if np.isfinite(v) else None)
                  for k, v in row.items()} for row in bad.to_dict('records')]
    comparisons, pins, histories = {}, {}, {}
    for pair in pairs:
        src = recovered/(pair+'.parquet')
        alt = lake/pair/'1h.parquet'
        pins[pair] = dict(recovered=sha(src), lake=sha(alt))
        original = pd.read_parquet(src, filters=[('timestamp', '<', cutoff)]).set_index('timestamp')
        alternate = pd.read_parquet(alt, filters=[('datetime', '<', cutoff)])
        # Frozen lake clock identity: Europe/London wall time minus five hours.
        wall = pd.DatetimeIndex(alternate.pop('datetime')).tz_convert('UTC').tz_localize(None)
        alternate.index = (wall+pd.Timedelta(hours=5)).tz_localize(
            'Europe/London', ambiguous='NaT', nonexistent='NaT').tz_convert('UTC')
        alternate = alternate[alternate.index.notna()]
        histories[pair] = dict(first=str(original.index.min()),
                              pre_train_rows=int((original.index < pd.Timestamp('2020-01-01', tz='UTC')).sum()))
        dates = [pd.Timestamp('2023-11-29T13:00:00Z')]
        if pair == 'usdchf':
            dates.append(pd.Timestamp('2022-05-19T21:00:00Z'))
        comparisons[pair] = {str(t): dict(recovered=source_window(original, t, 'close'),
                                         lake=source_window(alternate, t, 'close')) for t in dates}
    return dict(schema='stl_source_support_audit.v1', identity=cfg['identity'],
                source_sha256=pins, cross_asset_sha256=sha(cross_asset),
                empty_months=empty, invalid_windows_by_column=per_column,
                upstream_return_incidents=incidents, candidate_source_comparisons=comparisons,
                raw_source_history=histories, engineered_store_first_available=str(times.min()),
                no_imputation=True, source_changes=0, training_started=False,
                decision='RECOVERY_NOT_QUALIFIED',
                remaining=['USDCHF missing observation requires a verified real source',
                           'November candidate source requires clock and boundary qualification',
                           'Pre-TRAIN feature history requires frozen-transform reconstruction'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cross-asset', 'recovered', 'lake', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    result = audit(a.root, a.cross_asset, a.recovered, a.lake)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('empty_months', 'decision', 'source_changes')}))
