"""Non-governing COT identity sensitivity under the frozen legacy clock."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd

COLUMNS = ('cot.eur.net_long', 'cot.eur.net_share', 'cot.eur.net_z156')


def verified_weekly(doc):
    if doc['market'] != '099741':
        raise ValueError('COT_MARKET')
    digest = hashlib.sha256(json.dumps(doc['rows'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    if digest != doc['content_sha256']:
        raise ValueError('COT_DIGEST')
    frame = pd.DataFrame(doc['rows'])
    ts = pd.DatetimeIndex(pd.to_datetime(frame['date'], utc=True)).as_unit('ns')
    if len(frame) != doc['rows_count'] or not len(frame) or not (np.diff(ts.asi8) > 0).all():
        raise ValueError('COT_DATE_ORDER')
    if not (ts == ts.normalize()).all():
        raise ValueError('COT_DATE_TIME')
    counts = frame[['long', 'short']].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not (counts == np.floor(counts)).all():
        raise ValueError('COT_COUNTS')
    net = pd.Series(counts[:, 0]-counts[:, 1])
    gross = pd.Series(counts.sum(axis=1)).replace(0, np.nan)
    rolling = net.rolling(156, min_periods=52)
    values = np.column_stack([net, net/gross, (net-rolling.mean())/rolling.std(ddof=1)])
    stated = frame[['net_long','net_share','net_z156']].to_numpy(float)
    warmup = 52
    if not np.allclose(values, stated, atol=1e-10, rtol=1e-12, equal_nan=True):
        # Accept only the precisely identified full-window variant, and report
        # it. Recompute the legacy min52 series; never accept arbitrary gaps.
        full = net.rolling(156, min_periods=156)
        alternative = values.copy()
        alternative[:,2] = (net-full.mean())/full.std(ddof=1)
        if not np.allclose(alternative, stated, atol=1e-10, rtol=1e-12, equal_nan=True):
            raise ValueError('COT_ARITHMETIC')
        warmup = 156
    result = pd.DataFrame(values, index=ts, columns=COLUMNS)
    result.attrs['observed_min_periods'] = warmup
    result.attrs['recomputed_min_periods'] = 52
    result.attrs['warmup_cells_recomputed'] = int((np.isnan(stated[:,2]) & np.isfinite(values[:,2])).sum())
    return result


def legacy_asof(weekly, decision_ns):
    """Historical93h/14d lookup, NOT a verified publication calendar."""
    releases = weekly.index.asi8 + pd.Timedelta(hours=93).value
    index = np.searchsorted(releases, decision_ns, side='right')-1
    safe = np.maximum(index, 0)
    values = weekly.to_numpy()[safe].copy()
    missing = (index < 0) | (decision_ns-releases[safe] > pd.Timedelta(days=14).value)
    values[missing] = np.nan
    return values


def replace_channels(inputs, weekly):
    columns = inputs.recipe['tensor']['columns']
    names = [c['store_column'] for c in columns]
    indices = [names.index(c) for c in COLUMNS]
    if any(columns[j]['consumer_ops'] for j in indices):
        raise ValueError('COT_UNEXPECTED_TRANSFORM')
    raw = legacy_asof(weekly, inputs.slot_ts_ns)
    sc = inputs.recipe['scaler']
    normalized = ((raw-np.asarray(sc['mean'])[indices])/np.asarray(sc['std'])[indices]).astype(np.float32)
    if not np.isfinite(normalized[inputs.valid]).all():
        raise ValueError('COT_CHANGED_SUPPORT')
    windows = inputs.windows.copy()
    windows[:,:,indices] = normalized
    windows.flags.writeable = False
    return replace(inputs, windows=windows)
