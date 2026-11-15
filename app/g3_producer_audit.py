"""Exact dependency and temporal support checks for offline G3 qualification."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


TECHNICAL = ('ATR', 'BB_Width', 'ATR_ratio', 'rolling_std_24',
             'Stochastic_D', 'CCI', 'BB_position')
CROSSES = ('gbpusd', 'usdchf', 'usdjpy', 'eurgbp', 'eurjpy', 'audusd')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def dependency_inventory(columns, activity_features):
    """Enumerate the reviewed selected299 graph; never guess unknown producers."""
    sources = {
        'eurusd': ['EURUSD_BID_OHLC'],
        'calendar': ['ADMITTED_TIMESTAMPS', 'CLOSURES', 'TZ_DATABASE'],
        'technical': ['eurusd'], 'raw': ['eurusd'], 'price': ['eurusd'],
        'candle': ['eurusd'], 'ssa': ['eurusd'], 'wavelet': ['eurusd'],
        'hilbert': ['eurusd'], 'kalman': ['eurusd', 'FROZEN_KALMAN'],
        'gmm': ['eurusd', 'FROZEN_GMM'],
        'cross_prices': [p.upper() + '_CLOSE' for p in CROSSES],
        'factor': ['cross_prices', 'FROZEN_FACTOR'],
        'strength': ['cross_prices'],
        'rates': ['US10Y_RELEASES'], 'cot': ['EUR_COT_RELEASES'],
        'activity': ['FROZEN_ACTIVITY_OOF'],
        'dax': ['DAX_RELEASES'], 'spx': ['SPX_RELEASES'],
    }
    known = {}
    for group, names in {
        'raw': ['OPEN', 'RAW_BAR_LOG_RETURN', 'RAW_LOG_RANGE', 'RAW_CLOSE_LOG_RETURN'],
        'technical': TECHNICAL,
        'calendar': ['cal.utc_session_europe', 'cal.london_hod_sin', 'cal.london_hod_cos',
                     'px.hours_since_session_open', 'ev.tokyo_fix.hours_since',
                     'ev.tokyo_fix.hours_to', 'ev.london_fix.hours_to', 'ev.ny_cut.hours_to'],
        'price': [f'px.logret_{k}' for k in (1,4,24)] +
                 [f'px.range_{k}' for k in (1,2,4,6,48)] +
                 [f'px.close_loc_{k}' for k in (1,4,12,24,48)] +
                 [f'px.er_{k}' for k in (2,6,12,48)],
        'candle': ['cdl.body_pips', 'cdl.lower_wick_pips', 'cdl.range_pips',
                   'cdl.close_minus_low_pips', 'cdl.high_minus_close_pips'],
        'rates': ['dly.us10y.chg_1d', 'dly.us10y.chg_5d'],
        'cot': ['cot.eur.net_long', 'cot.eur.net_share', 'cot.eur.net_z156'],
        'factor': ['fac.fx_f1.ret_4', 'fac.fx_f1.ret_24'],
        'activity': ['act.p_hit'], 'ssa': ['ssa.c2_pips', 'ssa.c3_pips'],
        'wavelet': ['wv.lc.A4', 'wv.lc.D4', 'wv.lc.D3', 'wv.lc.D2',
                    'wv.lc.energy_D2', 'wv.lc.energy_D1', 'wv.r1.D3',
                    'wv.r1.energy_D3', 'wv.r1.energy_D2', 'wv.r1.energy_D1'],
        'hilbert': ['hb.r1.phase_cos', 'hb.r1.phase_sin', 'hb.lc_detr.inst_freq'],
        'strength': ['wv.usd.A4', 'hb.eur.inst_freq'],
        'kalman': ['kf.lc.level_minus_close'],
        'gmm': ['rg.gmm.p0', 'rg.gmm.p1', 'rg.gmm.p2', 'rg.gmm.entropy'],
    }.items():
        known.update({name: group for name in names})
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('G3_DUPLICATE_OR_EMPTY_COLUMNS')
    activity = []
    activity_bases = {
        'cal.utc_dow_cos': 'calendar', 'cal.utc_dow_sin': 'calendar',
        'dow_cos': 'calendar', 'dow_sin': 'calendar',
        'ev.tokyo_fix.hours_since': 'calendar', 'ev.tokyo_fix.hours_to': 'calendar',
        'px.hours_since_session_open': 'calendar',
        'dly.dax.ret_5d': 'dax', 'dly.spx.ret_1d': 'spx',
    }
    for name in activity_features:
        base, sep, lag = name.partition('@')
        if not sep or not lag.isdigit() or base not in activity_bases:
            raise ValueError('G3_UNKNOWN_ACTIVITY_DEPENDENCY:' + name)
        activity.append(dict(name=name, source=activity_bases[base], lag_rows=int(lag)))
        sources['activity'].append(activity_bases[base])
    if 'act.p_hit' in columns and not activity:
        raise ValueError('G3_ACTIVITY_DEPENDENCIES_EMPTY')
    graph = {k: sorted(set(v)) for k, v in sources.items()}
    primitive = set(v for vs in graph.values() for v in vs if v not in graph)

    def leaves(name, seen=()):
        if name in seen:
            raise ValueError('G3_DEPENDENCY_CYCLE')
        if name in primitive:
            return {name}
        if name not in graph:
            raise ValueError('G3_MISSING_DEPENDENCY:' + name)
        return set().union(*(leaves(x, seen+(name,)) for x in graph[name]))

    rows = []
    for name in columns:
        base, sep, transform = name.partition('@')
        if base not in known or (sep and transform not in ('diff', 'log', 'asinh')):
            raise ValueError('G3_UNKNOWN_FEATURE_OR_TRANSFORM:' + name)
        producer = known[base]
        rows.append(dict(column=name, producer=producer, transform=transform if sep else 'none',
                         primitive_dependencies=sorted(leaves(producer)),
                         price_calendar_replay=producer not in
                         ('rates', 'cot', 'factor', 'strength', 'activity')))
    return dict(columns=rows, activity_predictors=activity, graph=graph,
                primitive_dependencies=sorted(set().union(*(leaves(known[c.partition('@')[0]])
                                                            for c in columns))))


def require_grid(frame):
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None
            or not frame.index.is_unique or not frame.index.is_monotonic_increasing
            or frame.index.hasnans or not frame.columns.is_unique or len(frame) == 0):
        raise ValueError('G3_INVALID_GRID')


def support_report(frame, columns, offsets):
    require_grid(frame)
    if (not offsets or sorted(set(offsets)) != list(offsets) or offsets[-1] != 0
            or any(type(x) is not int or x > 0 for x in offsets)):
        raise ValueError('G3_INVALID_WINDOW')
    if any(c not in frame for c in columns):
        raise ValueError('G3_MISSING_COLUMN')
    finite = np.isfinite(frame[columns].to_numpy(float))
    good = finite.all(axis=1)
    valid = np.zeros(len(frame), bool)
    positions = np.arange(-offsets[0], len(frame))
    if len(positions):
        valid[positions] = good[positions[:, None] + np.asarray(offsets)].all(axis=1)
    def span(mask):
        idx = np.flatnonzero(mask)
        return dict(count=len(idx), first=str(frame.index[idx[0]]) if len(idx) else None,
                    last=str(frame.index[idx[-1]]) if len(idx) else None)
    return dict(rows=len(frame), start=str(frame.index[0]), end=str(frame.index[-1]),
                finite_rows=span(good), finite_windows=span(valid),
                window_support_sha256=hashlib.sha256(frame.index.asi8[valid].tobytes()).hexdigest(),
                by_column={c: dict(span(finite[:,i]), missing=int((~finite[:,i]).sum()))
                           for i,c in enumerate(columns)})


def parity(actual, expected, *, tolerance=1e-9):
    require_grid(actual)
    require_grid(expected)
    if not actual.index.equals(expected.index) or list(actual) != list(expected):
        raise ValueError('G3_PARITY_AXES')
    out = {}
    for c in actual:
        a, b = actual[c].to_numpy(float), expected[c].to_numpy(float)
        finite = np.isfinite(a) & np.isfinite(b)
        mask_equal = np.array_equal(np.isfinite(a), np.isfinite(b))
        error = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else None
        out[c] = dict(mask_equal=bool(mask_equal), max_error=error,
                      exact=bool(np.array_equal(a, b, equal_nan=True)),
                      passed=bool(mask_equal and finite.any() and np.allclose(
                          a,b,rtol=tolerance,atol=tolerance,equal_nan=True)))
    return out
