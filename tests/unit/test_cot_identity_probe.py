"""Isolated identity diagnostic; a legacy clock is not source certification."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from app.cot_identity_probe import verified_weekly, legacy_asof, replace_channels


def document():
    dates = pd.date_range('2018-01-02', periods=170, freq='7D')
    net = pd.Series(np.arange(170, dtype=float) + 1)
    z = (net-net.rolling(156, min_periods=52).mean())/net.rolling(156, min_periods=52).std()
    rows = [dict(date=str(d.date()), long=201+i, short=200, net_long=i+1,
                 net_share=(i+1)/(401+i), net_z156=None if pd.isna(z[i]) else float(z[i]))
            for i,d in enumerate(dates)]
    result = dict(market='099741', rows=rows, rows_count=len(rows))
    resign(result)
    return result


def resign(doc):
    doc['content_sha256'] = hashlib.sha256(json.dumps(doc['rows'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def test_recompute_all_columns_and_reject_bad_values_even_with_resigned_digest():
    doc = document()
    frame = verified_weekly(doc)
    assert len(frame) == 170
    assert frame.iloc[-1, 0] == 170
    bad = copy.deepcopy(doc); bad['rows'][-1]['net_z156'] += .01; resign(bad)
    with pytest.raises(ValueError, match='ARITHMETIC'): verified_weekly(bad)
    bad = copy.deepcopy(doc); bad['rows'][0]['long'] += 1
    with pytest.raises(ValueError, match='DIGEST'): verified_weekly(bad)


@pytest.mark.parametrize('kind', ['market', 'duplicate', 'order'])
def test_identity_and_weekly_order(kind):
    doc = document()
    if kind == 'market': doc['market'] = '299741'
    elif kind == 'duplicate': doc['rows'][1]['date'] = doc['rows'][0]['date']
    else: doc['rows'].reverse()
    resign(doc)
    with pytest.raises(ValueError): verified_weekly(doc)


def test_boundary_stale_missing_and_future_prefix():
    frame = verified_weekly(document()).iloc[160:]
    first = frame.index[0] + pd.Timedelta(hours=93)
    ts = np.array([first.value-1, first.value, first.value+1], dtype=np.int64)
    old = legacy_asof(frame, ts)
    assert np.isnan(old[0]).all()
    np.testing.assert_array_equal(old[1], frame.iloc[0].values)
    altered = frame.copy(); altered.iloc[1:] = 999
    np.testing.assert_array_equal(old, legacy_asof(altered, ts))
    stale = first + pd.Timedelta(days=14, nanoseconds=1)
    assert np.isnan(legacy_asof(frame.iloc[:1], np.array([stale.value]))).all()


def test_full156_warmup_is_explicit_but_an_arbitrary_late_gap_is_rejected():
    doc = document()
    for row in doc['rows'][:155]: row['net_z156'] = None
    resign(doc)
    frame = verified_weekly(doc)
    assert frame.attrs == dict(observed_min_periods=156, recomputed_min_periods=52,
                               warmup_cells_recomputed=104)
    assert np.isfinite(frame.iloc[51,2])
    doc['rows'][160]['net_z156'] = None; resign(doc)
    with pytest.raises(ValueError, match='ARITHMETIC'): verified_weekly(doc)


def test_only_requested_channels_and_source_untouched():
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Inputs:
        recipe: dict
        windows: np.ndarray
        slot_ts_ns: np.ndarray
        valid: np.ndarray
    frame = verified_weekly(document()).iloc[160:]
    ts = (frame.index[0]+pd.Timedelta(hours=93)).value
    cols = ['other']+list(frame.columns)
    rec = dict(tensor=dict(columns=[dict(store_column=x, consumer_ops=[]) for x in cols]),
               scaler=dict(mean=[0]*4, std=[1]*4))
    x = Inputs(rec, np.ones((2,3,4), np.float32), np.full((2,3),ts), np.ones(2,bool))
    source = x.windows.copy()
    changed = replace_channels(x, frame)
    np.testing.assert_array_equal(x.windows, source)
    np.testing.assert_array_equal(changed.windows[:,:,0], source[:,:,0])
    assert np.all(changed.windows[:,:,1] == 161)
    assert changed.valid is x.valid
    bad = Inputs(rec, x.windows, x.slot_ts_ns-10**18, x.valid)
    with pytest.raises(ValueError, match='SUPPORT'): replace_channels(bad, frame)
