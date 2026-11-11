import pandas as pd
import pytest
from tools.qualify_stl_recovery_sources import aggregate_complete, repair_interval


def frame():
    t = pd.Timestamp('2023-11-29T13:00Z')
    return t, pd.DataFrame(dict(open=[2.]*12, high=[3.]*12, low=[1.]*12, close=[2.5]*12),
                           index=pd.date_range(t, periods=12, freq='5min'))


def test_exact_aggregate_no_input_mutation():
    t, f = frame()
    before = f.copy()
    assert aggregate_complete(f, t, 5) == dict(open=2., high=3., low=1., close=2.5)
    pd.testing.assert_frame_equal(f, before)


def test_missing_minute_never_interpolated():
    t, f = frame()
    with pytest.raises(ValueError, match='INCOMPLETE'):
        aggregate_complete(f.drop(f.index[3]), t, 5)


def test_wrong_labels_rejected():
    t, f = frame()
    f.index += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match='INCOMPLETE'):
        aggregate_complete(f, t, 5)


def test_duplicate_and_invalid_prices_rejected():
    t, f = frame()
    with pytest.raises(ValueError, match='DUPLICATE'):
        aggregate_complete(pd.concat([f, f.iloc[:1]]), t, 5)
    f.iloc[0, f.columns.get_loc('high')] = 0.
    with pytest.raises(ValueError, match='OHLC'):
        aggregate_complete(f, t, 5)


def interval_frames():
    times = pd.date_range('2023-11-29T11:00Z', periods=5, freq='h')
    hourly = pd.DataFrame(dict(open=2., high=3., low=1., close=2.5), index=times)
    native = hourly.drop(times[2]).copy()
    native.loc[times[1], 'close'] = 2.2
    native.loc[times[3], 'open'] = 2.3
    fine = {}
    for step in (5, 15):
        fine[step] = pd.DataFrame(dict(open=2., high=3., low=1., close=2.5),
            index=pd.date_range(times[0], periods=5*60//step, freq=f'{step}min'))
    return native, hourly, fine, times[1:4]


def test_interval_repairs_neighbors_without_mutating_source():
    native, hourly, fine, window = interval_frames()
    before = native.copy()
    fixed, proof = repair_interval(native, hourly, fine, window)
    pd.testing.assert_frame_equal(fixed, hourly)
    pd.testing.assert_frame_equal(native, before)
    assert proof['inserted_rows'] == 1
    assert proof['replaced_rows'] == 2
    assert proof['outside_exact']


def test_interval_rejects_unexplained_outside_changes():
    native, hourly, fine, window = interval_frames()
    native.iloc[0, 0] = 2.1
    with pytest.raises(ValueError, match='OUTSIDE'):
        repair_interval(native, hourly, fine, window)


def test_interval_rejects_incomplete_finer_source():
    native, hourly, fine, window = interval_frames()
    fine[5] = fine[5].drop(window[1])
    with pytest.raises(ValueError, match='INCOMPLETE'):
        repair_interval(native, hourly, fine, window)


def test_interval_requires_both_boundary_anchors_and_grid():
    native, hourly, fine, window = interval_frames()
    with pytest.raises(ValueError, match='BOUNDARY'):
        repair_interval(native.iloc[1:], hourly.iloc[1:], fine, window)
    with pytest.raises(ValueError, match='GRID'):
        repair_interval(native, hourly.drop(window[1]), fine, window)
