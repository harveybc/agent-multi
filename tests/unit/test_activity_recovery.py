import numpy as np
import pandas as pd
import pytest

from tools.recover_activity_producer import assert_parity


def sample():
    t = pd.date_range('2020-01-01', periods=3, freq='h', tz='UTC')
    p = np.array([.1, .5, .9])
    return t, p, pd.DataFrame({'timestamp': t, 'p_hit': p})


def test_exact_probabilities_accepted():
    assert_parity(*sample())


def test_smallest_representable_change_rejected():
    t, p, ref = sample()
    p[0] = np.nextafter(p[0], 1.)
    with pytest.raises(ValueError, match='PROBABILITY'):
        assert_parity(t, p, ref)


def test_same_values_wrong_clock_rejected():
    t, p, ref = sample()
    with pytest.raises(ValueError, match='TIMESTAMP'):
        assert_parity(t + pd.Timedelta(hours=1), p, ref)


def test_duplicate_timestamps_rejected():
    t, p, ref = sample()
    ref['timestamp'] = [t[0], t[0], t[2]]
    with pytest.raises(ValueError, match='TIMESTAMP'):
        assert_parity(pd.DatetimeIndex(ref.timestamp), p, ref)


def test_nonfinite_rejected():
    t, p, ref = sample()
    p[0] = np.nan
    ref.loc[0, 'p_hit'] = np.nan
    with pytest.raises(ValueError, match='PROBABILITY'):
        assert_parity(t, p, ref)
