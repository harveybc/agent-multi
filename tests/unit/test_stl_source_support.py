import numpy as np
import pandas as pd
import pytest
from tools.audit_stl_source_support import finite_windows, source_window


def test_window_does_not_wrap_before_history():
    assert finite_windows(np.ones(5), [0, 2, 4], [-2, 0]).tolist() == [False, True, True]


def test_missing_observation_invalidates_only_referenced_slots():
    x = np.array([1., np.nan, 2., 3., 4.])
    assert finite_windows(x, [2, 3, 4], [-2, 0]).tolist() == [True, False, True]
    assert np.isnan(x[1])


def test_source_audit_preserves_absence_and_neighbor_values():
    t = pd.Timestamp('2023-11-29T13:00Z')
    frame = pd.DataFrame({'close': [2., 4.]}, index=[t-pd.Timedelta(hours=1), t+pd.Timedelta(hours=1)])
    result = source_window(frame, t, 'close')
    assert list(result.values()) == [2., None, 4.]
    assert len(frame) == 2


def test_ambiguous_source_is_rejected():
    t = pd.Timestamp('2023-11-29T13:00Z')
    with pytest.raises(ValueError, match='DUPLICATE'):
        source_window(pd.DataFrame({'close': [1., 2.]}, index=[t, t]), t, 'close')
