import numpy as np
import pandas as pd
import pytest

from tools.recover_stl_price_warmup import frozen_columns, compare


def test_frozen_transform_is_past_only_and_preserves_domain_failures():
    frame = pd.DataFrame({'a': [1., 2., 0., 4.], 'b': [-2., 0., 3., 4.]})
    names = ['a@diff', 'a@log', 'b@asinh', 'missing']
    result = frozen_columns(frame, names)
    assert list(result) == names[:-1]
    np.testing.assert_array_equal(result['a@diff'], [np.nan, 1., -2., 4.])
    assert np.isnan(result['a@log'].iloc[2])
    pd.testing.assert_frame_equal(frozen_columns(frame.iloc[:3], names), result.iloc[:3])


def test_unknown_transform_never_defaults_to_raw():
    with pytest.raises(ValueError, match='UNKNOWN_FROZEN_TRANSFORM'):
        frozen_columns(pd.DataFrame({'a': [1., 2.]}), ['a@unknown'])


def test_parity_checks_nan_support_not_just_finite_values():
    a = pd.DataFrame({'a': [1., np.nan]})
    b = pd.DataFrame({'a': [1., 2.]})
    result = compare(a, b)['a']
    assert result['max_error'] == 0.
    assert not result['equal'] and not result['mask_equal']
