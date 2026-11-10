import numpy as np
import pytest

from tools.audit_surface_shared_observations import observation_groups, sign_oracle


def test_merges_identical_windows_across_rows_but_separates_direction():
    windows = np.array([[[1., 2.]], [[1., 2.]], [[1., 3.]]], dtype=np.float32)
    records = [dict(row=0, direction='long'), dict(row=1, direction='long'),
               dict(row=0, direction='short'), dict(row=2, direction='long')]
    assert observation_groups(windows, records) == [[0, 1], [2], [3]]


def test_rejects_nonfinite_observation():
    with pytest.raises(ValueError, match='NONFINITE'):
        observation_groups(np.array([[[np.nan]]]), [dict(row=0, direction='long')])


def test_same_query_opposite_signs_is_infeasible():
    assert not sign_oracle([[1., 0.], [1., 0.]], [1., -1.])['feasible']


def test_same_sign_needs_no_large_exact_value_coefficients():
    got = sign_oracle([[.5, .5], [.5+1e-13, .5-1e-13]], [-.4, -.5])
    assert got['feasible'] and got['float32_wrong_actions'] == 0
    assert got['max_node_magnitude'] == pytest.approx(1.)


def test_separate_queries_can_have_opposite_actions():
    got = sign_oracle([[1., 0.], [0., 1.]], [1., -1.])
    assert got['feasible'] and got['float32_wrong_actions'] == 0
    assert got['minimum_float64_margin'] >= 1.-1e-8
