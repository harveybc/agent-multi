import numpy as np
import pytest

from app.critic_coverage_drift import choose_panel, drift_metrics, nearest_fit_rms


def test_selection_is_deterministic_excludes_old_and_never_ranks_by_return():
    trace = [dict(row=i, year=2020+i//40, kind='flat_both_allowed', timestamp=i+100)
             for i in range(160)]
    a, counts = choose_panel(trace, {101,102})
    b, _ = choose_panel([dict(r, future_profit=-r['row']) for r in trace], {101,102})
    assert [r['row'] for r in a] == [r['row'] for r in b]
    assert len(a) == 48
    assert not {101,102}.intersection(r['timestamp'] for r in a)
    assert counts['2020:open_long']['selected'] == 0
    with pytest.raises(ValueError): choose_panel(trace+[trace[0]], set())
    with pytest.raises(ValueError): choose_panel([dict(trace[0], year=2024)], set())


def test_drift_is_zero_for_same_policy_and_measures_changed_optimal_sets():
    values = np.array([[0.,2.,2.], [3.,3.,0.]])
    same = drift_metrics(values, values, np.array([2,0]))
    assert same['gap_rmse'] == same['mean_actor_regret'] == 0
    changed = np.array([[4.,2.,2.], [3.,3.,0.]])
    out = drift_metrics(values, changed, np.array([2,0]))
    assert out['optimal_set_changed'] == 1
    assert out['mean_actor_regret'] == 1
    assert out['mean_actor_regret_under_source'] == 0
    with pytest.raises(ValueError): drift_metrics(values, changed, np.array([3,0]))


def test_nearest_distance_uses_train_windows_not_outcomes():
    x = np.array([[[1.,2.]],[[3.,4.]],[[10.,20.]]])
    distance = nearest_fit_rms(x, x[:2])
    np.testing.assert_allclose(distance[:2], 0, atol=1e-8)
    assert distance[2] == pytest.approx(np.sqrt((7**2+16**2)/2))
    with pytest.raises(ValueError): nearest_fit_rms(x, x[:0])
