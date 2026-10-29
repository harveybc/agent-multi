import pytest

from tools.analyze_crossed_account_results import details, paired_clusters, skyline


def world(seed, ret, mirror=1):
    return dict(spec=dict(seed=seed, pattern='trend', mirror=mirror), return_pct=ret)


def test_paired_clusters_do_not_count_mirrors_as_independent():
    a = [world(1, 2), world(1, 4, -1), world(2, 6)]
    b = [world(1, 1), world(1, 1, -1), world(2, 2)]
    r = paired_clusters(a, b)
    assert r['clusters'] == 2 and r['mean_delta_pp'] == 3
    assert r['ci95_descriptive_pp'] == [2, 4]


def test_pair_mismatch_refused():
    with pytest.raises(ValueError, match='SUPPORT'):
        paired_clusters([world(1, 2)], [world(2, 2)])


def test_skyline_keeps_duplicates_and_tradeoffs_only():
    assert skyline({'a': [3, -3], 'b': [3, -3], 'c': [2, -2],
                    'd': [1, -4], 'e': [3, -4]}) == {'a', 'b', 'c'}


def test_forged_aggregate_rejected_before_reading_ledger():
    result = dict(worlds=[dict(orders=3, return_pct=1, drawdown_pct=2)],
                  objectives=[2, -2], mean_orders=3, eligible=True)
    with pytest.raises(ValueError, match='AGGREGATE_METRICS'):
        details(result)
