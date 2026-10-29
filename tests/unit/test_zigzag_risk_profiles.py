from types import SimpleNamespace

import numpy as np
import pytest

from app.zigzag_risk_profiles import describe, profile, size_from_risk, verify_paths
from app.zigzag_opportunities import HOUR, OpportunityEvaluator, Tape


def fixture(mirror=False):
    o = np.array([100., 101., 103., 102., 100., 99.])
    hi, lo = o + 1, o - 1
    pivots = [(2, 1, 104.), (4, -1, 99.)]
    if mirror:
        o, hi, lo = 200-o, 200-lo, 200-hi
        pivots = [(i, -s, 200-p) for i, s, p in pivots]
    t = Tape(np.arange(6)*HOUR, o, hi, lo, np.zeros(6, int),
             np.arange(6)<5, np.full(6, 3), np.full(6, .1),
             np.zeros((2, 6)), [(0, 6)], {})
    z = SimpleNamespace(Params=lambda *x: x, hindsight_pivots=lambda *x: pivots)
    r = OpportunityEvaluator(t, z, 3).evaluate((2, 5, 1), True)['records']
    return t, r


def test_statistics_empty_nonfinite_and_population_std():
    assert describe([]) == {'n': 0, 'mean': None, 'std': None, 'min': None,
                            'p10': None, 'p25': None, 'median': None,
                            'p75': None, 'p90': None, 'p95': None, 'max': None}
    assert describe([1, 3])['std'] == 1
    with pytest.raises(ValueError, match='NONFINITE'):
        describe([np.nan])


def test_reference_is_mae_not_drawdown_and_boundary_is_explicit():
    t, r = fixture()
    assert verify_paths(t, r)['paths'] == 4
    p = profile(t, r)
    buy = p['directions']['buy']
    assert buy['oracle_all']['tp_pips']['max'] == pytest.approx(40000)
    assert buy['oracle_all']['mae_pips']['max'] == pytest.approx(10000)
    assert buy['oracle_all']['drawdown_pips']['max'] == pytest.approx(20000)
    assert buy['positive_calendar']['n'] == 2  # release exactly at flat
    assert p['directions']['sell']['positive_calendar']['n'] == 0
    assert p['support']['censored'] == 3


def test_mirror_swaps_profiles():
    a = profile(*fixture())
    b = profile(*fixture(True))
    assert a['directions']['buy'] == b['directions']['sell']
    assert a['directions']['sell'] == b['directions']['buy']


def test_exact_positive_tie_is_disclosed_not_silently_symmetric():
    t, r = fixture()
    r['score'][1, 0] = r['score'][0, 0]
    p = profile(t, r)
    assert p['support']['positive_exact_ties'] == 1
    assert p['directions']['buy']['oracle_winners']['n'] == 2
    assert p['directions']['sell']['oracle_winners']['n'] == 0


def test_empty_complete_support_stays_empty():
    t, r = fixture()
    r['pair_valid'][:] = False
    p = profile(t, r)
    assert p['support']['complete'] == 0
    assert p['support']['complete_nop'] == 0
    assert p['directions']['buy']['positive_calendar']['tp_pips']['mean'] is None


def test_record_digest_is_checked_before_loading_arrays(tmp_path, monkeypatch):
    import json
    import tools.zigzag_risk_profiles as runner

    (tmp_path/'SPEC.private.json').write_text(json.dumps(dict(future_pivots=3, horizons=[3, 6, 12, 24])))
    (tmp_path/'bad.npz').write_bytes(b'changed')
    monkeypatch.setattr(runner, 'identity', lambda spec: 'identity')
    monkeypatch.setattr(runner, 'load_tape', lambda spec: fixture()[0])
    monkeypatch.setattr(runner, 'authenticated', lambda *a: dict(rows=[
        dict(horizon=h, records_file='bad.npz', records_sha256='wrong') for h in [3, 6, 12, 24]]))
    with pytest.raises(ValueError, match='RECORD_HASH'):
        runner.build(tmp_path)


def test_margin_cap_and_loss_budget_over_grid():
    for budget in [0., 1., 10., 100., 1000.]:
        for loss in [1., 17., 190.]:
            for quality in [0., .2, .8, 1.]:
                s = size_from_risk(10000., .1, budget, loss, quality)
                assert 0 <= s['q'] <= 1
                assert 0 <= s['margin'] <= 1000
                assert s['estimated_loss'] <= budget + 1e-12


@pytest.mark.parametrize('field', ['mae', 'net', 'duration', 'release'])
def test_corrupted_path_rejected(field):
    t, r = fixture()
    r[field][0, 0] += 1
    with pytest.raises(ValueError):
        verify_paths(t, r)


def test_q_uses_rel_volume_once_and_respects_budget():
    assert size_from_risk(10000, .1, 100, 100, 1)['margin'] == 1000
    s = size_from_risk(10000, .1, 30, 100, 1)
    assert s['q'] == pytest.approx(.3)
    assert s['margin'] == pytest.approx(300)
    assert s['estimated_loss'] <= 30
    assert size_from_risk(10000, .1, 30, 100, .5)['q'] == pytest.approx(.15)
    assert size_from_risk(10000, .1, 30, 100, 1, minimum_q=.4)['q'] == 0
    assert size_from_risk(10000, .1, 0, 100, 1)['q'] == 0


@pytest.mark.parametrize('args', [
    (10000, .1, 30, 0, 1), (10000, .1, 30, 100, 1.1),
    (10000, .1, -1, 100, 1), (0, .1, 30, 100, 1),
    (10000, 2, 30, 100, 1), (10000, .1, np.nan, 100, 1),
])
def test_invalid_sizing_rejected(args):
    with pytest.raises(ValueError):
        size_from_risk(*args)
