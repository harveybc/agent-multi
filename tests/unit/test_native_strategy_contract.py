from types import SimpleNamespace

import pytest

from app.native_strategy_contract import RiskParameters, executed_behavior


def test_parameter_domain_and_midpoint():
    p = RiskParameters()
    assert p.levels([(0.25, 3), (0.5, 3)]) == (1.625, 1.75)
    for args in ({'tp_fraction':0}, {'sl_fraction':1}, {'order_fraction':0},
                 {'order_fraction':1.01}, {'tp_fraction':float('nan')}):
        with pytest.raises(ValueError): RiskParameters(**args)


def test_requested_close_does_not_prove_execution():
    counts = dict(NOP=1, HOLD_LONG=1, HOLD_SHORT=1, CLOSE_LONG=2, CLOSE_SHORT=2)
    trades = [dict(direction=1, close_reason='SL'), dict(direction=-1, close_reason='TP')]
    r = executed_behavior(trades, counts, dict(valid=True, orders_per_week=2))
    assert not r['eligible']
    assert set(r['reasons']) == {'NO_EXECUTED_CLOSE_LONG', 'NO_EXECUTED_CLOSE_SHORT'}
    for t in trades: t['close_reason'] = 'AGENT_CLOSE'
    assert executed_behavior(trades, counts, dict(valid=True, orders_per_week=2))['eligible']
    assert not executed_behavior(trades, counts, dict(valid=False, orders_per_week=2))['eligible']


@pytest.mark.parametrize('activity', [float('nan'), float('inf'), -1, '2', None, True])
def test_activity_must_be_a_finite_nonnegative_measurement(activity):
    with pytest.raises(ValueError, match='INVALID_ACTIVITY_MEASUREMENT'):
        executed_behavior([], {}, dict(valid=True, orders_per_week=activity))


@pytest.mark.parametrize('valid', ['true', 1, None])
def test_native_validity_is_not_coerced(valid):
    with pytest.raises(ValueError, match='INVALID_NATIVE_VALIDITY'):
        executed_behavior([], {}, dict(valid=valid, orders_per_week=2))


@pytest.mark.parametrize('count', [float('nan'), float('inf'), -1, .5, '1', True])
def test_action_evidence_requires_nonnegative_integer_counts(count):
    with pytest.raises(ValueError, match='INVALID_ACTION_COUNT:NOP'):
        executed_behavior([], dict(NOP=count), dict(valid=True, orders_per_week=2))


def test_activity_boundary_and_absent_actions():
    trades = [dict(direction=side, close_reason='AGENT_CLOSE') for side in (1, -1)]
    counts = dict(NOP=1, HOLD_LONG=1, HOLD_SHORT=1)
    assert executed_behavior(trades, counts, dict(valid=True, orders_per_week=2))['eligible']
    below = executed_behavior(trades, counts, dict(valid=True, orders_per_week=1.999999))
    assert below['reasons'] == ['ACTIVITY_BELOW_TWO']
    assert executed_behavior(trades, {}, dict(valid=True, orders_per_week=2))['reasons'] == [
        'NO_NOP', 'NO_HOLD_LONG', 'NO_HOLD_SHORT']
