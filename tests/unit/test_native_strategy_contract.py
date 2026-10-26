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
