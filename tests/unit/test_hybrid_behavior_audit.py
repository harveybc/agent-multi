import pytest
from tools.hybrid_behavior_audit import trade_summary


def trade(direction, hours, gross, costs, reason='AGENT_CLOSE'):
    return dict(direction=direction, entry_time='2023-01-01T00:00Z',
                exit_time=f'2023-01-01T{hours:02d}:00Z', close_reason=reason,
                gross_usd=gross,entry_cost_usd=costs/2,exit_cost_usd=costs/2,
                net_usd=gross-costs,units=100)


def test_cost_breakdown_empty_population_and_reconciliation():
    assert trade_summary([])['win_rate'] is None
    s=trade_summary([trade(1,1,10,2),trade(-1,4,-5,3,'SL')])
    assert s['gross_usd']==5 and s['cost_usd']==5 and s['net_usd']==0
    assert s['one_hour_share']==.5 and s['win_rate']==.5
    assert s['directions']=={'long':1,'short':1}


def test_reject_bad_cost_accounting_and_negative_duration():
    t=trade(1,1,10,2);t['net_usd']=10
    with pytest.raises(ValueError,match='TRADE_ACCOUNTING'):trade_summary([t])
    t=trade(1,1,10,2);t['exit_time']='2022-01-01T00:00Z'
    with pytest.raises(ValueError,match='TRADE_DURATION'):trade_summary([t])
