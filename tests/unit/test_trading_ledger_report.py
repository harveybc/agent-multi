import pytest
from tools.trading_ledger_report import summarize,ledger_checks


def trade(gross=10.,cost=2.):
    return dict(direction=1,entry_time='2023-01-01T10:00:00+00:00',exit_time='2023-01-01T11:00:00+00:00',
        entry_price=100.,exit_price=100.+gross,units=1,stop_loss=90.,take_profit=120.,
        gross_usd=gross,net_usd=gross-cost,entry_cost_usd=cost/2,exit_cost_usd=cost/2,
        spread_usd=cost,slippage_usd=0.,commission_usd=0.,financing_usd=0.,ambiguous=False)


def test_wins_are_net_and_empty_direction_is_undefined():
    a=summarize([trade(1,2),trade(2,2),trade(4,2)])
    assert (a['wins'],a['losses'],a['breakeven'])==(1,1,1)
    assert a['win_pct']==pytest.approx(100/3)
    assert a['gross_positive_net_nonpositive']==2
    assert a['profit_factor']==2
    assert summarize([])['win_pct'] is None
    assert summarize([])['expectancy_usd'] is None


def test_checks_detect_wrong_direction_pnl_and_missing_protection():
    t=trade();r=dict(trades=1,net_profit_usd=8.,win_rate=1.)
    assert ledger_checks([t],r)=={}
    t['direction']=-1
    faults=ledger_checks([t],r)
    assert faults['gross_mismatch']==1 and faults['invalid_initial_protection']==1
