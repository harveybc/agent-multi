import copy
from app.hybrid_fitness import compare


def rows():
    return [dict(arm='A',seed=s,period=p,trades=12,weeks_with_decisions=4,
                 orders_per_week=3.,valid=True,ruined=False,max_entries_per_day=1,
                 mean_daily_log_return=.001,max_drawdown_pct=5.,initial_capital=10000.,
                 net_profit_usd=500.,sharpe_daily_sqrt365=1.,average_entries_per_business_day=.6)
            for s in (7,19,43) for p in ('TRAIN','INNER_A','OBS_2024')]


def test_selection_does_not_read_observation_year_and_requires_seeds():
    r=rows(); a=compare(r)
    for row in r:
        if row['period']=='OBS_2024': row['net_profit_usd']=1e9
    assert compare(r)==a
    assert a['rankings']['sharpe'][0]['arm']=='A'
    assert compare(r[:-3])['rankings']['sharpe']==[]


def test_no_activity_never_becomes_winner():
    r=rows()
    for row in r:
        row.update(trades=0,orders_per_week=0.)
    assert compare(r)['rankings']['sharpe']==[]
