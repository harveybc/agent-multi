from tools.report_action_ranking_contrast import trading_rows


def test_empty_side_is_not_reported_as_zero_win_probability():
    metrics=dict(net_profit_usd=1.,return_pct=.01,sharpe=.2,max_drawdown_pct=1.,trades=2,
        win_rate=.5,orders_per_week=.1,buy=dict(trades=0,wins=0),sell=dict(trades=2,wins=1),
        close_reasons=dict(AGENT_CLOSE=1))
    result=dict(arms=dict(control=dict(accounts=[dict(updates=1,eligible=False,metrics=dict(TRAIN=metrics))])))
    rows=trading_rows(result)
    assert len(rows)==1 and rows[0]['buy_win_rate'] is None
    assert rows[0]['sell_win_rate']==.5 and rows[0]['early_closes']==1
    assert not rows[0]['eligible']


def test_authenticated_export_creates_tables_and_nonempty_plots(tmp_path):
    from tools.report_action_ranking_contrast import export
    from tools.run_discrete_curve_probe import write,read
    from app.frozen_recipe_inputs import file_sha
    metrics=dict(net_profit_usd=1.,return_pct=.01,sharpe=.2,max_drawdown_pct=1.,trades=2,
        win_rate=.5,orders_per_week=.1,buy=dict(trades=0,wins=0),sell=dict(trades=2,wins=1),close_reasons={})
    rank=dict(rmse=.2,pairwise_accuracy=.5,mean_regret=.1,pairwise_comparisons=2)
    panel={r:dict(all={t:rank for t in ('greedy','stochastic')}) for r in ('fit','monitor')}
    result=dict(arms=dict(control=dict(accounts=[dict(updates=1,eligible=False,metrics={p:metrics for p in ('TRAIN','VALIDATION')})],
        calibration=[dict(updates=0,metrics=panel)])))
    write(tmp_path/'fit/RESULT.json',result)
    write(tmp_path/'fit/DONE.json',dict(identity='fixture',files={'RESULT.json':file_sha(tmp_path/'fit/RESULT.json')}))
    export(tmp_path,tmp_path/'reports')
    receipt=read(tmp_path/'reports/REPORT.json')
    for name,sha in receipt['files'].items():
        assert file_sha(tmp_path/'reports'/name)==sha
    assert (tmp_path/'reports/ACTION_REGRET.png').stat().st_size>1000
