import numpy as np
import pytest


def test_panels_are_fixed_disjoint_and_cover_directions():
    from app.multiscale_strategy_search import panel
    a,b=panel('calibration'),panel('confirmation')
    assert len(a)==len(b)==24 and a==panel('calibration')
    assert not {tuple(sorted(x.items())) for x in a}&{tuple(sorted(x.items())) for x in b}
    assert len({(np.sign(x['drift_before']),np.sign(x['drift_after'])) for x in a})==8


def test_objective_scaling_is_fixed_and_has_no_confirmation_input():
    from app.multiscale_strategy_search import score
    c={'sharpe':2.,'return_fraction':.1,'orders_per_day':.4}
    scales={'sharpe':2.,'return_fraction':.05,'orders_per_day':.2}
    assert score('sharpe',c,scales)==1
    assert score('sharpe_return',c,scales)==3
    assert score('sharpe_return_activity',c,scales)==5
    with pytest.raises(ValueError):score('sharpe',{**c,'sharpe':None},scales)


def test_final_audit_requires_every_finalist_matrix():
    from app.multiscale_strategy_search import require_final_evidence
    with pytest.raises(ValueError):require_final_evidence([],[],[])
    with pytest.raises(ValueError):require_final_evidence([{'id':str(i)} for i in range(9)],[],[])


def test_empty_trading_has_null_sharpe_not_success():
    from app.multiscale_strategy_search import aggregate
    row=dict(daily_returns=[0.,0.],trades=[],return_fraction=0.,weeks=2,business_days=10,
        ruined=False,net_profit_usd=0.,max_drawdown_pct=0.,effective_rr=[],barrier_breakeven=[],cost_pips=[])
    result=aggregate([row])
    assert result['components']['sharpe'] is None
    assert not result['eligible'] and 'UNDEFINED_SHARPE' in result['reasons']
    assert result['win_rate'] is None


def test_matrix_command_consumes_finalist_config_without_smoke_limit():
    from types import SimpleNamespace
    from tools.multiscale_deap_strategy import matrix_command,parameter_json
    a=SimpleNamespace(recipe='r',source_spec='s',engine_root='e',design='d',workers=6)
    cmd=matrix_command(a,'winner.json','fresh-matrix')
    assert cmd[cmd.index('--strategy-config')+1]=='winner.json'
    assert '--limit' not in cmd
    j=parameter_json([0.,1.,.2,1.,0.,.4])
    assert j['long']['order_fraction']==.2 and j['short']['sl_fraction']==0.


def test_final_audit_refuses_missing_matrix_even_with_nine_searches():
    from app.multiscale_strategy_search import require_final_evidence
    from app.rl_subset import digest
    searches=[dict(id=f'{name}{seed}',objective=name,genes=[.5,.5,1.]*2,fitness=1.,
        report={'mean_drawdown_pct':1.}) for name in ('sharpe','sharpe_return','sharpe_return_activity')
        for seed in (7,19,43)]
    confirmation=[{'id':r['id']} for r in searches]
    with pytest.raises(ValueError):require_final_evidence(searches,confirmation,[])
    m=[dict(genome_sha256=digest(searches[0]['genes']),terminal=True,cells=47628)]
    require_final_evidence(searches,confirmation,m)
    m[0]['cells']=12
    with pytest.raises(ValueError):require_final_evidence(searches,confirmation,m)


def test_no_final_audit_file_is_emitted_before_complete_evidence(tmp_path):
    from tools.multiscale_deap_strategy import emit_final
    with pytest.raises(ValueError):emit_final(tmp_path,{},[],[],[],[],[],{})
    assert not (tmp_path/'FINAL_AUDIT_REQUEST.md').exists()
    assert not (tmp_path/'FINAL_REPORT.md').exists()


def test_contract_success_cannot_select_a_fitness_or_qualify_learning(tmp_path):
    import json
    from app.rl_subset import digest
    from tools.multiscale_deap_strategy import emit_final
    report=dict(components=dict(sharpe=20.,return_fraction=.1,orders_per_day=.5),
        eligible=True,mean_drawdown_pct=1.,worst_drawdown_pct=2.,
        orders_per_week=2.5,mean_profit_usd=1000.,win_rate=.8,
        sides={'1':{'trades':10},'-1':{'trades':10}})
    searches=[dict(id=f'{name}{seed}',objective=name,genes=[.5,.5,1.]*2,
        fitness=1.,generation=100,stop_reason='MAX_GENERATIONS',report=report)
        for name in ('sharpe','sharpe_return','sharpe_return_activity')
        for seed in (7,19,43)]
    confirmation=[dict(id=r['id'],report=report) for r in searches]
    matrices=[dict(genome_sha256=digest(searches[0]['genes']),terminal=True,
        cells=47628,state='VERIFIED',failures=0)]
    emit_final(tmp_path,{},searches,confirmation,matrices,[],[],{})
    decision=json.loads((tmp_path/'DECISION.json').read_text())
    assert decision['recommended'] is None
    assert decision['state']=='STUDY_COMPLETE_OPTIMIZATION_UNQUALIFIED'
    assert decision['matrix_scope']=='EXECUTION_CONTRACT_ONLY'
    assert decision['learning_qualification']=='NOT_EVALUATED_BY_THIS_STUDY'
    assert not decision['financial_admission']
    assert not decision['learned_policy_admission']
    text=(tmp_path/'FINAL_REPORT.md').read_text()
    for required in ('LEARNING_QUALIFICATION_FAILED', 'deterministic',
                     'midpoint', 'bid/ask', 'not financial annualized performance'):
        assert required in text
    assert 'all-scenario verification' not in text
    request=(tmp_path/'FINAL_AUDIT_REQUEST.md').read_text()
    assert 'not a final optimized-strategy acceptance request' in request
