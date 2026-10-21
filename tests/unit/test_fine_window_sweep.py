import copy

import pytest

from tests.unit.test_selected_rl_capacity_sequence import paired_request
from tests.unit.test_selected_rl_comparison import write


def test_fine_recipes_and_conditional_refinement():
    from tools import temporal_capacity_sweep as m
    spec=dict(schema=m.FINE_WINDOW_SCHEMA)
    base=dict(m.BASE_ARCH,observation_stride_bars=2,window_size=13)
    assert [a['window_size'] for a in m.stage_recipes(spec,0,base)]==[9,11,15,17]
    assert m.stage_recipes(spec,1,base)==[]
    for winner,expected in [(9,[8,10]),(11,[10,12]),(15,[14,16]),(17,[16,18])]:
        assert m.stage_recipes(spec,1,dict(base,window_size=winner))==[
            dict(base,window_size=n) for n in expected]


def test_local_windows_and_shared_seeds_do_not_depend_on_promotion():
    from tools import temporal_capacity_sweep as m
    spec=dict(schema=m.LOCAL_WINDOW_SCHEMA)
    base=dict(m.BASE_ARCH,observation_stride_bars=2,window_size=13)
    assert [a['window_size'] for a in m.stage_recipes(spec,0,base)]==[12,14]
    with pytest.raises(ValueError):m.stage_recipes(spec,1,base)
    recipes=m.confirmation_recipes(spec,dict(incumbent=dict(architecture=base)))
    assert [(r['architecture']['window_size'],r['seed']) for r in recipes]==[
        (n,seed) for seed in (17,29) for n in (13,12,14)]


def test_local_queue_runs_all_eight_fits_even_if_control_wins(tmp_path,monkeypatch):
    from tools import temporal_capacity_sweep as m,selected_rl_comparison as selected
    from tools import window_common_support as common
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    baseline=tmp_path/'baseline';bp=tmp_path/'baseline-request.json'
    write(bp,dict(request,schema='selected_rl_comparison_request.v9',experiment_seed=7,
        temporal_encoder=m.BASE_ARCH,early_stopping=SETTINGS))
    selected.materialize(bp,baseline)
    def result(node):
        metric=dict(net_profit=-100,total_return=-.01,max_drawdown_pct=10,
            sharpe_ratio=-.1,trades_closed=100,win_rate_pct=45,profit_factor=.9)
        return dict(train_growth=-.001,validation_growth=-.002,max_drawdown_pct=10,
            trades_closed=100,wall_seconds=4200,metrics=dict(train=metric,validation=metric))
    monkeypatch.setattr(m,'outcome',result)
    monkeypatch.setattr(m,'scored_outcome',lambda spec,node:result(node))
    monkeypatch.setattr(m,'context_predecessor',lambda s:'READY')
    monkeypatch.setattr(m,'resources_ready',lambda *a:'READY')
    monkeypatch.setattr(m,'smoke',lambda *a:None)
    ready=[False]
    monkeypatch.setattr(common,'publication_ready',lambda *a:ready[0])
    source=tmp_path/'source';source.mkdir();write(source/'SPEC.json',{})
    root=tmp_path/'local'
    m.prepare(path,root,baseline,m.digest(baseline/'CONTRACT.json'),source,
        m.digest(source/'SPEC.json'),'local-window')
    write(root/'COMMON_PUBLICATION_STATUS.json',{});write(root/'COMMON_REPORTS.json',{})
    calls=[]
    def native(r):calls.append(r);write(r/'CLOSE.json',{});return 0
    monkeypatch.setattr(m.selected,'run',native)
    for _ in range(12):assert m.work(root)==0
    assert len(calls)==8 and not (root/'SWEEP_COMPLETE.json').exists()
    assert m.read(root/'SWEEP_STATE.json')['state']=='WAIT_COMMON_PUBLICATION'
    ready[0]=True
    assert m.work(root)==0
    state=m.read(root/'SWEEP_STATE.json');summary=m.read(root/'SUMMARY.json')
    assert state['state']=='ALL_TRIALS_CLOSED' and len(calls)==8
    assert {(t['architecture']['window_size'],t['seed']) for t in state['trials']}=={
        (n,seed) for n in (12,13,14) for seed in (7,17,29)}-{(13,7)}
    assert m.status(root)['maximum_new_fits']==8
    assert m.status(root)['estimated_remaining_seconds']==[0,0]
    assert len(summary['window_comparisons'])==2
    assert all(len(row['paired_seeds'])==3 for row in summary['window_comparisons'])
    assert state['incumbent']['architecture']['window_size']==13
    spec=m.read(root/'SPEC.json');spec['maximum_new_fits']=7;write(root/'SPEC.json',spec)
    with pytest.raises(ValueError):m.plan(root)


def test_local_predecessor_requires_the_closed_fine_study(tmp_path):
    from tools import temporal_capacity_sweep as m
    source=tmp_path/'source';source.mkdir()
    pin=write(source/'SPEC.json',dict(schema=m.FINE_WINDOW_SCHEMA))
    spec=dict(schema=m.LOCAL_WINDOW_SCHEMA,predecessor=dict(root=str(source),spec_sha256=pin))
    state=dict(spec_sha256=pin,trials=[]);write(source/'SWEEP_STATE.json',state)
    summary=write(source/'SUMMARY.json',{})
    write(source/'SWEEP_COMPLETE.json',dict(schema=m.FINE_WINDOW_SCHEMA,state='ALL_TRIALS_CLOSED',
        spec_sha256=pin,state_sha256=m.digest(source/'SWEEP_STATE.json'),summary_sha256=summary,trials=0,successful=0))
    assert m.context_predecessor(spec)=='READY'
    write(source/'SUMMARY.json',dict(tampered=True))
    with pytest.raises(ValueError):m.context_predecessor(spec)


def test_v12_only_declared_window_changes(tmp_path):
    from tools import selected_rl_comparison as s,temporal_capacity_sweep as m
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v9',experiment_seed=7,
        temporal_encoder=m.BASE_ARCH,early_stopping=SETTINGS)
    write(path,request);base,_=s._build(path,tmp_path/'base')
    for n in range(8,19):
        write(path,dict(request,schema='selected_rl_comparison_request.v12',window_size=n))
        body,_=s._build(path,tmp_path/str(n))
        expected=copy.deepcopy(base['common_config']);expected['window_size']=n
        assert body['common_config']==expected
        assert body['selected_input_binding']==base['selected_input_binding']


@pytest.mark.parametrize('n',[True,7,19,11.,'11'])
def test_fine_request_rejects_other_windows(tmp_path,n):
    from tools import selected_rl_comparison as s,temporal_capacity_sweep as m
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    write(path,dict(request,schema='selected_rl_comparison_request.v12',experiment_seed=7,
        temporal_encoder=m.BASE_ARCH,early_stopping=SETTINGS,window_size=n))
    with pytest.raises(ValueError):s._build(path,tmp_path/'bad')


@pytest.mark.parametrize('bad',[dict(core_filters=8),dict(net_arch=[16])])
def test_fine_cannot_change_capacity(tmp_path,bad):
    from tools import selected_rl_comparison as s,temporal_capacity_sweep as m
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    write(path,dict(request,schema='selected_rl_comparison_request.v12',experiment_seed=7,
        temporal_encoder=dict(m.BASE_ARCH,**bad),early_stopping=SETTINGS,window_size=11))
    with pytest.raises(ValueError):s._build(path,tmp_path/'bad')


def test_common_suffix_score_anchors_before_first_kept_row(tmp_path):
    from tools.window_common_support import trace_score
    p=tmp_path/'trace.csv'
    p.write_text('timestamp,equity\n2024-01-01T00:00:00Z,90\n'
        '2024-01-01T01:00:00Z,81\n2024-01-02T00:00:00Z,90\n')
    result=trace_score(p,100,'UTC',1)
    assert result['rows']==2 and result['days']==2
    assert result['return_fraction']==pytest.approx(0)
    assert result['max_drawdown_pct']==pytest.approx(10)
    assert result['growth']==pytest.approx(0)
    assert result['anchor_equity']==90


def test_common_suffix_rejects_bad_equity_and_missing_support(tmp_path):
    from tools.window_common_support import trace_score,require_equal_support
    p=tmp_path/'trace.csv';p.write_text('timestamp,equity\n2024-01-01T00:00:00Z,100\n')
    a=trace_score(p,100,'UTC',0)
    with pytest.raises(ValueError):require_equal_support(a,dict(a,rows=2))
    with pytest.raises(ValueError):require_equal_support(a,dict(a,support_sha256='bad'))
    with pytest.raises(ValueError):trace_score(p,100,'UTC',1)
    p.write_text('timestamp,equity\n2024-01-01T00:00:00Z,nan\n')
    with pytest.raises(ValueError):trace_score(p,100,'UTC',0)


@pytest.mark.parametrize('promotion',[False,True])
def test_fine_sequence_is_finite_and_advances_without_approval(tmp_path,monkeypatch,promotion):
    from tools import temporal_capacity_sweep as m,selected_rl_comparison as selected
    from tools import window_common_support as common
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    baseline=tmp_path/'baseline';bp=tmp_path/'baseline-request.json'
    write(bp,dict(request,schema='selected_rl_comparison_request.v9',experiment_seed=7,
        temporal_encoder=m.BASE_ARCH,early_stopping=SETTINGS))
    selected.materialize(bp,baseline)
    def result(node):
        n=node['architecture'].get('window_size',13)
        growth=-.001+({11:.0001,12:.0002}.get(n,0) if promotion else 0)
        metric=dict(net_profit=growth*10000,total_return=growth,max_drawdown_pct=10,
            sharpe_ratio=.1,trades_closed=100,win_rate_pct=50,profit_factor=.9)
        return dict(train_growth=-.002,validation_growth=growth,max_drawdown_pct=10,
            trades_closed=100,wall_seconds=4200,metrics=dict(train=metric,validation=metric))
    monkeypatch.setattr(m,'outcome',result)
    monkeypatch.setattr(m,'scored_outcome',lambda spec,node:result(node))
    monkeypatch.setattr(m,'context_predecessor',lambda s:'READY')
    monkeypatch.setattr(m,'resources_ready',lambda *a:'READY')
    monkeypatch.setattr(m,'smoke',lambda *a:None)
    monkeypatch.setattr(common,'publication_ready',lambda *a:True)
    source=tmp_path/'source';source.mkdir();write(source/'SPEC.json',{})
    root=tmp_path/'sweep'
    m.prepare(path,root,baseline,m.digest(baseline/'CONTRACT.json'),source,m.digest(source/'SPEC.json'),'fine-window')
    write(root/'COMMON_PUBLICATION_STATUS.json',{});write(root/'COMMON_REPORTS.json',{})
    calls=[]
    def native(r):calls.append(r);write(r/'CLOSE.json',{});return 0
    monkeypatch.setattr(m.selected,'run',native)
    for _ in range(12):assert m.work(root)==0
    state=m.read(root/'SWEEP_STATE.json')
    assert state['state']=='ALL_TRIALS_CLOSED' and len(state['stages'])==2
    assert len(calls)==(10 if promotion else 4)
    assert len({(t['architecture']['window_size'],t['seed']) for t in state['trials']})==len(calls)
    assert m.status(root)['maximum_new_fits']==10
    assert m.status(root)['estimated_remaining_seconds']==[0,0]
    assert state['incumbent']['architecture']['window_size']==(12 if promotion else 13)


def test_common_readback_must_cover_current_closed_nodes(tmp_path):
    from tools import window_common_support as m
    write(tmp_path/'SPEC.json',{})
    spec=dict(baseline=dict(contract_sha256='base'))
    state=dict(trials=[dict(state='CLOSED',attempts=[dict(contract_sha256='new')])])
    assert not m.publication_ready(tmp_path,spec,state)
    body=dict(spec_sha256=m.digest(tmp_path/'SPEC.json'),contracts=['base'],warehouse=dict(pending=0,mismatch=0))
    body['reports_sha256']=write(tmp_path/'COMMON_REPORTS.json',{})
    write(tmp_path/'COMMON_PUBLICATION_STATUS.json',body)
    assert not m.publication_ready(tmp_path,spec,state)
    body['contracts']=['base','new'];write(tmp_path/'COMMON_PUBLICATION_STATUS.json',body)
    assert m.publication_ready(tmp_path,spec,state)
    body['warehouse']['pending']=1;write(tmp_path/'COMMON_PUBLICATION_STATUS.json',body)
    assert not m.publication_ready(tmp_path,spec,state)


@pytest.mark.parametrize('n,actor,critic',[(8,902,1318),(9,934,1382),(11,998,1510),
    (12,1030,1574),(14,1094,1702),(15,1126,1766),(17,1190,1894),(18,1222,1958)])
def test_fine_native_shape_gradient_reload(n,actor,critic):
    from tools import temporal_capacity_sweep as m
    result=m.smoke_model(dict(m.BASE_ARCH,window_size=n),'cpu')
    assert result['gradient_updates']==16 and result['policy_reload_equal']
    assert (result['actor_parameters'],result['critic_parameters'])==(actor,critic)


def test_fine_predecessor_is_the_closed_coarse_window_sweep(tmp_path):
    from tools import temporal_capacity_sweep as m
    source=tmp_path/'source';source.mkdir()
    pin=write(source/'SPEC.json',dict(schema=m.WINDOW_SCHEMA))
    spec=dict(schema=m.FINE_WINDOW_SCHEMA,predecessor=dict(root=str(source),spec_sha256=pin))
    state=dict(spec_sha256=pin,trials=[]);write(source/'SWEEP_STATE.json',state)
    summary=write(source/'SUMMARY.json',{})
    write(source/'SWEEP_COMPLETE.json',dict(schema=m.WINDOW_SCHEMA,state='ALL_TRIALS_CLOSED',
        spec_sha256=pin,state_sha256=m.digest(source/'SWEEP_STATE.json'),summary_sha256=summary,trials=0,successful=0))
    assert m.context_predecessor(spec)=='READY'
    write(source/'SUMMARY.json',dict(tampered=True))
    with pytest.raises(ValueError):m.context_predecessor(spec)
