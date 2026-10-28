"""Context contrasts keep model size fixed and preserve the native queue."""
import copy

import pytest

from tests.unit.test_selected_rl_capacity_sequence import paired_request
from tests.unit.test_selected_rl_comparison import write


def test_v10_changes_only_stride_and_keeps_the_full_selected_binding(tmp_path):
    from tools import selected_rl_comparison as selected
    from tools.temporal_capacity_sweep import BASE_ARCH
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path, request = paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v9', experiment_seed=7,
                   temporal_encoder=BASE_ARCH, early_stopping=SETTINGS)
    write(path, request)
    old, _ = selected._build(path, tmp_path/'old')
    for stride in (1, 2, 4):
        new = dict(request, schema='selected_rl_comparison_request.v10',
                   observation_stride_bars=stride)
        write(path, new)
        body, _ = selected._build(path, tmp_path/f'new-{stride}')
        expected = copy.deepcopy(old['common_config'])
        expected['observation_stride_bars'] = stride
        assert body['common_config'] == expected
        assert body['common_config']['window_size'] == 13
        assert body['selected_input_binding'] == old['selected_input_binding']
        assert body['seeds'] == [7]


@pytest.mark.parametrize('stride', [True, 0, 3, 8, 1.0, '2'])
def test_context_stride_refuses_undeclared_values(tmp_path, stride):
    from tools import selected_rl_comparison as selected
    from tools.temporal_capacity_sweep import BASE_ARCH
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path, request = paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v10', experiment_seed=7,
                   temporal_encoder=BASE_ARCH, early_stopping=SETTINGS,
                   observation_stride_bars=stride)
    write(path, request)
    with pytest.raises(ValueError):
        selected._build(path, tmp_path/'bad')


def test_context_candidates_and_replication_keep_architecture_fixed():
    from tools import temporal_capacity_sweep as m
    spec = dict(schema=m.CONTEXT_SCHEMA)
    base = dict(m.BASE_ARCH, observation_stride_bars=2)
    assert m.stage_recipes(spec, 0, base) == [dict(base, observation_stride_bars=s) for s in (1, 4)]
    assert m.replications(base, base) == []
    runs = m.replications(dict(base, observation_stride_bars=4), base)
    assert [(r['seed'], r['role'], r['architecture']['observation_stride_bars']) for r in runs] == [
        (17, 'baseline', 2), (17, 'finalist', 4), (29, 'baseline', 2), (29, 'finalist', 4)]


def test_predecessor_is_pinned_and_must_have_all_cells_terminal(tmp_path):
    from tools import temporal_capacity_sweep as m
    from tools.curated_rl_campaign import digest
    source = tmp_path/'predecessor'; source.mkdir()
    write(source/'SPEC.json', {'identity': 'fixed'})
    spec = dict(predecessor=dict(root=str(source), spec_sha256=digest(source/'SPEC.json')))
    assert m.context_predecessor(spec) == 'WAIT_PREDECESSOR'
    write(source/'SCIENTIFIC_COMPLETE.json', {'state': 'SCIENTIFIC_COMPLETE', 'closed': 10, 'total': 10})
    write(source/'STATE.json', {'cells': []})
    with pytest.raises(ValueError):
        m.context_predecessor(spec)
    write(source/'SPEC.json', {'identity': 'changed'})
    with pytest.raises(ValueError):
        m.context_predecessor(spec)


def test_predecessor_accepts_only_authenticated_terminal_state(tmp_path):
    from tools import temporal_capacity_sweep as m
    from tools.curated_rl_campaign import digest
    source=tmp_path/'predecessor'; source.mkdir()
    cells=[dict(id='one'),dict(id='two')]
    pin=write(source/'SPEC.json',dict(cells=cells))
    state=dict(spec_sha256=pin,cells=[dict(cell=c,state='CLOSED') for c in cells])
    write(source/'STATE.json',state)
    write(source/'SCIENTIFIC_COMPLETE.json',dict(state='SCIENTIFIC_COMPLETE',state_sha256=digest(source/'STATE.json')))
    spec=dict(predecessor=dict(root=str(source),spec_sha256=pin))
    assert m.context_predecessor(spec)=='READY'
    state['cells'][0]['state']='RUNNING';write(source/'STATE.json',state)
    with pytest.raises(ValueError):m.context_predecessor(spec)


@pytest.mark.parametrize('promotion', [False, True])
@pytest.mark.parametrize('study',['context','window'])
def test_complete_context_queue_and_conditional_paired_replication(tmp_path,monkeypatch,promotion,study):
    from tools import temporal_capacity_sweep as m,selected_rl_comparison as selected
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    baseline=tmp_path/'baseline';bp=tmp_path/'baseline-request.json'
    write(bp,dict(request,schema='selected_rl_comparison_request.v9',experiment_seed=7,
        temporal_encoder=m.BASE_ARCH,early_stopping=SETTINGS))
    selected.materialize(bp,baseline)
    def result(node):
        chosen=(node['architecture'].get('window_size')==25 if study=='window' else node['architecture']['observation_stride_bars']==4)
        growth=-.001+(.0001 if promotion and chosen else 0)
        metric=dict(net_profit=growth*10000,total_return=growth,max_drawdown_pct=10,
            sharpe_ratio=.1,trades_closed=100,win_rate_pct=50,profit_factor=.9)
        return dict(train_growth=.001,validation_growth=growth,max_drawdown_pct=10,
            trades_closed=100,wall_seconds=4200,metrics=dict(train=metric,validation=metric))
    monkeypatch.setattr(m,'outcome',result)
    monkeypatch.setattr(m,'context_predecessor',lambda s:'READY')
    monkeypatch.setattr(m,'resources_ready',lambda *a:'READY')
    monkeypatch.setattr(m,'smoke',lambda *a:None)
    source=tmp_path/'source';source.mkdir();write(source/'SPEC.json',{})
    root=tmp_path/'sweep'
    m.prepare(path,root,baseline,m.digest(baseline/'CONTRACT.json'),source,m.digest(source/'SPEC.json'),study)
    calls=[]
    def native(r):
        calls.append(r);write(r/'CLOSE.json',{});return 0
    monkeypatch.setattr(m.selected,'run',native)
    monkeypatch.setattr(m,'context_predecessor',lambda s:'WAIT_PREDECESSOR')
    assert m.work(root)==0 and calls==[]
    assert m.read(root/'SWEEP_STATE.json')['trials']==[]
    assert m.status(root)['estimated_remaining_seconds'] is None
    units=root/'units';m.render(root,units,tmp_path/'private.env')
    assert (units/f'rl-temporal-{study}-worker.service').exists()
    assert not (units/'rl-temporal-capacity-worker.service').exists()
    monkeypatch.setattr(m,'context_predecessor',lambda s:'READY')
    for _ in range(7):assert m.work(root)==0
    state=m.read(root/'SWEEP_STATE.json')
    assert len(calls)==(6 if promotion else 2)
    assert state['state']=='ALL_TRIALS_CLOSED'
    assert all(t['architecture']['net_arch']==[8] for t in state['trials'])
    assert len(state['stages'])==1
    assert m.status(root)['estimated_remaining_seconds']==[0,0]
    assert m.status(root)['maximum_new_fits']==6
    assert len(m.read(root/'SUMMARY.json')['replication'])==(3 if promotion else 0)


def test_v10_rejects_architecture_change_even_if_allowed_by_capacity_schema(tmp_path):
    from tools import selected_rl_comparison as selected
    from tools.temporal_capacity_sweep import BASE_ARCH
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v10',experiment_seed=7,
        temporal_encoder=dict(BASE_ARCH,core_filters=8),early_stopping=SETTINGS,observation_stride_bars=1)
    write(path,request)
    with pytest.raises(ValueError,match='CONTEXT_REQUIRED'):selected._build(path,tmp_path/'bad')


@pytest.mark.parametrize('stride,n',[(1,13),(2,13),(4,13),(2,7),(2,25),
    (2,8),(2,9),(2,11),(2,15),(2,17),(2,18)])
def test_real_preprocessor_origin_weekend_and_future_invariance(stride,n):
    import importlib.util
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    root=os.environ.get('GYM_FX_CHECKOUT')
    if not root:pytest.skip('matching GYM_FX_CHECKOUT required')
    source=Path(root)/'preprocessor_plugins/feature_window_preprocessor.py'
    spec=importlib.util.spec_from_file_location('context_preprocessor_witness',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stamps=pd.date_range('2023-01-01',periods=800,freq='h',tz='UTC')
    stamps=stamps[stamps.dayofweek<5]
    frame=pd.DataFrame(dict(DATE_TIME=stamps,feat=np.arange(len(stamps),dtype=float)))
    cfg=dict(window_size=n,observation_stride_bars=stride,require_full_observation_window=True,
        feature_columns=['feat'],feature_scaling='none',feature_clip=0,include_price_window=False,
        include_agent_state=False)
    def observe(data,config):
        return module.Plugin(config).make_observation(data=data,step=300,bridge_state={},config=config)['features']
    obs=observe(frame,cfg)
    np.testing.assert_array_equal(obs[:,0],np.arange(299-(n-1)*stride,300,stride))
    assert obs.shape==(n,1) and obs[-1,0]==299
    scaled=dict(cfg,feature_scaling='rolling_zscore',feature_scaling_window=256)
    expected=observe(frame,scaled)
    changed=frame.copy();changed.loc[300:,'feat']=1e8
    np.testing.assert_array_equal(expected,observe(changed,scaled))
    changed.loc[299,'feat']+=100
    assert observe(changed,scaled)[-1,0]!=expected[-1,0]
