import copy
import json
from pathlib import Path

import pytest

from tests.unit.test_selected_rl_capacity_sequence import paired_request
from tests.unit.test_selected_rl_comparison import write


def test_new_request_only_changes_approved_architecture_and_seed(tmp_path):
    from tools import selected_rl_comparison as selected
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    arch=dict(net_arch=[8],branch_filters=2,core_filters=4,kernel_size=3)
    request.update(schema='selected_rl_comparison_request.v8',temporal_encoder=arch,early_stopping=SETTINGS)
    write(path,request);old,_=selected._build(path,tmp_path/'old')
    request.update(schema='selected_rl_comparison_request.v9',experiment_seed=17,
        temporal_encoder=dict(arch,core_filters=8,net_arch=[16],core_layers=2))
    write(path,request);body,_=selected._build(path,tmp_path/'new')
    expected=copy.deepcopy(old['common_config']);expected.update(net_arch=[16],temporal_core_filters=8,temporal_core_layers=2)
    assert body['common_config']==expected and body['seeds']==[17]
    assert body['selected_input_binding']==old['selected_input_binding']
    request['schema']='selected_rl_comparison_request.v8';request.pop('experiment_seed');write(path,request)
    with pytest.raises(ValueError):selected._build(path,tmp_path/'legacy')


@pytest.mark.parametrize('field,bad',[('core_layers',3),('branch_filters',16),('core_filters',32),('net_arch',[16,16]),('kernel_size',5),('core_layers',True)])
def test_undeclared_dimensions_fail_closed(tmp_path,field,bad):
    from tools import selected_rl_comparison as selected
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v9',experiment_seed=7,early_stopping=SETTINGS,
        temporal_encoder=dict(net_arch=[8],branch_filters=2,core_filters=4,kernel_size=3,core_layers=1))
    request['temporal_encoder'][field]=bad;write(path,request)
    with pytest.raises(ValueError):selected._build(path,tmp_path/'bad')


def test_depth_preserves_legacy_and_proves_origin_and_prefix():
    import torch
    from agent_plugins.temporal_features import PerFeatureTemporalExtractor
    from tests.unit.test_temporal_sac_features import ToyEnv
    torch.manual_seed(7);old=PerFeatureTemporalExtractor(ToyEnv.observation_space,2,4,3)
    torch.manual_seed(7);same=PerFeatureTemporalExtractor(ToyEnv.observation_space,2,4,3,core_layers=1)
    assert old.state_dict().keys()==same.state_dict().keys()
    x=torch.ones(2,13,3)
    assert torch.equal(old.temporal_forward(x),same.temporal_forward(x))
    extra=PerFeatureTemporalExtractor(ToyEnv.observation_space,2,4,3,core_layers=2)
    for p in extra.parameters():torch.nn.init.constant_(p,.1)
    y=extra.temporal_forward(x);changed=x.clone();changed[:,8:,:]+=1
    assert y.shape==(2,4,13)
    assert torch.equal(y[:,:,:8],extra.temporal_forward(changed)[:,:,:8])
    changed=x.clone();changed[:,-1,:]+=1
    assert not torch.equal(y[:,:,-1],extra.temporal_forward(changed)[:,:,-1])


def test_promotion_is_validation_gain_without_worse_gap_or_risk():
    from tools.temporal_capacity_sweep import improves
    old=dict(train_growth=.02,validation_growth=.01,max_drawdown_pct=10,trades_closed=100)
    assert improves(dict(old,validation_growth=.011),old)
    assert not improves(dict(old,validation_growth=.011,train_growth=.03),old)
    assert not improves(dict(old,validation_growth=.011,max_drawdown_pct=11),old)
    assert not improves(dict(old,validation_growth=.011,trades_closed=0),old)
    assert not improves(old,old)


def test_stage_changes_and_replication_are_exact():
    from tools.temporal_capacity_sweep import stage_architectures,BASE_ARCH,replications
    core=stage_architectures(0,BASE_ARCH)
    assert [a['core_filters'] for a in core]==[8,16]
    chosen=dict(core[1])
    heads=stage_architectures(1,chosen)
    assert [a['net_arch'] for a in heads]==[[16],[32]]
    assert all(a['core_filters']==16 and a['branch_filters']==2 for a in heads)
    branches=stage_architectures(2,heads[0]);assert [a['branch_filters'] for a in branches]==[4,8]
    depth=stage_architectures(3,branches[0]);assert depth==[dict(branches[0],core_layers=2)]
    assert replications(BASE_ARCH)==[]
    assert [(r['role'],r['seed']) for r in replications(depth[0])]==[
        ('baseline',17),('finalist',17),('baseline',29),('finalist',29)]


@pytest.mark.parametrize('arch',[
    dict(net_arch=[8],branch_filters=2,core_filters=8,kernel_size=3,core_layers=1),
    dict(net_arch=[32],branch_filters=8,core_filters=16,kernel_size=3,core_layers=2)])
def test_native_variable_temporal_updates_and_exact_reload(arch):
    from tools.temporal_capacity_sweep import smoke_model
    result=smoke_model(arch,'cpu')
    assert result['gradient_updates']==16 and result['policy_reload_equal']


@pytest.fixture
def sweep(tmp_path,monkeypatch):
    from tools import temporal_capacity_sweep as m,selected_rl_comparison as selected
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    baseline=tmp_path/'baseline';baseline_request=tmp_path/'baseline-request.json'
    write(baseline_request,dict(request,schema='selected_rl_comparison_request.v8',early_stopping=SETTINGS,
        temporal_encoder={k:v for k,v in m.BASE_ARCH.items() if k!='core_layers'}))
    selected.materialize(baseline_request,baseline)
    def result(node):
        a=node['architecture'];g=-.001+(.0006 if a['core_filters']==8 else .0007 if a['core_filters']==16 else 0)
        g+=.0001*(a['net_arch'][0]//16)+.00005*(a['branch_filters']//4)+.0001*(a['core_layers']-1)
        metric=dict(net_profit=g*10000,total_return=g,max_drawdown_pct=10-g*100,
            sharpe_ratio=.1,trades_closed=100,win_rate_pct=50,profit_factor=1.1)
        return dict(train_growth=.001,validation_growth=g,max_drawdown_pct=metric['max_drawdown_pct'],
            trades_closed=100,wall_seconds=4200,metrics=dict(train=dict(metric),validation=dict(metric)))
    monkeypatch.setattr(m,'outcome',result)
    monkeypatch.setattr(m,'resources_ready',lambda *a:'READY')
    monkeypatch.setattr(m,'smoke',lambda *a:None)
    root=tmp_path/'sweep';m.prepare(path,root,baseline,m.digest(baseline/'CONTRACT.json'))
    return m,root,result


def test_entire_adaptive_ladder_runs_and_closes_without_manual_decisions(sweep,monkeypatch):
    m,root,result=sweep;calls=[]
    def native(r):
        calls.append(r);write(r/'CLOSE.json',dict(state='synthetic-terminal'));return 0
    monkeypatch.setattr(m.selected,'run',native)
    for _ in range(12):assert m.work(root)==0
    assert len(calls)==11
    state=m.read(root/'SWEEP_STATE.json')
    assert state['state']=='ALL_TRIALS_CLOSED' and len(state['stages'])==4
    assert [t['phase'] for t in state['trials']]==[0,0,1,1,2,2,3,4,4,4,4]
    assert state['trials'][2]['architecture']['core_filters']==16
    assert state['trials'][4]['architecture']['net_arch']==[32]
    assert (root/'SWEEP_COMPLETE.json').exists() and (root/'RESULTS.csv').exists()
    assert m.work(root)==0 and len(calls)==11
    assert m.status(root)['estimated_remaining_seconds']==[0,0]
    summary=m.read(root/'SUMMARY.json')
    assert [r['seed'] for r in summary['replication']]==[7,17,29]
    assert summary['confirmation']=='PROMISING_SHARED_SEEDS_DEVELOPMENT_ONLY'


def test_bounded_failed_training_fresh_roots_then_next_candidate(sweep,monkeypatch):
    m,root,_=sweep;calls=[]
    def fail(r):
        calls.append(r);write(r/'STATUS.json',dict(tasks=[dict(split='train',state='FAILED_EXECUTION',attempt=1)]));return 2
    monkeypatch.setattr(m.selected,'run',fail)
    assert m.work(root)==2 and m.work(root)==2 and m.work(root)==2
    state=m.read(root/'SWEEP_STATE.json');first=state['trials'][0]
    assert first['state']=='FAILED_EXHAUSTED' and len(first['attempts'])==2
    assert calls[0]!=calls[1] and calls[2]!=calls[1]
    assert all((Path(n['root'])/'SWEEP_FAILURE.json').exists() for n in first['attempts'])
    assert state['trials'][1]['attempts'][0]['root']==str(calls[2])


def test_publication_wait_does_not_refit_and_validation_retry_reuses_root(sweep,monkeypatch):
    m,root,_=sweep;calls=[]
    def native(r):
        calls.append(r)
        write(r/'STATUS.json',dict(tasks=[dict(split=role,state='ENQUEUED',attempt=1) for role in ('train','validation')]))
        return 0
    monkeypatch.setattr(m.selected,'run',native)
    assert m.work(root)==0 and m.work(root)==0 and len(calls)==1
    assert m.read(root/'WORK_STATUS.json')['state']=='WAIT_PUBLICATION'
    write(calls[0]/'STATUS.json',dict(tasks=[dict(split='train',state='ENQUEUED',attempt=1),
        dict(split='validation',state='FAILED_EXECUTION',attempt=1)]))
    assert m.work(root)==0 and calls[0]==calls[1]
    assert len(m.read(root/'SWEEP_STATE.json')['trials'][0]['attempts'])==1


def test_resource_wait_and_existing_lock_never_dispatch(sweep,monkeypatch):
    import fcntl
    m,root,_=sweep
    monkeypatch.setattr(m,'resources_ready',lambda *a:'WAIT_RAM')
    monkeypatch.setattr(m.selected,'run',lambda *a:pytest.fail('must not train'))
    assert m.work(root)==0 and m.status(root)['estimated_remaining_seconds'] is None
    with (root/'.worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert m.work(root)==0
    assert m.read(root/'SWEEP_STATE.json')['trials']==[]


def test_pinned_plan_drift_refuses_before_dispatch(sweep):
    m,root,_=sweep;spec=m.read(root/'SPEC.json');spec['rule']={}
    write(root/'SPEC.json',spec)
    with pytest.raises(ValueError,match='DRIFT'):m.work(root)


def test_units_are_single_worker_finite_and_identified(sweep,tmp_path):
    m,root,_=sweep
    paths=m.render(root,root/'units',tmp_path/'private.env')
    assert len(paths)==4
    worker=(root/'units/rl-temporal-capacity-worker.service').read_text()
    publisher=(root/'units/rl-temporal-capacity-publication.service').read_text()
    assert 'MemoryMax=6G' in worker and 'RuntimeMaxSec=21600' in worker
    assert 'SWEEP_COMPLETE.json' in worker and 'BLOCKED.json' in worker
    assert '"work"' in worker and '"publish"' in publisher


def test_second_machine_discovers_closed_trials_without_new_training(sweep,monkeypatch,tmp_path):
    import subprocess
    from tools import collect_closed_raw_rl as collector
    m,root,_=sweep
    monkeypatch.setattr(m.selected,'run',lambda r:(write(r/'CLOSE.json',{}),0)[1])
    for _ in range(12):m.work(root)
    packet=m.export(root)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:json.dumps(packet).encode())
    calls=[]
    def collect(host,source,dest,state):
        calls.append(source);dest.mkdir(parents=True,exist_ok=True)
        write(dest/'CLOSE.json',dict(contract_sha256=m.digest(Path(source)/'CONTRACT.json')))
        return 0
    monkeypatch.setattr(collector,'collect',collect)
    archive=tmp_path/'physical';state=tmp_path/'collection'
    assert m.collect('worker.example.invalid',root,tmp_path/'runtime',archive,state)==0
    assert len(calls)==12
    assert m.read(state/'COLLECTION_COMPLETE.json')['copied_closed_runs']==12
    assert (archive/'RESULTS.csv').exists()
