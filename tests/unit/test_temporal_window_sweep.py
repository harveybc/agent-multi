import copy
import pytest
from tests.unit.test_selected_rl_capacity_sequence import paired_request
from tests.unit.test_selected_rl_comparison import write


def test_window_request_changes_only_declared_window(tmp_path):
    from tools import selected_rl_comparison as s,temporal_capacity_sweep as m
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v9',experiment_seed=7,
        temporal_encoder=m.BASE_ARCH,early_stopping=SETTINGS)
    write(path,request);base,_=s._build(path,tmp_path/'base')
    for n in (7,13,25):
        write(path,dict(request,schema='selected_rl_comparison_request.v11',window_size=n))
        body,_=s._build(path,tmp_path/str(n))
        expected=copy.deepcopy(base['common_config']);expected['window_size']=n
        assert body['common_config']==expected
        assert body['selected_input_binding']==base['selected_input_binding']


@pytest.mark.parametrize('n',[True,0,6,26,7.,'7'])
def test_undeclared_window_is_rejected(tmp_path,n):
    from tools import selected_rl_comparison as s,temporal_capacity_sweep as m
    from agent_plugins.fixed_budget_monitor import SETTINGS
    path,request=paired_request(tmp_path)
    request.update(schema='selected_rl_comparison_request.v11',experiment_seed=7,
        temporal_encoder=m.BASE_ARCH,early_stopping=SETTINGS,window_size=n)
    write(path,request)
    with pytest.raises(ValueError):s._build(path,tmp_path/'bad')


def test_window_recipes_keep_stride_and_encoder_fixed():
    from tools import temporal_capacity_sweep as m
    base=dict(m.BASE_ARCH,observation_stride_bars=2,window_size=13)
    assert m.stage_recipes(dict(schema=m.WINDOW_SCHEMA),0,base)==[
        dict(base,window_size=n) for n in (7,25)]
    assert m.replications(base,base)==[]


@pytest.mark.parametrize('n,actor,critic',[(7,870,1254),(13,1062,1638),(25,1446,2406)])
def test_native_actual_window_gradient_reload_and_size(n,actor,critic):
    from tools import temporal_capacity_sweep as m
    result=m.smoke_model(dict(m.BASE_ARCH,window_size=n),'cpu')
    assert result['gradient_updates']==16 and result['policy_reload_equal']
    assert (result['actor_parameters'],result['critic_parameters'])==(actor,critic)


def test_completed_sweep_predecessor_and_tampered_state(tmp_path):
    from tools import temporal_capacity_sweep as m
    p=tmp_path/'source';p.mkdir()
    pin=write(p/'SPEC.json',dict(schema=m.CONTEXT_SCHEMA))
    spec=dict(schema=m.WINDOW_SCHEMA,predecessor=dict(root=str(p),spec_sha256=pin))
    assert m.context_predecessor(spec)=='WAIT_PREDECESSOR'
    state=dict(spec_sha256=pin,trials=[])
    write(p/'SWEEP_STATE.json',state);summary_pin=write(p/'SUMMARY.json',{})
    write(p/'SWEEP_COMPLETE.json',dict(schema=m.CONTEXT_SCHEMA,state='ALL_TRIALS_CLOSED',
        spec_sha256=pin,state_sha256=m.digest(p/'SWEEP_STATE.json'),summary_sha256=summary_pin,
        trials=0,successful=0))
    assert m.context_predecessor(spec)=='READY'
    write(p/'SWEEP_STATE.json',dict(state,forged=True))
    with pytest.raises(ValueError):m.context_predecessor(spec)
