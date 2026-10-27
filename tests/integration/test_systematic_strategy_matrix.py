"""Real pinned-engine witnesses; paths supplied by the operator, never inferred."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from app.strategy_scenario_matrix import cells,WARMUP
from tools.systematic_strategy_matrix import setup,environment,run_cell,step,action_from_head
from tools.rl_subset_bridge import build_model


@pytest.fixture(scope='module')
def runtime():
    engine=os.environ.get('PINNED_ENGINE_ROOT');recipe=os.environ.get('PINNED_RECIPE_PATH')
    if not engine or not recipe:pytest.skip('explicit pinned synthetic runtime paths required')
    torch.set_num_threads(1)
    r,c=setup(Path(recipe),Path(engine))
    env=environment(next(cells()),c,(13,r['list']['n']))
    m=build_model(env,r,7,'cpu',dict(n_steps=8,batch_size=8,n_epochs=1))
    return r,c,m


@pytest.mark.parametrize('side',[-1,1])
def test_actual_head_and_engine_choose_correct_monotonic_direction(runtime,side):
    r,c,m=runtime
    cell=dict(before=side,after=side,vol_before=0.,vol_after=0.,switch=60,period=24,phase=0,cost='zero_cost')
    result=run_cell(cell,c,r,m,constant_sigma=True)
    for row in result['comparisons']:
        if row['state']=='FLAT':
            assert row['chosen']==(1 if side==1 else 2)
            assert row['values'][str(row['chosen'])]>0
            assert row['values'][str(2 if side==1 else 1)]<0


def test_sell_wins_when_buy_is_positive_but_lower(runtime):
    r,c,m=runtime
    env=environment(list(cells())[-1],c,(13,r['list']['n']))
    while env.bar<WARMUP+30:step(env,0)
    obs=env._observe()
    assert obs['action_mask'][1]==obs['action_mask'][2]==1
    action=action_from_head(m,obs,[0.,1.,2.,3.,4.])
    assert action[0]==2
    env.step(action)
    assert env.engine.pos is not None and env.engine.pos.direction==-1


def test_causal_scales_have_warmup_and_ignore_future_regime(runtime):
    r,c,_=runtime
    a=next(cells());b={**a,'after':1,'vol_after':.0006}
    left=environment(a,c,(13,r['list']['n']));right=environment(b,c,(13,r['list']['n']))
    assert np.isfinite(left.engine.sigma[WARMUP:]).all()
    assert np.isfinite(left.engine.sigma_rule[WARMUP:]).all()
    end=WARMUP+a['switch']+1
    np.testing.assert_array_equal(left.engine.sigma[:end],right.engine.sigma[:end])
    np.testing.assert_array_equal(left.engine.sigma_rule[:end],right.engine.sigma_rule[:end])


def test_explicit_json_midpoint_matches_existing_execution(runtime):
    from app.directional_risk_calibration import DirectionalRisk
    r,c,m=runtime
    config=Path(__file__).resolve().parents[2]/'configs/validation/directional_strategy_midpoint.json'
    params=DirectionalRisk.from_json(config)
    cell=list(cells())[-1]
    assert run_cell(cell,c,r,m)==run_cell(cell,c,r,m,parameters=params)


def test_broker_witnesses_use_candidate_directional_parameters(runtime):
    from app.directional_risk_calibration import DirectionalRisk,DirectionalOrders
    from tools.native_strategy_contract_study import witnesses
    r,c,_=runtime;params=DirectionalRisk((.25,.75,.2,.75,.25,.4))
    cases=witnesses(c,(13,r['list']['n']),lambda engine,inputs:DirectionalOrders(engine,inputs,params))
    assert len(cases)==22
    for case in cases:
        trade=case['trades'][0]
        assert trade['percentual_order_size']==(.2 if case['direction']==1 else .4)


@pytest.mark.parametrize('horizon',[1,4,12,24,48,72])
@pytest.mark.parametrize('side',[-1,1])
def test_multiscale_horizons_execute_native_commands(runtime,horizon,side):
    r,c,_=runtime
    cell=dict(drift_before=side*.00015,drift_after=side*.00015,
        vol_before=0.,vol_after=0.,switch=32,period=24,phase=0,cost='zero_cost',
        transition_bars=12,horizon=horizon,target_bars=240)
    result=run_cell(cell,c,r,None,constant_sigma=True)
    assert result['prescribed']
    for row in result['comparisons']:
        assert row['horizon']==horizon
        if row['state']=='FLAT':assert row['chosen']==(1 if side==1 else 2)


def test_independent_prescribed_checks_detect_wrong_sell_decoder(runtime,monkeypatch):
    from app.directional_risk_calibration import DirectionalOrders,DirectionalRisk
    from tools.systematic_strategy_matrix import prescribed_execution
    r,c,_=runtime
    env=environment(list(cells())[-1],c,(13,r['list']['n']),parameters=DirectionalRisk((.5,.5,1)*2))
    while env.bar<WARMUP+30:step(env,0)
    original=DirectionalOrders.decode
    def mutant(self,action):
        cmd,tp,sl,label=original(self,action)
        return (1 if cmd==2 else cmd),tp,sl,label
    monkeypatch.setattr(DirectionalOrders,'decode',mutant)
    with pytest.raises(ValueError,match='PRESCRIBED_'):
        prescribed_execution(env,[2],env.parameters)


def test_independent_cost_check_detects_removed_spread(runtime):
    from tools.systematic_strategy_matrix import outcome,check_costs
    r,c,_=runtime;env=environment({**list(cells())[-1],'cost':'configured'},c,(13,r['list']['n']))
    while env.bar<WARMUP+30:step(env,0)
    _,end=outcome(env,1,24,True)
    t=end.engine.trades[0];assert t['spread_usd']>0
    with pytest.raises(ValueError,match='INDEPENDENT_spread_usd'):
        check_costs({**t,'spread_usd':0},end.engine)


def test_invalid_output_is_rejected_without_state_change(runtime):
    r,c,_=runtime;env=environment(list(cells())[-1],c,(13,r['list']['n']))
    while env.bar<WARMUP+30:step(env,0)
    for action in ([1.2,.5,.5,.5],[np.nan,.5,.5,.5],[3,.5,.5,.5],[1,2,.5,.5]):
        before=(env.bar,env.engine.balance,len(env.engine.trades),dict(env.order_counts))
        with pytest.raises(ValueError):env.step(action)
        assert (env.bar,env.engine.balance,len(env.engine.trades),env.order_counts)==before


@pytest.mark.parametrize('mutation',['ignored_close','ignored_stop','calendar_bypass'])
def test_broker_contract_kills_execution_mutants(runtime,monkeypatch,mutation):
    from app.short_horizon.expert_guided.env import BarEnv
    from tools.native_strategy_contract_study import witnesses
    r,c,_=runtime
    if mutation=='ignored_close':
        original=BarEnv.step
        def step_mutant(self,b,action,*args,**kw):
            return original(self,b,0 if action==3 else action,*args,**kw)
        monkeypatch.setattr(BarEnv,'step',step_mutant)
    elif mutation=='ignored_stop':
        original=BarEnv._risk_hit
        def stop_mutant(self,*args,**kw):
            hit=original(self,*args,**kw)
            return None if hit is not None and hit[0]=='SL' else hit
        monkeypatch.setattr(BarEnv,'_risk_hit',stop_mutant)
    else:
        monkeypatch.setattr(BarEnv,'entry_gate',lambda *args:(True,'OK'))
    with pytest.raises(ValueError):witnesses(c,(13,r['list']['n']))


def test_removed_costs_in_engine_are_detected(runtime,monkeypatch):
    from app.short_horizon.expert_guided.env import BarEnv
    from tools.systematic_strategy_matrix import outcome
    r,c,_=runtime;env=environment({**list(cells())[-1],'cost':'configured'},c,(13,r['list']['n']))
    while env.bar<WARMUP+30:step(env,0)
    monkeypatch.setattr(BarEnv,'_fill_cost',lambda *args,**kw:dict(spread=0.,slippage=0.,commission=0.))
    with pytest.raises(ValueError,match='INDEPENDENT_'):outcome(env,1,4,True)
