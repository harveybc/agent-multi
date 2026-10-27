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
