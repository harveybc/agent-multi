import numpy as np
import pytest

from app.state_compatibility import compatibility, history_banks
from app.decision_attribution import state_view, continuation
from app.three_signal_learning import synthetic_env
from tests.unit.test_discrete_curve_native import curve_env
from tests.unit.test_double_tcn_probe import probe_recipe


def test_offsets_do_not_imply_choice_conflict():
    m=compatibility([[0,1,2],[5,6,7]])
    assert m['raw_rmse_floor']==pytest.approx(2.5)
    assert m['centered_rmse_floor']==0 and m['gap_rmse_floor']==0
    assert m['common_optimal_actions']==[2] and m['best_common_regret']==0
    assert not m['strict_reversal_pairs']


def test_reversed_optima_and_randomization_mean_bound():
    y=np.array([[0,2,0],[0,0,2]])
    m=compatibility(y)
    assert m['common_optimal_actions']==[]
    assert m['best_common_regret']==pytest.approx(1)
    assert m['deterministic_minimax_regret']==pytest.approx(2)
    for p in ([1,0,0],[0,1,0],[0,0,1],[0,.5,.5],[.2,.4,.4]):
        assert (y.max(1)-y@p).mean()>=m['best_common_regret']-1e-12


def test_q_conflict_can_coexist_with_shared_optimum():
    m=compatibility([[0,2,-1],[0,0,-1],[0,2,0]])
    assert m['gap_rmse_floor']>0 and m['common_optimal_actions']==[1]
    assert m['best_common_regret']==0
    lower=compatibility([[3,2,1],[3,1,2]])
    assert lower['strict_reversal_pairs']==[[1,2]]
    assert lower['common_optimal_actions']==[0] and lower['best_common_regret']==0
    same=compatibility([[1,1,1],[1,1,1]])
    assert same['gap_rmse_floor']==0 and same['common_optimal_actions']==[0,1,2]
    for bad in ([[1,2,3]],[[1,2],[2,3]],[[0,1,2],[0,float('nan'),2]]):
        with pytest.raises(ValueError):compatibility(bad)


def test_native_histories_use_identical_observations_without_parent_mutation(curve_env):
    env,regime=synthetic_env(probe_recipe(),curve_env.risk_config,17,1,.99)
    env.reset()
    while not env.engine.entry_gate(env.bar)[0]:env.step(0)
    before=state_view(env);banks=list(history_banks(env,(1,4)))
    assert state_view(env)==before and banks
    for lag,states,info in banks:
        assert lag in (1,4) and set(states)<= {'flat','long','short'}
        assert info['observation_equal']
        assert states['flat'].engine.pos is None
        for name,state in states.items():
            if name!='flat':
                assert state.engine.pos.direction==({'long':1,'short':-1}[name])
                assert state.engine.counts['entries']-before['counts']['entries']==1
        zero=np.zeros(len(regime),dtype=int)
        a=continuation(states['flat'],zero,0);b=continuation(states['flat'],zero,0)
        assert a==b
