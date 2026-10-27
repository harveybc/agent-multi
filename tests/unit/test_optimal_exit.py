import numpy as np
import pytest

from app.optimal_exit import targets,outcome


def test_wait_for_peak_then_close_before_stop_not_close_immediately():
    t=targets([0.,70.,100.,40.,-20.],-100.,100.)
    assert t['best_net']==100.
    assert t['oracle_close'].tolist()==[False,False,True,True,True]
    assert t['regret_close'].tolist()==[1.,.3,0.,.6,1.2]
    assert t['close_advantage'][0]<0<t['close_advantage'][2]


def test_hold_to_target_can_be_optimal_and_least_loss_is_still_optimal():
    assert not targets([10.,20.],30.,100.)['oracle_close'].any()
    t=targets([-60.,-10.,-40.],-100.,100.)
    assert t['best_net']==-10. and t['oracle_close'].tolist()==[False,True,True]


@pytest.mark.parametrize('actions',[[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
def test_dense_action_regrets_telescope_to_actual_exit_error(actions):
    case=dict(close_net=[10.,40.,20.],terminal_net=-50.,risk_usd=100.,direction=-1)
    t=targets(case['close_net'],case['terminal_net'],case['risk_usd'])
    total=0.
    for i,a in enumerate(actions):
        total+=t['reward_close' if a else 'reward_hold'][i]
        if a:break
    assert np.isclose(total,-outcome(case,actions)['regret_risk'])


def test_tie_earliest_exit_and_global_error_remembers_missed_peak():
    t=targets([50.,50.,10.],0.,10.)
    assert t['oracle_close'].tolist()==[True,True,True]
    assert t['regret_close'][2]==4.


@pytest.mark.parametrize('close,terminal,risk', [([],0,1),([np.nan],0,1),([1],np.inf,1),([1],0,0)])
def test_invalid_targets_rejected(close,terminal,risk):
    with pytest.raises(ValueError):targets(close,terminal,risk)
