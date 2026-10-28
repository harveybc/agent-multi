import numpy as np

from env_plugins.optimal_exit_diagnostic import OptimalExitDiagnostic


def test_no_oracle_or_position_feedback_in_observation_and_reward_telescopes():
    cases=[dict(close_net=[0.,10.,20.,-10.],terminal_net=-30.,risk_usd=10.,direction=-1)]
    observations=[np.arange(4*3*2,dtype=np.float32).reshape(4,3,2)]
    env=OptimalExitDiagnostic(cases,observations)
    obs,_=env.reset(options={'case_index':0})
    assert set(obs)=={'features','action_mask'}
    assert np.array_equal(obs['features'],observations[0][0])
    rewards=[]
    for cmd in (0,0,0,4):
        _,r,done,_,info=env.step([cmd,.5,.5,.5]);rewards.append(r)
    assert done and sum(rewards)==-info['outcome']['regret_risk']==-3.


def test_future_label_mutation_cannot_change_present_observation():
    a=dict(close_net=[0.,10.],terminal_net=-10.,risk_usd=1.,direction=1)
    b=dict(a,close_net=[0.,100.])
    x=[np.ones((2,3,2),np.float32)]
    e=OptimalExitDiagnostic([a],x);f=OptimalExitDiagnostic([b],x)
    eo,_=e.reset();fo,_=f.reset()
    assert all(np.array_equal(eo[k],fo[k]) for k in eo)


def test_terminal_and_dense_rewards_rank_exits_identically():
    case=dict(close_net=[0.,5.,-10.],terminal_net=-20.,risk_usd=10.,direction=1)
    x=[np.zeros((3,3,2),np.float32)]
    sums=[]
    for mode in ('dense_regret','terminal_regret'):
        e=OptimalExitDiagnostic([case],x,mode);e.reset();total=0.
        for cmd in (0,3):
            _,r,done,_,info=e.step([cmd,.5,.5,.5]);total+=r
        assert done;sums.append(total)
    assert sums==[0.,0.]
