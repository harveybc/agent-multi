import copy
import json
import numpy as np
import pytest

from app.decision_attribution import fork_env,continuation,state_view,reconcile
from tests.unit.test_discrete_curve_native import curve_env
from tests.unit.test_double_tcn_probe import probe_recipe
from app.three_signal_learning import synthetic_env


@pytest.mark.parametrize('direction',[1,-1])
def test_native_same_state_and_clone_isolation(curve_env,direction):
    env,regime=synthetic_env(probe_recipe(),curve_env.risk_config,17,direction,.99)
    env.reset();signals=np.where(regime>0,1,np.where(regime<0,2,0))
    while not env.engine.entry_gate(env.bar)[0]:env.step(0)
    env.step(1 if direction==1 else 2)
    before=copy.deepcopy(state_view(env));child=fork_env(env)
    json.dumps(before,allow_nan=False)
    child.engine.counts['entries']+=99
    assert state_view(env)==before
    a=continuation(env,signals,0);b=continuation(env,signals,0)
    assert a==b and state_view(env)==before
    if env.engine.pos:
        close=continuation(env,signals,2 if env.engine.pos.direction==1 else 1)
        assert close['first_effect']['command']==3 and close['first_effect']['closes']>=1


def test_chronological_reconciliation_not_isolated_sum():
    assert reconcile([3,-1,5],10,17)['error_usd']==0
    with pytest.raises(ValueError):reconcile([3,1],10,17)


def test_native_telescoping_and_entropy_convention(curve_env):
    env,regime=synthetic_env(probe_recipe(),curve_env.risk_config,21,1,.9)
    env.reset();zero=np.zeros(len(regime),dtype=int)
    p=np.tile([.5,.25,.25],(len(regime),1));alpha=.02
    before=state_view(env)
    actual=continuation(env,zero,0,p,alpha)
    trial=fork_env(env);discount=1.;expected=0.;first=True
    while not trial.finished:
        if not first:expected+=discount*alpha*(-np.log(.5))
        _,reward,_,duration=trial.step(0)
        expected+=discount*reward;discount*=trial.gamma**duration;first=False
    assert actual['soft_return']==pytest.approx(expected,abs=1e-12)
    assert state_view(env)==before
    base=continuation(env,zero,0)['net_usd'];previous=base;contributions=[]
    while not env.finished:
        action=1 if env.engine.entry_gate(env.bar)[0] and not env.engine.pos else 0
        if action:
            old=continuation(env,zero,0)['net_usd'];new=continuation(env,zero,action)['net_usd']
            assert old==pytest.approx(previous,abs=1e-8)
            contributions.append(new-old);previous=new
        env.step(action)
    reconcile(contributions,base,env.engine.balance-env.engine.cfg.capital)
