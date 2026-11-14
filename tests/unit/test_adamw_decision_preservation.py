from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from app.adamw_decision_preservation import protected_step
from app.native_head_warm_start import configure_scope,fit_step
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest


def setup():
    h=RoleHeads(nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,3)),1,19)
    opt,_=configure_scope(h,'full_head',3e-5,.01)
    torch.manual_seed(3)
    return h,opt,torch.randn(2,32,208),torch.randn(32,3)


def observation(value):return dict(regret=value,actions=np.array([0,1,2],dtype=np.uint8))


def test_full_acceptance_matches_actual_adamw_exactly():
    h,opt,z,y=setup();ref,ropt,_,_=setup()
    fit_step(ref,ropt,z,y)
    r=protected_step(h,opt,z,y,lambda:observation(0.),0.)
    assert r['accepted'] and r['scale']==1 and r['delta_norm']>0
    assert tree_digest(h.state_dict())==tree_digest(ref.state_dict())
    assert tree_digest(opt.state_dict())==tree_digest(ropt.state_dict())


def test_reduced_step_uses_real_displacement_and_keeps_proposed_moments():
    h,opt,z,y=setup();before=deepcopy(h.state_dict());ref,ropt,_,_=setup()
    fit_step(ref,ropt,z,y);proposal=deepcopy(ref.state_dict())
    distance=lambda:float(torch.cat([(v-before[k]).flatten() for k,v in h.state_dict().items()]).double().norm())
    full=float(torch.cat([(v-before[k]).flatten() for k,v in proposal.items()]).double().norm())
    r=protected_step(h,opt,z,y,lambda:observation(distance()),full*.55,tolerance=0)
    assert r['accepted'] and r['scale']==.5 and len(r['trial_regrets'])==2
    assert all(torch.equal(v,before[k]+.5*(proposal[k]-before[k])) for k,v in h.state_dict().items())
    assert tree_digest(opt.state_dict())==tree_digest(ropt.state_dict())


@pytest.mark.parametrize('mode',['reject','nan','exception'])
def test_rejection_and_errors_restore_nonempty_optimizer(mode):
    h,opt,z,y=setup();fit_step(h,opt,z,y)
    before=deepcopy(h.state_dict());optimizer=deepcopy(opt.state_dict());calls=[]
    def evaluate():
        calls.append(1)
        if len(calls)==1:return observation(0.)
        if mode=='exception':raise RuntimeError('injected evaluator error')
        return observation(float('nan') if mode=='nan' else 1.)
    if mode=='reject':
        r=protected_step(h,opt,z,y,evaluate,0.)
        assert not r['accepted'] and r['scale']==0 and len(r['trial_regrets'])==8
    else:
        with pytest.raises((ValueError,RuntimeError)):protected_step(h,opt,z,y,evaluate,0.)
    assert tree_digest(h.state_dict())==tree_digest(before)
    assert tree_digest(opt.state_dict())==tree_digest(optimizer)


def test_zero_motion_is_not_success(monkeypatch):
    import app.adamw_decision_preservation as module
    h,opt,z,y=setup()
    monkeypatch.setattr(module,'fit_step',lambda *args:False)
    r=protected_step(h,opt,z,y,lambda:observation(0.),0.)
    assert not r['accepted'] and r['rounded_zero'] and r['delta_norm']==0


def test_resume_protected_optimizer_exact():
    h,opt,z,y=setup();evaluate=lambda:observation(0.)
    for _ in range(3):protected_step(h,opt,z,y,evaluate,0.)
    resumed,ropt,_,_=setup();resumed.load_state_dict(deepcopy(h.state_dict()));ropt.load_state_dict(deepcopy(opt.state_dict()))
    for _ in range(3):
        protected_step(h,opt,z,y,evaluate,0.);protected_step(resumed,ropt,z,y,evaluate,0.)
    assert tree_digest(h.state_dict())==tree_digest(resumed.state_dict())
    assert tree_digest(opt.state_dict())==tree_digest(ropt.state_dict())


def test_invalid_contract_and_infeasible_start_refused():
    h,opt,z,y=setup()
    for scales in ([],[.5,1],[1,0],[1,float('nan')]):
        with pytest.raises(ValueError):protected_step(h,opt,z,y,lambda:observation(0.),0.,scales=scales)
    with pytest.raises(ValueError):protected_step(h,opt,z,y,lambda:observation(1.),0.)
