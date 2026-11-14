from copy import deepcopy
import numpy as np
import pytest
import torch

from app.interior_adamw_repair import repair_direction,transactional_step,extension_gate
from app.training_forensics import tree_digest
from tests.unit.test_adamw_decision_preservation import setup
from app.native_head_warm_start import fit_step


def test_interior_target_keeps_fixed_acceptance_guard():
    linear=lambda p:(np.array([p[0]]),np.array([[1.]]),p)
    evaluate=lambda p:dict(passed=bool(p[0]-1e-7>=1e-5),motion=float(np.linalg.norm(p)))
    d=np.array([-1e-4]);margin=np.array([1e-5])
    tangent=repair_direction(d,linear,evaluate,margin,1.,1)
    interior=repair_direction(d,linear,evaluate,margin,2.,3)
    assert not tangent['accepted'] and interior['accepted']
    assert interior['delta'][0]>=2e-5-1e-12


def test_nonlinear_relinearization_without_losing_norm_bound():
    linear=lambda p:(np.array([p[0]-p[0]**2]),np.array([[1-2*p[0]]]),p)
    evaluate=lambda p:dict(passed=bool(p[0]-p[0]**2>=.1),motion=float(np.linalg.norm(p)))
    r=repair_direction(np.array([-.1]),linear,evaluate,np.array([.1]),1.01,3)
    assert r['accepted'] and len(r['repairs'])>1 and np.linalg.norm(r['delta'])<=.4


def test_cap_zero_and_nonfinite_fail_closed():
    linear=lambda p:(np.array([p[0]]),np.array([[1.]]),p)
    evaluate=lambda p:dict(passed=True,motion=float(np.linalg.norm(p)))
    assert not repair_direction(np.array([.01]),linear,evaluate,np.array([1.]),2.,3)['accepted']
    for d in (np.array([0.]),np.array([np.nan])):
        with pytest.raises(ValueError):repair_direction(d,linear,evaluate,np.array([1.]),2.,3)


@pytest.mark.parametrize('kind',['reject','exception'])
def test_transaction_rolls_back_nonempty_optimizer(kind):
    h,opt,z,y=setup();fit_step(h,opt,z,y)
    before=tree_digest(h.state_dict());obefore=tree_digest(opt.state_dict())
    def repair(*args):
        if kind=='exception':raise RuntimeError('injected')
        return None,dict(accepted=False)
    if kind=='exception':
        with pytest.raises(RuntimeError):transactional_step(h,opt,z,y,repair)
    else:assert not transactional_step(h,opt,z,y,repair)['accepted']
    assert tree_digest(h.state_dict())==before and tree_digest(opt.state_dict())==obefore


def test_transaction_accepts_actual_proposal_and_resume_exact():
    h,opt,z,y=setup();ref,ropt,_,_=setup()
    repair=lambda before,proposed:(proposed,dict(accepted=True))
    transactional_step(h,opt,z,y,repair);fit_step(ref,ropt,z,y)
    assert tree_digest(h.state_dict())==tree_digest(ref.state_dict())
    assert tree_digest(opt.state_dict())==tree_digest(ropt.state_dict())
    resumed,r_opt,_,_=setup();resumed.load_state_dict(deepcopy(h.state_dict()));r_opt.load_state_dict(deepcopy(opt.state_dict()))
    for _ in range(2):
        transactional_step(h,opt,z,y,repair);transactional_step(resumed,r_opt,z,y,repair)
    assert tree_digest(h.state_dict())==tree_digest(resumed.state_dict())
    assert tree_digest(opt.state_dict())==tree_digest(r_opt.state_dict())


def test_extension_is_automatic_but_not_loss_only():
    arms=[dict(robust=True,learning=True,regret_improved=True,accepted=32) for _ in range(4)]
    assert extension_gate(arms)
    arms[0]['regret_improved']=False
    assert extension_gate(arms)
    arms[1]['regret_improved']=False
    assert not extension_gate(arms)
    assert not extension_gate(arms[:3])
    arms[1]['regret_improved']=True;arms[0]['robust']=False
    assert not extension_gate(arms)


def test_false_acceptance_of_zero_motion_rolls_back():
    h,opt,z,y=setup();before=tree_digest(h.state_dict());obefore=tree_digest(opt.state_dict())
    with pytest.raises(ValueError,match='ZERO_MOTION'):
        transactional_step(h,opt,z,y,lambda b,p:(b,dict(accepted=True)))
    assert tree_digest(h.state_dict())==before and tree_digest(opt.state_dict())==obefore
