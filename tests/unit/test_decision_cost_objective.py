from copy import deepcopy
import numpy as np
import pytest
import torch

from app.decision_cost_objective import fit_temperature,objective_parts,cost_objective,stable_added_mask,qualifies_extension
from app.native_head_warm_start import fit_step
from app.interior_adamw_repair import transactional_step
from app.training_forensics import tree_digest
from tests.unit.test_adamw_decision_preservation import setup
from tools.run_decision_cost_contrast import initialize_optimizer


def test_temperature_and_degenerate_contract():
    assert fit_temperature([[0.,1.,3.]])==2.5
    assert fit_temperature([[10.,11.,13.]])==2.5
    with pytest.raises(ValueError):fit_temperature(np.ones((2,3)))
    with pytest.raises(ValueError):fit_temperature([[0.,np.nan,1.]])


def test_cost_shift_invariance_hybrid_identity_and_improvement():
    q=torch.zeros((2,1,3),dtype=torch.float64);y=torch.tensor([[0.,1.,3.]],dtype=torch.float64)
    p=objective_parts(q,y,2.5);shifted=objective_parts(q+7,y,2.5)
    assert torch.equal(p['cost'],shifted['cost']) and torch.equal(p['hybrid'],p['mse']+p['cost'])
    q[:,:,2]=1
    assert objective_parts(q,y,2.5)['cost']<p['cost']


def test_cost_gradient_finite_difference_and_common_shift_null():
    q=torch.tensor([[[.1,.2,.3]],[[.4,.5,.6]]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[0.,1.,3.]],dtype=torch.float64)
    loss=cost_objective(q,y,'cost',2.5);g,=torch.autograd.grad(loss,q)
    for index in range(q.numel()):
        d=torch.zeros_like(q);d.flatten()[index]=1e-6
        numerical=(cost_objective(q+d,y,'cost',2.5)-cost_objective(q-d,y,'cost',2.5))/2e-6
        assert abs(float(numerical.detach())-float(g.flatten()[index]))<1e-8
    assert abs(float(g.sum()))<1e-12


def test_mse_adamw_default_parity():
    h,opt,z,y=setup();ref,ropt,_,_=setup()
    fit_step(h,opt,z,y);fit_step(ref,ropt,z,y,lambda q,y:cost_objective(q,y,'mse',1.))
    assert tree_digest(h.state_dict())==tree_digest(ref.state_dict())
    assert tree_digest(opt.state_dict())==tree_digest(ropt.state_dict())


def test_custom_objective_transaction_and_resume():
    h,opt,z,y=setup();loss=lambda q,y:cost_objective(q,y,'hybrid',1.)
    accept=lambda b,p:(p,dict(accepted=True))
    transactional_step(h,opt,z,y,accept,loss)
    before=tree_digest(h.state_dict());obefore=tree_digest(opt.state_dict())
    transactional_step(h,opt,z,y,lambda b,p:(None,dict(accepted=False)),loss)
    assert tree_digest(h.state_dict())==before and tree_digest(opt.state_dict())==obefore
    ref,ropt,_,_=setup();ref.load_state_dict(deepcopy(h.state_dict()));ropt.load_state_dict(deepcopy(opt.state_dict()))
    transactional_step(h,opt,z,y,accept,loss);transactional_step(ref,ropt,z,y,accept,loss)
    assert tree_digest(h.state_dict())==tree_digest(ref.state_dict())
    assert tree_digest(opt.state_dict())==tree_digest(ropt.state_dict())


def test_stability_excludes_protected_and_disagreeing_rows():
    a=np.array([[2.,1.,0.],[0.,2.,1.],[1.,0.,2.]])
    b=a.copy();b[1]=[2.,0.,1.]
    np.testing.assert_array_equal(stable_added_mask(np.array([0]),[a,b]),[False,False,True])


def test_gate_requires_cost_of_decisions_not_loss_or_numerical_repair():
    rows=[dict(fold=f,seed=s,robust=True,accepted=32,before=.2,after=.1,control=.15)
          for f in ('A','B') for s in (19,43)]
    assert qualifies_extension(rows)
    rows[0]['after']=.200001
    assert not qualifies_extension(rows)
    rows[0]['after']=.2
    assert qualifies_extension(rows)
    rows[1]['after']=.15
    assert not qualifies_extension(rows)


def test_optimizer_reset_changes_only_optimizer_and_preserves_inherited_mode():
    head,opt,z,y=setup();fit_step(head,opt,z,y)
    weights=tree_digest(head.state_dict());state=deepcopy(opt.state_dict())
    inherited,step=initialize_optimizer(head,state,3e-5,.01,'inherited')
    assert step==1 and tree_digest(inherited.state_dict())==tree_digest(state)
    fresh,step=initialize_optimizer(head,state,3e-5,.01,'fresh')
    assert step==0 and not fresh.state
    assert fresh.param_groups[0]['lr']==3e-5 and fresh.param_groups[0]['weight_decay']==.01
    assert tree_digest(head.state_dict())==weights and tree_digest(opt.state_dict())==tree_digest(state)
    with pytest.raises(ValueError):initialize_optimizer(head,state,3e-5,.01,'unknown')
    fit_step(head,fresh,z,y)
    assert {int(v['step']) for v in fresh.state.values()}=={1}
