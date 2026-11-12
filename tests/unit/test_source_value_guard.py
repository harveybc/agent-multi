import numpy as np
import pytest
import torch

from app.optimal_action_sets import SetGuard, SourceValueGuard, correction_loss, error_targets
from app.fit_decision_preservation import project_step


def fixture():
    z=torch.tensor([[.5,.4,.1],[.1,.8,.1]],dtype=torch.double).log()
    native=torch.tensor([[.9,.6,1.16],[0.,1.,0.]],dtype=torch.double)
    return z,native,[{'role':'fit'}]*2


def test_rejects302_wrong_to_worse_even_when_optimal_mass_increases():
    z,y,records=fixture();after=torch.tensor([[.4,.48,.12],[.1,.8,.1]],dtype=torch.double).log()
    mask,w=error_targets(z,y,records);ix=torch.arange(2)
    assert correction_loss(after,mask,w,ix)<correction_loss(z,mask,w,ix)
    assert SetGuard(z,y,records).check(after)['passed']
    check=SourceValueGuard(z,y,records).check(after)
    assert not check['passed'] and check['source_value_regressions']==1


def test_allows_better_action_without_requiring_immediate_global_optimum():
    z,y,records=fixture();g=SourceValueGuard(z,y,records)
    after=z.clone();after[0,2]=1
    assert g.check(after)['passed'] and g.check(after)['source_value_regressions']==0
    # A source-worst action has no worse alternative; improvement stays legal.
    z[0]=torch.tensor([0.,2.,0.]);g=SourceValueGuard(z,y,records)
    after=z.clone();after[0,0]=3
    assert g.check(after)['passed'] and g.check(after)['suboptimal_actions']==1


@pytest.mark.parametrize('native,forbidden', [([1.,1.,0.],2),([1.,0.,1.],1)])
def test_retains_hold_equivalence_and_rejects_bad_close(native,forbidden):
    z=torch.tensor([[2.,0.,0.]]);g=SourceValueGuard(z,torch.tensor([native]),[{'role':'fit'}])
    after=z.clone();after[0,forbidden]=3
    assert not g.check(after)['passed']
    after=z.clone();other=next(i for i in (1,2) if i!=forbidden);after[0,other]=3
    assert g.check(after)['passed']


def test_no_monitor_constraints_no_all_tied_constraints_and_finite_check():
    z=torch.tensor([[2.,0.,0.],[2.,0.,0.],[0.,0.,3.]])
    y=torch.tensor([[.9,.6,1.16],[.9,.6,1.16],[1.,1.,1.]])
    r=[{'role':'fit'},{'role':'monitor'},{'role':'fit'}];g=SourceValueGuard(z,y,r)
    assert g.rows.tolist()==[0] and g.constraint_rows.tolist()==[0] and g.bad.tolist()==[1]
    after=z.clone();after[1,1]=3
    assert g.check(after)['passed']
    after[2,0]=float('nan')
    assert not g.check(after)['passed']


def test_native_tolerance_is_separate_from_logit_tolerance_and_fixed_source():
    z=torch.tensor([[2.,0.,0.]],dtype=torch.double)
    y=torch.tensor([[1.,1.-5e-9,1.-2e-8]],dtype=torch.double)
    g=SourceValueGuard(z,y,[{'role':'fit'}]);assert g.allowed.tolist()==[[True,True,False]]
    assert g.bad.tolist()==[2]
    y[0,2]=100
    assert not g.allowed[0,2]


def test_new_constraints_have_nonzero_feasible_improving_direction():
    z,y,r=fixture();g=SourceValueGuard(z,y,r)
    params=z.detach().clone().requires_grad_()
    jac=torch.autograd.functional.jacobian(lambda p:g.margins(p),params).reshape(len(g.floor),-1)
    mask,w=error_targets(z,y,r)
    loss=correction_loss(params,mask,w,torch.arange(2));grad=torch.autograd.grad(loss,params)[0]
    matrix=torch.cat([jac,-grad.flatten()[None]])
    bounds=torch.cat([g.buffer,torch.zeros(1,dtype=torch.double)])
    step,info=project_step(-grad.detach().numpy().flatten()*.001,matrix.numpy(),bounds.numpy())
    assert info['success'] and info['nonzero']
    after=params.detach()+torch.tensor(step).reshape(z.shape)
    assert g.check(after)['passed']
    assert correction_loss(after,mask,w,torch.arange(2))<loss
    assert np.min(matrix.numpy()@step-bounds.numpy())>=-1e-10
