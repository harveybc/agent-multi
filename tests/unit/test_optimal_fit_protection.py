import numpy as np
import pytest
import torch
from app.role_routed_critic import RoleHeads
from app.optimal_fit_protection import optimal_support,optimal_pairs,complete_pair_jacobian,finite_quality,bounded_repair


def test_optimal_set_excludes_wrong_and_respects_ties():
    y=np.array([[2.,2.,0.],[0.,1.,2.],[1.,1.,1.]])
    q=np.array([[[2.,1.,0.],[2.,1.,0.],[0.,1.,2.]]]*2)
    allowed,eligible=optimal_support(y,{'a':q,'b':q.copy()})
    assert eligible.tolist()==[0]
    assert optimal_pairs(q,np.arange(3),allowed,eligible)==[(0,0,2)]
    other=q.copy();other[:,0]=[0.,2.,-1.]
    assert optimal_pairs(other,np.arange(3),allowed,eligible)==[(0,1,2)]
    assert optimal_support(y,{'a':q,'b':other})[1].tolist()==[0]
    other[:,0]=[0.,1.,2.]
    assert optimal_support(y,{'a':q,'b':other})[1].tolist()==[]


def test_complete_pair_manual_derivative_against_autograd():
    head=RoleHeads(torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.SiLU(),torch.nn.Linear(3,3)).double(),1,17)
    z=np.random.default_rng(9).normal(size=(2,5,4));pairs=[(0,1,0),(4,2,1),(2,0,2)]
    gaps,jac=complete_pair_jacobian(head.state_dict(),z,pairs)
    q=head(torch.tensor(z)).mean(2);actual=[];grads=[]
    for row,a,b in pairs:
        for twin in range(2):
            gap=q[twin,row,a]-q[:,row,b].min();actual.append(gap.item())
            grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(gap,list(head.parameters()),retain_graph=True)]).numpy())
    np.testing.assert_allclose(gaps,actual,atol=1e-13);np.testing.assert_allclose(jac,grads,atol=1e-13)
    with pytest.raises(ValueError):complete_pair_jacobian(head.state_dict(),z,[(0,1,1)])


def test_finite_quality_allows_tied_switch_and_wrong_action_improvement():
    y=np.array([[2.,2.,0.],[0.,1.,2.]])
    q=np.array([[[2.,1.,0.],[2.,1.,0.]]]*2);after=np.array([[[1.,3.,-1.],[0.,1.,3.]]]*2)
    allowed,eligible=optimal_support(y,{'x':q})
    def check(x,original=True,motion=.1):return finite_quality(y,{'x':x},{'x':q},allowed,eligible,1.,original,motion,1.)
    assert check(after)['passed']
    assert not check(after,False)['passed'] and not check(after,motion=0)['passed']
    assert not check(q)['passed']
    bad=after.copy();bad[:,0]=[0.,1.,4.]
    assert not check(bad)['passed']


def test_zero_boundary_and_bounded_finite_checks():
    def linear(center):return center.copy(),np.eye(1),center.copy(),np.zeros(1)
    def evaluation(d):return dict(passed=bool(d[0]>=0),motion=float(abs(d[0])))
    result,receipt=bounded_repair(np.array([1.]),linear,evaluation)
    np.testing.assert_array_equal(result,[1.]);assert receipt['accepted']
    assert bounded_repair(np.array([0.]),linear,evaluation)[1]['reason']=='zero_motion'
    bad=lambda center:(center.copy(),np.eye(1),center.copy(),np.array([-1.]))
    with pytest.raises(ValueError,match='OPTIMAL_THRESHOLD'):bounded_repair(np.array([1.]),bad,evaluation)
