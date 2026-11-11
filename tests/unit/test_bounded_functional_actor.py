import pytest
import torch

from app.bounded_functional_actor import BUDGET, admissible, backtrack, budget_metrics, desired_margins, projected_direction


def test_independent_projection_and_zero():
    j=torch.eye(4,dtype=torch.double).reshape(2,2,4)
    target=torch.tensor([[1.,-1.],[2.,-2.]],dtype=torch.double)
    d,m=projected_direction(j,target,torch.ones(2),1e-8)
    torch.testing.assert_close(d,target.flatten(),atol=1e-7,rtol=1e-7)
    assert m['residual_norm']<m['target_norm']
    zero,_=projected_direction(j,target*0,torch.ones(2))
    assert zero.abs().max()==0


def test_rank_deficient_and_duplicate_weights():
    j=torch.tensor([[[1.],[0.]],[[1.],[0.]]],dtype=torch.double)
    desired=torch.tensor([[1.,0.],[-1.,0.]],dtype=torch.double)
    d,_=projected_direction(j,desired,torch.tensor([2.,1.]),1e-8)
    torch.testing.assert_close(d,torch.tensor([1/3],dtype=torch.double),atol=1e-7,rtol=1e-7)
    repeated,_=projected_direction(j[[0,0,1]],desired[[0,0,1]],torch.ones(3),1e-8)
    torch.testing.assert_close(d,repeated,atol=1e-7,rtol=1e-7)
    zero,m=projected_direction(j*0,desired,torch.ones(2))
    assert m['zero_sensitivity'] and zero.abs().max()==0


def test_common_jacobian_scale():
    j=torch.eye(4,dtype=torch.double).reshape(2,2,4);t=torch.ones(2,2,dtype=torch.double)
    a,_=projected_direction(j,t,torch.ones(2));b,_=projected_direction(j*3,t,torch.ones(2))
    torch.testing.assert_close(a,b*3)


def test_target_rms_and_trust_boundary():
    g=torch.tensor([[1.,2.,-3.],[2.,1.,-3.]],dtype=torch.double);w=torch.tensor([2.,1.])
    d=desired_margins(g,w,.002)
    torch.testing.assert_close((d.square().mean(1)*w).sum()/w.sum(),torch.tensor(.002**2,dtype=torch.double))
    z=torch.zeros(2,3,dtype=torch.double);theta=torch.ones(4,dtype=torch.double)
    m=budget_metrics(z,z,z,theta,theta)
    assert admissible(m,1.,1.) and not admissible(m,1.,1.1)
    for k in m:
        invalid=dict(m);invalid[k]=BUDGET[k]*1.01
        assert not admissible(invalid,1.,.9)
    assert not admissible(m,1.,float('nan'))


def test_invalid_projection():
    j=torch.ones(2,2,3);d=torch.ones(2,2)
    with pytest.raises(ValueError):projected_direction(j,d,torch.tensor([1.,-1.]))
    with pytest.raises(ValueError):projected_direction(j*float('nan'),d,torch.ones(2))


def test_backtracking_and_restore_on_rejection_or_exception():
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.double))
    result=backtrack([p],torch.tensor([1.],dtype=torch.double),lambda:dict(accepted=float(p.detach())<1.3))
    assert result['fraction']==.25 and float(p.detach())==1.25
    previous=p.detach().clone()
    result=backtrack([p],torch.ones(1),lambda:dict(accepted=False),trials=3)
    assert not result['accepted'] and torch.equal(p,previous)
    def fail():raise RuntimeError('failed evaluator')
    with pytest.raises(RuntimeError):backtrack([p],torch.ones(1),fail)
    assert torch.equal(p,previous)
