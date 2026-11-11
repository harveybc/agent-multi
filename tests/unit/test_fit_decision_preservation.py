import numpy as np
import pytest
import torch

from app.fit_decision_preservation import protected_actions, margins, preservation, project_step


def test_only_fit_strict_source_optima_are_protected():
    z = torch.tensor([[2.,0.,0.],[0.,2.,0.],[0.,0.,2.],[2.,0.,0.]])
    y = torch.tensor([[1.,0.,0.],[0.,2.,0.],[0.,0.,3.],[1.,1.,0.]])
    records = [{'role':r} for r in ('fit','monitor','fit','fit')]
    rows, actions = protected_actions(z,y,records)
    assert rows.tolist()==[0,2] and actions.tolist()==[0,2]
    y[1] = torch.tensor([1e9,-1e9,0.])
    again,_ = protected_actions(z,y,records)
    assert torch.equal(rows,again)


def test_source_floor_not_previous_step_floor_and_no_flip():
    z = torch.tensor([[1.,0.,-1.]],dtype=torch.double)
    rows=torch.tensor([0]); acts=torch.tensor([0]); ref=margins(z,rows,acts)
    assert preservation(z,ref,rows,acts)['passed']
    down=z.clone();down[0,0]-=8e-8
    assert preservation(down,ref,rows,acts)['passed']
    down[0,0]-=8e-8
    assert not preservation(down,ref,rows,acts)['passed']
    tiny=torch.tensor([[0.,5e-8,-1.]],dtype=torch.double)
    reference=torch.tensor([[0.,1.]],dtype=torch.double)
    assert not preservation(tiny,reference,rows,acts)['passed']


def test_projection_halfspace_and_already_feasible():
    d=np.array([1.,-2.]); a=np.array([[0.,1.]]); b=np.array([0.])
    projected, info=project_step(d,a,b)
    assert info['success']
    np.testing.assert_allclose(projected,[1.,0.],atol=1e-8)
    again,info=project_step(projected,a,b)
    assert info['success']
    np.testing.assert_allclose(again,projected,atol=1e-8)


def test_duplicate_constraints_and_nonpositive_slope():
    d=np.array([1.,-2.,.5]); a=np.array([[0.,1.,0.],[0.,2.,0.],[-1.,0.,0.]])
    got, info=project_step(d,a,np.zeros(3))
    assert info['success']
    np.testing.assert_allclose(got,[0.,0.,.5],atol=1e-8)
    assert np.min(a@got)>=-1e-8


def test_infeasible_and_zero_only_are_not_reported_as_learning():
    got,info=project_step(np.ones(2),np.zeros((1,2)),np.ones(1))
    assert not info['success'] and np.all(got==0)
    got,info=project_step(np.ones(2),np.vstack([np.eye(2),-np.eye(2)]),np.zeros(4))
    assert info['success'] and np.linalg.norm(got)<1e-8
    assert not info['nonzero']


def test_invalid_projection_rejected():
    with pytest.raises(ValueError):project_step(np.array([np.nan]),np.ones((1,1)),np.zeros(1))
