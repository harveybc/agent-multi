import numpy as np
import pytest
import torch

from app.fit_decision_preservation import interior_buffer, margins, preservation, project_step, validate_fresh_calibration, protected_actions
from app.targeted_fit_errors import error_targets


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_buffer_is_explicit_dtype_scaled_and_protected_only(dtype):
    z=torch.tensor([[2.,0.,-1.],[0.,3.,1.]],dtype=dtype)
    rows=torch.tensor([0]);actions=torch.tensor([0])
    got=interior_buffer(z,rows,actions)
    expected=8*torch.finfo(dtype).eps*torch.tensor([[3.,4.]],dtype=torch.double)
    assert torch.equal(got,expected) and (got>0).all()
    z[1]=torch.tensor([1e12,-1e12,1e10],dtype=dtype)
    assert torch.equal(interior_buffer(z,rows,actions),expected)


@pytest.mark.parametrize('z,rows,actions',[
    (torch.tensor([[1,2,3]]),torch.tensor([0]),torch.tensor([2])),
    (torch.tensor([[1.,float('nan'),3.]]),torch.tensor([0]),torch.tensor([2])),
    (torch.zeros(1,3),torch.tensor([],dtype=torch.long),torch.tensor([],dtype=torch.long)),
    (torch.zeros(1,3),torch.tensor([0]),torch.tensor([3])),
])
def test_invalid_buffer_refuses(z,rows,actions):
    with pytest.raises(ValueError):interior_buffer(z,rows,actions)


def test_interior_halfspace_and_nonlinear_guard():
    z=torch.tensor([[2.,0.,-1.]])
    rows=torch.tensor([0]);actions=torch.tensor([0]);floor=margins(z.double(),rows,actions)
    cushion=interior_buffer(z,rows,actions).numpy().flatten()
    # x changes the protected margin; y improves the objective but has curvature.
    matrix=np.array([[1.,0.],[1.,0.],[0.,1.]])
    proposal=np.array([-1e-4,1e-3])
    boundary,_=project_step(proposal,matrix,np.zeros(3))
    interior,info=project_step(proposal,matrix,np.r_[cushion,0.])
    assert info['success'] and np.min(matrix@interior-np.r_[cushion,0.])>=-1e-10
    def actual(d):
        out=z.double().clone();out[0,0]+=d[0]-d[1]**2;return out
    assert not preservation(actual(boundary),floor,rows,actions)['passed']
    assert preservation(actual(interior),floor,rows,actions)['passed']


def test_infeasible_interior_does_not_fall_back_to_boundary():
    result,info=project_step(np.ones(1),np.array([[1.],[-1.]]),np.ones(2)*1e-5)
    assert not info['success'] and np.array_equal(result,np.zeros(1))


def test_fresh_source_rejects_old_or_mismatched_diagnosis():
    source=dict(schema='selected_input_critic_calibration.v1',actor_updates=0,modes=[],
                identity='current',source_sha256='actor72',support_sha256='support')
    diagnosis=dict(schema='selected_calibrated_boundary_diagnosis.v1',source='current',
                   source_sha256='actor72',support='support',actor_updates=0,posthoc_timestamps=[])
    validate_fresh_calibration(source,diagnosis)
    for key,value in [('source','previous'),('source_sha256','actor69'),('support','other'),
                      ('actor_updates',1),('posthoc_timestamps',[123])]:
        with pytest.raises(ValueError):validate_fresh_calibration(source,dict(diagnosis,**{key:value}))
    with pytest.raises(ValueError):validate_fresh_calibration(dict(source,modes=['calibrated_adamw']),diagnosis)


def test_new_actor_rebuilds_error_and_guard_sets_not_old_indices():
    native=torch.tensor([[0.,2.,1.],[0.,1.,3.],[0.,2.,1.]])
    records=[{'role':'fit'},{'role':'fit'},{'role':'monitor'}]
    old=torch.tensor([[3.,0.,0.],[0.,0.,3.],[0.,3.,0.]])
    new=torch.tensor([[0.,3.,0.],[3.,0.,0.],[0.,3.,0.]])
    assert protected_actions(old,native,records)[0].tolist()==[1]
    assert protected_actions(new,native,records)[0].tolist()==[0]
    assert error_targets(old,native,records)[0].tolist()==[1,-1,-1]
    assert error_targets(new,native,records)[0].tolist()==[-1,2,-1]
