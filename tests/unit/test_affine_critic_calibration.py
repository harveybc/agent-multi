from copy import deepcopy
import numpy as np
import pytest
import torch
from app.affine_critic_calibration import fit_shared_affine,required_scale,transform_head
from app.training_forensics import tree_digest
from tests.unit.test_adamw_decision_preservation import setup


def test_affine_recovers_known_shared_transform_and_constrained_optimum():
    x=np.arange(12,dtype=float).reshape(4,3);q=np.stack([x,x]);y=2*x+3
    r=fit_shared_affine(q,y,.1,'affine')
    assert r['scale']==2 and r['offset']==3 and r['theoretical_after_mse']==0
    r=fit_shared_affine(q,y,3.,'affine')
    assert r['scale']==3 and r['bound_active']
    for scale in np.linspace(3,8,30):
        offset=y.mean()-scale*q.mean()
        assert np.mean((scale*q+offset-y)**2)>=r['theoretical_after_mse']-1e-12


def test_shift_is_optimal_offset_and_does_not_scale():
    q=np.arange(24,dtype=float).reshape(2,4,3);y=np.ones((4,3))*5
    r=fit_shared_affine(q,y,.5,'shift')
    assert r['scale']==1 and r['offset']==float(np.mean(y-q))
    assert r['theoretical_after_mse']<r['theoretical_before_mse']


def test_negative_unconstrained_slope_respects_positive_margin_floor():
    x=np.arange(12,dtype=float).reshape(4,3);q=np.stack([x,x]);r=fit_shared_affine(q,-x,.25,'affine')
    assert r['unconstrained_scale']==-1 and r['scale']==.25


def test_margin_floor_uses_every_view():
    views=[np.array([[0.,2.,1.],[3.,0.,1.]]),np.array([[0.,2.,1.5],[3.,0.,2.]])]
    assert required_scale(views,np.array([1,0]),np.array([.1,.3]))==.3
    with pytest.raises(ValueError):required_scale([np.zeros((2,3))],np.array([1,0]),np.ones(2))


def test_baked_twin_transform_preserves_hidden_weights_and_ranks():
    head,_,z,_=setup();head=head.double();z=z.double();before=deepcopy(head.state_dict());original=tree_digest(before)
    with torch.no_grad():q=head(z).mean(2)
    transformed=transform_head(before,2.,.7);head.load_state_dict(transformed)
    with torch.no_grad():actual=head(z).mean(2)
    torch.testing.assert_close(actual,2*q+.7,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(actual.min(0).values,2*q.min(0).values+.7,rtol=1e-12,atol=1e-12)
    assert torch.equal(actual.min(0).values.argmax(1),q.min(0).values.argmax(1))
    assert tree_digest(before)==original
    for twin in range(2):
        for part in ('weight','bias'):
            key=f'twins.{twin}.0.0.{part}'
            assert torch.equal(before[key],transformed[key])


@pytest.mark.parametrize('change',['nan','shape','zero_scale','bad_kind','constant','shift_infeasible'])
def test_invalid_calibration_refused(change):
    q=np.ones((2,4,3));q[:,:,1]=2;y=np.zeros((4,3));minimum=.1;kind='affine'
    if change=='nan':q[0,0,0]=np.nan
    if change=='shape':y=y[:2]
    if change=='zero_scale':minimum=0
    if change=='bad_kind':kind='unknown'
    if change=='constant':q[:]=1
    if change=='shift_infeasible':kind='shift';minimum=2
    with pytest.raises(ValueError):fit_shared_affine(q,y,minimum,kind)


def test_invalid_transform_refused():
    h,_,_,_=setup();state=h.state_dict()
    for a,b in [(0.,0.),(-1.,0.),(1.,float('inf'))]:
        with pytest.raises(ValueError):transform_head(state,a,b)
    with pytest.raises(ValueError):transform_head({},1.,0.)
