import numpy as np
import pytest
import torch

from app.critic_coverage_geometry import geometry,distances,nearest_values,label_energy,encoder_identity


def test_encoder_identity_distinguishes_copy_change_and_storage_sharing():
    a=torch.ones(2,3)
    s={'actor_network_params.module.0.module.encoder.weight':a,
       'qvalue_network_params.module.encoder.weight':torch.stack([a,a])}
    r=encoder_identity(s)
    assert all(v['identical_tensors']==1 and v['storage_aliases']==0 for v in r)
    s['qvalue_network_params.module.encoder.weight'][1,0,0]=2
    r=encoder_identity(s)
    assert r[0]['maximum_float_difference']==0 and r[1]['maximum_float_difference']==1
    s['qvalue_network_params.module.encoder.weight']=a.expand(2,*a.shape)
    assert all(v['storage_aliases']==1 for v in encoder_identity(s))


def test_rank_one_duplicates_and_no_query_fit_contamination():
    x=np.array([[0.,0.,7.],[1.,2.,7.],[1.,2.,7.],[2.,4.,7.]])
    q=np.array([[2.,4.,8.]])
    a,b,info=geometry(x,q,True)
    _,_,changed=geometry(x,q*1000,True)
    assert info['center']==changed['center'] and info['scale']==changed['scale']
    assert info['unique_fit_rows']==3 and info['constant_fit_dimensions']==1
    assert info['query_rows_with_constant_dimension_drift']==1
    assert info['spectrum']['effective_rank']==pytest.approx(1.)
    assert distances(a,b)[0,3]==pytest.approx(0.)


def test_constant_geometry_is_explicit_not_infinite():
    a,b,info=geometry(np.ones((4,3)),np.zeros((2,3)),True)
    assert info['spectrum']['effective_rank']==0
    assert info['constant_fit_dimensions']==3 and info['query_rows_with_constant_dimension_drift']==2
    assert (distances(a,b)==0).all()


def test_neighbor_ties_stable_and_mean_exact():
    d=np.array([[0.,0.,2.],[2.,1.,0.]])
    y=np.eye(3)
    pred,ix=nearest_values(d,y,1)
    assert ix.tolist()==[[0],[2]] and np.array_equal(pred,y[[0,2]])
    pred,_=nearest_values(d,y,3)
    np.testing.assert_allclose(pred,np.full((2,3),1/3))
    with pytest.raises(ValueError):nearest_values(d,y,4)


def test_common_return_and_action_information_are_separate():
    common=np.arange(4.)[:,None]*np.ones((1,3))
    assert label_energy(common)['common_value_energy_share']==1
    assert label_energy(common)['all_actions_tied']==4
    actions=np.array([[1.,-1.,0.],[-1.,1.,0.]])
    assert label_energy(actions)['common_value_energy_share']==0
    assert label_energy(actions)['unambiguous_optimal_counts']==[1,1,0]
