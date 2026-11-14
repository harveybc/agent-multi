import numpy as np
import pytest
from app.function_local_projection import locality_factor,metric_projection,gap_drift,pairwise_qualified,gap_jacobian
from app.primal_direction_projection import project_primal


def test_unconstrained_closed_form():
    g=np.array([[1.,0.]])
    l=locality_factor(g)
    x,r=metric_projection(np.array([3.,2.]),np.array([[0.,1.]]),np.array([0.]),l)
    assert r['feasible']
    np.testing.assert_allclose(x,[1.,2.],atol=1e-12)


def test_constrained_closed_form():
    l=locality_factor(np.array([[1.,0.]]))
    x,r=metric_projection(np.array([3.,2.]),np.eye(2),np.array([2.,0.]),l)
    assert r['feasible']
    np.testing.assert_allclose(x,[2.,2.],atol=1e-9)


def test_oblique_constraint_and_metric_scale_invariance():
    g=np.array([[1.,0.]])
    l=locality_factor(g)
    np.testing.assert_array_equal(l,locality_factor(2*g))
    x,r=metric_projection(np.array([3.,2.]),np.array([[1.,1.]]),np.array([5.]),l)
    assert r['feasible']
    np.testing.assert_allclose(x,[1.5,3.5],atol=1e-9)


def test_empty_motion_rejected():
    with pytest.raises(ValueError):metric_projection(np.zeros(2),np.eye(2),np.ones(2),np.eye(2))


def test_gap_jacobian_exact_support():
    import torch
    state={}
    for t in range(2):
        prefix=f'twins.{t}.0.'
        state.update({prefix+'0.weight':torch.tensor([[.3,.2]]),prefix+'0.bias':torch.tensor([.1]),
            prefix+'2.weight':torch.tensor([[.1],[.3],[-.2]]),prefix+'2.bias':torch.tensor([.1,.2,.3])})
    z=np.arange(16.).reshape(2,4,2)/16
    g,ix=gap_jacobian(state,z,[0,2,3],2)
    assert ix.tolist()==[0,3] and g.shape[0]==6
    zz=z.copy();zz[:,1:3]=999
    gg,_=gap_jacobian(state,zz,[0,2,3],2)
    np.testing.assert_array_equal(g,gg)


def test_identity_metric_control():
    d=np.array([1.,-2.]);j=np.eye(2);b=np.array([2.,1.])
    x,r=metric_projection(d,j,b,locality_factor(np.zeros((2,2))))
    ref,rr=project_primal(d,j,b)
    assert r['feasible']==rr['feasible']
    np.testing.assert_array_equal(x,ref)


def test_gap_drift_excludes_target_and_common_value():
    q=np.arange(18.).reshape(2,3,3)
    assert gap_drift(q,q+4,1)==0
    x=q.copy();x[:,1,0]+=100
    assert gap_drift(q,x,1)==0
    x[:,0,0]+=1
    assert gap_drift(q,x,1)>0


@pytest.mark.parametrize('g',[np.array([[np.nan]]),np.array([1.,2.]),np.empty((0,2))])
def test_bad_metric(g):
    with pytest.raises(ValueError):locality_factor(g)


def test_pairwise_all_views_required():
    control={'cpu':{'regret':1.},'gpu':{'regret':1.}}
    a=dict(passed=True,metrics={'cpu':{'regret':1.},'gpu':{'regret':.9}})
    assert pairwise_qualified(a,control,{'cpu':1.,'gpu':1.},{'cpu':2.,'gpu':2.})
    assert not pairwise_qualified(a,control,{'cpu':1.,'gpu':2.},{'cpu':2.,'gpu':2.})
    a['metrics']['gpu']['regret']=1.01
    assert not pairwise_qualified(a,control,{'cpu':1.,'gpu':1.},{'cpu':2.,'gpu':2.})
    a['passed']=False
    assert not pairwise_qualified(a,control,{'cpu':1.,'gpu':1.},{'cpu':2.,'gpu':2.})
