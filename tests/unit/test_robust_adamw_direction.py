import numpy as np
import pytest

from app.robust_adamw_direction import project_displacement, decision_guard, numpy_heads


def test_projection_rotates_instead_of_only_shrinking():
    d=np.array([-1.,2.]);j=np.array([[1.,0.]])
    p,r=project_displacement(d,j,np.array([.2]))
    assert r['feasible']
    np.testing.assert_allclose(p,[.2,2.],atol=1e-8)
    assert r['correction_norm']>0


def test_feasible_identity_and_scale_invariance():
    d=np.array([1.,2.]);j=np.eye(2);b=np.array([.1,.2])
    p,r=project_displacement(d,j,b)
    np.testing.assert_array_equal(p,d)
    p2,_=project_displacement(d,j*1e6,b*1e6)
    np.testing.assert_allclose(p2,p)
    assert r['feasible']


def test_inconsistent_constraints_rejected():
    _,r=project_displacement(np.array([1.]),np.array([[1.],[-1.]]),np.array([2.,2.]))
    assert not r['feasible']


@pytest.mark.parametrize('d,j,b',[
    ([float('nan')],[[1.]],[1.]),([1.],[[1.,2.]],[1.]),([0.],[[1.]],[1.])])
def test_bad_projection_contract(d,j,b):
    with pytest.raises(ValueError):project_displacement(np.array(d),np.array(j),np.array(b))


def test_fragile_tie_and_nonlinear_endpoint_refused():
    y=np.array([[0.,2.,1.]])
    args=(np.array([1]),np.array([1e-5]),y,0.)
    assert not decision_guard(np.array([[1.,1.+1e-8,0.]]),*args)['passed']
    assert decision_guard(np.array([[1.,1.1,0.]]),*args)['passed']
    assert not decision_guard(np.array([[2.,1.,0.]]),*args)['passed']


def test_wrong_action_even_if_regret_equal_refused():
    result=decision_guard(np.array([[2.,1.,0.]]),np.array([1]),np.array([1e-5]),np.ones((1,3)),0.)
    assert result['regret']==0 and not result['passed']


def test_nonfinite_scores_refused():
    with pytest.raises(ValueError):
        decision_guard(np.array([[0.,np.nan,1.]]),np.array([2]),np.array([1e-5]),np.ones((1,3)),0.)


def test_numpy_forward_matches_torch():
    import torch
    from app.role_routed_critic import RoleHeads
    from tools.run_training_forensics import template
    head=RoleHeads(template('cpu'),1,19).double()
    z=torch.randn(2,7,208,dtype=torch.float64)
    with torch.no_grad():expected=head(z).mean(2).numpy()
    np.testing.assert_allclose(numpy_heads(head.state_dict(),z.numpy()),expected,atol=1e-14)


def test_jacobian_has_finite_difference_witness_and_state_moves_exactly():
    import torch
    from app.role_routed_critic import RoleHeads
    from tools.run_training_forensics import template
    from tools.probe_robust_adamw_direction import gap_jacobian,vector,moved
    torch.manual_seed(7)
    head=RoleHeads(template('cpu'),1,19).double()
    z=torch.randn(2,239,208,dtype=torch.float64);old=np.array([0,9])
    with torch.no_grad():actions=head(z).mean(2).min(0).values[old].argmax(1).numpy()
    gaps,j=gap_jacobian(head,z,old,actions)
    initial={k:v.clone() for k,v in head.state_dict().items()}
    direction=np.random.default_rng(19).normal(size=len(vector(initial)))
    direction/=np.linalg.norm(direction)
    sides=[]
    for sign in (-1,1):
        head.load_state_dict(moved(initial,direction,sign*1e-6))
        sides.append(gap_jacobian(head,z,old,actions)[0])
    np.testing.assert_allclose((sides[1]-sides[0])/2e-6,j@direction,atol=1e-8)
    np.testing.assert_allclose(vector(moved(initial,direction,1e-3))-vector(initial),1e-3*direction,atol=1e-16)
    assert gaps.shape==(4,)
