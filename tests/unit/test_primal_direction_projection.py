import numpy as np
import pytest
from app.primal_direction_projection import project_primal,select_stalled_probes,qualify_continuation
from app.robust_adamw_direction import project_displacement
from app.interior_adamw_repair import repair_direction


def test_known_primal_projection_and_kkt():
    d=np.array([-1.,2.]);j=np.array([[1.,0.]])
    p,r=project_primal(d,j,np.array([.2]))
    assert r['feasible'] and r['normalized_violation']<=1e-7
    np.testing.assert_allclose(p,[.2,2.],atol=1e-10)
    assert max(r['kkt'].values())<=1e-7


def test_row_scaling_and_duplicate_constraints_do_not_change_solution():
    d=np.array([-1.,-2.,3.]);j=np.array([[1.,0.,0.],[0.,1.,0.],[1.,0.,0.]])
    b=np.array([.1,.2,.1]);p,r=project_primal(d,j,b)
    scales=np.array([.01,1e5,2.]);p2,r2=project_primal(d,j*scales[:,None],b*scales)
    assert r['feasible'] and r2['feasible']
    np.testing.assert_allclose(p,[.1,.2,3.],atol=1e-8);np.testing.assert_allclose(p,p2,atol=1e-8)


def test_feasible_identity_and_infeasible_zero_gradient():
    d=np.array([1.,2.]);p,r=project_primal(d,np.eye(2),np.array([.1,.2]))
    assert r['feasible'];np.testing.assert_array_equal(p,d)
    _,r=project_primal(d,np.zeros((1,2)),np.array([1.]))
    assert not r['feasible']
    _,r=project_primal(np.array([1.]),np.array([[1.],[-1.]]),np.array([2.,2.]))
    assert not r['feasible']


@pytest.mark.parametrize('d,j,b', [([0.],[[1.]],[1.]),([float('nan')],[[1.]],[1.]),([1.],[[1.,2.]],[1.])])
def test_invalid_projection_refused(d,j,b):
    with pytest.raises(ValueError):project_primal(d,j,b)


def test_default_repair_is_exact_legacy_behavior():
    linear=lambda p:(np.array([p[0]]),np.array([[1.]]),p)
    evaluate=lambda p:dict(passed=bool(p[0]>=1e-5),motion=float(np.linalg.norm(p)))
    args=(np.array([-1e-4]),linear,evaluate,np.array([1e-5]))
    a=repair_direction(*args);b=repair_direction(*args,projector=project_displacement)
    np.testing.assert_array_equal(a.pop('delta'),b.pop('delta'));assert a==b
    c=repair_direction(*args,projector=project_primal)
    assert c['accepted'] and np.linalg.norm(c['delta'])<=4e-4


def test_selection_uses_only_trailing_rejection_strata():
    trace=[dict(step=1,accepted=True,reason='finite_guard')]
    trace.extend(dict(step=i,accepted=False,reason='finite_guard' if i%2 else 'solver_or_norm_cap') for i in range(2,42))
    last,selected=select_stalled_probes(trace)
    assert last==1 and len(selected)==len(set(selected))==16 and selected==sorted(selected)
    assert len([i for i in selected if i%2])==8
    with pytest.raises(ValueError):select_stalled_probes(trace[:4])


def test_gate_needs_actual_decision_learning_and_numeric_safety():
    rows=[dict(attempt=i,accepted=True,robust=True,geometry_agrees=True,before=.3,after=.3) for i in range(16)]
    assert not qualify_continuation(rows)
    rows[0]['after']=.29;assert qualify_continuation(rows)
    rows[0]['geometry_agrees']=False;assert not qualify_continuation(rows)
    rows[0]['geometry_agrees']=True
    for r in rows[3:]:r['accepted']=False
    assert not qualify_continuation(rows)
