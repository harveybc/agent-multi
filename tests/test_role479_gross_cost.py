import numpy as np
from tools.role479_gross_cost import estimate_cost, known_friction, lock_direction, ridge
from tools.role478_direction import preferences


def test_known_cost_at_entry_and_future_exit_estimated_only_from_fit():
    base=known_friction(np.array([.001,.002]),.0001,.00005)
    np.testing.assert_allclose(base[1,:3],.0022)
    np.testing.assert_allclose(base[:,3:],.0002)
    residual=np.array([[-.0002]*6,[.0004]*6])
    cost=estimate_cost(residual,base,np.array([1.,2.]))
    np.testing.assert_allclose(cost[0,3:],15.)
    np.testing.assert_allclose(cost[1,:3],57.5)


def test_gross_direction_may_abstain_but_never_flip():
    g=np.array([[10,8,9,7,6,5],[1,0,2,4,3,5],[1,3,0,2,3,1]])
    n=np.array([[-2,-1,-8,3,2,1],[1,0,1,0,2,1],[1,1,1,1,2,1]])
    np.testing.assert_array_equal(preferences(lock_direction(g,n)),[0,-1,0])


def test_gross_minus_same_ridge_cost_is_net_ridge():
    rng=np.random.default_rng(479)
    x=rng.normal(size=(90,5));g=rng.normal(size=(70,6));c=rng.uniform(size=(70,6))
    run=lambda y:ridge(x[:70],y,x[70:])
    np.testing.assert_allclose(run(g)-run(c),run(g-c),atol=1e-12)
