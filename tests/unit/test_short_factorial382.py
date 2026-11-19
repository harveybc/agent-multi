from docs.work_plan.generalization.short_factorial382 import match, regret
import numpy as np

def test_equal_support_counts_not_same_market_rows():
    a={'q1':[('a',i) for i in range(100)],'q2':[('a',100)]}
    b={'q1':[('b',i) for i in range(4)],'q3':[('b',99)]}
    x,y=match(a,b)
    assert len(set(x))==len(set(y))==4
    assert all(i<100 for _,i in x)
    assert [x,y]==match(a,b)

def test_regret_is_directional_and_ties_cost_nothing():
    np.testing.assert_array_equal(regret(np.array([-2.,3.,0.]),np.array([True,False,True])),[2,3,0])
