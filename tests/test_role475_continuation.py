import numpy as np
import pytest
from tools.role474_close_advantage import versus_wait


def test_first_strictly_future_exit_not_best_exit_or_current_tick():
    # Two independent paths; a negative next exit must not be replaced with HOLD.
    y=np.array([4.,-1.,100.,5.,6.])
    owner=np.array([2,2,2,7,7])
    gain,nxt=versus_wait(owner,y,np.array([True,True,True,False,True]))
    np.testing.assert_array_equal(nxt,[1,2,-1,4,-1])
    np.testing.assert_allclose(gain,[5.,-101.,100.,-1.,6.])


@pytest.mark.parametrize('n',[0,1,9])
def test_terminal_and_never_stop_policy_are_passive(n):
    y=np.arange(n,dtype=float)
    gain,nxt=versus_wait(np.zeros(n,np.int64),y,np.zeros(n,bool))
    np.testing.assert_array_equal(gain,y)
    assert (nxt==-1).all()


def test_randomized_parity_with_forward_scan():
    rng=np.random.default_rng(475)
    owner=np.repeat(np.arange(30),rng.integers(0,20,30))
    y=rng.normal(size=len(owner));close=rng.random(len(owner))<.3
    gain,nxt=versus_wait(owner,y,close)
    for i in range(len(y)):
        take=next((k for k in range(i+1,len(y)) if owner[k]==owner[i] and close[k]),-1)
        assert nxt[i]==take and gain[i]==pytest.approx(y[i]-(y[take] if take>=0 else 0))
