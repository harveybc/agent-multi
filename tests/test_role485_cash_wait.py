import numpy as np
import pytest

from tools.role483_fixed_gate import replay
from tools.role485_cash_wait import durations


def test_calibration_must_be_mature_and_paired_by_scale():
    t=np.array([1,5,9])
    ex=t[:,None]+np.array([[2,5,8],[2,6,9],[1,7,10]])
    np.testing.assert_array_equal(durations(t,ex,20),[2,6,9])
    with pytest.raises(ValueError):
        durations(t,ex,19)


def test_cash_wait_does_not_use_rejected_trade_outcome_or_extend_when_busy():
    t=np.arange(5)
    acts=np.array([3,0,0,3,0])
    score=np.tile([-1.,0.,-3.,1.,2.],(6,1)).T
    ends=np.tile(t[:,None],(1,6))
    for impossible in (-1000000,1000000):
        pnl=np.full((5,6),100.)
        pnl[1,0]=impossible
        ends[1,0]=impossible
        pauses=[]
        def wait(r,k):
            pauses.append((r,k))
            return r+2
        trades,ret,dd=replay(t,t,acts,score,pnl,ends,True,rejected_buy=wait)
        assert pauses==[(1,0)]
        assert [x['entry_ns'] for x in trades]==[4]
        assert ret==pytest.approx(1.) and dd==0.
