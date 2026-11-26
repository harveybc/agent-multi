import numpy as np
import pytest

from tools.role483_fixed_gate import replay


def test_zero_score_trade_can_displace_positive_trade_through_exit_bar():
    ts=np.arange(10,14)
    rows=np.arange(4)
    actions=np.zeros(4,dtype=int)
    net=np.array([[0],[2],[3],[4]])
    pnl=np.array([[-100],[300],[500],[100]])
    ends=np.array([[12],[12],[12],[13]])
    on,ret_on,dd_on=replay(ts,rows,actions,net,pnl,ends,True)
    off,ret_off,dd_off=replay(ts,rows,actions,net,pnl,ends,False)
    assert [x['entry_ns'] for x in on]==[11,13]
    assert [x['entry_ns'] for x in off]==[10,13]
    assert all(x['q']==1. for x in on+off)
    assert (ret_on,dd_on)==pytest.approx((4.03,0.))
    assert (ret_off,dd_off)==pytest.approx((-.01,1.))
