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


@pytest.mark.parametrize('buy_gate,sell_gate,entries', [
    (True,True,[12,13]),(False,True,[10,13]),(True,False,[11,13]),(False,False,[10,13])])
def test_side_specific_gate_shares_single_slot(buy_gate,sell_gate,entries):
    ts=np.arange(10,14)
    k=np.array([0,3,3,0])
    scores=np.tile(np.array([0.,-2,3,5])[:,None],(1,6))
    exits=np.tile(np.array([12,12,12,13])[:,None],(1,6))
    trades,_,_=replay(ts,np.arange(4),k,scores,scores,exits,buy_gate,sell_gate=sell_gate)
    assert [trade['entry_ns'] for trade in trades]==entries
