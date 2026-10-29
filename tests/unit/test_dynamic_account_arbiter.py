import numpy as np
import pytest

from app.dynamic_account_arbiter import decode, choose, objectives, frontier, interval_best, bounds


def base():
    return [1.,1.,1.,1.,0.,0.,0.,0.,1.,0.,1.]


def test_direction_order_and_ties():
    g=decode(base())
    bids=[dict(horizon=3,direction=1,gain_risk=1.,uncertainty_risk=0.,duration=2),
          dict(horizon=12,direction=-1,gain_risk=2.,uncertainty_risk=0.,duration=6)]
    assert choose(bids,g,0.)['direction']==-1
    assert choose(bids[::-1],g,0.)['direction']==-1
    bids[1]['gain_risk']=1.
    assert choose(bids,g,0.) is None
    bids[0]['gain_risk']=bids[1]['gain_risk']=-1
    assert choose(bids,g,0.) is None


def test_equity_drawdown_and_frontier():
    r=objectives([100.,110.,90.,105.],100.)
    assert r==pytest.approx((5.,100*20/110))
    rows={'a':(8.,-2.),'b':(12.,-5.),'c':(7.,-4.),'a2':(8.,-2.)}
    assert set(frontier(rows))=={'a','a2','b'}
    with pytest.raises(ValueError): objectives([100.,np.nan],100.)


def test_opportunity_blocking_is_joint_not_greedy():
    # Start/end/profit: holding the slot for6 blocks a later profit20.
    assert interval_best([(0,5,6),(1,3,20)])[0]==20
    assert interval_best([(0,2,4),(2,5,8),(0,5,10)])[0]==12
    # A short transaction can also block the better long candidate.
    assert interval_best([(0,2,3),(1,8,20)])[0]==20


def test_decode_bounds_and_allocation_parameters_have_effect():
    g=decode(base()); assert len(bounds())==11
    with pytest.raises(ValueError): decode(base()[:-1])
    with pytest.raises(ValueError): decode([np.nan]+base()[1:])
    bid=dict(horizon=3,direction=1,gain_risk=1.,uncertainty_risk=.5,duration=6)
    control=choose([bid],g,.5)['score']
    for index in (0,4,5,6,7):
        x=base();x[index]+=.1
        assert choose([bid],decode(x),.5)['score']!=control
    assert decode(base()[:-1]+[2.8])[-1]==3
    for i,h in enumerate((3,6,12,24)):
        x=base();x[i]=1.1
        item=dict(bid,horizon=h)
        assert choose([item],decode(x),0.)['score']>choose([item],g,0.)['score']
