import numpy as np
import pytest
from tools.role471_net_selector import account, criterion


def test_exclusive_slot_and_compounding():
    i=np.array([1,2,3,4])
    s=np.zeros((4,6));s[:,0]=10;s[3,4]=20
    r=np.full_like(s,100.);r[3,4]=-50.
    e=np.tile([3,3,3,4],(6,1)).T.copy()
    ret,dd,n,w=account(i,s,r,e,0.,0.,10.,1.)
    assert ret==pytest.approx(.495) and dd==pytest.approx(.5)
    assert n.tolist()==[1,0,0,0,1,0] and sum(w)==1


def test_zero_edge_tie_or_insufficient_margin_abstains():
    s=np.zeros((3,6));s[1,[0,3]]=20;s[2,0]=15;s[2,3]=14
    r=np.ones_like(s);e=np.arange(18).reshape(3,6)
    ret,_,n,_=account(np.arange(3),s,r,e,0.,1.,10.,1.)
    assert ret==0 and sum(n)==0


def test_no_trade_cannot_win_over_loss_with_activity():
    assert criterion([dict(orders=0,ret=0.)],True)<criterion([dict(orders=12,ret=-15.)],True)
