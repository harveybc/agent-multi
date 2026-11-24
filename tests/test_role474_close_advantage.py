import numpy as np
import pytest
from tools.role474_close_advantage import NS,candidates,first_close,fit


def test_first_close_uses_only_first_positive_score():
    assert first_close(np.array([-.2,0.,.1,3.]))==2
    assert first_close(np.array([-.1,0.])) is None
    assert first_close(np.array([])) is None


@pytest.mark.parametrize('k',[0,3])
def test_future_settlement_changes_label_not_inputs(k):
    t=np.arange(7)*NS;o=np.ones(7);atr=np.full(7,.01)
    data=(t,o,o,o,o,atr,None,None,None,np.full(7,6*NS))
    pred=np.arange(56.).reshape(7,8)/1000
    fees=np.zeros(6);spread=np.zeros(7);roll=np.zeros(7)
    ix=np.array([1]);ks=np.array([k]);end=np.full((1,6),4);net=np.zeros((1,6))
    def run():
        return candidates(ix,ks,data,spread,roll,fees,pred,1.,1.,net,end)
    _,j,x,y,c=run()
    np.testing.assert_array_equal(j,[2,3,4]);assert c[0]==3
    net[:]=100.
    _,_,x2,y2,_=run();np.testing.assert_array_equal(x,x2);assert (y2<y).all()
    pred[2:]*=2
    _,_,x3,_,_=run();np.testing.assert_array_equal(x[0],x3[0])


def test_weighted_ridge_matches_direct_solution():
    v=np.array([[1.,2.],[0.,2.],[5.,2.],[7.,2.]])
    y=np.array([2.,-3.,1.,4.]);w=np.array([1.,.5,.5,1.])
    z=np.c_[np.ones(4),v];m=fit((z.T@(z*w[:,None]),z.T@(w*y)),2)
    x=(v-m['mean'])/m['scale']
    want=np.linalg.solve(x.T@(w[:,None]*x)+10*np.eye(2),x.T@(w*(y-np.average(y,weights=w))))
    np.testing.assert_allclose(m['coef'],want,atol=1e-11)
