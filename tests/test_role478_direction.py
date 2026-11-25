import numpy as np
import pytest
from tools.role478_direction import quartiles, preferences
from tools.role471_net_selector import outcomes


def test_cuts_use_only_calibration_magnitudes_within_side():
    x=np.array([-40,-30,-20,-10,1,2,3,4,0.,-2,5])
    fit=np.arange(len(x))<8
    sign,q,rows=quartiles(x,fit)
    assert [r['q50'] for r in rows]==[25.,2.5]
    assert list(q[-3:])==[0,1,4] and list(sign[-3:])==[0,-1,1]
    x[-3:]=[1e6,0,-1e8]
    assert quartiles(x,fit)[2]==rows
    with pytest.raises(ValueError):quartiles(np.ones(8),np.ones(8,bool))


def test_skip_tied_and_negative_scores():
    a=np.array([[2,1,0,2,0,0],[-1,-2,-3,-4,-5,-6],
                [2,0,0,1,0,0],[0,0,1,0,2,0]])
    np.testing.assert_array_equal(preferences(a),[0,0,1,-1])


def test_extra_horizons_dont_change_original_probes():
    t=np.arange(6);o=np.ones(6)*10;atr=np.ones(6)*.1
    hi=np.array([10,10.12,10.13,10.26,10.4,10.5]);lo=o-.01
    args=(np.array([1]),t,o,hi,lo,atr,np.full(6,5),t*0.,t*0.,np.zeros(6),1.,1.)
    few,ef=outcomes(*args)
    all_h,ea=outcomes(*args,(1,2,3,4,5,6,24,48))
    cols=[5,6,7,13,14,15]
    np.testing.assert_array_equal(all_h[:,cols],few)
    np.testing.assert_array_equal(ea[:,cols],ef)
    assert ea[0,0]==ea[0,8]==1 and all_h[0,0]>0>all_h[0,8]
    assert ea[0,5]==3 and ea[0,15]==5
