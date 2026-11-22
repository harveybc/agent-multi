import numpy as np
import pytest
from tools.role408_router import apply_map,calibrate,ns,opportunity,terminal


def block():
    x=np.broadcast_to(np.array([.2,.4,.6,.8])[:,None,None],(4,2,4)).copy()
    return dict(year=2022,end=ns('2022-12-31'),p=x,y=5*x-2)


def test_affine_support_and_direction():
    b=block();b['y'][:,1,1]*=-1
    f=calibrate([b],2023)
    np.testing.assert_allclose(f[0,0],[-2,5,.2,.8])
    assert f[1,1,1]==0
    np.testing.assert_allclose(apply_map(b['p']*0+1,f)[:,0,0],2)
    b['end']=ns('2023-01-01')
    with pytest.raises(ValueError):calibrate([b],2023)


def test_matched_abstention_and_missing_outcome():
    raw=np.zeros((3,2,4));raw[:,0,1]=1
    cal=raw*0;y=raw*0;y[:,0,1]=[3,-4,np.nan]
    got=opportunity(raw,cal,y)
    assert (got['joint_filled_or_nop'],got['vetoes'],got['wins_rejected'],got['losses_avoided'])==(2,2,1,1)
    assert got['gains_rejected_bp_sum']==3 and got['losses_avoided_bp_sum']==4


def test_equal_continuation_blocked_while_occupied():
    extra=lambda anchor,cash: .01*cash
    assert terminal(1000,8,8,extra)==1010
    assert terminal(1000,9,8,extra)==1000
    assert terminal(1000,9,-1,extra)==1000
