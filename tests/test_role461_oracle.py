import numpy as np
import pytest

from tools.role461_seasonal_account import DEFAULT, entry, exit_signal, hit


@pytest.mark.parametrize('side',[1,-1])
def test_entry_is_mirrored_and_short_cannot_vote(side):
    path=np.array([99,99,99,99,99,3,1,2.])*side
    act=entry(path,path,1,None,DEFAULT)
    assert act[0]==side and act[-1]==6 and 0<act[3]<=1
    path[:5]*=-1
    assert entry(path,path,1,None,DEFAULT)==act
    path[-1]=np.nan
    assert entry(path,path,1,None,DEFAULT) is None


@pytest.mark.parametrize('side',[1,-1])
def test_sl_before_tp_and_pivot(side):
    def decision(vals):
        return exit_signal(side,10,10-side*2,10+side*3,side*np.array(vals),1,DEFAULT)
    assert decision([-2,4])=='early_sl'
    assert decision([3,-4]) is None
    assert decision([1,2,-1])=='wait'
    assert decision([-1,-1.5])=='early_pivot'
    assert decision([np.nan,np.nan]) is None
    assert decision([0,0]) is None


@pytest.mark.parametrize('side',[1,-1])
def test_ohlc_ambiguity_and_gap(side):
    sl=10-side;tp=10+side
    assert hit(side,10,12,8,sl,tp)==('sl_dual',sl)
    assert hit(side,10-2*side,14,6,sl,tp)==('sl_gap',10-2*side)


def test_flat_forecast_no_order():
    path=np.zeros(8)
    assert entry(path,path,1,None,DEFAULT) is None
