import numpy as np
import pytest
from tools.role461_seasonal_account import DEFAULT,exit_signal,hit


@pytest.mark.parametrize('s',[1,-1])
def test_margin_only_moves_forecast_trigger(s):
    def signal(path,m):
        return exit_signal(s,100,100-2*s,100+3*s,np.array(path)*s,1,
                           dict(DEFAULT,early_margin_atr=m))
    assert signal([-1.6],.5)=='early_sl' and signal([-1.6],0)!='early_sl'
    assert signal([-2.2],0)=='early_sl' and signal([-2.2],-.5)!='early_sl'
    assert signal([3,-4],1) is None
    assert signal([.1],10)!='early_sl'
    assert hit(s,100,102,98,100-2*s,100+3*s)==('sl',100-2*s)
