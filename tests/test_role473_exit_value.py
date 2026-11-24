import numpy as np
import pytest
from tools.role461_seasonal_account import DEFAULT, NS
from tools.role473_exit_value import compare_position


@pytest.mark.parametrize('side',[1,-1])
def test_prior_forecast_can_close_before_intrabar_stop_but_not_gap(side):
    t=np.arange(5)*NS;o=np.ones(5);hi=o.copy();lo=o.copy()
    hi[2]=1.02;lo[2]=.98
    sig=np.full((5,8),np.nan);sig[1,:]=-side*.02
    data=(t,o,hi,lo,np.ones(5),np.full(5,.01),None,None,np.ones(5,bool),np.full(5,t[4]))
    prof=dict(slippage_market=0.,slippage_stop=0.,swap_long_per_lot=0.,swap_short_per_lot=0.,
              lot_units=1,commission_per_lot_side=0.,limit_through=0.)
    def run():
        return compare_position(1,side,1,1,10000,1,1,data,(prof,np.zeros(5),np.zeros(5)),sig,DEFAULT)
    choices,ticks=run()
    assert choices['hold'][1]=='sl_dual' and choices['seasonal_sl'][1]=='early_sl'
    assert choices['seasonal_sl'][2]==0 and ticks[0]['minus_hold']>0
    sig[1,:]=np.nan
    assert run()[0]['seasonal_sl']==choices['hold']
    sig[1,:]=-side*.02;o[2]=1-side*.02
    choices,ticks=run()
    assert not ticks and choices['seasonal_sl']==choices['hold'] and choices['hold'][1]=='sl_gap'
