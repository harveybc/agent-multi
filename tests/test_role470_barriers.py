import numpy as np
import pytest

from tools.role470_barriers import NS, probe


def setup_case():
    t=np.arange(4)*NS
    o=np.ones(4)
    hi=o.copy();lo=o.copy()
    data=(t,o,hi,lo,None,None,None,None,np.ones(4,dtype=bool),np.full(4,3*NS))
    prof=dict(slippage_market=0.,slippage_stop=.001,commission_per_lot_side=0.,
              lot_units=1,swap_long_per_lot=0.,swap_short_per_lot=0.,limit_through=0.)
    return data,(prof,np.zeros(4),np.zeros(4))


@pytest.mark.parametrize('side',[-1,1])
def test_dual_bounds_and_exit_candle(side):
    data,cost=setup_case()
    data[2][1],data[3][1]=2.,.5
    r=probe(1,side,.02,.01,1,24,data,cost)
    assert r['reason']=='sl_dual' and r['end']==1 and r['mfe_lower_r']==0
    assert r['usd']==pytest.approx(-550.) and r['usd_tp_first']==pytest.approx(1000.)


def test_calendar_precedes_intrabar_and_swap_uses_elapsed_rolls():
    data,cost=setup_case()
    data[2][3]=2.
    cost[2][3]=1.;cost[0]['swap_long_per_lot']=-.0001
    r=probe(1,1,.02,.01,.5,48,data,cost)
    assert r['reason']=='calendar' and r['end']==3 and not r['tp']
    assert r['usd']==pytest.approx(-2.5) and r['gross_same_fills_usd']==pytest.approx(0.)
