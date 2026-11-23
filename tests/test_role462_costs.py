import numpy as np
import pandas as pd
import pytest

from tools.role461_seasonal_account import DEFAULT, NS, cost_arrays, hit, simulate


@pytest.mark.parametrize("side",[1,-1])
def test_flat_bid_round_trip(side):
    t=pd.Timestamp("2024-02-06",tz="UTC").value+np.arange(4)*NS
    b=np.ones(4); ds=np.zeros((4,8)); ds[0,5:]=side*np.array([.004,.001,.002])
    data=(t,b,b,b,b,b*.001,ds,ds,np.array([False,True,False,False]),np.full(4,t[3]))
    prof=dict(slippage_market=.00002,slippage_stop=.00005,limit_through=.0001,
              commission_per_lot_side=3.5,lot_units=100000,swap_long_per_lot=-7,swap_short_per_lot=2.5)
    spr=np.array([.0001,.0001,.0002,.0003])
    _,book=simulate(data,(np.ones(8),{}),DEFAULT,2024,'bridge',False,
                    (prof,spr,np.array([0,0,3,3])))
    r,=book
    assert r['reason']=='calendar'
    fee=sum(r[k] for k in ('commission_usd','slippage_usd','spread_usd'))-r['swap_usd']
    assert r['pnl']==pytest.approx(-fee)
    assert r['spread_usd']/r['units']==pytest.approx(spr[1 if side>0 else 3])
    assert r['swap_usd']/r['units']==pytest.approx(3*(-7 if side>0 else 2.5)/100000)


def test_schedule_and_rollover():
    op=pd.Timestamp('2024-02-04T22:00Z')
    closures=[(op-pd.Timedelta(days=2),op),(op+pd.Timedelta(days=5),op+pd.Timedelta(days=7))]
    prof=dict(spread_schedule=[[0,1,.0003],[1,24,.0001]],weekly_open_spread=.0004,
              triple_swap_rollover_index=3)
    s,r=cost_arrays(op.value+np.arange(100)*NS,closures,prof)
    assert s[[0,1,24]].tolist()==[.0004,.0001,.0003]
    assert r[[23,24,48,71,72,96]].tolist()==[0,1,2,2,5,6]


def test_limit_through():
    assert hit(1,1,1.01005,1,.99,1.01,.0001) is None
    assert hit(-1,1,1,.98995,1.01,.99,.0001) is None
