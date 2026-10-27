"""Finite synthetic market design and independent trade arithmetic assertions."""
import itertools
import math

import numpy as np
import pandas as pd

from app.frozen_recipe_inputs import require

WARMUP=600

def cells():
    keys=('before','after','vol_before','vol_after','switch','period','phase','cost')
    values=((-1,0,1),(-1,0,1),(.00002,.00015,.00060),(.00002,.00015,.00060),
            (32,60,88),(8,24,72),(0,5,11,19),('configured','zero_cost'))
    for row in itertools.product(*values):yield dict(zip(keys,row))


def market_frame(cell):
    n=np.arange(-WARMUP,120);before=n<cell['switch']
    drift=.00015*np.where(before,cell['before'],cell['after'])
    amplitude=np.where(before,cell['vol_before'],cell['vol_after'])
    phase=n+cell['phase']
    innovation=amplitude*(np.sin(2*np.pi*phase/cell['period'])+
                          .3*np.sin(2*np.pi*phase/7.3))
    close=1.1+np.cumsum(drift+innovation)
    open_=np.r_[1.1,close[:-1]]
    wick=amplitude*.25+1e-6
    return pd.DataFrame(dict(OPEN=open_,HIGH=np.maximum(open_,close)+wick,
        LOW=np.minimum(open_,close)-wick,CLOSE=close),
        index=pd.DatetimeIndex([t for week in range(-5,1) for t in pd.date_range(
            pd.Timestamp('2020-01-05T22:00Z')+pd.Timedelta(weeks=week),periods=120,freq='h')]))


def check_trade(trade):
    t=trade;d=t['direction'];u=t['units']
    require(d in (-1,1) and u>0,'SIDE_AND_UNITS')
    expected=d*u*(t['exit_price']-t['entry_price'])
    require(math.isclose(t['gross_usd'],expected,rel_tol=1e-10,abs_tol=1e-8),'SIGNED_GROSS')
    costs=sum(t[k] for k in ('spread_usd','slippage_usd','commission_usd','financing_usd'))
    require(math.isclose(t['net_usd'],expected-costs,rel_tol=1e-10,abs_tol=1e-8),'NET_ACCOUNTING')
    require(d*(t['take_profit']-t['entry_price'])>0 and
            d*(t['entry_price']-t['stop_loss'])>0,'DIRECTIONAL_PROTECTIONS')
