"""Finite synthetic market design and independent trade arithmetic assertions."""
import itertools
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from app.frozen_recipe_inputs import require

WARMUP=600

def cells(design=None):
    if design is not None:
        keys=('drift_before','drift_after','vol_before','vol_after','transition_bars','period','horizon','cost')
        require(set(design)=={'schema','axes','constants','expected_cells'} and
                design['schema']=='strategy_multiscale_matrix.v1','DESIGN_SCHEMA')
        axes=design['axes'];constants=design['constants']
        require(set(axes)==set(keys) and set(constants)=={'switch','phase','target_bars'},'DESIGN_AXES')
        require(all(isinstance(axes[k],list) and axes[k] and len(set(axes[k]))==len(axes[k]) for k in keys),'DESIGN_VALUES')
        require(math.prod(len(axes[k]) for k in keys)==design['expected_cells'],'DESIGN_SIZE')
        for k in keys[:-1]:
            require(all(type(x) in (int,float) and math.isfinite(x) for x in axes[k]),'DESIGN_FINITE')
        for k in ('horizon','period','transition_bars'):
            require(all(type(x) is int and x>=(0 if k=='transition_bars' else 1) for x in axes[k]),'DESIGN_TIME')
        require(all(x>=0 for k in ('vol_before','vol_after') for x in axes[k]),'DESIGN_VOLATILITY')
        require(set(axes['cost'])<={'configured','zero_cost'},'DESIGN_COST')
        require(all(type(constants[k]) is int for k in constants),'DESIGN_CONSTANTS')
        require(constants['target_bars']%120==0 and constants['target_bars']>=120 and
                2<constants['switch']<constants['target_bars']-max(axes['horizon'])-max(axes['transition_bars'])-4,'DESIGN_SUPPORT')
        for row in itertools.product(*(axes[k] for k in keys)):
            yield {**constants,**dict(zip(keys,row))}
        return
    keys=('before','after','vol_before','vol_after','switch','period','phase','cost')
    values=((-1,0,1),(-1,0,1),(.00002,.00015,.00060),(.00002,.00015,.00060),
            (32,60,88),(8,24,72),(0,5,11,19),('configured','zero_cost'))
    for row in itertools.product(*values):yield dict(zip(keys,row))


@lru_cache(maxsize=8)
def market_clock(target_bars):
    return pd.DatetimeIndex([t for week in range(-5,target_bars//120) for t in pd.date_range(
        pd.Timestamp('2020-01-05T22:00Z')+pd.Timedelta(weeks=week),periods=120,freq='h')])


def market_frame(cell):
    target=cell.get('target_bars',120)
    n=np.arange(-WARMUP,target);before=n<cell['switch']
    if 'drift_before' in cell:
        speed=cell['transition_bars']
        weight=np.clip((n-cell['switch'])/speed,0,1) if speed else (~before).astype(float)
        drift=cell['drift_before']+weight*(cell['drift_after']-cell['drift_before'])
        amplitude=cell['vol_before']+weight*(cell['vol_after']-cell['vol_before'])
    else:
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
        index=market_clock(target))


def check_trade(trade):
    t=trade;d=t['direction'];u=t['units']
    require(d in (-1,1) and u>0,'SIDE_AND_UNITS')
    expected=d*u*(t['exit_price']-t['entry_price'])
    require(math.isclose(t['gross_usd'],expected,rel_tol=1e-10,abs_tol=1e-8),'SIGNED_GROSS')
    costs=sum(t[k] for k in ('spread_usd','slippage_usd','commission_usd','financing_usd'))
    require(math.isclose(t['net_usd'],expected-costs,rel_tol=1e-10,abs_tol=1e-8),'NET_ACCOUNTING')
    require(d*(t['take_profit']-t['entry_price'])>0 and
            d*(t['entry_price']-t['stop_loss'])>0,'DIRECTIONAL_PROTECTIONS')
