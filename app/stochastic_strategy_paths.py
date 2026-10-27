"""Seeded MID-price engineering paths; not market data or admissible features."""
import math

import numpy as np
import pandas as pd

from app.strategy_scenario_matrix import WARMUP


def market_path(spec,target_bars=240):
    required={'seed','drift_before','drift_after','vol_before','vol_after',
              'switch','transition','overlay','noise','shock','mirror'}
    if (set(spec)!=required or spec['noise'] not in ('gaussian','student','clustered')
        or spec['shock'] not in ('none','gap','jump') or type(spec['seed']) is not int
        or type(spec['mirror']) is not bool or type(target_bars) is not int
        or target_bars<240 or target_bars%120 or not 0<spec['switch']<240
        or not 0<=spec['transition']<72
        or not all(math.isfinite(spec[k]) for k in ('drift_before','drift_after',
                          'vol_before','vol_after','overlay','switch','transition'))
        or min(spec['vol_before'],spec['vol_after'])<0):
        raise ValueError('STOCHASTIC_SCENARIO_SPEC')
    n=WARMUP+target_bars;t=np.arange(-WARMUP,target_bars)
    # Repeat the specified transition and its reverse, not one fixed change.
    phase=np.mod(t,240)
    width=spec['transition']
    w=np.clip((phase-spec['switch'])/max(width,1),0,1)
    if not width:w=(phase>=spec['switch']).astype(float)
    drift=spec['drift_before']+(spec['drift_after']-spec['drift_before'])*w
    amplitude=spec['vol_before']+(spec['vol_after']-spec['vol_before'])*w
    drift=drift+spec['overlay']*(np.sin(2*np.pi*t/72)-.5*np.sin(2*np.pi*t/8))
    seeds=np.random.SeedSequence(spec['seed']).spawn(3)
    rng=np.random.default_rng(seeds[0]);wick_rng=np.random.default_rng(seeds[1])
    shock_rng=np.random.default_rng(seeds[2])
    z=rng.standard_t(5,n)/np.sqrt(5/3) if spec['noise']=='student' else rng.normal(size=n)
    if spec['noise']=='clustered':
        v=1.;innovation=np.empty(n)
        for i in range(n):
            innovation[i]=np.sqrt(v)*z[i]
            v=.05+.9*v+.05*innovation[i]**2
    else:innovation=z
    moves=drift+amplitude*innovation
    event=(t>=0)&(np.mod(t,80)==40)
    jump=np.where(event,shock_rng.choice([-1.,1.],n)*.003,0.)
    gap=jump if spec['shock']=='gap' else np.zeros(n)
    if spec['shock']=='jump':moves=moves+jump
    close=1.1+np.cumsum(moves+gap)
    opened=np.r_[1.1,close[:-1]]+gap
    wick=amplitude*abs(wick_rng.normal(size=n))*.25+1e-6
    high=np.maximum(opened,close)+wick;low=np.minimum(opened,close)-wick
    if spec['mirror']:
        opened,high,low,close=2.2-opened,2.2-low,2.2-high,2.2-close
    stamps=[]
    origin=pd.Timestamp('2020-03-01T17:00',tz='America/New_York')
    for week in range(-5,target_bars//120):
        start=origin+pd.DateOffset(weeks=week)
        stamps.extend(pd.date_range(start,periods=120,freq='h').tz_convert('UTC'))
    frame=pd.DataFrame(dict(OPEN=opened,HIGH=high,LOW=low,CLOSE=close),
                       index=pd.DatetimeIndex(stamps).as_unit('ns'))
    if not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy()<=0).any():
        raise ValueError('SYNTHETIC_NONPOSITIVE_PRICE')
    return frame
