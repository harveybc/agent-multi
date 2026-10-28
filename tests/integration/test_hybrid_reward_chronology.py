"""Independent native witnesses for economic reward and hidden-position ambiguity."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_native_reward_chronology_and_observation_aliasing():
    root=os.environ.get('RECIPE_ENGINE_ROOT')
    if not root:
        pytest.skip('RECIPE_ENGINE_ROOT required')
    script=r'''
import copy, sys
import numpy as np
import pandas as pd
from app.recipe_runtime import attach_engine
attach_engine(sys.argv[1])
from app.short_horizon.expert_guided.env import BarEnv,EnvConfig
from app.short_horizon.expert_guided.data import calendar_from_closures
from env_plugins.hybrid_orders import HybridOrderEnv
from env_plugins.recipe_native import equity
from app.frozen_recipe_inputs import FrozenRecipeInputs
from tests.unit.test_frozen_recipe_inputs import recipe
ts=pd.date_range('2020-01-05T22:00Z',periods=120,freq='h')
cal=calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],['2020-01-10T22:00Z','2020-01-12T22:00Z']])
frame=pd.DataFrame({'timestamp':ts+pd.Timedelta(hours=1),'x':np.sin(np.arange(120)),'y':np.cos(np.arange(120))})
def make(mode='equity',gamma=1.):
    bars=pd.DataFrame({'OPEN':1.1,'HIGH':1.10001,'LOW':1.09999,'CLOSE':1.1},index=ts)
    inputs=FrozenRecipeInputs.from_frame(recipe(),frame)
    e=HybridOrderEnv(BarEnv(bars,cal,np.full(120,.001),np.full(120,.002),ts[0],ts[-1]+pd.Timedelta(hours=1),EnvConfig(max_entries_per_day=1)),inputs,gamma=gamma,reward_mode=mode)
    obs,_=e.reset()
    while not obs['action_mask'][1]:obs,*_=e.step([0,.5,.5,.5])
    return e
for direction in (-1,1):
 for mode in ('equity','log'):
  for voluntary in (False,True):
    e=make(mode)
    start=equity(e.engine,e.bar)
    # Constant market makes the cost-only loss independently calculable.
    total=0.; t=0
    while not e.finished:
        cmd=(1 if direction==1 else 2) if t==0 else ((3 if direction==1 else 4) if voluntary and t==30 else 0)
        before=equity(e.engine,e.bar)
        _,r,terminated,truncated,info=e.step([cmd,.5,.5,.5])
        after=equity(e.engine,e.bar)
        expected=100*(after-before)/start if mode=='equity' else 100*np.log(after/before)
        assert abs(r-expected)<1e-10
        assert not truncated and terminated==e.finished
        total+=r;t+=1
    assert e.engine.pos is None and len(e.engine.trades)==1
    trade=e.engine.trades[0]
    assert trade['gross_usd']==0 and trade['rollovers']>0, trade
    prof=e.engine.cfg.profile
    rate=prof.swap_long_per_lot if direction>0 else prof.swap_short_per_lot
    rolls=e.engine.cal.rollover_ns(trade['segment'])
    entry=pd.Timestamp(trade['entry_time']).value;exit=pd.Timestamp(trade['exit_time']).value
    expected_fin=-rate*trade['units']/prof.lot_units*sum(3 if i==prof.triple_swap_rollover_index else 1
        for i,r in enumerate(rolls,start=1) if entry<r<=exit)
    assert abs(trade['financing_usd']-expected_fin)<1e-8
    costs=sum(trade[k] for k in ('spread_usd','slippage_usd','commission_usd','financing_usd'))
    assert abs(trade['net_usd']+costs)<1e-8
    assert abs(e.engine.balance-start+costs)<1e-8
    expected=100*(-costs)/start if mode=='equity' else 100*np.log((start-costs)/start)
    assert abs(total-expected)<1e-8
    assert trade['close_reason']==('AGENT_CLOSE' if voluntary else 'FORCED_FLAT')

# A non-observable interval still accrues reward on every engine bar.
e=make(gamma=.5)
e.step([1,.5,.5,.5]); start=e.bar
e.valid[start+1:start+3]=False
e.engine.c=e.engine.c.copy()
e.engine.c[start+1:start+4]=[1.1001,1.1002,1.1003]
trace=[]; original=e.reward.step
def spy(before,after):
    r=original(before,after);trace.append(r);return r
e.reward.step=spy
_,r,_,_,info=e.step([0,.5,.5,.5])
assert info['elapsed_bars']==3 and info['discount']==.5**3
assert abs(r-sum(.5**i*x for i,x in enumerate(trace)))<1e-12

# Same history/mask, different executable stops: one should exit, one hold.
for direction in (-1,1):
 e=make(); e.step([1 if direction==1 else 2,.5,.5,.5])
 variants=[]
 for tight in (True,False):
    v=copy.deepcopy(e);p=v.engine.pos
    p.sl=p.entry_price-direction*(.0006 if tight else .002)
    p.tp=p.entry_price+direction*.001
    k=v.bar+1
    v.engine.o=v.engine.o.copy();v.engine.h=v.engine.h.copy();v.engine.l=v.engine.l.copy();v.engine.c=v.engine.c.copy()
    # First bar is adverse, second reaches the take profit of the wide stop.
    for j,movement in ((k,-direction*.001),(k+1,direction*.0012)):
        price=p.entry_price+movement
        v.engine.h[j]=max(p.entry_price,price)
        v.engine.l[j]=min(p.entry_price,price)
        v.engine.c[j]=price
    variants.append(v)
 a,b=variants
 assert all(np.array_equal(a._observe()[key],b._observe()[key]) for key in a._observe())
 deltas=[]
 for v in variants:
    close=copy.deepcopy(v); hold=copy.deepcopy(v)
    close.step([3 if direction==1 else 4,.5,.5,.5])
    hold.step([0,.5,.5,.5])
    if hold.engine.pos is not None:hold.step([0,.5,.5,.5])
    assert close.engine.pos is None and hold.engine.pos is None
    deltas.append(close.engine.balance-hold.engine.balance)
 assert deltas[0]>0 and deltas[1]<0, deltas
print('NATIVE_REWARD_CHRONOLOGY_AND_ALIASING_PASS')
'''
    p=subprocess.run([sys.executable,'-c',script,root],cwd=Path(__file__).resolve().parents[2],
                     env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                     text=True,capture_output=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
