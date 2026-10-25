"""Native counterfactual witnesses: discretionary exit versus protective exit."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_native_early_close_and_hold_have_correct_directional_economics():
    root = os.environ.get('RECIPE_ENGINE_ROOT')
    if not root:
        pytest.skip('RECIPE_ENGINE_ROOT required')
    script = r'''
import sys
import numpy as np
import pandas as pd
from app.recipe_runtime import attach_engine
attach_engine(sys.argv[1])
from app.short_horizon.expert_guided.env import BarEnv,EnvConfig
from app.short_horizon.expert_guided.data import calendar_from_closures
from env_plugins.hybrid_orders import HybridOrderEnv
from app.frozen_recipe_inputs import FrozenRecipeInputs
from tests.unit.test_frozen_recipe_inputs import recipe
ts=pd.date_range('2020-01-05T22:00Z',periods=120,freq='h')
cal=calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],['2020-01-10T22:00Z','2020-01-12T22:00Z']])
frame=pd.DataFrame({'timestamp':ts+pd.Timedelta(hours=1),'x':np.sin(np.arange(120)),'y':np.cos(np.arange(120))})
inputs=FrozenRecipeInputs.from_frame(recipe(),frame)
def outcome(direction,favorable,close,q):
    bars=pd.DataFrame({'OPEN':1.1,'HIGH':1.10001,'LOW':1.09999,'CLOSE':1.1},index=ts)
    e=HybridOrderEnv(BarEnv(bars,cal,np.full(120,.001),np.full(120,.002),ts[0],ts[-1]+pd.Timedelta(hours=1),EnvConfig(max_entries_per_day=1)),inputs,gamma=1.)
    obs,_=e.reset()
    while not obs['action_mask'][1]: obs,*_=e.step([0,.5,.5,.5])
    obs,*_=e.step([1 if direction==1 else 2,.5,.5,q])
    pos=e.engine.pos
    assert pos is not None and pos.direction==direction
    next_bar=e.bar+1
    target=pos.tp if favorable else pos.sl
    # Scenario is synthetic; both branches receive identical future bars.
    e.engine.h=e.engine.h.copy();e.engine.l=e.engine.l.copy();e.engine.c=e.engine.c.copy()
    through=max(1e-5,2*e.engine.cfg.profile.limit_through)
    beyond=target + (through if target>pos.entry_price else -through)
    e.engine.h[next_bar]=max(pos.entry_price,beyond)
    e.engine.l[next_bar]=min(pos.entry_price,beyond)
    e.engine.c[next_bar]=beyond
    obs,reward,*_=e.step([3 if close and direction==1 else 4 if close else 0,.5,.5,.5])
    assert e.engine.pos is None, (direction,favorable,close,target,beyond,e.engine.cfg.profile.limit_through)
    trade=e.engine.trades[-1]
    assert trade['direction']==direction and trade['units']>0
    assert trade['close_reason']==('AGENT_CLOSE' if close else 'TP' if favorable else 'SL')
    assert abs(trade['gross_usd']-direction*trade['units']*(trade['exit_price']-trade['entry_price']))<1e-8
    assert abs(trade['net_usd']-(trade['gross_usd']-trade['entry_cost_usd']-trade['exit_cost_usd']))<1e-8
    assert e.engine.counts['levels_clipped']==0
    return trade
for direction in (-1,1):
 for favorable in (False,True):
  for q in (.1,.8):
   close=outcome(direction,favorable,True,q)
   hold=outcome(direction,favorable,False,q)
   assert close['units']==hold['units']
   assert (hold['net_usd']>close['net_usd']) if favorable else (close['net_usd']>hold['net_usd'])
print('NATIVE_CLOSE_HOLD_8_PAIRED_SCENARIOS_PASS')
'''
    p = subprocess.run([sys.executable, '-c', script, root],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
        text=True, capture_output=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
