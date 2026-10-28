"""Candidate corrected engine only: actual categorical wrapper, no market data."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_candidate_rollover_rewards_and_zero_cost(tmp_path):
    root = os.environ.get('CORRECTED_ENGINE_ROOT')
    if not root:
        pytest.skip('CORRECTED_ENGINE_ROOT required; do not substitute a frozen legacy engine')
    script = r'''
import sys
import numpy as np
import pandas as pd
from app.recipe_runtime import attach_engine
attach_engine(sys.argv[1])
from app.short_horizon.expert_guided.env import BarEnv, EnvConfig
from app.short_horizon.expert_guided.data import calendar_from_closures
from env_plugins.hybrid_orders import HybridOrderEnv
from env_plugins.recipe_native import equity
from app.frozen_recipe_inputs import FrozenRecipeInputs
from tests.unit.test_frozen_recipe_inputs import recipe
ts = pd.date_range('2020-01-05T22:00Z', periods=120, freq='h')
cal = calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],
                            ['2020-01-10T22:00Z','2020-01-12T22:00Z']])
frame = pd.DataFrame({'timestamp':ts+pd.Timedelta(hours=1),
                      'x':np.sin(np.arange(120)), 'y':np.cos(np.arange(120))})
inputs = FrozenRecipeInputs.from_frame(recipe(), frame)
for mode in ('configured','zero_cost'):
 for command in (1,2):
    bars = pd.DataFrame(dict(OPEN=1.1,HIGH=1.10001,LOW=1.09999,CLOSE=1.1),index=ts)
    engine = BarEnv(bars,cal,np.full(120,.001),np.full(120,.002),ts[0],
                    ts[-1]+pd.Timedelta(hours=1),EnvConfig(cost_mode=mode,max_entries_per_day=1))
    env = HybridOrderEnv(engine,inputs,gamma=1.,learned_risk=False)
    obs,_ = env.reset()
    while not obs['action_mask'][command]: obs,*_ = env.step([0,.5,.5,.5])
    obs,total,*_ = env.step([command,.5,.5,.5])
    p = engine.pos
    assert p is not None
    rate = engine.cfg.profile.swap_long_per_lot if command==1 else engine.cfg.profile.swap_short_per_lot
    while env.bar < 22:
        _,r,*_ = env.step([0,.5,.5,.5]); total += r
    # End of bar 23 is exactly the first rollover. No close occurs.
    _,r,*_ = env.step([0,.5,.5,.5]); total += r
    expected = 100 * rate * p.units / engine.cfg.profile.lot_units / engine.cfg.capital
    assert abs(r-expected)<1e-10, (r, expected, mode, command)
    assert equity(engine,env.bar)==engine.equity_at(env.bar)
    assert 'features' in obs and set(obs)=={'features','action_mask'}
    while not env.finished:
        _,r,*_ = env.step([0,.5,.5,.5]); total += r
    assert abs(total - 100*(engine.balance-engine.cfg.capital)/engine.cfg.capital)<1e-9
    assert engine.result()['end_equity_usd']==engine.balance
    if mode=='zero_cost': assert abs(total)<1e-12
print('CORRECTED_REWARD_AND_ZERO_COST_PASS')
'''
    p = subprocess.run([sys.executable, '-c', script, root],
                       cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
                       timeout=120, env={**os.environ, 'CUDA_VISIBLE_DEVICES':'',
                                        'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'})
    assert p.returncode == 0, p.stdout + p.stderr
