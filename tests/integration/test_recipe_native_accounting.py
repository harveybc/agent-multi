"""Same existing business engine, all bars, native SAC, no expert data path."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_engine_parity_and_native_training(tmp_path):
    root = os.environ.get('RECIPE_ENGINE_ROOT')
    if not root:
        pytest.skip('RECIPE_ENGINE_ROOT required for cross-repository engine witness')
    code = r'''
import sys
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from app.recipe_runtime import attach_engine, build_sac
attach_engine(sys.argv[1])
from app.short_horizon.expert_guided.env import BarEnv, EnvConfig
from app.short_horizon.expert_guided.data import calendar_from_closures
from app.short_horizon.expert_guided.actions import decode
from app.short_horizon.execution import Sizing
from app.frozen_recipe_inputs import FrozenRecipeInputs
from env_plugins.recipe_native import RecipeNativeEnv
from tests.unit.test_frozen_recipe_inputs import recipe
ts = pd.date_range('2020-01-05T22:00Z', periods=120, freq='h')
bars = pd.DataFrame({'OPEN':1.1, 'HIGH':1.10001, 'LOW':1.09999, 'CLOSE':1.1}, index=ts)
cal = calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],
                             ['2020-01-10T22:00Z','2020-01-12T22:00Z']])
f = pd.DataFrame({'timestamp':ts, 'x':np.sin(np.arange(120)), 'y':np.cos(np.arange(120))})
f.loc[[30,31,32,119], 'x'] = np.nan
inputs = FrozenRecipeInputs.from_frame(recipe(),f)
cfg = EnvConfig(max_entries_per_day=1, min_orders_per_week=2,
                sizing=Sizing(rel_volume=.1,leverage=50))
def engine():
    return BarEnv(bars,cal,np.full(120,.001),np.full(120,.002),ts[0],ts[-1]+pd.Timedelta(hours=1),cfg)
env = RecipeNativeEnv(engine(),inputs)
control = engine()
obs,_=env.reset()
for b in range(control.first,env.bar): control.step(b,0)
total=0.
while True:
    b = env.bar
    action=np.array([1. if b<60 else -1.,0.,0.],np.float32)
    d,tp,sl=decode(action,control.pos.direction if control.pos else 0)
    obs,r,done,trunc,info=env.step(action)
    control.step(b,d,tp,sl)
    for k in range(b+1,env.bar): control.step(k,0)
    assert env.engine.balance==control.balance
    assert env.engine.trades==control.trades
    assert set(obs)=={'features'} and np.isfinite(obs['features']).all()
    total+=info['elapsed_bars']
    if done: break
assert total+env.coverage['prefix_bars']==119
assert env.engine.pos is None and env.engine.trades
assert all(t['entry_time']>=ts[12] for t in env.engine.trades)
assert max(env.engine.entries_day.values())<=1
assert all(t['commission_usd']>0 and t['spread_usd']>0 for t in env.engine.trades)
assert env.engine.counts['close_FORCED_FLAT']>=1
assert any(t['financing_usd']!=0 for t in env.engine.trades)
a,b=env.engine.result(),control.result()
for key in ['net_profit_usd','sharpe_daily_sqrt365','max_drawdown_pct','trades','win_rate']:
    assert a[key]==b[key],key
design=dict(seed=7,learning_rate=.0003,buffer_size=64,batch_size=4,learning_starts=2,gamma=.99,tau=.005)
model=build_sac(RecipeNativeEnv(engine(),inputs),recipe(),design,'GROUPED','cpu')
model.learn(total_timesteps=40)
assert model._n_updates>0 and model.replay_buffer.discounts[:40].min()<.99
obs,_=env.reset()
before=model.predict(obs,deterministic=True)[0]
model.save(sys.argv[2]+'/model')
from stable_baselines3 import SAC
loaded=SAC.load(sys.argv[2]+'/model',device='cpu')
np.testing.assert_array_equal(before,loaded.predict(obs,deterministic=True)[0])
assert len(model.actor.features_extractor.branches)==1
assert all('features' in model.replay_buffer.observations for _ in [0])
print('ACCOUNTING_AND_NATIVE_TRAINING_PASS')
'''
    result = subprocess.run([sys.executable, '-c', code, root, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True,
                            text=True, timeout=120, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
