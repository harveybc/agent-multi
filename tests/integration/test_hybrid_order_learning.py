import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('open_command,close_command,direction', [(1, 3, 1), (2, 4, -1)])
def test_hybrid_native_contract_ppo_and_reload(tmp_path, open_command, close_command, direction):
    root=os.environ.get('RECIPE_ENGINE_ROOT')
    if not root:
        pytest.skip('RECIPE_ENGINE_ROOT required')
    script=r'''
import sys
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from app.recipe_runtime import attach_engine
attach_engine(sys.argv[1])
from app.short_horizon.expert_guided.env import BarEnv,EnvConfig
from app.short_horizon.expert_guided.data import calendar_from_closures
from env_plugins.hybrid_orders import HybridOrderEnv
from agent_plugins.hybrid_orders import OrderPolicy,ElapsedRolloutBuffer
from app.frozen_recipe_inputs import FrozenRecipeInputs
from tests.unit.test_frozen_recipe_inputs import recipe
torch.set_num_threads(1)
open_command,close_command,direction=map(int,sys.argv[3:6])
ts=pd.date_range('2020-01-05T22:00Z',periods=120,freq='h')
bars=pd.DataFrame({'OPEN':1.1,'HIGH':1.10001,'LOW':1.09999,'CLOSE':1.1},index=ts)
cal=calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],['2020-01-10T22:00Z','2020-01-12T22:00Z']])
frame=pd.DataFrame({'timestamp':ts+pd.Timedelta(hours=1),'x':np.sin(np.arange(120)),'y':np.cos(np.arange(120))})
frame.loc[[30,31,32],'x']=np.nan
inputs=FrozenRecipeInputs.from_frame(recipe(),frame)
def env():
    return HybridOrderEnv(BarEnv(bars,cal,np.full(120,.001),np.full(120,.002),ts[0],ts[-1]+pd.Timedelta(hours=1),EnvConfig(max_entries_per_day=1)),inputs,gamma=1.)
e=env(); obs,_=e.reset(); total=0.
while not obs['action_mask'][1]:
    obs,r,done,_,_=e.step([0,.5,.5,.5]); total+=r
obs,r,done,_,_=e.step([open_command,.5,.5,.2]); total+=r
expected_mask=[1,0,0,0,0]; expected_mask[close_command]=1
assert e.engine.pos.direction==direction and obs['action_mask'].tolist()==expected_mask
# Consuming the daily entry allowance must not disable a discretionary exit.
assert not e.engine.entry_gate(e.bar)[0]
assert e.engine.cfg.percentual_order_size<.21
assert e.engine.counts['levels_clipped']==0
assert e.engine.pos.units*e.engine.pos.entry_price/e.engine.cfg.sizing.leverage <= e.engine.cfg.percentual_order_size*e.engine.cfg.sizing.rel_volume*e.engine.balance
obs,r,done,_,_=e.step([close_command,.5,.5,.5]);total+=r
assert e.engine.pos is None and e.engine.trades[-1]['net_usd']<0
assert e.engine.trades[-1]['close_reason']=='AGENT_CLOSE'
assert e.engine.counts['close_AGENT_CLOSE']==1
assert all(e.engine.counts[k]==0 for k in ('close_SL','close_TP','close_FORCED_FLAT'))
while not done:
    obs,r,done,_,_=e.step([0,.5,.5,.5]); total+=r
assert abs(total*100-sum(t['net_usd'] for t in e.engine.trades))<1e-8
class Duration(BaseCallback):
    def _on_step(self):
        self.model.rollout_buffer.pending_duration=np.array([i['elapsed_bars'] for i in self.locals['infos']])
        return True
model=PPO(OrderPolicy,env(),policy_kwargs={'recipe':recipe()},n_steps=32,batch_size=16,n_epochs=2,
          rollout_buffer_class=ElapsedRolloutBuffer,seed=7,device='cpu')
model.learn(64,callback=Duration())
obs,_=env().reset()
before=model.predict(obs,deterministic=True)[0]
model.save(sys.argv[2]+'/model')
loaded=PPO.load(sys.argv[2]+'/model',device='cpu')
np.testing.assert_array_equal(before,loaded.predict(obs,deterministic=True)[0])
obs_t,_=model.policy.obs_to_tensor(obs)
model.policy.set_training_mode(False)
a,v,lp=model.policy(obs_t)
model.policy.set_training_mode(True)
v2,lp2,entropy=model.policy.evaluate_actions(obs_t,a)
torch.testing.assert_close(lp,lp2,atol=0,rtol=0)
assert torch.isfinite(entropy).all()
print('HYBRID_NATIVE_AND_PPO_PASS')
'''
    p=subprocess.run([sys.executable,'-c',script,root,str(tmp_path),str(open_command),str(close_command),str(direction)],cwd=Path(__file__).resolve().parents[2],
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        text=True,capture_output=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
