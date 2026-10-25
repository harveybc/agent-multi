import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_business_witness_and_commands_native_sac(tmp_path):
    engine=os.environ.get('RECIPE_ENGINE_ROOT')
    if not engine:
        pytest.skip('RECIPE_ENGINE_ROOT required')
    script=r'''
import sys
import torch
import numpy as np
import pandas as pd
torch.set_num_threads(1)
from tools.order_contract_study import witness
report=witness(sys.argv[1])
assert report['state']=='PASS' and len(report['cases'])==16
from app.recipe_runtime import build_sac
from app.short_horizon.expert_guided.env import BarEnv,EnvConfig
from app.short_horizon.expert_guided.data import calendar_from_closures
from env_plugins.order_contract import OrderContractEnv,CONTRACTS
from app.frozen_recipe_inputs import FrozenRecipeInputs
from tests.unit.test_frozen_recipe_inputs import recipe
ts=pd.date_range('2020-01-05T22:00Z',periods=120,freq='h')
bars=pd.DataFrame({'OPEN':1.1,'HIGH':1.10001,'LOW':1.09999,'CLOSE':1.1},index=ts)
cal=calendar_from_closures([['2020-01-03T22:00Z','2020-01-05T22:00Z'],
                          ['2020-01-10T22:00Z','2020-01-12T22:00Z']])
frame=pd.DataFrame({'timestamp':ts,'x':np.sin(np.arange(120)),'y':np.cos(np.arange(120))})
frame.loc[[30,31,32,119],'x']=np.nan
inputs=FrozenRecipeInputs.from_frame(recipe(),frame)
e=BarEnv(bars,cal,np.full(120,.001),np.full(120,.002),ts[0],ts[-1]+pd.Timedelta(hours=1),EnvConfig())
env=OrderContractEnv(e,inputs,contract=CONTRACTS['COMMANDS5'])
d=dict(seed=7,learning_rate=.0003,buffer_size=64,batch_size=4,learning_starts=2,gamma=.99,tau=.005)
m=build_sac(env,recipe(),d,'GROUPED','cpu')
m.learn(total_timesteps=40)
assert m._n_updates>0 and m.replay_buffer.discounts[:40].min()<.99
obs,_=env.reset()
before=m.predict(obs,deterministic=True)[0]
m.save(sys.argv[2]+'/model')
from stable_baselines3 import SAC
loaded=SAC.load(sys.argv[2]+'/model',device='cpu')
np.testing.assert_array_equal(before,loaded.predict(obs,deterministic=True)[0])
print('ORDER_ALPHA_AND_NATIVE_SAC_PASS')
'''
    result=subprocess.run([sys.executable,'-c',script,engine,str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,timeout=120,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    assert result.returncode==0,result.stdout+result.stderr
