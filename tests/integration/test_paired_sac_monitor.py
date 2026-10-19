"""Synthetic real-engine subprocess replay; no financial data or GPU required."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_real_monitor_does_not_modify_learning_and_restores_best(tmp_path):
    gym = os.environ.get('GYM_FX_CHECKOUT')
    if not gym:
        pytest.skip('matching GYM_FX_CHECKOUT required')
    code = r'''
import json, random, pickle, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from tests.unit.test_selected_rl_comparison import fixture, write, digest
from tools import selected_rl_comparison as selected, curated_rl_campaign as queue
from agent_plugins.paired_early_stopping import SETTINGS, monitor_pair, make_callback
from agent_plugins.sac_agent import Plugin as Agent, _policy_tensor_hash
from env_plugins.gym_fx_env import Plugin as Env
root, gym = map(Path, sys.argv[1:])
path, request, _ = fixture(root)
request.update(schema='selected_rl_comparison_request.v4', leverage=50,
    early_stopping=SETTINGS, order_budget=dict(rel_volume=.1,
        min_order_volume=1, max_order_volume=1e12, volume_step=1))
request['execution'].update(gym_root=str(gym), gym_code_commit=subprocess.check_output(
    ['git','-C',str(gym),'rev-parse','HEAD'],text=True).strip())
write(path,request)
out = root/'campaign'
contract = selected.materialize(path,out)
body = queue.load_contract(contract)
d = out/'training'/'attempt-1'; d.mkdir(parents=True)
cfg=queue._task_config(body,dict(arm='SAC_FLAT',split='train',seed=7,id='test',attempt=1),d,d/'policy.zip')
queue.verify_execution_runtime(body['execution'],cfg)
agent=Agent(cfg); owner=Env(cfg); env=agent.wrap_env(owner.make_env(cfg),cfg)
model=agent.build(env,{**cfg,'learning_starts':0,'batch_size':2,'buffer_size':100})
try:
    model.learn(total_timesteps=8)
    before=_policy_tensor_hash(model.policy)
    replay=model.replay_buffer.size()
    rng=(random.getstate(),pickle.dumps(np.random.get_state()),torch.get_rng_state().clone())
    record=monitor_pair(cfg,model,8)
    assert _policy_tensor_hash(model.policy)==before
    assert model.replay_buffer.size()==replay
    assert random.getstate()==rng[0] and pickle.dumps(np.random.get_state())==rng[1]
    assert torch.equal(torch.get_rng_state(),rng[2])
    assert set(record['growth'])=={'train','validation'}
    for task in record['reports']:
        summary=json.loads(Path(task['summary']).read_text())
        assert summary['mode']=='inference'
        assert summary['static_policy']==dict(before=before,after=before)
        assert record['growth'][task['split']]['rows']==15
    model.learn(total_timesteps=8,reset_num_timesteps=False)
    assert _policy_tensor_hash(model.policy)!=before
    steps,updates=model.num_timesteps,model._n_updates
    callback=make_callback(cfg); callback.model=model; callback.best=record
    callback.records=[record]; callback.monitor.observe(8,0.,0.)
    callback._on_training_end()
    assert _policy_tensor_hash(model.policy)==before
    assert model.num_timesteps==steps and model._n_updates==updates
    restored_source=SAC.load(record['checkpoint'],device='cpu')
    for name in ('actor','critic'):
        actual=getattr(model,name).optimizer.state_dict()
        expected=getattr(restored_source,name).optimizer.state_dict()
        assert actual['param_groups']==expected['param_groups']
        assert actual['state'].keys()==expected['state'].keys()
        for key,values in actual['state'].items():
            for field,value in values.items():
                wanted=expected['state'][key][field]
                assert torch.equal(value,wanted) if isinstance(value,torch.Tensor) else value==wanted
    assert model.paired_early_stopping_evidence['best_step']==8
    assert not any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout))
                   for m in model.policy.modules())
    print('PAIRED_MONITOR_REAL_ENGINE_VERIFIED')
finally:
    owner.close()
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path), gym],
        cwd=Path(__file__).resolve().parents[2], env={**os.environ,
            'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'},
        text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    assert 'PAIRED_MONITOR_REAL_ENGINE_VERIFIED' in result.stdout
