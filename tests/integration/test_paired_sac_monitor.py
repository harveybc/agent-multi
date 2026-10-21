"""Synthetic real-engine subprocess replay; no financial data or GPU required."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('profile',[None,'dropout005','l2_1e6','dropout005_l2_1e6','batchnorm','TEMPORAL','FIXED','CAPACITY','WINDOW7','WINDOW25'])
def test_real_monitor_does_not_modify_learning_and_restores_best(tmp_path,profile):
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
root, gym = map(Path, sys.argv[1:3])
profile=sys.argv[3]
path, request, _ = fixture(root)
window_profile=profile in ('WINDOW7','WINDOW25')
if window_profile:
    import csv, hashlib
    from datetime import datetime, timedelta
    receipt_path=Path(request['receipt_file']); receipt=json.loads(receipt_path.read_text())
    for item in receipt['splits'].values():
        data=receipt_path.parent/item['input_file']; rows=list(csv.DictReader(data.open()))
        first=datetime.fromisoformat(rows[0]['DATE_TIME'])
        expanded=[]
        for i in range(80):
            row=dict(rows[i%len(rows)]);stamp=(first+timedelta(hours=i)).isoformat()
            row.update(DATE_TIME=stamp,SELECTED_AVAILABLE_AT=stamp);expanded.append(row)
        with data.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(expanded[0]));writer.writeheader();writer.writerows(expanded)
        manifest_path=receipt_path.parent/item['manifest_file'];manifest=json.loads(manifest_path.read_text())
        manifest.update(input_sha256=digest(data),row_count=len(expanded),expected_timestamps_sha256=
            hashlib.sha256(''.join(r['DATE_TIME']+'\n' for r in expanded).encode()).hexdigest())
        write(manifest_path,manifest);item['manifest_sha256']=digest(manifest_path)
    write(receipt_path,receipt);request['receipt_sha256']=digest(receipt_path)
request.update(schema='selected_rl_comparison_request.v4', leverage=50,
    early_stopping=SETTINGS, order_budget=dict(rel_volume=.1,
        min_order_volume=1, max_order_volume=1e12, volume_step=1))
if profile in ('TEMPORAL','FIXED','CAPACITY') or window_profile:
    request.update(schema='selected_rl_comparison_request.v7',temporal_encoder=dict(
        net_arch=[8],branch_filters=2,core_filters=4,kernel_size=3))
    if profile in ('FIXED','CAPACITY') or window_profile:
        from agent_plugins.fixed_budget_monitor import SETTINGS as FIXED_SETTINGS
        request.update(schema='selected_rl_comparison_request.v8',early_stopping=FIXED_SETTINGS)
    if profile=='CAPACITY':
        request.update(schema='selected_rl_comparison_request.v9',experiment_seed=17,
            temporal_encoder=dict(net_arch=[16],branch_filters=4,core_filters=8,kernel_size=3,core_layers=2))
    if window_profile:
        request.update(schema='selected_rl_comparison_request.v11',experiment_seed=7,
            window_size=int(profile.removeprefix('WINDOW')),
            temporal_encoder=dict(net_arch=[8],branch_filters=2,core_filters=4,kernel_size=3,core_layers=1))
elif profile!='NONE':
    from agent_plugins.sac_regularization import PROFILES
    request.update(schema='selected_rl_comparison_request.v5',regularization=PROFILES[profile])
request['execution'].update(gym_root=str(gym), gym_code_commit=subprocess.check_output(
    ['git','-C',str(gym),'rev-parse','HEAD'],text=True).strip())
write(path,request)
out = root/'campaign'
contract = selected.materialize(path,out)
body = queue.load_contract(contract)
d = out/'training'/'attempt-1'; d.mkdir(parents=True)
arm=body['arms'][0]
cfg=queue._task_config(body,dict(arm=arm,split='train',seed=body['seeds'][0],id='test',attempt=1),d,d/'policy.zip')
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
        assert task['arm']==arm
        task_cfg=json.loads(Path(task['config']).read_text())
        assert task_cfg['observation_encoder']==cfg['observation_encoder']
        if profile=='CAPACITY':
            assert task_cfg['temporal_core_layers']==2 and task_cfg['train_seed']==17
            assert task['seed']==17
        summary=json.loads(Path(task['summary']).read_text())
        assert summary['mode']=='inference'
        assert summary['static_policy']==dict(before=before,after=before)
        assert record['growth'][task['split']]['rows']==(80-(request['window_size']-1)*2-1 if window_profile else 15)
    model.learn(total_timesteps=8,reset_num_timesteps=False)
    assert _policy_tensor_hash(model.policy)!=before
    steps,updates=model.num_timesteps,model._n_updates
    callback=make_callback(cfg); callback.model=model
    if profile in ('FIXED','CAPACITY') or window_profile:
        for tick in (25000,50000,75000):
            model.num_timesteps=tick
            assert callback._on_step() is True
        model.num_timesteps=100000
        assert callback._on_step() is True and len(callback.records)==3
        model.train(gradient_steps=1,batch_size=2)
        final_hash=_policy_tensor_hash(model.policy)
        final_updates=model._n_updates
        callback._on_training_end()
        assert _policy_tensor_hash(model.policy)==final_hash
        assert model._n_updates==final_updates
        assert model.paired_early_stopping_evidence['selected_step']==100000
        assert model.paired_early_stopping_evidence['stop_reason']=='FIXED_BUDGET'
        assert 'best_step' not in model.paired_early_stopping_evidence
        record=callback.records[-1]
    else:
        callback.best=record;callback.records=[record];callback.monitor.observe(8,0.,0.)
        callback._on_training_end()
        assert _policy_tensor_hash(model.policy)==before
        assert model.num_timesteps==steps and model._n_updates==updates
        assert model.paired_early_stopping_evidence['best_step']==8
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
    if profile in ('NONE','TEMPORAL','FIXED','CAPACITY') or window_profile:
        assert not any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout))
                       for m in model.policy.modules())
    else:
        assert model.policy.regularization==PROFILES[profile]
        assert not any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout))
                       for m in model.actor.modules())
    print('PAIRED_MONITOR_REAL_ENGINE_VERIFIED')
finally:
    owner.close()
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path), gym,profile or 'NONE'],
        cwd=Path(__file__).resolve().parents[2], env={**os.environ,
            'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'},
        text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    assert 'PAIRED_MONITOR_REAL_ENGINE_VERIFIED' in result.stdout
