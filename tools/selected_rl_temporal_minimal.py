"""Bind the existing causal Conv1D SAC to the tested finite paired queue."""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess

from tools import selected_rl_comparison as selected
from tools import selected_rl_regularization_sequence as sequence
from tools.curated_rl_campaign import digest
from tools.verify_rl_predecessor import predecessor_ready
from tools.verify_rl_static_engineering import save_status

ARCH = dict(net_arch=[8], branch_filters=2, core_filters=4, kernel_size=3)
MODULE = 'tools.selected_rl_temporal_minimal'


def prepare(base_request, root, previous, previous_pin, *, fixed_budget=False):
    root = selected._output_root(root); previous = Path(previous).resolve()
    if (not re.fullmatch('[0-9a-f]{64}', previous_pin)
            or previous.is_relative_to(root) or root.is_relative_to(previous)):
        raise ValueError('TEMPORAL_SEPARATE_PINNED_PREDECESSOR_REQUIRED')
    base_request = Path(base_request).resolve()
    request = copy.deepcopy(json.loads(base_request.read_text()))
    if request['schema'] != 'selected_rl_comparison_request.v4':
        raise ValueError('TEMPORAL_PAIRED_BASE_REQUIRED')
    request.update(schema='selected_rl_comparison_request.v8' if fixed_budget else 'selected_rl_comparison_request.v7', temporal_encoder=ARCH)
    if fixed_budget:
        from agent_plugins.fixed_budget_monitor import SETTINGS
        request['early_stopping']=SETTINGS
    request['publication']['code_commit'] = subprocess.check_output(
        ['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()
    request['publication']['experiment_set_key'] += '-temporal-minimal-2x4-n8'
    if fixed_budget:request['publication']['experiment_set_key']+='-fixed100k'
    path = root.parent/(root.name+'-request.json')
    sequence.immutable(path, request)
    output = root/'conv1d2x4-n8'
    contract = selected.materialize(path, output)
    node = dict(profile='conv1d2x4-n8', root=str(output), contract_sha256=digest(contract),
        request_file=str(path), request_sha256=digest(path), predecessor_root=str(previous),
        predecessor_sha256=previous_pin)
    spec = dict(schema=sequence.SCHEMA, experiment_kind='MINIMAL_CAUSAL_TEMPORAL_SAC',
        architecture=ARCH, nodes=[node])
    if fixed_budget:spec['monitoring']=request['early_stopping']
    sequence.immutable(root/'SEQUENCE.json', spec)
    return spec


def validate(spec):
    if 'monitoring' in spec:
        from agent_plugins.fixed_budget_monitor import SETTINGS
        if spec['monitoring']!=SETTINGS:raise ValueError('FIXED_MONITOR_SPEC_INVALID')
    if (spec.get('experiment_kind') != 'MINIMAL_CAUSAL_TEMPORAL_SAC'
            or spec.get('architecture') != ARCH or len(spec.get('nodes', [])) != 1
            or spec['nodes'][0]['profile'] != 'conv1d2x4-n8'):
        raise ValueError('MINIMAL_TEMPORAL_SPEC_REQUIRED')
    list(sequence.checked_nodes(spec))


def render(spec, directory, environment_file, spec_path, runtime_root=None):
    validate(spec)
    return sequence.render(spec, directory, environment_file, spec_path, runtime_root,
        prefix='rl-selected-temporal-fixed' if 'monitoring' in spec else 'rl-selected-temporal-minimal', sequence_module=MODULE, smoke_module=MODULE)


def publish(spec, spec_path):
    validate(spec)
    node = spec['nodes'][0]; root = Path(node['root'])
    if 'monitoring' in spec and not (root/'CLOSE.json').exists():
        from tools.fixed_budget_diagnostics import refresh
        refresh(root)
    if selected.publish_and_close(root) != 0:
        return 2
    if predecessor_ready(root, node['contract_sha256']) != 'READY':
        return 2
    save_status(Path(spec_path).parent/'SEQUENCE_COMPLETE.json',dict(
        state='TEMPORAL_COMPLETE_AND_WAREHOUSE_READBACK_VERIFIED',
        spec_sha256=digest(spec_path), architecture=ARCH))
    return 0


def smoke_model(device):
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from agent_plugins.sac_agent import Plugin, _policy_tensor_hash
    import tempfile
    class Synthetic(gym.Env):
        observation_space = gym.spaces.Dict(dict(
            features=gym.spaces.Box(-10.,10.,(13,15),dtype=np.float32),
            position=gym.spaces.Box(-1.,1.,(1,),dtype=np.float32)))
        action_space = gym.spaces.Box(-1.,1.,(1,),dtype=np.float32)
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {k:np.zeros(s.shape,dtype=np.float32) for k,s in self.observation_space.spaces.items()}, {}
        def step(self, action):
            obs,_=self.reset()
            return obs,-float(action[0]**2),False,False,{}
    cfg=dict(observation_encoder='per_feature_conv1d',net_arch=[8],temporal_branch_filters=2,
        temporal_core_filters=4,temporal_kernel_size=3,device=device,train_seed=7,
        buffer_size=64,batch_size=8,learning_starts=0)
    env=Synthetic();model=Plugin(cfg).build(env,cfg)
    before=_policy_tensor_hash(model.policy)
    model.learn(16)
    if before==_policy_tensor_hash(model.policy) or not all(torch.isfinite(p).all().item() for p in model.policy.parameters()):
        raise ValueError('TEMPORAL_SMOKE_UPDATE_FAILED')
    obs,_=env.reset();action=model.predict(obs,deterministic=True)[0]
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'policy.zip';model.save(path)
        restored=SAC.load(path,env=env,device=device)
        if (_policy_tensor_hash(model.policy)!=_policy_tensor_hash(restored.policy)
                or not np.array_equal(action,restored.predict(obs,deterministic=True)[0])):
            raise ValueError('TEMPORAL_SMOKE_RELOAD_MISMATCH')
    result=dict(architecture=ARCH,gradient_updates=model._n_updates,
        synthetic_actor_parameters=sum(p.numel() for p in model.actor.parameters()),
        synthetic_critic_parameters=sum(p.numel() for p in model.critic.parameters()),
        synthetic_observation_shape=[13,15],synthetic_auxiliary_size=1,
        policy_reload_equal=True,updates_finite=True)
    env.close()
    return result


def smoke(root):
    import os
    import torch
    from tools.curated_rl_campaign import validate_gpu_evidence
    root=Path(root);body=selected.validate(root)
    validate_gpu_evidence(body['execution'])
    uuid=body['execution']['gpu_evidence']['assigned_gpu_uuid']
    if (body['arms']!=['SAC_PER_FEATURE_CONV1D'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid
            or not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        return 78
    path=root/'TEMPORAL_GPU_SMOKE.json'
    identity=dict(contract_sha256=digest(root/'CONTRACT.json'),gpu_uuid=uuid)
    old=json.loads(path.read_text()) if path.exists() else {}
    if old and old.get('identity')!=identity: return 78
    if old.get('state')=='VERIFIED': return 0
    attempt=old.get('attempt',0)+1
    if attempt>2:return 78
    save_status(path,dict(identity=identity,state='STARTED',attempt=attempt))
    try:
        result=smoke_model('cuda')
    except (ValueError,RuntimeError) as exc:
        save_status(path,dict(identity=identity,state='FAILED',attempt=attempt,error_type=type(exc).__name__))
        return 1 if attempt<2 else 78
    save_status(path,dict(identity=identity,state='VERIFIED',attempt=attempt,
        classification='SYNTHETIC_ENGINEERING_ONLY',**result))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',choices=('prepare','render','status','publish'))
    p.add_argument('--smoke-root',type=Path)
    p.add_argument('--base-request',type=Path);p.add_argument('--root',type=Path)
    p.add_argument('--predecessor-root',type=Path);p.add_argument('--predecessor-sha256')
    p.add_argument('--spec',type=Path);p.add_argument('--environment-file',type=Path)
    p.add_argument('--units-dir',type=Path);p.add_argument('--runtime-root',type=Path)
    p.add_argument('--fixed-budget',action='store_true')
    a=p.parse_args()
    if a.smoke_root:return smoke(a.smoke_root)
    if a.action=='prepare':result=prepare(a.base_request,a.root,a.predecessor_root,a.predecessor_sha256,fixed_budget=a.fixed_budget)
    else:
        spec=json.loads(a.spec.read_text());validate(spec)
        if a.action=='publish':return publish(spec,a.spec)
        if a.action=='status':result=sequence.status(spec)
        else:result=[str(v) for v in render(spec,a.units_dir or a.spec.parent/'units',
            a.environment_file,a.spec,a.runtime_root)]
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':
    raise SystemExit(main())
