"""Finite small-capacity contrasts using the existing selected SAC queue."""
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

SCHEMA = 'selected_rl_capacity_sequence.v1'
CAPACITIES = {'n8': [8], 'n16': [16], 'n16x16': [16, 16]}


def prepare(base_request, root, previous, previous_pin):
    root = selected._output_root(root)
    previous = Path(previous).resolve()
    if (not re.fullmatch('[0-9a-f]{64}', previous_pin)
            or previous.is_relative_to(root) or root.is_relative_to(previous)):
        raise ValueError('CAPACITY_SEPARATE_PINNED_PREDECESSOR_REQUIRED')
    base_request = Path(base_request).resolve()
    base = json.loads(base_request.read_text())
    if base.get('schema') != 'selected_rl_comparison_request.v4':
        raise ValueError('CAPACITY_PAIRED_UNREGULARIZED_BASE_REQUIRED')
    commit = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]),
                                     'rev-parse', 'HEAD'], text=True).strip()
    nodes = []
    for name, arch in CAPACITIES.items():
        request = copy.deepcopy(base)
        request.update(schema='selected_rl_comparison_request.v6', net_arch=arch)
        request['publication']['code_commit'] = commit
        request['publication']['experiment_set_key'] = base['publication']['experiment_set_key']+'-capacity-'+name
        path = root.parent/(root.name+'-requests')/(name+'.json')
        sequence.immutable(path, request)
        output = root/name
        contract = selected.materialize(path, output)
        pin = digest(contract)
        nodes.append(dict(profile=name, net_arch=arch, root=str(output), contract_sha256=pin,
            request_file=str(path), request_sha256=digest(path),
            predecessor_root=str(previous), predecessor_sha256=previous_pin))
        previous, previous_pin = output, pin
    spec = dict(schema=SCHEMA, base_request_file=str(base_request),
                base_request_sha256=digest(base_request), nodes=nodes)
    sequence.immutable(root/'SEQUENCE.json', spec)
    return spec


def validate(spec):
    if (spec.get('schema') != SCHEMA
            or [(n['profile'], n['net_arch']) for n in spec['nodes']] != list(CAPACITIES.items())):
        raise ValueError('CAPACITY_SEQUENCE_IDENTITY_INVALID')
    list(sequence.checked_nodes({**spec, 'schema': sequence.SCHEMA}))


def render(spec, directory, environment_file, spec_path, runtime_root=None):
    validate(spec)
    return sequence.render({**spec, 'schema': sequence.SCHEMA}, directory, environment_file, spec_path, runtime_root,
        prefix='rl-selected-capacity', sequence_module='tools.selected_rl_capacity_sequence',
        smoke_module='tools.selected_rl_capacity_sequence')


def status(spec):
    validate(spec)
    result = sequence.status({**spec, 'schema': sequence.SCHEMA})
    return {**result, 'schema': SCHEMA}


def publish(spec, spec_path):
    validate(spec)
    complete = True
    for node, root in sequence.checked_nodes({**spec, 'schema': sequence.SCHEMA}):
        if (selected.publish_and_close(root) != 0
                or predecessor_ready(root, node['contract_sha256']) != 'READY'):
            complete = False
    if complete:
        save_status(Path(spec_path).parent/'SEQUENCE_COMPLETE.json', dict(schema=SCHEMA,
            state='ALL_THREE_CLOSED_AND_WAREHOUSE_READBACK_VERIFIED',
            spec_sha256=digest(spec_path), capacities=CAPACITIES))
    return 0 if complete else 2


def smoke_model(arch, device):
    """Native synthetic updates and exact policy roundtrip, CPU-testable."""
    import gymnasium as gym
    import numpy as np
    import torch
    from agent_plugins.sac_agent import Plugin, _policy_tensor_hash
    if arch not in CAPACITIES.values():
        raise ValueError('CAPACITY_SMOKE_ARCHITECTURE_INVALID')
    class Synthetic(gym.Env):
        observation_space = gym.spaces.Box(-1., 1., shape=(201,), dtype=np.float32)
        action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(201, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(201, dtype=np.float32), -float(action[0]**2), False, False, {}
    cfg = dict(net_arch=arch, device=device, train_seed=7, buffer_size=64,
               batch_size=8, learning_starts=0)
    env = Synthetic(); model = Plugin().build(env, cfg)
    actor_widths = [m.out_features for m in model.actor.latent_pi if isinstance(m, torch.nn.Linear)]
    critic_widths = [m.out_features for m in model.critic.qf0 if isinstance(m, torch.nn.Linear)][:-1]
    other_widths = [m.out_features for m in model.critic.qf1 if isinstance(m, torch.nn.Linear)][:-1]
    if actor_widths != arch or critic_widths != arch or other_widths != arch:
        raise ValueError('CAPACITY_SMOKE_LAYER_MISMATCH')
    original = _policy_tensor_hash(model.policy)
    model.learn(total_timesteps=16)
    if original == _policy_tensor_hash(model.policy):
        raise ValueError('CAPACITY_SMOKE_NO_UPDATE')
    if not all(torch.isfinite(p).all().item() for p in model.policy.parameters()):
        raise ValueError('CAPACITY_SMOKE_NONFINITE')
    import tempfile
    from stable_baselines3 import SAC
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'smoke.zip'; model.save(path)
        loaded = SAC.load(path, env=env, device=device)
        obs, _ = env.reset()
        if (_policy_tensor_hash(model.policy) != _policy_tensor_hash(loaded.policy)
                or not np.array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])):
            raise ValueError('CAPACITY_SMOKE_RELOAD_MISMATCH')
    result = dict(net_arch=arch,
        actor_parameters=sum(p.numel() for p in model.actor.parameters()),
        critic_parameters=sum(p.numel() for p in model.critic.parameters()),
        synthetic_observation_dimension=201, updates_finite=True, reload_equal=True,
        gradient_updates=model._n_updates)
    env.close()
    return result


def smoke(root):
    """Bounded, idempotent synthetic CUDA admission; never financial data."""
    import os
    import torch
    from tools.curated_rl_campaign import validate_gpu_evidence
    root = Path(root)
    body = selected.validate(root)
    uuid = body['execution']['gpu_evidence']['assigned_gpu_uuid']
    validate_gpu_evidence(body['execution'])
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != uuid or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1):
        return 78
    path = root/'CAPACITY_GPU_SMOKE.json'
    identity = dict(contract_sha256=digest(root/'CONTRACT.json'), gpu_uuid=uuid,
                    net_arch=body['common_config']['net_arch'])
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and previous.get('identity') != identity:
        return 78
    if previous.get('state') == 'VERIFIED':
        return 0
    attempt = previous.get('attempt', 0)+1
    if attempt > 2:
        return 78
    save_status(path, dict(schema=SCHEMA, identity=identity, attempt=attempt, state='STARTED'))
    try:
        result = smoke_model(identity['net_arch'], 'cuda')
    except (ValueError, RuntimeError) as exc:
        save_status(path, dict(schema=SCHEMA, identity=identity, attempt=attempt,
                              state='FAILED', error_type=type(exc).__name__))
        return 1 if attempt < 2 else 78
    save_status(path, dict(schema=SCHEMA, identity=identity, attempt=attempt,
        state='VERIFIED', classification='SYNTHETIC_ENGINEERING_ONLY', **result))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', choices=('prepare', 'render', 'status', 'publish'))
    p.add_argument('--smoke-root', type=Path)
    p.add_argument('--spec', type=Path)
    p.add_argument('--base-request', type=Path)
    p.add_argument('--root', type=Path)
    p.add_argument('--predecessor-root', type=Path)
    p.add_argument('--predecessor-sha256')
    p.add_argument('--environment-file', type=Path)
    p.add_argument('--units-dir', type=Path)
    p.add_argument('--runtime-root', type=Path)
    a = p.parse_args()
    if a.smoke_root:
        return smoke(a.smoke_root)
    if a.action == 'prepare':
        result = prepare(a.base_request, a.root, a.predecessor_root, a.predecessor_sha256)
    else:
        spec = json.loads(a.spec.read_text())
        if a.action == 'publish': return publish(spec, a.spec)
        if a.action == 'status': result = status(spec)
        else: result = [str(v) for v in render(spec, a.units_dir or a.spec.parent/'units',
            a.environment_file, a.spec, a.runtime_root)]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
