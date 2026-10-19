"""Synthetic mechanics only: real registered SAC train/evaluate, no financial fit."""
import copy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pytest

from tools import curated_rl_campaign as queue


def write(path, body):
    path.write_text(json.dumps(body))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract(tmp_path):
    splits = {}
    for role, day in [('train', 8), ('validation', 9)]:
        start = datetime(2024, 1, day, tzinfo=timezone.utc)
        stamps = [start + timedelta(hours=i) for i in range(8)]
        data = tmp_path / f'{role}.csv'
        data.write_text('DATE_TIME,OPEN,HIGH,LOW,CLOSE,feature\n' + ''.join(
            f'{stamp.isoformat()},1,2,0.5,1.1,{i / 10}\n'
            for i, stamp in enumerate(stamps)))
        receipt = tmp_path / f'{role}.json'
        pin = write(receipt, {
            'schema': 'curated_rl_input.v1', 'status': 'ADMITTED',
            'availability_verified': True, 'open_session_coverage_verified': True,
            'input_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
            'split': role, 'feature_columns': ['feature'],
            'selection_sha256': 'a' * 64, 'calendar_sha256': 'b' * 64,
            'inventory_sha256': 'c' * 64,
            'timestamp_semantics': 'completed_bar_available_at',
            'period_start': stamps[0].isoformat(),
            'period_end': (stamps[-1] + timedelta(hours=1)).isoformat(),
            'row_count': len(stamps), 'expected_timestamps_sha256': hashlib.sha256(
                ''.join(s.isoformat() + '\n' for s in stamps).encode()).hexdigest(),
        })
        splits[role] = {'input_data_file': str(data), 'curated_input_manifest': str(receipt),
                        'curated_input_manifest_sha256': pin,
                        'curated_input_split': role, 'feature_columns': ['feature']}
    body = {'schema': 'curated_rl_campaign.v1', 'classification': 'NON_GOVERNING',
            'seeds': [7], 'splits': splits,
            'publication': {'campaign_sha256': 'd' * 64,
                            'deliveries': {'train': [], 'validation': []},
                            'unit_ids': {f'7/{arm}/{split}': f'synthetic-7-{arm}-{split}'
                                         for arm in queue.ARMS for split in ('train', 'validation')}},
            'common_config': {
                'env_plugin': 'gym_fx_env', 'pipeline_plugin': 'rl_pipeline',
                'total_timesteps': 4, 'training_transition_cap': 4,
                'evaluation_transition_cap': 3, 'learning_starts': 2,
                'buffer_size': 32, 'batch_size': 2, 'net_arch': [8],
                'require_feature_aware_preprocessor': True,
                'preprocessor_plugin': 'feature_window_preprocessor',
                'feature_scaling': 'rolling_zscore', 'include_price_window': False,
                'window_size': 13, 'observation_stride_bars': 2,
                'require_full_observation_window': True,
                'max_entries_per_day': 1, 'execution_transition': 'next_open_v2',
                'strategy_plugin': 'shared_execution_envelope'}}
    body['common_config'].update(action_space_mode='continuous',
                                continuous_action_contract='target_exposure_hysteresis_v2')
    path = tmp_path / 'campaign.json'
    write(path, body)
    return path, body


class ToyEnv(gym.Env):
    observation_space = spaces.Dict({
        'features': spaces.Box(-10, 10, (13, 1), dtype=np.float32),
        'position': spaces.Box(-1, 1, (1,), dtype=np.float32)})
    action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

    def __init__(self, cfg):
        self.config = cfg
        self.dataframe = pd.read_csv(cfg['input_data_file'])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.count = 0
        self.action_space.seed(seed)
        return {k: np.zeros(s.shape, dtype=np.float32)
                for k, s in self.observation_space.spaces.items()}, {'equity': 10000.}

    def step(self, action):
        self.count += 1
        return {k: np.zeros(s.shape, dtype=np.float32)
                for k, s in self.observation_space.spaces.items()}, 0., self.count >= 3, False, {
                    'bar_index': self.count, 'equity': 10000.}

    def summary(self):
        return {'net_profit': 0., 'sharpe_ratio': None}


class Outbox:
    def __init__(self):
        self.items = {}

    def put(self, envelope):
        self.items[envelope['unit_id']] = copy.deepcopy(envelope)

    def status(self):
        return {'pending': list(self.items)}


@pytest.fixture
def mechanics(monkeypatch):
    from env_plugins.gym_fx_env import Plugin as Env
    from agent_plugins.sac_agent import Plugin as SAC
    calls = []
    original = SAC.train

    def train(self, model, cfg):
        calls.append((cfg['observation_encoder'], cfg['curated_input_split']))
        return original(self, model, cfg)

    monkeypatch.setattr(SAC, 'train', train)
    monkeypatch.setattr(Env, 'make_env', lambda self, cfg: ToyEnv(cfg))
    return calls


def execute(cfg, directory, timeout):
    queue.execute_task(cfg)
    return 0


def test_sequential_real_trainer_checkpoint_and_frozen_evaluator(tmp_path, monkeypatch, mechanics):
    from stable_baselines3 import SAC
    from agent_plugins.sac_agent import Plugin as Agent
    path, body = contract(tmp_path)
    outbox = Outbox()
    output = tmp_path / 'output'
    assert queue.run(path, output, outbox=outbox, executor=execute) == 0
    assert mechanics == [('flat', 'train'), ('per_feature_conv1d', 'train')]
    state = queue.status(output)
    assert state['state'] == 'COMPLETED_OUTBOX_PENDING'
    assert len(state['tasks']) == len(outbox.items) == 12
    assert all(t['state'] == 'ENQUEUED' for t in state['tasks'])
    assert all(t.get('policy_tensor_sha256') for t in state['tasks'][:4:2])
    assert [t['arm'] for t in state['tasks']][::2] == list(queue.ARMS)
    for task in state['tasks'][:4]:
        cfg = json.loads(Path(task['config']).read_text())
        assert cfg['device'] == 'cpu' and cfg['max_entries_per_day'] == 1
        summary = json.loads(Path(task['summary']).read_text())
        if task['split'] == 'train':
            assert summary['compute_contract']['training_transitions_observed'] == 4
            assert summary['compute_contract']['gradient_updates'] == 2
        else:
            assert summary['static_policy']['before'] == summary['static_policy']['after']
            train = next(t for t in state['tasks'] if t['arm'] == task['arm'] and t['split'] == 'train')
            assert summary['static_policy']['before'] == train['policy_tensor_sha256']
            assert summary['mode'] == 'inference'
        assert all(m['metric'] != 'sharpe_ratio' for envelope in outbox.items.values()
                   for m in envelope['terminal']['metrics'])

    def forbidden(*args, **kwargs):
        raise AssertionError('evaluator tried to learn')

    monkeypatch.setattr(SAC, 'learn', forbidden)
    monkeypatch.setattr(Agent, 'train', forbidden)
    check_dir = tmp_path / 'frozen-validation-check'
    check_dir.mkdir()
    validation = queue._task_config(body, state['tasks'][1], check_dir,
                                    Path(state['tasks'][0]['checkpoint']))
    queue.execute_task(validation)
    before = (output / 'STATUS.json').read_bytes()
    assert queue.run(path, output, resume=True, outbox=outbox, executor=forbidden) == 0
    assert queue.status(output)['tasks'] == state['tasks']
    queue.status(output)
    assert (output / 'STATUS.json').read_bytes() == before


def test_typed_failure_resume_and_outbox_retry_do_not_retrain(tmp_path, mechanics):
    path, _ = contract(tmp_path)
    output, outbox = tmp_path / 'output', Outbox()
    calls = []

    def fail_second(cfg, directory, timeout):
        calls.append(cfg['curated_input_split'])
        if len(calls) == 2:
            raise subprocess.TimeoutExpired('synthetic', timeout)
        return execute(cfg, directory, timeout)

    assert queue.run(path, output, outbox=outbox, executor=fail_second) == 2
    assert queue.status(output)['tasks'][1]['state'] == 'FAILED_TIMEOUT'
    failed_record = Path(queue.status(output)['tasks'][1]['config']).parent / 'EXECUTION.json'
    failed_bytes = failed_record.read_bytes()
    assert queue.run(path, output, resume=True, outbox=outbox, executor=execute) == 0
    assert mechanics == [('flat', 'train'), ('per_feature_conv1d', 'train')]
    assert queue.status(output)['tasks'][1]['attempt'] == 2
    assert failed_record.read_bytes() == failed_bytes
    assert json.loads(failed_bytes)['state'] == 'FAILED_TIMEOUT'

    class FailingOutbox(Outbox):
        def put(self, envelope):
            raise OSError('synthetic outbox unavailable')

    other = tmp_path / 'other'
    assert queue.run(path, other, outbox=FailingOutbox(), executor=execute) == 2
    assert queue.status(other)['tasks'][0]['state'] == 'PUBLICATION_PENDING'
    trained = len(mechanics)
    assert queue.run(path, other, resume=True, outbox=Outbox(), executor=execute) == 0
    assert len(mechanics) == trained + 1  # only the remaining temporal arm


@pytest.mark.parametrize('mutation', ['digest', 'bytes', 'split', 'governing', 'optimizer'])
def test_refusal_before_dispatch(tmp_path, mutation):
    path, body = contract(tmp_path)
    if mutation == 'digest':
        body['splits']['train']['curated_input_manifest_sha256'] = '0' * 64
    if mutation == 'bytes':
        Path(body['splits']['validation']['input_data_file']).write_text('corrupt')
    if mutation == 'split':
        body['splits']['validation']['curated_input_split'] = 'test'
    if mutation == 'governing':
        body['classification'] = 'GOVERNING'
    if mutation == 'optimizer':
        body['common_config']['use_optimizer'] = True
    write(path, body)
    with pytest.raises(ValueError):
        queue.run(path, tmp_path / 'output', outbox=Outbox(), executor=lambda *a: pytest.fail('dispatched'))


def test_resume_refuses_contract_and_checkpoint_drift(tmp_path, mechanics):
    path, body = contract(tmp_path)
    output = tmp_path / 'output'
    assert queue.run(path, output, outbox=Outbox(), executor=execute) == 0
    body['seeds'] = [8]
    write(path, body)
    with pytest.raises(ValueError, match='CONTRACT_DRIFT'):
        queue.run(path, output, resume=True, outbox=Outbox(), executor=execute)
    body['seeds'] = [7]
    write(path, body)
    state = queue.status(output)
    Path(state['tasks'][0]['checkpoint']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='CHECKPOINT_DRIFT'):
        queue.run(path, output, resume=True, outbox=Outbox(), executor=execute)


def test_registered_controls_and_short_sign(tmp_path, mechanics):
    path, _ = contract(tmp_path)
    output = tmp_path / 'output'
    assert queue.run(path, output, outbox=Outbox(), executor=execute) == 0
    tasks = queue.status(output)['tasks']
    for arm, sign in [('NO_TRADE', 0), ('PASSIVE_LONG', 1), ('PASSIVE_SHORT', -1)]:
        task = next(t for t in tasks if t['arm'] == arm and t['split'] == 'validation')
        summary = json.loads(Path(task['summary']).read_text())
        assert summary['action_raw_mean'] == sign


def test_existing_outbox_adapter_delegates(monkeypatch, tmp_path):
    seen = []
    class Existing:
        def __init__(self, root):
            seen.append(root)
    monkeypatch.setattr(queue.governed_run, '_governed_exec',
                        lambda: type('Shared', (), {'TerminalOutbox': Existing}))
    assert isinstance(queue.open_outbox(tmp_path), Existing)
    assert seen == [tmp_path / 'outbox']


def test_real_durable_outbox_without_services(tmp_path, mechanics):
    path, _ = contract(tmp_path)
    output = tmp_path / 'output'
    real = queue.open_outbox(output)
    class AmbiguousWrite:
        def put(self, envelope):
            real.put(envelope)
            raise OSError('synthetic lost enqueue acknowledgment')
    assert queue.run(path, output, outbox=AmbiguousWrite(), executor=execute) == 2
    assert len(list((output / 'outbox' / 'pending').glob('*.json'))) == 1
    assert queue.run(path, output, resume=True, executor=execute) == 0
    assert mechanics == [('flat', 'train'), ('per_feature_conv1d', 'train')]
    pending = sorted((output / 'outbox' / 'pending').glob('*.json'))
    assert len(pending) == 12
    before = {p.name: p.read_bytes() for p in pending}
    assert queue.run(path, output, resume=True, executor=lambda *a: pytest.fail('retrained')) == 0
    assert {p.name: p.read_bytes() for p in pending} == before
    for path in pending:
        envelope = json.loads(path.read_text())
        assert isinstance(envelope['terminal']['artifacts'], list)
        assert all(m['split'] in ('train', 'validation')
                   for m in envelope['terminal']['metrics'])


@pytest.mark.parametrize('encoder', ['flat', 'per_feature_conv1d'])
def test_real_child_entry_with_synthetic_env(tmp_path, encoder):
    path, body = contract(tmp_path)
    directory = tmp_path / 'child'
    directory.mkdir()
    arm = 'SAC_FLAT' if encoder == 'flat' else 'SAC_PER_FEATURE_CONV1D'
    cfg = queue._task_config(body, {'arm': arm, 'split': 'train', 'seed': 7},
                             directory, directory / 'policy.zip')
    write(directory / 'config.json', cfg)
    bootstrap = '''
import sys
from env_plugins.gym_fx_env import Plugin
from tests.unit.test_curated_rl_campaign import ToyEnv
from tools.curated_rl_campaign import main
Plugin.make_env = lambda self, cfg: ToyEnv(cfg)
raise SystemExit(main(sys.argv[1:]))
'''
    result = subprocess.run([sys.executable, '-c', bootstrap, '_task', '--config',
                             str(directory / 'config.json')], cwd=queue.ROOT,
                            capture_output=True, text=True, timeout=30,
                            env={**os.environ, **queue.CPU_ENV})
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['compute_contract']['training_transitions_observed'] == 4
    assert summary['compute_contract']['gradient_updates'] == 2


def test_writer_lock_and_nonzero_child_are_typed(tmp_path):
    path, _ = contract(tmp_path)
    output = tmp_path / 'output'
    output.mkdir()
    with (output / '.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='ALREADY_RUNNING'):
            queue.run(path, output, outbox=Outbox())
    assert queue.run(path, output, outbox=Outbox(), executor=lambda *a: 9) == 2
    state = queue.status(output)
    assert state['state'] == 'FAILED_EXECUTION'
    assert state['tasks'][0]['returncode'] == 9
    assert state['tasks'][1]['state'] == 'READY'


def test_interrupted_stage_retries_new_attempt(tmp_path, mechanics):
    path, _ = contract(tmp_path)
    output = tmp_path / 'output'
    def interrupt(*args):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        queue.run(path, output, outbox=Outbox(), executor=interrupt)
    assert queue.status(output)['tasks'][0]['state'] == 'RUNNING'
    assert queue.run(path, output, resume=True, outbox=Outbox(), executor=execute) == 0
    assert queue.status(output)['tasks'][0]['attempt'] == 2


def test_default_executor_is_bounded_cpu_registered_child(tmp_path, monkeypatch):
    seen = []
    def child(command, **kwargs):
        seen.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(queue.subprocess, 'run', child)
    assert queue._subprocess({}, tmp_path, 3, lock_fd=42) == 0
    cmd, kwargs = seen[0]
    assert cmd[1:5] == ['-m', 'tools.curated_rl_campaign', '_task', '--config']
    assert kwargs['timeout'] == 3 and kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert kwargs['cwd'] == queue.ROOT
    assert kwargs['pass_fds'] == (42,)


def test_unavailable_status_is_read_only_and_setup_does_not_train(tmp_path, monkeypatch):
    output = tmp_path / 'not-created'
    assert queue.status(output)['state'] == 'UNAVAILABLE'
    assert not output.exists()
    path, _ = contract(tmp_path)
    def unavailable(*args):
        raise SystemExit('synthetic missing dependency')
    monkeypatch.setattr(queue, 'open_outbox', unavailable)
    assert queue.run(path, output, executor=lambda *a: pytest.fail('trained')) == 2
    assert queue.status(output)['state'] == 'PUBLICATION_UNAVAILABLE'
    assert all(t['attempt'] == 0 for t in queue.status(output)['tasks'])


@pytest.mark.parametrize('change', [
    {'train_freq': 8}, {'max_entries_per_day': 4}, {'action_space_mode': 'discrete'},
    {'final_stage_c_evaluation': True}, {'training_transition_cap': 2}])
def test_protocol_bypass_refused_before_dispatch(tmp_path, change):
    path, body = contract(tmp_path)
    body['common_config'].update(change)
    write(path, body)
    with pytest.raises(ValueError):
        queue.run(path, tmp_path / 'output', outbox=Outbox(),
                  executor=lambda *a: pytest.fail('dispatched'))


@pytest.mark.parametrize('mutation', ['overlap', 'lineage', 'sealed', 'unit_id'])
def test_receipt_and_publication_contract_refusals(tmp_path, mutation):
    path, body = contract(tmp_path)
    split = body['splits']['validation']
    receipt_path = Path(split['curated_input_manifest'])
    receipt = json.loads(receipt_path.read_text())
    if mutation == 'overlap':
        receipt['period_start'] = '2024-01-08T00:00:00+00:00'
    if mutation == 'lineage':
        receipt['selection_sha256'] = 'f' * 64
    if mutation == 'sealed':
        receipt['period_end'] = '2025-01-02T00:00:00+00:00'
    if mutation == 'unit_id':
        body['publication']['unit_ids'] = {}
    split['curated_input_manifest_sha256'] = write(receipt_path, receipt)
    write(path, body)
    with pytest.raises(ValueError):
        queue.run(path, tmp_path / 'output', outbox=Outbox(),
                  executor=lambda *a: pytest.fail('dispatched'))


def development(tmp_path, device='cpu'):
    path, body = contract(tmp_path)
    body.update(scientific_scope='REAL_DATA_DEVELOPMENT', execution_backend='local_subprocess',
                execution={'device': device, 'gym_root': str(tmp_path / 'gym'),
                           'gym_code_commit': 'a' * 40,
                           'task_timeout_seconds': {'train': 14400, 'validation': 3600},
                           'max_attempts_per_task': 2})
    body['publication'] = {'mode': 'WAREHOUSE_REPORTS', 'predictor_root': str(tmp_path / 'predictor'),
        'predictor_code_commit': 'b' * 40, 'code_commit': 'c' * 40,
        'actor': 'synthetic-rl', 'lake': 'synthetic-fixtures',
        'experiment_set_key': 'synthetic-development',
        'resources': {'train': 'train.csv', 'validation': 'validation.csv'}}
    if device == 'cuda':
        uuid = 'GPU-00000000-0000-0000-0000-000000000001'
        heartbeat = {'schema': 'agent_multi.gpu_readiness_heartbeat.v1',
            'hostname': '<synthetic-host>', 'generated_utc': datetime.now(timezone.utc).isoformat(),
            'classification': 'GPU_READY', 'running_kernel': {'release': 'synthetic-kernel',
                'modinfo_nvidia': {'ok': True}},
            'driver': {'nvidia_smi_ok': True, 'driver_version': 'synthetic-driver'},
            'gpus': {'observed_uuids': [uuid], 'expected_uuids': [uuid], 'exact_match': True},
            'framework': {'status': 'TORCH_CUDA_OK', 'per_uuid': {uuid: {
                'status': 'TORCH_CUDA_OK', 'torch_present': True, 'cuda_available': True,
                'cuda_device_count': 1, 'cuda_visible_devices': uuid,
                'torch_version': 'synthetic-torch', 'torch_cuda_version': 'synthetic-cuda'}}},
            'disk': {'classification': 'HOST_DISK_SUFFICIENT'}}
        evidence = tmp_path / 'gpu.json'
        body['execution']['gpu_evidence'] = {'path': str(evidence), 'sha256': write(evidence, heartbeat),
            'assigned_gpu_uuid': uuid, 'max_age_seconds': 86400}
    write(path, body)
    return path, body


def test_real_development_contract_order_device_timeouts_and_custody(tmp_path):
    path, body = development(tmp_path, 'cuda')
    body['common_config'].update(total_timesteps=100000, training_transition_cap=100000)
    write(path, body)
    loaded = queue.load_contract(path)
    assert queue.arm_order(loaded) == ['NO_TRADE', 'PASSIVE_LONG', 'PASSIVE_SHORT', 'RANDOM',
                                      'SAC_FLAT', 'SAC_PER_FEATURE_CONV1D']
    for arm, device in [('NO_TRADE', 'cpu'), ('SAC_FLAT', 'cuda')]:
        directory = tmp_path / arm
        directory.mkdir()
        cfg = queue._task_config(body, {'arm': arm, 'split': 'train', 'seed': 7},
                                 directory, directory / 'policy.zip')
        assert cfg['device'] == device
        assert cfg['session_flatten_custody_root'] == str(directory / 'session_custody')
        assert cfg['scientific_scope'] == 'REAL_DATA_DEVELOPMENT'
        assert cfg['total_timesteps'] == (100000 if device == 'cuda' else 0)


@pytest.mark.parametrize('field,value', [('classification', 'GOVERNING'),
    ('scientific_scope', 'REAL_DATA_RELEASE'), ('execution_backend', 'remote')])
def test_development_does_not_invent_a_governance_class_or_backend(tmp_path, field, value):
    path, body = development(tmp_path)
    body[field] = value
    write(path, body)
    with pytest.raises(ValueError):
        queue.load_contract(path)


@pytest.mark.parametrize('change', ['digest', 'fallback', 'driver', 'stale', 'assignment'])
def test_cuda_evidence_fail_closed_without_hardware_probe(tmp_path, change):
    path, body = development(tmp_path, 'cuda')
    proof = body['execution']['gpu_evidence']
    evidence_path = Path(proof['path'])
    hb = json.loads(evidence_path.read_text())
    if change == 'fallback': hb['framework']['status'] = 'TORCH_CPU_FALLBACK'
    if change == 'driver': hb['driver']['nvidia_smi_ok'] = False
    if change == 'stale': hb['generated_utc'] = '2000-01-01T00:00:00Z'
    if change == 'assignment': proof['assigned_gpu_uuid'] = 'GPU-other'
    proof['sha256'] = write(evidence_path, hb)
    if change == 'digest': proof['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='GPU'):
        queue.validate_gpu_evidence(body['execution'])


def test_cuda_child_preserves_assignment_and_rejects_ambient_mismatch(tmp_path, monkeypatch):
    _, body = development(tmp_path, 'cuda')
    uuid = body['execution']['gpu_evidence']['assigned_gpu_uuid']
    cfg = {'device': 'cuda', 'queue_execution': body['execution']}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', uuid)
    assert queue.child_environment(cfg)['CUDA_VISIBLE_DEVICES'] == uuid
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='GPU'):
        queue.child_environment(cfg)


def test_gpu_runtime_refuses_cpu_fallback_before_environment(tmp_path):
    _, body = development(tmp_path, 'cuda')
    fake = type('Torch', (), {'cuda': type('Cuda', (), {'is_available': lambda: False})})
    with pytest.raises(ValueError, match='GPU'):
        queue.runtime_device(body['execution'], torch_module=fake)


def test_retry_cap_and_sealed_timeout_are_used_without_training(tmp_path, monkeypatch):
    path, body = development(tmp_path)
    seen = []
    monkeypatch.setattr(queue, 'verify_execution_runtime', lambda *a: None)
    monkeypatch.setattr(queue, 'prepare_warehouse', lambda *a: None)
    for attempt in (1, 2):
        assert queue.run(path, tmp_path / 'output', resume=attempt > 1, outbox=Outbox(),
                         executor=lambda cfg, d, limit: seen.append(limit) or 9) == 2
    assert seen == [14400, 14400]
    assert queue.run(path, tmp_path / 'output', resume=True, outbox=Outbox(),
                     executor=lambda *a: pytest.fail('retry budget bypassed')) == 2
    assert queue.status(tmp_path / 'output')['state'] == 'ATTEMPTS_EXHAUSTED'
    with pytest.raises(ValueError, match='TIMEOUT'):
        queue.run(path, tmp_path / 'other', timeout=120, outbox=Outbox())


def test_warehouse_report_has_real_local_identity_and_no_flow_v3_claim(tmp_path):
    _, body = development(tmp_path)
    body.update(input_policy='RAW_CORE_PRICES', cost_scope='ANALYTICAL_NOT_BROKER_VERIFIED')
    task = {'id': 'local-task', 'arm': 'SAC_FLAT', 'seed': 7, 'split': 'validation',
            'wall_seconds': 2., 'artifacts': {'config': '1' * 64, 'summary': '2' * 64,
                'checkpoint': '3' * 64, 'effective_config': '4' * 64}}
    report = queue.warehouse_report(body, {'contract_sha256': 'f' * 64}, task,
                                    {'net_profit': 5., 'sharpe_ratio': None, 'episode_length': 8,
                                     'execution_diagnostics': {'fills': {'rejected': 2}}})
    assert report['tags']['classification'] == 'NON_GOVERNING'
    assert report['tags']['scientific_scope'] == 'REAL_DATA_DEVELOPMENT'
    assert report['tags']['local_campaign_sha256'] == 'f' * 64
    assert report['tags']['checkpoint_archive_sha256'] == '3' * 64
    assert report['lineage'] == 'UNVERIFIED'
    assert report['tags']['feature_selection_complete'] == 'false'
    assert report['tags']['cost_scope'] == body['cost_scope']
    assert next(m for m in report['metrics'] if m['metric'] == 'net_profit')['unit'] == 'USD'
    assert next(m for m in report['metrics'] if m['metric'] == 'execution_diagnostics.fills.rejected')['value'] == 2
    assert next(m for m in report['metrics'] if m['metric'] == 'sharpe_ratio')['value'] is None
    assert all(m['split'] == 'validation' for m in report['metrics'])
    assert 'campaign_sha256' not in report and 'delivery_id' not in json.dumps(report)


@pytest.mark.parametrize('arm', ['SAC_FLAT', 'SAC_PER_FEATURE_CONV1D'])
def test_pinned_registered_real_gym_entry_not_a_mock(tmp_path, arm):
    gym_root = os.environ.get('GYM_FX_CHECKOUT')
    if not gym_root:
        pytest.skip('set GYM_FX_CHECKOUT to a committed gym-fx checkout')
    path, body = development(tmp_path)
    commit = subprocess.run(['git', '-C', gym_root, 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    body['execution'].update(gym_root=gym_root, gym_code_commit=commit)
    body['common_config'].update(timeframe='1h', timestamp_semantics='completed_bar_available_at',
        metrics_plugin='trading_metrics', include_agent_state=True, agent_state_contract='live_stationary_v2')
    for role, day in [('train', 8), ('validation', 15)]:
        split = body['splits'][role]
        stamps = [datetime(2024, 1, day, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(80)]
        data = Path(split['input_data_file'])
        data.write_text('DATE_TIME,OPEN,HIGH,LOW,CLOSE,VOLUME,feature\n' + ''.join(
            f'{stamp.isoformat()},1.1,1.1001,1.0999,1.1,1,{i / 80}\n'
            for i, stamp in enumerate(stamps)))
        receipt_path = Path(split['curated_input_manifest'])
        receipt = json.loads(receipt_path.read_text())
        receipt.update(row_count=80, input_sha256=queue.digest(data),
            period_start=stamps[0].isoformat(), period_end=(stamps[-1] + timedelta(hours=1)).isoformat(),
            expected_timestamps_sha256=hashlib.sha256(''.join(s.isoformat() + '\n' for s in stamps).encode()).hexdigest())
        split['curated_input_manifest_sha256'] = write(receipt_path, receipt)
    write(path, body)
    queue.load_contract(path)
    train_dir, val_dir = tmp_path / 'fit', tmp_path / 'eval'
    train_dir.mkdir(); val_dir.mkdir()
    for role, directory in [('train', train_dir), ('validation', val_dir)]:
        cfg = queue._task_config(body, {'arm': arm, 'split': role, 'seed': 7}, directory,
                                 train_dir / 'policy.zip')
        write(directory / 'config.json', cfg)
    bootstrap = '''
import json, sys
from pathlib import Path
from tools.curated_rl_campaign import main
assert main(['_task', '--config', sys.argv[1]]) == 0
import app.main, app.env, gym_fx.env
assert Path(app.main.__file__).resolve() == Path.cwd() / 'app/main.py'
assert Path(app.env.__file__).resolve() == Path(sys.argv[2]) / 'app/env.py'
assert Path(gym_fx.env.__file__).resolve() == Path(sys.argv[2]) / 'gym_fx/env.py'
from stable_baselines3 import SAC
SAC.learn = lambda *a, **k: (_ for _ in ()).throw(AssertionError('evaluation learned'))
assert main(['_task', '--config', sys.argv[3]]) == 0
'''
    result = subprocess.run([sys.executable, '-c', bootstrap, str(train_dir / 'config.json'),
                             gym_root, str(val_dir / 'config.json')], cwd=queue.ROOT,
                            capture_output=True, text=True, timeout=60,
                            env={**os.environ, **queue.CPU_ENV})
    assert result.returncode == 0, result.stdout + result.stderr
    trained = json.loads((train_dir / 'summary.json').read_text())
    evaluated = json.loads((val_dir / 'summary.json').read_text())
    assert trained['compute_contract']['training_transitions_observed'] == 4
    assert evaluated['static_policy']['before'] == evaluated['static_policy']['after']
    assert evaluated['static_policy']['before'] == trained['checkpoint_policy_tensor_sha256']
    assert evaluated['runtime_identity']['gym_code_commit'] == commit


def test_existing_predictor_outbox_real_sqlite_and_idempotence_without_auth(tmp_path):
    import sqlite3
    from tools import rl_warehouse
    runtime = os.environ.get('PREDICTOR_CURATION_RUNTIME')
    if not runtime:
        pytest.skip('set PREDICTOR_CURATION_RUNTIME to the existing predictor checkout')
    _, body = development(tmp_path)
    body['publication'].update(predictor_root=runtime, predictor_code_commit=subprocess.run(
        ['git', '-C', runtime, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip())
    task = {'id': 'synthetic-local', 'arm': 'NO_TRADE', 'seed': 7, 'split': 'train', 'wall_seconds': 1.,
            'artifacts': {k: 'a' * 64 for k in ('config', 'summary', 'checkpoint', 'effective_config')}}
    report = queue.warehouse_report(body, {'contract_sha256': 'c' * 64}, task,
                                    {'net_profit': 0., 'sharpe_ratio': None, 'episode_length': 3})
    request = tmp_path / 'request.json'
    write(request, {'report': report, 'source_unit': task['id']})
    first = rl_warehouse.invoke(body['publication'], tmp_path, 'enqueue', request)
    second = rl_warehouse.invoke(body['publication'], tmp_path, 'enqueue', request)
    assert first['report_sha256'] == second['report_sha256']
    with sqlite3.connect(tmp_path / 'warehouse-outbox.sqlite') as conn:
        rows = conn.execute('SELECT body,state,attempts FROM items').fetchall()
    assert len(rows) == 1 and rows[0][1:] == ('PENDING', 0)
    sealed = json.loads(rows[0][0])
    assert sealed['report_sha256'] == first['report_sha256']
    assert sealed['lineage'] == 'UNVERIFIED'
    assert all(m['split'] == 'train' for m in sealed['metrics'])
