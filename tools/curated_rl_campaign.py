"""Finite CPU queue over the existing governed replay and registered RL pipeline.

Engineering-only, no admission issuance, network sends, promotion or sealed TEST.
Invoke as `python -m tools.curated_rl_campaign run|resume|status ...`.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from app.curated_rl_input import validate_curated_rl_input
from tools import governed_run
from tools.governed_offline_replay import compute_metrics, merged_config, metrics_of
from tools.verify_rl_static_engineering import save_status

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    'SAC_FLAT': ('sac_agent', 'flat'),
    'SAC_PER_FEATURE_CONV1D': ('sac_agent', 'per_feature_conv1d'),
    'NO_TRADE': ('no_trade_agent', 'flat'),
    'PASSIVE_LONG': ('buy_hold_agent', 'flat'),
    'PASSIVE_SHORT': ('buy_hold_agent', 'flat'),
    'RANDOM': ('random_agent', 'flat'),
}
SPLIT_KEYS = {'input_data_file', 'curated_input_manifest',
              'curated_input_manifest_sha256', 'curated_input_split', 'feature_columns'}
CPU_ENV = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _positive(value):
    return type(value) is int and value > 0


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_contract(path):
    body = json.loads(Path(path).read_text())
    if body.get('schema') != 'curated_rl_campaign.v1' or body.get('classification') != 'NON_GOVERNING':
        raise ValueError('QUEUE_ENGINEERING_CONTRACT_REQUIRED')
    seeds, common, splits = body['seeds'], body['common_config'], body['splits']
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 32
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('QUEUE_INVALID_SEEDS')
    for key in ('total_timesteps', 'training_transition_cap', 'evaluation_transition_cap'):
        if not _positive(common.get(key)):
            raise ValueError('QUEUE_FINITE_COMPUTE_CONTRACT_REQUIRED')
    if common['total_timesteps'] > common['training_transition_cap']:
        raise ValueError('QUEUE_TRAINING_CAP_EXCEEDED')
    if common.get('train_freq', 1) != 1:
        raise ValueError('QUEUE_SINGLE_TRANSITION_TRAIN_FREQ_REQUIRED')
    forbidden = ('use_optimizer', 'load_model', 'warm_start_model', 'warm_start_path',
                 'oracle_behavior_pretrain_enabled', 'remote_log',
                 'final_stage_c_evaluation', 'stage_c_acknowledged')
    if any(common.get(k) for k in forbidden):
        raise ValueError('QUEUE_UNSUPPORTED_FITTING_OR_REMOTE_PATH')
    if common.get('env_plugin') != 'gym_fx_env' or common.get('pipeline_plugin') != 'rl_pipeline':
        raise ValueError('QUEUE_REGISTERED_STATIC_PATH_REQUIRED')
    if (common.get('action_space_mode') != 'continuous'
            or common.get('continuous_action_contract') != 'target_exposure_hysteresis_v2'):
        raise ValueError('QUEUE_MATCHED_CONTINUOUS_ACTION_CONTRACT_REQUIRED')
    if (common.get('max_entries_per_day') != 1
            or (common.get('execution_envelope') or {}).get('max_entries_per_day', 1) != 1):
        raise ValueError('QUEUE_BASELINE_DAILY_CAP_REQUIRED')
    if set(splits) != {'train', 'validation'}:
        raise ValueError('QUEUE_TRAIN_VALIDATION_ONLY')
    publication = body['publication']
    units = publication.get('unit_ids', {})
    expected_units = {f'{seed}/{arm}/{role}' for seed in seeds for arm in ARMS
                      for role in ('train', 'validation')}
    if (not _pin(publication.get('campaign_sha256'))
            or set(publication.get('deliveries', {})) != set(splits)
            or any(not isinstance(v, list) for v in publication['deliveries'].values())
            or set(units) != expected_units
            or any(not isinstance(v, str) or not v.strip() for v in units.values())
            or len(set(units.values())) != len(units)):
        raise ValueError('QUEUE_EXISTING_PUBLICATION_IDENTITY_REQUIRED')
    receipts = []
    from env_plugins.gym_fx_env import Plugin as Env
    from pipeline_plugins._observation_contract import validate_observation_contract
    for role in ('train', 'validation'):
        split = splits[role]
        if set(split) != SPLIT_KEYS or split['curated_input_split'] != role:
            raise ValueError('QUEUE_SPLIT_CONTRACT_MISMATCH')
        if any(not Path(split[k]).is_absolute() for k in ('input_data_file', 'curated_input_manifest')):
            raise ValueError('QUEUE_ABSOLUTE_INPUT_PATH_REQUIRED')
        cfg = {**common, **split, 'require_curated_rl_input': True, 'max_rows': None}
        validate_curated_rl_input(cfg)
        validate_observation_contract(cfg)
        Env()._build_env_config(cfg)
        receipts.append(json.loads(Path(split['curated_input_manifest']).read_text()))
    train, validation = receipts
    from pipeline_plugins._return_trace import HELDOUT_BOUNDARY
    boundary = datetime.fromisoformat(HELDOUT_BOUNDARY).replace(tzinfo=timezone.utc)
    if any(datetime.fromisoformat(r['period_end'].replace('Z', '+00:00')) > boundary
           for r in receipts):
        raise ValueError('QUEUE_SEALED_TEST_SUPPORT_FORBIDDEN')
    if (datetime.fromisoformat(train['period_end'].replace('Z', '+00:00'))
            > datetime.fromisoformat(validation['period_start'].replace('Z', '+00:00'))):
        raise ValueError('QUEUE_OVERLAPPING_SPLITS_REQUIRE_SCORED_ORIGIN_PROTOCOL')
    for key in ('feature_columns', 'selection_sha256', 'calendar_sha256', 'inventory_sha256'):
        if train[key] != validation[key]:
            raise ValueError('QUEUE_SPLIT_LINEAGE_MISMATCH')
    return body


def open_outbox(output_root):
    # Shared durable publication API; sending/reconciliation stays with data-gov.
    return governed_run._governed_exec().TerminalOutbox(output_root / 'outbox')


def execute_task(config):
    """One existing registered pipeline call; no alternate learning/evaluation loop."""
    from app.main import _load_plugin_instance
    from agent_plugins.sac_agent import _policy_tensor_hash

    if config.get('device') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('QUEUE_CPU_REQUIRED')
    identity = validate_curated_rl_input(config)
    if identity is None:
        raise ValueError('QUEUE_CURATED_RECEIPT_REQUIRED')
    env = _load_plugin_instance('env.plugins', config['env_plugin'], config)
    agent = _load_plugin_instance('agent.plugins', config['agent_plugin'], config)
    pipeline = _load_plugin_instance('pipeline.plugins', config['pipeline_plugin'], config)
    defaults = {}
    for instance in (env, agent, pipeline):
        defaults.update(getattr(instance, 'plugin_params', {}))
    for key, value in defaults.items():
        config.setdefault(key, value)
    mode = config['mode']
    loaded = []
    saved = []
    original_save = agent.save

    def save(model, path):
        original_save(model, path)
        if hasattr(model, 'policy'):
            saved.append(_policy_tensor_hash(model.policy))

    agent.save = save
    original_build = agent.build

    def build(wrapped, cfg):
        wrapped.action_space.seed(cfg['train_seed'])
        return original_build(wrapped, cfg)

    agent.build = build
    if mode == 'inference':
        original_load = agent.load

        def load(path, wrapped):
            wrapped.action_space.seed(config['eval_seed'])
            model = original_load(path, wrapped)
            if hasattr(model, 'policy'):
                loaded.append((model, _policy_tensor_hash(model.policy)))
            return model

        def no_train(*args, **kwargs):
            raise RuntimeError('QUEUE_STATIC_EVALUATOR_CANNOT_TRAIN')

        agent.load, agent.train = load, no_train
    if config['queue_arm'] == 'PASSIVE_SHORT':
        original_predict = agent.predict
        agent.predict = lambda *a, **kw: -original_predict(*a, **kw)
    summary = pipeline.run_pipeline(config=config, env_plugin=env, agent_plugin=agent, mode=mode)
    summary['curated_input_identity'] = identity
    if saved:
        summary['checkpoint_policy_tensor_sha256'] = saved[0]
    for model, before in loaded:
        after = _policy_tensor_hash(model.policy)
        if before != after:
            raise RuntimeError('QUEUE_STATIC_POLICY_CHANGED')
        summary['static_policy'] = {'before': before, 'after': after}
    save_status(Path(config['results_file']), summary)
    save_status(Path(config['save_config']), config)
    return summary


def _subprocess(config, directory, timeout, *, lock_fd=None):
    with (directory / 'stdout.log').open('wb') as log:
        return subprocess.run(
            [sys.executable, '-m', 'tools.curated_rl_campaign', '_task',
             '--config', str(directory / 'config.json')],
            cwd=ROOT, env={**os.environ, **CPU_ENV}, stdout=log,
            stderr=subprocess.STDOUT, timeout=timeout, check=False,
            pass_fds=() if lock_fd is None else (lock_fd,)).returncode


def status(output_root):
    """Read only the local journal; never probes services, processes or the network."""
    try:
        body = json.loads((Path(output_root) / 'STATUS.json').read_text())
    except (OSError, ValueError) as exc:
        return {'schema': 'curated_rl_queue_status.v1', 'state': 'UNAVAILABLE',
                'reason': type(exc).__name__}
    if (not isinstance(body, dict) or body.get('schema') != 'curated_rl_queue_status.v1'
            or not isinstance(body.get('tasks'), list) or not body.get('state')):
        return {'schema': 'curated_rl_queue_status.v1', 'state': 'UNAVAILABLE',
                'reason': 'INVALID_STATUS_SCHEMA'}
    return body


def _verify_artifacts(state):
    for task in state['tasks']:
        for key, pin in task.get('artifacts', {}).items():
            path = Path(task[key])
            if not path.is_file() or digest(path) != pin:
                raise ValueError(f'QUEUE_{key.upper()}_DRIFT')


def _task_config(body, task, directory, checkpoint):
    agent, encoder = ARMS[task['arm']]
    base = copy.deepcopy(body['common_config'])
    base.update(body['splits'][task['split']])
    base.update(agent_plugin=agent, observation_encoder=encoder, queue_arm=task['arm'],
                mode='train' if task['split'] == 'train' else 'inference',
                require_curated_rl_input=True, max_rows=None, device='cpu',
                use_optimizer=False, train_seed=task['seed'], eval_seed=task['seed'],
                eval_deterministic=True, eval_split=task['split'],
                save_model=str(checkpoint) if task['split'] == 'train' else None,
                load_model=str(checkpoint) if task['split'] != 'train' else None)
    if agent != 'sac_agent':
        base['total_timesteps'] = 0  # registered controls' train hook is initialization only
    # Override all standard writable runtime paths; caller paths cannot escape the attempt.
    for key, name in {'results_file': 'summary.json', 'save_config': 'effective_config.json',
                      'resolved_config_file': 'resolved_config.json',
                      'config_manifest_file': 'config_manifest.json',
                      'training_progress_file': 'progress.json', 'progress_file': 'progress.json',
                      'return_trace_file': 'returns.csv'}.items():
        base[key] = str(directory / name)
    template = directory / 'template.json'
    save_status(template, base)
    # The legacy merge coerces max_rows before applying curated full-population mode.
    flat = {key: value for key, value in base.items() if key != 'max_rows'}
    return merged_config(flat, template, directory)


def _enqueue(body, state, task, outbox):
    summary = json.loads(Path(task['summary']).read_text())
    measured = metrics_of(summary, task['wall_seconds'], {})
    measured.update(compute_metrics(summary))
    shared = governed_run._governed_exec()
    directory = Path(task['summary']).parent
    save_status(directory / 'metrics.json', measured)
    spec = {'metrics': {'kind': 'json_numbers', 'path': 'metrics.json',
                        'keys': governed_run.METRIC_KEYS},
            'artifacts': {key: key for key in task['artifacts']}}
    metrics = shared.collect_metrics(spec, directory)
    for metric in metrics:
        metric['split'] = task['split']
    terminal = {'schema': 'governed_terminal.v1', 'generation': task['attempt'],
                'status': 'COMPLETED', 'reason': None,
                'started_at': task['started_at'], 'finished_at': task['finished_at'],
                'costs': {'wall_seconds': task['wall_seconds']},
                'deliveries': body['publication']['deliveries'][task['split']],
                'artifacts': shared.collect_artifacts(spec, directory, task),
                'metrics': metrics,
                'tags': {'classification': body['classification'], 'arm': task['arm'],
                         'split': task['split'], 'seed': str(task['seed']),
                         'contract_sha256': state['contract_sha256'],
                         'curated_manifest_sha256': body['splits'][task['split']][
                             'curated_input_manifest_sha256']}}
    outbox.put({'campaign_sha256': body['publication']['campaign_sha256'],
                'unit_id': body['publication']['unit_ids'][
                    f'{task["seed"]}/{task["arm"]}/{task["split"]}'], 'terminal': terminal})


def run(contract_path, output_root, *, resume=False, timeout=120, outbox=None, executor=None):
    contract_path, output_root = Path(contract_path).resolve(), Path(output_root).resolve()
    live_roots = (Path.home() / '.local/share/agent-multi',
                  Path.home() / '.local/state/agent-multi')
    if (not _positive(timeout) or output_root.is_relative_to(ROOT)
            or any(output_root.is_relative_to(p) for p in live_roots)):
        raise ValueError('QUEUE_PRIVATE_OUTPUT_AND_POSITIVE_TIMEOUT_REQUIRED')
    pin = digest(contract_path)
    if resume:
        previous = status(output_root)
        if previous['state'] == 'UNAVAILABLE':
            raise ValueError('QUEUE_NO_RESUME_STATE')
        if previous['contract_sha256'] != pin:
            raise ValueError('QUEUE_CONTRACT_DRIFT')
    body = load_contract(contract_path)  # validate ALL support before env/model/outputs
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / '.queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('QUEUE_ALREADY_RUNNING') from exc
        journal = output_root / 'STATUS.json'
        if resume:
            state = status(output_root)
            if state['runner_sha256'] != digest(Path(__file__)):
                raise ValueError('QUEUE_RUNNER_SOURCE_DRIFT')
            _verify_artifacts(state)
        else:
            if journal.exists():
                raise ValueError('QUEUE_EXISTS_USE_RESUME')
            state = {'schema': 'curated_rl_queue_status.v1', 'state': 'READY',
                     'contract_sha256': pin, 'classification': body['classification'],
                     'runner_sha256': digest(Path(__file__)),
                     'tasks': [{'id': f'{pin[:16]}-{seed}-{arm}-{split}', 'arm': arm,
                                'seed': seed, 'split': split, 'state': 'READY', 'attempt': 0}
                               for seed in body['seeds'] for arm in ARMS
                               for split in ('train', 'validation')]}
        save_status(journal, state)
        if outbox is None:
            try:
                outbox = open_outbox(output_root)
            except (OSError, RuntimeError, ImportError, SystemExit) as exc:
                state.update(state='PUBLICATION_UNAVAILABLE', error_type=type(exc).__name__)
                save_status(journal, state)
                return 2
        # Keep the claim alive in the child even if the queue parent is killed.
        if executor is None:
            executor = lambda cfg, directory, limit: _subprocess(
                cfg, directory, limit, lock_fd=lock.fileno())
        for task in state['tasks']:
            _verify_artifacts(state)
            if task['state'] == 'ENQUEUED':
                continue
            if task['state'] != 'PUBLICATION_PENDING':
                task.pop('returncode', None)
                task['attempt'] += 1
                directory = output_root / task['id'] / f'attempt-{task["attempt"]}'
                directory.mkdir(parents=True, exist_ok=False)
                if task['split'] == 'train':
                    checkpoint = directory / 'policy.zip'
                else:
                    train = next(t for t in state['tasks'] if t['seed'] == task['seed']
                                 and t['arm'] == task['arm'] and t['split'] == 'train')
                    checkpoint = Path(train['checkpoint'])
                cfg = _task_config(body, task, directory, checkpoint)
                task.update(config=str(directory / 'config.json'), checkpoint=str(checkpoint),
                            summary=cfg['results_file'], state='RUNNING', started_at=_now())
                save_status(Path(task['config']), cfg)
                state['state'] = 'RUNNING'
                save_status(journal, state)
                started = time.monotonic()
                try:
                    code = executor(cfg, directory, timeout)
                    task['returncode'] = code
                    if code != 0:
                        raise RuntimeError(f'QUEUE_CHILD_EXIT_{code}')
                    if not Path(task['summary']).is_file() or not checkpoint.is_file():
                        raise RuntimeError('QUEUE_MISSING_RUN_ARTIFACT')
                    summary = json.loads(Path(task['summary']).read_text())
                    paths = {key: task[key] for key in ('config', 'summary', 'checkpoint')}
                    paths['effective_config'] = cfg['save_config']
                    for key in ('return_trace_file', 'return_trace_metadata_file'):
                        if summary.get(key):
                            paths[key] = summary[key]
                    artifacts = {key: digest(path) for key, path in paths.items()}
                    task.update(paths, artifacts=artifacts)
                    if 'checkpoint_policy_tensor_sha256' in summary:
                        task['policy_tensor_sha256'] = summary['checkpoint_policy_tensor_sha256']
                    task['state'] = 'PUBLICATION_PENDING'
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    task.update(state='FAILED_TIMEOUT' if isinstance(exc, subprocess.TimeoutExpired)
                                else 'FAILED_EXECUTION', error_type=type(exc).__name__)
                task.update(wall_seconds=time.monotonic() - started, finished_at=_now())
                save_status(directory / 'EXECUTION.json', task)
                save_status(journal, state)
                if task['state'].startswith('FAILED'):
                    state['state'] = task['state']
                    save_status(journal, state)
                    return 2
            try:
                _enqueue(body, state, task, outbox)
            except (OSError, ValueError, RuntimeError) as exc:
                task['error_type'] = type(exc).__name__
                state['state'] = 'PUBLICATION_PENDING'
                save_status(journal, state)
                return 2
            task['state'] = 'ENQUEUED'
            task.pop('error_type', None)
            save_status(journal, state)
        state['state'] = 'COMPLETED_OUTBOX_PENDING'
        state.pop('error_type', None)
        save_status(journal, state)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'resume', 'status', '_task'))
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args(argv)
    if args.action == '_task':
        if args.config is None:
            parser.error('--config is required')
        os.environ.update(CPU_ENV)
        execute_task(json.loads(args.config.read_text()))
        return 0
    if args.output_root is None:
        parser.error('--output-root is required')
    if args.action == 'status':
        current = status(args.output_root)
        print(json.dumps(current, indent=2))
        return 2 if current['state'] == 'UNAVAILABLE' else 0
    if args.contract is None:
        parser.error('--contract is required')
    return run(args.contract, args.output_root, resume=args.action == 'resume', timeout=args.timeout)


if __name__ == '__main__':
    raise SystemExit(main())
