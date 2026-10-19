"""Finite, sealed queue over the existing replay and registered RL pipeline.

No admission issuance, promotion, TEST or network calls inside training steps.
Invoke as `python -m tools.curated_rl_campaign run|resume|status ...`.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import re

from app.curated_rl_input import validate_curated_rl_input
from tools import governed_run
from tools.governed_offline_replay import compute_metrics, merged_config, metrics_of, TRADING_METRIC_KEYS
from tools.verify_rl_static_engineering import save_status
from tools import rl_warehouse

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
DEVELOPMENT_ARMS = ['NO_TRADE', 'PASSIVE_LONG', 'PASSIVE_SHORT', 'RANDOM',
                    'SAC_FLAT', 'SAC_PER_FEATURE_CONV1D']
METRIC_UNITS = {
    **dict.fromkeys(('initial_cash', 'final_equity', 'net_profit', 'max_drawdown_money',
                     'trade_costs_total', 'avg_trade_pnl'), 'USD'),
    **dict.fromkeys(('max_drawdown_pct', 'win_rate_pct'), 'percent'),
    'total_return': 'fraction', 'sharpe_ratio': 'ratio', 'profit_factor': 'ratio',
    'episode_reward': 'reward', 'final_reward': 'reward', 'wall_seconds': 'seconds',
    'mean_daily_log_return': 'log_fraction_per_observed_day',
    'monitor_score': 'log_fraction_per_observed_day',
    'monitor_generalization_gap': 'log_fraction_per_observed_day',
    'monitor_step': 'transitions', 'observed_days': 'observed_days',
    'monitor_patience_used': 'evaluations', 'monitor_best_step': 'transitions',
    **dict.fromkeys(('trades_total', 'trades_closed', 'trades_won', 'trades_lost',
                     'entry_submissions', 'filled_entries', 'max_entries_filled_in_day',
                     'daily_entry_limit', 'observed_updates', 'no_trade_min_trades'), 'count'),
    **dict.fromkeys(('episode_length', 'observed_timesteps', 'requested_timesteps'), 'transitions'),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _positive(value):
    return type(value) is int and value > 0


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def arm_order(body):
    return body.get('arms', DEVELOPMENT_ARMS if body.get('scientific_scope') == 'REAL_DATA_DEVELOPMENT'
                    else list(ARMS))


def execution_settings(body):
    real = body.get('scientific_scope', 'SYNTHETIC_ENGINEERING') == 'REAL_DATA_DEVELOPMENT'
    execution = copy.deepcopy(body.get('execution', {'device': 'cpu'}))
    if body.get('execution_backend', 'local_subprocess') != 'local_subprocess':
        raise ValueError('QUEUE_UNSUPPORTED_EXECUTION_BACKEND')
    if execution.get('device') not in ('cpu', 'cuda'):
        raise ValueError('QUEUE_EXPLICIT_DEVICE_REQUIRED')
    if real and (body.get('execution_backend') != 'local_subprocess'
                 or not execution.get('gym_root') or not execution.get('gym_code_commit')):
        raise ValueError('QUEUE_PINNED_REAL_RUNTIME_REQUIRED')
    if real or 'task_timeout_seconds' in execution:
        times = execution.get('task_timeout_seconds', {})
        if set(times) != {'train', 'validation'} or any(not _positive(v) or v > 86400 for v in times.values()):
            raise ValueError('QUEUE_SEALED_TASK_TIMEOUT_REQUIRED')
    if 'max_attempts_per_task' in execution or real:
        if type(execution.get('max_attempts_per_task')) is not int or not 1 <= execution['max_attempts_per_task'] <= 2:
            raise ValueError('QUEUE_BOUNDED_ATTEMPTS_REQUIRED')
    if body['common_config'].get('device', execution['device']) != execution['device']:
        raise ValueError('QUEUE_CONFLICTING_DEVICE_DECLARATIONS')
    if execution['device'] == 'cuda':
        validate_gpu_evidence(execution, freshness=False)
    return execution


def validate_gpu_evidence(execution, *, freshness=True):
    from tools.gpu_readiness_probe import launch_gate
    try:
        proof = execution['gpu_evidence']
        path, uuid = Path(proof['path']), proof['assigned_gpu_uuid']
        if (not path.is_absolute() or not _pin(proof['sha256']) or digest(path) != proof['sha256']
                or not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uuid)
                or not _positive(proof['max_age_seconds']) or proof['max_age_seconds'] > 86400):
            raise ValueError('QUEUE_GPU_EVIDENCE_IDENTITY_INVALID')
        hb = json.loads(path.read_text())
        stamp = datetime.fromisoformat(hb['generated_utc'].replace('Z', '+00:00'))
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
        build = hb['framework']['per_uuid'][uuid]
        if (hb['schema'] != 'agent_multi.gpu_readiness_heartbeat.v1'
                or hb['classification'] != 'GPU_READY' or not hb['gpus']['exact_match']
                or uuid not in hb['gpus']['expected_uuids'] or uuid not in hb['gpus']['observed_uuids']
                or hb['driver']['nvidia_smi_ok'] is not True
                or hb['running_kernel']['modinfo_nvidia']['ok'] is not True
                or hb['framework']['status'] != 'TORCH_CUDA_OK'
                or build['status'] != 'TORCH_CUDA_OK' or build['cuda_available'] is not True
                or not build['torch_present'] or build['cuda_device_count'] < 1
                or build['cuda_visible_devices'] != uuid
                or not build['torch_version'] or not build['torch_cuda_version']
                or (freshness and not 0 <= age <= proof['max_age_seconds'])):
            raise ValueError('QUEUE_GPU_EVIDENCE_REFUSED')
        if launch_gate(hb, uuid)[1] != 0:
            raise ValueError('QUEUE_GPU_GATE_REFUSED')
        return hb
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError('QUEUE_GPU_EVIDENCE_INVALID') from exc


def child_environment(config):
    env = {**os.environ, **CPU_ENV}
    if config.get('device', 'cpu') == 'cuda':
        execution = config['queue_execution']
        validate_gpu_evidence(execution)
        uuid = execution['gpu_evidence']['assigned_gpu_uuid']
        if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
            raise ValueError('QUEUE_GPU_AMBIENT_ASSIGNMENT_MISMATCH')
        env['CUDA_VISIBLE_DEVICES'] = uuid
    return env


def runtime_device(execution, *, torch_module=None):
    if execution['device'] == 'cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('QUEUE_CPU_ISOLATION_REQUIRED')
        return {'device': 'cpu'}
    hb = validate_gpu_evidence(execution)
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise ValueError('QUEUE_GPU_RUNTIME_UNAVAILABLE_NO_CPU_FALLBACK')
    uuid = execution['gpu_evidence']['assigned_gpu_uuid']
    observed = str(torch_module.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower()
    build = hb['framework']['per_uuid'][uuid]
    if (observed != uuid.removeprefix('GPU-').lower()
            or str(torch_module.__version__) != build['torch_version']
            or str(torch_module.version.cuda) != build['torch_cuda_version']):
        raise ValueError('QUEUE_GPU_RUNTIME_IDENTITY_MISMATCH')
    return {'device': 'cuda', 'gpu_evidence_sha256': execution['gpu_evidence']['sha256']}


def verify_execution_runtime(execution, config=None):
    if not execution.get('gym_root'):
        return {}
    root = Path(execution['gym_root']).resolve()
    prefixes = ('app', 'gym_fx', 'broker_plugins', 'data_feed_plugins', 'strategy_plugins',
                'preprocessor_plugins', 'reward_plugins', 'metrics_plugins')
    pins = rl_warehouse.pin_sources(root, execution['gym_code_commit'], prefixes)
    import app
    from app import main as agent_main
    if not Path(agent_main.__file__).resolve().is_relative_to(ROOT):
        raise ValueError('QUEUE_AGENT_RUNTIME_IMPORT_MISMATCH')
    if str(root / 'app') not in app.__path__:
        app.__path__.insert(0, str(root / 'app'))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    modules = ['gym_fx', 'gym_fx.env', 'app.env', 'app.bt_bridge']
    if config is not None:
        from importlib.metadata import entry_points
        for group, key, default in (
            ('broker.plugins', 'broker_plugin', 'default_broker'),
            ('data_feed.plugins', 'data_feed_plugin', 'default_data_feed'),
            ('strategy.plugins', 'strategy_plugin', 'shared_execution_envelope'),
            ('preprocessor.plugins', 'preprocessor_plugin', 'feature_window_preprocessor'),
            ('reward.plugins', 'reward_plugin', 'pnl_reward'),
            ('metrics.plugins', 'metrics_plugin', 'default_metrics')):
            entry = next(e for e in entry_points(group=group) if e.name == config.get(key, default))
            modules.append(entry.load().__module__)
    for name in modules:
        module = importlib.import_module(name)
        path = Path(inspect.getfile(module)).resolve()
        if not path.is_relative_to(root) or str(path.relative_to(root)) not in pins:
            raise ValueError('QUEUE_GYM_RUNTIME_IMPORT_MISMATCH')
    return {'gym_code_commit': execution['gym_code_commit']}


def prepare_warehouse(publication, output_root):
    rl_warehouse.prepare(publication, output_root)


def load_contract(path):
    body = json.loads(Path(path).read_text())
    if body.get('schema') != 'curated_rl_campaign.v1' or body.get('classification') != 'NON_GOVERNING':
        raise ValueError('QUEUE_ENGINEERING_CONTRACT_REQUIRED')
    if body.get('scientific_scope', 'SYNTHETIC_ENGINEERING') not in ('SYNTHETIC_ENGINEERING', 'REAL_DATA_DEVELOPMENT'):
        raise ValueError('QUEUE_UNSUPPORTED_SCIENTIFIC_SCOPE')
    execution_settings(body)
    paired = body['common_config'].get('paired_early_stopping')
    if paired:
        from agent_plugins.paired_early_stopping import validate_settings
        validate_settings(paired)
        if body.get('arms') != ['SAC_FLAT']:
            raise ValueError('QUEUE_PAIRED_ES_DIRECT_ONLY')
    elif len(arm_order(body)) != len(ARMS) or set(arm_order(body)) != set(ARMS):
        raise ValueError('QUEUE_EXACT_ORDERED_ARM_SET_REQUIRED')
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
    warehouse = publication.get('mode') == 'WAREHOUSE_REPORTS'
    if publication.get('mode', 'FLOW_V3') not in ('FLOW_V3', 'WAREHOUSE_REPORTS'):
        raise ValueError('QUEUE_PUBLICATION_MODE_INVALID')
    if body.get('scientific_scope') == 'REAL_DATA_DEVELOPMENT' and not warehouse:
        raise ValueError('QUEUE_REAL_DEVELOPMENT_REQUIRES_WAREHOUSE_REPORTS')
    units = publication.get('unit_ids', {})
    expected_units = {f'{seed}/{arm}/{role}' for seed in seeds for arm in arm_order(body)
                      for role in ('train', 'validation')}
    if not warehouse and (not _pin(publication.get('campaign_sha256'))
            or set(publication.get('deliveries', {})) != set(splits)
            or any(not isinstance(v, list) for v in publication['deliveries'].values())
            or set(units) != expected_units
            or any(not isinstance(v, str) or not v.strip() for v in units.values())
            or len(set(units.values())) != len(units)):
        raise ValueError('QUEUE_EXISTING_PUBLICATION_IDENTITY_REQUIRED')
    if warehouse:
        if (not publication.get('predictor_root') or not Path(publication['predictor_root']).is_absolute()
                or not re.fullmatch('[0-9a-f]{40}', publication.get('predictor_code_commit', ''))
                or not all(isinstance(publication.get(k), str) and publication[k] for k in ('actor', 'lake'))
                or set(publication.get('resources', {})) != {'train', 'validation'}):
            raise ValueError('QUEUE_WAREHOUSE_CONFIG_REQUIRED')
        if units and (set(units) != expected_units or len(set(units.values())) != len(units)):
            raise ValueError('QUEUE_LOCAL_UNIT_IDS_INVALID')
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

    execution = config.get('queue_execution', {'device': config.get('device', 'cpu')})
    if config.get('device') != execution['device']:
        raise ValueError('QUEUE_DEVICE_CONFIG_MISMATCH')
    device_identity = runtime_device(execution)
    runtime_identity = verify_execution_runtime(execution, config)
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
        model = original_build(wrapped, cfg)
        check_model_device(model)
        return model

    def check_model_device(model):
        if hasattr(model, 'policy'):
            actual = getattr(getattr(model, 'device', None), 'type', None)
            if actual != execution['device'] or any(p.device.type != actual for p in model.policy.parameters()):
                raise ValueError('QUEUE_MODEL_DEVICE_MISMATCH_NO_FALLBACK')

    agent.build = build
    if mode == 'inference':
        original_load = agent.load

        def load(path, wrapped):
            wrapped.action_space.seed(config['eval_seed'])
            model = original_load(path, wrapped)
            check_model_device(model)
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
    summary['execution_device'] = device_identity
    summary['runtime_identity'] = runtime_identity
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
            cwd=ROOT, env=child_environment(config), stdout=log,
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
    execution = execution_settings(body)
    if agent != 'sac_agent':
        execution['device'] = 'cpu'
    base.update(body['splits'][task['split']])
    base.update(agent_plugin=agent, observation_encoder=encoder, queue_arm=task['arm'],
                mode='train' if task['split'] == 'train' else 'inference',
                require_curated_rl_input=True, max_rows=None, device=execution['device'],
                queue_execution=execution,
                scientific_scope=body.get('scientific_scope', 'SYNTHETIC_ENGINEERING'),
                execution_backend=body.get('execution_backend', 'local_subprocess'),
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
                      'return_trace_file': 'returns.csv',
                      'session_flatten_custody_root': 'session_custody'}.items():
        base[key] = str(directory / name)
    if base.get('paired_early_stopping') and task['split'] == 'train':
        campaign_root = directory.parents[1]
        base.update(paired_campaign_root=str(campaign_root),
                    paired_campaign_sha256=digest(campaign_root / 'CONTRACT.json'),
                    paired_source_unit=f"{task['id']}:a{task['attempt']}")
    template = directory / 'template.json'
    save_status(template, base)
    # The legacy merge coerces max_rows before applying curated full-population mode.
    flat = {key: value for key, value in base.items() if key != 'max_rows'}
    return merged_config(flat, template, directory)


def warehouse_report(body, state, task, summary):
    publication = body['publication']
    measured = metrics_of(summary, task['wall_seconds'], {})
    measured.update(compute_metrics(summary))
    for key, value in summary.items():
        if type(value) in (int, float) and math.isfinite(value):
            measured[key] = value
        elif value is None and key in TRADING_METRIC_KEYS:
            measured[key] = None
    def diagnostics(prefix, values):
        for key, value in values.items():
            name = prefix + '.' + key
            if isinstance(value, dict):
                diagnostics(name, value)
            elif type(value) in (int, float) and math.isfinite(value):
                measured[name] = value
    for field in ('execution_diagnostics', 'execution_counters'):
        if isinstance(summary.get(field), dict):
            diagnostics(field, summary[field])
    if isinstance(summary.get('paired_early_stopping'), dict):
        diagnostics('paired_early_stopping', summary['paired_early_stopping'])
    split = body['splits'][task['split']]
    if digest(split['curated_input_manifest']) != split['curated_input_manifest_sha256']:
        raise ValueError('QUEUE_REPORT_INPUT_IDENTITY_DRIFT')
    manifest = json.loads(Path(split['curated_input_manifest']).read_text())
    tags = {'classification': body['classification'],
            'scientific_scope': body.get('scientific_scope', 'SYNTHETIC_ENGINEERING'),
            'publication_protocol': 'WAREHOUSE_REPORTS_NOT_FLOW_V3',
            'local_campaign_sha256': state['contract_sha256'], 'arm': task['arm'],
            'seed': str(task['seed']), 'split': task['split'],
            'curated_manifest_sha256': split['curated_input_manifest_sha256']}
    for key in ('input_policy', 'cost_scope'):
        if key in body:
            tags[key] = body[key] if isinstance(body[key], str) else json.dumps(body[key], sort_keys=True)
    if body['common_config'].get('paired_early_stopping'):
        tags['checkpoint_selection_protocol'] = json.dumps(body['common_config']['paired_early_stopping'], sort_keys=True)
        tags['validation_role'] = 'DEVELOPMENT_CHECKPOINT_TUNING_NOT_HOLDOUT'
    raw_label = str(body.get('input_policy', body.get('scope', ''))).upper()
    if 'RAW' in raw_label or 'CORE_PRICES' in raw_label or body.get('feature_selection_complete') is False:
        tags['feature_selection_complete'] = 'false'
    for key, label in (('config', 'config_sha256'), ('summary', 'summary_sha256'),
                       ('checkpoint', 'checkpoint_archive_sha256'), ('effective_config', 'effective_config_sha256')):
        tags[label] = task['artifacts'][key]
    if task.get('policy_tensor_sha256'):
        tags['policy_tensor_sha256'] = task['policy_tensor_sha256']
    static = summary.get('static_policy', {})
    for key in ('before', 'after'):
        if static.get(key):
            tags['static_policy_' + key] = static[key]
    unit_key = f'{task["seed"]}/{task["arm"]}/{task["split"]}'
    local_id = publication.get('unit_ids', {}).get(unit_key, task['id'])
    return {'experiment_key': local_id, 'experiment_set_key': publication.get('experiment_set_key'),
            'actor': publication['actor'], 'lake': publication['lake'], 'lineage': 'UNVERIFIED',
            'project': 'agent-multi', 'phase': body.get('scientific_scope', 'SYNTHETIC_ENGINEERING'),
            'config_sha256': task['artifacts']['effective_config'], 'code_commit': publication.get('code_commit'),
            'tags': tags, 'datasets': [{'lake': publication['lake'],
                'resource': publication['resources'][task['split']], 'sha256': manifest['input_sha256'],
                'role': task['split'], 'lineage': 'UNVERIFIED',
                'reason': 'Local admitted content pin; no Flow-v3 delivery lineage claim'}],
            'metrics': [{'metric': key, 'value': value, 'split': task['split'], 'horizon': None,
                         'unit': metric_unit(key)}
                        for key, value in sorted(measured.items())]}


def metric_unit(key):
    if key in METRIC_UNITS:
        return METRIC_UNITS[key]
    if key.startswith('compute_'):
        if 'transition' in key and not key.endswith(('respected', 'reset')):
            return 'transitions'
        if key in ('compute_cap_respected', 'compute_evaluation_cap_respected',
                   'compute_partial_rollout', 'compute_timestep_counter_was_reset'):
            return 'flag'
        return 'count'
    if key.startswith(('execution_diagnostics.', 'execution_counters.')):
        return 'seconds' if key.endswith('_seconds') else 'count'
    if key.startswith('action_'):
        return 'fraction' if key.endswith(('fraction', 'pct')) else None
    return None


def pump_warehouse(body, output_root):
    publication, output_root = body['publication'], Path(output_root)
    if not publication.get('warehouse_url') or not publication.get('token_env'):
        raise ValueError('QUEUE_WAREHOUSE_PUMP_CONFIG_REQUIRED')
    request = output_root / 'warehouse_pump_request.json'
    save_status(request, {k: publication[k] for k in ('warehouse_url', 'token_env')})
    try:
        result = rl_warehouse.invoke(publication, output_root, 'pump', request)
        save_status(output_root / 'WAREHOUSE_STATUS.json', result)
        return 0 if result['warehouse']['pending'] == result['warehouse']['mismatch'] == 0 else 2
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        save_status(output_root / 'WAREHOUSE_STATUS.json', {'state': 'PUMP_RETRY_PENDING',
                                                         'error_type': type(exc).__name__})
        return 2


def _enqueue(body, state, task, outbox, *, warehouse_root=None):
    summary = json.loads(Path(task['summary']).read_text())
    if body['publication'].get('mode') == 'WAREHOUSE_REPORTS':
        directory = Path(task['summary']).parent
        report = warehouse_report(body, state, task, summary)
        report_path = directory / 'warehouse_report.json'
        save_status(report_path, report)
        request = directory / 'warehouse_request.json'
        save_status(request, {'report': report, 'source_unit': task['id']})
        root = Path(warehouse_root) if warehouse_root is not None else directory.parents[1]
        result = rl_warehouse.invoke(body['publication'], root, 'enqueue', request)
        task['warehouse_report_sha256'] = result['report_sha256']
        task['warehouse_report'] = str(report_path)
        task['artifacts']['warehouse_report'] = digest(report_path)
        return
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


def run(contract_path, output_root, *, resume=False, timeout=None, outbox=None, executor=None):
    contract_path, output_root = Path(contract_path).resolve(), Path(output_root).resolve()
    live_roots = (Path.home() / '.local/share/agent-multi',
                  Path.home() / '.local/state/agent-multi')
    if ((timeout is not None and not _positive(timeout)) or output_root.is_relative_to(ROOT)
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
    execution = execution_settings(body)
    real = body.get('scientific_scope') == 'REAL_DATA_DEVELOPMENT'
    if real and timeout is not None:
        raise ValueError('QUEUE_REAL_TIMEOUT_MUST_BE_SEALED_IN_CONTRACT')
    warehouse = body['publication'].get('mode') == 'WAREHOUSE_REPORTS'
    if real:
        verify_execution_runtime(execution)
    if warehouse:
        prepare_warehouse(body['publication'], output_root)
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
                     'scientific_scope': body.get('scientific_scope', 'SYNTHETIC_ENGINEERING'),
                     'runner_sha256': digest(Path(__file__)),
                     'tasks': [{'id': f'{pin[:16]}-{seed}-{arm}-{split}', 'arm': arm,
                                'seed': seed, 'split': split, 'state': 'READY', 'attempt': 0}
                               for seed in body['seeds'] for arm in arm_order(body)
                               for split in ('train', 'validation')]}
        save_status(journal, state)
        if outbox is None and not warehouse:
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
                if task['attempt'] >= execution.get('max_attempts_per_task', 2**31):
                    state['state'] = task['state'] = 'ATTEMPTS_EXHAUSTED'
                    save_status(journal, state)
                    return 2
                task.pop('returncode', None)
                task['attempt'] += 1
                save_status(journal, state)
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
                    limit = execution.get('task_timeout_seconds', {}).get(task['split'], timeout or 120)
                    task['timeout_seconds'] = limit
                    task['device'] = cfg['device']
                    code = executor(cfg, directory, limit)
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
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                task['error_type'] = type(exc).__name__
                state['state'] = 'PUBLICATION_PENDING'
                save_status(journal, state)
                return 2
            task['state'] = 'ENQUEUED'
            task.pop('error_type', None)
            save_status(journal, state)
            if warehouse and body['publication'].get('pump_between_stages', False):
                pump_warehouse(body, output_root)
        state['state'] = 'COMPLETED_OUTBOX_PENDING'
        state.pop('error_type', None)
        save_status(journal, state)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'resume', 'status', 'pump', '_task'))
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--timeout', type=int)
    args = parser.parse_args(argv)
    if args.action == '_task':
        if args.config is None:
            parser.error('--config is required')
        config = json.loads(args.config.read_text())
        os.environ.update(child_environment(config))
        execute_task(config)
        return 0
    if args.output_root is None:
        parser.error('--output-root is required')
    if args.action == 'status':
        current = status(args.output_root)
        print(json.dumps(current, indent=2))
        return 2 if current['state'] == 'UNAVAILABLE' else 0
    if args.contract is None:
        parser.error('--contract is required')
    if args.action == 'pump':
        body = json.loads(args.contract.read_text())
        if (body['publication'].get('mode') != 'WAREHOUSE_REPORTS'
                or status(args.output_root).get('contract_sha256') != digest(args.contract)):
            raise ValueError('QUEUE_PUMP_CONTRACT_MISMATCH')
        return pump_warehouse(body, args.output_root)
    return run(args.contract, args.output_root, resume=args.action == 'resume', timeout=args.timeout)


if __name__ == '__main__':
    raise SystemExit(main())
