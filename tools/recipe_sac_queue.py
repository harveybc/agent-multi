"""Finite owner-authorized native SAC queue on the frozen shared market recipe."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from app.frozen_recipe_inputs import file_sha, require
from tools.verify_rl_static_engineering import save_status

ROOT = Path(__file__).resolve().parents[1]
DESIGN = dict(seed=7, total_timesteps=100000, checkpoint_interval=25000,
              learning_rate=.0003, buffer_size=20000, batch_size=64,
              learning_starts=1000, gamma=.99, tau=.005,
              early_stopping=False, refit=False, expert=False, agent_feedback=False)
CRITIC_DROPOUT = {'CRITIC_BASE': .3, 'CRITIC_NODROP': 0.}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_status(path, value)


def source_pins(root, directories):
    return {str(p.relative_to(root)): file_sha(p) for directory in directories
            for p in sorted((root / directory).rglob('*.py'))}


def software():
    return {name: importlib.metadata.version(name) for name in
            ('torch','stable-baselines3','gymnasium','numpy','pandas','pyarrow')}


def prepare(a):
    if getattr(a, 'critic_parent', None):
        return prepare_critic(a)
    root = a.root.resolve()
    require(not root.exists(), 'NEW_CAMPAIGN_REQUIRED')
    engine = a.engine_root.resolve()
    spec = dict(schema='recipe_sac_queue.v1', recipe_sha256=file_sha(a.recipe),
                engine_root=str(engine), agent_root=str(ROOT), design=DESIGN,
                arms=['GROUPED', 'FLAT'], device='cuda', max_attempts=2,
                software=software(),
                task_timeout_seconds=43200,
                inputs={key: str(getattr(a, key).resolve()) for key in
                        ('recipe', 'store', 'manifest', 'bars', 'closures')},
                code={'agent': source_pins(ROOT, ['app', 'env_plugins', 'agent_plugins', 'tools']),
                      'engine': source_pins(engine, ['app/short_horizon'])})
    if getattr(a, 'order_parent', None):
        from env_plugins.order_contract import CONTRACTS
        parent = a.order_parent.resolve()
        require(a.parent_spec_sha and file_sha(parent / 'SPEC.private.json') == a.parent_spec_sha,
                'PARENT_SPEC')
        parent_state = read(parent / 'CLOSE.json')
        require(parent_state['state'] == 'COMPLETE', 'PARENT_NOT_COMPLETE')
        result_path = parent / parent_state['completed']['GROUPED']
        authenticated_result(result_path, a.parent_spec_sha)
        spec.update(schema='recipe_sac_order_queue.v1', arms=['TARGET','COMMANDS5'],
                    order_contracts=CONTRACTS, order_parent=dict(root=str(parent),
                        spec_sha256=a.parent_spec_sha, result=str(result_path.relative_to(parent)),
                        result_sha256=file_sha(result_path)))
    root.mkdir(parents=True)
    write(root / 'SPEC.private.json', spec)
    write(root / 'IDENTITY.json', {'spec_sha256': file_sha(root / 'SPEC.private.json'),
                                  'created_at': now(), 'state': 'PREPARED'})
    print(json.dumps({'state': 'PREPARED', 'spec_sha256': file_sha(root / 'SPEC.private.json')}))


def prepare_critic(a):
    parent, root = a.critic_parent.resolve(), a.root.resolve()
    require(not root.exists(), 'NEW_CAMPAIGN_REQUIRED')
    require(file_sha(parent/'SPEC.private.json') == a.parent_spec_sha, 'PARENT_SPEC')
    p = read(parent/'SPEC.private.json')
    closed = read(parent/'CLOSE.json')
    require(p['schema'] == 'recipe_sac_order_queue.v1' and closed['state'] == 'COMPLETE'
            and closed['spec_sha256'] == a.parent_spec_sha, 'CRITIC_PARENT')
    result = parent/closed['completed']['COMMANDS5']
    authenticated_result(result, a.parent_spec_sha)
    require(p['software'] == software() and p['design'] == DESIGN, 'CRITIC_PARENT_DESIGN')
    reg = read(p['inputs']['recipe'])['regularization']
    require(reg['dropout'] == .3 and reg['wd'] == .01,
            'CRITIC_PARENT_REGULARIZATION')
    spec = dict(schema='recipe_sac_critic_queue.v1', recipe_sha256=p['recipe_sha256'],
        engine_root=p['engine_root'], agent_root=str(ROOT), design=DESIGN,
        arms=list(CRITIC_DROPOUT), critic_dropout=CRITIC_DROPOUT,
        device='cuda', max_attempts=2, software=software(), task_timeout_seconds=43200,
        inputs=p['inputs'], order_contracts={arm:p['order_contracts']['COMMANDS5'] for arm in CRITIC_DROPOUT},
        critic_parent=dict(root=str(parent), spec_sha256=a.parent_spec_sha,
                           result=str(result.relative_to(parent)), result_sha256=file_sha(result)),
        code={'agent':source_pins(ROOT, ['app','env_plugins','agent_plugins','tools']),
              'engine':p['code']['engine']})
    root.mkdir(parents=True)
    write(root/'SPEC.private.json', spec)
    write(root/'IDENTITY.json', dict(spec_sha256=file_sha(root/'SPEC.private.json'), created_at=now(), state='PREPARED'))
    print(json.dumps({'state':'PREPARED', 'spec_sha256':file_sha(root/'SPEC.private.json')}))


def verify_critic_parent(spec):
    parent = Path(spec['critic_parent']['root'])
    pin = spec['critic_parent']
    require(file_sha(parent/'SPEC.private.json') == pin['spec_sha256'], 'CRITIC_PARENT_SPEC')
    p = read(parent/'SPEC.private.json')
    for key in ('recipe_sha256','inputs','design','software','engine_root'):
        require(spec[key] == p[key], 'CRITIC_PARENT_CHANGED_'+key)
    require(spec['code']['engine'] == p['code']['engine'], 'CRITIC_ENGINE_CHANGED')
    result = (parent/pin['result']).resolve()
    require(result.is_relative_to(parent.resolve()) and file_sha(result) == pin['result_sha256'], 'CRITIC_PARENT_RESULT')
    authenticated_result(result, pin['spec_sha256'])
    return result


def verify(root):
    spec = read(root / 'SPEC.private.json')
    require(file_sha(root / 'SPEC.private.json') == read(root / 'IDENTITY.json')['spec_sha256'], 'SPEC_PIN')
    from env_plugins.order_contract import CONTRACTS
    standard = spec['schema'] == 'recipe_sac_queue.v1' and spec['arms'] == ['GROUPED','FLAT']
    orders = (spec['schema'] == 'recipe_sac_order_queue.v1' and spec['arms'] == ['TARGET','COMMANDS5']
              and spec.get('order_contracts') == CONTRACTS and bool(spec.get('order_parent')))
    critics = (spec['schema'] == 'recipe_sac_critic_queue.v1' and spec['arms'] == list(CRITIC_DROPOUT)
               and spec.get('critic_dropout') == CRITIC_DROPOUT
               and spec.get('order_contracts') == {a:CONTRACTS['COMMANDS5'] for a in CRITIC_DROPOUT}
               and bool(spec.get('critic_parent')))
    require((standard or orders or critics) and spec['design'] == DESIGN and spec['max_attempts'] == 2
            and spec['device'] == 'cuda', 'DESIGN')
    require(spec['software'] == software(), 'SOFTWARE_PIN')
    for role, path in [('agent', spec['agent_root']), ('engine', spec['engine_root'])]:
        for name, digest in spec['code'][role].items():
            require(file_sha(Path(path) / name) == digest, 'CODE_PIN')
    if critics:
        verify_critic_parent(spec)
    return spec


def clean(value):
    """Preserve undefined metrics as null with explicit names, never NaN -> zero."""
    import numpy as np
    import pandas as pd
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def tensor_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def preserve_rng():
    import numpy as np
    import torch
    py, ns, ts = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cs = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(ns)
        torch.set_rng_state(ts)
        if cs is not None:
            torch.cuda.set_rng_state_all(cs)


def evaluate(model, market, period, output, diagnostic=False):
    import numpy as np
    import torch
    from app.short_horizon.expert_guided.env import summary
    env = market.env(period)
    py, np_state, th = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training = model.policy.training
    before = tensor_hash(model)
    started = time.monotonic()
    try:
        obs, _ = env.reset()
        selected = []
        if diagnostic:
            from tools.order_policy_diagnosis import sample_indices, policy_probe
            available = np.flatnonzero(env.valid[env.engine.first:env.engine.last]) + env.engine.first
            probe_bars = set(available[sample_indices(len(available))].tolist())
        while True:
            if diagnostic and env.bar in probe_bars:
                selected.append((env.bar, {k:v.copy() for k,v in obs.items()}))
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        native = env.engine.result()
        report = summary(native)
        daily = native['daily_equity'].to_numpy(dtype=float)
        previous = np.r_[env.engine.cfg.capital, daily[:-1]]
        mean_log = (float(np.log(daily / previous).mean())
                    if (daily > 0).all() and (previous > 0).all() else None)
        report.update(period=period, period_limits=market.recipe['split']['periods'][period],
                      coverage=env.coverage, seconds=time.monotonic() - started,
                      initial_capital=env.engine.cfg.capital,
                      total_return_pct=100 * native['net_profit_usd'] / env.engine.cfg.capital,
                      mean_daily_log_return=mean_log,
                      policy_tensor_sha256=before,
                      undefined_metrics=[k for k, v in report.items()
                                         if isinstance(v, float) and not math.isfinite(v)])
        if hasattr(env, 'order_counts'):
            durations = [(t['exit_time']-t['entry_time']).total_seconds()/3600 for t in env.engine.trades]
            report.update(duration_hours_median=float(np.median(durations)) if durations else None,
                          duration_hours_p90=float(np.quantile(durations,.9)) if durations else None,
                          n_long=sum(t['direction']==1 for t in env.engine.trades),
                          n_short=sum(t['direction']==-1 for t in env.engine.trades))
            write(output / (period + '.orders.json'), dict(contract=env.contract,counts=env.order_counts))
        require(tensor_hash(model) == before, 'EVAL_CHANGED_WEIGHTS')
        write(output / (period + '.json'), clean(report))
        write(output / (period + '.trades.json'), clean(env.engine.trades))
        native['daily_equity'].to_csv(output / (period + '.daily_equity.csv'))
        if diagnostic:
            require(len(selected) == len(probe_bars), 'CRITIC_PROBE_SUPPORT')
            observations = {k:np.stack([r[1][k] for r in selected]) for k in selected[0][1]}
            probe = policy_probe(model, observations)
            probe.update(policy_tensor_sha256=before,
                support_sha256=hashlib.sha256(np.asarray([r[0] for r in selected], np.int64).tobytes()).hexdigest(),
                scope='DIAGNOSTIC_NOT_FINANCIAL_SCORE')
            write(output / (period + '.probe.json'), clean(probe))
        return clean(report)
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(th)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        model.policy.set_training_mode(training)
        env.close()


def outbox_report(root, spec, arm, step, report, checkpoint):
    period = report['period']
    key = f'{read(root / "IDENTITY.json")["spec_sha256"][:16]}-{arm}-{step}-{period}'
    metrics = []
    for k, v in report.items():
        if type(v) in (int, float) or v is None:
            unit = ('USD' if k.endswith('_usd') or k == 'initial_capital' else
                    'percent' if k.endswith('_pct') else
                    'fraction' if k in ('win_rate', 'weeks_below_min_share', 'exposure_bar_share') else
                    'seconds' if k == 'seconds' else 'count' if k.startswith('n_') or k == 'trades' else 'ratio')
            metrics.append(dict(metric=k, value=v, split=period.lower(), horizon=None, unit=unit))
    payload = dict(experiment_key=key, experiment_set_key=(
                       'recipe-r3-critic-dropout' if 'critic_dropout' in spec else
                       'recipe-r3-order-contract' if 'order_contracts' in spec else 'recipe-r3-native-sac'), actor='musashi',
                   lake='financial-data', lineage='UNVERIFIED', project='agent-multi', phase='DEVELOPMENT',
                   config_sha256=read(root / 'IDENTITY.json')['spec_sha256'],
                   tags=dict(recipe_sha256=spec['recipe_sha256'], arm=arm, seed='7',
                             checkpoint=str(step), checkpoint_archive_sha256=file_sha(checkpoint),
                             policy_tensor_sha256=report['policy_tensor_sha256'],
                             costs='ASSUMED_NOT_BROKER_VERIFIED',
                             selection='FIXED_FINAL_100000', test='CLOSED',
                             expert='NONE', feedback='NONE', scaling='FROZEN_TRAIN'),
                   datasets=[dict(lake='financial-data',resource='shared-feature-store-r3',
                                  sha256=read(spec['inputs']['recipe'])['store']['sha256'], role=period.lower(),
                                  lineage='UNVERIFIED')], metrics=metrics)
    if 'order_contracts' in spec:
        payload['tags']['order_contract'] = json.dumps(spec['order_contracts'][arm],sort_keys=True)
    if 'critic_dropout' in spec:
        payload['tags'].update(actor_dropout='0.3', critic_dropout=str(spec['critic_dropout'][arm]),
                               weight_decay='0.01', contrast='CRITIC_ONLY_DROPOUT')
    write(root / 'outbox' / (key + '.json'), dict(report=payload, source_unit=key))


def worker(root, arm, attempt):
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from app.recipe_runtime import RecipeMarket, build_sac
    spec = verify(root)
    require(torch.cuda.is_available(), 'CUDA_REQUIRED')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    folder = root / arm / f'attempt-{attempt}'
    folder.mkdir(parents=True, exist_ok=False)
    market = RecipeMarket(spec, spec.get('order_contracts', {}).get(arm))
    env = market.env('TRAIN', DESIGN['gamma'])
    model = build_sac(env, market.recipe, DESIGN, 'GROUPED' if 'order_contracts' in spec else arm, spec['device'],
                      critic_dropout=spec.get('critic_dropout', {}).get(arm))
    if 'critic_dropout' in spec:
        write(folder/'EFFECTIVE_MODEL.json', dict(actor_dropout=model.actor.features_extractor.drop.p,
            critic_dropout=model.critic.features_extractor.drop.p,
            target_critic_dropout=model.critic_target.features_extractor.drop.p,
            weight_decay=model.critic.optimizer.param_groups[0]['weight_decay'],
            initial_policy_tensor_sha256=tensor_hash(model), design=DESIGN,
            order_contract=spec['order_contracts'][arm]))
        if arm == 'CRITIC_NODROP':
            base = root/read(root/'STATUS.json')['completed']['CRITIC_BASE']
            authenticated_result(base, file_sha(root/'SPEC.private.json'))
            require(read(base.parent/'EFFECTIVE_MODEL.json')['initial_policy_tensor_sha256'] == tensor_hash(model),
                    'CRITIC_INITIAL_PARITY')
    start = time.monotonic()
    progress = dict(state='TRAINING', arm=arm, attempt=attempt, step=0, budget=DESIGN['total_timesteps'],
                    recipe=market.recipe['identity'], device=torch.cuda.get_device_name(),
                    spec_sha256=file_sha(root / 'SPEC.private.json'), started_at=now())

    class Heartbeat(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 250 == 0:
                progress.update(step=self.num_timesteps, heartbeat_at=now(), elapsed_seconds=time.monotonic()-start,
                                gradient_updates=self.model._n_updates,
                                gpu_peak_bytes=torch.cuda.max_memory_allocated())
                write(root / 'PROGRESS.json', progress)
                print(json.dumps(progress), flush=True)
            return True

    write(root / 'PROGRESS.json', progress)
    reports = []
    for step in range(DESIGN['checkpoint_interval'], DESIGN['total_timesteps'] + 1, DESIGN['checkpoint_interval']):
        # SB3 continuation preserves replay/optimizer/environment; no reset or refit between checkpoints.
        model.learn(total_timesteps=DESIGN['checkpoint_interval'], reset_num_timesteps=(step == DESIGN['checkpoint_interval']),
                    callback=Heartbeat())
        require(model.num_timesteps == step, 'STEP_COUNT')
        checkpoint_dir = folder / f'step-{step}'
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / 'model.zip'
        model.save(checkpoint)
        with preserve_rng():
            probe_env = market.env('TRAIN')
            obs, _ = probe_env.reset()
            original = model.predict(obs, deterministic=True)[0]
            from stable_baselines3 import SAC
            loaded = SAC.load(checkpoint, device=spec['device'])
            require(np.array_equal(original, loaded.predict(obs, deterministic=True)[0]), 'MODEL_RELOAD')
            del loaded
            probe_env.close()
        checkpoint_reports = {}
        for period in ('TRAIN', 'INNER_A', 'OBS_2024'):
            progress.update(state='EVALUATING', step=step, period=period, heartbeat_at=now())
            write(root / 'PROGRESS.json', progress)
            report = evaluate(model, market, period, checkpoint_dir,
                              diagnostic='critic_dropout' in spec and period in ('TRAIN','INNER_A'))
            checkpoint_reports[period] = report
            outbox_report(root, spec, arm, step, report, checkpoint)
            reports.append(dict(step=step, **report))
        base = checkpoint_reports['TRAIN']['mean_daily_log_return']
        write(checkpoint_dir / 'GAPS.json', {
            'metric': 'TRAIN minus other mean daily log return (same daily units)',
            'gaps': {period: None if base is None or r['mean_daily_log_return'] is None
                     else base - r['mean_daily_log_return'] for period, r in checkpoint_reports.items()}})
        write(checkpoint_dir / 'RECEIPT.json', dict(
            spec_sha256=file_sha(root / 'SPEC.private.json'), policy_tensor_sha256=tensor_hash(model),
            artifacts={p.name: file_sha(p) for p in checkpoint_dir.iterdir() if p.is_file()}))
        progress.update(state='TRAINING', heartbeat_at=now(), elapsed_seconds=time.monotonic()-start,
                        eta_arm_seconds=(time.monotonic()-start) / step * (DESIGN['total_timesteps']-step))
        write(root / 'PROGRESS.json', progress)
    result = dict(state='COMPLETE', arm=arm, attempt=attempt, spec_sha256=file_sha(root / 'SPEC.private.json'),
                  finished_at=now(), elapsed_seconds=time.monotonic()-start, reports=reports,
                  artifacts={str(p.relative_to(folder)): file_sha(p) for p in folder.rglob('*') if p.is_file()})
    if arm == 'TARGET':
        parent = Path(spec['order_parent']['root']) / spec['order_parent']['result']
        original = read(parent.parent/'step-100000'/'TRAIN.json')
        require(tensor_hash(model) == original['policy_tensor_sha256'], 'FRESH_CONTROL_POLICY_REPLAY')
        result['fresh_control_policy_exact'] = True
    if arm == 'CRITIC_BASE':
        parent = verify_critic_parent(spec)
        original = read(parent.parent/'step-100000'/'TRAIN.json')
        require(tensor_hash(model) == original['policy_tensor_sha256'], 'FRESH_CRITIC_CONTROL_REPLAY')
        result['fresh_control_policy_exact'] = True
    write(folder / 'RESULT.json', result)
    progress.update(state='COMPLETE',period=None,heartbeat_at=now(),eta_arm_seconds=0.)
    write(root / 'PROGRESS.json', progress)
    env.close()
    return 0


def authenticated_result(path, spec_sha):
    result = read(path)
    require(result['state'] == 'COMPLETE' and result['spec_sha256'] == spec_sha, 'RESULT_IDENTITY')
    for name, digest in result['artifacts'].items():
        p = (path.parent / name).resolve()
        require(p.is_relative_to(path.parent.resolve()) and file_sha(p) == digest, 'RESULT_ARTIFACT')
    require(len(result['reports']) == 12 and {r['step'] for r in result['reports']} == {25000,50000,75000,100000},
            'RESULT_REPORTS')
    return result


def admission(root):
    # Resource wait is not an experiment failure and never changes its settings.
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
           if line.startswith('MemAvailable:')}
    if mem['MemAvailable'] < 5 * 1024**2 or shutil.disk_usage(root).free < 5 * 1024**3:
        return False
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible:
        return False
    result = subprocess.run(['nvidia-smi', '-i', visible, '--query-gpu=memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if result.returncode != 0 or int(result.stdout.strip()) < 4096:
        return False
    active = subprocess.run(['nvidia-smi','-i',visible,'--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    return active.returncode == 0 and not active.stdout.strip()


def gpu_witness(root):
    """Real recipe shape, synthetic observations/rewards only, fresh disposable model."""
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from app.recipe_runtime import build_sac
    spec = verify(root)
    require(torch.cuda.is_available(), 'CUDA_REQUIRED')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    rec = read(spec['inputs']['recipe'])
    require(file_sha(spec['inputs']['recipe']) == spec['recipe_sha256'], 'RECIPE_PIN')

    class Synthetic(gym.Env):
        observation_space = gym.spaces.Dict({'features': gym.spaces.Box(-np.inf, np.inf,
            (rec['tensor']['window']['observations'], rec['list']['n']), np.float32)})
        action_space = gym.spaces.Box(-1.,1.,(3,),np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.n = 0
            return {'features': np.ones(self.observation_space['features'].shape,np.float32)}, {}

        def step(self, action):
            self.n += 1
            return {'features': np.full(self.observation_space['features'].shape,self.n / 20,np.float32)}, \
                float(-np.square(action).sum()), self.n == 20, False, {'discount':.99}

    for arm in spec['arms']:
        with preserve_rng():
            env = Synthetic()
            design = {**DESIGN, 'learning_starts':2, 'batch_size':4, 'buffer_size':32}
            model = build_sac(env, rec, design, 'GROUPED' if 'order_contracts' in spec else arm, 'cuda',
                              critic_dropout=spec.get('critic_dropout', {}).get(arm))
            if 'critic_dropout' in spec:
                with preserve_rng():
                    legacy = build_sac(env, rec, design, 'GROUPED', 'cuda')
                    require(tensor_hash(legacy) == tensor_hash(model), 'CRITIC_INITIAL_PARITY')
                    del legacy
            model.learn(total_timesteps=16)
            obs, _ = env.reset()
            before = model.predict(obs, deterministic=True)[0]
            path = root / ('ADMISSION_' + arm + '.zip')
            model.save(path)
            replay = SAC.load(path, device='cuda')
            require(np.array_equal(before,replay.predict(obs,deterministic=True)[0]), 'GPU_RELOAD')
            require(model._n_updates == 14, 'GPU_UPDATES')
            if 'critic_dropout' in spec:
                require(replay.actor.features_extractor.drop.p == .3
                        and replay.critic.features_extractor.drop.p == spec['critic_dropout'][arm]
                        and replay.critic_target.features_extractor.drop.p == spec['critic_dropout'][arm],
                        'GPU_CRITIC_DROPOUT_RELOAD')
            del model, replay
            env.close()
    write(root / 'GPU_ADMISSION.json', dict(state='PASS', scope='SYNTHETIC_ONLY', scientific_fits=0,
          spec_sha256=file_sha(root / 'SPEC.private.json'), created_at=now(),
          device=torch.cuda.get_device_name(), torch=torch.__version__,
          updates_per_arm=14, peak_bytes=torch.cuda.max_memory_allocated()))
    torch.cuda.empty_cache()
    return 0


def run(root):
    spec = verify(root)
    lock = (root / 'QUEUE.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    spec_sha = file_sha(root / 'SPEC.private.json')
    state = read(root / 'STATUS.json') if (root / 'STATUS.json').exists() else dict(
        state='PREPARED', spec_sha256=spec_sha, attempts={}, completed={})
    require(state['spec_sha256'] == spec_sha, 'QUEUE_IDENTITY')
    if state['state'] in ('ADMISSION_FAILED','ORDER_GATE_FAILED','CONTRAST_CONTROL_FAILED'):
        return 2
    if state['state'] in ('COMPLETE','COMPLETE_WITH_FAILURES'):
        for path in state['completed'].values():
            authenticated_result(root/path,spec_sha)
        return 0
    if spec['schema'] == 'recipe_sac_order_queue.v1':
        if not admission(root):
            state.update(state='WAIT_RESOURCES',updated_at=now())
            write(root/'STATUS.json',state)
            return 75
        for mode, name, limit in [('witness','ORDER_ALPHA.json',180),('diagnose','DIAGNOSTIC.json',1800)]:
            receipt = root/name
            if receipt.exists():
                evidence=read(receipt)
                require(evidence['state']=='PASS' and evidence['spec_sha256']==spec_sha,'ORDER_GATE_IDENTITY')
                for p,digest in evidence.get('artifacts',{}).items():
                    require(file_sha(root/'diagnostic'/p)==digest,'DIAGNOSTIC_ARTIFACT')
                continue
            state.update(state=mode.upper(),updated_at=now())
            write(root/'STATUS.json',state)
            with (root/(mode+'.log')).open('a') as log:
                try:
                    gate=subprocess.run([sys.executable,'-m','tools.recipe_sac_queue',mode,'--root',str(root)],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=limit,
                        env={**os.environ,**({'CUDA_VISIBLE_DEVICES':''} if mode=='witness' else {})})
                    rc=gate.returncode
                except subprocess.TimeoutExpired:
                    rc=124
            if rc or not receipt.exists():
                state.update(state='ORDER_GATE_FAILED',gate=mode,returncode=rc,updated_at=now())
                write(root/'STATUS.json',state)
                return 2
            evidence=read(receipt)
            require(evidence['state']=='PASS' and evidence['spec_sha256']==spec_sha,'ORDER_GATE_IDENTITY')
    for arm in spec['arms']:
        control = {'COMMANDS5':'TARGET', 'CRITIC_NODROP':'CRITIC_BASE'}.get(arm)
        if control and control not in state['completed']:
            state.update(state='CONTRAST_CONTROL_FAILED',updated_at=now())
            write(root/'STATUS.json',state)
            return 2
        if arm in state['completed']:
            authenticated_result(root / state['completed'][arm], spec_sha)
            continue
        old = state['attempts'].get(arm, [])
        # Recover a completed child even if the supervisor died before adopting it.
        for row in old:
            path = root / arm / f'attempt-{row["number"]}' / 'RESULT.json'
            if path.exists():
                authenticated_result(path, spec_sha)
                state['completed'][arm] = str(path.relative_to(root))
                break
            if row['state'] == 'RUNNING':
                row.update(state='INTERRUPTED', ended_at=now())
        if arm in state['completed']:
            write(root / 'STATUS.json', state)
            continue
        while len(old) < spec['max_attempts']:
            if not admission(root):
                state.update(state='WAIT_RESOURCES', updated_at=now())
                write(root / 'STATUS.json', state)
                return 75
            # Isolated process: GPU engineering does not retain a CUDA context in the supervisor.
            with (root / 'ADMISSION.log').open('a') as log:
                witness = subprocess.run([sys.executable, '-m', 'tools.recipe_sac_queue', 'admit',
                    '--root', str(root)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if witness.returncode != 0:
                state.update(state='ADMISSION_FAILED', updated_at=now())
                write(root / 'STATUS.json', state)
                return 2
            number = len(old) + 1
            row = dict(number=number, state='RUNNING', started_at=now())
            old.append(row)
            state['attempts'][arm] = old
            state.update(state='RUNNING', current_arm=arm, updated_at=now())
            write(root / 'STATUS.json', state)
            with (root / f'{arm}-attempt-{number}.log').open('x') as log:
                try:
                    completed = subprocess.run([sys.executable, '-m', 'tools.recipe_sac_queue', 'worker',
                        '--root', str(root), '--arm', arm, '--attempt', str(number)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=spec['task_timeout_seconds'])
                    row['returncode'] = completed.returncode
                except subprocess.TimeoutExpired:
                    row['returncode'] = 124
            path = root / arm / f'attempt-{number}' / 'RESULT.json'
            row.update(state='FAILED', ended_at=now())
            if row['returncode'] == 0 and path.exists():
                authenticated_result(path, spec_sha)
                row['state'] = 'COMPLETE'
                state['completed'][arm] = str(path.relative_to(root))
            write(root / 'STATUS.json', state)
            if row['state'] == 'COMPLETE':
                break
            time.sleep(30)
    state.update(state='COMPLETE' if len(state['completed']) == len(spec['arms']) else 'COMPLETE_WITH_FAILURES',
                 finished_at=now())
    if 'order_contracts' in spec:
        rows=[]
        for arm,path in state['completed'].items():
            result=authenticated_result(root/path,spec_sha)
            rows.extend(dict(arm=arm,**r) for r in result['reports'] if r['step']==100000)
        limits = (['single seed','partial observability without agent feedback',
                   'critic-only dropout contrast','assumed broker costs'] if 'critic_dropout' in spec else
                  ['single seed','partial observability without agent feedback',
                   'action package contrast including reversal semantics','assumed broker costs'])
        write(root/'ORDER_COMPARISON.json',dict(spec_sha256=spec_sha,rows=rows,
            selection='NONE_REPORT_ALL_FINAL_100000',finished_at=now(),
            limits=limits))
        if 'critic_dropout' in spec and len(state['completed']) == 2:
            folders = {a:(root/p).parent for a,p in state['completed'].items()}
            initial = [read(p/'EFFECTIVE_MODEL.json')['initial_policy_tensor_sha256'] for p in folders.values()]
            require(len(set(initial)) == 1, 'CRITIC_INITIAL_PARITY')
            probes = []
            for step in range(25000,100001,25000):
                for period in ('TRAIN','INNER_A'):
                    pair = {a:read(p/f'step-{step}'/(period+'.probe.json')) for a,p in folders.items()}
                    require(len({r['support_sha256'] for r in pair.values()}) == 1, 'CRITIC_PROBE_UNPAIRED')
                    probes.append(dict(step=step,period=period,arms=pair))
            write(root/'CRITIC_DIAGNOSTICS.json', dict(spec_sha256=spec_sha, pairs=probes,
                selection='NONE', initial_policy_exact=True, scope='FROZEN_PROBES_NOT_TRADING_SCORES'))
    write(root / 'STATUS.json', state)
    write(root / 'CLOSE.json', state)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'worker', 'status', 'admit','witness','diagnose'])
    parser.add_argument('--root', type=Path, required=True)
    for name in ('engine-root', 'recipe', 'store', 'manifest', 'bars', 'closures'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--arm', choices=['GROUPED','FLAT','TARGET','COMMANDS5',*CRITIC_DROPOUT])
    parser.add_argument('--order-parent',type=Path)
    parser.add_argument('--critic-parent',type=Path)
    parser.add_argument('--parent-spec-sha')
    parser.add_argument('--attempt', type=int)
    a = parser.parse_args()
    if a.mode == 'prepare':
        prepare(a)
        return 0
    if a.mode == 'status':
        print(json.dumps({p.stem: read(p) for p in (a.root / 'STATUS.json', a.root / 'PROGRESS.json', a.root / 'CLOSE.json')
                          if p.exists()}, indent=2))
        return 0
    if a.mode == 'worker':
        return worker(a.root.resolve(), a.arm, a.attempt)
    if a.mode == 'admit':
        return gpu_witness(a.root.resolve())
    if a.mode == 'witness':
        from tools.order_contract_study import witness
        spec=verify(a.root)
        result=witness(spec['engine_root'])
        result.update(spec_sha256=file_sha(a.root/'SPEC.private.json'),finished_at=now())
        write(a.root/'ORDER_ALPHA.json',result)
        return 0
    if a.mode == 'diagnose':
        from tools.order_contract_study import diagnose
        return diagnose(a.root.resolve())
    return run(a.root.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
