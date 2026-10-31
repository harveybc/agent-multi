"""Bounded discrete-SAC qualification -> real probe -> sealed emission replay."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from agent_plugins.discrete_curve_sac import CurveSAC, OBS_KEYS, route_surface, tensor_observation
from app.frozen_recipe_inputs import file_sha, validate_recipe
from app.rl_subset import digest, seal_recipe, support_hash
from app.recipe_runtime import RecipeMarket
from env_plugins.discrete_curve_native import native_environment

ROOT = Path(__file__).resolve().parents[1]
DESIGN = dict(steps=32768, warmup=1024, update_every=4, batch_size=64,
              replay_size=8192, gamma=.99, alpha=.01, learning_rate=.0003,
              qualification_updates=2048, qualification_batch=96,
              early_stopping=False, agent_feedback=False, learned_risk=False,
              algorithm='TorchRL_DiscreteSACLoss', branch_width=8, core_width=16,
              head_width=8, close_curve_knots=9)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def versions():
    return {n: importlib.metadata.version(n) for n in
            ('torch', 'torchrl', 'tensordict', 'numpy', 'pandas', 'gymnasium', 'pyarrow')}


def pins(root, directories):
    return {str(p.relative_to(root)): file_sha(p) for d in directories
            for p in sorted((root / d).rglob('*.py'))}


def prepare(args):
    root = args.root.resolve()
    if root.exists():
        raise ValueError('NEW_CAMPAIGN_REQUIRED')
    parent = read(args.source_spec)
    data = root / 'data'
    data.mkdir(parents=True)
    inputs = {}
    for k, p in parent['inputs'].items():
        if k == 'recipe':
            continue
        dst = data / (k + Path(p).suffix)
        shutil.copy2(p, dst)
        inputs[k] = str(dst)
    original = read(parent['inputs']['recipe'])
    validate_recipe(original)
    rec = json.loads(json.dumps(original))
    rec['split']['test_closed_from'] = '2024-01-01'
    rec['split']['periods'] = {'TRAIN': ['2020-01-01', '2023-07-01'],
                             'INNER_A': ['2023-07-01', '2024-01-01']}
    if rec['scaler']['fit_period'] != rec['split']['periods']['TRAIN']:
        raise ValueError('SCALER_FIT_SCOPE')
    cfg = rec['business']['evaluation_and_policy']
    cfg.update(max_entries_per_day=None, execution_price_mode='bid_ask_v3',
               cost_mode='configured', min_orders_per_week=2.)
    cfg['sizing'].update(rel_volume=.1, leverage=50., basis='margin')
    rec['business']['calendar'].update(entry_delay_after_open_hours=12,
        force_flat_before_close_hours=12, entry_cutoff_before_flat_hours=0)
    rec['provenance'] = dict(parent_recipe_identity=original['identity'],
        scope='bounded RL probe; inherited phase-1 selection; no final floating selection',
        exposure='weights/scaler fit before2023H2; feature/risk design includes2023H2; development only')
    seal_recipe(rec)
    write(data / 'RECIPE.json', rec)
    inputs['recipe'] = str(data / 'RECIPE.json')
    shutil.copy2(ROOT / 'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json', data / 'RISK.json')
    spec = dict(schema='discrete_curve_probe.v1', inputs=inputs,
        recipe_sha256=file_sha(inputs['recipe']), engine_root=str(args.engine_root.resolve()),
        source_parent_sha256=file_sha(args.source_spec), design=DESIGN, software=versions(),
        risk=str(data / 'RISK.json'), risk_sha256=file_sha(data / 'RISK.json'),
        input_sha256={k: file_sha(v) for k, v in inputs.items()},
        code=pins(ROOT, ['app', 'agent_plugins', 'env_plugins', 'tools']),
        native=pins(args.engine_root.resolve(), ['app/short_horizon']),
        units=[dict(id=f'h{h}_s{s}', horizon=h, seed=s) for h in (3, 6, 12, 24) for s in (7, 19, 43)],
        max_attempts=2, task_timeout_seconds=43200, created_at=now())
    spec['identity'] = digest(spec)
    write(root / 'SPEC.private.json', spec)
    market = RecipeMarket(spec)
    report = dict(identity=spec['identity'], recipe_identity=rec['identity'],
        features=rec['list']['n'], groups=len(rec['representation']['resolved']['names']),
        window=rec['tensor']['window'], scaler_fit=rec['scaler']['fit_period'],
        support_hash=support_hash(market.inputs), validation_read=False, test_read=False,
        rows={p: int(((market.inputs.period == p) & market.inputs.valid).sum()) for p in ('TRAIN', 'INNER_A')},
        caveat=rec['provenance']['exposure'])
    for h in (3, 6, 12, 24):
        for p in ('TRAIN', 'INNER_A'):
            env = native_environment(market, read(spec['risk']), h, p)
            obs = env.reset()
            for _ in range(32):
                obs, _, done, _ = env.step(0)
                if done:
                    break
    write(root / 'DATA_PREFLIGHT.json', report)
    print(json.dumps(report), flush=True)


def authenticate(root):
    spec = read(root / 'SPEC.private.json')
    if digest({k: v for k, v in spec.items() if k != 'identity'}) != spec['identity']:
        raise ValueError('SPEC_IDENTITY')
    if spec['software'] != versions() or spec['design'] != DESIGN:
        raise ValueError('RUNTIME_VERSION_OR_DESIGN')
    for base, values in ((ROOT, spec['code']), (Path(spec['engine_root']), spec['native'])):
        for p, sha in values.items():
            if file_sha(base / p) != sha:
                raise ValueError('SOURCE_CHANGED:' + p)
    for key, sha in spec['input_sha256'].items():
        if file_sha(spec['inputs'][key]) != sha:
            raise ValueError('INPUT_CHANGED:' + key)
    if file_sha(spec['risk']) != spec['risk_sha256']:
        raise ValueError('RISK_CHANGED')
    return spec


def configure(seed):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    torch.manual_seed(seed)


def qualification_batch(recipe, rng, n, device):
    shape = (recipe['tensor']['window']['observations'], recipe['list']['n'])
    x = np.zeros((n, *shape), np.float32)
    signals = rng.integers(-1, 2, n)
    lags = rng.choice([0, 4, 8, 12], n)
    x[np.arange(n), shape[0] - 1 - lags, :] = signals[:, None]
    route = rng.integers(0, 3, n)
    progress = rng.choice([.125, .875], n).astype(np.float32)
    expected = np.where(route == 0, np.where(signals == 1, 1, np.where(signals == -1, 2, 0)),
        np.where(((route == 1) & (signals == -1)) | ((route == 2) & (signals == 1))
                 | ((signals == 0) & (progress < .5)), 1, 0))
    mask = np.ones((n, 3), bool)
    mask[route != 0, 2] = False
    obs = TensorDict(dict(features=torch.tensor(x, device=device),
        route=torch.tensor(route[:, None], device=device),
        progress=torch.tensor(progress[:, None], device=device),
        mask=torch.tensor(mask, device=device)), [n])
    return obs, torch.tensor(expected, device=device)


def qualify(model, recipe, seed, output):
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    for update in range(DESIGN['qualification_updates']):
        obs, expected = qualification_batch(recipe, rng, DESIGN['qualification_batch'], model.device)
        with torch.no_grad():
            action = model.distribution(obs).sample()
        nxt = obs.clone()
        nxt['reward'] = torch.where(action == expected, 1., -1.)[:, None]
        nxt['done'] = torch.ones(len(action), 1, dtype=torch.bool, device=model.device)
        nxt['terminated'] = nxt['done'].clone()
        obs['action'], obs['next'] = action, nxt
        loss = model.update(obs)
        if update % 64 == 0:
            write(output / 'PROGRESS.json', dict(stage='SYNTHETIC_SAC_QUALIFICATION',
                update=update, total=DESIGN['qualification_updates'], seconds=time.monotonic() - start, **loss))
    rows = []
    shape = (recipe['tensor']['window']['observations'], recipe['list']['n'])
    for lag in (0, 4, 8, 12):
        for route in range(3):
            for signal in (-1, 0, 1):
                for progress in (.125, .875):
                    feature = np.zeros(shape, np.float32)
                    feature[-1 - lag, :] = signal
                    obs = tensor_observation(dict(features=feature, route=np.array([route]),
                        progress=np.array([progress], np.float32), mask=np.array([1, 1, route == 0], bool)), model.device)
                    expected = ({-1: 2, 0: 0, 1: 1}[signal] if route == 0 else
                                int(signal == (-1 if route == 1 else 1) or (signal == 0 and progress < .5)))
                    with torch.no_grad():
                        probs = model.distribution(obs).probs[0].cpu().numpy()
                    rows.append(dict(lag=lag, route=route, signal=signal, progress=progress,
                        expected=expected, chosen=int(probs.argmax()), probabilities=probs.tolist()))
    passed = all(r['expected'] == r['chosen'] and r['probabilities'][r['expected']] > .6 for r in rows)
    write(output / 'QUALIFICATION.json', dict(scope='SYNTHETIC_LEARNABILITY_NOT_TRADING',
        passed=passed, seed=seed, seconds=time.monotonic() - start, rows=rows))
    model.save(output / 'qualification.pt')
    return passed


def transition(obs, action, nxt, reward, done, duration):
    td = tensor_observation(obs)
    td['action'] = torch.tensor([action], dtype=torch.long)
    td['steps_to_next_obs'] = torch.tensor([[duration]])
    following = tensor_observation(nxt)
    following['reward'] = torch.tensor([[reward]], dtype=torch.float32)
    following['done'] = torch.tensor([[done]], dtype=torch.bool)
    following['terminated'] = following['done'].clone()
    td['next'] = following
    return td


def train(model, env, output, seed):
    rng = np.random.default_rng(seed)
    replay = TensorDictReplayBuffer(storage=LazyTensorStorage(DESIGN['replay_size'], device='cpu'),
                                   batch_size=DESIGN['batch_size'])
    obs = env.reset()
    start, updates = time.monotonic(), 0
    for step in range(DESIGN['steps']):
        if step < DESIGN['warmup']:
            action = int(rng.choice(np.flatnonzero(obs['mask'])))
        else:
            with torch.no_grad():
                action = int(model.distribution(tensor_observation(obs, model.device)).sample()[0])
        nxt, reward, done, duration = env.step(action)
        replay.extend(transition(obs, action, nxt, reward, done, duration))
        obs = env.reset() if done else nxt
        if step >= DESIGN['warmup'] and step % DESIGN['update_every'] == 0:
            loss = model.update(replay.sample().to(model.device))
            updates += 1
        if step % 128 == 0:
            write(output / 'PROGRESS.json', dict(stage='REAL_TRAIN', step=step, total=DESIGN['steps'],
                updates=updates, seconds=time.monotonic() - start,
                replay_transitions=len(replay), peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                loss=loss if updates else None))
    model.save(output / 'model.pt')
    return dict(steps=DESIGN['steps'], updates=updates, seconds=time.monotonic() - start,
                replay_size=len(replay), replay_policy='uniform off-policy; no expert or validation transitions')


def seal_emissions(model, market, output):
    indices = np.flatnonzero(market.inputs.valid & np.isin(market.inputs.period, ['TRAIN', 'INNER_A']))
    surfaces, qvalues = [], []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            x = torch.tensor(market.inputs.windows[indices[start:start + 128]], device=model.device)
            surfaces.append(model.surfaces(x).cpu().numpy())
            qvalues.append(model.q_surfaces(x).cpu().numpy())
    surfaces, qvalues = np.concatenate(surfaces), np.concatenate(qvalues, axis=1)
    if not np.isfinite(surfaces).all() or not np.isfinite(qvalues).all():
        raise ValueError('NONFINITE_EMISSION')
    np.savez_compressed(output / 'EMISSIONS.npz', indices=indices, surfaces=surfaces, qvalues=qvalues,
                        timestamp=market.inputs.row_ts_ns[indices])
    write(output / 'EMISSION_SEAL.json', dict(sha256=file_sha(output / 'EMISSIONS.npz'),
        sealed_at=now(), before_account_replay=True, rows=len(indices), dimensions=39,
        mean=surfaces.mean(0).tolist(), std=surfaces.std(0).tolist(),
        quantiles=np.quantile(surfaces, [.01, .5, .99], axis=0).tolist(),
        interpretation='policy logits, not calibrated success probabilities; critics are soft returns'))
    return dict(zip(indices.tolist(), surfaces)), dict(zip(indices.tolist(), qvalues.transpose(1, 0, 2)))


def evaluate(env, surfaces, output, period):
    import pandas as pd
    obs, commands = env.reset(), Counter()
    steps = []
    while True:
        row = int(env.rows[env.bar])
        raw = torch.tensor(surfaces[row]).unsqueeze(0)
        logits = route_surface(raw, torch.tensor(obs['route'])[None], torch.tensor(obs['progress'])[None])[0]
        logits[~torch.tensor(obs['mask'])] = -1e9
        probs = logits.softmax(-1)
        action = int(logits.argmax())
        route = int(obs['route'][0])
        commands[f'{route}:{action}'] += 1
        stamp = int(env.engine.v[env.bar]) + 3600 * 10**9
        obs, reward, done, duration = env.step(action)
        steps.append(dict(timestamp=stamp, route=route, action=action, probabilities=probs.tolist(),
                          reward=reward, duration=duration))
        if done:
            break
    engine = env.engine
    engine.flatten_at_end()
    trace = np.asarray(env.equity_trace)
    daily = pd.Series(trace[:, 1], index=pd.to_datetime(trace[:, 0].astype(np.int64), utc=True)).resample('1D').last().ffill()
    returns = daily.pct_change().dropna().to_numpy()
    peak = np.maximum.accumulate(trace[:, 1])
    profit = float(engine.balance - engine.cfg.capital)
    accounted = sum(t['net_usd'] for t in engine.trades)
    if not np.isclose(profit, accounted, rtol=0, atol=1e-6):
        raise ValueError('CASH_LEDGER_MISMATCH')
    def side(direction):
        trades = [t for t in engine.trades if t['direction'] == direction]
        return dict(trades=len(trades), win_rate=None if not trades else
                    sum(t['net_usd'] > 0 for t in trades) / len(trades),
                    profit=sum(t['net_usd'] for t in trades))
    weeks = len({tuple(t.isocalendar()[:2]) for t in pd.to_datetime(trace[:, 0].astype(np.int64), utc=True)})
    result = dict(period=period, net_profit_usd=profit, return_pct=100 * profit / engine.cfg.capital,
        sharpe_daily_sqrt365=float(np.sqrt(365) * returns.mean() / returns.std(ddof=1))
            if len(returns) > 1 and returns.std(ddof=1) > 0 else None,
        max_drawdown_pct=float(np.max((peak - trace[:, 1]) / peak) * 100),
        trades=len(engine.trades), buy=side(1), sell=side(-1),
        orders_per_week=len(engine.trades) / weeks, activity_eligible=len(engine.trades) / weeks > 2,
        win_rate=sum(t['net_usd'] > 0 for t in engine.trades) / len(engine.trades) if engine.trades else None,
        close_reasons=dict(Counter(t['close_reason'] for t in engine.trades)),
        commands=dict(commands), automatic_hold_bars=env.auto_holds, native_counts=engine.counts,
        costs={key: sum(t[key] for t in engine.trades) for key in
               ('spread_usd', 'commission_usd', 'slippage_usd', 'financing_usd')},
        final_balance=engine.balance, cash_error=abs(profit - accounted))
    write(output / f'{period}_TRADES.json', json.loads(json.dumps(engine.trades, default=str)))
    write(output / f'{period}_ACTIONS.json', steps)
    np.savez_compressed(output / f'{period}_EQUITY.npz', timestamp=trace[:, 0].astype(np.int64), equity=trace[:, 1])
    return result


def receipt_valid(output, identity):
    path = output / 'RESULT.json'
    if not path.exists():
        return False
    result = read(path)
    if result['identity'] != identity:
        raise ValueError('RESULT_IDENTITY')
    for name, sha in result['artifacts'].items():
        if Path(name).name != name or file_sha(output / name) != sha:
            raise ValueError('RESULT_ARTIFACT_CHANGED')
    return True


def worker(args):
    spec = authenticate(args.root)
    unit = next(u for u in spec['units'] if u['id'] == args.unit)
    output = args.root / 'runs' / unit['id'] / f'attempt{args.attempt}'
    output.mkdir(parents=True, exist_ok=False)
    configure(unit['seed'])
    if not torch.cuda.is_available():
        raise ValueError('CUDA_REQUIRED_NO_CPU_FALLBACK')
    recipe = read(spec['inputs']['recipe'])
    cache = args.root / 'qualifications' / str(unit['seed'])
    if receipt_valid(cache, spec['identity']):
        cached = read(cache / 'RESULT.json')
        passed = cached['passed']
        for name in cached['artifacts']:
            shutil.copy2(cache / name, output / name)
    else:
        model = CurveSAC(recipe, 'cuda', seed=unit['seed'])
        passed = qualify(model, recipe, unit['seed'], output)
        if passed:
            from app.curve_probe_control import planted_control
            from app.recipe_runtime import attach_engine
            attach_engine(spec['engine_root'])
            controls = [planted_control(model, recipe, read(spec['risk']), m) for m in (1, -1)]
            write(output / 'NATIVE_POSITIVE_CONTROLS.json', controls)
            passed = all(c['passed'] for c in controls)
        cache.mkdir(parents=True, exist_ok=True)
        files = ['QUALIFICATION.json', 'qualification.pt']
        if (output / 'NATIVE_POSITIVE_CONTROLS.json').exists():
            files.append('NATIVE_POSITIVE_CONTROLS.json')
        for name in files:
            shutil.copy2(output / name, cache / name)
        write(cache / 'RESULT.json', dict(identity=spec['identity'], passed=passed,
            seed=unit['seed'], artifacts={name: file_sha(cache / name) for name in files}))
        del model
        torch.cuda.empty_cache()
    result = dict(identity=spec['identity'], unit=unit, algorithm='SAC_DISCRETE',
                  qualification_passed=passed, financial_training=False)
    if passed:
        # Discard synthetic-trained weights and optimizer before real training.
        model = CurveSAC(recipe, 'cuda', seed=unit['seed'])
        market = RecipeMarket(spec)
        risk = read(spec['risk'])
        result['training'] = train(model, native_environment(market, risk, unit['horizon'], 'TRAIN'), output, unit['seed'])
        sample = torch.tensor(market.inputs.windows[np.flatnonzero(market.inputs.valid)[:8]], device='cuda')
        with torch.no_grad():
            before = model.surfaces(sample).clone()
        model.load(output / 'model.pt')
        with torch.no_grad():
            if not torch.equal(before, model.surfaces(sample)):
                raise ValueError('MODEL_RELOAD_MISMATCH')
        surfaces, _ = seal_emissions(model, market, output)
        result['metrics'] = {p: evaluate(native_environment(market, risk, unit['horizon'], p), surfaces, output, p)
                             for p in ('TRAIN', 'INNER_A')}
        result.update(financial_training=True, reload_exact=True,
            scope='DEVELOPMENT_PROBE_NOT_FINAL_STRATEGY_OR_FLOATING_SELECTION',
            feature_and_risk_selection_exposure=recipe['provenance']['exposure'])
    result['finished_at'] = now()
    result['artifacts'] = {p.name: file_sha(p) for p in output.iterdir() if p.is_file() and p.name != 'PROGRESS.json'}
    write(output / 'RESULT.json', result)


def run(args):
    spec = authenticate(args.root)
    assigned = read(args.assignment)['units']
    all_units = {u['id'] for u in spec['units']}
    if not assigned or len(assigned) != len(set(assigned)) or not set(assigned) <= all_units:
        raise ValueError('ASSIGNMENT')
    with (args.root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        outcomes = {}
        for unit in assigned:
            done = False
            for attempt in range(1, spec['max_attempts'] + 1):
                folder = args.root / 'runs' / unit / f'attempt{attempt}'
                if receipt_valid(folder, spec['identity']):
                    outcomes[unit] = read(folder / 'RESULT.json')
                    done = True
                    break
                if folder.exists():
                    continue
                while True:
                    memory = {line.split(':')[0]: int(line.split()[1]) for line in
                              Path('/proc/meminfo').read_text().splitlines() if ':' in line}
                    free_gpu = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                        '--format=csv,noheader,nounits', '-i', os.environ['CUDA_VISIBLE_DEVICES']], text=True).strip())
                    if memory['MemAvailable'] > 3 * 1024**2 and free_gpu > 3072:
                        break
                    write(args.root / 'STATUS.json', dict(state='WAIT_RESOURCES', unit=unit,
                        free_ram_kib=memory['MemAvailable'], free_vram_mib=free_gpu, time=now()))
                    time.sleep(30)
                write(args.root / 'STATUS.json', dict(identity=spec['identity'], state='RUNNING', unit=unit,
                      attempt=attempt, completed=len(outcomes), total=len(assigned), time=now()))
                log = args.root / f'{unit}_attempt{attempt}.log'
                with log.open('a') as stream:
                    try:
                        proc = subprocess.run([sys.executable, '-m', 'tools.run_discrete_curve_probe', 'worker',
                            '--root', str(args.root), '--unit', unit, '--attempt', str(attempt)],
                            stdout=stream, stderr=subprocess.STDOUT, timeout=spec['task_timeout_seconds'])
                        code = proc.returncode
                    except subprocess.TimeoutExpired:
                        code = 'TIMEOUT'
                write(args.root / f'{unit}_attempt{attempt}_EXIT.json', dict(exit_code=code, time=now()))
                if code == 0 and receipt_valid(folder, spec['identity']):
                    outcomes[unit] = read(folder / 'RESULT.json')
                    done = True
                    break
                time.sleep(30 * attempt)
            if not done:
                outcomes[unit] = dict(status='PROCESS_FAILED', financial_training=False)
            write(args.root / 'SUMMARY.json', outcomes)
        write(args.root / 'STATUS.json', dict(identity=spec['identity'], state='COMPLETE',
              completed=len(outcomes), total=len(assigned), time=now(),
              financial_completed=sum(r.get('financial_training', False) for r in outcomes.values()),
              failures=sum(r.get('status') == 'PROCESS_FAILED' for r in outcomes.values())))


def smoke(args):
    """Engineering only: short real replay/update/reload/emission integration."""
    spec = read(args.root / 'SPEC.private.json')
    configure(7)
    output = args.root / 'SMOKE'
    output.mkdir(exist_ok=False)
    recipe = read(spec['inputs']['recipe'])
    market, risk = RecipeMarket(spec), read(spec['risk'])
    model = CurveSAC(recipe, 'cuda', seed=7)
    env = native_environment(market, risk, 3, 'TRAIN')
    replay = TensorDictReplayBuffer(storage=LazyTensorStorage(128), batch_size=32)
    obs = env.reset()
    rng = np.random.default_rng(7)
    for _ in range(128):
        action = int(rng.choice(np.flatnonzero(obs['mask'])))
        nxt, reward, done, duration = env.step(action)
        replay.extend(transition(obs, action, nxt, reward, done, duration))
        obs = env.reset() if done else nxt
    start = time.monotonic()
    losses = [model.update(replay.sample().to('cuda')) for _ in range(8)]
    update_seconds = (time.monotonic() - start) / 8
    sample = tensor_observation(obs, 'cuda')
    with torch.no_grad():
        before = model.distribution(sample).probs.clone()
    model.save(output / 'model.pt')
    model.load(output / 'model.pt')
    with torch.no_grad():
        if not torch.equal(before, model.distribution(sample).probs):
            raise ValueError('SMOKE_RELOAD')
    surfaces, _ = seal_emissions(model, market, output)
    report = evaluate(native_environment(market, risk, 3, 'INNER_A'), surfaces, output, 'INNER_A')
    write(output / 'RESULT.json', dict(scope='ENGINEERING_ONLY_NOT_A_CANDIDATE',
        passed=True, update_seconds=update_seconds, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        metrics=report, losses=losses, source=pins(ROOT, ['app', 'agent_plugins', 'env_plugins', 'tools'])))
    print(json.dumps(dict(passed=True, update_seconds=update_seconds,
                         peak_cuda_bytes=torch.cuda.max_memory_allocated(), cash_error=report['cash_error'])), flush=True)


def collect(args):
    config = read(args.assignment)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    requested = [u for node in config['nodes'] for u in node['units']]
    if len(requested) != 12 or len(set(requested)) != 12:
        raise ValueError('FLEET_PARTITION')
    while True:
        rows, issues = {}, []
        for node in config['nodes']:
            mirror = root / node['role']
            mirror.mkdir(exist_ok=True)
            source = node['root'] + '/'
            if node.get('host'):
                source = node['host'] + ':' + source
            command = ['rsync', '-a', '--exclude=data/', '--exclude=*.tmp', '--exclude=*.log',
                       '--exclude=queue.lock', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                       source, str(mirror) + '/']
            try:
                subprocess.run(command, check=True, timeout=90, capture_output=True)
            except (subprocess.SubprocessError, OSError):
                issues.append(node['role'] + ':TRANSPORT_RETRY')
                continue
            spec = read(mirror / 'SPEC.private.json')
            for unit in node['units']:
                for attempt in range(1, spec['max_attempts'] + 1):
                    folder = mirror / 'runs' / unit / f'attempt{attempt}'
                    if receipt_valid(folder, spec['identity']):
                        rows[unit] = read(folder / 'RESULT.json')
                        break
            if (mirror / 'STATUS.json').exists() and read(mirror / 'STATUS.json')['state'] == 'COMPLETE':
                for unit in node['units']:
                    if unit not in rows:
                        rows[unit] = dict(status='PROCESS_FAILED', financial_training=False)
        write(root / 'AGGREGATE.json', dict(time=now(), complete=len(rows), expected=12,
              issues=issues, results=rows))
        lines = ['# Discrete SAC Probe', '', 'Development only. No 2024 or TEST evaluation.', '',
                 '| Unit | Status | TRAIN return % | Internal return % | Internal Sharpe | Orders/week | Buy / Sell |',
                 '|---|---|---:|---:|---:|---:|---:|']
        for unit, row in sorted(rows.items()):
            if row.get('financial_training'):
                a, b = row['metrics']['TRAIN'], row['metrics']['INNER_A']
                lines.append(f"| {unit} | complete | {a['return_pct']:.4f} | {b['return_pct']:.4f} | "
                    f"{b['sharpe_daily_sqrt365']} | {b['orders_per_week']:.3f} | {b['buy']['trades']} / {b['sell']['trades']} |")
            else:
                lines.append(f"| {unit} | {row.get('status', 'qualification failed')} | - | - | - | - | - |")
        temp = root / 'RESULTS.md.tmp'
        temp.write_text('\n'.join(lines) + '\n')
        temp.replace(root / 'RESULTS.md')
        if len(rows) == 12:
            write(root / 'CLOSE.json', dict(completed_at=now(), results_sha256=file_sha(root / 'AGGREGATE.json'),
                  financial_completed=sum(r.get('financial_training', False) for r in rows.values()),
                  strategy_promoted=False, floating_selection_started=False))
            return
        time.sleep(120)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'worker', 'smoke', 'collect'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--source-spec', type=Path)
    p.add_argument('--engine-root', type=Path)
    p.add_argument('--assignment', type=Path)
    p.add_argument('--unit')
    p.add_argument('--attempt', type=int, default=1)
    args = p.parse_args()
    {'prepare': prepare, 'run': run, 'worker': worker, 'smoke': smoke, 'collect': collect}[args.command](args)


if __name__ == '__main__':
    main()
