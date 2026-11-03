"""Fresh selected-channel SAC fit, full annual monitoring and reusable weights."""
import argparse
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.recipe_runtime import RecipeMarket
from app.rl_subset import digest, seal_recipe, subset_recipe, refit_train_scaler, project_inputs, support_hash
from app.structural_refit import (EpochPatience, capture_rng, restore_rng, build_model,
    export_bundle, load_bundle, save_recovery, load_recovery)
from app.three_signal_learning import environment, emit, evaluate
from agent_plugins.discrete_curve_sac import tensor_observation
from tools.run_discrete_curve_probe import read, write, now, configure, transition
from tools.run_three_signal_baseline import synthetic_controls

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/run_structural_sac_refit.py', 'app/structural_refit.py',
    'app/frozen_recipe_inputs.py', 'app/frozen_signal_masks.py', 'app/rl_subset.py',
    'app/recipe_runtime.py', 'app/three_signal_learning.py', 'app/horizon_risk_rules.py',
    'app/train_native_pareto.py', 'app/dynamic_account_panel.py',
    'agent_plugins/three_signal_sac.py', 'agent_plugins/discrete_curve_sac.py',
    'agent_plugins/double_tcn_probe.py', 'env_plugins/three_signal_native.py',
    'env_plugins/discrete_curve_native.py', 'tools/run_discrete_curve_probe.py',
    'tools/run_three_signal_baseline.py']
SOFTWARE = ['torch', 'torchrl', 'tensordict', 'numpy', 'pandas', 'pyarrow']


def market_for(root, spec):
    paths = {k:str(root/v) for k,v in spec['inputs'].items()}
    paths['recipe'] = str(root/'SUPPORT_RECIPE.json')
    market = RecipeMarket(dict(inputs=paths, engine_root=str(root/'engine'),
        recipe_sha256=spec['files']['SUPPORT_RECIPE.json']))
    rec = read(root/'RECIPE.json')
    market.inputs = project_inputs(market.inputs, rec)
    market.recipe = rec
    return market


def prepare(args):
    root = args.root.resolve()
    if root.exists():
        raise ValueError('NEW_REFIT_ROOT_REQUIRED')
    parent = read(args.parent/'SPEC.private.json')
    close = read(args.selection/'CLOSE.json')
    result = read(args.selection/'RESULT.json')
    if (close['termination'] != 'TWO_PASSES_COMPLETE' or result['identity'] != close['identity']
            or file_sha(args.selection/'RESULT.json') != close['result_sha256']):
        raise ValueError('SELECTION_RECEIPT')
    selected = result['result']['selected']
    if len(selected) != 108:
        raise ValueError('EXPECTED_108_SELECTED_FEATURES')
    for name,path in parent['inputs'].items():
        if file_sha(path) != parent['input_sha256'][name]:
            raise ValueError('PARENT_INPUT:' + name)
    original = read(parent['inputs']['recipe'])
    baseline = read(args.baseline/'SPEC.private.json')
    if file_sha(args.baseline/'RISK.json') != baseline['risk_sha256']:
        raise ValueError('BASELINE_RISK')
    root.mkdir(); (root/'data').mkdir()
    for key in ('store', 'manifest', 'bars', 'closures'):
        shutil.copy2(parent['inputs'][key], root/'data'/Path(parent['inputs'][key]).name)
    shutil.copy2(args.baseline/'RISK.json', root/'RISK.json')
    shutil.copy2(args.selection/'RESULT.json', root/'SELECTION_RESULT.json')
    shutil.copy2(args.selection/'CLOSE.json', root/'SELECTION_CLOSE.json')
    # Export the exact native source tree, not a moving checkout or symlink.
    for name,sha in parent['native'].items():
        src = Path(parent['engine_root'])/name
        if file_sha(src) != sha:
            raise ValueError('NATIVE_SOURCE:' + name)
        dst = root/'engine'/name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    rec = copy.deepcopy(original)
    rec['split'] = dict(arm='owner_full_4plus1',
        periods={'TRAIN':['2020-01-01','2024-01-01'], 'VALIDATION':['2024-01-01','2025-01-01']},
        train=['2020-01-01','2024-01-01'], validation=['2024-01-01','2025-01-01'],
        test_closed_from='2025-01-01', selection_period='VALIDATION',
        status='development_validation_previously_exposed', support_status='full_parent_support_projected')
    frame = pd.read_parquet(parent['inputs']['store'], columns=['timestamp']+rec['list']['columns'],
                            filters=[('timestamp','<',pd.Timestamp('2025-01-01',tz='UTC'))])
    rec = refit_train_scaler(rec, frame)
    write(root/'SUPPORT_RECIPE.json', rec)
    reduced = subset_recipe(rec, selected)
    write(root/'RECIPE.json', reduced)
    d0 = baseline['design']
    design = {k:d0[k] for k in ('batch_size','warmup','replay_size','update_every','horizon')}
    design.update(max_epochs=2000, patience=20, min_delta=1e-8, seeds=[7,19],
        epoch='one_complete_TRAIN_traversal', monitor='mean_full_TRAIN_VALIDATION_return_per_week',
        hyperparameters_source='fixed_mask_baseline_not_completed_DEAP_optimum')
    effective = dict(architecture=dict(branch_width=8,core_width=16,branch_blocks=2,core_blocks=2,head_width=8),
        gamma=d0['gamma'],alpha=d0['alpha'],learning_rate=d0['learning_rate'],
        entropy_fraction=d0['entropy_fraction'],tau=.005,weight_decay=rec['regularization']['wd'],
        dropout=0.,batch_norm=False)
    inputs = {k:str(Path('data')/Path(parent['inputs'][k]).name) for k in ('store','manifest','bars','closures')}
    spec = dict(schema='structural_sac.refit.v1', created_at=now(), design=design, effective=effective,
        inputs=inputs, files={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()},
        sources={n:file_sha(ROOT/n) for n in SOURCES},
        software={n:importlib.metadata.version(n) for n in SOFTWARE},
        selection_identity=result['identity'], parent_recipe_identity=original['identity'])
    spec['identity'] = digest(spec)
    write(root/'SPEC.json', spec)
    market = market_for(root, spec)
    risk = read(root/'RISK.json')
    controls = synthetic_controls(reduced, risk, dict(gamma=effective['gamma']))
    if not controls['passed']:
        raise ValueError('NATIVE_CONTROL_FAILED')
    counts = {}
    for period,(lo,hi) in reduced['split']['periods'].items():
        chosen = (market.inputs.period == period)
        stamps = pd.to_datetime(market.inputs.row_ts_ns[chosen],utc=True)
        months = sorted(set(stamps.strftime('%Y-%m')))
        expected = list(pd.period_range(lo,pd.Timestamp(hi)-pd.Timedelta(days=1),freq='M').astype(str))
        if months != expected:
            raise ValueError('MISSING_MONTH:' + period)
        env = environment(market,risk,design['horizon'],[lo,hi],effective['gamma'])
        counts[period] = dict(months=len(months), rows=int(chosen.sum()),
            valid_windows=int((chosen & market.inputs.valid).sum()),
            first=str(stamps.min()),last=str(stamps.max()),
            decision_count=int(env.valid[env.engine.first:env.engine.last].sum()),
            native_bars=int(env.engine.last-env.engine.first))
    write(root/'PREFLIGHT.json', dict(identity=spec['identity'], periods=counts,
        features=reduced['list']['n'],groups=len(reduced['representation']['resolved']['names']),
        tensor_shape=list(market.inputs.windows.shape),support_sha256=support_hash(market.inputs),
        controls=controls, test_read=False, fresh_training=True))
    print(json.dumps(read(root/'PREFLIGHT.json')),flush=True)


def authenticate(root):
    spec = read(root/'SPEC.json')
    if spec['identity'] != digest({k:v for k,v in spec.items() if k != 'identity'}):
        raise ValueError('REFIT_SPEC_IDENTITY')
    for name,sha in spec['files'].items():
        if file_sha(root/name) != sha:
            raise ValueError('REFIT_INPUT:' + name)
    for name,sha in spec['sources'].items():
        if file_sha(ROOT/name) != sha:
            raise ValueError('REFIT_SOURCE:' + name)
    for name,version in spec['software'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('REFIT_SOFTWARE:' + name)
    preflight = read(root/'PREFLIGHT.json')
    if preflight['identity'] != spec['identity'] or not preflight['controls']['passed']:
        raise ValueError('REFIT_PREFLIGHT')
    return spec


def assess_full(model, market, risk, design, save=None, diagnose=False):
    logits,_ = emit(model,market.inputs)
    prepared = prepared_signals(logits)
    metrics = {p:evaluate(environment(market,risk,design['horizon'],interval,model.effective['gamma']),
        logits, save=save/p if save else None, diagnose=diagnose, prepared=prepared)
        for p,interval in market.recipe['split']['periods'].items()}
    return metrics


def fit_epochs(model, env_factory, assess, design, seed, output, identity, probe):
    output.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(seed)
    replay = TensorDictReplayBuffer(storage=LazyTensorStorage(design['replay_size'],device='cpu'),
                                   batch_size=design['batch_size'])
    stopping = EpochPatience(design['patience'],design['min_delta'])
    state = dict(identity=identity,seed=seed,epoch=0,steps=0,updates=0,history=[],
                 best_bundle=None,stopping=vars(stopping).copy())
    if (output/'RESUME.json').exists():
        state = load_recovery(output,model,replay,rng,identity)
        stopping.__dict__.update(state['stopping'])
    # Anything newer than the atomic pointer is an interrupted, uncommitted epoch.
    for p in (output/'recovery').glob('epoch_*') if (output/'recovery').exists() else []:
        if int(p.name.split('_')[1].split('.')[0]) > state['epoch']:
            shutil.rmtree(p)
    for epoch in range(state['epoch']+1,design['max_epochs']+1):
        if stopping.bad >= stopping.patience:
            break
        start = time.monotonic(); env = env_factory(); obs = env.reset()
        last_loss = None; decisions = 0
        while not env.finished:
            step = state['steps']+1
            if step <= design['warmup']:
                action = int(rng.integers(0,3))
            else:
                with torch.no_grad():
                    action = int(model.distribution(tensor_observation(obs,model.device)).sample()[0])
            nxt,reward,done,duration = env.step(action)
            replay.extend(transition(obs,action,nxt,reward,done,duration))
            obs = nxt; state['steps'] = step; decisions += 1
            if step >= design['warmup'] and step % design['update_every'] == 0:
                last_loss = model.update(replay.sample().to(model.device)); state['updates'] += 1
            if step % 256 == 0:
                write(output/'PROGRESS.json',dict(identity=identity,stage='TRAIN',epoch=epoch,
                    completed_epochs=state['epoch'],steps=step,updates=state['updates'],
                    decisions_this_epoch=decisions,epoch_seconds=time.monotonic()-start,
                    native_bar=int(env.bar),last_native_bar=int(env.engine.last),
                    bad=stopping.bad,best=stopping.best,loss=last_loss))
        training_seconds = time.monotonic()-start
        write(output/'PROGRESS.json',dict(identity=identity,stage='FULL_TRAIN_VALIDATION',epoch=epoch,
                                         training_seconds=training_seconds))
        before = capture_rng(rng); replay_count = len(replay)
        metrics = assess(model)
        if not torch.equal(before['torch'],torch.get_rng_state()) or replay_count != len(replay):
            raise ValueError('ASSESSMENT_MUTATED_TRAINING')
        if before['cuda'] and any(not torch.equal(a,b) for a,b in zip(before['cuda'],torch.cuda.get_rng_state_all())):
            raise ValueError('ASSESSMENT_MUTATED_CUDA_RNG')
        verdict = stopping.observe(metrics['TRAIN']['return_per_week'], metrics['VALIDATION']['return_per_week'])
        row = dict(epoch=epoch,steps=state['steps'],updates=state['updates'],decisions=decisions,
            training_seconds=training_seconds,evaluation_seconds=time.monotonic()-start-training_seconds,
            metrics=metrics,monitor=verdict,finished_at=now())
        if verdict['improved']:
            bundle = output/'bundles'/('epoch_%04d' % epoch)
            if bundle.exists():
                shutil.rmtree(bundle)  # Only the uncommitted epoch after interruption.
            receipt = export_bundle(bundle,model,probe,dict(epoch=epoch,score=verdict['score'],
                run_identity=identity,seed=seed))
            state['best_bundle'] = dict(relative_path=str(bundle.relative_to(output)),**receipt)
        state['history'].append(row); state.update(epoch=epoch,stopping=vars(stopping).copy())
        save_recovery(output,model,replay,rng,state)
        row['epoch_seconds'] = time.monotonic()-start
        write(output/'HISTORY.json',state['history'])
        write(output/'BEST.json',dict(identity=identity,best=state['best_bundle'],score=stopping.best))
        write(output/'PROGRESS.json',dict(identity=identity,stage='EPOCH_COMPLETE',**row))
        print(json.dumps(dict(epoch=epoch,seed=seed,monitor=verdict,seconds=row['epoch_seconds'])),flush=True)
        if verdict['stop']:
            break
    model.load(output/state['best_bundle']['relative_path']/'model.pt')
    write(output/'TRAINING.json',dict(identity=identity,epochs=state['epoch'],steps=state['steps'],
        stop_reason='patience' if stopping.bad>=stopping.patience else 'epoch_budget',
        best=state['best_bundle'],best_score=stopping.best,restored_best=True))
    return state


def worker(args):
    root = args.root.resolve(); spec = authenticate(root); d = spec['design']
    if args.seed not in d['seeds']:
        raise ValueError('REFIT_SEED')
    output = root/'runs'/('seed_%d' % args.seed); output.mkdir(parents=True,exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'RESULT.json').exists():
            result = read(output/'RESULT.json')
            if result['identity'] != spec['identity']:
                raise ValueError('COMPLETED_REFIT_IDENTITY')
            load_bundle(output/result['best_bundle']['relative_path'], 'cpu')
            return
        configure(args.seed); torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise ValueError('CUDA_REQUIRED_FOR_REAL_REFIT')
        market = market_for(root,spec); risk=read(root/'RISK.json')
        if support_hash(market.inputs) != read(root/'PREFLIGHT.json')['support_sha256']:
            raise ValueError('REFIT_SUPPORT_CHANGED')
        model = build_model(market.recipe,spec['effective'],args.seed,'cuda')
        train = market.recipe['split']['periods']['TRAIN']
        ix = np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
        probe = market.inputs.windows[ix[np.linspace(0,len(ix)-1,32,dtype=int)]].copy()
        write(output/'EFFECTIVE.json',dict(identity=spec['identity'],effective=model.effective,
            feature_count=market.recipe['list']['n'],groups=len(market.recipe['representation']['resolved']['names']),
            parameters=sum(p.numel() for p in model.loss.parameters()),seed=args.seed,
            gpu=torch.cuda.get_device_name(),input_shape=list(probe.shape[1:])))
        state = fit_epochs(model,lambda:environment(market,risk,d['horizon'],train,model.effective['gamma']),
            lambda m:assess_full(m,market,risk,d),d,args.seed,output,spec['identity'],probe)
        metrics = assess_full(model,market,risk,d,save=output/'final',diagnose=True)
        write(output/'RESULT.json',dict(identity=spec['identity'],seed=args.seed,
            best_bundle=state['best_bundle'],metrics=metrics,finished_at=now(),
            activity_admissible=all(m['orders_per_week']>2 for m in metrics.values()),
            scope='fresh_structural_108_development_validation_2024_TEST_closed'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ('root','parent','selection','baseline'):
        p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('worker'); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    args=parser.parse_args(); globals()[args.command](args)


if __name__ == '__main__':
    main()
