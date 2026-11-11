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
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, LazyMemmapStorage

from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.recipe_runtime import RecipeMarket
from app.rl_subset import digest, seal_recipe, subset_recipe, refit_train_scaler, project_inputs, support_hash, coherent_blocks
from app.structural_refit import (EpochPatience, capture_rng, restore_rng, build_model, verify_files,
    export_bundle, load_bundle, save_recovery, load_recovery)
from app.training_diagnostics import render_diagnostics
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
    'tools/run_three_signal_baseline.py', 'agent_plugins/temporal_features.py',
    'agent_plugins/causal_patch_attention.py', 'agent_plugins/sac_critic_regularization.py',
    'app/training_diagnostics.py','app/multiscale_context.py','agent_plugins/multiscale_context.py']
SOFTWARE = ['torch', 'torchrl', 'tensordict', 'numpy', 'pandas', 'pyarrow']


def market_for(root, spec):
    paths = {k:str(root/v) for k,v in spec['inputs'].items()}
    paths['recipe'] = str(root/'SUPPORT_RECIPE.json')
    market = RecipeMarket(dict(inputs=paths, engine_root=str(root/'engine'),
        recipe_sha256=spec['files']['SUPPORT_RECIPE.json']))
    rec = read(root/'RECIPE.json')
    market.inputs = project_inputs(market.inputs, rec)
    context=spec['effective']['architecture'].get('long_context')
    if context is not None:
        from app.multiscale_context import pack_context
        market.inputs=pack_context(market.inputs,context)
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
    mode = getattr(args,'monitor_mode','mean')
    seeds = getattr(args,'seeds',[7,19])
    if mode not in EpochPatience.MODES or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('REFIT_MONITOR_OR_SEEDS')
    design.update(max_epochs=2000, patience=20, min_delta=1e-8, seeds=seeds,
        monitor_mode=mode,monitor_start_epoch=getattr(args,'monitor_start_epoch',1),
        epoch='one_complete_TRAIN_traversal',
        monitor=mode+'_full_accounts_return_per_week',
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


def authenticate(root, source_root=None):
    spec = read(root/'SPEC.json')
    if spec['identity'] != digest({k:v for k,v in spec.items() if k != 'identity'}):
        raise ValueError('REFIT_SPEC_IDENTITY')
    for name,sha in spec['files'].items():
        if file_sha(root/name) != sha:
            raise ValueError('REFIT_INPUT:' + name)
    for name,sha in spec['sources'].items():
        if file_sha((source_root or ROOT)/name) != sha:
            raise ValueError('REFIT_SOURCE:' + name)
    for name,version in spec['software'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('REFIT_SOFTWARE:' + name)
    preflight = read(root/'PREFLIGHT.json')
    if preflight['identity'] != spec['identity'] or not preflight['controls']['passed']:
        raise ValueError('REFIT_PREFLIGHT')
    return spec


def selected_recipe_for_round(parent_recipe, selection_spec, result, close):
    identity = selection_spec['identity']
    if (identity != digest({k:v for k,v in selection_spec.items() if k != 'identity'})
            or result['identity'] != identity or close['identity'] != identity
            or close['termination'] != 'TWO_PASSES_COMPLETE'
            or result['result']['termination'] != 'TWO_PASSES_COMPLETE'
            or not result['original_support_preserved'] or not result['unmasked_after_pass_exact']
            or result['test_read'] or not result['final']['eligible']):
        raise ValueError('SELECTED_REFIT_RECEIPT')
    columns = parent_recipe['list']['columns']
    if (selection_spec['dataset']['features'] != columns
            or selection_spec['dataset']['scaler_hash'] != digest(parent_recipe['scaler'])):
        raise ValueError('SELECTED_REFIT_PARENT_RECIPE')
    selected = result['result']['selected']
    keep = set(selected)
    if any(keep.intersection(b) and not set(b) <= keep for b in coherent_blocks(columns)):
        raise ValueError('SELECTED_REFIT_PARTIAL_ENCODING')
    return subset_recipe(parent_recipe, selected)


def prepare_selected(args):
    source, root = args.source_root.resolve(), args.root.resolve()
    if root.exists():
        raise ValueError('NEW_REFIT_ROOT_REQUIRED')
    original = authenticate(source,args.source_runtime)
    selection = args.selection.resolve()
    ss, result, close = [read(selection/n) for n in ('SPEC.private.json','RESULT.json','CLOSE.json')]
    if (file_sha(selection/'RESULT.json') != close['result_sha256']
            or file_sha(source/'SPEC.json') != ss['parent_spec_sha256']
            or file_sha(args.selection_runtime/'tools/run_refit_mask_pass.py') != ss['source_sha256']):
        raise ValueError('SELECTED_REFIT_LINEAGE')
    run = source/'runs'/('seed_%d' % ss['seed'])
    for name,key in (('RESULT.json','parent_result_sha256'),('TRAINING.json','parent_training_sha256')):
        if file_sha(run/name) != ss[key]:
            raise ValueError('SELECTED_REFIT_PARENT_RESULT')
    parent_result = read(run/'RESULT.json')
    if (parent_result['identity'] != original['identity'] or
            parent_result['best_bundle'] != ss['best_bundle'] or
            result['parent_bundle'] != ss['best_bundle']):
        raise ValueError('SELECTED_REFIT_PARENT_BUNDLE')
    reduced = selected_recipe_for_round(read(source/'RECIPE.json'),ss,result,close)
    preflight = read(source/'PREFLIGHT.json')
    if preflight['support_sha256'] != ss['dataset']['support_hash']:
        raise ValueError('SELECTED_REFIT_SUPPORT_LINEAGE')
    root.mkdir()
    for name in original['files']:
        dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,dest)
    write(root/'RECIPE.json',reduced)
    for src,dst in (('RESULT.json','SELECTION_RESULT.json'),('CLOSE.json','SELECTION_CLOSE.json'),
                    ('SPEC.private.json','SELECTION_SPEC.private.json')):
        shutil.copy2(selection/src,root/dst)
    spec=copy.deepcopy(original)
    spec.pop('identity',None); spec.pop('warmup_migration',None)
    spec['created_at']=now()
    spec['design'].update(seeds=args.seeds,monitor_mode='validation',monitor_start_epoch=20,
        monitor='validation_full_accounts_return_per_week',max_epochs=2000,patience=20,min_delta=1e-8,
        select_during_warmup=True,replay_backend='memmap',require_activity=True)
    if not args.seeds or len(set(args.seeds))!=len(args.seeds):
        raise ValueError('SELECTED_REFIT_SEEDS')
    spec['sources']={n:file_sha(ROOT/n) for n in SOURCES+['tools/run_generalization_queue.py']}
    spec['selection_identity']=ss['identity']
    spec['parent_recipe_identity']=read(source/'RECIPE.json')['identity']
    spec['structural_round']=dict(variant='STRUCTURAL_SELECTED',parent_identity=original['identity'],
        selection_result_sha256=close['result_sha256'],features_before=len(ss['dataset']['features']),
        features_after=reduced['list']['n'],fresh_weights=True,fresh_replay=True,
        scaler='projected unchanged from parent TRAIN constants',
        checkpoint_source='new initialization, not the masked winner weights')
    write(root/'REFERENCE.json',dict(label='R1 masked selected inputs; frozen weights',
        matched_monitor=dict(selected=dict(epoch=ss['checkpoint_epoch'],metrics=result['final']['metrics'])),
        original_result=parent_result,selection_identity=ss['identity'],
        scope='comparison to frozen masked selection, not a trained reduced-input model'))
    spec['files']={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()}
    spec['identity']=digest(spec)
    write(root/'SPEC.json',spec)
    market=market_for(root,spec)
    if support_hash(market.inputs)!=preflight['support_sha256']:
        raise ValueError('SELECTED_REFIT_SUPPORT_CHANGED')
    for period,(lo,hi) in reduced['split']['periods'].items():
        chosen=market.inputs.period==period
        stamps=pd.to_datetime(market.inputs.row_ts_ns[chosen],utc=True)
        expected=list(pd.period_range(lo,pd.Timestamp(hi)-pd.Timedelta(days=1),freq='M').astype(str))
        if sorted(set(stamps.strftime('%Y-%m')))!=expected:
            raise ValueError('SELECTED_REFIT_MISSING_MONTH:'+period)
        env=environment(market,read(root/'RISK.json'),spec['design']['horizon'],[lo,hi],spec['effective']['gamma'])
        expected_counts=preflight['periods'][period]
        if (int(chosen.sum())!=expected_counts['rows'] or
                int((chosen & market.inputs.valid).sum())!=expected_counts['valid_windows'] or
                int(env.valid[env.engine.first:env.engine.last].sum())!=expected_counts['decision_count']):
            raise ValueError('SELECTED_REFIT_ROW_COUNTS')
    controls=synthetic_controls(reduced,read(root/'RISK.json'),dict(gamma=spec['effective']['gamma']))
    if not controls['passed']:
        raise ValueError('SELECTED_REFIT_NATIVE_CONTROLS')
    preflight.pop('warmup_migration',None); preflight.pop('continuation_of_fresh_fit',None)
    preflight.update(identity=spec['identity'],features=reduced['list']['n'],
        groups=len(reduced['representation']['resolved']['names']),
        tensor_shape=list(market.inputs.windows.shape),controls=controls,
        fresh_training=True,structural_round=spec['structural_round'])
    write(root/'PREFLIGHT.json',preflight)
    print(json.dumps(preflight),flush=True)


def compact_selected_recipe(source_recipe, selection, close, config):
    if (selection['identity']!=close['identity'] or close['termination']!='TWO_PASSES_COMPLETE'
            or selection['result']['termination']!='TWO_PASSES_COMPLETE'
            or selection['test_read'] or selection['training']
            or not selection['source_unchanged'] or not selection['original_support_preserved']
            or not selection['full_reload_exact'] or not selection['compact_parity']['qualified']
            or not selection['final']['eligible']
            or not all(all(v.values()) for v in selection['ledgers'].values())
            or config['source_recipe']!=source_recipe
            or config['lineage']['selection_identity']!=selection['identity']):
        raise ValueError('COMPACT_REFIT_RECEIPT')
    selected=selection['result']['selected']
    for block in coherent_blocks(source_recipe['list']['columns']):
        if set(selected).intersection(block) and not set(block)<=set(selected):
            raise ValueError('COMPACT_REFIT_PARTIAL_ENCODING')
    reduced=subset_recipe(source_recipe,selected)
    if reduced!=config['input_recipe']: raise ValueError('COMPACT_REFIT_RECIPE')
    return reduced


def prepare_compact(args):
    root,source,selection=args.root.resolve(),args.source_root.resolve(),args.selection.resolve()
    if root.exists(): raise ValueError('NEW_REFIT_ROOT_REQUIRED')
    original=authenticate(source,args.source_runtime)
    ss,result,close,config=[read(selection/n) for n in
        ('SPEC.private.json','RESULT.json','CLOSE.json','bundle/CONFIG.json')]
    verify_files(selection,close['files']); verify_files(args.selection_runtime,ss['sources'])
    if (ss['identity']!=digest({k:v for k,v in ss.items() if k!='identity'})
            or file_sha(selection/'RESULT.json')!=close['result_sha256']
            or file_sha(source/'SPEC.json')!=ss['parent_spec_sha256']
            or file_sha(source/'runs'/f"seed_{ss['seed']}"/'RESULT.json')!=ss['parent_result_sha256']):
        raise ValueError('COMPACT_REFIT_LINEAGE')
    reduced=compact_selected_recipe(read(source/'RECIPE.json'),result,close,config)
    preflight=read(source/'PREFLIGHT.json')
    if preflight['support_sha256']!=ss['dataset']['support_hash']: raise ValueError('COMPACT_REFIT_SUPPORT')
    root.mkdir()
    for name in original['files']:
        dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source/name,dest)
    write(root/'RECIPE.json',reduced)
    for name in ('RESULT.json','CLOSE.json','SPEC.private.json'):
        shutil.copy2(selection/name,root/('SELECTION_'+name))
    spec=copy.deepcopy(original); spec.pop('identity',None); spec.pop('warmup_migration',None)
    spec['created_at']=now()
    spec['design'].update(seeds=[7],monitor_mode='validation',monitor_start_epoch=20,
        monitor='validation_full_accounts_return_per_week',max_epochs=2000,patience=20,min_delta=1e-8)
    spec['effective']['architecture'].pop('temporal_attention',None)
    if args.temporal_attention=='causal_patch':
        spec['effective']['architecture']['temporal_attention']=dict(kind='causal_patch',
            patch_len=3,embedding=8,heads=2,feedforward=16,gate_init=.1)
    spec['sources']={n:file_sha(ROOT/n) for n in SOURCES}
    spec['selection_identity']=ss['identity']
    spec['structural_round']=dict(parent_identity=original['identity'],selection_result_sha256=close['result_sha256'],
        features_before=len(ss['dataset']['features']),features_after=reduced['list']['n'],
        fresh_weights=True,fresh_replay=True,variant=args.temporal_attention,
        scaler='projected unchanged',checkpoint_source='fresh initialization; compact reference preserved')
    spec['files']={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()}
    spec['identity']=digest(spec); write(root/'SPEC.json',spec)
    market=market_for(root,spec)
    if support_hash(market.inputs)!=preflight['support_sha256']: raise ValueError('COMPACT_REFIT_SUPPORT_CHANGED')
    for period,(lo,hi) in reduced['split']['periods'].items():
        chosen=market.inputs.period==period; stamps=pd.to_datetime(market.inputs.row_ts_ns[chosen],utc=True)
        expected=list(pd.period_range(lo,pd.Timestamp(hi)-pd.Timedelta(days=1),freq='M').astype(str))
        if sorted(set(stamps.strftime('%Y-%m')))!=expected: raise ValueError('COMPACT_REFIT_MONTHS')
        env=environment(market,read(root/'RISK.json'),spec['design']['horizon'],[lo,hi],spec['effective']['gamma'])
        old=preflight['periods'][period]
        if (int(chosen.sum())!=old['rows'] or int((chosen&market.inputs.valid).sum())!=old['valid_windows']
                or int(env.valid[env.engine.first:env.engine.last].sum())!=old['decision_count']):
            raise ValueError('COMPACT_REFIT_ROWS')
    controls=synthetic_controls(reduced,read(root/'RISK.json'),dict(gamma=spec['effective']['gamma']))
    if not controls['passed']: raise ValueError('COMPACT_REFIT_NATIVE_CONTROLS')
    preflight.pop('warmup_migration',None)
    preflight.update(identity=spec['identity'],features=reduced['list']['n'],
        groups=len(reduced['representation']['resolved']['names']),tensor_shape=list(market.inputs.windows.shape),
        controls=controls,fresh_training=True,structural_round=spec['structural_round'])
    write(root/'PREFLIGHT.json',preflight); print(json.dumps(preflight),flush=True)


def prepare_regularized(args):
    root, source = args.root.resolve(), args.source_root.resolve()
    if root.exists(): raise ValueError('NEW_REFIT_ROOT_REQUIRED')
    original = authenticate(source, args.source_runtime)
    result = read(source/'runs/seed_7/RESULT.json')
    if result['identity'] != original['identity']:
        raise ValueError('REGULARIZATION_PARENT_RESULT')
    if original['effective']['architecture'].get('temporal_attention',{}).get('kind') != 'causal_patch':
        raise ValueError('ATTENTION_PARENT_REQUIRED')
    root.mkdir()
    for name in original['files']:
        dest = root/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,dest)
    spec = copy.deepcopy(original); spec.pop('identity',None); spec.pop('warmup_migration',None)
    spec['created_at'] = now()
    config = dict(kind=args.profile)
    if 'dropout' in args.profile: config['p'] = .1
    model = build_model(read(root/'RECIPE.json'), original['effective'] | {'critic_regularization':config})
    spec['effective'] = model.effective
    spec['design'].update(select_during_warmup=True, monitor_start_epoch=20,
        diagnostics=True, seeds=[7], monitor_mode='validation', patience=20, min_delta=1e-8,
        max_epochs=2000)
    spec['structural_round'] = dict(parent_identity=original['identity'],
        parent_result_sha256=file_sha(source/'runs/seed_7/RESULT.json'),
        variant=args.profile,fresh_weights=True,fresh_replay=True,
        scaler='unchanged',checkpoint_contract='best from epoch1; patience counted from epoch21')
    spec['sources'] = {n:file_sha(ROOT/n) for n in SOURCES}
    spec['files'] = {str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()}
    spec['identity'] = digest(spec); write(root/'SPEC.json',spec)
    market = market_for(root,spec); preflight = read(source/'PREFLIGHT.json')
    if support_hash(market.inputs) != preflight['support_sha256']:
        raise ValueError('REGULARIZATION_SUPPORT_CHANGED')
    controls = synthetic_controls(market.recipe,read(root/'RISK.json'),dict(gamma=spec['effective']['gamma']))
    if not controls['passed']: raise ValueError('REGULARIZATION_NATIVE_CONTROLS')
    preflight.update(identity=spec['identity'],controls=controls,structural_round=spec['structural_round'])
    write(root/'PREFLIGHT.json',preflight)
    print(json.dumps(preflight),flush=True)


def assess_full(model, market, risk, design, save=None, diagnose=False):
    logits,_ = emit(model,market.inputs)
    prepared = prepared_signals(logits)
    metrics = {p:evaluate(environment(market,risk,design['horizon'],interval,model.effective['gamma']),
        logits, save=save/p if save else None, diagnose=diagnose, prepared=prepared)
        for p,interval in market.recipe['split']['periods'].items()}
    return metrics


def rebase_warmup_state(original, design, identity):
    """Replay only monitor bookkeeping; never update weights, optimizer or RNG."""
    start = design['monitor_start_epoch']
    if not 1 <= original['epoch'] < start <= design['max_epochs']:
        raise ValueError('WARMUP_MIGRATION_REQUIRES_PREMONITOR_PREFIX')
    if [r['epoch'] for r in original['history']] != list(range(1,original['epoch']+1)):
        raise ValueError('WARMUP_MIGRATION_HISTORY')
    state = copy.deepcopy(original)
    stopping = EpochPatience(design['patience'],design['min_delta'],design['monitor_mode'],start)
    shadows = {m:EpochPatience(design['patience'],design['min_delta'],m,start) for m in EpochPatience.MODES}
    for row in state['history']:
        row['source_monitor'] = row['monitor']
        row['source_shadow_monitors'] = row.get('shadow_monitors',{})
        t,v = (row['metrics'][p]['return_per_week'] for p in ('TRAIN','VALIDATION'))
        row['monitor'] = stopping.observe(t,v)
        row['shadow_monitors'] = {m:p.observe(t,v) for m,p in shadows.items()}
    state.update(identity=identity,best_bundle=None,stopping=vars(stopping).copy(),
                 shadows={m:vars(p).copy() for m,p in shadows.items()})
    return state


def migrate_warmup(args):
    source, root = args.source_root.resolve(), args.root.resolve()
    original = read(source/'SPEC.json')
    if original['identity'] != digest({k:v for k,v in original.items() if k != 'identity'}):
        raise ValueError('WARMUP_SOURCE_IDENTITY')
    verify_files(source,original['files'])
    verify_files(args.source_runtime,original['sources'])
    if args.seed not in original['design']['seeds'] or root.exists():
        raise ValueError('WARMUP_SEED_OR_NEW_ROOT')
    run = source/'runs'/('seed_%d' % args.seed)
    with (run/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pointer = read(run/'RESUME.json')
        folder = run/'recovery'/pointer['directory']
        if folder.parent.resolve() != (run/'recovery').resolve():
            raise ValueError('WARMUP_SOURCE_PATH')
        if file_sha(folder/'MANIFEST.json') != pointer['manifest_sha256']:
            raise ValueError('WARMUP_SOURCE_MANIFEST')
        manifest = read(folder/'MANIFEST.json')
        if manifest['identity'] != original['identity']:
            raise ValueError('WARMUP_RECOVERY_IDENTITY')
        verify_files(folder,manifest['files'])
        prior = read(folder/'STATE.json')
        if prior['identity'] != original['identity'] or prior['seed'] != args.seed:
            raise ValueError('WARMUP_SOURCE_STATE')
        spec = copy.deepcopy(original)
        spec['design']['monitor_start_epoch'] = args.monitor_start_epoch
        spec['sources'] = {n:file_sha(ROOT/n) for n in SOURCES}
        spec['created_at'] = now()
        spec['warmup_migration'] = dict(source_identity=original['identity'],seed=args.seed,
            completed_epoch=prior['epoch'],steps=prior['steps'],updates=prior['updates'],
            source_recovery_manifest_sha256=pointer['manifest_sha256'],
            model_sha256=manifest['files']['model.pt'],learning_sha256=manifest['files']['learning.pt'],
            change='monitor bookkeeping only; full learning state copied byte-for-byte')
        spec['identity'] = digest({k:v for k,v in spec.items() if k != 'identity'})
        state = rebase_warmup_state(prior,spec['design'],spec['identity'])
        root.mkdir()
        for name in original['files']:
            dest = root/name; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,dest)
        write(root/'SPEC.json',spec)
        preflight = read(source/'PREFLIGHT.json')
        preflight.update(identity=spec['identity'],fresh_training=False,
                         continuation_of_fresh_fit=True,warmup_migration=spec['warmup_migration'])
        write(root/'PREFLIGHT.json',preflight)
        out = root/'runs'/('seed_%d' % args.seed)
        dest = out/'recovery'/folder.name; dest.mkdir(parents=True)
        for name in ('model.pt','learning.pt'):
            shutil.copy2(folder/name,dest/name)
            if file_sha(dest/name) != manifest['files'][name]:
                raise ValueError('WARMUP_COPY_CHANGED')
        write(dest/'STATE.json',state)
        files = {p.name:file_sha(p) for p in dest.iterdir() if p.is_file()}
        write(dest/'MANIFEST.json',dict(identity=spec['identity'],files=files))
        from app.structural_refit import durable_files
        durable_files(dest)
        write(out/'RESUME.json',dict(directory=dest.name,manifest_sha256=file_sha(dest/'MANIFEST.json')))
        write(out/'HISTORY.json',state['history'])
        write(out/'BEST.json',dict(identity=spec['identity'],best=None,score=None))
        write(out/'MIGRATION.json',spec['warmup_migration'])
        write(out/'PROGRESS.json',dict(identity=spec['identity'],stage='MONITOR_WARMUP',
            completed_epochs=state['epoch'],bad=0,best=None,monitor_start_epoch=args.monitor_start_epoch))
        durable_files(out)
        print(json.dumps(dict(identity=spec['identity'],**spec['warmup_migration'])),flush=True)


def fit_epochs(model, env_factory, assess, design, seed, output, identity, probe):
    output.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(seed)
    backend=design.get('replay_backend','memory')
    if backend not in ('memory','memmap'): raise ValueError('REPLAY_BACKEND')
    storage=(LazyMemmapStorage(design['replay_size'],scratch_dir=str(output/'replay_live'),
                              existsok=True,auto_cleanup=False) if backend=='memmap'
             else LazyTensorStorage(design['replay_size'],device='cpu'))
    replay = TensorDictReplayBuffer(storage=storage,
                                   batch_size=design['batch_size'])
    start_epoch = design.get('monitor_start_epoch',1)
    select_early = design.get('select_during_warmup',False)
    stopping = EpochPatience(design['patience'],design['min_delta'],design.get('monitor_mode','mean'),start_epoch,select_early)
    shadows = {mode:EpochPatience(design['patience'],design['min_delta'],mode,start_epoch,select_early) for mode in EpochPatience.MODES}
    state = dict(identity=identity,seed=seed,epoch=0,steps=0,updates=0,history=[],
                 best_bundle=None,stopping=vars(stopping).copy(),
                 shadows={k:vars(v).copy() for k,v in shadows.items()})
    if (output/'RESUME.json').exists():
        state = load_recovery(output,model,replay,rng,identity)
        stopping.__dict__.update(state['stopping'])
        for mode,monitor in shadows.items():
            monitor.__dict__.update(state['shadows'][mode])
    elif design.get('initialize_epoch_zero',False):
        metrics=assess(model)
        if not all(metrics[p]['orders_per_week']>2 for p in ('TRAIN','VALIDATION')):
            raise ValueError('EPOCH_ZERO_ACTIVITY')
        for monitor in [stopping,*shadows.values()]:
            monitor.observe(metrics['TRAIN']['return_per_week'],metrics['VALIDATION']['return_per_week'])
            monitor.epoch=0
        bundle=output/'bundles'/'epoch_0000'
        if bundle.exists():
            restored,manifest=load_bundle(bundle,str(model.device))
            if (manifest['checkpoint']['run_identity']!=identity or
                    any(not torch.equal(v,restored.loss.state_dict()[k]) for k,v in model.loss.state_dict().items())):
                raise ValueError('EPOCH_ZERO_RESTART_CHANGED')
            receipt=dict(path=bundle.name,manifest_sha256=file_sha(bundle/'MANIFEST.json'),
                         model_sha256=file_sha(bundle/'model.pt'),reload_exact=True)
        else:
            receipt=export_bundle(bundle,model,probe,dict(epoch=0,score=stopping.best,run_identity=identity,seed=seed))
        state.update(best_bundle=dict(relative_path=str(bundle.relative_to(output)),**receipt),
                     stopping=vars(stopping).copy(),shadows={k:vars(m).copy() for k,m in shadows.items()})
        write(output/'EPOCH_ZERO.json',dict(identity=identity,metrics=metrics,best_bundle=state['best_bundle']))
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
            if step <= design['warmup'] and not design.get('warmup_policy',False):
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
        if design.get('head_only',False):
            from app.selected_warm_start import verify_frozen
            verify_frozen(model)
        write(output/'PROGRESS.json',dict(identity=identity,stage='FULL_TRAIN_VALIDATION',epoch=epoch,
                                         training_seconds=training_seconds))
        before = capture_rng(rng); replay_count = len(replay)
        metrics = assess(model)
        if not torch.equal(before['torch'],torch.get_rng_state()) or replay_count != len(replay):
            raise ValueError('ASSESSMENT_MUTATED_TRAINING')
        if before['cuda'] and any(not torch.equal(a,b) for a,b in zip(before['cuda'],torch.cuda.get_rng_state_all())):
            raise ValueError('ASSESSMENT_MUTATED_CUDA_RNG')
        eligible=(not design.get('require_activity',False) or
                  all(metrics[p]['orders_per_week']>2 for p in ('TRAIN','VALIDATION')))
        verdict = stopping.observe(metrics['TRAIN']['return_per_week'], metrics['VALIDATION']['return_per_week'],eligible)
        shadow_verdicts = {}
        for mode,monitor in shadows.items():
            if monitor.bad < monitor.patience:
                shadow_verdicts[mode] = monitor.observe(metrics['TRAIN']['return_per_week'],
                                                      metrics['VALIDATION']['return_per_week'],eligible)
            else:
                shadow_verdicts[mode] = dict(stop=True,already_stopped=True,best=monitor.best,bad=monitor.bad)
        row = dict(epoch=epoch,steps=state['steps'],updates=state['updates'],decisions=decisions,
            training_seconds=training_seconds,evaluation_seconds=time.monotonic()-start-training_seconds,
            metrics=metrics,monitor=verdict,shadow_monitors=shadow_verdicts,finished_at=now(),last_update=last_loss or {})
        if verdict['improved']:
            bundle = output/'bundles'/('epoch_%04d' % epoch)
            if bundle.exists():
                shutil.rmtree(bundle)  # Only the uncommitted epoch after interruption.
            receipt = export_bundle(bundle,model,probe,dict(epoch=epoch,score=verdict['score'],
                run_identity=identity,seed=seed))
            state['best_bundle'] = dict(relative_path=str(bundle.relative_to(output)),**receipt)
        if select_early and epoch >= start_epoch and eligible:
            late = state.get('best_after_warmup')
            if late is None or verdict['score'] > late['score'] + design['min_delta']:
                bundle = output/'bundles'/('epoch_%04d' % epoch)
                if verdict['improved']:
                    receipt = state['best_bundle'].copy()
                else:
                    if bundle.exists(): shutil.rmtree(bundle)
                    receipt = export_bundle(bundle,model,probe,dict(epoch=epoch,score=verdict['score'],
                        run_identity=identity,seed=seed,diagnostic='best_after_warmup'))
                state['best_after_warmup'] = dict(relative_path=str(bundle.relative_to(output)),
                    **{k:v for k,v in receipt.items() if k!='relative_path'},score=verdict['score'],epoch=epoch)
        state['history'].append(row); state.update(epoch=epoch,stopping=vars(stopping).copy(),
            shadows={k:vars(v).copy() for k,v in shadows.items()})
        save_recovery(output,model,replay,rng,state)
        row['epoch_seconds'] = time.monotonic()-start
        write(output/'HISTORY.json',state['history'])
        write(output/'BEST.json',dict(identity=identity,best=state['best_bundle'],score=stopping.best))
        if select_early:
            write(output/'BEST_AFTER_WARMUP.json',dict(identity=identity,best=state.get('best_after_warmup')))
        if design.get('diagnostics',False):
            render_diagnostics(state['history'],output/'diagnostics',design)
        write(output/'PROGRESS.json',dict(identity=identity,stage='EPOCH_COMPLETE',**row))
        print(json.dumps(dict(epoch=epoch,seed=seed,monitor=verdict,seconds=row['epoch_seconds'])),flush=True)
        if verdict['stop']:
            break
    if state['best_bundle'] is None:
        write(output/'TRAINING.json',dict(identity=identity,epochs=state['epoch'],
            steps=state['steps'],stop_reason='epoch_budget_before_monitor',restored_best=False))
        return state
    if design.get('diagnostics',False):
        render_diagnostics(state['history'],output/'diagnostics',design)
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
            best_after_warmup=state.get('best_after_warmup'),
            activity_admissible=all(m['orders_per_week']>2 for m in metrics.values()),
            scope=f"fresh_structural_{market.recipe['list']['n']}_development_validation_2024_TEST_closed"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ('root','parent','selection','baseline'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--monitor-mode',choices=EpochPatience.MODES,default='mean')
    p.add_argument('--seeds',nargs='+',type=int,default=[7,19])
    p.add_argument('--monitor-start-epoch',type=int,default=1)
    p=sub.add_parser('prepare-selected')
    for name in ('source-root','source-runtime','selection','selection-runtime','root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[7])
    p=sub.add_parser('prepare-compact')
    for name in ('source-root','source-runtime','selection','selection-runtime','root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--temporal-attention',choices=['none','causal_patch'],required=True)
    p=sub.add_parser('migrate-warmup')
    for name in ('source-root','source-runtime','root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--monitor-start-epoch',type=int,default=20)
    p=sub.add_parser('prepare-regularized')
    for name in ('source-root','source-runtime','root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',choices=['none','core_dropout','head_dropout','head_batchnorm','core_first_batchnorm'],required=True)
    p=sub.add_parser('worker'); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    args=parser.parse_args(); globals()[args.command.replace('-','_')](args)


if __name__ == '__main__':
    main()
