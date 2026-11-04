"""One ordered removal/reinsertion round over an authenticated compact actor."""
import argparse
import fcntl
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from feature_selector.contracts import subset_view
from feature_selector.plugins import resolve
from app.frozen_actor_compaction import (load_compact, effective_actor, compact_actor,
    compare_logits, save_compact)
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import cached_measurement, masked_emit, prepared_signals
from app.rl_subset import digest, project_inputs, subset_recipe, support_hash
from app.structural_refit import verify_files
from app.three_signal_learning import emit, environment, evaluate
from tools.run_discrete_curve_probe import read, write, now, configure
from tools.run_refit_mask_pass import dataset_for, account_equal, score_metrics
from tools.run_frozen_actor_compaction import completed_model, ledger_parity
from tools.run_structural_sac_refit import authenticate, market_for, SOURCES

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SOURCES = sorted(set(SOURCES + ['tools/run_compact_mask_pass.py',
    'tools/run_refit_mask_pass.py','tools/run_frozen_actor_compaction.py',
    'app/frozen_actor_compaction.py','agent_plugins/temporal_features.py']))


def authenticated_compaction(root):
    close = read(root/'COMPLETE.json')
    required = {'RESULT.json','SPEC.private.json','bundle/CONFIG.json','bundle/MANIFEST.json',
                'bundle/actor.pt','bundle/PROBE.npz','bundle/RELOAD.json'}
    if not required <= set(close['files']):
        raise ValueError('COMPACT_PARENT_FILES')
    verify_files(root,close['files'])
    spec,result,config = [read(root/n) for n in ('SPEC.private.json','RESULT.json','bundle/CONFIG.json')]
    if (spec['identity'] != digest({k:v for k,v in spec.items() if k!='identity'})
            or close['identity'] != spec['identity'] or result['identity'] != spec['identity']
            or not result['compact_inference_qualified'] or not result['full_reload_exact']
            or not result['source_unchanged'] or result['test_read'] or result['training']
            or config['lineage']['diagnostic_identity'] != spec['identity']
            or config['lineage']['source_checkpoint_sha256'] != spec['checkpoint_sha256']):
        raise ValueError('COMPACT_PARENT_QUALIFICATION')
    return spec,result,config


def prepare(args):
    if args.root.exists(): raise ValueError('NEW_COMPACT_MASK_ROOT_REQUIRED')
    old,result,config = authenticated_compaction(args.compaction)
    parent = authenticate(args.parent,args.parent_runtime)
    source_result = read(args.parent/'runs'/f'seed_{args.seed}'/'RESULT.json')
    if (old['parent_identity'] != parent['identity']
            or source_result['best_bundle']['model_sha256'] != old['checkpoint_sha256']
            or config['source_recipe'] != read(args.parent/'RECIPE.json')):
        raise ValueError('COMPACT_MASK_LINEAGE')
    market = market_for(args.parent,parent)
    market.inputs = project_inputs(market.inputs,config['input_recipe'])
    market.recipe = config['input_recipe']
    dataset = dataset_for(market)
    if dataset['support_hash'] != old['support_hash']:
        raise ValueError('COMPACT_MASK_SUPPORT')
    _,plugin = resolve('selectors','two_pass_greedy')
    spec = dict(schema='compact_actor.mask_pass.v1',created_at=now(),
        parent=str(args.parent.resolve()),parent_runtime=str(args.parent_runtime.resolve()),
        compaction=str(args.compaction.resolve()),compaction_identity=old['identity'],
        parent_spec_sha256=file_sha(args.parent/'SPEC.json'),
        parent_result_sha256=file_sha(args.parent/'runs'/f'seed_{args.seed}'/'RESULT.json'),
        compaction_close_sha256=file_sha(args.compaction/'COMPLETE.json'),
        dataset=dataset,plugin=plugin,seed=args.seed,device=args.device,
        selector_params=dict(min_features=1,tolerance=0.,remove_on_tie=True),
        sources={n:file_sha(ROOT/n) for n in LOCAL_SOURCES},
        scope='DEVELOPMENT_COMPACT_FROZEN_TWO_PASS; no training or TEST',
        initial_validation_return_pct=result['arms']['compact94']['VALIDATION']['return_pct'])
    spec['identity']=digest(spec)
    args.root.mkdir(parents=True); write(args.root/'SPEC.private.json',spec)
    print(dict(identity=spec['identity'],features=len(dataset['features']),blocks=len(dataset['blocks'])),flush=True)


def run(args):
    root=args.root
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=read(root/'SPEC.private.json')
        if s['identity'] != digest({k:v for k,v in s.items() if k!='identity'}):
            raise ValueError('COMPACT_MASK_IDENTITY')
        for name,sha in s['sources'].items():
            if file_sha(ROOT/name)!=sha: raise ValueError('COMPACT_MASK_SOURCE:'+name)
        parent,previous=Path(s['parent']),Path(s['compaction'])
        for path,key in ((parent/'SPEC.json','parent_spec_sha256'),
                         (parent/'runs'/f"seed_{s['seed']}"/'RESULT.json','parent_result_sha256'),
                         (previous/'COMPLETE.json','compaction_close_sha256')):
            if file_sha(path)!=s[key]: raise ValueError('COMPACT_MASK_PARENT_CHANGED')
        old,archived,config=authenticated_compaction(previous)
        spec=authenticate(parent,Path(s['parent_runtime']))
        selector,plugin=resolve('selectors','two_pass_greedy')
        if plugin!=s['plugin']: raise ValueError('COMPACT_MASK_PLUGIN')
        configure(s['seed']); torch.backends.cudnn.allow_tf32=False
        if old['torch'] != torch.__version__: raise ValueError('COMPACT_MASK_SOFTWARE')
        actor,_=load_compact(previous/'bundle',s['device'])
        device=next(actor.parameters()).device
        model=SimpleNamespace(surfaces=actor.surface,device=device)
        state={k:v.clone() for k,v in actor.state_dict().items()}
        probe=np.load(previous/'bundle'/'PROBE.npz')
        with torch.inference_mode():
            logits=actor.surface(torch.tensor(probe['inputs'],device=device)).cpu().numpy()
        if not np.array_equal(logits,probe['logits']): raise ValueError('COMPACT_MASK_PROBE')
        market=market_for(parent,spec)
        market.inputs=project_inputs(market.inputs,config['input_recipe']); market.recipe=config['input_recipe']
        if dataset_for(market)!=s['dataset']: raise ValueError('COMPACT_MASK_DATASET')
        columns=s['dataset']['features']; risk=read(parent/'RISK.json')
        context=dict(identity=s['identity'],device=s['device'],
            device_name=torch.cuda.get_device_name() if device.type=='cuda' else 'cpu')
        started=time.monotonic(); times=[]
        def accounts(logits,save=None,diagnose=False):
            prepared=prepared_signals(logits)
            return {p:evaluate(environment(market,risk,spec['design']['horizon'],interval,
                spec['effective']['gamma']),logits,save=save/p if save else None,
                diagnose=diagnose,prepared=prepared) for p,interval in market.recipe['split']['periods'].items()}
        def assess(keep,save=None,diagnose=False):
            logits=masked_emit(model,market.inputs,[c in keep for c in columns])
            return score_metrics(accounts(logits,save,diagnose)),logits
        control,base_logits=assess(columns,root/'control',True)
        ordinary,_=emit(model,market.inputs); valid=market.inputs.valid
        if not np.array_equal(base_logits[valid],ordinary[valid]): raise ValueError('COMPACT_MASK_EMIT_PARITY')
        for p,m in control['metrics'].items(): account_equal(m,archived['arms']['compact94'][p])
        if not all(all(v.values()) for v in ledger_parity(previous/'compact94',root/'control').values()):
            raise ValueError('COMPACT_MASK_CONTROL_LEDGER')
        write(root/'PREFLIGHT.json',dict(passed=True,identity=s['identity'],probe_exact=True,
            control=control,support_hash=s['dataset']['support_hash'],test_read=False))
        if args.preflight_only: return
        def measure(keep):
            ordered=subset_view(s['dataset'],keep)['features']
            def fresh():
                begin=time.monotonic()
                value=control if ordered==columns else assess(ordered)[0]
                times.append(time.monotonic()-begin)
                write(root/'PROGRESS.json',dict(stage='EVALUATING',at=now(),features=len(ordered),
                    fresh_evaluations=len(times),median_seconds=float(np.median(times)),
                    elapsed_seconds=time.monotonic()-started))
                return value
            return cached_measurement(root/'cache',context,ordered,fresh)
        result=selector.select(s['dataset'],measure,s['selector_params'],
            progress=lambda r:write(root/'PASSES.json',dict(r,at=now())))
        if result['termination']!='TWO_PASSES_COMPLETE': raise ValueError('COMPACT_MASK_INCOMPLETE')
        selected=result['selected']; final,final_logits=assess(selected,root/'final',True)
        for p,m in final['metrics'].items(): account_equal(m,measure(selected)['metrics'][p])
        if final['score']<control['score'] or not final['eligible']: raise ValueError('COMPACT_MASK_REGRESSION')
        restored,restored_logits=assess(columns)
        if not np.array_equal(base_logits[valid],restored_logits[valid]): raise ValueError('COMPACT_MASK_SOURCE_DRIFT')
        if not all(torch.equal(v,actor.state_dict()[k]) for k,v in state.items()):
            raise ValueError('COMPACT_MASK_WEIGHTS_CHANGED')
        for p,m in restored['metrics'].items(): account_equal(m,control['metrics'][p])

        # Export from the original source graph; do not lose earlier constant
        # fusion slots by treating the compact input recipe as its architecture.
        source,_,best,_=completed_model(parent,spec,s['device'],s['seed'])
        if best['model_sha256']!=old['checkpoint_sha256']: raise ValueError('COMPACT_MASK_SOURCE_CHECKPOINT')
        compact,graph=compact_actor(effective_actor(source),config['source_recipe'],selected)
        final_recipe=subset_recipe(config['source_recipe'],selected)
        final_inputs=project_inputs(market.inputs,final_recipe)
        if support_hash(final_inputs)!=s['dataset']['support_hash']: raise ValueError('COMPACT_FINAL_SUPPORT')
        new_model=SimpleNamespace(surfaces=compact.surface,device=device)
        compact_logits,_=emit(new_model,final_inputs)
        parity=compare_logits(final_logits[valid],compact_logits[valid])
        compact_metrics=accounts(compact_logits,root/'compact_final',True)
        ledgers=ledger_parity(root/'final',root/'compact_final')
        if not parity['qualified'] or not all(all(v.values()) for v in ledgers.values()):
            raise ValueError('COMPACT_FINAL_PARITY')
        lineage=dict(source_checkpoint_sha256=best['model_sha256'],selection_identity=s['identity'])
        if not (root/'bundle').exists():
            save_compact(root/'bundle',compact,config['source_recipe'],final_recipe,source.effective,
                lineage,final_inputs.windows[np.flatnonzero(valid)[:256]],graph)
        else:
            cfg=read(root/'bundle'/'CONFIG.json')
            if cfg['lineage']!=lineage or cfg['input_recipe']!=final_recipe:
                raise ValueError('COMPACT_FINAL_BUNDLE_IDENTITY')
        reload,_=load_compact(root/'bundle',s['device'])
        replay,_=emit(SimpleNamespace(surfaces=reload.surface,device=device),final_inputs)
        if not np.array_equal(replay[valid],compact_logits[valid]): raise ValueError('COMPACT_FINAL_RELOAD')
        payload=dict(identity=s['identity'],result=result,control=control,final=final,
            removed=[c for c in columns if c not in selected],groups=subset_view(s['dataset'],selected)['groups'],
            graph=graph,compact_metrics=compact_metrics,compact_parity=parity,ledgers=ledgers,
            full_reload_exact=True,source_unchanged=True,original_support_preserved=True,
            test_read=False,training=False,scope=s['scope'])
        if (root/'CLOSE.json').exists():
            close=read(root/'CLOSE.json'); verify_files(root,close['files'])
            if read(root/'RESULT.json')!=payload: raise ValueError('COMPACT_COMPLETED_REPLAY_CHANGED')
        else:
            write(root/'RESULT.json',payload)
            write(root/'FEATURE_COUNTS.json',dict(**result['counts'],groups=len(payload['groups']),
                initial_groups=len(s['dataset']['groups']),phase1_admitted=120))
            sealed=['RESULT.json','FEATURE_COUNTS.json','SPEC.private.json']
            sealed += [str(p.relative_to(root)) for p in (root/'bundle').iterdir() if p.is_file()]
            write(root/'CLOSE.json',dict(identity=s['identity'],termination=result['termination'],
                finished_at=now(),seconds=time.monotonic()-started,evaluations=result['evaluations'],
                files={n:file_sha(root/n) for n in sealed},result_sha256=file_sha(root/'RESULT.json')))
        write(root/'PROGRESS.json',dict(stage='COMPLETE',at=now(),identity=s['identity'],counts=result['counts']))
        print(dict(counts=result['counts'],evaluations=result['evaluations'],
                   validation_return_pct=final['metrics']['VALIDATION']['return_pct']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for name in ('root','parent','parent-runtime','compaction'): a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--seed',type=int,default=7); a.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=sub.add_parser('run'); a.add_argument('--root',type=Path,required=True); a.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(); (prepare if args.command=='prepare' else run)(args)


if __name__=='__main__': main()
