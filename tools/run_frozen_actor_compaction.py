"""Sequential, no-fit actor compaction and native account parity diagnostic."""
import argparse
import copy
import fcntl
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from app.frozen_actor_compaction import (ConstantResponse, PackedActor, compact_actor,
    effective_actor, compare_logits, save_compact, load_compact)
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import masked_emit, prepared_signals
from app.rl_subset import project_inputs, support_hash, digest
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import emit, environment, evaluate
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import authenticate, market_for, selected_recipe_for_round, SOURCES

ROOT = Path(__file__).resolve().parents[1]


def completed_model(root, spec, device, seed):
    run = root/'runs'/f'seed_{seed}'
    result, training = read(run/'RESULT.json'), read(run/'TRAINING.json')
    best = result['best_bundle']
    if (result['identity'] != spec['identity'] or training['identity'] != spec['identity']
            or not training['restored_best'] or best != training['best']):
        raise ValueError('COMPACTION_SOURCE_NOT_COMPLETE')
    path = (run/best['relative_path']).resolve()
    if not path.is_relative_to(run.resolve()):
        raise ValueError('COMPACTION_BUNDLE_PATH')
    if (file_sha(path/'MANIFEST.json') != best['manifest_sha256']
            or file_sha(path/'model.pt') != best['model_sha256']):
        raise ValueError('COMPACTION_BUNDLE_IDENTITY')
    model, manifest = load_bundle(path, device)
    probe = np.load(path/'PROBE.npz')
    with torch.inference_mode():
        y = model.surfaces(torch.tensor(probe['inputs'], device=device)).cpu().numpy()
    if not np.array_equal(y, probe['logits']):
        raise ValueError('COMPACTION_SOURCE_PROBE')
    return model, result, best, path


def ledger_parity(reference, candidate):
    result = {}
    for period in ('TRAIN', 'VALIDATION'):
        a, b = reference/period, candidate/period
        equity_a, equity_b = np.load(a/'EQUITY.npz'), np.load(b/'EQUITY.npz')
        result[period] = dict(
            actions=read(a/'ACTIONS.json') == read(b/'ACTIONS.json'),
            trades=read(a/'TRADES.json') == read(b/'TRADES.json'),
            equity=all(np.array_equal(equity_a[k], equity_b[k]) for k in equity_a.files))
    return result


def activation_profile(actor, x, names):
    with torch.inference_mode():
        enc = actor.encoder
        branch_values = [b(x.transpose(1,2).index_select(1,getattr(enc,f'indices_{i}')))
                         for i,b in enumerate(enc.branches)]
        def stats(v):
            return dict(rms=float(v.square().mean().sqrt()), mean=float(v.mean()),
                std=float(v.std(unbiased=False)), max_abs=float(v.abs().max()),
                temporal_range=float((v.max(-1).values-v.min(-1).values).abs().max()))
        return dict(branches={n:stats(v) for n,v in zip(names,branch_values)},
                    core=stats(enc.sequence({'features':x})), logits=stats(actor.surface(x)))


def run(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'COMPLETE.json').exists():
            close = read(out/'COMPLETE.json'); verify_files(out, close['files'])
            print(dict(stage='ALREADY_COMPLETE', identity=close['identity']), flush=True)
            return
        if (out/'SPEC.private.json').exists():
            raise ValueError('INCOMPLETE_DIAGNOSTIC_USE_NEW_OUTPUT')
        configure(args.seed); torch.backends.cudnn.allow_tf32 = False
        started = time.monotonic()
        parent = authenticate(args.parent, args.parent_runtime)
        fresh_spec = authenticate(args.refit, args.refit_runtime)
        s, selection, close = [read(args.selection/n) for n in
                              ('SPEC.private.json','RESULT.json','CLOSE.json')]
        if (file_sha(args.selection/'RESULT.json') != close['result_sha256']
                or file_sha(args.parent/'SPEC.json') != s['parent_spec_sha256']
                or file_sha(args.selection_runtime/'tools/run_refit_mask_pass.py') != s['source_sha256']):
            raise ValueError('COMPACTION_SELECTION_LINEAGE')
        source_run = args.parent/'runs'/f'seed_{args.seed}'
        for name,key in (('RESULT.json','parent_result_sha256'),('TRAINING.json','parent_training_sha256')):
            if file_sha(source_run/name) != s[key]:
                raise ValueError('COMPACTION_PARENT_RESULT')
        projected = selected_recipe_for_round(read(args.parent/'RECIPE.json'), s, selection, close)
        if (projected != read(args.refit/'RECIPE.json') or parent['effective'] != fresh_spec['effective']
                or parent['design']['horizon'] != fresh_spec['design']['horizon']
                or read(args.parent/'RISK.json') != read(args.refit/'RISK.json')):
            raise ValueError('COMPACTION_REFIT_CONTRACT')
        for key in parent['inputs']:
            if parent['files'][parent['inputs'][key]] != fresh_spec['files'][fresh_spec['inputs'][key]]:
                raise ValueError('COMPACTION_DATA_CHANGED')
        market = market_for(args.parent, parent)
        inputs = project_inputs(market.inputs, projected)
        support = support_hash(inputs)
        if (support != s['dataset']['support_hash'] or
                support != read(args.refit/'PREFLIGHT.json')['support_sha256'] or
                market.recipe['split']['periods'] != {
                    'TRAIN':['2020-01-01','2024-01-01'], 'VALIDATION':['2024-01-01','2025-01-01']}):
            raise ValueError('COMPACTION_SUPPORT_OR_SPLIT')
        model, archived, best, source_bundle = completed_model(args.parent,parent,args.device,args.seed)
        fresh, fresh_archived, fresh_best, _ = completed_model(args.refit,fresh_spec,args.device,args.seed)
        if best != s['best_bundle'] or best != selection['parent_bundle']:
            raise ValueError('COMPACTION_SELECTED_CHECKPOINT')
        sources = sorted(set(SOURCES + ['app/frozen_actor_compaction.py',
            'tools/run_frozen_actor_compaction.py','agent_plugins/temporal_features.py',
            'tools/run_refit_mask_pass.py','app/suppressed_close_counterfactuals.py']))
        spec = dict(schema='frozen_actor.diagnostic.v1', created_at=now(),
            parent_identity=parent['identity'], refit_identity=fresh_spec['identity'],
            selection_identity=selection['identity'], selection_result_sha256=file_sha(args.selection/'RESULT.json'),
            checkpoint_sha256=best['model_sha256'], fresh_checkpoint_sha256=fresh_best['model_sha256'],
            support_hash=support, source_recipe_identity=market.recipe['identity'],
            input_recipe_identity=projected['identity'], device=args.device,
            torch=torch.__version__, tolerance=dict(atol=1e-5,rtol=1e-5),
            sources={name:file_sha(ROOT/name) for name in sources},
            training=False,test_read=False, source_manifest_sha256=file_sha(source_bundle/'MANIFEST.json'))
        spec['identity'] = digest(spec); write(out/'SPEC.private.json',spec)
        original_state = {k:v.clone() for k,v in model.loss.state_dict().items()}
        actor = effective_actor(model)
        compact, info = compact_actor(actor, market.recipe, projected['list']['columns'])
        take = [market.recipe['list']['columns'].index(c) for c in projected['list']['columns']]
        packed = PackedActor(actor,take,market.recipe['list']['n'])
        zeroed = copy.deepcopy(compact)
        for branch in zeroed.encoder.branches:
            if isinstance(branch,ConstantResponse): branch.response.zero_()
        risk = read(args.parent/'RISK.json'); valid = market.inputs.valid
        arms, logits_by_arm = {}, {}
        def assess(name, logits, expected=None):
            write(out/'PROGRESS.json',dict(stage=name, at=now(), seconds=time.monotonic()-started))
            prepared = prepared_signals(logits)
            metrics = {p:evaluate(environment(market,risk,parent['design']['horizon'],interval,
                parent['effective']['gamma']),logits,save=out/name/p,diagnose=True,prepared=prepared)
                for p,interval in market.recipe['split']['periods'].items()}
            if expected is not None:
                for p,m in metrics.items(): account_equal(m,expected[p])
            np.save(out/name/'LOGITS.npy',logits)
            arms[name] = metrics; logits_by_arm[name] = logits
            print(dict(arm=name,validation_return_pct=metrics['VALIDATION']['return_pct']),flush=True)
        ordinary,_ = emit(model,market.inputs)
        assess('unmasked108',ordinary,archived['metrics'])
        masked = masked_emit(model,market.inputs,[c in projected['list']['columns'] for c in market.recipe['list']['columns']])
        assess('masked108',masked,selection['final']['metrics'])
        for name, net in (('packed94',packed),('compact94',compact),('constant_zero94',zeroed)):
            logits,_ = emit(SimpleNamespace(surfaces=net.surface,device=model.device),inputs)
            assess(name,logits)
        fresh_logits,_ = emit(fresh,inputs)
        assess('fresh94',fresh_logits,fresh_archived['metrics'])
        parity = {name:compare_logits(masked[valid],logits_by_arm[name][valid])
                  for name in ('packed94','compact94','constant_zero94','fresh94')}
        ledgers = {name:ledger_parity(out/'masked108',out/name) for name in parity}
        unchanged = all(torch.equal(v,model.loss.state_dict()[k]) for k,v in original_state.items())
        if not unchanged or not parity['packed94']['bitwise_exact']:
            raise ValueError('COMPACTION_REFERENCE_OR_SOURCE_CHANGED')
        profiles = {}
        names = market.recipe['representation']['resolved']['names']
        for period in ('TRAIN','VALIDATION'):
            ix = np.flatnonzero(valid & (market.inputs.period==period))
            # Uniformly cover the period, rather than cherry-picking activations.
            ix = ix[np.linspace(0,len(ix)-1,min(256,len(ix)),dtype=int)]
            x = torch.tensor(inputs.windows[ix],device=model.device)
            full = x.new_zeros(len(ix),x.shape[1],market.recipe['list']['n'])
            full[:,:,take] = x
            profiles[period] = dict(rows=len(ix),
                masked=activation_profile(actor,full,names),
                compact=activation_profile(compact,x,names),
                fresh=activation_profile(effective_actor(fresh),x,projected['representation']['resolved']['names']))
        write(out/'ACTIVATIONS.json',profiles)
        ix = np.flatnonzero(valid)[:256]
        save_compact(out/'bundle',compact,market.recipe,projected,model.effective,
            dict(source_checkpoint_sha256=best['model_sha256'],diagnostic_identity=spec['identity']),
            inputs.windows[ix],info)
        restored,_ = load_compact(out/'bundle',args.device)
        replay,_ = emit(SimpleNamespace(surfaces=restored.surface,device=model.device),inputs)
        reload_exact = bool(np.array_equal(replay[valid],logits_by_arm['compact94'][valid]))
        if not reload_exact: raise ValueError('COMPACTION_FULL_RELOAD_NOT_EXACT')
        qualified = parity['compact94']['qualified'] and all(
            all(p.values()) for p in ledgers['compact94'].values())
        result = dict(identity=spec['identity'],finished_at=now(),seconds=time.monotonic()-started,
            arms=arms,parity=parity,ledgers=ledgers,graph=info,
            compact_inference_qualified=qualified,full_reload_exact=reload_exact,
            source_unchanged=unchanged,test_read=False,training=False,
            scope='Previously exposed development; no new generalization or trainable SAC claim')
        write(out/'RESULT.json',result)
        files = {str(p.relative_to(out)):file_sha(p) for p in out.rglob('*')
                 if p.is_file() and p.name not in ('run.lock','PROGRESS.json','COMPLETE.json')}
        write(out/'COMPLETE.json',dict(identity=spec['identity'],files=files,finished_at=now()))
        write(out/'PROGRESS.json',dict(stage='COMPLETE',at=now(),qualified=qualified))
        print(dict(qualified=qualified,parity=parity['compact94'],seconds=result['seconds']),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','parent-runtime','selection','selection-runtime','refit','refit-runtime','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(p.parse_args())


if __name__ == '__main__':
    main()
