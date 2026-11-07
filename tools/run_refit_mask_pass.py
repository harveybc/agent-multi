"""Two ordered mask passes over a completed structural SAC inference bundle."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from feature_selector.contracts import subset_view, validate_dataset
from feature_selector.plugins import resolve
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import cached_measurement, masked_emit, prepared_signals
from app.rl_subset import digest, support_hash, coherent_blocks, groups_for_selector
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment, evaluate
from tools.run_discrete_curve_probe import read, write, now, configure
from tools.run_structural_sac_refit import authenticate, market_for

ROOT = Path(__file__).resolve().parents[1]


def authenticated_parent(parent, seed):
    spec = authenticate(parent)
    out = parent/'runs'/f'seed_{seed}'
    result, training = read(out/'RESULT.json'), read(out/'TRAINING.json')
    best = result['best_bundle']
    if (result['identity'] != spec['identity'] or training['identity'] != spec['identity']
            or not training['restored_best'] or best != training['best']
            or not result['activity_admissible'] or result['seed'] != seed):
        raise ValueError('MASK_PARENT_NOT_COMPLETED')
    bundle = (out/best['relative_path']).resolve()
    if not bundle.is_relative_to(out.resolve()):
        raise ValueError('MASK_BUNDLE_PATH')
    if (file_sha(bundle/'MANIFEST.json') != best['manifest_sha256']
            or file_sha(bundle/'model.pt') != best['model_sha256']):
        raise ValueError('MASK_PARENT_BUNDLE_CHANGED')
    return spec, out, result, bundle


def dataset_for(market):
    rec = market.recipe
    return validate_dataset(dict(schema='feature_selector.dataset.v1',scope='development',
        features=rec['list']['columns'],groups=groups_for_selector(rec),
        blocks=coherent_blocks(rec['list']['columns']),support_hash=support_hash(market.inputs),
        scaler_hash=digest(rec['scaler']),representation=rec['representation'],
        splits=dict(train=rec['split']['periods']['TRAIN'],
                    selection=rec['split']['periods']['VALIDATION'],test_start='2025-01-01'),
        constraints=dict(agent_state='none',scaling='fixed_train',clipping=False,
                         fit_scope='train_only',test_access=False)))


def account_equal(actual, expected):
    ignored = {'mean_signal_entropy','mean_requested_command_entropy','early_close_diagnostic'}
    if {k:v for k,v in actual.items() if k not in ignored} != {
            k:v for k,v in expected.items() if k not in ignored}:
        raise ValueError('REFIT_MASK_ACCOUNT_PARITY')


def score_metrics(metrics):
    train, val = [float(metrics[p]['return_per_week']) for p in ('TRAIN','VALIDATION')]
    if not np.isfinite([train,val]).all():
        raise ValueError('REFIT_MASK_NONFINITE_SCORE')
    return dict(score=val,train_score=train,gap=abs(train-val),
        eligible=all(metrics[p]['orders_per_week']>2 for p in ('TRAIN','VALIDATION')),
        metrics=metrics,frozen_weights=True,fresh_training=False)


def verify_probe(actual, expected, cross_device=False):
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('MASK_PROBE_SHAPE_OR_FINITE')
    exact=np.array_equal(actual,expected)
    if not exact and not cross_device:
        raise ValueError('MASK_PROBE_NOT_EXACT')
    if not np.allclose(actual,expected,rtol=1e-5,atol=1e-6):
        raise ValueError('MASK_PROBE_TOLERANCE')
    if not np.array_equal(actual.argmax(-1),expected.argmax(-1)):
        raise ValueError('MASK_PROBE_ACTION')
    return dict(exact=exact,cross_device_allowed=cross_device,
                max_abs_error=float(np.max(np.abs(actual-expected))),
                rtol=1e-5,atol=1e-6,actions_exact=True)


def prepare(args):
    if args.root.exists():
        raise ValueError('NEW_MASK_ROOT_REQUIRED')
    parent = args.parent.resolve()
    spec, out, result, bundle = authenticated_parent(parent,args.seed)
    market = market_for(parent,spec)
    dataset = dataset_for(market)
    if dataset['support_hash'] != read(parent/'PREFLIGHT.json')['support_sha256']:
        raise ValueError('MASK_SUPPORT_CHANGED')
    _, plugin = resolve('selectors','two_pass_greedy')
    s = dict(schema='structural_sac.mask_pass.v1',created_at=now(),parent=str(parent),
        parent_spec_sha256=file_sha(parent/'SPEC.json'),seed=args.seed,
        parent_result_sha256=file_sha(out/'RESULT.json'),
        parent_training_sha256=file_sha(out/'TRAINING.json'),best_bundle=result['best_bundle'],
        dataset=dataset,plugin=plugin,device=args.device,
        cross_device_probe=getattr(args,'cross_device_probe',False),
        selector_params=dict(min_features=1,tolerance=0.,remove_on_tie=True),
        source_sha256=file_sha(Path(__file__)),
        checkpoint_epoch=read(bundle/'MANIFEST.json')['checkpoint']['epoch'],
        ranking='full VALIDATION net return/week; TRAIN reported; activity >2/week in both',
        scope='DEVELOPMENT_FROZEN_MASK; no new training or TEST; no structural promotion')
    s['identity']=digest(s)
    args.root.mkdir(parents=True)
    write(args.root/'SPEC.private.json',s)
    print(s['identity'],flush=True)


def run(args):
    root=args.root
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=read(root/'SPEC.private.json')
        if s['identity'] != digest({k:v for k,v in s.items() if k!='identity'}):
            raise ValueError('MASK_IDENTITY')
        if file_sha(Path(__file__)) != s['source_sha256']:
            raise ValueError('MASK_RUNNER_CHANGED')
        selector,plugin=resolve('selectors','two_pass_greedy')
        if plugin != s['plugin']:
            raise ValueError('MASK_SELECTOR_CHANGED')
        parent=Path(s['parent'])
        spec,out,archived,bundle=authenticated_parent(parent,s['seed'])
        for path,key in ((parent/'SPEC.json','parent_spec_sha256'),
                         (out/'RESULT.json','parent_result_sha256'),
                         (out/'TRAINING.json','parent_training_sha256')):
            if file_sha(path)!=s[key]:
                raise ValueError('MASK_PARENT_CHANGED')
        configure(s['seed']); torch.backends.cudnn.allow_tf32=False
        model,manifest=load_bundle(bundle,s['device'])
        if manifest['backend']['torch'] != torch.__version__:
            raise ValueError('MASK_TORCH_CHANGED')
        market=market_for(parent,spec)
        if dataset_for(market)!=s['dataset'] or model.recipe!=market.recipe:
            raise ValueError('MASK_DATASET_CHANGED')
        probe=np.load(bundle/'PROBE.npz')
        with torch.inference_mode():
            logits=model.surfaces(torch.tensor(probe['inputs'],device=model.device)).cpu().numpy()
        probe_check=verify_probe(logits,probe['logits'],s.get('cross_device_probe',False))
        columns=s['dataset']['features']; risk=read(parent/'RISK.json')
        context=dict(identity=s['identity'],device=s['device'],
            device_name=torch.cuda.get_device_name() if s['device']=='cuda' else 'cpu')
        started=time.monotonic(); times=[]

        def assess(keep,save=None,diagnose=False):
            logits=masked_emit(model,market.inputs,[c in keep for c in columns])
            prepared=prepared_signals(logits)
            metrics={p:evaluate(environment(market,risk,spec['design']['horizon'],interval,
                spec['effective']['gamma']),logits,prepared=prepared,diagnose=diagnose,
                save=save/p if save else None) for p,interval in market.recipe['split']['periods'].items()}
            return score_metrics(metrics),logits

        # Always replay the unmasked control, including after restart. Never let
        # a cached subset mask a changed backend, support or account convention.
        control,logits=assess(columns,diagnose=True)
        ordinary,_=emit(model,market.inputs)
        ix=market.inputs.valid
        if not np.array_equal(logits[ix],ordinary[ix]):
            raise ValueError('MASK_UNMASKED_EMISSION_PARITY')
        for p,m in control['metrics'].items():
            account_equal(m,archived['metrics'][p])
        write(root/'PREFLIGHT.json',dict(identity=s['identity'],passed=True,control=control,
            probe_exact=probe_check['exact'],probe_check=probe_check,
            unmasked_account_exact=True,features=len(columns),
            support_hash=s['dataset']['support_hash'],test_read=False))
        if args.preflight_only:
            return

        def measure(keep):
            ordered=subset_view(s['dataset'],keep)['features']
            def fresh():
                begin=time.monotonic()
                value=control if ordered==columns else assess(ordered)[0]
                elapsed=time.monotonic()-begin; times.append(elapsed)
                write(root/'PROGRESS.json',dict(stage='EVALUATING',features=len(ordered),
                    fresh_evaluations=len(times),seconds=time.monotonic()-started,
                    last_seconds=elapsed,median_seconds=float(np.median(times)),at=now()))
                return value
            return cached_measurement(root/'cache',context,ordered,fresh)

        result=selector.select(s['dataset'],measure,s['selector_params'],
            progress=lambda r:write(root/'PASSES.json',dict(r,at=now())))
        if result['termination']!='TWO_PASSES_COMPLETE':
            raise ValueError('MASK_PASSES_INCOMPLETE:'+result['termination'])
        selected=result['selected']
        final,_=assess(selected,save=root/'final',diagnose=True)
        for p,m in final['metrics'].items():
            account_equal(m,measure(selected)['metrics'][p])
        if final['score']<control['score']:
            raise ValueError('MASK_FINAL_REGRESSION')
        # Verify the full control again after all interventions, not just at entry.
        restored,_=assess(columns)
        for p,m in restored['metrics'].items():
            account_equal(m,control['metrics'][p])
        view=subset_view(s['dataset'],selected)
        payload=dict(identity=s['identity'],result=result,control=control,final=final,
            groups=view['groups'],removed=[c for c in columns if c not in selected],
            parent_bundle=s['best_bundle'],scope=s['scope'],test_read=False,
            original_support_preserved=True,unmasked_after_pass_exact=True)
        if (root/'CLOSE.json').exists():
            close=read(root/'CLOSE.json')
            if (close['identity']!=s['identity'] or
                    file_sha(root/'RESULT.json')!=close['result_sha256'] or
                    read(root/'RESULT.json')!=payload):
                raise ValueError('MASK_COMPLETED_REPLAY_CHANGED')
        else:
            write(root/'RESULT.json',payload)
            write(root/'FEATURE_COUNTS.json',dict(**result['counts'],groups=len(view['groups']),
                initial_groups=len(s['dataset']['groups']),phase1_admitted=120))
            write(root/'CLOSE.json',dict(identity=s['identity'],termination=result['termination'],
                result_sha256=file_sha(root/'RESULT.json'),finished_at=now(),
                evaluations=result['evaluations'],structural_refit_performed=False))
        write(root/'PROGRESS.json',dict(stage='COMPLETE',identity=s['identity'],at=now(),
            counts=result['counts'],evaluations=result['evaluations']))
        print(dict(counts=result['counts'],evaluations=result['evaluations']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--seed',type=int,default=7)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--cross-device-probe',action='store_true')
    p=sub.add_parser('run'); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    (prepare if args.command=='prepare' else run)(args)


if __name__=='__main__':
    main()
