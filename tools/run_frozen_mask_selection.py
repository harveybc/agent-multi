"""Native frozen-checkpoint adapter for feature_selector floating backward search."""
import argparse
import fcntl
import copy
from pathlib import Path
import time

import numpy as np
import torch

from feature_selector.contracts import subset_view, validate_dataset
from feature_selector.plugins import resolve
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import cached_measurement, masked_emit, prepared_signals
from app.recipe_runtime import RecipeMarket
from app.rl_subset import digest, support_hash, coherent_blocks
from app.three_signal_learning import environment, evaluate
from tools.run_discrete_curve_probe import read, write, now, pins, configure
from tools.run_paired_ablation import verified as verify_pilot
from tools.run_three_signal_baseline import make_model

ROOT = Path(__file__).resolve().parents[1]


def check_identity(s):
    if s.get('identity') != digest({k:v for k,v in s.items() if k != 'identity'}):
        raise ValueError('MASK_SPEC_IDENTITY')


def prepare(args):
    if args.root.exists(): raise ValueError('NEW_MASK_ROOT_REQUIRED')
    pilot = read(args.pilot/'SPEC.private.json'); check_identity(pilot)
    parent = read(args.parent/'SPEC.private.json'); check_identity(parent)
    baseline = read(Path(pilot['baseline_root'])/'SPEC.private.json')
    risk_path = Path(pilot['baseline_root'])/'RISK.json'
    if file_sha(risk_path) != baseline['risk_sha256']: raise ValueError('MASK_RISK_CHANGED')
    witnesses = []
    for role, seed in (('primary', 7), ('secondary', 19)):
        out = args.collection/role/'runs'/f'full__{seed}'/'attempt1'
        if not verify_pilot(out/'RESULT.json', pilot, f'full__{seed}'):
            raise ValueError('MASK_BASELINE_RECEIPT')
        r = read(out/'RESULT.json')
        step = r['selectors']['train_best']['selected_step']
        if step is None: raise ValueError('MASK_NO_CHECKPOINT')
        witnesses.append(dict(seed=seed, folder=str(out.resolve()), step=step,
                              receipt_sha256=file_sha(out/'RESULT.json')))
    _, plugin = resolve('selectors', 'floating_backward')
    dataset = copy.deepcopy(pilot['dataset'])
    dataset['blocks'] = coherent_blocks(dataset['features'])
    s = dict(schema='frozen_sac_mask_search.v1', parent=parent, pilot=pilot,
        baseline=witnesses, dataset=dataset, design=pilot['design'],
        risk=read(risk_path), selector='floating_backward', plugin=plugin,
        selector_params=dict(min_features=1, max_evaluations=16384, tolerance=1e-5),
        checkpoint_rule='paired full-input pilot TRAIN-best; fixed 524288 budget',
        ranking='paired mean INNER_A net return/week; absolute TRAIN-INNER_A gap tie-break',
        periods=['TRAIN', 'INNER_A'], device=args.device, created_at=now(),
        torch_version=torch.__version__, numpy_version=np.__version__,
        inference_precision='float32; cudnn TF32 disabled; canonical device control re-evaluated',
        sources=pins(ROOT, ['app', 'agent_plugins', 'env_plugins', 'tools']),
        scope='FROZEN_MASK_DEVELOPMENT_NOT_STRUCTURAL_SELECTION; no TEST or new fitting')
    validate_dataset(s['dataset'])
    s['identity'] = digest(s)
    args.root.mkdir(parents=True)
    write(args.root/'SPEC.private.json', s)
    print(s['identity'], flush=True)


def authenticate(root):
    s = read(root/'SPEC.private.json'); check_identity(s)
    for name, sha in s['sources'].items():
        if file_sha(ROOT/name) != sha: raise ValueError('MASK_SOURCE_CHANGED:'+name)
    plugin, witness = resolve('selectors', s['selector'])
    if witness != s['plugin']: raise ValueError('MASK_PLUGIN_CHANGED')
    if torch.__version__ != s['torch_version'] or np.__version__ != s['numpy_version']:
        raise ValueError('MASK_NUMERIC_RUNTIME_CHANGED')
    p = s['parent']
    for name, sha in p['native'].items():
        if file_sha(Path(p['engine_root'])/name) != sha: raise ValueError('MASK_ENGINE_CHANGED')
    for name, sha in p['input_sha256'].items():
        if file_sha(p['inputs'][name]) != sha: raise ValueError('MASK_INPUT_CHANGED')
    for b in s['baseline']:
        f = Path(b['folder'])/'RESULT.json'
        if file_sha(f) != b['receipt_sha256'] or not verify_pilot(f, s['pilot'], f"full__{b['seed']}"):
            raise ValueError('MASK_BASELINE_CHANGED')
    return s, plugin


def exact_account_check(actual, expected):
    # Entropies are descriptive floating-point reductions, not trading actions.
    ignored = {'mean_signal_entropy', 'mean_requested_command_entropy'}
    if {k:v for k,v in actual.items() if k not in ignored} != {k:v for k,v in expected.items() if k not in ignored}:
        raise ValueError('MASK_FULL_ACCOUNT_PARITY')


def run(args):
    with (args.root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s, selector = authenticate(args.root)
        configure(7)
        torch.backends.cudnn.allow_tf32 = False
        market = RecipeMarket(s['parent']); d = s['design']; columns = s['dataset']['features']
        if (support_hash(market.inputs) != s['dataset']['support_hash']
                or columns != market.recipe['list']['columns']
                or digest(market.recipe['scaler']) != s['dataset']['scaler_hash']):
            raise ValueError('MASK_PAIRED_SUPPORT')
        models = []
        for b in s['baseline']:
            out = Path(b['folder']); rec = read(out/'RECIPE.json')
            if (rec['tensor'] != market.recipe['tensor'] or rec['scaler']['mean'] != market.recipe['scaler']['mean']
                    or rec['scaler']['std'] != market.recipe['scaler']['std']
                    or rec['representation']['resolved'] != market.recipe['representation']['resolved']):
                raise ValueError('MASK_FULL_REPRESENTATION')
            model = make_model(rec, d, b['seed'], s['device'])
            model.load(out/'checkpoints'/f"{b['step']}.pt")
            models.append(model)
        context = dict(spec=s['identity'], device=s['device'],
                       device_name=torch.cuda.get_device_name() if s['device']=='cuda' else 'cpu')
        completed = 0; times = []; started = time.monotonic()

        def compute(keep_columns):
            nonlocal completed
            start = time.monotonic(); inference_seconds = account_seconds = 0.
            keep = [c in keep_columns for c in columns]; full = all(keep); seeds = []
            for b, model in zip(s['baseline'], models):
                t = time.monotonic(); logits = masked_emit(model, market.inputs, keep)
                inference_seconds += time.monotonic()-t
                prepared = prepared_signals(logits)
                replay = None
                if full:
                    archive = np.load(Path(b['folder'])/'accounts'/str(b['step'])/'EMISSIONS.npz')
                    ix = archive['indices']
                    if not np.array_equal(ix, np.flatnonzero(market.inputs.valid)):
                        raise ValueError('MASK_CONTROL_SUPPORT')
                    # The archived GPU may have used TF32. Check the new FP32
                    # implementation against CPU, not a different arithmetic mode.
                    reference = make_model(read(Path(b['folder'])/'RECIPE.json'), d, b['seed'], 'cpu')
                    reference.load(Path(b['folder'])/'checkpoints'/f"{b['step']}.pt")
                    cpu = masked_emit(reference, market.inputs, keep)
                    if not np.allclose(logits[ix], cpu[ix], rtol=1e-4, atol=1e-4):
                        raise ValueError('MASK_FP32_CPU_REFERENCE_DRIFT')
                    del reference
                    # Requalify cross-device inference once. Never reuse an old
                    # account as the control when near ties change its actions.
                    model.load(Path(b['folder'])/'checkpoints'/f"{b['step']}.pt")
                    repeated = masked_emit(model, market.inputs, keep)
                    if not np.array_equal(logits[ix], repeated[ix]):
                        raise ValueError('MASK_RELOAD_LOGITS_CHANGED')
                    old = np.full_like(logits, np.nan); old[ix] = archive['logits']
                    replay = dict(changed_actions=int(np.sum(prepared[0][ix] != prepared_signals(old)[0][ix])),
                        maximum_logit_difference=float(np.abs(logits[ix]-archive['logits']).max()),
                        maximum_cpu_logit_difference=float(np.abs(logits[ix]-cpu[ix]).max()),
                        reload_exact=True, historical_account_reused=False, historical_metrics={})
                metrics = {}
                for period in s['periods']:
                    t = time.monotonic()
                    metrics[period] = evaluate(environment(market, s['risk'], d['horizon'],
                        market.recipe['split']['periods'][period], d['gamma']), logits,
                        diagnose=True, prepared=prepared)
                    account_seconds += time.monotonic()-t
                    if full:
                        expected = read(Path(b['folder'])/'accounts'/str(b['step'])/period/'METRICS.json')
                        archived = evaluate(environment(market, s['risk'], d['horizon'],
                            market.recipe['split']['periods'][period], d['gamma']), old,
                            diagnose=True, prepared=prepared_signals(old))
                        exact_account_check(archived, expected)
                        replay['historical_metrics'][period] = archived
                seeds.append(dict(seed=b['seed'], metrics=metrics, baseline_replay=replay))
            train = float(np.mean([r['metrics']['TRAIN']['return_per_week'] for r in seeds]))
            val = float(np.mean([r['metrics']['INNER_A']['return_per_week'] for r in seeds]))
            elapsed = time.monotonic()-start; times.append(elapsed); completed += 1
            write(args.root/'PROGRESS.json', dict(stage='FROZEN_FLOATING', new_evaluations=completed,
                features=len(keep_columns), seconds=time.monotonic()-started,
                median_seconds_per_subset=float(np.median(times)), last_inference_seconds=inference_seconds,
                last_account_seconds=account_seconds, train_return_per_week=train,
                validation_return_per_week=val, at=now()))
            return dict(eligible=all(r['metrics'][p]['orders_per_week']>2 for r in seeds for p in s['periods']),
                score=val, gap=abs(train-val), train_score=train, combined_score=.5*(train+val),
                seeds=seeds, frozen_weights=True, fresh_training=False,
                seconds=elapsed, inference_seconds=inference_seconds, account_seconds=account_seconds)

        def measure(keep_columns):
            ordered = subset_view(s['dataset'], keep_columns)['features']
            return cached_measurement(args.root/'cache', context, ordered, lambda: compute(ordered))

        # Full control must pass before any deletion or reinclusion is inspected.
        control = measure(columns)
        write(args.root/'PREFLIGHT.json', dict(identity=s['identity'], passed=True,
            paired_seeds=[b['seed'] for b in s['baseline']], control=control,
            archived_emission_native_parity=True, canonical_device_reload_exact=True,
            historical_account_reused=False, original_rows_and_windows=True))
        if args.preflight_only:
            return
        result = selector.select(s['dataset'], measure, s['selector_params'])
        write(args.root/'RESULT.json', dict(identity=s['identity'], result=result))
        write(args.root/'CLOSE.json', dict(identity=s['identity'], finished_at=now(),
            termination=result['termination'], evaluations=result['evaluations'], selected=result['selected'],
            result_sha256=file_sha(args.root/'RESULT.json'), structural_confirmation=False,
            scope=s['scope']))


def main():
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--root', type=Path, required=True)
    for name in ('pilot', 'collection', 'parent'): a.add_argument('--'+name, type=Path, required=True)
    a.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    a = sub.add_parser('run'); a.add_argument('--root', type=Path, required=True)
    a.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
