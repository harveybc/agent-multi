"""Authenticate and cold-replay explicitly pinned saved actor profiles; no fit."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.actor_checkpoint_qualification import (
    actor_difference, development_gate, exact_state, exact_trace, saved_point,
    qualification_profile,
)
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest, project_inputs, support_hash
from app.structural_refit import export_bundle, load_bundle, verify_files
from app.three_signal_learning import emit, environment, evaluate
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, now, read, write
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import market_for

ROOT = Path(__file__).resolve().parents[1]


def run(a):
    start = time.monotonic(); configure(7); torch.set_num_threads(a.threads)
    torch.backends.cudnn.allow_tf32 = False
    parent = read(a.parent/'SPEC.json'); verify_files(a.parent, parent['files'])
    source = read(a.source/'CONFIG.json'); verify_done(a.source, source['identity'])
    arm, updates = qualification_profile(source, a.profile)
    checkpoint = f'ACTOR_{updates:04d}.pt'
    trace_name = f'actor_{updates:04d}'
    model, _ = load_bundle(a.bundle, 'cpu')
    if (parent['identity'] != source['parent_identity'] or
            file_sha(a.bundle/'model.pt') != source['source_sha256']):
        raise ValueError('QUALIFY_LINEAGE')
    market = market_for(a.parent, parent)
    market.inputs = project_inputs(market.inputs, model.recipe); market.recipe = model.recipe
    if (support_hash(market.inputs) != source['support_sha256'] or
            set(model.recipe['split']['periods']) != {'TRAIN', 'VALIDATION'} or
            len(model.recipe['list']['columns']) != 69):
        raise ValueError('QUALIFY_SUPPORT')
    code = {str(p.relative_to(ROOT)): file_sha(p)
            for folder in ('app', 'agent_plugins', 'env_plugins', 'tools')
            for p in (ROOT/folder).rglob('*.py')}
    config = dict(schema='one_update_actor_qualification.v1', source=source['identity'],
                  parent=parent['identity'], source_receipt=file_sha(a.source/'DONE.json'),
                  parent_spec=file_sha(a.parent/'SPEC.json'),
                  bundle_manifest=file_sha(a.bundle/'MANIFEST.json'), source_code=code,
                  seeds=[19, 43], updates=updates, arm=arm, device='cpu', threads=a.threads,
                  recipe_identity=model.recipe['identity'], support=source['support_sha256'],
                  training=False, model_promoted=False)
    if a.profile != 'one_update':
        config.update(schema='saved_actor_qualification.v2', profile=a.profile)
    identity = digest(config)
    if (a.output/'CONFIG.json').exists():
        if read(a.output/'CONFIG.json') != config:
            raise ValueError('QUALIFY_RESUME_IDENTITY')
    elif any(p.name != 'worker.lock' for p in a.output.iterdir()):
        raise ValueError('QUALIFY_UNKNOWN_OUTPUT')
    write(a.output/'CONFIG.json', config)
    if (a.output/'DONE.json').exists():
        verify_done(a.output, identity)
        if not read(a.output/'RESULT.json')['qualified']:
            raise ValueError('QUALIFY_DEVELOPMENT_GATE')
        return

    def progress(stage, **kw):
        value = dict(stage=stage, at=now(), seconds=time.monotonic()-start, **kw)
        write(a.output/'PROGRESS.json', value); print(value, flush=True)

    risk = read(a.parent/'RISK.json')
    def accounts(logits, output, expected, trace_root):
        result = {}; prepared = prepared_signals(logits)
        for period, interval in model.recipe['split']['periods'].items():
            folder = output/period
            metrics = evaluate(environment(market, risk, parent['design']['horizon'], interval,
                                          model.effective['gamma']), logits, prepared=prepared,
                               save=folder, diagnose=True)
            account_equal(metrics, expected[period])
            for name in ('ACTIONS.json', 'TRADES.json', 'CLOSE_COUNTERFACTUALS.json'):
                exact_trace(read(folder/name), read(trace_root/period/name))
            with np.load(folder/'EQUITY.npz') as actual, np.load(trace_root/period/'EQUITY.npz') as old:
                if set(actual.files) != set(old.files) or any(not np.array_equal(actual[k], old[k]) for k in actual.files):
                    raise ValueError('QUALIFY_EQUITY_PARITY')
            result[period] = metrics
        return result

    progress('REFERENCE_REPLAY')
    baseline, _ = emit(model, market.inputs)
    expected_reference = (read(a.source/'RESULT.json')['reference'] if a.profile == 'bounded_critic8'
                          else read(a.source/'REFERENCE.json'))
    reference = accounts(baseline, a.output/'reference', expected_reference, a.source/'reference')
    ref_state = {n: v.detach().clone() for n, v in model.loss.state_dict().items()}
    records = []; first_actor = None
    for seed in config['seeds']:
        out = a.output/f'{arm}_{seed}'
        if (out/'DONE.json').exists():
            verify_done(out, identity); records.append(read(out/'RESULT.json')); continue
        origin = a.source/f'{arm}_{seed}'
        expected = saved_point(read(origin/'POINTS.json'), updates)
        progress('CANDIDATE_REPLAY', seed=seed)
        model, _ = load_bundle(a.bundle, 'cpu'); model.load(origin/checkpoint)
        differences = actor_difference(ref_state, model.loss.state_dict())
        actor_state = {n: v.detach().clone() for n, v in model.loss.state_dict().items() if n.startswith('actor_')}
        same_as_first = None if first_actor is None else all(torch.equal(v, first_actor[n]) for n, v in actor_state.items())
        if first_actor is None: first_actor = actor_state
        logits, _ = emit(model, market.inputs)
        metrics = accounts(logits, out/'native', expected['metrics'], origin/trace_name)
        ix = np.flatnonzero(market.inputs.valid & (market.inputs.period == 'TRAIN'))
        probe = market.inputs.windows[ix[np.linspace(0, len(ix)-1, 32, dtype=int)]].copy()
        if not (out/'bundle').exists():
            export_bundle(out/'bundle', model, probe, dict(identity=identity,
                kind='QUALIFIED_SAVED_ACTOR', profile=a.profile, seed=seed, updates=updates,
                source_checkpoint_sha256=file_sha(origin/checkpoint),
                new_optimizer_steps=0, research_only=True, model_promoted=False))
        reloaded, _ = load_bundle(out/'bundle', 'cpu')
        exact_state(reloaded.loss.state_dict(), model.loss.state_dict())
        if reloaded.effective != model.effective or reloaded.recipe != model.recipe:
            raise ValueError('QUALIFY_RELOAD_CONFIG')
        if {n:p.requires_grad for n,p in reloaded.loss.named_parameters()} != {
                n:p.requires_grad for n,p in model.loss.named_parameters()}:
            raise ValueError('QUALIFY_RELOAD_FLAGS')
        progress('COLD_REPLAY', seed=seed)
        cold, _ = emit(reloaded, market.inputs)
        if not np.array_equal(cold, logits, equal_nan=True):
            raise ValueError('QUALIFY_RELOAD_LOGITS')
        accounts(cold, out/'cold', metrics, out/'native')
        changes = {p:int((baseline[market.inputs.valid & (market.inputs.period == p)].argmax(-1) !=
                          logits[market.inputs.valid & (market.inputs.period == p)].argmax(-1)).sum())
                   for p in metrics}
        record = dict(seed=seed, metrics=metrics, actor_difference=differences,
                      same_actor_as_first_seed=same_as_first, changed_signals=changes,
                      source_checkpoint_sha256=file_sha(origin/checkpoint),
                      bundle_manifest_sha256=file_sha(out/'bundle/MANIFEST.json'),
                      model_sha256=file_sha(out/'bundle/model.pt'),
                      exact_native_accounts=True, exact_reload_logits=True,
                      exact_reload_accounts=True, new_optimizer_steps=0,
                      **development_gate(metrics, reference))
        write(out/'RESULT.json', record); seal(out, identity); records.append(record)
        progress('CANDIDATE_COMPLETE', seed=seed, qualified=record['qualified'])
    verify_done(a.source, source['identity']); verify_files(a.parent, parent['files'])
    if (file_sha(a.bundle/'model.pt') != source['source_sha256'] or
            file_sha(a.source/'DONE.json') != config['source_receipt'] or
            file_sha(a.parent/'SPEC.json') != config['parent_spec'] or
            file_sha(a.bundle/'MANIFEST.json') != config['bundle_manifest']):
        raise ValueError('QUALIFY_SOURCE_CHANGED')
    verify_files(ROOT, code)
    result = dict(identity=identity, reference=reference, candidates=records,
                  qualified=all(r['qualified'] for r in records),
                  full_support_rows=int(market.inputs.valid.sum()),
                  new_optimizer_steps=0, model_promoted=False, source_unchanged=True,
                  seconds=time.monotonic()-start, finished_at=now())
    write(a.output/'RESULT.json', result); seal(a.output, identity)
    progress('COMPLETE', qualified=result['qualified'])
    if not result['qualified']:
        raise ValueError('QUALIFY_DEVELOPMENT_GATE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'source', 'bundle', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--threads', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--profile', choices=['one_update', 'boundary_absolute8', 'bounded_critic8'], default='one_update')
    args = parser.parse_args()
    for source_path in (args.parent, args.source, args.bundle, ROOT):
        if args.output.resolve().is_relative_to(source_path.resolve()):
            raise ValueError('QUALIFY_OUTPUT_OVERLAPS_SOURCE')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
