"""Authenticated finite phase2 closure, reusing only the original evaluation context."""
import argparse
import copy
import fcntl
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from app.phase2_closure import close_phase2
from app.frozen_actor_compaction import compare_logits
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import cached_measurement, masked_emit, prepared_signals
from app.recipe_runtime import RecipeMarket
from app.rl_subset import digest, project_inputs, subset_recipe, support_hash
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import emit, environment, evaluate
from tools.run_cumulative_mask_pass import authenticate, make_expanded, load_selected
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_refit_mask_pass import account_equal, score_metrics
from tools.export_cumulative_sac_actor import run as export_actor

ROOT = Path(__file__).resolve().parents[1]


def old_cache(source, spec):
    rows = {}
    context = dict(identity=spec['identity'], device=spec['device'])
    for path in sorted((source / 'cache').glob('*.json')):
        row = read(path)
        identity = digest(dict(context=context, columns=row['columns']))
        if (row['identity'] != identity or path.stem != identity or
                row['sha256'] != digest(row['result'])):
            raise ValueError('CLOSURE_CACHE_IDENTITY')
        key = tuple(row['columns'])
        if key in rows:
            raise ValueError('CLOSURE_DUPLICATE_CACHE')
        rows[key] = (row['result'], str(path), file_sha(path))
    return rows


def run(a):
    source = authenticate(a.source)
    done = read(a.source / 'DONE.json')
    verify_files(a.source, done['files'])
    if done['identity'] != source['identity']:
        raise ValueError('CLOSURE_SOURCE_IDENTITY')
    reference_result = read(a.reference.parent / 'RESULT.json')
    verify_files(a.reference, read(a.reference / 'MANIFEST.json')['files'])
    original = read(a.source / 'RESULT.json')
    cfg = read(a.source / 'bundle/CONFIG.json')
    initial = original['result']['selected']
    if read(a.reference / 'RECIPE.json')['list']['columns'] != initial:
        raise ValueError('CLOSURE_REFERENCE_COLUMNS')
    spec = copy.deepcopy(source)
    spec.update(schema='phase2_closure.v1', source_selection=source['identity'],
                initial=initial, max_corrective_cycles=2)
    for name in ('app/phase2_closure.py', 'tools/run_phase2_closure.py'):
        spec['runtime_files'][name] = file_sha(ROOT / name)
    for p in (a.source / 'DONE.json', a.reference.parent / 'RESULT.json'):
        spec['source_files'][str(p)] = file_sha(p)
    for path in a.reference.iterdir():
        if path.is_file():
            spec['source_files'][str(path)] = file_sha(path)
    spec.pop('identity')
    spec['identity'] = digest(spec)
    if (a.output / 'SPEC.private.json').exists():
        if read(a.output / 'SPEC.private.json') != spec:
            raise ValueError('CLOSURE_RESUME_IDENTITY')
    else:
        write(a.output / 'SPEC.private.json', spec)
    if (a.output / 'DONE.json').exists():
        verify_files(a.output, read(a.output / 'DONE.json')['files'])
        return
    configure(7)
    torch.backends.cudnn.allow_tf32 = False
    if source['device'] != 'cuda' or torch.cuda.get_device_name() != read(a.source / 'PREFLIGHT.json')['device_name']:
        raise ValueError('CLOSURE_ORIGINAL_DEVICE')
    parent = Path(source['parent'])
    ps = read(parent / 'SPEC.json')
    paths = {k: str(parent / v) for k, v in ps['inputs'].items()}
    paths['recipe'] = str(parent / 'SUPPORT_RECIPE.json')
    market = RecipeMarket(dict(inputs=paths, engine_root=str(parent / 'engine'),
                               recipe_sha256=file_sha(paths['recipe'])))
    if support_hash(market.inputs) != read(a.source / 'PREFLIGHT.json')['support_hash']:
        raise ValueError('CLOSURE_SUPPORT')
    _, actor, restoration, _ = make_expanded(source)
    saved = torch.load(a.source / 'bundle/actor.pt', map_location=source['device'], weights_only=True)
    if set(saved) != set(actor.state_dict()) or any(not torch.equal(v, saved[k]) for k, v in actor.state_dict().items()):
        raise ValueError('CLOSURE_EXPANDED_WEIGHTS')
    model = SimpleNamespace(device=torch.device(source['device']), surfaces=actor.surface)
    risk = read(parent / 'RISK.json')
    universe = source['universe']
    start = time.monotonic()
    def assess(cols, save=None, diagnose=False):
        z = masked_emit(model, market.inputs, [c in cols for c in universe])
        prepared = prepared_signals(z)
        metrics = {p: evaluate(environment(market, risk, ps['design']['horizon'], interval,
                                           ps['effective']['gamma']), z, prepared=prepared,
                                diagnose=diagnose, save=save / p if save else None)
                   for p, interval in market.recipe['split']['periods'].items()}
        return score_metrics(metrics), z
    control, logits = assess(initial, diagnose=True)
    for p, m in control['metrics'].items():
        account_equal(m, reference_result['metrics'][p])
        account_equal(m, original['final']['metrics'][p])
    ref, _ = load_bundle(a.reference, source['device'])
    rz, _ = emit(ref, project_inputs(market.inputs, ref.recipe))
    parity = compare_logits(rz[market.inputs.valid], logits[market.inputs.valid])
    if not parity['qualified']:
        raise ValueError('CLOSURE_REFERENCE_PARITY')
    prior = old_cache(a.source, source)
    used, fresh = {}, []
    context = dict(identity=source['identity'], device=source['device'])
    def measure(cols):
        key = tuple(cols)
        if key in prior:
            value, path, sha = prior[key]
            used[key] = dict(columns=cols, origin='source299', path=path, sha256=sha)
            return value
        def compute():
            before = time.monotonic()
            value = assess(cols)[0]
            fresh.append(dict(columns=cols, seconds=time.monotonic() - before))
            return value
        value = cached_measurement(a.output / 'cache', context, cols, compute)
        used[key] = dict(columns=cols, origin='closure351', result_sha256=digest(value))
        write(a.output / 'PROGRESS.json', dict(at=now(), measured=len(used),
              fresh=len(fresh), seconds=time.monotonic() - start,
              last_seconds=fresh[-1]['seconds'] if fresh else None))
        return value
    result = close_phase2(universe, source['blocks'], initial, measure,
                         progress=lambda r: write(a.output / 'PROGRESS_PASS.json', r))
    # A second deterministic decision replay must require no new inference.
    n = len(fresh)
    if close_phase2(universe, source['blocks'], initial, measure) != result or len(fresh) != n:
        raise ValueError('CLOSURE_DECISION_REPLAY')
    selected = result['selected']
    final, _ = assess(selected, save=a.output / 'final', diagnose=True)
    for p, m in final['metrics'].items():
        account_equal(m, measure(selected)['metrics'][p])
    if not final['eligible'] or final['score'] < control['score']:
        raise ValueError('CLOSURE_REGRESSION')
    bundle = a.output / 'bundle'
    bundle.mkdir(exist_ok=True)
    cfg.update(selected=selected, selection_identity=spec['identity'],
               input_recipe=subset_recipe(cfg['universe'], selected))
    write(bundle / 'CONFIG.json', cfg)
    torch.save(actor.state_dict(), bundle / 'actor.pt')
    write(bundle / 'MANIFEST.json', dict(schema='cumulative_frozen_actor.v1',
          files={n: file_sha(bundle / n) for n in ('CONFIG.json', 'actor.pt')}))
    cold, _ = load_selected(bundle, source['device'])
    z, _ = emit(cold, project_inputs(market.inputs, cfg['input_recipe']))
    expected = masked_emit(model, market.inputs, [c in selected for c in universe])
    if not np.array_equal(z, expected, equal_nan=True):
        raise ValueError('CLOSURE_COLD_LOGITS')
    authenticate(a.output)
    verify_files(a.source, done['files'])
    write(a.output / 'CACHE_REUSE.json', dict(records=list(used.values()), fresh=fresh,
          context=context, original_actor_tensors_exact=len(saved),
          original_runtime_verified=True, original_support_verified=True))
    write(a.output / 'RESULT.json', dict(identity=spec['identity'], result=result,
          control=control, final=final, baseline_parity=parity, counts=dict(initial=len(initial),
          final=len(selected), groups=len(cfg['input_recipe']['representation']['resolved']['names']),
          absent=len(universe)-len(selected)), fresh_evaluations=len(fresh),
          reused_evaluations=sum(r['origin']=='source299' for r in used.values()),
          seconds=time.monotonic()-start, optimizer_steps=0, test_read=False,
          cold_exact=True, decision_replay_exact=True, source_unchanged=True,
          model_promoted=False, mode=source['mode']))
    write(a.output / 'DONE.json', dict(identity=spec['identity'], files={
          str(p.relative_to(a.output)): file_sha(p) for p in a.output.rglob('*')
          if p.is_file() and p.name not in ('DONE.json', 'worker.lock', 'PROGRESS.json', 'PROGRESS_PASS.json')
          and p.suffix != '.lock'}))
    print(result['status'], len(selected), 'fresh', len(fresh), flush=True)


def verify_and_export(a):
    spec = authenticate(a.output)
    verify_files(a.output, read(a.output / 'DONE.json')['files'])
    result = read(a.output / 'RESULT.json')
    old = old_cache(a.source, authenticate(a.source))
    local = old_cache(a.output, dict(identity=spec['source_selection'], device=spec['device']))
    rows = {**old, **local}
    replay = close_phase2(spec['universe'], spec['blocks'], spec['initial'],
                          lambda cols: rows[tuple(cols)][0])
    if replay != result['result']:
        raise ValueError('CLOSURE_VERIFY_REPLAY')
    # Independent explicit finite-neighborhood comparison, not the selector decision.
    selected = replay['selected']
    baseline = rows[tuple(selected)][0]
    qualifications = []
    for block in spec['blocks']:
        remove = set(block) <= set(selected)
        wanted = set(selected) - set(block) if remove else set(selected) | set(block)
        cols = [c for c in spec['universe'] if c in wanted]
        if not cols:
            qualifications.append(False)
            continue
        value = rows[tuple(cols)][0]
        qualifications.append(value['eligible'] and
                              (value['score'] >= baseline['score'] if remove else value['score'] > baseline['score']))
    if qualifications != [r['qualifies'] for r in replay['coverage'][-1]['neighbors']]:
        raise ValueError('CLOSURE_VERIFY_NEIGHBORS')
    export = a.output.parent / (a.output.name + '-export')
    if selected != spec['initial']:
        export.mkdir(exist_ok=True)
        export_actor(SimpleNamespace(selection=a.output, output=export))
        model_path = export / 'bundle'
    else:
        model_path = a.reference
    configure(7)
    model, _ = load_bundle(model_path, 'cpu')
    from app.selected_input_market import market_for
    parent = Path(spec['parent'])
    ps = read(parent / 'SPEC.json')
    market = market_for(parent, ps, selected_recipe=model.recipe)
    market.inputs = project_inputs(market.inputs, model.recipe)
    market.recipe = model.recipe
    z, _ = emit(model, market.inputs)
    risk = read(parent / 'RISK.json')
    for p, interval in market.recipe['split']['periods'].items():
        m = evaluate(environment(market, risk, ps['design']['horizon'], interval,
                                 ps['effective']['gamma']), z, prepared=prepared_signals(z), diagnose=True)
        account_equal(m, result['final']['metrics'][p])
    verify_files(a.output, read(a.output / 'DONE.json')['files'])
    verification = dict(identity=spec['identity'], status=replay['status'],
        final_context_neighbors=len(qualifications), qualifying_moves=sum(qualifications),
        cycles=len(replay['cycles']), selected=selected, cpu_accounts_exact=2,
        decision_replay_exact=True, independent_neighbor_check=True,
        retained_reference=selected==spec['initial'], physical_bundle=str(model_path),
        ready_for_architecture_search=True, globally_optimal=False, live_promoted=False)
    # Verification is a separate artifact, outside the already sealed producer.
    write(a.output.parent / (a.output.name + '-VERIFIED.json'), verification)
    print('VERIFIED', replay['status'], len(selected), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    if any(a.output.resolve().is_relative_to(p.resolve()) for p in (a.source, a.reference.parent)):
        raise ValueError('CLOSURE_SOURCE_OVERLAP')
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not a.verify_only:
            run(a)
        verify_and_export(a)
