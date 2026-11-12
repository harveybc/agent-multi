"""Nonlinear CPU check of budget-eligible local witnesses; always restore weights."""
import argparse
from pathlib import Path

import numpy as np
import torch

from app.bounded_functional_actor import BUDGET, admissible, budget_metrics
from app.decision_boundary_reachability import temporary_head_step
from app.frozen_recipe_inputs import file_sha
from app.optimal_action_sets import SourceValueGuard, correction_loss, error_targets, error_progress
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from tools.diagnose_selected_refinement import flat, head
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write


def run(a):
    configure(304)
    cfg = read(a.diagnostic/'CONFIG.json')
    verify_done(a.diagnostic, cfg['identity'])
    floor = read(a.floor/'CONFIG.json')
    verify_done(a.floor, floor['identity'])
    if cfg['floor'] != floor['identity'] or cfg['source_sha256'] != file_sha(a.bundle/'model.pt'):
        raise ValueError('BOUNDARY_WITNESS_LINEAGE')
    exported = read(a.bundle.parent/'RESULT.json')
    verify_done(a.bundle.parent, exported['identity'])
    if cfg['export'] != exported['identity']:
        raise ValueError('BOUNDARY_WITNESS_EXPORT')
    parent = read(a.parent/'SPEC.json')
    verify_files(a.parent, parent['files'])
    if parent['identity'] != floor['parent_identity']:
        raise ValueError('BOUNDARY_WITNESS_PARENT')
    model, _ = load_bundle(a.bundle, 'cpu')
    market = market_for(a.parent, parent, model.recipe)
    if support_hash(market.inputs) != floor['support_sha256']:
        raise ValueError('BOUNDARY_WITNESS_SUPPORT')
    records = read(a.floor/'panel/RECORDS.json')
    rows = np.array([r['row'] for r in records])
    if (not market.inputs.valid[rows].all() or not (market.inputs.period[rows]=='TRAIN').all()
            or market.inputs.row_ts_ns[rows].tolist() != [r['timestamp'] for r in records]):
        raise ValueError('BOUNDARY_WITNESS_PANEL')
    x = torch.tensor(market.inputs.windows[rows])
    params = [p for _, p in head(model)]
    theta = flat(params)
    saved = {k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    with torch.no_grad():
        z = model.surfaces(x).detach()
    native = torch.tensor([[o['reward_return'] for o in r['outcomes']] for r in records], dtype=torch.double)
    guard = SourceValueGuard(z, native, records)
    targets, weights = error_targets(z, native, records)
    fit = torch.tensor([i for i,r in enumerate(records) if r['role']=='fit'])
    before = float(correction_loss(z, targets, weights, fit))
    candidates = [r for r in read(a.diagnostic/'ROWS.json')
        if r['proposals']['guarded']['linear_witness']
        and r['proposals']['guarded']['relative_head_drift'] <= BUDGET['relative_parameter_drift']
        and r['proposals']['guarded']['linear_fit_margin_rms'] <= BUDGET['step_margin_rms']]
    directions = np.load(a.diagnostic/'LOCAL_DIRECTIONS.npz', allow_pickle=False)
    results = []
    for candidate in candidates:
        i, target = candidate['panel_index'], candidate['target']
        if records[i]['role'] != 'fit' or float(weights[i]) <= 0 or not targets[i,target]:
            raise ValueError('BOUNDARY_WITNESS_TARGET')
        step = directions[f'{i}_{target}_guarded']
        with temporary_head_step(params, step), torch.no_grad():
            after = model.surfaces(x).detach()
            metrics = budget_metrics(z[fit], z[fit], after[fit], theta, flat(params))
            loss = float(correction_loss(after, targets, weights, fit))
            preservation = guard.check(after)
            crossed = int(after[i].argmax()) == target
            results.append(dict(panel_index=i, target=target, crossed=crossed,
                source_action=int(z[i].argmax()), actual_action=int(after[i].argmax()),
                before_loss=before, after_loss=loss, metrics=metrics,
                guard=preservation, progress=error_progress(z, after, native, records),
                qualified=bool(crossed and preservation['passed'] and admissible(metrics,before,loss))))
        if not torch.equal(flat(params), theta):
            raise ValueError('BOUNDARY_WITNESS_HEAD_NOT_RESTORED')
    for key,value in saved.items():
        if not torch.equal(model.loss.state_dict()[key], value):
            raise ValueError('BOUNDARY_WITNESS_STATE_CHANGED:'+key)
    if file_sha(a.bundle/'model.pt') != cfg['source_sha256']:
        raise ValueError('BOUNDARY_WITNESS_FILE_CHANGED')
    root = Path(__file__).resolve().parents[1]
    report = dict(schema='boundary_nonlinear_cpu_probe.v1', source=cfg['identity'],
        source_sha256=cfg['source_sha256'], candidate_rule='all guarded linear witnesses within drift and RMS budgets',
        candidates=len(candidates), results=results, source_restored_exactly=True,
        state_tensors_verified=len(saved), optimizer_steps=0, model_exported=False,
        account_evaluated=False, validation_inference=False, test_read=False,
        qualified_candidates=sum(r['qualified'] for r in results),
        code={p:file_sha(root/p) for p in ('tools/verify_boundary_witness.py',
            'app/decision_boundary_reachability.py')})
    identity = digest(report)
    report['identity'] = identity
    if (a.output/'REPORT.json').exists() and read(a.output/'REPORT.json') != report:
        raise ValueError('BOUNDARY_WITNESS_OUTPUT_COLLISION')
    write(a.output/'REPORT.json', report)
    seal(a.output, identity)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic','floor','bundle','parent','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    run(parser.parse_args())
