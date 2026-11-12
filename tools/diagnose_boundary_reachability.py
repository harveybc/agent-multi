"""Finite FIT-only local witnesses from authenticated301/303; no actor update."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from app.bounded_functional_actor import BUDGET
from app.decision_boundary_reachability import target_problem, witness
from app.frozen_recipe_inputs import file_sha
from app.optimal_action_sets import SourceValueGuard, error_targets
from app.rl_subset import digest
from app.structural_refit import load_bundle
from tools.diagnose_selected_refinement import flat, head
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now


def run(args):
    started = time.monotonic()
    configure(304)
    source = read(args.diagnosis/'CONFIG.json')
    floor = read(args.floor/'CONFIG.json')
    export = read(args.bundle.parent/'RESULT.json')
    for path, identity in ((args.diagnosis, source['identity']),
                           (args.floor, floor['identity']),
                           (args.bundle.parent, export['identity'])):
        verify_done(path, identity)
    if (floor['diagnosis_identity'] != source['identity']
            or floor['diagnosis_receipt_sha256'] != file_sha(args.diagnosis/'DONE.json')
            or source['export'] != export['identity']
            or source['source_sha256'] != floor['source_sha256']
            or source['source_sha256'] != file_sha(args.bundle/'model.pt')
            or source['support'] != floor['support_sha256']):
        raise ValueError('BOUNDARY_LINEAGE')
    arrays = np.load(args.diagnosis/'PROJECTIONS.npz', allow_pickle=False)
    records = read(args.floor/'panel/RECORDS.json')
    if arrays['rows'].tolist() != [r['row'] for r in records]:
        raise ValueError('BOUNDARY_ROW_ORDER')
    fit = np.array([i for i, r in enumerate(records) if r['role']=='fit'])
    if arrays['fit'].tolist() != fit.tolist():
        raise ValueError('BOUNDARY_FIT_SUPPORT')
    z = torch.tensor(arrays['logits'], dtype=torch.float32)
    native = torch.tensor(arrays['native_returns'], dtype=torch.double)
    expected = torch.tensor([[o['reward_return'] for o in r['outcomes']] for r in records], dtype=torch.double)
    if not torch.equal(native, expected):
        raise ValueError('BOUNDARY_NATIVE_RETURNS')
    jac = arrays['jacobian']
    guard = SourceValueGuard(z, native, records)
    targets, weights = error_targets(z, native, records)
    model, _ = load_bundle(args.bundle, 'cpu')
    theta = flat([p for _, p in head(model)])
    if jac.shape != (len(records), 3, len(theta)):
        raise ValueError('BOUNDARY_HEAD_DIMENSION')
    theta_norm = float(theta.norm())
    target_p = z.double().masked_fill(~targets, -torch.inf).softmax(-1)
    gz = (z.double().softmax(-1)-target_p)*(weights/weights.sum())[:, None]
    gradient = np.einsum('na,nap->p', gz.numpy(), jac)
    guard_matrix = jac[guard.constraint_rows, guard.preferred]-jac[guard.constraint_rows, guard.bad]
    constraints = np.concatenate((guard_matrix, -gradient[None]), axis=0)
    lower = np.concatenate((guard.buffer.numpy(), [0.]))
    root = Path(__file__).resolve().parents[1]
    config = dict(schema='selected_boundary_reachability.v1', source=source['identity'],
        floor=floor['identity'], export=export['identity'], source_sha256=source['source_sha256'],
        source_arrays_sha256=file_sha(args.diagnosis/'PROJECTIONS.npz'),
        budget=BUDGET, fit_rows=len(fit), protected_rows=len(guard.rows),
        guard_constraints=len(guard.constraint_rows), head_parameters=len(theta),
        target_rule='every tied native-optimal action of each source-incorrect FIT row',
        claim='local linear witnesses only; solver failure is unresolved',
        actor_updates=0, validation_read=False, test_read=False,
        code={p:file_sha(root/p) for p in ('tools/diagnose_boundary_reachability.py',
            'app/decision_boundary_reachability.py','app/fit_decision_preservation.py',
            'app/optimal_action_sets.py')})
    config['identity'] = digest(config)
    if (args.output/'CONFIG.json').exists():
        if read(args.output/'CONFIG.json') != config:
            raise ValueError('BOUNDARY_OUTPUT_COLLISION')
        if (args.output/'DONE.json').exists():
            verify_done(args.output, config['identity'])
            return
    write(args.output/'CONFIG.json', config)
    rows = []; witnesses = {}
    for i in torch.nonzero(weights > 0).flatten().tolist():
        assert records[i]['role'] == 'fit'
        for target in torch.nonzero(targets[i]).flatten().tolist():
            direction, matrix, rhs, bounds = target_problem(z[i].numpy(), jac[i], target, len(fit))
            row = dict(panel_index=i, market_row=records[i]['row'], target=target,
                       position=records[i]['position'], **bounds, proposals={})
            for mode in ('unguarded', 'guarded'):
                step, result = witness(direction, matrix, rhs,
                    constraints if mode=='guarded' else None,
                    lower if mode=='guarded' else None)
                local_logits = z.double().numpy()+np.einsum('nap,p->na',jac,step)
                margins = np.einsum('nap,p->na',jac[fit,1:]-jac[fit,:1],step)
                result.update(head_step_l2=float(np.linalg.norm(step)),
                    relative_head_drift=float(np.linalg.norm(step)/theta_norm),
                    linear_fit_margin_rms=float(np.sqrt(np.mean(margins**2))),
                    linear_target_selected=int(local_logits[i].argmax())==target)
                row['proposals'][mode] = result
                witnesses[f'{i}_{target}_{mode}'] = step
            rows.append(row)
    write(args.output/'ROWS.json', rows)
    np.savez_compressed(args.output/'LOCAL_DIRECTIONS.npz', **witnesses)
    errors = sorted({r['panel_index'] for r in rows})
    feasible = {}
    for mode in ('unguarded','guarded'):
        solved = [r for r in rows if r['proposals'][mode]['linear_witness']]
        within = [r for r in solved if r['proposals'][mode]['relative_head_drift'] <= BUDGET['relative_parameter_drift']
                  and r['proposals'][mode]['linear_fit_margin_rms'] <= BUDGET['step_margin_rms']]
        feasible[mode] = dict(linear_witnesses=len(solved), unresolved=len(rows)-len(solved),
            witnesses_within_linear_rms_and_head_budgets=len(within),
            error_rows_with_such_witness=len({r['panel_index'] for r in within}))
    summary = dict(identity=config['identity'], fit_error_rows=len(errors),
        target_cases=len(rows), fit_rows=len(fit), head_parameters=len(theta), head_l2=theta_norm,
        gap_quantiles=np.quantile([r['gap'] for r in rows],[0,.25,.5,.75,1]).tolist(),
        error_rows_with_one_step_rms_bound_above_budget=int(sum(
            min(r['one_step_fit_margin_rms_lower_bound'] for r in rows if r['panel_index']==i)
            > BUDGET['step_margin_rms'] for i in errors)),
        proposals=feasible, native_updates_tested=0, actor_updates=0,
        nonlinear_verified=False, account_evaluated=False, model_promoted=False,
        validation_read=False, test_read=False, finished_at=now(),
        seconds=time.monotonic()-started)
    if file_sha(args.bundle/'model.pt') != source['source_sha256']:
        raise ValueError('BOUNDARY_SOURCE_CHANGED')
    write(args.output/'SUMMARY.json', summary)
    seal(args.output, config['identity'])
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnosis','floor','bundle','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
