"""Check one oracle field per exact market observation, across native banks."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import linprog

from app.frozen_recipe_inputs import file_sha
from app.position_value_surface import EPS, evaluate_surface, query_weights, oracle_fit
from app.rl_subset import digest, support_hash
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_action_ranking_contrast import ensure_config, seal
from tools.run_discrete_curve_probe import read, write
from tools.run_structural_sac_refit import authenticate, market_for


def observation_groups(windows, records):
    groups = {}
    for i, record in enumerate(records):
        x = np.ascontiguousarray(windows[record['row']])
        if not np.isfinite(x).all():
            raise ValueError('SHARED_OBSERVATION_NONFINITE')
        key = (x.dtype.str, x.shape, hashlib.sha256(x.tobytes()).hexdigest(), record['direction'])
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def sign_oracle(weights, gaps, device='cpu'):
    # Fit action signs separately: unstable exact-value fitting is not a
    # certificate that the interface cannot express the required decisions.
    w = np.asarray(weights, float); y = np.asarray(gaps, float)
    if w.ndim != 2 or y.shape != (len(w),) or not np.isfinite(w).all() or not np.isfinite(y).all():
        raise ValueError('SIGN_ORACLE_INPUT')
    active = np.abs(y) > EPS; n = w.shape[1]
    a = np.vstack([np.column_stack([-np.sign(y[active, None]) * w[active], np.zeros(active.sum())]),
                   np.column_stack([np.eye(n), -np.ones(n)]),
                   np.column_stack([-np.eye(n), -np.ones(n)])])
    b = np.r_[-np.ones(active.sum()), np.zeros(2*n)]
    fit = linprog(np.r_[np.zeros(n), 1.], A_ub=a, b_ub=b,
                  bounds=[(None, None)]*n + [(0, None)], method='highs')
    if fit.status == 2:
        return dict(feasible=False, solver_status=2, future_label_oracle_only=True)
    if not fit.success:
        raise ValueError('SIGN_ORACLE_SOLVER:' + fit.message)
    nodes = fit.x[:-1]
    field = torch.tensor(nodes, dtype=torch.float32, device=device)[None,None,None,:].expand(2,len(w),2,-1)
    pred = evaluate_surface(field, torch.zeros(len(w),dtype=torch.long,device=device),
                            torch.tensor(w,dtype=torch.float32,device=device))[0].cpu().numpy()
    return dict(feasible=True, solver_status=0, max_node_magnitude=float(np.max(np.abs(nodes))),
                minimum_float64_margin=float(np.min(np.sign(y[active])*(w@nodes)[active])) if active.any() else None,
                float32_wrong_actions=int(((pred>EPS)!=(y>EPS))[active].sum()),
                future_label_oracle_only=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'source-runtime', 'primary', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args(); started = time.monotonic()
    receipt = verify_collection(args.primary); pc = read(args.primary / 'CONFIG.json')
    spec = authenticate(args.source, args.source_runtime)
    market = market_for(args.source, spec)
    support = support_hash(market.inputs)
    if spec['identity'] != pc['source'] or support != read(args.source / 'PREFLIGHT.json')['support_sha256']:
        raise ValueError('SHARED_OBSERVATION_SOURCE')
    config = dict(primary=receipt['identity'], source=spec['identity'], support_sha256=support,
                  device=args.device, post_primary_audit=True, optimizer_steps=0,
                  source_sha256=file_sha(Path(__file__)),
                  interface_sha256=file_sha(Path(__file__).resolve().parents[1] / 'app/position_value_surface.py'))
    config['identity'] = digest(config); ensure_config(args.output, config)
    if (args.output / 'DONE.json').exists():
        verify_collection(args.output); return
    panel = read(args.primary / 'panel/RESULT.json'); records = panel['records']
    groups = observation_groups(market.inputs.windows, records)
    owner = {i: j for j, bank in enumerate(panel['banks']) for i in bank}
    if sorted(owner) != list(range(len(records))):
        raise ValueError('SHARED_OBSERVATION_BANK_COVERAGE')
    coords = np.array([r['coords'] for r in records])
    dirs = np.array([0 if r['direction'] == 'long' else 1 for r in records])
    gaps = np.array([r['gap'] for r in records]); axes = read(args.primary / 'AXES.json')
    detail = []; results = {}
    for mode in ('scalar', 'mean81', 'progress9', 'surface81'):
        weights, _ = query_weights(coords, dirs, axes, mode)
        fits = [oracle_fit(weights[g], gaps[g], device=args.device) for g in groups]
        results[mode] = dict(groups=len(groups),
            wrong_groups=sum(f['wrong_actions'] > 0 for f in fits),
            wrong_actions=sum(f['wrong_actions'] for f in fits),
            float32_wrong_groups=sum(f['float32_wrong_actions'] > 0 for f in fits),
            float32_wrong_actions=sum(f['float32_wrong_actions'] for f in fits),
            max_error=max(f['max_error'] for f in fits),
            float32_max_error=max(f['float32_max_error'] for f in fits),
            max_node_magnitude=max(f['max_node_magnitude'] for f in fits))
        signs = [sign_oracle(weights[g], gaps[g], device=args.device) for g in groups]
        feasible = [s for s in signs if s['feasible']]
        results[mode]['sign_oracle'] = dict(infeasible_groups=len(signs)-len(feasible),
            float32_wrong_actions_in_feasible_groups=sum(s['float32_wrong_actions'] for s in feasible),
            max_node_magnitude=max(s['max_node_magnitude'] for s in feasible))
        detail.append(dict(mode=mode, fits=fits, sign_oracles=signs))
    result = dict(identity=config['identity'], records=len(records), native_banks=len(panel['banks']),
        observation_groups=len(groups), merged_groups=sum(len({owner[i] for i in g}) > 1 for g in groups),
        mixed_role_groups=sum(len({records[i]['role'] for i in g}) > 1 for g in groups),
        max_group_size=max(map(len, groups)), results=results, future_label_oracle_only=True,
        does_not_establish_learnability=True, model_promoted=False)
    write(args.output / 'GROUPS.json', groups); write(args.output / 'ORACLES.json', detail)
    write(args.output / 'RESULT.json', result); seal(args.output, config['identity'], started)
    print(result)


if __name__ == '__main__':
    main()
