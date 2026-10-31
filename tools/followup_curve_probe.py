"""Sequential frozen-probe account ablations and output/entropy diagnostics."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict

from agent_plugins.discrete_curve_sac import CurveSAC
from app.frozen_recipe_inputs import file_sha
from app.recipe_runtime import RecipeMarket
from env_plugins.discrete_curve_native import native_environment
from tools.run_discrete_curve_probe import authenticate, evaluate, read, write, now, receipt_valid


def hold_surfaces(values):
    held = values.copy()
    curves = held[:, 3:].reshape(-1, 2, 9, 2)
    curves[..., 0], curves[..., 1] = 1., 0.
    return held


def verify_closure(root, name, key):
    closure = read(root / 'CLOSE.json')
    if closure[key] != file_sha(root / name):
        raise ValueError('CLOSURE_CHANGED:' + name)
    return closure


def entropy_witness(recipe, alpha=1., gamma=.99):
    model = CurveSAC(recipe, alpha=alpha, gamma=gamma)
    with torch.no_grad():
        for params in (model.loss.actor_network_params, model.loss.qvalue_network_params,
                       model.loss.target_qvalue_network_params):
            for value in params.values(True, True):
                value.zero_()
    shape = (2, recipe['tensor']['window']['observations'], recipe['list']['n'])
    obs = TensorDict(dict(features=torch.zeros(shape), route=torch.tensor([[0], [1]]),
        progress=torch.full((2, 1), .5), mask=torch.tensor([[True, True, True], [True, True, False]])), [2])
    batch = obs.clone()
    batch['action'] = torch.zeros(2, dtype=torch.long)
    batch['next'] = obs.clone()
    batch['next', 'reward'] = torch.zeros(2, 1)
    batch['next', 'done'] = torch.zeros(2, 1, dtype=torch.bool)
    batch['next', 'terminated'] = torch.zeros(2, 1, dtype=torch.bool)
    with torch.no_grad():
        targets = model.loss._compute_target(batch).cpu().numpy()
    difference = float(targets[0] - targets[1])
    expected = gamma * alpha * math.log(1.5)
    return dict(scope='ZERO_ECONOMIC_REWARD_STRUCTURAL_WITNESS_NOT_ACTUAL_CAUSE',
        alpha=alpha, gamma=gamma, targets=targets.tolist(), observed_difference=difference,
        expected_difference=expected, passed=bool(np.isclose(difference, expected, atol=1e-6, rtol=1e-5)))


def characterize(archive, market, indices, surfaces, qvalues, period, alpha, gamma):
    selected = market.inputs.period[indices] == period
    x, q = surfaces[selected], qvalues[:, selected]
    actions = read(archive / f'{period}_ACTIONS.json')
    rewards = np.array([r['reward'] for r in actions])
    nonzero = np.abs(rewards[rewards != 0])
    close = [r['probabilities'][1] for r in actions if r['route'] and r['action'] == 1]
    return dict(rows=int(selected.sum()), logit_mean=x.mean(0).tolist(),
        logit_std=x.std(0).tolist(), twin_q_mean=q.mean(1).tolist(),
        twin_q_std=q.std(1).tolist(),
        close_policy_probability_quantiles=np.quantile(close, [0, .5, 1]).tolist() if close else None,
        reward_nonzero_fraction=float(np.mean(rewards != 0)),
        abs_nonzero_reward_quantiles=np.quantile(nonzero, [.1, .5, .9]).tolist() if len(nonzero) else None,
        alpha=alpha, equal_logit_action_count_target_difference=gamma * alpha * math.log(1.5),
        caveat='soft policy/Q statistics, not calibrated success probability or a measured causal entropy effect')


def account_pair(folder, market, risk, period, output, spec, diagnosis, source):
    result = read(folder / 'RESULT.json')
    unit = result['unit']
    diagnostic = diagnosis / f"{unit['id']}_{period}.json"
    report = read(diagnostic)
    if report['binding']['result_sha256'] != file_sha(folder / 'RESULT.json') or not report['baseline_exact']:
        raise ValueError('DIAGNOSIS_BINDING')
    binding = dict(campaign=spec['identity'], source=source, period=period,
        result_sha256=file_sha(folder / 'RESULT.json'), diagnosis_sha256=file_sha(diagnostic))
    if (output / 'RESULT.json').exists():
        previous = read(output / 'RESULT.json')
        if previous['binding'] != binding:
            raise ValueError('FOLLOWUP_BINDING')
        for name, sha in previous['artifacts'].items():
            if file_sha(output / name) != sha:
                raise ValueError('FOLLOWUP_ARTIFACT_CHANGED')
        return previous
    output.mkdir(parents=True, exist_ok=True)
    with np.load(folder / 'EMISSIONS.npz', allow_pickle=False) as data:
        indices, surfaces, qvalues = data['indices'], data['surfaces'], data['qvalues']
    baseline_dir, held_dir = output / 'baseline', output / 'no_early_close'
    baseline_dir.mkdir(exist_ok=True)
    held_dir.mkdir(exist_ok=True)
    baseline = evaluate(native_environment(market, risk, unit['horizon'], period),
                        dict(zip(indices.tolist(), surfaces)), baseline_dir, period)
    if baseline != result['metrics'][period]:
        raise ValueError('FOLLOWUP_BASELINE_METRICS')
    for suffix in ('ACTIONS', 'TRADES'):
        name = f'{period}_{suffix}.json'
        if read(baseline_dir / name) != read(folder / name):
            raise ValueError('FOLLOWUP_BASELINE_' + suffix)
    held = evaluate(native_environment(market, risk, unit['horizon'], period),
                    dict(zip(indices.tolist(), hold_surfaces(surfaces))), held_dir, period)
    if held['close_reasons'].get('AGENT_CLOSE', 0):
        raise ValueError('ABLATION_STILL_CLOSES')
    checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
    alpha = float(checkpoint['loss']['log_alpha'].exp())
    summary = dict(binding=binding, unit=unit, period=period,
        scope='FROZEN_ACCOUNT_DIAGNOSTIC_NOT_PRODUCTION_STRATEGY_OR_SELECTION',
        baseline_exact=True, baseline=baseline, no_early_close=held,
        no_close_minus_baseline=dict(net_profit_usd=held['net_profit_usd'] - baseline['net_profit_usd'],
            return_pp=held['return_pct'] - baseline['return_pct'],
            drawdown_pp=held['max_drawdown_pct'] - baseline['max_drawdown_pct'],
            orders=held['trades'] - baseline['trades']),
        fixed_position_diagnosis=report['summary'],
        outputs=characterize(folder, market, indices, surfaces, qvalues, period, alpha, spec['design']['gamma']))
    summary['artifacts'] = {str(p.relative_to(output)): file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output / 'RESULT.json', summary)
    return summary


def render(root, results, witness):
    lines = ['# Frozen SAC Follow-Up', '',
        'TRAIN:2020-2023H1. INNER_A:2023H2; upstream selection/risk design exposed to INNER_A.',
        'No2024/TEST access. Suppressing close is a diagnostic, not an admissible replacement strategy.', '']
    for period in ('TRAIN', 'INNER_A'):
        lines += [f'## {period}', '',
            '| Unit | Return % | No-close return % | Sharpe | DD % | Orders | Buy / Sell | Win % | Harmful close % |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in results:
            if row['period'] != period:
                continue
            a, b, d = row['baseline'], row['no_early_close'], row['fixed_position_diagnosis']
            fmt = lambda x: '-' if x is None else f'{x:.3f}'
            lines.append(f"| {row['unit']['id']} | {a['return_pct']:.3f} | {b['return_pct']:.3f} | "
                f"{fmt(a['sharpe_daily_sqrt365'])} | {a['max_drawdown_pct']:.3f} | {a['trades']} | "
                f"{a['buy']['trades']} / {a['sell']['trades']} | {fmt(None if a['win_rate'] is None else 100*a['win_rate'])} | "
                f"{fmt(None if d['harmful_fraction'] is None else 100*d['harmful_fraction'])} |")
        lines.append('')
    grouped = {}
    for row in results:
        key = f"{row['period']}_h{row['unit']['horizon']}"
        grouped.setdefault(key, []).append(row)
    seed_summary = {k: dict(seeds=[r['unit']['seed'] for r in v],
        mean_return=float(np.mean([r['baseline']['return_pct'] for r in v])),
        std_return=float(np.std([r['baseline']['return_pct'] for r in v])),
        active_seeds=sum(r['baseline']['activity_eligible'] for r in v)) for k, v in grouped.items()}
    write(root / 'SEED_SUMMARY.json', seed_summary)
    write(root / 'RESULTS.json', dict(scope='DIAGNOSTIC_ONLY', results=results,
        entropy_witness=witness, seed_summary=seed_summary))
    lines += ['## Interpretation', '',
        'All account pairs reproduce original metrics, actions and ledgers before intervention.',
        'Whole-account no-close deltas include changed opportunity and exposure; they are not sums of position attribution.',
        'The entropy witness tests action-count dependence with zero economic reward. It does not establish the actual cause.',
        'No threshold, holding-time rule, new training, feature selection or strategy promotion is triggered.', '',
        '## Next Controlled Correction', '',
        'Use a paired TRAIN-only sequential positive control with beneficial HOLD, protective CLOSE and NOP in both directions.',
        'Compare unchanged SAC against legal-action entropy normalization only; keep data, heads, seeds and budget matched.',
        'Require native accounting and preserved protective early close before any new financial comparison.',
        'Do not choose a minimum holding interval or confidence threshold from INNER_A.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    (root / 'AUDIT_REQUEST.md').write_text('\n'.join([
        '# Audit Request: Frozen Discrete SAC Follow-Up', '',
        'Audit RESULTS.json, SEED_SUMMARY.json, every per-unit RESULT receipt and CLOSE manifest.',
        'Attack baseline identity, invariance of opening logits, protection/calendar preservation,',
        'changed exposure/opportunity in no-close accounts, entropy units and action masks,',
        'TRAIN versus INNER_A exposure, missing activity and unsupported probabilities.',
        'Distinguish reproducible diagnostic evidence from a qualified policy correction.',
        'No claim that the entropy witness proves the source of actual early closing.',
        'No corrected learned strategy or floating-selection readiness is claimed.', '']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'collection', 'diagnosis', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try:
        execute(args)
    except Exception as error:
        write(args.output / 'STATUS.json', dict(state='FAILED', error_type=type(error).__name__,
            error=str(error), time=now()))
        raise


def execute(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    while not all((r / 'CLOSE.json').exists() for r in (args.collection, args.diagnosis)):
        write(args.output / 'STATUS.json', dict(state='WAIT_AUTOMATIC_CLOSURE', time=now()))
        time.sleep(60)
    spec = authenticate(args.campaign)
    verify_closure(args.collection, 'AGGREGATE.json', 'results_sha256')
    diagnostic_close = verify_closure(args.diagnosis, 'SUMMARY.json', 'summary_sha256')
    if diagnostic_close['campaign'] != spec['identity']:
        raise ValueError('DIAGNOSTIC_CAMPAIGN')
    aggregate = read(args.collection / 'AGGREGATE.json')
    if aggregate['complete'] != 12 or aggregate['issues']:
        raise ValueError('INCOMPLETE_FLEET')
    folders = {}
    for path in args.collection.glob('*/runs/*/attempt*/RESULT.json'):
        if receipt_valid(path.parent, spec['identity']):
            r = read(path)
            if r.get('financial_training'):
                if r['unit']['id'] in folders:
                    raise ValueError('DUPLICATE_FINISHED_UNIT')
                folders[r['unit']['id']] = path.parent
    if len(folders) != sum(r.get('financial_training', False) for r in aggregate['results'].values()):
        raise ValueError('FOLLOWUP_COVERAGE')
    market, risk = RecipeMarket(spec), read(spec['risk'])
    witness = entropy_witness(market.recipe)
    if not witness['passed']:
        raise ValueError('ENTROPY_TARGET_WITNESS')
    results, source = [], file_sha(__file__)
    for unit, folder in sorted(folders.items()):
        for period in ('TRAIN', 'INNER_A'):
            write(args.output / 'STATUS.json', dict(state='ACCOUNT_PAIR', unit=unit, period=period,
                completed=len(results), total=2*len(folders), time=now()))
            results.append(account_pair(folder, market, risk, period,
                args.output / unit / period, spec, args.diagnosis, source))
    render(args.output, results, witness)
    manifest = {str(path.relative_to(args.output)): file_sha(path) for path in sorted(args.output.rglob('*'))
                if path.is_file() and path.name not in ('STATUS.json', 'CLOSE.json')}
    write(args.output / 'CLOSE.json', dict(time=now(), campaign=spec['identity'], source=source,
        account_pairs=len(results), files=manifest, strategy_promoted=False, new_training=False))
    write(args.output / 'STATUS.json', dict(state='COMPLETE', account_pairs=len(results), time=now()))


if __name__ == '__main__':
    main()
