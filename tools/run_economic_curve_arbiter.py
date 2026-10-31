"""Bounded TRAIN economic calibration -> all-horizon hourly account pilot."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from app.economic_curve_arbiter import Reports, EconomicController, labelled_state, calibrate, economic_vector, rollout_reference
from app.dynamic_account_arbiter import HORIZONS
from app.sequential_account_pareto import clone
from app.suppressed_close_counterfactuals import fixed_position_exits
from app.frozen_recipe_inputs import file_sha
from app.recipe_runtime import RecipeMarket
from app.rl_subset import digest
from env_plugins.discrete_curve_native import CurveNativeEnv, native_environment, HOUR
from tools.run_discrete_curve_probe import authenticate, evaluate, read, write, now, receipt_valid
from tools.followup_curve_probe import verify_closure

ROOT = Path(__file__).resolve().parents[1]


def load_reports(collection, seed, market, risk, identity):
    arrays, common = {}, None
    for h in HORIZONS:
        candidates = list(collection.glob(f'*/runs/h{h}_s{seed}/attempt*/RESULT.json'))
        finished = [p for p in candidates if receipt_valid(p.parent, identity)]
        if len(finished) != 1 or not read(finished[0]).get('financial_training'):
            raise ValueError('FROZEN_MODEL_REQUIRED')
        with np.load(finished[0].parent / 'EMISSIONS.npz', allow_pickle=False) as data:
            indices = data['indices']
            if common is not None and not np.array_equal(indices, common):
                raise ValueError('UNPAIRED_REPORT_ROWS')
            common = indices.copy()
            values = np.concatenate((data['surfaces'][None], data['qvalues']), axis=0)
            arrays[h] = {int(row): values[:, i].copy() for i, row in enumerate(indices)}
    return Reports(arrays, {h: native_environment(market, risk, h, 'TRAIN') for h in HORIZONS})


def session_origins(reports, offsets):
    e = reports.template.engine
    for segment in sorted(set(e.seg_of[e.first:e.last])):
        deadline = int(e.ff[int(segment)])
        if deadline > int(e.v[e.last]) + HOUR:
            continue
        bars = [b for b in range(e.first, e.last) if e.seg_of[b] == segment
                and reports.template.valid[b] and int(e.v[b]) + HOUR < deadline and e.entry_gate(b)[0]]
        if not bars:
            continue
        origins = sorted({next((b for b in bars if b >= bars[0] + offset), None) for offset in offsets} - {None})
        yield int(segment), origins


def generate_session(reports, origins, ages):
    rows, anchors = [], 0
    for b in origins:
        flat = clone(reports.template.engine)
        flat.reset()
        flat.next_decision = b
        rows.extend(labelled_state(flat, None, reports))
        for h in HORIZONS:
            for side in ('BUY', 'SELL'):
                e = clone(flat)
                owner, _ = reports.step(e, None, side + ':' + str(h))
                while e.pos is not None:
                    age = e.next_decision - e.pos.entry_bar + 1
                    if age > max(ages):
                        break
                    if age in ages and reports.values(e, 3) is not None:
                        rows.extend(labelled_state(e, owner, reports))
                        anchors += 1
                    if age == max(ages):
                        break
                    owner, _ = reports.step(e, owner, 'HOLD')
    return dict(rows=rows, anchored_positions=anchors, origins=origins)


class JointEnv(CurveNativeEnv):
    def reset(self):
        self.owner = None
        self.selected = None
        self.opening_horizons = {}
        return super().reset()

    def step(self, action):
        b = self.bar
        flat = self.engine.pos is None
        if flat and action in (1, 2):
            if self.selected not in HORIZONS:
                raise ValueError('ENTRY_HORIZON_REQUIRED')
            self.horizon = self.selected
            self.refs = {1 if r['direction'] == 'buy' else -1: r['reference']['median']
                for r in self.risk_config['rows'] if r['horizon'] == self.selected}
            self.opening_horizons[b + 1] = self.selected
        result = super().step(action)
        if self.engine.pos is None:
            self.owner = None
        elif flat:
            self.owner = self.selected
        return result


class DecisionMarkers:
    """Pure command adapter for the existing evaluator, not policy probabilities."""
    def __init__(self, env, reports, controller=None):
        self.env, self.reports, self.controller = env, reports, controller
        self.trace = []

    def __getitem__(self, row):
        env, e = self.env, self.env.engine
        if int(row) != int(env.rows[env.bar]):
            raise ValueError('CURRENT_REPORT_ONLY')
        decision = (self.controller.decide(e, env.owner) if self.controller else
                    dict(action=self.reports.reference(e, env.owner), supported=False))
        action = decision['action']
        values = np.zeros(39, np.float32)
        if ':' in action:
            side, h = action.split(':')
            env.selected = int(h)
            values[1 if side == 'BUY' else 2] = 1.
        elif action == 'NOP':
            values[0] = 1.
        else:
            values[3:].reshape(2, 9, 2)[..., int(action == 'CLOSE')] = 1.
        self.trace.append(dict(decision, timestamp=int(e.v[env.bar]) + HOUR,
            deadline=int(e.ff[int(e.seg_of[env.bar])]), owner=env.owner,
            position_direction=0 if e.pos is None else e.pos.direction))
        return values


def audit_closes(env, reports, controller, actions, trace, expected_trades):
    """Post-account interventions; no diagnostic future is returned to the controller."""
    class Continuation:
        def reference(self, engine, owner):
            return controller.decide(engine, owner)['action']

        def step(self, engine, owner, action):
            return reports.step(engine, owner, action)

    env.reset()
    diagnostics = []
    if len(actions) != len(trace):
        raise ValueError('AUDIT_TRACE_LENGTH')
    for row, decision in zip(actions, trace):
        e = env.engine
        if row['timestamp'] != int(e.v[env.bar]) + HOUR:
            raise ValueError('AUDIT_REPLAY_CLOCK')
        if e.pos is not None and row['action'] == 1:
            risk = abs(e.pos.entry_price - e.pos.sl) * e.pos.units
            item = fixed_position_exits(e, risk)
            deadline = int(e.ff[int(e.seg_of[env.bar])])
            if deadline <= int(e.v[e.last]) + HOUR:
                outcomes = {a: rollout_reference(e, env.owner, a, Continuation())
                            for a in ('CLOSE', 'HOLD')}
                item['account_delta_close_minus_hold'] = (
                    np.array(outcomes['CLOSE']['vector']) - outcomes['HOLD']['vector']).tolist()
                item['account_outcomes'] = outcomes
            else:
                item['account_delta_close_minus_hold'] = None
                item['account_scope'] = 'CENSORED_WEEK_END_OUTSIDE_PERIOD'
            diagnostics.append(item)
        if ':' in decision['action']:
            env.selected = int(decision['action'].split(':')[1])
        env.step(row['action'])
    env.engine.flatten_at_end()
    if json.loads(json.dumps(env.engine.trades, default=str)) != expected_trades:
        raise ValueError('AUDIT_ACCOUNT_REPLAY_CHANGED')
    return diagnostics


def run_account(market, risk, reports, models, period, output, mode, weight):
    output.mkdir(parents=True, exist_ok=True)
    base = native_environment(market, risk, 3, period)
    env = JointEnv(base.engine, market.inputs, risk, 3)
    controller = EconomicController(reports, models, weight) if mode == 'joint' else None
    markers = DecisionMarkers(env, reports, controller)
    result = evaluate(env, markers, output, period)
    actions = read(output / f'{period}_ACTIONS.json')
    for row in actions:
        row.pop('probabilities')
        row['decision_source'] = mode + '_controller_not_a_policy_probability'
    write(output / f'{period}_ACTIONS.json', actions)
    write(output / 'DECISIONS.json', markers.trace)
    residuals = []
    timeline = np.array(env.equity_trace)
    for row in markers.trace:
        if not row.get('supported'):
            continue
        selected = (timeline[:, 0] >= row['timestamp']) & (timeline[:, 0] <= row['deadline'])
        if not selected.any() or timeline[selected][-1, 0] < row['deadline']:
            continue
        actual = economic_vector(timeline[selected, 1])
        residuals.append(np.array(actual) - np.array(row['vector']))
    result.update(mode=mode, horizon_counts={str(h): sum(env.opening_horizons[t['entry_bar_abs']] == h
        for t in env.engine.trades) for h in HORIZONS},
        supported_decisions=sum(row.get('supported', False) for row in markers.trace),
        risk_projection_count=sum(row.get('risk_projection', 0) for row in markers.trace),
        continuation_gap=dict(n=len(residuals),
            mean=np.mean(residuals, axis=0).tolist() if residuals else None,
            rmse=np.sqrt(np.mean(np.array(residuals)**2, axis=0)).tolist() if residuals else None,
            interpretation='calibration plus changed continuation; not pure forecast error'),
        scope='REFERENCE_RELATIVE_PILOT_NOT_GLOBAL_PARETO_OR_PROMOTED_STRATEGY')
    if controller is not None:
        diagnostics = audit_closes(env, reports, controller, actions, markers.trace,
                                  read(output / f'{period}_TRADES.json'))
        write(output / 'CLOSE_COUNTERFACTUALS.json', diagnostics)
        delta = [r['account_delta_close_minus_hold'] for r in diagnostics
                 if r['account_delta_close_minus_hold'] is not None]
        result['close_diagnostic'] = dict(proposals=len(diagnostics),
            harmful_fixed_position=sum(r['delta_vs_hold'] < -1e-8 for r in diagnostics),
            helpful_fixed_position=sum(r['delta_vs_hold'] > 1e-8 for r in diagnostics),
            preempted=sum(r['preempted'] for r in diagnostics),
            mean_regret_risk=float(np.mean([r['regret_risk'] for r in diagnostics])) if diagnostics else None,
            full_account_comparisons=len(delta),
            harmful_account_profit=sum(d[0] < -1e-8 for d in delta),
            helpful_account_profit=sum(d[0] > 1e-8 for d in delta),
            account_mean_return_dd_delta=np.mean(delta, axis=0).tolist() if delta else None,
            exact_replay=True)
    write(output / 'METRICS.json', result)
    return result


def valid_design(design):
    if (design['schema'] != 'economic_curve_arbiter.pilot.v1' or design['seeds'] != [7, 19, 43]
            or design['periods'] != ['TRAIN', 'INNER_A']
            or design['calibration_cutoff'] != '2023-01-01'
            or design['ridge_alpha'] <= 0 or design['minimum_fit_rows'] < 2
            or design['account_risk_weight'] < 0
            or not all(np.isfinite(design[k]) for k in ('ridge_alpha', 'account_risk_weight', 'minimum_fit_rows'))
            or not design['origin_offsets'] or any(type(v) is not int or v < 0 for v in design['origin_offsets'])
            or not design['position_ages'] or any(type(v) is not int or v < 1 for v in design['position_ages'])):
        raise ValueError('ECONOMIC_PILOT_DESIGN')


def closed_and_verified(root, identity):
    path = root / 'CLOSE.json'
    if not path.exists():
        return False
    closed = read(path)
    if closed['identity'] != identity:
        raise ValueError('ECONOMIC_CLOSURE_IDENTITY')
    for name, sha in closed['files'].items():
        child = (root / name).resolve()
        if not child.is_relative_to(root.resolve()) or file_sha(child) != sha:
            raise ValueError('ECONOMIC_CLOSURE_CHANGED')
    return True


def execute(args):
    torch.set_num_threads(1)
    design = read(args.config)
    valid_design(design)
    spec = authenticate(args.campaign)
    verify_closure(args.collection, 'AGGREGATE.json', 'results_sha256')
    binding = dict(campaign=spec['identity'], design=design, smoke=args.smoke,
        sources={name: file_sha(ROOT / name) for name in
                 ('tools/run_economic_curve_arbiter.py', 'app/economic_curve_arbiter.py')},
        sklearn=importlib.metadata.version('scikit-learn'),
        collection_close_sha256=file_sha(args.collection / 'CLOSE.json'))
    identity = digest(binding)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'SPEC.json').exists() and read(args.output / 'SPEC.json')['identity'] != identity:
        raise ValueError('ECONOMIC_IDENTITY_CHANGED')
    if closed_and_verified(args.output, identity):
        return
    write(args.output / 'SPEC.json', dict(identity=identity, binding=binding))
    market, risk = RecipeMarket(spec), read(spec['risk'])
    finished = {}
    for seed in design['seeds']:
        seed_root = args.output / f'seed_{seed}'
        if (seed_root / 'RESULT.json').exists():
            old = read(seed_root / 'RESULT.json')
            if old['identity'] != identity:
                raise ValueError('ECONOMIC_RESULT_IDENTITY')
            for name, sha in old['artifacts'].items():
                if file_sha(seed_root / name) != sha:
                    raise ValueError('ECONOMIC_RESULT_CHANGED')
            finished[str(seed)] = old
            continue
        reports = load_reports(args.collection, seed, market, risk, spec['identity'])
        sessions = list(session_origins(reports, design['origin_offsets']))
        all_rows = []
        for number, (segment, origins) in enumerate(sessions[:1] if args.smoke else sessions):
            started = time.monotonic()
            path = seed_root / 'sessions' / f'{segment:04d}.json'
            write(args.output / 'STATUS.json', dict(state='TRAIN_COUNTERFACTUAL_LABELS', seed=seed,
                completed_sessions=number, total_sessions=len(sessions), time=now()))
            if path.exists():
                data = read(path)
                if data['identity'] != identity or data['digest'] != digest(data['payload']):
                    raise ValueError('SESSION_IDENTITY')
                payload = data['payload']
            else:
                payload = generate_session(reports, origins, design['position_ages'])
                write(path, dict(identity=identity, payload=payload, digest=digest(payload)))
            all_rows.extend(payload['rows'])
            write(args.output / 'STATUS.json', dict(state='TRAIN_COUNTERFACTUAL_LABELS', seed=seed,
                completed_sessions=number + 1, total_sessions=len(sessions),
                label_rows=len(all_rows), last_session_seconds=time.monotonic() - started, time=now()))
        if args.smoke:
            write(args.output / 'SMOKE.json', dict(scope='TARGET_GENERATION_ONLY', rows=len(all_rows),
                keys=sorted({r['key'] for r in all_rows}), finite=all(np.isfinite(r['y']).all() for r in all_rows)))
            write(args.output / 'STATUS.json', dict(state='SMOKE_COMPLETE', time=now()))
            return
        write(args.output / 'STATUS.json', dict(state='FIT_ECONOMIC_CALIBRATION', seed=seed, time=now()))
        models, calibration = calibrate(all_rows, design['minimum_fit_rows'],
                                       design['calibration_cutoff'], design['ridge_alpha'])
        expected = {f'open_{h}_{a}' for h in HORIZONS for a in (0, 1, 2)} | {
            f'close_{owner}_{side}_{h}_{action}' for owner in HORIZONS for side in (-1, 1)
            for h in HORIZONS for action in ('HOLD', 'CLOSE')}
        if set(models) != expected:
            raise ValueError('CALIBRATION_ROLE_COVERAGE')
        joblib.dump(models, seed_root / 'calibrators.joblib')
        write(seed_root / 'CALIBRATION.json', calibration)
        metrics = {}
        for period in design['periods']:
            for mode in ('reference', 'joint'):
                write(args.output / 'STATUS.json', dict(state='ACCOUNT_REPLAY', seed=seed,
                    period=period, mode=mode, time=now()))
                metrics[f'{period}_{mode}'] = run_account(market, risk, reports, models,
                    period, seed_root / period / mode, mode, design['account_risk_weight'])
        outcome = dict(identity=identity, seed=seed, label_rows=len(all_rows), sessions=len(sessions),
            metrics=metrics, artifacts={str(p.relative_to(seed_root)): file_sha(p)
                for p in sorted(seed_root.rglob('*')) if p.is_file()})
        write(seed_root / 'RESULT.json', outcome)
        finished[str(seed)] = outcome
        write(args.output / 'SUMMARY.json', dict(identity=identity, finished=finished))
    lines = ['# Economic Multihorizon Pilot', '',
        'Reference-relative calibration, not a certified global optimum. No2024/TEST access.',
        'Calibration fits2020-2022; underlying actor weights saw2023H1; upstream design saw2023H2.', '',
        '| Seed | Period / policy | Return % | Sharpe | DD % | Orders | Buy / Sell | Win % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for seed, outcome in finished.items():
        for key, m in outcome['metrics'].items():
            fmt = lambda v: '-' if v is None else f'{v:.3f}'
            lines.append(f"| {seed} | {key} | {m['return_pct']:.3f} | {fmt(m['sharpe_daily_sqrt365'])} | "
                f"{m['max_drawdown_pct']:.3f} | {m['trades']} | {m['buy']['trades']} / {m['sell']['trades']} | "
                f"{fmt(None if m['win_rate'] is None else 100*m['win_rate'])} |")
    lines += ['', 'Full calibration errors, native costs, horizon usage and continuation gaps are in JSON.',
        'The future is used only in TRAIN targets and retrospective scoring, never runtime selection.',
        'No automatic model promotion, parameter sweep or floating selection follows these results.']
    (args.output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    (args.output / 'AUDIT_REQUEST.md').write_text('\n'.join([
        '# Audit Request: Economic Multihorizon Pilot', '',
        'Attack target-unit equality, common deadlines, reference-policy identity and split purging.',
        'Verify all reporters query the identical position, immutable SL/TP and no learned state inputs.',
        'Inspect calibration failure against constant targets, risk projection and continuation-gap reports.',
        'Verify native accounting, both directions, activity and no buys-first priority.',
        'This is reference-relative approximate improvement, not exact policy evaluation or global Pareto.', '']))
    write(args.output / 'CLOSE.json', dict(identity=identity, time=now(), seeds=len(finished),
        files={str(p.relative_to(args.output)): file_sha(p) for p in sorted(args.output.rglob('*'))
               if p.is_file() and p.name not in ('STATUS.json', 'CLOSE.json')}, strategy_promoted=False))
    write(args.output / 'STATUS.json', dict(state='COMPLETE', seeds=len(finished), time=now()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'collection', 'output', 'config'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    try:
        execute(args)
    except Exception as error:
        write(args.output / 'STATUS.json', dict(state='FAILED', error_type=type(error).__name__,
              error=str(error), time=now()))
        if isinstance(error, (ValueError, AssertionError)):
            raise SystemExit(78) from error
        raise


if __name__ == '__main__':
    main()
