"""Retrospective frozen-position CLOSE attribution; never changes policy or fits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.recipe_runtime import RecipeMarket
from app.suppressed_close_counterfactuals import fixed_position_exits
from env_plugins.discrete_curve_native import native_environment, HOUR
from tools.run_discrete_curve_probe import authenticate, read, write, now, receipt_valid


def check_record(record, timestamp, route, mask):
    action = record['action']
    if (record['timestamp'] != timestamp or record['route'] != route or
            type(action) is not int or not 0 <= action < len(mask) or not mask[action]):
        raise ValueError('BASELINE_ACTION_TRACE')


def summarize(rows):
    valid = [r for r in rows if not r['preempted']]
    n = len(valid)
    helpful = sum(r['delta_vs_hold'] > 1e-9 for r in valid)
    harmful = sum(r['delta_vs_hold'] < -1e-9 for r in valid)
    return dict(actual_closes=n, preempted=len(rows) - n, helpful=helpful,
        harmful=harmful, tied=n - helpful - harmful,
        harmful_fraction=harmful / n if n else None,
        at_first_hour=sum(r['holding_hours'] <= 1 for r in valid),
        sum_position_delta_usd_not_portfolio_profit=sum(r['delta_vs_hold'] for r in valid),
        mean_delta_risk=float(np.mean([r['delta_vs_hold_risk'] for r in valid])) if n else None,
        mean_regret_risk=float(np.mean([r['regret_risk'] for r in valid])) if n else None,
        passive_exit_reasons=dict(Counter(r['hold_reason'] for r in valid)))


def diagnose(env, actions, archived):
    obs = env.reset()
    rows = []
    done = False
    for record in actions:
        if done:
            raise ValueError('EXTRA_BASELINE_ACTION')
        e, b = env.engine, env.bar
        check_record(record, int(e.v[b]) + HOUR, int(obs['route'][0]), obs['mask'])
        if int(obs['route'][0]) and record['action'] == 1:
            p = e.pos
            risk = p.units * p.direction * (p.entry_price - p.sl)
            before = (e.balance, e.next_decision, p.entry_bar, len(e.trades))
            row = fixed_position_exits(e, risk)
            if before != (e.balance, e.next_decision, e.pos.entry_bar, len(e.trades)):
                raise ValueError('COUNTERFACTUAL_MUTATED_BASELINE')
            row['holding_hours'] = float((e.v[b + 1] - e.v[p.entry_bar]) / HOUR)
            row['policy_probabilities'] = record['probabilities']
            rows.append(row)
        obs, reward, done, duration = env.step(record['action'])
        if duration != record['duration'] or not np.isclose(reward, record['reward'], atol=1e-12, rtol=0):
            raise ValueError('BASELINE_REWARD_OR_DURATION')
    if not done:
        raise ValueError('INCOMPLETE_BASELINE')
    env.engine.flatten_at_end()
    actual = json.loads(json.dumps(env.engine.trades, default=str))
    if actual != archived:
        raise ValueError('BASELINE_LEDGER_NOT_EXACT')
    if sum(not r['preempted'] for r in rows) != sum(t['close_reason'] == 'AGENT_CLOSE' for t in actual):
        raise ValueError('CLOSE_COUNT_MISMATCH')
    return dict(scope='FIXED_POSITION_RETROSPECTIVE_NOT_PORTFOLIO_COUNTERFACTUAL',
        baseline_exact=True, risk_denominator='original units * entry-to-SL distance; excludes fees',
        summary=summarize(rows),
        by_direction={str(d): summarize([r for r in rows if r['direction'] == d]) for d in (-1, 1)},
        rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = authenticate(args.campaign)
    source = file_sha(__file__)
    market = RecipeMarket(spec)
    risk = read(spec['risk'])
    while True:
        if file_sha(__file__) != source:
            raise ValueError('DIAGNOSTIC_SOURCE_CHANGED')
        results = {}
        for path in sorted(args.collection.glob('*/runs/*/attempt*/RESULT.json')):
            folder = path.parent
            if not receipt_valid(folder, spec['identity']):
                continue
            result = read(path)
            if not result.get('financial_training'):
                continue
            unit = result['unit']
            for period in ('TRAIN', 'INNER_A'):
                key = f"{unit['id']}_{period}"
                output = args.output / (key + '.json')
                binding = dict(campaign=spec['identity'], source=source, result_sha256=file_sha(path))
                if output.exists():
                    report = read(output)
                    if report['binding'] != binding:
                        raise ValueError('DIAGNOSTIC_IDENTITY_CHANGED')
                else:
                    report = diagnose(native_environment(market, risk, unit['horizon'], period),
                        read(folder / f'{period}_ACTIONS.json'), read(folder / f'{period}_TRADES.json'))
                    report.update(binding=binding, unit=unit, period=period, finished_at=now())
                    write(output, report)
                results[key] = report['summary']
                write(args.output / 'SUMMARY.json', dict(time=now(), periods=len(results), results=results))
        aggregate = read(args.collection / 'AGGREGATE.json')
        if aggregate['complete'] == aggregate['expected']:
            expected = 2 * sum(r.get('financial_training', False) for r in aggregate['results'].values())
            # A receipt can arrive after this iteration's directory snapshot.
            if len(results) == expected:
                write(args.output / 'CLOSE.json', dict(time=now(), periods=len(results),
                    summary_sha256=file_sha(args.output / 'SUMMARY.json'), campaign=spec['identity']))
                return
        if not args.watch:
            return
        time.sleep(120)


if __name__ == '__main__':
    main()
