#!/usr/bin/env python3
"""Exact small native account trees; retrospective design evidence, no training."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.sequential_account_pareto import native_case, replay
from app.zigzag_opportunities import engine


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_check(leaves, front):
    vectors = {r['vector'] for r in leaves}
    fv = {r['vector'] for r in front}
    def dominates(a, b):
        return a != b and all(x >= y for x, y in zip(a, b))
    if not fv <= vectors:
        raise AssertionError('FRONT_NOT_IN_TREE')
    if any(any(dominates(v, f) for v in vectors) for f in fv):
        raise AssertionError('DOMINATED_FRONT')
    if any(not any(dominates(f, v) for f in fv) for v in vectors - fv):
        raise AssertionError('MISSING_FRONT_VECTOR')


def calculate(engine_root, output, steps=5):
    engine(engine_root)
    config_path = ROOT / 'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    risk = json.loads(config_path.read_text())
    sources = {p: sha(ROOT / p) for p in (
        'app/sequential_account_pareto.py', 'app/dynamic_account_panel.py',
        'app/dynamic_account_arbiter.py', 'app/horizon_risk_rules.py',
        'tools/calculate_sequential_pareto_reference.py',
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json')}
    sources.update({'native/' + str(p.relative_to(engine_root)): sha(p)
                    for p in sorted((engine_root / 'app/short_horizon').rglob('*.py'))})
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    max_error = 0.
    replay_count = 0
    for pattern in ('smooth', 'oscillating', 'reversal'):
        for mirror in (1, -1):
            for initial in (0, 1, -1):
                case, leaves = native_case(risk, pattern, mirror, initial, steps=steps)
                independent_check(leaves, case['frontier'])
                for first, front in case['conditional'].items():
                    independent_check([r for r in leaves if r['trace'][0] == first], front)
                best = max(leaves, key=lambda r: r['vector'])
                case['max_return'] = best
                case['best_return_no_voluntary_close'] = max(r['vector'][0] for r in leaves if 'CLOSE' not in r['trace'])
                if not initial:
                    one_trade = lambda r: sum(':' in a for a in r['trace']) <= 1
                    case['best_return_at_most_one_entry'] = max(r['vector'][0] for r in leaves if one_trade(r))
                    case['best_return_single_horizon'] = {str(h): max(
                        r['vector'][0] for r in leaves if all(':' not in a or a.endswith(':' + str(h)) for a in r['trace']))
                        for h in (3, 6, 12, 24)}
                case['risk_budget_choices'] = {}
                for dd in (0., .1, .5, 1., 5.):
                    eligible = [r for r in leaves if -r['vector'][1] <= dd]
                    case['risk_budget_choices'][str(dd)] = max(eligible, key=lambda r: r['vector']) if eligible else None
                # Re-execute every conditional-frontier witness from a fresh native state.
                witnesses = {r['trace'] for f in case['conditional'].values() for r in f} | {best['trace']}
                for trace in sorted(witnesses):
                    actual = replay(risk, pattern, mirror, initial, trace, steps=steps)
                    expected = next(r for r in leaves if r['trace'] == trace)
                    if actual['vector'] != expected['vector']:
                        raise AssertionError('COLD_REPLAY_VECTOR')
                    max_error = max(max_error, abs(actual['cash_error']))
                    replay_count += 1
                    if trace == best['trace']:
                        case['max_return_ledger'] = actual['trades']
                payload = encoded(leaves)
                name = f'{pattern}_{mirror}_{initial}.json.gz'
                (output / name).write_bytes(gzip.compress(payload, mtime=0))
                case['all_terminal_payoffs_sha256'] = hashlib.sha256(payload).hexdigest()
                case['all_terminal_payoffs_file'] = name
                cases.append(case)
                print(pattern, mirror, initial, case['terminal_accounts'], 'complete', flush=True)
    report = dict(schema='exact_sequential_account_reference.v1',
        scope='RETROSPECTIVE_FINITE_SYNTHETIC_REFERENCE_NOT_FULL_TRAIN_NOT_CAUSAL',
        source_hashes=sources, cases=cases, steps=steps,
        terminal_accounts=sum(c['terminal_accounts'] for c in cases),
        nodes=sum(c['nodes'] for c in cases), cold_replays=replay_count,
        max_cash_error_usd=max_error, independent_frontier_check=True,
        current_deap_modified=False, feature_selection_authorized=False)
    (output / 'REPORT.json').write_bytes(encoded(report))
    lines = ['# Exact Sequential Account Reference Results', '',
        '2026-10-30. Retrospective synthetic calculation, not trading performance.', '',
        'See [derivation152](152_TWO_PHASE_SEQUENTIAL_PARETO_DERIVATION.md).', '',
        f'Complete finite trees: {len(cases)}; terminal accounts: {report["terminal_accounts"]}; '
        f'nodes: {report["nodes"]}; cold witness replays: {replay_count}.',
        f'Maximum cash conservation error: {max_error:.12g} USD. Independent pairwise dominance checks passed.', '',
        'All four saved horizon profiles, both directions, NOP and occupied CLOSE/HOLD are enumerated. '
        f'The {steps} decisions end at native weekly forced flat. Each case starts flat or with a 12h-profile BUY/SELL. '
        'Fees, bid/ask, funding, integer sizing and protections remain native. '
        'Drawdown is measured on hourly close equity, not unobserved intrabar equity extrema.', '',
        '## Flat Initial State', '',
        'The displayed member maximizes return within each finite tree; it is NOT the sole Pareto choice. '
        'Alternative return/DD points and conditional action frontiers are in REPORT.json.', '',
        '| Scenario | Net return % | DD % | Orders | BUY / SELL | Early closes | Best <=1 entry % | Best single horizon % |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for c in cases:
        if c['initial_direction']:
            continue
        ledger = c['max_return_ledger']
        buy = sum(t['direction'] == 1 for t in ledger)
        early = sum(t['reason'] == 'AGENT_CLOSE' for t in ledger)
        lines.append(f'| {c["pattern"]} / {c["mirror"]:+d} | {c["max_return"]["vector"][0]:.6f} | '
            f'{-c["max_return"]["vector"][1]:.6f} | {len(ledger)} | {buy} / {len(ledger)-buy} | {early} | '
            f'{c["best_return_at_most_one_entry"]:.6f} | {max(c["best_return_single_horizon"].values()):.6f} |')
    lines += ['', '## Initially Occupied', '',
        'Returns include the initial position, whose entry, 12h SL/TP and volume stay fixed. '
        'Subsequent entries can use any horizon. The two columns include optimal legal continuations '
        'after the first decision; they are not the isolated close-regret metric.', '',
        '| Scenario | Position | Best return after CLOSE % | Best return after HOLD % |',
        '|---|---|---:|---:|']
    for c in cases:
        if not c['initial_direction']:
            continue
        cv = max(r['vector'][0] for r in c['conditional']['CLOSE'])
        hv = max(r['vector'][0] for r in c['conditional']['HOLD'])
        lines.append(f'| {c["pattern"]} / {c["mirror"]:+d} | {"BUY" if c["initial_direction"] == 1 else "SELL"} | {cv:.6f} | {hv:.6f} |')
    lines += ['', '## Scope And Next Integration', '',
        'This calculates the exact frontier for these finite trees at fixed quality=1 and the saved risk recipe. '
        'It neither optimizes risk coefficients nor computes a four-year TRAIN frontier. '
        'Mirror rankings need not be identical: saved BUY/SELL geometry, funding and integer units differ. '
        'A real policy cannot choose using these known future paths; it needs a common action over '
        'indistinguishable report histories, then an explicit risk preference to select a Pareto member.', '',
        'Next: retain all K=3 TRAIN endpoint alternatives in an opportunity graph and qualify bounded exact '
        'native subproblems there; then qualify non-anticipative scenario-tree allocation against this oracle and the running '
        'local-score DEAP baseline. No live search was restarted or edited; no floating selection is released.', '',
        'Evidence: [REPORT.json](../audits/evidence/sequential_pareto_20261030/REPORT.json). '
        'The report identifies each complete terminal ledger file by digest for independent reconstruction.', '']
    (output / '153_SEQUENTIAL_PARETO_REFERENCE_RESULTS.md').write_text('\n'.join(lines))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    calculate(args.engine_root, args.output)
