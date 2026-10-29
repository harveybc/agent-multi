#!/usr/bin/env python3
"""Authenticate a finished crossed run and export paired, ledger-based diagnostics."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.crossed_account_panel import execution_specs, QUALITIES
from app.crossed_account_workers import ShardPool
from app.zigzag_opportunities import engine, sha
from tools.optimize_dynamic_account import Receipts, compact, digest
from tools.run_crossed_account_queue import load, pins, verify_completed


def skyline(vectors):
    # Independent two-objective sweep; equal vectors remain distinct policies.
    groups = defaultdict(list)
    for key, vector in vectors.items():
        groups[tuple(vector)].append(key)
    best = float('-inf')
    kept = set()
    for vector in sorted(groups, reverse=True):
        if vector[1] > best:
            kept.update(groups[vector])
            best = vector[1]
    return kept


def paired_clusters(a, b):
    groups = defaultdict(list)
    if len(a) != len(b):
        raise ValueError('PAIRED_SUPPORT')
    for x, y in zip(a, b):
        if x['spec'] != y['spec']:
            raise ValueError('PAIRED_SUPPORT')
        s = x['spec']
        groups[(s['seed'], s['pattern'])].append(x['return_pct'] - y['return_pct'])
    values = np.array([np.mean(v) for v in groups.values()])
    if not len(values):
        raise ValueError('EMPTY_SUPPORT')
    rng = np.random.default_rng(20261029)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return dict(clusters=len(values), mean_delta_pp=float(values.mean()),
                positive_clusters=int((values > 0).sum()),
                ci95_descriptive_pp=np.quantile(draws, [.025, .975]).tolist())


def details(result):
    worlds = result['worlds']
    mean_orders = float(np.mean([w['orders'] for w in worlds]))
    expected = [float(np.mean([w['return_pct'] for w in worlds])),
                -max(w['drawdown_pct'] for w in worlds)]
    if (result['objectives'] != expected or result['mean_orders'] != mean_orders or
            result['eligible'] != (mean_orders >= 3)):
        raise ValueError('AGGREGATE_METRICS')
    trades = [t for w in worlds for t in w['ledger']]
    for w in worlds:
        if (len(w['ledger']) != w['orders'] or
                sum(t['direction'] == 1 for t in w['ledger']) != w['buys'] or
                sum(t['net'] > 0 for t in w['ledger']) != w['wins'] or
                sum(t['direction'] == 1 and t['net'] > 0 for t in w['ledger']) != w['buy_wins'] or
                sum(t['direction'] == -1 and t['net'] > 0 for t in w['ledger']) != w['sell_wins'] or
                dict(Counter(t['reason'] for t in w['ledger'])) != w['closes']):
            raise ValueError('LEDGER_COUNTS')
        if abs(sum(t['net'] for t in w['ledger']) - w['net_usd']) > 1e-8:
            raise ValueError('LEDGER_CASH')
    return dict(**compact(result),
        closes=dict(Counter(t['reason'] for t in trades)),
        horizons=dict(Counter(str(t['horizon']) for t in trades)),
        mean_holding_bars=float(np.mean([t['exit_bar']-t['entry_bar'] for t in trades])) if trades else None,
        one_bar_trades=sum(t['exit_bar']-t['entry_bar'] == 1 for t in trades),
        losing_early_closes=sum(t['reason'] == 'AGENT_CLOSE' and t['net'] < 0 for t in trades),
        mean_costs_usd={k: float(np.mean([w['costs'][k] for w in worlds])) for k in worlds[0]['costs']},
        zero_order_worlds=sum(w['orders'] == 0 for w in worlds),
        under_three_order_worlds=sum(w['orders'] < 3 for w in worlds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--engine-root', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--replay-p1', action='store_true')
    args = parser.parse_args()
    run, out = args.run, args.export
    if run.resolve() == out.resolve():
        raise ValueError('EXPORT_MUST_NOT_MODIFY_RUN')
    manifest = json.loads((run/'MANIFEST.json').read_text())
    body = dict(manifest)
    identity = body.pop('identity')
    if digest(body) != identity or pins(args.engine_root) != manifest['sources']:
        raise ValueError('MANIFEST_OR_SOURCE')
    config = ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    if sha(config) != manifest['config_sha256']:
        raise ValueError('CONFIG')
    complete = load(run/'COMPLETE.json', identity)
    verify_completed(run, identity, complete, manifest['seeds'])
    receipts = Receipts(run/'CANDIDATES.jsonl', identity)
    genes = {r['key']: r['genes'] for r in map(json.loads, (run/'CANDIDATES.jsonl').read_text().splitlines())}
    count = 0
    for seed in manifest['seeds']:
        history = json.loads((run/f'SEED_{seed}.json').read_text())['history']
        if len(history) != 300:
            raise ValueError('GENERATION_COUNT')
        for step in history:
            if len(step['genes']) != 20 or len(step['fitness']) != 20:
                raise ValueError('POPULATION')
            for g, fitness in zip(step['genes'], step['fitness']):
                r = receipts.rows[digest(g)]
                if fitness != (r['objectives'] if r['eligible'] else [-1e6, -1e6]):
                    raise ValueError('HISTORY_FITNESS')
                count += 1
    front = load(run/'FRONTIER.json', identity)
    vectors = {k: r['objectives'] for k, r in receipts.rows.items() if r['eligible']}
    selected = skyline(vectors)
    if selected != set(front['keys']):
        raise ValueError('INDEPENDENT_FRONTIER')
    groups = defaultdict(list)
    for k in selected:
        groups[tuple(vectors[k])].append(k)
    ordered = sorted(groups, reverse=True)
    ranks = sorted({round(x) for x in np.linspace(0, len(ordered)-1, min(5, len(ordered)))})
    reps = [min(groups[ordered[i]], key=lambda k: tuple(genes[k])) for i in ranks]
    if reps != front['representatives']:
        raise ValueError('REPRESENTATIVE_RULE')
    for item in complete['controls']:
        control = load(run/f"CONTROL_{item['quality']}_{item['name']}.json", identity)
        if {k: v for k, v in control.items() if k != 'worlds'} != item:
            raise ValueError('CONTROL_SUMMARY')
    archive, summaries = {}, {}
    for item in complete['confirmation']:
        q, name = item['quality'], item['name']
        final = load(run/f'FINAL_{q}_{name}.json', identity)
        result = json.loads(gzip.decompress((run/f'REPLAY_{q}_{name}.json.gz').read_bytes()))
        if final != item or any(final[k] != v for k, v in compact(result).items()):
            raise ValueError('FINAL_SUMMARY')
        if digest([w['trace'] for w in result['worlds']]) != final['action_signature']:
            raise ValueError('TRACE_SIGNATURE')
        archive[q, name] = result
        summaries[q+' / '+name] = details(result)
    p1 = front['representatives'][0]
    comparisons, size_checks = {}, {}
    for q in QUALITIES:
        comparisons[q] = {name: paired_clusters(archive[q, p1]['worlds'], result['worlds'])
            for (quality, name), result in archive.items() if quality == q and len(name) < 20}
        for name in front['representatives']:
            if (q, name+'_size1') in archive:
                a, b = archive[q, name]['worlds'], archive[q, name+'_size1']['worlds']
                exact = all(x['spec'] == y['spec'] and x['trace'] == y['trace'] for x, y in zip(a, b))
                size_checks[q+' / '+name] = exact
    fresh = 0
    ablation = None
    if args.replay_p1:
        engine(args.engine_root)
        for q in QUALITIES:
            pool = ShardPool(execution_specs('confirmation', q, False), json.loads(config.read_text()), args.engine_root, 4)
            try:
                result = pool.evaluate(genes[p1], verify=True, details=True)
                if result != archive[q, p1]:
                    raise ValueError('FRESH_P1_REPLAY')
                fresh += len(result['worlds'])
                print('EXACT_P1_REPLAY', q, fresh, flush=True)
                if q == 'correlated':
                    diagnostic = pool.evaluate(genes[p1], {(6, 1), (6, -1)}, verify=True, details=True)
                    ablation = dict(label='POST_HOC_NOT_SELECTION_P1_ONLY6_SAME_GENES',
                        result=diagnostic, summary=details(diagnostic),
                        p1_minus_only6=paired_clusters(result['worlds'], diagnostic['worlds']))
            finally:
                pool.close()
    out.mkdir(parents=True, exist_ok=True)
    exported = []
    if ablation is not None:
        dest = out/'POST_HOC_P1_ONLY6.json.gz'
        dest.write_bytes(gzip.compress(json.dumps(ablation, sort_keys=True, allow_nan=False).encode(), mtime=0))
        exported.append(dict(file=dest.name, sha256=sha(dest)))
    patterns = ['MANIFEST.json', 'COMPLETE.json', 'FRONTIER.json', 'TESTS.json', 'SEED*.json',
                'CANDIDATES.jsonl', 'CONTROL_*.json', 'FINAL_*.json', 'REPLAY_*.gz', 'RESULTS.md', 'CONTROLS.md']
    for path in sorted({p for pattern in patterns for p in run.glob(pattern)}):
        dest = out/path.name
        if path.name.startswith('SEED_') and not path.name.startswith('SEED_DONE') or path.suffix == '.jsonl':
            dest = dest.with_name(dest.name+'.gz')
            dest.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        else:
            shutil.copyfile(path, dest)
        exported.append(dict(file=dest.name, sha256=sha(dest), original_file=path.name, original_sha256=sha(path)))
    report = dict(identity=identity, analysis_source_sha256=sha(Path(__file__)),
        authenticated_candidates=len(receipts.rows), generation_fitness_checked=count,
        independent_frontier_policies=len(selected), unique_vectors=len({tuple(vectors[k]) for k in selected}),
        summaries=summaries, paired_p1_vs_controls=comparisons, fixed_size_trace_checks=size_checks,
        fresh_native_p1_replays_exact=fresh, archived_accounts_checked=sum(len(r['worlds']) for r in archive.values()),
        post_hoc_p1_only6=None if ablation is None else {k: v for k, v in ablation.items() if k != 'result'},
        inference_scope='Descriptive paired cluster bootstrap; 30 draws, no multiplicity correction, synthetic only',
        files=exported)
    (out/'ANALYSIS.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if isinstance(v, (str, int))}), flush=True)


if __name__ == '__main__':
    main()
