"""G: prospectively declared biased / missing-mode controls (see G_PROTOCOL_PROSPECTIVE.md)."""
import gzip, json, sys
from dataclasses import asdict, replace
from pathlib import Path
AM, HS = sys.argv[1], sys.argv[2]
sys.path.insert(0, AM)
import numpy as np
from app.zigzag_opportunities import engine
engine(HS)
from app.absolute_exit_corpus import create_case, _funding, specs
from app.close_hold_verification import build_report, position_view, HOUR
from app.close_hold_controller import CloseConfig, CloseController
from app.protection_reference import protection_exit
from app.early_close_metrics import evaluate_exit, summarize_exits

config = json.loads(Path(AM, 'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
refs = {r['spec']['id']: r for r in json.loads(gzip.decompress(
    Path(AM, 'docs/audits/evidence/absolute_early_close_20261029/CASES.json.gz').read_bytes()))}
committed = json.loads(gzip.decompress(Path(AM, 'docs/audits/evidence/close_hold_controller_20261029/CASES.json.gz').read_bytes()))
committed = {(r['spec']['id'], r['arm']): r for r in committed}
CFG = dict(immediate=CloseConfig(), guarded=CloseConfig(uncertainty_penalty=1., margin_risk=.02, persistence=2, urgent_risk=.25))
stats = dict(mean_preservation_max_abs=0., missing_all_sl_ticks=0, missing_partial_ticks=0, missing_none_ticks=0)


def sample_ends_sl(p, path, spread):
    q = path - p.direction * spread[:, None] / 2
    return protection_exit(p.direction, p.tp, p.sl, *q.T, p.limit_through)[2] == 'SL'


def family_report(e, p, fam, seed):
    base_kind = 'ideal' if fam == 'IDEAL' or '_IDEAL_' in fam else 'noisy'
    r = build_report(e, base_kind, seed)
    if r is None:
        return None
    if base_kind == 'noisy':
        truth = build_report(e, 'ideal', seed).paths[0]
        stats['mean_preservation_max_abs'] = max(stats['mean_preservation_max_abs'],
                                                 float(np.max(np.abs(np.tensordot(r.weights, r.paths, 1) - truth))))
    if fam in ('IDEAL', 'NOISY'):
        return r
    if fam.startswith('BIAS'):
        k = float(fam.split('_k')[1]); sign = 1. if '_FAV_' in fam else -1.
        h = np.arange(1, r.paths.shape[1] + 1, dtype=float)
        if fam.startswith('BIAS_SL_'):
            shift = sign * p.direction * k * abs(p.entry - p.sl) * np.minimum(1., h / 6)
        else:
            shift = sign * p.direction * k * 1e-4 * np.sqrt(h)
        return replace(r, paths=r.paths + shift[None, :, None])
    ends = np.array([sample_ends_sl(p, x, s) for x, s in zip(r.paths, r.spreads)])
    if fam == 'MISSING_DROP':
        if ends.all():
            stats['missing_all_sl_ticks'] += 1; return 'EMPTY'
        stats['missing_partial_ticks' if ends.any() else 'missing_none_ticks'] += 1
        keep = ~ends; w = r.weights[keep] / r.weights[keep].sum()
        return replace(r, paths=r.paths[keep], spreads=r.spreads[keep], weights=w)
    if fam == 'MISSING_UNDERWEIGHT':
        w = r.weights * np.where(ends, .25, 1.); return replace(r, weights=w / w.sum())
    raise ValueError(fam)


def run(spec, fam, cfg_name):
    e, risk = create_case(spec, config)
    c = CloseController(CFG[cfg_name], allow_synthetic=True); p = position_view(e, risk)
    while e.pos is not None:
        b = e.next_decision
        r = family_report(e, p, fam, spec['seed'])
        if r is None or r == 'EMPTY':
            action = 'HOLD'
        else:
            times = r.first_fill + np.arange(r.paths.shape[1]) * r.step
            fo = np.array([_funding(e, e.pos, int(t)) / e.pos.units for t in times])
            fe = np.array([_funding(e, e.pos, int(t) + HOUR) / e.pos.units for t in times])
            action = c.decide(p, r, r.issued_at, fo, fe)['action']
        e.step(b, 3 if action == 'CLOSE' else 0)
    t = e.trades[0]; ref = refs[spec['id']]
    ix = ref['fill_bars'].index(t['exit_bar_abs']) if t['close_reason'] == 'AGENT_CLOSE' else None
    return evaluate_exit([v['close_net'] for v in ref['tick_metrics']], ref['native_passive_net'], risk, ix,
                         actual_net=t['net_usd'])


sample = list(specs())[::9]
fams = [f'BIAS_SL_{d}_{b}_k{k}' for d in ('FAV', 'ADV') for b in ('IDEAL', 'NOISY') for k in (0.25, 0.5)] if len(sys.argv) > 3 else ['IDEAL', 'NOISY', 'BIAS_FAV_IDEAL_k0.5', 'BIAS_FAV_IDEAL_k1', 'BIAS_FAV_NOISY_k0.5', 'BIAS_FAV_NOISY_k1',
        'BIAS_ADV_NOISY_k0.5', 'BIAS_ADV_NOISY_k1', 'MISSING_DROP', 'MISSING_UNDERWEIGHT']
table = []; paired_ok = True
for fam in fams:
    for cfg in ('immediate', 'guarded'):
        rows = [run(s, fam, cfg) for s in sample]
        if fam in ('IDEAL', 'NOISY'):
            arm = f"{fam.lower()}_{cfg}"
            paired_ok &= all(abs(m['actual_net_usd'] - committed[(s['id'], arm)]['metrics']['actual_net_usd']) < 1e-9
                             for s, m in zip(sample, rows))
        sm = summarize_exits(rows)
        table.append(dict(family=fam, config=cfg, positions=sm['positions'], early_closes=sm['early_closes'],
                          destructive_closes=sm['destructive_closes'], missed_optimal_exits=sm['missed_opportunities'],
                          never_closed_with_opportunity=sm['missed_without_closing'],
                          captured_improvement=sm['aggregate_capture'], regret_risk_mean=sm['regret_risk_mean']))
modes = {}
for s in sample: modes[s['mode']] = modes.get(s['mode'], 0) + 1
print(json.dumps(dict(sample_positions=len(sample), sample_modes=modes, baseline_matches_committed_cases=paired_ok,
                      stats=stats, table=table), indent=1))
