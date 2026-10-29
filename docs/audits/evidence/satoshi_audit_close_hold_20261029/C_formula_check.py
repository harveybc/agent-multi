"""C: recompute advantage / decision from scratch on stored CASES traces (read-only)."""
import gzip, json, sys
from pathlib import Path
AM, HS = sys.argv[1], sys.argv[2]
sys.path.insert(0, AM); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from app.zigzag_opportunities import engine
engine(HS)
from app.absolute_exit_corpus import create_case, _funding
from app.close_hold_verification import build_report, position_view, arms, HOUR
from app.close_hold_controller import CloseConfig
from audit_lib import independent_values, independent_advantage, IndependentPersistence

ev = Path(AM, 'docs/audits/evidence/close_hold_controller_20261029')
cases = json.loads(gzip.decompress((ev / 'CASES.json.gz').read_bytes()))
config = json.loads(Path(AM, 'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
armcfg = {a['name']: a for a in arms()}
# Deterministic choice: first long 'gain', first short 'loss', first long 'tp' with 24h horizon.
want = [dict(mode='gain', direction=1), dict(mode='loss', direction=-1), dict(mode='tp', direction=1, horizon=24)]
chosen = []
for w in want:
    chosen.append(next(r['spec']['id'] for r in cases if r['arm'] == 'noisy_guarded'
                       and all(r['spec'][k] == v for k, v in w.items())))
out = dict(chosen=chosen, rows=[], max_abs_adv_err=0., max_abs_close_err=0., max_abs_wait_err=0.,
           decision_mismatch=0, ticks=0)
ALL = len(sys.argv) > 3
if ALL: chosen = sorted({r['spec']['id'] for r in cases})
for cid in chosen:
    for arm in (tuple(armcfg) if ALL else ('noisy_guarded', 'ideal_guarded', 'noisy_immediate')):
        rec = next(r for r in cases if r['arm'] == arm and r['spec']['id'] == cid)
        e, risk = create_case(rec['spec'], config)
        p = position_view(e, risk)
        cfg = CloseConfig(**armcfg[arm]['config'])
        pers = IndependentPersistence(cfg.margin_risk, cfg.persistence, cfg.urgent_risk)
        for t in rec['trace']:
            assert e.next_decision == t['decision_bar']
            r = build_report(e, armcfg[arm]['report'], rec['spec']['seed'])
            if r is None:
                assert t.get('reason') == 'MANDATORY_DEADLINE'; e.step(t['decision_bar'], 0); continue
            times = r.first_fill + np.arange(r.paths.shape[1]) * r.step
            fo = [_funding(e, e.pos, int(x)) / e.pos.units for x in times]
            fe = [_funding(e, e.pos, int(x) + HOUR) / e.pos.units for x in times]
            vals = independent_values(p.direction, p.entry, p.units, p.sl, p.tp, p.entry_slip, p.market_slip,
                                      p.stop_slip, p.commission_roundtrip, p.limit_through,
                                      r.paths.tolist(), r.spreads.tolist(), fo, fe)
            w = r.weights.tolist()
            adv, per = independent_advantage(vals, w, cfg.uncertainty_penalty, p.risk)
            close_net = sum(wi * v[0] for wi, v in zip(w, vals))
            wait_best = max(sum(wi * v[j] for wi, v in zip(w, vals)) for j in range(1, len(vals[0])))
            d = pers.step(p.identity, r.issued_at, r.step, adv)
            ea = abs(adv - t['advantage_risk']); ec = abs(close_net - t['expected_close_net'])
            ew = abs(wait_best - t['best_expected_wait_net'])
            mism = any(d[k] != t[k] for k in ('qualifies', 'urgent', 'streak', 'action'))
            out['max_abs_adv_err'] = max(out['max_abs_adv_err'], ea)
            out['max_abs_close_err'] = max(out['max_abs_close_err'], ec)
            out['max_abs_wait_err'] = max(out['max_abs_wait_err'], ew)
            out['decision_mismatch'] += mism; out['ticks'] += 1
            if not ALL: out['rows'].append(dict(case=cid, arm=arm, bar=t['decision_bar'], samples=len(w),
                                    stored_adv=t['advantage_risk'], audit_adv=adv, abs_err=ea,
                                    argmin_wait_action=int(np.argmin(per)) + 1, action=t['action'],
                                    audit_action=d['action'], streak=t['streak'], urgent=t['urgent']))
            e.step(t['decision_bar'], 3 if t['action'] == 'CLOSE' else 0)
        assert e.pos is None and e.trades[0]['close_reason'] == rec['native_reason']
out['tolerance'] = 1e-12
out['pass'] = out['max_abs_adv_err'] <= 1e-12 and out['decision_mismatch'] == 0
out.pop('chosen') if ALL else None
print(json.dumps(out, indent=1))
