"""Re-review: F07 exact boundary (pure-function sweep + extra native fixtures) and DA24 fixtures on the full menu.

Usage: AM_ROOT=<am> HS_ROOT=<hs> python -I C_F07_DA24.py
"""
import copy, json, os, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fixtures_lib import *  # noqa: F401,F403  (make_env, evaluate, native_menu, AM, HS)
import numpy as np
from app.short_horizon import execution as X
from app.protection_reference import protection_exit
from app.absolute_exit_corpus import create_case, specs
from app.close_hold_verification import build_report, position_view, HOUR

out = {}

# 1. Pure-function sweep: protection_exit (reference) vs native X._protect on exact, +1ulp and -1ulp boundaries.
rng = np.random.default_rng(20261029)
mism = 0; n = 0; examples = []
for _ in range(20000):
    d = int(rng.choice([1, -1]))
    tp = round(float(rng.uniform(0.6, 1.9)) / 1e-5) * 1e-5
    through = float(rng.choice([0., 1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 1.7e-5]))
    sl = tp - d * 0.01
    o = tp - d * 0.002
    edge = tp + through if d == 1 else tp - through
    for fav in (edge, np.nextafter(edge, np.inf), np.nextafter(edge, -np.inf)):
        h, l = (max(o, fav), min(o, fav)) if d == 1 else (max(o, fav), min(o, fav))
        if d == -1:
            h, l = max(o, fav), min(o, fav)
        nat = X._protect(d, tp, sl, o, h, l, through)
        nat_reason = None if nat is None else nat[0]
        ref = protection_exit(d, tp, sl, [o], [h], [l], [o], through)[2]
        ref_reason = None if ref == 'DEADLINE' else ref
        n += 1
        if nat_reason != ref_reason:
            mism += 1
            if len(examples) < 5:
                examples.append(dict(d=d, tp=tp, through=through, fav=fav, native=nat_reason, reference=ref_reason))
out['F07_pure_sweep'] = dict(cases=n, mismatches=mism, examples=examples)
if os.environ.get('SWEEP_ONLY'):
    print(json.dumps(out, indent=1, default=float)); sys.exit(0)

# 2. Extra native fixtures: short INSIDE the boundary (our earlier 'F6s below' was actually 1e-6 BEYOND),
#    and long BEYOND by 1e-6, plus a tick-grid exact case where tp+through is representable on the 1e-5 grid.
rows = []
for d, tag, extra in ((-1, 'inside', -1e-6), (1, 'beyond', 1e-6)):
    e0, _ = make_env(d, 10, 10, 12, [])
    if d == 1:
        q = e0.pos.tp + 1e-4 + extra + 3e-5 / 2
        fut = [(1.1002, 1.1003, 1.1001, 1.1002), (1.1004, q, 1.1003, 1.1004)]
    else:
        q = e0.pos.tp - 1e-4 - extra - 3e-5 / 2
        fut = [(1.0998, 1.0999, 1.0997, 1.0998), (1.0996, 1.0997, q, 1.0996)]
    e, risk = make_env(d, 10, 10, 12, fut)
    e.step(e.next_decision, 0)
    r = evaluate(f'F6x_{"long" if d == 1 else "short"}_TP_through_{tag}', e, risk, 'extra boundary side')
    k = e.next_decision + 1
    hi = e.quote_price(k, e.h[k], -d); lo = e.quote_price(k, e.l[k], -d)
    r['quote_extreme_minus_boundary'] = (hi - (e.pos.tp + e.cfg.profile.limit_through)) if d == 1 else (lo - (e.pos.tp - e.cfg.profile.limit_through))
    rows.append(r)
out['F07_extra_native'] = rows

# 3. DA24: Musashi's eight fixtures (construction copied from tests/unit/test_dynamic_account_panel.py), checked
#    on EVERY common action against the native engine with my menu, not only the passive value.
config = json.loads(Path(AM, 'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
da24 = []
for direction in (1, -1):
    for mode in ('gap_close', 'ambiguous', 'through', 'deadline_spread'):
        spec = dict(next(specs()), direction=direction)
        e, risk = create_case(spec, config)
        b = e.next_decision; k = b + 1; p = e.pos; f = e.cfg.profile
        mid = p.entry_price
        for name in ('o', 'h', 'l', 'c'):
            values = getattr(e, name).copy(); values[k:] = mid; setattr(e, name, values)
        if mode == 'deadline_spread':
            f = replace(f, spread_schedule=((0, 10, .00005), (10, 24, .0004)))
            e.cfg = replace(e.cfg, profile=f)
        spread = f.spread_at(int(e.v[k]), int(e.mo[p.seg]))
        if mode == 'gap_close':
            for name in ('o', 'h', 'l', 'c'): getattr(e, name)[k:] = p.sl - direction * .001 + direction * spread / 2
        elif mode == 'ambiguous':
            e.h[k] = max(p.sl, p.tp) + .001 + direction * spread / 2
            e.l[k] = min(p.sl, p.tp) - .001 + direction * spread / 2
        elif mode == 'through':
            edge = p.tp + direction * f.limit_through + direction * spread / 2
            if direction == 1: e.h[k] = edge
            else: e.l[k] = edge
        row = evaluate(f'DA24_{mode}_{"long" if direction == 1 else "short"}', e, risk, 'Musashi fixture, full menu')
        if mode == 'through':
            q = e.quote_price(k, e.h[k] if direction == 1 else e.l[k], -direction)
            bound = p.tp + f.limit_through if direction == 1 else p.tp - f.limit_through
            row['quote_extreme_minus_boundary'] = q - bound
        if mode == 'deadline_spread':
            r = build_report(e, 'ideal', 0)
            last = int(np.searchsorted(e.v, e.ff[p.seg])) - 1
            row['bar_open_spread_last_bar'] = f.spread_at(int(e.v[last]), int(e.mo[p.seg]))
            row['terminal_spread_in_report'] = float(r.terminal_spreads[0])
        # Musashi's own assertion target: native reason of the passive / requested first action.
        t = copy.deepcopy(e)
        t.step(b, 3 if mode == 'gap_close' else 0)
        while t.pos is not None: t.step(t.next_decision, 0)
        row['musashi_scenario_reason'] = t.trades[0]['close_reason']
        da24.append(row)
out['DA24'] = da24
print(json.dumps(out, indent=1, default=float))
