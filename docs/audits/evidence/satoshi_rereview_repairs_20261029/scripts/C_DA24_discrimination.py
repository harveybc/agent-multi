"""Run Musashi's 8 DA24 fixtures (his test assertion verbatim) against a given agent-multi tree, plus a
discriminating deadline-spread variant (schedule boundary at hour 12 = the flat instant).
Usage: python -I C_DA24_discrimination.py <am> <hs>"""
import json, sys
from dataclasses import replace
from pathlib import Path
AM, HS = sys.argv[1], sys.argv[2]; sys.path.insert(0, AM)
import numpy as np
from app.zigzag_opportunities import engine; engine(Path(HS))
from app.absolute_exit_corpus import create_case, specs, _funding
from app.close_hold_verification import build_report, position_view, HOUR
from app.close_hold_controller import value_paths
config = json.loads(Path(AM, 'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
out = []
for direction in (1, -1):
    for mode in ('gap_close', 'ambiguous', 'through', 'deadline_spread', 'deadline_spread_boundary12'):
        spec = dict(next(specs()), direction=direction)
        e, risk = create_case(spec, config)
        b = e.next_decision; k = b+1; p = e.pos; f = e.cfg.profile; mid = p.entry_price
        for name in ('o', 'h', 'l', 'c'):
            values = getattr(e, name).copy(); values[k:] = mid; setattr(e, name, values)
        if mode.startswith('deadline_spread'):
            sched = ((0, 10, .00005), (10, 24, .0004)) if mode == 'deadline_spread' else ((0, 12, .00005), (12, 24, .0004))
            f = replace(f, spread_schedule=sched); e.cfg = replace(e.cfg, profile=f)
        spread = f.spread_at(int(e.v[k]), int(e.mo[p.seg]))
        if mode == 'gap_close':
            for name in ('o', 'h', 'l', 'c'): getattr(e, name)[k:] = p.sl-direction*.001+direction*spread/2
        elif mode == 'ambiguous':
            e.h[k] = max(p.sl, p.tp)+.001+direction*spread/2; e.l[k] = min(p.sl, p.tp)-.001+direction*spread/2
        elif mode == 'through':
            edge = p.tp+direction*f.limit_through+direction*spread/2
            if direction == 1: e.h[k] = edge
            else: e.l[k] = edge
        r = build_report(e, 'ideal', 0)
        times = r.first_fill+np.arange(len(r.paths[0]))*HOUR
        fo = np.array([_funding(e, p, int(t))/p.units for t in times]); fe = np.array([_funding(e, p, int(t)+HOUR)/p.units for t in times])
        expected = value_paths(position_view(e, risk), r, fo, fe)[0, -1]
        e.step(b, 3 if mode == 'gap_close' else 0)
        while e.pos is not None: e.step(e.next_decision, 0)
        reason = {'gap_close': 'SL', 'ambiguous': 'SL', 'through': 'TP'}.get(mode, 'FORCED_FLAT')
        t = e.trades[0]
        last = int(np.searchsorted(e.v, e.ff[p.seg]))-1
        out.append(dict(mode=mode, direction=direction, native_reason=t['close_reason'], reason_ok=t['close_reason'] == reason,
                        native_net=t['net_usd'], controller_passive=float(expected), abs_err=abs(t['net_usd']-expected),
                        test_would_pass=bool(t['close_reason'] == reason and abs(t['net_usd']-expected) <= 1e-9),
                        last_bar_open_spread=f.spread_at(int(e.v[last]), int(e.mo[p.seg])),
                        flat_time_spread=f.spread_at(int(e.ff[p.seg]), int(e.mo[p.seg]))))
print(json.dumps(out, indent=1, default=float))
