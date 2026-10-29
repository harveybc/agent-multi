"""Re-review of F10/DA11, DA12 and DA13 against 2318b0e3, adapted from our E_adversarial.py.

Usage: AM_WORKTREE=<am> HS_ENGINE_ROOT=<hs> python -I C_F10_DA12_DA13.py <scripts-dir> <decompressed-run-dir>
Call-site changes vs E_adversarial.py are marked 'MIGRATION'. Everything else is the original witness.
"""
import json, sys, copy
from dataclasses import replace
import numpy as np
sys.path.insert(0, sys.argv[1]); from common import *
from app.dynamic_account_panel import World, panel_specs, evaluate, BASE, HOUR, report_identity
from app.close_hold_controller import (CloseController, CloseConfig, PathReport, PositionView, value_paths)
run = sys.argv[2]; front = json.loads((Path(run)/'FRONTIER.json').read_text())
G = {r['key'][:12]: r['genes'] for r in front['rows']}
out = {}


def attempt(f):
    try: return ('ACCEPTED', f())
    except Exception as x: return ('REFUSED', f'{type(x).__name__}:{x}')


# ---------------------------------------------------------------- F10 / DA11, unit level (no engine)
# Long, entry 100, SL 95, TP 120, R 5, zero costs. Full path to the true (calendar) deadline 13:
# close now 0, close at open1 -1, passive (deadline close 112) +12  -> HOLD.
# Truncated self-consistent report (2 bars, deadline 12): passive = close of bar 1 = 99 -> CLOSE.
p = PositionView('u', 1, 100., 1., 95., 120., 5., 0., 0., 0., 0., 0.)
def rep(prices, t=10, deadline=None, step=1, tag='causal'):
    a = np.asarray(prices, float)[None, :]
    paths = np.repeat(a[:, :, None], 4, axis=2)
    return PathReport(t, t, t, step, t + a.shape[1]*step if deadline is None else deadline, paths,
                      np.zeros(a.shape), np.ones(1), tag, np.zeros(1))
full = rep([100., 99., 112.]); trunc = rep([100., 99.])
z3, z2 = np.zeros(3), np.zeros(2)
CAL_DEADLINE = 13   # the position's calendar deadline, independent of any report
c = lambda: CloseController(CloseConfig(), allow_synthetic=True)
u10 = {}
u10['full_report_calendar_deadline'] = attempt(lambda: (lambda d: (d['action'], d['advantage_risk']))(c().decide(p, full, 10, z3, z3, expected_deadline=CAL_DEADLINE)))
u10['truncated_self_consistent_calendar_deadline'] = attempt(lambda: c().decide(p, trunc, 10, z2, z2, expected_deadline=CAL_DEADLINE)['action'])
# deadline field matches the calendar but the bars stop early
short_bars = rep([100., 99.], deadline=CAL_DEADLINE)
u10['deadline_matches_bars_stop_early'] = attempt(lambda: c().decide(p, short_bars, 10, z2, z2, expected_deadline=CAL_DEADLINE)['action'])
# full report, caller passes a mismatched expected deadline
u10['full_report_expected_deadline_minus_1'] = attempt(lambda: c().decide(p, full, 10, z3, z3, expected_deadline=12)['action'])
u10['full_report_expected_deadline_plus_1'] = attempt(lambda: c().decide(p, full, 10, z3, z3, expected_deadline=14)['action'])
# step-mismatched: same bar count but step 2 claims deadline 16
u10['report_step_2_vs_calendar_13'] = attempt(lambda: c().decide(p, rep([100., 99., 112.], step=2), 10, z3, z3, expected_deadline=CAL_DEADLINE)['action'])
# the same attacks through the cached synthetic adapter
st = rep([100., 99.], tag='synthetic_ideal'); vt = value_paths(p, st, z2, z2)
u10['cache_adapter_truncated_calendar_deadline'] = attempt(lambda: c().decide_synthetic_values('u', 10, 1, vt, st.weights, 5., report=st, expected_deadline=CAL_DEADLINE)['action'])
# residual misuse: caller sources the deadline from the report under test (contract violation, API cannot detect)
u10['MISUSE_truncated_with_expected_deadline_from_report'] = attempt(lambda: c().decide(p, trunc, 10, z2, z2, expected_deadline=trunc.deadline)['action'])
# value_paths alone (public helper) has no external-deadline argument
u10['value_paths_direct_truncated'] = attempt(lambda: value_paths(p, trunc, z2, z2).tolist())
out['F10_unit'] = u10

# ---------------------------------------------------------------- native world (E_adversarial setup)
w = World(panel_specs(43)[0], CFG); e = w.env
R = replays(run); P1 = [k for k in R if k.startswith('08fca639649c')][0]
tr = R[P1]['confirmation']['worlds'][0]['ledger'][0]; k = tr['entry_bar']; d = tr['direction']; b = k+3
s = .0003; E = e.o[k]+d*s/2
pos = type('P', (), dict(direction=d, entry_bar=k, entry_price=E, seg=0))()   # MIGRATION: seg (close_values reads e.pos.seg)
unit = w.position(tr['horizon'], d, E, k); pos.sl, pos.tp = unit.sl, unit.tp
def fake_env(position):   # MIGRATION: close_values now reads e.cfg and e.ff besides e.pos and e.v
    return type('E', (), {'pos': position, 'v': e.v, 'cfg': e.cfg, 'ff': e.ff})()
u, vals, fo, fe = w.close_values(fake_env(pos), tr['horizon'], b); r = w.report(b)
now = int(e.v[b])+HOUR; DL = int(e.ff[0])   # engine clock and calendar deadline
cc = lambda: CloseController(CloseConfig(.1, 0., 1, .25), allow_synthetic=True)
nw = {}
nw['full_route_current'] = attempt(lambda: cc().decide(u, r, now, fo, fe, expected_deadline=DL)['action'])
Hs = 3; rt = replace(r, paths=r.paths[:, :Hs], spreads=r.spreads[:, :Hs], terminal_spreads=r.terminal_spreads,
                     deadline=r.first_fill+Hs*HOUR)
nw['F10_truncated_report_native_world'] = attempt(lambda: cc().decide(u, rt, now, fo[:Hs], fe[:Hs], expected_deadline=DL)['action'])
rs_short = replace(rt, deadline=DL)
nw['F10_deadline_matches_bars_stop_early_native_world'] = attempt(lambda: cc().decide(u, rs_short, now, fo[:Hs], fe[:Hs], expected_deadline=DL)['action'])
nw['F10_full_report_mismatched_expected_deadline'] = attempt(lambda: cc().decide(u, r, now, fo, fe, expected_deadline=DL-HOUR)['action'])
out['F10_native'] = nw

# ---------------------------------------------------------------- DA12 stale / future on both routes
d12 = {}
rs = w.report(b-1); ts = rs.first_fill+np.arange(rs.paths.shape[1])*HOUR
fos = w.funding(int(e.v[k]), d, ts); fes = w.funding(int(e.v[k]), d, ts+HOUR)
rf = w.report(b+1); tf = rf.first_fill+np.arange(rf.paths.shape[1])*HOUR
fof = w.funding(int(e.v[k]), d, tf); fef = w.funding(int(e.v[k]), d, tf+HOUR)
d12['direct_stale_at_env_clock'] = attempt(lambda: cc().decide(u, rs, now, fos, fes, expected_deadline=DL)['action'])
d12['direct_future_at_env_clock'] = attempt(lambda: cc().decide(u, rf, now, fof, fef, expected_deadline=DL)['action'])
vs = value_paths(u, rs, fos, fes); vf = value_paths(u, rf, fof, fef)
d12['cache_adapter_stale_at_env_clock'] = attempt(lambda: cc().decide_synthetic_values(u.identity, now, HOUR, vs, rs.weights, u.risk, report=rs, expected_deadline=DL)['action'])
d12['cache_adapter_future_at_env_clock'] = attempt(lambda: cc().decide_synthetic_values(u.identity, now, HOUR, vf, rf.weights, u.risk, report=rf, expected_deadline=DL)['action'])
# cache adapter given CURRENT report but values computed from the stale one (values are not bound to the report)
d12['cache_adapter_current_report_with_stale_values'] = attempt(lambda: cc().decide_synthetic_values(u.identity, now, HOUR, vs[:, 1:], r.weights, u.risk, report=r, expected_deadline=DL)['action'])
d12['MISUSE_direct_stale_at_report_clock'] = attempt(lambda: cc().decide(u, rs, int(rs.issued_at), fos, fes, expected_deadline=int(rs.deadline))['action'])
# close_values with a stale report injected into the warm report cache
w2 = World(panel_specs(43)[0], CFG); w2.close_values(fake_env(pos), tr['horizon'], b)
w2.reports[b] = rs
d12['close_values_warm_cache_stale_report'] = attempt(lambda: w2.close_values(fake_env(pos), tr['horizon'], b) and 'served')
w2.reports[b] = rf
d12['close_values_warm_cache_future_report'] = attempt(lambda: w2.close_values(fake_env(pos), tr['horizon'], b) and 'served')
# entry bids route, warm cache
w3 = World(panel_specs(43)[0], CFG); fb = 20; w3.candidate_bids(fb); w3.reports[fb] = World(panel_specs(43)[0], CFG).report(fb-1)
d12['bids_warm_cache_stale_report'] = attempt(lambda: len(w3.candidate_bids(fb)))
w3.reports[fb] = World(panel_specs(43)[0], CFG).report(fb+1)
d12['bids_warm_cache_future_report'] = attempt(lambda: len(w3.candidate_bids(fb)))

# whole-panel runs (the 1.795 -> 1.881 witness), each route separately, verify False and True
class Stale(World):
    def report(self, b): return super().report(max(self.env.first, b-1))
class Future(World):
    def report(self, b): return super().report(min(self.end-2, b+1))
class StaleCloseOnly(World):          # stale only for the occupied (close) route, via close_values
    flag = False
    def report(self, b): return super().report(b-1) if self.flag else super().report(b)
    def close_values(self, e, h, b):
        self.flag = True
        try: return super().close_values(e, h, b)
        finally: self.flag = False
class StaleAdapterOnly(World):        # close_values sees the current report; the adapter/direct call sees b-1
    flag = False
    def report(self, b): return super().report(b-1) if self.flag else super().report(b)
    def close_values(self, e, h, b):
        out = super().close_values(e, h, b); self.flag = True; return out
    def candidate_bids(self, b):
        self.flag = False; return super().candidate_bids(b)
    def run(self, *a, **kw):
        self.flag = False; return super().run(*a, **kw)
class StaleBidsOnly(World):
    flag = False
    def report(self, b): return super().report(b-1) if self.flag else super().report(b)
    def candidate_bids(self, b):
        self.flag = True
        try: return super().candidate_bids(b)
        finally: self.flag = False
g1 = G['08fca639649c']
d12['panel_true_P1_conf_trend_up'] = World(panel_specs(43)[0], CFG).run(g1, verify=True)['return_pct']
for cls in (Stale, Future, StaleCloseOnly, StaleAdapterOnly, StaleBidsOnly):
    for v in (False, True):
        d12[f'panel_{cls.__name__}_verify_{v}'] = attempt(lambda: cls(panel_specs(43)[0], CFG).run(g1, verify=v)['return_pct'])
out['DA12'] = d12

# ---------------------------------------------------------------- DA13 cache binding
d13 = {}
def fresh_values(position, h, mutate=None):
    wf = World(panel_specs(43)[0], CFG)
    if mutate: mutate(wf)
    env = type('E', (), {'pos': position, 'v': wf.env.v, 'cfg': wf.env.cfg, 'ff': wf.env.ff})()
    return wf.close_values(env, h, b)[1]
wc = World(panel_specs(43)[0], CFG)
def warm_then(position, h, mutate=None, env_cfg_follow=True):
    global wc
    wc = World(panel_specs(43)[0], CFG)   # fresh World per case (no carry-over of earlier mutations)
    base = wc.close_values(type('E', (), {'pos': pos, 'v': e.v, 'cfg': wc.env.cfg, 'ff': e.ff})(), tr['horizon'], b)[1]
    if mutate: mutate(wc)
    env = type('E', (), {'pos': position, 'v': e.v, 'cfg': wc.env.cfg, 'ff': e.ff})()
    try:
        got = wc.close_values(env, h, b)[1]
    except Exception as x:
        return dict(result='REFUSED', error=f'{type(x).__name__}:{x}')
    try: ref = fresh_values(position, h, mutate)
    except Exception as x: ref = None
    return dict(result='SERVED', equals_warm_baseline=bool(np.array_equal(got, base)),
                equals_fresh_recompute=None if ref is None else bool(np.array_equal(got, ref)),
                fresh_differs_from_baseline=None if ref is None else bool(not np.array_equal(ref, base)))
# (a) changed entry price, original geometry kept on the position (E_adversarial witness)
pos2 = type('P', (), dict(direction=d, entry_bar=k, entry_price=E+.0010, sl=unit.sl, tp=unit.tp, seg=0))()
d13['hit_changed_entry_old_geometry'] = warm_then(pos2, tr['horizon'])
# (b) changed entry price by 1e-7 with geometry recomputed for the new entry (consistent position)
u3 = w.position(tr['horizon'], d, E+1e-7, k)
pos3 = type('P', (), dict(direction=d, entry_bar=k, entry_price=E+1e-7, sl=u3.sl, tp=u3.tp, seg=0))()
d13['hit_changed_entry_consistent_geometry'] = warm_then(pos3, tr['horizon'])
# (c) changed profile (horizon) with consistent geometry
h2 = 6 if tr['horizon'] != 6 else 3; u4 = w.position(h2, d, E, k)
pos4 = type('P', (), dict(direction=d, entry_bar=k, entry_price=E, sl=u4.sl, tp=u4.tp, seg=0))()
d13['hit_changed_profile'] = warm_then(pos4, h2)
# (d) changed side with consistent geometry
u5 = w.position(tr['horizon'], -d, E, k)
pos5 = type('P', (), dict(direction=-d, entry_bar=k, entry_price=E, sl=u5.sl, tp=u5.tp, seg=0))()
d13['hit_changed_side'] = warm_then(pos5, tr['horizon'])
# (e) changed funding schedule (rollover weights) on the same World
def m_fund(x): x.rolls = [(t, n*2) for t, n in x.rolls]
d13['hit_changed_funding'] = warm_then(pos, tr['horizon'], m_fund)
# (f) changed costs: market slippage in the World's execution config
def m_slip(x):
    f = x.env.cfg.profile; x.env.cfg = replace(x.env.cfg, profile=replace(f, slippage_market=f.slippage_market+.00001))
d13['hit_changed_market_slippage'] = warm_then(pos, tr['horizon'], m_slip)
# (g) changed costs: spread schedule (affects report spreads and terminal spread) -- report cache keyed only by bar
def m_spread(x):
    f = x.env.cfg.profile; x.env.cfg = replace(x.env.cfg, profile=replace(f, spread_schedule=((0, 24, .0006),), weekly_open_spread=.0006))
d13['hit_changed_spread_schedule'] = warm_then(pos, tr['horizon'], m_spread)
# (h) changed report content
def m_rep(x): x.reports[b] = replace(x.report(b), paths=x.report(b).paths+.0001)
d13['hit_changed_report_paths'] = warm_then(pos, tr['horizon'], m_rep)
# (i) env cfg differs from the World cfg
wc.cache.clear()
f0 = e.cfg.profile
envx = type('E', (), {'pos': pos, 'v': e.v, 'cfg': replace(wc.env.cfg, profile=replace(f0, slippage_stop=f0.slippage_stop+1e-5)), 'ff': e.ff})()
d13['env_cfg_differs_from_world_cfg'] = attempt(lambda: wc.close_values(envx, tr['horizon'], b) and 'served')
# (j) entry-bid cache with changed costs (bids keyed by (bar, report identity) only)
wb = World(panel_specs(43)[0], CFG); fb = 20
base_b = copy.deepcopy(wb.candidate_bids(fb)); m_slip(wb); got_b = wb.candidate_bids(fb)
wf = World(panel_specs(43)[0], CFG); m_slip(wf); ref_b = wf.candidate_bids(fb)
d13['bids_hit_changed_market_slippage'] = dict(equals_warm_baseline=got_b == base_b, equals_fresh=got_b == ref_b,
                                               fresh_differs=ref_b != base_b)
wb = World(panel_specs(43)[0], CFG); base_b = copy.deepcopy(wb.candidate_bids(fb)); m_spread(wb); got_b = wb.candidate_bids(fb)
wf = World(panel_specs(43)[0], CFG); m_spread(wf); ref_b = wf.candidate_bids(fb)
d13['bids_hit_changed_spread_schedule'] = dict(equals_warm_baseline=got_b == base_b, equals_fresh=got_b == ref_b,
                                               fresh_differs=ref_b != base_b)
out['DA13'] = d13
print(json.dumps(out, indent=1, default=str))
