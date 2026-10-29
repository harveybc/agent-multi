"""D: auditor fixtures; controller valuation of every common action vs the native engine (read-only)."""
import copy, json, sys
from dataclasses import replace
from pathlib import Path
AM, HS = sys.argv[1], sys.argv[2]
sys.path.insert(0, AM); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from app.zigzag_opportunities import engine
engine(HS)
from app.short_horizon import execution as X
from app.short_horizon.expert_guided.absolute import AbsoluteBarEnv, AbsoluteOrder
from app.short_horizon.expert_guided.env import EnvConfig
from app.short_horizon.expert_guided.data import calendar_from_closures
from app.absolute_exit_corpus import _funding
from app.close_hold_verification import build_report, position_view, HOUR
from app.close_hold_controller import value_paths, CloseController, CloseConfig
from audit_lib import independent_values

PIP = 1e-4


def make_env(direction, sl_pips, tp_pips, entry_hour, future, profile=None):
    """future: list of (o,h,l,c) mid bars for bars entry_hour+1 ... ; rest flat at the last close."""
    ts = pd.date_range('2020-01-05T22:00Z', periods=120, freq='h')
    bars = pd.DataFrame(dict(OPEN=1.1, HIGH=1.1, LOW=1.1, CLOSE=1.1), index=ts)
    last = 1.1
    for i, b in enumerate(future):
        bars.iloc[entry_hour + 1 + i] = list(b); last = b[3]
    for j in range(entry_hour + 1 + len(future), 120):
        bars.iloc[j] = [last] * 4
    cal = calendar_from_closures([['2020-01-03T22:00Z', '2020-01-05T22:00Z'],
                                  ['2020-01-10T22:00Z', '2020-01-12T22:00Z']], cutoff=0)
    profile = profile or X.RealisticProfile()
    cfg = EnvConfig(profile=profile, sizing=X.Sizing(rel_volume=.1, leverage=50), execution_price_mode='bid_ask_v3',
                    max_entries_per_day=None, min_orders_per_week=0)
    e = AbsoluteBarEnv(bars, cal, np.full(120, .001), np.full(120, .002), ts[0], ts[-1] + pd.Timedelta(hours=1), cfg,
                       tick=.00001)

    def plan(ctx):
        q = ctx.entry_quote
        r = lambda v: round(v / 1e-5) * 1e-5
        return AbsoluteOrder(r(q + direction * tp_pips * PIP), r(q - direction * sl_pips * PIP), .5, 1e9, 0.)
    while e.next_decision < entry_hour - 1:
        e.step(e.next_decision, 0)
    info = e.step(entry_hour - 1, 1 if direction == 1 else 2, entry_plan=plan)
    assert info['entry'] and e.pos is not None, info
    return e, info['absolute_order']['modeled_loss']


def native_menu(e):
    """Native net for: close at open j (after HOLD j times), and passive (HOLD forever)."""
    H = None; res = []; reasons = []
    r = build_report(e, 'ideal', 0); H = r.paths.shape[1]
    for j in range(H + 1):
        t = copy.deepcopy(e); n = 0
        while t.pos is not None:
            b = t.next_decision
            t.step(b, 3 if (j < H and n == j) else 0); n += 1
        res.append(float(t.trades[0]['net_usd'])); reasons.append(t.trades[0]['close_reason'])
    return res, reasons


def evaluate(name, e, risk, note):
    p = position_view(e, risk)
    r = build_report(e, 'ideal', 0)
    times = r.first_fill + np.arange(r.paths.shape[1]) * r.step
    fo = np.array([_funding(e, e.pos, int(x)) / e.pos.units for x in times])
    fe = np.array([_funding(e, e.pos, int(x) + HOUR) / e.pos.units for x in times])
    ctrl = value_paths(p, r, fo, fe)[0]
    ind = independent_values(p.direction, p.entry, p.units, p.sl, p.tp, p.entry_slip, p.market_slip, p.stop_slip,
                             p.commission_roundtrip, p.limit_through, r.paths.tolist(), r.spreads.tolist(),
                             fo.tolist(), fe.tolist())[0]
    nat, why = native_menu(e)
    # MIGRATION (2318b0e3): clock from the engine, deadline from the position's calendar, never from r.
    clock = int(e.v[e.next_decision]) + HOUR; deadline = int(e.ff[e.pos.seg])
    dec = CloseController(CloseConfig(), allow_synthetic=True).decide(p, r, clock, fo, fe, expected_deadline=deadline)
    err_c = [abs(a - b) for a, b in zip(ctrl, nat)]; err_i = [abs(a - b) for a, b in zip(ind, nat)]
    worst = int(np.argmax(err_c))
    return dict(fixture=name, note=note, direction=p.direction, units=p.units, entry=p.entry, sl=p.sl, tp=p.tp,
                risk=risk, H=len(nat) - 1, native_reason_per_action=why,
                native_net=nat[:6] + ['...'] + nat[-2:], controller_net=ctrl.tolist()[:6] + ['...'] + ctrl.tolist()[-2:],
                max_abs_ctrl_vs_native=max(err_c), worst_action=worst,
                worst_ctrl=float(ctrl[worst]), worst_native=nat[worst],
                max_abs_independent_vs_native=max(err_i),
                passive_ctrl=float(ctrl[-1]), passive_native=nat[-1],
                close_now_ctrl=float(ctrl[0]), close_now_native=nat[0], close_now_native_reason=why[0],
                immediate_decision=dec['action'], advantage_risk=dec['advantage_risk'],
                funding_open_nonzero=int((fo != 0).sum()), funding_end_nonzero=int((fe != 0).sum()))


rows = []
# F1 long, next-open gap through SL: decision at the tick whose fill bar gaps.
e, risk = make_env(1, 10, 20, 12, [(1.1005, 1.1006, 1.1004, 1.1005), (1.0980, 1.0981, 1.0979, 1.0980)])
e.step(e.next_decision, 0)  # now the next fill bar is the gap bar
rows.append(evaluate('F1_long_gap_through_SL_at_open', e, risk, 'next open bid below SL; CLOSE cannot pre-empt'))
# F2 short, next-open gap through TP.
e, risk = make_env(-1, 10, 15, 12, [(1.0995, 1.0996, 1.0994, 1.0995), (1.0975, 1.0976, 1.0974, 1.0975)])
e.step(e.next_decision, 0)
rows.append(evaluate('F2_short_gap_through_TP_at_open', e, risk, 'next open ask beyond TP; fills at open, better than TP'))
# F3 long, open above SL but the same bar's low touches SL: CLOSE at open must win.
e, risk = make_env(1, 10, 20, 12, [(1.1003, 1.1004, 1.1002, 1.1003), (1.0995, 1.0996, 1.0970, 1.0975)])
e.step(e.next_decision, 0)
rows.append(evaluate('F3_long_close_at_open_vs_same_bar_SL', e, risk, 'agent close at open precedes intrabar SL'))
# F3b short mirror with ambiguous bar (both SL and TP touched): stop must win.
e, risk = make_env(-1, 10, 10, 12, [(1.0998, 1.0999, 1.0997, 1.0998), (1.1004, 1.1030, 1.0960, 1.1000)])
e.step(e.next_decision, 0)
rows.append(evaluate('F3b_short_ambiguous_bar_stop_wins', e, risk, 'both barriers inside one bar'))
# F4 long crossing the Monday rollover (ts index 24 = rollover instant) then intrabar SL on Tuesday.
fut = [(1.1002, 1.1003, 1.1001, 1.1002)] * 14 + [(1.0998, 1.0999, 1.0960, 1.0965)]
e, risk = make_env(1, 10, 30, 12, fut)
rows.append(evaluate('F4_long_rollover_between_now_and_wait', e, risk, 'swap charged when exit_ns >= rollover'))
# F4b short crossing the Wednesday triple rollover (index 3) with flat path to forced flat after Fri? (flat to deadline)
fut = [(1.0999, 1.1000, 1.0998, 1.0999)] * 20
e, risk = make_env(-1, 30, 30, 60, fut)
rows.append(evaluate('F4b_short_triple_rollover_positive_swap', e, risk, 'short swap is positive; triple on index 3'))
# F5 forced-flat deadline with an hour-dependent spread that differs between the last bar open and the flat time.
prof = replace(X.RealisticProfile(), spread_schedule=((0, 12, .5e-4), (12, 24, 3e-4)), weekly_open_spread=.5e-4)
e, risk = make_env(1, 30, 30, 100, [(1.1001, 1.1002, 1.1000, 1.1001)] * 7, profile=prof)
rows.append(evaluate('F5_forced_flat_close_time_spread', e, risk,
                     'native FORCED_FLAT quotes the spread at bar close (v+1h); report carries bar-open spread'))
# F5c same as F5 with the default schedule (business calendar: last bar 09:00 -> flat 10:00, both in the 9-21h band).
e, risk = make_env(1, 30, 30, 100, [(1.1001, 1.1002, 1.1000, 1.1001)] * 7)
rows.append(evaluate('F5c_forced_flat_default_schedule', e, risk, 'default schedule; no spread boundary at the flat time'))
# F6 long TP intrabar exactly at limit_through boundary (bid high = TP + through) and just below.
for tag, extra in (('at', 0.), ('below', -1e-6)):
    e0, _ = make_env(1, 10, 10, 12, [])
    q = e0.pos.tp + 1e-4 + extra + 3e-5 / 2  # bid high = mid - spread/2 at 10:00+ (spread 0.3 pip)
    e, risk = make_env(1, 10, 10, 12, [(1.1002, 1.1003, 1.1001, 1.1002), (1.1004, q, 1.1003, 1.1004)])
    e.step(e.next_decision, 0)
    rows.append(evaluate(f'F6_long_TP_through_{tag}', e, risk, 'limit_through boundary'))
print(json.dumps(rows, indent=1, default=float))
