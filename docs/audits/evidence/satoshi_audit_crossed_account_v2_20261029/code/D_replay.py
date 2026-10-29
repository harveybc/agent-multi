"""D+F+RR-N1: fresh native re-execution of every archived confirmation account of one quality rung,
warm (one World per cell, all policies, as the queue) and cold (fresh World per policy, every 10th
cell); independent accounting (cash, costs, financing, units, margin, gates, MTM DD) and report
clock/deadline/spread checks; size=1 trace comparison."""
import json, math, sys, os, time, gzip
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from native_lib import *
H = 3600*10**9
q = sys.argv[1]
names_ = names()
pol = policies(FRONT['rows'], REPS)
assert set(names_) <= set(pol), (set(names_)-set(pol))
arch = {n: replay(q, n)['worlds'] for n in names_}
if q == 'correlated':
    arch['P1only6'] = posthoc()['worlds']; pol['P1only6'] = (pol[REPS[0]][0], {(6, 1), (6, -1)}); names_ = names_+['P1only6']
specs = execution_specs('confirmation', q, False)[:int(os.environ.get('LIMIT', '360'))]
SW = dict(slip_m=.2e-4, slip_s=.5e-4, through=1e-4, comm=3.5, lot=1e5, swap_l=-7., swap_s=2.5, cap=1e4)

def my_spread(t, o, s):
    since = (t-o)/H
    if since < 1: return s
    hr = since % 24
    return 2*s if 10 <= hr < 11 else s

def my_protect(d, tp, sl, o, h, l, through):
    if d > 0:
        if o <= sl: return 'SL', o, True
        if o >= tp: return 'TP', o, True
        hs, ht = l <= sl, h >= tp+through
    else:
        if o >= sl: return 'SL', o, True
        if o <= tp: return 'TP', o, True
        hs, ht = h >= sl, l <= tp-through
    if hs: return 'SL', sl, False
    if ht: return 'TP', tp, False
    return None

def account_check(spec, e, led, arch_w):
    """Independent re-derivation from generated bars + declared constants."""
    s = spec['spread_pips']*1e-4
    X = market(spec); o, h, l, c = X.T
    v = e.v; mo = int(e.mo[0]); ff = int(e.ff[0])
    rolls = [(mo+k*24*H, 3 if k == 3 else 1) for k in range(1, 10) if mo+k*24*H < mo+120*H]
    err = dict(bars=0, entry=0, units=0, margin=0, exitpx=0, path=0, cost=0, fin=0, net=0, gate=0, overlap=0, ledger=0, ret=0, dd=0)
    if not (np.array_equal(o, e.o) and np.array_equal(c, e.c) and np.array_equal(h, e.h) and np.array_equal(l, e.l)): err['bars'] += 1
    B = SW['cap']; rows = []; prev_exit = -1
    for t, a in zip(e.trades, led):
        d = t['direction']; k = t['entry_bar_abs']; ek = t['exit_bar_abs']; u = t['units']
        if (a['entry_bar'], a['exit_bar'], a['direction'], a['units'], a['reason']) != (k, ek, d, u, t['close_reason']) or abs(a['net']-t['net_usd']) > 1e-9: err['ledger'] += 1
        if not (v[k]-mo >= 12*H and v[k] < ff and ek <= 107 and k > prev_exit): err['gate'] += 1
        if k <= prev_exit: err['overlap'] += 1
        prev_exit = ek
        eq = o[k]+d*my_spread(v[k], mo, s)/2
        if abs(eq-t['entry_price']) > 1e-12: err['entry'] += 1
        qq = a['q']
        mu = int(math.floor(qq*int(math.floor(B*.1*50/eq))))   # floor(q*units_max) as Sizing.units
        mu2 = int(math.floor(qq*.1*B*50/eq))                    # floor(q*0.1*B*50/p) closed form
        if mu != u: err['units'] += 1
        margin = u*eq/50
        if margin > qq*.1*B+1e-9 or margin > .1*B+1e-9: err['margin'] += 1
        tp, sl = t['take_profit'], t['stop_loss']
        # first passage, bid/ask quotes with bar-open spread
        hit = None
        for j in range(k, ek+1):
            sj = my_spread(v[j], mo, s)
            oq, hq, lq = o[j]-d*sj/2, h[j]-d*sj/2, l[j]-d*sj/2
            r = my_protect(d, tp, sl, oq, hq, lq, SW['through'])
            if r is not None: hit = (j,)+r; break
        reason = t['close_reason']
        if reason in ('TP', 'SL'):
            if hit is None or hit[0] != ek or hit[1] != reason: err['path'] += 1; continue
            px = hit[2]; at_open = hit[3]; exit_ns = int(v[ek]) + (0 if at_open else H)
            cost_t = int(v[ek])
        else:
            if hit is not None and (hit[0] < ek or (hit[0] == ek and reason == 'AGENT_CLOSE' and hit[3])): err['path'] += 1
            if reason == 'AGENT_CLOSE': exit_ns = int(v[ek]); px = o[ek]-d*my_spread(v[ek], mo, s)/2
            elif reason == 'FORCED_FLAT':
                exit_ns = int(v[ek])+H; px = c[ek]-d*my_spread(v[ek]+H, mo, s)/2
                if ek != 107: err['gate'] += 1
            else: err['path'] += 1; continue
            cost_t = exit_ns
        if abs(px-t['exit_price']) > 1e-12: err['exitpx'] += 1
        sp_cost = u*(my_spread(v[k], mo, s)+my_spread(cost_t, mo, s))/2
        slip = u*(SW['slip_m']+{'SL': SW['slip_s'], 'TP': 0.}.get(reason, SW['slip_m']))
        comm = 2*u/SW['lot']*SW['comm']
        rate = -(SW['swap_l'] if d == 1 else SW['swap_s'])/SW['lot']
        fin = sum(rate*u*m for r, m in rolls if v[k] < r <= exit_ns)
        if max(abs(sp_cost-t['spread_usd']), abs(slip-t['slippage_usd']), abs(comm-t['commission_usd'])) > 1e-8: err['cost'] += 1
        if abs(fin-t['financing_usd']) > 1e-8: err['fin'] += 1
        net = d*u*(px-eq)-slip-comm-fin
        if abs(net-t['net_usd']) > 1e-8: err['net'] += 1
        rows.append(dict(k=k, ek=ek, d=d, u=u, eq=eq, B=B, net=net, entry_cost=u*SW['slip_m']+u/SW['lot']*SW['comm'], rate=rate))
        B += net
    # mark-to-market equity after each decision b (bar k=b+1 close), independent
    eqs = [SW['cap']]; bal = SW['cap']; ri = 0
    for k in range(1, 120):
        while ri < len(rows) and rows[ri]['ek'] <= k:
            bal = SW['cap']+sum(r['net'] for r in rows[:ri+1]); ri += 1
        cur = [r for r in rows if r['k'] <= k < r['ek']]
        if cur:
            r = cur[0]; when = int(v[k])+H
            mark = c[k]-r['d']*my_spread(when, mo, s)/2
            fin = sum(r['rate']*r['u']*m for t0, m in rolls if v[r['k']] < t0 <= when)
            eqs.append(bal+r['d']*r['u']*(mark-r['eq'])-r['entry_cost']-fin)
        else: eqs.append(bal)
    x = np.r_[SW['cap'], np.array(eqs)]; pk = np.maximum.accumulate(x)
    dd = float(100*np.max((pk-x)/pk)); ret = float(100*(x[-1]/SW['cap']-1))
    if abs(ret-arch_w['return_pct']) > 1e-9: err['ret'] += 1
    if abs(dd-arch_w['drawdown_pct']) > 1e-9: err['dd'] += 1
    days = {}
    for t in e.trades: days[(int(e.v[t['entry_bar_abs']])-mo)//(24*H)] = days.get((int(e.v[t['entry_bar_abs']])-mo)//(24*H), 0)+1
    return err, max(days.values(), default=0), dd, ret

def report_check(w):
    e = w.env; mo = int(e.mo[0]); ff = int(e.ff[0]); s = w.spec['spread_pips']*1e-4
    bad = dict(clock=0, deadline=0, support=0, spreads=0, terminal=0); n = 0; disc = 0
    for b in range(0, w.end-1):
        if not w.report_available(b): continue
        r = w.report(b); n += 1
        if not (r.issued_at == r.first_fill == int(e.v[b])+H and r.observed_through <= r.issued_at): bad['clock'] += 1
        if r.deadline != ff or ff != int(e.v[108]): bad['deadline'] += 1
        if r.paths.shape[1] != 107-b or r.first_fill+r.paths.shape[1]*H != ff: bad['support'] += 1
        exp = np.array([my_spread(int(t), mo, s) for t in e.v[b+1:w.end]])
        if not np.allclose(r.spreads, exp[None], atol=0, rtol=0): bad['spreads'] += 1
        if not np.all(r.terminal_spreads == my_spread(ff, mo, s)): bad['terminal'] += 1
        if my_spread(ff, mo, s) != my_spread(int(e.v[107]), mo, s): disc += 1
    return bad, n, disc

def strip(x): return {k: v for k, v in x.items()}
t0 = time.time()
out = dict(quality=q, accounts=0, warm_mismatch=0, cold_checked=0, cold_mismatch=0, report_identity_warm_vs_cold_mismatch=0,
           acct_err={}, max_entries_per_day=0, report_bad={}, reports_checked=0, terminal_discriminating_reports=0,
           size_traces={}, per_policy_mismatch={})
from app.dynamic_account_panel import report_identity
size_rows = {k: [] for k in REPS if k+'_size1' in pol}
for i, spec in enumerate(specs):
    w = CrossedWorld(spec, CONFIG); cfg0 = w.env.cfg
    res = {}
    for n in names_:
        g, en = pol[n]
        r, e = run_capture(w, g, en)
        res[n] = r; out['accounts'] += 1
        if r != arch[n][i]:
            out['warm_mismatch'] += 1; out['per_policy_mismatch'][n] = out['per_policy_mismatch'].get(n, 0)+1
        err, mx, dd, ret = account_check(spec, e, r['ledger'], arch[n][i])
        for k2, v2 in err.items(): out['acct_err'][k2] = out['acct_err'].get(k2, 0)+v2
        out['max_entries_per_day'] = max(out['max_entries_per_day'], mx)
    assert w.env.cfg == cfg0
    bad, nrep, disc = report_check(w)
    for k2, v2 in bad.items(): out['report_bad'][k2] = out['report_bad'].get(k2, 0)+v2
    out['reports_checked'] += nrep; out['terminal_discriminating_reports'] += disc
    if i % 10 == 0:
        cold = CrossedWorld(spec, CONFIG)
        ids_warm = {b: report_identity(w.reports[b]) for b in w.reports}
        for n in names_:
            cw = CrossedWorld(spec, CONFIG); g, en = pol[n]
            r, _ = run_capture(cw, g, en); out['cold_checked'] += 1
            if r != arch[n][i]: out['cold_mismatch'] += 1
        for b, hid in ids_warm.items():
            if report_identity(cold.report(b)) != hid: out['report_identity_warm_vs_cold_mismatch'] += 1
    for k in size_rows:
        a, b = res[k], res[k+'_size1']
        la, lb = a['ledger'], b['ledger']
        size_rows[k].append(dict(trace_eq=a['trace'] == b['trace'],
            seq_eq=[(t['entry_bar'], t['exit_bar'], t['direction'], t['reason'], t['horizon']) for t in la] == [(t['entry_bar'], t['exit_bar'], t['direction'], t['reason'], t['horizon']) for t in lb],
            ret=a['return_pct'], ret1=b['return_pct'], dd=a['drawdown_pct'], dd1=b['drawdown_pct'],
            q=float(np.mean([t['q'] for t in la])) if la else None, q1=float(np.mean([t['q'] for t in lb])) if lb else None,
            unit_ratio=float(np.mean([x['units']/y['units'] for x, y in zip(la, lb)])) if la and len(la) == len(lb) else None))
    if i % 60 == 0: print(q, i, round(time.time()-t0, 1), out['warm_mismatch'], out['acct_err'], flush=True)
for k, rows in size_rows.items():
    out['size_traces'][LABEL[k]] = dict(worlds=len(rows), trace_equal=sum(r['trace_eq'] for r in rows), seq_equal=sum(r['seq_eq'] for r in rows),
        mean_ret=float(np.mean([r['ret'] for r in rows])), mean_ret_size1=float(np.mean([r['ret1'] for r in rows])),
        worst_dd=float(max(r['dd'] for r in rows)), worst_dd_size1=float(max(r['dd1'] for r in rows)),
        mean_q=float(np.mean([r['q'] for r in rows if r['q'] is not None])), mean_q_size1=float(np.mean([r['q1'] for r in rows if r['q1'] is not None])),
        mean_unit_ratio=float(np.nanmean([r['unit_ratio'] for r in rows if r['unit_ratio'] is not None])),
        ret_ratio=float(np.mean([r['ret'] for r in rows])/np.mean([r['ret1'] for r in rows])),
        dd_ratio_worst=float(max(r['dd'] for r in rows)/max(r['dd1'] for r in rows)))
out['seconds'] = time.time()-t0
dump(f'D_replay_{q}.json', out)
print(json.dumps(out, default=str))
