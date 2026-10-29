"""E: fixed-position native counterfactuals for every P1 early close on one rung.
Trace-driven replay of the archived account on the pinned engine; at each executed AGENT_CLOSE decision
the position is cloned and (a) held passively to its native outcome, (b) closed at every later executable
open up to that outcome. O* = max over (b) and (a)."""
import copy, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from native_lib import *
q = sys.argv[1]
P1 = REPS[0]; genes = decode(dict((r['key'], r['genes']) for r in FRONT['rows'])[P1])
arch = replay(q, P1)['worlds']; specs = execution_specs('confirmation', q, False)
SIG = .001; EPS = .02*SIG

def clone(e):
    c = copy.copy(e)
    c.trades = list(e.trades); c.log = list(e.log); c.entries_day = dict(e.entries_day)
    c.counts = dict(e.counts); c.decisions = list(e.decisions); c.pos = copy.copy(e.pos)
    return c

def step(w, e, t):
    a = t['action']; plan = None
    if a in (1, 2): plan = w.entry_plan(t['horizon'], 1 if a == 1 else -1, genes[8])
    return e.step(t['bar'], a, entry_plan=plan)

rows = []; replay_bad = 0; clone_bad = 0; preempted = 0
t0 = time.time()
for i, (spec, aw) in enumerate(zip(specs, arch)):
    w = CrossedWorld(spec, CONFIG); e = copy.copy(w.env); e.reset()
    tr = aw['trace']; risk = None; entry_info = None
    for idx, t in enumerate(tr):
        if t['action'] == 3 and e.pos is not None:
            pos = e.pos; R = risk; base_n = len(e.trades)
            # (a) hold passively to the native outcome
            hcl = clone(e); closes = []; j = idx
            while hcl.pos is not None:
                # (b) executable close at this decision (open of bar j+1), native protections first
                c2 = clone(hcl); c2.step(tr[j]['bar'], 3)
                closes.append((tr[j]['bar']+1, c2.trades[-1]['net_usd'], c2.trades[-1]['close_reason']))
                hcl.step(tr[j]['bar'], 0); j += 1
            hold = hcl.trades[-1]
            info = step(w, e, t)
            ex = e.trades[-1]
            if e.trades[-1]['net_usd'] != closes[0][1]: clone_bad += 1
            if info['close'] != 'AGENT_CLOSE': preempted += 1
            vals = [c[1] for c in closes]+[hold['net_usd']]
            ostar = max(vals); u = ex['units']
            rows.append(dict(world=i, pattern=spec['pattern'], volatility=spec['volatility'], spread=spec['spread_pips'], mirror=spec['mirror'],
                entry_bar=ex['entry_bar_abs'], close_bar=ex['exit_bar_abs'], reason=ex['close_reason'], units=u, R=R,
                executed=ex['net_usd'], hold=hold['net_usd'], hold_reason=hold['close_reason'], hold_exit_bar=hold['exit_bar_abs'],
                ostar=ostar, ostar_bar=([c[0] for c in closes]+[hold['exit_bar_abs']])[int(np.argmax(vals))],
                ostar_is_hold=bool(np.argmax(vals) == len(vals)-1), n_alternatives=len(vals)))
            continue
        info = step(w, e, t)
        if info.get('entry'):
            risk = info['absolute_order']['modeled_loss']
    led = aw['ledger']
    if [(x['entry_bar_abs'], x['exit_bar_abs'], x['units'], x['close_reason']) for x in e.trades] != [(x['entry_bar'], x['exit_bar'], x['units'], x['reason']) for x in led] or any(abs(x['net_usd']-y['net']) > 1e-9 for x, y in zip(e.trades, led)):
        replay_bad += 1
    if i % 60 == 0: print(q, i, len(rows), round(time.time()-t0, 1), flush=True)

pre_rows = [r for r in rows if r['reason'] != 'AGENT_CLOSE']
rows = [r for r in rows if r['reason'] == 'AGENT_CLOSE']

def classify(r):
    d = r['executed']-r['hold']; band = EPS*r['units']
    return 'helpful' if d > band else 'destructive' if d < -band else 'neutral'
for r in rows:
    r['delta_vs_hold_usd'] = r['executed']-r['hold']; r['delta_sigma'] = r['delta_vs_hold_usd']/(SIG*r['units'])
    r['class'] = classify(r); r['regret_R'] = (r['ostar']-r['executed'])/r['R']; r['delta_R'] = r['delta_vs_hold_usd']/r['R']

def agg(rs):
    if not rs: return None
    c = {k: sum(r['class'] == k for r in rs) for k in ('helpful', 'destructive', 'neutral')}
    reg = np.array([r['regret_R'] for r in rs]); dR = np.array([r['delta_R'] for r in rs])
    return dict(n=len(rs), **c, share_destructive=c['destructive']/len(rs), share_helpful=c['helpful']/len(rs),
        hold_outcome={k: sum(r['hold_reason'] == k for r in rs) for k in ('TP', 'SL', 'FORCED_FLAT')},
        sum_delta_vs_hold_usd=float(sum(r['delta_vs_hold_usd'] for r in rs)),
        sum_delta_vs_hold_pct_of_total_capital=float(sum(r['delta_vs_hold_usd'] for r in rs)/(1e4*len(specs))*100),
        mean_delta_R=float(dR.mean()), median_delta_R=float(np.median(dR)),
        regret_R_mean=float(reg.mean()), regret_R_median=float(np.median(reg)), regret_R_p90=float(np.quantile(reg, .9)),
        regret_zero_share=float((reg <= 1e-12).mean()), ostar_is_hold_share=float(np.mean([r['ostar_is_hold'] for r in rs])),
        executed_equals_ostar=int(sum(abs(r['ostar']-r['executed']) < 1e-9 for r in rs)))
neg = [r for r in rows if r['executed'] < 0]; nonneg = [r for r in rows if r['executed'] >= 0]
def even(rs, n):
    rs = sorted(rs, key=lambda r: (r['world'], r['entry_bar']))
    if len(rs) <= n: return rs
    return [rs[int(k)] for k in np.linspace(0, len(rs)-1, n).round().astype(int)]
sample = even(neg, 150)+even(nonneg, 150)
out = dict(quality=q, accounts=len(specs), replay_mismatch_accounts=replay_bad, clone_mismatch=clone_bad,
    close_actions_preempted_by_native=preempted, preempted_reasons={k: sum(r['reason'] == k for r in pre_rows) for k in ('SL', 'TP')}, early_closes=len(rows),
    all=agg(rows), negative_realized=agg(neg), nonnegative_realized=agg(nonneg), stratified_sample_300=agg(sample),
    by_hold_outcome={k: agg([r for r in rows if r['hold_reason'] == k]) for k in ('TP', 'SL', 'FORCED_FLAT')},
    by_pattern={p: agg([r for r in rows if r['pattern'] == p]) for p in ('trend', 'reversal', 'overlap')},
    p1_total_return_pct_mean=float(np.mean([a['return_pct'] for a in arch])), seconds=time.time()-t0)
dump(f'E_close_{q}.json', out)
(S/'out'/f'E_close_rows_{q}.json').write_text(json.dumps(rows)+'\n')
print(json.dumps({k: v for k, v in out.items() if k in ('quality', 'replay_mismatch_accounts', 'clone_mismatch', 'early_closes', 'all', 'negative_realized', 'stratified_sample_300')}, default=float))
