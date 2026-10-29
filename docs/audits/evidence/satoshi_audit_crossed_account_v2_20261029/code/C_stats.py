"""C: independent summaries + paired cluster statistics (auditor implementation)."""
import itertools, math, sys
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from common import *

def summarize(res):
    W = res['worlds']; n = len(W)
    trades = [t for w in W for t in w['ledger']]
    orders = len(trades); buys = sum(t['direction'] == 1 for t in trades)
    out = dict(accounts=n, return_pct=float(np.mean([w['return_pct'] for w in W])),
               worst_dd_pct=float(max(w['drawdown_pct'] for w in W)),
               mean_dd_pct=float(np.mean([w['drawdown_pct'] for w in W])),
               orders=orders, buys=buys, sells=orders-buys,
               wins=sum(t['net'] > 0 for t in trades),
               buy_wins=sum(t['net'] > 0 and t['direction'] == 1 for t in trades),
               sell_wins=sum(t['net'] > 0 and t['direction'] == -1 for t in trades),
               early_closes=sum(t['reason'] == 'AGENT_CLOSE' for t in trades),
               reasons={r: sum(t['reason'] == r for t in trades) for r in sorted({t['reason'] for t in trades})},
               horizons={str(h): sum(t['horizon'] == h for t in trades) for h in (3, 6, 12, 24)},
               mean_orders=orders/n, eligible=orders/n >= 3,
               neg_early=sum(t['reason'] == 'AGENT_CLOSE' and t['net'] < 0 for t in trades),
               # return recomputed from ledger cash (capital 10000 checked below)
               ret_from_ledger=float(np.mean([100*sum(t['net'] for t in w['ledger'])/1e4 for w in W])))
    return out

def clusters(a, b, field='return_pct'):
    assert len(a) == len(b)
    g = {}
    for x, y in zip(a, b):
        assert x['spec'] == y['spec'], 'UNPAIRED'
        g.setdefault((x['spec']['seed'], x['spec']['pattern']), []).append(x[field]-y[field])
    assert all(len(v) == 12 for v in g.values()), 'CLUSTER_SIZE'
    keys = sorted(g)
    return keys, np.array([np.mean(g[k]) for k in keys])

def stats(v, rng, B=20000):
    n = len(v); m = v.mean(); se = v.std(ddof=1)/math.sqrt(n) if n > 1 else float('nan')
    idx = rng.integers(0, n, size=(B, n)); boot = v[idx].mean(axis=1)
    flips = rng.choice([-1., 1.], size=(100000, n)); perm = (flips*v).mean(axis=1)
    p = float((np.abs(perm) >= abs(m)-1e-15).mean())
    tq = 2.045229642132703  # t_{0.975,29}
    return dict(clusters=n, mean_pp=float(m), positive=int((v > 0).sum()), negative=int((v < 0).sum()),
                zero=int((v == 0).sum()), boot95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                t95=[float(m-tq*se), float(m+tq*se)], signflip_p_two_sided=p)

def worst_dd_boot(a, b, rng, B=5000):
    # resample clusters, recompute worst-account DD for each policy
    ca, cb = {}, {}
    for x, y in zip(a, b):
        k = (x['spec']['seed'], x['spec']['pattern'])
        ca.setdefault(k, []).append(x['drawdown_pct']); cb.setdefault(k, []).append(y['drawdown_pct'])
    keys = sorted(ca); A = np.array([max(ca[k]) for k in keys]); Bm = np.array([max(cb[k]) for k in keys])
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    d = A[idx].max(axis=1)-Bm[idx].max(axis=1)
    return dict(point=float(A.max()-Bm.max()), boot95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                share_p1_lower=float((d < 0).mean()))

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.]*len(ps); run = 0.
    for r, i in enumerate(order):
        run = max(run, min(1., (len(ps)-r)*ps[i])); adj[i] = run
    return adj

rng = np.random.default_rng(777)
summ, mism = {}, []
for q in QUALITIES:
    for n in names():
        f = unseal(RUN/f'FINAL_{q}_{n}.json'); r = replay(q, n); s = summarize(r)
        summ[f'{q}/{label(n)}'] = s
        chk = dict(objectives=[s['return_pct'], -s['worst_dd_pct']], orders=s['orders'], buys=s['buys'],
                   sells=s['sells'], wins=s['wins'], buy_wins=s['buy_wins'], sell_wins=s['sell_wins'],
                   early_closes=s['early_closes'], eligible=s['eligible'])
        for k, v in chk.items():
            fv = f[k]
            same = (abs(v[0]-fv[0]) < 1e-9 and abs(v[1]-fv[1]) < 1e-9) if k == 'objectives' else v == fv
            if not same: mism.append((q, n, k, v, fv))
        if abs(s['ret_from_ledger']-s['return_pct']) > 1e-9: mism.append((q, n, 'ledger_return', s['ret_from_ledger'], s['return_pct']))
        del r
print('summary mismatches vs FINAL:', len(mism), mism[:5])
# paired comparisons
P1 = REPS[0]; comp = {}; plist = []
for q in QUALITIES:
    a = replay(q, P1)['worlds']
    others = [n for n in names() if n != P1]
    for n in others:
        b = replay(q, n)['worlds']; keys, v = clusters(a, b)
        st = stats(v, rng); st['worst_dd'] = worst_dd_boot(a, b, rng)
        _, vd = clusters(a, b, 'drawdown_pct'); st['mean_account_dd_delta_pp'] = float(vd.mean())
        comp[f'{q}/P1-{label(n)}'] = st
        if n in CONTROLS: plist.append(f'{q}/P1-{label(n)}')
    if q == 'correlated':
        b = posthoc()['worlds']; keys, v = clusters(a, b); st = stats(v, rng)
        st['worst_dd'] = worst_dd_boot(a, b, rng); comp['correlated/P1-P1only6_posthoc'] = st
        st['per_cluster'] = {f'{k[0]}:{k[1]}': float(x) for k, x in zip(keys, v)}
        st['accounts_identical_return'] = int(sum(abs(x['return_pct']-y['return_pct']) < 1e-12 for x, y in zip(a, b)))
        st['accounts_identical_trace'] = int(sum(x['trace'] == y['trace'] for x, y in zip(a, b)))
adj = holm([comp[k]['signflip_p_two_sided'] for k in plist])
for k, p in zip(plist, adj): comp[k]['holm_adj_p_over_35_control_contrasts'] = p
dump('C_summaries.json', summ); dump('C_paired.json', comp)
for k in ['correlated/P1-only_6', 'correlated/P1-greedy', 'correlated/P1-only_24', 'correlated/P1-only_3', 'correlated/P1-P1only6_posthoc', 'ideal/P1-only_3']:
    c = comp[k]; print(k, {x: c[x] for x in ('mean_pp', 'positive', 'boot95', 't95', 'signflip_p_two_sided')}, c.get('holm_adj_p_over_35_control_contrasts'), c['worst_dd'])
print('rows with P1 < control (mean):', [k for k in plist if comp[k]['mean_pp'] < 0])
print('rows not significant after Holm (adj p>0.05):', [k for k in plist if comp[k]['holm_adj_p_over_35_control_contrasts'] > .05])
