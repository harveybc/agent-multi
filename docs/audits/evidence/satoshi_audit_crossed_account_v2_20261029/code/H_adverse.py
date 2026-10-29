"""H: omitted-reversion failure anatomy + causal track-record error signal (auditor)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from native_lib import *
H = 3600*10**9
out = {}
P1 = REPS[0]
for q in ('correlated', 'omitted_reversal'):
    A = replay(q, P1)['worlds']; G = replay(q, 'greedy')['worlds']; O24 = replay(q, 'only_24')['worlds']
    tab = {}
    for key in ('pattern', 'volatility', 'spread_pips', 'mirror'):
        for val in sorted({w['spec'][key] for w in A}, key=str):
            ws = [w for w in A if w['spec'][key] == val]; tr = [t for w in ws for t in w['ledger']]
            tab[f'{key}={val}'] = dict(accounts=len(ws), mean_return=float(np.mean([w['return_pct'] for w in ws])),
                worst_dd=float(max(w['drawdown_pct'] for w in ws)), orders=len(tr), sl=sum(t['reason'] == 'SL' for t in tr),
                tp=sum(t['reason'] == 'TP' for t in tr), agent=sum(t['reason'] == 'AGENT_CLOSE' for t in tr),
                losing_accounts=sum(w['return_pct'] < 0 for w in ws),
                greedy_mean_return=float(np.mean([w['return_pct'] for w in G if w['spec'][key] == val])))
    tr = [t for w in A for t in w['ledger']]
    out[q] = dict(by=tab, orders=len(tr), reasons={r: sum(t['reason'] == r for t in tr) for r in ('TP', 'SL', 'AGENT_CLOSE', 'FORCED_FLAT')},
        losing_accounts=sum(w['return_pct'] < 0 for w in A), worst_account=max(A, key=lambda w: w['drawdown_pct'])['spec'],
        horizons={h: sum(t['horizon'] == h for t in tr) for h in (3, 6, 12, 24)},
        mean_holding=float(np.mean([t['exit_bar']-t['entry_bar'] for t in tr])))
# causal track record: at each P1 entry decision b, using ONLY reports issued at b-j (j=1..6) and closes up to b,
# realized error of those reports' predicted close at b, scaled by their own sample dispersion.
sig = {}
for q in ('correlated', 'omitted_reversal', 'delayed'):
    A = replay(q, P1)['worlds']; specs = execution_specs('confirmation', q, False)
    rows = []
    for spec, aw in zip(specs, A):
        w = CrossedWorld(spec, CONFIG); c = w.env.c
        for t in aw['ledger']:
            b = t['entry_bar']-1; errs = []; sds = []; signed = []
            for j in range(1, 7):
                bb = b-j
                if bb < 0 or not w.report_available(bb): continue
                P = w.report(bb).paths[:, j-1, 3]
                errs.append(abs(c[b]-P.mean())); sds.append(P.std()); signed.append(t['direction']*(c[b]-P.mean()))
            if not errs: continue
            rows.append(dict(z=float(np.mean(errs)/max(np.mean(sds), 1e-9)), err_pips=float(np.mean(errs)*1e4),
                             signed_pips=float(np.mean(signed)*1e4), sl=t['reason'] == 'SL', loss=t['net'] < 0, pattern=spec['pattern']))
    def auc(x, y):
        x = np.asarray(x); y = np.asarray(y, bool)
        if y.all() or (~y).all(): return None
        r = x.argsort().argsort()+1.; n1 = y.sum(); n0 = (~y).sum()
        return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0))
    z = [r['z'] for r in rows]
    sig[q] = dict(trades=len(rows), median_z=float(np.median(z)), p90_z=float(np.quantile(z, .9)),
        median_err_pips=float(np.median([r['err_pips'] for r in rows])),
        auc_z_predicts_SL=auc(z, [r['sl'] for r in rows]), auc_z_predicts_loss=auc(z, [r['loss'] for r in rows]),
        auc_err_predicts_loss=auc([r['err_pips'] for r in rows], [r['loss'] for r in rows]),
        auc_signed_error_predicts_loss=auc([-r['signed_pips'] for r in rows], [r['loss'] for r in rows]),
        by_pattern={p: dict(n=sum(r['pattern'] == p for r in rows), median_z=float(np.median([r['z'] for r in rows if r['pattern'] == p])),
                    loss_share=float(np.mean([r['loss'] for r in rows if r['pattern'] == p]))) for p in ('trend', 'reversal', 'overlap')})
out['track_record_signal'] = sig
dump('H_adverse.json', out)
print(json.dumps(out, default=str)[:5000])
