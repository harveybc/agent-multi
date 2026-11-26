"""Paired directional opportunity and ridge attribution for the 2022 anomaly."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.role470_barriers import probe
from tools.role471_net_selector import outcomes
from tools.role477_accounts import opportunity_panel
from tools.role478_direction import LIVE_H, load
from tools.role479_gross_cost import ridge


def run(root, history, market, bundle, out):
    ts, pred, _ = load(root, 2022)
    idx, fit, basic, signals, data, costs, fees, _ = opportunity_panel(
        ts, pred, 2022, history, market, bundle)
    t, op, hi, lo, _, atr, _, _, _, dead = data
    _, spread, roll = costs
    pnl, ends = outcomes(idx, t, op, hi, lo, atr, dead, spread, roll, fees, 1., 1.)
    assert t[ends[fit]].max() < t[idx[~fit]].min()
    gross = np.empty_like(pnl)
    for row, i in enumerate(idx):
        for k, h in enumerate(LIVE_H * 2):
            w = atr[i-1] * np.sqrt(h)
            f = probe(i, 1 if k < 3 else -1, w, w, 1., h, data, costs)
            assert f['end'] == ends[row, k] and abs(f['usd'] - pnl[row, k]) < 1e-8
            gross[row, k] = f['gross_same_fills_usd']
    records, shifts = [], []
    base = ['sin_hour','cos_hour','sin_week','cos_week','time_to_flat','log_vol','log_vol_sq']
    signal_names = [f'{prefix}_{h}' for prefix in ('forecast','abs_forecast')
                    for h in (1,2,3,4,5,6,24,48)]
    for arm, x, names in [('context', basic[:, :7], base),
                          ('seasonal', np.c_[basic[:, :7], signals, abs(signals)], base + signal_names)]:
        mu = x[fit].mean(0)
        pred_g = ridge(x[fit], gross[fit], x)
        delta = x[~fit].mean(0) - mu
        # Isolate each linear term's contribution to the mean score change.
        partial = ridge(x[fit], gross[fit], mu + np.diag(delta)) - gross[fit].mean(0)
        np.testing.assert_allclose(partial.sum(0), pred_g[~fit].mean(0)-pred_g[fit].mean(0), atol=1e-7)
        for j, h in enumerate(LIVE_H):
            for name, value in zip(names, partial[:, j]-partial[:, j+3]):
                shifts.append(dict(arm=arm, h=h, term=name, mean_shift_usd=value))
            for split, mask in [('fit',fit),('eval',~fit)]:
                g, n, p = [v[mask, j]-v[mask, j+3] for v in (gross,pnl,pred_g)]
                better = (n > 0) & (pnl[mask, j] > 0)
                records.append(dict(arm=arm,h=h,split=split,origins=len(g),
                    gross_buy_better=int(sum(g>0)),gross_ties=int(sum(g==0)),
                    net_buy_better=int(sum(n>0)),net_profitable_buy_better=int(sum(better)),
                    predicted_gross_buy_better=int(sum(p>0)),
                    profitable_buy_ranked_buy=int(sum(better & (p>0))),
                    sign_accuracy=np.mean(np.sign(p)==np.sign(g)),
                    gross_margin_mean=g.mean(),net_margin_mean=n.mean(),
                    predicted_margin_mean=p.mean()))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/'paired.csv',index=False)
    pd.DataFrame(shifts).to_csv(out/'shift.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','history','market','bundle','out'):
        parser.add_argument(key,type=Path)
    run(**vars(parser.parse_args()))
