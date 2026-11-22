"""TRAIN-only diagnostic for the unchanged 409 direct estimator."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from role409_signed_advantage import base, fit


def table(score, value, ts):
    valid = np.isfinite(value)
    s, v, t = score[valid], value[valid], ts[valid]
    assert np.isfinite(s).all() and len(s) >= 10
    # Equal scores stay together. These retrospective bins never route orders.
    group = np.minimum(9, ((rankdata(s) - .5) * 10 / len(s)).astype(int))
    week = ((t // 86400000000000 + 3) // 7).astype(int)
    _, wi = np.unique(week, return_inverse=True)
    rows = []
    for d in range(10):
        k = group == d
        assert k.any()
        rows.append(dict(decile=d+1, n=int(k.sum()), predicted_bp=s[k].mean(),
                         realized_bp=v[k].mean(), positive_fraction=np.mean(v[k]>0)))
    boot = np.random.default_rng(410).multinomial(len(set(wi)), np.ones(len(set(wi)))/len(set(wi)), size=2000)
    def weekly_mean(d):
        k = group == d
        w = boot @ np.bincount(wi[k], minlength=boot.shape[1])
        return np.divide(boot @ np.bincount(wi[k], weights=v[k], minlength=boot.shape[1]),
                         w, where=w>0, out=np.full(len(w), np.nan))
    lo, hi = weekly_mean(0), weekly_mean(9)
    return rows,dict(rho=float(spearmanr(s,v).statistic),
                     spread=rows[-1]['realized_bp']-rows[0]['realized_bp'],
                     spread_interval=np.nanquantile(hi-lo,[.025,.975]).tolist(),
                     top_interval=np.nanquantile(hi,[.025,.975]).tolist())


def main():
    out = Path('/var/tmp/role410-train-deciles')
    out.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz') == base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:
        bank, times, valid = z['x'], z['t'], z['valid']
    assert np.all(np.diff(times)>0)
    stats, rows = {}, []
    for seed in base.SEEDS:
        for year in (2022, 2023):
            path = base.OUT/f'score-{seed}-{year}'
            rec = base.read(path/'RECEIPT.json')
            assert rec['sha'] == base.file_sha(path/'labels.npz') and rec['spec']['year']==year and rec['spec']['seed']==seed
            assert rec['end'] < base.ns(f'{year+1}-01-01')
            with np.load(path/'labels.npz') as z:
                t, y = z['origin_ns'], z['y']
            i = np.searchsorted(times,t)
            assert np.array_equal(times[i],t) and valid[i].all() and np.all(np.diff(t)>0)
            assert base.ns(f'{year}-01-01')<=t[0]<=t[-1]<base.ns(f'{year+1}-01-01')
            x = bank[i]
            if year==2022:
                mu,sd,w,b,_ = fit([x],[y])
                continue
            score = np.tensordot((x.astype(float)-mu)/sd,w,axes=1)+b
            side,horizon = base.decisions(score)
            selected = np.flatnonzero(side)
            pay = y[selected,side[selected]-1,horizon[selected]]
            n, ref = {7:(2555,-4.2825793522500915),19:(2795,-3.6644036762070438),43:(2681,-3.887856430435991)}[seed]
            assert np.isfinite(pay).sum()==n and abs(np.nanmean(pay)-ref)<1e-9, '409 decision parity'
            for r,j in np.ndindex(2,4):
                panel,s = table(score[:,r,j], y[:,r,j],t)
                key = f'{seed}:{base.ROLES[r]}:{base.HS[j]}'
                s.update(top=panel[-1]['realized_bp'], predicted_top=panel[-1]['predicted_bp'])
                stats[key] = s
                rows.extend(dict(seed=seed,side=base.ROLES[r],hours=base.HS[j],**row) for row in panel)
    assert len(rows)==240
    with (out/'DECILES.csv').open('w') as f:
        csvout = csv.DictWriter(f, rows[0].keys())
        csvout.writeheader()
        csvout.writerows(rows)
    (out/'RANK.json').write_text(json.dumps(stats, indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
