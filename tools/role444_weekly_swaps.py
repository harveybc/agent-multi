"""Paired weekly SELL substitutions and a two-factor refit attribution."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from role437_sell_attribution import engine_parts, PARTS, selected
from role443_weekly_refit import atomic, base, cell, week, SRC

OUT = Path('/var/tmp/role444-weekly-swaps')


def analyse(key,pn,ix,dates,y,ref,scores):
    year,seed = key
    tag = f'{year}-{seed}'
    masks = {k:selected(v,dates) for k,v in scores.items()}
    f,w = masks['frozen'],masks['weekly']
    for name in ('frozen','weekly','shifted'):
        m = masks[name]
        arm = ref['arms'][name]
        assert m.sum()==arm['matched_n'] and abs(y[m].mean()-arm['bp'])<1e-7
    changed = (f!=w).any(axis=1)
    parts = np.zeros((*y.shape,len(PARTS)))
    parts[changed] = engine_parts(pn,pn.bars[ix[changed]])
    assert np.abs(parts[changed].sum(-1)-y[changed]).max()<1e-7
    rows = []
    swaps = []
    for tick in np.unique(week(dates)):
        ii = np.flatnonzero(week(dates)==tick)
        for j,h in enumerate(base.HS):
            picked = {k:v[ii,j] for k,v in masks.items()}
            sums = {k:float(y[ii[v],j].sum()) for k,v in picked.items()}
            n = int(picked['frozen'].sum())
            assert n>=1 and all(v.sum()==n for v in picked.values())
            adds = ii[w[ii,j]&~f[ii,j]]
            drops = ii[f[ii,j]&~w[ii,j]]
            assert len(adds)==len(drops)
            delta = parts[adds,j].sum(0)-parts[drops,j].sum(0)
            assert abs(delta.sum()-(sums['weekly']-sums['frozen']))<1e-6
            for sign,indices in [('in',adds),('out',drops)]:
                swaps.extend(dict(origin_ns=int(dates[i]),h=h,group=sign,net_bp=y[i,j],
                                  **dict(zip(PARTS,parts[i,j]))) for i in indices)
            # Average each factor's marginal contribution over the two update orders.
            geometry = .5*((sums['geometry']-sums['frozen'])+(sums['weekly']-sums['labels']))
            labels = .5*((sums['labels']-sums['frozen'])+(sums['weekly']-sums['geometry']))
            rows.append(dict(year=year,seed=seed,h=int(h),week=int(tick),n=n,swaps=len(adds),
                             added=float(y[adds,j].sum()),removed=float(y[drops,j].sum()),
                             geometry=geometry,labels=labels,**dict(zip(PARTS,delta))))
    total = sum(r['added']-r['removed'] for r in rows)/f.sum()
    assert abs(total-(ref['arms']['weekly']['bp']-ref['arms']['frozen']['bp']))<1e-7
    assert abs(total-sum(r['geometry']+r['labels'] for r in rows)/f.sum())<1e-7
    pd.DataFrame(swaps).to_csv(OUT/f'{tag}-swaps.csv.gz',index=False)
    atomic(OUT/f'{tag}.json',rows)
    return rows


def run(key):
    return cell(key,analyse)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(SRC/'PROTOCOL.json')['market']
    with ProcessPoolExecutor(3) as pool:
        rows = [r for block in pool.map(run,[(y,s) for y in (2022,2023) for s in (7,19,43)]) for r in block]
    atomic(OUT/'DONE.json',rows)
