"""Attribute sealed 435 nonlinear top offers and their equal-count random control."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from role411_payoff_parts import COST, OUT as LEDGERS
from role424_sell_bias import atomic, base, common, panel, selected

SRC = Path('/var/tmp/role435-within-week')
OUT = Path('/var/tmp/role437-sell-attribution')
PARTS = ['barrier_gross', 'close_gross', *COST, 'continuation']


def engine_parts(pn, bars):
    e = pn.e
    e.reset()
    cap = e.cfg.capital
    anchors = {}
    for b in pn.bars:
        g = e.seg_of[b]
        if b >= e.first + 3 and e.v[b] + base.HOUR >= e.op[g] + 36 * base.HOUR:
            anchors.setdefault(g, b)

    @lru_cache(4096)
    def extra(a, cash):
        side = np.sign(e.c[a] - e.c[a-3])
        return pn.isolated(a, 1 if side > 0 else 2, base.HS.index(6), cash)[1]-cash if side else 0.

    def trade(b, j, managed):
        eng = base.clone(e)
        eng.next_decision = b
        env = pn.envs[j]
        env.bar = b
        assert eng.step(b, 2, entry_plan=env.entry_plan(-1))['entry']
        k = b + 1
        while eng.pos is not None:
            assert k < eng.last
            eng.step(k, pn.close(eng, k, j) if managed else 0)
            k += 1
        tr, = eng.trades
        return tr, k, eng.balance

    values = np.empty((len(bars), len(base.HS), len(PARTS)))
    for i, b in enumerate(bars):
        a = anchors.get(e.seg_of[b], -1)
        wait = extra(a, cap) if a >= b+1 else 0.
        for j in range(len(base.HS)):
            m, k, cash = trade(b, j, True)
            p, _, _ = trade(b, j, False)
            assert all(m[f] == p[f] for f in ('units','direction','entry_price','entry_bar_abs','stop_loss','take_profit'))
            cont = (extra(a, cash) if a >= k else 0.) - wait
            values[i,j] = [p['gross_usd'], m['gross_usd']-p['gross_usd'],
                           *(-m[c+'_usd'] for c in COST), cont]
    return values*(1e4/cap)


def cell(key):
    year, seed = key
    tag = f'{year}-{seed}'
    base.torch.set_num_threads(1)
    with np.load(base.DEST/'market.npz') as bank:
        t = bank['t']
        ids,end,_,y,z = common(year,seed,t)
        dates = t[ids]
        ts = z['origin_ns']
        ix = np.searchsorted(ts,dates)
        assert np.array_equal(ts[ix],dates)
        jul = base.ns(f'{year}-07-01')
        fit = (dates<jul)&(end<jul)
        ev = dates>=jul
        with np.load(SRC/f'{tag}.scores.npz') as d:
            assert str(d['sha']) == base.file_sha(SRC/f'{tag}.npz')
            raw = d['nonlinear']
        pred = raw+(y[fit]-raw[ix[fit]]).mean(0)
        dates = dates[ev]
        picked = selected(pred[ix[ev]],dates)
        if year == 2023:
            f = pd.read_csv(LEDGERS/f'offers-{seed}.csv.gz')
            f = f[f.side=='OPEN_SELL'].set_index(['origin_ns','h'],verify_integrity=True)
            idx = pd.MultiIndex.from_product([dates,base.HS],names=f.index.names)
            parts = f.loc[idx,PARTS].to_numpy().reshape(len(dates),4,-1)
        else:
            pn = panel(year,seed,bank['x'],t,bank['valid'])
            assert np.array_equal(pn.p,z['p']) and np.array_equal(pn.bars,z['bars'])
            assert np.array_equal(pn.e.v[pn.bars]+base.HOUR,ts)
            parts = engine_parts(pn,pn.bars[ix[ev]])
    residual = float(np.abs(parts.sum(-1)-y[ev]).max())
    assert residual < 1e-7
    week = (dates//86400000000000+3)//7
    ref = base.read(SRC/f'{tag}.json')['nonlinear']
    rows = []
    for j,h in enumerate(base.HS):
        random_weights = np.zeros(len(week))
        for w in np.unique(week):
            at = week==w
            k = int(picked[at,j].sum())
            assert k==max(1,at.sum()//10)
            random_weights[at] = k/at.sum()
        a,b = [np.average(parts[:,j],weights=weights,axis=0) for weights in (picked[:,j],random_weights)]
        assert abs(a.sum()-ref['per_horizon'][str(h)]) < .001
        rows.append(dict(year=year,seed=seed,h=int(h),n=len(dates),k=int(picked[:,j].sum()),
                         model=dict(zip(PARTS,a)),random=dict(zip(PARTS,b)),residual=residual))
    for arm,metric in [('model','bp'),('random','random_bp')]:
        assert abs(np.average([sum(r[arm].values()) for r in rows],weights=[r['k'] for r in rows])-ref[metric]) < .001
    atomic(OUT/f'{tag}.json',rows)
    print(tag,'OK',flush=True)
    return rows


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz') == base.read(base.DEST/'CACHE.json')['sha']
    with ProcessPoolExecutor(3) as pool:
        rows = [r for group in pool.map(cell,[(y,s) for y in (2023,2022) for s in (7,19,43)]) for r in group]
    assert len(rows)==24
    atomic(OUT/'DONE.json',rows)
