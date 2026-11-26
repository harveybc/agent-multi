"""Remove the two explicit volatility terms from side selection only."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.role471_net_selector import account, audit_book, outcomes
from tools.role477_accounts import opportunity_panel
from tools.role478_direction import LIVE_H, load
from tools.role479_gross_cost import estimate_cost, fill_components, known_friction, lock_direction, ridge


def evaluate(a, year, reference, ledgers, observe=None):
    ts, pred, _ = load(a.root, year)
    idx, fit, basic, sig, data, costs, fees, _ = opportunity_panel(ts, pred, year, a.hist, a.market, a.bundle)
    t, op, hi, lo, close, atr, _, _, _, dead = data
    _, spread, roll = costs
    pnl, end = outcomes(idx, t, op, hi, lo, atr, dead, spread, roll, fees, 1., 1.)
    tr, ev = np.flatnonzero(fit), np.flatnonzero(~fit)
    assert t[end[tr]].max() < pd.Timestamp(str(year), tz='UTC').value <= t[idx[ev]].min()
    train_g, friction = fill_components(tr, idx, data, costs, end, pnl)
    known = known_friction(spread[idx], fees[0], fees[3])
    factor = np.where(np.arange(6)<3, op[idx[tr], None]+spread[idx[tr], None]+fees[0],
                      op[idx[tr], None]-fees[0]) / 5e4
    cost = estimate_cost(friction*factor-known[tr], known[ev], close[idx[ev]-1])
    accounts, books, rankings, funnel = [], [], [], []
    for arm in ('context','seasonal'):
        x = basic[:, :7] if arm == 'context' else np.c_[basic[:, :7], sig, abs(sig)]
        reduced = np.delete(x, [5,6], axis=1)
        original = ridge(x[tr], train_g, x[ev])
        net = original - cost
        for variant, direction in [('control',original), ('no_vol_side',ridge(reduced[tr],train_g,reduced[ev]))]:
            scores = lock_direction(direction, net)
            arr = (idx[ev], scores, pnl[ev], end[ev])
            genes = (1.,1.,0.,0.,15.,1.)
            ret, dd, n, wins = account(*arr,*genes[2:])
            book, verified = audit_book(arr,genes,t)
            np.testing.assert_allclose(verified,ret,atol=1e-8)
            if observe is not None:
                observe(year,arm,variant,t[idx[ev]],direction,net,pnl[ev])
            if variant == 'control':
                key = (year, arm, 'gross_side_then_cost')
                archived = ledgers.loc[key].reset_index(drop=True)
                for col in ('entry_ns','exit_bar_open_ns','side','hours'):
                    np.testing.assert_array_equal(pd.DataFrame(book)[col],archived[col])
                for col in ('net_usd','q','balance'):
                    np.testing.assert_allclose(pd.DataFrame(book)[col],archived[col],atol=1e-8,rtol=1e-11)
                np.testing.assert_allclose([ret,dd],reference.loc[key,['ret','dd_realized']],atol=1e-8)
            accounts.append(dict(year=year,arm=arm,variant=variant,origins=len(ev),fit=len(tr),
                                 ret=ret,dd_realized=dd,buy=sum(n[:3]),sell=sum(n[3:]),
                                 buy_wins=sum(wins[:3]),sell_wins=sum(wins[3:])))
            paired_buy = direction[:,:3] > direction[:,3:]
            gross_buy = direction[:,:3].max(1) > direction[:,3:].max(1)
            net_buy = gross_buy & (net[:,:3].max(1)>0)
            assert np.all(~gross_buy | paired_buy.any(1)) and net_buy.sum() >= n[:3].sum()
            funnel.append(dict(year=year,arm=arm,variant=variant,
                               any_scale_buy=paired_buy.any(1).sum(),best_gross_buy=gross_buy.sum(),
                               positive_net_buy=net_buy.sum(),buy_executed=n[:3].sum()))
            books.extend(dict(year=year,arm=arm,variant=variant,**r) for r in book)
            rankings.append((arm, variant, direction))
    real_g, _ = fill_components(ev, idx, data, costs, end, pnl)
    paired = []
    for arm, variant, direction in rankings:
        for k,h in enumerate(LIVE_H):
            actual = real_g[:,k]-real_g[:,k+3]
            predicted = direction[:,k]-direction[:,k+3]
            good_buy = (pnl[ev,k]>0) & (pnl[ev,k]>pnl[ev,k+3])
            good_sell = (pnl[ev,k+3]>0) & (pnl[ev,k+3]>pnl[ev,k])
            paired.append(dict(year=year,arm=arm,variant=variant,h=h,origins=len(ev),
                               correct=int(sum(np.sign(predicted)==np.sign(actual))),
                               profitable_better_buy=sum(good_buy),recognized_buy=sum(good_buy & (predicted>0)),
                               profitable_better_sell=sum(good_sell),recognized_sell=sum(good_sell & (predicted<0))))
    return accounts,books,paired,funnel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','hist','market','bundle','reference','out'):
        parser.add_argument(key,type=Path)
    a = parser.parse_args()
    keys = ['year','arm','variant']
    ref = pd.read_csv(a.reference/'accounts.csv').set_index(keys)
    books = pd.read_csv(a.reference/'books.csv.gz').set_index(keys).sort_index()
    results = [evaluate(a,y,ref,books) for y in range(2020,2024)]
    a.out.mkdir(exist_ok=True,parents=True)
    for i, name in enumerate(('accounts.csv','books.csv.gz','paired.csv','buy_funnel.csv')):
        pd.DataFrame(r for result in results for r in result[i]).to_csv(a.out/name,index=False)
