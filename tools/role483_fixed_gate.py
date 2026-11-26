"""Sequential gate comparison at the same fixed exposure; no policy fitting."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.role481_direction_volatility import LIVE_H, evaluate
from tools.role482_net_gate import side_horizon


def replay(ts, rows, actions, net, pnl, ends, gate, *, sell_gate=None, rejected_buy=None):
    sell_gate = gate if sell_gate is None else sell_gate
    balance = peak = 10000.
    dd = 0.
    occupied_until = -1
    trades = []
    for r,k in zip(rows,actions):
        side_gate = gate if k<3 else sell_gate
        if ts[r] <= occupied_until:
            continue
        if side_gate and net[r,k] <= 0:
            if k<3 and rejected_buy is not None:
                occupied_until = rejected_buy(r,k)
                assert occupied_until >= ts[r]
            continue
        occupied_until = int(ends[r,k])
        assert occupied_until >= ts[r]
        gain = balance*pnl[r,k]/10000.
        balance += gain
        assert balance > 0
        peak = max(peak,balance)
        dd = max(dd,1.-balance/peak)
        trades.append(dict(entry_ns=ts[r],exit_bar_open_ns=occupied_until,
                           hours=LIVE_H[k%3],side='buy' if k<3 else 'sell',
                           score=net[r,k],q=1.,net_usd=gain,balance=balance))
    return trades,100*(balance/10000.-1.),100*dd


def run(a):
    keys = ['year','arm','variant']
    def read(folder,name):
        return pd.read_csv(folder/name).set_index(keys).sort_index()
    history = read(a.control,'books.csv.gz')
    ref = read(a.control,'accounts.csv')
    archived = read(a.previous,'books.csv.gz')
    split = getattr(a,'side_reference',None)
    modes = [('both_filtered',True,True),('neither_filtered',False,False)]
    if split:
        fixed_books = pd.read_csv(split/'books.csv.gz').set_index(keys+['gate']).sort_index()
        fixed_accounts = pd.read_csv(split/'accounts.csv').set_index(keys+['gate']).sort_index()
        modes.extend([('allow_buy',False,True),('allow_sell',True,False)])
    accounts,books = [],[]
    def observe(year,arm,variant,ts,direction,net,pnl,ends):
        rows,k = side_horizon(direction,net)
        baseline = archived.loc[(year,arm,variant)]
        prior_entries = set(baseline.entry_ns)
        for mode,buy_gate,sell_gate in modes:
            trades,ret,dd = replay(ts,rows,k,net,pnl,ends,buy_gate,sell_gate=sell_gate)
            df = pd.DataFrame(trades)
            if buy_gate and sell_gate:
                for col in ('entry_ns','exit_bar_open_ns','hours','side'):
                    np.testing.assert_array_equal(df[col],baseline[col])
                # Recover q=1 trade returns independently from the original sized book.
                fractions = baseline.net_usd / ((baseline.balance-baseline.net_usd)*baseline.q)
                np.testing.assert_allclose(df.balance,10000.*(1.+fractions).cumprod(),rtol=1e-11)
            if split and buy_gate == sell_gate:
                key = (year,arm,variant,buy_gate)
                ref_book = fixed_books.loc[key]
                for col in ('entry_ns','exit_bar_open_ns','hours','side'):
                    np.testing.assert_array_equal(df[col],ref_book[col])
                for col in ('q','score','net_usd','balance'):
                    np.testing.assert_allclose(df[col],ref_book[col],atol=1e-8,rtol=1e-11)
                np.testing.assert_allclose([ret,dd],fixed_accounts.loc[key,['ret','dd_realized']],
                                           atol=1e-8)
            admitted = df[df.score<=0]
            assert not (buy_gate and (admitted.side=='buy').any())
            assert not (sell_gate and (admitted.side=='sell').any())
            label = dict(mode=mode,nonpositive_buys=sum(admitted.side=='buy'),
                         nonpositive_sells=sum(admitted.side=='sell')) if split else dict(gate=buy_gate)
            assert (df.entry_ns.to_numpy()[1:] > df.exit_bar_open_ns.to_numpy()[:-1]).all()
            common = int(df.entry_ns.isin(prior_entries).sum())
            accounts.append(dict(year=year,arm=arm,variant=variant,**label,
                origins=len(ts),ret=ret,dd_realized=dd,orders=len(df),wins=sum(df.net_usd>0),
                buys=sum(df.side=='buy'),buy_wins=sum((df.side=='buy') & (df.net_usd>0)),
                sells=sum(df.side=='sell'),sell_wins=sum((df.side=='sell') & (df.net_usd>0)),
                nonpositive=sum(df.score<=0),shared_with_gate=common,
                displaced_gated=len(prior_entries)-common))
            books.append(df.assign(year=year,arm=arm,variant=variant,
                                   **({'mode':mode} if split else {'gate':buy_gate})))
    for year in range(2020,2024):
        evaluate(a,year,ref,history,observe)
    a.out.mkdir(exist_ok=True,parents=True)
    annual = pd.DataFrame(accounts)
    annual.to_csv(a.out/'accounts.csv',index=False)
    if split:
        effects = {}
        for metric in ('ret','dd_realized'):
            v = annual.pivot(index=keys,columns='mode',values=metric)
            effects[metric+'_allow_buy_pp'] = v.allow_buy-v.both_filtered
            effects[metric+'_allow_sell_pp'] = v.allow_sell-v.both_filtered
            effects[metric+'_interaction_pp'] = v.neither_filtered-v.allow_buy-v.allow_sell+v.both_filtered
        pd.DataFrame(effects).to_csv(a.out/'effects.csv')
    pd.concat(books,ignore_index=True).to_csv(a.out/'books.csv.gz',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('root','hist','market','bundle','control','previous','out'):
        parser.add_argument(arg,type=Path)
    parser.add_argument('--side-reference',type=Path,help='483 output to verify when running side-specific gates')
    run(parser.parse_args())
