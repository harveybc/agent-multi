"""Sequential gate comparison at the same fixed exposure; no policy fitting."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.role481_direction_volatility import LIVE_H, evaluate
from tools.role482_net_gate import side_horizon


def replay(ts, rows, actions, net, pnl, ends, gate):
    balance = peak = 10000.
    dd = 0.
    occupied_until = -1
    trades = []
    for r,k in zip(rows,actions):
        if ts[r] <= occupied_until or (gate and net[r,k] <= 0):
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
    accounts,books = [],[]
    def observe(year,arm,variant,ts,direction,net,pnl,ends):
        rows,k = side_horizon(direction,net)
        baseline = archived.loc[(year,arm,variant)]
        prior_entries = set(baseline.entry_ns)
        for gate in (True,False):
            trades,ret,dd = replay(ts,rows,k,net,pnl,ends,gate)
            df = pd.DataFrame(trades)
            if gate:
                for col in ('entry_ns','exit_bar_open_ns','hours','side'):
                    np.testing.assert_array_equal(df[col],baseline[col])
                # Recover q=1 trade returns independently from the original sized book.
                fractions = baseline.net_usd / ((baseline.balance-baseline.net_usd)*baseline.q)
                np.testing.assert_allclose(df.balance,10000.*(1.+fractions).cumprod(),rtol=1e-11)
            assert (df.entry_ns.to_numpy()[1:] > df.exit_bar_open_ns.to_numpy()[:-1]).all()
            common = int(df.entry_ns.isin(prior_entries).sum())
            accounts.append(dict(year=year,arm=arm,variant=variant,gate=gate,
                origins=len(ts),ret=ret,dd_realized=dd,orders=len(df),wins=sum(df.net_usd>0),
                buys=sum(df.side=='buy'),buy_wins=sum((df.side=='buy') & (df.net_usd>0)),
                sells=sum(df.side=='sell'),sell_wins=sum((df.side=='sell') & (df.net_usd>0)),
                nonpositive=sum(df.score<=0),shared_with_gate=common,
                displaced_gated=len(prior_entries)-common))
            books.append(df.assign(year=year,arm=arm,variant=variant,gate=gate))
    for year in range(2020,2024):
        evaluate(a,year,ref,history,observe)
    a.out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(accounts).to_csv(a.out/'accounts.csv',index=False)
    pd.concat(books,ignore_index=True).to_csv(a.out/'books.csv.gz',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('root','hist','market','bundle','control','previous','out'):
        parser.add_argument(arg,type=Path)
    run(parser.parse_args())
