"""Compare staying available, opening rejected BUYs, and a causal cash timeout."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.role481_direction_volatility import evaluate, LIVE_H
from tools.role482_net_gate import side_horizon
from tools.role483_fixed_gate import replay

NS_H = 3_600_000_000_000


def durations(origins, buy_exits, boundary):
    elapsed = buy_exits - origins[:,None]
    if not len(origins) or elapsed.shape!=(len(origins),3) or elapsed.min()<0 or buy_exits.max()>=boundary:
        raise ValueError('invalid or immature calibration durations')
    return np.median(elapsed,axis=0).astype(np.int64)


def run(a):
    keys=['year','arm','variant']
    def read(folder,name,index):
        return pd.read_csv(folder/name).set_index(index).sort_index()
    ref=read(a.control,'accounts.csv',keys)
    old=read(a.control,'books.csv.gz',keys)
    archived=read(a.previous,'books.csv.gz',keys+['mode'])
    accounts=[]; books=[]; pauses=[]; calibration=[]
    for y in range(2020,2024):
        def calibrate(year,origins,exits,dead):
            nonlocal med,cutoffs
            med=durations(origins,exits,pd.Timestamp(str(year),tz='UTC').value)
            cutoffs=dead
            calibration.extend(dict(year=year,h=h,n=len(origins),median_hours=m/NS_H,
                                    last_calibration_exit_ns=int(exits[:,k].max()))
                               for k,(h,m) in enumerate(zip(LIVE_H,med)))
        def observe(year,arm,variant,ts,direction,net,pnl,ends):
            keys_now=dict(year=year,arm=arm,variant=variant)
            rows,actions=side_horizon(direction,net)
            def wait(r,k):
                until=min(ts[r]+med[k],cutoffs[r])
                pauses.append(dict(**keys_now,start_ns=ts[r],until_ns=until,h=LIVE_H[k]))
                return until
            for mode in ('both_filtered','allow_buy','cash_wait'):
                start=len(pauses)
                trades,ret,dd=replay(ts,rows,actions,net,pnl,ends,mode!='allow_buy',
                                     sell_gate=True,rejected_buy=wait if mode=='cash_wait' else None)
                df=pd.DataFrame(trades)
                if mode!='cash_wait':
                    baseline=archived.loc[(year,arm,variant,mode)]
                    for col in ('entry_ns','exit_bar_open_ns','side','hours'):
                        np.testing.assert_array_equal(df[col],baseline[col])
                    for col in ('score','net_usd','balance','q'):
                        np.testing.assert_allclose(df[col],baseline[col],rtol=1e-11,atol=1e-8)
                else:
                    assert (df.score>0).all()
                    occupied=sorted([(p['start_ns'],p['until_ns']) for p in pauses[start:]]
                                    +list(zip(df.entry_ns,df.exit_bar_open_ns)))
                    assert all(b<c for (_,b),(c,_) in zip(occupied,occupied[1:]))
                accounts.append(dict(**keys_now,mode=mode,origins=len(ts),pauses=len(pauses)-start,
                                     ret=ret,dd_realized=dd,orders=len(df),wins=sum(df.net_usd>0),
                                     buys=sum(df.side=='buy'),sells=sum(df.side=='sell')))
                books.append(df.assign(**keys_now,mode=mode))
        med=cutoffs=None
        evaluate(a,y,ref,old,observe,calibrate)
    a.out.mkdir(parents=True,exist_ok=True)
    for name,data in [('accounts.csv',accounts),('calibration.csv',calibration),('pauses.csv.gz',pauses)]:
        pd.DataFrame(data).to_csv(a.out/name,index=False)
    pd.concat(books,ignore_index=True).to_csv(a.out/'books.csv.gz',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','hist','market','bundle','control','previous','out'):
        p.add_argument(key,type=Path)
    run(p.parse_args())
