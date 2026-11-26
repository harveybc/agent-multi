"""Freeze 485 account opportunities and verify the reference before randomization."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tools.role481_direction_volatility import evaluate
from tools.role482_net_gate import side_horizon


def fold(a,y):
    def read(folder,name):
        return pd.read_csv(folder/name).set_index(['year','arm','variant']).sort_index()
    ref=read(a.control,'accounts.csv')
    old=read(a.control,'books.csv.gz')
    key=(y,'seasonal','no_vol_side')
    p=read(a.previous,'pauses.csv.gz').loc[key]
    book=read(a.previous,'books.csv.gz').loc[key].query('mode=="cash_wait"')
    def calibrate(year,t,exits,dead):
        nonlocal week
        week=dead
    def observe(year,arm,variant,t,direction,net,pnl,ends):
        if (year,arm,variant)!=key:
            return
        rows,k=side_horizon(direction,net)
        np.testing.assert_array_equal(rows,np.arange(len(t)))
        score=net[rows,k]
        ex=ends[rows,k]
        outcome=pnl[rows,k]
        start=p.start_ns.to_numpy()
        until=p.until_ns.to_numpy()
        deadlines=dict(zip(start,until))
        taken=[]; waited=[]; free=-1
        for i,now in enumerate(t):
            if now<=free:continue
            if now in deadlines:
                assert k[i]<3 and score[i]<=0
                free=deadlines[now]; waited.append(now)
            elif score[i]>0:
                taken.append(i);free=ex[i]
        np.testing.assert_array_equal(waited,start)
        np.testing.assert_array_equal(t[taken],book.entry_ns)
        np.testing.assert_array_equal(ex[taken],book.exit_bar_open_ns)
        np.testing.assert_allclose(10000*np.cumprod(1+outcome[taken]/10000),book.balance,rtol=1e-11)
        np.savez_compressed(a.out/f'{year}.npz',ts=t,week=week,score=score,exit=ex,pnl=outcome,
                            starts=start,until=until)
    week=None
    evaluate(a,y,ref,old,observe,calibrate)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','hist','market','bundle','control','previous','out'):
        p.add_argument(key,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(fold,[a]*4,range(2020,2024)))
