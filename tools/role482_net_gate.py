"""Retrospective gate attribution; no threshold search or portfolio counterfactual."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.role481_direction_volatility import evaluate, LIVE_H


def proposals(ts, direction, net, pnl, book):
    b, s = direction[:,:3].max(1), direction[:,3:].max(1)
    rows = np.flatnonzero(b != s)
    offset = 3*(s[rows] > b[rows]).astype(int)
    k = offset + net[rows[:,None],offset[:,None]+np.arange(3)].argmax(1)
    ts = ts[rows]
    score = net[rows,k]
    actual = pnl[rows,k]
    en, ex = book.entry_ns.to_numpy(), book.exit_bar_open_ns.to_numpy()
    assert (np.diff(en)>0).all() and (en[1:]>ex[:-1]).all()
    prior = en.searchsorted(ts,side='left')-1
    busy = (prior>=0) & (ts<=ex[prior]) if len(ex) else np.zeros(len(ts),bool)
    out = pd.DataFrame(dict(ts=ts,side=np.where(k<3,'buy','sell'),h=np.array(LIVE_H)[k%3],
                            approved=score>0,busy=busy,score=score,net_usd_q1=actual,win=actual>0))
    fills = out.loc[out.approved & ~out.busy]
    for col, ref in [('ts','entry_ns'),('h','hours'),('side','side')]:
        np.testing.assert_array_equal(fills[col],book[ref])
    np.testing.assert_allclose(fills.score,book.edge_bp,atol=1e-8)
    capital = book.balance - book.net_usd
    np.testing.assert_allclose(fills.net_usd_q1.to_numpy()*book.q*capital/1e4,book.net_usd,atol=1e-8)
    return out


def summarize(df, keys):
    return df.groupby(keys,as_index=False).agg(n=('win','size'),wins=('win','sum'),
        mean_net_usd=('net_usd_q1','mean'),median_net_usd=('net_usd_q1','median'),
        mean_score=('score','mean'))


def run(a):
    keys=['year','arm','variant']
    ref=pd.read_csv(a.control/'accounts.csv').set_index(keys)
    old=pd.read_csv(a.control/'books.csv.gz').set_index(keys).sort_index()
    archived=pd.read_csv(a.previous/'books.csv.gz').set_index(keys).sort_index()
    frames=[]
    counts=[]
    def observe(year,arm,variant,ts,direction,net,pnl):
        df=proposals(ts,direction,net,pnl,archived.loc[(year,arm,variant)])
        frames.append(df.assign(year=year,arm=arm,variant=variant))
        counts.append(dict(year=year,arm=arm,variant=variant,origins=len(ts),side_ties=len(ts)-len(df)))
    for year in range(2020,2024):
        evaluate(a,year,ref,old,observe)
    data=pd.concat(frames,ignore_index=True)
    a.out.mkdir(exist_ok=True,parents=True)
    for name, extra in [('sides',[]),('horizons',['h']),('availability',['busy']),
                        ('detail',['h','busy'])]:
        summarize(data,keys+['side','approved']+extra).to_csv(a.out/(name+'.csv'),index=False)
    pd.DataFrame(counts).to_csv(a.out/'counts.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','hist','market','bundle','control','previous','out'):
        p.add_argument(k,type=Path)
    run(p.parse_args())
