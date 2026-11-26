"""Exact log-return attribution on realized, compounded account paths."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def attribute(books):
    rows=[]
    for (year,arm,variant),g in books.groupby(['year','arm','variant']):
        def path(mode):
            df=g[g['mode']==mode].set_index(['entry_ns','side','hours'],verify_integrity=True)
            return df.assign(logpp=100*np.log1p(df.net_usd/(df.balance-df.net_usd)))
        control=path('both_filtered')
        for mode in ('allow_buy','allow_sell','neither_filtered'):
            alt=path(mode)
            shared=control.index.intersection(alt.index)
            np.testing.assert_allclose(control.loc[shared,'logpp'],alt.loc[shared,'logpp'],atol=1e-10)
            added=alt.loc[~alt.index.isin(shared)]
            lost=control.loc[~control.index.isin(shared)]
            nonpos=added.score<=0
            parts=[added.loc[nonpos,'logpp'].sum(),added.loc[~nonpos,'logpp'].sum(),-lost.logpp.sum()]
            delta=100*np.log(alt.balance.iloc[-1]/control.balance.iloc[-1])
            np.testing.assert_allclose(sum(parts),delta,atol=1e-9)
            rows.append(dict(year=year,arm=arm,variant=variant,mode=mode,
                             new_nonpositive=sum(nonpos),new_positive=sum(~nonpos),displaced=len(lost),
                             new_nonpositive_logpp=parts[0],new_positive_logpp=parts[1],
                             displaced_logpp=parts[2],total_logpp=delta))
    return pd.DataFrame(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    out=parser.parse_args().out
    attribute(pd.read_csv(out/'books.csv.gz')).to_csv(out/'displacement.csv',index=False)
