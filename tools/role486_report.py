"""Descriptive distributions of matched randomized schedules, not holdout p-values."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def run(draws,previous,out):
    ref=pd.read_csv(previous/'accounts.csv').query("arm=='seasonal' and variant=='no_vol_side' and mode=='cash_wait'").set_index('year')
    ledger=pd.read_csv(previous/'books.csv.gz').query("arm=='seasonal' and variant=='no_vol_side' and mode=='cash_wait'")
    hours=((ledger.exit_bar_open_ns-ledger.entry_ns)/3.6e12).groupby(ledger.year).sum()
    annual=[];compound=[]
    for strict in (0,1):
        for nop in (0,1):
            pooled={False:[],True:[]}
            for y in range(2020,2024):
                with np.load(draws/f'{y}-{strict}-{nop}.npz') as d:
                    r,dd,n,h,mv=[d[k][:-1] for k in ('ret','dd','orders','hours','moved')]
                    row=ref.loc[y]
                    np.testing.assert_allclose([d[k][-1] for k in ('ret','dd','orders','hours')],
                                               [row.ret,row.dd_realized,row.orders,hours[y]],atol=1e-8)
                    assert row.pauses==d['pauses'] and len(r)==8192 and d['moved'][-1]==0
                    ok=(abs(n-row.orders)<=.05*row.orders)&(abs(h-hours[y])<=.05*hours[y])
                    for matched,take in [(False,np.ones(len(r),bool)),(True,ok)]:
                        v=r[take]
                        if len(v)<200:raise ValueError('insufficient conditional schedules')
                        pooled[matched].append(v)
                        annual.append(dict(year=y,same_day=strict,nop_only=nop,matched=matched,draws=len(v),
                            actual=row.ret,actual_dd=row.dd_realized,actual_orders=row.orders,actual_hours=hours[y],
                            q05=np.quantile(v,.05),median=np.median(v),q95=np.quantile(v,.95),
                            placebo_ge_pct=100*np.mean(v>=row.ret-1e-9),dd_median=np.median(dd[take]),
                            median_hours=np.median(h[take]),median_orders=np.median(n[take]),
                            median_pauses_moved=np.median(mv[take]),pauses=row.pauses,
                            log10_schedules=float(d['log10_schedules'])))
            actual=100*(np.prod(1+ref.ret/100)-1)
            for matched,years in pooled.items():
                # Independent schedules across years, conditional on each year's exposure if requested.
                rng=np.random.default_rng([1486,strict,nop,int(matched)])
                comp=100*(np.prod([1+rng.choice(v,65536)/100 for v in years],axis=0)-1)
                compound.append(dict(same_day=strict,nop_only=nop,matched=matched,actual=actual,
                    q05=np.quantile(comp,.05),median=np.median(comp),q95=np.quantile(comp,.95),
                    placebo_ge_pct=100*np.mean(comp>=actual-1e-9)))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(annual).to_csv(out/'annual.csv',index=False)
    pd.DataFrame(compound).to_csv(out/'compound.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('draws','previous','out'):p.add_argument(k,type=Path)
    run(**vars(p.parse_args()))
