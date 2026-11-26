"""Compare frozen net-utility and explicit-cost selectors in the 477 accounts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tools.role470_barriers import probe
from tools.role471_net_selector import account, audit_book, outcomes
from tools.role477_accounts import opportunity_panel
from tools.role478_direction import load, preferences, LIVE_H


def ridge(xfit, yfit, x):
    mu=xfit.mean(0); sd=xfit.std(0); sd[sd<1e-10]=1.
    z=(xfit-mu)/sd; off=yfit.mean(0)
    return (x-mu)/sd@np.linalg.solve(z.T@z+10*np.eye(z.shape[1]),z.T@(yfit-off))+off


def known_friction(spread,market,commission):
    # Bid-price attribution of 470: buy pays spread on entry, sell on exit.
    return np.asarray(spread)[:,None]*np.array([1,1,1,0,0,0])+market+2*commission


def estimate_cost(residual_fit, known_eval, last_close):
    # Unknown exit slippage, exit spread (SELL) and carry in price units.
    # Convert at last observed close; execution still settles at actual fill.
    return (known_eval+residual_fit.mean(axis=0))*5e4/last_close[:,None]


def lock_direction(gross,net):
    buy=gross[:,:3].max(1);sell=gross[:,3:].max(1)
    out=net.copy()
    out[buy<=sell,:3]=0.
    out[sell<=buy,3:]=0.
    return out


def fill_components(rows,idx,data,costs,end,pnl):
    atr=data[5]
    friction=np.empty((len(rows),6));gross=np.empty_like(friction)
    for j,r in enumerate(rows):
        i=idx[r]
        for k,h in enumerate(LIVE_H*2):
            w=atr[i-1]*np.sqrt(h)
            ref=probe(i,1 if k<3 else -1,w,w,1.,h,data,costs)
            if ref['end']!=end[r,k] or abs(ref['usd']-pnl[r,k])>1e-8:
                raise ValueError('fill or PnL disagrees')
            gross[j,k]=ref['gross_same_fills_usd'];friction[j,k]=ref['friction_usd']
    np.testing.assert_allclose(gross-friction,pnl[rows],atol=1e-8)
    return gross,friction


def one(a,y):
    ts,pred,hashes=load(a.root,y); folder=a.root/str(y)
    idx,fit,basic,sig,data,costs,fees,_=opportunity_panel(ts,pred,y,a.hist,a.market,a.bundle)
    t,o,hi,lo,close,atr,_,_,_,dead=data
    _,spread,roll=costs
    pnl,end=outcomes(idx,t,o,hi,lo,atr,dead,spread,roll,fees,1.,1.)
    if t[end[fit]].max()>=pd.Timestamp(str(y),tz='UTC').value:raise ValueError('uncleared calibration positions')
    market,_,_,commission,_,_=fees
    known=known_friction(spread[idx],market,commission)
    tr=np.flatnonzero(fit);ev=np.flatnonzero(~fit);ii=idx[ev]
    gross,friction=fill_components(tr,idx,data,costs,end,pnl)
    factors=np.where(np.arange(6)<3, o[idx[tr],None]+spread[idx[tr],None]+market,
                     o[idx[tr],None]-market)/5e4
    residual=friction*factors-known[tr]
    costs_hat=estimate_cost(residual,known[ev],close[ii-1])
    assert np.isfinite(costs_hat).all()
    booked=pd.read_csv(folder/'book.csv')
    ref_account=json.loads((folder/'accounts.json').read_text())
    records=[];books=[]; models=[]
    for arm in ('context','seasonal'):
        x=basic if arm=='context' else np.column_stack([basic,sig,abs(sig)])
        clean=np.delete(x,[7,8],axis=1)
        direct=ridge(x[tr],pnl[tr],x[ev])
        clean_direct=ridge(clean[tr],pnl[tr],clean[ev])
        gross_hat=ridge(clean[tr],gross,clean[ev])
        net=gross_hat-costs_hat
        # Linearity control: modelling both parts by identical ridge must
        # reproduce the direct net model, not an apparent advantage.
        np.testing.assert_allclose(gross_hat-ridge(clean[tr],friction,clean[ev]),clean_direct,atol=1e-8)
        for variant,v in [('baseline',direct),('net_without_cost_inputs',clean_direct),
                          ('gross_minus_cost',net),('gross_side_then_cost',lock_direction(gross_hat,net))]:
            arr=(ii,v,pnl[ev],end[ev]);g=(1.,1.,0.,0.,15.,1.)
            ret,dd,n,w=account(*arr,*g[2:]);book,verified=audit_book(arr,g,t)
            np.testing.assert_allclose(verified,ret,atol=1e-8)
            assert len(book)==sum(n)
            if variant=='baseline':
                orig=booked[booked.selector==arm]; rep=pd.DataFrame(book)
                for k in ('entry_ns','side','hours','exit_bar_open_ns'):np.testing.assert_array_equal(rep[k],orig[k])
                for k in ('net_usd','balance','q'):np.testing.assert_allclose(rep[k],orig[k],rtol=1e-11,atol=1e-8)
                np.testing.assert_allclose(ret,ref_account[arm]['return_pct'],atol=1e-8)
            pref=preferences(v)
            records.append(dict(year=y,arm=arm,variant=variant,ret=ret,dd_realized=dd,
                buy=int(sum(n[:3])),sell=int(sum(n[3:])),buy_wins=int(sum(w[:3])),sell_wins=int(sum(w[3:])),
                propose_buy=int(sum(pref==1)),propose_sell=int(sum(pref==-1)),nop=int(sum(pref==0)),
                origins=len(ii),fitted=len(tr)))
            books.extend(dict(year=y,arm=arm,variant=variant,**r) for r in book)
    # Realized evaluation costs are accessed only after all decisions are made.
    _,real_cost=fill_components(ev,idx,data,costs,end,pnl)
    for k,h in enumerate(LIVE_H*2):
        models.append(dict(year=y,h=h,side='buy' if k<3 else 'sell',
            mean_residual_price_fit=residual[:,k].mean(),mean_predicted_bp=costs_hat[:,k].mean(),
            mean_realized_bp=real_cost[:,k].mean(),mae_bp=np.abs(costs_hat[:,k]-real_cost[:,k]).mean()))
    return records,books,models,dict(year=y,prediction_hashes=hashes,checked_fills=6*len(idx),
        verified_original_orders=len(booked),fit_last_exit_ns=int(t[end[fit]].max()),first_eval_ns=int(t[ii[0]]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','hist','market','bundle','out'):p.add_argument(k,type=Path)
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    runs=[one(a,y) for y in range(2020,2024)]
    for i,f in enumerate(('accounts.csv','books.csv.gz','costs.csv')):
        pd.DataFrame([row for run in runs for row in run[i]]).to_csv(a.out/f,index=False)
    (a.out/'audit.json').write_text(json.dumps([r[-1] for r in runs],indent=2)+'\n')
