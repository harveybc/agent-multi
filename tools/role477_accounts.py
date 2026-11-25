"""Matched fold-local selectors at fixed 471 settings; paired early exits at fixed entries."""
import argparse
import json
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
from tools.role461_seasonal_account import NS, DEFAULT, cost_arrays
from tools.role471_net_selector import account, audit_book, outcomes
from tools.role473_exit_value import compare_position


def evaluate(ts, pred, year, hist, market, bundle):
    bounds = [pd.Timestamp(s,tz='UTC') for s in
              (f'{year-1}-07-01',f'{year}-01-01',f'{year+1}-01-01')]
    if not (np.diff(ts)>0).all() or not np.isfinite(pred).all() or pred.shape!=(len(ts),8):
        raise ValueError('invalid forecast panel')
    if ts.min()<bounds[0].value or ts.max()>=bounds[-1].value:
        raise ValueError('dates outside fold')
    names=['BidOpen','BidHigh','BidLow','BidClose']
    b=pd.read_parquet(market/'bars.parquet',columns=['timestamp',*names],
        filters=[('timestamp','<',bounds[-1])]).set_index('timestamp')
    old=pd.read_parquet(hist/'OHLC.parquet')[['OPEN','HIGH','LOW','CLOSE']].set_axis(names,axis=1)
    b=pd.concat([old,b]).loc[bounds[0]-pd.Timedelta(days=14):]
    t=b.index.as_unit('ns').asi8
    o,hi,lo,c=b[names].to_numpy().T
    tr=np.maximum(hi-lo,np.maximum(abs(hi-np.r_[c[0],c[:-1]]),abs(lo-np.r_[c[0],c[:-1]])))
    atr=pd.Series(tr).rolling(DEFAULT['atr_window']).mean().to_numpy()
    closures=json.loads((hist/'CLOSURES.json').read_text())
    prof=json.loads((bundle/'RECIPE.json').read_text())['business']['evaluation_and_policy']['profile']
    spread,roll=cost_arrays(t,closures,prof)
    gate=np.zeros(len(t),bool);dead=np.zeros(len(t),np.int64)
    for (_,op),(stop,_) in zip(closures,closures[1:]):
        start=pd.Timestamp(op).value;flat=pd.Timestamp(stop).value-12*NS
        active=(t>=start+12*NS)&(t<flat)
        gate[active]=True;dead[active]=flat
    idx=np.searchsorted(t,ts)
    np.testing.assert_array_equal(t[idx],ts)
    idx0=idx.copy()
    eligible=gate[idx]&(t[idx]-t[idx-1]==NS)&(atr[idx-1]>0)
    eligible &= dead[idx]<np.where(ts<bounds[1].value,bounds[1].value,bounds[2].value)
    idx=idx[eligible];times=pd.to_datetime(t[idx],utc=True)
    train=t[idx]<bounds[1].value; test=~train
    p=np.log(atr[idx-1]/c[idx-1])
    theta=times.hour.to_numpy()*np.pi/12; day=times.dayofweek.to_numpy()*2*np.pi/7
    fees=np.array([prof[k] for k in ('slippage_market','slippage_stop','limit_through')]+
        [prof[k]/prof['lot_units'] for k in ('commission_per_lot_side','swap_long_per_lot','swap_short_per_lot')])
    basic=np.column_stack([np.sin(theta),np.cos(theta),np.sin(day),np.cos(day),
           (dead[idx]-t[idx])/NS,p,p*p,spread[idx]/atr[idx-1],fees[0]/atr[idx-1]])
    signals=pred[eligible]/atr[idx-1,None]
    pnl,ends=outcomes(idx,t,o,hi,lo,atr,dead,spread,roll,fees,1.,1.)
    if t[ends[train]].max()>=bounds[1].value:raise ValueError('selector label crosses boundary')
    data=(t,o,hi,lo,c,atr,None,None,gate,dead)
    paths=np.full((len(t),8),np.nan);paths[idx0-1]=pred
    result={}; ledger=[];exits=[]
    for name,features in [('context',basic),('seasonal',np.column_stack([basic,signals,abs(signals)]))]:
        xx=features[train];m=xx.mean(0);s=xx.std(0);s[s<1e-10]=1.
        xx=(xx-m)/s; offset=pnl[train].mean(0)
        coef=np.linalg.solve(xx.T@xx+10.*np.eye(xx.shape[1]),xx.T@(pnl[train]-offset))
        scores=(features[test]-m)/s@coef+offset
        arr=(idx[test],scores,pnl[test],ends[test])
        g=(1.,1.,0.,0.,15.,1.)
        ret,dd,n,w=account(*arr,*g[2:])
        book,verified=audit_book(arr,g,t)
        np.testing.assert_allclose(ret,verified,atol=1e-8)
        early=0;delta=0.
        for row in book:
            ledger.append(dict(selector=name,**row))
            i=np.searchsorted(t,row['entry_ns']);capital=row['balance']-row['net_usd']
            choices,_=compare_position(i,1 if row['side']=='buy' else -1,row['q'],row['hours'],
                capital,1.,1.,data,(prof,spread,roll),paths,DEFAULT)
            np.testing.assert_allclose(choices['hold'][2],row['net_usd'],atol=1e-8)
            j,reason,value=choices['seasonal_sl_pivot']
            gain=value-row['net_usd'];early+=int(reason.startswith('early_'));delta+=gain
            exits.append(dict(selector=name,entry_ns=row['entry_ns'],hold_usd=row['net_usd'],
                exit_ns=int(t[j]),reason=reason,paired_delta_usd=gain))
        result[name]=dict(return_pct=ret,realized_dd_pct=dd,buy=int(sum(n[:3])),
            sell=int(sum(n[3:])),buy_wins=int(sum(w[:3])),sell_wins=int(sum(w[3:])),
            early=early,early_minus_hold_usd=delta,fit_origins=int(train.sum()),eval_origins=int(test.sum()))
    return result,ledger,exits


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','history','market','bundle'):p.add_argument(key,type=Path)
    a=p.parse_args();pred=[]
    for task in ('short','long'):
        folder=a.root/task
        done=json.loads((folder/'done.json').read_text())
        with (folder/'pred.npz').open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=done['predictions']:raise ValueError('hash')
        with np.load(folder/'pred.npz') as ar:
            if pred:np.testing.assert_array_equal(ts,ar['ts'])
            ts=ar['ts'];pred.append(ar['pred'])
    stats,book,exits=evaluate(ts,np.column_stack(pred),int(a.root.name),a.history,a.market,a.bundle)
    pd.DataFrame(book).to_csv(a.root/'book.csv',index=False)
    pd.DataFrame(exits).to_csv(a.root/'exits.csv',index=False)
    tmp=a.root/'accounts.tmp';tmp.write_text(json.dumps(stats,indent=2)+'\n');tmp.replace(a.root/'accounts.json')
