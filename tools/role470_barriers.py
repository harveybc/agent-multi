"""Paired offline first-barrier attribution. Counterfactual probes are not a portfolio."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tools.role461_seasonal_account import NS, load, cost_arrays, hit, entry, simulate


def probe(i, side, gain, risk, q, horizon, data, costs):
    t,o,hi,lo,_,_,_,_,gate,flat=data
    profile,spread,roll=costs
    assert gate[i] and gain>0 and risk>0 and q>0
    market=profile['slippage_market']
    en=o[i]+(spread[i] if side>0 else 0)+side*market
    tp,sl=en+side*gain,en-side*risk
    units=10000*.1*50*q/en
    mfe=mae=0.
    for j in range(i,len(t)):
        ask=spread[j] if side<0 else 0.
        if t[j]>=flat[i]:
            assert t[j]==flat[i]
            reason,px='calendar',o[j]+ask
        else:
            touch=hit(side,o[j]+ask,hi[j]+ask,lo[j]+ask,sl,tp,profile['limit_through'])
            if not touch:
                ex=side*(np.array([hi[j],lo[j]])+ask-en)
                mfe=max(mfe,ex.max());mae=max(mae,-ex.min())
                continue
            reason,px=touch
        slip=profile['slippage_stop'] if reason.startswith('sl') else market if reason=='calendar' else 0.
        carry=(roll[j]-roll[i])*profile['swap_long_per_lot' if side>0 else 'swap_short_per_lot']/profile['lot_units']
        net=side*(px-en)-slip-2*profile['commission_per_lot_side']/profile['lot_units']+carry
        friction=market+slip+(spread[i] if side>0 else spread[j])+2*profile['commission_per_lot_side']/profile['lot_units']-carry
        # Include the exit fill but never assume the full exit candle preceded it.
        mfe=max(mfe,side*(px-en));mae=max(mae,-side*(px-en))
        return dict(start=i,end=j,side=side,h=horizon,q=q,reason=reason,
                    win=int(net>0),tp=int(reason.startswith('tp')),dual=int(reason=='sl_dual'),
                    usd=net*units,r=net/risk,friction_usd=friction*units,gross_same_fills_usd=(net+friction)*units,
                    mfe_lower_r=mfe/risk,mae_lower_r=mae/risk,
                    usd_tp_first=(net+(gain+risk+slip if reason=='sl_dual' else 0))*units)
    raise ValueError('missing mandatory settlement')


def main(a):
    p=json.loads(a.policy.read_text())['policy']
    data=list(load(a.src,a.bars,p))
    t,atr=data[0],data[5]
    prof=json.loads((a.bars/'RECIPE.json').read_text())['business']['evaluation_and_policy']['profile']
    costs=(prof,*cost_arrays(t,json.loads((a.bars/'closures.json').read_text()),prof))
    known=pd.read_csv(a.previous).set_index('ts').direct
    d=np.load(a.signals);keep=np.isin(d['ts'],known.index)
    ts=d['ts'][keep];x=d['x'][keep]
    np.testing.assert_array_equal(ts,known.index)
    ii=np.searchsorted(t,ts)
    np.testing.assert_array_equal(t[ii],ts)
    assert np.all(t[ii]-t[ii-1]==NS) and data[8][ii].all()
    dates=pd.to_datetime(ts,utc=True)
    month=np.asarray(dates.month);hour=np.asarray(dates.hour)
    assert (dates.year==2023).all()
    # Purge even unresolved positions at each monthly validation boundary.
    ends=pd.to_datetime(data[9][ii],utc=True)
    safe=(month==np.asarray(ends.month)) & (dates.year==ends.year)
    ttl=(data[9][ii]-t[ii])//NS
    vol=atr[ii-1]/data[4][ii-1]
    observations=[];purged=0;unmatched=0
    for z,i in enumerate(ii):
        v=entry(x[z],x[z],atr[i-1],None,p)
        assert (v[0] if v else 0)==known.iloc[z]
        if not v:continue
        if not safe[z]:
            purged+=1;continue
        side,gain,risk,q,h=v
        # Identical absolute barriers, size fraction and horizon; match without outcomes.
        peers=np.flatnonzero(safe & (month==month[z]) & (hour==hour[z]) &
                             (data[9][ii]!=data[9][i]) & (abs(ttl-ttl[z])<=24) &
                             (abs(np.log(vol/vol[z]))<=np.log(1.25)))
        unmatched+=int(not len(peers))
        for arm,orig,direction in [('signal',i,side),('mirror',i,-side)]+[
                                  ('other_time',ii[k],side) for k in peers]:
            row=probe(orig,direction,gain,risk,q,h,data,costs)
            observations.append(dict(proposal_ns=int(t[i]),probe_ns=int(t[orig]),arm=arm,
                                     month=int(month[z]),proposal_side=side,peers=len(peers),**row))
    df=pd.DataFrame(observations)
    # Independent parity against the production sequential engine, on every overlapping fill.
    path=np.full_like(data[6],np.nan);path[ii-1]=x;data[6]=path
    _,book=simulate(data,(None,None),p,2023,'bridge',False,costs)
    actual=df[df.arm=='signal'].set_index('start')
    checked=0
    for r in book:
        if r['entry_index'] not in actual.index: continue
        row=actual.loc[r['entry_index']]
        assert row.reason==r['reason'] and row.end==r['exit_index'] and row.side==r['side']
        fixed_units=10000*.1*50*r['q']/r['entry_price']
        np.testing.assert_allclose(row.usd,r['pnl']/r['units']*fixed_units,atol=1e-9)
        checked+=1
    assert checked>0
    a.out.mkdir(parents=True,exist_ok=True)
    df.to_csv(a.out/'probes.csv.gz',index=False)
    metrics=['win','tp','dual','usd','r','friction_usd','gross_same_fills_usd','mfe_lower_r','mae_lower_r','usd_tp_first']
    one=df.groupby(['month','proposal_ns','proposal_side','h','peers','arm'])[metrics].mean().reset_index()
    summary=[]
    for sample,sub in [('all',one[one.arm!='other_time']),('matched',one[one.peers>0])]:
        for keys in [[],['proposal_side','h'],['month']]:
            for group,g in sub.groupby(['arm']+keys):
                summary.append(dict(sample=sample,**dict(zip(['arm']+keys,group)),n=len(g),**g[metrics].mean().to_dict()))
    pd.DataFrame(summary).to_csv(a.out/'summary.csv',index=False)
    comp=[]
    for ctrl in ('mirror','other_time'):
        pair=one.pivot(index=['month','proposal_ns'],columns='arm',values='usd')[['signal',ctrl]].dropna()
        delta=(pair.signal-pair[ctrl]).groupby(level='month').agg(['sum','count'])
        rng=np.random.default_rng(470)
        draws=rng.integers(len(delta),size=(10000,len(delta)))
        boot=delta['sum'].to_numpy()[draws].sum(1)/delta['count'].to_numpy()[draws].sum(1)
        comp.append(dict(control=ctrl,n=int(delta['count'].sum()),months=len(delta),
                         positive_months=int((delta['sum']>0).sum()),mean_usd=float(delta['sum'].sum()/delta['count'].sum()),
                         monthly_bootstrap_95=np.quantile(boot,[.025,.975]).tolist()))
    (a.out/'report.json').write_text(json.dumps(dict(opportunities=len(ii),proposals=int((known!=0).sum()),
        purged=purged,unmatched=unmatched,reference_fills=checked,contrasts=comp),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('src','bars','signals','policy','previous','out'):parser.add_argument(key,type=Path)
    main(parser.parse_args())
