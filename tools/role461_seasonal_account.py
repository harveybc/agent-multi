"""NON_GOVERNING: zero-cost hourly account, ideal forecast inputs only."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

H = np.array([1, 2, 3, 4, 5, 6, 24, 48])
NS = 3600 * 10**9
DEFAULT = dict(first_long=6, atr_window=24, bins=[0.5, 1.5], min_cell=30,
               favorable_q=0.5, adverse_q=0.5, tp_mult=0.9, sl_mult=1.0,
               dd_floor=0.5, min_rr=1.0, full_rr=3.0, size_power=1.0,
               reversal_atr=0.25, wait_atr=0.05)


def load(root, bars_dir, p):
    b = pd.read_parquet(bars_dir/'bars.parquet')
    b = b[(b.timestamp.dt.year >= 2020) & (b.available_at.dt.year < 2025)].copy()
    t = pd.DatetimeIndex(b.timestamp).as_unit('ns').asi8
    assert np.all(np.diff(t)>0) and np.array_equal(t+NS, pd.DatetimeIndex(b.available_at).as_unit('ns').asi8)
    o,hi,lo,c = b[['BidOpen','BidHigh','BidLow','BidClose']].to_numpy().T
    assert np.isfinite([o,hi,lo,c]).all() and (hi>=np.maximum(o,c)).all() and (lo<=np.minimum(o,c)).all()
    tr=np.maximum(hi-lo,np.maximum(abs(hi-np.r_[c[0],c[:-1]]),abs(lo-np.r_[c[0],c[:-1]])))
    atr=pd.Series(tr).rolling(p['atr_window']).mean().to_numpy()
    # Attribute signals to the completed bar, never to the bar it predicts.
    ds=np.full((len(t),len(H)),np.nan); dp=ds.copy()
    with np.load(root/'dataset_sdelta.npz') as d, np.load(root/'stl_series.npz') as s:
        si=np.searchsorted(s['avail_ns'],t+NS)
        assert np.array_equal(s['avail_ns'][si],t+NS)
        np.testing.assert_allclose(c,s['close'][si],atol=1e-7,rtol=0)
        keep=np.isin(d['ts'],t+NS)
        ix=np.searchsorted(t+NS,d['ts'][keep])
        ds[ix]=np.column_stack((d['y_short'][keep],d['y_long'][keep]))
        for k,h in enumerate(H):
            rows=np.flatnonzero(np.isfinite(ds[:,k]))
            dest=np.searchsorted(t,t[rows]+h*NS)
            inc=dest<len(t)
            ds[rows[~inc],k]=np.nan
            rows,dest=rows[inc],dest[inc]
            assert np.array_equal(t[rows]+h*NS,t[dest]) and np.array_equal(dest,rows+h)
            np.testing.assert_allclose(ds[rows,k],s['seasonal'][si[dest]]-s['seasonal'][si[rows]],atol=1e-8)
            dp[rows,k]=c[dest]-c[rows]
    gate=np.zeros(len(t),bool);deadline=np.zeros(len(t),dtype='int64')
    closures=json.loads((bars_dir/'closures.json').read_text())
    for (_,op),(end,_) in zip(closures,closures[1:]):
        start=pd.Timestamp(op).value; flat=pd.Timestamp(end).value-12*NS
        active=(t>=start+12*NS)&(t<flat)
        gate[active]=True; deadline[active]=flat
    return t,o,hi,lo,c,atr,ds,dp,gate,deadline


def calibrate(data,p):
    t,o,hi,lo,c,atr,ds,dp,_,_=data
    beta=[]; cache=defaultdict(list)
    for k,h in enumerate(H):
        ix=np.flatnonzero((t+(h+1)*NS<pd.Timestamp('2023',tz='UTC').value)&
                          np.isfinite(ds[:,k]) & (atr>0))
        x=ds[ix,k]/atr[ix]; y=dp[ix,k]/atr[ix]
        beta.append(float(x@y/(x@x)))
        for i,v in zip(ix,x):
            for side in (1,-1):
                a=(max(hi[i+1:i+h+1])-c[i])/atr[i]
                b=(c[i]-min(lo[i+1:i+h+1]))/atr[i]
                pair=(max(0,a),max(0,b)) if side==1 else (max(0,b),max(0,a))
                key=int(np.digitize(side*v,p['bins']))
                cache[k,side,key].append(pair);cache[k,side,-1].append(pair)
    table={key:(float(np.quantile(np.asarray(vals)[:,0],p['favorable_q'])),
                float(np.quantile(np.asarray(vals)[:,1],p['adverse_q'])))
           for key,vals in cache.items() if len(vals)>=p['min_cell']}
    return np.asarray(beta),table


def entry(path,raw,atr,cal,p):
    # Candidate exits at a favorable high (long) / low (short), with drawdown
    # only over the path up to that endpoint, not after the proposed exit.
    ids=np.flatnonzero(H>=p['first_long'])
    if not np.isfinite(path[ids]).all() or not atr>0:return None
    winners=[]
    for side in (1,-1):
        peaks=np.r_[0,side*path[ids]]
        dd=np.maximum.accumulate(np.maximum.accumulate(peaks)-peaks)
        for n,k in enumerate(ids,1):
            if peaks[n]<=peaks[n-1] or (n<len(ids) and peaks[n]<peaks[n+1]):continue
            gain=peaks[n];risk=max(dd[n],p['dd_floor']*atr)
            if cal is not None:
                key=int(np.digitize(side*raw[k]/atr,p['bins']))
                f,a=cal.get((k,side,key),cal[k,side,-1])
                gain=min(gain,f*atr);risk=max(risk,a*atr)
            tp=gain*p['tp_mult'];sl=risk*p['sl_mult']
            if tp>0 and sl>0:winners.append((tp/sl,side,tp,sl,int(H[k])))
    if not winners:return None
    ranked=sorted(winners,key=lambda x:-x[0]);rr,side,tp,sl,h=ranked[0]
    if rr<=p['min_rr'] or any(z[1]!=side and np.isclose(rr,z[0]) for z in ranked):return None
    q=np.clip((rr-p['min_rr'])/(p['full_rr']-p['min_rr']),0,1)**p['size_power']
    return side,tp,sl,float(q),h


def exit_signal(side,cur,sl,tp,path,atr,p):
    vals=side*path[np.isfinite(path)]
    if not len(vals):return None
    stop=side*(sl-cur);target=side*(tp-cur)
    for v in vals:
        if v<=stop:return 'early_sl'
        if v>=target:return None
    v=np.r_[0,vals]
    if np.max(np.maximum.accumulate(v)-v)<p['reversal_atr']*atr:return None
    if np.isfinite(path[0]) and side*path[0]>p['wait_atr']*atr:return 'wait'
    return 'early_pivot'


def hit(side,o,hi,lo,sl,tp,through=0.):
    if side*(o-sl)<=0:return 'sl_gap',o
    if side*(o-tp)>=0:return 'tp_gap',o
    stop=(lo<=sl if side==1 else hi>=sl);take=(hi>=tp+through if side==1 else lo<=tp-through)
    if stop:return ('sl_dual' if take else 'sl'),sl
    if take:return 'tp',tp
    return None


def cost_arrays(t,closures,profile):
    spread=np.full(len(t),np.nan); roll=np.zeros(len(t))
    for (_,op),(end,_) in zip(closures,closures[1:]):
        start=pd.Timestamp(op).value; stop=pd.Timestamp(end).value
        a,b=np.searchsorted(t,[start,stop])
        hours=(t[a:b]-start)/NS
        for low,high,value in profile['spread_schedule']:
            spread[a:b][(hours%24>=low)&(hours%24<high)]=value
        spread[a:b][hours<1]=profile['weekly_open_spread']
        for j,r in enumerate(range(start+24*NS,stop,24*NS),1):
            k=np.searchsorted(t,r)
            if a<=k<b:
                roll[k]+=3 if j==profile['triple_swap_rollover_index'] else 1
    assert np.isfinite(spread).all() and (spread>=0).all()
    return spread,np.cumsum(roll)


def simulate(data,cal,p,year,mode,early,costs=None):
    t,o,hi,lo,c,atr,ds,dp,gate,deadline=data
    beta,table=cal
    if mode=='calibrated':path=ds*beta
    elif mode=='price_oracle':path=dp
    elif mode=='persistence':path=ds*0
    else:path=ds
    idx=np.flatnonzero((t>=pd.Timestamp(str(year),tz='UTC').value)&(t<pd.Timestamp(str(year+1),tz='UTC').value))
    if costs is None:
        spread=roll=np.zeros(len(t));market=stop_slip=through=commission=swap_buy=swap_sell=0.
    else:
        profile,spread,roll=costs
        market,stop_slip,through=(profile[k] for k in ('slippage_market','slippage_stop','limit_through'))
        commission=profile['commission_per_lot_side']/profile['lot_units']
        swap_buy,swap_sell=(profile[k]/profile['lot_units'] for k in ('swap_long_per_lot','swap_short_per_lot'))
    def slip(reason):
        return stop_slip if reason.startswith('sl') else (0. if reason.startswith('tp') else market)
    def liquidation(k,s):
        return spread[k] if s<0 else 0.
    def financing(start,end,s,u):
        return (roll[end]-roll[start])*(swap_buy if s>0 else swap_sell)*u
    bal=10000.;equity=[bal];log=[];pos=None;waits=0
    def finish(reason,px):
        nonlocal bal,pos
        s,en,sl,tp,units,flat,rec=pos
        start=rec['entry_index'];swap=financing(start,i,s,units)
        comm=2*units*commission; slip_cost=units*(market+slip(reason))
        spread_cost=units*(spread[start] if s>0 else spread[i])
        px-=s*slip(reason)
        pnl=s*(px-en)*units-comm+swap;bal+=pnl
        rec.update(exit_index=int(i),exit_time=str(pd.Timestamp(t[i],tz='UTC')),exit_price=px,
                   reason=reason,pnl=pnl,spread_usd=spread_cost,slippage_usd=slip_cost,
                   commission_usd=comm,swap_usd=swap)
        # Same position without discretionary close, through original barriers/deadline.
        if reason.startswith('early_'):
            for j in range(i,idx[-1]+1):
                add=liquidation(j,s)
                if t[j]>=flat:later=o[j]+add-s*market;break
                h=hit(s,o[j]+add,hi[j]+add,lo[j]+add,sl,tp,through)
                if h:later=h[1]-s*slip(h[0]);break
            else:later=c[idx[-1]]+liquidation(idx[-1],s)-s*market
            rec['early_minus_hold']=s*(px-later)*units+swap-financing(start,j,s,units)
        log.append(rec);pos=None
    for i in idx:
        prev=i-1;fresh=prev>=0 and t[i]-t[prev]==NS
        existing=pos is not None
        if pos:
            side,en,sl,tp,units,flat,_=pos
            ex=o[i]+liquidation(i,side)
            if t[i]>=flat:
                assert t[i]==flat, 'missing forced fill'
                finish('calendar',ex)
            elif side*(ex-sl)<=0:finish('sl_gap',ex)
            elif side*(ex-tp)>=0:finish('tp_gap',ex)
            elif early and fresh:
                reason=exit_signal(side,c[prev]+liquidation(prev,side),sl,tp,path[prev],atr[prev],p)
                if reason=='wait' and early=='immediate':reason='early_pivot'
                if reason=='wait' and early!='sl_only':waits+=1
                elif reason in ('early_pivot','early_sl'):
                    if early!='sl_only' or reason=='early_sl':
                        if early!='pivot_only' or reason=='early_pivot':finish(reason,ex)
        if not existing and gate[i] and fresh:
            choice=entry(path[prev],ds[prev],atr[prev],table if mode=='calibrated' else None,p)
            if choice:
                side,dist,risk,q,h=choice
                en=o[i]+(spread[i] if side>0 else 0)+side*market;sl=en-side*risk;tp=en+side*dist
                units=bal*0.1*q*50/en
                rec=dict(entry_index=int(i),entry_time=str(pd.Timestamp(t[i],tz='UTC')),side=side,
                         entry_price=en,sl=sl,tp=tp,q=q,h=h,units=units)
                pos=(side,en,sl,tp,units,deadline[i],rec)
        if pos:
            side,en,sl,tp,units,_,_=pos
            add=liquidation(i,side)
            h=hit(side,o[i]+add,hi[i]+add,lo[i]+add,sl,tp,through)
            if h:finish(*h)
        equity.append(bal if not pos else bal+pos[0]*(c[i]+liquidation(i,pos[0])-pos[1])*pos[4]
                      -commission*pos[4]+financing(pos[-1]['entry_index'],i,pos[0],pos[4]))
    assert pos is None and abs(bal-10000-sum(r['pnl'] for r in log))<1e-6
    e=np.asarray(equity);why=Counter(r['reason'] for r in log)
    cf=[r['early_minus_hold'] for r in log if 'early_minus_hold' in r]
    return dict(trades=len(log),buy=sum(r['side']==1 for r in log),wins=int(sum(r['pnl']>0 for r in log)),
                wins_buy=int(sum(r['side']==1 and r['pnl']>0 for r in log)),
                return_pct=(bal/10000-1)*100,dd_close_pct=max(1-e/np.maximum.accumulate(e))*100,
                reasons=dict(why),wait_decisions=waits,early_harmful=int(sum(x<0 for x in cf)),
                early_vs_hold_usd=sum(cf),
                costs_usd={key:sum(r[key] for r in log) for key in
                           ('spread_usd','slippage_usd','commission_usd','swap_usd')}),log


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('source','bars','out'):ap.add_argument(k,type=Path)
    ap.add_argument('--config',type=Path)
    ap.add_argument('--cost-recipe',type=Path)
    a=ap.parse_args();p=DEFAULT.copy()
    if a.config:
        cfg=json.loads(a.config.read_text());assert set(cfg)<=set(p);p.update(cfg)
    assert p['full_rr']>p['min_rr']>=0 and p['dd_floor']>0 and p['size_power']>0
    a.out.mkdir(parents=True,exist_ok=True);data=load(a.source,a.bars,p);cal=calibrate(data,p)
    report=dict(parameters=p,beta=cal[0].tolist(),panels={})
    costs=None
    if a.cost_recipe:
        profile=json.loads(a.cost_recipe.read_text())['business']['evaluation_and_policy']['profile']
        assert profile['entry_delay_bars']==0
        costs=(profile,*cost_arrays(data[0],json.loads((a.bars/'closures.json').read_text()),profile))
        report['execution_profile']=profile
    for year in (2023,2024):
        for mode in ('price_oracle','persistence','bridge','calibrated'):
            for early in ((False,True,'sl_only','pivot_only','immediate') if mode in ('bridge','calibrated') else (True,)):
                label=int(early) if isinstance(early,bool) else early
                key=f'{year}-{mode}-{label}';report['panels'][key],log=simulate(data,cal,p,year,mode,early,costs)
                pd.DataFrame(log).to_csv(a.out/(key+'.csv'),index=False)
    (a.out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
