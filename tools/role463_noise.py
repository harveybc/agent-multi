"""Paired Gaussian factorial on the fixed 462 bridge/sl_only account."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from tools.role461_seasonal_account import load, DEFAULT, H, NS, cost_arrays, simulate

LEVELS=np.array([0,.125,.25,.375,.5,.625,.75,.875,1,1.25,1.5,2,3,4.])
PANEL='bridge-sl_only'


@njit
def run(t,o,hi,lo,c,atr,gate,deadline,spread,roll,sig,fees):
    market,stops,through,comm,swapb,swaps=fees
    side=0;balance=10000.;peak=balance;dd=0.;trades=0;buys=0;wins=0
    en=sl=tp=u=0.;start=flat=0;day=-1;daily=np.empty(len(t)+1);nd=1;daily[0]=balance
    for i in range(len(t)):
        prev=i-1;fresh=i>0 and t[i]-t[prev]==NS;existing=side!=0;done=False;px=0.
        if existing:
            ex=o[i]+(spread[i] if side<0 else 0.)
            if t[i]>=flat:done=True;px=ex-side*market
            elif side*(ex-sl)<=0:done=True;px=ex-side*stops
            elif side*(ex-tp)>=0:done=True;px=ex
            elif fresh:
                cur=c[prev]+(spread[prev] if side<0 else 0.)
                for k in range(8):
                    v=side*sig[prev,k]
                    if v<=side*(sl-cur):done=True;px=ex-side*market;break
                    if v>=side*(tp-cur):break
        if not existing and fresh and gate[i] and atr[prev]>0 and np.all(np.isfinite(sig[prev,5:])):
            best=0.;second=0.;direction=0;td=sd=0.
            for s in (1,-1):
                top=draw=last=0.;sb=0.;st=ss=0.
                for k in range(5,8):
                    v=s*sig[prev,k];top=max(top,v);draw=max(draw,top-v)
                    if v>last and (k==7 or v>=s*sig[prev,k+1]):
                        gain=.9*v;risk=max(draw,.5*atr[prev]);score=gain/risk
                        if score>sb:sb=score;st=gain;ss=risk
                    last=v
                if sb>best:second=best;best=sb;td=st;sd=ss;direction=s
                else:second=sb
            if best>1 and abs(second-best)>1e-8+1e-5*best:
                side=direction;en=o[i]+(spread[i] if side>0 else 0.)+side*market
                sl=en-side*sd;tp=en+side*td;u=balance*.1*min(1.,(best-1.)/2)*50/en
                start=i;flat=deadline[i];buys+=int(side==1)
        if side and not done:
            a=spread[i] if side<0 else 0.;ex=o[i]+a
            if side*(ex-sl)<=0:done=True;px=ex-side*stops
            elif side*(ex-tp)>=0:done=True;px=ex
            elif (lo[i]+a<=sl if side>0 else hi[i]+a>=sl):done=True;px=sl-side*stops
            elif (hi[i]+a>=tp+through if side>0 else lo[i]+a<=tp-through):done=True;px=tp
        if done:
            pnl=(side*(px-en)-2*comm+(roll[i]-roll[start])*(swapb if side>0 else swaps))*u
            balance+=pnl;trades+=1;wins+=int(pnl>0);side=0
        eq=balance
        if side:
            ex=c[i]+(spread[i] if side<0 else 0.)
            eq+=u*(side*(ex-en)-comm+(roll[i]-roll[start])*(swapb if side>0 else swaps))
        peak=max(peak,eq);dd=max(dd,1-eq/peak)
        today=(t[i]+NS-1)//(24*NS)
        if day!=today:nd+=1;day=today
        daily[nd-1]=eq
    assert side==0
    r=daily[1:nd]/daily[:nd-1]-1
    sharpe=np.sqrt(252.)*np.mean(r)/np.sqrt(np.sum((r-np.mean(r))**2)/(len(r)-1)) if np.std(r)>0 else np.nan
    return (balance/100.,100.*dd,trades,buys,wins,sharpe)


def main(a):
    a.out.mkdir(exist_ok=True,parents=True)
    data=load(a.src,a.bars,DEFAULT);t,o,hi,lo,c,atr,ds,dp,gate,dead=data
    prof=json.loads(a.recipe.read_text())['business']['evaluation_and_policy']['profile']
    spr,roll=cost_arrays(t,json.loads((a.bars/'closures.json').read_text()),prof)
    fees=np.array([prof[k] for k in ('slippage_market','slippage_stop','limit_through')]+
                  [prof[k]/prof['lot_units'] for k in ('commission_per_lot_side','swap_long_per_lot','swap_short_per_lot')])
    rows=[];checks=[];noise=[];controls=[];matched=[];real=np.load(a.real)
    for year in (2023,2024):
        m=(t>=pd.Timestamp(str(year),tz='UTC').value)&(t<pd.Timestamp(str(year+1),tz='UTC').value)
        cols=[np.ascontiguousarray(col[m]) for col in (t,o,hi,lo,c,atr,gate,dead,spr,roll)]
        truth=ds[m].copy();naive=np.nanmean(abs(truth),axis=0)
        def evaluate(pred,check=False):
            val=list(run(*cols,np.ascontiguousarray(pred),fees));val[0]-=100
            if check:
                d=list(data);d[6]=ds.copy();d[6][m]=pred
                v,_=simulate(d,(None,None),DEFAULT,year,'bridge','sl_only',(prof,spr,roll))
                np.testing.assert_allclose(val[:5],[v[k] for k in ('return_pct','dd_close_pct','trades','buy','wins')],rtol=0,atol=1e-7)
                checks.append([year,*val[:5]])
            return val
        def control(name,pred):
            e=np.nanmean(abs(pred-truth),axis=0)
            controls.append([year,name,*evaluate(pred,True),*e,*naive])
        control('oracle',truth);control('persistence',truth*0)
        profiles={}
        if year==2024:
            for mode in ('frozen','tuned'):
                pred=np.full(truth.shape,np.nan)
                ix=np.searchsorted(t[m]+NS,real['ts']);assert np.array_equal((t[m]+NS)[ix],real['ts'])
                pred[ix]=real[mode];pred[~np.isfinite(truth)]=np.nan
                assert np.array_equal(np.isfinite(pred),np.isfinite(truth))
                control(mode,pred)
                for part,sl in [('short',slice(0,5)),('long',slice(5,8))]:
                    hybrid=truth.copy();hybrid[:,sl]=pred[:,sl]
                    control(f'{mode}_{part}_other_ideal',hybrid)
                profiles[mode]=np.nanmean(abs(pred-truth),axis=0)
        for seed in range(a.seeds):
            rng=np.random.default_rng(np.random.SeedSequence([463,year,seed]))
            unit=rng.normal(size=truth.shape)*naive*np.sqrt(np.pi/2)
            unit[~np.isfinite(truth)]=np.nan
            mae1=np.nanmean(abs(unit),axis=0)
            for name,profile in profiles.items():
                pred=truth+unit*(profile/mae1)
                np.testing.assert_allclose(np.nanmean(abs(pred-truth),axis=0),profile,rtol=1e-10)
                matched.append([year,seed,name,*evaluate(pred)])
            noise.extend([year,seed,int(h),n,int(np.isfinite(truth[:,j]).sum()),err]
                         for j,(h,n,err) in enumerate(zip(H,naive,mae1)))
            preds=[truth+x*unit for x in LEVELS]
            buf=truth.copy()
            for j,s in enumerate(LEVELS):
                buf[:,:5]=preds[j][:,:5]
                for k,l in enumerate(LEVELS):
                    buf[:,5:]=preds[k][:,5:]
                    val=evaluate(buf,seed==0 and (s,l) in ((1,0),(0,1),(1,1),(2,3)))
                    rows.append([year,seed,s,l,*val])
    metric=['return_pct','dd_pct','orders','buys','wins','sharpe']
    pd.DataFrame(rows,columns=['year','seed','short','long',*metric]).to_csv(a.out/'sweep.csv.gz',index=False)
    pd.DataFrame(noise,columns=['year','seed','h','naive','n','achieved_mae_at_1']).to_csv(a.out/'noise.csv.gz',index=False)
    pd.DataFrame(controls,columns=['year','control',*metric,*[f'mae_{h}' for h in H],*[f'naive_{h}' for h in H]]).to_csv(a.out/'controls.csv',index=False)
    pd.DataFrame(matched,columns=['year','seed','model',*metric]).to_csv(a.out/'matched.csv',index=False)
    (a.out/'reference_checks.json').write_text(json.dumps(checks,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('src','bars','recipe','real','out'):p.add_argument(k,type=Path)
    p.add_argument('--seeds',type=int,default=64);main(p.parse_args())
