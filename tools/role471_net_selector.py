"""Matched net utility regressors with prequential selector predictions and DEAP."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from deap import base, creator
from tools.role461_seasonal_account import DEFAULT, NS, cost_arrays, load
from tools.role464_deap import search
from tools.role470_barriers import probe


@njit
def outcomes(ii,t,o,hi,lo,atr,dead,spread,roll,fees,tm,sm,horizons=(6,24,48)):
    market,stop,through,comm,swapb,swaps=fees
    n=len(ii)
    nh=len(horizons)
    net=np.zeros((n,2*nh));end=np.zeros((n,2*nh),np.int64)
    for z,i in enumerate(ii):
        for k in range(2*nh):
            side=1 if k<nh else -1
            width=atr[i-1]*np.sqrt(horizons[k%nh])
            en=o[i]+(spread[i] if side>0 else 0.)+side*market
            tp=en+side*width*tm;sl=en-side*width*sm
            for j in range(i,len(t)):
                add=spread[j] if side<0 else 0.
                op=o[j]+add
                if t[j]>=dead[i]:
                    assert t[j]==dead[i]
                    px=op-side*market
                elif side*(op-sl)<=0:px=op-side*stop
                elif side*(op-tp)>=0:px=op
                elif (lo[j]+add<=sl if side>0 else hi[j]+add>=sl):px=sl-side*stop
                elif (hi[j]+add>=tp+through if side>0 else lo[j]+add<=tp-through):px=tp
                else:continue
                net[z,k]=(side*(px-en)-2*comm+(roll[j]-roll[i])*(swapb if side>0 else swaps))*5e4/en
                end[z,k]=j
                break
            assert end[z,k]>=i
    return net,end


@njit
def account(ii,score,pnl,ends,threshold,margin,full,power):
    bal=peak=10000.;dd=0.;free=-1
    counts=np.zeros(6,np.int64);wins=np.zeros(6,np.int64)
    for j,i in enumerate(ii):
        if i<=free:continue
        k=np.argmax(score[j]);s=score[j,k]
        opposite=np.max(score[j,3:]) if k<3 else np.max(score[j,:3])
        if s<=threshold or s-opposite<=margin:continue
        q=min(1.,(s-threshold)/full)**power
        earned=q*pnl[j,k]
        bal*=1+earned/10000
        counts[k]+=1;wins[k]+=int(earned>0)
        peak=max(peak,bal);dd=max(dd,1-bal/peak)
        free=ends[j,k]
    return (bal/100-100),100*dd,counts,wins


def criterion(metrics,monitor):
    need=12 if monitor else 24
    low=min(v['orders'] for v in metrics)
    return float(np.mean([v['ret'] for v in metrics])) if low>=need else -1000.+low


def prepare(a,cfg):
    data=load(a.src,a.bars,DEFAULT,a.prefix)
    t,o,hi,lo,c,atr,_,_,gate,dead=data
    prof=json.loads((a.bars/'RECIPE.json').read_text())['business']['evaluation_and_policy']['profile']
    spread,roll=cost_arrays(t,json.loads((a.prefix/'CLOSURES.json').read_text()),prof)
    fees=np.array([prof[k] for k in ('slippage_market','slippage_stop','limit_through')]+
                  [prof[k]/prof['lot_units'] for k in ('commission_per_lot_side','swap_long_per_lot','swap_short_per_lot')])
    d=np.load(a.signals);ts=d['ts'];ii=np.searchsorted(t,ts)
    np.testing.assert_array_equal(t[ii],ts)
    dt=pd.to_datetime(ts,utc=True)
    split=np.where(dt.year==2023,20231+(dt.month>6).astype(int),dt.year)
    bound=np.array([pd.Timestamp('2023-07-01' if y==20231 else '2024' if y==20232 else str(y+1),tz='UTC').value for y in np.unique(split)])
    cut=bound[np.searchsorted(np.unique(split),split)]
    m=gate[ii] & (ii>0) & (t[ii]-t[ii-1]==NS) & (dead[ii]<cut) & (atr[ii-1]>0)
    ii=ii[m];split=split[m];dt=dt[m]
    assert set(split)=={2019,2020,2021,2022,20231,20232}
    v=atr[ii-1]/c[ii-1]
    phase=dt.hour.to_numpy()*2*np.pi/24; dow=dt.dayofweek.to_numpy()*2*np.pi/7
    basic=np.column_stack([np.sin(phase),np.cos(phase),np.sin(dow),np.cos(dow),
        (dead[ii]-t[ii])/NS,np.log(v),np.log(v)**2,spread[ii]/atr[ii-1],fees[0]/atr[ii-1]])
    signal=d['x'][m]/atr[ii-1,None]
    xs={'context':basic,'seasonal':np.column_stack([basic,signal,np.abs(signal)])}
    arrays={}; fits={}; reference_checks=0
    for tm in cfg['genes']['tp_atr_sqrt_h']['values']:
        for sm in cfg['genes']['sl_atr_sqrt_h']['values']:
            net,end=outcomes(ii,t,o,hi,lo,atr,dead,spread,roll,fees,tm,sm)
            for j in np.linspace(0,len(ii)-1,18,dtype=int):
                for k in range(6):
                    h=(6,24,48)[k%3];width=atr[ii[j]-1]*np.sqrt(h)
                    ref=probe(ii[j],1 if k<3 else -1,tm*width,sm*width,1.,h,data,(prof,spread,roll))
                    np.testing.assert_allclose(net[j,k],ref['usd'],atol=1e-8)
                    assert end[j,k]==ref['end'];reference_checks+=1
            for name,features in xs.items():
                for fold in [*cfg['train_years'],20231,20232]:
                    train=split<min(fold,20231);test=split==fold
                    assert t[end[train].max()]<t[ii[test].min()]
                    x=features[train];mu=x.mean(0);std=x.std(0);std[std<1e-10]=1.
                    x=(x-mu)/std;y=net[train];offset=y.mean(0)
                    co=np.linalg.solve(x.T@x+cfg['ridge']*np.eye(x.shape[1]),x.T@(y-offset))
                    scores=(features[test]-mu)/std@co+offset
                    assert np.isfinite(scores).all()
                    arrays[name,tm,sm,fold]=(ii[test],scores,net[test],end[test])
                    if fold==20231:
                        fits[name,tm,sm]=dict(mean=mu.tolist(),scale=std.tolist(),offset=offset.tolist(),coef=co.tolist())
    return arrays,{int(s):int((split==s).sum()) for s in np.unique(split)},reference_checks,t,fits


def audit_book(arr,g,t):
    ii,pred,pnl,ends=arr
    _,_,threshold,margin,full,power=g
    ledger=[];bal=10000.;busy=-1
    for row,i in enumerate(ii):
        if i<=busy:continue
        v=pred[row];buy=int(np.argmax(v[:3]));sell=int(np.argmax(v[3:]))+3
        k=buy if v[buy]>v[sell] else sell
        if v[k]<=threshold or abs(v[buy]-v[sell])<=margin:continue
        q=min(1.,(v[k]-threshold)/full)**power
        busy=ends[row,k];gain=bal*q*pnl[row,k]/10000;bal+=gain
        ledger.append(dict(entry_ns=int(t[i]),exit_bar_open_ns=int(t[busy]),side='buy' if k<3 else 'sell',
                           hours=(6,24,48)[k%3],edge_bp=v[k],q=q,net_usd=gain,balance=bal))
    return ledger,bal/100-100


def main(a):
    cfg=json.loads(a.config.read_text());out=a.out;out.mkdir(exist_ok=True,parents=True)
    arrays,rows,checks,t,fits=prepare(a,cfg)
    specs=cfg['genes'];baseline=tuple(g['baseline'] for g in specs.values())
    def evaluate(name,genes,fold):
        tm,sm,*g=genes
        ret,dd,n,w=account(*arrays[name,tm,sm,fold],*g)
        return dict(ret=ret,dd_realized=dd,orders=int(sum(n)),wins=int(sum(w)),
                    by_action=n.tolist(),wins_by_action=w.tolist())
    names=['context','seasonal']
    folds=cfg['train_years']+[cfg['validation_year']]
    fixed={name:{y:evaluate(name,baseline,y) for y in folds} for name in names}
    (out/'fixed.json').write_text(json.dumps(fixed,indent=2)+'\n')
    creator.create('Fit464',base.Fitness,weights=(1.,));creator.create('Strategy464',list,fitness=creator.Fit464)
    winners={};runs={}
    for name in names:
        folder=out/name;folder.mkdir(exist_ok=True)
        def ev(g,y,paid):return evaluate(name,g,y)
        runs[name]=[search(cfg,ev,True,seed,folder,score_fn=criterion) for seed in cfg['seeds']]
        winners[name]=max(runs[name],key=lambda r:criterion([r['val']],True))
    (out/'locked.json').write_text(json.dumps(winners,indent=2)+'\n')
    report={};books=[];models={}
    for name in names:
        report[name]={}
        for arm,g in [('fixed',baseline),('deap',tuple(winners[name]['effective'].values()))]:
            report[name][arm]={}
            models[name+'_'+arm]=fits[name,*g[:2]]
            for y in folds+[cfg['confirmation']]:
                stats=evaluate(name,g,y)
                book,ret=audit_book(arrays[name,*g[:2],y],g,t)
                np.testing.assert_allclose(ret,stats['ret'],atol=1e-8)
                assert len(book)==stats['orders']
                books.extend(dict(selector=name,arm=arm,fold=y,**r) for r in book)
                report[name][arm][y]=stats
    (out/'selector_weights.json').write_text(json.dumps(models)+'\n')
    pd.DataFrame(books).to_csv(out/'accounts.csv.gz',index=False)
    (out/'results.json').write_text(json.dumps(dict(rows=rows,reference_checks=checks,panels=report,runs=runs),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('src','bars','prefix','signals','config','out'):p.add_argument(k,type=Path)
    main(p.parse_args())
