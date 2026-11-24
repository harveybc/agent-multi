"""Learn CLOSE-now minus passive HOLD in risk units. Score only the first exit."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from tools.role461_seasonal_account import DEFAULT, NS, cost_arrays, load
from tools.role471_net_selector import outcomes

STATE=['buy','sell','h6','h24','h48','to_stop_R','to_target_R','profit_R','abs_profit_R',
       'held_days','remaining_days','fraction_elapsed','atr_R','spread_R','profit_x_elapsed']
NAMES=STATE+[f'{name}_{h}' for name in ('aligned_R','abs_R') for h in (1,2,3,4,5,6,24,48)]


def candidates(ii,ks,data,spread,roll,fees,pred,tm,sm,net,end):
    t,o,_,_,c,atr,_,_,_,flat=data
    rows=np.arange(len(ii))
    length=end[rows,ks]-ii
    owner=np.repeat(rows,length)
    i=ii[owner];k=ks[owner]
    j=i+1+np.arange(len(i))-np.repeat(np.cumsum(length)-length,length)
    side=np.where(k<3,1,-1);h=np.array([6,24,48])[k%3]
    w=atr[i-1]*np.sqrt(h);risk=sm*w
    en=o[i]+np.where(side>0,spread[i],0)+side*fees[0]
    # Calendar and opening gaps execute before a discretionary exit. The opening
    # quote is used for eligibility and labels, never as a regression input.
    op=o[j]+np.where(side<0,spread[j],0)
    move=side*(op-en)
    valid=(t[j]<flat[i])&(move>-risk)&(move<tm*w)&(t[j]-t[j-1]==NS)&np.isfinite(pred[j-1]).all(axis=1)
    owner,i,k,j,side,h,w,risk,en,op=(v[valid] for v in (owner,i,k,j,side,h,w,risk,en,op))
    cur=c[j-1]+np.where(side<0,spread[j-1],0)
    profit=side*(cur-en)/risk
    fraction=(t[j]-t[i])/(flat[i]-t[i])
    aligned=side[:,None]*pred[j-1]/risk[:,None]
    x=np.column_stack([side>0,side<0,h==6,h==24,h==48,1+profit,tm/sm-profit,profit,abs(profit),
       (t[j]-t[i])/(24*NS),(flat[i]-t[j])/(24*NS),fraction,atr[j-1]/risk,spread[j-1]/risk,
       profit*fraction,aligned,abs(aligned)])
    early=side*(op-en)-fees[0]-2*fees[3]+(roll[j]-roll[i])*np.where(side>0,fees[4],fees[5])
    y=(early-net[owner,k]*en/5e4)/risk
    counts=np.bincount(owner,minlength=len(ii))
    return owner,j,x,y,counts


def fit(stats,n,penalty=10.):
    gram,cross=stats
    weight=gram[0,0]
    cols=slice(1,n+1);mu=gram[0,cols]/weight;offset=cross[0]/weight
    cov=gram[cols,cols]-weight*np.outer(mu,mu)
    scale=np.sqrt(np.maximum(0.,np.diag(cov)/weight));scale[scale<1e-8]=1
    coef=np.linalg.solve(cov/np.outer(scale,scale)+penalty*np.eye(n),(cross[cols]-weight*mu*offset)/scale)
    return dict(features=NAMES[:n],mean=mu.tolist(),scale=scale.tolist(),coef=coef.tolist(),offset=float(offset))


def first_close(scores):
    hits=np.flatnonzero(scores>0)
    return int(hits[0]) if len(hits) else None


@njit
def versus_wait(owner, passive_advantage, close):
    """Realized CLOSE minus the first policy exit strictly after this decision."""
    label=passive_advantage.copy()
    following=np.full(len(label),-1,np.int64)
    next_exit=-1
    for j in range(len(label)-1,-1,-1):
        if j==len(label)-1 or owner[j]!=owner[j+1]:
            next_exit=-1
        if next_exit!=-1:
            label[j]-=passive_advantage[next_exit]
            following[j]=next_exit
        if close[j]:
            next_exit=j
    return label,following


def predict(x,model):
    n=len(model['coef'])
    return (x[:,:n]-model['mean'])/model['scale']@model['coef']+model['offset']


def main(a):
    cfg=json.loads((a.prev/'locked.json').read_text())['context']['effective']
    tm,sm=cfg['tp_atr_sqrt_h'],cfg['sl_atr_sqrt_h']
    data=load(a.src,a.bars,DEFAULT,a.prefix)
    t,o,hi,lo,_,atr,_,_,gate,flat=data
    p=json.loads((a.bars/'RECIPE.json').read_text())['business']['evaluation_and_policy']['profile']
    spread,roll=cost_arrays(t,json.loads((a.prefix/'CLOSURES.json').read_text()),p)
    fees=np.array([p[k] for k in ('slippage_market','slippage_stop','limit_through')]+
                  [p[k]/p['lot_units'] for k in ('commission_per_lot_side','swap_long_per_lot','swap_short_per_lot')])
    d=np.load(a.signals);ts=d['ts'];assert ts.max()<pd.Timestamp('2024',tz='UTC').value
    ii=np.searchsorted(t,ts);np.testing.assert_array_equal(t[ii],ts)
    pred=np.full((len(t),8),np.nan);pred[ii-1]=d['x']
    np.testing.assert_array_equal(t[ii-1]+NS,ts)
    n=len(NAMES)+1;gram=np.zeros((n,n));cross=np.zeros((2,n))
    arms=[('state',len(STATE)),('seasonal',len(NAMES))]
    sequential=a.target=='continuation'
    blocks=[];latest=-1
    models={};manifest={}
    for yr in range(2019,2023):
        left,right=(pd.Timestamp(str(v),tz='UTC').value for v in (yr,yr+1))
        ix=ii[(ts>=left)&(ts<right)&gate[ii]&(flat[ii]<right)&(atr[ii-1]>0)]
        total=active=0;unique=set()
        week=(t[ix]-pd.Timestamp('1970-01-05',tz='UTC').value)//(168*NS) if sequential else np.zeros(len(ix),int)
        for wk in np.unique(week):
            batch=ix[week==wk]
            assert latest<t[batch[0]]
            prior=[fit((gram,cross[b]),w) if sequential and gram[0,0]>0 else None for b,(_,w) in enumerate(arms)]
            log=dict(year=yr,block=int(wk),prior_latest_exit_ns=int(latest),start_ns=int(t[batch[0]]),
                     ticks=0,state_deferred_exit=0,seasonal_deferred_exit=0)
            for ids in np.array_split(batch,(len(batch)+95)//96):
                net,end=outcomes(ids,t,o,hi,lo,atr,flat,spread,roll,fees,tm,sm)
                assert t[end.max()]<right
                latest=max(latest,t[end.max()])
                ix6=np.repeat(ids,6);ks=np.tile(np.arange(6),len(ids))
                owner,j,x,y,cnt=candidates(ix6,ks,data,spread,roll,fees,pred,tm,sm,
                                            np.repeat(net,6,axis=0),np.repeat(end,6,axis=0))
                unique.update(j.tolist());total+=len(j);active+=int((cnt>0).sum())
                z=np.column_stack([np.ones(len(x)),x]);weight=1/cnt[owner]
                gram+=z.T@(weight[:,None]*z)
                log['ticks']+=len(j)
                for b,(arm,_) in enumerate(arms):
                    target=y
                    if prior[b] is not None:
                        target,next_exit=versus_wait(owner,y,predict(x,prior[b])>0)
                        log[arm+'_deferred_exit']+=int((next_exit>=0).sum())
                    cross[b]+=z.T@(weight*target)
            blocks.append(dict(log,last_exit_ns=int(latest)))
        manifest[yr]=dict(origins=len(ix),positions=len(ix)*6,with_decisions=active,position_ticks=total,
                          distinct_decision_bars=len(unique),label_cutoff_exclusive_ns=int(right))
        models[yr+1]={arm:fit((gram,cross[b]),w) for b,(arm,w) in enumerate(arms)}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'weights.json').write_text(json.dumps(models,indent=2)+'\n')
    (a.out/'train_counts.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if sequential:
        pd.DataFrame(blocks).to_csv(a.out/'weekly_provenance.csv',index=False)
    archived=pd.read_csv(a.exit_dir/'alternatives.csv.gz').groupby('entry_ns')
    baseline=pd.read_csv(a.exit_dir/'positions.csv.gz')
    holds=baseline.query('arm=="hold"')
    records=[];signed=[]
    for r in holds.itertuples():
        i=int(np.searchsorted(t,r.entry_ns));s=1 if r.side=='buy' else -1;k=(0 if s==1 else 3)+(6,24,48).index(r.h)
        ix=np.array([i]);net,end=outcomes(ix,t,o,hi,lo,atr,flat,spread,roll,fees,tm,sm)
        np.testing.assert_allclose(net[0,k]*r.original_capital*r.q/10000,r.usd,atol=1e-8)
        _,j,x,y,_=candidates(ix,np.array([k]),data,spread,roll,fees,pred,tm,sm,net,end)
        assert len(j)==r.opportunities and t[end[0,k]]==r.exit_ns
        en=o[i]+(spread[i] if s>0 else 0)+s*fees[0]
        money_r=r.original_capital*5*r.q/en*sm*atr[i-1]*np.sqrt(r.h)
        if len(j):
            old=archived.get_group(r.entry_ns)
            np.testing.assert_array_equal(t[j],old.tick_ns)
            np.testing.assert_allclose(y*money_r,old.minus_hold,atol=1e-8)
        for arm,m in models[min(2023,r.fold)].items():
            scores=predict(x,m)
            at=first_close(scores);delta=0. if at is None else float(y[at]*money_r)
            records.append(dict(fold=r.fold,arm=arm,entry_ns=r.entry_ns,side=r.side,hold_reason=r.hold_reason,
                early=at is not None,exit_ns=r.exit_ns if at is None else t[j[at]],usd=r.usd+delta,
                delta_usd=delta,regret_usd=r.regret_usd-delta))
            target,_=versus_wait(np.zeros(len(j),np.int64),y,scores>0) if sequential else (y,None)
            signed.extend(dict(fold=r.fold,arm=arm,entry_ns=r.entry_ns,tick_ns=t[b],estimate_R=v,label_R=u)
                          for b,v,u in zip(j,scores,target))
    df=pd.DataFrame(records);df.to_csv(a.out/'positions.csv.gz',index=False)
    pd.DataFrame(signed).to_csv(a.out/'scores.csv.gz',index=False)
    allrows=pd.concat([df,baseline[baseline.arm.isin(['hold','clock_half','clock_first'])]])
    rows=[];pairs=[]
    for fold,g in allrows.groupby('fold'):
        panel=g.pivot(index='entry_ns',columns='arm',values='usd')
        for arm,v in g.groupby('arm'):
            rows.append(dict(fold=fold,arm=arm,n=len(v),ret_pct=v.usd.sum()/100,delta_usd=v.delta_usd.sum(),
                closes=int(v.early.sum()),helpful=int((v.delta_usd>1e-8).sum()),harmful=int((v.delta_usd < -1e-8).sum()),
                regret_usd=v.regret_usd.sum()))
        weeks=pd.to_datetime(panel.index,utc=True).tz_localize(None).to_period('W-SUN')
        for arm in ('state','seasonal'):
            for ctrl in ('hold','state','clock_half','clock_first'):
                if ctrl==arm:continue
                v=(panel[arm]-panel[ctrl]).groupby(weeks).agg(['sum','count'])
                z=np.random.default_rng(474).integers(len(v),size=(10000,len(v)))
                draws=v['sum'].to_numpy()[z].sum(1)/v['count'].to_numpy()[z].sum(1)
                pairs.append(dict(fold=fold,arm=arm,control=ctrl,mean_usd=v['sum'].sum()/len(panel),
                                  lo=np.quantile(draws,.025),hi=np.quantile(draws,.975),weeks=len(v)))
    pd.DataFrame(rows).to_csv(a.out/'summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(a.out/'paired.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('src','bars','prefix','signals','prev','exit_dir','out'):p.add_argument(key,type=Path)
    p.add_argument('--target',choices=['passive','continuation'],default='passive')
    main(p.parse_args())
