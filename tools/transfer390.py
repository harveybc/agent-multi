"""Cheap frozen-protocol probe of whether synthetic features retain exit information."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from tools.audit_g3_producers import write


def examples(p, z, h, side, sl, tp):
    # Position opened on previous open, still alive. Decide at current open.
    # Feature row immediately before decision is available; no future receiver values.
    ts=p.index.as_unit('ns').asi8
    op,hi,lo,cl=p.to_numpy().T
    w=lambda x:np.lib.stride_tricks.sliding_window_view(x,h)
    start=w(op)[:,0,None]
    fav,adv=(w(hi)-start,start-w(lo)) if side==1 else (start-w(lo),w(hi)-start)
    hits,stops=fav>=tp*1e-4,adv>=sl*1e-4
    j=np.where(hits.any(1),hits.argmax(1),h)
    k=np.where(stops.any(1),stops.argmax(1),h)
    ok=(ts[h-1:]-ts[:-h+1]==(h-1)*3600*10**9)&(j>0)&(k>0)&~((j==k)&(k<h))
    hold=np.where(j<k,tp,np.where(k<j,-sl,side*(w(cl)[:,-1]-start[:,0])*1e4))
    gain=side*(w(op)[:,1]-start[:,0])*1e4-hold
    valid=np.isfinite(z.to_numpy()[:len(ok)]).all(1)&ok&(abs(gain)>1e-9)
    return z.to_numpy()[:len(ok)][valid],(gain[valid]>0).astype(int),p.index[:len(ok)][valid]


def run(a):
    spec=json.loads(a.spec.read_text())['producer_settings']
    with np.load(a.business/'19.npz') as d:
        ts=pd.to_datetime(d['ts'],utc=True); risk=d['risk']
        real=pd.DataFrame(d['real'],index=ts,columns=['OPEN','HIGH','LOW','CLOSE'])
    rows=[]
    for name,n in [('price',61),('context',71)]:
        syn=[]
        for seed in (7,19):
            path=a.root/(name+str(seed))
            syn.append((pd.read_parquet(path/'OHLC.parquet'),pd.read_parquet(path/f'FEATURES{n}.parquet')))
        cols=syn[0][1].columns.tolist()
        z=pd.read_parquet(spec['store'],columns=['timestamp',*cols],filters=[('timestamp','>=',ts[0]),
                          ('timestamp','<=',ts[-1])]).set_index('timestamp').loc[ts]
        for h,side,sl,tp in risk:
            if h!=6:continue
            args=(int(h),int(side),sl,tp)
            xr,yr,_=examples(real[ts.year<2023],z[ts.year<2023],*args)
            xv,yv,t=examples(real[ts.year==2023],z[ts.year==2023],*args)
            pieces=[examples(p,f,*args) for p,f in syn]
            xs,ys=[np.concatenate([piece[i] for piece in pieces]) for i in (0,1)]
            for fitseed in (7,19):
                rng=np.random.default_rng(fitseed)
                picks=rng.choice(len(xr),len(xr)//10,replace=False)
                ix=rng.integers(len(xs),size=len(xr))
                for variant in ('real','mix10','synthetic_only','permuted'):
                    x,y=xr.copy(),yr.copy()
                    if variant=='mix10':x[picks],y[picks]=xs[ix[picks]],ys[ix[picks]]
                    if variant=='synthetic_only':x,y=xs[ix],ys[ix]
                    if variant=='permuted':y=rng.permutation(y)
                    m=HistGradientBoostingClassifier(max_iter=40,max_leaf_nodes=7,learning_rate=.05,
                            early_stopping=False,random_state=fitseed).fit(x,y)
                    prob=m.predict_proba(xv)[:,1]
                    for q in (0,1,2,3,4):
                        take=t.quarter==q if q else np.ones(len(t),bool)
                        rows.append(dict(features=n,side=int(side),seed=fitseed,fit=variant,quarter=q,
                                n_train=len(x),n_eval=int(take.sum()),auc=roc_auc_score(yv[take],prob[take]),
                                brier=brier_score_loss(yv[take],prob[take])))
    df=pd.DataFrame(rows)
    df.to_csv(a.root/'TRANSFER.csv',index=False)
    summary={}
    for (n,side),g in df[df.quarter==0].groupby(['features','side']):
        avg=g.groupby('fit')[['auc','brier']].mean()
        summary[f'{n}_{side}']=avg.to_dict(orient='index')
    write(a.root/'TRANSFER.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','business','spec'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
