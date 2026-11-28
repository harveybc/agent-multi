"""Freeze the matched 2021/2022 action-value probes across both halves of 2023."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from role491_recency import HASH, SEEDS, paired


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(panel, out, side):
    assert sha(panel)==HASH
    with np.load(panel) as d:
        mask=d['side']==side
        w=d['w'][mask]; x=d['x'][mask]; c=d['cash'][mask]; a=d['actor'][mask]
        if side:
            close=2 if side==1 else 1
            np.testing.assert_allclose(c[:,0],c[:,3-close],atol=1e-6)
            c=c[:,[0,close]]; a=(a==close).astype(int); x=np.c_[x,d['ctx'][mask]]
    q=pd.PeriodIndex(w,freq='Q').asi8; t=pd.Period('2023Q1','Q').ordinal
    y=c-c.mean(1,keepdims=True); informative=np.ptp(c,axis=1)>1e-8
    out.mkdir(exist_ok=True,parents=True)
    manifest=[]; rows=[]
    for seed in SEEDS:
        ids,_=paired(q,w,informative,t,seed)
        assert len(ids[0])==len(ids[1])
        fitted=[]
        for year,ii in zip((2021,2022),ids):
            assert all(v.startswith(str(year)) for v in w[ii])
            entry=dict(year=year,seed=seed,role=side,n=len(ii),weeks=len(set(w[ii])),
                       train_index_sha256=hashlib.sha256(ii.astype('<i8').tobytes()).hexdigest())
            if len(ii)>=16:
                model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=8,max_features=1.,
                                          random_state=seed,n_jobs=8).fit(x[ii],y[ii])
                path=out/f'r{side}_{year}_{seed}.joblib'
                joblib.dump(model,path,compress=3)
                fitted.append((path,sha(path)))
                entry.update(checkpoint=path.name,sha256=fitted[-1][1])
            manifest.append(entry)
        if not fitted:continue
        assert len(fitted)==2
        for half in (0,1):
            test=np.flatnonzero((q>=t+2*half)&(q<t+2*half+2))
            net=[]
            for path,h in fitted:
                assert sha(path)==h
                action=joblib.load(path).predict(x[test]).argmax(1)
                net.append(c[test,action])
            actor=c[test,a[test]]
            frame=pd.DataFrame(dict(recent_old=net[1]-net[0],recent_actor=net[1]-actor,
                                    old_actor=net[0]-actor),index=w[test])
            for week,g in frame.groupby(level=0):
                rows.append(dict(role=side,seed=seed,half=half+1,week=week,n=len(g),**g.mean().to_dict()))
    (out/f'manifest_{side}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if rows:pd.DataFrame(rows).to_csv(out/f'weekly_{side}.csv',index=False)


def report(out,previous):
    dfs=[]
    for side in (0,-1,1):
        manifest=json.loads((out/f'manifest_{side}.json').read_text())
        if not any('checkpoint' in r for r in manifest):
            assert all(r['n']<16 for r in manifest)
            continue
        f=pd.read_csv(out/f'weekly_{side}.csv')
        old=pd.read_csv(previous/f'weeks_{side}.csv')
        old=old[(old.start=='2023Q1')&(old['size']=='full')].set_index(['seed','week']).sort_index()
        first=f[f.half==1].set_index(['seed','week']).sort_index()
        assert first.index.equals(old.index)
        cols=['recent_old','recent_actor','old_actor']
        np.testing.assert_allclose(first[cols],old[cols],rtol=0,atol=1e-8)
        dfs.append(f)
    score=pd.concat(dfs)
    wk=score.groupby(['role','half','week'])[['n',*cols]].mean().reset_index()
    rows=[]
    for side,g in wk.groupby('role'):
        boot=[];means=[]
        for h,s in g.groupby('half'):
            v=s.recent_old.to_numpy(); m=v.mean(); means.append(m)
            b=np.random.default_rng(492+h).choice(v,(10000,len(v))).mean(1);boot.append(b)
            ci=np.quantile(b,[.025,.975])
            rows.append(dict(role=side,period=str(h),decisions=s.n.sum(),weeks=len(v),
                             recent_old=m,lo=ci[0],hi=ci[1],
                             recent_actor=s.recent_actor.mean(),old_actor=s.old_actor.mean()))
        ci=np.quantile(boot[1]-boot[0],[.025,.975])
        rows.append(dict(role=side,period='H2-H1',recent_old=means[1]-means[0],lo=ci[0],hi=ci[1]))
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False,float_format='%.6f')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--panel',type=Path)
    p.add_argument('--previous',type=Path);p.add_argument('--role',type=int,choices=(-1,0,1))
    args=p.parse_args()
    if args.role is None:report(args.out,args.previous)
    else:run(args.panel,args.out,args.role)
