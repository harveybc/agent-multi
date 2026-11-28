"""Paired role-specific vintage and sample-size contrasts on the sealed 490 panel."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

HASH = '15229cdb310c7c88eec15ee198ad1ae1e3396526c5422ba82ba2c3e2e4e3c866'
SEEDS = (7, 19, 41)


def paired(folds, w, informative, t, seed):
    rng = np.random.default_rng(seed)
    full, half = [[], []], [[], []]
    for quarter in range(4):
        groups = []
        for lag in (8, 4):
            ids = np.flatnonzero((folds == t-lag+quarter) & informative)
            weeks = rng.permutation(np.unique(w[ids]))
            groups.append(sorted([ids[w[ids]==wk] for wk in weeks], key=len, reverse=True))
        for a,b in zip(*groups):
            n = min(len(a),len(b))
            # Identical whole-week allocation in both arms; half is nested.
            for j,ids in enumerate((a,b)):
                take = rng.permutation(ids)[:n]
                full[j].extend(take)
                half[j].extend(take[:(n+1)//2])
    return [np.array(v,dtype=int) for v in full], [np.array(v,dtype=int) for v in half]


def run(root, panel, side):
    assert hashlib.sha256(panel.read_bytes()).hexdigest()==HASH
    with np.load(panel) as data:
        mask=data['side']==side
        w=data['w'][mask];x=data['x'][mask];c=data['cash'][mask];a=data['actor'][mask]
        if side:
            close=2 if side==1 else 1
            np.testing.assert_allclose(c[:,0],c[:,3-close],atol=1e-6)
            c=c[:,[0,close]];a=(a==close).astype(int)
            x=np.c_[x,data['ctx'][mask]]
    y=c-c.mean(1,keepdims=True)
    inf=np.ptp(c,axis=1)>1e-8
    quarter=pd.PeriodIndex(w,freq='Q').asi8
    scores=[]; coverage=[]
    for start in ('2022Q1','2022Q3','2023Q1','2023Q3'):
        t=pd.Period(start,freq='Q').ordinal
        test=np.flatnonzero((quarter>=t)&(quarter<t+2))
        for seed in SEEDS:
            full,half=paired(quarter,w,inf,t,seed)
            for size,tr in [('full',full),('half',half)]:
                n=len(tr[0]); nw=len(set(w[tr[0]]))
                assert n==len(tr[1]) and nw==len(set(w[tr[1]]))
                counts=[np.sort(np.unique(w[ids],return_counts=True)[1]) for ids in tr]
                np.testing.assert_array_equal(*counts)
                if n:assert max(quarter[tr[0]])<min(quarter[tr[1]])<=max(quarter[tr[1]])<t
                use=n>=16 and len(test)>0
                coverage.append(dict(start=start,role=side,size=size,seed=seed,n_per_arm=n,
                    weeks_per_arm=nw,test_n=len(test),test_informative=int(inf[test].sum()),
                    test_weeks=len(set(w[test])),ran=use))
                if not use:continue
                pp={}
                for vintage,ids in zip(('old','recent'),tr):
                    forest=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=8,max_features=1.,
                                               random_state=seed,n_jobs=8)
                    forest.fit(x[ids],y[ids]);pp[vintage]=forest.predict(x[test]).argmax(1)
                assert set(w[tr[0]]).isdisjoint(w[test]) and set(w[tr[1]]).isdisjoint(w[test])
                values={k:c[test,action] for k,action in dict(actor=a[test],**pp).items()}
                for wk,group in pd.DataFrame(values,index=w[test]).groupby(level=0):
                    vals=group.mean()
                    scores.append(dict(start=start,role=side,size=size,seed=seed,week=wk,
                        recent_old=vals.recent-vals.old,recent_actor=vals.recent-vals.actor,
                        old_actor=vals.old-vals.actor))
    root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(scores).to_csv(root/f'weeks_{side}.csv',index=False)
    pd.DataFrame(coverage).to_csv(root/f'coverage_{side}.csv',index=False)


def report(root):
    df=pd.concat([pd.read_csv(root/f'weeks_{s}.csv') for s in (-1,0,1)])
    cov=pd.concat([pd.read_csv(root/f'coverage_{s}.csv') for s in (-1,0,1)])
    groups=['role','start','size']
    out=[]
    for key,c in cov.groupby(groups):
        assert len(c)==3 and c.n_per_arm.nunique()==c.weeks_per_arm.nunique()==1
        d=c.iloc[0].drop(['seed']).to_dict()
        if c.ran.all():
            rows=df[(df.role==key[0])&(df.start==key[1])&(df['size']==key[2])]
            assert len(rows)==3*c.test_weeks.iloc[0] and not rows.duplicated(['week','seed']).any()
            wk=rows.groupby('week')[['recent_old','recent_actor','old_actor']].mean()
            v=wk.recent_old.to_numpy()
            ci=np.quantile(np.random.default_rng(491).choice(v,(10000,len(v))).mean(1),[.025,.975])
            d.update(wk.mean().to_dict(),lo=ci[0],hi=ci[1])
        out.append(d)
    pd.DataFrame(out).to_csv(root/'summary.csv',index=False,float_format='%.6f')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--role',type=int,choices=[-1,0,1]);p.add_argument('--panel',type=Path)
    args=p.parse_args()
    if args.role is None:report(args.root)
    else:run(args.root,args.panel,args.role)
