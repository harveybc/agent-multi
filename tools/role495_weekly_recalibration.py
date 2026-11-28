"""Settled-week intercept-only correction for frozen 494 short-close scores."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from role492_fixed_vintage import HASH, SEEDS, sha


def offsets(starts, eligible, labels, scores, now):
    # Labels settle at the end of their Sunday-to-Sunday simulated account.
    use=eligible & (starts>=now-np.timedelta64(28,'D')) & (starts+np.timedelta64(7,'D')<=now)
    weeks=np.unique(starts[use])
    if not len(weeks):
        return 0.,0.,0,0
    means=[(labels[use & (starts==w)].mean(),
            (labels[use & (starts==w)]-scores[use & (starts==w)]).mean()) for w in weeks]
    avg,shift=np.mean(means,axis=0)
    return float(avg),float(shift),len(weeks),int(use.sum())


def run(panel,models,refs,prior,out):
    assert sha(panel)==HASH
    d=np.load(panel); mask=(d['side']==-1)&(d['w']>='2023')&(d['w']<'2024')
    w=d['w'][mask].astype('datetime64[D]');ctx=d['ctx'][mask]
    x=np.c_[d['x'][mask],ctx];c=d['cash'][mask]
    np.testing.assert_allclose(c[:,0],c[:,2],atol=1e-6)
    actor=(d['actor'][mask]==1).astype(int);c=c[:,:2];ix=np.arange(len(c))
    manifest=json.loads((refs/'manifest_-1.json').read_text())
    fits=json.loads((prior/'fits.json').read_text());events=[];log=[]
    for seed in SEEDS:
        preds=[]
        for year in (2021,2022):
            m,=[m for m in manifest if m['year']==year and m['seed']==seed]
            file=models/m['checkpoint'];assert sha(file)==m['sha256']
            preds.append(joblib.load(file).predict(x))
        old,new=[v.argmax(1) for v in preds]
        elig=(old==0)&(new==1)
        labels=c[:,1]-c[:,0];base=c[ix,new]-c[ix,actor]
        margin=preds[1][:,1]-preds[1][:,0]
        for variant,feat in [('margin',margin[:,None]),('position_margin',np.c_[margin,ctx[:,3:6]])]:
            cfg,=[i for i in fits if i['role']==-1 and i['seed']==seed and i['variant']==variant]
            score=(feat-cfg['mean'])/cfg['scale']@cfg['coef']+cfg['intercept']
            for now in np.unique(w[w>=np.datetime64('2023-07-01')]):
                avg,shift,nw,n=offsets(w,elig,labels,score,now)
                corrupt=np.where(w>=now,np.nan,labels)
                assert offsets(w,elig,corrupt,score,now)==(avg,shift,nw,n)
                ids=ix[w==now]
                adj=score[ids]+shift
                if elig[ids].sum()>1:
                    assert np.array_equal(np.argsort(adj[elig[ids]]),np.argsort(score[ids][elig[ids]]))
                log.append(dict(week=str(now),variant=variant,seed=seed,n_prior=n,
                                settled_weeks=nw,offset=shift,recent_close_value=avg))
                for policy,sc in [('frozen',score[ids]),('corrected',adj),('recent_only',np.full(len(ids),avg))]:
                    for i,s in zip(ids,sc):
                        veto=bool(elig[i] and s<0)
                        events.append(dict(week=str(now),variant=variant,policy=policy,seed=seed,
                            eligible=bool(elig[i]),score=s,adv=labels[i],veto=veto,
                            vs_close=-labels[i] if veto else 0.,
                            vs_hold=labels[i] if elig[i] and not veto else 0.,
                            vs_actor=base[i]-(labels[i] if veto else 0.)))
    df=pd.DataFrame(events);summary=[];weekstats=[]
    reference=pd.read_csv(prior/'summary.csv').query('role==-1').set_index('variant')
    for (var,pol),g in df.groupby(['variant','policy']):
        e=g[g.eligible];v=e[e.veto]; weights=1/e.week.map(g.week.value_counts()).to_numpy()
        validate=e[e.adv.abs()>1e-8]
        auc=np.mean([roc_auc_score(h.adv>0,h.score) for _,h in validate.groupby('seed')])
        row=dict(variant=var,policy=pol,eligible=len(e),veto=len(v),
                 bad_avoided=int((v.adv<0).sum()),good_blocked=int((v.adv>0).sum()),auc=auc,
                 bias_usd=np.average(e.score-e.adv,weights=weights),
                 mae_usd=np.average(abs(e.score-e.adv),weights=weights))
        wk=g.groupby('week')[['vs_close','vs_hold','vs_actor']].mean()
        ctrl=df[(df.variant==var)&(df.policy=='recent_only')].groupby('week').vs_close.mean()
        wk['vs_recent_only']=wk.vs_close-ctrl
        # Exploratory paired two-week blocks of observed role weeks, not independent seeds.
        k=len(wk);starts=np.random.default_rng(495).integers(0,k,size=(10000,(k+1)//2))
        draw=np.stack([starts,(starts+1)%k],axis=-1).reshape(len(starts),-1)[:,:k]
        for key,s in wk.items():
            row[key]=s.mean()
            row[key+'_lo'],row[key+'_hi']=np.quantile(s.to_numpy()[draw].mean(1),[.025,.975])
        if pol=='frozen':
            r=reference.loc[var]
            np.testing.assert_allclose([len(e),len(v),auc,row['vs_actor'],row['bias_usd']],
                [r.eligible_seed_states,r.veto,r.auc,r.gain_actor,r.expected_conditional_usd-r.realized_conditional_usd],atol=1e-7)
        weekstats.extend(wk.assign(variant=var,policy=pol).reset_index().to_dict('records'))
        summary.append(row)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('summary',summary),('offsets',log),('weekly',weekstats)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False,float_format='%.8f')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('panel','models','refs','prior','out'):p.add_argument(k,type=Path)
    run(**vars(p.parse_args()))
