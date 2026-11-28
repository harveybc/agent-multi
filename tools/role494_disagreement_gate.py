"""Read-only shadow veto on two 493 transitions. H1 calibration, H2 unchanged."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from role492_fixed_vintage import HASH, SEEDS, sha


def run(panel,models,refs,prior,out):
    assert sha(panel)==HASH
    data=np.load(panel)
    fits=[]; events=[];checks=pd.read_csv(prior/'transitions.csv')
    for role,transition in [(0,'NOP_SELL'),(-1,'HOLD_CLOSE')]:
        mask=(data['side']==role)&(data['w']>='2023-01')&(data['w']<'2024')
        w=data['w'][mask]; z=data['x'][mask];ctx=data['ctx'][mask];cash=data['cash'][mask]
        actor=data['actor'][mask]; idx=np.arange(len(w));recent=2 if role==0 else 1
        if role==-1:
            z=np.c_[z,ctx];cash=cash[:,:2];actor=(actor==1).astype(int)
        manifest=json.loads((refs/f'manifest_{role}.json').read_text())
        for seed in SEEDS:
            pred=[]
            for year in (2021,2022):
                m,=[v for v in manifest if v['year']==year and v['seed']==seed]
                path=models/m['checkpoint'];assert sha(path)==m['sha256']
                pred.append(joblib.load(path).predict(z))
            old,new=[v.argmax(1) for v in pred]
            candidate=(old==0)&(new==recent)
            margin=pred[1][:,recent]-pred[1][:,0]
            inputs={'margin':margin[:,None]}
            if role==-1:inputs['position_margin']=np.c_[margin,ctx[:,3:6]]
            advantage=cash[idx,new]-cash[idx,old]
            train=candidate&(w<'2023-07'); held=w>='2023-07'
            # Match 493's equal-week economic objective, conditional on eligibility.
            all_counts=pd.Series(w).value_counts()
            weight=np.array([1/all_counts[s] for s in w[train]])
            weight/=weight.sum()
            for name,x in inputs.items():
                scale=StandardScaler().fit(x[train],sample_weight=weight)
                reg=Ridge(alpha=.1).fit(scale.transform(x[train]),advantage[train],sample_weight=weight)
                score=reg.predict(scale.transform(x))
                assert np.isfinite(score).all() and max(w[train])<min(w[held])
                fits.append(dict(role=role,seed=seed,variant=name,n_fit=int(train.sum()),
                    train_weeks=len(set(w[train])),mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),
                    coef=reg.coef_.tolist(),intercept=float(reg.intercept_)))
                for i in idx[held]:
                    d=advantage[i] if candidate[i] else 0.
                    veto=bool(candidate[i] and score[i]<0)
                    events.append(dict(role=role,transition=transition,week=w[i],idx=i,seed=seed,
                        variant=name,eligible=bool(candidate[i]),score=score[i],veto=veto,
                        d=d,gain_never=-d if veto else 0.,gain_always=0. if veto else d,
                        gain_actor=cash[i,old[i] if veto else new[i]]-cash[i,actor[i]]))
    df=pd.DataFrame(events);stats=[];perseed=[]
    for (role,var),g in df.groupby(['role','variant']):
        e=g[g.eligible]; v=e[e.veto];real=e[e.d.abs()>1e-8]
        a=[]
        for seed,h in real.groupby('seed'):
            auc=roc_auc_score(h.d>0,h.score) if h.d.min()<0<h.d.max() else None
            a.append(auc);perseed.append(dict(role=role,variant=var,seed=seed,auc=auc,
                gain_vs_never=g[g.seed==seed].groupby('week').gain_never.mean().mean()))
        wk=g.groupby('week')[['gain_never','gain_always','gain_actor','d']].mean()
        old,new=g.transition.iloc[0].split('_')
        ref=checks[(checks.role==role)&(checks.half==2)&(checks.old==old)&(checks.new==new)].iloc[0]
        assert len(e)==ref.seed_states and abs(wk.d.mean()-ref.contribution_usd)<1e-7
        res=dict(role=role,variant=var,states=g.idx.nunique(),eligible_unique=e.idx.nunique(),
            eligible_seed_states=len(e),veto=len(v),harm_avoided=int((v.d < -1e-8).sum()),
            useful_vetoed=int((v.d>1e-8).sum()),tied_vetoed=int((v.d.abs()<=1e-8).sum()),
            harmful_total=int((e.d< -1e-8).sum()),useful_total=int((e.d>1e-8).sum()),
            weeks=len(wk),auc=float(np.mean(a)))
        counts=g.week.value_counts()
        weights=e.week.map(lambda s:1/counts[s]).to_numpy()
        res.update(expected_conditional_usd=np.average(e.score,weights=weights),
                   realized_conditional_usd=np.average(e.d,weights=weights))
        for col in ('gain_never','gain_always','gain_actor'):
            b=np.random.default_rng(494).choice(wk[col],(10000,len(wk))).mean(1)
            res[col]=wk[col].mean();res[col+'_lo'],res[col+'_hi']=np.quantile(b,[.025,.975])
        stats.append(res)
    out.mkdir(parents=True,exist_ok=True)
    (out/'fits.json').write_text(json.dumps(fits,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(stats).to_csv(out/'summary.csv',index=False,float_format='%.8f')
    pd.DataFrame(perseed).to_csv(out/'seed_check.csv',index=False,float_format='%.8f')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('panel','models','refs','prior','out'):p.add_argument(key,type=Path)
    run(**vars(p.parse_args()))
