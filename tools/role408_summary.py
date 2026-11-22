"""Summarize completed pairs and check the frozen calibrator's attainable bids."""
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.role408_folds import OUT,SEEDS

read=lambda f:json.loads(f.read_text())


def summarize(root):
    if not (root/'DONE.json').is_file():raise ValueError('incomplete experiment')
    from tools.role404_data import assemble,environment,ROOT,PARENT
    joined,orig=assemble(PARENT,ROOT/'model/61/bundle',ROOT/'history397-61-r2')
    weeks={}
    for year,market in ((2023,joined),(2024,orig)):
        e=environment(market,read(PARENT/'RISK.json'),6,[f'{year}-01-01',f'{year+1}-01-01'],.99).engine
        weeks[year]=len(set(e.seg_of[e.first:e.last+1]))
    summary={}
    for seed in SEEDS:
        by_year={}
        for year in (2023,2024):
            d=root/f'score-{seed}-{year}'
            a=read(d/'ACCOUNT.json');m=read(d/'MAP.json')
            assert m['fit_years']==list(range(2022,year))
            for last,want in zip(m['fit_years'],m['fit_sha'],strict=True):
                old=root/f'score-{seed}-{last}'
                with (old/'labels.npz').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==want
                assert want==read(old/'RECEIPT.json')['sha']
            coef=np.array(m['coefficients']);inter,slope,lo,hi=coef.transpose(2,0,1)
            assert np.all(slope>=0) and np.all(lo<=hi)
            with np.load(d/'labels.npz') as z:
                p=z['p'];cap=inter+slope*hi;scores=inter+slope*np.clip(p,lo,hi)
            if scores.max()<=0:assert a['cal']['trades']==0
            for arm in ('raw','cal'):
                row=a[arm]
                row['civil_weeks']=row['weeks'];row['native_sessions']=weeks[year]
                row['native_orders_per_week']=row['trades']/weeks[year]
                row['native_activity_pass']=row['trades']>2*weeks[year]
            by_year[str(year)]=dict(raw=a['raw'],cal=a['cal'],opportunities=a['opportunities'],
                max_bid_on_domain_bp=cap.tolist(),positive_curves=int((cap>0).sum()),
                zero_slopes=int((slope==0).sum()),max_bid_observed_bp=float(scores.max()),
                ticks_positive_offer=int((scores.max((1,2))>0).sum()))
        # Descriptive TRAIN-only rank probe, not another opening threshold.
        folds={}
        for year in (2022,2023):
            with np.load(root/f'score-{seed}-{year}'/'labels.npz') as z:
                p,y=z['p'],z['y']
                top=np.empty((2,4));n=np.zeros((2,4),dtype=int)
                for r,j in np.ndindex(2,4):
                    ok=np.isfinite(y[:,r,j]);x=p[ok,r,j];v=y[ok,r,j]
                    sel=x>=np.quantile(x,.9);top[r,j]=v[sel].mean();n[r,j]=sel.sum()
                folds[year]=dict(top_decile_mean_bp=top.tolist(),count=n.tolist())
        a,b=[np.array(folds[y]['top_decile_mean_bp']) for y in (2022,2023)]
        by_year['train_rank_probe']=dict(folds=folds,positive_in_both=int(((a>0)&(b>0)).sum()))
        summary[seed]=by_year
    return summary


if __name__=='__main__':
    from tools.role404_data import save
    save(OUT/'INTERPRETATION.json',summarize(OUT))
