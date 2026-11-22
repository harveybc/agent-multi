"""Paired descriptive audit; no thresholds or production changes."""
import numpy as np
from role413_sequence import OUT, base, labels, ranks, atomic
from role412_campaign import LEDGER, SEEDS
import pandas as pd


def main():
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:
        x,t,valid=z['x'],z['t'],z['valid']
    result={}
    for seed in SEEDS:
        xs=[];ys=[]
        for year in range(2020,2024):
            data,_=labels(year,seed)
            dates=data['origin_ns']
            idx=np.searchsorted(t,dates)
            assert np.array_equal(t[idx],dates) and valid[idx].all()
            target=data['y'].reshape(-1,8)
            if year<2023:
                ok=np.isfinite(target).all(1)
                xs.append(x[idx[ok]].astype(float));ys.append(target[ok])
        ledger=pd.read_csv(LEDGER/f'offers-{seed}.csv.gz',usecols=['origin_ns','chosen'])
        picks=ledger.groupby('origin_ns').chosen.sum()
        assert np.array_equal(picks.index,dates) and picks.max()==1
        ev=(dates>=base.ns('2023-07-01')) & (picks.to_numpy()==1)
        u=x[idx[ev]].astype(float); y=target[ev]
        actions=[];pay=[]
        for arm,start in [('short',2),('extended',0)]:
            v=ranks(np.concatenate(xs[start:]),np.concatenate(ys[start:]),u)
            assert np.all((v==v.max(1)[:,None]).sum(1)==1)
            a=v.argmax(1);p=y[np.arange(len(y)),a]
            ref=base.read(OUT/f'{arm}-{seed}.json')['arms']['paired_timing']
            assert len(p)==ref['offers'] and abs(p.mean()-ref['offer_bp'])<1e-8
            actions.append(a);pay.append(p)
        a,b=actions; ch=a!=b; diff=pay[1]-pay[0]
        week=(dates[ev]//86_400_000_000_000+3)//7
        _,i=np.unique(week,return_inverse=True)
        n=np.bincount(i);s=np.bincount(i,weights=diff)
        counts=np.random.default_rng(414).multinomial(len(n),np.ones(len(n))/len(n),size=5000)
        result[seed]=dict(offers=len(a),switched=int(ch.sum()),direction_flips=int(((a//4)!=(b//4)).sum()),
                          changed_only_bp=float(diff[ch].mean()),all_delta_bp=float(diff.mean()),
                          weekly_interval=np.quantile((counts@s)/(counts@n),[.025,.975]).tolist(),
                          helped=int((diff>1e-8).sum()),hurt=int((diff< -1e-8).sum()))
    atomic(OUT/'SAME_TICK_414.json',result)
    print(result)


if __name__=='__main__':
    main()
