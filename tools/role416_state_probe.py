"""TRAIN only. Independent open offers and same-tick managed state probe."""
from pathlib import Path
import numpy as np
from role413_sequence import base, labels, atomic, ridge

OUT=Path('/var/tmp/role416-state-probe')


def openings():
    rows=[]
    for seed in (7,19,43):
        for year in (2022,2023):
            z,_=labels(year,seed)
            dt=z['origin_ns'];jul=base.ns(f'{year}-07-01')
            week=(dt//86400000000000+3)//7
            for r,j in np.ndindex(2,4):
                p=z['p'][:,r,j];y=z['y'][:,r,j]
                # Fix a probability threshold using H1 predictions, no outcomes.
                cut=np.quantile(p[dt<jul],.90)
                valid=(dt>=jul)&np.isfinite(y)
                sel=valid&(p>cut)
                n=int(sel.sum())
                if not n: continue
                # Exact expectation from drawing the same number of origins per week.
                ctrl=sum(y[valid&(week==w)].mean()*(sel&(week==w)).sum() for w in np.unique(week[sel]))/n
                rows.append(dict(seed=seed,year=year,role=base.ROLES[r],h=base.HS[j],offers=n,
                                 bp=float(y[sel].mean()),matched=float(ctrl),lift=float(y[sel].mean()-ctrl)))
    return rows


def probe(year,h,role,x,t,valid):
    root=base.DEST/f'labels-{year}-{h}'
    assert base.file_sha(root/'rows.npz')==base.read(root/'DONE.json')['sha']
    with np.load(root/'rows.npz') as z: ids=z[role+'_id']; v=z[role]
    i=ids[:,0]
    mask=valid[i]&(ids[:,2]<base.ns(f'{year+1}-01-01'))&(t[i]>=base.ns(f'{year}-01-01' if year==2022 else '2023-07-01'))
    assert np.all(ids[mask,1]<=t[i[mask]])
    i=i[mask];v=v[mask].astype(float);g=v[:,:3]
    return i,x[i].astype(float),g/(g+[1,1,24]),v[:,3]


def main():
    OUT.mkdir(exist_ok=True)
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:x,t,valid=z['x'],z['t'],z['valid']
    op=openings()
    records=[]
    for role in base.ROLES[2:]:
        for h in base.HS:
            _,a,g,b=probe(2022,h,role,x,t,valid)
            i,u,v,y=probe(2023,h,role,x,t,valid)
            # Add a signed payoff regressor after the frozen market encoder.
            inp=np.column_stack((a,g))
            state=np.column_stack((u,v))
            uniq,inv=np.unique(i,return_inverse=True)
            lo=np.full(len(uniq),np.inf);hi=-lo.copy()
            np.minimum.at(lo,inv,y);np.maximum.at(hi,inv,y)
            conflict=(lo<0)&(hi>0)
            # Roll the position state within each market tick; market features
            # and all outcomes stay unchanged. No outcome used for permutation.
            order=np.argsort(i,kind='stable')
            rotated=v.copy()
            for ix in np.split(order,np.flatnonzero(np.diff(i[order]))+1):
                rotated[ix]=v[np.roll(ix,1)]
            s=dict(market=ridge(a,b,u),direct=ridge(inp,b,state),
                   swapped=ridge(inp,b,np.column_stack((u,rotated))))
            scores={}
            for name,score in s.items():
                payoff=np.where(score>0,y,0)
                regret=np.maximum(y,0)-payoff
                bytick=np.bincount(inv,weights=regret)/np.bincount(inv)
                scores[name]=dict(regret=float(bytick.mean()),mixed_regret=float(bytick[conflict].mean()),
                                  close_fraction=float((score>0).mean()))
            records.append(dict(role=role,h=h,origins=len(i),ticks=len(uniq),conflicts=int(conflict.sum()),scores=scores))
    atomic(OUT/'RESULT.json',dict(openings=op,close=records))


if __name__=='__main__':main()
