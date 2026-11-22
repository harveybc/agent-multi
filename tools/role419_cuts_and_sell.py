"""Two independent TRAIN diagnostics: geometry thresholds and opener target parity."""
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from role417_paired_close import getrows
from role418_state_encoding import run, side, encode, tests, SEEDS, base, atomic
from role413_sequence import labels

OUT=Path('/var/tmp/role419-cuts-sell')
ARMS=('curve','level','gray','onehot','thermometer')


def sell():
    rows=[]
    for seed in SEEDS:
        for year in (2022,2023):
            z,_=labels(year,seed)
            ts=z['origin_ns'];h2=ts>=base.ns(f'{year}-07-01')
            for j,h in enumerate(base.HS):
                d=base.DEST/f'labels-{year}-{h}';path=d/'rows.npz'
                assert base.file_sha(path)==base.read(d/'DONE.json')['sha']
                with np.load(path) as a:
                    src=a['OPEN_SELL_id'][:,1];passive=a['OPEN_SELL'][:,3]
                assert len(np.unique(src))==len(src) and len(np.unique(ts))==len(ts)
                p0=z['p'][:,1,j];y0=z['y'][:,1,j]
                cut=np.quantile(p0[~h2],.90)
                _,ix,k=np.intersect1d(ts,src,return_indices=True)
                ok=h2[ix]&np.isfinite(y0[ix])&np.isfinite(passive[k])
                ix,k=ix[ok],k[ok]
                p=p0[ix];managed=y0[ix];v=passive[k];selected=p>cut
                assert selected.any()
                weeks=(ts[ix]//86400000000000+3)//7
                _,inv=np.unique(weeks,return_inverse=True)
                w=(np.bincount(inv,weights=selected)/np.bincount(inv))[inv]
                assert np.isclose(w.sum(),selected.sum())
                record=dict(seed=seed,year=year,h=h,n=len(ix),excluded=int((h2&np.isfinite(y0)).sum()-len(ix)),
                            selected=int(selected.sum()),sign_changed=float(np.mean((v[selected]>0)!=(managed[selected]>0))))
                for name,y in [('passive',v),('managed',managed)]:
                    yy=y[selected];pos=y>0;npos=pos.sum();nneg=len(pos)-npos
                    assert npos and nneg
                    record[name]=dict(bp=float(yy.mean()),win=float((yy>0).mean()),
                        win_control=float(np.average(pos,weights=w)),bp_control=float(np.average(y,weights=w)),
                        gains_bp=float(np.maximum(yy,0).mean()),losses_bp=float(np.maximum(-yy,0).mean()),
                        auc=float((rankdata(p)[pos].sum()-npos*(npos+1)/2)/(npos*nneg)))
                rows.append(record)
    atomic(OUT/'SELL.json',rows)


def main():
    OUT.mkdir(exist_ok=True);tests();base.torch.set_num_threads(1)
    sell()
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    states=np.concatenate([getrows(y,h,r,bank[1])[1] for y in range(2019,2023) for h in base.HS for r in base.ROLES[2:]])
    edges=np.quantile(states,np.arange(1,16)/16,axis=0).T.copy()
    assert np.all(np.diff(edges,axis=1)>=0)
    def bins(g):
        assert np.isfinite(g).all() and np.all((g>=0)&(g<=1))
        return np.stack([np.searchsorted(e,v,side='right') for e,v in zip(edges,g.T)],axis=1)
    q=bins(states)
    atomic(OUT/'CUTS.json',dict(fit_years=[2019,2020,2021,2022],fit_rows=len(q),
        physical=(edges/(1-edges)*np.array([1,1,24])[:,None]).tolist(),bounded=edges.tolist(),
        occupancy=[np.bincount(v,minlength=16).tolist() for v in q.T]))
    def design(arm,l,g):return encode(arm,l,(bins(g)+.5)/16)
    assert np.all(bins(np.zeros((1,3)))==0) and np.all(bins(np.ones((1,3)))==15)
    results=[]
    for seed in SEEDS:
        row=run(seed,bank,arms=ARMS,encode=design,out=OUT,route=side)
        ref=base.read(Path('/var/tmp/role418-state-encoding')/f'seed-{seed}.json')
        assert ref['account']['curve']==row['account']['curve'] and np.isclose(ref['same_entry_bp']['curve'],row['same_entry_bp']['curve'])
        atomic(OUT/f'seed-{seed}.json',row);results.append(row)
    def good(row,arm):
        cells=list(row['cells'].values())
        def loss(k):return np.average([c['regret'][k] for c in cells],weights=[c['conflict_ticks'] for c in cells])
        return loss(arm)<min(loss('curve'),loss(arm+'_swap')) and row['same_entry_bp'][arm]>row['same_entry_bp']['curve'] and row['account'][arm]['ret']>row['account']['curve']['ret']
    atomic(OUT/'DONE.json',dict(candidates=[arm for arm in ARMS[1:] if all(good(row,arm) for row in results)],promoted=False))


if __name__=='__main__':main()
