"""Bridge 404 training payoff to 408 managed payoff using the native 411 replays."""
import numpy as np
import pandas as pd
from role419_cuts_and_sell import OUT, base, labels, atomic, SEEDS
from role412_campaign import LEDGER
from role413_sequence import panel


def main():
    rows=[]
    base.torch.set_num_threads(1)
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as cache:bank=cache['x'],cache['t'],cache['valid']
    for seed in SEEDS:
        z,_=labels(2023,seed)
        pn=panel(2023,seed,*bank)
        assert np.array_equal(z['p'],pn.p) and np.array_equal(z['bars'],pn.bars)
        pn.close=lambda e,b,j:0
        pn.e.reset()
        df=pd.read_csv(LEDGER/f'offers-{seed}.csv.gz')
        ts=z['origin_ns'];h2=ts>=base.ns('2023-07-01')
        for j,h in enumerate(base.HS):
            d=base.DEST/f'labels-2023-{h}';path=d/'rows.npz'
            assert base.file_sha(path)==base.read(d/'DONE.json')['sha']
            with np.load(path) as a:
                src=a['OPEN_SELL_id'][:,1];train=a['OPEN_SELL'][:,3];standalone=a['OPEN_SELL'][:,4]
            p=z['p'][:,1,j]
            _,ix,k=np.intersect1d(ts,src,return_indices=True)
            ok=h2[ix]&(p[ix]>np.quantile(p[~h2],.90))
            ix,k=ix[ok],k[ok]
            q=df[(df.side=='OPEN_SELL')&(df.h==h)].set_index('origin_ns').loc[ts[ix]]
            np.testing.assert_allclose(q.target,z['y'][ix,1,j],atol=1e-9,rtol=0)
            a=train[k].astype(float);b=q.barrier_target.to_numpy();c=q.target.to_numpy()
            assert np.isfinite([a,b,c]).all()
            cap=pn.e.cfg.capital
            fills=[pn.isolated(pn.bars[ii],2,j,cap)[:2] for ii in ix]
            assert all(fill for fill,cash in fills)
            err=float(np.max(np.abs([(cash-cap)*1e4/cap-past for (_,cash),past in zip(fills,standalone[k].astype(float))])))
            assert err<1e-4, (seed,h,err)
            rows.append(dict(seed=seed,h=h,n=len(a),trained=float(a.mean()),passive_common_ref=float(b.mean()),
                             managed=float(c.mean()),baseline_change=float((b-a).mean()),closer_change=float((c-b).mean()),
                             standalone_max_error_bp=err))
    atomic(OUT/'SELL_BRIDGE_2023.json',rows)


if __name__=='__main__':main()
