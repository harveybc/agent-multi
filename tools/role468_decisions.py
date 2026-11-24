"""Compare weekly and fixed predictors on identical origins; audit actual decisions ex post."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tools.role461_seasonal_account import H, NS, cost_arrays, entry, exit_signal, load, simulate
from tools.role463_noise import run


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    p=json.loads((a.gate/'policy.json').read_text())['policy']
    prof=json.loads((a.bars/'RECIPE.json').read_text())['business']['evaluation_and_policy']['profile']
    data=list(load(a.src,a.bars,p))
    t,o,_,_,c,atr,truth,price,_,deadline=data
    spr,roll=cost_arrays(t,json.loads((a.bars/'closures.json').read_text()),prof)
    real=np.load(a.fixed)
    ix=np.searchsorted(t+NS,real['ts'])
    np.testing.assert_array_equal(t[ix]+NS,real['ts'])
    weeks=[np.load(a.week/f'{k}.npz') for k in ('short','long')]
    for w in weeks:
        np.testing.assert_array_equal(w['ts'],real['ts'])
    cases={'fixed':real['tuned'],'weekly':np.column_stack([w['weekly'] for w in weeks])}
    reports={}
    entries=[]; exits=[]
    old=pd.read_csv(a.gate/'tuned-all_archived_origins.csv.gz')
    for mode,signals in cases.items():
        assert np.isfinite(signals).all()
        pred=np.full_like(truth,np.nan)
        pred[ix]=signals
        data[6]=pred
        report,log=simulate(data,(None,None),p,2024,'bridge','sl_only',(prof,spr,roll))
        reports[mode]=report
        if mode=='fixed':
            for k in ('entry_index','exit_index','side','pnl','q','sl','tp'):
                np.testing.assert_allclose([r[k] for r in log],old[k],rtol=1e-12,atol=1e-9)
        for r in log:
            i,j,s=r['entry_index'],r['exit_index'],r['side']
            prev=i-1
            assert t[i]-t[prev]==NS
            k=int(np.flatnonzero(H==r['h'])[0])
            def oracle_side(values):
                if not np.isfinite(values[5:]).all():
                    return np.nan
                action=entry(values,values,atr[prev],None,p)
                return action[0] if action else 0
            entries.append(dict(model=mode,entry_ns=int(t[i]),side=s,h=r['h'],q=r['q'],
                            pnl=r['pnl'],exit_reason=r['reason'],ideal_seasonal_side=oracle_side(truth[prev]),
                            ideal_price_side=oracle_side(price[prev]),seasonal=s*truth[prev,k],
                            price=s*price[prev,k],forecast=s*pred[prev,k]))
            # Audit at identical actual positions. Oracle decisions here never enter the replay.
            for b in range(i+1,j+1):
                ex=o[b]+(spr[b] if s<0 else 0.)
                if t[b]-t[b-1]!=NS or t[b]>=deadline[i] or s*(ex-r['sl'])<=0 or s*(ex-r['tp'])>=0:
                    continue
                def close(values):
                    cur=c[b-1]+(spr[b-1] if s<0 else 0.)
                    return exit_signal(s,cur,r['sl'],r['tp'],values,atr[b-1],p)=='early_sl'
                actual=close(pred[b-1])
                assert actual==(b==j and r['reason']=='early_sl')
                exits.append(dict(model=mode,origin_ns=int(t[b]),entry_ns=int(t[i]),side=s,
                              closed=actual,has_signal=np.isfinite(pred[b-1]).any(),
                              all_truth=np.isfinite(truth[b-1]).all(),
                              seasonal_close=close(truth[b-1]),price_close=close(price[b-1]),
                              early_minus_hold=r.get('early_minus_hold') if actual else None))
    for name,rows in [('entries',entries),('exit_ticks',exits)]:
        pd.DataFrame(rows).to_csv(a.out/f'{name}.csv.gz',index=False)
    (a.out/'accounts.json').write_text(json.dumps(reports,indent=2)+'\n')
    # Offline negative control: preserve exact marginal errors and missingness;
    # break the link to the particular market origin, with or without cross-horizon pairing.
    yr=(t>=pd.Timestamp('2024',tz='UTC').value)&(t<pd.Timestamp('2025',tz='UTC').value)
    oracle=truth[yr]
    sig=np.full_like(truth,np.nan)
    sig[ix]=cases['fixed']
    residual=np.where(np.isfinite(oracle),sig[yr]-oracle,np.nan)
    assert np.array_equal(np.isfinite(residual),np.isfinite(oracle))
    masks,inv=np.unique(np.isfinite(oracle),axis=0,return_inverse=True)
    groups=[(np.flatnonzero(inv==i),np.flatnonzero(m)) for i,m in enumerate(masks) if m.any()]
    fees=np.array([prof[k] for k in ('slippage_market','slippage_stop','limit_through')]+
                  [prof[k]/prof['lot_units'] for k in ('commission_per_lot_side','swap_long_per_lot','swap_short_per_lot')])
    cols=[np.ascontiguousarray(a[yr]) for a in (*data[:6],*data[8:],spr,roll)]
    args=[p[k] for k in ('tp_mult','sl_mult','dd_floor','min_rr','full_rr','size_power','early_margin_atr')]
    stats=[]
    for mode in ('unaltered','joint','independent'):
        for seed in range(1 if mode=='unaltered' else 64):
            rng=np.random.default_rng(seed)
            r=residual.copy()
            for rows,hs in groups:
                if mode=='joint':
                    r[rows]=residual[rng.permutation(rows)]
                elif mode=='independent':
                    for h in hs:
                        r[rows,h]=residual[rng.permutation(rows),h]
            np.testing.assert_allclose(np.nanmean(abs(r),axis=0),np.nanmean(abs(residual),axis=0),atol=1e-14)
            v=run(*cols,np.ascontiguousarray(oracle+r),fees,*args)
            stats.append(dict(mode=mode,seed=seed,return_pct=v[0]-100,dd_pct=v[1],trades=v[2],sharpe=v[-1]))
    control=next(x for x in json.loads((a.gate/'accounts.json').read_text()) if x['model']=='tuned' and x['support']=='label_support')
    np.testing.assert_allclose(stats[0]['return_pct'],control['return_pct'],atol=1e-9)
    pd.DataFrame(stats).to_csv(a.out/'residual_controls.csv',index=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('src','bars','gate','fixed','week','out'):
        ap.add_argument(arg,type=Path)
    main(ap.parse_args())
