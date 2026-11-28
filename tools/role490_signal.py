"""Frozen role-specific action advantage probes; no actor updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def prepare(out):
    import role489_critic_factorial as f
    f.init_worker()
    m=f.market.inputs
    ix=np.flatnonzero(m.valid & (m.period=='TRAIN'))
    captured=[]
    hook=f.model.q_net.head.register_forward_pre_hook(
        lambda mod,args: captured.append(args[0].detach().cpu().numpy().copy()))
    try:
        _,q=f.emit(f.model,m,indices=ix,twins=True)
    finally:
        hook.remove()
    native=np.load(f.F/'signals.npz')['q']
    np.testing.assert_array_equal(native[ix],q[:,ix].min(0))
    z=np.column_stack([np.concatenate(captured[j::2]) for j in (0,1)])
    rows=[]; weekly=[]; margins=[]; example=None
    for w in f.cfg['weeks']:
        rs=f.cached(f.OUT/'weeks'/(w+'.json'),f.digest(f.cfg))['rows']
        rows.extend(dict(**r,w=w) for r in rs)
        if not w.startswith('2023'):continue
        e=f.environment(f.market,f.risk,f.spec['design']['horizon'],
                        [w,str((pd.Timestamp(w)+pd.Timedelta(days=7)).date())],f.cfg['gamma'])
        e.reset()
        while not e.finished:e.step(int(f.signal[e.rows[e.bar]]))
        eng=e.engine; capital=eng.cfg.capital; net=eng.balance-capital
        np.testing.assert_allclose([r['cash'][r['actor']] for r in rs],net,atol=1e-6)
        np.testing.assert_allclose(sum(t['net_usd'] for t in eng.trades),net,atol=1e-6)
        margins.extend(t['margin_usd'] for t in eng.trades)
        weekly.append(net)
        for r in rs:
            if example is None and r['side']==0 and max(r['cash'])-net>1:
                example=dict(time=str(pd.Timestamp(m.row_ts_ns[r['row']],tz='UTC')),
                             pnl=r['cash'],actor=r['actor'])
    i=np.array([r['row'] for r in rows])
    np.testing.assert_array_equal(ix[np.searchsorted(ix,i)],i)
    np.savez_compressed(out/'panel.npz',x=z[np.searchsorted(ix,i)],ctx=[r['ctx'] for r in rows],
        w=[r['w'] for r in rows],side=[r['side'] for r in rows],actor=[r['actor'] for r in rows],
        q=native[i],cash=[r['cash'] for r in rows])
    monitor=[r for r in rows if r['w'].startswith('2023')]
    def avg(kind):
        a=[(r['actor'] if kind=='actor' else 0 if kind=='nop' else native[r['row']].argmax()) for r in monitor]
        return float(pd.Series([max(r['cash'])-r['cash'][j] for r,j in zip(monitor,a)],
                    index=[r['w'] for r in monitor]).groupby(level=0).mean().mean())
    f.write(out/'account.json',dict(capital=capital,leverage=eng.cfg.sizing.leverage,
        rel_volume=f.risk['rel_volume'],trades=len(margins),margin_min_median_max=np.quantile(margins,[0,.5,1]).tolist(),
        weeks=len(weekly),mean_week_net=float(np.mean(weekly)),positive_weeks=int((np.array(weekly)>0).sum()),
        regret={k:avg(k) for k in ('actor','nop','critic')},example=example,panel_sha256=f.file_sha(out/'panel.npz')))


def run(out,role):
    from sklearn.ensemble import ExtraTreesRegressor
    v=np.load(out/'panel.npz'); sel=v['side']==role
    x=np.column_stack([v['x'][sel],v['ctx'][sel]]) if role else v['x'][sel]
    w=v['w'][sel]; years=np.array([int(t[:4]) for t in w]); a=v['actor'][sel]
    cash=v['cash'][sel]
    if role:
        close=2 if role==1 else 1
        np.testing.assert_allclose(cash[:,0],cash[:,3-close],atol=1e-6)
        cash=cash[:,[0,close]];a=(a==close).astype(int)
    y=cash-cash.mean(1,keepdims=True)
    results=[]
    for begin,end,held in [(2020,2021,2022),(2022,2022,2023),(2021,2022,2023),(2020,2022,2023)]:
        fit=(years>=begin)&(years<=end);test=years==held
        # Signal is a fixed function of one observed coordinate. Thresholds use fit only.
        col=np.argmax(x[fit].var(0)); k=y.shape[1]
        edges=np.quantile(x[fit,col],np.arange(1,k)/k)
        assert len(set(edges))==len(edges)
        planted=np.eye(k)[np.searchsorted(edges,x[:,col])]
        for seed in (7,19,41):
            rng=np.random.default_rng(seed); null=y.copy()
            for year in sorted(set(years[fit])):
                ids=np.flatnonzero(fit & (years==year))
                null[ids]=y[rng.permutation(ids)]
            for mode,label in [('true',y),('null',null),('planted',planted)]:
                forest=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=8,
                                           max_features=1.,random_state=seed,n_jobs=8)
                forest.fit(x[fit],label[fit])
                action=forest.predict(x).argmax(1)
                for part,mask in [('fit',fit),('later',test)]:
                    truth=planted[mask] if mode=='planted' else y[mask]
                    delta=truth.max(1)-truth[np.arange(len(truth)),action[mask]]
                    base=truth.max(1)-truth[np.arange(len(truth)),a[mask]]
                    weekly=pd.Series(delta-base,index=w[mask]).groupby(level=0).mean()
                    informative=np.ptp(truth,axis=1)>1e-8
                    results.append(dict(role=role,first=begin,last=end,later=held,seed=seed,
                        control=mode,part=part,n=int(mask.sum()),n_informative=int(informative.sum()),
                        best_rate=float((delta[informative]<1e-8).mean()),delta=float(weekly.mean()),
                        by_week=weekly.to_dict()))
    (out/f'role{role}.json').write_text(json.dumps(results,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('out',type=Path);parser.add_argument('stage',choices=['prepare','fit'])
    parser.add_argument('--role',type=int,choices=(-1,0,1),default=0)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare':prepare(args.out)
    else:run(args.out,args.role)
