"""Offline qualification pilot. Never launches a policy or reads 2024."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from tools.audit_g3_producers import authenticate, load_external, producers, rebuild, sha, write


def macroblocks(cat, bars, ts, width):
    records = []
    for _, chain in cat.groupby(cat.start.ne(cat.end.shift()).cumsum()):
        for j in range(0, len(chain) - width + 1, width):
            c = chain.iloc[j:j+width]
            a, b = int(c.start.iloc[0]), int(c.end.iloc[-1])
            assert (ts[b]-ts[a]) == pd.Timedelta(hours=b-a)
            r = np.diff(np.log(bars[a:b+1, 3]))
            records.append((a, b, r.sum()/max(abs(r).sum(), 1e-12),
                            np.log(max(r.std(), 1e-8)), np.log(len(r)),
                            c.retrace.mean()/max(abs(c.displacement).mean(), 1e-8)))
    return np.array(records)


def fit_regimes(blocks, ts):
    x = blocks[:, 2:]
    year = ts[blocks[:, 1].astype(int)].year
    scale = StandardScaler().fit(x[year < 2022])
    candidates = {}
    for k in (2, 3, 4, 5, 6):
        model = GaussianMixture(k, covariance_type='diag', n_init=8, random_state=17)
        model.fit(scale.transform(x[year < 2022]))
        candidates[k] = model.score(scale.transform(x[year == 2022]))
    k = max(candidates, key=candidates.get)
    scale.fit(x[year < 2023])
    model = GaussianMixture(k, covariance_type='diag', n_init=8, random_state=17)
    model.fit(scale.transform(x[year < 2023]))
    return model.predict(scale.transform(x)), candidates


def distance(x, y):
    return np.mean(np.minimum(abs(x-y), 6), axis=-1)


def transport_basis(c, fx, paths, series, donor):
    """Retain the donor's observed quote residual, not its cumulative drift."""
    pos = {k:i for i,k in enumerate(fx.columns)}
    if not {'eurjpy','usdjpy','eurgbp','gbpusd'} <= pos.keys():
        raise ValueError('missing FX legs')
    series[:,pos['eurjpy']] = (paths*c[donor]**-1 * series[:,pos['usdjpy']] *
                               (fx.eurjpy/fx.usdjpy).to_numpy()[donor])
    series[:,pos['eurgbp']] = (paths*c[donor]**-1 / series[:,pos['gbpusd']] *
                               (fx.eurgbp*fx.gbpusd).to_numpy()[donor])


def generate(blocks, labels, ts, bars, fx, context, seed, random_control, use_context=True, calendar_end=2023):
    rng = np.random.default_rng(seed)
    a, b = blocks[:, :2].astype(int).T
    fit = ts[b].year < 2023
    prices = np.column_stack((bars[:, 3], fx))
    valid = np.array([np.isfinite(prices[l:r+1]).all() for l,r in zip(a,b)])
    lib = np.flatnonzero(fit & valid)
    adjacent = (b[:-1] == a[1:]) & fit[:-1] & fit[1:] & valid[1:]
    pre, post = np.where(adjacent)[0], np.where(adjacent)[0]+1
    # Both variants have the same admission and calendar; only selection differs.
    eligible = post[np.isfinite(context[a[post]]).all(axis=1)] if use_context else post
    ctx = StandardScaler().fit(context[a[eligible]]).transform(context) if use_context else None
    logp = np.log(prices)
    out = np.full_like(prices, np.nan)
    outbars = np.full_like(bars, np.nan)
    last = np.flatnonzero(ts.year < calendar_end)[-1]+1
    starts = np.r_[0, np.flatnonzero(np.diff(ts[:last].as_unit('ns').asi8) != 3600*10**9)+1, last]
    trace = []
    donor_rows = np.arange(last)
    for l, end in zip(starts[:-1], starts[1:]):
        # Keep true closure gaps; do not create an ordinary hourly move across them.
        if l == 0:
            out[l] = prices[l]
            outbars[l] = bars[l]
        else:
            ratio = prices[l]/prices[l-1]
            out[l] = out[l-1]*ratio
            outbars[l] = bars[l]*(out[l,0]/prices[l,0])
        i, previous = l+1, None
        while i < end:
            choices = eligible
            if not random_control and previous is not None:
                nexts = post[labels[pre] == labels[previous]]
                if len(nexts):
                    state = labels[rng.choice(nexts)]
                    choices = eligible[labels[eligible] == state]
            if not len(choices):
                choices = lib
            if not random_control:
                h = abs(ts[a[choices]].hour.to_numpy()-ts[i-1].hour)
                cost = np.minimum(h, 24-h)/6
                if ctx is not None and np.isfinite(ctx[i-1]).all():
                    cost += distance(ctx[a[choices]], ctx[i-1])
                if previous is not None:
                    cost += abs(blocks[choices-1,2]-blocks[previous,2])
                choices = choices[np.argsort(cost, kind='stable')[:8]]
            d = int(rng.choice(choices))
            u = a[d]
            n = min(b[d]-u, end-i)
            relative = np.exp(logp[u+1:u+1+n]-logp[u])
            out[i:i+n] = out[i-1]*relative
            outbars[i:i+n] = bars[u+1:u+1+n]*(out[i-1,0]/prices[u,0])
            donor_rows[i:i+n] = np.arange(u+1,u+1+n)
            trace.append((i, i+n, u, u+n, int(labels[d]), int(d)))
            previous, i = d, i+n
    if set(('eurjpy','usdjpy','eurgbp','gbpusd'))<=set(fx):
        transport_basis(bars[:,3],fx,outbars[:last,3],out[:last,1:],donor_rows)
    return outbars[:last], out[:last,1:], np.asarray(trace)


def measures(frame, features, mask, risk, vol_cut):
    """Native price distances, no look across a missing hour or quarter."""
    f = frame.loc[mask]
    z = features.loc[mask]
    close = f.CLOSE.to_numpy()
    clock = f.index.as_unit('ns').asi8
    intra = np.diff(clock) == 3600*10**9
    r = pd.Series(np.r_[np.nan, np.where(intra,np.diff(np.log(close)),np.nan)],index=f.index)
    m = dict(abs_ret=float(r.abs().mean()), tail=float(r.abs().quantile(.95)),
             vol_cluster=float(r.abs().autocorr()), wick=float(((f.HIGH-f.LOW)/f.CLOSE).mean()))
    direction = r.rolling(12).sum().shift(1).to_numpy()
    vol = r.rolling(12).std().shift(1).to_numpy()
    for h,side,sl,tp in risk:
        h, side = int(h), int(side)
        v = np.lib.stride_tricks.sliding_window_view
        ok = clock[h-1:]-clock[:len(clock)-h+1] == (h-1)*3600*10**9
        op = f.OPEN.to_numpy()[:-h+1,None]
        hi,lo = v(f.HIGH,h),v(f.LOW,h)
        fav,adv = (hi-op,op-lo) if side==1 else (op-lo,hi-op)
        st,pt = adv>=sl*1e-4, fav>=tp*1e-4
        it = np.where(st.any(1),st.argmax(1),h)
        ip = np.where(pt.any(1),pt.argmax(1),h)
        m[f'sl_first_{h}_{side}'] = (it[ok]<ip[ok]).mean()
        m[f'tp_first_{h}_{side}'] = (ip[ok]<it[ok]).mean()
        # Exit at next open only if alive, excluding indeterminate double hits.
        alive = ok & (it>0) & (ip>0) & ~((it==ip)&(it<h))
        expiry = side*(v(f.CLOSE,h)[:,-1]-op[:,0])*1e4
        hold = np.where(it<ip,-sl,np.where(ip<it,tp,expiry))
        exit_now = side*(v(f.OPEN,h)[:,1]-op[:,0])*1e4
        m[f'early_minus_hold_pips_{h}_{side}'] = (exit_now-hold)[alive].mean()
        for sign in (-1,1):
            for high in (False,True):
                cell = alive & ((sign*direction[:len(ok)])>0) & np.isfinite(vol[:len(ok)]) & ((vol[:len(ok)]>vol_cut)==high)
                m[f'early_{h}_{side}_{sign}_{int(high)}'] = (exit_now-hold)[cell].mean() if cell.sum()>=20 else np.nan
    for col in z:
        if col.startswith(('dly.', 'cot.', 'fac.', 'wv.usd','hb.eur','act.')):
            m['vol_corr_'+col] = z[col].corr(r.abs())
    return pd.Series(m)


def gate_table(sample, target, envelope):
    table = pd.DataFrame(dict(generated=sample,held_2023=target.loc[sample.index],
                              tolerance=envelope.loc[sample.index]))
    table['pass'] = (table.generated-table.held_2023).abs()<=table.tolerance
    return table


def physics(c, fx, ts, joins):
    # Absolute triangle residuals catch drift that return correlations miss.
    def triangle(numerator, denominator):
        residual = abs(np.log(c*fx[denominator]/fx[numerator]))
        return float(np.nanquantile(residual,.95))
    j = joins[(joins>=2)&(joins<len(c))]
    j = j[(ts[j]-ts[j-2])==pd.Timedelta(hours=2)]
    curr,prev = np.log(c[j]/c[j-1]),np.log(c[j-1]/c[j-2])
    return pd.Series(dict(triangle_jpy=triangle('eurjpy','usdjpy'),
                triangle_gbp=float(np.nanquantile(abs(np.log(c/fx.gbpusd/fx.eurgbp)),.95)),
                join_reversal=float(np.mean(curr*prev<0)),join_abs_ret=float(abs(curr).mean())))


def blind_analogs(blocks, labels, ts, z):
    """Predict an unseen block's external mean using *only its previous block*.

    Empirical CRPS scores the whole donor ensemble, not just its mean.
    All donors are pre-2023; all scored targets are within held-out 2023.
    """
    a,b = blocks[:,:2].astype(int).T
    adj = np.flatnonzero(b[:-1]==a[1:])+1
    value = np.array([z.iloc[l+1:r+1].mean().to_numpy() for l,r in zip(a,b)])
    donors = adj[(ts[b[adj]].year<2023)&np.isfinite(value[adj]).all(1)]
    held = adj[(ts[a[adj-1]].year==2023)&np.isfinite(value[adj]).all(1)]
    assert ts[b[donors]].max()<ts[a[held-1]].min()
    scal = StandardScaler().fit(blocks[donors-1,2:])
    x = scal.transform(blocks[:,2:])
    unit = value[donors].std(axis=0)
    if np.any(unit==0): raise ValueError('constant held-out target scale')
    rng = np.random.default_rng(397)
    rows = []
    for i in held:
        group = donors[labels[donors-1]==labels[i-1]]
        hr = abs(ts[a[group]].hour.to_numpy()-ts[a[i]].hour)
        metric = distance(x[group-1],x[i-1])+np.minimum(hr,24-hr)/6
        chosen = group[np.argsort(metric,kind='stable')[:8]]
        if len(chosen)!=8: raise ValueError('insufficient analog support')
        pair = []
        for ix in (chosen,rng.choice(donors,8,replace=False)):
            v = value[ix]/unit
            pair.append(abs(v-value[i]/unit).mean(0)-.5*abs(v[:,None]-v[None,:]).mean((0,1)))
        rows.append(pair[1]-pair[0])
    delta = pd.DataFrame(rows,index=ts[a[held]],columns=z.columns)
    quarterly = delta.groupby(delta.index.quarter).mean()
    return dict(n=len(held),by_column=delta.mean().to_dict(),
                quarterly_advantage=quarterly.mean(axis=1).to_dict(),
                better_every_quarter=bool((quarterly.mean(axis=1)>0).all()))


def run(args):
    out = args.out
    out.mkdir(exist_ok=False, parents=True)
    spec = json.loads(args.spec.read_text())
    if sha(args.spec) != 'a37df7503445d0eddff3207b08e6f9f5c424d4f64fef1b8dd309472f15d95813':
        raise ValueError('spec')
    settings = spec['producer_settings']
    spec['pins'][str(args.factor)] = spec['pins'].pop(settings['factor_state'])
    settings['factor_state'] = str(args.factor)
    authenticate(spec)
    with np.load(args.business/'19.npz') as data:
        bars, ts, risk = data['real'], pd.to_datetime(data['ts'],utc=True), data['risk']
    assert ts.min().year == 2020 and ts.max().year == 2023
    ts.name = 'timestamp'
    train = ts.year < 2023
    real = pd.DataFrame(bars,index=ts,columns=['OPEN','HIGH','LOW','CLOSE'])
    cols = json.loads(args.recipe.read_text())['list']['columns']
    stored = pd.read_parquet(settings['store'], columns=['timestamp',*cols],
        filters=[('timestamp','>=',ts[0]),('timestamp','<=',ts[-1])]).set_index('timestamp').loc[ts]
    ext, loaded = load_external(spec), producers(spec)
    state = json.loads(Path(settings['scope']).read_text())
    closures = json.loads(Path(spec['closures']).read_text())
    extcols = [c for c in cols if c.startswith(('dly.','cot.','act.','fac.','wv.usd','hb.eur'))]
    pricecols = [c for c in cols if c not in extcols]
    assert (len(pricecols),len(extcols)) == (61,10)
    cat = pd.read_parquet(args.business/'real_catalog.parquet')
    blocks = macroblocks(cat, bars, ts, 4)
    labels, scores = fit_regimes(blocks,ts)
    fx = pd.DataFrame({k:v.reindex(ts) for k,v in ext['fx'].items()})
    # Use only previously published rates/COT to match context, never future price.
    context = stored[[c for c in extcols if c.startswith(('dly.','cot.'))]].shift(1).to_numpy()
    quarters = ts.year*4 + ts.quarter
    fit_close = real.CLOSE[ts.year<2022]
    valid_clock = fit_close.index.to_series().diff()==pd.Timedelta(hours=1)
    vol_cut = np.log(fit_close).diff().where(valid_clock).rolling(12).std().median()
    reference = pd.DataFrame([measures(real,stored,quarters==q,risk,vol_cut) for q in np.unique(quarters[train])])
    held = measures(real,stored,~train,risk,vol_cut)
    ref = reference.to_numpy()
    ii,jj = np.triu_indices(len(ref),k=1)
    tol = np.nanquantile(abs(ref[ii]-ref[jj]),.90,axis=0)
    report = dict(stage='NON_GOVERNING_GENERATOR_SCREEN', k_scores=scores,
                  blocks=len(blocks),fit_blocks=int((ts[blocks[:,1].astype(int)].year<2023).sum()),
                  arms={},training_released=False)
    report['blind_analogs'] = blind_analogs(blocks,labels,ts,stored[extcols])
    join = blocks[np.r_[False,blocks[1:,0]==blocks[:-1,1]],0].astype(int)+1
    def real_physics(mask):
        ix = np.where(mask)[0]
        return physics(bars[ix,3],fx.iloc[ix],ts[ix],join[(join>=ix[0])&(join<=ix[-1])]-ix[0])
    base = pd.DataFrame([real_physics(quarters==q) for q in np.unique(quarters[train])])
    held_phys = real_physics(~train)
    tol_phys = pd.Series(np.nanquantile(abs(base.values[ii]-base.values[jj]),.9,axis=0),index=base.columns)
    for seed in args.seeds:
        for mode in ('random','price','context'):
            name = mode+str(seed)
            p, crosses, trace = generate(blocks,labels,ts,bars,fx,context,seed,mode=='random',mode=='context')
            f = pd.DataFrame(p,index=ts[train],columns=real.columns)
            assert np.isfinite(p).all() and (p>0).all() and (p[:,1]>=p[:,[0,3]].max(1)).all() and (p[:,2]<=p[:,[0,3]].min(1)).all()
            # Restore observed cross availability. Missing crosses cannot be fabricated.
            crosses[~np.isfinite(fx.loc[train].to_numpy())] = np.nan
            e = dict(ext,fx={k:pd.Series(crosses[:,i],index=f.index) for i,k in enumerate(fx)})
            independent = rebuild(f,pricecols,state,closures,loaded)
            full = independent if mode=='price' else rebuild(f,cols,state,closures,loaded,e)
            pd.testing.assert_frame_equal(independent,full[pricecols],check_exact=True)
            dest = out/name
            dest.mkdir()
            for fname,table in [('OHLC',f),('FEATURES61',independent)] + ([] if mode=='price' else [('FEATURES71',full)]):
                table.to_parquet(dest/(fname+'.parquet'))
            np.save(dest/'DONORS.npy',trace)
            m = measures(f,full,np.ones(len(f),bool),risk,vol_cut)
            check = gate_table(m,held,pd.Series(tol,index=held.index))
            xfx = pd.DataFrame(crosses,index=f.index,columns=fx.columns)
            joins = trace[1:,0]
            joins = joins[joins==trace[:-1,1]]
            phys = physics(p[:,3],xfx,f.index,joins)
            if mode=='price':
                phys = phys[['join_reversal','join_abs_ret']]
            pcheck = gate_table(phys,held_phys,tol_phys)
            pcheck.to_csv(dest/'physics.csv')
            check.to_csv(dest/'gate.csv')
            report['arms'][name] = dict(rows=len(f),blocks=len(trace),used_donors=len(set(trace[:,5])),
                    features=len(full.columns),finite_rows=int(np.isfinite(full).all(axis=1).sum()),
                    price_distance=float(((m-held).abs()/pd.Series(tol,index=held.index)).loc[
                        [k for k in m.index if not k.startswith('vol_corr_')]].mean()),
                    physics_failed=pcheck.index[~pcheck['pass']].tolist(),
                    failed=check.index[~check['pass']].tolist())
            write(out/'PROGRESS.json',report)
    for mode in ('price','context'):
        report[mode+'_screen'] = all(not report['arms'][mode+str(s)]['failed'] and
            report['arms'][mode+str(s)]['price_distance'] < report['arms']['random'+str(s)]['price_distance']
            for s in args.seeds) and (mode=='price' or report['blind_analogs']['better_every_quarter'])
        report[mode+'_physical_screen'] = all(not report['arms'][mode+str(s)]['physics_failed'] for s in args.seeds)
    write(out/'RESULT.json',report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('out','spec','factor','business','recipe'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,19])
    run(parser.parse_args())
