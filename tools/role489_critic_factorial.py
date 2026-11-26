"""Weekly same-state counterfactuals on uninspected TRAIN sessions; fixed actor."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tools.run_frozen_critic_repair import (configure, cached, digest, read, write, file_sha,
    load_bundle, verify_files, market_for, support_hash, environment, emit, continuation,
    policy_signals, SOURCE, SUPPORT, TAPES)

P = Path('/var/tmp/sac70-warm-v3-20261108')
B = Path('/var/tmp/sac-selected299-export-20261112-r1/bundle')
OLD = Path('/var/tmp/frozen-critic-repair369-20261117')
OUT = Path('/var/tmp/critic489')
F = OUT/'frozen'
SEEDS = [7, 19, 41]


def load():
    global market, spec, model, risk
    configure(489)
    spec = read(P/'SPEC.json')
    verify_files(P, {k:v for k,v in spec['files'].items() if
                     k.startswith(('data/','engine/')) or k in ('RISK.json','SUPPORT_RECIPE.json')})
    assert file_sha(B/'model.pt') == SOURCE
    model, _ = load_bundle(B,'cpu')
    market = market_for(P,spec,model.recipe)
    assert support_hash(market.inputs) == SUPPORT
    risk = read(P/'RISK.json')


def prepare():
    load()
    ix = np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
    logits, q = emit(model,market.inputs,indices=ix,twins=True)
    panel = cached(OLD/'PANEL.json',digest(read(OLD/'CONFIG.json')))['anchors']
    stamps = [pd.Timestamp(r['timestamp'],tz='UTC') for r in panel]
    weeks = []
    for t in pd.date_range('2020-01-05','2023-12-24',freq='7D',tz='UTC'):
        end = t+pd.Timedelta(days=7)
        if t.year != (end-pd.Timedelta(seconds=1)).year or any(t<=s<end for s in stamps):
            continue
        valid = market.inputs.row_ts_ns[ix]
        if ((valid>=t.value)&(valid<end.value)).sum() >= 12:
            weeks.append(str(t.date()))
    F.mkdir(parents=True,exist_ok=True)
    np.savez(F/'signals.npz',logits=logits,q=q.min(0))
    write(F/'spec.json',dict(weeks=weeks,source=SOURCE,support=SUPPORT,
          signals=file_sha(F/'signals.npz'),parent=file_sha(P/'SPEC.json'),
          alpha=float(model.loss.log_alpha.detach().exp()),gamma=model.effective['gamma'],
          tapes=list(TAPES),seeds=SEEDS,quota=12,years=[2020,2021,2022,2023],
          forest=dict(n_estimators=128,min_samples_leaf=8,max_features=1.0),
          rule='all three seeds improve week-mean regret vs actor in all four held quarters and upper paired week bootstrap limit below zero for each seed'))


def init_worker():
    global cfg, signal, prob, tapes
    load()
    cfg=read(F/'spec.json')
    assert cfg['source']==SOURCE and cfg['support']==SUPPORT and cfg['parent']==file_sha(P/'SPEC.json')
    assert cfg['signals']==file_sha(F/'signals.npz')
    z=np.load(F/'signals.npz')['logits']
    greedy,prob=policy_signals(z,None)
    signal=greedy[0]
    tapes=[policy_signals(z,s)[0][0] for s in cfg['tapes']]


def week(w):
    path=OUT/'weeks'/(w+'.json')
    ident=digest(cfg)
    if path.exists():
        cached(path,ident)
        return w
    end=str((pd.Timestamp(w)+pd.Timedelta(days=7)).date())
    def new():
        e=environment(market,risk,spec['design']['horizon'],[w,end],cfg['gamma'])
        e.reset()
        return e
    def here(e):
        return int(e.rows[e.bar]),0 if e.engine.pos is None else int(e.engine.pos.direction)
    env=new()
    candidates={-1:[],0:[],1:[]}
    while not env.finished:
        row,d=here(env)
        assert market.inputs.period[row]=='TRAIN'
        candidates[d].append(row)
        env.step(int(signal[row]))
    terminal=env.engine.balance-env.engine.cfg.capital
    rng=np.random.default_rng(int(w.replace('-','')))
    selected={int(i) for v in candidates.values() for i in rng.choice(v,min(len(v),cfg['quota']),replace=False)}
    env=new(); rows=[]
    while not env.finished:
        row,d=here(env)
        if row in selected:
            e,b=env.engine,env.bar
            p=e.pos
            ctx=[int(d==0),int(d==1),int(d==-1),0.,0.,0.,
                 max(0.,float(e.ff[e.seg_of[b]]-e.v[b])/3.6e12-1)]
            if p is not None:
                ctx[3:6]=[d*(e.c[b]-p.sl)/p.sigma,d*(p.tp-e.c[b])/p.sigma,b-p.entry_bar+1]
            greedy=[continuation(env,signal,a) for a in range(3)]
            cash=[r['net_usd'] for r in greedy]
            assert abs(cash[signal[row]]-terminal)<1e-6
            soft=np.mean([[continuation(env,t,a,prob,cfg['alpha'])['soft_return']
                           for a in range(3)] for t in tapes],axis=0)
            rows.append(dict(row=row,side=d,ctx=ctx,actor=int(signal[row]),cash=cash,
                             greedy=[r['reward_return'] for r in greedy],soft=soft.tolist()))
        env.step(int(signal[row]))
    assert len(rows)==len(selected) and rows
    from tools.run_frozen_critic_repair import save
    save(path,ident,dict(week=w,rows=rows))
    return w


def worker(part,workers):
    init_worker()
    weeks=[w for i,w in enumerate(cfg['weeks']) if part=='all' or
           (part=='local' and i%3==0) or (part=='remote' and i%3!=0)]
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(week,w) for w in weeks]
        for i,f in enumerate(as_completed(futures),1):
            print(f.result(),i,len(weeks),flush=True)


def analyze():
    from sklearn.ensemble import ExtraTreesRegressor
    load(); cfg=read(F/'spec.json')
    rows=[dict(**r,week=w) for w in cfg['weeks'] for r in cached(OUT/'weeks'/(w+'.json'),digest(cfg))['rows']]
    ix=np.array([r['row'] for r in rows]); weeks=np.array([r['week'] for r in rows])
    fit=weeks<'2023'; held=~fit
    assert len(ix)==len(set(ix)) and held.any() and fit.any()
    # Hook the deployed sequential twins, not the gradient-training vmap.
    captured=[]
    hook=model.q_net.head.register_forward_pre_hook(
        lambda module,args: captured.append(args[0].detach().cpu().numpy().copy()))
    all_ix=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
    try:
        _,qs=emit(model,market.inputs,indices=all_ix,twins=True)
    finally:
        hook.remove()
    raw=np.load(F/'signals.npz')['q'][all_ix]
    np.testing.assert_array_equal(raw,qs[:,all_ix].min(0))
    assert len(captured)==2*((len(all_ix)+255)//256)
    x=np.column_stack([np.concatenate(captured[i::2]) for i in range(2)])[np.searchsorted(all_ix,ix)]
    error=float(np.max(np.abs(qs[:,all_ix].min(0)-raw)))
    ctx=np.array([r['ctx'] for r in rows]); cash=np.array([r['cash'] for r in rows])
    assert np.isfinite(x).all() and np.isfinite(ctx).all() and np.isfinite(cash).all()
    a=np.array([r['actor'] for r in rows]); idx=np.arange(len(a))
    baseline=cash.max(1)-cash[idx,a]
    dates=pd.DatetimeIndex(weeks); side=np.array([r['side'] for r in rows])
    native=np.load(F/'signals.npz')['q'][ix].argmax(1)
    results=[]
    for name in ('native','soft','greedy'):
        for conditional in ([False] if name=='native' else [False,True]):
            for seed in ([0] if name=='native' else SEEDS):
                act=native
                if name!='native':
                    features=np.c_[x,ctx if conditional else np.zeros_like(ctx)]
                    y=np.array([r[name] for r in rows]);y-=y.mean(1,keepdims=True)
                    forest=ExtraTreesRegressor(**cfg['forest'],random_state=seed,n_jobs=8)
                    forest.fit(features[fit],y[fit]);act=forest.predict(features).argmax(1)
                regret=cash.max(1)-cash[idx,act]
                diff=regret-baseline
                def byweek(mask):
                    return pd.Series(diff[mask],index=weeks[mask]).groupby(level=0).mean().to_numpy()
                v=byweek(held)
                ci=np.quantile(np.random.default_rng(489).choice(v,(5000,len(v))).mean(1),[.025,.975])
                quarters=[float(byweek(held&(dates.quarter==k)).mean()) for k in range(1,5)]
                results.append(dict(target=name,position=conditional,seed=seed,mean_week_delta_usd=float(v.mean()),
                    ci95=ci.tolist(),quarters=quarters,passed=bool(max(quarters)<0 and ci[1]<0),
                    roles={str(d):dict(n=int((held&(side==d)).sum()),delta=float(byweek(held&(side==d)).mean())) for d in (-1,0,1)}))
    write(OUT/'RESULT.json',dict(n=len(ix),n_fit=int(fit.sum()),n_monitor=int(held.sum()),weeks=len(set(weeks)),
        monitor_weeks=len(set(weeks[held])),embedding_error=error,arms=results,
        qualified=[(n,c) for n in ('soft','greedy') for c in (False,True)
                   if all(r['passed'] for r in results if r['target']==n and r['position']==c)],
        actor_updated=False,validation_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','worker','analyze'])
    parser.add_argument('--part',choices=['local','remote','all'],default='all')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='analyze':analyze()
    else:worker(args.part,args.workers)
