"""Offline role-conflict gate: frozen source, native TRAIN-only forks."""
import argparse
from collections import defaultdict
from pathlib import Path
import os

import numpy as np
import pandas as pd

from app.critic_identifiability import forked_state_test
from app.decision_attribution import fork_env,state_view
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest,support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment
from tools.diagnose_critic_coverage_drift import INTERVALS
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import configure,read,write


def regret(y):
    y=np.asarray(y,float)
    if y.ndim!=2 or y.shape[1]!=3 or not np.isfinite(y).all():raise ValueError('outcomes')
    return float((y.max(1)[:,None]-y).mean(0).min())


def value(env,a,signals):
    b=fork_env(env)
    for j in range(24):
        if b.finished:break
        b.step(a if j==0 else int(signals[b.rows[b.bar]]))
    return float(b.engine.equity_at(b.bar))


def states(env,signals):
    before=state_view(env);rows=[]; skipped=[]
    for tag,seq in [('flat',(0,0)),('long_old',(1,0)),('long_new',(0,1)),
                    ('short_old',(2,0)),('short_new',(0,2))]:
        b=fork_env(env)
        for a in seq:
            if not b.finished:b.step(a)
        p=b.engine.pos
        if b.finished or (tag!='flat' and p is None):
            skipped.append(tag);continue
        direction=p.direction if p else 0
        if direction!={'flat':0,'long':1,'short':-1}[tag.split('_')[0]]:raise ValueError('role')
        ix=int(b.rows[b.bar])
        if b.inputs.period[ix]!='TRAIN':raise ValueError('period')
        if rows and (rows[0]['row']!=ix or not np.array_equal(ref,b.observation()['features'])):
            raise ValueError('unequal market state')
        ref=b.observation()['features'].copy()
        outcomes=[value(b,i,signals) for i in range(3)]
        if p and outcomes[0]!=outcomes[1 if direction==1 else 2]:raise ValueError('hold equivalence')
        row=dict(tag=tag,side=direction,row=ix,y=outcomes)
        if p:
            mark=float(b.engine.c[b.bar]);ds=direction*(mark-p.sl);dt=direction*(p.tp-mark)
            row.update(sl_distance=ds,tp_distance=dt,near_sl=ds<dt,
                       entry_cost=sum(p.entry_cost.values()),risk=abs(p.entry_price-p.sl)*abs(p.units))
        rows.append(row)
    if state_view(env)!=before:raise ValueError('parent mutated')
    groups=defaultdict(list)
    for r in rows:groups[r['side']].append(r)
    pairs=[]
    for side,rs in groups.items():
        if side and len(rs)==2:
            g=regret([r['y'] for r in rs]);fee=np.mean([r['entry_cost'] for r in rs])
            pairs.append(dict(side=side,usd=g,material=bool(g>fee),cost=float(fee),
                risk_units=regret([np.asarray(r['y'])/r['risk'] for r in rs]),
                barriers='mixed' if rs[0]['near_sl']!=rs[1]['near_sl'] else ('near_SL' if rs[0]['near_sl'] else 'near_TP')))
    return dict(rows=rows,skipped=skipped,pairs=pairs,
        shared_usd=regret([r['y'] for r in rows]) if rows else None,
        routed_usd=sum(len(rs)*regret([r['y'] for r in rs]) for rs in groups.values())/len(rows) if rows else None)


def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    configure(378)
    old=read(a.old/'CONFIG.json');verify_done(a.old,old['identity'])
    verify_done(a.cache,old['diagnostic'])
    historic={r['anchor']['timestamp']:r for r in read(a.old/'RESULT.json')['origins']}
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    if file_sha(a.bundle/'model.pt')!=old['source_sha256']:raise ValueError('source')
    model,_=load_bundle(a.bundle,'cpu');market=market_for(a.parent,spec,model.recipe)
    if support_hash(market.inputs)!=old['support']:raise ValueError('support')
    risk=read(a.parent/'RISK.json')
    with np.load(a.cache/'SIGNALS.npz') as d:sig=d['source'].copy()
    def make(interval):
        e=environment(market,risk,spec['design']['horizon'],interval,model.effective['gamma'])
        e.reset();return e
    candidates=defaultdict(list);rng=np.random.default_rng(378)
    oldrows={r['anchor']['row'] for r in historic.values()}
    for _,interval in INTERVALS:
        env=make(interval)
        while not env.finished:
            ix=int(env.rows[env.bar]);t=int(market.inputs.row_ts_ns[ix])
            if env.engine.pos is None and all(abs(ix-o)>24 for o in oldrows):
                candidates[pd.Timestamp(t).year].append(t)
            env.step(int(sig[ix]))
    selected={y:sorted(rng.choice(sorted(ts),min(64,len(ts)),replace=False).tolist()) for y,ts in sorted(candidates.items())}
    a.out.mkdir(parents=True,exist_ok=True)
    write(a.out/'DESIGN.json',dict(selected=selected,source=old['source_sha256'],support=old['support'],
        protocol=file_sha(Path(__file__).parents[1]/'378_ROLE_CONFLICT_GATE.md'),script=file_sha(__file__)))
    wanted={t for ts in selected.values() for t in ts};evidence=[];checked=[]
    for _,interval in INTERVALS:
        env=make(interval)
        while not env.finished:
            ix=int(env.rows[env.bar]);t=int(market.inputs.row_ts_ns[ix])
            if t in historic:
                src=historic[t]
                if digest(state_view(env))!=src['anchor']['state_sha256']:raise ValueError('witness state')
                got=forked_state_test(env,sig)
                if got['outcomes']!=src['outcomes'] or got['ambiguity']!=src['ambiguity']:raise ValueError('witness outcomes')
                checked.append((t,not got['ambiguity']['common_optimal_actions']))
            if t in wanted:
                evidence.append(dict(timestamp=t,year=pd.Timestamp(t).year,**states(env,sig)))
            env.step(int(sig[ix]))
    if len(checked)!=16 or sum(v for _,v in checked)!=3 or len(evidence)!=len(wanted):raise ValueError('coverage')
    write(a.out/'PANEL.json',dict(origins=evidence,reproduced=checked))
    stats={}
    for label,part in [('development',[r for r in evidence if r['year']<2023]),('reserved',[r for r in evidence if r['year']==2023])]:
        pairs=[p for r in part for p in r['pairs']]
        eligible=[r for r in part if r['shared_usd'] is not None]
        stats[label]=dict(origins=len(part),shared_usd=np.mean([r['shared_usd'] for r in eligible]),routed_usd=np.mean([r['routed_usd'] for r in eligible]),strata={})
        for side in [1,-1]:
            for bucket in ['all','mixed','near_SL','near_TP']:
                ps=[p for p in pairs if p['side']==side and bucket in ('all',p['barriers'])]
                stats[label]['strata'][str(side)+':'+bucket]=dict(n=len(ps),conflicts=sum(p['usd']>1e-8 for p in ps),
                    material=sum(p['material'] for p in ps),mean_usd=float(np.mean([p['usd'] for p in ps])) if ps else None,
                    max_usd=max((p['usd'] for p in ps),default=None),mean_R=float(np.mean([p['risk_units'] for p in ps])) if ps else None)
    write(a.out/'RESULT.json',dict(witnesses=16,conflicts_reproduced=3,stats=stats))
    print(stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('old','cache','parent','bundle','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
