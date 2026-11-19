"""Non-training diagnostic for frozen 379 short heads; see protocol 381."""
import argparse
from collections import defaultdict
from math import fsum
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.close_surface import scores
from app.decision_attribution import fork_env, state_view
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import environment
from docs.work_plan.generalization.close_attribution380 import equal, PINS
from docs.work_plan.generalization.role_conflict378 import value
from docs.work_plan.generalization.role_surface379 import basis, query
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import read, write

def summarize(rows):
    risk=np.array([r['risk'] for r in rows])
    actions={h:np.array([r[h] for r in rows],bool) for h in ('scalar','table')}
    quarters=np.array([r['q'] for r in rows])
    out={};sign={}
    for k in 'ABC':
        usd=np.array([r[k] for r in rows])
        y=np.where(abs(usd)>1e-8,usd,0)/risk
        sign[k]=np.sign(y)
        regrets={h:np.maximum(0,y)-act*y for h,act in actions.items()}
        delta=regrets['table']-regrets['scalar']
        groups={q:delta[quarters==q] for q in sorted(set(quarters))}
        g=np.array([[v.sum(),len(v)] for v in groups.values()])
        sampled=g[np.random.default_rng(381).integers(len(g),size=(4000,len(g)))].sum(1)
        out[k]=dict(ties=int(sum(y==0)),close_better=int(sum(y>0)),
            **{h+'_regret_R':float(v.mean()) for h,v in regrets.items()},
            delta_R=float(delta.mean()),ci95=np.quantile(sampled[:,0]/sampled[:,1],[.025,.975]).tolist(),
            quarterly_delta_R={q:float(v.mean()) for q,v in groups.items()})
    pred=np.where(actions['table'],1,-1)
    wrong=sign['C']*pred<0
    prior=(sign['A']*pred)[wrong]
    return dict(n=len(rows),targets=out,flips={a+b:int(sum(sign[a]*sign[b]<0)) for a,b in ('AB','BC','AC')},
        table_C_errors_by_A=dict(wrong=int(sum(prior<0)),right=int(sum(prior>0)),tie=int(sum(prior==0))))

def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    heads={}
    for name,pin in PINS.items():
        if file_sha(a.panel/name)!=pin:raise ValueError(name)
    for side in (-1,1):
        with np.load(a.panel/f'HEAD_{side}.npz') as f:heads[side]=dict(f)
    old=read(a.old/'CONFIG.json');verify_done(a.old,old['identity']);verify_done(a.cache,old['diagnostic'])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    if file_sha(a.bundle/'model.pt')!=old['source_sha256']:raise ValueError('source')
    src,_=load_bundle(a.bundle,'cpu');m=market_for(a.parent,spec,src.recipe)
    if support_hash(m.inputs)!=old['support']:raise ValueError('support')
    with np.load(a.cache/'SIGNALS.npz') as f:sig=f['source'].copy()
    def make(year):
        e=environment(m,read(a.parent/'RISK.json'),spec['design']['horizon'],
            [f'{year}-01-01',f'{year+1}-01-01'],src.effective['gamma']);e.reset();return e
    scalar_cache={}
    def act(e,scalar=False):
        ix=int(e.rows[e.bar]);p=e.engine.pos
        if e.inputs.period[ix]!='TRAIN':raise ValueError('not TRAIN')
        if p is None:return int(sig[ix])
        key=(ix,p.direction)
        if scalar and key in scalar_cache:return scalar_cache[key]
        s,v=scores(heads[p.direction],basis(e)[None,:],query(e)[None,:])
        close=2 if p.direction==1 else 1
        scalar_cache[key]=close if s[0]>0 else 0
        return scalar_cache[key] if scalar else (close if v[0]>0 else 0)
    def walk(year):
        e=make(year)
        while not e.finished:
            stamp=int(e.inputs.row_ts_ns[e.rows[e.bar]])
            t=pd.Timestamp(stamp,unit='ns',tz='UTC')
            if t.year!=year:raise ValueError(('calendar',t))
            choice=act(e)
            if e.engine.pos is not None and e.engine.pos.direction==-1:
                yield e,stamp,f'{year}Q{t.quarter}',choice,act(e,True)
            e.step(choice)
    def branch(e,first):
        b=fork_env(e);idx=len(e.engine.trades);p=e.engine.pos
        initial=b.engine.cfg.capital
        for j in range(24):
            if b.finished:break
            b.step(first if j==0 else act(b,True))
        end=b.bar;mark=b.engine.equity_at(end)
        while not b.finished:b.step(act(b,True))
        equal(b.engine.balance-initial,fsum(t['net_usd'] for t in b.engine.trades))
        original=b.engine.trades[idx]
        if (original['entry_bar_abs'],original['direction'],original['units'])!=(p.entry_bar,-1,p.units):
            raise ValueError('wrong original order')
        return dict(B=float(mark),C=float(b.engine.balance),original=float(original['net_usd']),
                    end24=end,end_year=b.bar,A=value(e,first,sig))
    buckets=defaultdict(lambda:defaultdict(list))
    for year in range(2020,2023):
        for _,t,q,table,scalar in walk(year):
            buckets['general'][q].append(t)
            if table==1 and scalar==0:buckets['close_disagreement'][q].append(t)
    rng=np.random.default_rng(381)
    sets={k:{int(t) for q,ts in sorted(b.items()) for t in rng.choice(ts,min(12,len(ts)),False)}
          for k,b in buckets.items()}
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'DESIGN.json',dict(sample={k:sorted(v) for k,v in sets.items()},
          eligible={k:{q:len(v) for q,v in b.items()} for k,b in buckets.items()},pins=PINS,
          script=file_sha(__file__),protocol=file_sha(Path(__file__).parents[1]/'381_SHORT_TARGET_PROTOCOL.md')))
    wanted=set.union(*sets.values());rows=[]
    for year in range(2020,2023):
        for e,t,q,table,scalar in walk(year):
            if t not in wanted:continue
            before=state_view(e);p=e.engine.pos
            branches=[branch(e,i) for i in (0,1)]
            h,c=branches
            if before!=state_view(e) or any(h[k]!=c[k] for k in ('end24','end_year')):
                raise ValueError('unmatched forks')
            # Independent first-target reconstruction against the legacy label.
            for i,v in enumerate(branches):
                b=fork_env(e);b.step(i)
                for _ in range(23):
                    if b.finished:break
                    b.step(int(sig[b.rows[b.bar]]))
                equal(v['A'],b.engine.equity_at(b.bar))
                if b.bar!=h['end24']:raise ValueError('source clock')
            rows.append(dict(t=t,q=q,risk=abs(p.entry_price-p.sl)*abs(p.units),
                scalar=bool(scalar),table=bool(table),entry_bar=int(p.entry_bar),
                **{k:c[k]-h[k] for k in ('A','B','C','original')},branches=branches))
        print(year,len(rows),flush=True)
    if {r['t'] for r in rows}!=wanted or len(rows)!=len(wanted):raise ValueError('coverage')
    write(a.out/'ROWS.json',rows)
    result={k:summarize([r for r in rows if r['t'] in ts]) for k,ts in sets.items()}
    write(a.out/'RESULT.json',result)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('parent','bundle','old','cache','panel','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
