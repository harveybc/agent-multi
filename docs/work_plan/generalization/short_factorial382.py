"""Paired chronological 2x2; run on CPU with frozen 379 inputs."""
import argparse
from collections import defaultdict
from math import fsum
import os
from pathlib import Path
import numpy as np
import pandas as pd
from app.close_surface import fit, scores
from app.decision_attribution import fork_env
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment
from docs.work_plan.generalization.close_attribution380 import PINS,equal
from docs.work_plan.generalization.role_conflict378 import value
from docs.work_plan.generalization.role_surface379 import basis,query
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import read,write

def match(left,right):
    rng=np.random.default_rng(382)
    result=[[],[]]
    for q in sorted(left.keys() & right.keys()):
        n=min(96,len(left[q]),len(right[q]))
        for out,items in zip(result,(left[q],right[q])):
            out.extend(items[i] for i in rng.choice(len(items),n,replace=False))
    return result

def regret(y,close):
    return np.maximum(0,y)-close*y

def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    for name,pin in PINS.items():
        if file_sha(a.panel/name)!=pin:raise ValueError(name)
    old=read(a.old/'CONFIG.json')
    verify_done(a.old,old['identity']);verify_done(a.cache,old['diagnostic'])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    if file_sha(a.bundle/'model.pt')!=old['source_sha256']:raise ValueError('source')
    source,_=load_bundle(a.bundle,'cpu');m=market_for(a.parent,spec,source.recipe)
    if support_hash(m.inputs)!=old['support']:raise ValueError('support')
    with np.load(a.cache/'SIGNALS.npz') as f:sig=f['source'].copy()
    baseline={}
    for s in (-1,1):
        with np.load(a.panel/f'HEAD_{s}.npz') as f:baseline[s]=dict(f)
    risk=read(a.parent/'RISK.json')
    def make(year):
        if year not in (2020,2021,2022):raise ValueError('year')
        e=environment(m,risk,spec['design']['horizon'],[f'{year}-01-01',f'{year+1}-01-01'],source.effective['gamma'])
        e.reset();return e
    models={'incumbent':(baseline[-1],1)}
    def action(e,name):
        row=int(e.rows[e.bar])
        if e.inputs.period[row]!='TRAIN':raise ValueError('period')
        p=e.engine.pos
        if not p:return int(sig[row])
        head,which=(baseline[1],1) if p.direction==1 else models[name]
        v=scores(head,basis(e)[None,:],query(e)[None,:])[which][0]
        return (2 if p.direction==1 else 1) if v>0 else 0
    def walk(year):
        e=make(year)
        while not e.finished:
            i=int(e.rows[e.bar])
            t=pd.Timestamp(int(e.inputs.row_ts_ns[i]),unit='ns',tz='UTC')
            if t.year!=year:raise ValueError('year boundary')
            yield e,i,str(t.year)+'Q'+str(t.quarter)
            e.step(action(e,'incumbent'))
    def constructed(e,tag):
        b=fork_env(e)
        for cmd in ((2,0) if tag=='old' else (0,2)):
            if not b.finished:b.step(cmd)
        return b if not b.finished and b.engine.pos and b.engine.pos.direction==-1 else None
    def snapshot(e):
        p=e.engine.pos
        return dict(row=int(e.rows[e.bar]),x=basis(e).tolist(),q=query(e).tolist(),
                    entry=int(p.entry_bar),risk=float(abs(p.entry_price-p.sl)*abs(p.units)))
    def label(e,kind):
        if kind=='source':
            return value(e,1,sig)-value(e,0,sig)
        vals=[];ends=[]
        for cmd in (0,1):
            b=fork_env(e);b.step(cmd)
            for _ in range(23):
                if b.finished:break
                b.step(action(b,'scalar'))
            vals.append(b.engine.equity_at(b.bar));ends.append(b.bar)
        if ends[0]!=ends[1]:raise ValueError('unpaired clocks')
        return vals[1]-vals[0]
    flat=defaultdict(list);reached=defaultdict(list);art=defaultdict(list)
    for yr in (2020,2021):
        for e,i,q in walk(yr):
            if not e.engine.pos:flat[q].append(i)
            elif e.engine.pos.direction==-1:reached[q].append((i,'actual'))
    rng=np.random.default_rng(382)
    origins={int(i) for q,v in sorted(flat.items()) for i in rng.choice(v,min(128,len(v)),False)}
    for yr in (2020,2021):
        for e,i,_ in walk(yr):
            if i not in origins:continue
            for tag in ('old','new'):
                b=constructed(e,tag)
                if b:
                    stamp=pd.Timestamp(int(b.inputs.row_ts_ns[b.rows[b.bar]]),unit='ns',tz='UTC')
                    art[f'{yr}Q{stamp.quarter}'].append((i,tag))
    left,right=match(art,reached);wanted=set(left+right)
    if not left or len(wanted)!=2*len(left):raise ValueError('fit support')
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'DESIGN.json',dict(constructed=left,reachable=right,pins=PINS,
          protocol=file_sha(Path(__file__).parents[1]/'382_SHORT_FACTORIAL_PROTOCOL.md'),script=file_sha(__file__)))
    def examples():
        for yr in (2020,2021):
            for e,i,_ in walk(yr):
                for tag in ('actual','new','old'):
                    if (i,tag) in wanted:
                        yield (i,tag), e if tag=='actual' else constructed(e,tag)
    train={}
    for key,e in examples():
        r=snapshot(e);r['source']=label(e,'source')/r['risk'];train[key]=r
    if set(train)!=wanted:raise ValueError('missing fit states')
    def train_head(keys,target,name):
        rows=[train[k] for k in keys]
        h=fit(*(np.array([r[c] for r in rows]) for c in ('x','q',target)))
        path=a.out/(name+'.npz');np.savez(path,**h)
        with np.load(path) as f:models[name]=(dict(f),1)
        if any(not np.array_equal(v,models[name][0][k]) for k,v in h.items()):raise ValueError('reload')
    train_head(left,'source','constructed_source')
    models['scalar']=(models['constructed_source'][0],0)
    for key,e in examples():
        if train[key]['row']!=int(e.rows[e.bar]):raise ValueError('state replay')
        train[key]['scalar']=label(e,'scalar')/train[key]['risk']
    for pop,keys in [('constructed',left),('reachable',right)]:
        for target in ('source','scalar'):
            name=pop+'_'+target
            if name not in models:train_head(keys,target,name)
    write(a.out/'FIT.json',[dict(origin=k,**r) for k,r in train.items()])
    print('frozen four heads; fit states per arm',len(left),flush=True)
    panel=[]
    for e,i,_ in walk(2022):
        if not e.engine.pos or e.engine.pos.direction!=-1:continue
        r=snapshot(e);r.pop('x')
        r['week']=pd.Timestamp(int(e.inputs.row_ts_ns[i])).strftime('%G-%V')
        for target in ('source','scalar'):r['y_'+target]=label(e,target)/r['risk']
        r['choices']={name:action(e,name) for name in models};panel.append(r)
    write(a.out/'PANEL.json',panel)
    results={};weeks=np.array([r['week'] for r in panel])
    for name in models:
        decisions=np.array([r['choices'][name] for r in panel])
        y=np.array([r['y_scalar'] for r in panel])
        delta=regret(y,decisions)-regret(y,np.array([r['choices']['constructed_source'] for r in panel]))
        groups=np.array([[delta[weeks==w].sum(),sum(weeks==w)] for w in sorted(set(weeks))])
        draws=groups[np.random.default_rng(382).integers(len(groups),size=(4000,len(groups)))].sum(1)
        e=make(2022)
        while not e.finished:e.step(action(e,name))
        tr=e.engine.trades;net=e.engine.balance-e.engine.cfg.capital
        equal(net,fsum(t['net_usd'] for t in tr))
        eq=np.array([v for _,v in e.equity_trace])
        results[name]=dict(regret_scalar_R=float(regret(y,decisions).mean()),
              regret_source_R=float(regret(np.array([r['y_source'] for r in panel]),decisions).mean()),
              close=int(decisions.sum()),harmful=int(sum((decisions>0)&(y < -1e-8))),
              delta_ci95=np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist(),
              net_usd=net,return_pct=100*net/e.engine.cfg.capital,drawdown_pct=float(100*max(1-eq/np.maximum.accumulate(eq))),
              trades=len(tr),buy=sum(int(t['direction']==1) for t in tr),wins=sum(int(t['net_usd']>0) for t in tr))
    ref=results['constructed_source'];cut=max(results[n]['net_usd'] for n in ('constructed_source','incumbent','scalar'))
    passed=[n for n,v in results.items() if '_' in n and v['delta_ci95'][1]<0 and v['net_usd']>cut and v['drawdown_pct']<=ref['drawdown_pct']]
    write(a.out/'RESULT.json',dict(fit_per_arm=len(left),ticks=len(panel),weeks=len(set(weeks)),
        positions=len({r['entry'] for r in panel}),arms=results,follow_up=passed))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('parent','bundle','old','cache','panel','out'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
