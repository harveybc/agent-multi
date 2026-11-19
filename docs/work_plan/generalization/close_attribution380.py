"""Native, non-training counterfactual accounting for the frozen 379 heads."""
import argparse
from collections import defaultdict
from math import fsum, isclose
import os
from pathlib import Path

import numpy as np

from app.close_surface import scores
from app.decision_attribution import fork_env, state_view, reconcile
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import environment
from docs.work_plan.generalization.role_surface379 import basis, query
from tools.diagnose_critic_coverage_drift import INTERVALS
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import read, write

PINS = {
    'RESULT.json': '82b0113230186d61356d0dd9ecf80e3f64510c22988a036e39aa0b675926a68d',
    'HEAD_1.npz': 'd229c4d85a4a480ea7fe338456833fb691fd1e30f02729608d99ed20de371043',
    'HEAD_-1.npz': '2809e9dbbf9ff047f1139b7fc8608fe73d979cf078127003729147b571e31af2',
}

def equal(x,y):
    if not isclose(x,y,rel_tol=0,abs_tol=1e-6):
        raise ValueError(('account mismatch',x,y))

def partition(scalar, surface, identity):
    def parts(trades):
        if not trades or tuple(trades[0][k] for k in ('entry_bar_abs','direction','units'))!=identity:
            raise ValueError('missing original order settlement')
        later={(t['entry_bar_abs'],t['direction']):t['net_usd'] for t in trades[1:]}
        if len(later)!=len(trades)-1 or identity[:2] in later:
            raise ValueError('ambiguous trade identity')
        return trades[0]['net_usd'],later
    a,la=parts(scalar)
    b,lb=parts(surface)
    return dict(current=b-a, matched=fsum(lb[k]-la[k] for k in la.keys() & lb.keys()),
                changed_entries=fsum(lb[k] for k in lb.keys()-la.keys())
                                -fsum(la[k] for k in la.keys()-lb.keys()))

def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    for name,pin in PINS.items():
        if file_sha(a.panel/name)!=pin:raise ValueError(name)
    old=read(a.old/'CONFIG.json')
    verify_done(a.old,old['identity']);verify_done(a.cache,old['diagnostic'])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    if file_sha(a.bundle/'model.pt')!=old['source_sha256']:raise ValueError('source weights')
    source,_=load_bundle(a.bundle,'cpu')
    m=market_for(a.parent,spec,source.recipe)
    if support_hash(m.inputs)!=old['support']:raise ValueError('support')
    with np.load(a.cache/'SIGNALS.npz') as f:sig=f['source'].copy()
    heads={}
    for side in (-1,1):
        with np.load(a.panel/f'HEAD_{side}.npz') as f:heads[side]=dict(f)
    e=environment(m,read(a.parent/'RISK.json'),spec['design']['horizon'],INTERVALS[-1][1],source.effective['gamma'])
    e.reset()
    expected=read(a.panel/'RESULT.json')['accounts']
    initial=e.engine.cfg.capital
    last=expected['scalar']['net_usd']
    def actions(env):
        row=env.rows[env.bar]
        if env.inputs.period[row]!='TRAIN':raise ValueError('split')
        p=env.engine.pos
        if not p:return [int(sig[row])]*2
        values=scores(heads[p.direction],basis(env)[None,:],query(env)[None,:])
        return [(2 if p.direction==1 else 1) if v[0]>0 else 0 for v in values]
    def suffix(action):
        b=fork_env(e)
        b.step(action)
        while not b.finished:b.step(actions(b)[0])
        trades=b.engine.trades
        net=b.engine.balance-initial
        equal(net,fsum(t['net_usd'] for t in trades))
        return net,trades[len(e.engine.trades):]
    events=[]
    while not e.finished:
        scalar,table=actions(e)
        if scalar!=table:
            before=state_view(e);p=e.engine.pos
            ident=(int(p.entry_bar),int(p.direction),int(p.units))
            base,ta=suffix(scalar);new,tb=suffix(table)
            if state_view(e)!=before:raise ValueError('fork changed parent')
            equal(base,last)
            delta=partition(ta,tb,ident)
            equal(fsum(delta.values()),new-base)
            events.append(dict(time=before['timestamp'],side=ident[1],order=ident,
                table='CLOSE' if table else 'HOLD',delta=new-base,**delta,
                scalar_terminal=base,table_terminal=new,
                scalar_exit=ta[0]['close_reason'],table_exit=tb[0]['close_reason'],
                scalar_exit_bar=ta[0]['exit_bar_abs'],table_exit_bar=tb[0]['exit_bar_abs']))
            last=new
        e.step(table)
    terminal=e.engine.balance-initial
    equal(terminal,expected['surface']['net_usd']);equal(terminal,last)
    if len(e.engine.trades)!=expected['surface']['trades']:raise ValueError('trade count')
    recon=reconcile([r['delta'] for r in events],expected['scalar']['net_usd'],terminal)
    groups=defaultdict(list)
    for r in events:
        groups['all'].append(r);groups[f"{r['side']}:{r['table']}"].append(r)
    summary={k:dict(n=len(g),orders=len({r['order'] for r in g}),
                    **{col:fsum(r[col] for r in g) for col in ('delta','current','matched','changed_entries')})
             for k,g in groups.items()}
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'EVENTS.json',events)
    result=dict(groups=summary,reconciliation=recon,inputs=PINS,events_sha=file_sha(a.out/'EVENTS.json'))
    write(a.out/'RESULT.json',result)
    print(summary,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('parent','bundle','old','cache','panel','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())
