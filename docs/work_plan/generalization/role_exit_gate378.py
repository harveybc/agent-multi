"""Oracle-only native isolated-position exit vs frozen account continuation."""
import argparse
import os
from pathlib import Path
import numpy as np

from app.decision_attribution import fork_env,state_view
from app.frozen_recipe_inputs import file_sha
from app.selected_input_market import market_for
from app.structural_refit import load_bundle
from app.three_signal_learning import environment
from tools.diagnose_critic_coverage_drift import INTERVALS
from tools.run_discrete_curve_probe import configure,read,write


def exit_values(env):
    b=fork_env(env);p=b.engine.pos;direction=p.direction
    close=2 if direction==1 else 1
    values=[]
    for delay in range(24):
        if b.finished or b.engine.pos is None:break
        early=fork_env(b);early.step(close)
        if early.engine.pos is not None:raise ValueError('close not settled')
        values.append(float(early.engine.balance))
        b.step(0)
    values.append(float(b.engine.equity_at(b.bar)))
    return values


def no_reentry(env,action,sig):
    b=fork_env(env)
    for j in range(24):
        if b.finished or b.engine.pos is None:break
        b.step(action if j==0 else int(sig[b.rows[b.bar]]))
    return float(b.engine.equity_at(b.bar))


def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    configure(378);spec=read(a.parent/'SPEC.json');design=read(a.panel/'DESIGN.json')
    if file_sha(a.bundle/'model.pt')!=design['source']:raise ValueError('source')
    model,_=load_bundle(a.bundle,'cpu');m=market_for(a.parent,spec,model.recipe)
    with np.load(a.signals) as d:sig=d['source'].copy()
    old=read(a.panel/'PANEL.json')['origins'];targets={r['timestamp']:r for r in old}
    records=[];tested=0
    for _,interval in INTERVALS:
        e=environment(m,read(a.parent/'RISK.json'),spec['design']['horizon'],interval,model.effective['gamma'])
        e.reset()
        while not e.finished:
            ix=int(e.rows[e.bar]);t=int(m.inputs.row_ts_ns[ix])
            if t in targets:
                before=state_view(e);origin=targets[t];tested+=1
                for row in origin['rows']:
                    side=row['side']
                    if not side:continue
                    opening=1 if side==1 else 2;closing=3-opening
                    b=fork_env(e)
                    for act in ((opening,0) if row['tag'].endswith('old') else (0,opening)):b.step(act)
                    if int(b.rows[b.bar])!=row['row'] or b.engine.pos.direction!=side:raise ValueError('mismatch')
                    v=exit_values(b)
                    # Hold wins economic ties; clairvoyant exit schedule permits no re-entry.
                    close=v[0]>max(v[1:])+1e-8
                    y=row['y'];reg=max(y)-y[closing if close else 0]
                    isolated=[no_reentry(b,a,sig) for a in [0,closing]]
                    choose=isolated[1]>isolated[0]+1e-8
                    matched=max(y)-y[closing if choose else 0]
                    records.append(dict(timestamp=t,year=origin['year'],tag=row['tag'],side=side,
                        oracle_close=close,best_exit_delay=int(np.argmax(v)),max_position_advantage=v[0]-max(v[1:]),
                        account_regret_usd=reg,material=reg>row['entry_cost'],regret_R=reg/row['risk'],
                        no_reentry_close=choose,no_reentry_values=isolated,
                        no_reentry_material=matched>row['entry_cost'],no_reentry_account_regret=matched))
                if state_view(e)!=before:raise ValueError('mutated source')
            e.step(int(sig[ix]))
    if tested!=len(targets):raise ValueError('missing origin')
    write(a.panel/'EXIT_GATE.json',dict(panel_sha256=file_sha(a.panel/'PANEL.json'),records=records))
    for side in [1,-1]:
        rr=[r for r in records if r['year']==2023 and r['side']==side]
        print(side,len(rr),'close',sum(r['oracle_close'] for r in rr),'material misses',sum(r['material'] for r in rr),
            'mean regret USD',np.mean([r['account_regret_usd'] for r in rr]))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('panel','parent','bundle','signals'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
