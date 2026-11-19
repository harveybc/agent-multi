"""Execute the preregistered offline 379 gate. All timestamps are UTC."""
import argparse
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from app.close_surface import coordinates, fit, scores
from app.decision_attribution import fork_env
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment
from docs.work_plan.generalization.role_conflict378 import value
from tools.diagnose_critic_coverage_drift import INTERVALS
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import configure,read,write
from env_plugins.discrete_curve_native import HOUR

def basis(env):
    x=env.last_observation['features']
    if x.shape!=(13,71) or not np.isfinite(x).all():
        raise ValueError(('market window',x.shape))
    return np.r_[x[-1], x.mean(0)].astype(float)

def query(env):
    e,b,p=env.engine,env.bar,env.engine.pos
    now=int(e.v[b])+HOUR
    quote=e.quote_price(b,e.c[b],-p.direction,now)
    return coordinates(p.direction,quote,p.sl,p.tp,p.entry_price,(int(e.ff[p.seg])-now)/HOUR)

def collect(env,sig,t):
    states=[]
    for side,opening,closing in [(1,1,2),(-1,2,1)]:
        for age,seq in [('old',(opening,0)),('new',(0,opening))]:
            b=fork_env(env)
            for a in seq:
                if not b.finished:b.step(a)
            p=b.engine.pos
            if b.finished or p is None:continue
            if p.direction!=side:raise ValueError('direction')
            row=int(b.rows[b.bar])
            if b.inputs.period[row]!='TRAIN':raise ValueError('split')
            hold=value(b,0,sig)
            close=value(b,closing,sig)
            risk=abs(p.entry_price-p.sl)*abs(p.units)
            states.append(dict(t=t,side=side,age=age,row=row,q=query(b).tolist(),
                risk=risk,fee=sum(p.entry_cost.values()),hold=hold,close=close,y=(close-hold)/risk))
    if len({r['row'] for r in states})>1:raise ValueError('unpaired market')
    return states

def replay(env,model,kind,sig):
    while not env.finished:
        p=env.engine.pos
        a=int(sig[env.rows[env.bar]])
        if p and kind!='source':
            score=scores(model[p.direction],basis(env)[None,:],query(env)[None,:])[kind=='surface'][0]
            a=(2 if p.direction==1 else 1) if score>0 else 0
        env.step(a)
    e=env.engine
    net=e.balance-e.cfg.capital
    if abs(net-sum(r['net_usd'] for r in e.trades))>1e-6:raise ValueError('cash')
    eq=np.array([v for _,v in env.equity_trace])
    dd=np.max(1-eq/np.maximum.accumulate(eq))
    return dict(net_usd=net,return_pct=100*net/e.cfg.capital,drawdown_pct=100*dd,
        trades=len(e.trades),buy=sum(int(r['direction']==1) for r in e.trades),
        winners=sum(int(r['net_usd']>0) for r in e.trades))

def analyze(rows):
    d=pd.DataFrame(rows)
    d['material']=False
    for _,g in d.groupby(['t','side']):
        if len(g)==2:
            diff=g.close-g.hold
            if min(diff.clip(lower=0).mean(),(-diff).clip(lower=0).mean())>g.fee.mean():
                d.loc[g.index,'material']=True
    stats={}
    for k,g in [('long',d[d.side==1]),('short',d[d.side==-1]),('material',d[d.material]),('all',d)]:
        stats[k]={}
        for name in ('scalar','surface','always_hold','always_close'):
            pick=(g[name]>0) if name in ('scalar','surface') else np.full(len(g),name=='always_close')
            regret=np.maximum(g.y,0)-pick*g.y
            stats[k][name]=dict(n=len(g),closes=int(np.sum(pick)),harmful=int(np.sum(pick & (g.y<0))),
                regret_R=float(regret.mean()) if len(g) else None,
                regret_usd=float((regret*g.risk).mean()) if len(g) else None)
    delta=((d.scalar>0).astype(int)-(d.surface>0).astype(int))*d.y
    week=pd.to_datetime(d.t,unit='ns',utc=True).dt.strftime('%G-%V')
    groups=delta.groupby(week).agg(['sum','count']).to_numpy()
    samples=groups[np.random.default_rng(379).integers(len(groups),size=(4000,len(groups)))].sum(1)
    return stats,dict(weeks=len(groups),delta_R=float(delta.mean()),
        ci95=np.quantile(samples[:,0]/samples[:,1],[.025,.975]).tolist())

def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    configure(379)
    old=read(a.old/'CONFIG.json');verify_done(a.old,old['identity']);verify_done(a.cache,old['diagnostic'])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    if file_sha(a.bundle/'model.pt')!=old['source_sha256']:raise ValueError('source')
    source,_=load_bundle(a.bundle,'cpu');m=market_for(a.parent,spec,source.recipe)
    if support_hash(m.inputs)!=old['support']:raise ValueError('support')
    risk=read(a.parent/'RISK.json')
    with np.load(a.cache/'SIGNALS.npz') as f:sig=f['source'].copy()
    def make(interval):
        e=environment(m,risk,spec['design']['horizon'],interval,source.effective['gamma']);e.reset();return e
    oldts={r['anchor']['timestamp'] for r in read(a.old/'RESULT.json')['origins']}
    oldts.update(r['timestamp'] for r in read(a.previous/'PANEL.json')['origins'])
    rng=np.random.default_rng(379);candidates=defaultdict(list)
    for _,interval in INTERVALS:
        e=make(interval); decisions=[];flats=[]
        while not e.finished:
            row=int(e.rows[e.bar]); t=int(m.inputs.row_ts_ns[row]);decisions.append(t)
            if e.engine.pos is None:flats.append((len(decisions)-1,t))
            e.step(int(sig[row]))
        old_idx=np.where(np.isin(decisions,list(oldts)))[0]
        for i,t in flats:
            if not np.any(abs(old_idx-i)<=26):
                candidates[pd.Timestamp(t).year].append(t)
    wanted=sorted(int(t) for ts in candidates.values() for t in rng.choice(ts,min(160,len(ts)),False))
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'DESIGN.json',dict(origins=wanted,source=old['source_sha256'],support=old['support'],
        script=file_sha(__file__),protocol=file_sha(Path(__file__).parents[1]/'379_CLOSE_SURFACE_GATE.md'),
        lookup=file_sha(Path('app/close_surface.py')),old_panel=file_sha(a.previous/'PANEL.json')))
    wanted=set(wanted);model={};reserved=[];account={};saturation={}
    for part,interval in INTERVALS:
        rows=[];xs={};e=make(interval)
        while not e.finished:
            row=int(e.rows[e.bar]);t=int(m.inputs.row_ts_ns[row])
            if t in wanted:
                for r in collect(e,sig,t):
                    x=m.inputs.observation(r['row'])['features']
                    if x.shape!=(13,71):raise ValueError(x.shape)
                    xs[r['row']]=np.r_[x[-1],x.mean(0)].astype(float);rows.append(r)
            e.step(int(sig[row]))
        for side in (1,-1):
            g=[r for r in rows if r['side']==side]
            x=np.array([xs[r['row']] for r in g]);q=np.array([r['q'] for r in g]);y=np.array([r['y'] for r in g])
            if part=='calibration_fit':
                head=fit(x,q,y)
                path=a.out/f'HEAD_{side}.npz';np.savez(path,**head)
                with np.load(path) as f:model[side]=dict(f)
                if not np.array_equal(scores(head,x,q),scores(model[side],x,q)):raise ValueError('reload')
            else:
                bounds=model[side]['bounds']
                saturation[str(side)]=np.mean((q<bounds[0])|(q>bounds[1]),0).tolist()
                for r,s,v in zip(g,*scores(model[side],x,q)):
                    r.update(scalar=float(s),surface=float(v))
                reserved.extend(g)
        write(a.out/(part+'.json'),rows)
        print(part,len(rows),flush=True)
    stats,paired=analyze(reserved)
    for kind in ('source','scalar','surface'):
        account[kind]=replay(make(INTERVALS[-1][1]),model,kind,sig)
    passed=bool(all(stats[s]['surface']['regret_R']<stats[s]['scalar']['regret_R'] for s in ('long','short'))
        and paired['ci95'][1]<0 and stats['material']['scalar']['n']>0
        and stats['material']['surface']['regret_R']<stats['material']['scalar']['regret_R']
        and account['surface']['net_usd']>=account['scalar']['net_usd'])
    result=dict(stats=stats,paired=paired,accounts=account,saturation=saturation,advance=passed)
    write(a.out/'RESULT.json',result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('old','parent','bundle','cache','previous','out'):p.add_argument('--'+key,required=True,type=Path)
    run(p.parse_args())
