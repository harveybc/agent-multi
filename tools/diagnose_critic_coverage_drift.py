"""Finite TRAIN-only frozen-Q coverage and same-state policy drift queue."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from app.critic_coverage_drift import QUOTAS, choose_panel, drift_metrics, nearest_fit_rms
from app.critic_fit_probe import fit_metrics
from app.decision_attribution import continuation, state_view
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import emit, environment
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_stratified_critic import state_kind

INTERVALS = (('calibration_fit',['2020-01-01','2023-01-01']),
             ('later_train',['2023-01-01','2024-01-01']))


def run(a):
    started = time.monotonic(); configure(312)
    spec = read(a.parent/'SPEC.json'); verify_files(a.parent,spec['files'])
    cr = read(a.calibration/'CONFIG.json'); verify_done(a.calibration,cr['identity'])
    hc = read(a.capacity/'CONFIG.json'); verify_done(a.capacity,hc['identity'])
    ref = read(a.bundle.parent/'RESULT.json'); verify_done(a.bundle.parent,ref['identity'])
    sha = file_sha(a.bundle/'model.pt')
    if sha!=hc['source_sha256'] or sha!=cr['source_sha256'] or hc['calibration_identity']!=cr['identity']:
        raise ValueError('COVERAGE_SOURCE_LINEAGE')
    source,_ = load_bundle(a.bundle,'cpu'); market = market_for(a.parent,spec,source.recipe)
    if support_hash(market.inputs)!=cr['support_sha256']:
        raise ValueError('COVERAGE_SUPPORT')
    old = read(a.calibration/'panel/RECORDS.json')
    for r in old:
        if (not market.inputs.valid[r['row']] or market.inputs.period[r['row']]!='TRAIN'
                or int(market.inputs.row_ts_ns[r['row']])!=r['timestamp']):
            raise ValueError('COVERAGE_OLD_PANEL')
    cfg = dict(schema='frozen_critic_coverage_drift.v1',source_sha256=sha,
        calibration=cr['identity'],capacity=hc['identity'],support_sha256=cr['support_sha256'],
        quotas=QUOTAS,seed=312,intervals=INTERVALS,source_data='historical_mixed_COT_not_live_qualified',
        target='plain_discounted_native_reward_same_state_first_action_full_interval',
        optimizer_steps=0,validation_inference=False,test_read=False,model_promoted=False,
        code={p:file_sha(Path(__file__).resolve().parents[1]/p) for p in (
            'app/critic_coverage_drift.py','tools/diagnose_critic_coverage_drift.py',
            'app/decision_attribution.py','app/critic_fit_probe.py','tools/run_stratified_critic.py')})
    # JSON round trip keeps interval tuple/list identity consistent on resume.
    cfg['intervals'] = [[name,interval] for name,interval in INTERVALS]
    cfg['identity'] = digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:
        raise ValueError('COVERAGE_COLLISION')
    write(a.output/'CONFIG.json',cfg)
    if (a.output/'DONE.json').exists(): verify_done(a.output,cfg['identity']); return
    def progress(stage,**kw):
        value=dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw)
        write(a.output/'PROGRESS.json',value); print(value,flush=True)
    rows=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
    risk=read(a.parent/'RISK.json')
    def make(interval):
        e=environment(market,risk,spec['design']['horizon'],interval,source.effective['gamma'])
        e.reset();return e
    cache=a.output/'cache'
    if (cache/'DONE.json').exists():
        verify_done(cache,cfg['identity'])
        with np.load(cache/'SIGNALS.npz',allow_pickle=False) as data:
            policies={k:data[k].copy() for k in data.files}
    else:
        progress('INFERENCE_TRAIN_ONLY')
        z,_=emit(source,market.inputs,indices=rows)
        policies={'source':prepared_signals(z)[0]}
        for seed in (19,43):
            for width in (8,16):
                name=f'seed_{seed}_head_{width}'
                model,_=load_bundle(a.capacity/name/'bundle','cpu')
                if model.recipe!=source.recipe:raise ValueError('COVERAGE_CANDIDATE_RECIPE')
                calibrated,_=load_bundle(a.bundle,'cpu');calibrated.load(a.calibration/f'critic_{seed}/MODEL.pt')
                for key,value in calibrated.loss.state_dict().items():
                    if not key.startswith('actor_') or '.head.' not in key:
                        if not torch.equal(value,model.loss.state_dict()[key]):
                            raise ValueError('COVERAGE_FROZEN_PARAMETERS')
                z,_=emit(model,market.inputs,indices=rows)
                policies[name]=prepared_signals(z)[0]
                del model,calibrated
        cache.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache/'SIGNALS.npz',**policies)
        seal(cache,cfg['identity'])
    signals=policies['source']
    design=a.output/'design'
    if (design/'DONE.json').exists():
        verify_done(design,cfg['identity']);panel=read(design/'PANEL.json')
    else:
        progress('TRACE_AND_FREEZE_PANEL');trace=[];cash={}
        for segment,interval in INTERVALS:
            env=make(interval)
            while not env.finished:
                row=int(env.rows[env.bar]);ts=int(market.inputs.row_ts_ns[row])
                if market.inputs.period[row]!='TRAIN':raise ValueError('COVERAGE_NONTRAIN')
                trace.append(dict(row=row,timestamp=ts,year=pd.Timestamp(ts,tz='UTC').year,
                    kind=state_kind(env),segment=segment,state_sha256=digest(state_view(env)),
                    window_start=int(market.inputs.slot_ts_ns[row,0])))
                env.step(int(signals[row]))
            cash[segment]=float(env.engine.balance-env.engine.cfg.capital)
        selected,counts=choose_panel(trace,{r['timestamp'] for r in old})
        if not selected:raise ValueError('COVERAGE_EMPTY_PANEL')
        fitrows=[r['row'] for r in old if r['role']=='fit']
        distances=nearest_fit_rms(market.inputs.windows[[r['row'] for r in selected]],market.inputs.windows[fitrows])
        for r,d in zip(selected,distances):r['nearest_fit_window_rms']=float(d)
        panel=dict(anchors=selected,counts=counts,reference_cash=cash,
            trace_sha256=digest(trace),old_anchor_count=len(old),fit_anchor_count=len(fitrows))
        write(design/'PANEL.json',panel);seal(design,cfg['identity'])
    selected=panel['anchors'];wanted={r['timestamp']:r for r in selected}
    qdir=a.output/'critics'
    if (qdir/'DONE.json').exists():
        verify_done(qdir,cfg['identity'])
        with np.load(qdir/'Q.npz') as data:qpred={k:data[k].copy() for k in data.files}
    else:
        progress('CRITIC_PREDICTIONS');qpred={};ix=np.array([r['row'] for r in old+selected])
        for seed in (19,43):
            model,_=load_bundle(a.bundle,'cpu');model.load(a.calibration/f'critic_{seed}/MODEL.pt')
            _,q=emit(model,market.inputs,indices=ix,twins=True);qpred[str(seed)]=q[:,ix]
        qdir.mkdir(parents=True,exist_ok=True);np.savez_compressed(qdir/'Q.npz',**qpred);seal(qdir,cfg['identity'])
    records=[];durations=[]
    for segment,interval in INTERVALS:
        env=make(interval)
        while not env.finished:
            row=int(env.rows[env.bar]);ts=int(market.inputs.row_ts_ns[row])
            if ts in wanted:
                anchor=wanted[ts];out=a.output/'anchors'/str(row)
                if digest(state_view(env))!=anchor['state_sha256']:raise ValueError('COVERAGE_STATE_REPLAY')
                if (out/'DONE.json').exists():
                    verify_done(out,cfg['identity']);record=read(out/'RESULT.json')
                    if record['anchor']!=anchor:raise ValueError('COVERAGE_ANCHOR_CACHE')
                else:
                    begin=time.monotonic();outcomes={}
                    for name,tape in policies.items():
                        progress('CONTINUATIONS',completed=len(records),total=len(selected),row=row,policy=name,
                            remaining_seconds_estimate=float(np.median(durations[-8:])*(len(selected)-len(records))) if durations else None)
                        outcomes[name]=[continuation(env,tape,action) for action in range(3)]
                        if name!='source' and [v['first_effect'] for v in outcomes[name]]!=[v['first_effect'] for v in outcomes['source']]:
                            raise ValueError('COVERAGE_FIRST_ACTION_MISMATCH')
                    if abs(outcomes['source'][int(signals[row])]['net_usd']-panel['reference_cash'][segment])>1e-6:
                        raise ValueError('COVERAGE_REFERENCE_CASH')
                    if not records:
                        if [continuation(env,signals,i) for i in range(3)]!=outcomes['source']:
                            raise ValueError('COVERAGE_REPEAT_WITNESS')
                    record=dict(anchor=anchor,outcomes=outcomes,seconds=time.monotonic()-begin,
                                parent_state_unchanged=digest(state_view(env))==anchor['state_sha256'])
                    if not record['parent_state_unchanged']:raise ValueError('COVERAGE_PARENT_CHANGED')
                    write(out/'RESULT.json',record);seal(out,cfg['identity']);durations.append(record['seconds'])
                records.append(record)
            env.step(int(signals[row]))
    if [r['anchor'] for r in records]!=selected:raise ValueError('COVERAGE_FINAL_ANCHORS')
    def targets(values):return np.array([[v['reward_return'] for v in rr] for rr in values])
    old_y=targets([r['outcomes'] for r in old])
    y={k:targets([r['outcomes'][k] for r in records]) for k in policies}
    summary={}
    for seed in (19,43):
        q=qpred[str(seed)];old_q,new_q=q[:,:len(old)],q[:,len(old):]
        old_groups={role:fit_metrics(old_q[:,[i for i,r in enumerate(old) if r['role']==role]],
            old_y[[i for i,r in enumerate(old) if r['role']==role]]) for role in ('fit','monitor')}
        groups={}
        for kind in QUOTAS:
            for year in ('all',2020,2021,2022,2023):
                ix=np.array([i for i,r in enumerate(selected) if r['kind']==kind and (year=='all' or r['year']==year)],int)
                if not len(ix):continue
                values=dict(n=len(ix),source_critic=fit_metrics(new_q[:,ix],y['source'][ix]),candidates={})
                for width in (8,16):
                    name=f'seed_{seed}_head_{width}';actions=policies[name][[selected[i]['row'] for i in ix]]
                    values['candidates'][name]=dict(critic=fit_metrics(new_q[:,ix],y[name][ix]),
                        drift=drift_metrics(y['source'][ix],y[name][ix],actions))
                groups[f'{year}:{kind}']=values
        summary[str(seed)]=dict(old_labels=old_groups,new_groups=groups)
    if file_sha(a.bundle/'model.pt')!=sha:raise ValueError('COVERAGE_SOURCE_MUTATED')
    result=dict(identity=cfg['identity'],anchors=len(records),native_suffixes=len(records)*3*len(policies),
        summary=summary,counts=panel['counts'],seconds=time.monotonic()-started,model_promoted=False,
        historical_mixed_cot=True,validation_inference=False,test_read=False,optimizer_steps=0,
        first_effects_identical=True,parent_states_unchanged=True,
        limitation='Same source states, not candidate own-state visitation; sampled strata not population average.')
    write(a.output/'RESULT.json',result);seal(a.output,cfg['identity'])
    progress('COMPLETE',anchors=len(records),native_suffixes=result['native_suffixes'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','calibration','capacity','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)
