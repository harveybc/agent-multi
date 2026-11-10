"""Exact native attribution of the fixed paired-greedy endpoint, no fitting."""
import argparse
import fcntl
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd

from app.decision_attribution import continuation,fork_env,state_view,reconcile
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest,support_hash
from app.three_signal_learning import environment
from tools.attribute_changed_decisions import trace_check
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_discrete_curve_probe import read,write,configure,now
from tools.run_structural_sac_refit import authenticate,market_for


def account_check(env,metrics):
    net=float(env.engine.balance-env.engine.cfg.capital)
    if (not env.finished or abs(net-metrics['net_profit_usd'])>1e-8
            or len(env.engine.trades)!=metrics['trades'] or env.engine.counts!=metrics['native_counts']):
        raise ValueError('IMPROVEMENT_ACCOUNT')
    return net


def attribute_pair(make_env,row_timestamps,baseline,candidate,progress,calibration_signals=None,
                   soft_policies=None,alpha=0.,isolated_cache=None,cache_namespace=None):
    if isolated_cache is not None and not cache_namespace:
        raise ValueError('ATTRIBUTION_CACHE_NAMESPACE')
    if not np.isfinite(alpha) or alpha<0:raise ValueError('ATTRIBUTION_ALPHA')
    policies=soft_policies or {}
    for tape,probability in policies.values():
        if (np.asarray(tape).shape!=(len(row_timestamps),) or
                np.asarray(probability).shape!=(len(row_timestamps),3) or
                not np.isin(tape,[0,1,2]).all()):
            raise ValueError('ATTRIBUTION_SOFT_POLICY')
    if calibration_signals is not None:
        calibration_signals=np.asarray(calibration_signals)
        if (calibration_signals.shape!=(len(row_timestamps),)
                or not np.isin(calibration_signals,[0,1,2]).all()):
            raise ValueError('IMPROVEMENT_CALIBRATION_TAPE')
    traces=[read(p/'ACTIONS.json') for p in (baseline,candidate)]
    cache={} if isolated_cache is None else isolated_cache
    context=digest(dict(namespace=cache_namespace,trace=traces[0],alpha=alpha,
        policies={str(k):dict(tape=digest(np.asarray(v[0]).tolist()),
            probability_sha256=hashlib.sha256(np.asarray(v[1]).tobytes()).hexdigest())
                  for k,v in policies.items()}))
    metrics=[read(p/'METRICS.json') for p in (baseline,candidate)]
    ts=np.array([r['timestamp'] for r in traces[0]],dtype=np.int64)
    if not np.array_equal(ts,[r['timestamp'] for r in traces[1]]):raise ValueError('IMPROVEMENT_SUPPORT')
    ix=np.searchsorted(row_timestamps,ts)
    if (ix>=len(row_timestamps)).any() or not np.array_equal(row_timestamps[ix],ts):
        raise ValueError('IMPROVEMENT_SUPPORT')
    tapes=[]
    for trace in traces:
        tape=np.zeros(len(row_timestamps),dtype=np.int64);tape[ix]=[r['signal'] for r in trace];tapes.append(tape)
    changed=set(np.flatnonzero(tapes[0][ix]!=tapes[1][ix]).tolist())
    env=make_env();env.reset();snapshots={}
    for i,row in enumerate(traces[0]):
        if i in changed:snapshots[i]=fork_env(env)
        trace_check(env,row,row['signal'])
    reference=account_check(env,metrics[0]);previous=reference
    env=make_env();env.reset();events=[]
    for i,row in enumerate(traces[1]):
        if i in changed:
            old=int(traces[0][i]['signal']);new=int(row['signal'])
            key=(context,i,old,new)
            if key not in cache:
                cache[key]=dict(old=continuation(snapshots[i],tapes[0],old),
                    new=continuation(snapshots[i],tapes[0],new),
                    soft=[dict(seed=int(seed),old=continuation(snapshots[i],tape,old,p,alpha),
                               new=continuation(snapshots[i],tape,new,p,alpha))
                          for seed,(tape,p) in policies.items()])
            iso_old,iso_new=cache[key]['old'],cache[key]['new']
            before=continuation(env,tapes[0],old);after=continuation(env,tapes[0],new)
            if abs(iso_old['net_usd']-reference)>1e-6 or abs(before['net_usd']-previous)>1e-6:
                raise ValueError('IMPROVEMENT_CONTINUATION_CHAIN')
            contribution=after['net_usd']-before['net_usd'];previous=after['net_usd']
            extra=dict(isolated_state=state_view(snapshots[i]),soft_continuations=cache[key]['soft']) if policies else {}
            if calibration_signals is not None:
                extra=dict(isolated_state=state_view(snapshots[i]),
                    calibration_old=continuation(snapshots[i],calibration_signals,old),
                    calibration_new=continuation(snapshots[i],calibration_signals,new))
            events.append(dict(utc=pd.Timestamp(ts[i],tz='UTC').isoformat(),timestamp=int(ts[i]),
                old_signal=old,new_signal=new,state=state_view(env),isolated_old=iso_old,isolated_new=iso_new,
                prefix_old=before,prefix_new=after,contribution_usd=contribution,**extra))
            progress(events=len(events),total=len(changed))
        trace_check(env,row,row['signal'])
    new_net=account_check(env,metrics[1])
    recon=reconcile([e['contribution_usd'] for e in events],reference,new_net)
    return dict(reference_net_usd=reference,candidate_net_usd=new_net,delta_usd=new_net-reference,
        reference_trades=metrics[0]['trades'],candidate_trades=metrics[1]['trades'],
        reconciliation=recon,changed_signals=len(events),
        effective_changes=sum(abs(e['contribution_usd'])>1e-6 for e in events),events=events)


def run(args):
    started=time.monotonic();configure(7)
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    source_receipt=verify_collection(args.probe);ranking_receipt=verify_collection(args.ranking/'fit')
    parent=read(args.ranking/'CONFIG.json')
    if (parent['source_identity']!=spec['identity'] or parent['probe_identity']!=source_receipt['identity']
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']):
        raise ValueError('IMPROVEMENT_PARENT')
    config=dict(source=spec['identity'],ranking=ranking_receipt['identity'],probe=source_receipt['identity'],
        candidate='paired_greedy/actor_0032',references=['unchanged','control32'],periods=['TRAIN','VALIDATION'],
        sources={p:file_sha(Path(__file__).resolve().parents[1]/p) for p in
            ('tools/attribute_ranking_improvement.py','app/decision_attribution.py','tools/attribute_changed_decisions.py')})
    config['identity']=digest(config)
    if (args.output/'CONFIG.json').exists() and read(args.output/'CONFIG.json')!=config:
        raise ValueError('IMPROVEMENT_CONFIG')
    write(args.output/'CONFIG.json',config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    risk=read(args.source/'RISK.json');rows=[]
    for period in config['periods']:
        interval=market.recipe['split']['periods'][period]
        def make():return environment(market,risk,spec['design']['horizon'],interval,spec['effective']['gamma'])
        candidate=args.ranking/'fit/paired_greedy/actor_0032'/period
        for label,baseline in [('unchanged',args.probe/'reference'/period),('control32',args.ranking/'fit/control/actor_0032'/period)]:
            def progress(**kw):write(args.output/'PROGRESS.json',dict(period=period,reference=label,at=now(),**kw))
            value=attribute_pair(make,market.inputs.row_ts_ns,baseline,candidate,progress)
            write(args.output/f'{period}_{label}.json',value)
            rows.append(dict(period=period,reference=label,**{k:v for k,v in value.items() if k!='events'}))
    write(args.output/'RESULT.json',dict(identity=config['identity'],accounts=rows,no_training=True,model_promoted=False))
    write(args.output/'DONE.json',dict(identity=config['identity'],finished_at=now(),seconds=time.monotonic()-started,
        files={str(p.relative_to(args.output)):file_sha(p) for p in args.output.rglob('*.json')
            if p.name not in ('DONE.json','PROGRESS.json','FAILED.json')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-runtime','probe','ranking','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
