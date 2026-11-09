"""Frozen same-state and chronological native account attribution, no training."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from app.decision_attribution import continuation, fork_env, reconcile, state_view
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.policy_return_calibration import policy_signals
from app.rl_subset import digest, support_hash
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_structural_sac_refit import authenticate, market_for

ROOT=Path(__file__).resolve().parents[1]
ARMS=('retained','fresh')
UPDATES=(1,8,32)
SEEDS=(7,19,43)


def online_q(model):
    return {k:v.detach().cpu().clone() for k,v in model.loss.state_dict().items()
            if k.startswith('qvalue_network_params.')}


def trace_check(env, row, action):
    if int(env.engine.v[env.bar])+3600000000000!=row['timestamp']:
        raise ValueError('ATTRIBUTION_TIMESTAMP')
    if (0 if env.engine.pos is None else env.engine.pos.direction)!=row['position']:
        raise ValueError('ATTRIBUTION_POSITION')
    _,reward,_,duration=env.step(int(action))
    if (abs(reward-row['reward'])>1e-12 or duration!=row['duration']
            or env.last_command!=row['command']):
        raise ValueError('ATTRIBUTION_NATIVE_REPLAY')


def run(args):
    start=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    spec=authenticate(args.source,args.source_runtime)
    receipt=verify_collection(args.probe)
    market=market_for(args.source,spec)
    if set(market.recipe['split']['periods'])!={'TRAIN','VALIDATION'}:
        raise ValueError('ATTRIBUTION_PERIODS')
    if support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']:
        raise ValueError('ATTRIBUTION_SUPPORT')
    bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    source_sha=file_sha(bundle/'model.pt')
    if source_sha!=read(args.probe/'CONFIG.json')['source_sha256']:
        raise ValueError('ATTRIBUTION_SOURCE')
    config=dict(source=spec['identity'],probe=receipt['identity'],source_sha256=source_sha,
        arms=ARMS,updates=UPDATES,stochastic_seeds=SEEDS,
        sources={n:file_sha(ROOT/n) for n in ('tools/attribute_changed_decisions.py',
            'app/decision_attribution.py','app/three_signal_learning.py',
            'app/policy_return_calibration.py','env_plugins/three_signal_native.py')})
    config['identity']=digest(config)
    # JSON round-trip makes resumed tuple/list identity comparisons exact.
    if (args.output/'CONFIG.json').exists():
        if read(args.output/'CONFIG.json')['identity']!=config['identity']:
            raise ValueError('ATTRIBUTION_CONFIG_CHANGED')
    write(args.output/'CONFIG.json',config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):
        write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),
            seconds=time.monotonic()-start,identity=config['identity'],**kw))
    surfaces=args.output/'SURFACES.npz'
    if not (args.output/'SURFACES.json').exists():
        if not torch.cuda.is_available() or torch.cuda.get_device_name()!=args.expected_device_name:
            raise ValueError('ATTRIBUTION_GPU_IDENTITY')
        progress('FROZEN_GPU_SURFACES')
        model,_=load_bundle(bundle,'cuda');arrays={}
        arrays['reference'],_=emit(model,market.inputs)
        alpha=float(model.loss.log_alpha.exp().detach().cpu())
        for arm in ARMS:
            model.load(args.probe/arm/'CALIBRATED.pt');qstate=online_q(model)
            if not qstate:raise ValueError('ATTRIBUTION_EMPTY_CRITIC')
            _,arrays[arm+'_q']=emit(model,market.inputs,twins=True)
            for step in UPDATES:
                model.load(args.probe/arm/f'ACTOR_{step:04d}.pt')
                check=online_q(model)
                if check.keys()!=qstate.keys() or any(not torch.equal(v,check[k]) for k,v in qstate.items()):
                    raise ValueError('ATTRIBUTION_CRITIC_CHANGED')
                if float(model.loss.log_alpha.exp().detach().cpu())!=alpha:
                    raise ValueError('ATTRIBUTION_ALPHA_CHANGED')
                arrays[f'{arm}_{step}'],_=emit(model,market.inputs)
        del model;torch.cuda.empty_cache()
        np.savez(surfaces,**arrays)
        write(args.output/'SURFACES.json',dict(sha256=file_sha(surfaces),alpha=alpha,
            online_critics_unchanged=True,device=torch.cuda.get_device_name()))
    surface_receipt=read(args.output/'SURFACES.json')
    if file_sha(surfaces)!=surface_receipt['sha256']:raise ValueError('ATTRIBUTION_SURFACE_HASH')
    z=np.load(surfaces);alpha=surface_receipt['alpha']
    signals={k:prepared_signals(z[k])[0] for k in z.files if not k.endswith('_q')}
    stochastic={seed:policy_signals(z['reference'],seed) for seed in SEEDS}
    risk=read(args.source/'RISK.json');design=spec['design'];summary=[]
    # Shorter validation account first; both periods are mandatory for completion.
    for period in ('VALIDATION','TRAIN'):
        interval=market.recipe['split']['periods'][period]
        def make_env():
            env=environment(market,risk,design['horizon'],interval,spec['effective']['gamma'])
            env.reset();return env
        out=args.output/period
        if (out/'DONE.json').exists():
            verify_collection(out);summary.extend(read(out/'RESULT.json'));continue
        progress('AUTHENTICATE_NATIVE_ACCOUNTS',period=period)
        ref_trace=read(args.probe/'reference'/period/'ACTIONS.json')
        ref_metrics=read(args.probe/'reference'/period/'METRICS.json')
        ts=np.array([r['timestamp'] for r in ref_trace],dtype=np.int64)
        indices=np.searchsorted(market.inputs.row_ts_ns,ts)
        if not np.array_equal(market.inputs.row_ts_ns[indices],ts):
            raise ValueError('ATTRIBUTION_ROW_MAP')
        if not np.array_equal(signals['reference'][indices],[r['signal'] for r in ref_trace]):
            raise ValueError('ATTRIBUTION_REFERENCE_ARGMAX')
        traces={};changed={};union=set()
        for arm in ARMS:
            for step in UPDATES:
                key=f'{arm}_{step}'
                traces[key]=read(args.probe/arm/f'actor_{step:04d}'/period/'ACTIONS.json')
                if (not np.array_equal(ts,[r['timestamp'] for r in traces[key]]) or
                    not np.array_equal(signals[key][indices],[r['signal'] for r in traces[key]])):
                    raise ValueError('ATTRIBUTION_CANDIDATE_ARGMAX')
                changed[key]=set(np.flatnonzero(signals[key][indices]!=signals['reference'][indices]).tolist())
                union.update(changed[key])
        env=make_env();snapshots={}
        for i,r in enumerate(ref_trace):
            if i in union:snapshots[i]=fork_env(env)
            trace_check(env,r,r['signal'])
        reference=float(env.engine.balance-env.engine.cfg.capital)
        if (abs(reference-ref_metrics['net_profit_usd'])>1e-8 or not env.finished
                or len(env.engine.trades)!=ref_metrics['trades'] or env.engine.counts!=ref_metrics['native_counts']):
            raise ValueError('ATTRIBUTION_REFERENCE_ACCOUNT')
        cache={};records=[]
        for arm in ARMS:
            for step in UPDATES:
                key=f'{arm}_{step}';path=out/(key+'.json')
                # Recompute accounts on resume; immutable per-event cache avoids expensive suffix repeats.
                env=make_env();prefix=reference;events=[]
                for i,r in enumerate(traces[key]):
                    row=int(indices[i]);new=int(r['signal']);old=int(ref_trace[i]['signal'])
                    if i in changed[key]:
                        cache_key=f'{i}_{old}_{new}';cache_path=out/'isolated'/(cache_key+'.json')
                        progress('COUNTERFACTUALS',period=period,arm=arm,updates=step,
                            event=len(events)+1,events=len(changed[key]),completed_cells=len(summary)+len(records))
                        if cache_key not in cache:
                            if cache_path.exists():
                                saved=read(cache_path)
                                if saved['identity']!=config['identity'] or saved['sha256']!=digest(saved['result']):
                                    raise ValueError('ATTRIBUTION_CACHE_IDENTITY')
                                cache[cache_key]=saved['result']
                            else:
                                baseline=continuation(snapshots[i],signals['reference'],old)
                                intervention=continuation(snapshots[i],signals['reference'],new)
                                if abs(baseline['net_usd']-reference)>1e-6:
                                    raise ValueError('ATTRIBUTION_REFERENCE_SUFFIX')
                                samples=[]
                                for seed,(prepared,probs) in stochastic.items():
                                    a=continuation(snapshots[i],prepared[0],old,probs,alpha)
                                    b=continuation(snapshots[i],prepared[0],new,probs,alpha)
                                    samples.append(dict(seed=seed,old=a,new=b,
                                        soft_delta=b['soft_return']-a['soft_return'],net_delta=b['net_usd']-a['net_usd']))
                                cache[cache_key]=dict(state=state_view(snapshots[i]),old=baseline,new=intervention,
                                    delta_usd=intervention['net_usd']-baseline['net_usd'],stochastic=samples)
                                write(cache_path,dict(identity=config['identity'],result=cache[cache_key],
                                    sha256=digest(cache[cache_key])))
                        isolated=cache[cache_key]
                        a=continuation(env,signals['reference'],old)
                        b=continuation(env,signals['reference'],new)
                        if abs(a['net_usd']-prefix)>1e-6:
                            raise ValueError('ATTRIBUTION_PREFIX_CHAIN')
                        contribution=b['net_usd']-a['net_usd'];prefix=b['net_usd']
                        q=z[arm+'_q'][:,row,:];qmin=q.min(axis=0)
                        events.append(dict(index=i,timestamp=int(ts[i]),utc=pd.Timestamp(ts[i],tz='UTC').isoformat(),
                            old_signal=old,new_signal=new,reference_logits=z['reference'][row].tolist(),
                            candidate_logits=z[key][row].tolist(),twins=q.tolist(),
                            q_gap=float(qmin[new]-qmin[old]),isolated=isolated,
                            prefix_state=state_view(env),prefix_old=a,prefix_new=b,
                            chronological_contribution_usd=contribution))
                        write(out/(key+'_PROGRESS.json'),events)
                    trace_check(env,r,new)
                candidate=float(env.engine.balance-env.engine.cfg.capital)
                metrics=read(args.probe/arm/f'actor_{step:04d}'/period/'METRICS.json')
                if (abs(candidate-metrics['net_profit_usd'])>1e-8 or not env.finished
                        or len(env.engine.trades)!=metrics['trades'] or env.engine.counts!=metrics['native_counts']):
                    raise ValueError('ATTRIBUTION_CANDIDATE_ACCOUNT')
                recon=reconcile([e['chronological_contribution_usd'] for e in events],reference,candidate)
                cell=dict(period=period,arm=arm,updates=step,changed_signals=len(events),
                    reference_net_usd=reference,candidate_net_usd=candidate,reconciliation=recon,events=events)
                write(path,cell)
                records.append({k:v for k,v in cell.items() if k!='events'})
                write(args.output/'SUMMARY_PROGRESS.json',summary+records)
        write(out/'RESULT.json',records)
        write(out/'DONE.json',dict(identity=config['identity'],finished_at=now(),
            files={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*.json') if p.name!='DONE.json'}))
        summary.extend(records)
    if file_sha(bundle/'model.pt')!=source_sha:raise ValueError('ATTRIBUTION_SOURCE_MUTATED')
    write(args.output/'RESULT.json',dict(identity=config['identity'],accounts=summary,model_promoted=False,
        no_training=True,no_test=True,stochastic_draws=3,stochastic_not_exact_expectation=True))
    write(args.output/'DONE.json',dict(identity=config['identity'],finished_at=now(),seconds=time.monotonic()-start,
        files={str(p.relative_to(args.output)):file_sha(p) for p in args.output.rglob('*') if p.is_file()
            and p.name not in ('worker.lock','PROGRESS.json') and p!=args.output/'DONE.json'}))
    progress('COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','source-runtime','probe','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-device-name',default='NVIDIA GeForce RTX 5090')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise


if __name__=='__main__':main()
