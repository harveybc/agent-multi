"""Full-period frozen greedy/stochastic accounts and SAC return calibration."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest,support_hash
from app.structural_refit import load_bundle,build_model
from app.three_signal_learning import emit,environment,evaluate
from app.policy_return_calibration import policy_signals,discounted_returns,calibration
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_structural_sac_refit import authenticate,market_for
from tools.run_refit_mask_pass import account_equal
from tools.diagnose_warmstart_drift import state_value_drift

SEEDS=(7,19,43)
ROOT=Path(__file__).resolve().parents[1]


def metric_text(value,digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def account_calibration(folder,inputs,q,p,gamma,alpha,stochastic):
    trace=read(folder/'ACTIONS.json')
    ts=np.array([r['timestamp'] for r in trace],dtype=np.int64)
    rows=np.searchsorted(inputs.row_ts_ns,ts)
    if ((rows>=len(inputs.row_ts_ns)).any() or
            not np.array_equal(inputs.row_ts_ns[rows],ts) or not inputs.valid[rows].all()):
        raise ValueError('CALIBRATION_TRACE_SUPPORT')
    actions=np.array([r['signal'] for r in trace],dtype=np.int64)
    positions=np.array([r['position'] for r in trace],dtype=np.int64)
    reward=np.array([r['reward'] for r in trace]);duration=np.array([r['duration'] for r in trace])
    chosen=p[rows,actions]
    if stochastic and (chosen<=0).any():raise ValueError('ZERO_PROBABILITY_DRAW')
    surprise=-np.log(np.maximum(chosen,1e-30))
    plain=discounted_returns(reward,duration,surprise,gamma,0.)
    soft=discounted_returns(reward,duration,surprise,gamma,alpha) if stochastic else plain
    qa=q[:,rows,actions];minimum=qa.min(0)
    quarter=pd.to_datetime(ts,utc=True).tz_localize(None).to_period('Q').astype(str).to_numpy()
    strata=[]
    for category,values in [('signal',actions),('position_before_signal',positions),('quarter',quarter)]:
        for value in np.unique(values):
            ix=values==value
            strata.append(dict(category=category,value=str(value),**calibration(minimum[ix],soft[ix])))
    np.savez_compressed(folder/'RETURN_TRACE.npz',timestamp=ts,signal=actions,position=positions,
        reward=reward,duration=duration,action_probability=chosen,plain_return=plain,
        soft_return=soft,online_q=qa)
    result=dict(rows=len(trace),units='percentage-point reward units relative to initial capital',
        scope=('sampled on-policy soft-return calibration' if stochastic else
               'greedy-policy mismatch diagnostic, not on-policy calibration for stochastic Q'),
        alpha=alpha,gamma=gamma,min_q=calibration(minimum,soft),
        twins=[calibration(v,soft) for v in qa],ordinary_return_mean=float(plain.mean()),
        strata=strata)
    write(folder/'CALIBRATION.json',result)
    return result


def run(args):
    start=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError('CALIBRATION_GPU_REQUIRED')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    if set(market.recipe['split']['periods'])!={'TRAIN','VALIDATION'}:
        raise ValueError('CALIBRATION_PERIODS')
    pre=read(args.source/'PREFLIGHT.json')
    if support_hash(market.inputs)!=pre['support_sha256']:raise ValueError('CALIBRATION_SUPPORT')
    baseline_path=args.source/'runs/seed_7/bundles/epoch_0000'
    initial=read(args.source/'runs/seed_7/EPOCH_ZERO.json')
    snapshots={}
    for path in args.snapshot:
        manifest=read(path/'MANIFEST.json');state=read(path/'STATE.json')
        if manifest['identity']!=spec['identity'] or state['identity']!=spec['identity']:
            raise ValueError('CALIBRATION_SNAPSHOT_IDENTITY')
        for name in ('STATE.json','model.pt'):
            if file_sha(path/name)!=manifest['files'][name]:raise ValueError('CALIBRATION_SNAPSHOT_HASH')
        epoch=state['epoch']
        if epoch in snapshots or epoch==0:raise ValueError('CALIBRATION_DUPLICATE_EPOCH')
        snapshots[epoch]=(path,state,manifest)
    if set(snapshots)!={6,40}:raise ValueError('CALIBRATION_REQUIRED_EPOCHS')
    config=dict(source_identity=spec['identity'],seed_policy=list(SEEDS),periods=market.recipe['split']['periods'],
        support_sha256=pre['support_sha256'],initial_sha256=file_sha(baseline_path/'model.pt'),
        snapshot_sha256={str(e):m['files']['model.pt'] for e,(_,_,m) in snapshots.items()},
        sources={n:file_sha(ROOT/n) for n in ('tools/audit_frozen_policy_returns.py',
            'app/policy_return_calibration.py','app/three_signal_learning.py','tools/diagnose_warmstart_drift.py')},
        scope='frozen inference only; no fitting, promotion or TEST; three sampling seeds, not trained-model seeds')
    config['identity']=digest(config)
    if (args.output/'CONFIG.json').exists() and read(args.output/'CONFIG.json')!=config:
        raise ValueError('CALIBRATION_CONFIG_CHANGED')
    write(args.output/'CONFIG.json',config)
    if (args.output/'DONE.json').exists():return
    def progress(stage,**fields):
        write(args.output/'PROGRESS.json',dict(identity=config['identity'],stage=stage,
            seconds=time.monotonic()-start,updated_at=now(),**fields))
    risk=read(args.source/'RISK.json');all_results=[]
    for epoch in (0,6,40):
        label=f'epoch_{epoch:04d}';progress('FROZEN_INFERENCE',checkpoint=label)
        if epoch==0:
            model,_=load_bundle(baseline_path,'cuda');expected=initial['metrics']
        else:
            path,state,_=snapshots[epoch]
            model=build_model(market.recipe,spec['effective'],device='cuda');model.load(path/'model.pt')
            expected=state['history'][-1]['metrics']
        before={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
        logits,q=emit(model,market.inputs,twins=True)
        alpha=float(model.loss.log_alpha.detach().exp());folder=args.output/label;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'SURFACES.npz',timestamp=market.inputs.row_ts_ns,
            valid=market.inputs.valid,logits=logits,online_q=q)
        for seed in (None,*SEEDS):
            policy='greedy' if seed is None else f'stochastic_{seed}'
            prepared,p=policy_signals(logits,seed)
            action_sha=hashlib.sha256(prepared[0].tobytes()).hexdigest()
            for period,interval in market.recipe['split']['periods'].items():
                progress('NATIVE_ACCOUNT',checkpoint=label,policy=policy,period=period,completed=len(all_results))
                out=folder/policy/period
                env=environment(market,risk,spec['design']['horizon'],interval,model.effective['gamma'])
                metrics=evaluate(env,logits,save=out,prepared=prepared)
                if seed is None:account_equal(metrics,expected[period])
                calibration_result=account_calibration(out,market.inputs,q,p,model.effective['gamma'],alpha,seed is not None)
                if calibration_result['rows']!=pre['periods'][period]['decision_count']:
                    raise ValueError('CALIBRATION_DECISION_COUNT')
                all_results.append(dict(checkpoint=epoch,policy=policy,period=period,metrics=metrics,
                    calibration=calibration_result['min_q'],actions_sha256=action_sha,
                    scope=calibration_result['scope']))
                write(args.output/'ACCOUNTS.json',all_results)
        drift=state_value_drift(before,model.loss.state_dict())
        if drift['changed']:raise ValueError('CALIBRATION_MODEL_MUTATED')
        write(folder/'UNCHANGED.json',dict(exact_values=True,device_moves=drift['device_moves']))
        del model
    lines=['# Frozen policy account comparison','',config['scope'],'']
    for period in market.recipe['split']['periods']:
        lines.extend([f'## {period}','',
            '| Epoch | Policy | Return % | Sharpe | Orders | Win % | Max DD % | Total costs USD | Q bias |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
        for r in all_results:
            if r['period']!=period:continue
            m=r['metrics']
            win=None if m['win_rate'] is None else 100*m['win_rate']
            lines.append(f"| {r['checkpoint']} | {r['policy']} | {m['return_pct']:.4f} | {metric_text(m['sharpe'])} | {m['trades']} | {metric_text(win,2)} | {m['max_drawdown_pct']:.4f} | {sum(m['costs'].values()):.2f} | {r['calibration']['bias']:.5f} |")
        lines.append('')
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    write(args.output/'DONE.json',dict(identity=config['identity'],accounts=len(all_results),
        seconds=time.monotonic()-start,finished_at=now(),
        files={str(p.relative_to(args.output)):file_sha(p) for p in args.output.rglob('*')
               if p.is_file() and p.name not in ('worker.lock','PROGRESS.json','DONE.json')}))
    progress('COMPLETE',accounts=len(all_results))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-runtime','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,action='append',required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:
            write(args.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),updated_at=now()))
            raise


if __name__=='__main__':main()
