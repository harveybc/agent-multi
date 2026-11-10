"""Native quarterly TRAIN panels followed by paired frozen-actor critic fits."""
import argparse
from collections import Counter
import csv
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from app.action_ranking_contrast import advantage_loss
from app.actor_learning_diagnostics import clear_optimizer_memory
from app.critic_fit_probe import classify_anchor, probe_q, set_scope
from app.decision_attribution import continuation, fork_env, state_view
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest, support_hash
from app.selected_warm_start import reset_learning_rng
from app.stratified_critic import QUOTAS, grouped_metrics, select_strata, validate_support
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment
from env_plugins.discrete_curve_native import HOUR
from tools.audit_fixed_actor_critics import covering_schedule, verify_collection
from tools.diagnose_warmstart_drift import frozen_state, verify_frozen_state
from tools.run_action_ranking_contrast import ensure_config, seal
from tools.run_actor_preservation import collect
from tools.run_critic_fit_probe import train_logits
from tools.run_discrete_curve_probe import configure, now, read, write
from tools.run_structural_sac_refit import authenticate, market_for

ROOT=Path(__file__).resolve().parents[1]
POINTS=(0,64,256,1024,2048,4096)
BOUNDARY=pd.Timestamp('2023-01-01',tz='UTC').value


def state_kind(env):
    if env.engine.pos is not None:
        return 'open_long' if env.engine.pos.direction==1 else 'open_short'
    before=state_view(env);accepted=[]
    for action in (1,2):
        fork=fork_env(env);entries=fork.engine.counts['entries'];fork.step(action)
        accepted.append(fork.engine.counts['entries']>entries)
    if before!=state_view(env):raise ValueError('STRATIFIED_ADMISSION_MUTATION')
    return 'flat_both_allowed' if all(accepted) else 'flat_restricted'


def quarter_panel(args,spec,market,signals,year,quarter,config,progress):
    name=f'{year}Q{quarter}';root=args.output/'panel'/name;root.mkdir(parents=True,exist_ok=True)
    if (root/'DONE.json').exists():
        if verify_collection(root)['identity']!=config['identity']:raise ValueError('STRATIFIED_PANEL_IDENTITY')
        return read(root/'RESULT.json')
    started=time.monotonic();start=pd.Timestamp(year=year,month=3*quarter-2,day=1)
    interval=[start.isoformat(),(start+pd.DateOffset(months=3)).isoformat()]
    role='fit' if year<2023 else 'monitor'
    risk=read(args.source/'RISK.json')
    def new_env():
        e=environment(market,risk,spec['design']['horizon'],interval,spec['effective']['gamma'])
        e.reset();return e
    env=new_env();terminal=int(env.engine.v[env.engine.last])+HOUR
    window=market.recipe['tensor']['window'];lookback=(window['observations']-1)*window['stride_rows']
    candidates=[];trace=[];excluded_context=0
    while not env.finished:
        row=int(env.rows[env.bar]);timestamp=int(market.inputs.row_ts_ns[row])
        if market.inputs.period[row]!='TRAIN':raise ValueError('STRATIFIED_NONTRAIN_ROW')
        action=int(signals[row]);kind=state_kind(env)
        window_start=int(market.inputs.row_ts_ns[row-lookback]) if row>=lookback else -1
        anchor=dict(id=f'{name}:{len(trace)}',trace_index=len(trace),row=row,timestamp=timestamp,
            year=year,quarter=quarter,role=role,kind=kind,terminal=terminal,
            window_start=window_start,position=0 if env.engine.pos is None else env.engine.pos.direction)
        trace.append(dict(row=row,signal=action,state=state_view(env)))
        if window_start<0 or (role=='monitor' and window_start<BOUNDARY):excluded_context+=1
        else:candidates.append(anchor)
        env.step(action)
    reference_cash=float(env.engine.balance-env.engine.cfg.capital)
    if abs(reference_cash-sum(t['net_usd'] for t in env.engine.trades))>1e-6:
        raise ValueError('STRATIFIED_REFERENCE_CASH')
    selected,counts=select_strata(candidates,19+year*4+quarter)
    anchors=[candidates[i] for i in selected];chosen={r['trace_index']:r for r in anchors}
    write(root/'DESIGN.json',dict(interval=interval,role=role,terminal=terminal,counts=counts,
        anchors=anchors,excluded_context=excluded_context,reference_cash=reference_cash,
        reference_trace_sha256=digest(trace)))
    env=new_env();records=[];witness_error=0.
    for i,t in enumerate(trace):
        if state_view(env)!=t['state']:raise ValueError('STRATIFIED_REFERENCE_STATE')
        if i in chosen:
            anchor=chosen[i];path=root/'rows'/f'{i:05d}.json'
            if path.exists():
                record=read(path)
                if (record['identity']!=config['identity'] or record['anchor']!=anchor
                        or record['sha256']!=digest(record['results'])):
                    raise ValueError('STRATIFIED_CACHED_ROW')
            else:
                results=dict(greedy=[continuation(env,signals,a) for a in range(3)])
                record=dict(identity=config['identity'],anchor=anchor,results=results,sha256=digest(results))
                write(path,record)
            c=classify_anchor(record)
            if c['kind']!=anchor['kind']:raise ValueError('STRATIFIED_ADMISSION_MISMATCH')
            if abs(record['results']['greedy'][t['signal']]['net_usd']-reference_cash)>1e-6:
                raise ValueError('STRATIFIED_REFERENCE_BRANCH')
            # Rerun the first selected anchor through a separately cloned path.
            if not records:
                for action in range(3):
                    value=continuation(env,signals,action);expected=record['results']['greedy'][action]
                    for key in ('reward_return','net_usd'):
                        witness_error=max(witness_error,abs(value[key]-expected[key]))
                    if value['first_effect']!=expected['first_effect']:
                        raise ValueError('STRATIFIED_WITNESS_EFFECT')
                if witness_error>1e-8:raise ValueError('STRATIFIED_WITNESS_RETURN')
            records.append(record)
            progress('PANEL',quarter=name,completed=len(records),total=len(anchors))
        env.step(t['signal'])
    if float(env.engine.balance-env.engine.cfg.capital)!=reference_cash:
        raise ValueError('STRATIFIED_REFERENCE_REPLAY')
    result=dict(identity=config['identity'],records=records,counts=counts,interval=interval,
        witness_max_error=witness_error,reference_cash=reference_cash,excluded_context=excluded_context)
    write(root/'RESULT.json',result);seal(root,config['identity'],started);return result


def export_report(root,results):
    rows=[]
    for seed,arms in results.items():
        for arm,record in arms.items():
            for point in record['points']:
                for role,groups in point['metrics'].items():
                    for kind,twins in groups.items():
                        if twins is None:continue
                        for twin,m in twins.items():
                            rows.append(dict(seed=seed,arm=arm,updates=point['updates'],role=role,kind=kind,twin=twin,
                                **{k:m[k] for k in ('n','pairwise_accuracy','mean_regret','maximum_regret','gap_rmse','level_rmse')}))
    with (root/'CURVES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(role,kind) in zip(axes.flat,[(r,k) for r in ('fit','monitor') for k in tuple(QUOTAS)[:3]]):
        for seed,arms in results.items():
            for arm in arms:
                rr=[r for r in rows if r['seed']==seed and r['arm']==arm and r['role']==role
                    and r['kind']==kind and r['twin']=='minimum']
                ax.plot([r['updates'] for r in rr],[r['gap_rmse'] for r in rr],marker='.',label=f'{seed} {arm}')
        ax.set(title=f'{role}: {kind}',xlabel='Critic updates',ylabel='Action-gap RMSE')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'FIT_MONITOR.png',dpi=140);plt.close(fig)


def run(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    probe=verify_collection(args.probe);pc=read(args.probe/'CONFIG.json')
    bundle=args.source/'runs/seed_7/bundles/epoch_0000';source_sha=file_sha(bundle/'model.pt')
    if (pc['source_identity']!=spec['identity'] or source_sha!=pc['source_sha256']
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']):
        raise ValueError('STRATIFIED_SOURCE')
    if tuple(market.recipe['split']['periods']['TRAIN'])!=('2020-01-01','2024-01-01'):
        raise ValueError('STRATIFIED_TRAIN_CONTRACT')
    config=dict(source=spec['identity'],source_sha256=source_sha,probe=probe['identity'],
        calibrated_sha256=file_sha(args.probe/'fresh/CALIBRATED.pt'),quarters=16,quotas=QUOTAS,
        fit_years=[2020,2021,2022],monitor_years=[2023],panel_seed=19,seeds=[19,43],
        updates=4096,batch_size=128,scopes=['heads','critic_representation'],actor_updates=0,
        loss='pairwise_plus_mean',monitor_previously_used_by_source=True,model_promoted=False,
        sources={n:file_sha(ROOT/n) for n in ('tools/run_stratified_critic.py','app/stratified_critic.py',
            'app/critic_fit_probe.py','app/action_ranking_contrast.py','app/decision_attribution.py',
            'agent_plugins/three_signal_sac.py','env_plugins/three_signal_native.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):
        write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    progress('PREFLIGHT')
    model,_=load_bundle(bundle,'cpu')
    train=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
    logits,_=emit(model,market.inputs,indices=train);signals=prepared_signals(logits)[0];del model
    quarters=[]
    for year in range(2020,2024):
        for quarter in range(1,5):
            quarters.append(quarter_panel(args,spec,market,signals,year,quarter,config,progress))
            if args.stage=='canary':
                write(args.output/'CANARY.json',dict(identity=config['identity'],quarter=quarters[0]['counts'],
                    witness_max_error=quarters[0]['witness_max_error']));return
    records=[r for q in quarters for r in q['records']];anchors=[r['anchor'] for r in records]
    validate_support(anchors,BOUNDARY)
    if len({r['row'] for r in anchors})!=len(anchors):raise ValueError('STRATIFIED_DUPLICATE_ROW')
    for role in ('fit','monitor'):
        for kind in ('flat_both_allowed','open_long','open_short'):
            if sum(r['role']==role and r['kind']==kind for r in anchors)<32:
                raise ValueError('STRATIFIED_INSUFFICIENT_OPERATIONAL_COVERAGE')
    write(args.output/'PANEL.json',dict(identity=config['identity'],anchors=anchors,
        counts=dict(Counter(r['role']+':'+r['kind'] for r in anchors)),
        quarter_counts=[q['counts'] for q in quarters],native_suffixes=3*len(anchors),
        independent_holdout=False,actor_pretrained_on_monitor=True,
        max_fit_label_end=max(r['terminal'] for r in anchors if r['role']=='fit'),
        min_monitor_window_start=min(r['window_start'] for r in anchors if r['role']=='monitor')))
    if args.stage=='panel':return
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':
        raise ValueError('STRATIFIED_GPU_IDENTITY')
    features=torch.tensor(market.inputs.windows[[r['row'] for r in anchors]],device='cuda')
    targets=np.array([classify_anchor(r)['targets'] for r in records])
    target=torch.tensor(targets,device='cuda',dtype=torch.float32)
    fit_ix=np.array([i for i,r in enumerate(anchors) if r['role']=='fit'])
    reference,_=load_bundle(bundle,'cuda');parent=train_logits(reference,market.inputs.windows[train]);del reference
    results={}
    for seed in (19,43):
        results[str(seed)]={};schedule=fit_ix[covering_schedule(len(fit_ix),4096,128,seed)]
        for scope in ('heads','critic_representation'):
            lane=args.output/f'r{seed}'/scope;lane.mkdir(parents=True,exist_ok=True)
            if (lane/'DONE.json').exists():
                if verify_collection(lane)['identity']!=config['identity']:raise ValueError('STRATIFIED_ARM_IDENTITY')
                results[str(seed)][scope]=read(lane/'RESULT.json');continue
            model,_=load_bundle(bundle,'cuda');model.load(args.probe/'fresh/CALIBRATED.pt')
            set_scope(model,scope);clear_optimizer_memory(model.optimizer);reset_learning_rng(seed)
            frozen=frozen_state(model);initial={n:p.detach().clone() for n,p in model.loss.named_parameters() if p.requires_grad}
            write(lane/'EFFECTIVE.json',dict(identity=config['identity'],seed=seed,scope=scope,
                schedule_sha256=digest(schedule.tolist()),learning_rate=model.effective['learning_rate'],
                weight_decay=model.effective['weight_decay'],trainable={n:p.numel() for n,p in model.loss.named_parameters() if p.requires_grad}))
            points=[];losses=[]
            for step in range(4097):
                if step:
                    ix=schedule[step-1];model.optimizer.zero_grad(set_to_none=True)
                    q=probe_q(model,features[ix]);loss=advantage_loss(q,target[ix]);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.loss.parameters(),10.,error_if_nonfinite=True)
                    model.optimizer.step();losses.append(float(loss.detach()))
                if step in POINTS:
                    verify_frozen_state(model,frozen)
                    moved=[n for n,p in model.loss.named_parameters() if n in initial and not torch.equal(p,initial[n])]
                    if step and (not moved or (scope=='critic_representation' and not any('.encoder.' in n for n in moved))):
                        raise ValueError('STRATIFIED_PARAMETERS_NOT_MOVING')
                    with torch.no_grad():
                        q=torch.cat([probe_q(model,features[i:i+128]) for i in range(0,len(features),128)],dim=1).cpu().numpy()
                    points.append(dict(updates=step,changed_tensors=len(moved),metrics=grouped_metrics(q,targets,anchors)))
                    write(lane/'CURVE.json',points);progress('FIT',seed=seed,scope=scope,updates=step)
            if not np.array_equal(train_logits(model,market.inputs.windows[train]),parent):
                raise ValueError('STRATIFIED_ACTOR_CHANGED')
            model.save(lane/'DIAGNOSTIC.pt')
            record=dict(identity=config['identity'],points=points,losses=losses,
                full_train_actor_logits_exact=True,frozen_parameters_exact=True,model_promoted=False)
            write(lane/'RESULT.json',record);seal(lane,config['identity'],started)
            results[str(seed)][scope]=record;del model
    if (file_sha(bundle/'model.pt')!=source_sha
            or file_sha(args.probe/'fresh/CALIBRATED.pt')!=config['calibrated_sha256']):
        raise ValueError('STRATIFIED_SOURCE_MUTATED')
    export_report(args.output,results)
    write(args.output/'RESULT.json',dict(identity=config['identity'],arms=results,actor_updates=0,model_promoted=False))
    seal(args.output,config['identity'],started);progress('COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('run','canary','panel','collect'))
    for n in ('source','source-runtime','probe'):p.add_argument('--'+n,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-host')
    args=p.parse_args()
    if args.stage=='collect':
        if not args.worker_host:p.error('--worker-host required')
        collect(args);return
    if not all((args.source,args.source_runtime,args.probe)):p.error('source paths required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
