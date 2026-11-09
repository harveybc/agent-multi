"""Paired TRAIN action calibration, two CPU shards then sequential GPU arms."""
import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
import torch

from app.action_ranking_contrast import (select_anchors, target_loss, ranking_metrics, differentiable_q,
    replication_design,shard_owner,calibration_target)
from app.actor_learning_diagnostics import clear_optimizer_memory
from app.critic_refinement import sync_targets
from app.decision_attribution import continuation
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.policy_return_calibration import policy_signals
from app.rl_subset import digest, support_hash
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment
from tools.attribute_changed_decisions import trace_check
from tools.audit_fixed_actor_critics import verify_collection, covering_schedule, collect_full_replay
from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import authenticate, market_for, assess_full

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(7,19,43,73)
ARMS=('control','chosen_stochastic','paired_stochastic','paired_greedy')
SOURCES=('tools/run_action_ranking_contrast.py','app/action_ranking_contrast.py',
    'app/decision_attribution.py','tools/attribute_changed_decisions.py',
    'app/three_signal_learning.py','app/policy_return_calibration.py',
    'agent_plugins/three_signal_sac.py','tools/diagnose_warmstart_drift.py')


def seal(root,identity,started):
    write(root/'DONE.json',dict(identity=identity,finished_at=now(),seconds=time.monotonic()-started,
        files={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()
            and p.name not in ('worker.lock','PROGRESS.json','FAILED.json') and p!=root/'DONE.json'}))


def prepare(args):
    spec=authenticate(args.source,args.source_runtime)
    probe=verify_collection(args.probe);attribution=verify_collection(args.attribution)
    pc=read(args.probe/'CONFIG.json');ac=read(args.attribution/'CONFIG.json')
    if ac['probe']!=probe['identity'] or ac['source']!=spec['identity'] or pc['source_identity']!=spec['identity']:
        raise ValueError('RANKING_PARENT_CHAIN')
    config=dict(source_identity=spec['identity'],probe_identity=probe['identity'],
        attribution_identity=attribution['identity'],**replication_design(getattr(args,'replica','original')),
        updates=4096,batch_size=128,actor_updates=[1,8,32],per_stratum=32,fit_per_stratum=24,
        sources={n:file_sha(ROOT/n) for n in SOURCES})
    config['identity']=digest(config)
    market=market_for(args.source,spec)
    if (support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']
            or set(market.recipe['split']['periods'])!={'TRAIN','VALIDATION'}):
        raise ValueError('RANKING_SUPPORT')
    source_sha=file_sha(args.source/'runs/seed_7/bundles/epoch_0000/model.pt')
    if source_sha!=pc['source_sha256']:raise ValueError('RANKING_SOURCE_WEIGHT')
    surfaces=np.load(args.attribution/'SURFACES.npz')
    logits=surfaces['reference'];alpha=read(args.attribution/'SURFACES.json')['alpha']
    return spec,market,config,logits,alpha


def ensure_config(root,config):
    if (root/'CONFIG.json').exists() and read(root/'CONFIG.json')!=config:
        raise ValueError('RANKING_CONFIG_CHANGED')
    write(root/'CONFIG.json',config)


def progress(root,start,stage,**kw):
    write(root/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-start,**kw))


def panel(args):
    start=time.monotonic();configure(7)
    spec,market,config,logits,alpha=prepare(args)
    out=args.output/f'shard_{args.shard}';out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ensure_config(out,config)
        if (out/'DONE.json').exists():verify_collection(out);return
        trace=read(args.probe/'reference/TRAIN/ACTIONS.json')
        cohort=select_anchors(trace,32,24,config['anchor_seed'])
        if {r['year'] for r in cohort}!={2020,2021,2022,2023}:
            raise ValueError('RANKING_FOUR_TRAIN_YEARS')
        indices=np.searchsorted(market.inputs.row_ts_ns,[r['timestamp'] for r in trace])
        if not np.array_equal(market.inputs.row_ts_ns[indices],[r['timestamp'] for r in trace]):
            raise ValueError('RANKING_ROW_MAP')
        signals=prepared_signals(logits)[0]
        if not np.array_equal(signals[indices],[r['signal'] for r in trace]):
            raise ValueError('RANKING_REFERENCE_ACTIONS')
        chosen,_=policy_signals(logits,config['chosen_action_seed'])
        for j,r in enumerate(cohort):
            r['id']=j;r['row']=int(indices[r['trace_index']]);r['chosen_action']=int(chosen[0][r['row']])
        write(out/'COHORT.json',cohort)
        selected={r['trace_index']:r for r in cohort if shard_owner(r['id'],config['shard_weights'])==args.shard}
        tapes={s:policy_signals(logits,s) for s in config['seeds']}
        risk=read(args.source/'RISK.json')
        env=environment(market,risk,spec['design']['horizon'],market.recipe['split']['periods']['TRAIN'],spec['effective']['gamma'])
        env.reset();reference=read(args.probe/'reference/TRAIN/METRICS.json')['net_profit_usd'];done=0
        for i,t in enumerate(trace):
            if i in selected:
                r=selected[i];path=out/f"rows/{r['id']:04d}.json"
                progress(out,start,'COUNTERFACTUAL_PANEL',completed=done,total=len(selected))
                if path.exists():
                    record=read(path)
                    if (record['identity']!=config['identity'] or record['anchor']!=r
                            or record['sha256']!=digest(record['results'])):
                        raise ValueError('RANKING_CACHED_ROW')
                else:
                    result={'greedy':[], 'stochastic':{str(s):[] for s in config['seeds']}}
                    for a in range(3):
                        result['greedy'].append(continuation(env,signals,a))
                        for s,(prepared,probs) in tapes.items():
                            result['stochastic'][str(s)].append(continuation(env,prepared[0],a,probs,alpha))
                    if abs(result['greedy'][t['signal']]['net_usd']-reference)>1e-6:
                        raise ValueError('RANKING_REFERENCE_SUFFIX')
                    write(path,dict(identity=config['identity'],anchor=r,results=result,sha256=digest(result)))
                done+=1
            trace_check(env,t,t['signal'])
        if abs(env.engine.balance-env.engine.cfg.capital-reference)>1e-8:
            raise ValueError('RANKING_REFERENCE_ACCOUNT')
        write(out/'RESULT.json',dict(identity=config['identity'],shard=args.shard,completed=done,
            cohort_sha256=digest(cohort),year_counts={str(y):sum(r['year']==y for r in cohort) for y in range(2020,2024)}))
        seal(out,config['identity'],start);progress(out,start,'COMPLETE',completed=done,total=len(selected))


def load_panel(root,identity,seeds=SEEDS,weights=(1,1)):
    rows=[];cohort=None
    for shard in range(2):
        p=root/f'shard_{shard}';receipt=verify_collection(p)
        if receipt['identity']!=identity:raise ValueError('RANKING_SHARD_IDENTITY')
        c=read(p/'COHORT.json')
        if cohort is not None and cohort!=c:raise ValueError('RANKING_COHORT_MISMATCH')
        cohort=c
        expected={r['id'] for r in cohort if shard_owner(r['id'],weights)==shard};actual=set()
        for f in sorted((p/'rows').glob('*.json')):
            row=read(f);idx=row['anchor']['id']
            if (idx in actual or idx not in expected or row['anchor']!=cohort[idx]
                    or row['identity']!=identity or digest(row['results'])!=row['sha256']):
                raise ValueError('RANKING_SHARD_ROW')
            actual.add(idx);rows.append(row)
        if actual!=expected:raise ValueError('RANKING_SHARD_COVERAGE')
    rows.sort(key=lambda x:x['anchor']['id'])
    if len(rows)!=len(cohort):raise ValueError('RANKING_PANEL_COVERAGE')
    greedy=np.array([[v['reward_return'] for v in r['results']['greedy']] for r in rows])
    stochastic=np.array([[[v['soft_return'] for v in r['results']['stochastic'][str(s)]] for s in seeds] for r in rows])
    plain=np.array([[[v['reward_return'] for v in r['results']['stochastic'][str(s)]] for s in seeds] for r in rows])
    if not all(np.isfinite(x).all() for x in (greedy,stochastic,plain)):
        raise ValueError('RANKING_LABEL_FINITE')
    return cohort,greedy,stochastic,plain


def panel_metrics(model,features,cohort,greedy,stochastic,plain=None):
    with torch.no_grad():
        q=torch.cat([model.q_surfaces(features[i:i+128]).min(0).values for i in range(0,len(features),128)]).cpu().numpy()
    result={}
    for role in ('fit','monitor'):
        groups={'all':np.array([r['role']==role for r in cohort])}
        groups.update({str(p):np.array([r['role']==role and r['position']==p for r in cohort]) for p in (-1,0,1)})
        result[role]={}
        for group,mask in groups.items():
            if not mask.any():raise ValueError('RANKING_EMPTY_DIAGNOSTIC_GROUP')
            result[role][group]=dict(greedy=ranking_metrics(q[mask],greedy[mask]),
                stochastic=ranking_metrics(q[mask],stochastic.mean(1)[mask]))
            if plain is not None:
                result[role][group]['stochastic_reward_only']=ranking_metrics(q[mask],plain.mean(1)[mask])
    return result


def fit(args):
    start=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=args.expected_device_name:
        raise ValueError('RANKING_GPU_IDENTITY')
    spec,market,config,logits,_=prepare(args);ensure_config(args.output,config)
    out=args.output/'fit';out.mkdir(exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'DONE.json').exists():verify_collection(out);return
        cohort,greedy,stochastic,plain=load_panel(args.output,config['identity'],config['seeds'],config['shard_weights'])
        write(out/'PANEL.json',dict(anchors=len(cohort),fit=sum(r['role']=='fit' for r in cohort),
            monitor=sum(r['role']=='monitor' for r in cohort),cohort_sha256=digest(cohort),
            mean_future_entropy_contribution=float((stochastic-plain).mean()),
            monte_carlo_standard_deviation_mean=float(stochastic.std(1,ddof=1).mean()),
            independent_market_holdout=False,train_only=True))
        np.savez(out/'TARGETS.npz',greedy=greedy,stochastic=stochastic,stochastic_reward_only=plain)
        cal=verify_collection(args.calibration)
        if cal['identity']!=read(args.probe/'CONFIG.json')['calibration_identity']:
            raise ValueError('RANKING_FULL_REPLAY_PARENT')
        folder=args.calibration/'epoch_0000/stochastic_7/TRAIN'
        risk=read(args.source/'RISK.json');design=spec['design']
        env=environment(market,risk,design['horizon'],market.recipe['split']['periods']['TRAIN'],spec['effective']['gamma'])
        replay,receipt=collect_full_replay(env,read(folder/'ACTIONS.json'))
        if len(replay)!=read(args.source/'PREFLIGHT.json')['periods']['TRAIN']['decision_count']:
            raise ValueError('RANKING_FULL_TRAIN_REPLAY')
        write(out/'REPLAY.json',receipt)
        actor_schedule=covering_schedule(len(replay),4096,128,config['update_seed'])
        fit_ix=np.array([i for i,r in enumerate(cohort) if r['role']=='fit'])
        schedule=fit_ix[covering_schedule(len(fit_ix),4096,128,config['update_seed'])]
        write(out/'SCHEDULE.json',dict(critic_sha256=digest(schedule.tolist()),actor_sha256=digest(actor_schedule.tolist())))
        features=torch.tensor(market.inputs.windows[[r['row'] for r in cohort]],device='cuda')
        chosen=torch.tensor([r['chosen_action'] for r in cohort],device='cuda',dtype=torch.int64)
        bundle=args.source/'runs/seed_7/bundles/epoch_0000';results={}
        for arm in config['arms']:
            lane=out/arm;lane.mkdir(exist_ok=True)
            if (lane/'DONE.json').exists():
                if verify_collection(lane)['identity']!=config['identity']:raise ValueError('RANKING_ARM_IDENTITY')
                results[arm]=read(lane/'RESULT.json');continue
            model,_=load_bundle(bundle,'cuda');model.load(args.probe/'fresh/CALIBRATED.pt')
            set_arm(model,'critic_only');clear_optimizer_memory(model.optimizer);reset_learning_rng(config['update_seed'])
            frozen=frozen_state(model);initial_head={n:p.detach().clone() for n,p in model.loss.named_parameters() if p.requires_grad}
            target=torch.tensor(calibration_target(arm,greedy,stochastic,plain),device='cuda',dtype=torch.float32)
            points=[]
            for step in range((0 if arm=='control' else 4096)+1):
                if step:
                    ix=schedule[step-1];model.optimizer.zero_grad(set_to_none=True)
                    q=differentiable_q(model,features[ix])
                    loss=target_loss(q,target[ix],chosen[ix] if arm=='chosen_stochastic' else None)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.loss.parameters(),10.,error_if_nonfinite=True)
                    model.optimizer.step()
                if step in (0,64,256,1024,2048,4096):
                    verify_frozen_state(model,frozen)
                    moved=sum(not torch.equal(p,initial_head[n]) for n,p in model.loss.named_parameters() if n in initial_head)
                    if step and not moved:raise ValueError('RANKING_CRITIC_DID_NOT_MOVE')
                    point=dict(updates=step,head_tensors_changed=moved,metrics=panel_metrics(model,features,cohort,greedy,stochastic,plain))
                    points.append(point);write(lane/'CALIBRATION.json',points)
                    progress(out,start,'CALIBRATE',arm=arm,updates=step)
            sync_targets(model);model.save(lane/'CALIBRATED.pt')
            # Native actor updates are paired; no correction-fit optimizer history enters them.
            set_arm(model,'actor_only');clear_optimizer_memory(model.optimizer);reset_learning_rng(config['update_seed'])
            frozen=frozen_state(model);accounts=[]
            reference_logits,_=emit(model,market.inputs)
            if not np.array_equal(reference_logits[np.isfinite(logits).all(1)],logits[np.isfinite(logits).all(1)]):
                raise ValueError('RANKING_CALIBRATION_CHANGED_ACTOR')
            if arm=='control':
                zero=assess_full(model,market,risk,design)
                for period in zero:account_equal(zero[period],read(args.probe/'REFERENCE.json')[period])
                write(lane/'EPOCH_ZERO.json',dict(metrics=zero,source_exact=True))
            for step in range(1,33):
                model.update(replay[actor_schedule[step-1]].to('cuda'))
                if step in (1,8,32):
                    verify_frozen_state(model,frozen);progress(out,start,'ACTOR_ACCOUNTS',arm=arm,updates=step)
                    metrics=assess_full(model,market,risk,design,save=lane/f'actor_{step:04d}',diagnose=True)
                    if arm=='control' and config['update_seed']==7:
                        expected=read(args.probe/'fresh/RESULT.json')['accounts']
                        previous=next(x for x in expected if x['updates']==step)['metrics']
                        for period in metrics:account_equal(metrics[period],previous[period])
                    model.save(lane/f'ACTOR_{step:04d}.pt')
                    accounts.append(dict(updates=step,metrics=metrics,
                        eligible=all(m['orders_per_week']>2 for m in metrics.values()),model_promoted=False))
                    write(lane/'ACTOR_PROGRESS.json',accounts)
            results[arm]=dict(identity=config['identity'],arm=arm,calibration=points,accounts=accounts,model_promoted=False)
            write(lane/'RESULT.json',results[arm]);seal(lane,config['identity'],start);del model
        if file_sha(bundle/'model.pt')!=read(args.probe/'CONFIG.json')['source_sha256']:
            raise ValueError('RANKING_SOURCE_MUTATED')
        write(out/'RESULT.json',dict(identity=config['identity'],arms=results,
            original_seed7_endpoint_reproduced=True if config['update_seed']==7 else None,control_epoch_zero_exact=True,
            control_reproduced=True if config['update_seed']==7 else None,replica=config['replica'],
            no_test=True,fit_train_only=True,model_promoted=False))
        seal(out,config['identity'],start);progress(out,start,'COMPLETE')


def shared_args(args):
    return sum(([f'--{n.replace("_","-")}',str(getattr(args,n))]
        for n in ('source','source_runtime','probe','attribution','calibration','output')),[])+['--replica',getattr(args,'replica','original')]


def queue(args):
    start=time.monotonic();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        python=args.worker_python;common=shared_args(args)
        local=[python,'-u','-m','tools.run_action_ranking_contrast','panel',*common,'--shard','1']
        remote=['env',f'PYTHONPATH={args.worker_runtime}','CUDA_VISIBLE_DEVICES=',
            'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',python,'-u','-m','tools.run_action_ranking_contrast',
            'panel',*common,'--shard','0']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        progress(args.output,start,'CPU_SHARDS')
        with (args.output/'LOCAL.log').open('a') as log1,(args.output/'REMOTE.log').open('a') as log0:
            a=subprocess.Popen(local,cwd=ROOT,env=env,stdout=log1,stderr=subprocess.STDOUT)
            b=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',args.worker_host,
                shlex.join(remote)],stdout=log0,stderr=subprocess.STDOUT)
            while a.poll() is None or b.poll() is None:
                progress(args.output,start,'CPU_SHARDS',local_returncode=a.poll(),remote_returncode=b.poll())
                time.sleep(10)
            if a.returncode or b.returncode:raise ValueError(f'RANKING_SHARD_EXIT:{a.returncode}:{b.returncode}')
        subprocess.run(['rsync','-a',str(args.output/'shard_1')+'/',
            f'{args.worker_host}:{args.output}/shard_1/'],check=True)
        progress(args.output,start,'GPU_CORRECTIONS')
        command=['env',f'PYTHONPATH={args.worker_runtime}',f'CUDA_VISIBLE_DEVICES={args.gpu_id}',
            'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',python,'-u','-m','tools.run_action_ranking_contrast',
            'fit',*common,'--expected-device-name',args.expected_device_name]
        with (args.output/'GPU.log').open('a') as log:
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',args.worker_host,
                shlex.join(command)],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['rsync','-a',f'{args.worker_host}:{args.output}/shard_0/',str(args.output/'shard_0')+'/'],check=True)
        subprocess.run(['rsync','-a',f'{args.worker_host}:{args.output}/fit/',str(args.output/'fit')+'/'],check=True)
        receipt=verify_collection(args.output/'fit')
        write(args.output/'QUEUE_DONE.json',dict(identity=receipt['identity'],finished_at=now(),
            seconds=time.monotonic()-start,fit_done_sha256=file_sha(args.output/'fit/DONE.json'),
            verified_files=len(receipt['files']),model_promoted=False))
        progress(args.output,start,'COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('panel','fit','queue'))
    for n in ('source','source-runtime','probe','attribution','calibration','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--shard',type=int,choices=(0,1));p.add_argument('--expected-device-name',default='NVIDIA GeForce RTX 5090')
    p.add_argument('--replica',choices=('original','r19','r43'),default='original')
    p.add_argument('--worker-host');p.add_argument('--worker-runtime');p.add_argument('--worker-python');p.add_argument('--gpu-id')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=='panel' and a.shard is None:p.error('--shard required')
    if a.stage=='queue' and not all((a.worker_host,a.worker_runtime,a.worker_python,a.gpu_id)):
        p.error('queue worker identity required')
    try:{'panel':panel,'fit':fit,'queue':queue}[a.stage](a)
    except Exception as exc:
        write(a.output/f'FAILED_{a.stage}_{a.shard}.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
