"""Paired TRAIN action-gap versus absolute-return fits, then anchored actors."""
import argparse
import csv
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.action_ranking_contrast import (advantage_loss, calibration_indices,
    differentiable_q, target_loss)
from app.actor_learning_diagnostics import clear_optimizer_memory
from app.actor_preservation import anchored_update, arm_config
from app.critic_refinement import sync_targets
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, support_hash
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment
from tools.audit_fixed_actor_critics import verify_collection, covering_schedule, collect_full_replay
from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
from tools.run_action_ranking_contrast import load_panel, panel_metrics, seal, ensure_config
from tools.run_actor_preservation import measure
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import authenticate, market_for

ROOT=Path(__file__).resolve().parents[1]
ARMS=('absolute','advantage')
CRITIC_POINTS=(0,64,256,1024,2048,4096)
ACTOR_POINTS=(32,256)


def report(root,results):
    rows=[]
    for replica,result in results.items():
        for arm,record in result['arms'].items():
            for point in record['points']:
                for period,m in point['metrics'].items():
                    row=dict(replica=replica,arm=arm,updates=point['updates'],period=period,
                             eligible=point['eligible'])
                    row.update({k:m[k] for k in ('return_pct','net_profit_usd','sharpe',
                        'max_drawdown_pct','trades','win_rate','orders_per_week')})
                    for side in ('buy','sell'):
                        row[side+'_trades']=m[side]['trades']
                        row[side+'_win_rate']=m[side]['wins']/m[side]['trades'] if m[side]['trades'] else None
                    row['early_closes']=m['close_reasons'].get('AGENT_CLOSE',0)
                    row.update(point['policy_drift'][period]);rows.append(row)
    with (root/'TRADING.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(replica,period) in zip(axes.flat,[(r,p) for r in ('r19','r43') for p in ('TRAIN','VALIDATION')]):
        for arm in ARMS:
            chosen=[r for r in rows if r['replica']==replica and r['period']==period and r['arm']==arm]
            ax.plot([r['updates'] for r in chosen],[r['return_pct'] for r in chosen],marker='.',label=arm)
        ax.axhline(results[replica]['reference']['metrics'][period]['return_pct'],color='black',linestyle='--',label='parent')
        ax.set(title=f'{replica} {period}',xlabel='Actor updates',ylabel='Account return (%)')
        ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'RETURNS.png',dpi=140);plt.close(fig)
    write(root/'RESULT.json',dict(replicas=results,fixed_endpoint=256,fit_train_only=True,
        independent_market_holdout=False,model_promoted=False))


def run(args):
    start=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':
        raise ValueError('ADVANTAGE_GPU_IDENTITY')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    probe=verify_collection(args.probe);cal=verify_collection(args.calibration)
    preservation=verify_collection(args.preservation);pc=read(args.probe/'CONFIG.json')
    if (support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']
            or set(market.recipe['split']['periods'])!={'TRAIN','VALIDATION'}
            or pc['source_identity']!=spec['identity'] or pc['calibration_identity']!=cal['identity']
            or read(args.preservation/'CONFIG.json')['source']!=spec['identity']):
        raise ValueError('ADVANTAGE_LINEAGE')
    bundle=args.source/'runs/seed_7/bundles/epoch_0000';source_sha=file_sha(bundle/'model.pt')
    if source_sha!=pc['source_sha256']:raise ValueError('ADVANTAGE_SOURCE_WEIGHT')
    parents={r:verify_collection(args.ranking/r/'fit') for r in ('r19','r43')}
    config=dict(source=spec['identity'],source_sha256=source_sha,probe=probe['identity'],
        calibration=cal['identity'],preservation=preservation['identity'],
        parents={r:v['identity'] for r,v in parents.items()},replicas=['r19','r43'],
        arms=list(ARMS),critic_updates=4096,actor_updates=256,batch_size=128,beta=1.0,
        advantage_mean_coefficient=1.0,actor_points=list(ACTOR_POINTS),
        sources={n:file_sha(ROOT/n) for n in ('tools/run_action_advantage_contrast.py',
            'app/action_ranking_contrast.py','app/actor_preservation.py','tools/run_actor_preservation.py',
            'agent_plugins/three_signal_sac.py','agent_plugins/discrete_curve_sac.py',
            'tools/diagnose_warmstart_drift.py','app/three_signal_learning.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):
        write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-start,**kw))
    progress('PREFLIGHT')
    risk=read(args.source/'RISK.json');design=spec['design']
    reference,_=load_bundle(bundle,'cuda');parent,_=emit(reference,market.inputs)
    env=environment(market,risk,design['horizon'],market.recipe['split']['periods']['TRAIN'],spec['effective']['gamma'])
    replay,receipt=collect_full_replay(env,read(args.calibration/'epoch_0000/stochastic_7/TRAIN/ACTIONS.json'))
    if len(replay)!=read(args.source/'PREFLIGHT.json')['periods']['TRAIN']['decision_count']:
        raise ValueError('ADVANTAGE_FULL_REPLAY')
    write(args.output/'REPLAY.json',receipt)
    initial=measure(reference,market,risk,design,parent,args.output/'reference')
    for period,m in initial['metrics'].items():account_equal(m,read(args.probe/'REFERENCE.json')[period])
    results={}
    for seed in (19,43):
        replica=f'r{seed}';root=args.ranking/replica;rc=read(root/'CONFIG.json')
        if (rc['identity']!=parents[replica]['identity'] or rc['update_seed']!=seed
                or rc['source_identity']!=spec['identity'] or rc['probe_identity']!=probe['identity']):
            raise ValueError('ADVANTAGE_PARENT_CONFIG')
        cohort,greedy,stochastic,plain=load_panel(root,rc['identity'],rc['seeds'],rc['shard_weights'])
        ix=calibration_indices(cohort,market.inputs.row_ts_ns,market.inputs.valid,market.inputs.period)
        schedule=ix[covering_schedule(len(ix),4096,128,seed)]
        actor_schedule=covering_schedule(len(replay),4096,128,seed)
        schedules=dict(critic_sha256=digest(schedule.tolist()),actor_sha256=digest(actor_schedule.tolist()))
        if schedules!=read(root/'fit/SCHEDULE.json'):raise ValueError('ADVANTAGE_PAIRED_SCHEDULE')
        directory=args.output/replica;directory.mkdir(exist_ok=True)
        write(directory/'PANEL.json',dict(cohort=cohort,fit=len(ix),monitor=len(cohort)-len(ix),
            schedules=schedules,targets_sha256=file_sha(root/'fit/TARGETS.npz'),
            independent_market_holdout=False))
        features=torch.tensor(market.inputs.windows[[r['row'] for r in cohort]],device='cuda')
        target=torch.tensor(greedy,device='cuda',dtype=torch.float32)
        results[replica]=dict(reference=initial,arms={})
        for arm in ARMS:
            lane=directory/arm;lane.mkdir(exist_ok=True)
            if (lane/'DONE.json').exists():
                if verify_collection(lane)['identity']!=config['identity']:raise ValueError('ADVANTAGE_RESUME')
                results[replica]['arms'][arm]=read(lane/'RESULT.json');continue
            model,_=load_bundle(bundle,'cuda');model.load(args.probe/'fresh/CALIBRATED.pt')
            set_arm(model,'critic_only');clear_optimizer_memory(model.optimizer);reset_learning_rng(seed)
            frozen=frozen_state(model);head={n:p.detach().clone() for n,p in model.loss.named_parameters() if p.requires_grad}
            calibration=[];losses=[]
            for step in range(4097):
                if step:
                    batch=schedule[step-1];model.optimizer.zero_grad(set_to_none=True)
                    q=differentiable_q(model,features[batch])
                    loss=(target_loss if arm=='absolute' else advantage_loss)(q,target[batch])
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.loss.parameters(),10.,error_if_nonfinite=True)
                    model.optimizer.step();losses.append(float(loss.detach()))
                if step in CRITIC_POINTS:
                    verify_frozen_state(model,frozen)
                    moved=sum(not torch.equal(p,head[n]) for n,p in model.loss.named_parameters() if n in head)
                    if step and not moved:raise ValueError('ADVANTAGE_CRITIC_NOT_MOVING')
                    calibration.append(dict(updates=step,head_tensors_changed=moved,
                        metrics=panel_metrics(model,features,cohort,greedy,stochastic,plain)))
                    write(lane/'CALIBRATION.json',calibration)
                    progress('CRITIC_FIT',replica=replica,arm=arm,updates=step)
            sync_targets(model);model.save(lane/'CALIBRATED.pt')
            if arm=='absolute':
                expected,_=load_bundle(bundle,'cuda');expected.load(root/'fit/paired_greedy/CALIBRATED.pt')
                theirs=dict(expected.loss.named_parameters())
                for n,p in model.loss.named_parameters():
                    if not torch.equal(p,theirs[n]):raise ValueError('ADVANTAGE_CONTROL_PARAMETER_PARITY:'+n)
                del expected
            actual,_=emit(model,market.inputs)
            if not np.array_equal(actual[market.inputs.valid],parent[market.inputs.valid]):
                raise ValueError('ADVANTAGE_CRITIC_CHANGED_ACTOR')
            set_arm(model,'actor_only');effective=arm_config(model,'kl_parent')
            clear_optimizer_memory(model.optimizer);reset_learning_rng(seed);frozen=frozen_state(model)
            points=[dict(updates=0,**initial)];actor_losses=[]
            for step in range(1,257):
                batch=replay[actor_schedule[step-1]].to('cuda')
                with torch.no_grad():ref=reference.surfaces(batch['features']).detach()
                actor_losses.append(dict(updates=step,**anchored_update(model,batch,ref,effective['beta'])))
                if step in ACTOR_POINTS:
                    verify_frozen_state(model,frozen);progress('ACTOR_ACCOUNTS',replica=replica,arm=arm,updates=step)
                    value=measure(model,market,risk,design,parent,lane/f'actor_{step:04d}')
                    if arm=='absolute':
                        for period,m in value['metrics'].items():
                            account_equal(m,read(args.preservation/replica/'kl_parent'/f'actor_{step:04d}'/period/'METRICS.json'))
                    points.append(dict(updates=step,**value));model.save(lane/f'ACTOR_{step:04d}.pt')
                    write(lane/'ACTOR_PROGRESS.json',points)
            record=dict(identity=config['identity'],effective=effective,calibration=calibration,
                calibration_losses=losses,actor_losses=actor_losses,points=points,
                frozen_parameters_exact=True,absolute_archive_parity=True if arm=='absolute' else None)
            write(lane/'RESULT.json',record);seal(lane,config['identity'],start)
            results[replica]['arms'][arm]=record;del model
    if file_sha(bundle/'model.pt')!=source_sha:raise ValueError('ADVANTAGE_SOURCE_MUTATED')
    report(args.output,results);seal(args.output,config['identity'],start);progress('COMPLETE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('source','source-runtime','probe','calibration','ranking','preservation','output'):
        parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
