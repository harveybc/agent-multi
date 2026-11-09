"""Sequential fixed-critic actor preservation contrast, native accounts and collection."""
import argparse
import csv
import fcntl
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
import torch

from app.actor_preservation import arm_config, anchored_update
from app.actor_learning_diagnostics import clear_optimizer_memory
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest, support_hash
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment, evaluate
from tools.audit_fixed_actor_critics import verify_collection, covering_schedule, collect_full_replay
from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
from tools.run_action_ranking_contrast import seal, ensure_config
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import authenticate, market_for

ROOT=Path(__file__).resolve().parents[1]
POINTS=(0,1,8,32,64,128,256)
ARMS=('native','lr_div3','kl_parent')


def measure(model, market, risk, design, parent, output):
    logits,_=emit(model,market.inputs);prepared=prepared_signals(logits)
    metrics={};drift={}
    for period,interval in market.recipe['split']['periods'].items():
        metrics[period]=evaluate(environment(market,risk,design['horizon'],interval,model.effective['gamma']),
            logits,save=output/period,diagnose=True,prepared=prepared)
        rows=market.inputs.valid & (market.inputs.period==period)
        ref=torch.tensor(parent[rows],dtype=torch.float64).log_softmax(-1)
        cur=torch.tensor(logits[rows],dtype=torch.float64).log_softmax(-1)
        kl=(ref.exp()*(ref-cur)).sum(-1).numpy()
        drift[period]=dict(rows=int(rows.sum()),kl_mean=float(kl.mean()),kl_max=float(kl.max()),
            changed_actions=int((parent[rows].argmax(-1)!=logits[rows].argmax(-1)).sum()),
            maximum_logit_change=float(np.abs(parent[rows]-logits[rows]).max()))
    return dict(metrics=metrics,policy_drift=drift,
        eligible=all(m['orders_per_week']>2 for m in metrics.values()),model_promoted=False)


def report(root, results):
    rows=[]
    for replica,value in results.items():
        for arm,record in value['arms'].items():
            for point in record['points']:
                for period,m in point['metrics'].items():
                    row=dict(replica=replica,arm=arm,updates=point['updates'],period=period,eligible=point['eligible'])
                    row.update({k:m[k] for k in ('return_pct','net_profit_usd','sharpe','max_drawdown_pct',
                        'trades','win_rate','orders_per_week')})
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
        ax.axhline(results[replica]['reference']['metrics'][period]['return_pct'],color='black',linestyle='--',label='frozen parent')
        ax.set(title=f'{replica} {period}',xlabel='Actor updates',ylabel='Account return (%)')
        ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'RETURNS.png',dpi=140);plt.close(fig)
    write(root/'RESULT.json',dict(replicas=results,fixed_endpoint=256,model_promoted=False))


def run(args):
    start=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':
        raise ValueError('ACTOR_PRESERVATION_GPU')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    probe=verify_collection(args.probe);cal=verify_collection(args.calibration)
    if (support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']
            or set(market.recipe['split']['periods'])!={'TRAIN','VALIDATION'}
            or cal['identity']!=read(args.probe/'CONFIG.json')['calibration_identity']):
        raise ValueError('ACTOR_PRESERVATION_LINEAGE')
    bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    source_sha=file_sha(bundle/'model.pt')
    if source_sha!=read(args.probe/'CONFIG.json')['source_sha256']:raise ValueError('ACTOR_SOURCE_HASH')
    parents={s:verify_collection(args.ranking/s/'fit') for s in ('r19','r43')}
    config=dict(source=spec['identity'],probe=probe['identity'],source_sha256=source_sha,
        parents={s:d['identity'] for s,d in parents.items()},points=list(POINTS),arms=list(ARMS),
        seeds=[19,43],batch_size=128,beta=1.0,lr_divisor=3,
        sources={n:file_sha(ROOT/n) for n in ('tools/run_actor_preservation.py','app/actor_preservation.py',
            'agent_plugins/three_signal_sac.py','agent_plugins/discrete_curve_sac.py',
            'tools/diagnose_warmstart_drift.py','app/three_signal_learning.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):
        write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-start,**kw))
    progress('PREFLIGHT')
    risk=read(args.source/'RISK.json');design=spec['design']
    reference,_=load_bundle(bundle,'cuda');parent,_=emit(reference,market.inputs)
    folder=args.calibration/'epoch_0000/stochastic_7/TRAIN'
    env=environment(market,risk,design['horizon'],market.recipe['split']['periods']['TRAIN'],spec['effective']['gamma'])
    replay,replay_receipt=collect_full_replay(env,read(folder/'ACTIONS.json'))
    if len(replay)!=read(args.source/'PREFLIGHT.json')['periods']['TRAIN']['decision_count']:
        raise ValueError('ACTOR_FULL_REPLAY')
    write(args.output/'REPLAY.json',replay_receipt)
    results={}
    for seed in (19,43):
        replica=f'r{seed}';directory=args.output/replica;directory.mkdir(exist_ok=True)
        parentroot=args.ranking/replica
        pc=read(parentroot/'CONFIG.json')
        if (pc['source_identity']!=spec['identity'] or pc['probe_identity']!=probe['identity']
                or pc['identity']!=parents[replica]['identity'] or pc['update_seed']!=seed):
            raise ValueError('ACTOR_PARENT_CONFIG')
        schedule=covering_schedule(len(replay),4096,128,seed)
        if digest(schedule.tolist())!=read(parentroot/'fit/SCHEDULE.json')['actor_sha256']:
            raise ValueError('ACTOR_PAIRED_SCHEDULE')
        write(directory/'SCHEDULE.json',dict(seed=seed,sha256=digest(schedule.tolist()),
            used_updates=256,batch_size=128,replay_rows=len(replay)))
        initial=measure(reference,market,risk,design,parent,directory/'reference')
        for period in initial['metrics']:account_equal(initial['metrics'][period],read(args.probe/'REFERENCE.json')[period])
        write(directory/'REFERENCE.json',initial)
        results[replica]=dict(reference=initial,arms={})
        for arm in ARMS:
            lane=directory/arm;lane.mkdir(exist_ok=True)
            if (lane/'DONE.json').exists():
                if verify_collection(lane)['identity']!=config['identity']:raise ValueError('ACTOR_RESUME_IDENTITY')
                results[replica]['arms'][arm]=read(lane/'RESULT.json');continue
            model,_=load_bundle(bundle,'cuda');model.load(parentroot/'fit/paired_greedy/CALIBRATED.pt')
            set_arm(model,'actor_only');effective=arm_config(model,arm)
            clear_optimizer_memory(model.optimizer);reset_learning_rng(seed)
            frozen=frozen_state(model);points=[];losses=[]
            write(lane/'EFFECTIVE.json',effective)
            for step in range(257):
                if step:
                    batch=replay[schedule[step-1]].to('cuda')
                    with torch.no_grad():ref=reference.surfaces(batch['features']).detach()
                    losses.append(dict(updates=step,**anchored_update(model,batch,ref,effective['beta'])))
                if step in POINTS:
                    verify_frozen_state(model,frozen);progress('ACCOUNTS',replica=replica,arm=arm,updates=step)
                    value=initial if step==0 else measure(model,market,risk,design,parent,lane/f'actor_{step:04d}')
                    if step==0:
                        actual,_=emit(model,market.inputs)
                        if not np.array_equal(actual[market.inputs.valid],parent[market.inputs.valid]):
                            raise ValueError('ACTOR_INITIAL_PARITY')
                    if arm=='native' and step in (1,8,32):
                        for period in value['metrics']:
                            account_equal(value['metrics'][period],read(parentroot/f'fit/paired_greedy/actor_{step:04d}'/period/'METRICS.json'))
                    points.append(dict(updates=step,**value))
                    model.save(lane/f'ACTOR_{step:04d}.pt')
                    write(lane/'PROGRESS.json',dict(points=points,losses=losses))
            record=dict(identity=config['identity'],effective=effective,points=points,losses=losses,
                frozen_parameters_exact=True,native_archive_parity=True if arm=='native' else None)
            write(lane/'RESULT.json',record);seal(lane,config['identity'],start)
            results[replica]['arms'][arm]=record;del model
    if file_sha(bundle/'model.pt')!=source_sha:raise ValueError('ACTOR_SOURCE_MUTATED')
    report(args.output,results);seal(args.output,config['identity'],start);progress('COMPLETE')


def collect(args):
    start=time.monotonic()
    while time.monotonic()-start<7200:
        done=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',args.worker_host,
            shlex.join(['test','-f',str(args.output/'DONE.json')])])
        if done.returncode==0:
            subprocess.run(['rsync','-a',f'{args.worker_host}:{args.output}/',str(args.output)+'/'],check=True)
            receipt=verify_collection(args.output)
            print(dict(verified_files=len(receipt['files']),finished_at=receipt['finished_at']),flush=True);return
        failed=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',args.worker_host,
            shlex.join(['test','-f',str(args.output/'FAILED.json')])])
        if failed.returncode==0:raise ValueError('ACTOR_REMOTE_FAILED')
        time.sleep(15)
    raise TimeoutError('ACTOR_COLLECTION_TIMEOUT')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('fit','collect'))
    for n in ('source','source-runtime','probe','calibration','ranking'):p.add_argument('--'+n,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-host')
    args=p.parse_args()
    if args.stage=='collect':
        if not args.worker_host:p.error('--worker-host required')
        collect(args);return
    if not all(getattr(args,n) for n in ('source','source_runtime','probe','calibration','ranking')):
        p.error('fit requires all source paths')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
