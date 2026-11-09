"""Sequential bounded critic restoration and actor sensitivity on preserved70."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.actor_learning_diagnostics import action_calibration
from app.critic_refinement import POLICY, begin_stage, fit_anchor, sync_targets
from app.frozen_recipe_inputs import file_sha
from app.policy_return_calibration import calibration
from app.rl_subset import digest,support_hash
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle
from app.three_signal_learning import environment
from tools.audit_fixed_actor_critics import collect_full_replay, covering_schedule, verify_collection
from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import authenticate, market_for, assess_full

ROOT=Path(__file__).resolve().parents[1]


def calibration_panel(model,replay):
    chunks=[]
    with torch.no_grad():
        for first in range(0,len(replay),256):
            b=replay[first:first+256].to(model.device)
            q=model.q_surfaces(b['features']).min(0).values
            chunks.append(q.gather(-1,b['action'].reshape(-1,1)).flatten().cpu())
    q=torch.cat(chunks).numpy();g=replay['realized_soft_return'].flatten().numpy()
    return dict(**calibration(q,g),by_action=action_calibration(q,g,replay['action'].flatten().numpy()),
                zero=calibration(np.zeros_like(g),g))


def run(args):
    start=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=args.expected_device_name:
        raise ValueError('RESTORATION_GPU_IDENTITY')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    if set(market.recipe['split']['periods'])!={'TRAIN','VALIDATION'}:
        raise ValueError('RESTORATION_PERIODS')
    if support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']:
        raise ValueError('RESTORATION_SUPPORT')
    receipt=verify_collection(args.calibration)
    bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    source_sha=file_sha(bundle/'model.pt')
    if source_sha!=read(args.calibration/'CONFIG.json')['initial_sha256']:
        raise ValueError('RESTORATION_OWN_POLICY')
    config=dict(source_identity=spec['identity'],source_sha256=source_sha,
        calibration_identity=receipt['identity'],policy=POLICY,arms=['retained','fresh'],
        actor_updates=[1,8,32],bootstrap_updates=256,device=torch.cuda.get_device_name(),
        sources={n:file_sha(ROOT/n) for n in ('tools/run_preserved_policy_refinement_probe.py',
            'app/critic_refinement.py','app/actor_learning_diagnostics.py','app/fixed_actor_critic_audit.py',
            'tools/audit_fixed_actor_critics.py','tools/diagnose_warmstart_drift.py',
            'tools/run_structural_sac_refit.py','app/three_signal_learning.py')})
    config['identity']=digest(config)
    if (args.output/'CONFIG.json').exists() and read(args.output/'CONFIG.json')!=config:
        raise ValueError('RESTORATION_CONFIG_CHANGED')
    write(args.output/'CONFIG.json',config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):
        write(args.output/'PROGRESS.json',dict(stage=stage,updated_at=now(),
            seconds=time.monotonic()-start,identity=config['identity'],**kw))
    risk=read(args.source/'RISK.json');design=spec['design']
    interval=market.recipe['split']['periods']['TRAIN']
    folder=args.calibration/'epoch_0000/stochastic_7/TRAIN'
    trace=read(folder/'ACTIONS.json');z=np.load(folder/'RETURN_TRACE.npz')
    if (len(trace)!=read(args.source/'PREFLIGHT.json')['periods']['TRAIN']['decision_count']
            or not np.array_equal(z['timestamp'],[r['timestamp'] for r in trace])
            or not np.array_equal(z['signal'],[r['signal'] for r in trace])):
        raise ValueError('RESTORATION_TRACE_SUPPORT')
    progress('NATIVE_TRAIN_REPLAY')
    env=environment(market,risk,design['horizon'],interval,spec['effective']['gamma'])
    replay,replay_receipt=collect_full_replay(env,trace)
    expected=read(folder/'METRICS.json')
    if abs(env.engine.balance-env.engine.cfg.capital-expected['net_profit_usd'])>1e-8:
        raise ValueError('RESTORATION_NATIVE_ACCOUNT')
    replay['realized_soft_return']=torch.tensor(z['soft_return'],dtype=torch.float32).reshape(-1,1)
    schedule=covering_schedule(len(replay),POLICY['updates'],POLICY['batch_size'],7)
    write(args.output/'REPLAY.json',dict(**replay_receipt,schedule_sha256=digest(schedule.tolist()),
        source_policy=source_sha,period='TRAIN',full_rows=len(replay)))
    base,_=load_bundle(bundle,'cuda')
    progress('REFERENCE_ACCOUNTS')
    reference=assess_full(base,market,risk,design,save=args.output/'reference',diagnose=True)
    original=read(args.source/'runs/seed_7/EPOCH_ZERO.json')['metrics']
    for p,m in reference.items():account_equal(m,original[p])
    with torch.no_grad():base_probe=base.surfaces(replay['features'][:512].to('cuda')).clone()
    write(args.output/'REFERENCE.json',reference);del base
    results={}
    for arm in config['arms']:
        out=args.output/arm;out.mkdir(exist_ok=True)
        if (out/'DONE.json').exists():
            verify_collection(out);results[arm]=read(out/'RESULT.json');continue
        model,_=load_bundle(bundle,'cuda')
        initialization=begin_stage(model,config['identity']+'/'+arm,control=arm=='retained')
        def monitor(r):
            with torch.no_grad():
                if not torch.equal(base_probe,model.surfaces(replay['features'][:512].to('cuda'))):
                    raise ValueError('RESTORATION_ACTOR_CHANGED')
            write(out/'ANCHOR_PROGRESS.json',r);progress('CALIBRATE',arm=arm,updates=r['updates'])
        points=fit_anchor(model,replay,schedule,lambda:calibration_panel(model,replay),monitor)
        target_sync=sync_targets(model);model.save(out/'CALIBRATED.pt')
        write(out/'CALIBRATION.json',dict(initialization=initialization,points=points,target_sync=target_sync))
        # Native-bootstrap recurrence is separate from the actor intervention.
        set_arm(model,'critic_only');frozen=frozen_state(model);reset_learning_rng(7)
        for step in range(256):model.update(replay[schedule[step]].to('cuda'))
        verify_frozen_state(model,frozen)
        bootstrap=calibration_panel(model,replay)
        write(out/'BOOTSTRAP.json',dict(updates=256,calibration=bootstrap,actor_unchanged=True))
        model.load(out/'CALIBRATED.pt');set_arm(model,'actor_only')
        frozen=frozen_state(model);reset_learning_rng(7);accounts=[]
        for step in range(1,33):
            model.update(replay[schedule[step-1]].to('cuda'))
            if step in config['actor_updates']:
                verify_frozen_state(model,frozen);progress('ACTOR_ACCOUNTS',arm=arm,updates=step)
                metrics=assess_full(model,market,risk,design,save=out/f'actor_{step:04d}',diagnose=True)
                with torch.no_grad():drift=float((model.surfaces(replay['features'][:512].to('cuda'))-base_probe).abs().max())
                model.save(out/f'ACTOR_{step:04d}.pt')
                accounts.append(dict(updates=step,metrics=metrics,probe_logit_max_drift=drift,
                    active=all(m['orders_per_week']>2 for m in metrics.values()),model_promoted=False))
                write(out/'ACTOR_PROGRESS.json',accounts)
        result=dict(identity=config['identity'],arm=arm,calibration=points,bootstrap=bootstrap,
            accounts=accounts,source_unchanged=file_sha(bundle/'model.pt')==source_sha,model_promoted=False)
        if not result['source_unchanged']:raise ValueError('RESTORATION_SOURCE_MUTATED')
        write(out/'RESULT.json',result)
        write(out/'DONE.json',dict(identity=config['identity'],finished_at=now(),
            files={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*') if p.is_file() and p.name!='DONE.json'}))
        results[arm]=result;del model
    write(args.output/'COMPARISON.json',dict(identity=config['identity'],reference=reference,arms=results,
        no_test=True,model_promoted=False))
    write(args.output/'DONE.json',dict(identity=config['identity'],finished_at=now(),seconds=time.monotonic()-start,
        files={str(p.relative_to(args.output)):file_sha(p) for p in args.output.rglob('*')
               if p.is_file() and p.name not in ('worker.lock','PROGRESS.json') and p!=args.output/'DONE.json'}))
    progress('COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','source-runtime','calibration','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:
            write(args.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise


if __name__=='__main__':main()
