"""Paired critic-only mechanism audit; no trading-model promotion or refit."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.fixed_actor_critic_audit import ARMS, configure_arm, algebra_audit, checked_update, backup_components
from app.generalization_diagnostics import summary
from app.policy_return_calibration import calibration
from app.selected_warm_start import reset_learning_rng, target_surfaces
from app.structural_refit import load_bundle, build_model
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, support_hash
from agent_plugins.sac_critic_regularization import module_mode
from tools.diagnose_warmstart_drift import tensor_digest, frozen_state, verify_frozen_state
from tools.run_discrete_curve_probe import read, write, configure, now
from tools.run_structural_sac_refit import authenticate, market_for

ROOT = Path(__file__).resolve().parents[1]
POINTS = (0, 1, 64, 256, 1024)


def verify_collection(root):
    done = read(root/'DONE.json')
    for name, expected in done['files'].items():
        if file_sha(root/name) != expected:
            raise ValueError('CRITIC_AUDIT_COLLECTION:' + name)
    return done


def measure(model, batch, panel):
    with torch.random.fork_rng(), torch.no_grad():
        expected = model.loss._compute_target(batch.clone())
        q = model.q_surfaces(batch['features'])
        tq = target_surfaces(model, batch['next','features'])
        samples = []
        with module_mode(model.loss.qvalue_network, True):
            for _ in range(16):
                samples.append(model.loss._vmap_qnetworkN0(
                    batch.select('features'),model.loss.qvalue_network_params)['action_value'].clone())
        samples = torch.stack(samples)
        index = batch['action'].reshape(1,-1,1).expand(2,-1,1)
        chosen = q.gather(-1,index).squeeze(-1)
        components = backup_components(batch['next','reward'],batch['next','terminated'],
            batch['steps_to_next_obs'],model.surfaces(batch['next','features']).softmax(-1),
            tq,chosen.mean(0),model.loss.log_alpha.exp(),model.effective['gamma'])
        if getattr(model,'_audit_mean_target',False):
            components['residual'] -= components['minimum_twin_penalty']
            components['minimum_twin_penalty'] = torch.zeros_like(components['residual'])
        torch.testing.assert_close(components['residual'],expected-chosen.mean(0),atol=2e-5,rtol=2e-5)
        panel_q = model.q_surfaces(panel['features']).gather(-1,
            panel['action'].reshape(1,-1,1).expand(2,-1,1)).squeeze(-1).min(0).values
        return dict(backup_components={k:summary(v.cpu().numpy()) for k,v in components.items()},
            online_eval_q=summary(q.cpu().numpy()),target_q=summary(tq.cpu().numpy()),
            bellman_target=summary(expected.cpu().numpy()),
            eval_td_error=summary((chosen-expected).cpu().numpy()),
            train_minus_eval_q=summary((samples.mean(0)-q).cpu().numpy()),
            dropout_std=summary(samples.std(0).cpu().numpy()),
            fixed_policy_train_panel=calibration(panel_q.cpu().numpy(),panel['returns']))


def run(args):
    started = time.monotonic()
    configure(7)
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():raise ValueError('CRITIC_AUDIT_GPU_REQUIRED')
    spec = authenticate(args.source,args.source_runtime)
    market = market_for(args.source,spec)
    if set(market.recipe['split']['periods']) != {'TRAIN','VALIDATION'}:
        raise ValueError('CRITIC_AUDIT_SPLITS')
    if support_hash(market.inputs) != read(args.source/'PREFLIGHT.json')['support_sha256']:
        raise ValueError('CRITIC_AUDIT_SUPPORT')
    replay_receipt = verify_collection(args.replay)
    return_receipt = verify_collection(args.calibration)
    meta = read(args.replay/'TRAIN_REPLAY.json')
    bundle = args.source/'runs/seed_7/bundles/epoch_0000'
    if meta['source_checkpoint_sha256'] != file_sha(bundle/'model.pt'):
        raise ValueError('CRITIC_AUDIT_REPLAY_SOURCE')
    replay = torch.load(args.replay/'TRAIN_REPLAY.pt',map_location='cpu',weights_only=False)
    if meta['tensor_digest'] != tensor_digest(replay) or meta['period'] != 'TRAIN':
        raise ValueError('CRITIC_AUDIT_REPLAY')
    if read(args.calibration/'CONFIG.json')['initial_sha256'] != file_sha(bundle/'model.pt'):
        raise ValueError('CRITIC_AUDIT_RETURN_SOURCE')
    z = np.load(args.calibration/f'epoch_{args.start_epoch:04d}/stochastic_7/TRAIN/RETURN_TRACE.npz')
    ix = np.linspace(0,len(z['timestamp'])-1,512,dtype=int)
    rows = np.searchsorted(market.inputs.row_ts_ns,z['timestamp'][ix])
    if (not np.array_equal(market.inputs.row_ts_ns[rows],z['timestamp'][ix]) or
            not (market.inputs.period[rows] == 'TRAIN').all() or not market.inputs.valid[rows].all()):
        raise ValueError('CRITIC_AUDIT_PANEL_SUPPORT')
    panel = dict(features=torch.tensor(market.inputs.windows[rows],device='cuda'),
        action=torch.tensor(z['signal'][ix],device='cuda'),returns=z['soft_return'][ix])
    snapshots = {}
    for path in args.snapshot:
        m = read(path/'MANIFEST.json');s = read(path/'STATE.json')
        if m['identity'] != spec['identity'] or s['identity'] != spec['identity']:
            raise ValueError('CRITIC_AUDIT_SNAPSHOT_IDENTITY')
        for name in ('model.pt','STATE.json'):
            if file_sha(path/name) != m['files'][name]:raise ValueError('CRITIC_AUDIT_SNAPSHOT_HASH')
        snapshots[s['epoch']] = (path,m['files']['model.pt'])
    if set(snapshots) != {6,40}:raise ValueError('CRITIC_AUDIT_SNAPSHOTS')
    schedule = np.random.default_rng(7).integers(0,len(replay),size=(1024,128))
    config = dict(source_identity=spec['identity'],initial_sha256=file_sha(bundle/'model.pt'),
        replay_identity=replay_receipt['identity'],return_identity=return_receipt['identity'],
        snapshot_sha256={str(e):h for e,(_,h) in snapshots.items()},
        schedule_sha256=digest(schedule.tolist()),arms=list(ARMS),points=list(POINTS),
        train_panel_rows=512,train_panel_seed=7,updates=1024,batch_size=128,
        sources={n:file_sha(ROOT/n) for n in ('app/fixed_actor_critic_audit.py',
            'tools/audit_fixed_actor_critics.py','agent_plugins/three_signal_sac.py',
            'agent_plugins/discrete_curve_sac.py','agent_plugins/sac_critic_regularization.py')},
        inspection_only=args.inspection_only,start_epoch=args.start_epoch,
        scope='TRAIN-only critic-head interventions; fixed actor/alpha; no account promotion')
    if args.inspection_only:
        config.update(arms=[],points=[],updates=0,scope='read-only critic snapshot arithmetic; no updates')
    elif args.start_epoch==40:
        config.update(arms=['production','mean_target'],
            scope='exploratory epoch40 paired twin-minimum ablation; actor40 fixed; no promotion')
    config['identity'] = digest(config)
    if (args.output/'CONFIG.json').exists() and read(args.output/'CONFIG.json') != config:
        raise ValueError('CRITIC_AUDIT_CONFIG_CHANGED')
    write(args.output/'CONFIG.json',config)
    if (args.output/'DONE.json').exists():return
    def progress(stage,**fields):
        write(args.output/'PROGRESS.json',dict(identity=config['identity'],stage=stage,
            updated_at=now(),seconds=time.monotonic()-started,**fields))
    probe = replay[schedule[0]].to('cuda')
    inspection = {}
    for epoch in (0,6,40):
        progress('SNAPSHOT_ALGEBRA',epoch=epoch)
        model,_ = load_bundle(bundle,'cuda')
        if epoch:model.load(snapshots[epoch][0]/'model.pt')
        # Counterexample fixtures on real observations: terminal and duration variants.
        terminal = probe.clone();terminal['next','terminated'].fill_(True);terminal['next','done'].fill_(True)
        duration = probe.clone();duration['steps_to_next_obs'].fill_(4)
        inspection[str(epoch)] = dict(native=algebra_audit(model,probe),
            terminal=algebra_audit(model,terminal),duration4=algebra_audit(model,duration),
            measures=measure(model,probe,panel))
        # Epoch6/40 actor differs: their fixed-policy panel is off-policy, not calibration.
        inspection[str(epoch)]['panel_scope'] = (f'on-policy epoch{args.start_epoch}' if epoch==args.start_epoch else
            f'epoch{args.start_epoch}-policy return reference only; not on-policy calibration')
        del model
    write(args.output/'SNAPSHOT_AUDIT.json',inspection)
    records = {}
    for arm in config['arms']:
        destination = args.output/(arm+'.json')
        if destination.exists():
            record=read(destination)
            if record['identity']!=config['identity']:raise ValueError('CRITIC_AUDIT_ARM_IDENTITY')
            records[arm]=record;continue
        model,_=load_bundle(bundle,'cuda')
        if args.start_epoch:model.load(snapshots[args.start_epoch][0]/'model.pt')
        configure_arm(model,arm);reset_learning_rng(7)
        frozen=frozen_state(model)
        with torch.no_grad():actor=model.surfaces(panel['features']).clone()
        record=dict(identity=config['identity'],arm=arm,points=[])
        targets = {k:v.clone() for k,v in model.loss.target_qvalue_network_params.items(True,True)}
        for step in range(1025):
            if step and arm != 'no_update':
                batch=replay[schedule[step-1]].to('cuda')
                if step in (1,1024):
                    update=checked_update(model,batch,arm)
                else:
                    update=model.update(batch)
                    if arm=='fixed_target':
                        with torch.no_grad():
                            for k,v in targets.items():model.loss.target_qvalue_network_params[k].copy_(v)
            if step in POINTS:
                verify_frozen_state(model,frozen)
                with torch.no_grad():
                    if not torch.equal(actor,model.surfaces(panel['features'])):
                        raise ValueError('CRITIC_AUDIT_ACTOR_MUTATED')
                result=measure(model,probe,panel)
                result.update(updates=step if arm!='no_update' else 0,schedule_position=step,
                    actor_exact=True,frozen_parameters_exact=True,
                    update=None if not step or arm=='no_update' else update)
                record['points'].append(result)
                write(args.output/(arm+'.progress.json'),record)
                progress('CRITIC_INTERVENTIONS',arm=arm,updates=step,completed_arms=list(records))
        write(destination,record);records[arm]=record;del model
    write(args.output/'DONE.json',dict(identity=config['identity'],finished_at=now(),
        seconds=time.monotonic()-started,arms=len(records),
        files={p.name:file_sha(p) for p in args.output.iterdir()
               if p.is_file() and p.name not in ('DONE.json','PROGRESS.json','worker.lock')}))
    progress('COMPLETE',completed_arms=list(records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-runtime','replay','calibration','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,action='append',required=True)
    parser.add_argument('--inspection-only',action='store_true')
    parser.add_argument('--start-epoch',type=int,choices=[0,40],default=0)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:
            write(args.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),updated_at=now()))
            raise


if __name__=='__main__':main()
