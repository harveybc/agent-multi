"""Paired critic-only mechanism audit; no trading-model promotion or refit."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.fixed_actor_critic_audit import (ARMS, configure_arm, algebra_audit, checked_update,
    backup_components, verify_backup_identity)
from app.generalization_diagnostics import summary
from app.policy_return_calibration import calibration
from app.selected_warm_start import reset_learning_rng, target_surfaces
from app.structural_refit import load_bundle, build_model
from app.three_signal_learning import environment
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, support_hash
from agent_plugins.sac_critic_regularization import module_mode
from app.actor_learning_diagnostics import clear_optimizer_memory
from tools.diagnose_warmstart_drift import tensor_digest, frozen_state, verify_frozen_state
from tools.run_discrete_curve_probe import read, write, configure, now, transition
from tools.run_structural_sac_refit import authenticate, market_for

ROOT = Path(__file__).resolve().parents[1]
POINTS = (0, 1, 64, 256, 1024)


def covering_schedule(rows, updates, batch_size, seed):
    total=updates*batch_size
    if rows<1 or total<rows:raise ValueError('REPLAY_COVERAGE_BUDGET')
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.permutation(rows) for _ in range((total+rows-1)//rows)])[:total].reshape(updates,batch_size)


def collect_full_replay(env, trace):
    from env_plugins.discrete_curve_native import HOUR
    if not trace:raise ValueError('REPLAY_EMPTY')
    obs=env.reset();replay=None;maximum=0.
    for i,row in enumerate(trace):
        if env.finished:raise ValueError('REPLAY_TERMINAL_EARLY')
        position=0 if env.engine.pos is None else env.engine.pos.direction
        if int(env.engine.v[env.bar])+HOUR!=row['timestamp'] or position!=row['position']:
            raise ValueError('REPLAY_NATIVE_STATE')
        action=int(row['signal']);nxt,reward,done,duration=env.step(action)
        error=abs(reward-row['reward']);maximum=max(maximum,error)
        if error>1e-12:raise ValueError('REPLAY_REWARD')
        if duration!=row['duration'] or env.last_command!=row['command']:
            raise ValueError('REPLAY_NATIVE_TRANSITION')
        if bool(done)!=(i==len(trace)-1):raise ValueError('REPLAY_TERMINAL')
        td=transition(obs,action,nxt,reward,done,duration)
        if replay is None:replay=td.expand(len(trace)).clone()
        replay[i:i+1]=td;obs=nxt
    return replay,dict(rows=len(trace),terminal_count=int(replay['next','terminated'].sum()),
        reward_max_error=maximum,native_duration=int(replay['steps_to_next_obs'].sum()))


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
        bootstrap_expected=(model._audit_bootstrap_target(batch.clone())
                            if getattr(model,'_audit_return_target',False) else expected)
        residual_roundoff = verify_backup_identity(components,chosen.mean(0),bootstrap_expected)
        q_chunks=[]
        for first in range(0,len(panel['features']),256):
            q_chunks.append(model.q_surfaces(panel['features'][first:first+256]).cpu())
        panel_surface=torch.cat(q_chunks,dim=1)
        panel_q=panel_surface.gather(-1,
            panel['action'].cpu().reshape(1,-1,1).expand(2,-1,1)).squeeze(-1).min(0).values
        alternate={}
        for seed,reference in panel.get('alternate',{}).items():
            qa=panel_surface.gather(-1,reference['action'].reshape(1,-1,1).expand(2,-1,1)).squeeze(-1).min(0).values
            alternate[str(seed)]=calibration(qa.numpy(),reference['returns'])
            alternate[str(seed)]['error_summary']=summary(qa.numpy()-reference['returns'])
        primary=calibration(panel_q.numpy(),panel['returns'])
        primary['error_summary']=summary(panel_q.numpy()-panel['returns'])
        return dict(residual_roundoff_max=residual_roundoff,
            backup_components={k:summary(v.cpu().numpy()) for k,v in components.items()},
            online_eval_q=summary(q.cpu().numpy()),target_q=summary(tq.cpu().numpy()),
            bellman_target=summary(bootstrap_expected.cpu().numpy()),
            eval_td_error=summary((chosen-bootstrap_expected).cpu().numpy()),
            training_target=summary(expected.cpu().numpy()),
            training_target_kind='realized_soft_return' if getattr(model,'_audit_return_target',False) else 'bootstrap',
            training_target_error=summary((chosen-expected).cpu().numpy()),
            train_minus_eval_q=summary((samples.mean(0)-q).cpu().numpy()),
            dropout_std=summary(samples.std(0).cpu().numpy()),
            alternate_policy_trajectories=alternate,
            fixed_policy_train_panel=primary)


def run(args):
    started = time.monotonic()
    reset_optimizer = getattr(args,'reset_optimizer',False)
    if reset_optimizer and not args.return_anchor:
        raise ValueError('OPTIMIZER_CONTRAST_REQUIRES_RETURN_ANCHOR')
    if args.return_anchor:
        if not args.full_train_replay or args.start_epoch!=40 or args.anchor_control is None:
            raise ValueError('RETURN_ANCHOR_CONTRACT')
    elif args.anchor_control is not None or args.updates!=1024:
        raise ValueError('EXTENDED_BUDGET_ANCHOR_ONLY')
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
    anchor_receipt=verify_collection(args.anchor_control) if args.return_anchor else None
    anchor_config=read(args.anchor_control/'CONFIG.json') if args.return_anchor else None
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
    ix = (np.arange(len(z['timestamp'])) if args.full_train_replay else
          np.linspace(0,len(z['timestamp'])-1,512,dtype=int))
    rows = np.searchsorted(market.inputs.row_ts_ns,z['timestamp'][ix])
    if (not np.array_equal(market.inputs.row_ts_ns[rows],z['timestamp'][ix]) or
            not (market.inputs.period[rows] == 'TRAIN').all() or not market.inputs.valid[rows].all()):
        raise ValueError('CRITIC_AUDIT_PANEL_SUPPORT')
    panel = dict(features=torch.tensor(market.inputs.windows[rows],device='cuda'),
        action=torch.tensor(z['signal'][ix],device='cuda'),returns=z['soft_return'][ix])
    if args.full_train_replay:
        if args.start_epoch!=40:raise ValueError('FULL_REPLAY_REQUIRES_EPOCH40')
        panel['alternate']={}
        for seed in (19,43):
            alt=np.load(args.calibration/f'epoch_0040/stochastic_{seed}/TRAIN/RETURN_TRACE.npz')
            if not np.array_equal(alt['timestamp'],z['timestamp']):raise ValueError('FULL_PANEL_SUPPORT')
            panel['alternate'][seed]=dict(action=torch.tensor(alt['signal']),returns=alt['soft_return'])
    snapshots = {}
    for path in args.snapshot:
        m = read(path/'MANIFEST.json');s = read(path/'STATE.json')
        if m['identity'] != spec['identity'] or s['identity'] != spec['identity']:
            raise ValueError('CRITIC_AUDIT_SNAPSHOT_IDENTITY')
        for name in ('model.pt','STATE.json'):
            if file_sha(path/name) != m['files'][name]:raise ValueError('CRITIC_AUDIT_SNAPSHOT_HASH')
        snapshots[s['epoch']] = (path,m['files']['model.pt'])
    if set(snapshots) != {6,40}:raise ValueError('CRITIC_AUDIT_SNAPSHOTS')
    full_receipt=None
    if args.full_train_replay:
        folder=args.calibration/'epoch_0040/stochastic_7/TRAIN'
        trace=read(folder/'ACTIONS.json')
        expected_count=read(args.source/'PREFLIGHT.json')['periods']['TRAIN']['decision_count']
        if len(trace)!=expected_count or len(rows)!=expected_count:raise ValueError('FULL_REPLAY_COUNT')
        cache=args.anchor_control if args.return_anchor else args.output
        receipt_path=cache/'FULL_TRAIN_REPLAY.json';replay_path=cache/'FULL_TRAIN_REPLAY.pt'
        if receipt_path.exists():
            full_receipt=read(receipt_path)
            if (full_receipt['sha256']!=file_sha(replay_path) or
                    full_receipt['source_trace_sha256']!=file_sha(folder/'ACTIONS.json') or
                    full_receipt['source_checkpoint_sha256']!=snapshots[40][1] or
                    full_receipt['period']!='TRAIN' or full_receipt['rows']!=expected_count):
                raise ValueError('FULL_REPLAY_CACHE_IDENTITY')
            replay=torch.load(replay_path,map_location='cpu',weights_only=False)
            if tensor_digest(replay)!=full_receipt['tensor_digest']:raise ValueError('FULL_REPLAY_CACHE_CONTENT')
        else:
            if (args.output/'CONFIG.json').exists():raise ValueError('FULL_REPLAY_RECEIPT_MISSING')
            env=environment(market,read(args.source/'RISK.json'),spec['design']['horizon'],
                market.recipe['split']['periods']['TRAIN'],spec['effective']['gamma'])
            replay,full_receipt=collect_full_replay(env,trace)
            expected=read(folder/'METRICS.json')
            if (abs(env.engine.balance-env.engine.cfg.capital-expected['net_profit_usd'])>1e-8 or
                    len(env.engine.trades)!=expected['trades']):raise ValueError('FULL_REPLAY_ACCOUNT')
            torch.save(replay,replay_path)
            full_receipt.update(tensor_digest=tensor_digest(replay),sha256=file_sha(replay_path),
                source_trace_sha256=file_sha(folder/'ACTIONS.json'),period='TRAIN',sampling_seed=7,
                source_checkpoint_sha256=snapshots[40][1],account_net_profit_usd=expected['net_profit_usd'])
            write(receipt_path,full_receipt)
    if args.return_anchor:
        if (anchor_config['source_identity']!=spec['identity'] or
                anchor_config['return_identity']!=return_receipt['identity'] or
                anchor_config['snapshot_sha256']['40']!=snapshots[40][1] or
                anchor_config['full_train_replay']!=full_receipt or
                anchor_config['updates']!=1024 or anchor_config['batch_size']!=128):
            raise ValueError('RETURN_ANCHOR_CONTROL_IDENTITY')
        replay['realized_soft_return']=torch.tensor(z['soft_return'],dtype=torch.float32).reshape(-1,1)
    schedule = (covering_schedule(len(replay),args.updates,128,7) if args.full_train_replay else
                np.random.default_rng(7).integers(0,len(replay),size=(1024,128)))
    if args.return_anchor and digest(schedule[:1024].tolist())!=anchor_config['schedule_sha256']:
        raise ValueError('RETURN_ANCHOR_SCHEDULE_PREFIX')
    config = dict(source_identity=spec['identity'],initial_sha256=file_sha(bundle/'model.pt'),
        replay_identity=replay_receipt['identity'],return_identity=return_receipt['identity'],
        snapshot_sha256={str(e):h for e,(_,h) in snapshots.items()},
        schedule_sha256=digest(schedule.tolist()),arms=list(ARMS),points=list(POINTS),
        train_panel_rows=len(rows),train_panel_seed=7,updates=args.updates,batch_size=128,
        anchor_control_identity=anchor_receipt['identity'] if anchor_receipt else None,
        replay_with_targets_digest=tensor_digest(replay) if args.return_anchor else None,
        full_train_replay=full_receipt,schedule_unique_rows=len(np.unique(schedule)),
        sources={n:file_sha(ROOT/n) for n in ('app/fixed_actor_critic_audit.py',
            'tools/audit_fixed_actor_critics.py','agent_plugins/three_signal_sac.py',
            'agent_plugins/discrete_curve_sac.py','agent_plugins/sac_critic_regularization.py')},
        inspection_only=args.inspection_only,start_epoch=args.start_epoch,
        reset_optimizer=reset_optimizer,
        optimizer_diagnostic_sha256=file_sha(ROOT/'app/actor_learning_diagnostics.py'),
        scope='TRAIN-only critic-head interventions; fixed actor/alpha; no account promotion')
    if args.inspection_only:
        config.update(arms=[],points=[],updates=0,scope='read-only critic snapshot arithmetic; no updates')
    elif args.start_epoch==40:
        config.update(arms=['production','mean_target'],
            scope='exploratory epoch40 paired twin-minimum ablation; actor40 fixed; no promotion')
    if args.return_anchor:
        config.update(arms=['return_anchor'],points=sorted(set(POINTS)|{2048,args.updates}) if args.updates>1024 else list(POINTS),
            scope='realized TRAIN return critic anchor; actor40 fixed; no promotion')
    config['identity'] = digest(config)
    if (args.output/'CONFIG.json').exists() and read(args.output/'CONFIG.json') != config:
        raise ValueError('CRITIC_AUDIT_CONFIG_CHANGED')
    write(args.output/'CONFIG.json',config)
    if (args.output/'DONE.json').exists():
        verify_collection(args.output)
        return
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
        configure_arm(model,arm)
        optimizer_reset = (clear_optimizer_memory(model.optimizer) if reset_optimizer else
            dict(previous_states=len(model.optimizer.state),reset=False))
        reset_learning_rng(7)
        frozen=frozen_state(model)
        with torch.no_grad():actor=model.surfaces(panel['features'][:512]).clone()
        record=dict(identity=config['identity'],arm=arm,points=[],optimizer_reset=optimizer_reset)
        targets = {k:v.clone() for k,v in model.loss.target_qvalue_network_params.items(True,True)}
        for step in range(config['updates']+1):
            if step and arm != 'no_update':
                batch=replay[schedule[step-1]].to('cuda')
                if step in (1,1024,config['updates']):
                    update=checked_update(model,batch,arm)
                else:
                    update=model.update(batch)
                    if arm=='fixed_target':
                        with torch.no_grad():
                            for k,v in targets.items():model.loss.target_qvalue_network_params[k].copy_(v)
            if step in config['points']:
                verify_frozen_state(model,frozen)
                with torch.no_grad():
                    if not torch.equal(actor,model.surfaces(panel['features'][:512])):
                        raise ValueError('CRITIC_AUDIT_ACTOR_MUTATED')
                result=measure(model,probe,panel)
                if step==0 and args.full_train_replay:
                    for seed,measured in [(7,result['fixed_policy_train_panel']),
                            *[(s,result['alternate_policy_trajectories'][str(s)]) for s in (19,43)]]:
                        expected=read(args.calibration/f'epoch_0040/stochastic_{seed}/TRAIN/CALIBRATION.json')['min_q']
                        for field in ('bias','mae','rmse','q_mean','return_mean'):
                            if not np.isclose(measured[field],expected[field],rtol=1e-5,atol=1e-5):
                                raise ValueError('FULL_PANEL_INITIAL_PARITY:'+field)
                    result['initial_calibration_parity']=True
                result.update(updates=step if arm!='no_update' else 0,schedule_position=step,
                    actor_exact=True,frozen_parameters_exact=True,
                    update=None if not step or arm=='no_update' else update)
                record['points'].append(result)
                if args.return_anchor and step in (1024,config['updates']):
                    model.save(args.output/f'DIAGNOSTIC_CRITIC_{step:04d}.pt')
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
    parser.add_argument('--full-train-replay',action='store_true')
    parser.add_argument('--return-anchor',action='store_true')
    parser.add_argument('--reset-optimizer',action='store_true')
    parser.add_argument('--anchor-control',type=Path)
    parser.add_argument('--updates',type=int,choices=[1024,4096],default=1024)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:
            write(args.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),updated_at=now()))
            raise


if __name__=='__main__':main()
