"""Paired short SAC update diagnostics; immutable source, no model promotion."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import cat

from app.generalization_diagnostics import summary, soft_td_target
from app.selected_warm_start import reset_learning_rng, target_surfaces
from app.structural_refit import load_bundle, build_model
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, support_hash
from app.three_signal_learning import environment
from agent_plugins.discrete_curve_sac import tensor_observation
from agent_plugins.sac_critic_regularization import module_mode
from tools.run_discrete_curve_probe import configure, read, write, transition, now
from tools.run_structural_sac_refit import authenticate, market_for, assess_full
from tools.run_refit_mask_pass import account_equal

ARMS=('no_update','actor_only','critic_only','joint','joint_adaptive_alpha','joint_lr_div10')
POINTS=(0,1,16,64,256)


def set_arm(model,arm):
    if arm not in ARMS:raise ValueError('DIAGNOSTIC_ARM')
    for name,param in model.loss.named_parameters():
        train=False
        if '.head.' in name:
            train=((name.startswith('actor_') and arm not in ('no_update','critic_only')) or
                   (name.startswith('qvalue_') and arm not in ('no_update','actor_only')))
        if name=='log_alpha':train=arm=='joint_adaptive_alpha'
        param.requires_grad_(train)
    for group in model.optimizer.param_groups:
        group['lr']=model.effective['learning_rate']/(10 if arm=='joint_lr_div10' else 1)


def frozen_state(model):
    return {n:p.detach().clone() for n,p in model.loss.named_parameters() if not p.requires_grad}


def verify_frozen_state(model,state):
    params=dict(model.loss.named_parameters())
    for name,value in state.items():
        if not torch.equal(value,params[name]):raise ValueError('DIAGNOSTIC_FROZEN:'+name)


def state_value_drift(before,after):
    """Compare values, allowing lazy TD buffers to move from CPU to CUDA."""
    if before.keys()!=after.keys():raise ValueError('DIAGNOSTIC_STATE_KEYS')
    changed=[];moved=[];maximum=0.
    for name,old in before.items():
        new=after[name]
        if old.device!=new.device:moved.append(name)
        a,b=old.detach().cpu(),new.detach().cpu()
        if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):changed.append(name)
        if a.is_floating_point() and a.shape==b.shape:
            maximum=max(maximum,float((a-b).abs().max()))
    return dict(changed=changed,device_moves=moved,max_abs=maximum)


def tensor_digest(td):
    h=hashlib.sha256()
    for key in sorted(td.keys(True,True),key=str):
        value=td[key].detach().cpu().contiguous().numpy()
        h.update(str((key,value.shape,str(value.dtype))).encode());h.update(value.tobytes())
    return h.hexdigest()


def probes(model,market,reference,batch):
    out={}
    with torch.inference_mode():
        for period in ('TRAIN','VALIDATION'):
            rows=np.flatnonzero(market.inputs.valid & (market.inputs.period==period))
            rows=rows[np.linspace(0,len(rows)-1,min(512,len(rows)),dtype=int)]
            x=torch.tensor(market.inputs.windows[rows],device=model.device)
            logits=model.surfaces(x);p=logits.softmax(-1)
            base=reference.surfaces(x).softmax(-1)
            q=model.q_surfaces(x);minimum=q.min(0).values
            qnop=minimum[:,0]-minimum[:,1:].max(-1).values
            entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)
            out[period]=dict(rows=len(rows),entropy=summary(entropy.cpu().numpy()),
                mean_probabilities=p.mean(0).cpu().tolist(),
                greedy_share=[float((p.argmax(-1)==a).float().mean()) for a in range(3)],
                q_greedy_share=[float((minimum.argmax(-1)==a).float().mean()) for a in range(3)],
                q_nop_advantage=summary(qnop.cpu().numpy()),
                twin_disagreement=summary((q[0]-q[1]).abs().cpu().numpy()),
                kl_from_initial=summary((base*(base.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum(-1).cpu().numpy()),
                action_change=float((p.argmax(-1)!=base.argmax(-1)).float().mean()))
        x=batch['features'];nxt=batch['next','features']
        q=model.q_surfaces(x);p=model.surfaces(nxt).softmax(-1)
        target_q=target_surfaces(model,nxt);alpha=model.loss.log_alpha.exp()
        expected=soft_td_target(batch['next','reward'],batch['next','terminated'],
            batch['steps_to_next_obs'],p,target_q,alpha,model.effective['gamma'])
        with module_mode(model.loss.qvalue_network,False):
            native=model.loss._compute_target(batch.clone())
        torch.testing.assert_close(expected,native,rtol=2e-5,atol=2e-5)
        qa=q.gather(-1,batch['action'].long().reshape(1,-1,1).expand(2,-1,1)).squeeze(-1)
        out['TD']=dict(alpha=float(alpha),reward=summary(batch['next','reward'].cpu().numpy()),
            target=summary(expected.cpu().numpy()),
            abs_error=summary((qa-expected).abs().cpu().numpy()),
            native_target_max_error=float((expected-native).abs().max()))
    return out


def evaluate(model,market,reference,batch,risk,design):
    # Diagnostics must not change the update/dropout random stream.
    with torch.random.fork_rng():
        return dict(metrics=assess_full(model,market,risk,design),
                    probes=probes(model,market,reference,batch))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-runtime','snapshot','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(args)
        except Exception as exc:
            write(args.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),updated_at=now()))
            raise


def run(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError('DIAGNOSTIC_CUDA_REQUIRED')
    spec=authenticate(args.source,args.source_runtime)
    market=market_for(args.source,spec);risk=read(args.source/'RISK.json');design=spec['design']
    if support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']:
        raise ValueError('DIAGNOSTIC_SUPPORT')
    snap=read(args.snapshot/'MANIFEST.json');state=read(args.snapshot/'STATE.json')
    if snap['identity']!=spec['identity'] or state['identity']!=spec['identity']:
        raise ValueError('DIAGNOSTIC_SNAPSHOT_IDENTITY')
    for name in ('STATE.json','model.pt'):
        if file_sha(args.snapshot/name)!=snap['files'][name]:raise ValueError('DIAGNOSTIC_SNAPSHOT_HASH')
    bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    reference,_=load_bundle(bundle,'cuda')
    initial=read(args.source/'runs/seed_7/EPOCH_ZERO.json')
    config=dict(source_identity=spec['identity'],initial_checkpoint_sha256=file_sha(bundle/'model.pt'),
        snapshot_checkpoint_sha256=file_sha(args.snapshot/'model.pt'),snapshot_epoch=state['epoch'],
        arms=list(ARMS),points=list(POINTS),transitions=4096,batch_size=128,seed=7,
        sources={n:file_sha(Path(__file__).resolve().parents[1]/n) for n in
            ('tools/diagnose_warmstart_drift.py','agent_plugins/three_signal_sac.py','app/generalization_diagnostics.py')},
        scope='single-seed paired update mechanism probe; no model promotion; no TEST')
    config['identity']=digest(config)
    if (args.output/'CONFIG.json').exists() and read(args.output/'CONFIG.json')!=config:
        raise ValueError('DIAGNOSTIC_CONFIG_CHANGED')
    write(args.output/'CONFIG.json',config)
    if (args.output/'DONE.json').exists():return
    def progress(stage,**fields):
        write(args.output/'PROGRESS.json',dict(identity=config['identity'],stage=stage,
            updated_at=now(),seconds=time.monotonic()-started,**fields))
    progress('COLLECT_SHARED_TRAIN_REPLAY')
    replay_path=args.output/'TRAIN_REPLAY.pt';meta_path=args.output/'TRAIN_REPLAY.json'
    if replay_path.exists() and meta_path.exists():
        meta=read(meta_path)
        if meta['sha256']!=file_sha(replay_path):raise ValueError('DIAGNOSTIC_REPLAY_HASH')
        replay=torch.load(replay_path,weights_only=False,map_location='cpu')
        if meta['tensor_digest']!=tensor_digest(replay):raise ValueError('DIAGNOSTIC_REPLAY_CONTENT')
    else:
        env=environment(market,risk,design['horizon'],market.recipe['split']['periods']['TRAIN'],reference.effective['gamma'])
        obs=env.reset();items=[];reset_learning_rng(7)
        for step in range(4096):
            if env.finished:raise ValueError('DIAGNOSTIC_TRAIN_TOO_SHORT')
            with torch.no_grad():action=int(reference.distribution(tensor_observation(obs,'cuda')).sample()[0])
            nxt,reward,done,duration=env.step(action)
            items.append(transition(obs,action,nxt,reward,done,duration));obs=nxt
            if (step+1)%256==0:progress('COLLECT_SHARED_TRAIN_REPLAY',transitions=step+1)
        replay=cat(items,dim=0);torch.save(replay,replay_path)
        write(meta_path,dict(sha256=file_sha(replay_path),tensor_digest=tensor_digest(replay),
            period='TRAIN',rows=len(replay),source_checkpoint_sha256=config['initial_checkpoint_sha256']))
    schedule=np.random.default_rng(7).integers(0,len(replay),size=(256,128))
    schedule_sha=hashlib.sha256(schedule.tobytes()).hexdigest()
    write(args.output/'BATCH_SCHEDULE.json',dict(sha256=schedule_sha,indices=schedule.tolist()))
    probe_batch=replay[schedule[0]].to('cuda')
    baseline=evaluate(reference,market,reference,probe_batch,risk,design)
    for period in baseline['metrics']:account_equal(baseline['metrics'][period],initial['metrics'][period])
    write(args.output/'INITIAL.json',baseline)
    current=build_model(market.recipe,spec['effective'],device='cuda');current.load(args.snapshot/'model.pt')
    old=reference.loss.state_dict();new=current.loss.state_dict()
    frozen_changes=[n for n in old if ('.encoder.' in n and not n.startswith('target_') or n=='log_alpha')
                    and not torch.equal(old[n],new[n])]
    if frozen_changes:raise ValueError('DIAGNOSTIC_FROZEN_SNAPSHOT:'+str(frozen_changes))
    snapshot=evaluate(current,market,reference,probe_batch,risk,design)
    for period in snapshot['metrics']:
        account_equal(snapshot['metrics'][period],state['history'][-1]['metrics'][period])
    write(args.output/'SNAPSHOT.json',dict(epoch=state['epoch'],frozen_changes=frozen_changes,**snapshot))
    del current
    results={}
    for arm in ARMS:
        path=args.output/(arm+'.json')
        if path.exists():
            record=read(path)
            if record['identity']!=config['identity'] or record['schedule_sha256']!=schedule_sha:
                raise ValueError('DIAGNOSTIC_ARM_IDENTITY')
            results[arm]=record;continue
        model,_=load_bundle(bundle,'cuda');set_arm(model,arm);reset_learning_rng(7)
        frozen=frozen_state(model);original={k:v.clone() for k,v in model.loss.state_dict().items()}
        record=dict(identity=config['identity'],arm=arm,schedule_sha256=schedule_sha,
                    replay_tensor_digest=tensor_digest(replay),points=[])
        for step in range(257):
            if step and arm!='no_update':model.update(replay[schedule[step-1]].to('cuda'))
            if step in POINTS:
                verify_frozen_state(model,frozen)
                result=evaluate(model,market,reference,probe_batch,risk,design)
                if step==0 or arm=='no_update':
                    for period in result['metrics']:account_equal(result['metrics'][period],baseline['metrics'][period])
                drift=state_value_drift(original,model.loss.state_dict())
                if arm=='no_update' and drift['changed']:
                    raise ValueError('DIAGNOSTIC_CONTROL_MUTATED')
                result.update(updates=step if arm!='no_update' else 0,
                    equivalent_schedule_position=step,
                    max_state_value_change=drift['max_abs'],
                    state_device_moves=drift['device_moves'],
                    frozen_parameters_exact=True)
                record['points'].append(result)
                write(args.output/(arm+'.progress.json'),record)
                progress('PAIRED_OPERATORS',arm=arm,update=step,completed_arms=list(results))
        write(path,record);results[arm]=record;del model
    lines=['# Warm-start paired operator diagnosis','',config['scope'],'',
        '| Arm | Updates | TRAIN % | VAL % | VAL Sharpe | VAL orders | VAL NOP share | Alpha |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,record in results.items():
        for point in record['points']:
            m=point['metrics'];p=point['probes']
            lines.append(f"| {arm} | {point['updates']} | {m['TRAIN']['return_pct']:.4f} | {m['VALIDATION']['return_pct']:.4f} | {m['VALIDATION']['sharpe']:.4f} | {m['VALIDATION']['trades']} | {p['VALIDATION']['greedy_share'][0]:.4f} | {p['TD']['alpha']:.6f} |")
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(args.output/'DONE.json',dict(identity=config['identity'],finished_at=now(),
        seconds=time.monotonic()-started,files={p.name:file_sha(p) for p in args.output.iterdir()
            if p.is_file() and p.name not in ('worker.lock','PROGRESS.json','DONE.json')}))
    progress('COMPLETE',completed_arms=list(results))


if __name__=='__main__':main()
