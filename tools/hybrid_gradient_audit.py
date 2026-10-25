"""Finite no-update on-policy gradient audit on authenticated TRAIN checkpoints."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from app.frozen_recipe_inputs import file_sha, require
from app.hybrid_learning_diagnostics import gradient_geometry
from tools.recipe_hybrid_queue import Market, authenticated, verify, configure, DESIGN
from tools.recipe_sac_queue import read, write, now, tensor_hash, clean


def inspect_rollout(model):
    rows=[]
    for sample in model.rollout_buffer.get(batch_size=256):
        values,log_prob,entropy=model.policy.evaluate_actions(sample.observations,sample.actions)
        values=values.flatten()
        advantage=sample.advantages
        if model.normalize_advantage:
            advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        ratio=torch.exp(log_prob-sample.old_log_prob)
        actor=-torch.minimum(advantage*ratio,advantage*torch.clamp(ratio,1-.2,1+.2)).mean()
        critic=model.vf_coef*((sample.returns-values)**2).mean()
        ent=-model.ent_coef*entropy.mean()
        params=list(model.policy.features_extractor.parameters())
        residual=(sample.returns-values).detach().cpu().numpy()
        returns=sample.returns.detach().cpu().numpy()
        var=float(np.var(returns))
        with torch.no_grad():
            dist=model.policy.get_distribution(sample.observations)
            probabilities=dist.cat.probs
            route=sample.observations['action_mask'][:,3]+2*sample.observations['action_mask'][:,4]
        rows.append(dict(actor_vs_critic=gradient_geometry(actor,critic,params),
            actor_with_entropy_vs_critic=gradient_geometry(actor+ent,critic,params),
            actor_loss=float(actor.detach()),weighted_critic_loss=float(critic.detach()),
            weighted_entropy_loss=float(ent.detach()),value_rmse=float(np.sqrt(np.mean(residual**2))),
            explained_variance=1-float(np.var(residual))/var if var else None,
            ratio_max_abs_from_one=float((ratio.detach()-1).abs().max()),
            route_counts={str(r):int((route==r).sum()) for r in range(3)},
            probability_mean=probabilities.mean(0).cpu().tolist(),
            categorical_entropy_mean=float(dist.cat.entropy().mean()),
            advantage_std=float(sample.advantages.std()),return_std=float(sample.returns.std())))
    return rows


def run(root,out,device):
    root=root.resolve();out=out.resolve()
    require(not out.is_relative_to(root),'OUTPUT_ISOLATION')
    spec=verify(root);sha=file_sha(root/'SPEC.private.json')
    require(read(root/'CLOSE.json')['state']=='COMPLETE','PARENT_CLOSED')
    # Imports execute outside the frozen source; old scientific modules must match.
    runtime=Path(__file__).resolve().parents[1]
    for name,digest in spec['code']['agent'].items():
        require(file_sha(runtime/name)==digest,'IMPORTED_PARENT_CODE_PIN')
    design=dict(seeds=[7,19,43],anchor_fractions=[0.,1/3,2/3],transitions=512,
                arm='CAT_FIXED',period='TRAIN',checkpoint=102400,optimizer_steps=0,
                start='flat, initial capital at first valid bar after fixed TRAIN anchor',
                interpretation='diagnostic restart, not portfolio performance',device=device)
    identity=dict(parent_spec_sha256=sha,design=design,tool_sha256=file_sha(Path(__file__)),
        helper_sha256=file_sha(runtime/'app/hybrid_learning_diagnostics.py'))
    if (out/'IDENTITY.json').exists():require(read(out/'IDENTITY.json')==identity,'IDENTITY')
    else:write(out/'IDENTITY.json',identity)
    identity_sha=file_sha(out/'IDENTITY.json')
    configure();market=Market(spec,'CAT_FIXED');parent_status=read(root/'STATUS.json')
    results=[];started=time.monotonic()
    for seed in design['seeds']:
        parent_path=root/parent_status['completed'][f'CAT_FIXED_s{seed}']
        authenticated(parent_path,sha)
        archive=parent_path.parent/'model-102400.zip'
        for anchor in range(3):
            unit=f's{seed}-anchor{anchor}';dest=out/unit/'RESULT.json'
            if dest.exists():
                result=read(dest)
                require(result['identity_sha256']==identity_sha and result['unit']==unit
                        and result['checkpoint_sha256']==file_sha(archive),'RESULT_IDENTITY')
                results.append(result);continue
            e=market.env('TRAIN')
            span=e.engine.last-e.engine.first
            e.engine.first+=int(design['anchor_fractions'][anchor]*span)
            require(e.valid[e.engine.first:e.engine.last].sum()>512,'ANCHOR_SUPPORT')
            model=PPO.load(archive,env=e,device=device)
            model.set_random_seed(71000+seed*10+anchor)
            before=tensor_hash(model)
            support=hashlib.sha256();transitions=[]
            class Capture(BaseCallback):
                def _on_step(self):
                    info=self.locals['infos'][0]
                    self.model.rollout_buffer.pending_duration=np.array([info['elapsed_bars']],np.int32)
                    obs=self.locals['obs_tensor']
                    support.update(obs['features'].detach().cpu().numpy().tobytes())
                    support.update(np.asarray([e.engine.v[info['decision_bar']]],np.int64).tobytes())
                    transitions.append(dict(bar=info['decision_bar'],elapsed_bars=info['elapsed_bars'],
                        reward=float(self.locals['rewards'][0]),command=info['command']))
                    if len(transitions)%128==0:
                        write(out/'STATUS.json',dict(state='COLLECTING',unit=unit,complete=len(results),total=9,
                            decisions=len(transitions),heartbeat_at=now()))
                    return True
            _,cb=model._setup_learn(512,callback=Capture(),reset_num_timesteps=True)
            cb.on_training_start({}, {})
            require(model.collect_rollouts(model.env,cb,model.rollout_buffer,n_rollout_steps=512),'COLLECTION')
            batches=inspect_rollout(model)
            require(before==tensor_hash(model),'DIAGNOSTIC_CHANGED_WEIGHTS')
            require(len(transitions)==512,'TRANSITION_COUNT')
            rewards=np.array([r['reward'] for r in transitions])
            result=clean(dict(state='COMPLETE',unit=unit,identity_sha256=identity_sha,
                checkpoint_sha256=file_sha(archive),policy_tensor_sha256=before,
                source_result_sha256=file_sha(parent_path),seed=seed,anchor=anchor,
                support_sha256=support.hexdigest(),optimizer_steps=0,
                start_bar=transitions[0]['bar'],end_bar=e.bar,period='TRAIN',
                reward=dict(mean=float(rewards.mean()),std=float(rewards.std()),
                    min=float(rewards.min()),max=float(rewards.max()),nonzero=int(np.count_nonzero(rewards))),
                elapsed_bars=sum(r['elapsed_bars'] for r in transitions),
                command_counts={k:sum(r['command']==k for r in transitions) for k in sorted({r['command'] for r in transitions})},
                batches=batches,finished_at=now()))
            write(dest,result);results.append(result)
            print(unit,'COMPLETE',round(time.monotonic()-started,1),flush=True)
            del model,e
    write(out/'SUMMARY.json',results)
    write(out/'STATUS.json',dict(state='COMPLETE',units=len(results),seconds=time.monotonic()-started,finished_at=now()))
    write(out/'CLOSE.json',dict(state='COMPLETE',identity_sha256=identity_sha,
        artifacts={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*.json') if p.name!='CLOSE.json'},finished_at=now()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();run(a.root,a.output,a.device)


if __name__=='__main__':main()
