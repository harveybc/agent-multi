"""Finite full-SAC replay qualification and paired role-LR interventions."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import random
import time

import numpy as np
import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, support_hash
from app.sac_training_reliability import (ROLES, split_optimizer, snapshot, restore,
    signatures, weight_diagnostics)
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, build_model, verify_files
from app.three_signal_learning import environment
from app.training_forensics import rng_state, restore_rng, tree_digest
from agent_plugins.discrete_curve_sac import tensor_observation
from tools.run_discrete_curve_probe import configure, transition, write, read, now
from tools.run_structural_sac_refit import assess_full


ARMS = dict(control=dict.fromkeys(ROLES,1.), actor_lr01=dict(actor=.1,critic=1.,temperature=1.),
    critic_lr01=dict(actor=1.,critic=.1,temperature=1.), alpha_lr01=dict(actor=1.,critic=1.,temperature=.1),
    all_lr01=dict.fromkeys(ROLES,.1))
MEASURE = {0,1,8,32,64,128,256,512}


def atomic_save(path, value):
    tmp=path.with_suffix('.building')
    torch.save(value,tmp); tmp.replace(path)


def code_identity():
    root=Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)):file_sha(p)
        for folder in ('app','agent_plugins','env_plugins','tools')
        for p in sorted((root/folder).rglob('*.py'))})


def fresh(a):
    model,_=load_bundle(a.bundle,'cuda')
    reference=build_model(model.recipe,model.effective,a.seed,'cuda')
    flags={n:p.requires_grad for n,p in reference.loss.named_parameters()}
    del reference
    for n,p in model.loss.named_parameters():p.requires_grad_(flags[n])
    return model


def reseed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def collect(a,model,market,risk,design,identity):
    path=a.output/'DATA.pt'; receipt=a.output/'DATA.json'
    if receipt.exists():
        r=read(receipt)
        if r['identity']!=identity or file_sha(path)!=r['sha256']:raise ValueError('REPLAY_DATA_IDENTITY')
        return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    interval=model.recipe['split']['periods']['TRAIN']
    env=environment(market,risk,design['horizon'],interval,model.effective['gamma'])
    expected_rows=int(env.valid[env.engine.first:env.engine.last].sum())
    obs=env.reset(); rng=np.random.default_rng(a.seed); reseed(a.seed)
    storage=LazyMemmapStorage(len(market.inputs.valid),scratch_dir=str(a.output/'collection'),
        existsok=True,auto_cleanup=False)
    replay=TensorDictReplayBuffer(storage=storage,batch_size=design['batch_size'])
    count=0; started=time.monotonic()
    while not env.finished:
        if count<design['warmup'] and not design.get('warmup_policy',False):
            action=int(rng.integers(0,3))
        else:
            with torch.no_grad():
                action=int(model.distribution(tensor_observation(obs,'cuda')).sample()[0])
        nxt,reward,done,duration=env.step(action)
        replay.extend(transition(obs,action,nxt,reward,done,duration));obs=nxt;count+=1
        if count%512==0:
            write(a.output/'PROGRESS.json',dict(stage='COLLECT_FULL_TRAIN',rows=count,
                native_bar=int(env.bar),last_native_bar=int(env.engine.last),seconds=time.monotonic()-started,at=now()))
    data=replay._storage.get(torch.arange(len(replay))).clone()
    if count!=expected_rows:raise ValueError('INCOMPLETE_NATIVE_TRAIN_TRAVERSAL')
    if not torch.isfinite(data['next','reward']).all():raise ValueError('NONFINITE_REAL_REWARD')
    atomic_save(path,data)
    write(receipt,dict(identity=identity,sha256=file_sha(path),rows=count,expected_rows=expected_rows,interval=interval,
        source='fixed selected299 actor native TRAIN traversal; no optimizer steps',
        batch_size=design['batch_size'],support=support_hash(market.inputs)))
    return data


def advance(model,data,sampler,n,batch_size,start=0):
    trace=[]
    for step in range(start,start+n):
        ix=sampler.integers(len(data),size=batch_size)
        b=data[torch.as_tensor(ix)].clone().to('cuda')
        values=model.update(b)
        trace.append(dict(step=step+1,indices=tree_digest(ix),values=values))
    return trace


def qualify(a,data,design,identity):
    out=a.output/'QUALIFICATION.json'
    if out.exists():
        result=read(out)
        if result['identity']!=identity or not result['passed']:raise ValueError('PRIOR_REPLAY_QUALIFICATION_FAILED')
        return
    size=design['batch_size']; rng=np.random.default_rng(a.seed+500)
    indices=[rng.integers(len(data),size=size) for _ in range(8)]
    original=fresh(a); reseed(a.seed+900); before=rng_state()
    trace=[]
    for ix in indices:trace.append(original.update(data[torch.as_tensor(ix)].clone().to('cuda')))
    expected=tree_digest(original.loss.state_dict())
    moments={n:tree_digest(original.optimizer.state[p]) for n,p in original.loss.named_parameters()}
    del original
    grouped=fresh(a); split_optimizer(grouped,ARMS['control']);restore_rng(before)
    actual=[]
    for ix in indices:actual.append(grouped.update(data[torch.as_tensor(ix)].clone().to('cuda')))
    split_ok=(actual==trace and tree_digest(grouped.loss.state_dict())==expected and
        moments=={n:tree_digest(grouped.optimizer.state[p]) for n,p in grouped.loss.named_parameters()})
    del grouped
    model=fresh(a);split_optimizer(model,ARMS['control']);reseed(a.seed+1000)
    sampler=np.random.default_rng(a.seed+2000)
    head=advance(model,data,sampler,32,size)
    mid=snapshot(model,sampler,32,identity);atomic_save(a.output/'RESTART32.pt',mid)
    tail=advance(model,data,sampler,32,size,32);expected=signatures(snapshot(model,sampler,64,identity))
    del model
    cold=fresh(a);split_optimizer(cold,ARMS['control']);sampler=np.random.default_rng(0)
    restore(cold,sampler,torch.load(a.output/'RESTART32.pt',weights_only=False,map_location='cpu'),identity)
    replay=advance(cold,data,sampler,32,size,32)
    actual=signatures(snapshot(cold,sampler,64,identity));del cold
    mismatches=[k for k in expected if actual[k]!=expected[k]]
    first=next((x['step'] for x,y in zip(tail,replay) if x!=y),None)
    result=dict(identity=identity,passed=split_ok and not mismatches and first is None,
        original_vs_grouped_exact=split_ok,resume_state_mismatches=mismatches,first_trace_divergence=first,
        continuous_tail=tail,resumed_tail=replay,first32_trace=head,
        expected_signatures=expected,actual_signatures=actual)
    write(out,result)
    if not result['passed']:raise ValueError('REPLAY_QUALIFICATION_FAILED')


def run_arm(a,name,factors,data,market,risk,design,identity,probe):
    root=a.output/name;root.mkdir(exist_ok=True)
    armid=digest(dict(parent=identity,name=name,factors=factors))
    if (root/'RESULT.json').exists():
        result=read(root/'RESULT.json')
        if result['identity']!=armid or file_sha(root/'LATEST.pt')!=result['checkpoint_sha256']:
            raise ValueError('ARM_RESULT_IDENTITY')
        return result
    model=fresh(a);receipt=split_optimizer(model,factors)
    initial={n:p.detach().clone() for n,p in model.loss.named_parameters()}
    with torch.no_grad():initial_actions=model.surfaces(probe).argmax(-1)
    sampler=np.random.default_rng(a.seed+3000);reseed(a.seed+4000)
    history=[];trace=[];step=0;covered=set()
    if (root/'LATEST.pt').exists():
        saved=torch.load(root/'LATEST.pt',map_location='cpu',weights_only=False)
        restore(model,sampler,saved['state'],armid)
        history=saved['history'];trace=saved['trace'];step=saved['state']['step'];covered=set(saved['covered'])
    else:
        history=[dict(step=0,diagnostics=weight_diagnostics(model,initial,probe,initial_actions),
            metrics=read(a.output/'BASELINE.json')['metrics'])]
    started=time.monotonic();started_step=step
    for step in range(step+1,513):
        ix=sampler.integers(len(data),size=design['batch_size']);covered.update(ix.tolist())
        b=data[torch.as_tensor(ix)].clone().to('cuda');loss=model.update(b)
        td=b.get('td_error',None)
        trace.append(dict(step=step,indices_sha256=tree_digest(ix),loss=loss,
            td_error_mean=None if td is None else float(td.detach().mean())))
        if step in MEASURE:
            before=rng_state()
            row=dict(step=step,loss=loss,diagnostics=weight_diagnostics(model,initial,probe,initial_actions))
            if step in (64,256,512):
                row['metrics']=assess_full(model,market,risk,design)
            if tree_digest(before)!=tree_digest(rng_state()):raise ValueError('DIAGNOSTICS_MUTATED_RNG')
            history.append(row)
        if step%32==0:
            state=snapshot(model,sampler,step,armid)
            atomic_save(root/'LATEST.pt',dict(state=state,history=history,trace=trace,covered=sorted(covered)))
            if step in (64,256,512):atomic_save(root/f'STEP_{step:04d}.pt',state)
            write(root/'HISTORY.json',history)
            write(a.output/'PROGRESS.json',dict(stage='PAIRED_INTERVENTIONS',arm=name,step=step,total=512,
                completed_arms=sum((a.output/n/'RESULT.json').exists() for n in ARMS),at=now(),
                seconds=time.monotonic()-started,steps_this_session=step-started_step))
    final=snapshot(model,sampler,512,armid)
    cold=fresh(a);split_optimizer(cold,factors);other=np.random.default_rng(0)
    restore(cold,other,torch.load(root/'LATEST.pt',map_location='cpu',weights_only=False)['state'],armid)
    if signatures(final)!=signatures(snapshot(cold,other,512,armid)):raise ValueError('ARM_COLD_STATE')
    with torch.no_grad():
        if not torch.equal(model.surfaces(probe),cold.surfaces(probe)):raise ValueError('ARM_COLD_LOGITS')
    result=dict(identity=armid,name=name,factors=factors,actual_optimizer=receipt,updates=512,
        samples=512*design['batch_size'],distinct_rows_sampled=len(covered),replay_rows=len(data),
        checkpoint_sha256=file_sha(root/'LATEST.pt'),cold_exact=True,history=history,
        final_metrics=history[-1]['metrics'],promoted=False)
    write(root/'RESULT.json',result)
    return result


def run(a):
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        configure(a.seed);torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise ValueError('GPU_REQUIRED')
        spec=read(a.parent/'SPEC.json'); verify_files(a.parent,{k:v for k,v in spec['files'].items()
            if k.startswith(('data/','engine/')) or k in ('SUPPORT_RECIPE.json','RISK.json')})
        model=fresh(a);recipe=model.recipe;design=deepcopy(spec['design'])
        design.update(head_only=False,diagnostic_updates=512,early_stopping_applied=False)
        if set(recipe['split']['periods'])!={'TRAIN','VALIDATION'} or recipe['list']['n']!=71:
            raise ValueError('RELIABILITY_SUPPORT')
        binding=dict(schema='full_sac_reliability357.v1',seed=a.seed,source=file_sha(a.bundle/'model.pt'),
            manifest=file_sha(a.bundle/'MANIFEST.json'),parent=file_sha(a.parent/'SPEC.json'),
            code=code_identity(),design=design,effective=model.effective,arms=ARMS,updates_per_arm=512,
            hardware=torch.cuda.get_device_name(),torch=torch.__version__,trainability={n:p.requires_grad for n,p in model.loss.named_parameters()})
        binding['identity']=digest(binding);identity=binding['identity']
        if (a.output/'SPEC.json').exists() and read(a.output/'SPEC.json')!=binding:raise ValueError('WORKER_IDENTITY')
        write(a.output/'SPEC.json',binding)
        if (a.output/'RESULT.json').exists():
            if read(a.output/'RESULT.json')['identity']!=identity:raise ValueError('FINAL_IDENTITY')
            return
        market=market_for(a.parent,spec,recipe);risk=read(a.parent/'RISK.json')
        if not (a.output/'BASELINE.json').exists():
            write(a.output/'BASELINE.json',dict(identity=identity,metrics=assess_full(model,market,risk,design),
                support=support_hash(market.inputs)))
        data=collect(a,model,market,risk,design,identity);del model
        write(a.output/'PROGRESS.json',dict(stage='EXACT_REPLAY_GATE',at=now()))
        qualify(a,data,design,identity)
        ix=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
        probe=torch.tensor(market.inputs.windows[ix[np.linspace(0,len(ix)-1,128,dtype=int)]],device='cuda')
        results=[run_arm(a,n,f,data,market,risk,design,identity,probe) for n,f in ARMS.items()]
        write(a.output/'RESULT.json',dict(identity=identity,status='BOUNDED_DIAGNOSIS_COMPLETE',seed=a.seed,
            arms=results,phase2_closed=False,deap_released=False,promoted=False,
            scope='fixed full-TRAIN native replay; not full online convergence qualification'))
        write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),arms=len(results)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(7,19,43),required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as exc:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc),at=now()))
        raise
