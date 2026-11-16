"""Verified full-state fork: actor plateau, then equal-budget fixed-rate control."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import shutil

import numpy as np
import torch
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from app.frozen_recipe_inputs import file_sha
from app.g3_replay import JointReplay
from app.frozen_signal_masks import prepared_signals
from app.policy_margin_trace import policy_delta
from app.rl_subset import digest, support_hash
from app.sac_training_reliability import weight_diagnostics, role
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, load_recovery, verify_files
from app.three_signal_learning import environment, emit, evaluate
from app.training_forensics import tree_digest, rng_state
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_long_sac_continuation import SUPPORT
from tools.run_sac_training_reliability import code_identity, atomic_save
from tools.run_structural_sac_refit import fit_epochs, assess_full
from tools.seal_sac_recovery import seal


PLATEAU = dict(factor=.5, threshold=1e-8, cooldown=5, min_actor_lr=1e-7, post_reduction_epochs=20)


def source_model(source, device):
    model, _ = load_bundle(source/'initial_bundle', device)
    pointer = read(source/'RESUME.json')
    model.load(source/'recovery'/pointer['directory']/'model.pt')
    return model


def prepare_branch(source, output, identity, design):
    if (output/'RESUME.json').exists():
        state = read(output/'recovery'/read(output/'RESUME.json')['directory']/'STATE.json')
        if state['identity'] != identity or read(output/'DESIGN.json') != design:
            raise ValueError('FORK_RESUME_IDENTITY')
        return
    if output.exists():
        raise ValueError('UNCOMMITTED_FORK_DIRECTORY')
    temp = output.with_name(output.name+'.building')
    if temp.exists():shutil.rmtree(temp)
    temp.mkdir(parents=True)
    pointer = read(source/'RESUME.json')
    old = source/'recovery'/pointer['directory']
    new = temp/'recovery'/pointer['directory']
    shutil.copytree(old,new)
    state = read(new/'STATE.json'); oldid = state['identity']
    state['identity'] = identity
    write(new/'STATE.json',state)
    manifest = read(new/'MANIFEST.json'); manifest['identity'] = identity
    manifest['files']['STATE.json'] = file_sha(new/'STATE.json')
    write(new/'MANIFEST.json',manifest)
    pointer['manifest_sha256'] = file_sha(new/'MANIFEST.json')
    for folder in ('initial_bundle','bundles'):shutil.copytree(source/folder,temp/folder)
    for name in ('EFFECTIVE.json','EPOCH_ZERO.json'):shutil.copy2(source/name,temp/name)
    write(temp/'DESIGN.json',design); write(temp/'HISTORY.json',state['history'])
    write(temp/'RESUME.json',pointer)
    write(temp/'FORK.json',dict(source_identity=oldid,identity=identity,epoch=state['epoch'],
        source_seal_sha256=file_sha(source/'SEAL.json'),
        model_exact=file_sha(new/'model.pt')==file_sha(old/'model.pt'),
        learning_exact=file_sha(new/'learning.pt')==file_sha(old/'learning.pt'),
        migration='identity only in STATE and recovery manifest; all learning state inherited'))
    temp.rename(output)


def qualify(source, market, risk, design, output):
    source_id = read(source/'SEAL.json')['identity']
    receipt_path = output/'QUALIFICATION.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['source_seal_sha256'] != file_sha(source/'SEAL.json'):
            raise ValueError('FORK_QUALIFICATION_SOURCE')
        return receipt
    traces = []
    reference = read(source/'HISTORY.json')[-1]['metrics']
    for repetition in range(2):
        model = source_model(source,'cuda')
        names = {id(p):n for n,p in model.loss.named_parameters()}
        roles = [{role(names[id(p)]) for p in g['params']} for g in model.optimizer.param_groups]
        if roles != [{'actor'},{'critic'},{'temperature'}]:raise ValueError('FORK_OPTIMIZER_ROLES')
        real = TensorDictReplayBuffer(storage=LazyTensorStorage(design['replay_size'], device='cpu'),
                                     batch_size=design['batch_size'])
        replay = JointReplay(real,None,design['batch_size'],0,None)
        rng = np.random.default_rng(0)
        state = load_recovery(source,model,replay,rng,source_id)
        before = tree_digest(rng_state())
        native = assess_full(model,market,risk,design)
        if tree_digest(rng_state()) != before:raise ValueError('QUALIFY_INFERENCE_RNG')
        for period in ('TRAIN','VALIDATION'):
            if (abs(native[period]['return_pct']-reference[period]['return_pct'])>1e-9
                    or native[period]['trades']!=reference[period]['trades']):
                raise ValueError('FORK_NATIVE_SOURCE_PARITY')
        trace = dict(epoch=state['epoch'], native=native,
            initial_model=tree_digest(model.loss.state_dict()),
            initial_optimizer=tree_digest(model.optimizer.state_dict()),updates=[])
        for _ in range(4):
            batch = replay.sample().to('cuda')
            bs = tree_digest(batch)
            losses = model.update(batch)
            trace['updates'].append(dict(batch=bs,losses=losses,
                weights=tree_digest(model.loss.state_dict()),optimizer=tree_digest(model.optimizer.state_dict())))
        trace['final_rng'] = tree_digest(rng_state())
        trace['replay_rng'] = deepcopy(replay.rng.bit_generator.state)
        traces.append(trace)
        del replay, real, model
    if traces[0] != traces[1]:raise ValueError('FORK_NEXT_UPDATES_NOT_EXACT')
    receipt = dict(source_seal_sha256=file_sha(source/'SEAL.json'),
        native_exact=True,next_four_sampled_updates_exact=True,trace=traces[0],
        source_modified=False)
    write(receipt_path,receipt)
    return receipt


def arm(a, name, market, risk, source_design, source_epoch, probe, matched_end=None):
    root = a.output/name
    design = deepcopy(source_design)
    design.update(fork_epoch=source_epoch, minimum_epochs=max(100,source_epoch+20),
        max_epochs=2000 if matched_end is None else matched_end,
        matched_budget=name=='fixed', actor_plateau=PLATEAU if name=='plateau' else None)
    identity = digest(dict(source=file_sha(a.source/'SEAL.json'),arm=name,design=design,code=code_identity()))
    if (root/'RESULT.json').exists():
        result=read(root/'RESULT.json')
        if result['identity']!=identity:raise ValueError('FORK_COMPLETED_IDENTITY')
        return result
    prepare_branch(a.source,root,identity,design)
    model = source_model(a.source,'cuda')
    initial = {n:p.detach().clone() for n,p in model.loss.named_parameters()}
    x=torch.as_tensor(probe,device='cuda')
    with torch.no_grad():initial_actions=model.surfaces(x).argmax(-1)
    source_logits,_=emit(model,market.inputs)
    last_function={}
    def assess(m):
        logits,_=emit(m,market.inputs)
        prepared=prepared_signals(logits)
        last_function['value']=policy_delta(source_logits,logits,market.inputs.period)
        epoch=read(root/'PROGRESS.json')['epoch']
        save_trace=epoch in (source_epoch+1,source_epoch+8,source_epoch+9)
        trace=root/'decision_traces'/('epoch_%04d'%epoch)
        if save_trace:
            trace.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(trace/'LOGITS.npz',logits=logits)
        return {p:evaluate(environment(market,risk,design['horizon'],interval,m.effective['gamma']),
            logits,prepared=prepared,save=trace/p if save_trace else None)
            for p,interval in m.recipe['split']['periods'].items()}
    def hook(m,row):
        row['weight_diagnostics']=weight_diagnostics(m,initial,x,initial_actions)
        row['full_policy_delta_from_fork']=deepcopy(last_function['value'])
        path=root/'weights'/('epoch_%04d.pt'%row['epoch']);path.parent.mkdir(exist_ok=True)
        atomic_save(path,{n:v.detach().cpu() for n,v in m.loss.state_dict().items()})
        row['weight_snapshot']=dict(path=str(path.relative_to(root)),sha256=file_sha(path))
    write(a.output/'PROGRESS.json',dict(stage='CONTINUATION',arm=name,source_epoch=source_epoch,at=now()))
    interval=model.recipe['split']['periods']['TRAIN']
    state=fit_epochs(model,lambda:environment(market,risk,design['horizon'],interval,model.effective['gamma']),
        assess,design,a.seed,root,identity,probe,
        replay_factory=lambda real:JointReplay(real,None,design['batch_size'],a.seed+200,None),epoch_hook=hook)
    selected=assess_full(model,market,risk,design)
    result=dict(identity=identity,arm=name,source_epoch=source_epoch,end_epoch=state['epoch'],
        additional_epochs=state['epoch']-source_epoch,last_metrics=state['history'][-1]['metrics'],
        selected_metrics=selected,selected=state['best_bundle'],stop=read(root/'TRAINING.json'),
        promoted=False, phase2_closed=False, at=now())
    write(root/'RESULT.json',result)
    return result


def run(a):
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        configure(a.seed);torch.backends.cudnn.allow_tf32=False
        receipt=seal(a.source,a.source)
        parent=read(a.parent/'SPEC.json')
        verify_files(a.parent,{k:v for k,v in parent['files'].items() if k.startswith(('data/','engine/'))
            or k in ('SUPPORT_RECIPE.json','RISK.json')})
        model=source_model(a.source,'cuda')
        market=market_for(a.parent,parent,model.recipe); risk=read(a.parent/'RISK.json')
        if support_hash(market.inputs)!=SUPPORT:raise ValueError('FORK_SUPPORT')
        design=read(a.source/'DESIGN.json')
        binding=dict(schema='sealed_plateau362.v1',seed=a.seed,source_seal_sha256=file_sha(a.source/'SEAL.json'),
            code=code_identity(),plateau=PLATEAU,source_epoch=receipt['epoch'],source_identity=receipt['identity'])
        if (a.output/'SPEC.json').exists() and read(a.output/'SPEC.json')!=binding:raise ValueError('FORK_SPEC')
        write(a.output/'SPEC.json',binding)
        ix=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
        probe=market.inputs.windows[ix[np.linspace(0,len(ix)-1,128,dtype=int)]].copy()
        del model
        write(a.output/'PROGRESS.json',dict(stage='FULL_STATE_QUALIFICATION',at=now()))
        qualify(a.source,market,risk,design,a.output)
        if a.qualify_only:return
        scheduled=arm(a,'plateau',market,risk,design,receipt['epoch'],probe)
        fixed=arm(a,'fixed',market,risk,design,receipt['epoch'],probe,scheduled['end_epoch'])
        write(a.output/'RESULT.json',dict(status='MATCHED_FORK_COMPLETE',arms=[scheduled,fixed],promoted=False))
        write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','parent','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--qualify-only',action='store_true')
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        write(args.output/'FAILURE.json',dict(error=repr(exc),at=now()))
        raise
