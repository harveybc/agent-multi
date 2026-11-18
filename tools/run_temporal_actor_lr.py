"""Finite paired continuation with authenticated full-state actor temporal LR forks."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from app.structural_refit import build_model, verify_files, load_recovery, EpochPatience
from app.temporal_actor_lr import split_temporal_optimizer, named_moments, foreign_gpu_pids
from app.sac_training_reliability import weight_diagnostics
from app.three_signal_learning import environment
from app.training_forensics import tree_digest, rng_state
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_nested_history_training import market_for, input_parity
from tools.run_sac_training_reliability import code_identity, atomic_save
from tools.run_structural_sac_refit import fit_epochs, assess_full


ARMS={'control':1.,'temporal_lr_div5':.2}


def admission(c):
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu.startswith('GPU-') or ',' in gpu:raise ValueError('TEMPORAL_PHYSICAL_GPU_REQUIRED')
    raw=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True)
    pids=[int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()]
    blocked=foreign_gpu_pids(pids,lambda pid:str(Path('/proc',str(pid),'exe').resolve(strict=True)),
        c.get('allowed_gpu_process_executables',[]))
    if blocked:return 'GPU_BUSY'
    free=int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())*1024**2
    if free<c['minimum_gpu_free_bytes']:return 'GPU_MEMORY_PRESSURE'
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
        if s.startswith('MemAvailable:'))
    return 'HOST_MEMORY_PRESSURE' if available<c['minimum_available_bytes'] else None


def authenticate(c):
    if c['schema']!='temporal_actor_lr376.v1' or c['code_identity']!=code_identity():
        raise ValueError('TEMPORAL_CODE_IDENTITY')
    source=Path(c['source']); spec=read(source/'SPEC.json')
    for name,key in [('SPEC.json','source_spec_sha256'),('RESUME.json','source_resume_sha256')]:
        if file_sha(source/name)!=c[key]:raise ValueError('TEMPORAL_SOURCE_CHANGED')
    pointer=read(source/'RESUME.json'); folder=source/'recovery'/pointer['directory']
    if folder.parent.resolve()!=(source/'recovery').resolve():raise ValueError('TEMPORAL_SOURCE_PATH')
    if file_sha(folder/'MANIFEST.json')!=pointer['manifest_sha256']:
        raise ValueError('TEMPORAL_SOURCE_MANIFEST')
    manifest=read(folder/'MANIFEST.json'); verify_files(folder,manifest['files'])
    state=read(folder/'STATE.json')
    if (manifest['identity']!=spec['identity'] or state['identity']!=spec['identity']
            or state['epoch']!=101 or state['updates']!=600600 or spec['months']!=6 or spec['seed']!=19):
        raise ValueError('TEMPORAL_ANCHOR_CONTRACT')
    parent=Path(c['parent']); ps=read(parent/'SPEC.json')
    if file_sha(parent/'SPEC.json')!=c['parent_spec_sha256']:raise ValueError('TEMPORAL_PARENT')
    verify_files(parent,{k:v for k,v in ps['files'].items()
        if k.startswith(('data/','engine/')) or k in ('SUPPORT_RECIPE.json','RISK.json')})
    store=Path(c['stores'])/'months_6'
    if file_sha(store/'DONE.json')!=c['store_done_sha256']:raise ValueError('TEMPORAL_STORE')
    verify_files(store,read(store/'DONE.json')['files'])
    if file_sha(store/'RECIPE.json')!=spec['recipe']:raise ValueError('TEMPORAL_RECIPE')
    return spec,folder,state


def design_for(spec):
    d=deepcopy(spec['design'])
    d.update(max_epochs=121,minimum_epochs=121,matched_budget=True,
        initialize_epoch_zero=False,monitor_start_epoch=1,select_during_warmup=False)
    d.pop('actor_plateau',None)
    return d


def prepare(c,name):
    spec,source,state=authenticate(c)
    root=Path(c['output'])/name
    identity=digest(dict(config=c,arm=name,factor=ARMS[name]))
    if (root/'RESUME.json').exists():
        if read(root/'FORK.json')['identity']!=identity:raise ValueError('TEMPORAL_FORK_CHANGED')
        return root,identity
    temp=root.with_name(root.name+'.building')
    if temp.exists():shutil.rmtree(temp)
    temp.mkdir(parents=True)
    d=design_for(spec); recipe=read(Path(c['stores'])/'months_6/RECIPE.json')
    model=build_model(recipe,spec['effective'],device='cpu');model.load(source/'model.pt')
    before=dict(weights=tree_digest(model.loss.state_dict()),moments=tree_digest(named_moments(model)))
    groups=split_temporal_optimizer(model,ARMS[name])
    after=dict(weights=tree_digest(model.loss.state_dict()),moments=tree_digest(named_moments(model)))
    if before!=after:raise ValueError('TEMPORAL_FORK_MUTATED_LEARNING_STATE')
    dest=temp/'recovery'/source.name;dest.mkdir(parents=True)
    model.save(dest/'model.pt');shutil.copy2(source/'learning.pt',dest/'learning.pt')
    monitors={mode:EpochPatience(d['patience'],d['min_delta'],mode,1,False) for mode in EpochPatience.MODES}
    stopping=EpochPatience(d['patience'],d['min_delta'],d['monitor_mode'],1,False)
    state.update(identity=identity,history=[],best_bundle=None,
        stopping=vars(stopping).copy(),
        shadows={k:vars(v).copy() for k,v in monitors.items()})
    state.pop('actor_plateau',None);state.pop('best_after_warmup',None)
    write(dest/'STATE.json',state)
    write(dest/'MANIFEST.json',dict(identity=identity,files={p.name:file_sha(p) for p in dest.iterdir()}))
    write(temp/'RESUME.json',dict(directory=source.name,manifest_sha256=file_sha(dest/'MANIFEST.json')))
    write(temp/'DESIGN.json',d)
    write(temp/'FORK.json',dict(identity=identity,source_identity=spec['identity'],anchor_epoch=101,
        source_recovery_manifest_sha256=file_sha(source/'MANIFEST.json'),
        original=before,forked=after,groups=groups,
        replay_rng_exact=file_sha(dest/'learning.pt')==file_sha(source/'learning.pt'),
        monitor_reset='descriptive only; equal fixed budget; no promotion'))
    temp.rename(root)
    return root,identity


def qualify(c,device):
    spec,_,_=authenticate(c);market=market_for(c,6)
    parity=input_parity(c,6,market);d=design_for(spec)
    risk=read(Path(c['parent'])/'RISK.json')
    traces={}
    for name in ARMS:
        root,identity=prepare(c,name); series=[]
        for repetition in range(2):
            model=build_model(market.recipe,spec['effective'],device=device)
            replay=TensorDictReplayBuffer(storage=LazyTensorStorage(d['replay_size'],device='cpu'),batch_size=d['batch_size'])
            rng=np.random.default_rng(19)
            state=load_recovery(root,model,replay,rng,identity)
            before=tree_digest(rng_state())
            metrics=assess_full(model,market,risk,d)
            if before!=tree_digest(rng_state()):raise ValueError('TEMPORAL_ASSESSMENT_RNG')
            trace=dict(weights=tree_digest(model.loss.state_dict()),moments=tree_digest(named_moments(model)),
                metrics=metrics,epoch=state['epoch'],replay=tree_digest(replay.state_dict()),
                rng=tree_digest(rng_state()),updates=[])
            for _ in range(4):
                batch=replay.sample().to(device)
                bh=tree_digest(batch);loss=model.update(batch)
                trace['updates'].append(dict(batch=bh,loss=loss,weights=tree_digest(model.loss.state_dict()),
                    moments=tree_digest(named_moments(model))))
            series.append(trace)
            del replay,model
        if series[0]!=series[1]:raise ValueError('TEMPORAL_COLD_UPDATES_NOT_EXACT')
        traces[name]=series[0]
    for key in ('weights','moments','metrics','replay','rng'):
        if traces['control'][key]!=traces['temporal_lr_div5'][key]:
            raise ValueError('TEMPORAL_UNPAIRED_INITIAL_'+key)
    if [u['batch'] for u in traces['control']['updates']] != [u['batch'] for u in traces['temporal_lr_div5']['updates']]:
        raise ValueError('TEMPORAL_UNPAIRED_SAMPLE_SEQUENCE')
    original=build_model(market.recipe,spec['effective'],device=device)
    replay=TensorDictReplayBuffer(storage=LazyTensorStorage(d['replay_size'],device='cpu'),batch_size=d['batch_size'])
    load_recovery(Path(c['source']),original,replay,np.random.default_rng(19),spec['identity'])
    for expected in traces['control']['updates']:
        batch=replay.sample().to(device);bh=tree_digest(batch);loss=original.update(batch)
        actual=dict(batch=bh,loss=loss,weights=tree_digest(original.loss.state_dict()),
            moments=tree_digest(named_moments(original)))
        if actual!=expected:raise ValueError('TEMPORAL_GROUPING_CHANGED_CONTROL_UPDATE')
    result=dict(at=now(),code=c['code_identity'],config=digest(c),device=str(device),input_parity=parity,
        paired_initial_state=True,cold_four_updates_exact=True,grouped_control_updates_exact=True,
        traces=traces,promoted=False)
    write(Path(c['output'])/'QUALIFICATION.json',result)
    return result


def worker(c,name):
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError('TEMPORAL_GPU_REQUIRED')
    spec,_,_=authenticate(c);root,identity=prepare(c,name)
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'RESULT.json').exists():
        result=read(root/'RESULT.json')
        if result['identity']!=identity:raise ValueError('TEMPORAL_RESULT_IDENTITY')
        return
    q=read(Path(c['output'])/'QUALIFICATION.json')
    if q['config']!=digest(c) or q['device']!='cuda':raise ValueError('TEMPORAL_QUALIFICATION')
    market=market_for(c,6);model=build_model(market.recipe,spec['effective'],device='cuda')
    pointer=read(root/'RESUME.json');model.load(root/'recovery'/pointer['directory']/'model.pt')
    d=design_for(spec);risk=read(Path(c['parent'])/'RISK.json')
    # Reference remains the original anchor even after a service restart.
    anchor=build_model(market.recipe,spec['effective'],device='cuda')
    anchor.load(Path(c['source'])/'recovery/epoch_0101/model.pt')
    initial={n:p.detach().clone() for n,p in anchor.loss.named_parameters()}
    ix=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    probe=market.inputs.windows[ix[np.linspace(0,len(ix)-1,32,dtype=int)]].copy()
    x=torch.tensor(probe,device='cuda')
    with torch.no_grad():actions=anchor.surfaces(x).argmax(-1)
    del anchor
    def hook(m,row):
        row['weight_diagnostics']=weight_diagnostics(m,initial,x,actions)
        row['learning_rates_used']=[float(g['lr']) for g in m.optimizer.param_groups]
        path=root/'weights'/('epoch_%04d.pt'%row['epoch']);path.parent.mkdir(exist_ok=True)
        atomic_save(path,{n:v.detach().cpu() for n,v in m.loss.state_dict().items()})
        row['weight_snapshot']=dict(path=str(path.relative_to(root)),sha256=file_sha(path))
    state=fit_epochs(model,lambda:environment(market,risk,d['horizon'],
        market.recipe['split']['periods']['TRAIN'],model.effective['gamma']),
        lambda m:assess_full(m,market,risk,d),d,19,root,identity,probe,epoch_hook=hook)
    if state['epoch']!=121 or state['updates']!=720720 or len(state['history'])!=20:
        raise ValueError('TEMPORAL_BUDGET')
    write(root/'RESULT.json',dict(identity=identity,arm=name,epoch=state['epoch'],additional_epochs=20,
        additional_updates=120120,endpoint=state['history'][-1]['metrics'],
        selected_descriptive=state['best_bundle'],promoted=False,phase2_closed=False,test_read=False,at=now()))
    write(root/'PROGRESS.json',dict(identity=identity,stage='COMPLETE',at=now()))


def summary(c):
    rows={}
    for name in ARMS:
        r=read(Path(c['output'])/name/'RESULT.json')
        rows[name]={p:dict(return_pct=m['return_pct'],trades=m['trades'],sharpe=m['sharpe'],
            net_per_trade=m['net_profit_usd']/m['trades'] if m['trades'] else None,
            costs=m['costs'],close_reasons=m['close_reasons']) for p,m in r['endpoint'].items()}
    write(Path(c['output'])/'SUMMARY.json',dict(at=now(),endpoint=rows,
        activity_matched=False,interpretation='Recovery diagnostic; lower activity can confound return improvement',
        promoted=False,phase2_closed=False))


def run(c,path):
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    authenticate(c)
    reason=admission(c)
    if reason:raise ValueError('TEMPORAL_ADMISSION:'+reason)
    for name in ARMS:prepare(c,name)
    q=root/'QUALIFICATION.json'
    if not q.exists():
        write(root/'QUEUE_STATUS.json',dict(stage='QUALIFYING',at=now()))
        subprocess.run([sys.executable,'-u','-m','tools.run_temporal_actor_lr','--config',str(path),'--qualify'],check=True)
    elif read(q)['config']!=digest(c):raise ValueError('TEMPORAL_QUALIFICATION_CONFIG')
    for name in ARMS:
        write(root/'QUEUE_STATUS.json',dict(stage='RUNNING',arm=name,at=now()))
        with (root/name/'WORKER.log').open('a') as log:
            subprocess.run([sys.executable,'-u','-m','tools.run_temporal_actor_lr','--config',str(path),
                '--worker',name],check=True,stdout=log,stderr=subprocess.STDOUT)
    summary(c);write(root/'QUEUE_STATUS.json',dict(stage='COMPLETE',at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--qualify',action='store_true')
    modes.add_argument('--preflight',action='store_true')
    modes.add_argument('--worker',choices=list(ARMS))
    a=parser.parse_args();c=read(a.config);configure(19);torch.backends.cudnn.allow_tf32=False
    if a.preflight:
        spec,_,_=authenticate(c);m=market_for(c,6)
        receipt=dict(config=digest(c),input_parity=input_parity(c,6,m),design=design_for(spec),training_launched=False)
        write(Path(c['output'])/'PREFLIGHT.json',receipt)
        print(receipt)
    elif a.qualify:qualify(c,'cuda')
    elif a.worker:worker(c,a.worker)
    else:run(c,a.config)
