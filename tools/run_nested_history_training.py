"""Pinned fresh-network history-size qualification and sequential GPU queue."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from app.frozen_recipe_inputs import file_sha
from app.nested_history_training import training_design
from app.recipe_runtime import RecipeMarket
from app.rl_subset import digest, support_hash
from app.sac_training_reliability import weight_diagnostics
from app.structural_refit import build_model, export_bundle, load_bundle, verify_files
from app.three_signal_learning import environment
from app.training_forensics import tree_digest
from tools.run_discrete_curve_probe import configure, now, read, write
from tools.run_long_sac_continuation import SOURCE
from tools.run_sac_training_reliability import code_identity, reseed, atomic_save
from tools.run_structural_sac_refit import assess_full, fit_epochs


def authenticate(c):
    if c['schema'] != 'nested_history374.v1' or c['source'] != SOURCE:
        raise ValueError('HISTORY_CONFIG')
    if c['code_identity'] != code_identity():
        raise ValueError('HISTORY_CODE_CHANGED')
    if sorted(c['months']) != [6,12,24,48] or c['seed'] not in (7,19):
        raise ValueError('HISTORY_QUEUE_CONTRACT')
    parent=Path(c['parent']); bundle=Path(c['bundle']); stores=Path(c['stores'])
    ps=read(parent/'SPEC.json')
    if file_sha(parent/'SPEC.json') != c['parent_sha256']:
        raise ValueError('HISTORY_PARENT')
    verify_files(parent,{k:v for k,v in ps['files'].items()
        if k.startswith(('data/','engine/')) or k in ('SUPPORT_RECIPE.json','RISK.json')})
    verify_files(bundle,read(bundle/'MANIFEST.json')['files'])
    if file_sha(bundle/'model.pt') != SOURCE or file_sha(stores/'DONE.json') != c['stores_done_sha256']:
        raise ValueError('HISTORY_SOURCE_OR_STORES')
    top=read(stores/'DONE.json')
    for m in c['months']:
        arm=stores/f'months_{m}'; done=read(arm/'DONE.json')
        if done['identity'] != top['identity'] or done['summary'] not in top['arms']:
            raise ValueError('HISTORY_PRODUCER_IDENTITY')
        verify_files(arm,done['files'])
    if len({a['validation_support_sha256'] for a in top['arms']}) != 1:
        raise ValueError('HISTORY_VALIDATION_SUPPORT')
    return ps, read(bundle/'MODEL_CONFIG.json')


def market_for(c, months):
    arm=Path(c['stores'])/f'months_{months}'; parent=Path(c['parent'])
    paths=dict(recipe=str(arm/'RECIPE.json'),store=str(arm/'store.parquet'),
        manifest=str(arm/'manifest.json'),bars=str(parent/'data/bars.parquet'),
        closures=str(parent/'data/closures.json'))
    return RecipeMarket(dict(inputs=paths,engine_root=str(parent/'engine'),
        recipe_sha256=file_sha(paths['recipe'])))


def input_parity(c, months, market):
    r=market.recipe; inp=market.inputs; arm=Path(c['stores'])/f'months_{months}'
    frame=pd.read_parquet(arm/'store.parquet')
    if r['list']['n'] != 71 or len(r['representation']['resolved']['indices']) != 28:
        raise ValueError('HISTORY_INPUT_COUNT')
    if any(x['consumer_ops'] for x in r['tensor']['columns']):
        raise ValueError('HISTORY_UNEXPECTED_CONSUMER_OP')
    if set(r['split']['periods']) != {'TRAIN','VALIDATION'}:
        raise ValueError('HISTORY_SPLITS')
    close=pd.DatetimeIndex(frame.timestamp).as_unit('ns').asi8+3600*10**9
    raw=frame[r['list']['columns']].to_numpy(dtype=np.float64)
    scaled=((raw-np.asarray(r['scaler']['mean']))/np.asarray(r['scaler']['std'])).astype(np.float32)
    ix=np.flatnonzero(inp.valid)
    slots=np.searchsorted(close,inp.slot_ts_ns[ix])
    if not np.array_equal(close[slots],inp.slot_ts_ns[ix]):
        raise ValueError('HISTORY_WINDOW_CLOCK')
    if not np.array_equal(scaled[slots],inp.windows[ix]):
        raise ValueError('HISTORY_SCALING_PARITY')
    train=inp.valid&(inp.period=='TRAIN'); val=inp.valid&(inp.period=='VALIDATION')
    lo=pd.Timestamp(r['split']['periods']['TRAIN'][0],tz='UTC').value
    if np.any(inp.slot_ts_ns[train,0]<lo):
        raise ValueError('HISTORY_WINDOW_OUTSIDE_SUBSET')
    expected=read(arm/'RESULT.json')
    val_hash=hashlib.sha256(inp.row_ts_ns[val].astype('<i8').tobytes()).hexdigest()
    if (int(train.sum()) != expected['train_windows'] or int(val.sum()) != expected['validation_windows']
            or val_hash != expected['validation_support_sha256']):
        raise ValueError('HISTORY_SUPPORT_CHANGED')
    return dict(input_parity_exact=True,train_windows=int(train.sum()),validation_windows=int(val.sum()),
        validation_support_sha256=val_hash,support_sha256=support_hash(inp),
        constant_columns=r['scaler']['constant_columns'])


def qualification(c, months, effective, device, output):
    market=market_for(c,months); receipt=input_parity(c,months,market)
    model=build_model(market.recipe,effective,c['seed'],device)
    if model.optimizer.state:
        raise ValueError('HISTORY_FRESH_ADAM_NOT_EMPTY')
    reference=build_model(read(Path(c['stores'])/'months_48/RECIPE.json'),effective,c['seed'],device)
    initial_hash=tree_digest(model.loss.state_dict())
    if initial_hash != tree_digest(reference.loss.state_dict()):
        raise ValueError('HISTORY_UNPAIRED_INITIALIZATION')
    del reference
    ix=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    probe=market.inputs.windows[ix[np.linspace(0,len(ix)-1,32,dtype=int)]].copy()
    initial=output/'initial_bundle'
    if not initial.exists():
        export_bundle(initial,model,probe,dict(epoch=0,initialization='fresh_all',seed=c['seed']))
    cold,_=load_bundle(initial,device)
    if (tree_digest(cold.loss.state_dict()) != initial_hash
            or tree_digest(cold.optimizer.state_dict()) != tree_digest(model.optimizer.state_dict())):
        raise ValueError('HISTORY_COLD_RELOAD')
    del cold
    receipt.update(initial_parameter_sha256=initial_hash,optimizer_empty=True,cold_reload_exact=True,
        trainable_parameters=sum(p.numel() for p in model.loss.parameters() if p.requires_grad),
        fresh_actor_critics_targets=True,trained_source_loaded=False,seed=c['seed'],device=device)
    write(output/'QUALIFICATION.json',receipt)
    return model,market,probe,receipt


def worker(c, months):
    configure(c['seed']); torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise ValueError('HISTORY_CUDA_REQUIRED')
    ps,effective=authenticate(c)
    output=Path(c['output'])/f'months_{months}_seed_{c["seed"]}'
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    design=training_design(ps['design'])
    binding=dict(schema=c['schema'],config=digest(c),months=months,seed=c['seed'],
        code=c['code_identity'],recipe=file_sha(Path(c['stores'])/f'months_{months}/RECIPE.json'),
        design=design,effective=effective,device=torch.cuda.get_device_name(),torch=torch.__version__)
    binding['identity']=digest(binding); identity=binding['identity']
    if (output/'SPEC.json').exists() and read(output/'SPEC.json') != binding:
        raise ValueError('HISTORY_RESUME_IDENTITY')
    write(output/'SPEC.json',binding)
    if (output/'RESULT.json').exists():
        result=read(output/'RESULT.json')
        if result['identity'] != identity: raise ValueError('HISTORY_RESULT_IDENTITY')
        verify_files(output,result['files']);return
    write(output/'PROGRESS.json',dict(stage='QUALIFYING',identity=identity,at=now()))
    model,market,probe,receipt=qualification(c,months,effective,'cuda',output)
    risk=read(Path(c['parent'])/'RISK.json')
    initial={n:p.detach().clone() for n,p in model.loss.named_parameters()}
    x=torch.tensor(probe,device='cuda')
    with torch.no_grad(): actions=model.surfaces(x).argmax(-1)
    write(output/'DESIGN.json',design);write(output/'EFFECTIVE.json',model.effective)
    reseed(c['seed'])
    def hook(m,row):
        row['weight_diagnostics']=weight_diagnostics(m,initial,x,actions)
        path=output/'weights'/('epoch_%04d.pt'%row['epoch']);path.parent.mkdir(exist_ok=True)
        atomic_save(path,{n:v.detach().cpu() for n,v in m.loss.state_dict().items()})
        row['weight_snapshot']=dict(path=str(path.relative_to(output)),sha256=file_sha(path))
    state=fit_epochs(model,lambda:environment(market,risk,design['horizon'],
        market.recipe['split']['periods']['TRAIN'],effective['gamma']),
        lambda m:assess_full(m,market,risk,design),design,c['seed'],output,identity,probe,epoch_hook=hook)
    if state['epoch'] != 101 or state['updates'] != 600600:
        raise ValueError('HISTORY_FINAL_GRADIENT_BUDGET')
    selected=state['best_bundle']; files={n:file_sha(output/n) for n in
        ('SPEC.json','HISTORY.json','TRAINING.json','QUALIFICATION.json','EFFECTIVE.json','DESIGN.json')}
    if selected:
        folder=output/selected['relative_path']
        files.update({str(p.relative_to(output)):file_sha(p) for p in folder.iterdir() if p.is_file()})
    selected_epoch=None if selected is None else int(selected['path'].split('_')[-1])
    selected_metrics=(None if selected_epoch is None else read(output/'EPOCH_ZERO.json')['metrics']
        if selected_epoch==0 else next(r['metrics'] for r in state['history'] if r['epoch']==selected_epoch))
    write(output/'RESULT.json',dict(identity=identity,status='COMPLETE',months=months,seed=c['seed'],
        epochs=state['epoch'],updates=state['updates'],steps=state['steps'],best_bundle=selected,
        eligible_checkpoint=selected is not None,selected_epoch=selected_epoch,selected_metrics=selected_metrics,
        last_metrics=state['history'][-1]['metrics'],files=files,qualification=receipt,
        promoted=False,phase2_closed=False,test_read=False))
    write(output/'PROGRESS.json',dict(identity=identity,stage='COMPLETE',at=now()))


def admission(c):
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu or ',' in gpu:raise ValueError('HISTORY_SINGLE_GPU_REQUIRED')
    q=subprocess.run(['nvidia-smi','--id='+gpu,'--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    if any(line.strip().isdigit() for line in q.stdout.splitlines()):return 'GPU_BUSY'
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
        if s.startswith('MemAvailable:'))
    return 'HOST_MEMORY_PRESSURE' if available<c['minimum_available_bytes'] else None


def queue(c, config_path):
    authenticate(c);output=Path(c['output']);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'CONFIG.json').exists() and read(output/'CONFIG.json') != c:
        raise ValueError('HISTORY_QUEUE_CHANGED')
    write(output/'CONFIG.json',c)
    for months in c['months']:
        arm=output/f'months_{months}_seed_{c["seed"]}'
        if (arm/'RESULT.json').exists():
            result=read(arm/'RESULT.json');spec=read(arm/'SPEC.json')
            if spec['config'] != digest(c) or result['identity'] != spec['identity']:
                raise ValueError('HISTORY_COMPLETED_IDENTITY')
            verify_files(arm,result['files']);continue
        while True:
            reason=admission(c)
            if reason is None:break
            write(output/'QUEUE_STATUS.json',dict(stage='WAITING_RESOURCE',reason=reason,months=months,at=now()))
            time.sleep(30)
        write(output/'QUEUE_STATUS.json',dict(stage='TRAINING',months=months,seed=c['seed'],at=now()))
        arm.mkdir(exist_ok=True)
        with (arm/'WORKER.log').open('a') as log:
            subprocess.run([sys.executable,'-u','-m','tools.run_nested_history_training',
                '--config',str(config_path),'--worker',str(months)],check=True,stdout=log,stderr=subprocess.STDOUT)
    write(output/'QUEUE_STATUS.json',dict(stage='COMPLETE',at=now()))


def preflight(c):
    configure(c['seed']);_,effective=authenticate(c)
    output=Path(c['output'])/'preflight';output.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for months in c['months']:
        arm=output/f'months_{months}';arm.mkdir(exist_ok=True)
        model,market,probe,receipt=qualification(c,months,effective,'cpu',arm)
        env=environment(market,read(Path(c['parent'])/'RISK.json'),6,
            market.recipe['split']['periods']['TRAIN'],effective['gamma'])
        obs=env.reset();decisions=int(np.count_nonzero(env.valid[env.bar:env.engine.last]))
        if obs['features'].shape != (13,71) or not obs['mask'].all():raise ValueError('HISTORY_NATIVE_OBSERVATION')
        receipt.update(months=months,decisions_per_traversal=decisions)
        receipts.append(receipt);del model,market,env
    if len({r['initial_parameter_sha256'] for r in receipts}) != 1:raise ValueError('HISTORY_PREFLIGHT_INITIALIZATION')
    write(output/'RESULT.json',dict(config=digest(c),arms=receipts,training_launched=False))
    print(json.dumps(receipts),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--worker',type=int,choices=(6,12,24,48))
    modes.add_argument('--preflight',action='store_true')
    args=parser.parse_args();config=read(args.config)
    try:
        if args.preflight:preflight(config)
        elif args.worker:worker(config,args.worker)
        else:queue(config,args.config)
    except Exception as exc:
        root=Path(config['output'])
        if args.worker:root=root/f'months_{args.worker}_seed_{config["seed"]}'
        write(root/'FAILURE.json',dict(at=now(),error=repr(exc)));raise
