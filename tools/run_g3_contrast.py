"""Resumable sequential G3 control and matched-budget augmentation contrast."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchrl.data import TensorDictReplayBuffer,LazyMemmapStorage

from app.frozen_recipe_inputs import file_sha
from app.g3_producer_audit import identity
from app.g3_replay import JointReplay,NativeCollector
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.three_signal_learning import environment
from tools.audit_g3_producers import write
from tools.prepare_g3_joint import synthetic_market
from tools.run_discrete_curve_probe import configure
from tools.run_structural_sac_refit import fit_epochs,assess_full


def read(path):return json.loads(path.read_text())


def make_binding(parent,bundle,generated):
    source=Path(__file__).resolve().parents[1]
    d=read(parent/'SPEC.json')['design'].copy()
    d.update(head_only=False,initialize_epoch_zero=True,monitor_mode='validation',
             matched_budget=False,max_epochs=2000,patience=20,min_delta=1e-8,diagnostics=True)
    pinned={str(p):file_sha(p) for p in (parent/'SPEC.json',parent/'RISK.json',bundle/'MANIFEST.json',generated/'DONE.json')}
    for directory in ('app','agent_plugins','env_plugins','tools'):
        pinned.update({str(p):file_sha(p) for p in (source/directory).rglob('*.py')})
    return dict(schema='g3_training_contrast.v1',design=d,pins=pinned,seed=7,
                initial_checkpoint=file_sha(bundle/'model.pt'),
                trainability='all original online parameters; target parameters and alpha follow fresh model defaults',
                optimizer='same inherited optimizer state in both arms',
                budget='control early stopping determines exact augmentation epochs/updates',
                sampling='real-only vs cumulatively exact 90/10; separate native accounts')


def validate(parent,bundle,generated,binding):
    if any(file_sha(Path(p))!=s for p,s in binding['pins'].items()):raise ValueError('G3_TRAINING_PIN_CHANGED')
    spec=read(parent/'SPEC.json')
    verify_files(parent,{k:v for k,v in spec['files'].items() if k.startswith(('data/','engine/')) or k in ('SUPPORT_RECIPE.json','RISK.json')})
    verify_files(bundle,read(bundle/'MANIFEST.json')['files'])
    done=read(generated/'DONE.json');verify_files(generated,done['files'])
    if read(generated/'RESULT.json')['status']!='GENERATOR_QUALIFIED':raise ValueError('G3_GENERATOR_NOT_QUALIFIED')


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding=make_binding(args.parent,args.bundle,args.generated)
        binding['identity']=identity(binding)
        if (root/'SPEC.json').exists():
            if read(root/'SPEC.json')!=binding:raise ValueError('G3_CONTRAST_RESUME_IDENTITY')
        else:write(root/'SPEC.json',binding)
        validate(args.parent,args.bundle,args.generated,binding)
        if not torch.cuda.is_available() and not args.preflight:raise ValueError('G3_CUDA_REQUIRED')
        device='cpu' if args.preflight else 'cuda'
        configure(binding['seed']);torch.backends.cudnn.allow_tf32=False
        recipe=read(args.bundle/'RECIPE.json')
        if set(recipe['split']['periods'])!={'TRAIN','VALIDATION'}:raise ValueError('G3_EVAL_PERIODS')
        market=market_for(args.parent,read(args.parent/'SPEC.json'),recipe)
        bars=pd.read_parquet(args.generated/'SYNTHETIC_BARS.parquet').set_index('timestamp')
        features=pd.read_parquet(args.generated/'SYNTHETIC_FEATURES.parquet').set_index('timestamp')
        synmarket=synthetic_market(market,bars,features)
        risk=read(args.parent/'RISK.json');interval=recipe['split']['periods']['TRAIN']
        d=binding['design'];seed=binding['seed']
        ix=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
        probe=market.inputs.windows[ix[np.linspace(0,len(ix)-1,32,dtype=int)]].copy()
        model,_=load_bundle(args.bundle,device)
        # Restore fresh online trainability, not head-only restrictions of old refinements.
        from app.structural_refit import build_model
        fresh=build_model(recipe,model.effective,seed,device)
        trainability={k:p.requires_grad for k,p in fresh.loss.named_parameters()};del fresh
        preflight=dict(identity=binding['identity'],features=len(recipe['list']['columns']),
                       real_support=support_hash(market.inputs),synthetic_support=support_hash(synmarket.inputs),
                       initial_metrics=assess_full(model,market,risk,d),
                       online_trainability=trainability,device=device)
        write(root/'PREFLIGHT.json',preflight)
        if args.preflight:
            collector=NativeCollector(model,lambda:environment(synmarket,risk,d['horizon'],interval,model.effective['gamma']),seed)
            collector.reset()
            items=[collector.next() for _ in range(4)]
            if not all(torch.isfinite(x['next','reward']).all() for x in items):raise ValueError('G3_NATIVE_COLLECTOR_REWARD')
            write(root/'SMOKE.json',dict(passed=True,transitions=len(items),identity=binding['identity']))
            return
        model=None
        for arm in ('real_only','real90_synthetic10'):
            output=root/arm;output.mkdir(exist_ok=True)
            if (output/'RESULT.json').exists():continue
            validate(args.parent,args.bundle,args.generated,binding)
            np.random.seed(seed);torch.manual_seed(seed)
            model,_=load_bundle(args.bundle,device)
            for name,param in model.loss.named_parameters():param.requires_grad_(trainability[name])
            np.random.seed(seed);torch.manual_seed(seed)
            design=d.copy()
            if arm!='real_only':
                control=read(root/'real_only'/'TRAINING.json')
                design.update(max_epochs=control['epochs'],matched_budget=True)
            write(output/'DESIGN.json',design)
            collector=None;synthetic=None
            if arm!='real_only':
                synthetic=TensorDictReplayBuffer(storage=LazyMemmapStorage(design['replay_size'],
                    scratch_dir=str(output/'synthetic_replay'),existsok=True,auto_cleanup=False),batch_size=design['batch_size'])
                collector=NativeCollector(model,lambda:environment(synmarket,risk,design['horizon'],interval,model.effective['gamma']),seed+100)
            factory=lambda real:JointReplay(real,synthetic,design['batch_size'],seed+200,collector)
            state=fit_epochs(model,lambda:environment(market,risk,design['horizon'],interval,model.effective['gamma']),
                lambda m:assess_full(m,market,risk,design),design,seed,output,binding['identity']+':'+arm,probe,
                replay_factory=factory)
            metrics=assess_full(model,market,risk,design,save=output/'final',diagnose=True)
            write(output/'RESULT.json',dict(identity=binding['identity'],arm=arm,epochs=state['epoch'],
                updates=state['updates'],steps=state['steps'],best=state['best_bundle'],metrics=metrics,
                replay_mixture=state['history'][-1]['replay_mixture'],promoted=False))
            model=collector=synthetic=None
        a,b=(read(root/x/'RESULT.json') for x in ('real_only','real90_synthetic10'))
        if any(a[k]!=b[k] for k in ('epochs','steps','updates')):raise ValueError('G3_UNMATCHED_BUDGET')
        validate(args.parent,args.bundle,args.generated,binding)
        write(root/'RESULT.json',dict(identity=binding['identity'],status='MATCHED_CONTRAST_COMPLETE',
            arms=[a,b],model_promoted=False,stl_resumed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','generated','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    run(p.parse_args())
