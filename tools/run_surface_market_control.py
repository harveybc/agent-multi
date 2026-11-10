"""Post-primary paired constant-market control for position-value surfaces."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from app.frozen_recipe_inputs import file_sha
from app.position_value_surface import NODES,query_weights,GapSurfaceHeads,evaluate_surface
from app.rl_subset import digest,support_hash
from app.role_routed_critic import matched_schedule
from app.structural_refit import load_bundle
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_action_ranking_contrast import ensure_config,seal
from tools.run_actor_preservation import collect
from tools.run_critic_fit_probe import train_logits
from tools.run_discrete_curve_probe import configure,now,read,write
from tools.run_position_value_surface import describe,predictions,export
from tools.run_role_routed_critic import cache_embeddings
from tools.run_stratified_critic import POINTS
from tools.run_structural_sac_refit import authenticate,market_for

ROOT=Path(__file__).resolve().parents[1]


def fit_constant_embeddings(z,banks,records):
    fit=[b for b in banks if all(records[i]['role']=='fit' for i in b)]
    if not fit:raise ValueError('CONSTANT_MARKET_NO_FIT_BANK')
    mean=torch.stack([z[:,b].mean(1) for b in fit],dim=1).mean(1,keepdim=True)
    return mean.expand(-1,z.shape[1],-1).contiguous(),mean


def run(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':raise ValueError('MARKET_CONTROL_GPU')
    primary=verify_collection(args.primary);pc=read(args.primary/'CONFIG.json')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    probe=verify_collection(args.probe);bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    if (pc['source']!=spec['identity'] or pc['probe']!=probe['identity']
            or pc['source_sha256']!=file_sha(bundle/'model.pt')
            or pc['calibrated_sha256']!=file_sha(args.probe/'fresh/CALIBRATED.pt')
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']):raise ValueError('MARKET_CONTROL_SOURCE')
    for n,sha in pc['sources'].items():
        if file_sha(ROOT/n)!=sha:raise ValueError('MARKET_CONTROL_SOURCE_CODE:'+n)
    config=dict(primary=primary['identity'],source=spec['identity'],seeds=[19,43],arms=['progress9','surface81'],
        source_sha256=pc['source_sha256'],calibrated_sha256=pc['calibrated_sha256'],
        updates=4096,batch_size=128,learning_rate=pc['learning_rate'],weight_decay=pc['weight_decay'],
        control='fit_bank_balanced_constant_embedding',post_primary_design=True,model_promoted=False,
        sources={'tools/run_surface_market_control.py':file_sha(ROOT/'tools/run_surface_market_control.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    panel=read(args.primary/'panel/RESULT.json');records=panel['records'];banks=panel['banks']
    axes=read(args.primary/'AXES.json');rows=sorted({r['row'] for r in records});lookup={r:i for i,r in enumerate(rows)}
    model,_=load_bundle(bundle,'cuda');model.load(args.probe/'fresh/CALIBRATED.pt')
    initial={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    train=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'));reference=train_logits(model,market.inputs.windows[train])
    embeddings,_,errors=cache_embeddings(model,torch.tensor(market.inputs.windows[rows],device='cuda'))
    z=embeddings[:,[lookup[r['row']] for r in records]];z,mean=fit_constant_embeddings(z,banks,records)
    torch.save(mean.cpu(),args.output/'FIT_MEAN_EMBEDDING.pt')
    coords=np.array([r['coords'] for r in records]);dirs=np.array([0 if r['direction']=='long' else 1 for r in records])
    directions=torch.tensor(dirs,device='cuda');target=torch.tensor([r['gap'] for r in records],device='cuda')
    fit_banks=[b for b in banks if records[b[0]]['role']=='fit'];arms={}
    for seed in config['seeds']:
        schedule=matched_schedule(fit_banks,4096,128,seed)
        for mode in config['arms']:
            lane=args.output/f'r{seed}'/mode;lane.mkdir(parents=True,exist_ok=True)
            if (lane/'DONE.json').exists():
                if verify_collection(lane)['identity']!=config['identity']:raise ValueError('MARKET_CONTROL_ARM_IDENTITY')
                arms[f'{seed}:{mode}']=read(lane/'RESULT.json');continue
            start=time.monotonic();nodes=NODES[mode];heads=GapSurfaceHeads(z.shape[-1],nodes,seed,device='cuda')
            saved=deepcopy(heads.state_dict());torch.save(saved,lane/'INITIAL.pt');np.save(lane/'SCHEDULE.npy',schedule)
            weights,outside=query_weights(coords,dirs,axes,mode);w=torch.tensor(weights,dtype=torch.float32,device='cuda')
            effective=dict(identity=config['identity'],seed=seed,mode=mode,control=config['control'],nodes=nodes,
                axes_sha256=digest(axes),initialization_sha256=digest({k:v.cpu().tolist() for k,v in saved.items()}),
                schedule_sha256=digest(schedule.tolist()),parameters=sum(p.numel() for p in heads.parameters()))
            original=read(args.primary/f'r{seed}'/mode/'EFFECTIVE.json')
            if any(original[k]!=effective[k] for k in ('axes_sha256','initialization_sha256','schedule_sha256','parameters')):
                raise ValueError('MARKET_CONTROL_NOT_PAIRED')
            write(lane/'EFFECTIVE.json',effective);opt=torch.optim.AdamW(heads.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']);points=[]
            for step in range(4097):
                if step:
                    ix=schedule[step-1];opt.zero_grad(set_to_none=True)
                    gap=evaluate_surface(heads(z[:,ix]),directions[ix],w[ix]);loss=F.smooth_l1_loss(gap,target[ix][None].expand_as(gap))
                    if not torch.isfinite(loss):raise ValueError('MARKET_CONTROL_NONFINITE')
                    loss.backward();torch.nn.utils.clip_grad_norm_(heads.parameters(),10.,error_if_nonfinite=True);opt.step()
                if step in POINTS:
                    pred=predictions(heads,z,directions,w);points.append(dict(updates=step,metrics=describe(pred.mean(0),records,banks,outside)))
                    write(lane/'CURVE.json',points);progress('FIT',seed=seed,mode=mode,updates=step)
            np.save(lane/'PREDICTIONS.npy',pred);torch.save(dict(state=heads.state_dict(),effective=effective),lane/'HEADS.pt')
            reload=GapSurfaceHeads(z.shape[-1],nodes,seed,device='cuda');reload.load_state_dict(torch.load(lane/'HEADS.pt',weights_only=False)['state'])
            if not np.array_equal(pred,predictions(reload,z,directions,w)):raise ValueError('MARKET_CONTROL_RELOAD')
            record=dict(effective=effective,points=points,reload_exact=True,
                twins=[describe(v,records,banks,outside) for v in pred])
            write(lane/'RESULT.json',record);seal(lane,config['identity'],start);arms[f'{seed}:{mode}']=record
    if (any(not torch.equal(v,initial[k]) for k,v in model.loss.state_dict().items())
            or not np.array_equal(reference,train_logits(model,market.inputs.windows[train]))
            or file_sha(bundle/'model.pt')!=config['source_sha256']
            or file_sha(args.probe/'fresh/CALIBRATED.pt')!=config['calibrated_sha256']):raise ValueError('MARKET_CONTROL_SOURCE_CHANGED')
    result=dict(identity=config['identity'],arms=arms,actor_exact=True,source_exact=True,embedding_max_error=errors,
        post_primary_design=True,model_promoted=False)
    write(args.output/'RESULT.json',result);export(args.output,arms);seal(args.output,config['identity'],started);progress('COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('run','collect'))
    for name in ('source','source-runtime','primary','probe'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-host');args=p.parse_args()
    if args.stage=='collect':collect(args);return
    if not all((args.source,args.source_runtime,args.primary,args.probe)):p.error('source paths required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
