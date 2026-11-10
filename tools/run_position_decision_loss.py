"""Paired decision-loss contrasts on authenticated frozen position fields."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.position_value_surface import NODES,query_weights,GapSurfaceHeads,evaluate_surface
from app.position_decision_loss import fit_loss_constants,partner_schedule,decision_loss
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
from tools.run_surface_market_control import fit_constant_embeddings

ROOT=Path(__file__).resolve().parents[1]


def decision_description(scores,records,banks,outside):
    report=describe(scores,records,banks,outside)
    for groups in report.values():
        for metrics in groups.values():
            if metrics is not None:metrics.pop('gap_rmse')
    return report


def run(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':raise ValueError('DECISION_GPU')
    primary=verify_collection(args.primary);pc=read(args.primary/'CONFIG.json')
    constant=verify_collection(args.constant);cc=read(args.constant/'CONFIG.json')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    probe=verify_collection(args.probe);bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    if (pc['source']!=spec['identity'] or pc['probe']!=probe['identity']
            or cc['primary']!=primary['identity'] or cc['source']!=spec['identity']
            or pc['source_sha256']!=file_sha(bundle/'model.pt')
            or pc['calibrated_sha256']!=file_sha(args.probe/'fresh/CALIBRATED.pt')
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']):raise ValueError('DECISION_SOURCE')
    for contract in (pc,cc):
        for n,sha in contract['sources'].items():
            if file_sha(ROOT/n)!=sha:raise ValueError('DECISION_SOURCE_CODE:'+n)
    config=dict(primary=primary['identity'],constant=constant['identity'],source=spec['identity'],
        source_sha256=pc['source_sha256'],calibrated_sha256=pc['calibrated_sha256'],
        seeds=[19,43],interfaces=['progress9','surface81'],embeddings=['market','constant'],
        objectives=['sign','sign_pair'],updates=4096,batch_size=128,learning_rate=pc['learning_rate'],
        weight_decay=pc['weight_decay'],pair_coefficient=.5,partner_seed_offset=100000,
        score_semantics='decision_score_not_Q_or_probability',actor_updates=0,encoder_updates=0,
        sources={n:file_sha(ROOT/n) for n in ('tools/run_position_decision_loss.py','app/position_decision_loss.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    panel=read(args.primary/'panel/RESULT.json');records=panel['records'];banks=panel['banks'];axes=read(args.primary/'AXES.json')
    fit_banks=[b for b in banks if all(records[i]['role']=='fit' for i in b)]
    if sum(map(len,fit_banks))!=sum(r['role']=='fit' for r in records):raise ValueError('DECISION_FIT_COVERAGE')
    for b in banks:
        if len({(records[i]['row'],records[i]['direction'],records[i]['role']) for i in b})!=1:
            raise ValueError('DECISION_PAIR_MARKET')
    targets=np.array([r['gap'] for r in records]);constants=fit_loss_constants(targets,fit_banks)
    write(args.output/'LOSS_CONSTANTS.json',constants)
    rows=sorted({r['row'] for r in records});lookup={r:i for i,r in enumerate(rows)}
    model,_=load_bundle(bundle,'cuda');model.load(args.probe/'fresh/CALIBRATED.pt')
    initial={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    train=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'));reference=train_logits(model,market.inputs.windows[train])
    embedding,_,errors=cache_embeddings(model,torch.tensor(market.inputs.windows[rows],device='cuda'))
    actual=embedding[:,[lookup[r['row']] for r in records]];const,mean=fit_constant_embeddings(actual,banks,records)
    saved_mean=torch.load(args.constant/'FIT_MEAN_EMBEDDING.pt',map_location='cpu',weights_only=True)
    if not torch.equal(mean.cpu(),saved_mean):raise ValueError('DECISION_CONSTANT_PARITY')
    coords=np.array([r['coords'] for r in records]);dirs=np.array([0 if r['direction']=='long' else 1 for r in records])
    directions=torch.tensor(dirs,device='cuda');target=torch.tensor(targets,device='cuda');arms={};witness={}
    witness_ix=[i for i,r in enumerate(records) if (r['origin'],r['lag'],r['direction'])==('2023Q3:71',4,'short')]
    if len(witness_ix)!=3:raise ValueError('DECISION_WITNESS')
    for seed in config['seeds']:
        schedule=matched_schedule(fit_banks,4096,128,seed);partners=partner_schedule(schedule,fit_banks,seed)
        for mode in config['interfaces']:
            weights,outside=query_weights(coords,dirs,axes,mode);w=torch.tensor(weights,dtype=torch.float32,device='cuda')
            for context,z in (('market',actual),('constant',const)):
                for objective in config['objectives']:
                    key=f'{seed}:{mode}:{context}:{objective}';lane=args.output/f'r{seed}'/mode/context/objective
                    lane.mkdir(parents=True,exist_ok=True)
                    if (lane/'DONE.json').exists():
                        if verify_collection(lane)['identity']!=config['identity']:raise ValueError('DECISION_ARM_IDENTITY')
                        arms[key]=read(lane/'RESULT.json');witness[key]=arms[key]['witness'];continue
                    start=time.monotonic();heads=GapSurfaceHeads(z.shape[-1],NODES[mode],seed,device='cuda')
                    saved=deepcopy(heads.state_dict());torch.save(saved,lane/'INITIAL.pt')
                    np.save(lane/'SCHEDULE.npy',schedule);np.save(lane/'PARTNERS.npy',partners)
                    effective=dict(identity=config['identity'],seed=seed,mode=mode,embedding=context,objective=objective,
                        axes_sha256=digest(axes),initialization_sha256=digest({k:v.cpu().tolist() for k,v in saved.items()}),
                        schedule_sha256=digest(schedule.tolist()),partner_sha256=digest(partners.tolist()),
                        parameters=sum(p.numel() for p in heads.parameters()))
                    prior=read((args.primary if context=='market' else args.constant)/f'r{seed}'/mode/'EFFECTIVE.json')
                    if any(prior[k]!=effective[k] for k in ('axes_sha256','initialization_sha256','schedule_sha256','parameters')):
                        raise ValueError('DECISION_NOT_PAIRED')
                    write(lane/'EFFECTIVE.json',effective);opt=torch.optim.AdamW(heads.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']);points=[]
                    for step in range(4097):
                        if step:
                            ix=schedule[step-1];opt.zero_grad(set_to_none=True)
                            score=evaluate_surface(heads(z[:,ix]),directions[ix],w[ix]);other=partners[step-1]
                            paired=evaluate_surface(heads(z[:,other]),directions[other],w[other]) if objective=='sign_pair' else None
                            loss=decision_loss(score,target[ix],constants,objective,paired,target[other])
                            if not torch.isfinite(loss):raise ValueError('DECISION_NONFINITE')
                            loss.backward();torch.nn.utils.clip_grad_norm_(heads.parameters(),10.,error_if_nonfinite=True);opt.step()
                        if step in POINTS:
                            pred=predictions(heads,z,directions,w)
                            points.append(dict(updates=step,metrics=decision_description(pred.mean(0),records,banks,outside)))
                            write(lane/'CURVE.json',points);progress('FIT',arm=key,completed=len(arms),total=16,updates=step)
                    np.save(lane/'PREDICTIONS.npy',pred);torch.save(dict(state=heads.state_dict(),effective=effective),lane/'HEADS.pt')
                    reloaded=GapSurfaceHeads(z.shape[-1],NODES[mode],seed,device='cuda')
                    reloaded.load_state_dict(torch.load(lane/'HEADS.pt',weights_only=False)['state'])
                    if not np.array_equal(pred,predictions(reloaded,z,directions,w)):raise ValueError('DECISION_RELOAD')
                    if all(torch.equal(v,saved[k]) for k,v in heads.state_dict().items()):raise ValueError('DECISION_NO_UPDATE')
                    witnessed=dict(scores=pred[:,witness_ix].tolist(),close=(pred[:,witness_ix].mean(0)>1e-8).tolist(),
                        target_close=(targets[witness_ix]>1e-8).tolist())
                    witnessed['passed']=witnessed['close']==witnessed['target_close'];witness[key]=witnessed
                    record=dict(effective=effective,points=points,reload_exact=True,witness=witnessed,
                        twins=[decision_description(v,records,banks,outside) for v in pred])
                    write(lane/'RESULT.json',record);seal(lane,config['identity'],start);arms[key]=record
    if (any(not torch.equal(v,initial[k]) for k,v in model.loss.state_dict().items())
            or not np.array_equal(reference,train_logits(model,market.inputs.windows[train]))
            or file_sha(bundle/'model.pt')!=config['source_sha256']
            or file_sha(args.probe/'fresh/CALIBRATED.pt')!=config['calibrated_sha256']):raise ValueError('DECISION_SOURCE_CHANGED')
    result=dict(identity=config['identity'],arms=arms,actor_exact=True,source_exact=True,embedding_max_error=errors,
        records=len(records),banks=len(banks),model_promoted=False,score_semantics=config['score_semantics'])
    write(args.output/'RESULT.json',result);write(args.output/'WITNESS.json',witness);export(args.output,arms)
    for mode in config['interfaces']:
        for context in config['embeddings']:
            folder=args.output/'plots'/mode/context;folder.mkdir(parents=True,exist_ok=True)
            export(folder,{k:v for k,v in arms.items() if v['effective']['mode']==mode and v['effective']['embedding']==context})
    seal(args.output,config['identity'],started);progress('COMPLETE',completed=16,total=16)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('run','collect'))
    for name in ('source','source-runtime','primary','constant','probe'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-host');args=p.parse_args()
    if args.stage=='collect':collect(args);return
    if not all((args.source,args.source_runtime,args.primary,args.constant,args.probe)):p.error('source paths required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
