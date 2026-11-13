"""Four native-head fits against authenticated318 checkpoints, actor frozen."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.critic_blocked_stopping import RegretStopper,fold_indices
from app.critic_fit_probe import fit_metrics
from app.critic_native_capacity import native_from_ensemble
from app.critic_target_contract import contract_loss
from app.frozen_recipe_inputs import file_sha
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest,support_hash
from app.role_routed_critic import RoleHeads
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from tools.audit_fixed_actor_critics import covering_schedule
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_critic_blocked_stopping import FOLDS,report
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_role_routed_critic import cache_embeddings

ROOT=Path(__file__).resolve().parents[1]


def run(a):
    begun=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:
        raise ValueError('CAPACITY_DEVICE')
    pc=read(a.previous/'CONFIG.json');verify_done(a.previous,pc['identity'])
    cc=read(a.coverage/'CONFIG.json');verify_done(a.coverage,cc['identity'])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    source_sha=file_sha(a.bundle/'model.pt')
    if source_sha!=pc['source_sha256'] or cc['identity']!=pc['coverage']:
        raise ValueError('CAPACITY_LINEAGE')
    expected=dict(updates=4096,batch=128,check_every=64,patience=20,min_delta=1e-8,loss='mse',mode='shared_mean3',seeds=[19,43])
    if any(pc[k]!=v for k,v in expected.items()):raise ValueError('CAPACITY_CONTROL_PROTOCOL')
    model,_=load_bundle(a.bundle,'cuda');market=market_for(a.parent,spec,model.recipe)
    if support_hash(market.inputs)!=cc['support'] or model.recipe['list']['n']!=71:
        raise ValueError('CAPACITY_FEATURES')
    if model.effective['learning_rate']!=pc['lr']:raise ValueError('CAPACITY_LR')
    cfg=dict(schema='critic_native_capacity.v1',source_sha256=source_sha,previous=pc['identity'],
        coverage=cc['identity'],support=cc['support'],seeds=pc['seeds'],weight_decays=[.01],
        lr=pc['lr'],updates=4096,batch=128,check_every=64,patience=20,min_delta=1e-8,
        initialization='ensemble_head0_each_twin',code={n:file_sha(ROOT/n) for n in (
            'tools/run_native_critic_capacity.py','app/critic_native_capacity.py','app/critic_blocked_stopping.py',
            'tools/run_role_routed_critic.py','app/role_routed_critic.py','app/critic_target_contract.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('CAPACITY_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):
        write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-begun,**kw))
    progress('PREFLIGHT')
    flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    frozen={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    train=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    logits=emit_actor_reference(model,market.inputs,train,flags)[0][train].copy()
    records=read(a.previous/'panel/RESULT.json')['records']
    if len(records)!=239:raise ValueError('CAPACITY_ANCHORS')
    rows=np.array([r['row'] for r in records])
    if any(not market.inputs.valid[i] or market.inputs.period[i]!='TRAIN' for i in rows):raise ValueError('CAPACITY_NONTRAIN')
    z,template,errors=cache_embeddings(model,torch.tensor(market.inputs.windows[rows],device='cuda'))
    targets=np.array([[o['reward_return'] for o in r['outcomes']] for r in records])
    y=torch.tensor(targets,dtype=torch.float32,device='cuda')
    write(a.output/'CACHE.json',dict(errors=errors,dimensions=list(z.shape),records_sha256=digest(records)))
    results=[];comparisons=[]
    for fold,years,stop_year,audit_year in FOLDS:
        base_masks=fold_indices(records,years,stop_year,audit_year)
        masks={k:np.array(v,dtype=int) for k,v in read(a.previous/f'FOLD_{fold}.json').items()}
        if any(not np.array_equal(v,masks[k]) for k,v in base_masks.items()):raise ValueError('CAPACITY_FOLDS')
        write(a.output/f'FOLD_{fold}.json',{k:v.tolist() for k,v in masks.items()})
        for seed in cfg['seeds']:
            control_dir=a.previous/f'{fold}_{seed}_0.01';verify_done(control_dir,pc['identity'])
            control=read(control_dir/'RESULT.json')
            out=a.output/f'{fold}_{seed}_0.01';out.mkdir(parents=True,exist_ok=True)
            if (out/'DONE.json').exists():
                verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
            ensemble=RoleHeads(template,3,seed).to('cuda')
            init_hash=digest({k:v.cpu().tolist() for k,v in ensemble.state_dict().items()})
            schedule=masks['fit'][covering_schedule(len(masks['fit']),4096,128,seed)]
            if init_hash!=control['initialization_sha256'] or digest(schedule.tolist())!=control['schedule_sha256']:
                raise ValueError('CAPACITY_PAIRING')
            heads=native_from_ensemble(ensemble)
            if sum(p.numel() for p in heads.parameters())!=3398:raise ValueError('CAPACITY_PARAMETER_COUNT')
            with torch.no_grad():initial_delta=float((heads(z).mean(2)-ensemble(z).mean(2)).abs().max())
            # Authenticate every reused endpoint before any new fitting.
            for endpoint,name in (('stopped','SELECTED.pt'),('fixed','FIXED.pt')):
                ensemble.load_state_dict(torch.load(control_dir/name,weights_only=True)['state'])
                with torch.no_grad():q=ensemble(z).mean(2).cpu().numpy()
                replay={k:fit_metrics(q[:,ix],targets[ix]) for k,ix in masks.items()}
                if replay!=control[endpoint]:raise ValueError('CAPACITY_CONTROL_REPLAY')
            del ensemble
            initial=deepcopy(heads.state_dict());opt=torch.optim.AdamW(heads.parameters(),lr=cfg['lr'],weight_decay=.01)
            stopper=RegretStopper(20,1e-8);best=None;curve=[]
            for step in range(4097):
                if step:
                    ix=schedule[step-1];opt.zero_grad(set_to_none=True)
                    loss=contract_loss(heads(z[:,ix]).mean(2),y[ix],'mse');loss.backward()
                    torch.nn.utils.clip_grad_norm_(heads.parameters(),10.,error_if_nonfinite=True);opt.step()
                if step%64==0:
                    with torch.no_grad():
                        fm=fit_metrics(heads(z[:,masks['fit']]).mean(2).cpu().numpy(),targets[masks['fit']])['minimum']
                        sm=fit_metrics(heads(z[:,masks['stop']]).mean(2).cpu().numpy(),targets[masks['stop']])['minimum']
                    if step and stopper.observe(step,sm['mean_regret']):best=deepcopy(heads.state_dict())
                    curve.append(dict(updates=step,fit_regret=fm['mean_regret'],stop_regret=sm['mean_regret'],
                        fit_rmse=fm['rmse'],stop_rmse=sm['rmse'],patience_used=stopper.bad))
                    write(out/'CURVE.json',curve)
                    progress('FIT',fold=fold,seed=seed,updates=step,arms_complete=len(results),arms_total=4)
            if best is None or all(torch.equal(v,initial[k]) for k,v in heads.state_dict().items()):
                raise ValueError('CAPACITY_NO_FIT')
            torch.save(dict(state=best,updates=stopper.best_step),out/'SELECTED.pt')
            torch.save(dict(state=heads.state_dict(),updates=4096),out/'FIXED.pt')
            endpoints={}
            for endpoint,name in (('stopped','SELECTED.pt'),('fixed','FIXED.pt')):
                cold=RoleHeads(template,1,seed).to('cuda');cold.load_state_dict(torch.load(out/name,weights_only=True)['state'])
                reference=RoleHeads(template,1,seed).to('cuda')
                reference.load_state_dict(best if endpoint=='stopped' else heads.state_dict())
                with torch.no_grad():
                    q=cold(z).mean(2)
                    if not torch.equal(q,reference(z).mean(2)):raise ValueError('CAPACITY_COLD_RELOAD')
                    q=q.cpu().numpy()
                endpoints[endpoint]={k:fit_metrics(q[:,ix],targets[ix]) for k,ix in masks.items()}
            result=dict(identity=cfg['identity'],fold=fold,seed=seed,weight_decay=.01,parameters=3398,
                ensemble_initialization_sha256=init_hash,schedule_sha256=digest(schedule.tolist()),
                initial_function_max_delta=initial_delta,control_replay_exact=True,cold_exact=True,
                selected_update=stopper.best_step,stopped_at=stopper.stopped_at,curve=curve,
                baselines=control['baselines'],control_result_sha256=file_sha(control_dir/'RESULT.json'),**endpoints)
            write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
            del heads,cold,reference,opt
    for r in results:
        c=read(a.previous/f"{r['fold']}_{r['seed']}_0.01/RESULT.json")
        values={}
        for endpoint in ('stopped','fixed'):
            values[endpoint]={k:dict(native=r[endpoint][k]['minimum']['mean_regret'],
                control=c[endpoint][k]['minimum']['mean_regret']) for k in ('audit','audit_opening')}
        passed=(values['stopped']['audit']['native']<values['stopped']['audit']['control'] and
            values['stopped']['audit_opening']['native']<=values['stopped']['audit_opening']['control'])
        comparisons.append(dict(fold=r['fold'],seed=r['seed'],passes=passed,**values))
    if (any(not torch.equal(v,frozen[k]) for k,v in model.loss.state_dict().items()) or
            not np.array_equal(emit_actor_reference(model,market.inputs,train,flags)[0][train],logits) or
            file_sha(a.bundle/'model.pt')!=source_sha):raise ValueError('CAPACITY_SOURCE_MUTATED')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],arms=results,comparisons=comparisons,
        consistently_helpful=all(c['passes'] for c in comparisons),source_exact=True,actor_exact=True,
        model_promoted=False,validation_read=False,test_read=False,stl_included=False))
    report(a.output,results,cfg)
    lines=['# Native Versus Ensemble Critic Heads','',
        'Action regret, lower is better; diagnostic only, not trading returns.','',
        '| Fold | Seed | Native stopped | Control stopped | Native fixed | Control fixed | Pass with opening guard |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for c in comparisons:
        s=c['stopped']['audit'];f=c['fixed']['audit']
        lines.append(f"| {c['fold']} | {c['seed']} | {s['native']:.6f} | {s['control']:.6f} | {f['native']:.6f} | {f['control']:.6f} | {c['passes']} |")
    (a.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    seal(a.output,cfg['identity']);progress('COMPLETE',arms_complete=4,arms_total=4)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','coverage','previous','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise
