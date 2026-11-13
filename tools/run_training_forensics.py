"""Autonomous frozen-head reproducibility, optimizer and known-teacher diagnosis."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from app.critic_blocked_stopping import RegretStopper
from app.critic_fit_probe import fit_metrics
from app.critic_native_capacity import native_from_ensemble
from app.critic_target_contract import contract_loss
from app.frozen_recipe_inputs import file_sha
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest,support_hash
from app.role_routed_critic import RoleHeads
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.training_forensics import tree_digest,rng_state,restore_rng,gradient_geometry,sinusoid_lags,balance_teacher
from tools.audit_fixed_actor_critics import covering_schedule
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_role_routed_critic import cache_embeddings

ROOT=Path(__file__).resolve().parents[1]
DENSE={0,1,2,4,8,16,32,64}


def template(device):
    return nn.Sequential(nn.Linear(208,8),nn.SiLU(),nn.Linear(8,3)).to(device)


def checkpoint(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def prepare(a,cfg):
    out=a.output/'cache'
    if (out/'DONE.json').exists():verify_done(out,cfg['identity']);return
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    model,_=load_bundle(a.bundle,'cuda');market=market_for(a.parent,spec,model.recipe)
    pc=read(a.previous/'CONFIG.json');records=read(a.labels/'panel/RESULT.json')['records']
    if (file_sha(a.bundle/'model.pt')!=pc['source_sha256'] or support_hash(market.inputs)!=pc['support']
            or model.recipe['list']['n']!=71):raise ValueError('FORENSIC_SOURCE')
    flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    frozen={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    train=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    logits=emit_actor_reference(model,market.inputs,train,flags)[0][train].copy()
    rows=np.array([r['row'] for r in records])
    if len(rows)!=239 or any(not market.inputs.valid[i] or market.inputs.period[i]!='TRAIN' for i in rows):
        raise ValueError('FORENSIC_ROWS')
    z,t,errors=cache_embeddings(model,torch.tensor(market.inputs.windows[rows],device='cuda'))
    if repr(t)!=repr(template('cuda')):raise ValueError('FORENSIC_HEAD_ARCHITECTURE')
    targets=np.array([[o['reward_return'] for o in r['outcomes']] for r in records])
    initial={};replays=[]
    for seed in (19,43):
        original=native_from_ensemble(RoleHeads(t,3,seed))
        initial[seed]={k:v.cpu().clone() for k,v in original.state_dict().items()}
        for fold in ('A','B'):
            d=a.previous/f'{fold}_{seed}_0.01';r=read(d/'RESULT.json')
            masks=read(a.labels/f'FOLD_{fold}.json')
            for end,name in (('stopped','SELECTED.pt'),('fixed','FIXED.pt')):
                original.load_state_dict(torch.load(d/name,weights_only=True)['state'])
                with torch.no_grad():q=original(z).mean(2).cpu().numpy()
                if {k:fit_metrics(q[:,ix],targets[ix]) for k,ix in masks.items()}!=r[end]:
                    raise ValueError('FORENSIC_CACHE_REPLAY')
                replays.append(dict(fold=fold,seed=seed,endpoint=end,exact=True))
    if (any(not torch.equal(v,frozen[k]) for k,v in model.loss.state_dict().items()) or
            not np.array_equal(logits,emit_actor_reference(model,market.inputs,train,flags)[0][train])):
        raise ValueError('FORENSIC_SOURCE_MUTATED')
    checkpoint(out/'DATA.pt',dict(z=z.cpu(),targets=torch.from_numpy(targets),initial=initial))
    write(out/'VERIFY.json',dict(source_exact=True,actor_exact=True,train_rows=len(train),cache_errors=errors,
        checkpoint_replays=replays,records_sha256=digest(records),source_sha256=pc['source_sha256']))
    for fold in ('A','B'):write(out/f'FOLD_{fold}.json',read(a.labels/f'FOLD_{fold}.json'))
    seal(out,cfg['identity'])


def worker(a):
    configure(318);random.seed(318);torch.backends.cudnn.allow_tf32=False
    cfg=read(a.output/'CONFIG.json');verify_done(a.output/'cache',cfg['identity'])
    task=read(a.task);out=a.output/'units'/task['name'];out.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(parent=cfg['identity'],task=task))
    if (out/'DONE.json').exists():verify_done(out,identity);return
    write(out/'TASK.json',dict(identity=identity,task=task))
    data=torch.load(a.output/'cache/DATA.pt',weights_only=True)
    z=data['z'].to('cuda');targets=data['targets'].numpy()
    masks={k:np.array(v) for k,v in read(a.output/f"cache/FOLD_{task['fold']}.json").items() if k in ('fit','stop','audit')}
    model=RoleHeads(template('cuda'),1,task['seed']);model.load_state_dict(data['initial'][task['seed']])
    teacher_residual=None;teacher_balance=None;teacher_counts=None
    if task['target']!='real':
        if task['target'].endswith('sinusoid'):
            signal=torch.from_numpy(sinusoid_lags(768)).to('cuda');z=torch.stack([signal,signal])
            masks=dict(fit=np.arange(384),stop=np.arange(384,576),audit=np.arange(576,768))
        teacher=RoleHeads(template('cuda'),1,271)
        teacher.twins[1][0].load_state_dict(teacher.twins[0][0].state_dict())
        if task['target'].startswith('balanced'):teacher_balance=balance_teacher(teacher,z,masks['fit'])
        with torch.no_grad():
            tq=teacher(z).mean(2);targets=tq[0].cpu().numpy().astype(float)
            teacher_residual=float((tq-tq[0]).abs().max())
            teacher_counts={k:tq[0,ix].argmax(-1).bincount(minlength=3).cpu().tolist() for k,ix in masks.items()}
        if teacher_residual>1e-5:raise ValueError('FORENSIC_TEACHER_NOT_REALIZABLE')
        del teacher
    y=torch.tensor(targets,dtype=torch.float32,device='cuda');initial=deepcopy(model.state_dict())
    with torch.no_grad():
        initial_q=model(z).mean(2).detach()
        initial_hidden=torch.stack([m[0][1](m[0][0](z[i])) for i,m in enumerate(model.twins)])
    schedule=masks['fit'][covering_schedule(len(masks['fit']),task['updates'],128,task['seed'])]
    opt=torch.optim.AdamW(model.parameters(),lr=task['lr'],weight_decay=.01)
    stopper=RegretStopper(20,1e-8);curve=[];trace=[];step0=0;clipped=0;best=None
    latest=out/'LATEST.pt'
    if latest.exists():
        saved=torch.load(latest,weights_only=False)
        if saved['identity']!=identity:raise ValueError('FORENSIC_RESTART_IDENTITY')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        stopper.__dict__.update(saved['stopper']);restore_rng(saved['rng'])
        step0=saved['step'];clipped=saved['clipped'];best=saved['best']
        curve=saved['curve'];trace=saved['trace']
    limit=min(task['updates'],a.until or task['updates']);begun=time.monotonic()
    for step in range(step0,limit+1):
        if step==step0 and latest.exists():continue
        if step:
            ix=schedule[step-1];opt.zero_grad(set_to_none=True)
            loss=contract_loss(model(z[:,ix]).mean(2),y[ix],'mse');loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            clipped+=int(float(gn)>10)
            if task.get('trace'):
                trace.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(gn),
                    gradient_sha256=tree_digest([p.grad for p in model.parameters()])))
            # Save one actual update's parameter ratios at measurement steps.
            measure=step in DENSE or step%64==0 or step==limit
            before={k:v.detach().clone() for k,v in model.named_parameters()} if measure else None
            opt.step()
        else:measure=True;before=None
        if not measure:continue
        with torch.no_grad():
            q=model(z).mean(2);values=q.cpu().numpy()
            hidden=torch.stack([m[0][1](m[0][0](z[i])) for i,m in enumerate(model.twins)])
            metrics={k:fit_metrics(values[:,ix],targets[ix])['minimum'] for k,ix in masks.items()}
            sorted_q=q.min(0).values.sort(-1).values
            margins=sorted_q[:,-1]-sorted_q[:,-2]
            drift={k:float((v-initial[k]).norm()/initial[k].norm().clamp_min(1e-12)) for k,v in model.state_dict().items()}
            ratios={} if before is None else {k:float((v-before[k]).norm()/before[k].norm().clamp_min(1e-12)) for k,v in model.named_parameters()}
        if step and step%64==0 and stopper.observe(step,metrics['stop']['mean_regret']):
            best=deepcopy(model.state_dict());checkpoint(out/'SELECTED.pt',dict(state=best,step=step))
        row=dict(step=step,metrics=metrics,geometry=gradient_geometry(model,z[:,masks['fit']],y[masks['fit']]),
            hidden_relative_drift=float((hidden-initial_hidden).norm()/initial_hidden.norm().clamp_min(1e-12)),
            greedy_flips_from_initial=int((q.min(0).values.argmax(-1)!=initial_q.min(0).values.argmax(-1)).sum()),
            near_ties=int((margins<1e-5).sum()),mean_margin=float(margins.mean()),
            layer_relative_drift=drift,update_weight_ratio=ratios,clipped_updates=clipped,
            selected_step=stopper.best_step,stopped_at=stopper.stopped_at)
        row['optimizer_moment_norms']={n:{k:float(v.norm()) for k,v in opt.state[p].items() if k in ('exp_avg','exp_avg_sq')}
            for n,p in model.named_parameters()}
        save_full=step<=8192 or step%1024==0 or step==limit
        if save_full:
            snap=dict(identity=identity,step=step,model=model.state_dict(),optimizer=opt.state_dict(),
                rng=rng_state(),stopper=vars(stopper).copy(),clipped=clipped,best=best)
            snap['preupdate_gradients']={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
            row['model_sha256']=tree_digest(snap['model']);row['optimizer_sha256']=tree_digest(snap['optimizer'])
            row['rng_sha256']=tree_digest(snap['rng']);row['outputs_sha256']=tree_digest(q)
            checkpoint(out/'checkpoints'/f'{step:06d}.pt',snap)
        curve.append(row)
        if save_full:
            checkpoint(latest,dict(**snap,curve=curve,trace=trace))
            write(out/'CURVE.json',curve)
            write(out/'PROGRESS.json',dict(step=step,total=task['updates'],seconds=time.monotonic()-begun,at=now()))
    if limit<task['updates']:return
    cold=RoleHeads(template('cuda'),1,task['seed']);cold.load_state_dict(torch.load(latest,weights_only=False)['model'])
    with torch.no_grad():
        if not torch.equal(cold(z),model(z)):raise ValueError('FORENSIC_COLD_RELOAD')
    constant=np.tile(targets[masks['fit']].mean(0),(len(masks['audit']),1))
    interventions=[]
    if task['target']=='real' and not task.get('trace'):
        late=deepcopy(model.state_dict())
        for direction,base,donor in (('early_with_late_layer',best,late),('late_with_early_layer',late,best)):
            for twin in (0,1):
                for layer in (0,2):
                    state=deepcopy(base);prefix=f'twins.{twin}.0.{layer}.'
                    for k in state:
                        if k.startswith(prefix):state[k]=donor[k]
                    cold.load_state_dict(state)
                    with torch.no_grad():q=cold(z).mean(2).cpu().numpy()
                    interventions.append(dict(direction=direction,twin=twin,layer=layer,
                        metrics={k:fit_metrics(q[:,ix],targets[ix])['minimum'] for k,ix in masks.items()}))
        write(out/'LAYER_INTERVENTIONS.json',interventions)
    result=dict(identity=identity,task=task,selected_step=stopper.best_step,stopped_at=stopper.stopped_at,
        checkpoints=len(list((out/'checkpoints').glob('*.pt'))),teacher_residual=teacher_residual,
        audit_constant_rmse=float(np.sqrt(np.mean((constant-targets[masks['audit']])**2))),
        schedule_sha256=tree_digest(schedule),cold_exact=True,clipped_updates=clipped,
        initial=curve[0]['metrics'],final=curve[-1]['metrics'],teacher_balance=teacher_balance,teacher_counts=teacher_counts)
    write(out/'TRACE.json',trace);write(out/'RESULT.json',result);seal(out,identity)


def compare_runs(root,names):
    ref=root/'units'/names[0];differences=[]
    for name in names[1:]:
        d=root/'units'/name
        for file in ('TRACE.json','CURVE.json'):
            a=read(ref/file);b=read(d/file)
            if len(a)!=len(b):differences.append(dict(unit=name,file=file,reason='length'))
            for i,(x,y) in enumerate(zip(a,b)):
                if x!=y:
                    differences.append(dict(unit=name,file=file,index=i,step=x.get('step')));break
        # The identity is intentionally different; all optimizer/model/RNG values must match.
        a=torch.load(ref/'LATEST.pt',map_location='cpu',weights_only=False)
        b=torch.load(d/'LATEST.pt',map_location='cpu',weights_only=False)
        for key in ('model','optimizer','rng','stopper','best'):
            if tree_digest(a[key])!=tree_digest(b[key]):differences.append(dict(unit=name,key=key))
    return dict(exact=not differences,differences=differences,units=names)


def summarize(a,cfg,tasks,gate):
    results=[];lines=['# Training Forensics Results','','All scores below are internal TRAIN-block critic diagnostics, not trading returns.','',
        '| Unit | Initial stop regret | Selected step | Final FIT regret | Final stop regret | Final audit regret |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for task in tasks:
        d=a.output/'units'/task['name'];r=read(d/'RESULT.json');verify_done(d,r['identity'])
        c=read(d/'CURVE.json');results.append(r)
        lines.append(f"| {task['name']} | {r['initial']['stop']['mean_regret']:.6f} | {r['selected_step']} | {r['final']['fit']['mean_regret']:.6f} | {r['final']['stop']['mean_regret']:.6f} | {r['final']['audit']['mean_regret']:.6f} |")
        if task['target']=='real' and not task.get('trace'):
            ax=axes.flat[(0 if task['fold']=='A' else 2)+(0 if task['seed']==19 else 1)]
            x=[v['step'] for v in c]
            ax.plot(x,[v['metrics']['stop']['mean_regret'] for v in c],label=f"lr={task['lr']:.0e} stop")
            ax.plot(x,[v['metrics']['fit']['mean_regret'] for v in c],linestyle='--',label=f"lr={task['lr']:.0e} fit")
            ax.set(xscale='symlog',xlabel='Updates',ylabel='Action regret',title=f"{task['fold']}/{task['seed']}")
            ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(a.output/'TRAINING_CURVES.png',dpi=140);plt.close(fig)
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],reproducibility=gate,units=results,
        source_preserved=True,actor_updates=0,encoder_updates=0,model_promoted=False,
        validation_read=False,test_read=False,stl_included=False))


def run(a):
    begun=time.monotonic();configure(318);random.seed(318);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('FORENSIC_DEVICE')
    pc=read(a.previous/'CONFIG.json');verify_done(a.previous,pc['identity'])
    lc=read(a.labels/'CONFIG.json');verify_done(a.labels,lc['identity'])
    if pc['previous']!=lc['identity']:raise ValueError('FORENSIC_LABEL_LINEAGE')
    cfg=dict(schema='training_forensics.v1',previous=pc['identity'],labels=lc['identity'],
        source_sha256=file_sha(a.bundle/'model.pt'),budgets=[4096,8192,81920],lrs=[3e-5,3e-6],
        code={k:file_sha(ROOT/k) for k in ('tools/run_training_forensics.py','app/training_forensics.py',
            'app/critic_native_capacity.py','app/critic_blocked_stopping.py','app/critic_target_contract.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('FORENSIC_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg);prepare(a,cfg)
    tasks=[]
    def execute(task,until=None):
        path=a.output/'tasks'/f"{task['name']}.json";write(path,task)
        write(a.output/'PROGRESS.json',dict(stage='RUNNING',unit=task['name'],completed=len(tasks),at=now(),seconds=time.monotonic()-begun))
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--output',str(a.output),'--task',str(path)]
        if until:cmd+=['--until',str(until)]
        with (a.output/'WORKERS.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    base=dict(fold='B',seed=19,lr=pc['lr'],updates=4096,target='real',trace=True)
    for name in ('repeat1','repeat2','restart'):
        task=dict(**base,name=name)
        if name=='restart' and not (a.output/'units/restart/LATEST.pt').exists():execute(task,512)
        execute(task);tasks.append(task)
    gate=compare_runs(a.output,['repeat1','repeat2','restart'])
    original=torch.load(a.previous/'B_19_0.01/FIXED.pt',map_location='cpu',weights_only=True)['state']
    repeat=torch.load(a.output/'units/repeat1/LATEST.pt',map_location='cpu',weights_only=False)['model']
    gate['matches320_fixed']=tree_digest(original)==tree_digest(repeat)
    write(a.output/'REPRODUCIBILITY.json',gate)
    if not gate['exact'] or not gate['matches320_fixed']:raise ValueError('FORENSIC_REPRODUCIBILITY_GATE')
    for fold in ('A','B'):
        for seed in (19,43):
            for rate,budget,label in ((pc['lr'],8192,'base'),(pc['lr']/10,81920,'low')):
                task=dict(name=f'{fold}_{seed}_{label}',fold=fold,seed=seed,lr=rate,updates=budget,target='real')
                execute(task);tasks.append(task)
            task=dict(name=f'{fold}_{seed}_teacher',fold=fold,seed=seed,lr=pc['lr'],updates=8192,target='teacher')
            execute(task);tasks.append(task)
    task=dict(name='sinusoid_teacher',fold='A',seed=19,lr=pc['lr'],updates=8192,target='sinusoid')
    execute(task);tasks.append(task)
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('FORENSIC_SOURCE_CHANGED')
    summarize(a,cfg,tasks,gate);seal(a.output,cfg['identity'])
    write(a.output/'PROGRESS.json',dict(stage='COMPLETE',completed=len(tasks),seconds=time.monotonic()-begun,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','previous','labels','task'):p.add_argument('--'+name,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--until',type=int);p.add_argument('--expected-device-name');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.worker:worker(a)
    else:
        with (a.output/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:run(a)
            except Exception as exc:write(a.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise
