"""Nested FIT coverage on the unchanged known teacher; no actor or encoder fit."""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from app.critic_fit_probe import fit_metrics
from app.critic_target_contract import contract_loss
from app.frozen_recipe_inputs import file_sha
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest,support_hash
from app.role_routed_critic import RoleHeads
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from app.teacher_coverage import nested_support
from app.training_forensics import balance_teacher,tree_digest,rng_state,restore_rng
from tools.audit_fixed_actor_critics import covering_schedule
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_role_routed_critic import cache_embeddings
from tools.run_training_forensics import template,checkpoint

ROOT=Path(__file__).resolve().parents[1]


def prepare(a,cfg):
    out=a.output/'cache'
    if (out/'DONE.json').exists():verify_done(out,cfg['identity']);return
    old=torch.load(a.previous/'cache/DATA.pt',weights_only=True)
    records=read(a.labels/'panel/RESULT.json')['records'];oldrows=np.array([r['row'] for r in records])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    model,_=load_bundle(a.bundle,'cuda');market=market_for(a.parent,spec,model.recipe)
    if (support_hash(market.inputs)!=cfg['support'] or model.recipe['list']['n']!=71
            or file_sha(a.bundle/'model.pt')!=cfg['source_sha256']):raise ValueError('COVERAGE_SOURCE')
    train=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    flags={k:v.requires_grad for k,v in model.loss.named_parameters()}
    saved={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    logits=emit_actor_reference(model,market.inputs,train,flags)[0][train].copy()
    dates=pd.DatetimeIndex(market.inputs.row_ts_ns,tz='UTC')
    years=dates.year.to_numpy();quarters=np.array([f'{y}Q{q}' for y,q in zip(years,dates.quarter)])
    window_year=pd.DatetimeIndex(market.inputs.slot_ts_ns[:,0],tz='UTC').year.to_numpy()
    folds={};supports={};pools={}
    for fold in ('A','B'):
        masks=read(a.previous/f'cache/FOLD_{fold}.json');folds[fold]=masks
        base=oldrows[masks['fit']];fy=sorted({records[i]['year'] for i in masks['fit']})
        eligible=train[np.isin(years[train],fy)&(window_year[train]==years[train])]
        pool=eligible[~np.isin(eligible,oldrows)]
        supports[fold]=nested_support(base,pool,quarters[pool])
        pools[fold]=dict(eligible=len(eligible),new_candidates=len(pool),fit_years=fy)
    newrows=np.array(sorted(set().union(*[set(v.tolist()) for f in supports.values() for v in f.values()])-set(oldrows.tolist())))
    if not len(newrows) or not set(newrows)<=set(train):raise ValueError('COVERAGE_NEW_ROWS')
    if any(int(market.inputs.row_ts_ns[r['row']])!=r['timestamp'] for r in records):raise ValueError('COVERAGE_ORIGINAL_ROWS')
    rebuilt,_,old_errors=cache_embeddings(model,torch.tensor(market.inputs.windows[oldrows],device='cuda'))
    if not torch.equal(rebuilt.cpu(),old['z']):raise ValueError('COVERAGE_ORIGINAL_CACHE')
    added,_,errors=cache_embeddings(model,torch.tensor(market.inputs.windows[newrows],device='cuda'))
    # Preserve original tensor values exactly; concatenate only real new native windows.
    zold=old['z'].to('cuda');z=torch.cat([zold,added],dim=1)
    rows=np.r_[oldrows,newrows];lookup={int(v):i for i,v in enumerate(rows)}
    targets={};teachers={};metadata={};replays=[]
    for fold in ('A','B'):
        masks={k:np.array(v) for k,v in folds[fold].items() if k in ('fit','stop','audit')}
        teacher=RoleHeads(template('cuda'),1,271)
        teacher.twins[1][0].load_state_dict(teacher.twins[0][0].state_dict())
        balance=balance_teacher(teacher,zold,masks['fit']);teacher.requires_grad_(False)
        teacher_hash=tree_digest(teacher.state_dict())
        with torch.no_grad():
            original=teacher(zold).mean(2);extra=teacher(added).mean(2)
            residual=max(float((original[0]-original[1]).abs().max()),float((extra[0]-extra[1]).abs().max()))
            y=torch.cat([original[0],extra[0]]).cpu().numpy().astype(float)
        if residual>1e-5:raise ValueError('COVERAGE_TEACHER_PARITY')
        targets[fold]=torch.from_numpy(y);teachers[fold]={k:v.cpu() for k,v in teacher.state_dict().items()}
        if teacher_hash!=tree_digest(teacher.state_dict()):raise ValueError('COVERAGE_TEACHER_MUTATED')
        for seed in (19,43):
            d=a.previous/f'units/{fold}_{seed}_balanced';control=read(d/'RESULT.json');verify_done(d,control['identity'])
            if control['teacher_balance']!=balance:raise ValueError('COVERAGE_TEACHER_THRESHOLDS')
            head=RoleHeads(template('cuda'),1,seed)
            for endpoint in ('initial','final'):
                state=old['initial'][seed] if endpoint=='initial' else torch.load(d/'LATEST.pt',weights_only=False)['model']
                head.load_state_dict(state)
                with torch.no_grad():q=head(zold).mean(2).cpu().numpy()
                if {k:fit_metrics(q[:,ix],y[ix])['minimum'] for k,ix in masks.items()}!=control[endpoint]:
                    raise ValueError('COVERAGE_CONTROL_REPLAY')
                replays.append(dict(fold=fold,seed=seed,endpoint=endpoint,exact=True))
        expanded={str(f):[lookup[int(r)] for r in selected] for f,selected in supports[fold].items()}
        for f,ix in expanded.items():
            if set(ix)&(set(masks['stop'])|set(masks['audit'])):raise ValueError('COVERAGE_HELD_LEAK')
            if len(ix)!=int(f)*len(masks['fit']):raise ValueError('COVERAGE_SIZE')
        metadata[fold]=dict(masks={k:v.tolist() for k,v in masks.items()},expanded=expanded,
            teacher_balance=balance,teacher_state_sha256=teacher_hash,teacher_residual=residual,pool=pools[fold],
            expanded_quarter_counts={f:dict(Counter(quarters[rows[ix]])) for f,ix in expanded.items()},
            expanded_action_counts={f:np.bincount(y[ix].argmax(-1),minlength=3).tolist() for f,ix in expanded.items()})
    if (any(not torch.equal(v,saved[k]) for k,v in model.loss.state_dict().items()) or
            not np.array_equal(logits,emit_actor_reference(model,market.inputs,train,flags)[0][train])):
        raise ValueError('COVERAGE_SOURCE_MUTATED')
    checkpoint(out/'DATA.pt',dict(z=z.cpu(),targets=targets,initial=old['initial'],teachers=teachers))
    write(out/'SUPPORT.json',dict(rows=rows.tolist(),old_count=len(oldrows),folds=metadata))
    write(out/'VERIFY.json',dict(source_exact=True,actor_exact=True,original_embeddings_exact=True,
        replays=replays,old_cache_errors=old_errors,new_cache_errors=errors,new_windows=len(newrows),
        source_sha256=cfg['source_sha256'],support_sha256=cfg['support']))
    seal(out,cfg['identity'])


def metrics(head,z,y,fit,masks,nold):
    with torch.no_grad():
        qold=head(z[:,:nold]).mean(2).cpu().numpy()
        qfit=head(z[:,fit]).mean(2).cpu().numpy()
    values={k:fit_metrics(qold[:,ix],y[ix])['minimum'] for k,ix in masks.items()}
    values['expanded_fit']=fit_metrics(qfit,y[fit])['minimum']
    return values


def run(a):
    begun=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('COVERAGE_DEVICE')
    pc=read(a.previous/'CONFIG.json');verify_done(a.previous,pc['identity'])
    lc=read(a.labels/'CONFIG.json');verify_done(a.labels,lc['identity'])
    upstream=read(a.forensics/'CONFIG.json');verify_done(a.forensics,upstream['identity'])
    if pc['previous']!=upstream['identity'] or upstream['labels']!=lc['identity']:raise ValueError('COVERAGE_LINEAGE')
    prior_source=read(a.forensics/'cache/VERIFY.json')['source_sha256']
    if file_sha(a.bundle/'model.pt')!=prior_source:raise ValueError('COVERAGE_BUNDLE')
    # Reuse the support identity already authenticated by the original frozen-head series.
    source_cfg=read(a.native/'CONFIG.json');verify_done(a.native,source_cfg['identity'])
    if source_cfg['identity']!=upstream['previous']:raise ValueError('COVERAGE_NATIVE_LINEAGE')
    cfg=dict(schema='known_teacher_coverage.v1',previous=pc['identity'],forensics=upstream['identity'],
        labels=lc['identity'],source_sha256=prior_source,support=source_cfg['support'],
        factors=[4,16],seeds=[19,43],lr=pc['lr'],weight_decay=.01,base_updates=8192,batch=128,sampling_seed=323,
        primary='fixed8192',secondary='same_nominal_presentations_per_example',
        code={k:file_sha(ROOT/k) for k in ('tools/run_teacher_coverage.py','app/teacher_coverage.py',
            'app/training_forensics.py','app/critic_target_contract.py','tools/run_role_routed_critic.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('COVERAGE_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-begun,**kw))
    progress('PREPARE');prepare(a,cfg)
    data=torch.load(a.output/'cache/DATA.pt',weights_only=True);support=read(a.output/'cache/SUPPORT.json')
    z=data['z'].to('cuda');nold=support['old_count'];results=[]
    for fold in ('A','B'):
        info=support['folds'][fold];masks={k:np.array(v) for k,v in info['masks'].items()}
        targets=data['targets'][fold].numpy();y=torch.tensor(targets,dtype=torch.float32,device='cuda')
        for seed in cfg['seeds']:
            for factor in cfg['factors']:
                out=a.output/'units'/f'{fold}_{seed}_{factor}';out.mkdir(parents=True,exist_ok=True)
                if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
                fit=np.array(info['expanded'][str(factor)]);budget=8192*factor
                head=RoleHeads(template('cuda'),1,seed);head.load_state_dict(data['initial'][seed])
                if tree_digest(head.state_dict())!=tree_digest(data['initial'][seed]):raise ValueError('COVERAGE_INITIALIZATION')
                opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=.01)
                schedule=fit[covering_schedule(len(fit),budget,128,seed)]
                curve=[];step0=0;clipped=0
                latest=out/'LATEST.pt'
                if latest.exists():
                    saved=torch.load(latest,weights_only=False)
                    if saved['identity']!=cfg['identity']:raise ValueError('COVERAGE_ARM_RESUME')
                    head.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
                    step0=saved['step'];curve=saved['curve'];clipped=saved['clipped']
                progress('FIT',fold=fold,seed=seed,factor=factor,step=step0,total=budget,completed=len(results),arms=8)
                for step in range(step0,budget+1):
                    if step==step0 and latest.exists():continue
                    if step:
                        ix=schedule[step-1];opt.zero_grad(set_to_none=True)
                        loss=contract_loss(head(z[:,ix]).mean(2),y[ix],'mse');loss.backward()
                        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True)
                        clipped+=int(float(norm)>10);opt.step()
                    if step not in (0,64,256,1024) and step%2048:continue
                    values=metrics(head,z,targets,fit,masks,nold)
                    curve.append(dict(step=step,metrics=values,clipped_updates=clipped))
                    snap=dict(identity=cfg['identity'],state=head.state_dict(),optimizer=opt.state_dict(),
                        rng=rng_state(),step=step,curve=curve,clipped=clipped)
                    checkpoint(latest,snap);write(out/'CURVE.json',curve)
                    if step in (8192,budget):checkpoint(out/f'ENDPOINT_{step}.pt',snap)
                    progress('FIT',fold=fold,seed=seed,factor=factor,step=step,total=budget,completed=len(results),arms=8)
                endpoints={}
                for step,label in ((8192,'equal_updates'),(budget,'equal_presentations')):
                    saved=torch.load(out/f'ENDPOINT_{step}.pt',weights_only=False)
                    if any(g['lr']!=cfg['lr'] or g['weight_decay']!=.01 for g in saved['optimizer']['param_groups']):
                        raise ValueError('COVERAGE_OPTIMIZER')
                    cold=RoleHeads(template('cuda'),1,seed);cold.load_state_dict(saved['state'])
                    expected=next(v['metrics'] for v in curve if v['step']==step)
                    actual=metrics(cold,z,targets,fit,masks,nold)
                    if actual!=expected:raise ValueError('COVERAGE_COLD_RELOAD')
                    endpoints[label]=actual
                r=dict(identity=cfg['identity'],fold=fold,seed=seed,factor=factor,fit_rows=len(fit),
                    initialization_sha256=tree_digest(data['initial'][seed]),schedule_sha256=tree_digest(schedule),
                    teacher_state_sha256=info['teacher_state_sha256'],clipped_updates=clipped,cold_exact=True,
                    initial=curve[0]['metrics'],
                    constant_baselines={k:fit_metrics(np.tile(targets[fit].mean(0),(2,len(ix),1)),targets[ix])['minimum']
                        for k,ix in masks.items()},**endpoints)
                write(out/'RESULT.json',r);seal(out,cfg['identity']);results.append(r)
                del head,opt,cold
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('COVERAGE_SOURCE_FILE_CHANGED')
    comparisons=[]
    for r in results:
        control=read(a.previous/f"units/{r['fold']}_{r['seed']}_balanced/RESULT.json")['final']
        for endpoint in ('equal_updates','equal_presentations'):
            audit=r[endpoint]['audit'];baseline=control['audit']
            comparisons.append(dict(fold=r['fold'],seed=r['seed'],factor=r['factor'],endpoint=endpoint,
                audit_rmse=audit['rmse'],control_rmse=baseline['rmse'],audit_regret=audit['mean_regret'],
                control_regret=baseline['mean_regret'],passes=audit['rmse']<baseline['rmse'] and audit['mean_regret']<=baseline['mean_regret']))
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],arms=results,comparisons=comparisons,
        source_preserved=True,teacher_frozen=True,actor_updates=0,encoder_updates=0,
        validation_read=False,test_read=False,model_promoted=False,stl_included=False))
    report(a.output,results,comparisons)
    seal(a.output,cfg['identity']);progress('COMPLETE',completed=8,arms=8)


def report(root,results,comparisons):
    lines=['# Known Teacher Coverage Results','','Lower RMSE/regret is better; artificial targets on real frozen TRAIN embeddings.','',
        '| Fold | Seed | FIT factor | Endpoint | Control RMSE | Expanded RMSE | Control regret | Expanded regret | Pass |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for c in comparisons:
        lines.append(f"| {c['fold']} | {c['seed']} | {c['factor']} | {c['endpoint']} | {c['control_rmse']:.6f} | {c['audit_rmse']:.6f} | {c['control_regret']:.6f} | {c['audit_regret']:.6f} | {c['passes']} |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for r in results:
        ax=axes.flat[(0 if r['fold']=='A' else 2)+(0 if r['seed']==19 else 1)]
        curve=read(root/f"units/{r['fold']}_{r['seed']}_{r['factor']}/CURVE.json")
        for subset,style in (('expanded_fit','--'),('stop',':'),('audit','-')):
            ax.plot([v['step'] for v in curve],[v['metrics'][subset]['rmse'] for v in curve],
                linestyle=style,label=f"{r['factor']}x {subset}")
        ax.set(xscale='symlog',xlabel='Updates',ylabel='Value RMSE',title=f"{r['fold']}/{r['seed']}");ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(root/'COVERAGE_CURVES.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','previous','forensics','labels','native','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise
