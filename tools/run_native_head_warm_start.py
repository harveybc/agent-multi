"""Eight sequential paired native-head warm starts, no actor or encoder updates."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.head_component_recombination import fit_preservation_gate
from app.native_fit_coverage import validate_record
from app.native_head_warm_start import configure_scope,check_frozen,fit_step
from app.rl_subset import digest
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest,rng_state,restore_rng
from tools.audit_fixed_actor_critics import covering_schedule
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_teacher_coverage import metrics
from tools.run_training_forensics import template,checkpoint

ROOT=Path(__file__).resolve().parents[1]


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('WARM_DEVICE')
    pc,cc,rc=[read(p/'CONFIG.json') for p in (a.previous,a.coverage,a.recombination)]
    for p,c in ((a.previous,pc),(a.coverage,cc),(a.recombination,rc)):verify_done(p,c['identity'])
    if cc['previous']!=pc['identity'] or rc['coverage']!=cc['identity'] or rc['previous']!=pc['identity']:
        raise ValueError('WARM_LINEAGE')
    cfg=dict(schema='native_head_warm_start.v1',previous=pc['identity'],coverage=cc['identity'],
        recombination=rc['identity'],source_sha256=cc['source_sha256'],seeds=[19,43],
        scopes=['output_only','full_head'],updates=[8192,32768],batch=cc['batch'],lr=cc['lr'],
        weight_decay=cc['weight_decay'],gradient_clip=10.,optimizer_state='fresh_both_arms',
        tolerance=1e-8,primary='8192_additional_updates',
        code={k:file_sha(ROOT/k) for k in ('tools/run_native_head_warm_start.py','app/native_head_warm_start.py',
            'app/head_component_recombination.py','tools/run_teacher_coverage.py','app/critic_target_contract.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('WARM_SOURCE')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('WARM_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):
        write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True)
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    if support['old_count']!=239 or support['rows']!=[r['row'] for r in records]:raise ValueError('WARM_SUPPORT')
    targets=np.stack([validate_record(r) for r in records])
    z=data['z'].to('cuda');y=torch.tensor(targets,dtype=torch.float32,device='cuda')
    prior=read(a.coverage/'RESULT.json');results=[];replays=[]
    for fold,info in support['folds'].items():
        masks={k:np.array(v) for k,v in info['masks'].items()}
        for kind in ('flat_both_allowed','flat_restricted','open_long','open_short'):
            ix=np.array([i for i in masks['audit'] if records[i]['kind']==kind],dtype=int)
            if len(ix):masks['audit_'+kind]=ix
        fit=np.array(info['expanded'])
        if set(fit)&(set(masks['stop'])|set(masks['audit'])):raise ValueError('WARM_SPLIT_LEAK')
        for seed in cfg['seeds']:
            cp=a.previous/f'units/{fold}_{seed}_base/LATEST.pt';cp_sha=file_sha(cp)
            initial=torch.load(cp,weights_only=False)['model'];initial_hash=tree_digest(initial)
            control=RoleHeads(template('cuda'),1,seed);control.load_state_dict(initial)
            baseline=metrics(control,z,targets,fit,masks,239)
            old=next(r for r in prior['arms'] if (r['fold'],r['seed'])==(fold,seed))['control']
            if {k:v for k,v in baseline.items() if k!='expanded_fit'}!=old:raise ValueError('WARM_CONTROL_REPLAY')
            replays.append(dict(fold=fold,seed=seed,exact=True))
            schedule=fit[covering_schedule(len(fit),32768,cfg['batch'],seed)]
            schedule_hash=tree_digest(schedule)
            for scope in cfg['scopes']:
                out=a.output/'units'/f'{fold}_{seed}_{scope}';out.mkdir(parents=True,exist_ok=True)
                if (out/'DONE.json').exists():
                    verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
                head=RoleHeads(template('cuda'),1,seed);head.load_state_dict(initial)
                if tree_digest(head.state_dict())!=initial_hash:raise ValueError('WARM_INITIAL_WEIGHTS')
                opt,names=configure_scope(head,scope,cfg['lr'],cfg['weight_decay'])
                if opt.state:raise ValueError('WARM_OPTIMIZER_NOT_FRESH')
                curve=[];start=0;clipped=0;latest=out/'LATEST.pt'
                if latest.exists():
                    saved=torch.load(latest,weights_only=False)
                    if (saved['identity']!=cfg['identity'] or saved['scope']!=scope or
                        saved['initialization_sha256']!=initial_hash or saved['schedule_sha256']!=schedule_hash or
                        saved['trainable_names']!=names):raise ValueError('WARM_UNIT_RESUME')
                    head.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
                    start=saved['step'];curve=saved['curve'];clipped=saved['clipped']
                    if start<0 or start>32768 or curve[-1]['step']!=start:raise ValueError('WARM_RESUME_STEP')
                    if any(g['lr']!=cfg['lr'] or g['weight_decay']!=cfg['weight_decay'] for g in opt.param_groups):
                        raise ValueError('WARM_RESUME_OPTIMIZER')
                    check_frozen(head,initial,names)
                    if start in cfg['updates'] and not (out/f'ENDPOINT_{start}.pt').exists():
                        checkpoint(out/f'ENDPOINT_{start}.pt',saved)
                progress('FIT',fold=fold,seed=seed,scope=scope,step=start,total=32768,completed=len(results),arms=8)
                for step in range(start,32769):
                    if step==start and latest.exists():continue
                    if step:
                        ix=schedule[step-1];clipped+=int(fit_step(head,opt,z[:,ix],y[ix]))
                    if step not in (0,64,256,1024) and step%2048:continue
                    check_frozen(head,initial,names)
                    values=metrics(head,z,targets,fit,masks,239)
                    if step==0 and values!=baseline:raise ValueError('WARM_INITIAL_REPLAY')
                    gate=fit_preservation_gate(*[{k:m[k]['mean_regret'] for k in ('fit','expanded_fit')}
                                                for m in (baseline,values)],cfg['tolerance'])
                    delta={k:float((v-initial[k].to(v.device)).double().square().sum().sqrt())
                           for k,v in head.state_dict().items()}
                    curve.append(dict(step=step,metrics=values,fit_gate=gate,clipped_updates=clipped,weight_delta_l2=delta))
                    snap=dict(identity=cfg['identity'],scope=scope,initialization_sha256=initial_hash,
                        schedule_sha256=schedule_hash,trainable_names=names,state=head.state_dict(),
                        optimizer=opt.state_dict(),rng=rng_state(),step=step,curve=curve,clipped=clipped)
                    checkpoint(latest,snap);write(out/'CURVE.json',curve)
                    if step in cfg['updates']:checkpoint(out/f'ENDPOINT_{step}.pt',snap)
                    progress('FIT',fold=fold,seed=seed,scope=scope,step=step,total=32768,completed=len(results),arms=8)
                endpoints={}
                for step in cfg['updates']:
                    saved=torch.load(out/f'ENDPOINT_{step}.pt',weights_only=False)
                    cold=RoleHeads(template('cuda'),1,seed);cold.load_state_dict(saved['state'])
                    check_frozen(cold,initial,names)
                    values=metrics(cold,z,targets,fit,masks,239)
                    point=next(r for r in curve if r['step']==step)
                    if values!=point['metrics']:raise ValueError('WARM_COLD_RELOAD')
                    endpoints[str(step)]=dict(metrics=values,fit_gate=point['fit_gate'],cold_exact=True,
                        weight_delta_l2=point['weight_delta_l2'],state_sha256=tree_digest(saved['state']))
                r=dict(identity=cfg['identity'],fold=fold,seed=seed,scope=scope,
                    initialization_sha256=initial_hash,control_file_sha256=cp_sha,schedule_sha256=schedule_hash,
                    fit_support_sha256=tree_digest(fit),fit_rows=len(fit),trainable_names=names,
                    trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
                    optimizer_state='fresh',clipped_updates=clipped,control=baseline,endpoints=endpoints)
                write(out/'RESULT.json',r);seal(out,cfg['identity']);results.append(r)
            if file_sha(cp)!=cp_sha:raise ValueError('WARM_CONTROL_CHANGED')
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('WARM_SOURCE_CHANGED')
    for fold in ('A','B'):
        for seed in cfg['seeds']:
            pair=[r for r in results if (r['fold'],r['seed'])==(fold,seed)]
            if len(pair)!=2 or any(pair[0][k]!=pair[1][k] for k in
                    ('initialization_sha256','schedule_sha256','fit_support_sha256','control')):
                raise ValueError('WARM_PAIR_MISMATCH')
    gates={f'{scope}:{step}':dict(passes=sum(r['endpoints'][str(step)]['fit_gate'] for r in results if r['scope']==scope),total=4)
           for scope in cfg['scopes'] for step in cfg['updates']}
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],arms=results,replays=replays,fit_gates=gates,
        source_preserved=True,actor_updates=0,encoder_updates=0,model_promoted=False,validation_read=False,test_read=False))
    lines=['# Native Head Warm Start','', 'FIT gate is paired; audit is descriptive. No model promotion.','',
        '| Fold | Seed | Scope | Updates | Original FIT regret | Expanded FIT regret | Audit regret | FIT pass |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for r in results:
        for step,e in r['endpoints'].items():
            m=e['metrics'];lines.append(f"| {r['fold']} | {r['seed']} | {r['scope']} | {step} | {m['fit']['mean_regret']:.6f} | {m['expanded_fit']['mean_regret']:.6f} | {m['audit']['mean_regret']:.6f} | {e['fit_gate']} |")
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    seal(a.output,cfg['identity']);progress('COMPLETE',completed=8,arms=8)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','coverage','recombination','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)
