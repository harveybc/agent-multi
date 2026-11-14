"""Four resumable protected AdamW fits against authenticated unprotected327."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.adamw_decision_preservation import protected_step
from app.frozen_recipe_inputs import file_sha
from app.head_component_recombination import fit_preservation_gate
from app.native_fit_coverage import validate_record
from app.native_head_warm_start import configure_scope
from app.rl_subset import digest
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest,rng_state,restore_rng
from tools.audit_fixed_actor_critics import covering_schedule
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_teacher_coverage import metrics
from tools.run_training_forensics import template,checkpoint

ROOT=Path(__file__).resolve().parents[1]


def counts(trace,n):
    scales=trace['scale'][:n];streak=longest=0
    for v in scales:
        streak=streak+1 if v==0 else 0;longest=max(longest,streak)
    return dict(attempted=n,accepted=int((scales>0).sum()),full=int((scales==1).sum()),
        reduced=int(((scales>0)&(scales<1)).sum()),rejected=int((scales==0).sum()),
        rounded_zero=int(trace['rounded_zero'][:n].sum()),longest_rejection_streak=longest,
        stalled_256_observed=longest>=256,accepted_motion_sum=float(trace['delta_norm'][:n].sum()),
        clipped=int(trace['clipped'][:n].sum()))


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('PRESERVE_DEVICE')
    pc,cc,wc=[read(p/'CONFIG.json') for p in (a.previous,a.coverage,a.warm)]
    for path,c in ((a.previous,pc),(a.coverage,cc),(a.warm,wc)):verify_done(path,c['identity'])
    if cc['previous']!=pc['identity'] or wc['previous']!=pc['identity'] or wc['coverage']!=cc['identity']:
        raise ValueError('PRESERVE_LINEAGE')
    cfg=dict(schema='actual_adamw_preservation.v1',previous=pc['identity'],coverage=cc['identity'],warm=wc['identity'],
        source_sha256=wc['source_sha256'],seeds=wc['seeds'],updates=wc['updates'],batch=wc['batch'],
        lr=wc['lr'],weight_decay=wc['weight_decay'],scales=[2.**(-i) for i in range(8)],tolerance=1e-8,
        rejection='rollback_parameters_and_optimizer; advance_batch',accepted_damping='retain_proposed_moments',
        code={k:file_sha(ROOT/k) for k in ('tools/run_adamw_decision_preservation.py',
            'app/adamw_decision_preservation.py','app/native_head_warm_start.py','tools/run_teacher_coverage.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('PRESERVE_SOURCE')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('PRESERVE_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True)
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    if support['old_count']!=239 or support['rows']!=[r['row'] for r in records]:raise ValueError('PRESERVE_SUPPORT')
    z=data['z'].to('cuda');targets=np.stack([validate_record(r) for r in records])
    y=torch.tensor(targets,device='cuda',dtype=torch.float32)
    controls=[r for r in read(a.warm/'RESULT.json')['arms'] if r['scope']=='full_head'];results=[];replays=[]
    for control in controls:
        fold,seed=control['fold'],control['seed'];info=support['folds'][fold]
        masks={k:np.array(v) for k,v in info['masks'].items()}
        for kind in ('flat_both_allowed','flat_restricted','open_long','open_short'):
            ix=np.array([i for i in masks['audit'] if records[i]['kind']==kind],dtype=int)
            if len(ix):masks['audit_'+kind]=ix
        fit=np.array(info['expanded']);old=masks['fit'];oldy=targets[old]
        if set(fit)&(set(masks['stop'])|set(masks['audit'])):raise ValueError('PRESERVE_FIT_LEAK')
        initial_path=a.previous/f'units/{fold}_{seed}_base/LATEST.pt';initial_file_sha=file_sha(initial_path)
        initial=torch.load(initial_path,weights_only=False)['model'];initial_hash=tree_digest(initial)
        schedule=fit[covering_schedule(len(fit),32768,cfg['batch'],seed)]
        if (initial_hash!=control['initialization_sha256'] or tree_digest(schedule)!=control['schedule_sha256']
                or tree_digest(fit)!=control['fit_support_sha256']):raise ValueError('PRESERVE_PAIR')
        replay=RoleHeads(template('cuda'),1,seed)
        for step in (0,*cfg['updates']):
            state=initial if step==0 else torch.load(a.warm/f'units/{fold}_{seed}_full_head/ENDPOINT_{step}.pt',weights_only=False)['state']
            replay.load_state_dict(state);actual=metrics(replay,z,targets,fit,masks,239)
            expected=control['control'] if step==0 else control['endpoints'][str(step)]['metrics']
            if actual!=expected:raise ValueError('PRESERVE_CONTROL_REPLAY')
            replays.append(dict(fold=fold,seed=seed,step=step,exact=True))
        out=a.output/'units'/f'{fold}_{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
        head=RoleHeads(template('cuda'),1,seed);head.load_state_dict(initial)
        opt,names=configure_scope(head,'full_head',cfg['lr'],cfg['weight_decay'])
        baseline=control['control'];limit=baseline['fit']['mean_regret']
        def evaluate():
            with torch.no_grad():actions=head(z[:,:239]).mean(2).min(0).values.argmax(-1)[old].cpu().numpy()
            regret=float(np.mean(oldy.max(1)-oldy[np.arange(len(old)),actions]))
            return dict(regret=regret,actions=actions)
        if evaluate()['regret']!=limit:raise ValueError('PRESERVE_INITIAL_GUARD')
        trace=dict(scale=np.zeros(32768),delta_norm=np.zeros(32768),retained_regret=np.zeros(32768),
            trial_regrets=np.full((32768,8),np.nan),trial_count=np.zeros(32768,dtype=np.uint8),
            actions=np.zeros((32768,len(old)),dtype=np.uint8),proposed_actions=np.zeros((32768,len(old)),dtype=np.uint8),
            rounded_zero=np.zeros(32768,dtype=bool),clipped=np.zeros(32768,dtype=bool))
        curve=[];start=0;latest=out/'LATEST.pt'
        if latest.exists():
            saved=torch.load(latest,weights_only=False)
            if (saved['identity']!=cfg['identity'] or saved['initialization_sha256']!=initial_hash
                    or saved['schedule_sha256']!=control['schedule_sha256']):raise ValueError('PRESERVE_UNIT_RESUME')
            head.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
            start=saved['step'];curve=saved['curve']
            if start<0 or start>32768 or curve[-1]['step']!=start:raise ValueError('PRESERVE_RESUME_STEP')
            for k,v in saved['trace'].items():trace[k][:start]=v
            if start in cfg['updates'] and not (out/f'ENDPOINT_{start}.pt').exists():checkpoint(out/f'ENDPOINT_{start}.pt',saved)
            if evaluate()['regret']>limit+cfg['tolerance']:raise ValueError('PRESERVE_RESUME_GUARD')
        progress('FIT',fold=fold,seed=seed,step=start,total=32768,completed=len(results),arms=4)
        for step in range(start,32769):
            if step==start and latest.exists():continue
            if step:
                ix=schedule[step-1]
                record=protected_step(head,opt,z[:,ix],y[ix],evaluate,limit,cfg['tolerance'],cfg['scales'])
                i=step-1
                for k in ('scale','delta_norm','retained_regret','actions','proposed_actions','rounded_zero','clipped'):trace[k][i]=record[k]
                trace['trial_count'][i]=len(record['trial_regrets']);trace['trial_regrets'][i,:len(record['trial_regrets'])]=record['trial_regrets']
            if step not in (0,64,256,1024) and step%2048:continue
            values=metrics(head,z,targets,fit,masks,239);accounting=counts(trace,step)
            if values['fit']['mean_regret']>limit+cfg['tolerance']:raise ValueError('PRESERVE_GUARD_MISMATCH')
            if step==0 and values!=baseline:raise ValueError('PRESERVE_START_REPLAY')
            if opt.state and any(int(v['step'].item())!=accounting['accepted'] for v in opt.state.values()):
                raise ValueError('PRESERVE_OPTIMIZER_COUNT')
            gate=fit_preservation_gate(*[{k:m[k]['mean_regret'] for k in ('fit','expanded_fit')} for m in (baseline,values)],cfg['tolerance'])
            gate=gate and accounting['accepted']>0 and accounting['accepted_motion_sum']>0
            curve.append(dict(step=step,metrics=values,fit_gate=bool(gate),accounting=accounting))
            snap=dict(identity=cfg['identity'],initialization_sha256=initial_hash,schedule_sha256=control['schedule_sha256'],
                state=head.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),step=step,curve=curve,
                trace={k:v[:step].copy() for k,v in trace.items()})
            checkpoint(latest,snap);write(out/'CURVE.json',curve)
            if step in cfg['updates']:checkpoint(out/f'ENDPOINT_{step}.pt',snap)
            progress('FIT',fold=fold,seed=seed,step=step,total=32768,completed=len(results),arms=4,accounting=accounting)
        endpoints={}
        for step in cfg['updates']:
            saved=torch.load(out/f'ENDPOINT_{step}.pt',weights_only=False)
            cold=RoleHeads(template('cuda'),1,seed);cold.load_state_dict(saved['state'])
            actual=metrics(cold,z,targets,fit,masks,239);point=next(r for r in curve if r['step']==step)
            if actual!=point['metrics']:raise ValueError('PRESERVE_COLD_REPLAY')
            endpoints[str(step)]=dict(point,cold_exact=True,state_sha256=tree_digest(saved['state']))
        np.savez_compressed(out/'TRACE.npz',**trace)
        r=dict(identity=cfg['identity'],fold=fold,seed=seed,control=baseline,unprotected=control['endpoints'],
            initialization_sha256=initial_hash,initial_file_sha256=initial_file_sha,schedule_sha256=control['schedule_sha256'],
            fit_support_sha256=control['fit_support_sha256'],original_fit_indices=old.tolist(),
            original_regret_limit=limit,trainable_parameters=3398,endpoints=endpoints)
        write(out/'RESULT.json',r);seal(out,cfg['identity']);results.append(r)
        if file_sha(initial_path)!=initial_file_sha:raise ValueError('PRESERVE_CONTROL_CHANGED')
    if len(results)!=4 or file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('PRESERVE_SOURCE_CHANGED')
    gates={str(step):dict(passes=sum(r['endpoints'][str(step)]['fit_gate'] for r in results),total=4) for step in cfg['updates']}
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],arms=results,fit_gates=gates,replays=replays,
        model_promoted=False,source_preserved=True,actor_updates=0,encoder_updates=0,validation_read=False,test_read=False))
    lines=['# Actual AdamW Decision Preservation','','Fixed proposal budgets; audit descriptive; no model promotion.','',
        '| Fold | Seed | Proposals | Accepted | Reduced | Rejected | Old FIT regret | Expanded FIT regret | Audit regret | FIT pass |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in results:
        for step,e in r['endpoints'].items():
            c=e['accounting'];m=e['metrics']
            lines.append(f"| {r['fold']} | {r['seed']} | {step} | {c['accepted']} | {c['reduced']} | {c['rejected']} | {m['fit']['mean_regret']:.6f} | {m['expanded_fit']['mean_regret']:.6f} | {m['audit']['mean_regret']:.6f} | {e['fit_gate']} |")
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    seal(a.output,cfg['identity']);progress('COMPLETE',completed=4,arms=4)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','coverage','warm','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)
