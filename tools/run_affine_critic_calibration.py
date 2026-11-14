"""Calibrate saved cost critics on FIT, preserving measured decisions and margins."""
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
import torch

from app.affine_critic_calibration import required_scale,fit_shared_affine,transform_head
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.rl_subset import digest
from app.robust_adamw_direction import numpy_heads
from app.training_forensics import tree_digest
from tools.probe_robust_adamw_direction import fit_quality
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def views(ctx,state,indices,singles=True):
    ctx.head.load_state_dict(state);ctx.cpu.load_state_dict(state)
    with torch.no_grad():
        out=dict(gpu=ctx.head(ctx.z[:,indices]).mean(2).cpu().numpy(),
            cpu=ctx.cpu(ctx.zcpu[:,indices]).mean(2).numpy(),numpy=numpy_heads(state,ctx.zn[:,indices]))
        if singles:
            out['gpu_single']=np.concatenate([ctx.head(ctx.z[:,i:i+1]).mean(2).cpu().numpy() for i in indices],axis=1)
            out['cpu_single']=np.concatenate([ctx.cpu(ctx.zcpu[:,i:i+1]).mean(2).numpy() for i in indices],axis=1)
    return out


def compare(before,after,target):
    out={}
    for name,q in after.items():
        old=before[name];actions=q.min(0).argmax(1);previous=old.min(0).argmax(1)
        out[name]=dict(action_changes=int(np.sum(actions!=previous)),before=fit_quality(old,target),after=fit_quality(q,target))
    return out


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('CALIBRATION_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('source','probe','coverage')}
    for k,cfg in parents.items():verify_done(getattr(a,k),cfg['identity'])
    source=parents['source']
    if (source['probe']!=parents['probe']['identity'] or source['coverage']!=parents['coverage']['identity']
            or source['optimizer_state']!='fresh'):raise ValueError('CALIBRATION_LINEAGE')
    cfg=dict(schema='affine_critic_calibration.v1',parents={k:c['identity'] for k,c in parents.items()},
        source_actor_sha256=source['source_sha256'],endpoints=[256,1024],variants=['shift','affine'],
        fit='expanded_FIT_only',optimizer_updates=0,validation_read=False,test_read=False,
        code={k:file_sha(ROOT/k) for k in ('tools/run_affine_critic_calibration.py','app/affine_critic_calibration.py',
            'app/robust_adamw_direction.py','tools/run_decision_cost_contrast.py','tools/run_automated_robust_refinement.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('CALIBRATION_ACTOR_IDENTITY')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('CALIBRATION_RESUME_IDENTITY')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    records=read(a.coverage/'PANEL.json')['records'];support=read(a.coverage/'prepared/SUPPORT.json')
    data=torch.load(a.coverage/'prepared/DATA.pt',map_location='cpu',weights_only=True)
    if support['old_count']!=239 or support['rows']!=[r['row'] for r in records]:raise ValueError('CALIBRATION_SUPPORT')
    y=np.stack([validate_record(r) for r in records]);results=[];controls=0;cold=0
    for fold in ('A','B'):
        for seed in (19,43):
            snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
            old=np.array(support['folds'][fold]['masks']['fit']);actions=snap['actions']
            limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),actions]))
            ctx=CostContext(data,y,support,snap,fold,seed,limit)
            for step in cfg['endpoints']:
                saved=torch.load(a.source/'units'/f'{fold}_{seed}_cost'/f'ENDPOINT_{step}.pt',weights_only=False,map_location='cpu')
                state=saved['state'];before=ctx.observe(state,single=True)
                expected=next(p['assessment']['observation'] for p in saved['curve'] if p['step']==step)
                if before!=expected or not before['passed']:raise ValueError('CALIBRATION_CONTROL_REPLAY')
                controls+=1
                fit_before=views(ctx,state,ctx.fit);guard_before=views(ctx,state,np.arange(239),singles=False)
                oldviews=[q.min(0)[ctx.old_positions] for q in fit_before.values()]+[q.min(0)[old] for q in guard_before.values()]
                lower=required_scale(oldviews,actions,snap['margin'])
                held={panel:np.array(support['folds'][fold]['masks'][panel]) for panel in ('stop','audit')}
                if any(set(ix)&set(ctx.fit) for ix in held.values()):raise ValueError('CALIBRATION_HELD_OVERLAP')
                held_before={panel:views(ctx,state,ix) for panel,ix in held.items()}
                for kind in cfg['variants']:
                    out=a.output/'units'/f'{fold}_{seed}_{step}_{kind}'
                    coefficients=fit_shared_affine(fit_before['numpy'],y[ctx.fit],lower,kind)
                    changed=transform_head(state,coefficients['scale'],coefficients['offset'])
                    checkpoint(out/'MODEL.pt',dict(identity=cfg['identity'],source_state_sha256=tree_digest(state),
                        state=changed,coefficients=coefficients,fold=fold,seed=seed,source_step=step))
                    checkpoint(out/'FIT_WITNESS.pt',dict(q=fit_before['numpy'],target=y[ctx.fit],protected_views=oldviews,
                        protected_actions=actions,protected_margins=snap['margin'],fit=ctx.fit,old=old))
                    actual=ctx.observe(changed,single=True);fit_after=views(ctx,changed,ctx.fit)
                    comparisons=compare(fit_before,fit_after,y[ctx.fit]);held_reports={}
                    for panel,ix in held.items():held_reports[panel]=compare(held_before[panel],views(ctx,changed,ix),y[ix])
                    unchanged=all(r['action_changes']==0 for r in comparisons.values())
                    nonincrease=all(actual['fit'][k]['mse']<=before['fit'][k]['mse']+1e-10 for k in ('gpu','cpu','numpy'))
                    strict=all(actual['fit'][k]['mse']<before['fit'][k]['mse']-1e-10 for k in ('gpu','cpu','numpy'))
                    restored=torch.load(out/'MODEL.pt',weights_only=False,map_location='cpu')['state']
                    if tree_digest(restored)!=tree_digest(changed) or ctx.observe(restored,single=True)!=actual:raise ValueError('CALIBRATION_COLD_REPLAY')
                    cold+=1
                    record=dict(fold=fold,seed=seed,source_step=step,kind=kind,coefficients=coefficients,
                        source_state_sha256=tree_digest(state),calibrated_state_sha256=tree_digest(changed),
                        eligible=bool(actual['passed'] and unchanged and nonincrease),strict_mse_improvement=strict,
                        original_protection=actual['passed'],all_fit_actions_preserved=unchanged,
                        before=before,after=actual,fit_comparisons=comparisons,held=held_reports,
                        affine_forward_max_error={k:float(np.max(np.abs(q-(coefficients['scale']*fit_before[k]+coefficients['offset'])))) for k,q in fit_after.items()})
                    write(out/'RESULT.json',record);seal(out,cfg['identity']);results.append(record)
                    write(a.output/'PROGRESS.json',dict(stage='CALIBRATING',completed=len(results),total=16,at=now(),seconds=time.monotonic()-started))
    for k,parent in parents.items():verify_done(getattr(a,k),parent['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('CALIBRATION_ACTOR_CHANGED')
    result=dict(identity=cfg['identity'],seconds=time.monotonic()-started,results=results,source_controls_exact=controls,
        cold_replays_exact=cold,optimizer_updates=0,actor_updates=0,model_promoted=False)
    write(a.output/'RESULT.json',result)
    lines=['# Shared Affine Calibration Results','','Native TRAIN FIT error, not trading returns.','',
        '| Fold/seed | Step | Variant | Scale | Offset | MSE before | MSE after | FIT unchanged | Protected | Eligible |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |']
    for r in results:
        c=r['coefficients'];b=r['before']['fit']['gpu'];x=r['after']['fit']['gpu']
        lines.append(f"| {r['fold']}{r['seed']} | {r['source_step']} | {r['kind']} | {c['scale']:.8f} | {c['offset']:.8f} | {b['mse']:.6f} | {x['mse']:.6f} | {r['all_fit_actions_preserved']} | {r['original_protection']} | {r['eligible']} |")
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');seal(a.output,cfg['identity'])
    write(a.output/'PROGRESS.json',dict(stage='COMPLETE',completed=len(results),at=now(),seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','probe','coverage','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
