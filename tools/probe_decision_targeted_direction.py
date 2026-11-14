"""Bounded decision-targeted projections on immutable340 proposals, paired with341."""
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
import torch
from app.decision_targeted_direction import correction_candidates,strict_correction
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.optimal_fit_protection import optimal_support,optimal_pairs,complete_pair_jacobian,bounded_repair
from app.rl_subset import digest
from app.weight_decision_forensics import native_forward
from tools.probe_optimal_fit_protection import support_and_repair
from tools.probe_robust_adamw_direction import vector,moved
from tools.replay_first_harmful_crossings import require_equal
from tools.run_affine_critic_calibration import views
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def target_derivative_check(ctx,state,pairs):
    ctx.double.load_state_dict(state);q=ctx.double(ctx.zcpu.double()).mean(2);actual=[];grads=[]
    for row,a,b in pairs:
        for twin in range(2):
            gap=q[twin,row,a]-q[:,row,b].min();actual.append(gap.item())
            grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(gap,list(ctx.double.parameters()),retain_graph=True)]).detach().numpy())
    gaps,jac=complete_pair_jacobian(state,ctx.zn,pairs)
    error=max(float(np.max(abs(gaps-actual))),float(np.max(abs(jac-np.stack(grads)))))
    if error>1e-11:raise ValueError('TARGET_DERIVATIVE')
    return error


def run(a):
    configure(318);torch.backends.cudnn.allow_tf32=False;start=time.monotonic()
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('TARGET_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('crossings','control','verified','probe','coverage')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if parents['control']!=parents['verified']:raise ValueError('TARGET_VERIFIED')
    for k in ('crossings','probe','coverage'):
        if parents['control']['parents'][k]!=parents[k]['identity']:raise ValueError('TARGET_LINEAGE')
    for k,sha in parents['control']['code'].items():
        if file_sha(ROOT/k)!=sha:raise ValueError('TARGET_CONTROL_CODE')
    cfg=dict(schema='decision_targeted_direction.v1',parents={k:c['identity'] for k,c in parents.items()},
        source_actor_sha256=parents['control']['source_actor_sha256'],max_candidates=8,norm_cap=4,
        target_margin='64*float32_epsilon*max(1,max_abs_Q_in_five_views)',repairs=3,scales=8,
        acceptance='original_and_optimal_sets_and_target_correct_and_strictly_lower_FIT_cost_and_regret_all5',
        code={k:file_sha(ROOT/k) for k in ('tools/probe_decision_targeted_direction.py','app/decision_targeted_direction.py','app/optimal_fit_protection.py','app/primal_direction_projection.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('TARGET_ACTOR')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('TARGET_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');support=read(a.coverage/'prepared/SUPPORT.json')
    y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']]);results=[]
    for entry in read(a.control/'RESULT.json')['units']:
        fold,seed,offset=entry['fold'],entry['seed'],entry['offset'];name=f'{fold}_{seed}_{offset:03d}';out=a.output/'units'/name
        if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
        path=a.crossings/'cases'/f'{fold}_{seed}'/'witnesses'/f'{offset:03d}.pt'
        if file_sha(path)!=entry['witness_sha256']:raise ValueError('TARGET_WITNESS')
        w=torch.load(path,weights_only=False,map_location='cpu');before=w['before'];d=vector(w['proposed'])-vector(before);norm=float(np.linalg.norm(d))
        snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        old=np.array(support['folds'][fold]['masks']['fit']);limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,y,support,snap,fold,seed,limit)
        ctrl,rec,details,assess=support_and_repair(ctx,before,w['proposed'])
        control_cp=torch.load(a.control/'units'/name/'ENDPOINT.pt',weights_only=False,map_location='cpu')
        require_equal(ctrl,control_cp['state'],'TARGET_CONTROL_STATE');require_equal(rec,entry['record'],'TARGET_CONTROL_RECEIPT')
        require_equal(details['assessment'],entry['assessment'],'TARGET_CONTROL_QUALITY')
        baseline=views(ctx,before,ctx.fit);allowed,eligible=optimal_support(y[ctx.fit],baseline)
        candidates,counts=correction_candidates(before,ctx.zn,ctx.fit,y[ctx.fit],baseline,ctx.old,4*norm,cfg['max_candidates'])
        old_pairs=[(int(i),int(action),b) for i,action in zip(ctx.old,ctx.actions) for b in range(3) if b!=action]
        old_threshold=np.repeat(ctx.margin,4)*2;trials=[]
        for candidate in candidates:
            error=target_derivative_check(ctx,before,candidate['pairs'])
            def evaluate(delta):
                state=moved(before,delta,1.);assessment=assess(state);qs=views(ctx,state,ctx.fit)
                assessment['passed']=strict_correction(assessment,details['baseline'],qs,candidate['position'],allowed[candidate['position']])
                return assessment
            def linear(center):
                state=moved(before,center,1.);q=native_forward(state,ctx.zn)[0]
                protected=optimal_pairs(q,ctx.fit,allowed,eligible);pairs=old_pairs+protected+candidate['pairs']
                gaps,jac=complete_pair_jacobian(state,ctx.zn,pairs)
                threshold=np.concatenate([old_threshold,np.zeros(2*len(protected)),np.full(2*len(candidate['pairs']),candidate['margin'])])
                return gaps,jac,vector(state)-vector(before),threshold
            delta,record=bounded_repair(d,linear,evaluate);assessment=None;cold=False;repeat=False
            if delta is not None:
                state=moved(before,delta,1.);assessment=evaluate(delta)
                checkpoint(out/f"TARGET_{candidate['row']}.pt",dict(identity=cfg['identity'],witness_sha256=file_sha(path),state=state,target=candidate['row'],continuation_allowed=False))
                loaded=torch.load(out/f"TARGET_{candidate['row']}.pt",weights_only=False,map_location='cpu')
                require_equal(loaded['state'],state,'TARGET_COLD_STATE')
                require_equal(evaluate(vector(loaded['state'])-vector(before)),assessment,'TARGET_COLD_QUALITY');cold=True
                again,record2=bounded_repair(d,linear,evaluate)
                require_equal(again,delta,'TARGET_REPEAT_DIRECTION');require_equal(record2,record,'TARGET_REPEAT_RECEIPT');repeat=True
            trials.append(dict(candidate=candidate,derivative_max_error=error,record=record,assessment=assessment,cold_exact=cold,repeated_exact=repeat))
        useful=[t for t in trials if t['record']['accepted']]
        best=min(useful,key=lambda t:(t['assessment']['metrics']['numpy']['regret'],t['assessment']['metrics']['numpy']['cost'],t['candidate']['row'])) if useful else None
        result=dict(unit=name,fold=fold,seed=seed,offset=offset,witness_sha256=file_sha(path),proposal_norm=norm,
            control_exact=True,baseline=details['baseline'],control=entry['assessment'],counts=counts,trials=trials,
            best_row=best['candidate']['row'] if best else None,best_assessment=best['assessment'] if best else None)
        write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
        write(a.output/'PROGRESS.json',dict(stage='PROBING',completed=len(results),total=10,unit=name,successful=sum(r['best_row'] is not None for r in results),seconds=time.monotonic()-start,at=now()))
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('TARGET_ACTOR_CHANGED')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],seconds=time.monotonic()-start,units=results,actor_updates=0,optimizer_steps=0,model_promoted=False))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',completed=10,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('crossings','control','verified','probe','coverage','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
