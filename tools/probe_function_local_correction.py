"""Fixed lambda1 FIT-gap-local corrections paired with ten accepted344 controls."""
import argparse
import csv
import fcntl
from pathlib import Path
import time
import numpy as np
import torch
from app.decision_cost_objective import fit_temperature
from app.decision_targeted_direction import strict_correction
from app.function_local_projection import PAIRS,gap_jacobian,locality_factor,bounded_local_repair,gap_drift,pairwise_qualified
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.optimal_fit_protection import optimal_support,optimal_pairs,complete_pair_jacobian
from app.rl_subset import digest
from app.robust_adamw_direction import decision_guard
from app.weight_decision_forensics import native_forward
from tools.probe_decision_targeted_direction import target_derivative_check
from tools.probe_optimal_fit_protection import support_and_repair
from tools.probe_robust_adamw_direction import vector,moved
from tools.replay_first_harmful_crossings import require_equal
from tools.run_affine_critic_calibration import views
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import checkpoint
from tools.verify_optimal_protection_continuation import score

ROOT=Path(__file__).resolve().parents[1]


def gap_derivative_check(ctx,state,g,ix):
    ctx.double.load_state_dict(state);q=ctx.double(ctx.zcpu.double()).mean(2);error=0.
    for pair,(a,b) in enumerate(PAIRS):
        for pos in sorted(set((0,len(ix)//2,len(ix)-1))):
            row=int(ix[pos]);value=q[:,row,a].min()-q[:,row,b].min()
            actual=torch.cat([v.flatten() for v in torch.autograd.grad(value,list(ctx.double.parameters()),retain_graph=True)]).detach().numpy()
            error=max(error,float(np.max(abs(actual-g[pair*len(ix)+pos]))))
    if error>1e-11:raise ValueError('LOCAL_GAP_DERIVATIVE')
    return error


def run(a):
    configure(318);torch.backends.cudnn.allow_tf32=False;start=time.monotonic()
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('LOCAL_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('control','verified','crossings','probe','coverage')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if parents['verified']!=parents['control']:raise ValueError('LOCAL_VERIFIED_ID')
    if read(a.verified/'VERIFICATION.json')['accepted_candidates']!=10:raise ValueError('LOCAL_CONTROL_COUNT')
    for k in ('crossings','probe','coverage'):
        if parents['control']['parents'][k]!=parents[k]['identity']:raise ValueError('LOCAL_LINEAGE')
    for k,sha in parents['control']['code'].items():
        if file_sha(ROOT/k)!=sha:raise ValueError('LOCAL_CONTROL_CODE')
    cfg=dict(schema='function_local_correction.v1',parents={k:c['identity'] for k,c in parents.items()},lambda_fixed=1,
        objective='half_squared_parameter_distance_from_raw_plus_half_P_over_Gfro2_times_squared_other_FIT_gap_change',
        repairs=3,scales=8,norm_cap=4,candidates=10,scope='accepted344_targets_only',
        source_actor_sha256=parents['control']['source_actor_sha256'],
        code={k:file_sha(ROOT/k) for k in ('tools/probe_function_local_correction.py','app/function_local_projection.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('LOCAL_ACTOR')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('LOCAL_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');support=read(a.coverage/'prepared/SUPPORT.json')
    y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']]);results=[]
    for u in read(a.control/'RESULT.json')['units']:
        trials=[t for t in u['trials'] if t['record']['accepted']]
        if not trials:continue
        fold,seed=u['fold'],u['seed'];path=a.crossings/'cases'/f'{fold}_{seed}'/'witnesses'/f"{u['offset']:03d}.pt"
        if file_sha(path)!=u['witness_sha256']:raise ValueError('LOCAL_WITNESS')
        w=torch.load(path,weights_only=False,map_location='cpu');before=w['before'];d=vector(w['proposed'])-vector(before)
        snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        old=np.array(support['folds'][fold]['masks']['fit']);limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,y,support,snap,fold,seed,limit);_,_,details,assess=support_and_repair(ctx,before,w['proposed'])
        baseline=views(ctx,before,ctx.fit);allowed,eligible=optimal_support(y[ctx.fit],baseline)
        old_pairs=[(int(i),int(action),b) for i,action in zip(ctx.old,ctx.actions) for b in range(3) if b!=action]
        for trial in trials:
            c=trial['candidate'];row=c['row'];name=f"{u['unit']}_{row}";out=a.output/'units'/name
            if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
            control_path=a.control/'units'/u['unit']/f'TARGET_{row}.pt';control=torch.load(control_path,weights_only=False,map_location='cpu')
            def evaluate(delta):
                state=moved(before,delta,1.);assessment=assess(state);qs=views(ctx,state,ctx.fit)
                assessment['passed']=strict_correction(assessment,details['baseline'],qs,c['position'],allowed[c['position']])
                return assessment
            control_assessment=evaluate(vector(control['state'])-vector(before));require_equal(control_assessment,trial['assessment'],'LOCAL_CONTROL')
            cq=views(ctx,control['state'],ctx.fit);cd={k:gap_drift(baseline[k],q,c['position']) for k,q in cq.items()}
            def linear(center):
                state=moved(before,center,1.);q=native_forward(state,ctx.zn)[0];protected=optimal_pairs(q,ctx.fit,allowed,eligible)
                gaps,jac=complete_pair_jacobian(state,ctx.zn,old_pairs+protected+c['pairs'])
                threshold=np.concatenate([np.repeat(ctx.margin,4)*2,np.zeros(2*len(protected)),np.full(2*len(c['pairs']),c['margin'])])
                return gaps,jac,vector(state)-vector(before),threshold
            zero,zreceipt=bounded_local_repair(d,linear,evaluate,np.eye(len(d)))
            if zero is None:raise ValueError('LOCAL_IDENTITY_CONTROL_FAILED')
            require_equal(moved(before,zero,1.),control['state'],'LOCAL_IDENTITY_CONTROL_STATE')
            g,ix=gap_jacobian(before,ctx.zn,ctx.fit,row);derivative=gap_derivative_check(ctx,before,g,ix)
            target_error=target_derivative_check(ctx,before,c['pairs']);l=locality_factor(g)
            delta,receipt=bounded_local_repair(d,linear,evaluate,l);assessment=None;drift=None;held=None;cold=False;repeated=False
            observations=dict(before=baseline,control=cq)
            if delta is not None:
                state=moved(before,delta,1.);assessment=evaluate(delta);qs=views(ctx,state,ctx.fit);observations['local']=qs
                drift={k:gap_drift(baseline[k],q,c['position']) for k,q in qs.items()}
                checkpoint(out/'CANDIDATE.pt',dict(identity=cfg['identity'],state=state,target=row,witness_sha256=file_sha(path),continuation_allowed=False))
                loaded=torch.load(out/'CANDIDATE.pt',weights_only=False,map_location='cpu');require_equal(evaluate(vector(loaded['state'])-vector(before)),assessment,'LOCAL_COLD');cold=True
                again,rec2=bounded_local_repair(d,linear,evaluate,l);require_equal(again,delta,'LOCAL_REPEAT_DIRECTION');require_equal(rec2,receipt,'LOCAL_REPEAT_RECEIPT');repeated=True
                held={}
                for label,s in [('control',control['state']),('local',state)]:
                    q=native_forward(s,ctx.zn)[0];held[label]={k:score(q[:,support['folds'][fold]['masks'][k]],y[support['folds'][fold]['masks'][k]],ctx.temperature) for k in ('stop','audit')}
            checkpoint(out/'OBSERVATIONS.pt',observations)
            result=dict(unit=name,source_unit=u['unit'],fold=fold,seed=seed,offset=u['offset'],candidate=c,witness_sha256=file_sha(path),
                control_sha256=file_sha(control_path),temperature=ctx.temperature,control=control_assessment,control_drift=cd,
                record=receipt,assessment=assessment,drift=drift,qualified=pairwise_qualified(assessment,control_assessment['metrics'],drift,cd),
                gap_derivative_max_error=derivative,target_derivative_max_error=target_error,identity_control_exact=True,
                metric_rows=len(g),metric_columns=g.shape[1],metric_frobenius_squared=float(np.sum(g*g)),
                cold_exact=cold,repeated_exact=repeated,held_internal_train=held)
            write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
            write(a.output/'PROGRESS.json',dict(stage='PROBING',completed=len(results),total=10,unit=name,seconds=time.monotonic()-start,at=now()))
    if len(results)!=10:raise ValueError('LOCAL_FINAL_COUNT')
    for k,cfgp in parents.items():verify_done(getattr(a,k),cfgp['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('LOCAL_ACTOR_CHANGED')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],seconds=time.monotonic()-start,units=results,actor_updates=0,optimizer_steps=0,model_promoted=False))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',completed=10,at=now()))


def verify(a):
    cfg=read(a.output/'CONFIG.json');verify_done(a.output,cfg['identity']);result=read(a.output/'RESULT.json')
    for k,identity in cfg['parents'].items():verify_done(getattr(a,k),identity)
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('LOCAL_VERIFY_ACTOR')
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');z=data['z'].double().numpy()
    support=read(a.coverage/'prepared/SUPPORT.json');y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']]);rows=[]
    for u in result['units']:
        out=a.output/'units'/u['unit'];verify_done(out,cfg['identity']);fold=u['fold'];c=u['candidate'];pos=c['position']
        fit=np.array(support['folds'][fold]['expanded']);target=y[fit];temperature=fit_temperature(target)
        if fit[pos]!=c['row'] or temperature!=u['temperature']:raise ValueError('LOCAL_VERIFY_TARGET')
        wpath=a.crossings/'cases'/f"{fold}_{u['seed']}"/'witnesses'/f"{u['offset']:03d}.pt"
        if file_sha(wpath)!=u['witness_sha256']:raise ValueError('LOCAL_VERIFY_WITNESS')
        w=torch.load(wpath,weights_only=False,map_location='cpu');obs=torch.load(out/'OBSERVATIONS.pt',weights_only=False,map_location='cpu')
        control_path=a.control/'units'/u['source_unit']/f"TARGET_{c['row']}.pt"
        if file_sha(control_path)!=u['control_sha256']:raise ValueError('LOCAL_VERIFY_CONTROL')
        control=torch.load(control_path,weights_only=False,map_location='cpu');states={'before':w['before'],'control':control['state']}
        if u['record']['accepted']:
            cp=torch.load(out/'CANDIDATE.pt',weights_only=False,map_location='cpu')
            if cp['identity']!=cfg['identity'] or cp['target']!=c['row'] or cp['continuation_allowed'] or 'optimizer' in cp:raise ValueError('LOCAL_VERIFY_CP')
            states['local']=cp['state']
        allowed=target==target.max(1,keepdims=True);eligible=allowed.sum(1)<3
        for q in obs['before'].values():eligible&=allowed[np.arange(len(fit)),q.min(0).argmax(1)]
        old=np.array(support['folds'][fold]['masks']['fit']);old_pos=np.array([list(fit).index(i) for i in old])
        snap=torch.load(a.probe/'units'/f"{fold}_{u['seed']}_8192_0"/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        for label,state in states.items():
            actual=native_forward(state,z[:,fit])[0]
            if np.max(abs(actual-obs[label]['numpy']))>1e-11 or not np.array_equal(actual.min(0).argmax(1),obs[label]['numpy'].min(0).argmax(1)):raise ValueError('LOCAL_VERIFY_FORWARD')
            if label=='before':continue
            if u['held_internal_train'] is not None:
                full=native_forward(state,z)[0]
                for split in ('stop','audit'):
                    ix=np.array(support['folds'][fold]['masks'][split]);actual_held=score(full[:,ix],y[ix],temperature);expected_held=u['held_internal_train'][label][split]
                    if actual_held['actions_sha256']!=expected_held['actions_sha256']:raise ValueError('LOCAL_VERIFY_HELD_ACTIONS')
                    for k in ('cost','regret'):
                        if abs(actual_held[k]-expected_held[k])>1e-12:raise ValueError('LOCAL_VERIFY_HELD_METRIC')
            motion=float(np.linalg.norm(vector(state)-vector(w['before'])));norm=float(np.linalg.norm(vector(w['proposed'])-vector(w['before'])))
            if not 0<motion<=4*norm:raise ValueError('LOCAL_VERIFY_MOTION')
            expected=u['control'] if label=='control' else u['assessment'];expected_drift=u['control_drift'] if label=='control' else u['drift']
            for name,q in obs[label].items():
                actions=q.min(0).argmax(1);metrics=score(q,target,temperature);base=score(obs['before'][name],target,temperature)
                if not decision_guard(q.min(0)[old_pos],snap['actions'],snap['margin'],y[old],limit)['passed']:raise ValueError('LOCAL_VERIFY_GUARD')
                if not allowed[pos,actions[pos]] or not allowed[np.flatnonzero(eligible),actions[eligible]].all():raise ValueError('LOCAL_VERIFY_OPTIMAL')
                if not metrics['cost']<base['cost'] or not metrics['regret']<base['regret']:raise ValueError('LOCAL_VERIFY_QUALITY')
                for k in ('cost','regret'):
                    if abs(metrics[k]-expected['metrics'][name][k])>1e-12:raise ValueError('LOCAL_VERIFY_METRIC')
                if abs(gap_drift(obs['before'][name],q,pos)-expected_drift[name])>1e-14:raise ValueError('LOCAL_VERIFY_DRIFT')
        qualified=pairwise_qualified(u['assessment'],u['control']['metrics'],u['drift'],u['control_drift'])
        if qualified!=u['qualified']:raise ValueError('LOCAL_VERIFY_SELECTION')
        rows.append(dict(unit=u['unit'],target=c['row'],accepted=u['record']['accepted'],qualified=qualified,
            control_drift=u['control_drift']['numpy'],local_drift=u['drift']['numpy'] if u['drift'] else None,
            control_regret=u['control']['metrics']['numpy']['regret'],local_regret=u['assessment']['metrics']['numpy']['regret'] if u['assessment'] else None,
            reason=u['record']['reason']))
    report=dict(identity=cfg['identity'],controls=len(rows),accepted=sum(r['accepted'] for r in rows),qualified=sum(r['qualified'] for r in rows),
        cold_exact=sum(u['cold_exact'] for u in result['units']),repeated_exact=sum(u['repeated_exact'] for u in result['units']),rows=rows,actor_updates=0,model_promoted=False)
    write(a.verification_output/'CONFIG.json',cfg);write(a.verification_output/'VERIFICATION.json',report)
    with (a.verification_output/'SUMMARY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    seal(a.verification_output,cfg['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('control','verified','crossings','probe','coverage','bundle','output','verification-output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if not a.verify_only:run(a)
            if (a.verification_output/'DONE.json').exists():verify_done(a.verification_output,read(a.output/'CONFIG.json')['identity'])
            else:verify(a)
        except Exception as exc:
            if not a.verify_only:write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()))
            raise
