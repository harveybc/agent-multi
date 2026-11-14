"""Last bounded critic contrast: original345 target then optional348 local repair."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from app.decision_targeted_direction import correction_candidates,strict_correction
from app.function_local_projection import gap_jacobian,locality_factor,bounded_local_repair,gap_drift,pairwise_qualified
from app.hybrid_local_selection import choose_local,continuation_decision
from app.optimal_fit_protection import optimal_support,optimal_pairs,complete_pair_jacobian
from app.frozen_recipe_inputs import file_sha
from app.weight_decision_forensics import native_forward
from app.training_forensics import tree_digest
from tools.probe_optimal_fit_protection import support_and_repair
from tools.probe_robust_adamw_direction import vector,moved
from tools.run_affine_critic_calibration import views
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write,now
from tools.run_targeted_rescue_continuation import rescue,run as continue_run
from tools.verify_optimal_protection_continuation import run as verify_continuation

ROOT=Path(__file__).resolve().parents[1]


def hybrid(ctx,before,proposed,budget):
    control,record=rescue(ctx,before,proposed,budget)
    if record['arm']!='targeted':return control,record
    d=vector(proposed)-vector(before);baseline=views(ctx,before,ctx.fit)
    allowed,eligible=optimal_support(ctx.targets[ctx.fit],baseline)
    candidates,_=correction_candidates(before,ctx.zn,ctx.fit,ctx.targets[ctx.fit],baseline,ctx.old,4*float(np.linalg.norm(d)),8)
    matches=[c for c in candidates if c['row']==record['target']]
    if len(matches)!=1:raise ValueError('HYBRID_SAME_TARGET')
    c=matches[0];_,_,details,assess=support_and_repair(ctx,before,proposed)
    old_pairs=[(int(i),int(a),b) for i,a in zip(ctx.old,ctx.actions) for b in range(3) if b!=a]
    def evaluate(delta):
        state=moved(before,delta,1.);a=assess(state)
        a['passed']=strict_correction(a,details['baseline'],views(ctx,state,ctx.fit),c['position'],allowed[c['position']])
        return a
    def linear(center):
        state=moved(before,center,1.);q=native_forward(state,ctx.zn)[0]
        protected=optimal_pairs(q,ctx.fit,allowed,eligible)
        gaps,jac=complete_pair_jacobian(state,ctx.zn,old_pairs+protected+c['pairs'])
        threshold=np.concatenate([np.repeat(ctx.margin,4)*2,np.zeros(2*len(protected)),np.full(2*len(c['pairs']),c['margin'])])
        return gaps,jac,vector(state)-vector(before),threshold
    g,_=gap_jacobian(before,ctx.zn,ctx.fit,c['row'])
    delta,receipt=bounded_local_repair(d,linear,evaluate,locality_factor(g))
    local=moved(before,delta,1.) if delta is not None else None
    assessment=evaluate(delta) if delta is not None else None
    cq=views(ctx,control,ctx.fit);control_drift={k:gap_drift(baseline[k],q,c['position']) for k,q in cq.items()}
    drift={k:gap_drift(baseline[k],q,c['position']) for k,q in views(ctx,local,ctx.fit).items()} if local is not None else None
    state,choice=choose_local(control,local,assessment,record['details']['assessment']['metrics'],drift,control_drift)
    result=deepcopy(record)
    result['locality']=dict(choice,target=c['row'],candidate=c,receipt=receipt,assessment=assessment,
        original_assessment=record['details']['assessment'],drift=drift,control_drift=control_drift,
        raw_state=tree_digest(proposed),searches_used=budget[0])
    if choice['qualified']:
        result.update(receipt)
        result['details']['assessment']=assessment
    if tree_digest(state)!=choice['selected_state']:raise ValueError('HYBRID_CHOICE_STATE')
    return state,result


def verify_choices(a):
    cfg=read(a.output/'CONFIG.json');verify_done(a.output,cfg['identity'])
    if (a.verification_output/'DONE.json').exists():verify_done(a.verification_output,cfg['identity'])
    else:verify_continuation(SimpleNamespace(run=a.output,source=a.source,control=a.control,coverage=a.coverage,output=a.verification_output))
    if (a.decision_output/'DONE.json').exists():verify_done(a.decision_output,cfg['identity']);return
    result=read(a.output/'RESULT.json');rows=[]
    for u in result['units']:
        cp=torch.load(a.output/'units'/f"{u['fold']}_{u['seed']}"/'ENDPOINT_16.pt',weights_only=False,map_location='cpu')
        for r in cp['trace']:
            if 'locality' not in r:continue
            l=r['locality'];q=pairwise_qualified(l['assessment'],l['original_assessment']['metrics'],l['drift'],l['control_drift'])
            if q!=l['qualified'] or l['target']!=r['target'] or l['searches_used']!=r['searches_used']:raise ValueError('HYBRID_VERIFY_CHOICE')
            if l['selected_state']!=(l['local_state'] if q else l['control_state']):raise ValueError('HYBRID_VERIFY_FALLBACK')
            expected=l['assessment'] if q else l['original_assessment']
            if expected!=r['details']['assessment']:raise ValueError('HYBRID_VERIFY_ASSESSMENT')
            rows.append(dict(case=f"{u['fold']}_{u['seed']}",offset=r['offset'],target=r['target'],qualified=q,
                drift=l['drift'],control_drift=l['control_drift']))
    choice=continuation_decision(result['units'],sum(r['qualified'] for r in rows))
    write(a.decision_output/'CONFIG.json',cfg)
    write(a.decision_output/'DECISION.json',dict(identity=cfg['identity'],**choice,locality_attempts=len(rows),rows=rows,
        actor_updated=False,critic_diagnostics_closed=True,next_action='one_bounded_actor_utility_trial_then_close_or_export'))
    seal(a.decision_output,cfg['identity'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','control','probe','coverage','protection','verified','bundle','output','verification-output','decision-output','locality','locality-verified'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:verify_choices(a);return
    lc=read(a.locality/'CONFIG.json');vc=read(a.locality_verified/'CONFIG.json')
    verify_done(a.locality,lc['identity']);verify_done(a.locality_verified,vc['identity'])
    if lc!=vc or lc['parents']['control']!=read(a.protection/'CONFIG.json')['identity']:raise ValueError('HYBRID_LOCALITY_LINEAGE')
    for k,sha in lc['code'].items():
        if file_sha(ROOT/k)!=sha:raise ValueError('HYBRID_LOCALITY_CODE')
    extra=dict(schema='hybrid_local_continuation.v1',locality_identity=lc['identity'],lambda_fixed=1,
        selection='all_four_cases_five_views_no_higher_FIT_regret_and_at_least_one_paired_locality_gain',
        automatic_extension=False,critic_diagnostics_closed_after_this_run=True,
        hybrid_code={k:file_sha(ROOT/k) for k in ('tools/run_hybrid_local_continuation.py','app/hybrid_local_selection.py','app/function_local_projection.py')})
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            continue_run(a,repair_fn=hybrid,config_extra=extra);verify_choices(a)
        except Exception as exc:
            write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise


if __name__=='__main__':main()
