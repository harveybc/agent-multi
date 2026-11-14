"""Paired fixed-proposal expanded-FIT optimal-set protection; no new training."""
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
import torch
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.optimal_fit_protection import optimal_support,optimal_pairs,complete_pair_jacobian,finite_quality,bounded_repair
from app.primal_direction_projection import project_primal
from app.rl_subset import digest
from app.training_forensics import tree_digest
from app.twin_complete_protection import twin_complete_jacobian
from app.weight_decision_forensics import native_forward,cost_parts
from tools.probe_robust_adamw_direction import vector,moved
from tools.replay_first_harmful_crossings import require_equal
from tools.run_affine_critic_calibration import views
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def support_and_repair(ctx,before,proposed):
    baseline=views(ctx,before,ctx.fit);allowed,eligible=optimal_support(ctx.targets[ctx.fit],baseline)
    old_pairs=[(int(i),int(a),b) for i,a in zip(ctx.old,ctx.actions) for b in range(3) if b!=a]
    old_threshold=np.repeat(ctx.margin,4)*2
    d=vector(proposed)-vector(before);norm=float(np.linalg.norm(d));source_digest=tree_digest(before)
    # Independently witness the manual original-constraint Jacobian before using it.
    ctx.double.load_state_dict(before);g0,j0=twin_complete_jacobian(ctx.double,ctx.zcpu.double(),ctx.old,ctx.actions)
    g1,j1=complete_pair_jacobian(before,ctx.zn,old_pairs)
    if np.max(abs(g0-g1))>1e-11 or np.max(abs(j0-j1))>1e-11:raise ValueError('OPTIMAL_ORIGINAL_DERIVATIVE')
    def assess(state):
        original=ctx.observe(state,single=True);q=views(ctx,state,ctx.fit)
        return finite_quality(ctx.targets[ctx.fit],q,baseline,allowed,eligible,ctx.temperature,original['passed'],
            float(np.linalg.norm(vector(state)-vector(before))),norm)
    def linear(center):
        state=moved(before,center,1.);q=native_forward(state,ctx.zn)[0]
        pairs=old_pairs+optimal_pairs(q,ctx.fit,allowed,eligible)
        g,j=complete_pair_jacobian(state,ctx.zn,pairs)
        threshold=np.concatenate([old_threshold,np.zeros(len(g)-len(old_threshold))])
        return g,j,vector(state)-vector(before),threshold
    delta,rec=bounded_repair(d,linear,lambda delta:assess(moved(before,delta,1.)))
    state=moved(before,delta,1.) if delta is not None else None
    if tree_digest(before)!=source_digest:raise ValueError('OPTIMAL_SOURCE_MUTATION')
    details=dict(protected_optimal_rows=len(eligible),protected_indices=ctx.fit[eligible].tolist(),
        original_rows=len(ctx.old),expanded_rows=len(ctx.fit),original_jacobian_error=float(np.max(abs(j0-j1))),
        baseline={k:dict(cost=float(cost_parts(q,ctx.targets[ctx.fit],ctx.temperature)['row_cost'].mean()),
            regret=float(cost_parts(q,ctx.targets[ctx.fit],ctx.temperature)['row_regret'].mean())) for k,q in baseline.items()},
        assessment=assess(state) if state is not None else None)
    return state,rec,details,assess


def run(a):
    start=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('OPTIMAL_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('crossings','probe','coverage')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    for k in ('probe','coverage'):
        if parents['crossings']['parents'][k]!=parents[k]['identity']:raise ValueError('OPTIMAL_LINEAGE')
    cfg=dict(schema='optimal_fit_protection.v1',parents={k:c['identity'] for k,c in parents.items()},
        source_actor_sha256=parents['crossings']['source_actor_sha256'],eligible='all_five_geometries_target_optimal_not_all_tied',
        new_threshold=0,original_margin_factor=2,repairs=3,scales=8,norm_cap=4,
        acceptance='original_guards_and_optimal_sets_and_lower_cost_and_nonincreasing_regret_all_views',
        gate='at_least8of10_useful_and_each_case_represented',
        code={k:file_sha(ROOT/k) for k in ('tools/probe_optimal_fit_protection.py','app/optimal_fit_protection.py','app/primal_direction_projection.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('OPTIMAL_ACTOR')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('OPTIMAL_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    source=read(a.crossings/'RESULT.json');support=read(a.coverage/'prepared/SUPPORT.json')
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')
    targets=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']]);results=[]
    for case in source['cases']:
        fold,seed=case['fold'],case['seed'];snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        old=np.array(support['folds'][fold]['masks']['fit']);limit=float(np.mean(targets[old].max(1)-targets[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,targets,support,snap,fold,seed,limit)
        for entry in case['witnesses']:
            name=f"{fold}_{seed}_{entry['offset']:03d}";out=a.output/'units'/name
            if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
            path=a.crossings/'cases'/f'{fold}_{seed}'/'witnesses'/f"{entry['offset']:03d}.pt"
            if file_sha(path)!=entry['file_sha256']:raise ValueError('OPTIMAL_WITNESS_HASH')
            w=torch.load(path,weights_only=False,map_location='cpu')
            if w['identity']!=parents['crossings']['identity']:raise ValueError('OPTIMAL_WITNESS_ID')
            control,control_rec=ctx.repair(w['before'],w['proposed'],'interior',projector=project_primal,twin_complete=True,
                finite_evaluate=lambda state:ctx.observe(state,single=True))
            require_equal(control,w['accepted'],'CONTROL_STATE')
            expected={k:v for k,v in w['receipt'].items() if k not in ('offset','clipped')}
            require_equal(control_rec,expected,'CONTROL_RECEIPT')
            state,rec,details,assess=support_and_repair(ctx,w['before'],w['proposed'])
            control_quality=assess(control);cold=False;repeat=None
            if state is not None:
                checkpoint(out/'ENDPOINT.pt',dict(identity=cfg['identity'],state=state,witness_sha256=file_sha(path),
                    optimizer=w['optimizer_proposed'],rng=w['rng_after']))
                loaded=torch.load(out/'ENDPOINT.pt',weights_only=False,map_location='cpu')
                require_equal(assess(loaded['state']),details['assessment'],'COLD');cold=True
            if not results:
                again,rec2,details2,_=support_and_repair(ctx,w['before'],w['proposed'])
                require_equal(again,state,'REPEATED_STATE');require_equal(rec2,rec,'REPEATED_RECEIPT');require_equal(details2,details,'REPEATED_DETAILS');repeat=True
            result=dict(fold=fold,seed=seed,offset=w['offset'],witness_sha256=file_sha(path),control_exact=True,
                control_quality=control_quality,record=rec,**details,cold_exact=cold,repeated_exact=repeat)
            write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
            write(a.output/'PROGRESS.json',dict(stage='PROBING',completed=len(results),total=10,unit=name,
                accepted=sum(r['record']['accepted'] for r in results),seconds=time.monotonic()-start,at=now()))
    good=[r for r in results if r['record']['accepted'] and r['assessment']['passed']]
    cases={(r['fold'],r['seed']) for r in good};gate=len(good)>=8 and len(cases)==4 and len(results)==10
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('OPTIMAL_ACTOR_CHANGED')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],seconds=time.monotonic()-start,units=results,
        continuation_gate=gate,useful=len(good),actor_updates=0,optimizer_updates=0,model_promoted=False))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',accepted=len(good),total=10,continuation_gate=gate,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('crossings','probe','coverage','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
