"""Resume345 at16 through64 with no new targeted searches and saved342 controls."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from app.decision_cost_objective import cost_objective
from app.frozen_recipe_inputs import file_sha
from app.interior_adamw_repair import transactional_step
from app.native_fit_coverage import validate_record
from app.native_head_warm_start import configure_scope
from app.rl_subset import digest
from app.retention_contract import validate_resume,validate_retention_suffix
from app.training_forensics import rng_state,restore_rng,tree_digest
from app.weight_decision_forensics import native_forward,cost_parts
from tools.audit_fixed_actor_critics import covering_schedule
from tools.probe_optimal_fit_protection import support_and_repair
from tools.replay_first_harmful_crossings import cpu_state,require_equal
from tools.run_affine_critic_calibration import views
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def run(a):
    start=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('OPTIMAL_CONT_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('source','control','probe','coverage','resume','verified')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if parents['verified']!=parents['resume']:raise ValueError('RETENTION_VERIFY_ID')
    v=read(a.verified/'VERIFICATION.json')
    if v['checkpoints']!=8 or v['trace_steps']!=64 or v['resumed_exact_reported']!=16:raise ValueError('RETENTION_VERIFY_GATE')
    for k in ('source','probe','coverage'):
        if parents['control']['parents'][k]!=parents[k]['identity']:raise ValueError('OPTIMAL_CONT_LINEAGE')
    for k in ('source','control','probe','coverage'):
        if parents['resume']['parents'][k]!=parents[k]['identity']:raise ValueError('RETENTION_RESUME_LINEAGE')
    for k,sha in parents['resume']['code'].items():
        if file_sha(ROOT/k)!=sha:raise ValueError('OPTIMAL_CONT_PROTECTION_CODE')
    sc=parents['source'];cfg=dict(schema='targeted_rescue_retention.v1',parents={k:c['identity'] for k,c in parents.items()},
        budget=64,resume_offset=16,checkpoints=[8,16,32,48,64],control_budget=64,control_suffix='',searches_per_case=4,new_searches=0,
        cases=parents['control']['cases'],source_actor_sha256=sc['source_sha256'],lr=sc['lr'],weight_decay=sc['weight_decay'],batch=sc['batch'],
        code={k:file_sha(ROOT/k) for k in ('tools/run_targeted_rescue_retention.py','app/retention_contract.py','tools/probe_optimal_fit_protection.py','app/optimal_fit_protection.py','tools/verify_optimal_protection_continuation.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('OPTIMAL_CONT_ACTOR')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('OPTIMAL_CONT_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');support=read(a.coverage/'prepared/SUPPORT.json')
    y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']]);results=[]
    for fold,seed in cfg['cases']:
        name=f'{fold}_{seed}';out=a.output/'units'/name
        if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
        path=a.source/'units'/f'{name}_cost'/'LATEST.pt';source=torch.load(path,weights_only=False,map_location='cpu')
        snap=torch.load(a.probe/'units'/f'{name}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        prior=a.control/'units'/name;control=torch.load(prior/'ENDPOINT_64.pt',weights_only=False,map_location='cpu')
        resume_path=a.resume/'units'/name/'ENDPOINT_16.pt';resumed=torch.load(resume_path,weights_only=False,map_location='cpu')
        verify_done(resume_path.parent,parents['resume']['identity'])
        validate_resume(resumed,parents['resume']['identity'],file_sha(path),source['accepted'])
        verify_done(prior,parents['control']['identity'])
        if control['source_sha256']!=file_sha(path):raise ValueError('OPTIMAL_CONT_SOURCE')
        old=np.array(support['folds'][fold]['masks']['fit']);limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,y,support,snap,fold,seed,limit);require_equal(ctx.support_receipt,source['support'],'CONT_SUPPORT')
        schedule=ctx.fit[covering_schedule(len(ctx.fit),32768,cfg['batch'],seed)]
        require_equal(tree_digest(schedule),read(resume_path.parent/'RESULT.json')['schedule_sha256'],'RETENTION_SCHEDULE')
        def inspect(state):
            observation=ctx.observe(state,single=True);qs=views(ctx,state,ctx.fit)
            def score(q,target):
                p=cost_parts(q,target,ctx.temperature)
                return dict(cost=float(p['row_cost'].mean()),regret=float(p['row_regret'].mean()),actions_sha256=tree_digest(q.min(0).argmax(1)))
            fit={k:score(q,y[ctx.fit]) for k,q in qs.items()}
            q=native_forward(state,ctx.zn)[0];held={k:score(q[:,support['folds'][fold]['masks'][k]],y[support['folds'][fold]['masks'][k]]) for k in ('stop','audit')}
            return dict(original_passed=observation['passed'],fit=fit,held_internal_train=held)
        baseline=inspect(source['state']);control_metrics=inspect(control['state'])
        require_equal(control_metrics,control['curve'][-1]['metrics'],'RETENTION_CONTROL')
        require_equal(baseline,resumed['curve'][0]['metrics'],'RETENTION_BASELINE')
        require_equal(inspect(resumed['state']),resumed['curve'][-1]['metrics'],'RETENTION_START')
        def restore(state,optimizer,rng):
            ctx.head.load_state_dict(state);opt,_=configure_scope(ctx.head,'full_head',cfg['lr'],cfg['weight_decay'])
            opt.load_state_dict(deepcopy(optimizer));restore_rng(rng);return opt
        for offset in (8,16):
            inherited=torch.load(resume_path.parent/f'ENDPOINT_{offset}.pt',weights_only=False,map_location='cpu')
            checkpoint(out/f'ENDPOINT_{offset}.pt',dict(inherited,identity=cfg['identity'],resume_sha256=file_sha(resume_path)))
        opt=restore(resumed['state'],resumed['optimizer'],resumed['rng'])
        require_equal(opt.state_dict(),resumed['optimizer'],'RETENTION_LOADED_OPTIMIZER')
        require_equal(rng_state(),resumed['rng'],'RETENTION_LOADED_RNG')
        trace=deepcopy(resumed['trace']);curve=deepcopy(resumed['curve']);accepted=resumed['accepted']
        def step(offset):
            ix=schedule[1024+8192+offset-1]
            def repair(before,proposed):
                state,rec,details,_=support_and_repair(ctx,cpu_state(before),cpu_state(proposed))
                return state,dict(rec,details=details,arm='ordinary',searches_used=4)
            return transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],repair,
                loss_fn=lambda q,t:cost_objective(q,t,'cost',ctx.temperature))
        for offset in range(17,65):
            rec=step(offset);accepted+=int(rec['accepted']);trace.append(dict(offset=offset,**rec))
            if offset%16:continue
            validate_retention_suffix(trace[16:],16,offset)
            state=cpu_state(ctx.head.state_dict());metrics=inspect(state)
            if not metrics['original_passed']:raise ValueError('OPTIMAL_CONT_SAVED_GUARD')
            if any(int(v['step'])!=source['accepted']+accepted for v in opt.state.values()):raise ValueError('OPTIMAL_CONT_OPT_COUNT')
            curve.append(dict(offset=offset,accepted=accepted,metrics=metrics))
            cp=dict(identity=cfg['identity'],source_sha256=file_sha(path),offset=offset,accepted=accepted,
                state=state,optimizer=deepcopy(opt.state_dict()),rng=rng_state(),trace=trace.copy(),curve=curve.copy(),
                searches_used=4,resume_sha256=file_sha(resume_path))
            checkpoint(out/f'ENDPOINT_{offset}.pt',cp)
            write(a.output/'PROGRESS.json',dict(stage='FITTING',case=name,offset=offset,accepted=accepted,completed=len(results),seconds=time.monotonic()-start,at=now()))
        for offset in (8,16,32,48,64):
            cp=torch.load(out/f'ENDPOINT_{offset}.pt',weights_only=False,map_location='cpu');require_equal(inspect(cp['state']),cp['curve'][-1]['metrics'],'CONT_COLD')
        for start_offset in (16,32):
            cp=torch.load(out/f'ENDPOINT_{start_offset}.pt',weights_only=False,map_location='cpu');opt=restore(cp['state'],cp['optimizer'],cp['rng'])
            for offset in (start_offset+1,start_offset+2):require_equal(dict(offset=offset,**step(offset)),trace[offset-1],'RETENTION_REPLAY')
        control16=torch.load(prior/'ENDPOINT_16.pt',weights_only=False,map_location='cpu')
        equivalent=all(tree_digest(resumed[k])==tree_digest(control16[k]) for k in ('state','optimizer','rng'))
        if equivalent:
            final=torch.load(out/'ENDPOINT_64.pt',weights_only=False,map_location='cpu')
            for k in ('state','optimizer','rng'):require_equal(final[k],control[k],'RETENTION_NEGATIVE_CONTROL')
        result=dict(fold=fold,seed=seed,curve=curve,control=control_metrics,control_sha256=file_sha(prior/'ENDPOINT_64.pt'),
            accepted=accepted,new_accepted=accepted-resumed['accepted'],cold_exact=5,resumed_exact=4,
            resume_sha256=file_sha(resume_path),new_searches=0,negative_control_exact=equivalent,schedule_sha256=tree_digest(schedule))
        write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('OPTIMAL_CONT_ACTOR_CHANGED')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],seconds=time.monotonic()-start,units=results,actor_updates=0,model_promoted=False))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),completed=4))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','control','probe','coverage','resume','verified','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--verification-output',type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(a)
            from tools.verify_optimal_protection_continuation import run as verify
            if (a.verification_output/'DONE.json').exists():verify_done(a.verification_output,read(a.output/'CONFIG.json')['identity'])
            else:verify(SimpleNamespace(run=a.output,source=a.source,control=a.control,coverage=a.coverage,resume=a.resume,output=a.verification_output))
        except Exception as exc:write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
