"""Four gated64-step optimal-set continuations against saved identical-budget338 controls."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time
import numpy as np
import torch
from app.decision_cost_objective import cost_objective
from app.frozen_recipe_inputs import file_sha
from app.interior_adamw_repair import transactional_step
from app.native_fit_coverage import validate_record
from app.native_head_warm_start import configure_scope
from app.rl_subset import digest
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
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('source','control','probe','coverage','protection','verified')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if parents['verified']!=parents['protection']:raise ValueError('OPTIMAL_CONT_VERIFY_ID')
    if not read(a.protection/'RESULT.json')['continuation_gate'] or not read(a.verified/'VERIFICATION.json')['continuation_gate']:raise ValueError('OPTIMAL_CONT_GATE')
    for k in ('source','probe','coverage'):
        if parents['control']['parents'][k]!=parents[k]['identity']:raise ValueError('OPTIMAL_CONT_LINEAGE')
    for k in ('probe','coverage'):
        if parents['protection']['parents'][k]!=parents[k]['identity']:raise ValueError('OPTIMAL_CONT_PROTECTION_LINEAGE')
    for k,sha in parents['protection']['code'].items():
        if file_sha(ROOT/k)!=sha:raise ValueError('OPTIMAL_CONT_PROTECTION_CODE')
    sc=parents['source'];cfg=dict(schema='optimal_protection_continuation.v1',parents={k:c['identity'] for k,c in parents.items()},
        budget=64,cases=parents['control']['cells'],source_actor_sha256=sc['source_sha256'],lr=sc['lr'],weight_decay=sc['weight_decay'],batch=sc['batch'],
        code={k:file_sha(ROOT/k) for k in ('tools/run_optimal_protection_continuation.py','tools/probe_optimal_fit_protection.py','app/optimal_fit_protection.py')})
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
        prior=a.control/'units'/f'{name}_complete';control=torch.load(prior/'ENDPOINT_64.pt',weights_only=False,map_location='cpu')
        verify_done(prior,parents['control']['identity'])
        if control['source_sha256']!=file_sha(path):raise ValueError('OPTIMAL_CONT_SOURCE')
        old=np.array(support['folds'][fold]['masks']['fit']);limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,y,support,snap,fold,seed,limit);require_equal(ctx.support_receipt,source['support'],'CONT_SUPPORT')
        require_equal(ctx.observe(control['state'],single=True),control['curve'][-1]['assessment']['observation'],'CONT_CONTROL')
        schedule=ctx.fit[covering_schedule(len(ctx.fit),32768,cfg['batch'],seed)]
        require_equal(tree_digest(schedule),read(prior/'RESULT.json')['schedule_sha256'],'CONT_SCHEDULE')
        def inspect(state):
            observation=ctx.observe(state,single=True);qs=views(ctx,state,ctx.fit)
            def score(q,target):
                p=cost_parts(q,target,ctx.temperature)
                return dict(cost=float(p['row_cost'].mean()),regret=float(p['row_regret'].mean()),actions_sha256=tree_digest(q.min(0).argmax(1)))
            fit={k:score(q,y[ctx.fit]) for k,q in qs.items()}
            q=native_forward(state,ctx.zn)[0];held={k:score(q[:,support['folds'][fold]['masks'][k]],y[support['folds'][fold]['masks'][k]]) for k in ('stop','audit')}
            return dict(original_passed=observation['passed'],fit=fit,held_internal_train=held)
        baseline=inspect(source['state']);control_metrics=inspect(control['state'])
        def restore(state,optimizer,rng):
            ctx.head.load_state_dict(state);opt,_=configure_scope(ctx.head,'full_head',cfg['lr'],cfg['weight_decay'])
            opt.load_state_dict(deepcopy(optimizer));restore_rng(rng);return opt
        opt=restore(source['state'],source['optimizer'],source['rng']);trace=[];curve=[dict(offset=0,accepted=0,metrics=baseline)];accepted=0
        def step(offset):
            ix=schedule[1024+8192+offset-1]
            def repair(before,proposed):
                state,rec,details,_=support_and_repair(ctx,cpu_state(before),cpu_state(proposed))
                return state,dict(rec,details=details)
            return transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],repair,
                loss_fn=lambda q,t:cost_objective(q,t,'cost',ctx.temperature))
        for offset in range(1,65):
            rec=step(offset);accepted+=int(rec['accepted']);trace.append(dict(offset=offset,**rec))
            if offset%16:continue
            state=cpu_state(ctx.head.state_dict());metrics=inspect(state)
            if not metrics['original_passed']:raise ValueError('OPTIMAL_CONT_SAVED_GUARD')
            if any(int(v['step'])!=source['accepted']+accepted for v in opt.state.values()):raise ValueError('OPTIMAL_CONT_OPT_COUNT')
            curve.append(dict(offset=offset,accepted=accepted,metrics=metrics))
            cp=dict(identity=cfg['identity'],source_sha256=file_sha(path),offset=offset,accepted=accepted,
                state=state,optimizer=deepcopy(opt.state_dict()),rng=rng_state(),trace=trace.copy(),curve=curve.copy())
            checkpoint(out/f'ENDPOINT_{offset}.pt',cp)
            write(a.output/'PROGRESS.json',dict(stage='FITTING',case=name,offset=offset,accepted=accepted,completed=len(results),seconds=time.monotonic()-start,at=now()))
        for offset in (16,32,48,64):
            cp=torch.load(out/f'ENDPOINT_{offset}.pt',weights_only=False,map_location='cpu');require_equal(inspect(cp['state']),cp['curve'][-1]['metrics'],'CONT_COLD')
        cp=torch.load(out/'ENDPOINT_16.pt',weights_only=False,map_location='cpu');opt=restore(cp['state'],cp['optimizer'],cp['rng'])
        for offset in (17,18):require_equal(dict(offset=offset,**step(offset)),trace[offset-1],'CONT_REPLAY')
        result=dict(fold=fold,seed=seed,curve=curve,control=control_metrics,control_sha256=file_sha(prior/'ENDPOINT_64.pt'),
            accepted=accepted,cold_exact=4,resumed_exact=2)
        write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('OPTIMAL_CONT_ACTOR_CHANGED')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],seconds=time.monotonic()-start,units=results,actor_updates=0,model_promoted=False))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),completed=4))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','control','probe','coverage','protection','verified','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
