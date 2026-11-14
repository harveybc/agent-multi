"""Exact338 complete-arm replay with first-crossing stage attribution and witnesses."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time
import numpy as np
import torch
from app.crossing_step_forensics import attribute,harmful_rows
from app.decision_cost_objective import cost_objective
from app.frozen_recipe_inputs import file_sha
from app.interior_adamw_repair import transactional_step
from app.native_fit_coverage import validate_record
from app.native_head_warm_start import configure_scope
from app.primal_direction_projection import project_primal
from app.rl_subset import digest
from app.training_forensics import restore_rng,rng_state,tree_digest
from app.weight_decision_forensics import native_forward
from tools.audit_fixed_actor_critics import covering_schedule
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def cpu_state(state):return {k:v.detach().cpu().clone() for k,v in state.items()}


def require_equal(actual,expected,kind):
    if tree_digest(actual)!=tree_digest(expected):raise ValueError('CROSS_REPLAY_'+kind)


def run(a):
    start=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('CROSS_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('source','probe','coverage','previous')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    pc=parents['previous'];sc=parents['source']
    if any(pc['parents'][k]!=parents[k]['identity'] for k in ('source','probe','coverage')):raise ValueError('CROSS_LINEAGE')
    for name,sha in pc['code'].items():
        if file_sha(ROOT/name)!=sha:raise ValueError('CROSS_PRODUCER_CODE:'+name)
    cfg=dict(schema='first_crossing_replay.v1',parents={k:c['identity'] for k,c in parents.items()},
        source_actor_sha256=pc['source_actor_sha256'],budget=256,cases=pc['cells'],mode='complete',
        capture='first_harmful_transition_per_expanded_FIT_row',
        code={k:file_sha(ROOT/k) for k in ('tools/replay_first_harmful_crossings.py','app/crossing_step_forensics.py','app/weight_decision_forensics.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('CROSS_ACTOR')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('CROSS_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')
    z=data['z'].double().numpy();y=np.stack([validate_record(r) for r in records]);results=[]
    for fold,seed in cfg['cases']:
        out=a.output/'cases'/f'{fold}_{seed}'
        if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
        source_path=a.source/'units'/f'{fold}_{seed}_cost'/'LATEST.pt'
        source=torch.load(source_path,weights_only=False,map_location='cpu')
        snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        prior_dir=a.previous/'units'/f'{fold}_{seed}_complete';verify_done(prior_dir,pc['identity'])
        prior=torch.load(prior_dir/'ENDPOINT_256.pt',weights_only=False,map_location='cpu')
        if prior['source_sha256']!=file_sha(source_path):raise ValueError('CROSS_SOURCE_FILE')
        old=np.array(support['folds'][fold]['masks']['fit'])
        limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,y,support,snap,fold,seed,limit)
        require_equal(ctx.support_receipt,source['support'],'SUPPORT')
        schedule=ctx.fit[covering_schedule(len(ctx.fit),32768,sc['batch'],seed)]
        require_equal(tree_digest(schedule),read(prior_dir/'RESULT.json')['schedule_sha256'],'SCHEDULE')
        ctx.head.load_state_dict(source['state']);opt,_=configure_scope(ctx.head,'full_head',sc['lr'],sc['weight_decay'])
        opt.load_state_dict(deepcopy(source['optimizer']));restore_rng(source['rng'])
        seen=set();witnesses=[];trace=[];all_harmful=0
        for offset in range(1,257):
            ix=schedule[1024+8192+offset-1];before=cpu_state(ctx.head.state_dict());old_opt=deepcopy(opt.state_dict());rng=rng_state();capture={}
            def repair(b,p):
                capture['proposed']=cpu_state(p);capture['optimizer']=deepcopy(opt.state_dict())
                capture['gradient']=torch.cat([p.grad.detach().reshape(-1).cpu() for p in ctx.head.parameters()]).double().numpy()
                return ctx.repair(b,p,'interior',projector=project_primal,twin_complete=True,
                    finite_evaluate=lambda state:ctx.observe(state,single=True))
            rec=transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],repair,
                loss_fn=lambda q,t:cost_objective(q,t,'cost',ctx.temperature))
            expected=prior['trace'][offset-1];actual=dict(offset=offset,**rec)
            require_equal(actual,expected,'TRANSACTION');trace.append(actual)
            after=cpu_state(ctx.head.state_dict());q0=native_forward(before,z)[0];q1=native_forward(after,z)[0]
            bad=harmful_rows(q0,q1,y,ctx.fit);all_harmful+=len(bad);first=[i for i in bad if i not in seen];seen.update(first)
            if first:
                analysis=attribute(before,capture['proposed'],after,old_opt,capture['optimizer'],capture['gradient'],z,y,ctx.fit,ix,np.array(first),ctx.temperature)
                for event in analysis['events']:event['kind']=records[event['row']]['kind']
                witness=dict(identity=cfg['identity'],fold=fold,seed=seed,offset=offset,before=before,proposed=capture['proposed'],accepted=after,
                    optimizer_before=old_opt,optimizer_proposed=capture['optimizer'],optimizer_after=deepcopy(opt.state_dict()),
                    rng_before=rng,rng_after=rng_state(),batch=ix,rows=first,actual_gradient=capture['gradient'],receipt=actual)
                path=out/'witnesses'/f'{offset:03d}.pt';checkpoint(path,witness)
                witnesses.append(dict(offset=offset,file_sha256=file_sha(path),**analysis))
            if offset in (64,256):
                saved=torch.load(prior_dir/f'ENDPOINT_{offset}.pt',weights_only=False,map_location='cpu')
                for k,value in [('state',after),('optimizer',opt.state_dict()),('rng',rng_state())]:require_equal(value,saved[k],'ENDPOINT_'+k)
            if offset%32==0:write(a.output/'PROGRESS.json',dict(stage='REPLAYING',case=f'{fold}_{seed}',offset=offset,
                first_harmful_rows=len(seen),seconds=time.monotonic()-start,at=now()))
        result=dict(fold=fold,seed=seed,transactions_exact=len(trace),endpoints_exact=2,
            first_harmful_rows=len(seen),all_harmful_transition_occurrences=all_harmful,witnesses=witnesses)
        write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('CROSS_ACTOR_CHANGED')
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],seconds=time.monotonic()-start,cases=results,
        actor_updates=0,model_promoted=False,replayed_transactions=sum(r['transactions_exact'] for r in results)))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','probe','coverage','previous','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
