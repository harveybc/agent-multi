"""Eight paired fixed-budget fits isolating twin-complete versus active protection."""
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
from app.primal_direction_projection import project_primal
from app.rl_subset import digest
from app.training_forensics import rng_state,restore_rng,tree_digest
from tools.audit_fixed_actor_critics import covering_schedule
from tools.probe_robust_adamw_direction import fit_quality
from tools.run_affine_critic_calibration import views
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def run(a):
    start=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('REPLICATION_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('source','probe','coverage','previous')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    sc=parents['source']
    if (sc['coverage']!=parents['coverage']['identity'] or sc['probe']!=parents['probe']['identity']
            or parents['previous']['parents']['source']!=sc['identity']):raise ValueError('REPLICATION_LINEAGE')
    cfg=dict(schema='twin_protection_replication.v2',parents={k:c['identity'] for k,c in parents.items()},
        budget=256,methods=['active','complete'],cells=[['A',19],['A',43],['B',19],['B',43]],
        transaction_guard='original_batched_plus_cpu_gpu_single',
        lr=sc['lr'],weight_decay=sc['weight_decay'],batch=sc['batch'],source_actor_sha256=sc['source_sha256'],
        code={k:file_sha(ROOT/k) for k in ('tools/run_twin_protection_replication.py','app/twin_complete_protection.py',
            'app/primal_direction_projection.py','app/interior_adamw_repair.py',
            'tools/run_automated_robust_refinement.py','tools/run_decision_cost_contrast.py','app/decision_cost_objective.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('REPLICATION_ACTOR')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('REPLICATION_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-start,**kw))
    support=read(a.coverage/'prepared/SUPPORT.json');data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')
    targets=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']]);results=[]
    for fold,seed in cfg['cells']:
        source_path=a.source/'units'/f'{fold}_{seed}_cost'/'LATEST.pt'
        source=torch.load(source_path,weights_only=False,map_location='cpu')
        snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        old=np.array(support['folds'][fold]['masks']['fit'])
        limit=float(np.mean(targets[old].max(1)-targets[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,targets,support,snap,fold,seed,limit)
        if source['step']!=1024 or ctx.support_receipt!=source['support']:raise ValueError('REPLICATION_SOURCE')
        if ctx.observe(source['state'],single=True)!=source['curve'][-1]['assessment']['observation']:raise ValueError('REPLICATION_SOURCE_REPLAY')
        q0=views(ctx,source['state'],ctx.fit);actions=np.stack([q.min(0).argmax(1) for q in q0.values()])
        mask=ctx.stable&(actions==actions[0]).all(0)
        if not mask.any():raise ValueError('REPLICATION_STABLE')
        schedule=ctx.fit[covering_schedule(len(ctx.fit),32768,cfg['batch'],seed)]
        def inspect(state):
            obs=ctx.observe(state,single=True);qs=views(ctx,state,ctx.fit)
            acts=np.stack([q[:,mask].min(0).argmax(1) for q in qs.values()])
            return dict(robust=obs['passed'],geometry_agrees=bool((acts==acts[0]).all()),observation=obs,
                stable_added={k:fit_quality(q[:,mask],targets[ctx.fit][mask]) for k,q in qs.items()})
        def restore(state,optimizer,rng):
            ctx.head.load_state_dict(state);opt,_=configure_scope(ctx.head,'full_head',cfg['lr'],cfg['weight_decay'])
            opt.load_state_dict(deepcopy(optimizer));restore_rng(rng);return opt
        for method in cfg['methods']:
            name=f'{fold}_{seed}_{method}';out=a.output/'units'/name
            if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
            opt=restore(source['state'],source['optimizer'],source['rng']);begin=0;accepted=0;trace=[];curve=[]
            def step(offset):
                ix=schedule[1024+8192+offset-1]
                return transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],
                    lambda before,proposed:ctx.repair(before,proposed,'interior',projector=project_primal,
                        twin_complete=method=='complete',finite_evaluate=lambda state:ctx.observe(state,single=True)),
                    loss_fn=lambda q,t:cost_objective(q,t,'cost',ctx.temperature))
            if (out/'LATEST.pt').exists():
                cp=torch.load(out/'LATEST.pt',weights_only=False,map_location='cpu')
                if cp['identity']!=cfg['identity'] or cp['source_sha256']!=file_sha(source_path):raise ValueError('REPLICATION_CHECKPOINT')
                opt=restore(cp['state'],cp['optimizer'],cp['rng']);begin=cp['offset'];accepted=cp['accepted'];trace=cp['trace'];curve=cp['curve']
                if inspect(cp['state'])!=curve[-1]['assessment']:raise ValueError('REPLICATION_RESUME_METRICS')
                if begin in (64,256) and not (out/f'ENDPOINT_{begin}.pt').exists():checkpoint(out/f'ENDPOINT_{begin}.pt',cp)
            for offset in range(begin,257):
                if offset==begin and (out/'LATEST.pt').exists():continue
                if offset:
                    rec=step(offset);accepted+=int(rec['accepted']);trace.append(dict(offset=offset,**rec))
                if offset%32:continue
                state={k:v.detach().cpu().clone() for k,v in ctx.head.state_dict().items()};assessment=inspect(state)
                if not assessment['robust']:raise ValueError('REPLICATION_SAVED_GUARD')
                if any(int(v['step'])!=source['accepted']+accepted for v in opt.state.values()):raise ValueError('REPLICATION_OPT_COUNT')
                curve.append(dict(offset=offset,accepted=accepted,assessment=assessment))
                cp=dict(identity=cfg['identity'],source_sha256=file_sha(source_path),state=state,optimizer=deepcopy(opt.state_dict()),
                    rng=rng_state(),offset=offset,accepted=accepted,trace=trace,curve=curve)
                checkpoint(out/'LATEST.pt',cp)
                if offset in (64,256):checkpoint(out/f'ENDPOINT_{offset}.pt',cp)
                progress('FITTING',unit=name,offset=offset,accepted=accepted,completed=len(results),total=8)
            for offset in (64,256):
                cp=torch.load(out/f'ENDPOINT_{offset}.pt',weights_only=False,map_location='cpu')
                if inspect(cp['state'])!=cp['curve'][-1]['assessment']:raise ValueError('REPLICATION_COLD')
            cp=torch.load(out/'ENDPOINT_64.pt',weights_only=False,map_location='cpu');opt=restore(cp['state'],cp['optimizer'],cp['rng'])
            for offset in (65,66,67):
                if dict(offset=offset,**step(offset))!=trace[offset-1]:raise ValueError('REPLICATION_TRANSITION')
            result=dict(fold=fold,seed=seed,method=method,curve=curve,source_sha256=file_sha(source_path),
                schedule_sha256=tree_digest(schedule),stable_added_indices=ctx.fit[mask].tolist(),cold_replays=2,resumed_transitions=3)
            write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('REPLICATION_ACTOR_CHANGED')
    result=dict(identity=cfg['identity'],seconds=time.monotonic()-start,units=results,actor_updates=0,model_promoted=False)
    write(a.output/'RESULT.json',result);seal(a.output,cfg['identity']);progress('COMPLETE',completed=8,total=8)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','probe','coverage','previous','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
