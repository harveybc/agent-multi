"""Replay stalled AdamW proposals with paired dual/primal direction solvers."""
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
from app.primal_direction_projection import project_primal,select_stalled_probes,qualify_continuation
from app.twin_complete_protection import qualify_twin_continuation
from app.rl_subset import digest
from app.robust_adamw_direction import project_displacement
from app.training_forensics import tree_digest,rng_state,restore_rng
from tools.audit_fixed_actor_critics import covering_schedule
from tools.probe_robust_adamw_direction import fit_quality
from tools.run_affine_critic_calibration import views
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import checkpoint

ROOT=Path(__file__).resolve().parents[1]


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('SOLVER_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('source','probe','coverage')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    sc=parents['source']
    if sc['probe']!=parents['probe']['identity'] or sc['coverage']!=parents['coverage']['identity'] or sc['optimizer_state']!='fresh':raise ValueError('SOLVER_LINEAGE')
    source_file=a.source/'units/B_43_cost/LATEST.pt'
    saved=torch.load(source_file,weights_only=False,map_location='cpu')
    if saved['step']!=1024:raise ValueError('SOLVER_SOURCE_STEP')
    last,selected=select_stalled_probes(saved['trace'])
    cfg=dict(schema='stalled_solver_contrast.v1',parents={k:c['identity'] for k,c in parents.items()},
        source_file_sha256=file_sha(source_file),source_actor_sha256=sc['source_sha256'],last_accepted=last,
        selected_attempts=selected,continuation_budget=256,lr=sc['lr'],weight_decay=sc['weight_decay'],batch=sc['batch'],
        code={k:file_sha(ROOT/k) for k in ('tools/run_stalled_solver_contrast.py','app/primal_direction_projection.py',
            'app/interior_adamw_repair.py','app/robust_adamw_direction.py','tools/run_automated_robust_refinement.py',
            'tools/run_decision_cost_contrast.py','tools/run_affine_critic_calibration.py','app/decision_cost_objective.py')})
    complete=getattr(a,'twin_complete',False)
    if complete:
        if a.previous_solver is None:raise ValueError('SOLVER_PREVIOUS_REQUIRED')
        pc=read(a.previous_solver/'CONFIG.json');verify_done(a.previous_solver,pc['identity'])
        if pc['selected_attempts']!=selected or pc['source_file_sha256']!=cfg['source_file_sha256']:
            raise ValueError('SOLVER_UNPAIRED_PREVIOUS')
        cfg.update(schema='twin_complete_protection.v1',previous_solver=pc['identity'],twin_complete=True,
            continuation_gate='twelve_safe_nonworsening_cost_improved_probes',
            twin_complete_code=file_sha(ROOT/'app/twin_complete_protection.py'))
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('SOLVER_ACTOR_IDENTITY')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('SOLVER_RESUME_IDENTITY')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**extra):write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**extra))
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');y=np.stack([validate_record(r) for r in records])
    snap=torch.load(a.probe/'units/B_43_8192_0/PROPOSAL.pt',weights_only=False,map_location='cpu')
    old=np.array(support['folds']['B']['masks']['fit']);limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
    ctx=CostContext(data,y,support,snap,'B',43,limit)
    if ctx.support_receipt!=saved['support']:raise ValueError('SOLVER_SUPPORT')
    frozen=saved['state'];start=ctx.observe(frozen,single=True)
    if start!=saved['curve'][-1]['assessment']['observation'] or not start['passed']:raise ValueError('SOLVER_START_REPLAY')
    start_views=views(ctx,frozen,ctx.fit);acts=np.stack([q.min(0).argmax(1) for q in start_views.values()])
    mask=ctx.stable&(acts==acts[0]).all(0)
    if not mask.any():raise ValueError('SOLVER_STABLE_ROWS')
    schedule=ctx.fit[covering_schedule(len(ctx.fit),32768,cfg['batch'],43)]
    loss=lambda q,t:cost_objective(q,t,'cost',ctx.temperature)
    def restore(state=frozen,optimizer=saved['optimizer'],rng=saved['rng']):
        ctx.head.load_state_dict(state);opt,_=configure_scope(ctx.head,'full_head',cfg['lr'],cfg['weight_decay'])
        opt.load_state_dict(deepcopy(optimizer));restore_rng(rng);return opt
    def inspect(state):
        obs=ctx.observe(state,single=True);scores=views(ctx,state,ctx.fit)
        actions=np.stack([q[:,mask].min(0).argmax(1) for q in scores.values()])
        return dict(robust=obs['passed'],geometry_agrees=bool((actions==actions[0]).all()),observation=obs,
            stable_added={k:fit_quality(q[:,mask],y[ctx.fit][mask]) for k,q in scores.items()})
    baseline=inspect(frozen)
    def step(opt,attempt,method,capture=None):
        projector=project_displacement if method=='legacy' else project_primal
        def project(d,j,b):
            p,receipt=projector(d,j,b)
            if capture is not None and method=='primal':capture['systems'].append(dict(d=d,j=j,b=b,p=p,receipt=receipt))
            return p,receipt
        def repair(before,proposed):
            if capture is not None:
                capture['proposal']={k:v.detach().cpu().clone() for k,v in proposed.items()}
                capture['proposed_optimizer']=deepcopy(opt.state_dict())
            return ctx.repair(before,proposed,'interior',projector=project,twin_complete=complete and method=='primal')
        ix=schedule[8192+attempt-1]
        return transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],repair,loss_fn=loss)
    results=[];replayed=0
    for attempt in selected:
        for method in ('legacy','primal'):
            out=a.output/'probes'/f'{attempt}_{method}'
            if (out/'DONE.json').exists():
                verify_done(out,cfg['identity']);result=read(out/'RESULT.json')
                cp=torch.load(out/'ENDPOINT.pt',weights_only=False,map_location='cpu')
                if inspect(cp['state'])!=result['assessment']:raise ValueError('SOLVER_CACHED_PROBE')
                if method=='legacy':
                    if dict(step=attempt,**result['record'])!=saved['trace'][attempt-1]:raise ValueError('SOLVER_CACHED_CONTROL')
                    replayed+=1
                results.append(result);continue
            opt=restore();capture=dict(systems=[])
            record=step(opt,attempt,method,capture)
            if method=='legacy':
                if dict(step=attempt,**record)!=saved['trace'][attempt-1]:raise ValueError('SOLVER_LEGACY_REPLAY')
                if tree_digest(ctx.head.state_dict())!=tree_digest(frozen) or tree_digest(opt.state_dict())!=tree_digest(saved['optimizer']):raise ValueError('SOLVER_REJECTION_CHANGED_STATE')
                replayed+=1
            if not record['accepted'] and (tree_digest(ctx.head.state_dict())!=tree_digest(frozen)
                    or tree_digest(opt.state_dict())!=tree_digest(saved['optimizer'])):raise ValueError('SOLVER_REJECTION_CHANGED_STATE')
            if complete:
                prior=torch.load(a.previous_solver/'probes'/f'{attempt}_{method}'/'ENDPOINT.pt',weights_only=False,map_location='cpu')
                if (tree_digest(capture['proposal'])!=tree_digest(prior['proposal'])
                        or tree_digest(capture['proposed_optimizer'])!=tree_digest(prior['proposed_optimizer'])):
                    raise ValueError('SOLVER_UNPAIRED_ACTUAL_PROPOSAL')
            state={k:v.detach().cpu().clone() for k,v in ctx.head.state_dict().items()};assessment=inspect(state)
            checkpoint(out/'ENDPOINT.pt',dict(identity=cfg['identity'],state=state,optimizer=deepcopy(opt.state_dict()),
                record=record,assessment=assessment,batch_indices=schedule[8192+attempt-1],**capture))
            restored=torch.load(out/'ENDPOINT.pt',weights_only=False,map_location='cpu')
            if inspect(restored['state'])!=assessment:raise ValueError('SOLVER_PROBE_COLD')
            result=dict(attempt=attempt,source_reason=saved['trace'][attempt-1]['reason'],method=method,record=record,assessment=assessment)
            write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
            progress('PROBES',completed=len(results),total=32)
    gate_rows=[dict(attempt=r['attempt'],accepted=r['record']['accepted'],robust=r['assessment']['robust'],
        geometry_agrees=r['assessment']['geometry_agrees'],before=baseline['stable_added']['gpu']['regret'],
        after=r['assessment']['stable_added']['gpu']['regret']) for r in results if r['method']=='primal']
    if complete:
        for row,r in zip(gate_rows,[r for r in results if r['method']=='primal']):
            row.update(cost_before=baseline['observation']['fit']['gpu']['cost'],
                cost_after=r['assessment']['observation']['fit']['gpu']['cost'])
    qualifies=(qualify_twin_continuation if complete else qualify_continuation)(gate_rows)
    write(a.output/'GATE.json',dict(passed=qualifies,rows=gate_rows))
    continuation=[];resumed=0
    if qualifies:
        for method in ('legacy','primal'):
            out=a.output/'continuation'/method
            if (out/'DONE.json').exists():
                verify_done(out,cfg['identity']);continuation.append(read(out/'RESULT.json'));resumed+=3;continue
            opt=restore();trace=[];curve=[];accepted=0;begin=0;latest=out/'LATEST.pt'
            if latest.exists():
                cp=torch.load(latest,weights_only=False,map_location='cpu')
                if cp['identity']!=cfg['identity']:raise ValueError('SOLVER_CONTINUATION_IDENTITY')
                opt=restore(cp['state'],cp['optimizer'],cp['rng']);begin=cp['offset'];accepted=cp['accepted']
                trace=cp['trace'];curve=cp['curve']
                if inspect(cp['state'])!=curve[-1]['assessment']:raise ValueError('SOLVER_CONTINUATION_RESTORE')
                if begin in (64,256) and not (out/f'ENDPOINT_{begin}.pt').exists():checkpoint(out/f'ENDPOINT_{begin}.pt',cp)
            for offset in range(begin,257):
                if offset==begin and latest.exists():continue
                if offset:
                    rec=step(opt,1024+offset,method);accepted+=int(rec['accepted']);trace.append(dict(offset=offset,**rec))
                if offset%32:continue
                state={k:v.detach().cpu().clone() for k,v in ctx.head.state_dict().items()}
                point=dict(offset=offset,accepted=accepted,assessment=inspect(state));curve.append(point)
                if any(int(v['step'])!=saved['accepted']+accepted for v in opt.state.values()):raise ValueError('SOLVER_OPT_COUNTER')
                current=dict(identity=cfg['identity'],state=state,optimizer=deepcopy(opt.state_dict()),rng=rng_state(),
                    offset=offset,accepted=accepted,trace=trace,curve=curve)
                checkpoint(out/'LATEST.pt',current)
                if offset in (64,256):checkpoint(out/f'ENDPOINT_{offset}.pt',current)
                progress('CONTINUATION',method=method,offset=offset,accepted=accepted)
            for offset in (64,256):
                cp=torch.load(out/f'ENDPOINT_{offset}.pt',weights_only=False,map_location='cpu')
                if inspect(cp['state'])!=cp['curve'][-1]['assessment']:raise ValueError('SOLVER_CONTINUATION_COLD')
            cp=torch.load(out/'ENDPOINT_64.pt',weights_only=False,map_location='cpu');opt=restore(cp['state'],cp['optimizer'],cp['rng'])
            for offset in (65,66,67):
                rec=step(opt,1024+offset,method)
                if dict(offset=offset,**rec)!=trace[offset-1]:raise ValueError('SOLVER_CONTINUATION_RESUME')
                resumed+=1
            result=dict(method=method,curve=curve);write(out/'RESULT.json',result);seal(out,cfg['identity']);continuation.append(result)
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('SOLVER_ACTOR_CHANGED')
    result=dict(identity=cfg['identity'],seconds=time.monotonic()-started,baseline=baseline,stable_added_indices=ctx.fit[mask].tolist(),
        original_controls_exact=replayed,probe_cold_replays=len(results),resumed_transitions_exact=resumed,
        probes=results,continuation_gate=qualifies,continuation=continuation,model_promoted=False,actor_updates=0)
    write(a.output/'RESULT.json',result)
    lines=['# Stalled Solver Contrast','','Native FIT decisions, not trading performance.','',
        '| Attempt | Prior cause | Solver | Accepted | Robust | Stable regret | Reason |',
        '| ---: | --- | --- | --- | --- | ---: | --- |']
    for r in results:
        x=r['assessment'];lines.append(f"| {r['attempt']} | {r['source_reason']} | {r['method']} | {r['record']['accepted']} | {x['robust']} | {x['stable_added']['gpu']['regret']:.9f} | {r['record']['reason']} |")
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');seal(a.output,cfg['identity']);progress('COMPLETE',continuation=qualifies)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','probe','coverage','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--twin-complete',action='store_true');p.add_argument('--previous-solver',type=Path)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
