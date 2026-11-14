"""Sealed interior-margin feasibility -> paired short fits -> gated extension."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.interior_adamw_repair import repair_direction,transactional_step,extension_gate
from app.native_fit_coverage import validate_record
from app.native_head_warm_start import configure_scope
from app.rl_subset import digest
from app.robust_adamw_direction import decision_guard,numpy_heads
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest,rng_state,restore_rng
from tools.audit_fixed_actor_critics import covering_schedule
from tools.probe_robust_adamw_direction import vector,moved,gap_jacobian,scores,fit_quality
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import template,checkpoint

ROOT=Path(__file__).resolve().parents[1]


class Context:
    def __init__(self,data,targets,support,proposal,fold,seed,limit):
        self.zcpu=data['z'];self.z=self.zcpu.to('cuda');self.zn=self.zcpu.double().numpy()
        self.targets=targets;self.y=torch.tensor(targets,device='cuda',dtype=torch.float32)
        info=support['folds'][fold];self.old=np.array(info['masks']['fit']);self.fit=np.array(info['expanded'])
        if set(self.fit)&(set(info['masks']['stop'])|set(info['masks']['audit'])):raise ValueError('CHAIN_FIT_LEAK')
        self.actions=proposal['actions'];self.margin=proposal['margin'];self.limit=limit
        self.head=RoleHeads(template('cuda'),1,seed);self.head.load_state_dict(proposal['initial'])
        self.cpu=RoleHeads(template('cpu'),1,seed);self.double=deepcopy(self.cpu).double()
        self.initial=deepcopy(proposal['initial']);self.baseline=self.observe(self.initial)

    def observe(self,state):
        self.head.load_state_dict(state);self.cpu.load_state_dict(state)
        old,fit=scores(self.head,self.z,self.fit);cpuold,_=scores(self.cpu,self.zcpu,self.fit)
        npold=numpy_heads(state,self.zn[:,:239]);npfit=numpy_heads(state,self.zn[:,self.fit])
        guards={k:decision_guard(q.min(0)[self.old],self.actions,self.margin,self.targets[self.old],self.limit)
            for k,q in (('gpu32',old),('cpu32',cpuold),('numpy64',npold))}
        for g in guards.values():g['actions_sha256']=digest(g.pop('actions'))
        return dict(passed=bool(all(g['passed'] for g in guards.values())),guards=guards,
            fit=dict(gpu=fit_quality(fit,self.targets[self.fit]),numpy=fit_quality(npfit,self.targets[self.fit])))

    def repair(self,before,proposed,kind):
        before={k:v.detach().cpu().clone() for k,v in before.items()}
        d=vector(proposed)-vector(before)
        def linearize(center):
            state=moved(before,center,1.);self.double.load_state_dict(state)
            gaps,jac=gap_jacobian(self.double,self.zcpu.double(),self.old,self.actions)
            return gaps,jac,vector(state)-vector(before)
        def evaluate(delta):
            state=moved(before,delta,1.)
            return dict(self.observe(state),motion=float(np.linalg.norm(vector(state)-vector(before))))
        result=repair_direction(d,linearize,evaluate,np.repeat(self.margin,2),
            factor=1. if kind=='tangent' else 2.,max_repairs=1 if kind=='tangent' else 3)
        delta=result.pop('delta');state=moved(before,delta,1.) if result['accepted'] else None
        return state,dict(result,proposal_norm=float(np.linalg.norm(d)))

    def assess(self,state,accepted):
        observation=self.observe(state)
        learning=all(self.baseline['fit'][k]['mse']-observation['fit'][k]['mse']>1e-10 for k in ('gpu','numpy'))
        regret=all(self.baseline['fit'][k]['regret']-observation['fit'][k]['regret']>1e-8 for k in ('gpu','numpy'))
        return dict(robust=observation['passed'],learning=bool(learning),regret_improved=bool(regret),
            accepted=accepted,observation=observation)


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('CHAIN_DEVICE')
    oldcfg=read(a.probe/'CONFIG.json');coveragecfg=read(a.coverage/'CONFIG.json')
    verify_done(a.probe,oldcfg['identity']);verify_done(a.coverage,coveragecfg['identity'])
    if oldcfg['parents'][1]!=coveragecfg['identity']:raise ValueError('CHAIN_LINEAGE')
    config=dict(schema='automated_robust_refinement.v1',probe=oldcfg['identity'],coverage=coveragecfg['identity'],
        source_sha256=oldcfg['source_sha256'],lr=oldcfg['lr'],weight_decay=oldcfg['weight_decay'],batch=oldcfg['batch'],
        margin_relative=1e-5,interior_factor=2.,norm_cap=4.,max_repairs=3,
        short_budget=256,extended_budget=1024,acceptance='fixed_actions_and_margin_three_implementations',
        extension='four_robust_MSE_improved_32_accepted_and_three_regret_improved',
        code={k:file_sha(ROOT/k) for k in ('tools/run_automated_robust_refinement.py','app/interior_adamw_repair.py',
            'app/robust_adamw_direction.py','tools/probe_robust_adamw_direction.py','app/native_head_warm_start.py')})
    config['identity']=digest(config)
    if file_sha(a.bundle/'model.pt')!=config['source_sha256']:raise ValueError('CHAIN_SOURCE')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=config:raise ValueError('CHAIN_RESUME_IDENTITY')
    if (a.output/'DONE.json').exists():verify_done(a.output,config['identity']);return
    write(a.output/'CONFIG.json',config)
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    if support['old_count']!=239 or support['rows']!=[r['row'] for r in records]:raise ValueError('CHAIN_SUPPORT')
    targets=np.stack([validate_record(r) for r in records]);probe=read(a.probe/'RESULT.json')
    proposals=probe['proposals'];feasible=[]
    for r in proposals:
        fold,seed,start,offset=[r[k] for k in ('fold','seed','start','offset')]
        name=f'{fold}_{seed}_{start}_{offset}';out=a.output/'feasibility'/name
        if (out/'DONE.json').exists():verify_done(out,config['identity']);feasible.append(read(out/'RESULT.json'));continue
        snap=torch.load(a.probe/'units'/name/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        if tree_digest(snap['initial'])!=r['initial_sha256'] or tree_digest(snap['proposed'])!=r['proposed_sha256']:
            raise ValueError('CHAIN_PROPOSAL_AUTH')
        # Fixed starting actions imply this tighter bound, already below322.
        old=np.array(support['folds'][fold]['masks']['fit']);wanted=snap['actions']
        limit=float(np.mean(targets[old].max(1)-targets[old][np.arange(len(old)),wanted]))
        ctx=Context(data,targets,support,snap,fold,seed,limit)
        state,record=ctx.repair(snap['initial'],snap['proposed'],'interior')
        assessment=ctx.assess(state if state is not None else snap['initial'],int(record['accepted']))
        result=dict(fold=fold,seed=seed,start=start,offset=offset,record=record,assessment=assessment,
            initial_sha256=r['initial_sha256'],limit=limit)
        if state is not None:checkpoint(out/'ENDPOINT.pt',dict(state=state,identity=config['identity']))
        write(out/'RESULT.json',result);seal(out,config['identity']);feasible.append(result)
        progress('FEASIBILITY',completed=len(feasible),total=64)
    cells=[(fold,seed) for fold in ('A','B') for seed in (19,43)]
    ready=all(any(r['fold']==fold and r['seed']==seed and r['start']==8192
        and r['record']['accepted'] and r['assessment']['robust'] and r['assessment']['learning'] for r in feasible)
        for fold,seed in cells)
    write(a.output/'FEASIBILITY_GATE.json',dict(passed=ready,cells=4,accepted=sum(r['record']['accepted'] for r in feasible)))
    summaries=[];extension=False
    if ready:
        def run_unit(fold,seed,kind,budget):
            name=f'{fold}_{seed}_{kind}';out=a.output/'sequential'/name
            snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
            cell=next(r for r in feasible if (r['fold'],r['seed'],r['start'],r['offset'])==(fold,seed,8192,0))
            ctx=Context(data,targets,support,snap,fold,seed,cell['limit'])
            opt,_=configure_scope(ctx.head,'full_head',config['lr'],config['weight_decay']);opt.load_state_dict(snap['optimizer_before'])
            initial_opt_steps={int(v['step']) for v in opt.state.values()}
            if len(initial_opt_steps)!=1:raise ValueError('CHAIN_INITIAL_OPT_STEPS')
            initial_opt_step=next(iter(initial_opt_steps))
            schedule=ctx.fit[covering_schedule(len(ctx.fit),32768,config['batch'],seed)]
            trace=[];curve=[];start=0;accepted=0;latest=out/'LATEST.pt'
            if latest.exists():
                saved=torch.load(latest,weights_only=False,map_location='cpu')
                if saved['identity']!=config['identity'] or saved['initial_sha256']!=tree_digest(snap['initial']):raise ValueError('CHAIN_UNIT_IDENTITY')
                ctx.head.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
                start=saved['step'];trace=saved['trace'];curve=saved['curve'];accepted=saved['accepted']
                if ctx.assess(saved['state'],accepted)!=curve[-1]['assessment']:raise ValueError('CHAIN_RESUME_METRICS')
                if start in (64,256,1024) and not (out/f'ENDPOINT_{start}.pt').exists():checkpoint(out/f'ENDPOINT_{start}.pt',saved)
            if start>budget:return next(p for p in curve if p['step']==budget)
            for step in range(start,budget+1):
                if step==start and latest.exists():continue
                if step:
                    ix=schedule[8192+step-1]
                    rec=transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],lambda before,proposed:ctx.repair(before,proposed,kind))
                    accepted+=int(rec['accepted']);trace.append(dict(step=step,**rec))
                if step not in (0,1,8) and step%32:continue
                state={k:v.detach().cpu().clone() for k,v in ctx.head.state_dict().items()}
                point=dict(step=step,assessment=ctx.assess(state,accepted))
                if accepted and not point['assessment']['robust']:raise ValueError('CHAIN_RETAINED_GUARD')
                if any(int(v['step'])!=initial_opt_step+accepted for v in opt.state.values()):raise ValueError('CHAIN_OPT_COUNT')
                curve.append(point)
                saved=dict(identity=config['identity'],initial_sha256=tree_digest(snap['initial']),state=state,
                    optimizer=deepcopy(opt.state_dict()),rng=rng_state(),step=step,trace=trace,curve=curve,accepted=accepted)
                checkpoint(latest,saved);write(out/'CURVE.json',curve)
                if step in (64,256,1024):checkpoint(out/f'ENDPOINT_{step}.pt',saved)
                progress('SEQUENTIAL',unit=name,step=step,budget=budget,accepted=accepted)
            return curve[-1]
        for fold,seed in cells:
            for kind in ('tangent','interior'):
                point=run_unit(fold,seed,kind,256);summaries.append(dict(fold=fold,seed=seed,kind=kind,**point))
        extension=extension_gate([r['assessment'] for r in summaries if r['kind']=='interior'])
        write(a.output/'EXTENSION_GATE.json',dict(passed=extension,short_results=summaries))
        if extension:
            for fold,seed in cells:
                for kind in ('tangent','interior'):
                    point=run_unit(fold,seed,kind,1024);summaries.append(dict(fold=fold,seed=seed,kind=kind,**point))
        replayed=0;resumed=0
        for fold,seed in cells:
            snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
            cell=next(r for r in feasible if (r['fold'],r['seed'],r['start'],r['offset'])==(fold,seed,8192,0))
            for kind in ('tangent','interior'):
                out=a.output/'sequential'/f'{fold}_{seed}_{kind}'
                ctx=Context(data,targets,support,snap,fold,seed,cell['limit'])
                latest=torch.load(out/'LATEST.pt',weights_only=False,map_location='cpu')
                for path in sorted(out.glob('ENDPOINT_*.pt')):
                    saved=torch.load(path,weights_only=False,map_location='cpu')
                    expected=next(p['assessment'] for p in latest['curve'] if p['step']==saved['step'])
                    if ctx.assess(saved['state'],saved['accepted'])!=expected:raise ValueError('CHAIN_COLD_REPLAY')
                    replayed+=1
                saved=torch.load(out/'ENDPOINT_64.pt',weights_only=False,map_location='cpu')
                ctx.head.load_state_dict(saved['state']);opt,_=configure_scope(ctx.head,'full_head',config['lr'],config['weight_decay'])
                opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
                schedule=ctx.fit[covering_schedule(len(ctx.fit),32768,config['batch'],seed)]
                for step in (65,66,67):
                    ix=schedule[8192+step-1]
                    actual=transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],lambda b,p:ctx.repair(b,p,kind))
                    if dict(step=step,**actual)!=latest['trace'][step-1]:raise ValueError('CHAIN_RESUME_TRANSITION')
                    resumed+=1
                write(out/'RESULT.json',dict(fold=fold,seed=seed,kind=kind,curve=latest['curve'],accepted=latest['accepted']))
                seal(out,config['identity'])
        write(a.output/'REPLAY.json',dict(cold_endpoints_exact=replayed,resumed_transitions_exact=resumed))
    if file_sha(a.bundle/'model.pt')!=config['source_sha256']:raise ValueError('CHAIN_SOURCE_CHANGED')
    verify_done(a.probe,oldcfg['identity']);verify_done(a.coverage,coveragecfg['identity'])
    result=dict(identity=config['identity'],seconds=time.monotonic()-started,feasibility_ready=ready,
        feasibility=feasible,extension=extension,sequential=summaries,model_promoted=False,
        source_preserved=True,actor_updates=0,encoder_updates=0,validation_read=False,test_read=False)
    write(a.output/'RESULT.json',result);seal(a.output,config['identity'])
    progress('COMPLETE',feasibility_ready=ready,extended=extension,sequential_arms=8 if ready else 0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','coverage','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            write(a.output/'PROGRESS.json',dict(stage='FAILED',at=now(),error=repr(exc)));raise
