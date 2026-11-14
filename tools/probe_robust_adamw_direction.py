"""Bounded FIT-only projection feasibility on authentic saved AdamW states."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.robust_adamw_direction import project_displacement,decision_guard,numpy_heads
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.native_head_warm_start import configure_scope,fit_step
from app.rl_subset import digest
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest
from tools.audit_fixed_actor_critics import covering_schedule
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import template,checkpoint

ROOT=Path(__file__).resolve().parents[1]


def vector(state):
    return np.concatenate([v.detach().cpu().double().numpy().ravel() for v in state.values()])


def moved(state,delta,alpha):
    result={};offset=0
    for name,v in state.items():
        block=delta[offset:offset+v.numel()].reshape(v.shape);offset+=v.numel()
        result[name]=(v.detach().cpu().double()+torch.from_numpy(alpha*block)).to(v.dtype)
    if offset!=len(delta):raise ValueError('DIRECTION_VECTOR_SIZE')
    return result


def gap_jacobian(head,z,old,actions):
    q=head(z[:,:239]).mean(2).min(0).values[old]
    alternatives=np.array([[k for k in range(3) if k!=v] for v in actions])
    chosen=q[torch.arange(len(old)),torch.tensor(actions)]
    gaps=(chosen[:,None]-q.gather(1,torch.tensor(alternatives))).reshape(-1)
    parameters=list(head.parameters());rows=[]
    for gap in gaps:
        grad=torch.autograd.grad(gap,parameters,retain_graph=True)
        rows.append(torch.cat([g.reshape(-1) for g in grad]).detach().numpy())
    return gaps.detach().numpy(),np.stack(rows)


def scores(head,z,fit):
    with torch.no_grad():
        old=head(z[:,:239]).mean(2).cpu().double().numpy()
        expanded=head(z[:,fit]).mean(2).cpu().double().numpy()
    return old,expanded


def fit_quality(q,y):
    acts=q.min(0).argmax(1)
    return dict(mse=float(np.mean((q-y[None])**2)),
        regret=float(np.mean(y.max(1)-y[np.arange(len(y)),acts])))


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('DIRECTION_DEVICE')
    parents=[read(p/'CONFIG.json') for p in (a.previous,a.coverage,a.preserved)]
    for p,c in zip((a.previous,a.coverage,a.preserved),parents):verify_done(p,c['identity'])
    pc,cc,rc=parents
    if rc['previous']!=pc['identity'] or rc['coverage']!=cc['identity']:raise ValueError('DIRECTION_LINEAGE')
    cfg=dict(schema='robust_direction_feasibility.v1',parents=[c['identity'] for c in parents],
        source_sha256=rc['source_sha256'],starts=[0,8192],batches_per_start=8,lr=rc['lr'],
        weight_decay=rc['weight_decay'],batch=rc['batch'],margin_relative=1e-5,
        correction_norm_cap=4.,scales=[2.**(-i) for i in range(8)],regret_tolerance=1e-8,
        mse_learning_tolerance=1e-10,code={k:file_sha(ROOT/k) for k in (
            'tools/probe_robust_adamw_direction.py','app/robust_adamw_direction.py',
            'app/native_head_warm_start.py','app/role_routed_critic.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('DIRECTION_SOURCE')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('DIRECTION_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    if support['old_count']!=239 or support['rows']!=[r['row'] for r in records]:raise ValueError('DIRECTION_SUPPORT')
    zcpu=data['z'];z=zcpu.to('cuda');zn=zcpu.double().numpy()
    targets=np.stack([validate_record(r) for r in records]);y=torch.tensor(targets,dtype=torch.float32,device='cuda')
    results=[];basechecks=[]
    for control in read(a.preserved/'RESULT.json')['arms']:
        fold,seed=control['fold'],control['seed'];info=support['folds'][fold]
        old=np.array(info['masks']['fit']);fit=np.array(info['expanded']);limit=control['original_regret_limit']
        if set(fit)&(set(info['masks']['stop'])|set(info['masks']['audit'])):raise ValueError('DIRECTION_FIT_LEAK')
        schedule=fit[covering_schedule(len(fit),32768,cfg['batch'],seed)]
        if tree_digest(schedule)!=control['schedule_sha256']:raise ValueError('DIRECTION_SCHEDULE')
        for start in cfg['starts']:
            path=(a.previous/f'units/{fold}_{seed}_base/LATEST.pt' if start==0
                  else a.preserved/f'units/{fold}_{seed}/ENDPOINT_{start}.pt')
            source_hash=file_sha(path);saved=torch.load(path,weights_only=False,map_location='cpu')
            initial=saved['model'] if start==0 else saved['state'];initial_hash=tree_digest(initial)
            if start==0 and initial_hash!=control['initialization_sha256']:raise ValueError('DIRECTION_INITIAL')
            head=RoleHeads(template('cuda'),1,seed);head.load_state_dict(initial)
            optimizer,_=configure_scope(head,'full_head',cfg['lr'],cfg['weight_decay'])
            if start:optimizer.load_state_dict(saved['optimizer'])
            initial_opt=deepcopy(optimizer.state_dict());opt_hash=tree_digest(initial_opt)
            qold,qfit=scores(head,z,fit);q=qold.min(0)[old];actions=q.argmax(1)
            margin=cfg['margin_relative']*np.maximum(1.,np.abs(q).max(1))
            native_baseline=fit_quality(qfit,targets[fit]);npbaseline=fit_quality(numpy_heads(initial,zn[:,fit]),targets[fit])
            if decision_guard(q,actions,margin,targets[old],limit)['regret']>limit+1e-8:raise ValueError('DIRECTION_START')
            cpu=RoleHeads(template('cpu'),1,seed);cpu.load_state_dict(initial)
            double=deepcopy(cpu).double();gaps,jac=gap_jacobian(double,zcpu.double(),old,actions)
            required=np.repeat(margin,2)-gaps
            basechecks.append(dict(fold=fold,seed=seed,start=start,initial_sha256=initial_hash,
                optimizer_sha256=opt_hash,source_file_sha256=source_hash,fit_sha256=tree_digest(fit),
                old_sha256=tree_digest(old),native_baseline=native_baseline,numpy_baseline=npbaseline))
            for offset in range(cfg['batches_per_start']):
                name=f'{fold}_{seed}_{start}_{offset}';out=a.output/'units'/name
                if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
                out.mkdir(parents=True,exist_ok=True)
                write(a.output/'PROGRESS.json',dict(stage='PROPOSAL',at=now(),completed=len(results),total=64,unit=name))
                head.load_state_dict(initial);optimizer.load_state_dict(deepcopy(initial_opt))
                ix=schedule[start+offset];fit_step(head,optimizer,z[:,ix],y[ix])
                proposed={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                proposed_opt=deepcopy(optimizer.state_dict());delta=vector(proposed)-vector(initial)
                norm=float(np.linalg.norm(delta));corrected,solve=project_displacement(delta,jac,required)
                ratio=float(np.linalg.norm(corrected)/norm)
                arms={}
                for kind,direction in (('raw',delta),('projected',corrected)):
                    trials=[];accepted=None
                    if kind=='projected' and (not solve['feasible'] or ratio>cfg['correction_norm_cap']):
                        arms[kind]=dict(accepted=False,reason='linear_solver_or_norm_cap',trials=trials);continue
                    for alpha in cfg['scales']:
                        state=moved(initial,direction,alpha);motion=float(np.linalg.norm(vector(state)-vector(initial)))
                        if not np.isfinite(motion) or motion==0:raise ValueError('DIRECTION_NO_MOTION')
                        head.load_state_dict(state);cpu.load_state_dict(state)
                        gpuold,gpufit=scores(head,z,fit);cpuold,_=scores(cpu,zcpu,fit)
                        npold=numpy_heads(state,zn[:,:239]);npfit=numpy_heads(state,zn[:,fit])
                        guards={key:decision_guard(values.min(0)[old],actions,margin,targets[old],limit)
                            for key,values in (('gpu32',gpuold),('cpu32',cpuold),('numpy64',npold))}
                        values=dict(gpu=fit_quality(gpufit,targets[fit]),numpy=fit_quality(npfit,targets[fit]))
                        trial=dict(alpha=alpha,motion=motion,guards=guards,fit=values)
                        trials.append(trial)
                        if all(g['passed'] for g in guards.values()):
                            learning=all(before['mse']-values[key]['mse']>cfg['mse_learning_tolerance']
                                for key,before in (('gpu',native_baseline),('numpy',npbaseline)))
                            regret_improved=all(before['regret']-values[key]['regret']>cfg['regret_tolerance']
                                for key,before in (('gpu',native_baseline),('numpy',npbaseline)))
                            accepted=dict(alpha=alpha,motion=motion,learning_witness=learning,
                                regret_improved=regret_improved,fit=values,guards=guards,state_sha256=tree_digest(state))
                            checkpoint(out/f'{kind}.pt',dict(state=state,identity=cfg['identity']))
                            break
                    arms[kind]=dict(accepted=accepted is not None,endpoint=accepted,trials=trials)
                checkpoint(out/'PROPOSAL.pt',dict(identity=cfg['identity'],initial=initial,proposed=proposed,
                    optimizer_before=initial_opt,optimizer_proposed=proposed_opt,batch_indices=ix,
                    actions=actions,margin=margin,delta=delta,corrected=corrected,required=required))
                result=dict(identity=cfg['identity'],fold=fold,seed=seed,start=start,offset=offset,
                    initial_sha256=initial_hash,optimizer_sha256=opt_hash,batch_sha256=tree_digest(ix),
                    proposed_sha256=tree_digest(proposed),proposal_norm=norm,corrected_norm_ratio=ratio,
                    solver=solve,baseline=dict(gpu=native_baseline,numpy=npbaseline),arms=arms)
                write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
            if file_sha(path)!=source_hash:raise ValueError('DIRECTION_PARENT_MUTATED')
    if len(results)!=64 or file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('DIRECTION_SOURCE_CHANGED')
    summary={kind:dict(accepted=sum(r['arms'][kind]['accepted'] for r in results),
        learning=sum(bool(r['arms'][kind].get('endpoint',{})) and r['arms'][kind]['endpoint']['learning_witness'] for r in results),
        regret_improved=sum(bool(r['arms'][kind].get('endpoint',{})) and r['arms'][kind]['endpoint']['regret_improved'] for r in results))
        for kind in ('raw','projected')}
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],seconds=time.monotonic()-started,summary=summary,
        starts=basechecks,proposals=results,source_preserved=True,actor_updates=0,encoder_updates=0,
        model_promoted=False,validation_read=False,test_read=False))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),completed=64,total=64,summary=summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','coverage','preserved','bundle','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-device-name',required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
