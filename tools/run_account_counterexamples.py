"""Finite whole-TRAIN counterexample correction, frozen validation and cold export."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.account_counterexamples import new_constraints, account_qualifies
from app.bounded_functional_actor import BUDGET, budget_metrics, admissible
from app.decision_boundary_reachability import target_problem, witness, temporary_head_step
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.optimal_action_sets import SourceValueGuard, error_targets, correction_loss
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, export_bundle, verify_files
from app.three_signal_learning import emit, environment, evaluate
from tools.attribute_ranking_improvement import attribute_pair
from tools.diagnose_selected_refinement import flat, head
from tools.replay_boundary_account import fixed_candidates
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_refit_mask_pass import account_equal


def run(a):
    started=time.monotonic();configure(307);torch.backends.cudnn.allow_tf32=False
    if a.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name):
        raise ValueError('COUNTEREXAMPLE_GPU_IDENTITY')
    probe=read(a.probe/'REPORT.json');verify_done(a.probe,probe['identity'])
    diagnostic=read(a.diagnostic/'CONFIG.json');verify_done(a.diagnostic,diagnostic['identity'])
    floor=read(a.floor/'CONFIG.json');verify_done(a.floor,floor['identity'])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    source=read(a.bundle.parent/'RESULT.json');verify_done(a.bundle.parent,source['identity'])
    sha=file_sha(a.bundle/'model.pt')
    if (probe['source']!=diagnostic['identity'] or diagnostic['floor']!=floor['identity']
            or diagnostic['export']!=source['identity'] or probe['source_sha256']!=sha
            or spec['identity']!=floor['parent_identity']):raise ValueError('COUNTEREXAMPLE_LINEAGE')
    targets_fixed=fixed_candidates(probe)
    if len(targets_fixed)!=1:raise ValueError('COUNTEREXAMPLE_FIXED_SINGLE_WITNESS')
    target_index,target=targets_fixed[0]
    model,_=load_bundle(a.bundle,a.device);market=market_for(a.parent,spec,model.recipe)
    if support_hash(market.inputs)!=floor['support_sha256']:raise ValueError('COUNTEREXAMPLE_SUPPORT')
    periods=model.recipe['split']['periods']
    if set(periods)!={'TRAIN','VALIDATION'}:raise ValueError('COUNTEREXAMPLE_PERIODS')
    cfg=dict(schema='native_account_counterexamples.v1',probe=probe['identity'],floor=floor['identity'],
        source_sha256=sha,support_sha256=support_hash(market.inputs),target=[target_index,target],
        max_rounds=8,max_changed_train_signals=128,guard_mode=a.guard_mode,budget=BUDGET,
        correction_scope=periods['TRAIN'],validation_fit=False,test_read=False,
        device=a.device,device_name=torch.cuda.get_device_name() if a.device=='cuda' else 'cpu',
        code={p:file_sha(Path(__file__).resolve().parents[1]/p) for p in (
            'tools/run_account_counterexamples.py','app/account_counterexamples.py',
            'app/decision_boundary_reachability.py','app/fit_decision_preservation.py',
            'app/optimal_action_sets.py','app/three_signal_learning.py','tools/attribute_ranking_improvement.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:
        raise ValueError('COUNTEREXAMPLE_OUTPUT_COLLISION')
    write(a.output/'CONFIG.json',cfg)
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    risk=read(a.parent/'RISK.json')
    def make(period):return environment(market,risk,spec['design']['horizon'],periods[period],model.effective['gamma'])
    def measure(logits,period,path):return evaluate(make(period),logits,save=path,prepared=prepared_signals(logits))
    train_rows=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
    progress('SOURCE_TRAIN_REPLAY')
    baseline,_=emit(model,market.inputs,indices=train_rows)
    reference=measure(baseline,'TRAIN',a.output/'reference/TRAIN')
    account_equal(reference,source['metrics']['TRAIN'])
    records=read(a.floor/'panel/RECORDS.json');rows=np.array([r['row'] for r in records])
    if (not market.inputs.valid[rows].all() or not (market.inputs.period[rows]=='TRAIN').all()
            or market.inputs.row_ts_ns[rows].tolist()!=[r['timestamp'] for r in records]):
        raise ValueError('COUNTEREXAMPLE_PANEL')
    params=[p for _,p in head(model)];theta=flat(params).cpu()
    saved={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    x=torch.tensor(market.inputs.windows[rows],device=model.device)
    progress('SOURCE_GPU_JACOBIAN')
    z=model.surfaces(x)
    jac=np.stack([flat(torch.autograd.grad(z[i,c],params,retain_graph=True)).cpu().numpy()
                  for i in range(len(rows)) for c in range(3)]).reshape(len(rows),3,-1)
    z0=z.detach().cpu();del z
    native=torch.tensor([[o['reward_return'] for o in r['outcomes']] for r in records],dtype=torch.double)
    guard=SourceValueGuard(z0,native,records);target_mask,weights=error_targets(z0,native,records)
    fit=torch.tensor([i for i,r in enumerate(records) if r['role']=='fit'])
    if weights[target_index]<=0 or not target_mask[target_index,target]:raise ValueError('COUNTEREXAMPLE_TARGET')
    before_loss=float(correction_loss(z0,target_mask,weights,fit))
    direction,target_matrix,target_lower,_=target_problem(z0[target_index].numpy(),jac[target_index],target,len(fit))
    desired=z0.double().masked_fill(~target_mask,-torch.inf).softmax(-1)
    grad=np.einsum('na,nap->p',((z0.double().softmax(-1)-desired)*(weights/weights.sum())[:,None]).numpy(),jac)
    matrix=np.concatenate((jac[guard.constraint_rows,guard.preferred]-jac[guard.constraint_rows,guard.bad],-grad[None]))
    lower=np.concatenate((guard.buffer.numpy(),[0.]))
    constraints=[];rounds=[];selected_step=None;cache={};jac_cache={}
    for iteration in range(1,9):
        progress('PROJECT',round=iteration,constraints=len(constraints))
        out=a.output/f'round_{iteration:02d}'
        if (out/'DONE.json').exists():
            verify_done(out,cfg['identity']);r=read(out/'ROUND.json')
            if r['input_constraints']!=[list(t) for t in constraints]:raise ValueError('COUNTEREXAMPLE_RESUME')
            rounds.append(r);constraints.extend(tuple(v) for v in r['added_constraints'])
            if r['train_qualified']:
                selected_step=np.load(out/'STEP.npy',allow_pickle=False);break
            if r['stop']:break
            continue
        extra=[];rhs=[]
        for row,old,new in constraints:
            if row not in jac_cache:
                logits=model.surfaces(torch.tensor(market.inputs.windows[[row]],device=model.device))
                j=np.stack([flat(torch.autograd.grad(logits[0,c],params,retain_graph=True)).cpu().numpy() for c in range(3)])
                jac_cache[row]=(logits.detach().cpu().numpy()[0],j)
            rz,j=jac_cache[row];extra.append(j[old]-j[new])
            cushion=8*np.finfo(np.float32).eps*(1+abs(rz[old])+abs(rz[new]))
            rhs.append(cushion if a.guard_mode=='floor' else float(rz[new]-rz[old]+cushion))
        all_matrix=np.concatenate((matrix,np.asarray(extra))) if extra else matrix
        all_lower=np.concatenate((lower,rhs)) if extra else lower
        step,projection=witness(direction,target_matrix,target_lower,all_matrix,all_lower)
        r=dict(round=iteration,input_constraints=[list(t) for t in constraints],projection=projection,
            added_constraints=[],train_qualified=False,stop=False)
        write(out/'ROUND.json',r);np.save(out/'STEP.npy',step)
        if not projection['linear_witness']:
            r.update(stop=True,reason='NO_LINEAR_WITNESS')
        else:
            with temporary_head_step(params,step),torch.no_grad():
                after=model.surfaces(x).detach().cpu()
                metrics=budget_metrics(z0[fit],z0[fit],after[fit],theta,flat(params).cpu())
                loss=float(correction_loss(after,target_mask,weights,fit));preserved=guard.check(after)
                r.update(budget=metrics,fit_guard=preserved,loss_before=before_loss,loss_after=loss)
                if (int(after[target_index].argmax())!=target or not preserved['passed']
                        or not admissible(metrics,before_loss,loss)):
                    r.update(stop=True,reason='NONLINEAR_OR_BUDGET_REJECTION')
                else:
                    progress('TRAIN_ACCOUNT',round=iteration)
                    logits,_=emit(model,market.inputs,indices=train_rows)
                    changed=int((logits[train_rows].argmax(-1)!=baseline[train_rows].argmax(-1)).sum())
                    r['changed_train_signals']=changed
                    if changed>128:r.update(stop=True,reason='CHANGED_SIGNAL_BUDGET')
                    else:
                        train_metrics=measure(logits,'TRAIN',out/'TRAIN')
                        attr=attribute_pair(lambda:make('TRAIN'),market.inputs.row_ts_ns,
                            a.output/'reference/TRAIN',out/'TRAIN',
                            lambda **kw:progress('TRAIN_ATTRIBUTION',round=iteration,**kw),
                            isolated_cache=cache,cache_namespace=cfg['identity'])
                        write(out/'ATTRIBUTION.json',attr)
                        added=new_constraints(attr['events'],market.inputs.row_ts_ns,market.inputs.period,set(constraints))
                        r.update(metrics=train_metrics,added_constraints=[list(t) for t in added],
                            attribution={k:v for k,v in attr.items() if k!='events'},
                            train_qualified=bool(account_qualifies(reference,train_metrics,attr['events'])))
                        if r['train_qualified']:selected_step=step.copy();r['stop']=True
                        elif not added:r.update(stop=True,reason='NO_NEW_TRAIN_COUNTEREXAMPLE')
        if any(not torch.equal(v,model.loss.state_dict()[k]) for k,v in saved.items()):
            raise ValueError('COUNTEREXAMPLE_SOURCE_RESTORE')
        write(out/'ROUND.json',r);seal(out,cfg['identity']);rounds.append(r)
        constraints.extend(tuple(t) for t in r['added_constraints'])
        if r['stop']:break
    result=dict(identity=cfg['identity'],rounds=rounds,reference={'TRAIN':reference},
        train_qualified=selected_step is not None,qualified_for_selection=False,
        model_promoted=False,test_read=False,source_unchanged=True,optimizer_steps=0)
    if selected_step is not None:
        progress('FROZEN_VALIDATION_AND_COLD_REPLAY')
        source_logits,_=emit(model,market.inputs)
        ref_val=measure(source_logits,'VALIDATION',a.output/'reference/VALIDATION')
        account_equal(ref_val,source['metrics']['VALIDATION']);result['reference']['VALIDATION']=ref_val
        with temporary_head_step(params,selected_step):
            logits,_=emit(model,market.inputs)
            metrics={p:measure(logits,p,a.output/'final'/p) for p in periods}
            export=a.output/'bundle'
            if not export.exists():
                ix=np.flatnonzero(market.inputs.valid)[np.linspace(0,market.inputs.valid.sum()-1,32,dtype=int)]
                export_bundle(export,model,market.inputs.windows[ix],dict(source=sha,diagnostic=True,round=len(rounds)))
            cold,_=load_bundle(export,a.device);cold_logits,_=emit(cold,market.inputs)
            if not np.array_equal(logits[market.inputs.valid],cold_logits[market.inputs.valid]):raise ValueError('COUNTEREXAMPLE_COLD_LOGITS')
            for p in periods:account_equal(metrics[p],measure(cold_logits,p,a.output/'cold'/p))
            result.update(metrics=metrics,cold_exact=True,
                qualified_for_selection=bool(metrics['VALIDATION']['return_pct']>=ref_val['return_pct']-1e-8
                    and all(m['orders_per_week']>2 for m in metrics.values())))
    if any(not torch.equal(v,model.loss.state_dict()[k]) for k,v in saved.items()):raise ValueError('COUNTEREXAMPLE_FINAL_RESTORE')
    if file_sha(a.bundle/'model.pt')!=sha:raise ValueError('COUNTEREXAMPLE_SOURCE_FILE')
    result['seconds']=time.monotonic()-started
    write(a.output/'RESULT.json',result);seal(a.output,cfg['identity']);progress('COMPLETE')
    print({k:v for k,v in result.items() if k not in ('rounds','reference','metrics')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','floor','probe','diagnostic','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--expected-device-name')
    p.add_argument('--guard-mode',choices=['floor','boundary'],default='floor')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
