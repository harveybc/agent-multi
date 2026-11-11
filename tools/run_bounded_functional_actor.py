"""Paired bounded actor optimizers using sealed critics; no model promotion."""
import argparse
import copy
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.actor_gradient_coupling import logit_terms
from app.actor_learning_diagnostics import actor_terms, clear_optimizer_memory
from app.all_state_refinement import policy_effect, validate_panel
from app.bounded_functional_actor import BUDGET, admissible, backtrack, budget_metrics, desired_margins, projected_direction
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest, project_inputs, support_hash
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import environment, evaluate
from tools.attribute_ranking_improvement import attribute_pair
from tools.audit_fixed_actor_critics import covering_schedule
from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_operational_actor_transfer import save_canonical
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import market_for

ROOT=Path(__file__).resolve().parents[1]


def run(a):
    started=time.monotonic();configure(284);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:
        raise ValueError('FUNCTIONAL_DEVICE')
    source=read(a.source/'CONFIG.json');verify_done(a.source,source['identity'])
    if source['schema']!='opening_boundary_refinement.v1':raise ValueError('FUNCTIONAL_SOURCE')
    parent=read(a.parent/'SPEC.json');verify_files(a.parent,parent['files'])
    model,_=load_bundle(a.bundle,'cuda');flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    if (parent['identity']!=source['parent_identity'] or file_sha(a.bundle/'model.pt')!=source['source_sha256']
            or model.recipe['identity']!=source['recipe_identity']):raise ValueError('FUNCTIONAL_LINEAGE')
    market=market_for(a.parent,parent);market.inputs=project_inputs(market.inputs,model.recipe);market.recipe=model.recipe
    if support_hash(market.inputs)!=source['support_sha256']:raise ValueError('FUNCTIONAL_SUPPORT')
    records=read(a.source/'panel/RECORDS.json');validate_panel(records)
    rows=np.array([r['row'] for r in records]);fit=np.array([i for i,r in enumerate(records) if r['role']=='fit'])
    if (not market.inputs.valid[rows].all() or not (market.inputs.period[rows]=='TRAIN').all()
            or not np.array_equal(market.inputs.row_ts_ns[rows],[r['timestamp'] for r in records])):
        raise ValueError('FUNCTIONAL_PANEL')
    cfg=dict(schema='bounded_functional_actor.v1',source_identity=source['identity'],
        source_receipt_sha256=file_sha(a.source/'DONE.json'),source_sha256=source['source_sha256'],
        parent_identity=parent['identity'],recipe_identity=model.recipe['identity'],
        support_sha256=source['support_sha256'],effective=model.effective,budget=BUDGET,
        modes=['bounded_adamw','last_layer','function_project'],seeds=[19,43],points=[0,1,8],
        primary_endpoint=8,batch=128,critic='sealed282 absolute',model_promoted=False,test_read=False,
        sources={str(p.relative_to(ROOT)):file_sha(p) for folder in ('app','agent_plugins','env_plugins','tools')
                 for p in (ROOT/folder).rglob('*.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('FUNCTIONAL_CONFIG')
    write(a.output/'CONFIG.json',cfg)
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'panel/RECORDS.json',records)
    def progress(stage,**kw):
        d=dict(stage=stage,at=now(),seconds=time.monotonic()-started,identity=cfg['identity'],**kw)
        write(a.output/'PROGRESS.json',d);print(d,flush=True)
    risk=read(a.parent/'RISK.json');gamma=model.effective['gamma']
    def make(interval):return environment(market,risk,parent['design']['horizon'],interval,gamma)
    def logits(m):return emit_actor_reference(m,market.inputs,None,flags)[0]
    def measure(z,path):
        signals=prepared_signals(z)
        return {p:evaluate(make(interval),z,prepared=signals,save=path/p,diagnose=True)
                for p,interval in market.recipe['split']['periods'].items()}
    progress('REFERENCE');baseline=logits(model);reference=measure(baseline,a.output/'reference')
    for period,m in reference.items():account_equal(m,read(a.source/'REFERENCE.json')[period])
    x=torch.tensor(market.inputs.windows[rows],device='cuda');del model
    targets=np.array([[o['reward_return'] for o in r['outcomes']] for r in records])
    cash=np.array([[o['net_usd'] for o in r['outcomes']] for r in records])
    masks={role:np.array([i for i,r in enumerate(records) if r['role']==role]) for role in ('fit','monitor')}
    def diagnostic(z):
        return {role:dict(actor_reward=policy_effect(baseline[rows[ix]],z[rows[ix]],targets[ix]),
                         actor_cash=policy_effect(baseline[rows[ix]],z[rows[ix]],cash[ix])) for role,ix in masks.items()}
    flat=lambda ps:torch.cat([p.detach().flatten().double() for p in ps])
    results=[]
    for seed in cfg['seeds']:
        old=a.source/f'greedy_calibrated_native_{seed}'
        witness=read(old/'GRADIENT.json');schedule=covering_schedule(len(fit),8,128,seed)
        if digest(schedule.tolist())!=read(old/'EFFECTIVE.json')['schedule_sha256']:
            raise ValueError('FUNCTIONAL_SCHEDULE')
        for mode in cfg['modes']:
            key=f'{mode}_{seed}';out=a.output/key
            if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
            progress('ARM',arm=key)
            model,_=load_bundle(a.bundle,'cuda');model.load(a.source/f'critic_{seed}/MODEL.pt')
            if not np.array_equal(logits(model)[market.inputs.valid],baseline[market.inputs.valid]):
                raise ValueError('FUNCTIONAL_INITIAL_ACTOR')
            set_arm(model,'actor_only')
            if mode=='last_layer':
                for n,p in model.loss.named_parameters():
                    if p.requires_grad:p.requires_grad_('.head.2.' in n)
            clear_optimizer_memory(model.optimizer);reset_learning_rng(seed)
            named=[(n,p) for n,p in model.loss.named_parameters() if p.requires_grad]
            if len(named)!=(2 if mode=='last_layer' else 4):raise ValueError('FUNCTIONAL_SCOPE')
            params=[p for _,p in named]
            all_head=[p for n,p in model.loss.named_parameters() if n.startswith('actor_') and '.head.' in n]
            source_theta=flat(all_head);fixed=frozen_state(model)
            buffers={n:v.detach().clone() for n,v in model.loss.named_buffers()}
            with torch.no_grad():q=model.q_surfaces(x).min(0).values.detach();source_z=model.surfaces(x).detach()
            if not np.array_equal(q.cpu().numpy(),np.array(witness['q'],dtype=np.float32)):
                raise ValueError('FUNCTIONAL_Q_PARITY')
            alpha=model.loss.log_alpha.detach().exp();points=[dict(updates=0,metrics=reference,diagnostic=diagnostic(baseline))];steps=[]
            write(out/'EFFECTIVE.json',dict(mode=mode,budget=BUDGET,alpha=float(alpha),
                trainable=[n for n,_ in named],schedule_sha256=digest(schedule.tolist()),
                critic_sha256=file_sha(a.source/f'critic_{seed}/MODEL.pt'),
                learning_rate=model.effective['learning_rate'],weight_decay=model.effective['weight_decay']))
            for step in range(1,9):
                ix=torch.tensor(fit[schedule[step-1]],device='cuda')
                if any(records[i]['role']!='fit' for i in ix.tolist()):raise ValueError('FUNCTIONAL_FIT_ONLY')
                before=[p.detach().clone() for p in params];theta=flat(params)
                optimizer_before=copy.deepcopy(model.optimizer.state_dict())
                with torch.no_grad():before_z=model.surfaces(x).detach()
                def objective(z):
                    v,e,_=actor_terms(z[ix].double(),q[ix].double(),alpha.double());return float(v+e)
                before_loss=objective(before_z);projection={}
                if mode=='function_project':
                    unique,count=torch.unique(ix,sorted=True,return_counts=True)
                    z=model.surfaces(x[unique]);grad=logit_terms(z.detach().double(),q[unique].double(),alpha.double())['total']
                    weight=count.double();desired=desired_margins(grad,weight,BUDGET['step_margin_rms'])
                    jac=torch.stack([flat(torch.autograd.grad(z[i,d]-z[i,0],params,retain_graph=True))
                                     for i in range(len(unique)) for d in (1,2)]).reshape(len(unique),2,-1)
                    direction,projection=projected_direction(jac,desired,weight,BUDGET['damping_ratio'])
                    projection['sample_unique']=len(unique)
                    direction-=model.effective['learning_rate']*model.effective['weight_decay']*theta
                    del z
                else:
                    model.optimizer.zero_grad(set_to_none=True)
                    v,e,_=actor_terms(model.surfaces(x[ix]),q[ix],alpha);(v+e).backward()
                    norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
                    model.optimizer.step();direction=flat(params)-theta
                    projection['gradient_norm']=float(norm)
                    with torch.no_grad():
                        for p,b in zip(params,before):p.copy_(b)
                def check(current=model):
                    with torch.no_grad():after_z=current.surfaces(x).detach()
                    metrics=budget_metrics(source_z[fit],before_z[fit],after_z[fit],source_theta,flat(all_head))
                    after_loss=objective(after_z)
                    moved=any(not torch.equal(p,b) for p,b in zip(params,before))
                    return dict(accepted=bool(moved and metrics['step_margin_rms']>1e-12 and
                        admissible(metrics,before_loss,after_loss)),metrics=metrics,loss_before=before_loss,loss_after=after_loss)
                result=backtrack(params,direction,check,BUDGET['backtracks'])
                if not result['accepted']:model.optimizer.load_state_dict(optimizer_before)
                if mode=='function_project':
                    with torch.no_grad():actual=model.surfaces(x[unique]).double()
                    change=(actual[:,1:]-actual[:,:1])-(before_z[unique,1:]-before_z[unique,:1]).double()
                    wanted=desired*result['fraction']
                    projection['actual_scaled_target_rmse']=float((change-wanted).square().mean().sqrt())
                    projection['actual_target_cosine']=float((change*wanted).sum()/(change.norm()*wanted.norm())) if float(change.norm()*wanted.norm()) else None
                steps.append(dict(updates=step,projection=projection,**result));write(out/'STEPS.json',steps)
                verify_frozen_state(model,fixed)
                if any(not torch.equal(v,dict(model.loss.named_buffers())[n]) for n,v in buffers.items()):
                    raise ValueError('FUNCTIONAL_BUFFER')
                if step in (1,8):
                    z=logits(model);checkpoint=out/f'ACTOR_{step:04d}.pt';save_canonical(model,checkpoint,flags)
                    cold,_=load_bundle(a.bundle,'cuda');cold.load(checkpoint);cold_z=logits(cold)
                    if not np.array_equal(z[market.inputs.valid],cold_z[market.inputs.valid]):raise ValueError('FUNCTIONAL_COLD')
                    metrics=measure(z,out/f'actor_{step:04d}');cold_metrics=measure(cold_z,out/f'cold_{step:04d}')
                    for period in metrics:account_equal(metrics[period],cold_metrics[period])
                    del cold
                    points.append(dict(updates=step,metrics=metrics,diagnostic=diagnostic(z),
                        eligible=all(m['orders_per_week']>2 for m in metrics.values()),
                        changed_actions={p:int((z[market.inputs.valid&(market.inputs.period==p)].argmax(-1)!=
                            baseline[market.inputs.valid&(market.inputs.period==p)].argmax(-1)).sum()) for p in metrics}))
                    write(out/'POINTS.json',points);progress('POINT',arm=key,updates=step)
            result=dict(arm=key,seed=seed,mode=mode,critic='greedy_calibrated',entropy='native',
                points=points,steps=steps,accepted_steps=sum(s['accepted'] for s in steps),
                frozen_exact=True,cold_exact=True,model_promoted=False)
            write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result);del model
    attribution=[]
    for r in results:
        if r['seed']!=19:continue
        out=a.output/'attribution'/r['arm']
        if (out/'DONE.json').exists():verify_done(out,cfg['identity']);attribution.append(read(out/'SUMMARY.json'));continue
        value=attribute_pair(lambda:make(market.recipe['split']['periods']['TRAIN']),market.inputs.row_ts_ns,
            a.output/'reference/TRAIN',a.output/r['arm']/'actor_0008/TRAIN',lambda **kw:progress('ATTRIBUTION',arm=r['arm'],**kw))
        write(out/'EVENTS.json',value['events']);summary=dict(arm=r['arm'],**{k:v for k,v in value.items() if k!='events'})
        write(out/'SUMMARY.json',summary);seal(out,cfg['identity']);attribution.append(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r in results:
        for ax,period in zip(axes,('TRAIN','VALIDATION')):
            ax.plot([p['updates'] for p in r['points']],[p['metrics'][period]['return_pct'] for p in r['points']],label=r['arm'])
            ax.set(title=period,xlabel='Additional actor updates',ylabel='Net return %')
    axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(a.output/'RETURNS.png');plt.close(fig)
    if file_sha(a.bundle/'model.pt')!=source['source_sha256']:raise ValueError('FUNCTIONAL_SOURCE_CHANGED')
    verify_done(a.source,source['identity'])
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],reference=reference,arms=results,
        historical_unbounded=[r for r in read(a.source/'RESULT.json')['arms'] if r['critic']=='greedy_calibrated'],
        attribution=attribution,model_promoted=False,test_read=False,source_unchanged=True,
        seconds=time.monotonic()-started,finished_at=now()))
    seal(a.output,cfg['identity']);progress('COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
