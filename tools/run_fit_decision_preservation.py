"""Bounded actor refinement with source-correct FIT decision preservation."""
import argparse
import copy
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.actor_learning_diagnostics import actor_terms, clear_optimizer_memory
from app.all_state_refinement import policy_effect, validate_panel
from app.bounded_functional_actor import BUDGET, admissible, backtrack, budget_metrics
from app.fit_decision_preservation import protected_actions, margins, preservation, project_step
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle, verify_files
from app.targeted_fit_errors import correction_loss, error_targets, error_progress, requested_margins
from app.three_signal_learning import environment, evaluate
from tools.attribute_ranking_improvement import attribute_pair
from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_operational_actor_transfer import save_canonical
from tools.run_refit_mask_pass import account_equal

ROOT=Path(__file__).resolve().parents[1]


def flat(parameters):
    return torch.cat([p.detach().flatten().double() for p in parameters])


def run(a):
    started=time.monotonic();configure(292);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:
        raise ValueError('PRESERVE_DEVICE')
    source=read(a.source/'CONFIG.json');verify_done(a.source,source['identity'])
    if source['schema']!='selected_input_bounded_refinement.v1':raise ValueError('PRESERVE_SOURCE')
    parent=read(a.parent/'SPEC.json');verify_files(a.parent,parent['files'])
    exported=read(a.bundle.parent/'RESULT.json');verify_done(a.bundle.parent,exported['identity'])
    model,_=load_bundle(a.bundle,'cuda');flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    sha=file_sha(a.bundle/'model.pt')
    if (sha!=source['source_sha256'] or parent['identity']!=source['parent_identity']
            or not exported['qualified'] or model.recipe['identity']!=source['recipe_identity']):
        raise ValueError('PRESERVE_LINEAGE')
    market=market_for(a.parent,parent,model.recipe)
    if support_hash(market.inputs)!=source['support_sha256']:raise ValueError('PRESERVE_SUPPORT')
    records=read(a.source/'panel/RECORDS.json');validate_panel(records)
    rows=np.array([r['row'] for r in records]);fit=np.array([i for i,r in enumerate(records) if r['role']=='fit'])
    if (not market.inputs.valid[rows].all() or not (market.inputs.period[rows]=='TRAIN').all()
            or market.inputs.row_ts_ns[rows].tolist()!=[r['timestamp'] for r in records]):
        raise ValueError('PRESERVE_PANEL')
    x=torch.tensor(market.inputs.windows[rows],device='cuda')
    native=torch.tensor([[o['reward_return'] for o in r['outcomes']] for r in records],device='cuda',dtype=torch.double)
    # Match trainability/kernel dispatch before fixing a sub-micro logit floor.
    set_arm(model,'actor_only')
    with torch.no_grad():source_z=model.surfaces(x).detach()
    protected,actions=protected_actions(source_z,native,records)
    if not len(protected):raise ValueError('PRESERVE_NO_STRICT_FIT_OPTIMUM')
    source_margins=margins(source_z.double(),protected,actions).detach()
    guard=dict(indices=protected.cpu().tolist(),actions=actions.cpu().tolist(),
        rows=rows[protected.cpu().numpy()].tolist(),source_margins=source_margins.cpu().tolist(),
        native_returns=native[protected].cpu().tolist(),tolerance=1e-7)
    targeted=a.target_fit_errors
    error_labels,error_weights=error_targets(source_z,native,records)
    prior_guard=None
    if targeted:
        if a.prior_guard is None:raise ValueError('PRESERVE_PRIOR_REQUIRED')
        prior_guard=read(a.prior_guard/'CONFIG.json');verify_done(a.prior_guard,prior_guard['identity'])
        if (prior_guard['schema']!='fit_decision_preservation.v1'
                or prior_guard['source_identity']!=source['identity']
                or prior_guard['guard']!=guard):raise ValueError('PRESERVE_PRIOR_GUARD')
    cfg=dict(schema='fit_decision_preservation.v1',source_identity=source['identity'],
        source_receipt_sha256=file_sha(a.source/'DONE.json'),source_sha256=sha,
        export_identity=exported['identity'],parent_identity=parent['identity'],
        recipe_identity=model.recipe['identity'],support_sha256=source['support_sha256'],
        effective=model.effective,budget=BUDGET,seeds=[19,43],modes=['guard_only','project_guard'],
        points=[0,1,8],primary_endpoint=8,batch=128,guard=guard,minimum_step_margin_rms=1e-6,
        optimizer_rejection_restores_moments=True,critic_updates=0,model_promoted=False,test_read=False,
        sources={str(p.relative_to(ROOT)):file_sha(p) for folder in ('app','tools','agent_plugins','env_plugins')
                 for p in (ROOT/folder).rglob('*.py')})
    if targeted:
        cfg.update(schema='targeted_fit_error_correction.v1',modes=['error_project_guard'],
            objective='advantage_weighted_fixed_fit_error_cross_entropy',
            prior_guard_identity=prior_guard['identity'],
            prior_guard_receipt_sha256=file_sha(a.prior_guard/'DONE.json'),
            error_targets=error_labels.cpu().tolist(),error_weights=error_weights.cpu().tolist())
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('PRESERVE_CONFIG')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg);write(a.output/'GUARD.json',guard)
    write(a.output/'panel/RECORDS.json',records)
    def progress(stage,**kw):
        value=dict(stage=stage,at=now(),seconds=time.monotonic()-started,identity=cfg['identity'],**kw)
        write(a.output/'PROGRESS.json',value);print(value,flush=True)
    risk=read(a.parent/'RISK.json')
    def make(period):return environment(market,risk,parent['design']['horizon'],
        market.recipe['split']['periods'][period],model.effective['gamma'])
    def logits(current):return emit_actor_reference(current,market.inputs,None,flags)[0]
    def measure(z,path):
        signals=prepared_signals(z)
        return {p:evaluate(make(p),z,prepared=signals,save=path/p,diagnose=True) for p in ('TRAIN','VALIDATION')}
    progress('REFERENCE',protected=len(protected));baseline=logits(model)
    reference=measure(baseline,a.output/'reference')
    for period,value in reference.items():account_equal(value,exported['metrics'][period])
    prior=read(a.source/'RESULT.json');results=[]
    for seed in cfg['seeds']:
        original=f'calibrated_adamw_{seed}'
        effective=read(a.source/original/'EFFECTIVE.json')
        critic=a.source/f'critic_{seed}/MODEL.pt'
        if file_sha(critic)!=effective['critic_sha256']:raise ValueError('PRESERVE_CRITIC_IDENTITY')
        targets=read(a.source/original/'TARGETS.json')
        lookup={int(r):i for i,r in enumerate(rows)}
        schedule=np.array([[lookup[r] for r in batch] for batch in effective['sampled_rows']])
        if schedule.shape!=(8,128) or any(records[i]['role']!='fit' for i in schedule.flatten()):
            raise ValueError('PRESERVE_FIT_SCHEDULE')
        for mode in cfg['modes']:
            key=f'{mode}_{seed}';out=a.output/key
            if (out/'DONE.json').exists():verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
            progress('ARM',arm=key)
            model,_=load_bundle(a.bundle,'cuda');model.load(critic)
            if not np.array_equal(logits(model)[market.inputs.valid],baseline[market.inputs.valid]):
                raise ValueError('PRESERVE_ACTOR_START')
            set_arm(model,'actor_only');clear_optimizer_memory(model.optimizer);reset_learning_rng(seed)
            with torch.no_grad():initial_z=model.surfaces(x).detach()
            if not torch.equal(initial_z,source_z):raise ValueError('PRESERVE_PANEL_START')
            write(out/'INITIAL_GUARD.json',preservation(initial_z,source_margins,protected,actions))
            named=[(n,p) for n,p in model.loss.named_parameters() if p.requires_grad]
            if len(named)!=4 or any('.head.' not in n or not n.startswith('actor_') for n,_ in named):
                raise ValueError('PRESERVE_SCOPE')
            params=[p for _,p in named];theta0=flat(params);fixed=frozen_state(model)
            buffers={n:v.detach().clone() for n,v in model.loss.named_buffers()}
            with torch.no_grad():q=model.q_surfaces(x).min(0).values.detach()
            if not np.array_equal(q.cpu().numpy(),np.array(targets['effective'],dtype=np.float32)):
                raise ValueError('PRESERVE_Q_PARITY')
            alpha=model.loss.log_alpha.detach().exp();steps=[];points=[dict(updates=0,metrics=reference)]
            write(out/'EFFECTIVE.json',dict(critic_sha256=file_sha(critic),sampled_rows=effective['sampled_rows'],
                source_schedule_sha256=effective['schedule_sha256'],alpha=float(alpha),
                trainable=[n for n,_ in named],guard_sha256=file_sha(a.output/'GUARD.json')))
            for step, batch in enumerate(schedule,1):
                ix=torch.tensor(batch,device='cuda');theta=flat(params)
                before=[p.detach().clone() for p in params];optim_before=copy.deepcopy(model.optimizer.state_dict())
                with torch.no_grad():before_z=model.surfaces(x).detach()
                def objective(z):
                    if targeted:return float(correction_loss(z.double(),error_labels,error_weights,ix))
                    v,e,_=actor_terms(z[ix].double(),q[ix].double(),alpha.double());return float(v+e)
                loss_before=objective(before_z)
                model.optimizer.zero_grad(set_to_none=True)
                if targeted:correction_loss(model.surfaces(x).double(),error_labels,error_weights,ix).backward()
                else:
                    v,e,_=actor_terms(model.surfaces(x[ix]),q[ix],alpha);(v+e).backward()
                gradient=flat([p.grad for p in params])
                norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
                model.optimizer.step();direction=flat(params)-theta
                with torch.no_grad():
                    for p,old in zip(params,before):p.copy_(old)
                projection=dict(success=True,nonzero=True,mode='unprojected')
                if mode in ('project_guard','error_project_guard'):
                    z=model.surfaces(x)
                    current=margins(z.double(),protected,actions)
                    jac=torch.stack([flat(torch.autograd.grad(value,params,retain_graph=True))
                                     for value in current.flatten()])
                    lower=(source_margins-current.detach()).flatten()
                    matrix=torch.cat([jac,-gradient[None]],dim=0)
                    bounds=torch.cat([lower,torch.zeros(1,device='cuda',dtype=torch.double)])
                    projected,projection=project_step(direction.cpu().numpy(),matrix.cpu().numpy(),bounds.cpu().numpy())
                    direction=torch.tensor(projected,device='cuda',dtype=torch.double);del z,current
                def check():
                    with torch.no_grad():after=model.surfaces(x).detach()
                    metrics=budget_metrics(source_z[fit],before_z[fit],after[fit],theta0,flat(params))
                    loss_after=objective(after);guard_check=preservation(after,source_margins,protected,actions)
                    moved=any(not torch.equal(p,old) for p,old in zip(params,before))
                    return dict(accepted=bool(moved and metrics['step_margin_rms']>=cfg['minimum_step_margin_rms']
                        and guard_check['passed'] and admissible(metrics,loss_before,loss_after)),
                        metrics=metrics,guard=guard_check,loss_before=loss_before,loss_after=loss_after)
                if projection['success'] and projection['nonzero']:
                    result=backtrack(params,direction,check,BUDGET['backtracks'])
                else:result=dict(accepted=False,fraction=0.,attempts=0,reason='projection_failed_or_zero')
                if not result['accepted']:model.optimizer.load_state_dict(optim_before)
                final_check=check()
                if not final_check['guard']['passed']:raise ValueError('PRESERVE_POST_STEP_GUARD')
                verify_frozen_state(model,fixed)
                if any(not torch.equal(value,dict(model.loss.named_buffers())[name]) for name,value in buffers.items()):
                    raise ValueError('PRESERVE_FROZEN_BUFFER')
                corrective={}
                if targeted:
                    with torch.no_grad():after_z=model.surfaces(x).detach()
                    ids,requested=requested_margins(before_z,error_labels,error_weights,ix)
                    actual=margins(after_z.double(),ids,error_labels[ids])-margins(before_z.double(),ids,error_labels[ids])
                    def native_loss(z):
                        v,e,_=actor_terms(z[ix].double(),q[ix].double(),alpha.double());return float(v+e)
                    corrective=dict(error_rows=rows[ids.cpu().numpy()].tolist(),
                        requested_margin_derivative=requested.cpu().tolist(),actual_margin_change=actual.cpu().tolist(),
                        sign_agreement=float(((actual*requested)>0).double().mean()),
                        loss_before=loss_before,loss_after=objective(after_z),
                        native_loss_before=native_loss(before_z),native_loss_after=native_loss(after_z),
                        progress=error_progress(source_z,after_z,native,records))
                steps.append(dict(updates=step,gradient_norm=float(norm),projection=projection,
                    final_guard=final_check['guard'],corrective=corrective,**result))
                write(out/'STEPS.json',steps);progress('STEP',arm=key,updates=step,accepted=result['accepted'])
                if step in (1,8):
                    z=logits(model);checkpoint=out/f'ACTOR_{step:04d}.pt';save_canonical(model,checkpoint,flags)
                    cold,_=load_bundle(a.bundle,'cuda');cold.load(checkpoint);cold_z=logits(cold)
                    if not np.array_equal(z[market.inputs.valid],cold_z[market.inputs.valid]):raise ValueError('PRESERVE_COLD')
                    metrics=measure(z,out/f'actor_{step:04d}')
                    cold_metrics=measure(cold_z,out/f'cold_{step:04d}')
                    for period,value in metrics.items():account_equal(value,cold_metrics[period])
                    diagnostic={role:policy_effect(baseline[rows[ids]],z[rows[ids]],native.cpu().numpy()[ids])
                        for role in ('fit','monitor')
                        for ids in [np.array([i for i,r in enumerate(records) if r['role']==role])]}
                    points.append(dict(updates=step,metrics=metrics,diagnostic=diagnostic,
                        eligible=all(v['orders_per_week']>2 for v in metrics.values()),
                        changed_actions={p:int((z[market.inputs.valid&(market.inputs.period==p)].argmax(-1)!=
                            baseline[market.inputs.valid&(market.inputs.period==p)].argmax(-1)).sum()) for p in metrics}))
                    write(out/'POINTS.json',points);del cold
            result=dict(arm=key,seed=seed,critic='sealed290',entropy='native',points=points,steps=steps,
                accepted_steps=sum(s['accepted'] for s in steps),frozen_exact=True,cold_exact=True,model_promoted=False)
            write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result)
    attribution=[]
    for arm in results:
        if arm['seed']!=19:continue
        out=a.output/'attribution'/arm['arm']
        value=attribute_pair(lambda:make('TRAIN'),market.inputs.row_ts_ns,a.output/'reference/TRAIN',
            a.output/arm['arm']/'actor_0008/TRAIN',lambda **kw:progress('ATTRIBUTION',arm=arm['arm'],**kw))
        write(out/'EVENTS.json',value['events'])
        result=dict(arm=arm['arm'],**{k:v for k,v in value.items() if k!='events'})
        write(out/'SUMMARY.json',result);seal(out,cfg['identity']);attribution.append(result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm in results:
        for ax,period in zip(axes,('TRAIN','VALIDATION')):
            ax.plot([p['updates'] for p in arm['points']],[p['metrics'][period]['return_pct'] for p in arm['points']],label=arm['arm'])
            ax.set(title=period,xlabel='Attempted actor updates',ylabel='Net return %')
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(a.output/'RETURNS.png');plt.close(fig)
    if file_sha(a.bundle/'model.pt')!=sha:raise ValueError('PRESERVE_SOURCE_CHANGED')
    verify_done(a.source,source['identity']);verify_done(a.bundle.parent,exported['identity'])
    if targeted:verify_done(a.prior_guard,prior_guard['identity'])
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],reference=reference,arms=results,
        prior_controls=prior['arms'],attribution=attribution,protected_rows=len(protected),
        source_unchanged=True,test_read=False,model_promoted=False,seconds=time.monotonic()-started,finished_at=now()))
    seal(a.output,cfg['identity']);progress('COMPLETE')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','bundle','source','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expected-device-name',required=True)
    parser.add_argument('--target-fit-errors',action='store_true')
    parser.add_argument('--prior-guard',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
