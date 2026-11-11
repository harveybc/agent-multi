"""Bounded own-policy all-state critic x entropy diagnostic; no promotion."""
import argparse
from collections import Counter
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from app.actor_learning_diagnostics import clear_optimizer_memory
from app.all_state_refinement import BOUNDARY, choose_panel, validate_panel, policy_effect, qualification_identity
from app.action_ranking_contrast import target_loss
from app.critic_fit_probe import fit_metrics, probe_q, set_scope
from app.critic_refinement import begin_stage
from app.decision_attribution import continuation, state_view
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.operational_actor_transfer import actor_loss
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest, project_inputs, support_hash
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import environment, evaluate
from tools.attribute_ranking_improvement import attribute_pair
from tools.audit_fixed_actor_critics import covering_schedule
from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import read, write, configure, now
from tools.run_operational_actor_transfer import save_canonical
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import market_for

ROOT=Path(__file__).resolve().parents[1]
POINTS=(0,1,8)
INTERVALS={'fit':['2020-01-01','2023-01-01'], 'monitor':['2023-01-01','2024-01-01']}


def run(a):
    started=time.monotonic();configure(281);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:
        raise ValueError('ALL_STATE_GPU')
    parent=read(a.parent/'SPEC.json');verify_files(a.parent,parent['files'])
    qualification_config=read(a.bundle.parent.parent/'CONFIG.json')
    qualification_id=qualification_identity(qualification_config)
    verify_done(a.bundle.parent.parent,qualification_id)
    qualified=read(a.bundle.parent/'RESULT.json')
    model,_=load_bundle(a.bundle,'cuda');source_sha=file_sha(a.bundle/'model.pt')
    flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    market=market_for(a.parent,parent)
    market.inputs=project_inputs(market.inputs,model.recipe);market.recipe=model.recipe
    if len(model.recipe['list']['columns'])!=69 or set(model.recipe['split']['periods'])!={'TRAIN','VALIDATION'}:
        raise ValueError('ALL_STATE_RECIPE')
    cfg=dict(schema='all_state_refinement.v1',parent_identity=parent['identity'],source_sha256=source_sha,
        qualification_identity=qualification_id,
        recipe_identity=model.recipe['identity'],support_sha256=support_hash(market.inputs),
        effective=model.effective,seeds=[19,43],critics=['retained','greedy_calibrated'],entropy=['native','zero'],
        points=list(POINTS),critic_updates=4096,primary_endpoint=8,batch=128,anchor_seed=281,
        per_year={'flat':8,'long':4,'short':4},fit_intervals=INTERVALS,model_promoted=False,test_read=False,
        sources={str(p.relative_to(ROOT)):file_sha(p) for folder in ('app','agent_plugins','env_plugins','tools')
                 for p in (ROOT/folder).rglob('*.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:
        raise ValueError('ALL_STATE_CONFIG')
    write(a.output/'CONFIG.json',cfg)
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    def progress(stage,**kw):
        value=dict(stage=stage,at=now(),seconds=time.monotonic()-started,identity=cfg['identity'],**kw)
        write(a.output/'PROGRESS.json',value);print(value,flush=True)
    def logits(m):return emit_actor_reference(m,market.inputs,None,flags)[0]
    risk=read(a.parent/'RISK.json')
    def make(interval):return environment(market,risk,parent['design']['horizon'],interval,model_gamma)
    model_gamma=model.effective['gamma']
    def measure(z,path):
        prepared=prepared_signals(z)
        return {p:evaluate(make(interval),z,prepared=prepared,save=path/p,diagnose=True)
                for p,interval in market.recipe['split']['periods'].items()}
    progress('REFERENCE')
    baseline=logits(model);base_metrics=measure(baseline,a.output/'reference')
    for p,m in base_metrics.items():account_equal(m,qualified['metrics'][p])
    write(a.output/'REFERENCE.json',base_metrics)
    signals=prepared_signals(baseline)[0]
    del model

    panel_root=a.output/'panel'
    if (panel_root/'DONE.json').exists():
        verify_done(panel_root,cfg['identity']);records=read(panel_root/'RECORDS.json')
    else:
        trace=[]
        for role,interval in INTERVALS.items():
            env=make(interval);env.reset()
            while not env.finished:
                row=int(env.rows[env.bar]);ts=int(market.inputs.row_ts_ns[row])
                window_start=int(market.inputs.slot_ts_ns[row,0])
                if role!='monitor' or window_start>=BOUNDARY:
                    trace.append(dict(row=row,timestamp=ts,window_start=window_start,role=role,
                        year=pd.Timestamp(ts,tz='UTC').year,position=0 if env.engine.pos is None else env.engine.pos.direction))
                env.step(int(signals[row]))
        chosen=choose_panel(trace);wanted={r['timestamp']:r for r in chosen}
        write(panel_root/'ANCHORS.json',chosen)
        write(panel_root/'COVERAGE.json',dict(available=dict(Counter(f"{r['year']}:{r['position']}" for r in trace)),
                                            selected=dict(Counter(f"{r['year']}:{r['position']}" for r in chosen))))
        records=[]
        for role,interval in INTERVALS.items():
            env=make(interval);env.reset();terminal=int(env.engine.v[env.engine.last])+3600*10**9
            while not env.finished:
                row=int(env.rows[env.bar]);ts=int(market.inputs.row_ts_ns[row])
                if ts in wanted:
                    cell=panel_root/'cells'/f'{ts}.json'
                    if cell.exists():
                        rec=read(cell)
                        if rec['identity']!=cfg['identity'] or rec['state']!=state_view(env):
                            raise ValueError('ALL_STATE_CELL_REPLAY')
                    else:
                        rec=dict(wanted[ts],identity=cfg['identity'],terminal=terminal,state=state_view(env),
                                 outcomes=[continuation(env,signals,action) for action in (0,1,2)])
                        write(cell,rec)
                    records.append(rec)
                    progress('PANEL',anchors=len(records),total=len(chosen))
                env.step(int(signals[row]))
        validate_panel(records);write(panel_root/'RECORDS.json',records);seal(panel_root,cfg['identity'])
    validate_panel(records)
    rows=np.array([r['row'] for r in records])
    if (not market.inputs.valid[rows].all() or not (market.inputs.period[rows]=='TRAIN').all()
            or not np.array_equal(market.inputs.row_ts_ns[rows],[r['timestamp'] for r in records])):
        raise ValueError('ALL_STATE_ALIGNMENT')
    x=torch.tensor(market.inputs.windows[rows],device='cuda')
    targets=np.array([[v['reward_return'] for v in r['outcomes']] for r in records])
    cash=np.array([[v['net_usd'] for v in r['outcomes']] for r in records])
    y=torch.tensor(targets,dtype=torch.float32,device='cuda')
    masks={role:np.array([i for i,r in enumerate(records) if r['role']==role]) for role in INTERVALS}
    for role in INTERVALS:
        for pos in (-1,0,1):
            ix=np.array([i for i,r in enumerate(records) if r['role']==role and r['position']==pos])
            if not len(ix):raise ValueError('ALL_STATE_MISSING_POSITION')
            masks[f'{role}:position{pos}']=ix
    fit=masks['fit']
    def diagnostic(m,z):
        with torch.no_grad():q=m.q_surfaces(x).cpu().numpy()
        return {key:dict(critic=fit_metrics(q[:,ix],targets[ix]),
            actor_reward=policy_effect(baseline[rows[ix]],z[rows[ix]],targets[ix]),
            actor_cash=policy_effect(baseline[rows[ix]],z[rows[ix]],cash[ix])) for key,ix in masks.items()}

    results=[]
    for seed in cfg['seeds']:
        fitted=a.output/f'critic_{seed}'
        if (fitted/'DONE.json').exists():verify_done(fitted,cfg['identity'])
        else:
            model,_=load_bundle(a.bundle,'cuda')
            init=begin_stage(model,'all_state_greedy',seed=seed)
            set_scope(model,'heads');fixed=frozen_state(model)
            buffers={n:v.detach().clone() for n,v in model.loss.named_buffers()}
            reset_learning_rng(seed);schedule=covering_schedule(len(fit),cfg['critic_updates'],128,seed)
            curve=[]
            for step in range(cfg['critic_updates']+1):
                if step:
                    ix=fit[schedule[step-1]];model.optimizer.zero_grad(set_to_none=True)
                    loss=target_loss(probe_q(model,x[ix]),y[ix]);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.loss.parameters(),10.,error_if_nonfinite=True)
                    model.optimizer.step()
                if step in (0,64,256,1024,2048,4096):
                    verify_frozen_state(model,fixed)
                    if any(not torch.equal(v,dict(model.loss.named_buffers())[n]) for n,v in buffers.items()):
                        raise ValueError('ALL_STATE_CRITIC_BUFFER')
                    with torch.no_grad():q=model.q_surfaces(x).cpu().numpy()
                    curve.append(dict(updates=step,metrics={k:fit_metrics(q[:,ix],targets[ix]) for k,ix in masks.items()}))
                    write(fitted/'CURVE.json',curve);progress('CRITIC_FIT',seed=seed,updates=step)
            if not np.array_equal(logits(model)[market.inputs.valid],baseline[market.inputs.valid]):
                raise ValueError('ALL_STATE_CRITIC_CHANGED_ACTOR')
            save_canonical(model,fitted/'MODEL.pt',flags)
            write(fitted/'RESULT.json',dict(initialization=init,curve=curve,schedule_sha256=digest(schedule.tolist()),
                                          actor_unchanged=True,frozen_exact=True))
            seal(fitted,cfg['identity']);del model
        schedule=covering_schedule(len(fit),8,128,seed)
        for critic in cfg['critics']:
            for entropy in cfg['entropy']:
                key=f'{critic}_{entropy}_{seed}';out=a.output/key
                if (out/'DONE.json').exists():
                    verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
                model,_=load_bundle(a.bundle,'cuda')
                if critic=='greedy_calibrated':model.load(fitted/'MODEL.pt')
                if not np.array_equal(logits(model)[market.inputs.valid],baseline[market.inputs.valid]):
                    raise ValueError('ALL_STATE_INITIAL_ACTOR')
                set_arm(model,'actor_only');clear_optimizer_memory(model.optimizer);reset_learning_rng(seed)
                fixed=frozen_state(model);params={n:p for n,p in model.loss.named_parameters() if p.requires_grad}
                if not params or any(not n.startswith('actor_') or '.head.' not in n for n in params):
                    raise ValueError('ALL_STATE_ACTOR_SCOPE')
                initial={n:p.detach().clone() for n,p in params.items()}
                buffers={n:v.detach().clone() for n,v in model.loss.named_buffers()}
                with torch.no_grad():q=model.q_surfaces(x).min(0).values.detach()
                alpha=model.loss.log_alpha.detach().exp()*(1 if entropy=='native' else 0)
                points=[dict(updates=0,metrics=base_metrics,diagnostic=diagnostic(model,baseline))]
                write(out/'EFFECTIVE.json',dict(alpha=float(alpha),critic=critic,entropy=entropy,
                    schedule_sha256=digest(schedule.tolist()),fit_rows=len(fit),trainable=list(params),
                    lr=model.effective['learning_rate'],weight_decay=model.effective['weight_decay']))
                for step in range(1,9):
                    ix=fit[schedule[step-1]];model.optimizer.zero_grad(set_to_none=True)
                    loss=actor_loss(model.surfaces(x[ix]),q[ix],alpha);loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(params.values()),10.,error_if_nonfinite=True);model.optimizer.step()
                    if step in POINTS:
                        verify_frozen_state(model,fixed)
                        if any(not torch.equal(v,dict(model.loss.named_buffers())[n]) for n,v in buffers.items()):
                            raise ValueError('ALL_STATE_ACTOR_BUFFER')
                        if all(torch.equal(v,params[n]) for n,v in initial.items()):raise ValueError('ALL_STATE_NO_UPDATE')
                        z=logits(model);path=out/f'ACTOR_{step:04d}.pt';save_canonical(model,path,flags)
                        cold,_=load_bundle(a.bundle,'cuda');cold.load(path);cold_z=logits(cold)
                        if not np.array_equal(cold_z[market.inputs.valid],z[market.inputs.valid]):
                            raise ValueError('ALL_STATE_COLD_LOGITS')
                        metrics=measure(z,out/f'actor_{step:04d}')
                        cold_metrics=measure(cold_z,out/f'cold_{step:04d}')
                        for period in metrics:account_equal(metrics[period],cold_metrics[period])
                        del cold
                        points.append(dict(updates=step,metrics=metrics,diagnostic=diagnostic(model,z),
                            eligible=all(v['orders_per_week']>2 for v in metrics.values()),
                            changed_actions={p:int((z[market.inputs.valid & (market.inputs.period==p)].argmax(-1)!=
                                baseline[market.inputs.valid & (market.inputs.period==p)].argmax(-1)).sum()) for p in metrics}))
                        write(out/'POINTS.json',points);progress('ACTOR',arm=key,updates=step)
                result=dict(arm=key,seed=seed,critic=critic,entropy=entropy,points=points,
                            frozen_exact=True,cold_exact=True,model_promoted=False)
                write(out/'RESULT.json',result);seal(out,cfg['identity']);results.append(result);del model
    write(a.output/'FIT_RESULT.json',dict(identity=cfg['identity'],reference=base_metrics,arms=results))

    # Fixed endpoint attribution is diagnostic only; it cannot choose a training arm.
    attribution=[]
    for r in results:
        if r['seed']!=19:continue
        out=a.output/'attribution'/r['arm']
        if (out/'DONE.json').exists():verify_done(out,cfg['identity']);attribution.append(read(out/'SUMMARY.json'));continue
        def advance(**kw):progress('ATTRIBUTION',arm=r['arm'],**kw)
        value=attribute_pair(lambda:make(market.recipe['split']['periods']['TRAIN']),market.inputs.row_ts_ns,
                             a.output/'reference/TRAIN',a.output/r['arm']/'actor_0008/TRAIN',advance)
        write(out/'EVENTS.json',value['events'])
        summary=dict(arm=r['arm'],**{k:v for k,v in value.items() if k!='events'})
        write(out/'SUMMARY.json',summary);seal(out,cfg['identity']);attribution.append(summary)
    if file_sha(a.bundle/'model.pt')!=source_sha:raise ValueError('ALL_STATE_SOURCE_CHANGED')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r in results:
        for ax,period in zip(axes,('TRAIN','VALIDATION')):
            ax.plot([p['updates'] for p in r['points']],[p['metrics'][period]['return_pct'] for p in r['points']],label=r['arm'])
            ax.set(title=period,xlabel='Additional actor updates',ylabel='Net return %')
    axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(a.output/'RETURNS.png');plt.close(fig)
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],reference=base_metrics,arms=results,
        attribution=attribution,anchors=len(records),fit_anchors=len(fit),model_promoted=False,test_read=False,
        source_unchanged=True,seconds=time.monotonic()-started,finished_at=now()))
    seal(a.output,cfg['identity']);progress('COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
