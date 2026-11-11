"""Sequential selected69 actor-head transfer, sealed critics and native accounts."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.actor_learning_diagnostics import clear_optimizer_memory
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.operational_actor_transfer import actor_loss, fit_indices, check_lineage
from app.operational_opening_critic import emit_actor_reference
from app.rl_subset import digest, project_inputs, support_hash
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import environment, evaluate
from tools.audit_fixed_actor_critics import covering_schedule
from tools.diagnose_warmstart_drift import set_arm, frozen_state, verify_frozen_state
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import read, write, configure, now
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import market_for

ROOT=Path(__file__).resolve().parents[1]
POINTS=(0,1,8,32)


def save_canonical(model, path, flags):
    params=dict(model.loss.named_parameters())
    before={n:p.requires_grad for n,p in params.items()}
    try:
        for n,p in params.items():p.requires_grad_(flags[n])
        model.save(path)
    finally:
        for n,p in params.items():p.requires_grad_(before[n])


def run(a):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=a.expected_device_name:
        raise ValueError('ACTOR_TRANSFER_GPU')
    parent=read(a.parent/'SPEC.json');verify_files(a.parent,parent['files'])
    control=read(a.control/'CONFIG.json');verify_done(a.control,control['identity'])
    corrected=read(a.corrected/'CONFIG.json');verify_done(a.corrected,corrected['identity'])
    model,_=load_bundle(a.bundle,'cuda');source_sha=file_sha(a.bundle/'model.pt')
    flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    market=market_for(a.parent,parent)
    market.inputs=project_inputs(market.inputs,model.recipe);market.recipe=model.recipe
    if len(model.recipe['list']['columns'])!=69 or set(model.recipe['split']['periods'])!={'TRAIN','VALIDATION'}:
        raise ValueError('ACTOR_TRANSFER_RECIPE')
    check_lineage(source_sha,support_hash(market.inputs),control,corrected)
    records=read(a.corrected/'panel/RECORDS.json');take=fit_indices(records)
    rows=np.array([records[i]['row'] for i in take])
    if (not market.inputs.valid[rows].all() or not (market.inputs.period[rows]=='TRAIN').all()
            or not np.array_equal(market.inputs.row_ts_ns[rows],[records[i]['timestamp'] for i in take])):
        raise ValueError('ACTOR_TRANSFER_FIT_ALIGNMENT')
    cfg=dict(schema='operational_actor_transfer.v1',source_sha256=source_sha,
        parent_identity=parent['identity'],control_identity=control['identity'],corrected_identity=corrected['identity'],
        support_sha256=support_hash(market.inputs),recipe_identity=model.recipe['identity'],
        seeds=[19,43],arms=['control','corrected'],points=list(POINTS),batch=128,
        effective=model.effective,fit_timestamps=[records[i]['timestamp'] for i in take],
        fit='2020-2022 flat both-open-allowed anchors',actor_scope='head only',
        objective='categorical SAC with detached frozen minimum twin Q and alpha',
        primary_endpoint=32,model_promoted=False,test_read=False,
        sources={str(p.relative_to(ROOT)):file_sha(p) for folder in ('app','agent_plugins','env_plugins','tools')
                 for p in (ROOT/folder).rglob('*.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:
        raise ValueError('ACTOR_TRANSFER_CONFIG')
    write(a.output/'CONFIG.json',cfg)
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    def progress(stage,**kw):
        value=dict(stage=stage,at=now(),seconds=time.monotonic()-started,identity=cfg['identity'],**kw)
        write(a.output/'PROGRESS.json',value);print(value,flush=True)
    def logits_for(m):return emit_actor_reference(m,market.inputs,None,flags)[0]
    risk=read(a.parent/'RISK.json');design=parent['design']
    def measure(logits,folder):
        prepared=prepared_signals(logits)
        return {p:evaluate(environment(market,risk,design['horizon'],interval,model.effective['gamma']),
            logits,prepared=prepared,save=folder/p,diagnose=True)
            for p,interval in market.recipe['split']['periods'].items()}
    progress('REFERENCE_ACCOUNTS',fit_anchors=len(take))
    baseline=logits_for(model);reference=measure(baseline,a.output/'reference')
    for p,m in reference.items():account_equal(m,read(a.bundle.parent/'RESULT.json')['metrics'][p])
    write(a.output/'REFERENCE.json',reference)
    features=torch.tensor(market.inputs.windows[rows],device='cuda');del model
    results=[]
    for seed in cfg['seeds']:
        schedule=covering_schedule(len(rows),32,128,seed)
        for arm in cfg['arms']:
            out=a.output/f'{arm}_{seed}'
            if (out/'DONE.json').exists():
                verify_done(out,cfg['identity']);results.append(read(out/'RESULT.json'));continue
            checkpoint=(a.control/'common/CALIBRATED.pt' if arm=='control' else
                        a.corrected/f'critic_representation_absolute_{seed}/DIAGNOSTIC.pt')
            progress('LOAD_ARM',arm=arm,seed=seed)
            model,_=load_bundle(a.bundle,'cuda');model.load(checkpoint)
            if not np.array_equal(logits_for(model)[market.inputs.valid],baseline[market.inputs.valid]):
                raise ValueError('ACTOR_TRANSFER_INITIAL_POLICY')
            set_arm(model,'actor_only');clear_optimizer_memory(model.optimizer);reset_learning_rng(seed)
            fixed=frozen_state(model)
            trained={n:p for n,p in model.loss.named_parameters() if p.requires_grad}
            if not trained or any(not n.startswith('actor_') or '.head.' not in n for n in trained):
                raise ValueError('ACTOR_TRANSFER_SCOPE')
            initial={n:p.detach().clone() for n,p in trained.items()}
            frozen_buffers={n:v.detach().clone() for n,v in model.loss.named_buffers()}
            with torch.no_grad():q=model.q_surfaces(features).min(0).values.detach()
            alpha=model.loss.log_alpha.detach().exp()
            write(out/'EFFECTIVE.json',dict(checkpoint_sha256=file_sha(checkpoint),
                schedule_sha256=digest(schedule.tolist()),fit_rows=len(rows),trainable=list(trained),
                learning_rate=model.effective['learning_rate'],alpha=float(alpha),weight_decay=model.effective['weight_decay']))
            points=[dict(updates=0,metrics=reference,changed_actions={p:0 for p in reference},
                         eligible=all(m['orders_per_week']>2 for m in reference.values()))]
            losses=[]
            for step in range(1,33):
                batch=schedule[step-1];model.optimizer.zero_grad(set_to_none=True)
                loss=actor_loss(model.surfaces(features[batch]),q[batch],alpha)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(trained.values()),10.,error_if_nonfinite=True)
                model.optimizer.step();losses.append(dict(updates=step,loss=float(loss.detach()),grad=float(norm)))
                if step in POINTS:
                    verify_frozen_state(model,fixed)
                    if any(not torch.equal(v,dict(model.loss.named_buffers())[n]) for n,v in frozen_buffers.items()):
                        raise ValueError('ACTOR_TRANSFER_BUFFER_CHANGED')
                    if all(torch.equal(v,trained[n]) for n,v in initial.items()):
                        raise ValueError('ACTOR_TRANSFER_NO_UPDATE')
                    progress('ACCOUNTS',arm=arm,seed=seed,updates=step)
                    actual=logits_for(model)
                    path=out/f'ACTOR_{step:04d}.pt';save_canonical(model,path,flags)
                    if step==1:
                        reloaded,_=load_bundle(a.bundle,'cuda');reloaded.load(path)
                        if not np.array_equal(logits_for(reloaded)[market.inputs.valid],actual[market.inputs.valid]):
                            raise ValueError('ACTOR_TRANSFER_RELOAD_PARITY')
                        del reloaded
                    metrics=measure(actual,out/f'actor_{step:04d}')
                    changes={p:int((actual[market.inputs.valid & (market.inputs.period==p)].argmax(-1)!=
                                      baseline[market.inputs.valid & (market.inputs.period==p)].argmax(-1)).sum()) for p in metrics}
                    points.append(dict(updates=step,metrics=metrics,changed_actions=changes,
                                       eligible=all(m['orders_per_week']>2 for m in metrics.values())))
                    write(out/'POINTS.json',points);progress('POINT_COMPLETE',arm=arm,seed=seed,updates=step,
                        validation_return_pct=metrics['VALIDATION']['return_pct'])
            record=dict(identity=cfg['identity'],arm=arm,seed=seed,points=points,losses=losses,
                        frozen_exact=True,reload_exact=True,model_promoted=False)
            write(out/'RESULT.json',record);seal(out,cfg['identity']);results.append(record);del model
    if file_sha(a.bundle/'model.pt')!=source_sha:raise ValueError('ACTOR_TRANSFER_SOURCE_CHANGED')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r in results:
        for ax,p in zip(axes,('TRAIN','VALIDATION')):
            ax.plot([x['updates'] for x in r['points']],[x['metrics'][p]['return_pct'] for x in r['points']],
                    marker='.',label=f"{r['arm']} s{r['seed']}")
            ax.set(title=p,xlabel='Actor updates',ylabel='Net return %')
    axes[0].legend();fig.tight_layout();fig.savefig(a.output/'RETURNS.png');plt.close(fig)
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],reference=reference,arms=results,
                                    model_promoted=False,finished_at=now()))
    seal(a.output,cfg['identity']);progress('COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','control','corrected','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
