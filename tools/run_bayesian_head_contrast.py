"""Sequential, paired selected69 last-layer SAC Bayesian probe (no promotion)."""
import argparse
import fcntl
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from app.bayesian_actor_head import LastLayer, actor_objective, hidden, uncertainty
from app.critic_refinement import begin_stage, fit_anchor, sync_targets
from app.critic_moment_loss import moment_critic_objective
from app.frozen_actor_compaction import effective_actor, compare_logits
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.policy_return_calibration import policy_signals, discounted_returns
from app.rl_subset import digest, project_inputs, support_hash
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import emit, environment, evaluate
from tools.audit_fixed_actor_critics import collect_full_replay, covering_schedule
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_preserved_policy_refinement_probe import calibration_panel
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import market_for

ROOT = Path(__file__).resolve().parents[1]
DESIGN = dict(seeds=[7, 19, 43], arms=['deterministic', 'bayesian'], updates=256,
    points=[0, 8, 32, 128, 256], batch=128, lr=3e-5, weight_decay=.01,
    prior_std=.1, posterior_std=.001, train_draws=4, eval_draws=32,
    eval_seed=109, check_seed=211, kl_coefficient=1., critic_updates=4096,
    primary_endpoint=256, promotion=False, test_read=False,
    scope='bounded actor-only variational SAC with common frozen calibrated Q')


def seal(folder, identity):
    write(folder/'DONE.json', dict(identity=identity, at=now(), files={
        str(p.relative_to(folder)): file_sha(p) for p in folder.rglob('*')
        if p.is_file() and p != folder/'DONE.json' and p.name not in ('worker.lock', 'PROGRESS.json')}))


def verify_done(folder, identity):
    done = read(folder/'DONE.json')
    if done['identity'] != identity:
        raise ValueError('BAYES_RESULT_IDENTITY')
    verify_files(folder, done['files'])


def summary_report(output, config, results, reference):
    endpoint = []
    for result in results:
        last = result['points'][-1]
        endpoint.append(dict(seed=result['seed'], arm=result['arm'], metrics=last['metrics'],
            eligible=last['eligible'], uncertainty=last['uncertainty'],
            draw_action_disagreements=result['draw_action_disagreements']))
    paired = []
    if config['design']['arms']==['deterministic','bayesian']:
        for seed in config['design']['seeds']:
            a, b = [next(v for v in endpoint if v['seed']==seed and v['arm']==arm)
                    for arm in config['design']['arms']]
            paired.append(dict(seed=seed, both_eligible=a['eligible'] and b['eligible'],
                validation_return_delta_pp=b['metrics']['VALIDATION']['return_pct']-a['metrics']['VALIDATION']['return_pct'],
                train_return_delta_pp=b['metrics']['TRAIN']['return_pct']-a['metrics']['TRAIN']['return_pct']))
    write(output/'SUMMARY.json', dict(identity=config['identity'], reference=reference,
        endpoint=endpoint, paired=paired, model_promoted=False,
        interpretation='Three refinement seeds on one inherited representation; not independent pretrained-model replications. Uncertainty is not calibrated.'))
    lines = ['# Bayesian last-layer contrast', '',
        'Primary endpoint: 256 updates. No automatic promotion.', '',
        '| Seed | Arm | TRAIN return % | VAL return % | VAL Sharpe | VAL DD % | VAL trades | VAL win rate | Eligible |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in endpoint:
        t, v = r['metrics']['TRAIN'], r['metrics']['VALIDATION']
        lines.append(f"| {r['seed']} | {r['arm']} | {t['return_pct']:.6f} | {v['return_pct']:.6f} | {v['sharpe']} | {v['max_drawdown_pct']:.6f} | {v['trades']} | {v['win_rate']} | {r['eligible']} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for result in results:
        for ax, period in zip(axes, ('TRAIN', 'VALIDATION')):
            ax.plot([p['updates'] for p in result['points']],
                    [p['metrics'][period]['return_pct'] for p in result['points']],
                    marker='o', label=f"{result['arm']} s{result['seed']}")
            ax.set(title=period, xlabel='Actor updates (not epochs)', ylabel='Net return %')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(output/'LEARNING.png');plt.close(fig)


def run(args):
    start = time.monotonic()
    variance_weight={'smooth_l1':None,'mean_var_1':1.,'mean_var_05':.5}[args.critic_objective]
    design=dict(DESIGN,arms=args.arms,critic_objective=args.critic_objective,variance_weight=variance_weight)
    configure(7);torch.backends.cudnn.allow_tf32=False
    device=args.device
    if device=='cpu' and not args.preflight_only:
        raise ValueError('BAYES_CPU_PREFLIGHT_ONLY')
    if device=='cuda' and (not torch.cuda.is_available() or torch.cuda.get_device_name() != args.expected_device_name):
        raise ValueError('BAYES_GPU_IDENTITY')
    parent=read(args.parent/'SPEC.json');verify_files(args.parent,parent['files'])
    model, manifest=load_bundle(args.bundle,device)
    if len(model.recipe['list']['columns']) != 69:
        raise ValueError('BAYES_EXPECTED_69')
    market=market_for(args.parent,parent)
    market.inputs=project_inputs(market.inputs,model.recipe);market.recipe=model.recipe
    if set(market.recipe['split']['periods']) != {'TRAIN','VALIDATION'}:
        raise ValueError('BAYES_PERIODS')
    if support_hash(market.inputs)!=read(args.parent/'PREFLIGHT.json')['support_sha256']:
        raise ValueError('BAYES_SUPPORT')
    source_sha=file_sha(args.bundle/'model.pt')
    sources={str(p.relative_to(ROOT)):file_sha(p) for folder in ('app','agent_plugins','env_plugins','tools')
             for p in (ROOT/folder).rglob('*.py')}
    config=dict(schema='bayesian_actor_probe.v1', design=design, source_sha256=source_sha,
        manifest_sha256=file_sha(args.bundle/'MANIFEST.json'), parent_identity=parent['identity'],
        support_sha256=support_hash(market.inputs), recipe_identity=model.recipe['identity'],
        sources=sources, torch=torch.__version__, device=torch.cuda.get_device_name() if device=='cuda' else 'cpu')
    if args.control_diagnostics_from:
        old=read(args.control_diagnostics_from/'CONFIG.json')
        verify_done(args.control_diagnostics_from,old['identity'])
        if args.critic_objective!='smooth_l1' or any(config[k]!=old[k] for k in
                ('source_sha256','manifest_sha256','parent_identity','support_sha256','recipe_identity','torch','device')):
            raise ValueError('BAYES_DIAGNOSTIC_CONTROL_MISMATCH')
        if any(DESIGN[k]!=old['design'][k] for k in DESIGN):
            raise ValueError('BAYES_DIAGNOSTIC_CONTROL_DESIGN')
        config['control_diagnostics_identity']=old['identity']
    config['identity']=digest(config)
    if (args.output/'CONFIG.json').exists() and read(args.output/'CONFIG.json')!=config:
        raise ValueError('BAYES_CONFIG_CHANGED')
    write(args.output/'CONFIG.json',config)
    if (args.output/'DONE.json').exists():
        verify_done(args.output,config['identity']);return
    def progress(stage, **kw):
        write(args.output/'PROGRESS.json',dict(stage=stage, at=now(), seconds=time.monotonic()-start, **kw))
        print(stage, kw, flush=True)
    risk=read(args.parent/'RISK.json');d=parent['design'];gamma=model.effective['gamma']
    source_copy=args.output/'SOURCE_BUNDLE'
    if not source_copy.exists():shutil.copytree(args.bundle,source_copy)
    verify_files(source_copy,manifest['files'])
    periods=market.recipe['split']['periods']
    def accounts(logits, folder):
        prepared=prepared_signals(logits)
        return {p:evaluate(environment(market,risk,d['horizon'],interval,gamma), logits,
                prepared=prepared, save=folder/p, diagnose=True) for p,interval in periods.items()}
    progress('REFERENCE')
    logits,_=emit(model,market.inputs)
    reference=accounts(logits,args.output/'reference')
    expected=read(args.bundle.parent/'RESULT.json')['metrics']
    for p in periods:account_equal(reference[p],expected[p])
    write(args.output/'REFERENCE.json',reference)
    actor=effective_actor(model)
    actor_before={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    ix=np.flatnonzero(market.inputs.valid)
    with torch.no_grad():
        embeddings=torch.cat([hidden(actor,torch.tensor(market.inputs.windows[j],device=device))
                              for j in np.array_split(ix, max(1,(len(ix)+255)//256))])
        parity=compare_logits(logits[ix],actor.head[-1](embeddings).cpu().numpy())
    if not parity['qualified']:raise ValueError('BAYES_HIDDEN_PARITY')
    write(args.output/'PREFLIGHT.json',dict(reference_accounts_exact=True,hidden_parity=parity,
        columns=69,valid_windows=len(ix),source_sha256=source_sha,design=design))
    if args.preflight_only:
        progress('PREFLIGHT_PASS');return
    # Frozen policy generates the only critic-fitting trajectory, entirely in TRAIN.
    progress('TRAIN_REPLAY')
    prepared,prob=policy_signals(logits,7)
    env=environment(market,risk,d['horizon'],periods['TRAIN'],gamma)
    evaluate(env,logits,prepared=prepared,save=args.output/'train_replay')
    trace=read(args.output/'train_replay/ACTIONS.json')
    replay,receipt=collect_full_replay(environment(market,risk,d['horizon'],periods['TRAIN'],gamma),trace)
    rows=np.searchsorted(market.inputs.row_ts_ns,[r['timestamp'] for r in trace])
    if not np.array_equal(market.inputs.row_ts_ns[rows],[r['timestamp'] for r in trace]):
        raise ValueError('BAYES_REPLAY_TIMESTAMP')
    chosen=np.array([r['signal'] for r in trace]);alpha=float(model.loss.log_alpha.detach().exp())
    returns=discounted_returns([r['reward'] for r in trace],[r['duration'] for r in trace],
        -np.log(prob[rows,chosen].clip(1e-30)),gamma,alpha)
    replay['realized_soft_return']=torch.tensor(returns,dtype=torch.float32).reshape(-1,1)
    write(args.output/'REPLAY.json',dict(**receipt,period='TRAIN',alpha=alpha,
        trace_sha256=file_sha(args.output/'train_replay/ACTIONS.json'),source_sha256=source_sha))
    common=args.output/'common';common.mkdir(exist_ok=True)
    if args.control_diagnostics_from:
        progress('REUSE_SEALED_CONTROL_NO_UPDATES')
        source=args.control_diagnostics_from/'common'
        for name in ('CALIBRATED.pt','CALIBRATION.json'):
            shutil.copy2(source/name,common/name)
        model.load(common/'CALIBRATED.pt')
    elif (common/'DONE.json').exists():
        verify_done(common,config['identity']);model.load(common/'CALIBRATED.pt')
    else:
        initialization=begin_stage(model,config['identity'],seed=7)
        schedule=covering_schedule(len(replay),design['critic_updates'],design['batch'],7)
        with moment_critic_objective(model,variance_weight):
            points=fit_anchor(model,replay,schedule,lambda:calibration_panel(model,replay),
                lambda r:(write(common/'PROGRESS.json',r),progress('CRITIC_CALIBRATION',updates=r['updates'])))
        sync_targets(model);model.save(common/'CALIBRATED.pt')
        write(common/'CALIBRATION.json',dict(initialization=initialization,points=points))
        seal(common,config['identity'])
    unchanged=effective_actor(model).state_dict()
    if any(not torch.equal(v,unchanged[k].cpu()) for k,v in actor_before.items()):
        raise ValueError('BAYES_CRITIC_CHANGED_ACTOR')
    with torch.no_grad():
        q=torch.cat([model.q_surfaces(replay['features'][a:a+256].to('cuda')).min(0).values
                     for a in range(0,len(replay),256)])
    train_hidden=embeddings[torch.tensor(np.searchsorted(ix,rows),device='cuda')]
    if not torch.isfinite(q).all():raise ValueError('BAYES_Q_FINITE')
    write(common/'Q_TABLE.json',dict(shape=list(q.shape),note='Frozen Q common to both arms; no alternate-action ranking guarantee'))
    progress('CRITIC_VALIDATION_DIAGNOSTIC')
    # Diagnostic only: these transitions never enter an optimizer or fit schedule.
    validation_env=environment(market,risk,d['horizon'],periods['VALIDATION'],gamma)
    evaluate(validation_env,logits,prepared=prepared,save=args.output/'validation_replay')
    validation_trace=read(args.output/'validation_replay/ACTIONS.json')
    validation_replay,validation_receipt=collect_full_replay(
        environment(market,risk,d['horizon'],periods['VALIDATION'],gamma),validation_trace)
    vr=np.searchsorted(market.inputs.row_ts_ns,[r['timestamp'] for r in validation_trace])
    va=np.array([r['signal'] for r in validation_trace])
    validation_returns=discounted_returns([r['reward'] for r in validation_trace],
        [r['duration'] for r in validation_trace],-np.log(prob[vr,va].clip(1e-30)),gamma,alpha)
    validation_replay['realized_soft_return']=torch.tensor(validation_returns,dtype=torch.float32).reshape(-1,1)
    write(common/'ERRORS.json',dict(TRAIN=calibration_panel(model,replay),
        VALIDATION=calibration_panel(model,validation_replay),validation_receipt=validation_receipt,
        validation_fitting=False,variance_weight=variance_weight))
    del validation_replay
    # Reseal after the Q provenance was added, before any arm can consume it.
    seal(common,config['identity'])
    if args.control_diagnostics_from:
        write(args.output/'SUMMARY.json',dict(identity=config['identity'],optimizer_updates=0,
            source_identity=config['control_diagnostics_identity'],errors=read(common/'ERRORS.json'),
            model_promoted=False))
        seal(args.output,config['identity']);progress('DIAGNOSTICS_COMPLETE',optimizer_updates=0)
        return
    del replay, model
    period_indices={p:np.flatnonzero((market.inputs.row_ts_ns[ix]>=np.datetime64(interval[0],'ns').astype('int64')) &
        (market.inputs.row_ts_ns[ix]<np.datetime64(interval[1],'ns').astype('int64'))) for p,interval in periods.items()}
    results=[]
    for seed in design['seeds']:
        schedule=covering_schedule(len(train_hidden),design['updates'],design['batch'],seed)
        for arm in design['arms']:
            out=args.output/f'{arm}_{seed}';out.mkdir(exist_ok=True)
            if (out/'DONE.json').exists():
                verify_done(out,config['identity']);results.append(read(out/'RESULT.json'));continue
            head=LastLayer(actor.head[-1],arm=='bayesian',design['prior_std'],design['posterior_std']).cuda()
            opt=torch.optim.AdamW(head.parameters(),lr=design['lr'],weight_decay=design['weight_decay'])
            checkpoints=[];history=[]
            def infer(eval_seed, mean_only=False):
                chunks=[]
                noise=head.draws(design['eval_draws'],eval_seed) if head.bayesian else None
                with torch.no_grad():
                    for x in embeddings.split(512):
                        z=head.mean_logits(x)[None] if mean_only else head.logits(x,noise)
                        chunks.append({k:v.cpu() for k,v in uncertainty(z).items()})
                u={k:torch.cat([c[k] for c in chunks]).numpy() for k in chunks[0]}
                raw=np.full_like(logits,np.nan);raw[ix]=np.log(u['probability'].clip(1e-30))
                return raw,u
            for step in range(design['updates']+1):
                if step:
                    batch=torch.tensor(schedule[step-1],device='cuda')
                    noise=head.draws(design['train_draws'],seed*10000+step) if head.bayesian else None
                    sac=actor_objective(head.logits(train_hidden[batch],noise),q[batch],alpha)
                    kl=head.kl();total=sac+design['kl_coefficient']*kl/len(train_hidden)
                    if not torch.isfinite(total):raise ValueError('BAYES_LOSS_FINITE')
                    opt.zero_grad(set_to_none=True);total.backward()
                    grad=torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True);opt.step()
                    history.append(dict(updates=step,sac=float(sac.detach()),kl=float(kl.detach()),
                                        total=float(total.detach()),gradient_norm=float(grad)))
                if step in design['points']:
                    progress('ACCOUNTS',seed=seed,arm=arm,updates=step)
                    raw,u=infer(design['eval_seed'])
                    folder=out/f'update_{step:04d}'
                    metrics=accounts(raw,folder)
                    if arm=='deterministic' and step==0:
                        for p in periods:account_equal(metrics[p],reference[p])
                    np.savez_compressed(folder/'UNCERTAINTY.npz',row_indices=ix,**u)
                    torch.save(dict(head=head.state_dict(),optimizer=opt.state_dict(),updates=step,
                        source_sha256=source_sha,design=design,bayesian=head.bayesian),folder/'HEAD.pt')
                    restored=LastLayer(actor.head[-1],head.bayesian,design['prior_std'],design['posterior_std']).cuda()
                    restored.load_state_dict(torch.load(folder/'HEAD.pt',weights_only=False)['head'])
                    with torch.no_grad():
                        noise=head.draws(32,design['eval_seed']) if head.bayesian else None
                        if not torch.equal(head.logits(embeddings[:256],noise),restored.logits(embeddings[:256],noise)):
                            raise ValueError('BAYES_RELOAD')
                    us={p:{k:float(v[indices].mean()) for k,v in u.items() if k!='probability'}
                        for p,indices in period_indices.items()}
                    checkpoints.append(dict(updates=step,metrics=metrics,uncertainty=us,reload_exact=True,
                        eligible=all(m['orders_per_week']>2 for m in metrics.values())))
                    write(out/'POINTS.json',checkpoints);write(out/'LOSS.json',history)
            alternate,_=infer(design['check_seed'])
            disagreement=int(np.count_nonzero(prepared_signals(raw)[0][ix]!=prepared_signals(alternate)[0][ix]))
            mean_raw,_=infer(design['eval_seed'],True)
            mean_metrics=accounts(mean_raw,out/'endpoint_mean_weights')
            result=dict(seed=seed,arm=arm,points=checkpoints,mean_weight_endpoint=mean_metrics,
                draw_action_disagreements=disagreement,schedule_sha256=digest(schedule.tolist()),
                model_promoted=False,source_sha256=source_sha)
            write(out/'RESULT.json',result);seal(out,config['identity']);results.append(result)
    if file_sha(args.bundle/'model.pt')!=source_sha or any(
            not torch.equal(v,actor.state_dict()[k].cpu()) for k,v in actor_before.items()):
        raise ValueError('BAYES_FROZEN_SOURCE_CHANGED')
    summary_report(args.output,config,results,reference)
    seal(args.output,config['identity']);progress('COMPLETE',arms=len(results))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-device-name',required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--critic-objective',choices=['smooth_l1','mean_var_1','mean_var_05'],default='smooth_l1')
    parser.add_argument('--arms',nargs='+',choices=['deterministic','bayesian'],default=['deterministic','bayesian'])
    parser.add_argument('--control-diagnostics-from',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:
            write(args.output/'PROGRESS.json',dict(stage='FAILED',at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
