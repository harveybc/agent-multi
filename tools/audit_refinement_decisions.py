"""Frozen native attribution of selected69 Bayesian/moment-loss endpoints."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.bayesian_actor_head import LastLayer, hidden, uncertainty
from app.frozen_actor_compaction import effective_actor
from app.frozen_recipe_inputs import file_sha
from app.policy_return_calibration import policy_signals
from app.refinement_decision_audit import boundary_terms, classify_event, mechanism_summary
from app.rl_subset import digest, project_inputs, support_hash
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import emit, environment
from tools.attribute_anchored_actions import check_trace
from tools.attribute_ranking_improvement import attribute_pair
from tools.report_critic_moments import authenticated
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_structural_sac_refit import market_for

ROOT=Path(__file__).resolve().parents[1]
OBJECTIVES=('smooth_l1','mean_var_1','mean_var_05')
SEEDS=(7,19,43)


def audit(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if args.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.get_device_name()!=args.expected_device_name):
        raise ValueError('DECISION_AUDIT_GPU')
    parent=read(args.parent/'SPEC.json');verify_files(args.parent,parent['files'])
    control=authenticated(args.control)
    source=args.control if args.objective=='smooth_l1' else args.moments/args.objective
    candidate=authenticated(source)
    report=read(args.moments/'report/DONE.json');verify_files(args.moments/'report',report['files'])
    if report['identities'][args.objective]!=candidate['identity']:
        raise ValueError('DECISION_AUDIT_SOURCE')
    model,manifest=load_bundle(args.bundle,args.device)
    if file_sha(args.bundle/'model.pt')!=control['source_sha256']:
        raise ValueError('DECISION_AUDIT_MODEL')
    market=market_for(args.parent,parent)
    market.inputs=project_inputs(market.inputs,model.recipe);market.recipe=model.recipe
    if (set(market.recipe['split']['periods'])!={'TRAIN','VALIDATION'} or
            support_hash(market.inputs)!=control['support_sha256']):
        raise ValueError('DECISION_AUDIT_SUPPORT')
    sources={n:file_sha(ROOT/n) for n in ('tools/audit_refinement_decisions.py',
        'app/refinement_decision_audit.py','tools/attribute_ranking_improvement.py',
        'app/decision_attribution.py','app/anchored_action_attribution.py',
        'app/bayesian_actor_head.py','app/three_signal_learning.py')}
    cfg=dict(schema='refinement_decision_audit.v1',source=candidate['identity'],
        control=control['identity'],objective=args.objective,endpoint=256,seeds=list(SEEDS),
        periods=['VALIDATION','TRAIN'],continuation_seeds=list(SEEDS),
        source_sha256=control['source_sha256'],support_sha256=control['support_sha256'],
        recipe_identity=model.recipe['identity'],sources=sources,device=args.device,
        device_name=torch.cuda.get_device_name() if args.device=='cuda' else 'cpu',optimizer_updates=0)
    cfg['identity']=digest(cfg)
    if (args.output/'CONFIG.json').exists() and read(args.output/'CONFIG.json')!=cfg:
        raise ValueError('DECISION_AUDIT_CONFIG_CHANGED')
    write(args.output/'CONFIG.json',cfg)
    if (args.output/'DONE.json').exists():verify_done(args.output,cfg['identity']);return
    def progress(stage,**kw):
        write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
        if stage!='COUNTERFACTUALS' or kw.get('events',0)%10==0:print(stage,kw,flush=True)
    progress('FROZEN_SURFACES')
    base,_=emit(model,market.inputs)
    actor=effective_actor(model);ix=np.flatnonzero(market.inputs.valid)
    with torch.no_grad():
        embeddings=torch.cat([hidden(actor,torch.tensor(market.inputs.windows[j],device=args.device))
                              for j in np.array_split(ix,max(1,(len(ix)+255)//256))])
    pw=actor.head[-1].weight.cpu().numpy();pb=actor.head[-1].bias.cpu().numpy()
    model.load(source/'common/CALIBRATED.pt')
    _,twins=emit(model,market.inputs,twins=True)
    alpha=float(model.loss.log_alpha.detach().exp())
    stochastic={s:policy_signals(base,s) for s in SEEDS}
    soft={s:(t[0][0],t[1]) for s,t in stochastic.items()}
    calibration_positions={}
    for period,folder in [('TRAIN',args.control/'train_replay'),
                          ('VALIDATION',args.moments/'control_diagnostics/validation_replay')]:
        trace=read(folder/'ACTIONS.json')
        rows=np.searchsorted(market.inputs.row_ts_ns,[r['timestamp'] for r in trace])
        if not np.array_equal(soft[7][0][rows],[r['signal'] for r in trace]):
            raise ValueError('DECISION_AUDIT_CALIBRATION_POLICY')
        calibration_positions[period]={r['timestamp']:r['position'] for r in trace}
        check_trace(market.inputs,base,read(args.control/'reference'/period/'ACTIONS.json'))
    del model
    candidates={};heads={};checks={}
    for seed in SEEDS:
        folder=source/f'bayesian_{seed}/update_0256'
        saved=torch.load(folder/'HEAD.pt',map_location=args.device,weights_only=False)
        if saved['source_sha256']!=control['source_sha256'] or saved['updates']!=256 or not saved['bayesian']:
            raise ValueError('DECISION_AUDIT_HEAD_IDENTITY')
        head=LastLayer(actor.head[-1],True).to(args.device);head.load_state_dict(saved['head'])
        with np.load(folder/'UNCERTAINTY.npz') as z:
            if not np.array_equal(z['row_indices'],ix):raise ValueError('DECISION_AUDIT_ARRAY_SUPPORT')
            probs=z['probability'].copy()
        with torch.no_grad():
            noise=head.draws(32,109)
            reconstructed=torch.cat([uncertainty(head.logits(x,noise))['probability']
                                      for x in embeddings.split(512)]).cpu().numpy()
        if not np.allclose(probs,reconstructed,atol=1e-6,rtol=1e-5):
            raise ValueError('DECISION_AUDIT_HEAD_RELOAD')
        raw=np.full_like(base,np.nan);raw[ix]=np.log(probs.clip(1e-30))
        for period in cfg['periods']:check_trace(market.inputs,raw,read(folder/period/'ACTIONS.json'))
        heads[seed]=(head.weight.detach().cpu().numpy(),head.bias.detach().cpu().numpy())
        candidates[seed]=raw
        checks[str(seed)]=dict(max_probability_error=float(np.max(np.abs(probs-reconstructed))),
            checkpoint_sha256=file_sha(folder/'HEAD.pt'),archive_actions_exact=True)
    write(args.output/'PREFLIGHT.json',dict(source_weights_unchanged=True,head_reload=checks,
        source_sha256=file_sha(args.bundle/'model.pt'),valid_windows=len(ix),calibration_policy_exact=True))
    h=embeddings.cpu().numpy();del embeddings,actor,head
    if args.device=='cuda':torch.cuda.empty_cache()
    np.savez_compressed(args.output/'SURFACES.npz',indices=ix,parent=base[ix],twins=twins[:,ix],hidden=h,
                        **{f'candidate_{s}':candidates[s][ix] for s in SEEDS})
    risk=read(args.parent/'RISK.json');design=parent['design'];accounts=[];all_rows=[]
    for period in cfg['periods']:
        interval=market.recipe['split']['periods'][period]
        def make():return environment(market,risk,design['horizon'],interval,parent['effective']['gamma'])
        cache={}
        for seed in SEEDS:
            cell=args.output/f'{period}_{seed}'
            if (cell/'DONE.json').exists():
                verify_done(cell,cfg['identity']);r=read(cell/'SUMMARY.json')
                accounts.append(r);all_rows.extend(read(cell/'EVENTS.json'));continue
            progress('ACCOUNT_START',period=period,seed=seed,completed=len(accounts))
            result=attribute_pair(make,market.inputs.row_ts_ns,args.control/'reference'/period,
                source/f'bayesian_{seed}/update_0256'/period,
                lambda **kw:progress('COUNTERFACTUALS',period=period,seed=seed,completed=len(accounts),**kw),
                soft_policies=soft,alpha=alpha,isolated_cache=cache,cache_namespace=cfg['identity']+'/'+period)
            write(cell/'ATTRIBUTION.json',result);rows=[]
            for event in result['events']:
                row=int(np.searchsorted(market.inputs.row_ts_ns,event['timestamp']))
                hi=int(np.searchsorted(ix,row));old,new=event['old_signal'],event['new_signal']
                w,b=heads[seed]
                terms=boundary_terms(old,new,h[hi],pw,pb,w,b,base[row],candidates[seed][row])
                details=classify_event(event,base[row],candidates[seed][row],twins[:,row],alpha,
                    calibration_positions[period][event['timestamp']],terms)
                rows.append(dict(objective=args.objective,seed=seed,period=period,utc=event['utc'],
                    timestamp=event['timestamp'],**details))
            summary=dict(objective=args.objective,seed=seed,period=period,
                **mechanism_summary(rows),reconciliation=result['reconciliation'])
            if abs(summary['delta_usd']-result['delta_usd'])>1e-6:raise ValueError('DECISION_AUDIT_SUM')
            write(cell/'EVENTS.json',rows);write(cell/'SUMMARY.json',summary);seal(cell,cfg['identity'])
            accounts.append(summary);all_rows.extend(rows)
    if file_sha(args.bundle/'model.pt')!=control['source_sha256']:raise ValueError('DECISION_AUDIT_SOURCE_CHANGED')
    write(args.output/'EVENTS.json',all_rows)
    write(args.output/'SUMMARY.json',dict(identity=cfg['identity'],accounts=accounts,
        optimizer_updates=0,source_unchanged=True,model_promoted=False))
    seal(args.output,cfg['identity']);progress('COMPLETE',accounts=len(accounts),events=len(all_rows))


def aggregate(args):
    roots={name:args.root/name for name in OBJECTIVES};configs={};accounts=[];rows=[]
    for name,root in roots.items():
        config=read(root/'CONFIG.json');verify_done(root,config['identity'])
        if config['objective']!=name:raise ValueError('DECISION_AGGREGATE_OBJECTIVE')
        configs[name]=config
        summary=read(root/'SUMMARY.json')
        cells={(r['seed'],r['period']) for r in summary['accounts']}
        if cells!={(s,p) for s in SEEDS for p in ('TRAIN','VALIDATION')} or len(summary['accounts'])!=6:
            raise ValueError('DECISION_AGGREGATE_INCOMPLETE')
        accounts.extend(summary['accounts']);rows.extend(read(root/'EVENTS.json'))
    first=configs[OBJECTIVES[0]]
    for cfg in configs.values():
        for k in ('control','source_sha256','support_sha256','recipe_identity','sources','seeds','continuation_seeds','endpoint'):
            if cfg[k]!=first[k]:raise ValueError('DECISION_AGGREGATE_PAIR:'+k)
    by_period={p:mechanism_summary([r for r in rows if r['period']==p]) for p in ('TRAIN','VALIDATION')}
    train=by_period['TRAIN'];counts=train['harmful_counts'];losses=train['harmful_loss_by_flag']
    if not train['harmful']:
        recommendation='NO_HARMFUL_TRAIN_CHANGE: retain source and review endpoint validation separately; no automatic new fit'
    elif (counts['q_soft_all_draws_conflict']>=train['harmful']/2 and
          losses['q_soft_all_draws_conflict']>=train['harmful_loss_usd']/2):
        recommendation='PAIRED_ACTION_RANKING_DIAGNOSTIC: reproduce TRAIN opposite-action soft targets and state-alias strata before another actor update'
    elif counts['actor_change_against_q']>=train['harmful']/2:
        recommendation='ENTROPY_EXECUTION_ALIGNMENT: inspect stochastic SAC objective versus greedy action boundaries before another generic LR/KL sweep'
    elif counts['all_soft_draws_favor_change']>=train['harmful']/2:
        recommendation='REWARD_ACCOUNT_ALIGNMENT: quantify discounted soft reward versus account cash tradeoffs on TRAIN before adjusting losses'
    else:
        recommendation='MIXED_MECHANISMS: inspect matched TRAIN harmful events and alias strata; no justified single automatic training correction'
    output=args.root/'report';output.mkdir(exist_ok=True)
    payload=dict(identities={k:v['identity'] for k,v in configs.items()},accounts=accounts,
        by_period=by_period,recommendation=recommendation,recommendation_scope='TRAIN only; diagnostic, not a causal proof',
        unique_changed_events=len({(r['period'],r['timestamp'],r['old_signal'],r['new_signal']) for r in rows}),
        event_occurrences=len(rows),optimizer_updates=0,model_promoted=False)
    write(output/'SUMMARY.json',payload);write(output/'EVENTS.json',rows)
    lines=['# Selected69 refinement decision audit','','No training or policy promotion. Contributions telescope chronologically.',
        'Three stochastic continuations are a diagnostic, not an exact expectation.','',
        '| Objective | Seed | Period | Delta USD | Signals | Helpful | Harmful | Q/soft conflicts among harmful | Against Q among harmful |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in accounts:
        lines.append(f"| {r['objective']} | {r['seed']} | {r['period']} | {r['delta_usd']:.6f} | {r['events']} | {r['helpful']} | {r['harmful']} | {r['harmful_counts']['q_soft_all_draws_conflict']} | {r['harmful_counts']['actor_change_against_q']} |")
    lines+=['','## Next diagnostic',recommendation,'',
        'Counts across seeds overlap; they are not independent samples. Calibration-position mismatch is an association, not proof of causation.',
        'Weight/bias/posterior boundary terms and all neutral events are retained in EVENTS.json.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    seal(output,digest(payload))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run')
    for n in ('parent','bundle','control','moments','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--objective',choices=OBJECTIVES,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--expected-device-name')
    p=sub.add_parser('aggregate');p.add_argument('--root',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='aggregate':aggregate(a);return
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:audit(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
