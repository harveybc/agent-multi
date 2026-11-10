"""Frozen-policy native counterfactual state compatibility; no learning updates."""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from app.critic_fit_probe import classify_anchor, probe_q
from app.decision_attribution import continuation, state_view
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import digest, support_hash
from app.state_compatibility import compatibility, history_banks
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, environment
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_action_ranking_contrast import ensure_config, seal
from tools.run_actor_preservation import collect
from tools.run_critic_fit_probe import train_logits
from tools.run_discrete_curve_probe import configure, now, read, write
from tools.run_structural_sac_refit import authenticate, market_for

ROOT=Path(__file__).resolve().parents[1]
LAGS=(1,4,12)


def repeated_input_check(model,features):
    with torch.no_grad():
        duplicate=features[:1].expand(3,-1,-1).contiguous()
        actor=model.surfaces(duplicate);q=probe_q(model,duplicate)
        if not torch.equal(actor,actor[:1].expand_as(actor)) or not torch.equal(q,q[:,:1].expand_as(q)):
            raise ValueError('COMPATIBILITY_MODEL_INPUT_INVARIANCE')


def origin_banks(env,anchor,signals,identity,witness):
    before=state_view(env);banks=[];suffixes=0;negative_error=None
    for lag,states,info in history_banks(env,LAGS):
        flat=states['flat'];row=int(flat.rows[flat.bar]);view=state_view(flat)
        feature=flat.last_observation['features']
        item=dict(lag=lag,row=row,timestamp=view['timestamp'],role=anchor['role'],
            feature_sha256=hashlib.sha256(np.ascontiguousarray(feature).tobytes()).hexdigest(),
            states=list(states),history=info,outcomes={},compatibility=None)
        if len(states)>=2:
            for name,state in states.items():
                values=[continuation(state,signals,a) for a in range(3)];suffixes+=3
                a=dict(anchor,row=row,timestamp=view['timestamp'],position=0 if name=='flat' else 1 if name=='long' else -1)
                classify_anchor(dict(anchor=a,results=dict(greedy=values)))
                item['outcomes'][name]=values
            y=[[v['reward_return'] for v in values] for values in item['outcomes'].values()]
            item['compatibility']=compatibility(y)
            if witness and negative_error is None:
                repeat=[continuation(flat,signals,a) for a in range(3)];suffixes+=3
                negative_error=max(abs(v[k]-w[k]) for v,w in zip(repeat,item['outcomes']['flat'])
                                   for k in ('reward_return','net_usd','soft_return'))
                if negative_error>1e-8:raise ValueError('COMPATIBILITY_NEGATIVE_CONTROL')
        banks.append(item)
    if state_view(env)!=before:raise ValueError('COMPATIBILITY_ORIGIN_CHANGED')
    data=dict(anchor=anchor,initial_state=before,banks=banks,suffixes=suffixes,
        terminal_excluded_lags=[v for v in LAGS if v not in {b['lag'] for b in banks}],
        negative_control_error=negative_error)
    return dict(identity=identity,data=data,sha256=digest(data))


def quarter(args,spec,market,signals,source,config,progress):
    name=source.name;root=args.output/'quarters'/name;root.mkdir(parents=True,exist_ok=True)
    if (root/'DONE.json').exists():
        if verify_collection(root)['identity']!=config['identity']:raise ValueError('COMPATIBILITY_QUARTER_IDENTITY')
        return read(root/'RESULT.json')
    started=time.monotonic();design=read(source/'DESIGN.json')
    anchors=[a for a in design['anchors'] if a['kind']=='flat_both_allowed']
    chosen={a['trace_index']:a for a in anchors}
    env=environment(market,read(args.source/'RISK.json'),spec['design']['horizon'],design['interval'],spec['effective']['gamma'])
    env.reset();trace=[];results=[];witness_done=False
    while not env.finished:
        row=int(env.rows[env.bar]);action=int(signals[row]);index=len(trace)
        trace.append(dict(row=row,signal=action,state=state_view(env)))
        if index in chosen:
            anchor=chosen[index]
            if (anchor['row']!=row or anchor['timestamp']!=state_view(env)['timestamp']
                    or env.engine.pos is not None or market.inputs.period[row]!='TRAIN'):
                raise ValueError('COMPATIBILITY_ORIGIN_REPLAY')
            path=root/'origins'/f'{index:05d}.json'
            if path.exists():
                record=read(path)
                if (record['identity']!=config['identity'] or record['data']['anchor']!=anchor
                        or record['sha256']!=digest(record['data'])):
                    raise ValueError('COMPATIBILITY_CACHED_ORIGIN')
            else:
                record=origin_banks(env,anchor,signals,config['identity'],not witness_done);write(path,record)
            witness_done|=record['data']['negative_control_error'] is not None
            results.append(record['data'])
            progress('NATIVE_HISTORIES',quarter=name,completed=len(results),total=len(anchors))
        env.step(action)
    if (digest(trace)!=design['reference_trace_sha256']
            or abs(env.engine.balance-env.engine.cfg.capital-design['reference_cash'])>1e-8
            or len(results)!=len(anchors)):
        raise ValueError('COMPATIBILITY_REFERENCE_ACCOUNT')
    result=dict(identity=config['identity'],origins=results,reference_exact=True,
        negative_control_executed=witness_done,interval=design['interval'])
    write(root/'RESULT.json',result);seal(root,config['identity'],started);return result


def summarize(banks):
    groups={}
    for role in ('fit','monitor'):
        for lag in LAGS:
            all_rows=[b for b in banks if b['role']==role and b['lag']==lag]
            rows=[b for b in all_rows if b['compatibility'] is not None]
            group=dict(role=role,lag=lag,attempted_banks=len(all_rows),comparable_banks=len(rows),
                state_sets=dict(Counter('|'.join(b['states']) for b in all_rows)))
            for key in ('raw_rmse_floor','centered_rmse_floor','gap_rmse_floor','best_common_regret','deterministic_minimax_regret'):
                values=[b['compatibility'][key] for b in rows]
                group[key+'_mean']=float(np.mean(values)) if values else None
            group['conflicting_optima']=sum(not b['compatibility']['common_optimal_actions'] for b in rows)
            group['strict_reversal_banks']=sum(bool(b['compatibility']['strict_reversal_pairs']) for b in rows)
            group['positive_mean_regret_banks']=sum(b['compatibility']['best_common_regret']>1e-8 for b in rows)
            groups[f'{role}:{lag}']=group
    return groups


def model_report(args,bundle,market,paths,banks,signals,reference_logits,progress):
    eligible=[b for b in banks if b['compatibility'] is not None]
    rows=sorted({b['row'] for b in eligible});lookup={r:i for i,r in enumerate(rows)}
    if not rows:return dict(status='NO_COMPARABLE_BANKS',models={})
    features=torch.tensor(market.inputs.windows[rows],device='cuda');results={};csv_rows=[]
    for name,path in paths.items():
        model,_=load_bundle(bundle,'cuda');model.load(path)
        train=market.inputs.valid & (market.inputs.period=='TRAIN')
        if not np.array_equal(train_logits(model,market.inputs.windows[train]),reference_logits):
            raise ValueError('COMPATIBILITY_ACTOR_DRIFT')
        repeated_input_check(model,features)
        with torch.no_grad():
            q=torch.cat([probe_q(model,features[i:i+128]) for i in range(0,len(features),128)],dim=1).cpu().numpy()
        output=[]
        for b in eligible:
            y=np.array([[v['reward_return'] for v in outcomes] for outcomes in b['outcomes'].values()])
            qaction=int(q[:,lookup[b['row']]].min(0).argmax())
            choices={'critic_min':qaction,'actor':int(signals[b['row']])}
            values={}
            for choice,action in choices.items():
                regret=y.max(1)-y[:,action];floor=b['compatibility']['best_common_regret']
                if regret.mean()<floor-1e-8:raise ValueError('COMPATIBILITY_REGRET_BOUND')
                values[choice]=dict(action=action,mean_regret=float(regret.mean()),
                    excess_regret=float(regret.mean()-floor),by_state=dict(zip(b['states'],regret.tolist())))
                csv_rows.append(dict(model=name,choice=choice,role=b['role'],lag=b['lag'],row=b['row'],
                    state_set='|'.join(b['states']),action=action,mean_regret=float(regret.mean()),
                    best_common_regret=floor,excess_regret=float(regret.mean()-floor)))
            output.append(dict(row=b['row'],lag=b['lag'],role=b['role'],states=b['states'],choices=values))
        results[name]=dict(checkpoint_sha256=file_sha(path),actor_exact=True,duplicate_input_exact=True,choices=output)
        progress('FROZEN_MODELS',completed=len(results),total=len(paths));del model
    with (args.output/'MODEL_CHOICES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    write(args.output/'MODEL_REPORT.json',results)
    return dict(status='COMPLETE',models={name:{k:v[k] for k in ('checkpoint_sha256','actor_exact','duplicate_input_exact')} for name,v in results.items()})


def report(root,results,models):
    origins=[o for q in results for o in q['origins']];banks=[b for o in origins for b in o['banks']]
    groups=summarize(banks)
    rows=[]
    for origin in origins:
        for b in origin['banks']:
            c=b['compatibility']
            rows.append(dict(origin=origin['anchor']['id'],row=b['row'],role=b['role'],lag=b['lag'],
                elapsed_hours=b['history']['elapsed_hours'],states='|'.join(b['states']),
                common_optimal_actions=None if c is None else '|'.join(map(str,c['common_optimal_actions'])),
                best_common_regret=None if c is None else c['best_common_regret'],
                gap_rmse_floor=None if c is None else c['gap_rmse_floor'],
                strict_reversals=None if c is None else len(c['strict_reversal_pairs'])))
    with (root/'COMPATIBILITY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for role in ('fit','monitor'):
        rr=[groups[f'{role}:{lag}'] for lag in LAGS]
        axes[0].plot(LAGS,[r['conflicting_optima']/r['comparable_banks'] if r['comparable_banks'] else np.nan for r in rr],marker='o',label=role)
        axes[1].plot(LAGS,[r['best_common_regret_mean'] for r in rr],marker='o',label=role)
    for ax in axes:ax.set_xlabel('Native decision steps since opening');ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('Fraction without a shared optimal action');axes[1].set_ylabel('Best common-action mean regret')
    fig.tight_layout();fig.savefig(root/'COMPATIBILITY.png',dpi=140);plt.close(fig)
    summary=dict(origins=len(origins),banks=len(banks),comparable_banks=sum(b['compatibility'] is not None for b in banks),
        terminal_excluded_banks=sum(len(o['terminal_excluded_lags']) for o in origins),
        native_suffixes=sum(o['suffixes'] for o in origins),
        negative_controls=sum(o['negative_control_error'] is not None for o in origins),
        negative_control_max_error=max((o['negative_control_error'] or 0) for o in origins),
        groups=groups,models=models,optimizer_steps=0,model_promoted=False,
        distinct_market_rows=len({b['row'] for b in banks}),
        observational_sample=False,equal_state_weight_diagnostic=True)
    write(root/'RESULT.json',summary)


def run(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':
        raise ValueError('COMPATIBILITY_GPU_IDENTITY')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    prior=verify_collection(args.stratified);partial=verify_collection(args.partial);probe=verify_collection(args.probe)
    pc=read(args.stratified/'CONFIG.json');qc=read(args.partial/'CONFIG.json')
    bundle=args.source/'runs/seed_7/bundles/epoch_0000';source_sha=file_sha(bundle/'model.pt')
    if (pc['source']!=spec['identity'] or qc['source']!=spec['identity'] or qc['predecessor']!=prior['identity']
            or pc['probe']!=probe['identity'] or qc['probe']!=probe['identity'] or pc['source_sha256']!=source_sha
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']):
        raise ValueError('COMPATIBILITY_SOURCE_CHAIN')
    for n,sha in pc['sources'].items():
        if file_sha(ROOT/n)!=sha:raise ValueError('COMPATIBILITY_SOURCE_CODE:'+n)
    paths={'initial_calibrated':args.probe/'fresh/CALIBRATED.pt'}
    for seed in (19,43):
        for scope in ('heads','critic_representation'):
            paths[f'r{seed}_{scope}']=args.stratified/f'r{seed}'/scope/'DIAGNOSTIC.pt'
        paths[f'r{seed}_last_core']=args.partial/f'r{seed}/last_core/DIAGNOSTIC.pt'
    config=dict(source=spec['identity'],source_sha256=source_sha,stratified=prior['identity'],partial=partial['identity'],
        probe=probe['identity'],lags=list(LAGS),origin_rule='all_archived_flat_both_allowed',
        models={name:file_sha(path) for name,path in paths.items()},optimizer_steps=0,
        sources={n:file_sha(ROOT/n) for n in ('tools/run_state_compatibility.py','app/state_compatibility.py',
            'app/decision_attribution.py','app/critic_fit_probe.py','env_plugins/three_signal_native.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    model,_=load_bundle(bundle,'cuda');train=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
    repeated_input_check(model,torch.tensor(market.inputs.windows[train[:1]],device='cuda'))
    logits,_=emit(model,market.inputs,indices=train);signals=prepared_signals(logits)[0]
    reference_logits=train_logits(model,market.inputs.windows[train]);del model
    results=[]
    for source in sorted((args.stratified/'panel').iterdir()):
        if source.is_dir():results.append(quarter(args,spec,market,signals,source,config,progress))
    if len(results)!=16:raise ValueError('COMPATIBILITY_QUARTER_COVERAGE')
    banks=[b for q in results for o in q['origins'] for b in o['banks']]
    for b in banks:
        if market.inputs.period[b['row']]!='TRAIN':raise ValueError('COMPATIBILITY_NONTRAIN_BANK')
    models=model_report(args,bundle,market,paths,banks,signals,reference_logits,progress)
    if file_sha(bundle/'model.pt')!=source_sha or any(file_sha(p)!=config['models'][n] for n,p in paths.items()):
        raise ValueError('COMPATIBILITY_SOURCE_MUTATED')
    report(args.output,results,models);seal(args.output,config['identity'],started);progress('COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('run','collect'))
    for n in ('source','source-runtime','probe','stratified','partial'):p.add_argument('--'+n,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-host')
    args=p.parse_args()
    if args.stage=='collect':
        if not args.worker_host:p.error('--worker-host required')
        collect(args);return
    if not all((args.source,args.source_runtime,args.probe,args.stratified,args.partial)):p.error('source paths required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
