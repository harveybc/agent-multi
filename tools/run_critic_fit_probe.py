"""Audit native TRAIN labels, then fit small critic panels without actor updates."""
import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from app.action_ranking_contrast import advantage_loss, calibration_indices
from app.actor_learning_diagnostics import clear_optimizer_memory
from app.critic_fit_probe import classify_anchor, choose_small_set, fit_metrics, fit_passed, set_scope, probe_q
from app.decision_attribution import continuation
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, support_hash
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle
from app.three_signal_learning import environment
from tools.attribute_changed_decisions import trace_check
from tools.attribute_ranking_improvement import account_check
from tools.audit_fixed_actor_critics import verify_collection
from tools.diagnose_warmstart_drift import frozen_state, verify_frozen_state
from tools.run_action_ranking_contrast import load_panel, seal, ensure_config
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_structural_sac_refit import authenticate, market_for

ROOT=Path(__file__).resolve().parents[1]
POINTS=(0,64,256,1024,2048,4096)


def train_logits(model,features):
    with torch.inference_mode():
        return torch.cat([model.surfaces(torch.tensor(features[i:i+256],device='cuda')).cpu()
                          for i in range(0,len(features),256)]).numpy()


def audit(args,spec,market,root,records,chosen):
    groups=defaultdict(list);rows=[];duplicates=defaultdict(list)
    for i,r in enumerate(records):
        a=r['anchor'];c=classify_anchor(r)
        row=dict(id=a['id'],year=a['year'],role=a['role'],position=a['position'],**c)
        rows.append(row);groups[(a['role'],c['kind'])].append(row)
        x=np.ascontiguousarray(market.inputs.windows[a['row']])
        duplicates[hashlib.sha256(x.tobytes()).hexdigest()].append(i)
    summary=[]
    for (role,kind),rr in sorted(groups.items()):
        values=np.array([r['targets'] for r in rr]);refusals=Counter()
        for r in rr:refusals.update(r['refusals'])
        summary.append(dict(role=role,kind=kind,anchors=len(rr),
            years=dict(Counter(r['year'] for r in rr)),
            informative_pairs=sum(r['informative_pairs'] for r in rr),
            tied_pairs=sum(r['tied_pairs'] for r in rr),refusals=dict(refusals),
            target_quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist()))
    duplicate_groups=[ix for ix in duplicates.values() if len(ix)>1]
    conflicts=[ix for ix in duplicate_groups if np.ptp(np.array([rows[i]['targets'] for i in ix]),axis=0).max()>1e-8]
    # Reconstruct the native reference path and independently rerun one fit
    # anchor per year with all three actions before admitting any critic update.
    witnesses={next(i for i in chosen if records[i]['anchor']['year']==year) for year in range(2020,2024)}
    witness_map={records[i]['anchor']['trace_index']:i for i in witnesses}
    trace=read(args.probe/'reference/TRAIN/ACTIONS.json')
    timestamps=[r['timestamp'] for r in trace];ix=np.searchsorted(market.inputs.row_ts_ns,timestamps)
    if (ix>=len(market.inputs.row_ts_ns)).any() or not np.array_equal(market.inputs.row_ts_ns[ix],timestamps):
        raise ValueError('FIT_PROBE_TRACE_SUPPORT')
    tape=np.zeros(len(market.inputs.valid),dtype=np.int64);tape[ix]=[r['signal'] for r in trace]
    env=environment(market,read(args.source/'RISK.json'),spec['design']['horizon'],
                    market.recipe['split']['periods']['TRAIN'],spec['effective']['gamma'])
    env.reset();maximum=0.;witnessed=[]
    reference_cash=read(args.probe/'reference/TRAIN/METRICS.json')['net_profit_usd']
    for r in records:
        a=r['anchor'];t=trace[a['trace_index']]
        if t['timestamp']!=a['timestamp'] or t['position']!=a['position']:
            raise ValueError('FIT_PROBE_ANCHOR_STATE')
        if abs(r['results']['greedy'][t['signal']]['net_usd']-reference_cash)>1e-6:
            raise ValueError('FIT_PROBE_REFERENCE_RETURN')
    for i,t in enumerate(trace):
        if i in witness_map:
            r=records[witness_map[i]]
            for action in range(3):
                value=continuation(env,tape,action);expected=r['results']['greedy'][action]
                for key in ('reward_return','soft_return','net_usd'):
                    difference=abs(value[key]-expected[key]);maximum=max(maximum,difference)
                    if difference>1e-8:raise ValueError('FIT_PROBE_NATIVE_LABEL_REPLAY')
                if value['first_effect']!=expected['first_effect']:raise ValueError('FIT_PROBE_FIRST_EFFECT')
            witnessed.append(r['anchor']['id'])
        trace_check(env,t,t['signal'])
    account_check(env,read(args.probe/'reference/TRAIN/METRICS.json'))
    result=dict(groups=summary,rows=rows,duplicate_input_groups=duplicate_groups,
        contradictory_duplicate_groups=conflicts,chosen=chosen,
        chosen_anchors=[records[i]['anchor'] for i in chosen],
        native_witnesses=witnessed,native_suffixes=12,maximum_replay_difference=maximum,
        fit_only=True,independent_market_holdout=False)
    write(root/'AUDIT.json',result);return result


def report(root,results):
    rows=[]
    for replica,result in results.items():
        for arm,record in result['arms'].items():
            for point in record['points']:
                for population in ('small_set','other_fit','monitor'):
                    m=point['metrics'][population]['minimum']
                    rows.append(dict(replica=replica,arm=arm,updates=point['updates'],population=population,
                        **{k:m[k] for k in ('n','pairwise_accuracy','mean_regret','maximum_regret','gap_rmse','level_rmse','rmse')}))
    with (root/'CURVES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,replica in enumerate(('r19','r43')):
        for arm in results[replica]['arms']:
            for population in ('small_set','monitor'):
                rr=[r for r in rows if r['replica']==replica and r['arm']==arm and r['population']==population]
                for ax,key in ((axes[0,col],'gap_rmse'),(axes[1,col],'pairwise_accuracy')):
                    ax.plot([r['updates'] for r in rr],[r[key] for r in rr],marker='.',label=arm+' '+population)
                    ax.set(title=replica+' '+key,xlabel='Critic updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'FIT_CURVES.png',dpi=140);plt.close(fig)
    write(root/'RESULT.json',dict(replicas=results,actor_updates=0,fit_train_only=True,
        independent_market_holdout=False,model_promoted=False))


def run(args):
    start=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':
        raise ValueError('FIT_PROBE_GPU')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    probe=verify_collection(args.probe);pc=read(args.probe/'CONFIG.json')
    bundle=args.source/'runs/seed_7/bundles/epoch_0000';source_sha=file_sha(bundle/'model.pt')
    if (pc['source_identity']!=spec['identity'] or source_sha!=pc['source_sha256']
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']):
        raise ValueError('FIT_PROBE_SOURCE')
    parents={r:verify_collection(args.ranking/r/'fit') for r in ('r19','r43')}
    config=dict(source=spec['identity'],probe=probe['identity'],source_sha256=source_sha,
        parents={r:p['identity'] for r,p in parents.items()},replicas=['r19','r43'],
        updates=4096,small_set=32,per_year=8,scope_sequence=['heads','critic_representation_if_needed'],
        seeds=[19,43],loss='pairwise_plus_mean',actor_updates=0,
        pass_criterion=dict(pairwise_accuracy=1.,gap_rmse_max=.05,maximum_regret=1e-8),
        sources={n:file_sha(ROOT/n) for n in ('tools/run_critic_fit_probe.py','app/critic_fit_probe.py',
            'app/action_ranking_contrast.py','tools/run_action_ranking_contrast.py',
            'agent_plugins/three_signal_sac.py','tools/diagnose_warmstart_drift.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):
        write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-start,**kw))
    reference,_=load_bundle(bundle,'cuda');set_scope(reference,'heads')
    train=market.inputs.valid & (market.inputs.period=='TRAIN')
    parent=train_logits(reference,market.inputs.windows[train]);del reference
    results={}
    for seed in (19,43):
        replica=f'r{seed}';root=args.ranking/replica;rc=read(root/'CONFIG.json')
        if (rc['identity']!=parents[replica]['identity'] or rc['source_identity']!=spec['identity']
                or rc['probe_identity']!=probe['identity'] or rc['update_seed']!=seed):
            raise ValueError('FIT_PROBE_PANEL_IDENTITY')
        cohort,greedy,_,_=load_panel(root,rc['identity'],rc['seeds'],rc['shard_weights'])
        fit_ix=calibration_indices(cohort,market.inputs.row_ts_ns,market.inputs.valid,market.inputs.period)
        records=sorted([read(p) for p in root.glob('shard_*/rows/*.json')],key=lambda r:r['anchor']['id'])
        if [r['anchor'] for r in records]!=cohort:raise ValueError('FIT_PROBE_PANEL_ORDER')
        chosen=choose_small_set(records,seed);directory=args.output/replica;directory.mkdir(exist_ok=True)
        progress('AUDIT',replica=replica)
        audit_result=audit(args,spec,market,directory,records,chosen)
        if audit_result['contradictory_duplicate_groups']:
            # Report the conflict but do not silently relabel or remove rows.
            progress('AUDIT_CONFLICT_RECORDED',replica=replica)
        features=torch.tensor(market.inputs.windows[[r['row'] for r in cohort]],device='cuda')
        target=torch.tensor(greedy,device='cuda',dtype=torch.float32)
        masks=dict(small_set=np.array(chosen),other_fit=np.array([i for i in fit_ix if i not in chosen]),
                   monitor=np.array([i for i,r in enumerate(cohort) if r['role']=='monitor']))
        results[replica]=dict(audit=audit_result,arms={})
        for scope in ('heads','critic_representation'):
            if scope=='critic_representation' and results[replica]['arms']['heads']['passed']:
                write(directory/'REPRESENTATION_SKIPPED.json',dict(reason='heads_met_fixed_fit_criterion'))
                break
            lane=directory/scope;lane.mkdir(exist_ok=True)
            if (lane/'DONE.json').exists():
                if verify_collection(lane)['identity']!=config['identity']:raise ValueError('FIT_PROBE_RESUME')
                results[replica]['arms'][scope]=read(lane/'RESULT.json');continue
            model,_=load_bundle(bundle,'cuda');model.load(args.probe/'fresh/CALIBRATED.pt')
            set_scope(model,scope);clear_optimizer_memory(model.optimizer);reset_learning_rng(seed)
            frozen=frozen_state(model);initial={n:p.detach().clone() for n,p in model.loss.named_parameters() if p.requires_grad}
            points=[];losses=[];gradient=0.
            write(lane/'EFFECTIVE.json',dict(scope=scope,learning_rate=model.effective['learning_rate'],
                weight_decay=model.effective['weight_decay'],trainable={n:p.numel() for n,p in model.loss.named_parameters() if p.requires_grad}))
            for step in range(4097):
                if step:
                    model.optimizer.zero_grad(set_to_none=True)
                    q=probe_q(model,features[chosen]);loss=advantage_loss(q,target[chosen])
                    loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.loss.parameters(),10.,error_if_nonfinite=True))
                    model.optimizer.step();losses.append(float(loss.detach()))
                if step in POINTS:
                    verify_frozen_state(model,frozen)
                    moved=[n for n,p in model.loss.named_parameters() if n in initial and not torch.equal(p,initial[n])]
                    if step and (not moved or (scope=='critic_representation' and not any('.encoder.' in n for n in moved))):
                        raise ValueError('FIT_PROBE_PARAMETERS_NOT_MOVING')
                    with torch.no_grad():
                        q=torch.cat([probe_q(model,features[i:i+128]) for i in range(0,len(features),128)],dim=1).cpu().numpy()
                    metrics={name:fit_metrics(q[:,ix],greedy[ix]) for name,ix in masks.items()}
                    points.append(dict(updates=step,gradient_norm=gradient,changed_tensors=len(moved),metrics=metrics))
                    write(lane/'CURVE.json',points);progress('FIT',replica=replica,scope=scope,updates=step,
                        small_set_gap_rmse=metrics['small_set']['minimum']['gap_rmse'],
                        small_set_accuracy=metrics['small_set']['minimum']['pairwise_accuracy'])
            if not np.array_equal(train_logits(model,market.inputs.windows[train]),parent):
                raise ValueError('FIT_PROBE_ACTOR_CHANGED')
            model.save(lane/'DIAGNOSTIC.pt')
            record=dict(identity=config['identity'],scope=scope,points=points,losses=losses,
                passed=fit_passed(points[-1]['metrics']['small_set']),frozen_parameters_exact=True,
                full_train_actor_logits_exact=True,model_promoted=False)
            write(lane/'RESULT.json',record);seal(lane,config['identity'],start)
            results[replica]['arms'][scope]=record;del model
    if file_sha(bundle/'model.pt')!=source_sha:raise ValueError('FIT_PROBE_SOURCE_MUTATED')
    report(args.output,results);seal(args.output,config['identity'],start);progress('COMPLETE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('source','source-runtime','probe','ranking','output'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
