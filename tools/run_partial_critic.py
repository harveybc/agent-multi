"""Restore critic components diagnostically, then fit only last core and heads."""
import argparse
import csv
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from app.action_ranking_contrast import advantage_loss
from app.actor_learning_diagnostics import clear_optimizer_memory
from app.critic_fit_probe import classify_anchor, probe_q, set_scope
from app.frozen_recipe_inputs import file_sha
from app.partial_critic import module_name, restore_component, set_partial_scope, weight_drift
from app.rl_subset import digest, support_hash
from app.selected_warm_start import reset_learning_rng
from app.stratified_critic import grouped_metrics, validate_support
from app.structural_refit import load_bundle
from tools.audit_fixed_actor_critics import covering_schedule, verify_collection
from tools.diagnose_warmstart_drift import frozen_state, verify_frozen_state
from tools.run_action_ranking_contrast import ensure_config, seal
from tools.run_actor_preservation import collect
from tools.run_critic_fit_probe import train_logits
from tools.run_discrete_curve_probe import configure, now, read, write
from tools.run_stratified_critic import BOUNDARY, POINTS, export_report
from tools.run_structural_sac_refit import authenticate, market_for

ROOT=Path(__file__).resolve().parents[1]


def assert_metrics(actual,expected):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('PARTIAL_METRIC_KEYS')
        for key in expected:assert_metrics(actual[key],expected[key])
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('PARTIAL_METRIC_LIST')
        for a,b in zip(actual,expected):assert_metrics(a,b)
    elif expected is None:
        if actual is not None:raise ValueError('PARTIAL_METRIC_NULL')
    elif isinstance(expected,(int,float)):
        if not np.isfinite(actual) or abs(actual-expected)>1e-6:raise ValueError('PARTIAL_METRIC_PARITY')
    elif actual!=expected:raise ValueError('PARTIAL_METRIC_VALUE')


def metrics(model,features,targets,anchors):
    with torch.no_grad():
        q=torch.cat([probe_q(model,features[i:i+128]) for i in range(0,len(features),128)],dim=1).cpu().numpy()
    return grouped_metrics(q,targets,anchors)


def diagnose(args,model,initial,features,targets,anchors,archives,config,progress):
    root=args.output/'diagnosis';root.mkdir(exist_ok=True)
    if (root/'DONE.json').exists():
        if verify_collection(root)['identity']!=config['identity']:raise ValueError('PARTIAL_DIAGNOSIS_IDENTITY')
        return
    started=time.monotonic();results={};rows=[]
    for seed in (19,43):
        model.load(args.predecessor/f'r{seed}/critic_representation/DIAGNOSTIC.pt')
        set_scope(model,'critic_representation')
        full={n:p.detach().clone() for n,p in model.loss.named_parameters()}
        original=metrics(model,features,targets,anchors)
        assert_metrics(original,archives[str(seed)]['critic_representation']['points'][-1]['metrics'])
        drift=weight_drift(initial,full);restorations={}
        for component in sorted(drift):
            restore_component(model,initial,component)
            restored=metrics(model,features,targets,anchors)
            restore_component(model,full,component)
            if any(not torch.equal(p,full[n]) for n,p in model.loss.named_parameters()):
                raise ValueError('PARTIAL_RESTORE_ROUNDTRIP')
            restorations[component]=restored
            for role,groups in restored.items():
                for kind,m in groups.items():
                    if m is None:continue
                    baseline=original[role][kind]['minimum'];v=m['minimum']
                    rows.append(dict(seed=seed,component=component,role=role,kind=kind,
                        delta_l2=drift[component]['delta_l2'],relative_l2=drift[component]['relative_l2'],
                        gap_rmse_delta=v['gap_rmse']-baseline['gap_rmse'],
                        mean_regret_delta=v['mean_regret']-baseline['mean_regret'],
                        accuracy_delta=None if v['pairwise_accuracy'] is None else v['pairwise_accuracy']-baseline['pairwise_accuracy']))
            progress('RESTORATIONS',seed=seed,component=component,completed=len(restorations),total=len(drift))
        results[str(seed)]=dict(drift=drift,original=original,restorations=restorations,
            full_endpoint_metric_parity=True,parameter_roundtrip_exact=True)
    with (root/'RESTORATIONS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(root/'RESULT.json',results);seal(root,config['identity'],started)


def run(args):
    started=time.monotonic();configure(7);torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 5090':
        raise ValueError('PARTIAL_GPU_IDENTITY')
    spec=authenticate(args.source,args.source_runtime);market=market_for(args.source,spec)
    previous=verify_collection(args.predecessor);pc=read(args.predecessor/'CONFIG.json')
    probe=verify_collection(args.probe);bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    source_sha=file_sha(bundle/'model.pt');cal_sha=file_sha(args.probe/'fresh/CALIBRATED.pt')
    if (pc['identity']!=previous['identity'] or pc['source']!=spec['identity']
            or pc['probe']!=probe['identity'] or pc['source_sha256']!=source_sha
            or pc['calibrated_sha256']!=cal_sha or pc['updates']!=4096 or pc['batch_size']!=128
            or pc['seeds']!=[19,43] or pc['loss']!='pairwise_plus_mean'
            or support_hash(market.inputs)!=read(args.source/'PREFLIGHT.json')['support_sha256']):
        raise ValueError('PARTIAL_PARENT_CONTRACT')
    for name,sha in pc['sources'].items():
        if file_sha(ROOT/name)!=sha:raise ValueError('PARTIAL_ARCHIVED_CODE_CHANGED:'+name)
    records=[r for f in sorted((args.predecessor/'panel').glob('*/RESULT.json')) for r in read(f)['records']]
    anchors=[r['anchor'] for r in records];panel=read(args.predecessor/'PANEL.json')
    if anchors!=panel['anchors']:raise ValueError('PARTIAL_ANCHORS')
    validate_support(anchors,BOUNDARY)
    targets=np.array([classify_anchor(r)['targets'] for r in records])
    features=torch.tensor(market.inputs.windows[[r['row'] for r in anchors]],device='cuda')
    target=torch.tensor(targets,device='cuda',dtype=torch.float32)
    fit_ix=np.array([i for i,r in enumerate(anchors) if r['role']=='fit'])
    archives=read(args.predecessor/'RESULT.json')['arms']
    model,_=load_bundle(bundle,'cuda');model.load(args.probe/'fresh/CALIBRATED.pt')
    mapping=set_partial_scope(model)
    if sum(v['elements'] for v in mapping.values())!=6534:raise ValueError('PARTIAL_PARAMETER_COUNT')
    initial={n:p.detach().clone() for n,p in model.loss.named_parameters()}
    config=dict(source=spec['identity'],source_sha256=source_sha,calibrated_sha256=cal_sha,
        predecessor=previous['identity'],probe=probe['identity'],scope=mapping,
        seeds=[19,43],updates=4096,batch_size=128,loss='pairwise_plus_mean',actor_updates=0,
        reused_controls=['heads','critic_representation'],model_promoted=False,
        sources={n:file_sha(ROOT/n) for n in ('tools/run_partial_critic.py','app/partial_critic.py',
            'tools/run_stratified_critic.py','app/critic_fit_probe.py','app/stratified_critic.py',
            'app/action_ranking_contrast.py','agent_plugins/double_tcn_probe.py')})
    config['identity']=digest(config);ensure_config(args.output,config)
    if (args.output/'DONE.json').exists():verify_collection(args.output);return
    def progress(stage,**kw):
        write(args.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    train=market.inputs.valid & (market.inputs.period=='TRAIN')
    parent=train_logits(model,market.inputs.windows[train])
    baseline=metrics(model,features,targets,anchors)
    for seed in (19,43):
        for scope in ('heads','critic_representation'):
            assert_metrics(baseline,archives[str(seed)][scope]['points'][0]['metrics'])
    diagnose(args,model,initial,features,targets,anchors,archives,config,progress)
    del model
    results={s:dict(arms) for s,arms in archives.items()}
    for seed in (19,43):
        lane=args.output/f'r{seed}'/'last_core';lane.mkdir(parents=True,exist_ok=True)
        if (lane/'DONE.json').exists():
            if verify_collection(lane)['identity']!=config['identity']:raise ValueError('PARTIAL_ARM_IDENTITY')
            results[str(seed)]['last_core']=read(lane/'RESULT.json');continue
        model,_=load_bundle(bundle,'cuda');model.load(args.probe/'fresh/CALIBRATED.pt')
        actual=set_partial_scope(model)
        if actual!=mapping:raise ValueError('PARTIAL_MAP_CHANGED')
        clear_optimizer_memory(model.optimizer);reset_learning_rng(seed)
        frozen=frozen_state(model);schedule=fit_ix[covering_schedule(len(fit_ix),4096,128,seed)]
        for control in ('heads','critic_representation'):
            effective=read(args.predecessor/f'r{seed}'/control/'EFFECTIVE.json')
            if (digest(schedule.tolist())!=effective['schedule_sha256']
                    or effective['learning_rate']!=model.effective['learning_rate']
                    or effective['weight_decay']!=model.effective['weight_decay']):
                raise ValueError('PARTIAL_MATCHED_OPTIMIZATION')
        write(lane/'EFFECTIVE.json',dict(identity=config['identity'],scope=mapping,seed=seed,
            learning_rate=model.effective['learning_rate'],weight_decay=model.effective['weight_decay'],
            schedule_sha256=digest(schedule.tolist()),optimizer='fresh'))
        points=[];losses=[]
        for step in range(4097):
            if step:
                ix=schedule[step-1];model.optimizer.zero_grad(set_to_none=True)
                q=probe_q(model,features[ix]);loss=advantage_loss(q,target[ix]);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.loss.parameters(),10.,error_if_nonfinite=True)
                model.optimizer.step();losses.append(float(loss.detach()))
            if step in POINTS:
                verify_frozen_state(model,frozen)
                moved=[n for n,p in model.loss.named_parameters() if not torch.equal(p,initial[n])]
                if not set(moved)<=set(mapping):raise ValueError('PARTIAL_UNDECLARED_UPDATE')
                if step and (not any('.core.1.' in n for n in moved) or not any('.head.' in n for n in moved)):
                    raise ValueError('PARTIAL_NO_CORE_OR_HEAD_UPDATE')
                value=metrics(model,features,targets,anchors)
                if step==0:assert_metrics(value,baseline)
                points.append(dict(updates=step,changed_tensors=len(moved),metrics=value))
                write(lane/'CURVE.json',points);progress('FIT',seed=seed,scope='last_core',updates=step)
        if not np.array_equal(train_logits(model,market.inputs.windows[train]),parent):
            raise ValueError('PARTIAL_ACTOR_CHANGED')
        model.save(lane/'DIAGNOSTIC.pt')
        record=dict(identity=config['identity'],points=points,losses=losses,
            full_train_actor_logits_exact=True,frozen_parameters_exact=True,model_promoted=False)
        write(lane/'RESULT.json',record);seal(lane,config['identity'],started)
        results[str(seed)]['last_core']=record;del model
    if file_sha(bundle/'model.pt')!=source_sha or file_sha(args.probe/'fresh/CALIBRATED.pt')!=cal_sha:
        raise ValueError('PARTIAL_SOURCE_MUTATED')
    export_report(args.output,results)
    write(args.output/'RESULT.json',dict(identity=config['identity'],arms=results,
        reused_control_identity=previous['identity'],new_arms=2,actor_updates=0,model_promoted=False))
    seal(args.output,config['identity'],started);progress('COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('run','collect'))
    for n in ('source','source-runtime','probe','predecessor'):p.add_argument('--'+n,type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-host')
    args=p.parse_args()
    if args.stage=='collect':
        if not args.worker_host:p.error('--worker-host required')
        collect(args);return
    if not all((args.source,args.source_runtime,args.probe,args.predecessor)):p.error('source paths required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as exc:write(args.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
