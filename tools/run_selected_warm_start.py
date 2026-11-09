"""Priority selected-channel SAC weight transfer and conservative head-only fit."""
import argparse
import copy
import fcntl
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from app.structural_refit import build_model, load_bundle, verify_files
from app.selected_warm_start import transfer_selected, configure_head_only, target_surfaces, export_components, reset_learning_rng
from app.frozen_actor_compaction import compare_logits
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest,support_hash
from app.three_signal_learning import emit,environment
from app.critic_refinement import POLICY, begin_stage, fit_anchor, sync_targets
from tools.run_discrete_curve_probe import read,write,now,configure
from tools.run_structural_sac_refit import prepare_selected,authenticate,market_for,fit_epochs,assess_full
from tools.run_refit_mask_pass import account_equal

ROOT=Path(__file__).resolve().parents[1]
EXTRA=['app/selected_warm_start.py','tools/run_selected_warm_start.py',
       'app/frozen_actor_compaction.py','tools/run_refit_mask_pass.py',
       'app/critic_refinement.py','app/actor_learning_diagnostics.py',
       'tools/run_preserved_policy_refinement_probe.py','tools/audit_fixed_actor_critics.py',
       'app/fixed_actor_critic_audit.py','tools/diagnose_warmstart_drift.py',
       'tools/audit_frozen_policy_returns.py','app/policy_return_calibration.py']


def prepare(args):
    if args.root.exists(): raise ValueError('NEW_WARMSTART_ROOT_REQUIRED')
    prepare_selected(SimpleNamespace(source_root=args.parent,source_runtime=args.parent_runtime,
        selection=args.selection,selection_runtime=args.selection_runtime,root=args.root,seeds=[7]))
    spec=read(args.root/'SPEC.json');spec.pop('identity')
    ss=read(args.selection/'SPEC.private.json')
    bundle=args.parent/'runs'/f"seed_{ss['seed']}"/ss['best_bundle']['relative_path']
    if (file_sha(bundle/'model.pt')!=ss['best_bundle']['model_sha256'] or
            file_sha(bundle/'MANIFEST.json')!=ss['best_bundle']['manifest_sha256']):
        raise ValueError('TRANSFER_SOURCE_CHECKPOINT')
    manifest=read(bundle/'MANIFEST.json');verify_files(bundle,manifest['files'])
    shutil.copytree(bundle,args.root/'SOURCE_BUNDLE')
    spec['effective']['learning_rate']=spec['effective']['learning_rate']/10
    spec['design'].update(require_activity=True,initialize_epoch_zero=True,warmup_policy=True,head_only=True)
    spec['design']['critic_refinement']=dict(POLICY)
    spec['structural_round'].update(variant='WARMSTART_HEAD_ONLY',fresh_weights=False,
        checkpoint_source='selected masked parent checkpoint, structurally projected',
        source_checkpoint_sha256=ss['best_bundle']['model_sha256'])
    spec['warm_start']=dict(source_epoch=ss['checkpoint_epoch'],initial_bundle='SOURCE_BUNDLE',
        learning_rate_factor=.1,trainable='actor_and_twin_critic_heads_only',
        replay='fresh; first4096 transitions sampled from transferred policy, no updates',
        optimizer='fresh AdamW, no inherited moments',temperature='copied_and_frozen',
        activity='strictly >2 orders/week in both TRAIN and VALIDATION',
        epoch_zero='eligible transferred policy remains fallback unless improved')
    spec['sources'].update({n:file_sha(ROOT/n) for n in EXTRA})
    for path in (args.root/'SOURCE_BUNDLE').rglob('*'):
        if path.is_file():spec['files'][str(path.relative_to(args.root))]=file_sha(path)
    reference=read(args.root/'REFERENCE.json');reference['label']='Parent epoch6 with70 selected channels; frozen weights'
    write(args.root/'REFERENCE.json',reference);spec['files']['REFERENCE.json']=file_sha(args.root/'REFERENCE.json')
    spec['identity']=digest(spec);write(args.root/'SPEC.json',spec)
    pre=read(args.root/'PREFLIGHT.json');pre.update(identity=spec['identity'],fresh_training=False,
        structural_round=spec['structural_round'],transfer_validation='required_before_training')
    write(args.root/'PREFLIGHT.json',pre)
    print(spec['identity'],flush=True)


def check_transfer(source,model,market):
    names=source.recipe['list']['columns'];take=[names.index(c) for c in model.recipe['list']['columns']]
    def expand(x):
        full=x.new_zeros(x.shape[0],x.shape[1],len(names));full[:,:,take]=x;return full
    reference=SimpleNamespace(device=model.device,surfaces=lambda x:source.surfaces(expand(x)))
    a,_=emit(reference,market.inputs);b,_=emit(model,market.inputs);valid=market.inputs.valid
    report=compare_logits(a[valid],b[valid])
    if not report['qualified']:raise ValueError('TRANSFER_FULL_POLICY_PARITY')
    errors={'online_q_max_abs':0.,'target_q_max_abs':0.}
    ix=np.flatnonzero(valid)
    with torch.inference_mode():
        for start in range(0,len(ix),256):
            x=torch.tensor(market.inputs.windows[ix[start:start+256]],device=model.device)
            for name,fn in [('online_q_max_abs',lambda m,t:m.q_surfaces(t)),('target_q_max_abs',target_surfaces)]:
                old,new=fn(source,expand(x)),fn(model,x)
                torch.testing.assert_close(old,new,rtol=1e-5,atol=1e-6)
                errors[name]=max(errors[name],float((old-new).abs().max()))
    return dict(actor=report,**errors)


def qualify_new_critics(model,market,risk,design,out,identity):
    """Calibrate once on this transferred policy, never reset recovered learning."""
    if (out/'RESUME.json').exists():return dict(mode='resume',reset=False)
    if design.get('critic_refinement')!=POLICY:raise ValueError('WARM_CRITIC_POLICY_REQUIRED')
    from tools.audit_fixed_actor_critics import collect_full_replay, covering_schedule, verify_collection
    from tools.audit_frozen_policy_returns import account_calibration
    from tools.run_preserved_policy_refinement_probe import calibration_panel
    from app.policy_return_calibration import policy_signals
    from app.three_signal_learning import evaluate
    stage=out/'critic_initialization';stage.mkdir(exist_ok=True)
    if (stage/'DONE.json').exists():
        receipt=verify_collection(stage)
        if receipt['identity']!=identity:raise ValueError('WARM_CRITIC_STAGE_IDENTITY')
        model.load(stage/'CALIBRATED.pt')
        return dict(mode='completed_stage_reload',reset=False)
    logits,q=emit(model,market.inputs,twins=True)
    prepared,probabilities=policy_signals(logits,7)
    interval=market.recipe['split']['periods']['TRAIN'];gamma=model.effective['gamma']
    env=lambda:environment(market,risk,design['horizon'],interval,gamma)
    evaluate(env(),logits,save=stage/'policy_train',prepared=prepared)
    account_calibration(stage/'policy_train',market.inputs,q,probabilities,gamma,
                        float(model.loss.log_alpha.detach().exp()),True)
    replay,receipt=collect_full_replay(env(),read(stage/'policy_train/ACTIONS.json'))
    labels=np.load(stage/'policy_train/RETURN_TRACE.npz')['soft_return']
    replay['realized_soft_return']=torch.tensor(labels,dtype=torch.float32).reshape(-1,1)
    init=begin_stage(model,identity)
    schedule=covering_schedule(len(replay),POLICY['updates'],POLICY['batch_size'],7)
    points=fit_anchor(model,replay,schedule,lambda:calibration_panel(model,replay),
        lambda r:write(stage/'PROGRESS.json',r))
    sync=sync_targets(model);model.save(stage/'CALIBRATED.pt')
    write(stage/'RESULT.json',dict(identity=identity,initialization=init,replay=receipt,
        calibration=points,target_sync=sync,period='TRAIN',source_policy='transferred actor unchanged'))
    write(stage/'DONE.json',dict(identity=identity,files={str(p.relative_to(stage)):file_sha(p)
        for p in stage.rglob('*') if p.is_file() and p.name not in ('DONE.json','PROGRESS.json')}))
    return dict(mode='new_refinement',reset=True,calibrated=True)


def worker(args):
    spec=authenticate(args.root);out=args.root/'runs'/'seed_7';out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'RESULT.json').exists():
            r=read(out/'RESULT.json');load_bundle(out/r['best_bundle']['relative_path'])
            return
        if not torch.cuda.is_available():raise ValueError('CUDA_REQUIRED')
        configure(7);torch.backends.cudnn.allow_tf32=False
        market=market_for(args.root,spec)
        if support_hash(market.inputs)!=read(args.root/'PREFLIGHT.json')['support_sha256']:
            raise ValueError('WARMSTART_SUPPORT')
        source,_=load_bundle(args.root/'SOURCE_BUNDLE','cuda')
        model=build_model(market.recipe,spec['effective'],7,'cuda')
        copied=transfer_selected(source,model)
        modes=configure_head_only(model)
        parity=check_transfer(source,model,market)
        risk=read(args.root/'RISK.json');d=spec['design']
        initial=assess_full(model,market,risk,d,save=out/'initial',diagnose=True)
        expected=read(args.root/'SELECTION_RESULT.json')['final']['metrics']
        for p,m in initial.items():account_equal(m,expected[p])
        if not all(m['orders_per_week']>2 for m in initial.values()):raise ValueError('WARMSTART_ACTIVITY')
        export_components(model,out/'components_initial')
        write(out/'TRANSFER.json',dict(identity=spec['identity'],copied=copied,modes=modes,
            parity=parity,initial_metrics=initial,full_accounts_exact=True,
            source_checkpoint_sha256=spec['structural_round']['source_checkpoint_sha256']))
        del source
        critic_stage=qualify_new_critics(model,market,risk,d,out,spec['identity'])
        write(out/'CRITIC_POLICY.json',critic_stage)
        reset_learning_rng(7)
        ix=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
        probe=market.inputs.windows[ix[np.linspace(0,len(ix)-1,32,dtype=int)]].copy()
        write(out/'EFFECTIVE.json',dict(identity=spec['identity'],effective=model.effective,
            feature_count=market.recipe['list']['n'],groups=len(market.recipe['representation']['resolved']['names']),
            gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.loss.parameters()),
            trainable_parameters=sum(p.numel() for p in model.loss.parameters() if p.requires_grad),
            input_shape=list(probe.shape[1:]),seed=7,modes=modes))
        state=fit_epochs(model,lambda:environment(market,risk,d['horizon'],
            market.recipe['split']['periods']['TRAIN'],model.effective['gamma']),
            lambda m:assess_full(m,market,risk,d),d,7,out,spec['identity'],probe)
        metrics=assess_full(model,market,risk,d,save=out/'final',diagnose=True)
        if not all(m['orders_per_week']>2 for m in metrics.values()):raise ValueError('FINAL_ACTIVITY')
        if metrics['VALIDATION']['return_per_week']<initial['VALIDATION']['return_per_week']:
            raise ValueError('FINAL_WORSE_THAN_EPOCH_ZERO')
        export_components(model,out/'components_selected')
        write(out/'RESULT.json',dict(identity=spec['identity'],seed=7,best_bundle=state['best_bundle'],
            metrics=metrics,initial_metrics=initial,activity_admissible=True,finished_at=now(),
            scope='warmstarted70_head_only; development full4plus1; unchanged selected input support'))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('prepare')
    for n in ('parent','parent-runtime','selection','selection-runtime','root'):a.add_argument('--'+n,type=Path,required=True)
    a=s.add_parser('worker');a.add_argument('--root',type=Path,required=True)
    args=p.parse_args();globals()[args.cmd](args)


if __name__=='__main__':main()
