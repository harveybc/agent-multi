"""Export a selected actor into a trainable SAC bundle without actor fitting."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from app.selected_actor_export import transfer_expanded_actor
from app.structural_refit import build_model,export_bundle,load_bundle,verify_files
from app.frozen_actor_compaction import compare_logits
from app.rl_subset import digest,project_inputs,support_hash
from app.three_signal_learning import emit,environment,evaluate
from app.frozen_signal_masks import prepared_signals
from app.frozen_recipe_inputs import file_sha
from app.critic_refinement import sync_targets
from tools.run_cumulative_mask_pass import authenticate,make_expanded,load_selected
from tools.run_discrete_curve_probe import configure,read,write
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_refit_mask_pass import account_equal
from tools.run_structural_sac_refit import market_for
from app.actor_checkpoint_qualification import exact_trace,exact_state


def run(a):
    configure(290);start=time.monotonic();torch.backends.cudnn.allow_tf32=False
    spec=authenticate(a.selection);done=read(a.selection/'DONE.json');verify_files(a.selection,done['files'])
    if done['identity']!=spec['identity']:raise ValueError('EXPORT_SELECTION_IDENTITY')
    source=read(a.selection/'RESULT.json');cfg=read(a.selection/'bundle/CONFIG.json')
    config=dict(schema='selected_actor_sac_export.v1',selection=spec['identity'],
        receipt=file_sha(a.selection/'DONE.json'),parent=spec['parent'],
        recipe=cfg['input_recipe']['identity'],source_code={str(p.relative_to(Path(__file__).resolve().parents[1])):file_sha(p)
        for folder in ('app','agent_plugins','env_plugins','tools') for p in (Path(__file__).resolve().parents[1]/folder).rglob('*.py')},
        actor_training=False,critic_status='fresh_heads_actor_encoder_initialization_requires_calibration')
    identity=digest(config)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=config:raise ValueError('EXPORT_CONFIG')
    if (a.output/'DONE.json').exists():verify_done(a.output,identity);return
    write(a.output/'CONFIG.json',config)
    base,expanded,restoration,_=make_expanded(dict(spec,device='cpu'))
    expanded.load_state_dict(torch.load(a.selection/'bundle/actor.pt',map_location='cpu',weights_only=True))
    model=build_model(cfg['input_recipe'],base.effective,seed=19)
    with model.loss.actor_network_params.to_module(model.actor):
        transfer=transfer_expanded_actor(expanded,model.actor_net,cfg['universe'],model.recipe,restoration['group_order'])
        encoder={k:v.detach().clone() for k,v in model.actor_net.encoder.state_dict().items()}
    # The changed-input critics start with the actor representation, not mislabeled old columns.
    for i in range(2):
        with model.loss.qvalue_network_params[i].to_module(model.loss.qvalue_network):
            model.q_net.encoder.load_state_dict(encoder,strict=True)
    sync_targets(model)
    with torch.no_grad():model.loss.log_alpha.copy_(base.loss.log_alpha)
    parent=Path(spec['parent']);ps=read(parent/'SPEC.json');market=market_for(parent,ps)
    market.inputs=project_inputs(market.inputs,model.recipe);market.recipe=model.recipe
    reference,_=load_selected(a.selection/'bundle','cpu')
    expected,_=emit(reference,market.inputs);z,_=emit(model,market.inputs)
    parity=compare_logits(expected[market.inputs.valid],z[market.inputs.valid])
    if not parity['qualified']:raise ValueError('EXPORT_LOGITS')
    risk=read(parent/'RISK.json')
    def measure(logits,root,trace):
        metrics={}
        for period,interval in model.recipe['split']['periods'].items():
            folder=root/period
            m=evaluate(environment(market,risk,ps['design']['horizon'],interval,model.effective['gamma']),
                logits,prepared=prepared_signals(logits),save=folder,diagnose=True)
            account_equal(m,source['final']['metrics'][period])
            for f in ('ACTIONS.json','TRADES.json','CLOSE_COUNTERFACTUALS.json'):
                exact_trace(read(folder/f),read(trace/period/f))
            with np.load(folder/'EQUITY.npz') as left,np.load(trace/period/'EQUITY.npz') as right:
                if set(left.files)!=set(right.files) or any(not np.array_equal(left[k],right[k]) for k in left.files):
                    raise ValueError('EXPORT_EQUITY')
            metrics[period]=m
        return metrics
    metrics=measure(z,a.output/'native',a.selection/'final')
    ix=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    probe=market.inputs.windows[ix[np.linspace(0,len(ix)-1,32,dtype=int)]]
    receipt=export_bundle(a.output/'bundle',model,probe,dict(identity=identity,new_optimizer_steps=0,
        critic_calibrated=False,selection_identity=spec['identity'],research_only=True))
    cold,_=load_bundle(a.output/'bundle');exact_state(cold.loss.state_dict(),model.loss.state_dict())
    replay,_=emit(cold,market.inputs)
    if not np.array_equal(replay,z,equal_nan=True):raise ValueError('EXPORT_COLD_LOGITS')
    measure(replay,a.output/'cold',a.output/'native')
    authenticate(a.selection);verify_files(a.selection,done['files'])
    write(a.output/'RESULT.json',dict(identity=identity,qualified=True,metrics=metrics,
        transfer=transfer,parity=parity,export=receipt,source_unchanged=True,cold_exact=True,
        support_sha256=support_hash(market.inputs),critic_calibrated=False,new_optimizer_steps=0,
        seconds=time.monotonic()-start))
    seal(a.output,identity)
    print('EXPORT_QUALIFIED',identity,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve().is_relative_to(a.selection.resolve()):raise ValueError('EXPORT_SOURCE_OVERLAP')
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a)
