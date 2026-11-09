"""Lifecycle-scoped critic reset and own-policy return calibration."""
import torch

from app.actor_learning_diagnostics import initialize_critic_heads, clear_optimizer_memory

POLICY = dict(version=1, initialization='fresh_online_critic_heads',
              calibration='own_policy_realized_soft_train_return', updates=4096,
              batch_size=128, reset_actor=False, reset_encoders=False)


def begin_stage(model, stage, mode='new_refinement', seed=7, control=False):
    if mode not in ('new_refinement','resume','inference','frozen_selection','fresh_fit'):
        raise ValueError('CRITIC_REFINEMENT_MODE')
    if mode!='new_refinement':return dict(applied=False,mode=mode)
    if not isinstance(stage,str) or not stage:raise ValueError('CRITIC_REFINEMENT_STAGE')
    if getattr(model,'_critic_refinement_stage',None) is not None:
        raise ValueError('CRITIC_REFINEMENT_ALREADY_INITIALIZED')
    receipt=dict(applied=True,mode=mode,stage=stage,control=control,
        head=None if control else initialize_critic_heads(model,seed),
        optimizer=clear_optimizer_memory(model.optimizer))
    model._critic_refinement_stage=stage
    return receipt


def sync_targets(model):
    with torch.no_grad():
        online=model.loss.qvalue_network_params
        target=model.loss.target_qvalue_network_params
        if set(online.keys(True,True))!=set(target.keys(True,True)):
            raise ValueError('CRITIC_TARGET_KEYS')
        for key,value in online.items(True,True):
            target[key].copy_(value)
            if not torch.equal(target[key],value):raise ValueError('CRITIC_TARGET_COPY')
    return dict(exact=True,scope='all online critic parameters and buffers to target')


def fit_anchor(model,replay,schedule,measure,progress):
    from app.fixed_actor_critic_audit import configure_arm
    from tools.diagnose_warmstart_drift import frozen_state, verify_frozen_state
    from app.selected_warm_start import reset_learning_rng
    flags={n:p.requires_grad for n,p in model.loss.named_parameters()}
    native=model.loss._compute_target
    try:
        configure_arm(model,'return_anchor')
        frozen=frozen_state(model)
        reset_learning_rng(7)
        points=[]
        for step in range(len(schedule)+1):
            if step:model.update(replay[schedule[step-1]].to(model.device))
            if step in (0,1,64,256,1024,2048,len(schedule)):
                verify_frozen_state(model,frozen)
                record=dict(updates=step,calibration=measure(),frozen_parameters_exact=True)
                points.append(record);progress(record)
        return points
    finally:
        model.loss._compute_target=native
        for attr in ('_audit_bootstrap_target','_audit_return_target'):
            if hasattr(model,attr):delattr(model,attr)
        for n,p in model.loss.named_parameters():p.requires_grad_(flags[n])
