"""Bounded actor-head refinement with a detached frozen-parent KL penalty."""
import math

import torch

from agent_plugins.sac_critic_regularization import module_mode, sync_target_statistics


def policy_kl(reference, current):
    if (reference.shape!=current.shape or current.ndim!=2 or current.shape[-1]!=3
            or not torch.isfinite(reference).all() or not torch.isfinite(current).all()):
        raise ValueError('ACTOR_KL_INPUT')
    logref=reference.detach().log_softmax(-1)
    return (logref.exp()*(logref-current.log_softmax(-1))).sum(-1).mean()


def arm_config(model, arm):
    if arm not in ('native','lr_div3','kl_parent'):raise ValueError('ACTOR_PRESERVATION_ARM')
    lr=model.effective['learning_rate']/(3 if arm=='lr_div3' else 1)
    for group in model.optimizer.param_groups:group['lr']=lr
    return dict(arm=arm,learning_rate=lr,beta=1.0 if arm=='kl_parent' else 0.0,
                kl_direction='parent || current')


def anchored_update(model, batch, reference_logits, beta):
    if not math.isfinite(beta) or beta<0:raise ValueError('ACTOR_KL_BETA')
    if beta==0:return model.update(batch)
    if not batch['mask'].all() or not batch['next','mask'].all():
        raise ValueError('THREE_SYMBOLS_ALWAYS_AVAILABLE')
    model.optimizer.zero_grad(set_to_none=True)
    legal=batch['mask'].sum(-1).float()
    model.loss.target_entropy.copy_(model.entropy_fraction*legal.log().mean())
    values=model.loss(batch)
    with module_mode(model.actor,False):
        kl=policy_kl(reference_logits,model.surfaces(batch['features']))
    total=values['loss_actor']+values['loss_qvalue']+values['loss_alpha']+beta*kl
    if not torch.isfinite(total):raise ValueError('ACTOR_KL_NONFINITE')
    total.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.loss.parameters(),10.,error_if_nonfinite=True)
    model.optimizer.step();model.updater.step()
    if model.explicit_modes:
        with torch.no_grad():sync_target_statistics(model.loss)
    return dict(actor=float(values['loss_actor'].detach()),critic=float(values['loss_qvalue'].detach()),
        gradient_norm=float(grad),alpha=float(model.loss.log_alpha.detach().exp()),
        target_entropy=float(model.loss.target_entropy),kl=float(kl.detach()),beta=beta)
