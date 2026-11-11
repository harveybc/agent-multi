"""Small, explicit function-space projection and FIT-only trust checks."""
import math

import torch


BUDGET = dict(step_margin_rms=.002, mean_kl=1e-4, max_kl=1e-3,
              relative_parameter_drift=.01, damping_ratio=1e-3, backtracks=24)


def projected_direction(jacobian, desired, weights, damping_ratio=1e-3):
    if (jacobian.ndim!=3 or jacobian.shape[1]!=2 or desired.shape!=jacobian.shape[:2]
            or weights.shape!=(len(desired),) or (weights<0).any()
            or not torch.isfinite(jacobian).all() or not torch.isfinite(desired).all()
            or not torch.isfinite(weights).all() or float(weights.sum())<=0
            or not math.isfinite(damping_ratio) or damping_ratio<=0):
        raise ValueError('FUNCTIONAL_PROJECTION_INPUT')
    scale=(weights/weights.sum()).sqrt()[:,None]
    j=(jacobian*scale[:,:,None]).reshape(-1,jacobian.shape[-1]).double()
    target=(desired*scale).reshape(-1).double()
    gram=j@j.T;mean=float(gram.trace()/2)
    if mean==0:
        return torch.zeros(j.shape[-1],device=j.device,dtype=j.dtype),dict(
            damping=0.,target_norm=float(target.norm()),residual_norm=float(target.norm()),zero_sensitivity=True)
    damping=damping_ratio*mean
    delta=j.T@torch.linalg.solve(gram+damping*torch.eye(len(gram),device=j.device,dtype=j.dtype),target)
    if not torch.isfinite(delta).all():raise ValueError('FUNCTIONAL_PROJECTION_NONFINITE')
    return delta,dict(damping=damping,target_norm=float(target.norm()),
        residual_norm=float((j@delta-target).norm()),zero_sensitivity=False)


def desired_margins(local_gradient, weights, rms):
    margin=-(local_gradient[:,1:]-local_gradient[:,:1])
    current=((margin.square().mean(1)*weights).sum()/weights.sum()).sqrt()
    if not math.isfinite(rms) or rms<=0:raise ValueError('FUNCTIONAL_TARGET_RMS')
    return margin*(rms/current if float(current)>0 else 0.)


def budget_metrics(source_logits, before_logits, after_logits, source_theta, after_theta):
    tensors=(source_logits,before_logits,after_logits,source_theta,after_theta)
    if (any(not torch.isfinite(t).all() for t in tensors)
            or source_logits.ndim!=2 or source_logits.shape[1]!=3
            or before_logits.shape!=source_logits.shape or after_logits.shape!=source_logits.shape
            or source_theta.shape!=after_theta.shape or source_theta.ndim!=1
            or float(source_theta.norm())==0):raise ValueError('FUNCTIONAL_BUDGET_INPUT')
    log0=source_logits.double().log_softmax(-1);log1=after_logits.double().log_softmax(-1)
    kl=(log0.exp()*(log0-log1)).sum(-1).clamp_min(0)
    change=(after_logits[:,1:]-after_logits[:,:1])-(before_logits[:,1:]-before_logits[:,:1])
    return dict(step_margin_rms=float(change.double().square().mean().sqrt()),
        mean_kl=float(kl.mean()),max_kl=float(kl.max()),
        relative_parameter_drift=float((after_theta-source_theta).norm()/source_theta.norm()))


def admissible(metrics, before_loss, after_loss, budget=BUDGET):
    return (math.isfinite(before_loss) and math.isfinite(after_loss)
        and after_loss<=before_loss+1e-10
        and all(math.isfinite(metrics[k]) and metrics[k]<=budget[k]+1e-12
                for k in ('step_margin_rms','mean_kl','max_kl','relative_parameter_drift')))


def backtrack(parameters, delta, check, trials=24):
    initial=[p.detach().clone() for p in parameters]
    if (trials<1 or delta.ndim!=1 or delta.numel()!=sum(p.numel() for p in parameters)
            or not torch.isfinite(delta).all()):raise ValueError('FUNCTIONAL_TRIAL_INPUT')
    accepted=False;last={}
    try:
        for attempt in range(trials):
            fraction=2.**(-attempt);offset=0
            with torch.no_grad():
                for p,old in zip(parameters,initial):
                    part=delta[offset:offset+p.numel()].reshape(p.shape)
                    p.copy_((old.double()+fraction*part).to(p));offset+=p.numel()
            last=check()
            if last['accepted']:
                accepted=True
                return dict(last,fraction=fraction,attempts=attempt+1)
        return dict(accepted=False,fraction=0.,attempts=trials,last_rejection=last)
    finally:
        if not accepted:
            with torch.no_grad():
                for p,old in zip(parameters,initial):p.copy_(old)
