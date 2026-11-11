"""Exact categorical actor gradient projections; no model fitting or selection."""
import torch


def logit_terms(logits, q, alpha):
    if (logits.ndim != 2 or logits.shape[1] != 3 or q.shape != logits.shape
            or not torch.isfinite(logits).all() or not torch.isfinite(q).all()
            or not torch.isfinite(torch.as_tensor(alpha)).all() or float(alpha) < 0):
        raise ValueError('COUPLING_LOGITS')
    p = logits.softmax(-1)
    value = p * ((p*q).sum(-1, keepdim=True)-q)
    logp = logits.log_softmax(-1)
    entropy = alpha*p*(logp-(p*logp).sum(-1, keepdim=True))
    return dict(value=value, entropy=entropy, total=value+entropy)


def decompose(logits, q, alpha, jacobian, indices):
    """Indices select the actual batch with replacement; all other rows are probes."""
    n = len(logits)
    if (jacobian.ndim != 3 or jacobian.shape[:2] != logits.shape
            or not torch.isfinite(jacobian).all() or indices.ndim != 1
            or indices.dtype != torch.long or not len(indices)
            or int(indices.min()) < 0 or int(indices.max()) >= n):
        raise ValueError('COUPLING_JACOBIAN_OR_BATCH')
    weight = torch.bincount(indices, minlength=n).to(logits)/len(indices)
    jm = jacobian[:, 1:]-jacobian[:, :1]
    out = {}
    for name, gz in logit_terms(logits, q, alpha).items():
        per_row = torch.einsum('nap,na->np', jacobian, gz)
        batch = torch.einsum('n,np->p', weight, per_row)
        own = -torch.einsum('ndp,np->nd', jm, per_row)
        total = -torch.einsum('ndp,p->nd', jm, batch)
        self_part = own*weight[:, None]
        out[name] = dict(local_logit=-(gz[:, 1:]-gz[:, :1]), self_alone=own,
                         batch_self=self_part, cross=total-self_part, total=total,
                         per_row_gradient=per_row, batch_gradient=batch)
    return out, weight, jm


def first_adamw_delta(theta, gradient, lr, weight_decay, eps):
    """Fresh moments only: bias-corrected first AdamW step, before FP rounding."""
    if (theta.shape != gradient.shape or not torch.isfinite(theta).all()
            or not torch.isfinite(gradient).all() or lr <= 0
            or weight_decay < 0 or eps <= 0):
        raise ValueError('COUPLING_ADAMW')
    adaptive = -lr*gradient/(gradient.abs()+eps)
    decay = -lr*weight_decay*theta
    return adaptive, decay


def classify_pressure(native, q, projections, mask):
    """Counts are direction/observation comparisons, never trade counts."""
    good = native[:, 1:]-native[:, :1] > 1e-8
    qgood = q[:, 1:]-q[:, :1] > 1e-8
    bad = native[:, 1:]-native[:, :1] < -1e-8
    take = mask[:, None].expand_as(good)
    both = take & good & qgood
    return dict(rows=int(mask.sum()), beneficial=int((take & good).sum()),
        harmful=int((take & bad).sum()), native_and_q_beneficial=int(both.sum()),
        stages={name:dict(suppresses_helpful=int((take & good & (v < -1e-10)).sum()),
            suppresses_native_and_q_helpful=int((both & (v < -1e-10)).sum()),
            favors_harmful=int((take & bad & (v > 1e-10)).sum()),
            favors_helpful=int((take & good & (v > 1e-10)).sum()))
            for name, v in projections.items()})
