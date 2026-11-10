"""Cost-weighted decision objectives for externally queried market-only fields."""
import numpy as np
import torch
from torch.nn import functional as F

EPS=1e-8


def fit_loss_constants(target,banks):
    y=np.asarray(target,float)
    if not banks or y.ndim!=1 or not np.isfinite(y).all():raise ValueError('DECISION_LOSS_FIT')
    sign=[];pair=[]
    for bank in banks:
        v=y[bank]; active=np.abs(v)>EPS
        sign.append(np.mean(np.abs(v)*active))
        opposite=(v[:,None]*v[None,:]<0) & active[:,None] & active[None,:]
        pair.append(np.mean(np.abs(v[:,None]-v[None,:])*opposite))
    c=dict(sign=float(np.mean(sign)),pair=float(np.mean(pair)))
    if not all(np.isfinite(v) and v>EPS for v in c.values()):raise ValueError('DECISION_LOSS_NO_SIGNAL')
    return c


def partner_schedule(schedule,banks,seed):
    a=np.asarray(schedule); owner={}
    for bank in banks:
        for i in bank:
            if i in owner:raise ValueError('DECISION_PAIR_DUPLICATE')
            owner[i]=bank
    if a.ndim!=2 or not np.issubdtype(a.dtype,np.integer) or any(i not in owner for i in np.unique(a)):
        raise ValueError('DECISION_PAIR_SUPPORT')
    rng=np.random.default_rng(seed+100000)
    result=np.empty_like(a)
    for i in np.unique(a):
        mask=a==i; result[mask]=rng.choice(owner[i],size=int(mask.sum()))
    return result


def decision_loss(scores,target,constants,objective,partner_scores=None,partner_target=None):
    if scores.ndim!=2 or scores.shape[0]!=2 or target.shape!=(scores.shape[1],):
        raise ValueError('DECISION_LOSS_SHAPE')
    active=target.abs()>EPS
    loss=(target.abs()*active*F.softplus(-target.sign()*scores)).mean()/constants['sign']
    if objective=='sign':return loss
    if objective!='sign_pair' or partner_scores is None or partner_target is None:
        raise ValueError('DECISION_LOSS_OBJECTIVE')
    if partner_scores.shape!=scores.shape or partner_target.shape!=target.shape:
        raise ValueError('DECISION_PAIR_SHAPE')
    delta=target-partner_target
    opposite=(target*partner_target<0) & active & (partner_target.abs()>EPS)
    pair=(delta.abs()*opposite*F.softplus(-delta.sign()*(scores-partner_scores))).mean()/constants['pair']
    return .5*loss+.5*pair
