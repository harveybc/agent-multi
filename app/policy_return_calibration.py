"""Frozen categorical-policy sampling and duration-aware return diagnostics."""
import numpy as np
from app.frozen_signal_masks import prepared_signals


def policy_signals(logits,seed):
    prepared=prepared_signals(logits)
    valid=prepared[3]
    raw=np.where(valid[:,None],logits,0.)
    p=np.exp(raw-raw.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    p[~valid]=np.nan
    if seed is not None:
        u=np.random.default_rng(seed).random(len(logits))
        cdf=np.cumsum(p,axis=1);cdf[valid,-1]=1.
        actions=(u[:,None]>=cdf).sum(1).astype(np.int64)
        prepared=(actions,*prepared[1:])
    return prepared,p


def discounted_returns(rewards,durations,surprisal,gamma,alpha):
    r,d,s=[np.asarray(x,dtype=float) for x in (rewards,durations,surprisal)]
    if (r.ndim!=1 or not len(r) or r.shape!=d.shape or r.shape!=s.shape or
            not all(np.isfinite(x).all() for x in (r,d,s)) or
            (d<1).any() or (d!=np.floor(d)).any() or (s<0).any() or
            not 0<=gamma<=1 or not np.isfinite(alpha) or alpha<0):
        raise ValueError('RETURN_INPUT')
    result=np.empty_like(r);result[-1]=r[-1]
    for i in range(len(r)-2,-1,-1):
        result[i]=r[i]+gamma**d[i]*(result[i+1]+alpha*s[i+1])
    return result


def calibration(predicted,observed):
    q,g=[np.asarray(x,dtype=float) for x in (predicted,observed)]
    if q.ndim!=1 or q.shape!=g.shape or not len(q) or not np.isfinite([q,g]).all():
        raise ValueError('CALIBRATION_INPUT')
    error=q-g
    bins=[]
    for ix in np.array_split(np.argsort(q,kind='stable'),min(10,len(q))):
        bins.append(dict(n=len(ix),q_mean=float(q[ix].mean()),return_mean=float(g[ix].mean()),
                         bias=float(error[ix].mean())))
    return dict(n=len(q),q_mean=float(q.mean()),return_mean=float(g.mean()),
        bias=float(error.mean()),mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.mean(error**2))),bins=bins)
