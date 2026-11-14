"""Closed-form shared twin-head calibration with explicit margin constraints."""
from copy import deepcopy
import math
import numpy as np
import torch


def required_scale(views,actions,margin):
    actions=np.asarray(actions);margin=np.asarray(margin,dtype=float)
    if (actions.ndim!=1 or not len(actions) or actions.dtype.kind not in 'iu'
            or ((actions<0)|(actions>2)).any() or margin.shape!=actions.shape
            or not np.isfinite(margin).all() or (margin<=0).any() or not views):
        raise ValueError('CALIBRATION_MARGIN_CONTRACT')
    lower=0.
    for view in views:
        q=np.asarray(view,dtype=float)
        if q.shape!=(len(actions),3) or not np.isfinite(q).all():raise ValueError('CALIBRATION_VIEW')
        gaps=q[np.arange(len(q)),actions,None]-q;gaps[np.arange(len(q)),actions]=np.inf
        if (gaps<=0).any():raise ValueError('CALIBRATION_NONPOSITIVE_SOURCE_GAP')
        lower=max(lower,float(np.max(margin[:,None]/gaps)))
    if not math.isfinite(lower) or lower<=0:raise ValueError('CALIBRATION_SCALE_BOUND')
    return lower


def fit_shared_affine(q,y,min_scale,kind):
    q=np.asarray(q,dtype=float);y=np.asarray(y,dtype=float)
    if (q.ndim!=3 or q.shape[0]!=2 or q.shape[2]!=3 or not q.shape[1]
            or y.shape!=q.shape[1:] or not np.isfinite(q).all() or not np.isfinite(y).all()
            or not math.isfinite(min_scale) or min_scale<=0 or kind not in ('shift','affine')):
        raise ValueError('CALIBRATION_FIT_CONTRACT')
    x=q.ravel();target=np.broadcast_to(y,q.shape).ravel()
    xm=float(x.mean());ym=float(target.mean());var=float(np.mean((x-xm)**2))
    slope=float(np.mean((x-xm)*(target-ym))/var) if var>0 else None
    if kind=='shift':
        if min_scale>1:raise ValueError('CALIBRATION_SHIFT_INFEASIBLE')
        scale=1.
    else:
        if slope is None:raise ValueError('CALIBRATION_CONSTANT_SCORES')
        scale=max(slope,min_scale)
    offset=ym-scale*xm
    if not all(math.isfinite(v) for v in (scale,offset)):raise ValueError('CALIBRATION_NONFINITE_COEFFICIENT')
    return dict(kind=kind,scale=scale,offset=offset,min_scale=min_scale,
        unconstrained_scale=slope,bound_active=bool(kind=='affine' and slope<min_scale),
        theoretical_before_mse=float(np.mean((x-target)**2)),
        theoretical_after_mse=float(np.mean((scale*x+offset-target)**2)))


def transform_head(state,scale,offset):
    if not math.isfinite(scale) or scale<=0 or not math.isfinite(offset):raise ValueError('CALIBRATION_COEFFICIENT')
    expected={f'twins.{t}.0.{layer}.{part}' for t in range(2) for layer in (0,2) for part in ('weight','bias')}
    if set(state)!=expected:raise ValueError('CALIBRATION_HEAD_KEYS')
    out=deepcopy(state)
    for twin in range(2):
        for part in ('weight','bias'):
            key=f'twins.{twin}.0.2.{part}';value=state[key]
            adjusted=value.double()*scale+(offset if part=='bias' else 0.)
            out[key]=adjusted.to(dtype=value.dtype)
            if not torch.isfinite(out[key]).all():raise ValueError('CALIBRATION_PARAMETER_OVERFLOW')
    return out
