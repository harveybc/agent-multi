"""FIT-scaled decision-cost surrogates; discrete regret remains the endpoint."""
import math
import numpy as np
import torch

from app.critic_target_contract import contract_loss


def fit_temperature(targets):
    y=np.asarray(targets,dtype=float)
    if y.ndim!=2 or y.shape[1]!=3 or not len(y) or not np.isfinite(y).all():
        raise ValueError('COST_TEMPERATURE_TARGETS')
    costs=y.max(1,keepdims=True)-y;positive=costs[costs>0]
    if not positive.size:raise ValueError('COST_NO_DECISION_CONTRAST')
    value=float(np.median(positive))
    if not math.isfinite(value) or value<=1e-12:raise ValueError('COST_TEMPERATURE_SCALE')
    return value


def objective_parts(q,y,temperature):
    if not math.isfinite(temperature) or temperature<=1e-12:raise ValueError('COST_TEMPERATURE_SCALE')
    mse=contract_loss(q,y,'mse')
    costs=y.max(-1,keepdim=True).values-y
    probabilities=torch.softmax(q.min(0).values/temperature,dim=-1)
    cost=temperature*(probabilities*costs).sum(-1).mean()
    return dict(mse=mse,cost=cost,hybrid=mse+cost)


def cost_objective(q,y,kind,temperature):
    if kind not in ('mse','cost','hybrid'):raise ValueError('COST_OBJECTIVE_KIND')
    return objective_parts(q,y,temperature)[kind]


def stable_added_mask(original,views):
    arrays=[np.asarray(v) for v in views]
    if len(arrays)<2 or any(a.ndim!=2 or a.shape!=arrays[0].shape or a.shape[1]!=3
            or not len(a) or not np.isfinite(a).all() for a in arrays):raise ValueError('COST_STABILITY_VIEWS')
    original=np.asarray(original)
    if original.ndim!=1 or original.dtype.kind not in 'iu' or (original<0).any() or (original>=len(arrays[0])).any():
        raise ValueError('COST_STABILITY_ORIGINAL')
    actions=np.stack([a.argmax(1) for a in arrays]);stable=(actions==actions[0]).all(0)
    stable[original]=False
    return stable


def qualifies_extension(rows):
    if len(rows)!=4 or {(r['fold'],r['seed']) for r in rows}!={('A',19),('A',43),('B',19),('B',43)}:
        return False
    good=0
    for r in rows:
        before,after,control=[r[k] for k in ('before','after','control')]
        if (r['robust'] is not True or r['accepted']<32 or not all(math.isfinite(v) for v in (before,after,control))
                or after>before+1e-8):return False
        good+=int(after<before-1e-8 and after<control-1e-8)
    return good>=3
