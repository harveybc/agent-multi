"""TRAIN-visitation weighting and explicit mean versus robust critic targets."""
from collections import Counter

import numpy as np
import torch.nn.functional as F

from app.action_ranking_contrast import target_loss
from app.critic_representation_control import state_kind


def stratum(record):
    return f"{record['year']}:{state_kind(record)}"


def visitation_probabilities(records,fit,counts):
    if not len(fit) or len(set(fit))!=len(fit):raise ValueError('TARGET_FIT')
    if any(records[i]['role']!='fit' or records[i]['year'] not in (2020,2021,2022) for i in fit):
        raise ValueError('TARGET_FIT_YEAR')
    if any(k.split(':')[0] not in ('2020','2021','2022') or type(v) is not int or v<0 for k,v in counts.items()):
        raise ValueError('TARGET_VISITATION_COUNTS')
    sampled=Counter(stratum(records[i]) for i in fit)
    if (any(v>0 and k not in sampled for k,v in counts.items())
            or any(counts.get(k,0)<v for k,v in sampled.items())):
        raise ValueError('TARGET_STRATUM_COVERAGE')
    p=np.array([counts[stratum(records[i])]/sampled[stratum(records[i])] for i in fit],float)
    if not np.isfinite(p).all() or (p<=0).any():raise ValueError('TARGET_PROBABILITY')
    return p/p.sum()


def contract_loss(q,y,kind):
    if kind=='smooth_l1':return target_loss(q,y)
    if kind!='mse':raise ValueError('TARGET_LOSS')
    # Reuse the established shape/finiteness checks before squared error.
    target_loss(q,y)
    return F.mse_loss(q,y.unsqueeze(0).expand_as(q))
