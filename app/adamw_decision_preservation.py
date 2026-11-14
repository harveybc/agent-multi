"""Transactional backtracking of the actual AdamW parameter displacement."""
from copy import deepcopy
import math

import numpy as np
import torch

from app.native_head_warm_start import fit_step


def protected_step(head,optimizer,z,y,evaluate,limit,tolerance=1e-8,scales=None):
    scales=list(scales if scales is not None else [2.**(-i) for i in range(8)])
    if (not math.isfinite(limit) or limit<0 or not math.isfinite(tolerance) or tolerance<0
            or not scales or scales[0]!=1 or any(not math.isfinite(v) or v<=0 for v in scales)
            or any(a<=b for a,b in zip(scales,scales[1:]))):raise ValueError('PRESERVE_CONTRACT')
    def checked():
        result=evaluate()
        if not math.isfinite(result['regret']) or result['regret']<0:raise ValueError('PRESERVE_NONFINITE')
        actions=np.asarray(result['actions'])
        if actions.ndim!=1 or not len(actions) or actions.dtype.kind not in 'iu' or ((actions<0)|(actions>2)).any():
            raise ValueError('PRESERVE_ACTIONS')
        return dict(regret=float(result['regret']),actions=actions.astype(np.uint8,copy=True))
    initial=checked()
    if initial['regret']>limit+tolerance:raise ValueError('PRESERVE_START')
    before=deepcopy(head.state_dict());old_opt=deepcopy(optimizer.state_dict())
    trials=[];proposal_actions=None;rounded=False;clipped=False
    try:
        clipped=fit_step(head,optimizer,z,y)
        proposed=deepcopy(head.state_dict());delta={k:proposed[k]-before[k] for k in before}
        for alpha in scales:
            if alpha!=1:
                head.load_state_dict({k:before[k]+alpha*delta[k] for k in before})
            motion=torch.cat([(v-before[k]).reshape(-1) for k,v in head.state_dict().items()]).double().norm()
            norm=float(motion)
            if not math.isfinite(norm):raise ValueError('PRESERVE_STEP_NONFINITE')
            if norm==0:rounded=True;break
            observed=checked();trials.append(observed['regret'])
            if proposal_actions is None:proposal_actions=observed['actions']
            if observed['regret']<=limit+tolerance:
                return dict(accepted=True,scale=alpha,delta_norm=norm,trial_regrets=trials,
                    before_regret=initial['regret'],retained_regret=observed['regret'],actions=observed['actions'],
                    proposed_actions=proposal_actions,rounded_zero=rounded,clipped=clipped)
        head.load_state_dict(before);optimizer.load_state_dict(old_opt)
        return dict(accepted=False,scale=0.,delta_norm=0.,trial_regrets=trials,
            before_regret=initial['regret'],retained_regret=initial['regret'],actions=initial['actions'],
            proposed_actions=initial['actions'] if proposal_actions is None else proposal_actions,
            rounded_zero=rounded,clipped=clipped)
    except BaseException:
        head.load_state_dict(before);optimizer.load_state_dict(old_opt);raise
