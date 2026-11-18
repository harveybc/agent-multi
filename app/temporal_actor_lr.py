"""Name-checked actor temporal LR intervention preserving named Adam state."""
from copy import deepcopy
import math

import torch

from app.sac_training_reliability import module_key


def named_moments(model):
    names={id(p):n for n,p in model.loss.named_parameters()}
    return {names[id(p)]:deepcopy(s) for p,s in model.optimizer.state.items()}


def foreign_gpu_pids(pids, executable_for_pid, allowed_executables):
    """Only exact operator-declared desktop executables may coexist with a worker."""
    blocked=[]
    for pid in pids:
        try:
            executable=executable_for_pid(pid)
        except FileNotFoundError:
            continue  # The enumerated process exited.
        if executable not in allowed_executables:blocked.append(pid)
    return blocked


def split_temporal_optimizer(model, factor):
    if not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError('TEMPORAL_LR_FACTOR')
    old=model.optimizer
    if len(old.param_groups)!=1:
        raise ValueError('EXPECTED_ORIGINAL_SINGLE_OPTIMIZER_GROUP')
    names={id(p):n for n,p in model.loss.named_parameters()}
    original=old.param_groups[0]
    ids=[id(p) for p in original['params']]
    if len(ids)!=len(set(ids)) or set(ids)!=set(names):
        raise ValueError('TEMPORAL_PARAMETER_COVERAGE')
    temporal={'actor.attention','actor.core'}
    if not temporal.issubset({module_key(n) for n in names.values()}):
        raise ValueError('TEMPORAL_COMPONENTS_MISSING')
    groups=[]; receipt=[]
    for label, selected in [('actor_temporal',True),('unchanged',False)]:
        params=[p for p in original['params'] if (module_key(names[id(p)]) in temporal)==selected]
        if not params:raise ValueError('EMPTY_TEMPORAL_GROUP')
        options={k:deepcopy(v) for k,v in original.items() if k!='params'}
        options.update(params=params,lr=original['lr']*(factor if selected else 1.))
        groups.append(options)
        receipt.append(dict(group=label,lr=options['lr'],names=[names[id(p)] for p in params]))
    optimizer=torch.optim.AdamW(groups)
    for p,state in old.state.items():optimizer.state[p]=deepcopy(state)
    model.optimizer=optimizer
    return receipt
