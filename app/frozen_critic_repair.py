"""Bounded TRAIN-only selection, qualification and diagnostic actor trust step."""
from copy import deepcopy
import math

import numpy as np
import torch


def choose(trace, seed=369, quota=4):
    rng = np.random.default_rng(seed)
    keys = ('row', 'timestamp', 'year', 'kind', 'window_start', 'role', 'state_sha256', 'segment')
    rows = [{k: r[k] for k in keys if k in r} for r in trace
            if r['year'] in (2020, 2021, 2022, 2023)
            and (r['role'] != 'monitor' or r['window_start'] >= 1672531200000000000)]
    selected = []
    for group in sorted({(r['year'], r['kind']) for r in rows}):
        candidates = [r for r in rows if (r['year'], r['kind']) == group]
        selected.extend(candidates[i] for i in rng.permutation(len(candidates))[:quota])
    return sorted(selected, key=lambda r: r['row'])


def passed(candidate, source, actor, nop):
    keys = ('mean_regret', 'pairwise_accuracy', 'rmse')
    if any(x.get(k) is None or not math.isfinite(x[k]) for x in (candidate, source) for k in keys):
        return False
    return (candidate['mean_regret'] < source['mean_regret'] - 1e-8
            and candidate['mean_regret'] <= min(actor['mean_regret'], nop['mean_regret']) + 1e-8
            and candidate['pairwise_accuracy'] >= source['pairwise_accuracy']
            and candidate['rmse'] <= source['rmse'])


def mean_kl(reference, current):
    lp = reference.double().log_softmax(-1)
    return float((lp.exp() * (lp - current.double().log_softmax(-1))).sum(-1).mean().clamp_min(0))


def constrained_step(optimizer, params, loss, logits, reference, limit, max_backtracks=12):
    if limit is not None and (not math.isfinite(limit) or limit < 0):
        raise ValueError('REPAIR_KL_LIMIT')
    old = [p.detach().clone() for p in params]
    state = deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
    optimizer.step()
    proposed = [p.detach().clone() for p in params]
    with torch.no_grad():
        for attempt in range(max_backtracks + 1):
            fraction = 2. ** -attempt
            if attempt:
                for p, a, b in zip(params, old, proposed):
                    p.copy_(a + fraction * (b-a))
            value = logits()
            if torch.isfinite(value).all():
                kl = mean_kl(reference, value)
                if limit is None or kl <= limit:
                    return dict(accepted=True, fraction=fraction, kl=kl)
        for p, a in zip(params, old):
            p.copy_(a)
        optimizer.load_state_dict(state)
        return dict(accepted=False, fraction=0., kl=mean_kl(reference, logits()))
