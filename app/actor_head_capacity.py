"""Isolated fresh actor-head capacity on a preserved SAC representation."""
import copy

import numpy as np
import torch

from app.structural_refit import build_model


def is_actor_head(name):
    return name.startswith('actor_') and '.head.' in name


def make_candidate(source, width, seed):
    if type(width) is not int or width not in (8, 16):
        raise ValueError('HEAD_CAPACITY_WIDTH')
    effective = copy.deepcopy(source.effective)
    effective['architecture']['actor_head_width'] = width
    model = build_model(source.recipe, effective, seed=seed, device=source.device)
    src, dst = source.loss.state_dict(), model.loss.state_dict()
    if set(src) != set(dst):
        raise ValueError('HEAD_CAPACITY_STATE_KEYS')
    with torch.no_grad():
        for key, value in src.items():
            if not is_actor_head(key):
                if value.shape != dst[key].shape:
                    raise ValueError('HEAD_CAPACITY_FROZEN_SHAPE')
                dst[key].copy_(value)
    flags = {n:p.requires_grad for n,p in source.loss.named_parameters()}
    for name, param in model.loss.named_parameters():
        param.requires_grad_(True if is_actor_head(name) else flags[name])
    # Optimize this standalone head on cached features. No gradient can reach
    # the encoder, critic or temperature, whose checkpoint tensors stay intact.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = torch.nn.Sequential(torch.nn.Linear(model.actor_net.encoder.features_dim, width),
                                   torch.nn.SiLU(), torch.nn.Linear(width, 3))
    install_head(model, head)
    verify_fixed(model, src)
    return model, head


def install_head(model, head):
    weights = head.state_dict()
    with torch.no_grad():
        found = 0
        for name, param in model.loss.named_parameters():
            if is_actor_head(name):
                key = name.split('.head.', 1)[1]
                if key not in weights or param.shape != weights[key].shape:
                    raise ValueError('HEAD_CAPACITY_LAYOUT')
                param.copy_(weights[key].to(param)); found += 1
    if found != 4:
        raise ValueError('HEAD_CAPACITY_PARAMETER_COUNT')


def verify_fixed(model, reference):
    current = model.loss.state_dict()
    if set(current) != set(reference):
        raise ValueError('HEAD_CAPACITY_STATE_KEYS')
    for name, value in reference.items():
        if not is_actor_head(name) and not torch.equal(value, current[name]):
            raise ValueError('HEAD_CAPACITY_FROZEN_CHANGED:'+name)


def full_schedule(rows, steps, batch, seed):
    if min(rows, steps, batch) < 1:
        raise ValueError('HEAD_CAPACITY_SCHEDULE')
    rng = np.random.default_rng(seed)
    needed = steps*batch
    values = np.concatenate([rng.permutation(rows) for _ in range((needed+rows-1)//rows)])[:needed]
    return values.reshape(steps, batch)
