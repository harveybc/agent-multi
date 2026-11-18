"""Exact optimizer-state interventions for bounded full-SAC diagnostics."""
from copy import deepcopy

import numpy as np
import torch

from app.training_forensics import rng_state, restore_rng, tree_digest


ROLES = ('actor', 'critic', 'temperature')


def role(name):
    if name.startswith('actor_network_params.'):
        return 'actor'
    if name.startswith('qvalue_network_params.'):
        return 'critic'
    if name == 'log_alpha':
        return 'temperature'
    raise ValueError('UNCLASSIFIED_SAC_PARAMETER:' + name)


def split_optimizer(model, factors):
    if set(factors) != set(ROLES) or any(not np.isfinite(v) or v <= 0 for v in factors.values()):
        raise ValueError('ROLE_LR_FACTORS')
    old = model.optimizer
    if len(old.param_groups) != 1:
        raise ValueError('EXPECTED_ORIGINAL_SINGLE_OPTIMIZER_GROUP')
    group = old.param_groups[0]
    names = {id(p): n for n, p in model.loss.named_parameters()}
    groups = []
    receipt = []
    for r in ROLES:
        params = [p for p in group['params'] if role(names[id(p)]) == r]
        if not params:
            raise ValueError('EMPTY_SAC_ROLE:' + r)
        options = {k: deepcopy(v) for k, v in group.items() if k != 'params'}
        options.update(params=params, lr=group['lr'] * factors[r])
        groups.append(options)
        receipt.append(dict(role=r, lr=options['lr'], names=[names[id(p)] for p in params]))
    optimizer = torch.optim.AdamW(groups)
    for p, state in old.state.items():
        optimizer.state[p] = deepcopy(state)
    model.optimizer = optimizer
    return receipt


def snapshot(model, sampler, step, identity):
    return deepcopy(dict(identity=identity, step=step, loss=model.loss.state_dict(),
        optimizer=model.optimizer.state_dict(), rng=rng_state(),
        sampler=sampler.bit_generator.state,
        trainability={n: p.requires_grad for n, p in model.loss.named_parameters()},
        modes={n: m.training for n, m in model.loss.named_modules()}))


def restore(model, sampler, state, identity):
    if state['identity'] != identity:
        raise ValueError('RELIABILITY_RESTART_IDENTITY')
    model.loss.load_state_dict(state['loss'])
    model.optimizer.load_state_dict(state['optimizer'])
    params = dict(model.loss.named_parameters())
    if set(params) != set(state['trainability']):
        raise ValueError('RELIABILITY_TRAINABILITY')
    for name, param in params.items():
        param.requires_grad_(state['trainability'][name])
    modules = dict(model.loss.named_modules())
    if set(modules) != set(state['modes']):
        raise ValueError('RELIABILITY_MODES')
    for name, module in modules.items():
        module.training = state['modes'][name]
    sampler.bit_generator.state = deepcopy(state['sampler'])
    restore_rng(state['rng'])


def signatures(state):
    return {k: tree_digest(state[k]) for k in
            ('loss', 'optimizer', 'rng', 'sampler', 'trainability', 'modes')}


def module_key(name):
    r = role(name)
    if '.encoder.post_fusion.' in name:
        return r + '.attention'
    if '.encoder.branches.' in name:
        return r + '.branch.' + name.split('.encoder.branches.')[1].split('.')[0]
    for component in ('fusion', 'core', 'attention', 'head'):
        if '.' + component + '.' in name:
            return r + '.' + component
    return r + '.other'


def weight_diagnostics(model, initial, features, initial_actions):
    buckets = {}
    with torch.no_grad():
        for name, p in model.loss.named_parameters():
            key = module_key(name)
            row = buckets.setdefault(key, dict(weight2=0., delta2=0., initial2=0., gradient2=0.))
            row['weight2'] += float(p.square().sum())
            row['initial2'] += float(initial[name].square().sum())
            row['delta2'] += float((p - initial[name]).square().sum())
            if p.grad is not None:
                row['gradient2'] += float(p.grad.square().sum())
        logits = model.surfaces(features)
        q = model.q_surfaces(features)
        prob = logits.softmax(-1)
        entropy = -(prob * prob.clamp_min(1e-30).log()).sum(-1)
    for row in buckets.values():
        row['relative_drift'] = (row['delta2'] / max(row['initial2'], 1e-30)) ** .5
    return dict(modules=buckets, action_flips=int((logits.argmax(-1) != initial_actions).sum()),
        q_abs_mean=float(q.abs().mean()), twin_gap_mean=float((q[0]-q[1]).abs().mean()),
        entropy_mean=float(entropy.mean()), alpha=float(model.loss.log_alpha.detach().exp()),
        logits_sha256=tree_digest(logits), q_sha256=tree_digest(q))
