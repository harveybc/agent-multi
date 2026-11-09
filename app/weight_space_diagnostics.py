"""Descriptive weight-space measurements and a softmax-invariant head edit."""
from __future__ import annotations

import torch


def tensor_summary(value):
    x = value.detach().cpu().double()
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError('WEIGHT_NONFINITE_OR_EMPTY')
    result = dict(shape=list(x.shape), count=x.numel(), mean=float(x.mean()),
        std=float(x.std(unbiased=False)), norm=float(x.norm()),
        max_abs=float(x.abs().max()), zero_fraction=float((x == 0).double().mean()))
    if x.ndim >= 2:
        matrix = x.reshape(x.shape[0], -1)
        singular = torch.linalg.svdvals(matrix)
        power = singular.square()
        result['spectral_norm'] = float(singular.max())
        result['stable_rank'] = float(power.sum() / power.max()) if power.max() > 0 else 0.
        result['singular_values'] = singular.tolist()
    return result


def component(name):
    role = ('target_critic' if name.startswith('target_qvalue_') else
            'critic' if name.startswith('qvalue_') else
            'actor' if name.startswith('actor_') else 'other')
    if '.branches.' in name:
        group = 'branch_' + name.split('.branches.', 1)[1].split('.')[0]
    elif '.fusion.' in name:
        group = 'fusion'
    elif '.post_fusion.' in name:
        group = 'attention'
    elif '.core.' in name:
        group = 'core_' + name.split('.core.', 1)[1].split('.')[0]
    elif '.head.' in name:
        group = 'head_' + name.split('.head.', 1)[1].split('.')[0]
    else:
        group = 'other'
    return role + '/' + group


def state_delta(before, after):
    if before.keys() != after.keys():
        raise ValueError('WEIGHT_STATE_KEYS')
    tensors, groups = {}, {}
    for name, old in before.items():
        new = after[name]
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError('WEIGHT_STATE_SHAPE')
        if not old.is_floating_point():
            if not torch.equal(old.cpu(), new.cpu()):
                raise ValueError('WEIGHT_BUFFER_CHANGED:' + name)
            continue
        a, b = old.detach().cpu().double(), new.detach().cpu().double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('WEIGHT_NONFINITE')
        delta = b - a
        row = dict(changed=not torch.equal(a, b), norm=float(b.norm()),
            delta_norm=float(delta.norm()), max_abs_delta=float(delta.abs().max()))
        tensors[name] = row
        key = component(name)
        group = groups.setdefault(key, dict(tensors=0, changed=0, delta_squared=0.))
        group['tensors'] += 1
        group['changed'] += int(row['changed'])
        group['delta_squared'] += row['delta_norm'] ** 2
    for group in groups.values():
        group['delta_norm'] = group.pop('delta_squared') ** .5
    return dict(tensors=tensors, groups=groups)


def center_categorical_head(weight, bias):
    """Remove only the common action component; never change critic Q this way."""
    if weight.ndim != 2 or bias.shape != (weight.shape[0],) or weight.shape[0] != 3:
        raise ValueError('CATEGORICAL_HEAD_SHAPE')
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError('WEIGHT_NONFINITE')
    return weight - weight.mean(0, keepdim=True), bias - bias.mean()


def transplant_actor_heads(reference, donor, layers):
    """Isolate head interventions; never borrow an encoder or critic implicitly."""
    if not set(layers) <= {0, 2}:
        raise ValueError('ACTOR_HEAD_LAYER')
    state_delta(reference, donor)
    result = {k: v.clone() for k, v in reference.items()}
    keys = []
    for layer in layers:
        prefix = f'actor_network_params.module.0.module.head.{layer}.'
        names = [prefix + suffix for suffix in ('weight', 'bias')]
        if any(k not in result for k in names):
            raise ValueError('ACTOR_HEAD_KEYS')
        for name in names:
            result[name] = donor[name].clone()
            keys.append(name)
    return result, keys
