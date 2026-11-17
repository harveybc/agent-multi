"""Read-only SAC objective pressure and named actor component interventions."""
import math

import torch


def actor_pressure(logits, q, alpha):
    if (logits.ndim != 2 or logits.shape[-1] != 3 or q.shape != logits.shape
            or not math.isfinite(alpha) or alpha <= 0
            or not torch.isfinite(logits).all() or not torch.isfinite(q).all()):
        raise ValueError('ACTOR_PRESSURE_CONTRACT')
    logp = logits.double().log_softmax(-1)
    p = logp.exp()
    q = q.double()
    return p * (alpha * (logp - (p * logp).sum(-1, keepdim=True))
                - (q - (p * q).sum(-1, keepdim=True)))


def component(key):
    if not key.startswith('actor_network_params.'):
        return None
    for token, part in (('.branches.', 'branches'), ('.post_fusion.', 'attention'),
                        ('.fusion.', 'fusion'), ('.core.', 'core'), ('.head.', 'head')):
        if token in key:
            return part
    if '.encoder.indices_' in key:
        return 'indices'
    raise ValueError('UNCLASSIFIED_ACTOR_TENSOR:' + key)


def swap_component(current, best, part):
    if part not in ('branches', 'attention', 'fusion', 'core', 'head', 'all') or set(current) != set(best):
        raise ValueError('COMPONENT_SWAP_CONTRACT')
    result = {k: v.clone() for k, v in current.items()}
    replaced = []
    for key in current:
        if current[key].shape != best[key].shape:
            raise ValueError('COMPONENT_SWAP_SHAPE')
        kind = component(key)
        if kind is not None and (part == 'all' or kind == part):
            result[key] = best[key].clone()
            replaced.append(key)
    if not replaced:
        raise ValueError('COMPONENT_SWAP_EMPTY')
    return result, replaced
