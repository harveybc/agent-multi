"""Read-only functional measurements and explicit actor component swaps."""
import math

import torch


def actor_component(name):
    if not name.startswith('actor_network_params.'):
        return None
    if '.head.' in name:
        return 'head'
    if '.encoder.' in name:
        return 'encoder'
    raise ValueError('UNCLASSIFIED_ACTOR_COMPONENT:' + name)


def swap_actor(current, initial, component):
    if component not in ('head', 'encoder') or set(current) != set(initial):
        raise ValueError('TRANSIENT_SWAP_CONTRACT')
    out = {k: v.clone() for k, v in current.items()}
    replaced = []
    for key in current:
        if current[key].shape != initial[key].shape:
            raise ValueError('TRANSIENT_SWAP_SHAPE')
        if actor_component(key) == component:
            out[key] = initial[key].clone()
            replaced.append(key)
    if not replaced:
        raise ValueError('TRANSIENT_SWAP_EMPTY')
    return out, replaced


def functional_metrics(logits, twins, baseline_logits, alpha):
    if (logits.ndim != 2 or logits.shape[-1] != 3 or twins.shape != (2, *logits.shape)
            or baseline_logits.shape != logits.shape or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError('TRANSIENT_PROBE_SHAPE_OR_ALPHA')
    if not all(torch.isfinite(v).all() for v in (logits, twins, baseline_logits)):
        raise ValueError('TRANSIENT_PROBE_NONFINITE')
    logits, twins, baseline_logits = [v.double() for v in (logits, twins, baseline_logits)]
    logp = logits.log_softmax(-1)
    p = logp.exp()
    logbase = baseline_logits.log_softmax(-1)
    q = twins.min(0).values
    level = q.mean(-1, keepdim=True)
    centered = q - level
    action = logits.argmax(-1)
    target_logp = (centered / alpha).log_softmax(-1)
    top = logits.topk(2, dim=-1).values
    qtop = q.topk(2, dim=-1).values
    return dict(alpha=float(alpha), entropy=float(-(p * logp).sum(-1).mean()),
        max_probability=float(p.max(-1).values.mean()),
        action_flips_fraction=float((action != baseline_logits.argmax(-1)).double().mean()),
        greedy_signal_share=[float((action == i).double().mean()) for i in range(3)],
        logit_top_margin=float((top[:, 0]-top[:, 1]).mean()),
        kl_to_initial=float((p * (logp-logbase)).sum(-1).mean()),
        kl_to_soft_q=float((p * (logp-target_logp)).sum(-1).mean()),
        q_disagrees_actor=float((q.argmax(-1) != action).double().mean()),
        q_common_mean=float(level.mean()), q_common_abs=float(level.abs().mean()),
        q_centered_rms=float(centered.square().mean().sqrt()),
        q_action_gap=float((qtop[:, 0]-qtop[:, 1]).mean()),
        q_action_gap_over_alpha=float((qtop[:, 0]-qtop[:, 1]).mean()/alpha),
        twin_disagreement=float((twins[0]-twins[1]).abs().mean()))


def checkpoint_drift(current, initial):
    buckets = {}
    for name, value in current.items():
        if not value.is_floating_point() or name not in initial:
            continue
        if name.startswith('actor_network_params.'):
            key = 'actor.' + actor_component(name)
        elif name.startswith('target_qvalue_network_params.'):
            key = 'target_critic'
        elif name.startswith('qvalue_network_params.'):
            key = 'critic.head' if '.head.' in name else 'critic.encoder'
        else:
            continue
        row = buckets.setdefault(key, dict(delta2=0., initial2=0., values=0))
        a, b = value.double(), initial[name].double()
        row['delta2'] += float((a-b).square().sum())
        row['initial2'] += float(b.square().sum())
        row['values'] += value.numel()
    for row in buckets.values():
        row['relative_l2'] = (row['delta2']/max(row['initial2'], 1e-30))**.5
    return buckets
