"""Descriptive rank and frozen-function sensitivity, never a pruning rule."""
import math

import torch


def energy_summary(energy):
    energy = energy.double().clamp_min(0)
    total = float(energy.sum())
    if total == 0:
        return dict(energy=0., stable_rank=0., participation_rank=0., rank90=0, rank99=0)
    fraction = energy/total
    cumulative = fraction.cumsum(0)
    return dict(energy=total, stable_rank=total/float(energy.max()),
                participation_rank=1/float(fraction.square().sum()),
                rank90=int(torch.searchsorted(cumulative, .90))+1,
                rank99=int(torch.searchsorted(cumulative, .99))+1)


def spectrum(matrix):
    x = torch.as_tensor(matrix).detach().double()
    if x.ndim != 2 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError('CAPACITY_MATRIX')
    return dict(rows=x.shape[0], columns=x.shape[1],
                **energy_summary(torch.linalg.svdvals(x).square()))


def activation_spectrum(values):
    x = torch.as_tensor(values).detach().double()
    if x.ndim != 2 or x.shape[0] < 2 or not torch.isfinite(x).all():
        raise ValueError('CAPACITY_ACTIVATIONS')
    z = x-x.mean(0)
    variance = z.square().mean(0)
    covariance = z.T @ z / len(z)
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0)
    return dict(samples=len(x), channels=x.shape[1],
                constant_channels=int((variance == 0).sum()),
                channel_std=variance.sqrt().tolist(), **energy_summary(eigenvalues))


def low_rank(weight, fraction):
    if not 0 < fraction <= 1 or weight.ndim < 2 or not torch.isfinite(weight).all():
        raise ValueError('CAPACITY_LOW_RANK')
    matrix = weight.detach().double().reshape(weight.shape[0], -1)
    rank = max(1, math.ceil(min(matrix.shape)*fraction))
    if rank == min(matrix.shape):
        return weight.detach().clone()
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return ((u[:, :rank]*s[:rank]) @ vh[:rank]).reshape(weight.shape).to(weight)


def signal_change(reference, candidate):
    if (reference.shape != candidate.shape or reference.ndim != 2
            or reference.shape[1] != 3 or len(reference) == 0
            or not torch.isfinite(reference).all() or not torch.isfinite(candidate).all()):
        raise ValueError('CAPACITY_LOGITS')
    a, b = reference.double(), candidate.double()
    p = b.softmax(-1)
    changed = a.argmax(-1) != b.argmax(-1)
    margins = a.topk(2, dim=-1).values
    return dict(rows=len(a), flips=int(changed.sum()),
                flip_fraction=float(changed.double().mean()),
                mean_kl=float((p*(b.log_softmax(-1)-a.log_softmax(-1))).sum(-1).mean()),
                logit_rms_delta=float((a-b).square().mean().sqrt()),
                changed_reference_margin_median=None if not changed.any() else
                float((margins[:, 0]-margins[:, 1])[changed].median()))
