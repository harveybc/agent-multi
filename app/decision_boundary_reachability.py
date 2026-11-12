"""Local necessary bounds and projected witnesses, never a policy certificate."""
from contextlib import contextmanager

import numpy as np
import torch

from app.fit_decision_preservation import project_step


def target_problem(logits, jacobian, target, fit_count):
    z = np.asarray(logits, dtype=np.float64)
    j = np.asarray(jacobian, dtype=np.float64)
    if (z.shape != (3,) or j.ndim != 2 or j.shape[0] != 3
            or target not in (0, 1, 2) or fit_count < 1
            or not np.isfinite(z).all() or not np.isfinite(j).all()):
        raise ValueError('BOUNDARY_INPUT')
    winner = int(z.argmax())
    if winner == target:
        raise ValueError('BOUNDARY_TARGET_ALREADY_CHOSEN')
    others = [k for k in range(3) if k != target]
    matrix = j[target] - j[others]
    cushion = 8*np.finfo(np.float32).eps*(1+abs(z[target])+np.abs(z[others]))
    lower = z[others]-z[target]+cushion
    pair = others.index(winner)
    g = matrix[pair]
    norm = float(np.linalg.norm(g))
    gap = float(z[winner]-z[target])
    # The margins are (BUY-NOP, SELL-NOP). A non-NOP pair has norm sqrt(2).
    pair_norm = 1. if 0 in (winner, target) else np.sqrt(2.)
    bounds = dict(winner=winner, gap=gap, head_linear_distance_lower_bound=
                  None if norm == 0 else gap/norm,
                  one_step_fit_margin_rms_lower_bound=gap/(pair_norm*np.sqrt(2*fit_count)),
                  zero_pair_sensitivity=norm == 0)
    direction = np.zeros(j.shape[1]) if norm == 0 else g*(lower[pair]/norm**2)
    return direction, matrix, lower, bounds


def witness(direction, matrix, lower, guard_matrix=None, guard_lower=None):
    if guard_matrix is not None:
        matrix = np.concatenate((matrix, guard_matrix), axis=0)
        lower = np.concatenate((lower, guard_lower), axis=0)
    step, report = project_step(direction, matrix, lower)
    # Explicit absolute slack supplements the solver's normalized tolerance.
    slack = float(np.min(matrix@step-lower))
    report.update(absolute_minimum_slack=slack,
                  linear_witness=bool(report['success'] and slack >= -1e-8),
                  nonlinear_verified=False)
    return step, report


@contextmanager
def temporary_head_step(parameters, step):
    delta = torch.as_tensor(step, dtype=torch.double)
    if delta.ndim != 1 or len(delta) != sum(p.numel() for p in parameters) or not torch.isfinite(delta).all():
        raise ValueError('BOUNDARY_TEMPORARY_STEP')
    saved = [p.detach().clone() for p in parameters]
    try:
        offset = 0
        with torch.no_grad():
            for p, original in zip(parameters, saved):
                part = delta[offset:offset+p.numel()].reshape(p.shape).to(p.device)
                p.copy_((original.double()+part).to(p))
                offset += p.numel()
        yield
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, saved):
                p.copy_(original)
