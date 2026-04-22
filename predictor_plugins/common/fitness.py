"""Shared fitness computation for optimizer and DON evaluator.

Penalized Asymmetric Delta (NDA):
    base = 0.4 * train_delta + 0.6 * val_delta
    penalty += train_delta   if train_delta > 0
    penalty += val_delta * 2 if val_delta > 0
    fitness = base + penalty

Lower fitness is better (negative = beating naive).
"""

import numpy as np


def compute_fitness(train_mae, train_naive_mae, val_mae, val_naive_mae):
    """Full penalized asymmetric delta fitness (lower is better).

    Used by the optimizer's candidate_worker during training.
    """
    if (train_naive_mae is None or val_naive_mae is None
            or not np.isfinite(train_naive_mae) or not np.isfinite(val_naive_mae)):
        return float("inf")

    train_delta = train_mae - train_naive_mae
    val_delta = val_mae - val_naive_mae
    base = 0.4 * train_delta + 0.6 * val_delta
    penalty = 0.0
    if train_delta > 0:
        penalty += train_delta
    if val_delta > 0:
        penalty += val_delta * 2
    return base + penalty


def compute_val_only_fitness(val_mae, val_naive_mae):
    """Val-only fitness for DON evaluator (no training data available).

    Returns val_mae - naive_mae when naive is available, else val_mae.
    """
    if val_naive_mae is not None and np.isfinite(val_naive_mae) and val_naive_mae > 0:
        return val_mae - val_naive_mae
    return val_mae
