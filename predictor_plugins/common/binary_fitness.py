"""Binary classification fitness computation for optimizer and DON evaluator.

F1-only fitness with optimal threshold search:
    1. Sweep thresholds [0.1 .. 0.9] on validation to find best F1
    2. Apply that threshold to train/val/test
    fitness = 0.4 * train_F1 + 0.6 * val_F1 - overfitting_penalty

    F1 is the sole optimisation target (handles class imbalance).

Higher fitness is better (positive = good model).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    brier_score_loss,
)


def _safe_auc(y_true, y_prob):
    """Compute AUC-ROC, returning 0.5 on constant class or failure."""
    try:
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def _safe_f1(y_true, y_prob, threshold=0.5):
    """Compute F1, returning 0.0 on failure."""
    try:
        y_true = np.asarray(y_true).flatten().astype(int)
        y_hat = (np.asarray(y_prob).flatten() >= threshold).astype(int)
        return float(f1_score(y_true, y_hat, zero_division=0))
    except Exception:
        return 0.0


def find_best_threshold(y_true, y_prob, thresholds=None):
    """Sweep thresholds to find the one that maximises F1 on the given split.

    Parameters
    ----------
    y_true : array-like — ground-truth binary labels (0/1)
    y_prob : array-like — predicted probabilities
    thresholds : list[float] | None — candidates to try (default: 0.10 .. 0.90)

    Returns
    -------
    float : best threshold (maximising F1)
    """
    if thresholds is None:
        thresholds = [round(t * 0.05 + 0.10, 2) for t in range(17)]  # 0.10 .. 0.90
    y_true = np.asarray(y_true).flatten().astype(int)
    y_prob = np.asarray(y_prob).flatten()
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f = _safe_f1(y_true, y_prob, t)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t


def compute_binary_metrics_for_split(y_true, y_prob, threshold=0.5):
    """Compute full binary classification metrics for a single split.

    Parameters
    ----------
    y_true : array-like, shape (N,)  — ground-truth binary labels (0/1)
    y_prob : array-like, shape (N,)  — predicted probabilities

    Returns
    -------
    dict with keys: auc_roc, f1, accuracy, precision, recall, mcc, brier, pos_rate_true, pos_rate_pred
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_prob = np.asarray(y_prob, dtype=np.float32).flatten()
    n = min(len(y_true), len(y_prob))
    y_true = y_true[:n]
    y_prob = y_prob[:n]
    y_int = y_true.astype(int)
    y_hat = (y_prob >= threshold).astype(int)

    auc = _safe_auc(y_int, y_prob)
    f1 = _safe_f1(y_int, y_prob, threshold)

    try:
        acc = float(accuracy_score(y_int, y_hat))
    except Exception:
        acc = 0.0
    try:
        prec = float(precision_score(y_int, y_hat, zero_division=0))
    except Exception:
        prec = 0.0
    try:
        rec = float(recall_score(y_int, y_hat, zero_division=0))
    except Exception:
        rec = 0.0
    try:
        mcc = float(matthews_corrcoef(y_int, y_hat))
    except Exception:
        mcc = 0.0
    try:
        brier = float(brier_score_loss(y_int, np.clip(y_prob, 1e-7, 1 - 1e-7)))
    except Exception:
        brier = 1.0

    return {
        "auc_roc": auc,
        "f1": f1,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "mcc": mcc,
        "brier": brier,
        "pos_rate_true": float(np.mean(y_int)),
        "pos_rate_pred": float(np.mean(y_hat)),
    }


def _composite_score(metrics):
    """Return F1 score from a metrics dict. Range [0, 1]."""
    return metrics.get("f1", 0.0)


def compute_binary_fitness(train_metrics, val_metrics):
    """F1-based binary fitness (higher is better).

    Parameters
    ----------
    train_metrics : dict from compute_binary_metrics_for_split
    val_metrics   : dict from compute_binary_metrics_for_split

    Returns
    -------
    float : fitness value (higher is better, positive = good model)
    """
    train_f1 = train_metrics.get("f1", 0.0)
    val_f1 = val_metrics.get("f1", 0.0)

    if not np.isfinite(train_f1) or not np.isfinite(val_f1):
        return float("-inf")

    # Base: weighted F1 (higher F1 = higher fitness)
    fitness = 0.4 * train_f1 + 0.6 * val_f1

    # Penalty: overfitting (train F1 >> val F1)
    overfit = train_f1 - val_f1
    if overfit > 0.05:
        fitness -= overfit * 2.0

    return fitness


def compute_binary_val_only_fitness(val_metrics):
    """Val-only F1 fitness for DON evaluator (no training data available).

    Returns F1 when available, else 0.
    """
    f1 = val_metrics.get("f1", 0.0)
    if np.isfinite(f1):
        return f1
    return 0.0
