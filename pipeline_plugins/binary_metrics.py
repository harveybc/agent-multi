"""
Metrics computation and aggregation for binary classification pipelines.

Provides:
- compute_binary_metrics  (train / val / test — single split)
- aggregate_and_save_binary_results
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

# Canonical metric list (order used everywhere)
BINARY_METRIC_NAMES = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC_ROC",
    "AUC_PR",
    "MCC",
    "Brier",
    "LogLoss",
    "Pos_Rate_True",
    "Pos_Rate_Pred",
    "Uncertainty",
]


def _safe_metric(fn, *args, default=np.nan, **kwargs):
    """Call *fn* and return default on any failure (constant-class, empty, etc.)."""
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_unc: Optional[np.ndarray],
    split_name: str,
    horizon: int,
    metrics_results: Dict,
) -> None:
    """Compute all binary metrics for one split / horizon and store in *metrics_results*.

    Parameters
    ----------
    y_true : (N,) or (N,1) float32  — ground-truth 0/1 labels
    y_prob : (N,) or (N,1) float32  — predicted probabilities
    y_unc  : (N,) or (N,1) float32 or None — MC uncertainty (std)
    split_name : "Train" | "Validation" | "Test"
    horizon : int — always 1 for binary predictors
    metrics_results : nested dict[split][metric][horizon] -> list of floats
    """
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_prob = np.asarray(y_prob, dtype=np.float32).flatten()

    n = min(len(y_true), len(y_prob))
    y_true = y_true[:n]
    y_prob = y_prob[:n]

    y_hat = (y_prob >= 0.5).astype(int)
    y_int = y_true.astype(int)

    acc   = _safe_metric(accuracy_score, y_int, y_hat)
    prec  = _safe_metric(precision_score, y_int, y_hat, zero_division=0)
    rec   = _safe_metric(recall_score, y_int, y_hat, zero_division=0)
    f1    = _safe_metric(f1_score, y_int, y_hat, zero_division=0)
    auc   = _safe_metric(roc_auc_score, y_int, y_prob)
    ap    = _safe_metric(average_precision_score, y_int, y_prob)
    mcc   = _safe_metric(matthews_corrcoef, y_int, y_hat)
    brier = _safe_metric(brier_score_loss, y_int, y_prob)
    ll    = _safe_metric(log_loss, y_int, np.clip(y_prob, 1e-7, 1 - 1e-7))
    pos_true = float(np.mean(y_int))
    pos_pred = float(np.mean(y_hat))

    unc_mean = np.nan
    if y_unc is not None:
        y_unc = np.asarray(y_unc, dtype=np.float32).flatten()[:n]
        unc_mean = float(np.mean(np.abs(y_unc)))

    values = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC_ROC": auc,
        "AUC_PR": ap,
        "MCC": mcc,
        "Brier": brier,
        "LogLoss": ll,
        "Pos_Rate_True": pos_true,
        "Pos_Rate_Pred": pos_pred,
        "Uncertainty": unc_mean,
    }

    for metric, val in values.items():
        metrics_results[split_name][metric][horizon].append(val)

    # Pretty-print one-liner
    print(
        f"  {split_name} H{horizon} | "
        f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} "
        f"AUC={auc:.4f} AP={ap:.4f} MCC={mcc:.4f} Brier={brier:.4f} "
        f"Unc={unc_mean:.6f}"
    )


def aggregate_and_save_binary_results(
    metrics_results: Dict,
    predicted_horizons: List[int],
    results_file: str,
) -> None:
    """Aggregate across iterations and save to CSV (same format as regression pipeline)."""
    print("\n--- Aggregating Binary Classification Results ---")
    data_sets = ["Train", "Validation", "Test"]

    results_list = []
    for ds in data_sets:
        for mn in BINARY_METRIC_NAMES:
            for h in predicted_horizons:
                values = metrics_results[ds][mn].get(h, [])
                valid = [v for v in values if not np.isnan(v)]
                if valid:
                    results_list.append({
                        "Metric": f"{ds} {mn} H{h}",
                        "Average": np.mean(valid),
                        "Std Dev": np.std(valid),
                        "Min": np.min(valid),
                        "Max": np.max(valid),
                    })
                else:
                    results_list.append({
                        "Metric": f"{ds} {mn} H{h}",
                        "Average": np.nan,
                        "Std Dev": np.nan,
                        "Min": np.nan,
                        "Max": np.nan,
                    })

    results_df = pd.DataFrame(results_list)
    try:
        results_df.to_csv(results_file, index=False, float_format="%.6f")
        print(f"Results saved: {results_file}")
        print(results_df.to_string())
    except Exception as e:
        print(f"ERROR saving results: {e}")
