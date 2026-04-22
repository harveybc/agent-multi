"""
Plotting helpers for binary classification pipelines.

Provides:
- plot_and_save_loss       (training / val loss curves)
- plot_binary_predictions  (probability timeseries + true labels)
- plot_confusion_matrix    (confusion matrix heatmap)
- plot_roc_pr_curves       (ROC + PR curves side by side)
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def plot_and_save_loss(history, loss_plot_file: str, iteration: int) -> None:
    """Standard train/val loss curves (same as regression)."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title(f"Binary CrossEntropy Loss — Iter {iteration}")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.savefig(loss_plot_file)
    plt.close()
    print(f"Loss plot saved: {loss_plot_file}")


def plot_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    final_dates: List,
    num_test_points: int,
    predictions_plot_file: str,
    params: Dict,
) -> None:
    """Timeseries plot of predicted probabilities vs true binary labels."""
    print("\nGenerating binary prediction plot …")
    try:
        n = min(len(y_true), len(y_prob), num_test_points)
        n_plot = params.get("plot_points", 480)
        start = max(0, n - n_plot)

        y_t = y_true[start:n].flatten()
        y_p = y_prob[start:n].flatten()
        dates = final_dates[start:n]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(dates, 0, 1, where=(y_t == 1), alpha=0.15, color="green", label="True = 1")
        ax.fill_between(dates, 0, 1, where=(y_t == 0), alpha=0.08, color="red", label="True = 0")
        ax.plot(dates, y_p, color="blue", lw=1.2, label="P(y=1)")
        ax.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Binary Prediction Probabilities")
        ax.set_xlabel("Time")
        ax.set_ylabel("Probability")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        fig.savefig(predictions_plot_file, dpi=300)
        plt.close(fig)
        print(f"Binary prediction plot saved: {predictions_plot_file}")
    except Exception as e:
        print(f"ERROR generating binary prediction plot: {e}")
        import traceback; traceback.print_exc()
        plt.close("all")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cm_plot_file: str,
) -> None:
    """Confusion-matrix heatmap saved to file."""
    try:
        y_hat = (y_prob.flatten() >= 0.5).astype(int)
        y_int = y_true.flatten().astype(int)
        cm = confusion_matrix(y_int, y_hat, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=["0", "1"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix (test)")
        fig.tight_layout()
        fig.savefig(cm_plot_file, dpi=200)
        plt.close(fig)
        print(f"Confusion matrix plot saved: {cm_plot_file}")
    except Exception as e:
        print(f"ERROR generating confusion matrix plot: {e}")
        plt.close("all")


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_pr_plot_file: str,
) -> None:
    """Side-by-side ROC and Precision-Recall curves."""
    try:
        y_int = y_true.flatten().astype(int)
        y_p = y_prob.flatten()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ROC
        fpr, tpr, _ = roc_curve(y_int, y_p)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
        ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax1.set_xlabel("FPR")
        ax1.set_ylabel("TPR")
        ax1.set_title("ROC Curve")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.4)

        # Precision-Recall
        prec, rec, _ = precision_recall_curve(y_int, y_p)
        avg_prec = average_precision_score(y_int, y_p)
        ax2.plot(rec, prec, color="navy", lw=2, label=f"AP = {avg_prec:.4f}")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.4)

        fig.tight_layout()
        fig.savefig(roc_pr_plot_file, dpi=200)
        plt.close(fig)
        print(f"ROC + PR curves saved: {roc_pr_plot_file}")
    except Exception as e:
        print(f"ERROR generating ROC/PR curves: {e}")
        plt.close("all")
