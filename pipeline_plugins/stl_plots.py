"""
Plotting helpers for the STL pipeline.

Provides:
- plot_and_save_loss
- plot_predictions
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from .stl_norm import denormalize, denormalize_returns


def plot_and_save_loss(history, loss_plot_file: str, iteration: int) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title(f"Loss-Iter {iteration}")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.savefig(loss_plot_file)
    plt.close()
    print(f"Loss plot saved: {loss_plot_file}")


def plot_predictions(
    predicted_horizons: List[int],
    plotted_horizon: int,
    list_test_preds: List[np.ndarray],
    list_test_unc: List[np.ndarray],
    y_test_list: List[np.ndarray],
    final_dates: List,
    num_test_points: int,
    final_baseline: Optional[np.ndarray],
    predictions_plot_file: str,
    params: Dict,
) -> None:
    print(f"\nGenerating prediction plot for H={plotted_horizon}...")
    try:
        plotted_index = predicted_horizons.index(plotted_horizon)
        preds_plot_raw = list_test_preds[plotted_index][:num_test_points]
        target_plot_raw = y_test_list[plotted_index][:num_test_points]
        unc_plot_raw = list_test_unc[plotted_index][:num_test_points]

        # Denormalize and flatten
        pred_plot_price_flat = denormalize(preds_plot_raw, params).flatten()
        target_plot_price_flat = denormalize(target_plot_raw, params).flatten()
        unc_plot_denorm_flat = denormalize_returns(unc_plot_raw, params).flatten()
        true_plot_price_flat = (
            denormalize(final_baseline, params).flatten() if final_baseline is not None else None
        )

        # Determine slice for last N points
        n_plot = params.get("plot_points", 480)
        num_avail_plot = len(pred_plot_price_flat)
        plot_slice = slice(max(0, num_avail_plot - n_plot), num_avail_plot)

        dates_plot_final = final_dates[plot_slice]
        pred_plot_final = pred_plot_price_flat[plot_slice]
        target_plot_final = target_plot_price_flat[plot_slice]
        unc_plot_final = unc_plot_denorm_flat[plot_slice]
        true_plot_final = (
            true_plot_price_flat[plot_slice] if true_plot_price_flat is not None else None
        )

        plt.figure(figsize=(14, 7))
        plt.plot(
            dates_plot_final,
            pred_plot_final,
            label=f"Pred Price H{plotted_horizon}",
            color=params.get("plot_color_predicted", "red"),
            lw=1.5,
            zorder=3,
        )
        plt.plot(
            dates_plot_final,
            target_plot_final,
            label=f"Target Price H{plotted_horizon}",
            color=params.get("plot_color_target", "orange"),
            lw=1.5,
            zorder=2,
        )
        if true_plot_final is not None:
            plt.plot(
                dates_plot_final,
                true_plot_final,
                label="Actual Price",
                color=params.get("plot_color_true", "blue"),
                lw=1,
                ls="--",
                alpha=0.7,
                zorder=1,
            )
        plt.fill_between(
            dates_plot_final,
            pred_plot_final - np.abs(unc_plot_final),
            pred_plot_final + np.abs(unc_plot_final),
            color=params.get("plot_color_uncertainty", "green"),
            alpha=0.2,
            label=f"Uncertainty H{plotted_horizon}",
            zorder=0,
        )
        plt.title(f"Predictions vs Target/Actual (H={plotted_horizon})")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.6)
        plt.tight_layout()
        plt.savefig(predictions_plot_file, dpi=300)
        plt.close()
        print(f"Prediction plot saved: {predictions_plot_file}")
    except Exception as e:
        print(f"ERROR generating prediction plot: {e}")
        import traceback
        traceback.print_exc()
        plt.close()
