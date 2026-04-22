"""
Metrics computation and aggregation for the STL pipeline (use_returns=False).

Provides:
- compute_train_val_metrics
- compute_test_metrics
- aggregate_and_save_results
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .stl_norm import denormalize, denormalize_returns


def compute_train_val_metrics(
    metrics_results: Dict,
    predicted_horizons: List[int],
    list_train_preds: List[np.ndarray],
    list_train_unc: List[np.ndarray],
    list_val_preds: List[np.ndarray],
    list_val_unc: List[np.ndarray],
    y_train_list: List[np.ndarray],
    y_val_list: List[np.ndarray],
    baseline_train: Optional[np.ndarray],
    baseline_val: Optional[np.ndarray],
    metric_names: List[str],
    params: Dict,
) -> None:
    num_outputs = len(predicted_horizons)
    can_calc_train_stats = all(len(lst) == num_outputs for lst in [list_train_preds, list_train_unc])
    if not can_calc_train_stats:
        print("WARN: Skipping Train/Val stats calculation.")
        return

    print("Calculating Train/Validation metrics (all horizons)...")
    for idx, h in enumerate(predicted_horizons):
        try:
            train_preds_h = list_train_preds[idx].flatten()
            train_target_h = y_train_list[idx].flatten()
            train_unc_h = list_train_unc[idx].flatten()

            val_preds_h = list_val_preds[idx].flatten()
            val_target_h = y_val_list[idx].flatten()
            val_unc_h = list_val_unc[idx].flatten()

            # Keep baseline length in sync even if not used for metrics
            num_train_pts = min(
                len(train_preds_h), len(train_target_h), len(baseline_train) if baseline_train is not None else len(train_preds_h)
            )
            num_val_pts = min(
                len(val_preds_h), len(val_target_h), len(baseline_val) if baseline_val is not None else len(val_preds_h)
            )

            train_preds_h = train_preds_h[:num_train_pts]
            train_target_h = train_target_h[:num_train_pts]
            train_unc_h = train_unc_h[:num_train_pts]

            val_preds_h = val_preds_h[:num_val_pts]
            val_target_h = val_target_h[:num_val_pts]
            val_unc_h = val_unc_h[:num_val_pts]

            # Denormalize prices
            train_target_price = denormalize(train_target_h, params)
            train_pred_price = denormalize(train_preds_h, params)
            val_target_price = denormalize(val_target_h, params)
            val_pred_price = denormalize(val_preds_h, params)

            # Metrics (MAE calculated on real prices)
            train_mae_h = np.mean(np.abs(train_pred_price - train_target_price))
            train_r2_h = r2_score(train_target_price, train_pred_price)
            train_unc_mean_h = np.mean(np.abs(denormalize_returns(train_unc_h, params)))
            train_snr_h = np.mean(train_pred_price) / (train_unc_mean_h + 1e-9)

            # Naive MAE: baseline vs target price (both in real-world scale)
            train_naive_mae_h = np.nan
            if baseline_train is not None:
                baseline_train_h = baseline_train[:num_train_pts].flatten()
                train_naive_mae_h = np.mean(np.abs(denormalize(baseline_train_h, params) - train_target_price))

            val_mae_h = np.mean(np.abs(val_pred_price - val_target_price))
            val_r2_h = r2_score(val_target_price, val_pred_price)
            val_unc_mean_h = np.mean(np.abs(denormalize_returns(val_unc_h, params)))
            val_snr_h = np.mean(val_pred_price) / (val_unc_mean_h + 1e-9)
            val_naive_mae_h = np.nan
            if baseline_val is not None:
                baseline_val_h = baseline_val[:num_val_pts].flatten()
                val_naive_mae_h = np.mean(np.abs(denormalize(baseline_val_h, params) - val_target_price))

            metrics_results["Train"]["MAE"][h].append(train_mae_h)
            metrics_results["Train"]["Naive MAE"][h].append(train_naive_mae_h)
            metrics_results["Train"]["R2"][h].append(train_r2_h)
            metrics_results["Train"]["Uncertainty"][h].append(train_unc_mean_h)
            metrics_results["Train"]["SNR"][h].append(train_snr_h)

            metrics_results["Validation"]["MAE"][h].append(val_mae_h)
            metrics_results["Validation"]["Naive MAE"][h].append(val_naive_mae_h)
            metrics_results["Validation"]["R2"][h].append(val_r2_h)
            metrics_results["Validation"]["Uncertainty"][h].append(val_unc_mean_h)
            metrics_results["Validation"]["SNR"][h].append(val_snr_h)
        except Exception as e:
            print(f"WARN: Error Train/Val metrics H={h}: {e}")
            for ds in ["Train", "Validation"]:
                for m in metric_names:
                    metrics_results[ds][m][h].append(np.nan)


def compute_test_metrics(
    metrics_results: Dict,
    predicted_horizons: List[int],
    list_test_preds: List[np.ndarray],
    list_test_unc: List[np.ndarray],
    y_test_list: List[np.ndarray],
    baseline_test: Optional[np.ndarray],
    metric_names: List[str],
    params: Dict,
) -> None:
    num_outputs = len(predicted_horizons)
    if not all(len(lst) == num_outputs for lst in [list_test_preds, list_test_unc]):
        raise ValueError("Ioin predict mismatch outputs.")

    for idx, h in enumerate(predicted_horizons):
        try:
            test_preds_h = list_test_preds[idx].flatten()
            test_target_h = y_test_list[idx].flatten()
            test_unc_h = list_test_unc[idx].flatten()

            num_test_pts = min(
                len(test_preds_h),
                len(test_target_h),
                len(baseline_test) if baseline_test is not None else len(test_preds_h),
            )

            test_preds_h = test_preds_h[:num_test_pts]
            test_target_h = test_target_h[:num_test_pts]
            test_unc_h = test_unc_h[:num_test_pts]

            test_target_price = denormalize(test_target_h, params)
            test_pred_price = denormalize(test_preds_h, params)

            test_mae_h = np.mean(np.abs(denormalize_returns(test_preds_h - test_target_h, params)))
            test_r2_h = r2_score(test_target_price, test_pred_price)
            test_unc_mean_h = np.mean(np.abs(denormalize_returns(test_unc_h, params)))
            test_snr_h = np.mean(test_pred_price) / (test_unc_mean_h + 1e-9)

            test_naive_mae_h = np.nan
            if baseline_test is not None:
                baseline_test_h = baseline_test[:num_test_pts].flatten()
                test_naive_mae_h = np.mean(np.abs(denormalize(baseline_test_h, params) - test_target_price))

            metrics_results["Test"]["MAE"][h].append(test_mae_h)
            metrics_results["Test"]["Naive MAE"][h].append(test_naive_mae_h)
            metrics_results["Test"]["R2"][h].append(test_r2_h)
            metrics_results["Test"]["Uncertainty"][h].append(test_unc_mean_h)
            metrics_results["Test"]["SNR"][h].append(test_snr_h)
        except Exception as e:
            print(f"WARN: Error Test metrics H={h}: {e}")
            for m in metric_names:
                metrics_results["Test"][m][h].append(np.nan)


def aggregate_and_save_results(metrics_results: Dict, predicted_horizons: List[int], results_file: str) -> None:
    print("\n--- Aggregating Results Across Iterations (All Horizons) ---")
    metric_names = ["MAE", "Naive MAE", "R2", "Uncertainty", "SNR"]
    data_sets = ["Train", "Validation", "Test"]

    results_list = []
    for ds in data_sets:
        for mn in metric_names:
            for h in predicted_horizons:
                values = metrics_results[ds][mn][h]
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    results_list.append(
                        {
                            "Metric": f"{ds} {mn} H{h}",
                            "Average": np.mean(valid_values),
                            "Std Dev": np.std(valid_values),
                            "Min": np.min(valid_values),
                            "Max": np.max(valid_values),
                        }
                    )
                else:
                    results_list.append(
                        {
                            "Metric": f"{ds} {mn} H{h}",
                            "Average": np.nan,
                            "Std Dev": np.nan,
                            "Min": np.nan,
                            "Max": np.nan,
                        }
                    )
    results_df = pd.DataFrame(results_list)
    try:
        results_df.to_csv(results_file, index=False, float_format="%.6f")
        print(f"Aggregated results saved: {results_file}")
        print(results_df.to_string())
    except Exception as e:
        print(f"ERROR saving results: {e}")
