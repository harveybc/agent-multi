"""
I/O helpers for the STL pipeline.

Provides:
- save_final_outputs: save predictions/targets and uncertainties to CSVs and return plot context.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .stl_norm import denormalize, denormalize_returns
from app.data_handler import write_csv


def save_final_outputs(
    final_predictions: List[np.ndarray],
    final_uncertainties: List[np.ndarray],
    y_test_list: List[np.ndarray],
    test_dates: Optional[np.ndarray],
    baseline_test: Optional[np.ndarray],
    predicted_horizons: List[int],
    output_file: str,
    uncertainties_file: Optional[str],
    params: Dict,
) -> Tuple[List, int, Optional[np.ndarray]]:
    print("\n--- Saving Final Test Outputs (Predictions & Uncertainties Separately) ---")

    try:
        arrays_to_check_len = [final_predictions[0], baseline_test, test_dates]
        num_test_points = min(len(arr) for arr in arrays_to_check_len if arr is not None)
        print(f"Determined consistent output length: {num_test_points}")

        final_dates = list(test_dates[:num_test_points]) if test_dates is not None else list(range(num_test_points))
        final_baseline = baseline_test[:num_test_points].flatten() if baseline_test is not None else None

        output_data = {"DATE_TIME": final_dates}
        uncertainty_data = {"DATE_TIME": final_dates}

        # Add denormalized test CLOSE price for reference plot/CSV
        try:
            denorm_test_close = denormalize(final_baseline, params) if final_baseline is not None else np.full(num_test_points, np.nan)
        except Exception as e:
            print(f"WARN: Error denorm test_CLOSE: {e}")
            denorm_test_close = np.full(num_test_points, np.nan)
        output_data["test_CLOSE"] = denorm_test_close.flatten()

        # Horizon-wise export
        for idx, h in enumerate(predicted_horizons):
            preds_raw = final_predictions[idx][:num_test_points].flatten()
            target_raw = y_test_list[idx][:num_test_points].flatten()
            unc_raw = final_uncertainties[idx][:num_test_points].flatten()

            try:
                pred_price_denorm = denormalize(preds_raw, params)
                target_price_denorm = denormalize(target_raw, params)
                unc_denorm = denormalize_returns(unc_raw, params)
            except Exception as e:
                print(f"WARN: Error denorm H={h}: {e}")
                pred_price_denorm = np.full(num_test_points, np.nan)
                target_price_denorm = np.full(num_test_points, np.nan)
                unc_denorm = np.full(num_test_points, np.nan)

            output_data[f"Target_H{h}"] = target_price_denorm
            output_data[f"Prediction_H{h}"] = pred_price_denorm
            uncertainty_data[f"Uncertainty_H{h}"] = unc_denorm

        # Save predictions/targets CSV
        try:
            print("\nChecking final lengths for Predictions DataFrame:")
            for k, v in output_data.items():
                print(f"  - {k}: {len(v)}")
            if len(set(len(v) for v in output_data.values())) > 1:
                raise ValueError("Length mismatch (Predictions).")
            output_df = pd.DataFrame(output_data)
            cols_order = ["DATE_TIME"]
            if "test_CLOSE" in output_df:
                cols_order.append("test_CLOSE")
            for h in predicted_horizons:
                cols_order.extend([f"Target_H{h}", f"Prediction_H{h}"])
            output_df = output_df.reindex(columns=[c for c in cols_order if c in output_df.columns])
            write_csv(file_path=output_file, data=output_df, include_date=False, headers=True)
            print(f"Predictions/Targets saved: {output_file} ({len(output_df)} rows)")
        except ImportError:
            print(f"WARN: write_csv not found. Skip save: {output_file}.")
        except ValueError as ve:
            print(f"ERROR creating/saving predictions CSV: {ve}")
        except Exception as e:
            print(f"ERROR saving predictions CSV: {e}")

        # Save uncertainties CSV
        if uncertainties_file:
            try:
                print("\nChecking final lengths for Uncertainty DataFrame:")
                for k, v in uncertainty_data.items():
                    print(f"  - {k}: {len(v)}")
                if len(set(len(v) for v in uncertainty_data.values())) > 1:
                    raise ValueError("Length mismatch (Uncertainty).")
                uncertainty_df = pd.DataFrame(uncertainty_data)
                cols_order = ["DATE_TIME"] + [f"Uncertainty_H{h}" for h in predicted_horizons]
                uncertainty_df = uncertainty_df.reindex(columns=[c for c in cols_order if c in uncertainty_df.columns])
                write_csv(file_path=uncertainties_file, data=uncertainty_df, include_date=False, headers=True)
                print(f"Uncertainties saved: {uncertainties_file} ({len(uncertainty_df)} rows)")
            except ImportError:
                print(f"WARN: write_csv not found. Skip save: {uncertainties_file}.")
            except ValueError as ve:
                print(f"ERROR creating/saving uncertainties CSV: {ve}")
            except Exception as e:
                print(f"ERROR saving uncertainties CSV: {e}")
        else:
            print("INFO: No 'uncertainties_file' specified.")

        return final_dates, num_test_points, final_baseline
    except Exception as e:
        print(f"ERROR during final CSV saving: {e}")
        # Best-effort fallbacks
        return list(test_dates) if test_dates is not None else [], 0, None
