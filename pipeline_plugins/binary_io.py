"""
I/O helpers for the binary classification pipeline.

Provides:
- save_binary_outputs: save predicted probabilities, hard labels, and true labels to CSV.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from app.data_handler import write_csv


def save_binary_outputs(
    final_predictions: List[np.ndarray],
    final_uncertainties: List[np.ndarray],
    y_test_list: List[np.ndarray],
    test_dates: Optional[np.ndarray],
    predicted_horizons: List[int],
    output_file: str,
    uncertainties_file: Optional[str],
    params: Dict,
) -> Tuple[List, int]:
    """Save binary predictions CSV and (optionally) uncertainties CSV.

    Predictions CSV columns:
        DATE_TIME, True_Label_H1, Probability_H1, Predicted_Label_H1

    Returns
    -------
    (final_dates, num_test_points)
    """
    print("\n--- Saving Binary Classification Test Outputs ---")

    try:
        num_test_points = min(
            len(arr) for arr in [final_predictions[0], y_test_list[0]]
            if arr is not None
        )
        if test_dates is not None:
            num_test_points = min(num_test_points, len(test_dates))

        final_dates = list(test_dates[:num_test_points]) if test_dates is not None else list(range(num_test_points))

        output_data: Dict[str, np.ndarray] = {"DATE_TIME": final_dates}
        uncertainty_data: Dict[str, np.ndarray] = {"DATE_TIME": final_dates}

        for idx, h in enumerate(predicted_horizons):
            y_true = y_test_list[idx][:num_test_points].flatten()
            y_prob = final_predictions[idx][:num_test_points].flatten()
            y_hat = (y_prob >= 0.5).astype(int)

            output_data[f"True_Label_H{h}"] = y_true.astype(int)
            output_data[f"Probability_H{h}"] = y_prob
            output_data[f"Predicted_Label_H{h}"] = y_hat

            if idx < len(final_uncertainties):
                unc = final_uncertainties[idx][:num_test_points].flatten()
                uncertainty_data[f"Uncertainty_H{h}"] = unc

        # Save predictions CSV
        try:
            output_df = pd.DataFrame(output_data)
            cols = ["DATE_TIME"]
            for h in predicted_horizons:
                cols.extend([f"True_Label_H{h}", f"Probability_H{h}", f"Predicted_Label_H{h}"])
            output_df = output_df.reindex(columns=[c for c in cols if c in output_df.columns])
            write_csv(file_path=output_file, data=output_df, include_date=False, headers=True)
            print(f"Binary predictions saved: {output_file} ({len(output_df)} rows)")
        except Exception as e:
            print(f"ERROR saving binary predictions CSV: {e}")

        # Save uncertainties CSV
        if uncertainties_file:
            try:
                unc_df = pd.DataFrame(uncertainty_data)
                cols = ["DATE_TIME"] + [f"Uncertainty_H{h}" for h in predicted_horizons]
                unc_df = unc_df.reindex(columns=[c for c in cols if c in unc_df.columns])
                write_csv(file_path=uncertainties_file, data=unc_df, include_date=False, headers=True)
                print(f"Uncertainties saved: {uncertainties_file} ({len(unc_df)} rows)")
            except Exception as e:
                print(f"ERROR saving uncertainties CSV: {e}")
        else:
            print("INFO: No 'uncertainties_file' specified.")

        return final_dates, num_test_points

    except Exception as e:
        print(f"ERROR during binary output saving: {e}")
        return list(test_dates) if test_dates is not None else [], 0
