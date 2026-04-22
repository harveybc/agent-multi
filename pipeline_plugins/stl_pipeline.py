#!/usr/bin/env python
"""
STL Pipeline Plugin (Refactored for readability and maintainability)

This implementation preserves the exact behavior and outputs for the
use_returns = False configuration. Any logic for use_returns = True has been
removed by design to keep the code focused and simpler.

Steps overview (mirrored as section headers in code):
  1) Load and prepare datasets
  2) Build and train model (per iteration)
  3) Compute Train/Validation metrics
  4) Plot and save loss curve (per iteration)
  5) Predict on Test with uncertainty and compute metrics
  6) Aggregate metrics across iterations and save results
  7) Save final Test predictions/targets and uncertainties
  8) Plot predictions vs target/actual for selected horizon
  9) Optionally export model plot and save model
 10) Load-and-evaluate (utility) — evaluation on validation set
"""

from typing import Dict, List, Tuple, Optional
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app.data_handler import write_csv

# Conditional import for plot_model
try:
    from tensorflow.keras.utils import plot_model
except ImportError:  # pragma: no cover - optional dependency
    plot_model = None

# Local helpers
from .stl_norm import denormalize, denormalize_returns
from .stl_metrics import (
    compute_train_val_metrics,
    compute_test_metrics,
    aggregate_and_save_results,
)
from .stl_plots import plot_and_save_loss, plot_predictions
from .stl_io import save_final_outputs


class STLPipelinePlugin:
    """Refactored STL pipeline focused on use_returns=False only."""

    # Default parameters (unchanged defaults where applicable)
    plugin_params = {
        "iterations": 1,
        "batch_size": 32,
        "epochs": 50,
        "threshold_error": 0.001,
        "loss_plot_file": "loss_plot.png",
        "output_file": "test_predictions.csv",
        "uncertainties_file": "test_uncertainties.csv",
        "model_plot_file": "model_plot.png",
        "predictions_plot_file": "predictions_plot.png",
        "results_file": "results.csv",
        "plot_points": 480,
        "plotted_horizon": 6,
        "use_strategy": False,
        "predicted_horizons": [1, 6, 12, 24],
        "normalize_features": True,
        "window_size": 48,
        "target_column": "TARGET",
        "use_normalization_json": None,
        "mc_samples": 100,
        # Explicitly fixed to False (no returns logic in this refactor)
        "use_returns": False,
    }

    plugin_debug_vars = [
        "iterations",
        "batch_size",
        "epochs",
        "threshold_error",
        "output_file",
        "uncertainties_file",
        "results_file",
        "plotted_horizon",
        "plot_points",
    ]

    # ---------------------------------------------------------------------
    # 0) Construction and config helpers
    # ---------------------------------------------------------------------
    def __init__(self) -> None:
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self) -> Dict:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict) -> None:
        debug_info.update(self.get_debug_info())

    # ---------------------------------------------------------------------
    # 1) Load and prepare datasets
    # ---------------------------------------------------------------------
    def _prepare_datasets(self, preprocessor_plugin, target_plugin, config: Dict) -> Dict:
        print("Loading/processing datasets via Preprocessor...")
        datasets = preprocessor_plugin.run_preprocessing(target_plugin, config)
        print("Preprocessor finished.")
        return datasets

    @staticmethod
    def _targets_to_list(y_targets, predicted_horizons: List[int]) -> List[np.ndarray]:
        """Normalize targets to a list aligned with predicted_horizons.

        Accepts either a list/tuple of arrays (already ordered), or a dict with
        keys like 'output_horizon_{h}'. Returns a list of numpy arrays.
        """
        if isinstance(y_targets, (list, tuple)):
            return list(y_targets)
        if isinstance(y_targets, dict):
            out = []
            for h in predicted_horizons:
                key = f"output_horizon_{h}"
                arr = y_targets.get(key)
                if arr is None:
                    raise KeyError(f"Missing target for horizon {h} under key '{key}'")
                out.append(np.asarray(arr))
            return out
        raise TypeError("y_targets must be a list/tuple or a dict keyed by 'output_horizon_{h}'")

    # ---------------------------------------------------------------------
    # 2) Build and train model (per iteration)
    # ---------------------------------------------------------------------
    def _build_and_train(self,
                         predictor_plugin,
                         X_train: np.ndarray,
                         y_train_dict: Dict[str, np.ndarray],
                         X_val: np.ndarray,
                         y_val_dict: Dict[str, np.ndarray],
                         config: Dict):
        input_shape = (X_train.shape[1], X_train.shape[2]) if X_train.ndim == 3 else (X_train.shape[1],)
        predictor_plugin.build_model(input_shape=input_shape, x_train=X_train, config=config)
        history, list_train_preds, list_train_unc, list_val_preds, list_val_unc = predictor_plugin.train(
            X_train,
            y_train_dict,
            epochs=config.get("epochs", 50),
            batch_size=config.get("batch_size", 32),
            threshold_error=config.get("threshold_error", 0.001),
            x_val=X_val,
            y_val=y_val_dict,
            config=config,
        )
        return history, list_train_preds, list_train_unc, list_val_preds, list_val_unc

    # ---------------------------------------------------------------------
    # 5) Predict with uncertainty helper wrapper
    # ---------------------------------------------------------------------
    def _predict_with_uncertainty(self, predictor_plugin, X_test: np.ndarray, mc_samples: int):
        return predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)

    # ---------------------------------------------------------------------
    # 9) Optionally export model plot and save model
    # ---------------------------------------------------------------------
    def _plot_model(self, predictor_plugin, config: Dict) -> None:
        if plot_model is not None and hasattr(predictor_plugin, "model") and predictor_plugin.model is not None:
            try:
                model_plot_file = config.get("model_plot_file", "model_plot.png")
                plot_model(
                    predictor_plugin.model,
                    to_file=model_plot_file,
                    show_shapes=True,
                    show_layer_names=True,
                    dpi=300,
                )
                print(f"Model plot saved: {model_plot_file}")
            except Exception as e:
                print(f"WARN: Failed model plot: {e}")
        else:
            print("INFO: Skipping model plot.")

    def _save_model(self, predictor_plugin, config: Dict) -> None:
        if hasattr(predictor_plugin, "save") and callable(predictor_plugin.save):
            save_model_file = config.get("save_model", "pretrained_model.keras")
            try:
                predictor_plugin.save(save_model_file)
                print(f"Model saved: {save_model_file}")
            except Exception as e:
                print(f"ERROR saving model: {e}")
        else:
            print("WARN: Ioin has no save method.")

    # ---------------------------------------------------------------------
    # Main pipeline orchestrator
    # ---------------------------------------------------------------------
    def run_prediction_pipeline(self, config, predictor_plugin, preprocessor_plugin, target_plugin):
        start_time = time.time()

        import random
        import numpy as np
        import tensorflow as tf

        # Set deterministic seeds if enabled (default: True for reproducibility)
        if config.get("deterministic_training", True):
            seed = config.get("random_seed", 42)
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            # Enable TensorFlow deterministic operations
            tf.config.experimental.enable_op_determinism()

        # Merge params with provided config and pin use_returns=False
        run_config = self.params.copy()
        run_config.update(config)
        run_config["use_returns"] = False  # Explicitly enforced
        # Safety: always run inference in bounded batches.
        if "predict_batch_size" not in run_config or not run_config.get("predict_batch_size"):
            bs = run_config.get("batch_size", 32)
            try:
                bs = int(bs)
            except Exception:
                bs = 32
            run_config["predict_batch_size"] = max(64, bs)
        config = run_config

        iterations = config.get("iterations", 1)
        print(f"Iterations: {iterations}")

        predicted_horizons = config.get("predicted_horizons")
        num_outputs = len(predicted_horizons)
        metric_names = ["MAE", "Naive MAE", "R2", "Uncertainty", "SNR"]
        data_sets = ["Train", "Validation", "Test"]
        metrics_results = {ds: {mn: {h: [] for h in predicted_horizons} for mn in metric_names} for ds in data_sets}

        # 1) Load and prepare datasets
        datasets = self._prepare_datasets(preprocessor_plugin, target_plugin, config)
        X_train = datasets["x_train"]
        X_val = datasets["x_val"]
        X_test = datasets["x_test"]
        # Normalize target containers (dict -> list ordered by horizons)
        y_train_list = self._targets_to_list(datasets["y_train"], predicted_horizons)
        y_val_list = self._targets_to_list(datasets["y_val"], predicted_horizons)
        y_test_list = self._targets_to_list(datasets["y_test"], predicted_horizons)

        train_dates = datasets.get("y_train_dates")
        val_dates = datasets.get("y_val_dates")
        test_dates = datasets.get("y_test_dates")
        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")

        plotted_horizon = config.get("plotted_horizon")
        output_names = [f"output_horizon_{h}" for h in predicted_horizons]

        # Train/Val target dicts
        y_train_dict = {name: np.asarray(y).reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_train_list)}
        y_val_dict = {name: np.asarray(y).reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_val_list)}

        print(f"Input shapes: Train:{X_train.shape}, Val:{X_val.shape}, Test:{X_test.shape}")
        print(
            f"Target shapes(H={predicted_horizons[0]}): Train:{y_train_list[0].shape}, Val:{y_val_list[0].shape}, Test:{y_test_list[0].shape}"
        )
        print(f"Predicting Horizons: {predicted_horizons}, Plotting: H={plotted_horizon}")

        # Iterations loop
        list_test_preds = None
        list_test_unc = None
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            iter_start = time.time()

            # 2) Build and train
            history, list_train_preds, list_train_unc, list_val_preds, list_val_unc = self._build_and_train(
                predictor_plugin, X_train, y_train_dict, X_val, y_val_dict, config
            )

            # 3) Train/Val metrics
            compute_train_val_metrics(
                metrics_results,
                predicted_horizons,
                list_train_preds,
                list_train_unc,
                list_val_preds,
                list_val_unc,
                y_train_list,
                y_val_list,
                baseline_train,
                baseline_val,
                metric_names,
                self.params,
            )

            # 4) Loss plot
            plot_and_save_loss(history, config.get("loss_plot_file"), iteration)

            # 5) Test prediction metrics
            print("Evaluating test set & calculating metrics...")
            mc_samples = config.get("mc_samples", 100)
            list_test_preds, list_test_unc = self._predict_with_uncertainty(predictor_plugin, X_test, mc_samples)
            compute_test_metrics(
                metrics_results,
                predicted_horizons,
                list_test_preds,
                list_test_unc,
                y_test_list,
                baseline_test,
                metric_names,
                self.params,
            )

            # Iteration summary (for plotted horizon)
            try:
                plotted_idx = predicted_horizons.index(plotted_horizon)
                can_calc_train_val = all(len(lst) == num_outputs for lst in [list_val_preds, list_val_unc])
                train_mae_plot = metrics_results["Train"]["MAE"][plotted_horizon][-1] if can_calc_train_val else np.nan
                train_naive_mae_plot = metrics_results["Train"]["Naive MAE"][plotted_horizon][-1] if can_calc_train_val else np.nan
                train_r2_plot = metrics_results["Train"]["R2"][plotted_horizon][-1] if can_calc_train_val else np.nan
                val_mae_plot = metrics_results["Validation"]["MAE"][plotted_horizon][-1] if can_calc_train_val else np.nan
                val_naive_mae_plot = metrics_results["Validation"]["Naive MAE"][plotted_horizon][-1] if can_calc_train_val else np.nan
                val_r2_plot = metrics_results["Validation"]["R2"][plotted_horizon][-1] if can_calc_train_val else np.nan
                test_mae_plot = metrics_results["Test"]["MAE"][plotted_horizon][-1]
                test_naive_mae_plot = metrics_results["Test"]["Naive MAE"][plotted_horizon][-1]
                test_r2_plot = metrics_results["Test"]["R2"][plotted_horizon][-1]
                test_unc_plot = metrics_results["Test"]["Uncertainty"][plotted_horizon][-1]
                test_snr_plot = metrics_results["Test"]["SNR"][plotted_horizon][-1]
                print("*" * 72)
                print(f"Iter {iteration} Done|Time:{time.time() - iter_start:.2f}s|Plot H:{plotted_horizon}")
                print(
                    f"  Train MAE:{train_mae_plot:.6f}|NMAE:{train_naive_mae_plot:.6f}|R²:{train_r2_plot:.4f} -- "
                    f"Valid MAE:{val_mae_plot:.6f}|NMAE:{val_naive_mae_plot:.6f}|R²:{val_r2_plot:.4f}"
                )
                print(
                    f"  Test  MAE:{test_mae_plot:.6f}|NMAE:{test_naive_mae_plot:.6f}|R²:{test_r2_plot:.4f}|Unc:{test_unc_plot:.6f}|SNR:{test_snr_plot:.2f}"
                )
                print("*" * 72)
            except Exception as e:
                print(f"WARN: Error printing iter summary: {e}")

        # 6) Aggregate results across iterations
        aggregate_and_save_results(
            metrics_results,
            predicted_horizons,
            config.get("results_file", self.params["results_file"]),
        )

        # 7) Save final outputs
        final_dates, num_test_points, final_baseline = save_final_outputs(
            list_test_preds,
            list_test_unc,
            y_test_list,
            test_dates,
            baseline_test,
            predicted_horizons,
            config.get("output_file", self.params["output_file"]),
            config.get("uncertainties_file", self.params.get("uncertainties_file")),
            self.params,
        )

        # 8) Plot predictions for plotted horizon
        plot_predictions(
            predicted_horizons,
            plotted_horizon,
            list_test_preds,
            list_test_unc,
            y_test_list,
            final_dates,
            num_test_points,
            final_baseline,
            config.get("predictions_plot_file", self.params["predictions_plot_file"]),
            self.params,
        )

        # 9) Optional model plot and save
        self._plot_model(predictor_plugin, config)
        self._save_model(predictor_plugin, config)

        print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f} seconds")

    # ---------------------------------------------------------------------
    # 10) Load-and-evaluate model (validation set)
    # ---------------------------------------------------------------------
    def load_and_evaluate_model(self, config, predictor_plugin, preprocessor_plugin, target_plugin):
        from tensorflow.keras.models import load_model

        print(f"Loading pre-trained model from {config['load_model']}...")
        try:
            custom_objects = {}
            predictor_plugin.model = load_model(config["load_model"], custom_objects=custom_objects)
            print("Model loaded.")
        except Exception as e:
            print(f"Failed load model: {e}")
            return

        print("Loading/processing validation data for evaluation...")
        datasets = preprocessor_plugin.run_preprocessing(target_plugin, config)
        x_val = datasets["x_val"]
        val_dates = datasets.get("y_val_dates")
        print(f"Validation data X shape: {x_val.shape}")

        print("Making predictions on validation data...")
        try:
            mc_samples = config.get("mc_samples", 100)
            list_predictions, _ = predictor_plugin.predict_with_uncertainty(x_val, mc_samples=mc_samples)
            print(f"Preds list length: {len(list_predictions)}")
        except Exception as e:
            print(f"Failed predictions: {e}")
            return

        try:
            num_val_points = len(list_predictions[0])
            final_dates = list(val_dates[:num_val_points]) if val_dates is not None else list(range(num_val_points))
            output_data = {"DATE_TIME": final_dates}
            for idx, h in enumerate(config["predicted_horizons"]):
                preds_raw = list_predictions[idx][:num_val_points].flatten()
                denorm_pred_price = denormalize(preds_raw, config)
                output_data[f"Prediction_H{h}"] = denorm_pred_price.flatten()
            evaluate_df = pd.DataFrame(output_data)
            evaluate_filename = config.get("output_file", "eval_predictions.csv")
            write_csv(file_path=evaluate_filename, data=evaluate_df, include_date=False, headers=True)
            print(f"Validation predictions saved: {evaluate_filename}")
        except ImportError:
            print("WARN: write_csv not found.")
        except Exception as e:
            print(f"Failed save validation predictions: {e}")

# --- NO if __name__ == '__main__': block ---