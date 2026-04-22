#!/usr/bin/env python
"""
Binary Classification Pipeline Plugin

Full pipeline for binary classification predictors. Replaces
regression-specific metrics (MAE, R², Naive MAE, SNR) with proper
binary classification metrics (Accuracy, Precision, Recall, F1,
AUC-ROC, AUC-PR, MCC, Brier Score, Log Loss).

Output files
------------
- Predictions CSV : DATE_TIME, True_Label_H1, Probability_H1, Predicted_Label_H1
- Uncertainties CSV : DATE_TIME, Uncertainty_H1
- Results CSV : aggregated binary metrics (same Metric/Average/Std Dev/Min/Max)
- Loss plot, prediction probability plot, confusion matrix, ROC+PR curves
- OLAP upload of experiment config & results (when DB available)

Steps overview (mirrored as section headers in code):
  1) Load and prepare datasets
  2) Build and train model (per iteration)
  3) Compute Train/Validation binary metrics
  4) Plot and save loss curve (per iteration)
  5) Predict on Test with uncertainty and compute binary metrics
  6) Aggregate metrics across iterations and save results
  7) Save final binary predictions/targets and uncertainties
  8) Plot binary predictions, confusion matrix, ROC+PR
  9) Optionally export model plot and save model
 10) Upload experiment and results to OLAP cube
 11) Load-and-evaluate (utility)
"""

from typing import Dict, List, Optional
import time

import numpy as np
import pandas as pd

# Conditional import for plot_model
try:
    from tensorflow.keras.utils import plot_model
except ImportError:
    plot_model = None

from .binary_metrics import (
    BINARY_METRIC_NAMES,
    compute_binary_metrics,
    aggregate_and_save_binary_results,
)
from .binary_plots import (
    plot_and_save_loss,
    plot_binary_predictions,
    plot_confusion_matrix,
    plot_roc_pr_curves,
)
from .binary_io import save_binary_outputs
from .binary_olap import upload_binary_experiment


class BinaryPipelinePlugin:
    """Pipeline for binary classification predictors (single horizon H1)."""

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
        "cm_plot_file": "confusion_matrix.png",
        "roc_pr_plot_file": "roc_pr_curves.png",
        "results_file": "results.csv",
        "plot_points": 480,
        "plotted_horizon": 1,
        "use_strategy": False,
        "predicted_horizons": [1],
        "window_size": 48,
        "mc_samples": 100,
        "use_returns": False,
        # OLAP defaults
        "olap_upload": True,
        "olap_project_key": "ioin",
        "olap_phase_key": "phase_1b_binary",
    }

    plugin_debug_vars = [
        "iterations",
        "batch_size",
        "epochs",
        "threshold_error",
        "output_file",
        "uncertainties_file",
        "results_file",
        "plot_points",
        "olap_upload",
    ]

    # ------------------------------------------------------------------
    # 0) Construction and config helpers
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self) -> Dict:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict) -> None:
        debug_info.update(self.get_debug_info())

    # ------------------------------------------------------------------
    # 1) Load and prepare datasets
    # ------------------------------------------------------------------
    def _prepare_datasets(self, preprocessor_plugin, target_plugin, config: Dict) -> Dict:
        print("Loading/processing datasets via Preprocessor …")
        datasets = preprocessor_plugin.run_preprocessing(target_plugin, config)
        print("Preprocessor finished.")
        return datasets

    @staticmethod
    def _targets_to_list(y_targets, predicted_horizons: List[int]) -> List[np.ndarray]:
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
        raise TypeError("y_targets must be a list/tuple or dict keyed by 'output_horizon_{h}'")

    # ------------------------------------------------------------------
    # 2) Build and train model (per iteration)
    # ------------------------------------------------------------------
    def _build_and_train(self, predictor_plugin, X_train, y_train_dict, X_val, y_val_dict, config):
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

    # ------------------------------------------------------------------
    # 9) Optionally export model plot and save model
    # ------------------------------------------------------------------
    def _plot_model(self, predictor_plugin, config: Dict) -> None:
        if plot_model is not None and hasattr(predictor_plugin, "model") and predictor_plugin.model is not None:
            try:
                plot_model(
                    predictor_plugin.model,
                    to_file=config.get("model_plot_file", "model_plot.png"),
                    show_shapes=True,
                    show_layer_names=True,
                    dpi=300,
                )
                print(f"Model plot saved: {config.get('model_plot_file')}")
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

    # ------------------------------------------------------------------
    # Main pipeline orchestrator
    # ------------------------------------------------------------------
    def run_prediction_pipeline(self, config, predictor_plugin, preprocessor_plugin, target_plugin):
        start_time = time.time()

        import random
        import tensorflow as tf

        # Deterministic seeds
        if config.get("deterministic_training", True):
            seed = config.get("random_seed", 42)
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            tf.config.experimental.enable_op_determinism()

        # Merge params
        run_config = self.params.copy()
        run_config.update(config)
        run_config["use_returns"] = False
        if "predict_batch_size" not in run_config or not run_config.get("predict_batch_size"):
            bs = run_config.get("batch_size", 32)
            try:
                bs = int(bs)
            except Exception:
                bs = 32
            run_config["predict_batch_size"] = max(64, bs)
        config = run_config

        iterations = config.get("iterations", 1)
        predicted_horizons = config.get("predicted_horizons", [1])
        data_sets = ["Train", "Validation", "Test"]
        metrics_results = {
            ds: {mn: {h: [] for h in predicted_horizons} for mn in BINARY_METRIC_NAMES}
            for ds in data_sets
        }

        # 1) Load and prepare datasets
        datasets = self._prepare_datasets(preprocessor_plugin, target_plugin, config)
        X_train = datasets["x_train"]
        X_val   = datasets["x_val"]
        X_test  = datasets["x_test"]
        y_train_list = self._targets_to_list(datasets["y_train"], predicted_horizons)
        y_val_list   = self._targets_to_list(datasets["y_val"], predicted_horizons)
        y_test_list  = self._targets_to_list(datasets["y_test"], predicted_horizons)

        test_dates = datasets.get("y_test_dates")

        output_names = [f"output_horizon_{h}" for h in predicted_horizons]
        y_train_dict = {n: np.asarray(y).reshape(-1, 1).astype(np.float32) for n, y in zip(output_names, y_train_list)}
        y_val_dict   = {n: np.asarray(y).reshape(-1, 1).astype(np.float32) for n, y in zip(output_names, y_val_list)}

        print(f"Input shapes: Train:{X_train.shape}, Val:{X_val.shape}, Test:{X_test.shape}")
        print(f"Target shapes (H={predicted_horizons[0]}): Train:{y_train_list[0].shape}, Val:{y_val_list[0].shape}, Test:{y_test_list[0].shape}")
        print(f"Predicting Horizons: {predicted_horizons}")

        # Iterations loop
        list_test_preds = None
        list_test_unc = None
        for iteration in range(1, iterations + 1):
            print(f"\n{'='*60}")
            print(f"=== Binary Classification Iteration {iteration}/{iterations} ===")
            print(f"{'='*60}")
            iter_start = time.time()

            # 2) Build and train (or load pre-trained model)
            if config.get("load_model"):
                import zipfile, tempfile, os
                model_path = config["load_model"]
                print(f"\nLoading pre-trained weights from {model_path} …")
                # 1) Build the model architecture (creates Lambda layers fresh)
                input_shape = (X_train.shape[1], X_train.shape[2]) if X_train.ndim == 3 else (X_train.shape[1],)
                predictor_plugin.build_model(input_shape=input_shape, x_train=X_train, config=config)
                # 2) Extract weights from .keras zip and load them
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(model_path, "r") as zf:
                        zf.extractall(tmpdir)
                    weights_path = os.path.join(tmpdir, "model.weights.h5")
                    if not os.path.exists(weights_path):
                        raise FileNotFoundError(f"No model.weights.h5 inside {model_path}")
                    predictor_plugin.model.load_weights(weights_path)
                print("Weights loaded — skipping training, computing predictions …")
                history = None
                pred_bs = int(config.get("predict_batch_size", 256))
                list_train_preds = [predictor_plugin.model.predict(X_train, batch_size=pred_bs, verbose=0)]
                list_train_unc = [None]
                list_val_preds = [predictor_plugin.model.predict(X_val, batch_size=pred_bs, verbose=0)]
                list_val_unc = [None]
            else:
                history, list_train_preds, list_train_unc, list_val_preds, list_val_unc = self._build_and_train(
                    predictor_plugin, X_train, y_train_dict, X_val, y_val_dict, config
                )

            # 3) Train/Val binary metrics
            print("\nComputing Train/Validation binary metrics …")
            for idx, h in enumerate(predicted_horizons):
                try:
                    train_unc = list_train_unc[idx] if list_train_unc and idx < len(list_train_unc) else None
                    val_unc = list_val_unc[idx] if list_val_unc and idx < len(list_val_unc) else None

                    compute_binary_metrics(
                        y_train_list[idx], list_train_preds[idx], train_unc,
                        "Train", h, metrics_results,
                    )
                    compute_binary_metrics(
                        y_val_list[idx], list_val_preds[idx], val_unc,
                        "Validation", h, metrics_results,
                    )
                except Exception as e:
                    print(f"WARN: Error Train/Val metrics H={h}: {e}")
                    for ds in ["Train", "Validation"]:
                        for m in BINARY_METRIC_NAMES:
                            metrics_results[ds][m][h].append(np.nan)

            # 4) Loss plot
            if history is not None:
                plot_and_save_loss(history, config.get("loss_plot_file", "loss_plot.png"), iteration)

            # 5) Test prediction + binary metrics
            print("\nEvaluating test set with MC uncertainty …")
            mc_samples = config.get("mc_samples", 100)
            list_test_preds, list_test_unc = predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)

            for idx, h in enumerate(predicted_horizons):
                try:
                    test_unc = list_test_unc[idx] if list_test_unc and idx < len(list_test_unc) else None
                    compute_binary_metrics(
                        y_test_list[idx], list_test_preds[idx], test_unc,
                        "Test", h, metrics_results,
                    )
                except Exception as e:
                    print(f"WARN: Error Test metrics H={h}: {e}")
                    for m in BINARY_METRIC_NAMES:
                        metrics_results["Test"][m][h].append(np.nan)

            # Iteration summary
            h0 = predicted_horizons[0]
            try:
                print("*" * 72)
                print(f"Iter {iteration} Done | Time: {time.time() - iter_start:.2f}s")
                print(
                    f"  Train  Acc={metrics_results['Train']['Accuracy'][h0][-1]:.4f} "
                    f"F1={metrics_results['Train']['F1'][h0][-1]:.4f} "
                    f"AUC={metrics_results['Train']['AUC_ROC'][h0][-1]:.4f}"
                )
                print(
                    f"  Val    Acc={metrics_results['Validation']['Accuracy'][h0][-1]:.4f} "
                    f"F1={metrics_results['Validation']['F1'][h0][-1]:.4f} "
                    f"AUC={metrics_results['Validation']['AUC_ROC'][h0][-1]:.4f}"
                )
                print(
                    f"  Test   Acc={metrics_results['Test']['Accuracy'][h0][-1]:.4f} "
                    f"F1={metrics_results['Test']['F1'][h0][-1]:.4f} "
                    f"AUC={metrics_results['Test']['AUC_ROC'][h0][-1]:.4f} "
                    f"MCC={metrics_results['Test']['MCC'][h0][-1]:.4f} "
                    f"Brier={metrics_results['Test']['Brier'][h0][-1]:.4f}"
                )
                print("*" * 72)
            except Exception as e:
                print(f"WARN: Error printing iter summary: {e}")

        # 6) Aggregate results across iterations
        results_file = config.get("results_file", self.params["results_file"])
        aggregate_and_save_binary_results(metrics_results, predicted_horizons, results_file)

        # 7) Save final binary outputs
        final_dates, num_test_points = save_binary_outputs(
            list_test_preds,
            list_test_unc,
            y_test_list,
            test_dates,
            predicted_horizons,
            config.get("output_file", self.params["output_file"]),
            config.get("uncertainties_file", self.params.get("uncertainties_file")),
            self.params,
        )

        # 8) Binary-specific plots
        h0 = predicted_horizons[0]
        idx0 = 0
        test_true_h0 = y_test_list[idx0][:num_test_points].flatten()
        test_prob_h0 = list_test_preds[idx0][:num_test_points].flatten()

        plot_binary_predictions(
            test_true_h0, test_prob_h0,
            final_dates, num_test_points,
            config.get("predictions_plot_file", self.params["predictions_plot_file"]),
            self.params,
        )
        plot_confusion_matrix(
            test_true_h0, test_prob_h0,
            config.get("cm_plot_file", self.params["cm_plot_file"]),
        )
        plot_roc_pr_curves(
            test_true_h0, test_prob_h0,
            config.get("roc_pr_plot_file", self.params["roc_pr_plot_file"]),
        )

        # 9) Optional model plot and save
        self._plot_model(predictor_plugin, config)
        self._save_model(predictor_plugin, config)

        # 10) OLAP upload
        if config.get("olap_upload", True):
            print("\n--- OLAP Upload ---")
            experiment_key = config.get(
                "olap_experiment_key",
                f"{config.get('predictor_plugin', 'unknown')}_{config.get('signal_type', 'unknown')}",
            )
            upload_binary_experiment(
                config_json=config,
                results_csv=results_file,
                project_key=config.get("olap_project_key", "ioin"),
                phase_key=config.get("olap_phase_key", "phase_1b_binary"),
                experiment_key=experiment_key,
            )

        print(f"\nTotal Binary Pipeline Execution Time: {time.time() - start_time:.2f} seconds")

    # ------------------------------------------------------------------
    # 11) Load-and-evaluate model (validation set)
    # ------------------------------------------------------------------
    def load_and_evaluate_model(self, config, predictor_plugin, preprocessor_plugin, target_plugin):
        from tensorflow.keras.models import load_model
        from app.data_handler import write_csv

        print(f"Loading pre-trained model from {config['load_model']} …")
        try:
            predictor_plugin.model = load_model(config["load_model"])
            print("Model loaded.")
        except Exception as e:
            print(f"Failed load model: {e}")
            return

        print("Loading/processing data for evaluation …")
        datasets = preprocessor_plugin.run_preprocessing(target_plugin, config)
        x_val = datasets["x_val"]
        val_dates = datasets.get("y_val_dates")
        print(f"Validation data X shape: {x_val.shape}")

        print("Making predictions on validation data …")
        try:
            mc_samples = config.get("mc_samples", 100)
            list_predictions, _ = predictor_plugin.predict_with_uncertainty(x_val, mc_samples=mc_samples)
        except Exception as e:
            print(f"Failed predictions: {e}")
            return

        try:
            num_pts = len(list_predictions[0])
            final_dates = list(val_dates[:num_pts]) if val_dates is not None else list(range(num_pts))
            output_data = {"DATE_TIME": final_dates}
            for idx, h in enumerate(config.get("predicted_horizons", [1])):
                y_prob = list_predictions[idx][:num_pts].flatten()
                output_data[f"Probability_H{h}"] = y_prob
                output_data[f"Predicted_Label_H{h}"] = (y_prob >= 0.5).astype(int)
            eval_df = pd.DataFrame(output_data)
            out_file = config.get("output_file", "eval_predictions.csv")
            write_csv(file_path=out_file, data=eval_df, include_date=False, headers=True)
            print(f"Validation predictions saved: {out_file}")
        except Exception as e:
            print(f"Failed save validation predictions: {e}")
