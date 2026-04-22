#!/usr/bin/env python
"""
STL Decomposition Pipeline Plugin.

Orchestrates STL decomposition based training.
Modes:
1. Sequential: Train separate models for Trend, Seasonal, Residual, then sum.
2. Simultaneous: Train one model with internal branching and unified loss.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional

# Local helpers (assuming these exist in the environment as per stl_pipeline.py)
from .stl_norm import denormalize, denormalize_returns
from .stl_metrics import (
    compute_train_val_metrics,
    compute_test_metrics,
    aggregate_and_save_results,
)
from .stl_plots import plot_and_save_loss, plot_predictions
from .stl_io import save_final_outputs

try:
    from tensorflow.keras.utils import plot_model
except ImportError:
    plot_model = None

class STLPipelinePlugin:
    
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
        "use_returns": False,
        
        # New params
        "sequential_training": True, 
    }

    plugin_debug_vars = [
        "iterations", "batch_size", "epochs", "sequential_training"
    ]

    def __init__(self) -> None:
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self) -> Dict:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict) -> None:
        debug_info.update(self.get_debug_info())

    def _prepare_datasets(self, preprocessor_plugin, target_plugin, config: Dict) -> Dict:
        print("Loading/processing datasets via Preprocessor...")
        datasets = preprocessor_plugin.run_preprocessing(target_plugin, config)
        print("Preprocessor finished.")
        return datasets

    @staticmethod
    def _targets_to_list(y_targets, predicted_horizons: List[int]) -> List[np.ndarray]:
        if isinstance(y_targets, (list, tuple)):
            return list(y_targets)
        if isinstance(y_targets, dict):
            return [y_targets[f"output_horizon_{h}"] for h in predicted_horizons]
        raise TypeError("y_targets must be a list/tuple or a dict keyed by 'output_horizon_{h}'")

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

    def _predict_with_uncertainty(self, predictor_plugin, X_test, mc_samples):
        return predictor_plugin.predict_with_uncertainty(X_test, mc_samples=mc_samples)

    def run_prediction_pipeline(self, config, predictor_plugin, preprocessor_plugin, target_plugin):
        start_time = time.time()
        run_config = self.params.copy()
        run_config.update(config)
        run_config["use_returns"] = False
        config = run_config

        iterations = config.get("iterations", 1)
        predicted_horizons = config.get("predicted_horizons")
        sequential = config.get("sequential_training", True)
        
        # 1) Load Datasets
        datasets = self._prepare_datasets(preprocessor_plugin, target_plugin, config)
        X_train = datasets["x_train"]
        X_val = datasets["x_val"]
        X_test = datasets["x_test"]
        
        # Helper to extract targets for a specific component
        def get_targets(component_suffix=""):
            # component_suffix: "", "_trend", "_seasonal", "_resid"
            # The target plugin returns keys like "y_train_trend"
            
            def extract(split):
                key = f"y_{split}{component_suffix}"
                if key not in datasets:
                    # Fallback to standard keys if suffix is empty
                    if component_suffix == "":
                        key = f"y_{split}"
                    else:
                        raise ValueError(f"Missing component targets: {key}")
                return self._targets_to_list(datasets[key], predicted_horizons)

            y_tr = extract("train")
            y_v = extract("val")
            y_te = extract("test")
            return y_tr, y_v, y_te

        output_names = [f"output_horizon_{h}" for h in predicted_horizons]
        
        # Prepare metrics containers
        metric_names = ["MAE", "Naive MAE", "R2", "Uncertainty", "SNR"]
        data_sets = ["Train", "Validation", "Test"]
        metrics_results = {ds: {mn: {h: [] for h in predicted_horizons} for mn in metric_names} for ds in data_sets}

        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")
        test_dates = datasets.get("y_test_dates")

        # Main Loop
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            
            if sequential:
                print("--- Sequential Training Mode ---")
                components = ["trend", "seasonal", "resid"]
                
                # Store predictions for each component
                comp_preds = {"train": {}, "val": {}, "test": {}}
                comp_uncs = {"train": {}, "val": {}, "test": {}}
                
                # Train 3 separate models
                for comp in components:
                    print(f"Training component: {comp}")
                    y_tr_list, y_v_list, _ = get_targets(f"_{comp}")
                    
                    # Convert to dict for ioin
                    y_tr_dict = {name: np.asarray(y).reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_tr_list)}
                    y_v_dict = {name: np.asarray(y).reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_v_list)}
                    
                    # Configure ioin for standard MIMO (simultaneous=False)
                    comp_config = config.copy()
                    comp_config["use_simultaneous_stl"] = False
                    
                    history, tr_preds, tr_unc, val_preds, val_unc = self._build_and_train(
                        predictor_plugin, X_train, y_train_dict=y_tr_dict, X_val=X_val, y_val_dict=y_v_dict, config=comp_config
                    )
                    
                    # Predict on Test
                    mc_samples = config.get("mc_samples", 100)
                    te_preds, te_unc = self._predict_with_uncertainty(predictor_plugin, X_test, mc_samples)
                    
                    # Store predictions
                    comp_preds["train"][comp] = tr_preds
                    comp_uncs["train"][comp] = tr_unc
                    comp_preds["val"][comp] = val_preds
                    comp_uncs["val"][comp] = val_unc
                    comp_preds["test"][comp] = te_preds
                    comp_uncs["test"][comp] = te_unc
                    
                    # Save loss plot for component
                    plot_and_save_loss(history, f"loss_plot_iter{iteration}_{comp}.png", iteration)

                # Recompose
                def recompose(split_name):
                    preds_list = []
                    uncs_list = []
                    for i in range(len(predicted_horizons)):
                        # Sum means
                        sum_pred = (comp_preds[split_name]["trend"][i] + 
                                    comp_preds[split_name]["seasonal"][i] + 
                                    comp_preds[split_name]["resid"][i])
                        preds_list.append(sum_pred)
                        
                        # Sum uncertainties (sqrt sum squares)
                        u_trend = comp_uncs[split_name]["trend"][i]
                        u_seas = comp_uncs[split_name]["seasonal"][i]
                        u_resid = comp_uncs[split_name]["resid"][i]
                        sum_unc = np.sqrt(u_trend**2 + u_seas**2 + u_resid**2)
                        uncs_list.append(sum_unc)
                    return preds_list, uncs_list

                list_train_preds, list_train_unc = recompose("train")
                list_val_preds, list_val_unc = recompose("val")
                list_test_preds, list_test_unc = recompose("test")
                
                # Get Total Targets for metrics
                y_train_list, y_val_list, y_test_list = get_targets("")

            else:
                print("--- Simultaneous Training Mode ---")
                # Train 1 model with simultaneous=True
                # Target is the TOTAL target
                y_tr_list, y_v_list, y_test_list = get_targets("")
                
                y_tr_dict = {name: np.asarray(y).reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_tr_list)}
                y_v_dict = {name: np.asarray(y).reshape(-1, 1).astype(np.float32) for name, y in zip(output_names, y_v_list)}
                
                sim_config = config.copy()
                sim_config["use_simultaneous_stl"] = True
                
                history, list_train_preds, list_train_unc, list_val_preds, list_val_unc = self._build_and_train(
                    predictor_plugin, X_train, y_train_dict=y_tr_dict, X_val=X_val, y_val_dict=y_v_dict, config=sim_config
                )
                
                plot_and_save_loss(history, config.get("loss_plot_file"), iteration)
                
                mc_samples = config.get("mc_samples", 100)
                list_test_preds, list_test_unc = self._predict_with_uncertainty(predictor_plugin, X_test, mc_samples)

            # Compute Metrics
            compute_train_val_metrics(
                metrics_results, predicted_horizons, list_train_preds, list_train_unc,
                list_val_preds, list_val_unc, y_train_list, y_val_list,
                baseline_train, baseline_val, metric_names, self.params
            )
            
            compute_test_metrics(
                metrics_results, predicted_horizons, list_test_preds, list_test_unc,
                y_test_list, baseline_test, metric_names, self.params
            )

        # Aggregate and Save
        aggregate_and_save_results(
            metrics_results, predicted_horizons, config.get("results_file", self.params["results_file"])
        )

        # Save Final Outputs
        final_dates, num_test_points, final_baseline = save_final_outputs(
            list_test_preds, list_test_unc, y_test_list, test_dates, baseline_test,
            predicted_horizons, config.get("output_file", self.params["output_file"]),
            config.get("uncertainties_file", self.params.get("uncertainties_file")),
            self.params
        )

        # Plot
        plot_predictions(
            predicted_horizons, config.get("plotted_horizon", 6), list_test_preds, list_test_unc,
            y_test_list, final_dates, num_test_points, final_baseline,
            config.get("predictions_plot_file", self.params["predictions_plot_file"]),
            self.params
        )

        print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f} seconds")

    def load_and_evaluate_model(self, config, predictor_plugin, preprocessor_plugin, target_plugin):
        # Placeholder for load_and_evaluate - similar to original but simplified
        pass
