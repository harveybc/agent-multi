#!/usr/bin/env python
"""
Default Pipeline Plugin

Este plugin orquesta el flujo completo:
  1. Obtiene los datasets mediante el Preprocessor Plugin.
  2. Ejecuta iteraciones de entrenamiento, validación y evaluación utilizando el Ioin Plugin.
  3. Calcula métricas (MAE, R², incertidumbre, SNR), genera gráficos de pérdida y predicción, y guarda resultados en archivos CSV.
  4. Permite, adicionalmente, la carga y evaluación de modelos preentrenados.

La lógica de preprocesamiento (creación de ventanas deslizantes, etc.) se delega al Preprocessor Plugin,
manteniendo así la separación de preocupaciones.
"""

import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.utils import plot_model

# Se asume que las funciones de preprocesamiento y de manejo de archivos (como write_csv) se encuentran en sus respectivos plugins.

class PipelinePlugin:
    # Parámetros por defecto específicos del pipeline.
    plugin_params = {
        "iterations": 1,
        "batch_size": 32,
        "epochs": 50,
        "threshold_error": 0.001,
        "loss_plot_file": "loss_plot.png",
        "output_file": "test_predictions.csv",
        "model_plot_file": "model_plot.png",
        "uncertainties_file": "test_uncertainties.csv",
        "predictions_plot_file": "predictions_plot.png",
        "plot_points": 1575,
        "plotted_horizon": 6,
        "use_strategy": False  # Si se activa, se guardan estadísticas extendidas.
    }
    plugin_debug_vars = ["iterations", "batch_size", "epochs", "threshold_error"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        """
        Actualiza los parámetros del pipeline combinando los parámetros específicos con la configuración global.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Devuelve información de debug de los parámetros relevantes del pipeline.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Agrega la información de debug al diccionario proporcionado.
        """
        debug_info.update(self.get_debug_info())

    def run_prediction_pipeline(self, config, predictor_plugin, preprocessor_plugin):
        """
        Ejecuta el pipeline completo:
          1. Obtiene los datasets usando el Preprocessor Plugin.
          2. Realiza iteraciones de entrenamiento y evaluación utilizando el Ioin Plugin.
          3. Calcula y guarda métricas, gráficos y resultados (incluyendo la denormalización de predicciones si se configura).
          4. Guarda resultados consolidados en archivos CSV.

        Args:
            config (dict): Configuración global.
            predictor_plugin: Plugin encargado de construir, entrenar y predecir.
            preprocessor_plugin: Plugin encargado del preprocesamiento de datos.
        """
        start_time = time.time()
        iterations = config.get("iterations", self.params["iterations"])
        print(f"Number of iterations: {iterations}")

        # Inicialización de listas para almacenar métricas.
        training_mae_list, training_r2_list, training_unc_list, training_snr_list = [], [], [], []
        validation_mae_list, validation_r2_list, validation_unc_list, validation_snr_list = [], [], [], []
        test_mae_list, test_r2_list, test_unc_list, test_snr_list = [], [], [], []

        # 1. Obtener datasets mediante el Preprocessor Plugin.
        print("Loading and processing datasets using Preprocessor Plugin...")
        datasets = preprocessor_plugin.run_preprocessing(config)
        x_train, y_train = datasets["x_train"], datasets["y_train"]
        x_val, y_val = datasets["x_val"], datasets["y_val"]
        x_test, y_test = datasets["x_test"], datasets["y_test"]

        # Convertir targets a arrays 2D.
        y_train_array = y_train[0] if isinstance(y_train, list) and len(y_train) == 1 else np.stack(y_train, axis=1)
        y_val_array   = y_val[0] if isinstance(y_val, list) and len(y_val) == 1 else np.stack(y_val, axis=1)
        y_test_array  = y_test[0] if isinstance(y_test, list) and len(y_test) == 1 else np.stack(y_test, axis=1)

        train_dates = datasets.get("dates_train")
        val_dates = datasets.get("dates_val")
        test_dates = datasets.get("dates_test")
        test_close_prices = datasets.get("test_close_prices")
        if config.get("use_returns", False):
            baseline_train = datasets.get("baseline_train")
            baseline_val = datasets.get("baseline_val")
            baseline_test = datasets.get("baseline_test")

        print(f"Training data shapes: x_train: {x_train.shape}, y_train: {y_train_array.shape}")
        print(f"Validation data shapes: x_val: {x_val.shape}, y_val: {y_val_array.shape}")
        print(f"Test data shapes: x_test: {x_test.shape}, y_test: {y_test_array.shape}")

        time_horizon = config.get("time_horizon")
        window_size = config.get("window_size")
        if time_horizon is None:
            raise ValueError("`time_horizon` is not defined in the configuration.")
        if config["plugin"] in ["lstm", "cnn", "transformer", "ann"] and window_size is None:
            raise ValueError("`window_size` must be defined for CNN, Transformer and LSTM plugins.")
        print(f"Time Horizon: {time_horizon}")
        batch_size = config.get("batch_size", self.params["batch_size"])
        epochs = config.get("epochs", self.params["epochs"])
        threshold_error = config.get("threshold_error", self.params["threshold_error"])

        # Actualizar la configuración del Ioin Plugin (por ejemplo, establecer el horizonte).
        predictor_plugin.set_params(time_horizon=time_horizon)

        # Iteraciones de entrenamiento y evaluación.
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            iter_start = time.time()
            # Construir el modelo en función de la forma de entrada.
            if config["plugin"] in ["lstm", "cnn", "transformer", "ann"]:
                predictor_plugin.build_model(input_shape=(window_size, x_train.shape[2]), x_train=x_train, config=config)
            else:
                predictor_plugin.build_model(input_shape=x_train.shape[1], x_train=x_train, config=config)

            history, train_preds, train_unc, val_preds, val_unc = predictor_plugin.train(
                x_train, y_train, epochs=epochs, batch_size=batch_size,
                threshold_error=threshold_error, x_val=x_val, y_val=y_val, config=config
            )

            # Si se usa returns, realizar inversa escalada de las predicciones.
            if config.get("use_returns", False):
                inv_scale_factor = 1.0 / config.get("target_scaling_factor", 100.0)
                print(f"DEBUG: Inversely scaling predictions by factor {inv_scale_factor}.")
                train_preds = train_preds * inv_scale_factor
                val_preds = val_preds * inv_scale_factor

            # Calcular R² para entrenamiento y validación.
            if config.get("use_returns", False):
                train_r2 = r2_score(
                    (baseline_train[:, -1] + np.stack(y_train, axis=1)[:, -1]).flatten(),
                    (baseline_train[:, -1] + train_preds[:, 0]).flatten()
                )
                val_r2 = r2_score(
                    (baseline_val[:, -1] + np.stack(y_val, axis=1)[:, -1]).flatten(),
                    (baseline_val[:, -1] + val_preds[:, 0]).flatten()
                )
            else:
                train_r2 = r2_score(np.stack(y_train, axis=1)[:, -1].flatten(), train_preds[:, 0].flatten())
                val_r2 = r2_score(np.stack(y_val, axis=1)[:, -1].flatten(), val_preds[:, 0].flatten())

            # Calcular MAE.
            n_train = train_preds.shape[0]
            n_val = val_preds.shape[0]
            train_mae = np.mean(np.abs(train_preds[:, -1] - np.stack(y_train, axis=1)[:n_train, -1]))
            val_mae = np.mean(np.abs(val_preds[:, -1] - np.stack(y_val, axis=1)[:n_val, -1]))

            # Guardar gráfico de pérdida.
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f"Model Loss for {config['plugin'].upper()} - Iteration {iteration}")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Val"], loc="upper left")
            loss_plot_file = config.get("loss_plot_file", self.params["loss_plot_file"])
            plt.savefig(loss_plot_file)
            plt.close()
            print(f"Loss plot saved to {loss_plot_file}")

            print("\nEvaluating on test dataset...")
            mc_samples = config.get("mc_samples", 100)
            test_predictions, uncertainty_estimates = predictor_plugin.predict_with_uncertainty(x_test, mc_samples=mc_samples)
            n_test = test_predictions.shape[0]
            y_test_array = np.stack(y_test, axis=1)

            if config.get("use_returns", False) and "baseline_test" in datasets:
                print("DEBUG: baseline_test shape:", datasets["baseline_test"].shape)
            else:
                print("DEBUG: Not using returns or baseline_test not available")
            print("DEBUG: y_test_array shape:", y_test_array.shape)

            test_mae = np.mean(np.abs(test_predictions[:, -1] - y_test_array[:n_test, -1]))
            test_r2 = r2_score(y_test_array.flatten(), test_predictions[:, 0].flatten())

            # Calcular incertidumbre (media de la última columna).
            train_unc_last = np.mean(train_unc[:, -1])
            val_unc_last = np.mean(val_unc[:, -1])
            test_unc_last = np.mean(uncertainty_estimates[:, -1])

            # Calcular SNR (relación señal-ruido).
            if config.get("use_returns", False):
                train_mean = np.mean(baseline_train[:, -1] + train_preds[:, -1])
                val_mean = np.mean(baseline_val[:, -1] + val_preds[:, -1])
                test_mean = np.mean(baseline_test[:, -1] + test_predictions[:, -1])
            else:
                train_mean = np.mean(train_preds[:, -1])
                val_mean = np.mean(val_preds[:, -1])
                test_mean = np.mean(test_predictions[:, -1])
            train_snr = 1 / (train_unc_last / train_mean)
            val_snr = 1 / (val_unc_last / val_mean)
            test_snr = 1 / (test_unc_last / test_mean)

            # Almacenar métricas.
            training_mae_list.append(train_mae)
            training_r2_list.append(train_r2)
            training_unc_list.append(train_unc_last)
            training_snr_list.append(train_snr)
            validation_mae_list.append(val_mae)
            validation_r2_list.append(val_r2)
            validation_unc_list.append(val_unc_last)
            validation_snr_list.append(val_snr)
            test_mae_list.append(test_mae)
            test_r2_list.append(test_r2)
            test_unc_list.append(test_unc_last)
            test_snr_list.append(test_snr)

            print("************************************************************************")
            print(f"Iteration {iteration} completed in {time.time() - iter_start:.2f} seconds")
            print(f"Training MAE: {train_mae}, Training R²: {train_r2}, Training Uncertainty: {train_unc_last}, Training SNR: {train_snr}")
            print(f"Validation MAE: {val_mae}, Validation R²: {val_r2}, Validation Uncertainty: {val_unc_last}, Validation SNR: {val_snr}")
            print(f"Test MAE: {test_mae}, Test R²: {test_r2}, Test Uncertainty: {test_unc_last}, Test SNR: {test_snr}")
            print("************************************************************************")

        # Consolidar resultados.
        if config.get("use_strategy", False):
            results = {
                "Metric": ["Training MAE", "Training R²", "Training Uncertainty", "Training SNR",
                           "Validation MAE", "Validation R²", "Validation Uncertainty", "Validation SNR",
                           "Test MAE", "Test R²", "Test Uncertainty", "Test SNR"],
                "Average": [np.mean(training_mae_list), np.mean(training_r2_list),
                            np.mean(training_unc_list), np.mean(training_snr_list),
                            np.mean(validation_mae_list), np.mean(validation_r2_list),
                            np.mean(validation_unc_list), np.mean(validation_snr_list),
                            np.mean(test_mae_list), np.mean(test_r2_list),
                            np.mean(test_unc_list), np.mean(test_snr_list)],
                "Std Dev": [np.std(training_mae_list), np.std(training_r2_list),
                            np.std(training_unc_list), np.std(training_snr_list),
                            np.std(validation_mae_list), np.std(validation_r2_list),
                            np.std(validation_unc_list), np.std(validation_snr_list),
                            np.std(test_mae_list), np.std(test_r2_list),
                            np.std(test_unc_list), np.std(test_snr_list)]
            }
        else:
            results = {
                "Metric": ["Training MAE", "Training R²", "Training Uncertainty", "Training SNR",
                           "Validation MAE", "Validation R²", "Validation Uncertainty", "Validation SNR",
                           "Test MAE", "Test R²", "Test Uncertainty", "Test SNR"],
                "Average": [np.mean(training_mae_list), np.mean(training_r2_list),
                            np.mean(training_unc_list), np.mean(training_snr_list),
                            np.mean(validation_mae_list), np.mean(validation_r2_list),
                            np.mean(validation_unc_list), np.mean(validation_snr_list),
                            np.mean(test_mae_list), np.mean(test_r2_list),
                            np.mean(test_unc_list), np.mean(test_snr_list)],
                "Std Dev": [np.std(training_mae_list), np.std(training_r2_list),
                            np.std(training_unc_list), np.std(training_snr_list),
                            np.std(validation_mae_list), np.std(validation_r2_list),
                            np.std(validation_unc_list), np.std(validation_snr_list),
                            np.std(test_mae_list), np.std(test_r2_list),
                            np.std(test_unc_list), np.std(test_snr_list)]
            }
        results_file = config.get("results_file", "results.csv")
        pd.DataFrame(results).to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

        # Denormalizar predicciones finales (si se provee normalización).
        norm_json = config.get("use_normalization_json")
        if norm_json is None:
            norm_json = {}
        elif isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        if "CLOSE" in norm_json:
            close_info = norm_json["CLOSE"]
            if "min" in close_info and "max" in close_info:
                close_min = close_info["min"]; close_max = close_info["max"]
                denorm_test_close_prices = test_close_prices * (close_max - close_min) + close_min
            elif "mean" in close_info and "std" in close_info:
                mean = close_info["mean"]; std = close_info["std"]
                denorm_test_close_prices = test_close_prices * std + mean
            else:
                denorm_test_close_prices = test_close_prices
        else:
            denorm_test_close_prices = test_close_prices

        if config.get("use_normalization_json") is not None:
            norm_json = config.get("use_normalization_json")
            if isinstance(norm_json, str):
                with open(norm_json, 'r') as f:
                    norm_json = json.load(f)
            if config.get("use_returns", False):
                if "CLOSE" in norm_json:
                    close_info = norm_json["CLOSE"]
                    if baseline_test is not None:
                        if "min" in close_info and "max" in close_info:
                            diff = close_info["max"] - close_info["min"]
                            test_predictions = (test_predictions + baseline_test) * diff + close_info["min"]
                            y_test_array = np.stack(y_test, axis=1)
                            denorm_y_test = (y_test_array + baseline_test) * diff + close_info["min"]
                        elif "mean" in close_info and "std" in close_info:
                            std = close_info["std"]; mean = close_info["mean"]
                            test_predictions = (test_predictions + baseline_test) * std + mean
                            y_test_array = np.stack(y_test, axis=1)
                            denorm_y_test = (y_test_array + baseline_test) * std + mean
                        else:
                            denorm_y_test = np.stack(y_test, axis=1)
                    else:
                        print("Warning: Baseline test values not found; skipping returns denormalization.")
                        denorm_y_test = np.stack(y_test, axis=1)
                else:
                    print("Warning: 'CLOSE' not found; skipping denormalization for returns.")
                    denorm_y_test = np.stack(y_test, axis=1)
            else:
                if "CLOSE" in norm_json:
                    close_info = norm_json["CLOSE"]
                    if "min" in close_info and "max" in close_info:
                        test_predictions = test_predictions * (close_info["max"] - close_info["min"]) + close_info["min"]
                        denorm_y_test = np.stack(y_test, axis=1) * (close_info["max"] - close_info["min"]) + close_info["min"]
                    elif "mean" in close_info and "std" in close_info:
                        test_predictions = test_predictions * close_info["std"] + close_info["mean"]
                        denorm_y_test = np.stack(y_test, axis=1) * close_info["std"] + close_info["mean"]
                    else:
                        print("Warning: 'CLOSE' not found; skipping denormalization for non-returns mode.")
                        denorm_y_test = np.stack(y_test, axis=1)
                else:
                    print("Warning: 'CLOSE' not found; skipping denormalization for non-returns mode.")
                    denorm_y_test = np.stack(y_test, axis=1)
        else:
            denorm_y_test = np.stack(y_test, axis=1)

        # Recompute denorm_test_close_prices using the same auto-detection logic for consistency
        try:
            norm_json_tmp = config.get("use_normalization_json")
            if norm_json_tmp is None:
                denorm_test_close_prices = test_close_prices
            else:
                if isinstance(norm_json_tmp, str):
                    with open(norm_json_tmp, 'r') as f:
                        norm_json_tmp = json.load(f)
                if isinstance(norm_json_tmp, dict) and "CLOSE" in norm_json_tmp:
                    ci = norm_json_tmp["CLOSE"]
                    if "min" in ci and "max" in ci:
                        denorm_test_close_prices = test_close_prices * (ci["max"] - ci["min"]) + ci["min"]
                    elif "mean" in ci and "std" in ci:
                        denorm_test_close_prices = test_close_prices * ci["std"] + ci["mean"]
                    else:
                        denorm_test_close_prices = test_close_prices
                else:
                    denorm_test_close_prices = test_close_prices
        except Exception:
            denorm_test_close_prices = test_close_prices

        # Guardar predicciones finales en CSV.
        final_test_file = config.get("output_file", "test_predictions.csv")
        num_pred_steps = test_predictions.shape[1]
        pred_cols = [f"Prediction_{i+1}" for i in range(num_pred_steps)]
        test_predictions_df = pd.DataFrame(test_predictions, columns=pred_cols)
        if test_dates is not None:
            test_predictions_df['DATE_TIME'] = pd.Series(test_dates[:len(test_predictions_df)])
        else:
            test_predictions_df['DATE_TIME'] = pd.NaT
        cols = ['DATE_TIME'] + [col for col in test_predictions_df.columns if col != 'DATE_TIME']
        test_predictions_df = test_predictions_df[cols]
        denorm_y_test_df = pd.DataFrame(denorm_y_test, columns=[f"Target_{i+1}" for i in range(denorm_y_test.shape[1])])
        test_predictions_df = pd.concat([test_predictions_df, denorm_y_test_df], axis=1)
        test_predictions_df['test_CLOSE'] = denorm_test_close_prices
        from app.data_handler import write_csv
        write_csv(file_path=final_test_file, data=test_predictions_df, include_date=False, headers=config.get('headers', True))
        print(f"Final validation predictions saved to {final_test_file}")

        # Calcular y guardar incertidumbres (denormalizadas).
        print("Computing uncertainty estimates using MC sampling...")
        try:
            mc_samples = config.get("mc_samples", 100)
            _, uncertainty_estimates = predictor_plugin.predict_with_uncertainty(x_test, mc_samples=mc_samples)
            if config.get("use_normalization_json") is not None:
                norm_json = config.get("use_normalization_json")
                if isinstance(norm_json, str):
                    with open(norm_json, 'r') as f:
                        norm_json = json.load(f)
                if "CLOSE" in norm_json:
                    close_info = norm_json["CLOSE"]
                    if "min" in close_info and "max" in close_info:
                        diff = close_info["max"] - close_info["min"]
                        denorm_uncertainty = uncertainty_estimates * diff
                    elif "mean" in close_info and "std" in close_info:
                        denorm_uncertainty = uncertainty_estimates * close_info["std"]
                    else:
                        print("Warning: 'CLOSE' not found; uncertainties remain normalized.")
                        denorm_uncertainty = uncertainty_estimates
                else:
                    print("Warning: 'CLOSE' not found; uncertainties remain normalized.")
                    denorm_uncertainty = uncertainty_estimates
            else:
                denorm_uncertainty = uncertainty_estimates
            uncertainty_df = pd.DataFrame(denorm_uncertainty, columns=[f"Uncertainty_{i+1}" for i in range(denorm_uncertainty.shape[1])])
            if test_dates is not None:
                uncertainty_df['DATE_TIME'] = pd.Series(test_dates[:len(uncertainty_df)])
            else:
                uncertainty_df['DATE_TIME'] = pd.NaT
            cols = ['DATE_TIME'] + [col for col in uncertainty_df.columns if col != 'DATE_TIME']
            uncertainty_df = uncertainty_df[cols]
            uncertainties_file = config.get("uncertainties_file", "test_uncertainties.csv")
            uncertainty_df.to_csv(uncertainties_file, index=False)
            print(f"Uncertainty predictions saved to {uncertainties_file}")
        except Exception as e:
            print(f"Failed to compute or save uncertainty predictions: {e}")

        # Plot predictions para el horizonte configurado.
        plotted_horizon = config.get("plotted_horizon", 6)
        plotted_idx = plotted_horizon - 1
        if plotted_idx >= test_predictions.shape[1]:
            raise ValueError(f"Plotted horizon index {plotted_idx} is out of bounds for predictions shape {test_predictions.shape}")
        pred_plot = test_predictions[:, plotted_idx]
        n_plot = config.get("plot_points", 1575)
        if len(pred_plot) > n_plot:
            pred_plot = pred_plot[-n_plot:]
            test_dates_plot = test_dates[-n_plot:] if test_dates is not None else np.arange(len(pred_plot))
        else:
            test_dates_plot = test_dates if test_dates is not None else np.arange(len(pred_plot))
        true_plot = denorm_test_close_prices
        if len(true_plot) > len(test_dates_plot):
            true_plot = true_plot[-len(test_dates_plot):]
        uncertainty_plot = denorm_uncertainty[:, plotted_idx]
        if len(uncertainty_plot) > n_plot:
            uncertainty_plot = uncertainty_plot[-n_plot:]
        plot_color_predicted = config.get("plot_color_predicted", "blue")
        plot_color_true = config.get("plot_color_true", "red")
        plot_color_uncertainty = config.get("plot_color_uncertainty", "green")
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates_plot, pred_plot, label="Predicted Price", color=plot_color_predicted, linewidth=2)
        plt.plot(test_dates_plot, true_plot, label="True Price", color=plot_color_true, linewidth=2)
        plt.fill_between(test_dates_plot, pred_plot - uncertainty_plot, pred_plot + uncertainty_plot,
                         color=plot_color_uncertainty, alpha=0.15, label="Uncertainty")
        if config.get("use_daily", False):
            plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} days)")
        else:
            plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} hours)")
        plt.xlabel("Close Time")
        plt.ylabel("EUR Price [USD]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        try:
            predictions_plot_file = config.get("predictions_plot_file", self.params["predictions_plot_file"])
            plt.savefig(predictions_plot_file, dpi=300)
            plt.close()
            print(f"Prediction plot saved to {predictions_plot_file}")
        except Exception as e:
            print(f"Failed to generate prediction plot: {e}")

        # Plotear y guardar la gráfica del modelo.
        try:
            plot_model(predictor_plugin.model, to_file=config['model_plot_file'],
                       show_shapes=True, show_dtype=False, show_layer_names=True,
                       expand_nested=True, dpi=300, show_layer_activations=True)
            print(f"Model plot saved to {config['model_plot_file']}")
        except Exception as e:
            print(f"Failed to generate model plot: {e}")
            print("Download Graphviz from https://graphviz.org/download/")
        save_model_file = config.get("save_model", "pretrained_model.keras")
        try:
            predictor_plugin.save(save_model_file)
            print(f"Model saved to {save_model_file}")
        except Exception as e:
            print(f"Failed to save model to {save_model_file}: {e}")

        print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")

    def load_and_evaluate_model(self, config, predictor_plugin, preprocessor_plugin):
        """
        Carga un modelo preentrenado y lo evalúa usando datos de validación.
        Las predicciones se denormalizan y se guardan en un CSV junto con DATE_TIME.
        """
        from tensorflow.keras.models import load_model
        print(f"Loading pre-trained model from {config['load_model']}...")
        try:
            custom_objects = {"combined_loss": combined_loss, "mmd": mmd_metric, "huber": huber_metric}
            predictor_plugin.model = load_model(config['load_model'], custom_objects=custom_objects)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load the model from {config['load_model']}: {e}")
            return

        print("Loading and processing validation data for evaluation...")
        datasets = preprocessor_plugin.run_preprocessing(config)
        x_val = datasets["x_val"]
        val_dates = datasets.get("dates_val")
        print(f"Processed validation data: X shape: {x_val.shape}")

        print("Making predictions on validation data...")
        try:
            x_val_array = x_val if isinstance(x_val, np.ndarray) else x_val.to_numpy()
            mc_samples = config.get("mc_samples", 100)
            predictions, _ = predictor_plugin.predict_with_uncertainty(x_val_array, mc_samples=mc_samples)
            print(f"Predictions shape: {predictions.shape}")
        except Exception as e:
            print(f"Failed to make predictions: {e}")
            return

        if predictions.ndim == 1 or predictions.shape[1] == 1:
            predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        else:
            num_steps = predictions.shape[1]
            pred_cols = [f'Prediction_{i+1}' for i in range(num_steps)]
            predictions_df = pd.DataFrame(predictions, columns=pred_cols)
        if val_dates is not None:
            predictions_df['DATE_TIME'] = pd.Series(val_dates[:len(predictions_df)])
        else:
            predictions_df['DATE_TIME'] = pd.NaT
            print("Warning: DATE_TIME for validation predictions not captured.")
        cols = ['DATE_TIME'] + [col for col in predictions_df.columns if col != 'DATE_TIME']
        predictions_df = predictions_df[cols]
        evaluate_filename = config['output_file']
        from app.data_handler import write_csv
        try:
            write_csv(file_path=evaluate_filename, data=predictions_df,
                      include_date=False, headers=config.get('headers', True))
            print(f"Validation predictions with DATE_TIME saved to {evaluate_filename}")
        except Exception as e:
            print(f"Failed to save validation predictions to {evaluate_filename}: {e}")

# Debugging usage example: ejecutar el plugin directamente.
if __name__ == "__main__":
    pipeline_plugin = PipelinePlugin()
    # Configuración de prueba (ajustar según los datos disponibles)
    test_config = {
        "x_train_file": "data/train.csv",
        "x_validation_file": "data/val.csv",
        "x_test_file": "data/test.csv",
        "headers": True,
        "max_steps_train": 1000,
        "max_steps_val": 500,
        "max_steps_test": 500,
        "window_size": 24,
        "time_horizon": 1,
        "use_returns": false,
    "use_log1p_targets": false,
        "plugin": "ann",
        "loss_plot_file": "loss_plot.png",
        "output_file": "test_predictions.csv",
        "model_plot_file": "model_plot.png",
        "uncertainties_file": "test_uncertainties.csv",
        "predictions_plot_file": "predictions_plot.png",
        "batch_size": 32,
        "epochs": 50,
        "threshold_error": 0.001
    }
    from app.plugin_loader import load_plugin
    predictor_class, _ = load_plugin('ioin.plugins', test_config.get('plugin', 'default_predictor'))
    predictor_plugin = predictor_class()
    predictor_plugin.set_params(**test_config)
    from app.plugin_loader import load_plugin as load_preprocessor_plugin
    preprocessor_class, _ = load_preprocessor_plugin('preprocessor.plugins', 'default_preprocessor')
    preprocessor_plugin = preprocessor_class()
    preprocessor_plugin.set_params(**test_config)
    pipeline_plugin.run_prediction_pipeline(test_config, predictor_plugin, preprocessor_plugin)
