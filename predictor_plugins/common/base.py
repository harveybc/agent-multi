#!/usr/bin/env python
"""Base ioin plugin abstractions.

Provides documented base classes to eliminate duplication across concrete
ioin plugins (ANN, CNN, LSTM, Transformer, N-BEATS).

Interface expected by pipeline:
    build_model(input_shape, x_train, config)
    train(x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config)
    predict_with_uncertainty(x_test, mc_samples=...)
    save(path) / load(path)
    calculate_mae(y_true, y_pred)
    calculate_r2(y_true, y_pred)
    set_params / get_debug_info / add_debug_info

Design:
  - BasePredictorPlugin: parameter handling + generic metrics.
  - BaseKerasPredictor: adds generic Keras save/load + common callbacks & train loop.
  - BaseBayesianKerasPredictor: adds KL weight variable + MC uncertainty.
  - BaseDeterministicKerasPredictor: zero-uncertainty implementation.

Concrete plugins only implement:
  * plugin_params (class attr)
  * plugin_debug_vars (class attr)
  * build_model (sets self.model & self.output_names)

All other behaviors inherited, drastically reducing code duplication.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K

from .losses import (
    mae_magnitude, r2_metric, composite_loss_multihead as composite_loss,
    compute_mmd, composite_loss_noreturns
)
from .callbacks import (
    ReduceLROnPlateauWithCounter,
    EarlyStoppingWithPatienceCounter,
    MemoryUsageLogger,
    ResourceUsageLogger,
    BatchResourceUsageLogger,
    ResourceGuard,
    capture_resource_snapshot,
)
from .bayesian import build_kl_anneal_callback, predict_mc_welford


def _repo_root() -> Path:
    # base.py -> common/ -> predictor_plugins/ -> <repo_root>
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(p: str | None) -> str | None:
    if not p:
        return None
    try:
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        return str((_repo_root() / pp).resolve())
    except Exception:
        return str(p)

# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------
class BasePredictorPlugin:
    """Holds parameter management + generic metrics shared by all plugins."""
    plugin_params: Dict[str, Any] = {"predicted_horizons": [1]}
    plugin_debug_vars: List[str] = ["predicted_horizons"]

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.params.update(config)
        self.model: Model | None = None
        self.output_names: List[str] = []

    # --- Param / debug API ---
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v
    def get_debug_info(self) -> Dict[str, Any]:
        return {k: self.params.get(k) for k in self.plugin_debug_vars}
    def add_debug_info(self, debug_info: Dict[str, Any]):
        debug_info.update(self.get_debug_info())

    # --- Metrics (magnitude = first column) ---
    def _ensure_two_cols(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            arr = arr.reshape(-1, 1)
            arr = np.concatenate([arr, np.zeros_like(arr)], axis=1)
        return arr
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = self._ensure_two_cols(y_true)
        y_pred = self._ensure_two_cols(y_pred)
        mae = float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])))
        print(f"MAE (magnitude): {mae}")
        return mae
    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = self._ensure_two_cols(y_true)
        y_pred = self._ensure_two_cols(y_pred)
        ss_res = np.sum((y_true[:, 0] - y_pred[:, 0]) ** 2)
        ss_tot = np.sum((y_true[:, 0] - np.mean(y_true[:, 0])) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))
        print(f"R2 (magnitude): {r2}")
        return r2

    # --- Abstract placeholders (implemented in subclasses) ---
    def build_model(self, input_shape: Tuple[int, ...], x_train, config: Dict[str, Any]):  # noqa: D401
        raise NotImplementedError
    def predict_with_uncertainty(self, x_test, mc_samples: int = 50):  # noqa: D401
        raise NotImplementedError
    def save(self, file_path: str):  # noqa: D401
        raise NotImplementedError
    def load(self, file_path: str):  # noqa: D401
        raise NotImplementedError


class BaseKerasPredictor(BasePredictorPlugin):
    """Adds shared Keras training loop, callbacks, save/load.

    Subclasses must implement build_model to populate self.model & self.output_names.
    """
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)

    # --- Custom objects reused across all models ---
    def get_custom_objects(self):
        use_returns = self.params.get('use_log1p_targets', False)
        if use_returns:
            return {
                'composite_loss': composite_loss,
                'compute_mmd': compute_mmd,
                'r2_metric': r2_metric,
                'mae_magnitude': mae_magnitude,
            }
        else:
            return {
                'composite_loss': composite_loss_noreturns,
                'compute_mmd': compute_mmd,
                'r2_metric': r2_metric,
                'mae_magnitude': mae_magnitude,
            }
        

    # --- Callbacks factory (Bayesian variant will extend) ---
    def _build_callbacks(self):
        quiet = self.params.get('quiet', False) or self.params.get('quiet_mode', False)
        cb_verbose = 0 if quiet else 1
        callbacks = [
            EarlyStoppingWithPatienceCounter(
                monitor='val_loss',
                patience=self.params.get('early_patience', 10),
                restore_best_weights=True,
                min_delta=1e-10,
                verbose=cb_verbose
            ),
            ReduceLROnPlateauWithCounter(
                monitor='val_loss',
                factor=0.3,
                min_delta=1e-10,
                patience=max(1, self.params.get('early_patience', 10) // 4),
                verbose=cb_verbose
            ),
        ]
        if not quiet:
            callbacks.append(LambdaCallback(on_epoch_end=lambda e, l: print(
                f"Epoch {e+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}")))

        mem_log = _resolve_repo_path(self.params.get('memory_log_file'))
        mem_tag = self.params.get('memory_log_tag')
        mem_flush = int(self.params.get('memory_log_flush_every', 1) or 1)
        mem_gpu = bool(self.params.get('memory_log_gpu', True))
        mem_gc = bool(self.params.get('memory_log_gc', False))

        batch_mem_log = _resolve_repo_path(self.params.get('batch_memory_log_file'))
        batch_every = self.params.get('batch_memory_log_every')
        batch_flush = int(self.params.get('batch_memory_log_flush_every', mem_flush) or 1)

        # Prefer the richer logger if enabled; keep the old one for backward compatibility.
        if mem_log:
            try:
                callbacks.append(
                    ResourceUsageLogger(
                        str(mem_log),
                        tag=str(mem_tag) if mem_tag is not None else None,
                        flush_every=mem_flush,
                        include_gpu=mem_gpu,
                        include_gc=mem_gc,
                    )
                )
                print(f"Resource logging enabled: file={mem_log} tag={mem_tag} gpu={mem_gpu} gc={mem_gc}")
            except Exception as e:
                print(f"WARN: Failed to enable ResourceUsageLogger: {e}")
                try:
                    callbacks.append(MemoryUsageLogger(str(mem_log), flush_every=mem_flush, tag=str(mem_tag) if mem_tag else None))
                    print(f"Fallback memory logging enabled: file={mem_log} tag={mem_tag}")
                except Exception as e2:
                    print(f"WARN: Failed to enable fallback MemoryUsageLogger: {e2}")

        # Optional batch-level logging (use sparingly; can be large files).
        if batch_mem_log and batch_every:
            try:
                callbacks.append(
                    BatchResourceUsageLogger(
                        str(batch_mem_log),
                        tag=str(mem_tag) if mem_tag is not None else None,
                        every_n_batches=int(batch_every),
                        flush_every=batch_flush,
                        include_gpu=mem_gpu,
                        include_gc=mem_gc,
                    )
                )
                print(
                    "Batch resource logging enabled: "
                    f"file={batch_mem_log} tag={mem_tag} every_n_batches={batch_every} flush_every={batch_flush}"
                )
            except Exception as e:
                print(f"WARN: Failed to enable BatchResourceUsageLogger: {e}")

        # Optional pre-OOM guard to stop the run with a Python exception + logs.
        max_rss_mb = self.params.get('max_rss_mb')
        max_rss_gb = self.params.get('max_rss_gb')
        max_rss_mib = self.params.get('max_rss_mib')
        max_rss_gib = self.params.get('max_rss_gib')
        guard_check_every_batches = self.params.get('resource_guard_check_every_batches')
        if max_rss_mb or max_rss_gb or max_rss_mib or max_rss_gib:
            try:
                callbacks.append(
                    ResourceGuard(
                        max_rss_mb=int(max_rss_mb) if max_rss_mb else None,
                        max_rss_gb=float(max_rss_gb) if max_rss_gb else None,
                        max_rss_mib=int(max_rss_mib) if max_rss_mib else None,
                        max_rss_gib=float(max_rss_gib) if max_rss_gib else None,
                        include_gpu=mem_gpu,
                        print_every=int(self.params.get('resource_guard_print_every', 1) or 1),
                        check_every_batches=int(guard_check_every_batches) if guard_check_every_batches else None,
                    )
                )
                print(
                    "ResourceGuard enabled: "
                    f"max_rss_mb={max_rss_mb} max_rss_gb={max_rss_gb} "
                    f"max_rss_mib={max_rss_mib} max_rss_gib={max_rss_gib} "
                    f"resource_guard_check_every_batches={guard_check_every_batches}"
                )
            except Exception as e:
                print(f"WARN: Failed to enable ResourceGuard: {e}")

        return callbacks

    # --- Generic train loop ---
    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        if config:
            self.params.update(config)
        if 'predicted_horizons' not in self.params or 'plotted_horizon' not in self.params:
            raise ValueError("Config must contain 'predicted_horizons' and 'plotted_horizon'.")
        ph = self.params['predicted_horizons']
        plotted = self.params['plotted_horizon']
        if plotted not in ph:
            raise ValueError('plotted_horizon must be one of predicted_horizons')
        plotted_index = ph.index(plotted)
        if not isinstance(y_train, dict) or not isinstance(y_val, dict):
            raise TypeError('y_train/y_val must be dicts mapping output names -> arrays')
        callbacks = self._build_callbacks()

        # One-shot resource snapshot at fit start (printed and optionally logged by callbacks later).
        try:
            snap = capture_resource_snapshot(
                include_gpu=bool(self.params.get('memory_log_gpu', True)),
                include_gc=bool(self.params.get('memory_log_gc', False)),
            )
            print(
                f"[RESOURCE] fit_start ts={snap.ts:.3f} VmRSS_kB={snap.rss_kb} VmHWM_kB={snap.hwm_kb} "
                f"gpu_current_B={snap.gpu_current_bytes} gpu_peak_B={snap.gpu_peak_bytes} gc={snap.gc_counts}"
            )
        except Exception as e:
            print(f"[RESOURCE] fit_start snapshot failed: {e}")

        fit_verbose = 0 if self.params.get('quiet', False) else 1
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=fit_verbose,
            shuffle=False,
        )

        # Post-fit predictions/uncertainty can dominate memory/time during GA optimization.
        # When disabled, we only run a single batched deterministic predict.
        disable_postfit_uncertainty = bool(self.params.get('disable_postfit_uncertainty', False))
        pred_bs = int(self.params.get('predict_batch_size', 0) or 0)
        if pred_bs <= 0:
            pred_bs = int(batch_size) if isinstance(batch_size, int) and batch_size > 0 else 256

        if disable_postfit_uncertainty:
            train_preds = self.model.predict(x_train, batch_size=pred_bs, verbose=0)
            val_preds = self.model.predict(x_val, batch_size=pred_bs, verbose=0)
            train_preds = [train_preds] if isinstance(train_preds, np.ndarray) else train_preds
            val_preds = [val_preds] if isinstance(val_preds, np.ndarray) else val_preds
            train_unc = [np.zeros_like(p) for p in train_preds]
            val_unc = [np.zeros_like(p) for p in val_preds]
        else:
            mc = int(self.params.get('mc_samples', 50))
            train_preds, train_unc = self.predict_with_uncertainty(x_train, mc)
            val_preds, val_unc = self.predict_with_uncertainty(x_val, mc)
        try:
            self.calculate_mae(y_train[self.output_names[plotted_index]], train_preds[plotted_index])
            self.calculate_r2(y_train[self.output_names[plotted_index]], train_preds[plotted_index])
        except Exception as e:  # pragma: no cover (defensive)
            print(f'Metric calculation error: {e}')
        return history, train_preds, train_unc, val_preds, val_unc

    # --- Persistence ---
    def save(self, file_path: str):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")
    def load(self, file_path: str):
        self.model = load_model(file_path, custom_objects=self.get_custom_objects())
        print(f"Ioin model loaded from {file_path}")


class BaseBayesianKerasPredictor(BaseKerasPredictor):
    """Adds KL annealing variable + MC uncertainty to Keras base ioin."""
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_weight_var")
        # Patch DenseFlipout add_variable once (TFP quirk for some versions)
        if not hasattr(tfp.layers.DenseFlipout, "_already_patched_add_variable"):
            def _patched_add_variable(layer_instance, name, shape, dtype, initializer, trainable, **kwargs):
                return layer_instance.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
            tfp.layers.DenseFlipout.add_variable = _patched_add_variable  # type: ignore
            tfp.layers.DenseFlipout._already_patched_add_variable = True  # type: ignore

    def _build_callbacks(self):
        base = super()._build_callbacks()
        base.append(build_kl_anneal_callback(self, self.params.get('kl_weight', 1e-3), self.params.get('kl_anneal_epochs', 10)))
        return base

    def predict_with_uncertainty(self, x_test, mc_samples: int = 50):
        pred_bs = int(self.params.get('predict_batch_size', 0) or self.params.get('batch_size', 0) or 0)
        return predict_mc_welford(self.model, x_test, mc_samples, batch_size=pred_bs, training=False)


class BaseDeterministicKerasPredictor(BaseKerasPredictor):
    """Deterministic variant returning zero uncertainties."""
    def predict_with_uncertainty(self, x_test, mc_samples: int = 1):
        preds = self.model.predict(x_test, verbose=0)
        preds = [preds] if isinstance(preds, np.ndarray) else preds
        zeros = [np.zeros_like(p) for p in preds]
        return preds, zeros

__all__ = [
    'BasePredictorPlugin', 'BaseKerasPredictor', 'BaseBayesianKerasPredictor', 'BaseDeterministicKerasPredictor'
]
