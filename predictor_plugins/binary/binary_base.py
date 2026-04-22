"""Binary classification mixin for ioin plugins.

Provides train/save/load overrides for binary classification models.
Must be placed BEFORE the regression parent in MRO so that its ``train()``
takes precedence over ``BaseKerasPredictor.train()``::

    class Plugin(BinaryMixin, ANNPlugin):
        ...
"""
from __future__ import annotations

import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K


class BinaryF1Score(Metric):
    """Epoch-level binary F1 metric (threshold=0.5).

    Accumulates TP/FP/FN counts across batches and computes F1 at epoch end.
    """

    def __init__(self, name="f1", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) >= self.threshold, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        return 2 * precision * recall / (precision + recall + K.epsilon())

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

    def get_config(self):
        cfg = super().get_config()
        cfg["threshold"] = self.threshold
        return cfg

VALID_SIGNAL_TYPES = ("buy_entry", "sell_entry", "buy_exit", "sell_exit")

FEATURE_COLUMNS = [
    "ATR", "RSI", "MACD", "MACD_Histogram", "MACD_Signal",
    "ADX", "DI_plus", "DI_minus", "Stochastic_K", "Stochastic_D",
    "BB_Width", "CCI", "WilliamsR", "ROC",
    "ATR_ratio", "BB_position",
    "rolling_std_24", "price_minus_ema",
    "hod_sin", "hod_cos", "dow_sin", "dow_cos",
]


def _as_bool(v, default=False):
    """Safely cast various types to bool (handles string/int/float/None)."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return bool(int(v))
        except Exception:
            return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
        return default
    return bool(v)


class BinaryMixin:
    """Overrides train/save/load for binary classification plugins.

    The base ``BaseKerasPredictor.train()`` requires ``predicted_horizons``
    and ``plotted_horizon`` in config and computes MAE/R² (regression
    metrics).  This mixin replaces that with binary-appropriate logic
    (accuracy, positive rate) and drops the horizon requirement.
    """

    def _build_callbacks(self):
        """Override base callbacks to monitor val F1 (mode=max) for early stopping."""
        from ..common.callbacks import (
            EarlyStoppingWithPatienceCounter,
            ReduceLROnPlateauWithCounter,
            ResourceUsageLogger,
        )

        quiet = self.params.get('quiet', False) or self.params.get('quiet_mode', False)
        cb_verbose = 0 if quiet else 1

        # Keras logs the F1 metric as "val_f1" (no output-name prefix for
        # single-output models even when metrics are passed as a dict).
        val_f1_key = "val_f1"

        callbacks = [
            EarlyStoppingWithPatienceCounter(
                monitor=val_f1_key,
                mode='max',
                patience=self.params.get('early_patience', 10),
                restore_best_weights=True,
                min_delta=1e-6,
                verbose=cb_verbose,
            ),
            ReduceLROnPlateauWithCounter(
                monitor=val_f1_key,
                mode='max',
                factor=0.3,
                min_delta=1e-6,
                patience=max(1, self.params.get('early_patience', 10) // 4),
                verbose=cb_verbose,
            ),
        ]
        if not quiet:
            callbacks.append(LambdaCallback(on_epoch_end=lambda e, l: print(
                f"Epoch {e+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}")))

        # Resource logging (reuse from base pattern)
        mem_log = self.params.get('memory_log_file')
        if mem_log:
            try:
                mem_tag = self.params.get('memory_log_tag')
                mem_flush = int(self.params.get('memory_log_flush_every', 1) or 1)
                mem_gpu = bool(self.params.get('memory_log_gpu', True))
                mem_gc = bool(self.params.get('memory_log_gc', False))
                callbacks.append(
                    ResourceUsageLogger(
                        str(mem_log),
                        tag=str(mem_tag) if mem_tag is not None else None,
                        flush_every=mem_flush,
                        include_gpu=mem_gpu,
                        include_gc=mem_gc,
                    )
                )
            except Exception:
                pass

        return callbacks

    def train(self, x_train, y_train, epochs, batch_size, threshold_error,
              x_val, y_val, config):
        if config:
            self.params.update(config)
        if not isinstance(y_train, dict) or not isinstance(y_val, dict):
            raise TypeError(
                "y_train/y_val must be dicts mapping output names -> arrays"
            )

        # For single-output binary models, extract the array from the dict
        # to avoid optree/Keras3 tree_map issues with single-key dicts.
        if len(y_train) == 1:
            y_train_fit = next(iter(y_train.values()))
            y_val_fit = next(iter(y_val.values()))
        else:
            y_train_fit = y_train
            y_val_fit = y_val

        callbacks = self._build_callbacks()

        # Resource snapshot at fit start (same as base class).
        try:
            from ..common.callbacks import capture_resource_snapshot
            snap = capture_resource_snapshot(
                include_gpu=bool(self.params.get("memory_log_gpu", True)),
                include_gc=bool(self.params.get("memory_log_gc", False)),
            )
            print(
                f"[RESOURCE] fit_start ts={snap.ts:.3f} "
                f"VmRSS_kB={snap.rss_kb} VmHWM_kB={snap.hwm_kb} "
                f"gpu_current_B={snap.gpu_current_bytes} "
                f"gpu_peak_B={snap.gpu_peak_bytes} gc={snap.gc_counts}"
            )
        except Exception as e:
            print(f"[RESOURCE] fit_start snapshot failed: {e}")

        _quiet = self.params.get("quiet", False) or self.params.get("quiet_mode", False)
        fit_verbose = 0 if _quiet else 1

        # Compute balanced class weights to counteract class imbalance.
        # Without this, models converge to predicting the majority class only.
        y_flat = y_train_fit.flatten().astype(int)
        classes = np.array([0, 1])
        weights = compute_class_weight("balanced", classes=classes, y=y_flat)
        class_weight = {0: float(weights[0]), 1: float(weights[1])}
        print(f"[BINARY] class_weight: {{0: {class_weight[0]:.4f}, 1: {class_weight[1]:.4f}}}")

        history = self.model.fit(
            x_train,
            y_train_fit,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val_fit),
            callbacks=callbacks,
            verbose=fit_verbose,
            shuffle=False,
            class_weight=class_weight,
        )

        # Post-fit predictions (identical to base except metrics at the end).
        disable_postfit = bool(
            self.params.get("disable_postfit_uncertainty", False)
        )
        pred_bs = int(self.params.get("predict_batch_size", 0) or 0)
        if pred_bs <= 0:
            pred_bs = (
                int(batch_size)
                if isinstance(batch_size, int) and batch_size > 0
                else 256
            )

        if disable_postfit:
            train_preds = self.model.predict(
                x_train, batch_size=pred_bs, verbose=0
            )
            val_preds = self.model.predict(
                x_val, batch_size=pred_bs, verbose=0
            )
            train_preds = (
                [train_preds]
                if isinstance(train_preds, np.ndarray)
                else train_preds
            )
            val_preds = (
                [val_preds]
                if isinstance(val_preds, np.ndarray)
                else val_preds
            )
            train_unc = [np.zeros_like(p) for p in train_preds]
            val_unc = [np.zeros_like(p) for p in val_preds]
        else:
            mc = int(self.params.get("mc_samples", 50))
            train_preds, train_unc = self.predict_with_uncertainty(
                x_train, mc
            )
            val_preds, val_unc = self.predict_with_uncertainty(x_val, mc)

        # Binary classification metrics (replaces MAE/R²).
        try:
            from sklearn.metrics import f1_score, accuracy_score
            out_name = self.output_names[0]

            # Train metrics
            y_tr_true = y_train[out_name].flatten()
            y_tr_prob = train_preds[0].flatten()
            y_tr_hat = (y_tr_prob >= 0.5).astype(int)
            tr_f1 = float(f1_score(y_tr_true, y_tr_hat, zero_division=0))
            tr_acc = float(accuracy_score(y_tr_true, y_tr_hat))
            tr_pos = float(np.mean(y_tr_true))
            tr_pred_pos = float(np.mean(y_tr_hat))

            # Val metrics
            y_vl_true = y_val[out_name].flatten()
            y_vl_prob = val_preds[0].flatten()
            y_vl_hat = (y_vl_prob >= 0.5).astype(int)
            vl_f1 = float(f1_score(y_vl_true, y_vl_hat, zero_division=0))
            vl_acc = float(accuracy_score(y_vl_true, y_vl_hat))
            vl_pos = float(np.mean(y_vl_true))
            vl_pred_pos = float(np.mean(y_vl_hat))

            print(
                f"\n{'='*60}\n"
                f"  TRAIN  F1={tr_f1:.4f}  Acc={tr_acc:.4f}  "
                f"pos_rate={tr_pos:.3f}  pred_pos={tr_pred_pos:.3f}\n"
                f"  VAL    F1={vl_f1:.4f}  Acc={vl_acc:.4f}  "
                f"pos_rate={vl_pos:.3f}  pred_pos={vl_pred_pos:.3f}\n"
                f"{'='*60}"
            )
        except Exception as e:
            print(f"Binary metric calculation error: {e}")

        return history, train_preds, train_unc, val_preds, val_unc

    # -- Persistence with metadata -------------------------------------------

    def save(self, file_path):
        super().save(file_path)
        metadata = {
            "model_type": "binary",
            "signal_type": self.params.get("signal_type", "buy_entry"),
            "window_size": self.params.get("window_size"),
            "output_names": self.output_names,
            "feature_columns": FEATURE_COLUMNS,
        }
        meta_path = str(file_path).rsplit(".", 1)[0] + "_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {meta_path}")

    def load(self, file_path):
        super().load(file_path)
        meta_path = str(file_path).rsplit(".", 1)[0] + "_metadata.json"
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            self.output_names = metadata.get(
                "output_names", self.output_names
            )
            if "signal_type" in metadata:
                self.params["signal_type"] = metadata["signal_type"]
            if "window_size" in metadata:
                self.params["window_size"] = metadata["window_size"]
            print(f"Metadata loaded from {meta_path}")
        except FileNotFoundError:
            print(f"No metadata file found at {meta_path}")
