#!/usr/bin/env python
"""Candidate evaluation worker.

Runs a single (gen,candidate) evaluation in a fresh Python process to avoid
in-process TensorFlow / CUDA / allocator cache growth across GA candidates.

# Reverse mapping: GA encodes activation as int [0..7], model needs string.
ACTIVATION_INDEX_TO_NAME = [
    "relu",         # 0
    "elu",          # 1
    "selu",         # 2
    "tanh",         # 3
    "sigmoid",      # 4
    "swish",        # 5
    "gelu",         # 6
    "leaky_relu",   # 7
]

This module is invoked by `optimizer_plugins/default_optimizer.py` via:
  python -m optimizer_plugins.candidate_worker --input <json> --output <json>

Input JSON schema:
{
  "gen": int,
  "cand": int,
  "config": { ... },
  "hyper": { ... }
}

Output JSON schema:
{
    "ok": bool,
    "fitness": float,                 # VALIDATION MAE max horizon (pipeline parity)
    "naive_mae": float|null,          # VALIDATION Naive MAE max horizon
    "train_mae": float|null,          # TRAINING MAE max horizon
    "train_naive_mae": float|null,    # TRAINING Naive MAE max horizon
    "test_mae": float|null,           # TEST MAE max horizon
    "test_naive_mae": float|null,     # TEST Naive MAE max horizon
    "error": str|null
}
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

_QUIET = os.environ.get('PREDICTOR_QUIET', '0') == '1'


def _repo_root() -> Path:
    # candidate_worker.py -> optimizer_plugins/ -> <repo_root>
    return Path(__file__).resolve().parents[1]


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


def _read_proc_status_kb(key: str) -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    return int(parts[1])
    except Exception:
        return None
    return None


def _read_gpu_mem_bytes() -> tuple[int | None, int | None]:
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return (None, None)
        info = tf.config.experimental.get_memory_info("GPU:0")  # type: ignore[attr-defined]
        cur = int(info.get("current")) if isinstance(info, dict) and info.get("current") is not None else None
        peak = int(info.get("peak")) if isinstance(info, dict) and info.get("peak") is not None else None
        return (cur, peak)
    except Exception:
        return (None, None)


def _coerce_bool(v: object | None, *, default: bool = False) -> bool:
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


def _append_optimizer_resource_row(config: dict, stage: str, gen: int | None, cand: int | None, extra: dict | None = None) -> None:
    log_path = _resolve_repo_path(config.get("optimizer_resource_log_file"))
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("ts,stage,generation,candidate,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,extra\n")

        ts = time.time()
        rss = _read_proc_status_kb("VmRSS")
        hwm = _read_proc_status_kb("VmHWM")
        gpu_cur, gpu_peak = _read_gpu_mem_bytes() if bool(config.get("memory_log_gpu", True)) else (None, None)
        extra_json = ""
        if extra is not None:
            try:
                extra_json = json.dumps(extra, separators=(",", ":"))
            except Exception:
                extra_json = ""

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{ts:.3f},{stage},{gen if gen is not None else ''},{cand if cand is not None else ''},"
                f"{rss if rss is not None else ''},{hwm if hwm is not None else ''},"
                f"{gpu_cur if gpu_cur is not None else ''},{gpu_peak if gpu_peak is not None else ''},"
                f"{extra_json}\n"
            )
            f.flush()
    except Exception:
        # Never fail the worker due to logging.
        return


def evaluate_candidate(*, config: dict, hyper: dict, gen: int, cand: int) -> tuple[float, float | None]:
    """Run preprocessing, build model, train, and compute fitness + naive_mae.

    Fitness is the denormalized MAE for the max horizon (pipeline parity).
    """

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

    # Resolve log paths to repo root for consistency.
    for k in ("memory_log_file", "optimizer_resource_log_file", "batch_memory_log_file"):
        if config.get(k):
            config[k] = _resolve_repo_path(config.get(k))

    # Resolve data file paths to repo root for consistency.
    for k in ("x_train_file", "y_train_file", "x_validation_file", "y_validation_file",
              "x_test_file", "y_test_file", "use_normalization_json"):
        if config.get(k):
            config[k] = _resolve_repo_path(config.get(k))

    # Tag for per-epoch/batch logs.
    config.setdefault("memory_log_tag", f"ga_gen{int(gen)}_cand{int(cand)}")

    # Safety: avoid post-fit uncertainty during GA eval.
    config.setdefault("disable_postfit_uncertainty", True)
    config.setdefault("mc_samples", 1)
    config.setdefault("predict_batch_size", config.get("batch_size", 32))

    from app.plugin_loader import load_plugin

    # Instantiate plugins inside the worker process.
    target_plugin_name = config.get("target_plugin", "default_target")
    target_class, _ = load_plugin("target.plugins", target_plugin_name)
    target_plugin = target_class()
    target_plugin.set_params(**config)

    preprocessor_name = config.get("preprocessor_plugin", "default_preprocessor")
    preprocessor_class, _ = load_plugin("preprocessor.plugins", preprocessor_name)
    preprocessor_plugin = preprocessor_class()
    preprocessor_plugin.set_params(**config)

    predictor_name = config.get("plugin") or config.get("predictor_plugin")
    predictor_class, _ = load_plugin("ioin.plugins", predictor_name)
    predictor_plugin = predictor_class(config)
    predictor_plugin.set_params(**config)

    _append_optimizer_resource_row(config, "before_preprocess", gen, cand)
    datasets = preprocessor_plugin.run_preprocessing(target_plugin, config)
    if isinstance(datasets, tuple):
        datasets = datasets[0]
    _append_optimizer_resource_row(config, "after_preprocess", gen, cand)

    # FIX: Extract dates and feature names for Prophet/Time-aware predictors
    if "x_train_dates" in datasets:
        config["train_dates"] = datasets["x_train_dates"]
    else:
        print("WARNING: 'x_train_dates' not found in preprocessor output.")

    if "x_val_dates" in datasets:
        config["val_dates"] = datasets["x_val_dates"]
    if "x_test_dates" in datasets:
        config["test_dates"] = datasets["x_test_dates"]
    if "feature_names" in datasets:
        config["feature_names"] = datasets["feature_names"]

    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]
    x_test, y_test = datasets.get("x_test"), datasets.get("y_test")
    baseline_train = datasets.get("baseline_train")
    baseline_val = datasets.get("baseline_val")
    baseline_test = datasets.get("baseline_test")

    def _ensure_2d_targets(y):
        if isinstance(y, dict):
            out = {}
            for k, v in y.items():
                arr = __import__("numpy").asarray(v)
                out[k] = arr.reshape(-1, 1).astype("float32")
            return out
        return y

    y_train = _ensure_2d_targets(y_train)
    y_val = _ensure_2d_targets(y_val)

    window_size = config.get("window_size")
    _append_optimizer_resource_row(config, "before_build_model", gen, cand)
    
    # Try tuple input_shape first (for sequence models), fall back to scalar
    input_shape_tuple = (window_size, x_train.shape[2]) if len(x_train.shape) == 3 else (x_train.shape[1],)
    try:
        if not _QUIET: print(f"Attempting build_model with tuple input_shape: {input_shape_tuple}")
        predictor_plugin.build_model(input_shape=input_shape_tuple, x_train=x_train, config=config)
        if not _QUIET: print("build_model succeeded with tuple input_shape")
    except (ValueError, TypeError) as e:
        # Plugin expects scalar input_shape, try with flattened dimension
        if not _QUIET: print(f"Tuple input_shape failed ({type(e).__name__}: {e}), trying scalar...")
        input_shape_scalar = x_train.shape[1]
        if not _QUIET: print(f"Attempting build_model with scalar input_shape: {input_shape_scalar}")
        predictor_plugin.build_model(input_shape=input_shape_scalar, x_train=x_train, config=config)
        if not _QUIET: print("build_model succeeded with scalar input_shape")

    # Capture model summary as a string for dashboard display
    _model_summary_str = ""
    try:
        if hasattr(predictor_plugin, "model") and predictor_plugin.model is not None:
            _lines = []
            predictor_plugin.model.summary(line_length=120, print_fn=lambda line: _lines.append(line))
            _model_summary_str = "\n".join(_lines)
    except Exception:
        pass

    _append_optimizer_resource_row(config, "after_build_model", gen, cand)

    _append_optimizer_resource_row(config, "before_fit", gen, cand, extra={"params": hyper})
    if not _QUIET: print(f"Starting training with epochs={config.get('epochs', 10)}, batch_size={config.get('batch_size', 32)}")
    if not _QUIET: print(f"x_train shape: {x_train.shape}, y_train type: {type(y_train)}")
    if isinstance(y_train, dict):
        if not _QUIET: print(f"y_train keys: {y_train.keys()}, shapes: {[(k, v.shape) for k, v in y_train.items()]}")
    else:
        if not _QUIET: print(f"y_train shape: {y_train.shape}")
    
    try:
        history, train_preds, _, val_preds, _ = predictor_plugin.train(
            x_train,
            y_train,
            epochs=config.get("epochs", 10),
            batch_size=config.get("batch_size", 32),
            threshold_error=config.get("threshold_error", 0.001),
            x_val=x_val,
            y_val=y_val,
            config=config,
        )
        if not _QUIET: print("Training completed successfully")
    except Exception as e:
        print(f"ERROR during training: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    _append_optimizer_resource_row(config, "after_fit", gen, cand)

    import numpy as np

    # --- Detect binary classification mode ---
    _is_binary = config.get("target_plugin") in ("binary_target", "direction_target")

    if _is_binary:
        # ── BINARY CLASSIFICATION METRICS ─────────────────────────
        from predictor_plugins.common.binary_fitness import (
            compute_binary_metrics_for_split,
            compute_binary_fitness,
            find_best_threshold,
        )

        predicted_horizons = config.get("predicted_horizons", [1])
        max_h_idx = 0  # Binary always uses horizon 1

        def _extract_h(y_any, preds_list):
            if isinstance(y_any, dict):
                key = list(y_any.keys())[0]
                y_h = np.asarray(y_any[key]).flatten()
            elif isinstance(y_any, list):
                y_h = np.asarray(y_any[max_h_idx]).flatten()
            else:
                y_h = np.asarray(y_any).flatten()
            p_h = np.asarray(preds_list[max_h_idx]).flatten()
            n = min(len(y_h), len(p_h))
            return y_h[:n], p_h[:n]

        # Extract raw arrays
        y_tr, p_tr = _extract_h(y_train, train_preds)
        y_vl, p_vl = _extract_h(y_val, val_preds)

        # Find optimal threshold on validation set (maximises val F1)
        best_threshold = find_best_threshold(y_vl, p_vl)
        if not _QUIET:
            print(f"  Binary threshold search: best_threshold={best_threshold:.2f}")

        # TRAIN metrics (using optimal threshold from val)
        train_bin_metrics = compute_binary_metrics_for_split(y_tr, p_tr, threshold=best_threshold)
        if not _QUIET:
            print(f"  Binary TRAIN: AUC={train_bin_metrics['auc_roc']:.4f} F1={train_bin_metrics['f1']:.4f} "
                  f"Acc={train_bin_metrics['accuracy']:.4f} MCC={train_bin_metrics['mcc']:.4f}")

        # VALIDATION metrics (using optimal threshold)
        val_bin_metrics = compute_binary_metrics_for_split(y_vl, p_vl, threshold=best_threshold)
        if not _QUIET:
            print(f"  Binary VAL:   AUC={val_bin_metrics['auc_roc']:.4f} F1={val_bin_metrics['f1']:.4f} "
                  f"Acc={val_bin_metrics['accuracy']:.4f} MCC={val_bin_metrics['mcc']:.4f}")

        # FITNESS: Penalized Asymmetric AUC (binary-specific)
        fitness = compute_binary_fitness(train_bin_metrics, val_bin_metrics)

        # Map binary metrics into the standard wire-format keys for doin-node compatibility:
        #   val_mae → val Accuracy,  train_mae → train Accuracy
        #   val_naive_mae → val F1, train_naive_mae → train F1
        #   naive_mae → val F1 (used as the "baseline" reference in wire format)
        train_mae = train_bin_metrics["accuracy"]
        val_mae = val_bin_metrics["accuracy"]
        train_naive_mae = train_bin_metrics["f1"]
        naive_mae = val_bin_metrics["f1"]

        # TEST
        test_mae = None
        test_naive_mae = None
        if x_test is not None and y_test is not None:
            try:
                if hasattr(predictor_plugin, "model") and hasattr(predictor_plugin.model, "predict"):
                    pred_bs = int(config.get("predict_batch_size", 0) or config.get("batch_size", 32) or 256)
                    test_preds_raw = predictor_plugin.model.predict(x_test, batch_size=pred_bs, verbose=0)
                elif hasattr(predictor_plugin, "predict_with_uncertainty"):
                    test_preds_raw, _ = predictor_plugin.predict_with_uncertainty(x_test, mc_samples=config.get("mc_samples", 1))
                else:
                    test_preds_raw = None

                if test_preds_raw is not None:
                    test_preds_list = [test_preds_raw] if isinstance(test_preds_raw, np.ndarray) else test_preds_raw
                    y_ts, p_ts = _extract_h(y_test, test_preds_list)
                    test_bin_metrics = compute_binary_metrics_for_split(y_ts, p_ts, threshold=best_threshold)
                    test_mae = test_bin_metrics["accuracy"]
                    test_naive_mae = test_bin_metrics["f1"]
                    if not _QUIET:
                        print(f"  Binary TEST:  AUC={test_bin_metrics['auc_roc']:.4f} F1={test_bin_metrics['f1']:.4f}")
            except Exception as e:
                if not _QUIET:
                    print(f"  Binary TEST failed: {e}")
                test_mae = None
                test_naive_mae = None

    else:
        # ── REGRESSION METRICS (original path, unchanged) ─────────
        from pipeline_plugins.stl_norm import denormalize, denormalize_returns

        predicted_horizons = config.get("predicted_horizons", [1])
        max_horizon = max(predicted_horizons) if predicted_horizons else 1
        max_h_idx = predicted_horizons.index(max_horizon) if predicted_horizons else 0

        def _extract_max_h(y_any):
            if isinstance(y_any, dict):
                return np.asarray(y_any.get(f"output_horizon_{max_horizon}")).reshape(-1)
            if isinstance(y_any, list):
                return np.asarray(y_any[max_h_idx]).reshape(-1)
            return np.asarray(y_any).reshape(-1)

        def _split_metrics(preds_list, y_any, baseline_any):
            y_h = _extract_max_h(y_any)
            p_h = np.asarray(preds_list[max_h_idx]).reshape(-1)
            n = min(len(y_h), len(p_h))
            if baseline_any is not None:
                n = min(n, len(np.asarray(baseline_any).reshape(-1)))
            if n <= 0:
                return (float("inf"), None)
            y_h = y_h[:n]
            p_h = p_h[:n]

            # CORRECT MAE CALCULATION (Real Price Space)
            real_p = denormalize(p_h, config)
            real_y = denormalize(y_h, config)
            mae = float(np.mean(np.abs(real_p - real_y)))

            naive = None
            if baseline_any is not None:
                baseline_h = np.asarray(baseline_any).reshape(-1)[:n]
                real_baseline = denormalize(baseline_h, config)
                naive = float(np.mean(np.abs(real_baseline - real_y)))
            return (mae, naive)

        # TRAIN
        train_mae, train_naive_mae = _split_metrics(train_preds, y_train, baseline_train)

        # VALIDATION
        val_mae, naive_mae = _split_metrics(val_preds, y_val, baseline_val)

        # FITNESS: Penalized Asymmetric Delta (shared implementation)
        from predictor_plugins.common.fitness import compute_fitness
        fitness = compute_fitness(train_mae, train_naive_mae, val_mae, naive_mae)

        # TEST
        test_mae = None
        test_naive_mae = None
        if x_test is not None and y_test is not None:
            try:
                if hasattr(predictor_plugin, "model") and hasattr(predictor_plugin.model, "predict"):
                    pred_bs = int(config.get("predict_batch_size", 0) or config.get("batch_size", 32) or 256)
                    test_preds = predictor_plugin.model.predict(x_test, batch_size=pred_bs, verbose=0)
                elif hasattr(predictor_plugin, "predict_with_uncertainty"):
                    test_preds, _ = predictor_plugin.predict_with_uncertainty(x_test, mc_samples=config.get("mc_samples", 1))
                else:
                    test_preds = []

                test_preds = [test_preds] if isinstance(test_preds, np.ndarray) else test_preds
                test_mae, test_naive_mae = _split_metrics(test_preds, y_test, baseline_test)
            except Exception as e:
                test_mae = float("inf")
                test_naive_mae = float("inf")

    # Save trained model for DOIN evaluator verification (inference-only)
    try:
        model_save_path = config.get("_doin_model_save_path")
        if model_save_path and hasattr(predictor_plugin, "model"):
            predictor_plugin.model.save(model_save_path)
    except Exception:
        pass

    # Cleanup best-effort.
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

    # Stash extra metrics on the function for parent JSON (kept simple: return tuple plus attrs)
    evaluate_candidate.last_metrics = {
        "train_mae": train_mae,
        "train_naive_mae": train_naive_mae,
        "val_mae": val_mae,
        "test_mae": test_mae,
        "test_naive_mae": test_naive_mae,
        "model_summary": _model_summary_str,
    }
    return fitness, naive_mae


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Configure TF allocator before importing TF via plugins.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "1")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    payload = json.load(open(args.input, "r", encoding="utf-8"))
    gen = int(payload.get("gen", 0))
    cand = int(payload.get("cand", 0))
    config = payload.get("config", {})
    hyper = payload.get("hyper", {})

    # Apply hyperparams into config (this must match parent behavior).
    config = dict(config)
    config.update(hyper)

    # Normalize boolean-like params that are commonly encoded as 0/1 by optimizers.
    if "positional_encoding" in config:
        config["positional_encoding"] = _coerce_bool(config.get("positional_encoding"), default=False)
    if "use_temporal_features" in config:
        config["use_temporal_features"] = _coerce_bool(config.get("use_temporal_features"), default=True)
    if "add_window_stats" in config:
        config["add_window_stats"] = _coerce_bool(config.get("add_window_stats"), default=False)
    if "add_multi_scale_returns" in config:
        config["add_multi_scale_returns"] = _coerce_bool(config.get("add_multi_scale_returns"), default=False)

    # Convert activation from GA integer encoding [0..7] to string name.
    if "activation" in config:
        act_val = config["activation"]
        if isinstance(act_val, (int, float)):
            act_idx = int(round(act_val))
            act_idx = max(0, min(act_idx, len(ACTIVATION_INDEX_TO_NAME) - 1))
            config["activation"] = ACTIVATION_INDEX_TO_NAME[act_idx]

    # Convert encoding params from int to string name (safety: already done by to_hyper_dict).
    _ENCODING_NAMES = ["none", "sincos", "onehot"]
    for _enc_key in ("hod_encoding", "dow_encoding", "moy_encoding"):
        if _enc_key in config and isinstance(config[_enc_key], (int, float)):
            _ei = max(0, min(int(round(config[_enc_key])), len(_ENCODING_NAMES) - 1))
            config[_enc_key] = _ENCODING_NAMES[_ei]

    # Convert loss_type from int to string name.
    _LOSS_NAMES = ["mae", "trend_sigma", "pearson_structural", "soft_dtw", "combined_diff"]
    if "loss_type" in config and isinstance(config["loss_type"], (int, float)):
        _li = max(0, min(int(round(config["loss_type"])), len(_LOSS_NAMES) - 1))
        config["loss_type"] = _LOSS_NAMES[_li]

    out = {
        "ok": False,
        "fitness": float("inf"),
        "naive_mae": None,
        "val_mae": None,
        "train_mae": None,
        "train_naive_mae": None,
        "test_mae": None,
        "test_naive_mae": None,
        "model_summary": None,
        "error": None,
    }
    try:
        fitness, naive_mae = evaluate_candidate(config=config, hyper=hyper, gen=gen, cand=cand)
        extra = getattr(evaluate_candidate, "last_metrics", {}) or {}
        out.update(
            {
                "ok": True,
                "fitness": float(fitness),
                "naive_mae": naive_mae,
                "val_mae": extra.get("val_mae"),
                "train_mae": extra.get("train_mae"),
                "train_naive_mae": extra.get("train_naive_mae"),
                "test_mae": extra.get("test_mae"),
                "test_naive_mae": extra.get("test_naive_mae"),
                "model_summary": extra.get("model_summary"),
            }
        )
    except Exception as e:
        out.update({"ok": False, "fitness": float("inf"), "naive_mae": None, "error": str(e)})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f)

    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
