#!/usr/bin/env python3
"""
main.py

Punto de entrada de la aplicación de predicción de EUR/USD. Este script orquesta:
    - La carga y fusión de configuraciones (CLI, archivos locales y remotos).
    - La inicialización de los plugins: Ioin, Optimizer, Pipeline y Preprocessor.
    - La selección entre ejecutar la optimización de hiperparámetros o entrenar y evaluar directamente.
    - El guardado de la configuración resultante de forma local y/o remota.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Quiet mode: suppress verbose output when PREDICTOR_QUIET=1 or --quiet flag
# Only allows ERROR/WARN/final-metric lines through. Progress bars are killed
# by setting verbose=0 on model.fit (handled in common/base.py).
# ---------------------------------------------------------------------------
import builtins
_original_print = builtins.print

def _quiet_print(*args, **kwargs):
    """Filtered print that only passes through important messages."""
    if args:
        msg = str(args[0])
        # Always allow errors, warnings, and final metrics
        _pass = any(kw in msg.upper() for kw in [
            'ERROR', 'WARN', 'EXCEPTION', 'TRACEBACK', 'FATAL',
            'FINAL', 'BEST VAL', 'TEST MAE', 'VAL MAE', 'RESULT',
            'IMPROVEMENT', 'VERDICT', 'SUMMARY',
        ])
        if _pass:
            _original_print(*args, **kwargs)
        return
    _original_print(*args, **kwargs)

if os.environ.get('PREDICTOR_QUIET', '0') == '1' or '--quiet' in sys.argv:
    builtins.print = _quiet_print
import json
import pandas as pd
from typing import Any, Dict
from pathlib import Path

from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log
)
from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

# Se asume que los siguientes plugins se cargan desde sus respectivos namespaces:
# - ioin.plugins
# - optimizer.plugins
# - pipeline.plugins
# - preprocessor.plugins


def _repo_root() -> Path:
    # main.py -> app/ -> <repo_root>
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(p: Any) -> str | None:
    if not p:
        return None
    try:
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        return str((_repo_root() / pp).resolve())
    except Exception:
        return str(p)


def _ensure_csv_header(path: str, header_line: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header_line.rstrip("\n") + "\n")
            f.flush()
            os.fsync(f.fileno())


def _validate_logging_config(config: Dict[str, Any]) -> None:
    """Fail-fast-ish validation so remote runs never silently produce 'no logs'."""
    mem_log = _resolve_repo_path(config.get("memory_log_file"))
    opt_log = _resolve_repo_path(config.get("optimizer_resource_log_file"))
    if mem_log:
        config["memory_log_file"] = mem_log
    if opt_log:
        config["optimizer_resource_log_file"] = opt_log

    print(
        "[LOGGING_CONFIG] "
        f"memory_log_file={config.get('memory_log_file')} "
        f"optimizer_resource_log_file={config.get('optimizer_resource_log_file')} "
        f"memory_log_gpu={config.get('memory_log_gpu')} memory_log_gc={config.get('memory_log_gc')} "
        f"max_rss_gb={config.get('max_rss_gb')} max_rss_mb={config.get('max_rss_mb')}"
    )

    # Ensure files are writable and have headers (so tail/grep always works).
    try:
        if mem_log:
            _ensure_csv_header(
                mem_log,
                "ts,epoch,tag,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,gc0,gc1,gc2",
            )
    except Exception as e:
        print(f"[LOGGING_CONFIG] WARN: cannot initialize memory_log_file: {e}")

    try:
        if opt_log:
            _ensure_csv_header(
                opt_log,
                "ts,stage,generation,candidate,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,extra",
            )
    except Exception as e:
        print(f"[LOGGING_CONFIG] WARN: cannot initialize optimizer_resource_log_file: {e}")

def main():
    """
    Orquesta la ejecución completa del sistema, incluyendo la optimización (si se configura)
    y la ejecución del pipeline completo (preprocesamiento, entrenamiento, predicción y evaluación).
    """
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    # ------------------------------------------------------------------
    # TensorFlow memory safety (must run BEFORE any TF import in plugins)
    # ------------------------------------------------------------------
    # Helps prevent long-run fragmentation and makes GPU allocation behavior
    # consistent across plugins (CNN was attempting this too late).
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    try:
        import tensorflow as tf  # noqa: WPS433 (runtime import intentional)

        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # If TF already initialized somewhere, we can't change this.
                # Keep going; env var above still helps in many setups.
                pass
        if gpus:
            print(f"TensorFlow GPU memory growth configured for {len(gpus)} GPU(s).")
    except Exception as e:
        print(f"INFO: TensorFlow memory configuration skipped: {e}")

    print("Loading default configuration...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()

    file_config: Dict[str, Any] = {}
    # Carga remota de configuración si se solicita
    if args.remote_load_config:
        try:
            file_config = remote_load_config(args.remote_load_config, args.username, args.password)
            print(f"Loaded remote config: {file_config}")
        except Exception as e:
            print(f"Failed to load remote configuration: {e}")
            sys.exit(1)

    # Carga local de configuración si se solicita
    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local config: {file_config}")
        except Exception as e:
            print(f"Failed to load local configuration: {e}")
            sys.exit(1)

    # Primera fusión de la configuración (sin parámetros específicos de plugins)
    print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)

    # Selección del plugins
    if not cli_args.get('predictor_plugin'):
        cli_args['predictor_plugin'] = config.get('predictor_plugin', 'default_predictor')
    plugin_name = config.get('predictor_plugin', 'default_predictor')
    
    
    # --- CARGA DE PLUGINS ---
    # Carga del Ioin Plugin
    print(f"Loading Ioin Plugin: {plugin_name}")
    try:
        predictor_class, _ = load_plugin('ioin.plugins', plugin_name)
        predictor_plugin = predictor_class(config)
        predictor_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Ioin Plugin '{plugin_name}': {e}")
        sys.exit(1)

    # Carga del Optimizer Plugin (por defecto, se usa el de DEAP)
    # Selección del plugin si no se especifica
    plugin_name = config.get('optimizer_plugin', 'default_optimizer')
    print(f"Loading Plugin ..{plugin_name}")

    try:
        optimizer_class, _ = load_plugin('optimizer.plugins', plugin_name)
        optimizer_plugin = optimizer_class()
        optimizer_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Optimizer Plugin: {e}")
        sys.exit(1)

    # Carga del Pipeline Plugin (orquestador del flujo de entrenamiento y evaluación)
    plugin_name = config.get('pipeline_plugin', 'default_pipeline')
    print(f"Loading Plugin ..{plugin_name}")
    try:
        pipeline_class, _ = load_plugin('pipeline.plugins', plugin_name)
        pipeline_plugin = pipeline_class()
        pipeline_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Pipeline Plugin: {e}")
        sys.exit(1)

    # Carga del Target Plugin (para target and metrics calculation)
    plugin_name = config.get('target_plugin', 'default_target')
    print(f"Loading Plugin ..{plugin_name}")
    try:
        target_class, _ = load_plugin('target.plugins', plugin_name)
        target_plugin = target_class()
        target_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Target Plugin: {e}")
        sys.exit(1)

    # Carga del Preprocessor Plugin (para process_data, ventanas deslizantes y STL)
    plugin_name = config.get('preprocessor_plugin', 'default_preprocessor')
    print(f"Loading Plugin ..{plugin_name}")
    try:
        preprocessor_class, _ = load_plugin('preprocessor.plugins', plugin_name)
        preprocessor_plugin = preprocessor_class()
        preprocessor_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to load or initialize Preprocessor Plugin: {e}")
        sys.exit(1)

    # fusión de configuración, integrando parámetros específicos de plugin ioin
    print("Merging configuration with CLI arguments and unknown args (second pass, with plugin params)...")
    config = merge_config(config, predictor_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    # fusión de configuración, integrando parámetros específicos de plugin optimizer
    config = merge_config(config, optimizer_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    # fusión de configuración, integrando parámetros específicos de plugin pipeline
    config = merge_config(config, pipeline_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    # fusión de configuración, integrando parámetros específicos de plugin target
    config = merge_config(config, target_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)
    # fusión de configuración, integrando parámetros específicos de plugin preprocessor
    config = merge_config(config, preprocessor_plugin.plugin_params, {}, file_config, cli_args, unknown_args_dict)

    # Validate logging destinations after final config merge.
    _validate_logging_config(config)
    

    # --- DECISIÓN DE EJECUCIÓN ---
    if config.get('use_optimizer', False) and not config.get('load_model', False):
        print("Running hyperparameter optimization with Optimizer Plugin...")
        try:
            optimal_params = optimizer_plugin.optimize(predictor_plugin, preprocessor_plugin, config)
            optimizer_output_file = config.get("optimizer_output_file", "optimizer_output.json")
            with open(optimizer_output_file, "w") as f:
                json.dump(optimal_params, f, indent=4)
            print(f"Optimized parameters saved to {optimizer_output_file}.")
            config.update(optimal_params)
        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
            sys.exit(1)
    else:
        if not config.get('use_optimizer', False):
            print("Skipping hyperparameter optimization.")
        print("Running prediction pipeline...")

    # Pipeline Plugin orchestrates preprocessing, training (or model loading), evaluation
    pipeline_plugin.run_prediction_pipeline(
        config,
        predictor_plugin,
        preprocessor_plugin,
        target_plugin
    )
        
    # Guardado de la configuración local y remota
    if config.get('save_config'):
        try:
            save_config(config, config['save_config'])
            print(f"Configuration saved to {config['save_config']}.")
        except Exception as e:
            print(f"Failed to save configuration locally: {e}")

    if config.get('remote_save_config'):
        print(f"Remote saving configuration to {config['remote_save_config']}")
        try:
            remote_save_config(config, config['remote_save_config'], config.get('username'), config.get('password'))
            print("Remote configuration saved.")
        except Exception as e:
            print(f"Failed to save configuration remotely: {e}")

if __name__ == "__main__":
    main()
