#!/usr/bin/env python
"""
Default Optimizer Plugin

Este plugin utiliza algoritmos genéticos (DEAP) para optimizar los hiperparámetros
del Ioin Plugin. La función optimize realiza una búsqueda en el espacio de hiperparámetros
definidos y retorna un diccionario con los parámetros óptimos encontrados.

Se asume que:
  - Los hiperparámetros a optimizar están definidos en "hyperparameter_bounds".
  - Algunos parámetros deben ser tratados como enteros (por ejemplo, 'num_layers' o 'early_patience').

Nota: Se utiliza un número reducido de epochs para la evaluación en el proceso de optimización.
"""

import random
import numpy as np
import time
import json
import gc
import tensorflow as tf
import os
import sys
import subprocess
import tempfile
from collections import deque
import pty
import select
from pathlib import Path
from deap import base, creator, tools, algorithms

# Import optimizer helper modules
from .modules.incremental_logic import setup_incremental_optimization, adjust_params_for_resume
from .modules.resume_operations import load_resume_checkpoint, load_champion_parameters, save_resume_checkpoint
from .modules.genome_operations import validate_genome_bounds
from .modules.meta_logger import initialize_meta_log, log_candidate_evaluation, get_meta_log_path
from app.plugin_loader import load_plugin

from predictor_plugins.common.callbacks import capture_resource_snapshot

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


def _repo_root() -> Path:
    # default_optimizer.py -> optimizer_plugins/ -> <repo_root>
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

class Plugin:
    # Parámetros por defecto del optimizador.
    plugin_params = {
        "population_size": 20,
        "n_generations": 10,
        "cxpb": 0.5,      # Probabilidad de cruce.
        "mutpb": 0.2,     # Probabilidad de mutación.
        "optimization_patience": 3, # Paciencia para early stopping
        # Espacio de hiperparámetros a optimizar, con sus límites.
        "hyperparameter_bounds": {
            "learning_rate": (1e-5, 1e-2),
            "num_layers": (1, 5),
            "layer_size": (16, 256),
            "positional_encoding": (0, 1),
            "l2_reg": (1e-7, 1e-3),
            "mmd_lambda": (1e-5, 1e-2),
            "early_patience": (10, 100)
        }
    }
    plugin_debug_vars = ["population_size", "n_generations", "cxpb", "mutpb", "optimization_patience"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        """
        Actualiza los parámetros del optimizador combinando los parámetros específicos con la configuración global.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Devuelve información de debug de los parámetros relevantes del optimizador.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Agrega la información de debug del optimizador al diccionario proporcionado.
        """
        debug_info.update(self.get_debug_info())

    def optimize(self, predictor_plugin, preprocessor_plugin, config):
        """
        Realiza la optimización de hiperparámetros utilizando algoritmos genéticos (DEAP).

        Args:
            predictor_plugin: Plugin encargado del ioin, que se evaluará con los hiperparámetros.
            preprocessor_plugin: Plugin encargado del preprocesamiento de datos.
            config (dict): Configuración global.

        Returns:
            dict: Diccionario con los hiperparámetros óptimos.
        """
        # Fix: Ensure 'plugin' key matches 'predictor_plugin' if present, as 'plugin' might be stale from defaults
        if "predictor_plugin" in config:
            config["plugin"] = config["predictor_plugin"]
        elif "plugin" not in config:
            config["plugin"] = "default_predictor"

        # Load Target Plugin to pass to preprocessor
        target_plugin_name = config.get('target_plugin', 'default_target')
        try:
            target_class, _ = load_plugin('target.plugins', target_plugin_name)
            target_plugin = target_class()
            target_plugin.set_params(**config)
        except Exception as e:
            print(f"Failed to load Target Plugin inside optimizer: {e}")
            raise e

        # Extraer el espacio de búsqueda de hiperparámetros.
        # Use config directly to ensure we get the merged values from file_config
        full_bounds = config.get("hyperparameter_bounds", self.params.get("hyperparameter_bounds"))
        
        # --- INCREMENTAL OPTIMIZATION SETUP ---
        incremental_enabled, increment_size, all_param_keys, hyper_keys, total_stages, meta_mode = setup_incremental_optimization(full_bounds, config)
        
        # Initialize stage tracking variables (used for progress display)
        incremental_stage = 1
        current_meta_stage = 1 if meta_mode else None
        
        # Initialize meta-training log if meta mode is enabled
        if meta_mode:
            meta_log_path = _resolve_repo_path(get_meta_log_path(config))
            initialize_meta_log(meta_log_path, config)
            print(f"[META-OPTIMIZATION] Initialized meta-training log at: {meta_log_path}")
        
        bounds = {k: full_bounds[k] for k in hyper_keys}
        
        # Determine parameter types based on bounds (int vs float)
        param_types = {}
        lower_bounds = []
        upper_bounds = []
        
        for key in hyper_keys:
            low, up = bounds[key]
            lower_bounds.append(low)
            upper_bounds.append(up)
            # Heuristic: if both bounds are int, treat as int
            if isinstance(low, int) and isinstance(up, int):
                param_types[key] = 'int'
            else:
                param_types[key] = 'float'

        # Configuración de DEAP: se define el individuo y la función objetivo.
        # Fix: Check if classes already exist to avoid RuntimeError on re-run
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Generador de atributo: maneja int y float
        def make_attr(low, up, ptype):
            if ptype == 'int':
                return random.randint(low, up)
            else:
                return random.uniform(low, up)

        # Registrar atributos para cada hiperparámetro.
        for i, key in enumerate(hyper_keys):
            low = lower_bounds[i]
            up = upper_bounds[i]
            ptype = param_types[key]
            toolbox.register(f"attr_{key}", make_attr, low, up, ptype)

        # Definir el individuo como una lista de hiperparámetros.
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         [toolbox.__getattribute__(f"attr_{key}") for key in hyper_keys], n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Función de evaluación: construye y entrena el modelo con los hiperparámetros dados,
        # devolviendo la pérdida de validación del último epoch.
        
        # Global counters for progress tracking
        self.eval_counter = 0
        self.total_eval_counter = 0
        self.current_gen = 0
        # Fitness direction: binary/direction targets use higher-is-better
        _is_binary = config.get("target_plugin") in ("binary_target", "direction_target")
        _hib = config.get("higher_is_better", _is_binary)
        self._higher_is_better = _hib
        _worst_fitness = float("-inf") if _hib else float("inf")
        self.best_fitness_so_far = _worst_fitness
        self.patience_counter = 0
        self.best_val_mae_so_far = None  # ACTUAL validation MAE
        self.best_naive_mae_so_far = None  # Validation naive MAE
        self.best_test_mae_so_far = None
        self.best_test_naive_mae_so_far = None
        self.best_train_mae_so_far = None
        self.best_train_naive_mae_so_far = None
        self.best_params_so_far = {}  # Best champion hyper_dict for stage-end broadcast
        # Tracks the FITNESS to beat for the *current generation* (used by GA early stopping).
        self.best_at_gen_start = _worst_fitness

        def _extract_max_horizon_array(y_any, predicted_horizons, max_horizon):
            if isinstance(y_any, dict):
                key = f"output_horizon_{max_horizon}"
                return np.asarray(y_any.get(key))
            if isinstance(y_any, list):
                max_h_idx = predicted_horizons.index(max_horizon)
                return np.asarray(y_any[max_h_idx])
            return np.asarray(y_any)

        def _compute_split_metrics_max_h(*, split: str, preds_list, y_any, baseline_any, cfg) -> tuple[float, float]:
            """Return (mae, naive_mae) for max horizon using pipeline-parity formulas."""
            predicted_horizons = cfg.get("predicted_horizons", [1])
            max_horizon = max(predicted_horizons) if predicted_horizons else 1
            max_h_idx = predicted_horizons.index(max_horizon) if predicted_horizons else 0
            from pipeline_plugins.stl_norm import denormalize, denormalize_returns

            y_h = _extract_max_horizon_array(y_any, predicted_horizons, max_horizon).reshape(-1)
            p_h = np.asarray(preds_list[max_h_idx]).reshape(-1)
            n = min(len(y_h), len(p_h))
            if baseline_any is not None:
                n = min(n, len(np.asarray(baseline_any).reshape(-1)))
            if n <= 0:
                print(f"WARN: {split} metrics could not be computed (empty arrays).")
                return (float("inf"), float("inf"))
            y_h = y_h[:n]
            p_h = p_h[:n]
            
            # CORRECT MAE CALCULATION:
            # Always denormalize both predictions and targets to REAL PRICE SPACE first.
            # This handles both linear scaling and log1p (via expm1 in denormalize).
            # Old method denormalize_returns(p-y) is invalid for log-space differences.
            real_p = denormalize(p_h, cfg)
            real_y = denormalize(y_h, cfg)
            mae = float(np.mean(np.abs(real_p - real_y)))

            naive_mae = float("inf")
            if baseline_any is not None:
                baseline_h = np.asarray(baseline_any).reshape(-1)[:n]
                # baseline_h is in the same space as y_h (normalized/log).
                # Denormalize it to get real price baseline.
                real_baseline = denormalize(baseline_h, cfg)
                naive_mae = float(np.mean(np.abs(real_baseline - real_y)))
            return (mae, naive_mae)

        def _ensure_2d_targets(y):
            """Match the STL pipeline behavior: y arrays are column vectors (N,1)."""
            if isinstance(y, dict):
                out = {}
                for k, v in y.items():
                    arr = np.asarray(v)
                    out[k] = arr.reshape(-1, 1).astype(np.float32)
                return out
            return y

        def _array_stats(arr):
            arr = np.asarray(arr).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return {"n": 0}
            return {
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        def _json_sanitize(obj):
            """Recursively convert numpy/tensor types into JSON-serializable Python types."""
            # TensorFlow tensors
            try:
                if isinstance(obj, tf.Tensor):
                    obj = obj.numpy()
            except Exception:
                pass

            # Skip non-serializable types (callables, sets, etc.)
            if callable(obj) and not isinstance(obj, type):
                return None
            if isinstance(obj, set):
                return list(obj)

            # Numpy scalars
            if isinstance(obj, np.generic):
                return obj.item()

            # Numpy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()

            if isinstance(obj, dict):
                # Skip keys with non-serializable values
                return {str(k): _json_sanitize(v) for k, v in obj.items()
                        if k != "optimization_callbacks" and k != "_non_serializable_keys"}
            if isinstance(obj, (list, tuple)):
                return [_json_sanitize(v) for v in obj]

            # Plain Python types are fine (including None)
            return obj

        def _atomic_json_dump(path: str, payload: dict) -> None:
            """Write JSON atomically to avoid corrupt/truncated files on crashes/OOM."""
            tmp_path = f"{path}.tmp"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(_json_sanitize(payload), f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)

        def _append_resource_row(stage: str, gen: int | None = None, cand: int | None = None, extra: dict | None = None) -> None:
            log_path = _resolve_repo_path(config.get("optimizer_resource_log_file"))
            if not log_path:
                return
            try:
                snap = capture_resource_snapshot(include_gpu=bool(config.get("memory_log_gpu", True)), include_gc=bool(config.get("memory_log_gc", False)))
                os.makedirs(os.path.dirname(str(log_path)) or ".", exist_ok=True)
                new_file = not os.path.exists(str(log_path))
                with open(str(log_path), "a", encoding="utf-8") as f:
                    if new_file:
                        f.write(
                            "ts,stage,generation,candidate,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,extra\n"
                        )
                    extra_json = ""
                    if extra:
                        try:
                            extra_json = json.dumps(_json_sanitize(extra), separators=(",", ":"))
                        except Exception:
                            extra_json = ""
                    f.write(
                        f"{snap.ts:.3f},{stage},{gen if gen is not None else ''},{cand if cand is not None else ''},"
                        f"{snap.rss_kb if snap.rss_kb is not None else ''},{snap.hwm_kb if snap.hwm_kb is not None else ''},"
                        f"{snap.gpu_current_bytes if snap.gpu_current_bytes is not None else ''},{snap.gpu_peak_bytes if snap.gpu_peak_bytes is not None else ''},"
                        f"{extra_json}\n"
                    )
                    f.flush()
            except Exception as e:
                print(f"WARN: optimizer resource logging failed: {e}")
        
        def eval_individual(individual):
            # Note: eval_counter is incremented just below; use a stable candidate id for printing/logging.
            candidate_id = int(self.eval_counter + 1)
            _append_resource_row("candidate_start", gen=int(self.current_gen or 0), cand=candidate_id)

            # CRITICAL FIX: Clear Keras session and garbage collect to prevent OOM
            if hasattr(predictor_plugin, 'model'):
                del predictor_plugin.model
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Determinism: Set seeds before every evaluation
            # This is crucial for "exact same validation MAE" when re-evaluating the champion.
            seed_val = 42
            random.seed(seed_val)
            np.random.seed(seed_val)
            tf.random.set_seed(seed_val)

            # Optional: recreate ioin plugin per candidate to avoid hidden state accumulation
            predictor_for_eval = predictor_plugin
            if bool(config.get("optimizer_recreate_predictor_each_eval", False)):
                try:
                    predictor_for_eval = predictor_plugin.__class__(config)
                    predictor_for_eval.set_params(**config)
                except Exception as e:
                    print(f"WARN: could not recreate ioin plugin; using shared instance: {e}")
                    predictor_for_eval = predictor_plugin

            self.eval_counter += 1
            self.total_eval_counter += 1
            
            # Mapear el individuo a un diccionario de hiperparámetros.
            hyper_dict = {}
            for i, key in enumerate(hyper_keys):
                value = individual[i]
                ptype = param_types[key]
                
                # Specific handling for requested parameters
                if key == "use_log1p_features":
                    # 0 -> None, 1 -> ["typical_price"]
                    if isinstance(value, list):
                        hyper_dict[key] = value
                    else:
                        val_int = int(round(value))
                        hyper_dict[key] = ["typical_price"] if val_int == 1 else None
                elif key == "positional_encoding":
                    val_int = int(round(value))
                    hyper_dict[key] = bool(val_int)
                elif key == "activation":
                    act_idx = int(round(value))
                    act_idx = max(0, min(act_idx, len(ACTIVATION_INDEX_TO_NAME) - 1))
                    hyper_dict[key] = ACTIVATION_INDEX_TO_NAME[act_idx]
                elif ptype == 'int':
                    hyper_dict[key] = int(round(value))
                else:
                    hyper_dict[key] = value

            # Build stage context string for display
            if meta_mode and current_meta_stage:
                try:
                    from .modules.meta_stages import get_stage_info, get_total_stages
                    _stage_name, _ = get_stage_info(current_meta_stage, config)
                    _stage_ctx = f"Stage {current_meta_stage}/{get_total_stages(config)} [{_stage_name}] | "
                except Exception:
                    _stage_ctx = f"Stage {current_meta_stage} | "
            elif incremental_enabled:
                _stage_ctx = f"Stage {incremental_stage}/{total_stages} | "
            else:
                _stage_ctx = ""

            print(f"\n--- Evaluating Candidate {self.eval_counter}/{population_size} | {_stage_ctx}Gen {self.current_gen}/{self.end_gen - 1} | Params: {len(hyper_keys)} | Total Evals: {self.total_eval_counter} ---")
            print(f"Params: {hyper_dict}")

            # Combinar los hiperparámetros con la configuración actual.
            new_config = config.copy()
            new_config.update(hyper_dict)

            # -----------------------------------------------------------------
            # CRITICAL: Optional subprocess isolation per candidate.
            # This prevents host-RAM accumulation across candidates from killing
            # the main optimizer process (Linux OOM killer).
            # -----------------------------------------------------------------
            if bool(new_config.get("optimizer_isolate_candidate_process", True)):
                # Subprocess worker will compute train/val/test metrics for this candidate.
                # Resolve log paths deterministically to the ioin repo root.
                for k in ("memory_log_file", "optimizer_resource_log_file", "batch_memory_log_file"):
                    if new_config.get(k):
                        new_config[k] = _resolve_repo_path(new_config.get(k))

                # Tag epoch/batch logs for correlation.
                new_config.setdefault(
                    "memory_log_tag",
                    f"ga_gen{int(self.current_gen or 0)}_cand{int(self.eval_counter)}",
                )

                _append_resource_row(
                    "subprocess_start",
                    gen=int(self.current_gen or 0),
                    cand=int(self.eval_counter),
                    extra={"params": hyper_dict},
                )

                with tempfile.TemporaryDirectory(prefix="ga_cand_") as td:
                    in_path = os.path.join(td, "input.json")
                    out_path = os.path.join(td, "output.json")
                    model_path = os.path.join(td, "candidate_model.keras")
                    new_config["_doin_model_save_path"] = model_path
                    with open(in_path, "w", encoding="utf-8") as f:
                        # Strip non-serializable keys (e.g. DOIN callbacks)
                        _serial_config = _json_sanitize(new_config)
                        json.dump(
                            {
                                "gen": int(self.current_gen or 0),
                                "cand": int(self.eval_counter),
                                "config": _serial_config,
                                "hyper": hyper_dict,
                            },
                            f,
                        )

                    env = os.environ.copy()
                    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "1")
                    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
                    env.setdefault("PYTHONUNBUFFERED", "1")

                    cmd = [
                        sys.executable,
                        "-u",
                        "-m",
                        "optimizer_plugins.candidate_worker",
                        "--input",
                        in_path,
                        "--output",
                        out_path,
                    ]

                    try:
                        # Run worker under a pseudo-terminal so tqdm/progress-bars render correctly.
                        master_fd, slave_fd = pty.openpty()
                        p = subprocess.Popen(
                            cmd,
                            env=env,
                            stdin=slave_fd,
                            stdout=slave_fd,
                            stderr=slave_fd,
                            close_fds=True,
                        )
                        os.close(slave_fd)

                        tail_chunks: deque[bytes] = deque(maxlen=256)
                        while True:
                            # Read available output without blocking indefinitely.
                            r, _, _ = select.select([master_fd], [], [], 0.2)
                            if master_fd in r:
                                try:
                                    data = os.read(master_fd, 4096)
                                except OSError:
                                    data = b""
                                if not data:
                                    break
                                try:
                                    sys.stdout.buffer.write(data)
                                    sys.stdout.buffer.flush()
                                except Exception:
                                    pass
                                tail_chunks.append(data)

                            if p.poll() is not None:
                                # Drain remaining PTY output.
                                continue

                        returncode = p.wait()
                        try:
                            os.close(master_fd)
                        except Exception:
                            pass
                    except Exception as e:
                        _append_resource_row(
                            "subprocess_spawn_failed",
                            gen=int(self.current_gen or 0),
                            cand=int(self.eval_counter),
                            extra={"error": str(e)},
                        )
                        print(f"CRITICAL: Worker spawn failed: {e}")
                        individual.naive_mae = float("inf")
                        _append_resource_row("candidate_end", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
                        return (float("inf"),)

                    # Worker failure (including OOM-kill) must not kill the optimizer.
                    if returncode != 0:
                        stdout_tail = b"".join(list(tail_chunks))[-8000:].decode(errors="replace")
                        print(f"CRITICAL: Worker failed (returncode={returncode}). Tail of stdout:\n{stdout_tail}\n--- End of worker stdout ---")
                        _append_resource_row(
                            "subprocess_failed",
                            gen=int(self.current_gen or 0),
                            cand=int(self.eval_counter),
                            extra={"returncode": returncode, "stdout_tail": stdout_tail},
                        )
                        fitness = float("inf")
                        naive_mae = float("inf")
                        try:
                            if os.path.exists(out_path):
                                payload = json.load(open(out_path, "r", encoding="utf-8"))
                                fitness = float(payload.get("fitness", float("inf")))
                                nm = payload.get("naive_mae")
                                naive_mae = float(nm) if nm is not None else float("inf")
                        except Exception:
                            pass
                        individual.naive_mae = naive_mae
                        _append_resource_row("candidate_end", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
                        return (fitness,)

                    payload = json.load(open(out_path, "r", encoding="utf-8"))
                    fitness = float(payload.get("fitness", float("inf")))
                    nm = payload.get("naive_mae")
                    naive_mae = float(nm) if nm is not None else float("inf")
                    vm = payload.get("val_mae")
                    val_mae = float(vm) if vm is not None else float("inf")
                    test_mae = float(payload.get("test_mae", float("inf")))
                    tnm = payload.get("test_naive_mae")
                    test_naive_mae = float(tnm) if tnm is not None else float("inf")
                    train_mae = float(payload.get("train_mae", float("inf")))
                    trnm = payload.get("train_naive_mae")
                    train_naive_mae = float(trnm) if trnm is not None else float("inf")
                    individual.naive_mae = naive_mae
                    individual.val_mae = val_mae
                    individual.test_mae = test_mae
                    individual.test_naive_mae = test_naive_mae
                    individual.train_mae = train_mae
                    individual.train_naive_mae = train_naive_mae
                    individual.model_summary = payload.get("model_summary") or ""
                    individual.hyper_dict = hyper_dict

                    is_new_champion = False
                    _is_new_best = (fitness > float(self.best_fitness_so_far)) if _hib else (fitness < float(self.best_fitness_so_far))
                    if np.isfinite(fitness) and _is_new_best:
                        self.best_fitness_so_far = float(fitness)
                        self.best_val_mae_so_far = float(val_mae) if np.isfinite(val_mae) else self.best_val_mae_so_far
                        self.best_naive_mae_so_far = float(naive_mae) if np.isfinite(naive_mae) else self.best_naive_mae_so_far
                        self.best_test_mae_so_far = float(test_mae) if np.isfinite(test_mae) else self.best_test_mae_so_far
                        self.best_test_naive_mae_so_far = float(test_naive_mae) if np.isfinite(test_naive_mae) else self.best_test_naive_mae_so_far
                        self.best_train_mae_so_far = float(train_mae) if np.isfinite(train_mae) else self.best_train_mae_so_far
                        self.best_train_naive_mae_so_far = float(train_naive_mae) if np.isfinite(train_naive_mae) else self.best_train_naive_mae_so_far
                        is_new_champion = True
                        self.best_params_so_far = hyper_dict.copy()

                        # Save Champion Parameters Immediately
                        params_file = config.get("optimization_parameters_file", "optimization_parameters.json")
                        resolved_params_file = _resolve_repo_path(params_file) or params_file
                        try:
                            _atomic_json_dump(resolved_params_file, hyper_dict)
                            print(f"  [CHAMPION] Parameters saved to {resolved_params_file}")
                        except Exception as e:
                            print(f"  [CHAMPION] Failed to save parameters: {e}")

                        # --- CALLBACK: on_new_champion (migration OUT) ---
                        _cb_new_champ = _opt_callbacks.get("on_new_champion")
                        if _cb_new_champ:
                            try:
                                # Read trained model for DOIN evaluator verification
                                _model_b64 = None
                                if os.path.exists(model_path):
                                    import base64
                                    with open(model_path, "rb") as _mf:
                                        _model_b64 = base64.b64encode(_mf.read()).decode("ascii")
                                _champ_metrics = {
                                    "fitness": fitness,
                                    "val_mae": val_mae, "val_naive_mae": naive_mae,
                                    "train_mae": train_mae, "train_naive_mae": train_naive_mae,
                                    "test_mae": test_mae, "test_naive_mae": test_naive_mae,
                                    "_model_b64": _model_b64,
                                }
                                _champ_stage = {
                                    "stage": current_meta_stage if meta_mode else incremental_stage,
                                    "total_stages": total_stages,
                                    "generation": int(self.current_gen or 0),
                                    "candidate": int(self.eval_counter),
                                    "total_candidates_evaluated": int(self.total_eval_counter),
                                }
                                _cb_new_champ(hyper_dict, fitness, _champ_metrics, int(self.current_gen or 0), _champ_stage)
                            except Exception as _cb_err:
                                print(f"  [MIGRATION OUT] Champion broadcast error: {_cb_err}")

                    _append_resource_row(
                        "subprocess_ok",
                        gen=int(self.current_gen or 0),
                        cand=int(self.eval_counter),
                        extra={"fitness": fitness, "naive_mae": naive_mae},
                    )
                    _append_resource_row("candidate_end", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
                    # Build stage context for result display
                    if meta_mode and current_meta_stage:
                        try:
                            from .modules.meta_stages import get_stage_info, get_total_stages
                            _rs_name, _rs_desc = get_stage_info(current_meta_stage, config)
                            _rs_ctx = f"Stage {current_meta_stage}/{get_total_stages(config)} [{_rs_name}]"
                        except Exception:
                            _rs_ctx = f"Stage {current_meta_stage}"
                    elif incremental_enabled:
                        _rs_ctx = f"Stage {incremental_stage}/{total_stages}"
                    else:
                        _rs_ctx = "Standard"

                    _is_binary = config.get("target_plugin") in ("binary_target", "direction_target")
                    _lbl1 = "Accuracy" if _is_binary else "MAE maxH"
                    _lbl2 = "F1" if _is_binary else "Naive MAE maxH"
                    print(f"\n{'='*80}")
                    print(f"CANDIDATE RESULT | {_rs_ctx} | Gen {self.current_gen}/{self.end_gen - 1} | Candidate {self.eval_counter}/{population_size} | Total Evals: {self.total_eval_counter}")
                    print(f"Active Parameters ({len(hyper_keys)}): {', '.join(hyper_keys)}")
                    print(f"{'-'*80}")
                    print(
                        f"  TRAINING   -> {_lbl1}: {train_mae:.6f} | {_lbl2}: {train_naive_mae:.6f}\n"
                        f"  VALIDATION -> {_lbl1}: {val_mae:.6f} | {_lbl2}: {naive_mae:.6f}\n"
                        f"  TEST       -> {_lbl1}: {test_mae:.6f} | {_lbl2}: {test_naive_mae:.6f}\n"
                        f"  FITNESS (avg delta from naive): {fitness:.6f}"
                    )
                    champion_fitness = float(self.best_fitness_so_far) if self.best_fitness_so_far is not None else float("inf")
                    champion_val_mae = (
                        float(self.best_val_mae_so_far)
                        if self.best_val_mae_so_far is not None and np.isfinite(self.best_val_mae_so_far)
                        else float("inf")
                    )
                    champion_naive = (
                        float(self.best_naive_mae_so_far)
                        if self.best_naive_mae_so_far is not None and np.isfinite(self.best_naive_mae_so_far)
                        else float("inf")
                    )
                    champion_test_mae = (
                        float(self.best_test_mae_so_far)
                        if self.best_test_mae_so_far is not None and np.isfinite(self.best_test_mae_so_far)
                        else float("inf")
                    )
                    champion_test_naive = (
                        float(self.best_test_naive_mae_so_far)
                        if self.best_test_naive_mae_so_far is not None and np.isfinite(self.best_test_naive_mae_so_far)
                        else float("inf")
                    )
                    champion_train_mae = (
                        float(self.best_train_mae_so_far)
                        if self.best_train_mae_so_far is not None and np.isfinite(self.best_train_mae_so_far)
                        else float("inf")
                    )
                    champion_train_naive = (
                        float(self.best_train_naive_mae_so_far)
                        if self.best_train_naive_mae_so_far is not None and np.isfinite(self.best_train_naive_mae_so_far)
                        else float("inf")
                    )
                    print(f"{'-'*80}")
                    print(f"CHAMPION SO FAR{' | *** NEW CHAMPION ***' if is_new_champion else ''}")
                    print(
                        f"  TRAINING   -> {_lbl1}: {champion_train_mae:.6f} | {_lbl2}: {champion_train_naive:.6f}\n"
                        f"  VALIDATION -> {_lbl1}: {champion_val_mae:.6f} | {_lbl2}: {champion_naive:.6f}\n"
                        f"  TEST       -> {_lbl1}: {champion_test_mae:.6f} | {_lbl2}: {champion_test_naive:.6f}\n"
                        f"  FITNESS: {champion_fitness:.6f}"
                    )
                    print(f"{'-'*80}")
                    print(f"GA EARLY STOPPING: No-improve {self.patience_counter}/{patience} | Gen-start fitness to beat: {float(self.best_at_gen_start):.6f} | Global best: {float(self.best_fitness_so_far):.6f}")
                    print(f"{'='*80}")
                    return (fitness,)

            # Tag epoch-level resource logs so we can correlate with GA generation/candidate.
            new_config.setdefault("memory_log_tag", f"ga_gen{int(self.current_gen or 0)}_cand{int(self.eval_counter)}")

            # Resolve log paths deterministically to the ioin repo root.
            if new_config.get("memory_log_file"):
                new_config["memory_log_file"] = _resolve_repo_path(new_config.get("memory_log_file"))
            if new_config.get("optimizer_resource_log_file"):
                new_config["optimizer_resource_log_file"] = _resolve_repo_path(new_config.get("optimizer_resource_log_file"))

            # Print the effective logging destinations every candidate (critical for remote debugging).
            print(
                "[LOGGING] "
                f"memory_log_file={new_config.get('memory_log_file')} "
                f"optimizer_resource_log_file={new_config.get('optimizer_resource_log_file')} "
                f"memory_log_tag={new_config.get('memory_log_tag')} "
                f"max_rss_gb={new_config.get('max_rss_gb')} max_rss_mb={new_config.get('max_rss_mb')} "
                f"predict_batch_size={new_config.get('predict_batch_size')} disable_postfit_uncertainty={new_config.get('disable_postfit_uncertainty')}"
            )

            # CRITICAL: During GA optimization, avoid post-fit MC uncertainty passes.
            # The default BaseBayesianKerasPredictor does MC prediction over full train+val
            # after every fit, which can explode memory/time on large datasets.
            new_config.setdefault("disable_postfit_uncertainty", True)
            new_config.setdefault("mc_samples", 1)
            new_config.setdefault("predict_batch_size", new_config.get("batch_size", 32))

            # Obtener los datos de entrenamiento y validación usando el Preprocessor Plugin.
            _append_resource_row("before_preprocess", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
            datasets = preprocessor_plugin.run_preprocessing(target_plugin, new_config)
            _append_resource_row("after_preprocess", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
            
            # Handle tuple return from stl_preprocessor (datasets, params)
            if isinstance(datasets, tuple):
                datasets = datasets[0]

            # FIX: Extract dates and feature names for Prophet/Time-aware predictors (In-process)
            if "x_train_dates" in datasets:
                new_config["train_dates"] = datasets["x_train_dates"]
            if "x_val_dates" in datasets:
                new_config["val_dates"] = datasets["x_val_dates"]
            if "x_test_dates" in datasets:
                new_config["test_dates"] = datasets["x_test_dates"]
            if "feature_names" in datasets:
                new_config["feature_names"] = datasets["feature_names"]
                
            x_train, y_train = datasets["x_train"], datasets["y_train"]
            x_val, y_val = datasets["x_val"], datasets["y_val"]
            x_test, y_test = datasets.get("x_test"), datasets.get("y_test")
            baseline_train = datasets.get("baseline_train")
            baseline_val = datasets.get("baseline_val")
            baseline_test = datasets.get("baseline_test")

            # Match pipeline target shapes (N,1) for Keras.
            y_train = _ensure_2d_targets(y_train)
            y_val = _ensure_2d_targets(y_val)

            # FIX: Trim dates to match target lengths (Prophet requires exact length match)
            def _trim_dates(dates, y_dict):
                if dates is None or not isinstance(y_dict, dict):
                    return dates
                # Find min length of targets
                min_len = min(len(v) for v in y_dict.values())
                if len(dates) > min_len:
                    return dates[:min_len]
                return dates

            if "train_dates" in new_config:
                new_config["train_dates"] = _trim_dates(new_config["train_dates"], y_train)
            if "val_dates" in new_config:
                new_config["val_dates"] = _trim_dates(new_config["val_dates"], y_val)
            if "test_dates" in new_config:
                # y_test might be None or dict
                if y_test is not None:
                    y_test_fixed = _ensure_2d_targets(y_test)
                    new_config["test_dates"] = _trim_dates(new_config["test_dates"], y_test_fixed)

            # Construir y entrenar el modelo utilizando el Ioin Plugin.
            window_size = new_config.get("window_size")
            _append_resource_row("before_build_model", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
            if new_config["plugin"] in ["lstm", "cnn", "transformer", "ann", "mimo", "n_beats", "n_beats_b", "tft"]:
                # Handle 3D input for sequence models
                input_shape = (window_size, x_train.shape[2]) if len(x_train.shape) == 3 else (x_train.shape[1],)
                predictor_for_eval.build_model(input_shape=input_shape, x_train=x_train, config=new_config)
            else:
                predictor_for_eval.build_model(input_shape=x_train.shape[1], x_train=x_train, config=new_config)
            _append_resource_row("after_build_model", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
            
            # Calculate Naive MAE for the highest horizon
            # Naive prediction: use the last known value (t-1) as prediction for all future steps
            # For normalized data, this is often just 0 if using returns, or the last value if using prices.
            # Assuming standard time series where x_val contains history.
            # We need to know the target column index or assume it's the last one or specific one.
            # However, a simpler robust way for "Naive" in this context (often used in this repo) 
            # is Mean Absolute Error of (y_true[1:] - y_true[:-1]) for 1-step, but for N-step it varies.
            # The user defined Naive MAE as "using the current value as prediction".
            # We will calculate it using the last value of the input window vs the target.
            
            naive_mae = float("inf")
            try:
                # Get the highest horizon index
                predicted_horizons = new_config.get("predicted_horizons", [1])
                max_horizon = max(predicted_horizons)
                max_h_idx = predicted_horizons.index(max_horizon)
                
                # y_val is a dict for MIMO or list/array. Preprocessor usually returns dict for MIMO.
                # If it's a dict: {'output_horizon_1': ..., 'output_horizon_6': ...}
                # We need the target for the max horizon.
                
                y_true_max_h = None
                if isinstance(y_val, dict):
                    key = f"output_horizon_{max_horizon}"
                    if key in y_val:
                        y_true_max_h = y_val[key]
                elif isinstance(y_val, list):
                     if len(y_val) > max_h_idx:
                        y_true_max_h = y_val[max_h_idx]
                elif isinstance(y_val, np.ndarray):
                    # If single output or stacked
                    y_true_max_h = y_val
                
                if y_true_max_h is not None:
                    # Naive prediction: The last value in the input sequence x_val.
                    # x_val shape: (samples, window_size, features)
                    # We assume the target feature is at index 0 or we need to know which one.
                    # Config has "target_column". But x_val is numpy array.
                    # We'll assume the target is the first column (index 0) of the input if not specified,
                    # OR better, we use the "baseline" if available from preprocessor, but preprocessor returns x/y.
                    # Let's use the last value of the input window (index -1) of the first feature (index 0).
                    # This is a standard heuristic if we don't have column mapping here.
                    
                    # x_val[:, -1, 0] is the most recent value for each sample.
                    # We compare this against y_true_max_h.
                    
                    # Ensure shapes match
                    # preds_naive = x_val[:, -1, 0].reshape(-1, 1)
                    
                    # Calculate MAE
                    # naive_errors = np.abs(y_true_max_h - preds_naive)
                    # naive_mae = np.mean(naive_errors)
                    
                    # Use baseline_val if available (as per pipeline logic)
                    baseline_val = datasets.get("baseline_val")
                    if baseline_val is not None:
                        # Pipeline logic: denormalize(baseline) - denormalize(target)
                        # But here we are in normalized space for optimization fitness (val_loss).
                        # If we want to compare apples to apples with val_loss (which is usually MSE or MAE on normalized data),
                        # we should compute Naive MAE on normalized data too.
                        # Pipeline computes metrics on DENORMALIZED data.
                        # The user said: "Validation Naive MAE and for the validation MAE... verify that the MAE is claulated in the same exact way"
                        # The optimizer uses `history.history["val_loss"][-1]` as fitness.
                        # If the model loss is MSE/MAE on normalized data, then we should compute Naive MAE on normalized data.
                        # BUT, the pipeline prints Denormalized MAE.
                        # If the user sees "Validation MAE" in the pipeline output, that is Denormalized.
                        # If the user sees "Val Loss" in the optimizer output, that is Normalized (usually).
                        # The user complained "values are way too diferent".
                        # This confirms we are comparing Normalized Loss vs Denormalized Naive (or vice versa).
                        
                        # To fix this, we must calculate the Validation MAE exactly as the pipeline does:
                        # 1. Predict on x_val
                        # 2. Denormalize predictions and targets
                        # 3. Compute MAE
                        
                        # We cannot rely on history['val_loss'] if we want to match the pipeline's "Validation MAE".
                        # We must run a prediction step here.
                        pass # Logic moved below to "After training" block
            except Exception as e:
                print(f"Warning: Could not calculate Naive MAE pre-check: {e}")
                naive_mae = float("inf")

            try:
                # Para optimización, usar menos epochs.
                _append_resource_row("before_fit", gen=int(self.current_gen or 0), cand=int(self.eval_counter), extra={"params": hyper_dict})
                history, train_preds, _, val_preds, _ = predictor_for_eval.train(
                    x_train, y_train,
                    epochs=new_config.get("epochs", 10),
                    batch_size=new_config.get("batch_size", 32),
                    threshold_error=new_config.get("threshold_error", 0.001),
                    x_val=x_val, y_val=y_val, config=new_config
                )
                _append_resource_row("after_fit", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
                
                # --- Compute Metrics Exactly like Pipeline ---
                # We need to replicate compute_train_val_metrics logic for the max horizon
                
                # 1. Get predictions for max horizon
                # val_preds is a list of arrays (one per horizon)
                predicted_horizons = new_config.get("predicted_horizons", [1])
                max_horizon = max(predicted_horizons)
                max_h_idx = predicted_horizons.index(max_horizon)
                
                val_preds_h = val_preds[max_h_idx].flatten()
                
                # 2. Get targets for max horizon
                # y_val is dict or list. We need to extract the array.
                if isinstance(y_val, dict):
                    y_true_max_h = y_val[f"output_horizon_{max_horizon}"].flatten()
                elif isinstance(y_val, list):
                    y_true_max_h = y_val[max_h_idx].flatten()
                else:
                    y_true_max_h = y_val.flatten()
                
                # 3. Align lengths (VAL)
                # FIX: Ensure all arrays are the same length before slicing
                len_preds = len(val_preds_h)
                len_true = len(y_true_max_h)
                len_base = len(baseline_val) if baseline_val is not None else float('inf')
                
                num_val_pts = min(len_preds, len_true, len_base)
                
                val_preds_h = val_preds_h[:num_val_pts]
                y_true_max_h = y_true_max_h[:num_val_pts]
                if baseline_val is not None:
                    baseline_val_h = baseline_val[:num_val_pts].flatten()
                
                # 5. Denormalize (Crucial step from pipeline)
                # We need the 'denormalize' function. We can import it or implement simple version if params available.
                # Pipeline uses: from .stl_norm import denormalize
                # We should import it at top of file.
                from pipeline_plugins.stl_norm import denormalize, denormalize_returns
                
                val_target_price = denormalize(y_true_max_h, new_config)
                val_pred_price = denormalize(val_preds_h, new_config)
                
                # 6. Compute MAE (Pipeline: denormalize_returns(pred - target))
                # Wait, pipeline does: np.mean(np.abs(denormalize_returns(val_preds_h - val_target_h, params)))
                # This implies the model predicts returns/normalized values, and we denormalize the DIFFERENCE?
                # Or denormalize the values then subtract?
                # Pipeline code: val_mae_h = np.mean(np.abs(denormalize_returns(val_preds_h - val_target_h, params)))
                # If use_returns=False (which is enforced in pipeline), denormalize_returns might just be identity or simple scaling.
                # Let's look at stl_metrics.py again.
                # "train_mae_h = np.mean(np.abs(denormalize_returns(train_preds_h - train_target_h, params)))"
                # AND "train_naive_mae_h = np.mean(np.abs(denormalize(baseline_train_h, params) - train_target_price))"
                
                # So for Model MAE: denormalize_returns(pred - target)
                # For Naive MAE: denormalize(baseline) - denormalize(target)
                
                # Let's follow this exactly.
                
                # Model MAE
                # FIX: Use val_mae variable instead of fitness to ensure consistency
                val_mae = np.mean(np.abs(denormalize_returns(val_preds_h - y_true_max_h, new_config)))
                
                # Naive MAE
                naive_mae = float("inf")
                if baseline_val is not None:
                    # Note: pipeline uses denormalize(baseline) - val_target_price
                    # val_target_price was computed as denormalize(y_true_max_h)
                    naive_mae = np.mean(np.abs(denormalize(baseline_val_h, new_config) - val_target_price))

                # TRAIN metrics (max horizon)
                train_mae, train_naive_mae = _compute_split_metrics_max_h(
                    split="TRAINING",
                    preds_list=train_preds,
                    y_any=y_train,
                    baseline_any=baseline_train,
                    cfg=new_config,
                )

                # TEST metrics (max horizon)
                test_mae, test_naive_mae = (float("inf"), float("inf"))
                if x_test is not None and y_test is not None:
                    try:
                        if hasattr(predictor_for_eval, "model") and hasattr(predictor_for_eval.model, "predict"):
                            pred_bs = int(new_config.get("predict_batch_size", 0) or new_config.get("batch_size", 32) or 256)
                            test_preds = predictor_for_eval.model.predict(x_test, batch_size=pred_bs, verbose=0)
                        elif hasattr(predictor_for_eval, "predict_with_uncertainty"):
                            test_preds, _ = predictor_for_eval.predict_with_uncertainty(x_test, mc_samples=new_config.get("mc_samples", 1))
                        else:
                            test_preds = []
                        
                        test_preds = [test_preds] if isinstance(test_preds, np.ndarray) else test_preds
                        test_mae, test_naive_mae = _compute_split_metrics_max_h(
                            split="TEST",
                            preds_list=test_preds,
                            y_any=y_test,
                            baseline_any=baseline_test,
                            cfg=new_config,
                        )
                    except Exception as e:
                        print(f"WARN: Test prediction failed: {e}")
                        test_mae, test_naive_mae = (float("inf"), float("inf"))
                
                # FITNESS: Penalized Asymmetric Delta
                # Base: weighted average favoring validation (generalization matters more)
                # Penalty: harsh penalty if either split is ABOVE naive (val penalized 2x)
                # This prevents overfitting on training from masking poor validation
                
                if train_naive_mae is None or naive_mae is None or not np.isfinite(train_naive_mae) or not np.isfinite(naive_mae):
                    fitness = float("inf")
                else:
                    train_delta = float(train_mae - train_naive_mae)
                    val_delta = float(val_mae - naive_mae)
                    base = 0.4 * train_delta + 0.6 * val_delta
                    penalty = 0.0
                    if train_delta > 0:
                        penalty += train_delta
                    if val_delta > 0:
                        penalty += val_delta * 2
                    fitness = base + penalty
                train_loss = float(history.history.get("loss", [np.nan])[-1])
                val_loss = float(history.history.get("val_loss", [np.nan])[-1])

                # Scale diagnostics (catch mixed-unit bugs immediately)
                try:
                    diag = {
                        "y_true_max_h": _array_stats(y_true_max_h),
                        "val_preds_h": _array_stats(val_preds_h),
                        "baseline_val": _array_stats(baseline_val[:num_val_pts]) if baseline_val is not None else {"n": 0},
                    }
                    print(f"[DIAG] scales: {diag}")
                except Exception as _e:
                    print(f"[DIAG] scale stats failed: {_e}")

                _is_binary2 = new_config.get("target_plugin") in ("binary_target", "direction_target")
                _l1 = "Accuracy" if _is_binary2 else "MAE maxH"
                _l2 = "F1" if _is_binary2 else "Naive MAE maxH"
                print(
                    "Candidate Result -> "
                    f"TRAINING {_l1}: {train_mae:.6f} | TRAINING {_l2}: {train_naive_mae:.6f} || "
                    f"VALIDATION {_l1}: {val_mae:.6f} | VALIDATION {_l2}: {naive_mae:.6f} || "
                    f"TEST {_l1}: {test_mae:.6f} | TEST {_l2}: {test_naive_mae:.6f} || "
                    f"FITNESS (avg delta from naive): {fitness:.6f} || "
                    f"Keras VALIDATION loss (val_loss): {val_loss:.6f} | Keras TRAINING loss (loss): {train_loss:.6f}"
                )

                is_new_champion = False
                _is_new_best = (fitness > float(self.best_fitness_so_far)) if _hib else (fitness < float(self.best_fitness_so_far))
                if np.isfinite(fitness) and _is_new_best:
                    self.best_fitness_so_far = float(fitness)
                    self.best_val_mae_so_far = float(val_mae) if np.isfinite(val_mae) else self.best_val_mae_so_far
                    self.best_naive_mae_so_far = float(naive_mae) if np.isfinite(naive_mae) else self.best_naive_mae_so_far
                    self.best_test_mae_so_far = float(test_mae) if np.isfinite(test_mae) else self.best_test_mae_so_far
                    self.best_test_naive_mae_so_far = float(test_naive_mae) if np.isfinite(test_naive_mae) else self.best_test_naive_mae_so_far
                    self.best_train_mae_so_far = float(train_mae) if np.isfinite(train_mae) else self.best_train_mae_so_far
                    self.best_train_naive_mae_so_far = float(train_naive_mae) if np.isfinite(train_naive_mae) else self.best_train_naive_mae_so_far
                    is_new_champion = True
                    self.best_params_so_far = hyper_dict.copy()

                    # Save Champion Parameters Immediately
                    params_file = config.get("optimization_parameters_file", "optimization_parameters.json")
                    resolved_params_file = _resolve_repo_path(params_file) or params_file
                    try:
                        _atomic_json_dump(resolved_params_file, hyper_dict)
                        print(f"  [CHAMPION] Parameters saved to {resolved_params_file}")
                    except Exception as e:
                        print(f"  [CHAMPION] Failed to save parameters: {e}")

                champion_fitness = float(self.best_fitness_so_far) if self.best_fitness_so_far is not None else float("inf")
                champion_val_mae = (
                    float(self.best_val_mae_so_far)
                    if self.best_val_mae_so_far is not None and np.isfinite(self.best_val_mae_so_far)
                    else float("inf")
                )
                champion_naive = (
                    float(self.best_naive_mae_so_far)
                    if self.best_naive_mae_so_far is not None and np.isfinite(self.best_naive_mae_so_far)
                    else float("inf")
                )
                champion_test_mae = (
                    float(self.best_test_mae_so_far)
                    if self.best_test_mae_so_far is not None and np.isfinite(self.best_test_mae_so_far)
                    else float("inf")
                )
                champion_test_naive = (
                    float(self.best_test_naive_mae_so_far)
                    if self.best_test_naive_mae_so_far is not None and np.isfinite(self.best_test_naive_mae_so_far)
                    else float("inf")
                )
                champion_train_mae = (
                    float(self.best_train_mae_so_far)
                    if self.best_train_mae_so_far is not None and np.isfinite(self.best_train_mae_so_far)
                    else float("inf")
                )
                champion_train_naive = (
                    float(self.best_train_naive_mae_so_far)
                    if self.best_train_naive_mae_so_far is not None and np.isfinite(self.best_train_naive_mae_so_far)
                    else float("inf")
                )
                print(
                    "Champion so far -> "
                    f"TRAINING {_l1}: {champion_train_mae:.6f} | TRAINING {_l2}: {champion_train_naive:.6f} || "
                    f"VALIDATION {_l1}: {champion_val_mae:.6f} | VALIDATION {_l2}: {champion_naive:.6f} || "
                    f"TEST {_l1}: {champion_test_mae:.6f} | TEST {_l2}: {champion_test_naive:.6f} || "
                    f"FITNESS: {champion_fitness:.6f}"
                    + (" | NEW CHAMPION" if is_new_champion else "")
                )
            except Exception as e:
                print(f"Training/Metrics failed for individual {hyper_dict}: {e}")
                fitness = float("inf")

            # Hard cleanup after each candidate as well (prevents long-run accumulation)
            try:
                if hasattr(predictor_for_eval, 'model'):
                    del predictor_for_eval.model
            except Exception:
                pass
            tf.keras.backend.clear_session()
            gc.collect()

            _append_resource_row("candidate_end", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
            
            # Print optimization progress and state (ALWAYS, not just meta-mode)
            print("\n" + "="*80)
            print(f"OPTIMIZATION PROGRESS")
            print("="*80)
            if incremental_enabled or meta_mode:
                if meta_mode and current_meta_stage:
                    stage_name, stage_desc = "", ""
                    try:
                        from .modules.meta_stages import get_stage_info, get_total_stages
                        stage_name, stage_desc = get_stage_info(current_meta_stage, config)
                        total_meta_stages = get_total_stages(config)
                        print(f"Meta-Optimization Stage: {current_meta_stage}/{total_meta_stages} - {stage_name}")
                        print(f"  Description: {stage_desc}")
                    except Exception:
                        print(f"Stage: {current_meta_stage}")
                else:
                    print(f"Incremental Stage: {incremental_stage}/{total_stages}")
                print(f"Active Parameters ({len(hyper_keys)}): {', '.join(hyper_keys)}")
            print(f"Generation: {self.current_gen}/{self.end_gen - 1}")
            print(f"Candidate: {self.eval_counter}/{population_size}")
            print(f"Total Evaluations: {self.total_eval_counter}")
            print(f"\nGA Early Stopping:")
            print(f"  Patience: {patience} generations")
            print(f"  No-improve counter: {self.patience_counter}/{patience}")
            print(f"  Gen-start FITNESS to beat: {float(self.best_at_gen_start):.6f}")
            print(f"  Global champion FITNESS: {float(self.best_fitness_so_far):.6f}")
            print("="*80 + "\n")
            
            # Store naive_mae in individual for stats later (hacky but effective)
            individual.naive_mae = naive_mae
            try:
                individual.val_mae = val_mae
                individual.test_mae = test_mae
                individual.test_naive_mae = test_naive_mae
                individual.train_mae = train_mae
                individual.train_naive_mae = train_naive_mae
            except Exception:
                pass
            
            # Log to meta-training CSV if meta-mode is enabled
            if meta_mode:
                try:
                    # Gather all metrics for logging
                    metrics_dict = {
                        'train_mae': train_mae,
                        'train_rmse': float('inf'),  # Not computed in current code
                        'train_permutation_ok': 0,  # Not computed
                        'train_directional_accuracy': 0.0,  # Not computed
                        'train_naive_mae': train_naive_mae,
                        'val_mae': val_mae,
                        'val_rmse': float('inf'),  # Not computed
                        'val_permutation_ok': 0,  # Not computed
                        'val_directional_accuracy': 0.0,  # Not computed
                        'val_naive_mae': naive_mae,
                        'test_mae': test_mae,
                        'test_rmse': float('inf'),  # Not computed
                        'test_permutation_ok': 0,  # Not computed
                        'test_directional_accuracy': 0.0,  # Not computed
                        'test_naive_mae': test_naive_mae,
                        'stage': current_meta_stage,
                        'generation': int(self.current_gen or 0),
                        'candidate_id': int(self.eval_counter),
                        'fitness': fitness
                    }
                    
                    resolved_log_path = _resolve_repo_path(get_meta_log_path(config))
                    log_candidate_evaluation(
                        log_path=resolved_log_path,
                        params_dict=hyper_dict,
                        metrics_dict=metrics_dict,
                        stage=current_meta_stage,
                        generation=int(self.current_gen or 0),
                        candidate_id=int(self.eval_counter),
                        config=config
                    )
                except Exception as log_error:
                    import traceback
                    print(f"[META-LOG] Warning: Failed to log candidate: {log_error}")
                    traceback.print_exc()
            
            return (fitness,)

        # Toolbox will be registered inside the while loop to use current hyper_keys
        population_size = config.get("population_size", self.params.get("population_size", 20))
        n_generations = config.get("n_generations", self.params.get("n_generations", 10))
        patience = config.get("optimization_patience", self.params.get("optimization_patience", 3))
        
        # --- INCREMENTAL OPTIMIZATION OUTER LOOP ---
        # This wraps the entire GA optimization to progressively add parameters
        # Note: incremental_stage and current_meta_stage already initialized above
        if not incremental_enabled and not meta_mode:
            total_stages = 1  # Single stage for standard optimization
        
        # Initialize population once (will be expanded in-place during incremental stages)
        population = None
        hof = tools.HallOfFame(1)
        first_iteration = True  # Gate resume/recovery to first iteration only
        stats_history = []  # Persist across stages
        
        # --- DOIN CALLBACK HOOKS (must be defined before initial population eval) ---
        _opt_callbacks = config.get("optimization_callbacks", {})

        while True:
            print(f"\n{'='*80}")
            if incremental_enabled:
                print(f"[INCREMENTAL] STAGE {incremental_stage}/{total_stages}")
                print(f"[INCREMENTAL] Optimizing parameters: {hyper_keys}")
            print(f"{'='*80}\n")
            
            print(f"[DEBUG] Creating population of size {population_size} with {len(hyper_keys)} parameters...")
            print(f"[DEBUG] Parameters: {hyper_keys}")
            
            # Update bounds and param_types for current stage
            bounds = {k: full_bounds[k] for k in hyper_keys}
            param_types = {}
            lower_bounds = []
            upper_bounds = []
            
            for key in hyper_keys:
                low, up = bounds[key]
                lower_bounds.append(low)
                upper_bounds.append(up)
                if isinstance(low, int) and isinstance(up, int):
                    param_types[key] = 'int'
                else:
                    param_types[key] = 'float'
            
            # Register toolbox operations with current parameter set
            # Must be inside loop so closures capture updated hyper_keys/bounds after stage expansion
            def mutate(individual, indpb):
                for i, key in enumerate(hyper_keys):
                    if random.random() < indpb:
                        low = lower_bounds[i]
                        up = upper_bounds[i]
                        ptype = param_types[key]
                        
                        if ptype == 'int':
                            individual[i] = random.randint(low, up)
                        else:
                            # Gaussian mutation for floats
                            sigma = (up - low) * 0.1
                            val = individual[i] + random.gauss(0, sigma)
                            individual[i] = max(low, min(up, val))
                return individual,
            
            toolbox.register("evaluate", eval_individual)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", mutate, indpb=config.get("mutpb", self.params.get("mutpb", 0.2)))
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Create initial population only on first iteration
            if population is None:
                population = toolbox.population(n=population_size)
                print(f"\n{'='*80}")
                print(f"CREATED INITIAL POPULATION: {len(population)} individuals with {len(population[0])} parameters")
                print(f"Parameters: {hyper_keys}")
                print(f"{'='*80}\n")

            # --- RESUME / RECOVERY LOGIC (first iteration only) ---
            start_gen = 0
            loaded_indices = set()
            resume_enabled = config.get("optimization_resume", False)
            resume_path = _resolve_repo_path(config.get("optimization_resume_file"))
            params_path = _resolve_repo_path(config.get("optimization_parameters_file"))

            if first_iteration:
                actual_genome_size = len(hyper_keys)

                # 1. Try to load full population state ONLY if resume is explicitly enabled
                if resume_enabled and resume_path and os.path.exists(resume_path):
                    start_gen, loaded_count, loaded_indices, actual_genome_size, resumed_stage, resumed_params, _resumed_opt_state = load_resume_checkpoint(
                        resume_path, population, hyper_keys, full_bounds, incremental_enabled, config
                    )

                    # Restore optimizer state from checkpoint
                    if _resumed_opt_state:
                        if _resumed_opt_state.get("best_fitness_so_far") is not None:
                            self.best_fitness_so_far = float(_resumed_opt_state["best_fitness_so_far"])
                        if _resumed_opt_state.get("best_val_mae_so_far") is not None:
                            self.best_val_mae_so_far = float(_resumed_opt_state["best_val_mae_so_far"])
                        if _resumed_opt_state.get("best_naive_mae_so_far") is not None:
                            self.best_naive_mae_so_far = float(_resumed_opt_state["best_naive_mae_so_far"])
                        if _resumed_opt_state.get("best_test_mae_so_far") is not None:
                            self.best_test_mae_so_far = float(_resumed_opt_state["best_test_mae_so_far"])
                        if _resumed_opt_state.get("best_test_naive_mae_so_far") is not None:
                            self.best_test_naive_mae_so_far = float(_resumed_opt_state["best_test_naive_mae_so_far"])
                        if _resumed_opt_state.get("best_train_mae_so_far") is not None:
                            self.best_train_mae_so_far = float(_resumed_opt_state["best_train_mae_so_far"])
                        if _resumed_opt_state.get("best_train_naive_mae_so_far") is not None:
                            self.best_train_naive_mae_so_far = float(_resumed_opt_state["best_train_naive_mae_so_far"])
                        if _resumed_opt_state.get("total_eval_counter") is not None:
                            self.total_eval_counter = int(_resumed_opt_state["total_eval_counter"])
                        if _resumed_opt_state.get("patience_counter") is not None:
                            self.patience_counter = int(_resumed_opt_state["patience_counter"])
                        self.best_at_gen_start = float(self.best_fitness_so_far)

                    # Adjust hyper_keys if resume loaded smaller genome and incremental is enabled
                    if loaded_count > 0:
                        if resumed_params and len(resumed_params) < len(hyper_keys):
                            print(f"[RESUME] Using resumed parameters ({len(resumed_params)}), will add new ones incrementally")
                            hyper_keys = resumed_params
                        else:
                            hyper_keys = adjust_params_for_resume(hyper_keys, all_param_keys, actual_genome_size, incremental_enabled)

                    # Restore stage if meta-mode, but reconcile with actual genome size
                    # If genome was truncated (e.g. 14→3 genes), current_meta_stage must
                    # match the stage whose cumulative params equal the actual genome,
                    # not the stage from the resume file.
                    if meta_mode and resumed_stage:
                        from .modules.meta_stages import get_active_parameters_for_stage, get_total_stages
                        effective_stage = resumed_stage
                        n_active = len(hyper_keys)
                        ts = get_total_stages(config)
                        for s in range(1, ts + 1):
                            if len(get_active_parameters_for_stage(s, config)) >= n_active:
                                effective_stage = s
                                break
                        if effective_stage != resumed_stage:
                            print(f"[RESUME] Reconciled meta-mode stage: resume said {resumed_stage}, but genome has {n_active} params → stage {effective_stage}")
                        else:
                            print(f"[RESUME] Restored meta-mode stage {effective_stage}")
                        current_meta_stage = effective_stage

                        # Update bounds and param_types to match adjusted hyper_keys
                        bounds = {k: full_bounds[k] for k in hyper_keys}
                        lower_bounds = [full_bounds[k][0] for k in hyper_keys]
                        upper_bounds = [full_bounds[k][1] for k in hyper_keys]
                        param_types = {}
                        for key in hyper_keys:
                            low, up = full_bounds[key]
                            param_types[key] = 'int' if isinstance(low, int) and isinstance(up, int) else 'float'

                # 2. Inject Champion (Elitism/Continuity)
                # ONLY load champion if resume is enabled BUT resume file was NOT found
                # This avoids duplication (resume file already contains champion in population)
                resume_file_loaded = resume_enabled and resume_path and os.path.exists(resume_path)
                if resume_enabled and params_path and os.path.exists(params_path) and not resume_file_loaded:
                    champ_ind = load_champion_parameters(params_path, hyper_keys, full_bounds)
                    if champ_ind:
                        print(f"[CHAMPION LOAD] Injecting champion into population index 0")
                        for k, val in enumerate(champ_ind):
                            population[0][k] = val
                        loaded_indices.add(0)

                # Pause if requested and any resume/load ACTUALLY happened
                if config.get("optimization_pause_on_resume", False) and resume_enabled:
                    if (resume_path and os.path.exists(resume_path)) or (params_path and os.path.exists(params_path)):
                        print("\n[PAUSE] optimization_pause_on_resume is enabled.")
                        print("[PAUSE] Resume/Champion data loaded. Press Enter to continue optimization...")
                        try:
                            input()
                        except EOFError:
                            pass
            else:
                print(f"[STAGE TRANSITION] Skipping resume/recovery (stage {incremental_stage}, not first iteration)")

            # Calculate end generation: start_gen + n_generations
            # This way n_generations always means "run THIS MANY generations"
            # Fresh start: range(0, 0+10) = 10 generations (0-9)
            # Resume from 9: range(10, 10+2) = 2 generations (10-11)
            end_gen = start_gen + n_generations
            self.end_gen = end_gen  # Store for eval_individual to access
            print(f"Starting hyperparameter optimization (Gen {start_gen} to {end_gen - 1}, running {n_generations} generations)...")
            start_opt = time.time()

            # Custom optimization loop with early stopping
            # Evaluate the entire population
            self.current_gen = start_gen
            self.eval_counter = 0 # Reset per-generation counter
            if first_iteration:
                self.total_eval_counter = 0

            # Enforce bounds on initial population (just in case)
            print("[BOUNDS] Checking initial population against hyperparameter bounds...")
            for idx_ind, ind in enumerate(population):
                # Skip bounds enforcement if loaded from file (Resume or Champion)
                if idx_ind in loaded_indices:
                    # Optional: Warn if it IS out of bounds, just for info
                    for i, val in enumerate(ind):
                        low = lower_bounds[i]
                        up = upper_bounds[i]
                        try:
                            if not (low <= val <= up):
                                 print(f"  [BOUNDS] INFO: Loaded individual {idx_ind} parameter '{hyper_keys[i]}' value {val} is outside bounds [{low}, {up}] (kept as is).")
                        except TypeError:
                            pass
                    continue

                for i, val in enumerate(ind):
                    low = lower_bounds[i]
                    up = upper_bounds[i]
                    clamped = max(low, min(up, val))
                    if clamped != val:
                        ind[i] = clamped

            # Print Candidate 0 (Champion) for verification
            print(f"\n[VERIFY] Candidate 0 (Champion) Parameters to be evaluated:")
            champ_verify = {}
            for i, key in enumerate(hyper_keys):
                champ_verify[key] = population[0][i]
            print(json.dumps(champ_verify, indent=2))
            print("-" * 60)

            # Only evaluate individuals without valid fitness.
            # Individuals restored from checkpoint with saved fitness will be skipped.
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            valid_from_checkpoint = len(population) - len(invalid_ind)
            if valid_from_checkpoint > 0:
                print(f"[RESUME] {valid_from_checkpoint}/{len(population)} individuals have restored fitness — skipping re-evaluation")
            if invalid_ind:
                print(f"[EVAL] Evaluating {len(invalid_ind)} individuals without fitness...")
            fitnesses = map(toolbox.evaluate, invalid_ind)
            _cb_between_init = _opt_callbacks.get("on_between_candidates")
            _cand_num_init = 0
            _force_advance_init = False
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                _cand_num_init += 1
                if _cb_between_init:
                    try:
                        _bc_stage_init = {
                            "stage": current_meta_stage if meta_mode else incremental_stage,
                            "total_stages": total_stages,
                            "generation": start_gen,
                            "candidate_num": _cand_num_init,
                            "total_candidates": len(invalid_ind),
                            "total_candidates_evaluated": int(self.total_eval_counter),
                            "fitness": float(fit[0]) if fit else None,
                            "val_mae": getattr(ind, 'val_mae', None),
                            "train_mae": getattr(ind, 'train_mae', None),
                            "val_naive_mae": getattr(ind, 'naive_mae', None),
                            "train_naive_mae": getattr(ind, 'train_naive_mae', None),
                            "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far != float('inf') else None,
                        }
                        _between_init_result = _cb_between_init(start_gen, _cand_num_init, _bc_stage_init)
                        # Stage-sync: network signalled stage advance during initial evaluation
                        if isinstance(_between_init_result, dict) and _between_init_result.get("_force_stage_advance"):
                            print(f"  [STAGE SYNC] Network signalled stage advance during initial eval ({_cand_num_init}/{len(invalid_ind)} evaluated)")
                            _force_advance_init = True
                            break
                    except Exception as _cb_err:
                        print(f"  [EVAL SERVICE] Between-candidates callback error: {_cb_err}")

            hof.update(population)

            # Initialize best_*_so_far from hof[0] ONLY if not already restored
            # from checkpoint (resume with optimizer_state).  If resumed, the
            # checkpoint values are authoritative — hof[0] may differ.
            _resumed_has_state = (resume_enabled and '_resumed_opt_state' in dir() 
                                  and _resumed_opt_state 
                                  and _resumed_opt_state.get("best_fitness_so_far") is not None)
            if hof and hof[0].fitness.valid and not _resumed_has_state:
                self.best_fitness_so_far = hof[0].fitness.values[0]
                self.best_val_mae_so_far = getattr(hof[0], "val_mae", None)
                self.best_naive_mae_so_far = getattr(hof[0], "naive_mae", None)
                self.best_test_mae_so_far = getattr(hof[0], "test_mae", None)
                self.best_test_naive_mae_so_far = getattr(hof[0], "test_naive_mae", None)
                self.best_train_mae_so_far = getattr(hof[0], "train_mae", None)
                self.best_train_naive_mae_so_far = getattr(hof[0], "train_naive_mae", None)
            self.best_at_gen_start = float(self.best_fitness_so_far)

            if not _resumed_has_state:
                no_improve_counter = 0
                self.patience_counter = 0
            else:
                no_improve_counter = int(self.patience_counter)
                print(f"[RESUME] Restored patience_counter={self.patience_counter}, "
                      f"best_fitness={self.best_fitness_so_far:.6f}, "
                      f"total_evals={self.total_eval_counter}")

            # Mark first iteration complete (resume/recovery won't run again)
            first_iteration = False
            # Clear stale resume state so subsequent stages don't think we're resuming.
            # Without this, _resumed_has_state stays True on stage 2+, preventing
            # proper no_improve_counter reset and best_fitness_so_far initialization.
            _resumed_opt_state = None

            # Stage-sync: if force advance signalled during initial eval, skip gen loop
            if _force_advance_init:
                print(f"  [STAGE SYNC] Skipping generation loop — advancing to next stage")
                _gen_range = range(0, 0)  # empty range — skip all generations
            else:
                _gen_range = range(start_gen, end_gen)

            for gen in _gen_range:
                gen_start_time = time.time()
                self.current_gen = gen  # Current generation number (absolute)
                self.eval_counter = 0 # Reset per generation
                print(f"-- Generation {gen}/{end_gen - 1} --")

                best_at_gen_start = float(self.best_fitness_so_far)
                self.best_at_gen_start = best_at_gen_start

                # --- CALLBACK: on_generation_start (migration IN) ---
                _cb_gen_start = _opt_callbacks.get("on_generation_start")
                if _cb_gen_start:
                    try:
                        _stage_info = {
                            "stage": current_meta_stage if meta_mode else incremental_stage,
                            "total_stages": total_stages,
                            "meta_mode": meta_mode,
                            "total_candidates_evaluated": int(self.total_eval_counter),
                            "population_size": population_size,
                        }
                        _migrant_params = _cb_gen_start(population, hof, hyper_keys, gen, _stage_info)
                        # Stage-sync: network ordered us to advance to the next stage
                        if isinstance(_migrant_params, dict) and _migrant_params.get("_force_stage_advance"):
                            print(f"  [STAGE SYNC] Network signalled stage advance — ending current stage early")
                            break
                        if _migrant_params and isinstance(_migrant_params, dict):
                            # ── Deduplication: skip if this exact candidate already exists ──
                            _migrant_genome = []
                            for ki, key in enumerate(hyper_keys):
                                _migrant_genome.append(_migrant_params.get(key))
                            _already_in_pop = False
                            for _pop_ind in population:
                                if all(
                                    _pop_ind[ki] == _migrant_genome[ki]
                                    for ki in range(len(hyper_keys))
                                    if _migrant_genome[ki] is not None
                                ):
                                    _already_in_pop = True
                                    break
                            if _already_in_pop:
                                print(f"  [MIGRATION IN] Network champion already in population — skipping injection")
                            else:
                                # Replace worst individual in population with network champion
                                _worst_idx = min(range(len(population)),
                                                 key=lambda i: -population[i].fitness.values[0]
                                                 if population[i].fitness.valid else float("inf"))
                                for ki, key in enumerate(hyper_keys):
                                    if key in _migrant_params:
                                        population[_worst_idx][ki] = _migrant_params[key]
                                del population[_worst_idx].fitness.values  # Mark for re-evaluation
                                print(f"  [MIGRATION IN] Injected network champion into population[{_worst_idx}]")
                    except Exception as _cb_err:
                        print(f"  [MIGRATION IN] Callback error: {_cb_err}")

                # Select the next generation individuals
                offspring = toolbox.select(population, len(population))
                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))

                # Save pre-mutation snapshot of the current population so that
                # failed candidates (OOM, crash) can have their genome reverted
                # to the parent's version instead of keeping a broken mutant.
                _prev_population_genes = [list(ind) for ind in population]

                # PRESERVE CHAMPION: In first generation after resume, ensure champion stays in position 0
                # without any genetic operators applied, to enable exact reproducibility check
                is_first_gen_after_resume = (gen == start_gen and start_gen > 0)
                if is_first_gen_after_resume:
                    # Force the loaded champion into offspring[0] (it should already be there from selection)
                    # but we clone from population[0] to be absolutely sure
                    offspring[0] = toolbox.clone(population[0])
                    print(f"  [RESUME] Preserving loaded champion at offspring[0] for exact reproducibility")

                # Apply crossover and mutation on the offspring
                # Skip position 0 in first generation after resume to preserve champion
                start_idx = 1 if is_first_gen_after_resume else 0
                for child1, child2 in zip(offspring[start_idx::2], offspring[start_idx+1::2]):
                    if random.random() < self.params.get("cxpb", 0.5):
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for i, mutant in enumerate(offspring):
                    if i == 0 and is_first_gen_after_resume:
                        continue  # Skip mutation for champion in first gen after resume
                    if random.random() < self.params.get("mutpb", 0.2):
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Enforce bounds on all offspring (fix for cxBlend producing out-of-bounds values)
                for child in offspring:
                    for i, val in enumerate(child):
                        low = lower_bounds[i]
                        up = upper_bounds[i]
                        child[i] = max(low, min(up, val))

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                _cb_between = _opt_callbacks.get("on_between_candidates")
                _cand_num = 0
                _force_advance_flag = False
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                    _cand_num += 1

                    # Revert genome on evaluation failure: if the candidate crashed
                    # (OOM, etc.) its fitness is inf.  Replace such broken mutants
                    # with the corresponding parent genome from the previous
                    # generation so the population doesn't accumulate dead weight.
                    if fit[0] == float("inf"):
                        idx_in_offspring = offspring.index(ind)
                        if idx_in_offspring < len(_prev_population_genes):
                            for gi, gv in enumerate(_prev_population_genes[idx_in_offspring]):
                                ind[gi] = gv
                            del ind.fitness.values  # mark for re-evaluation next gen
                            print(f"  [GENOME REVERT] Candidate {_cand_num} failed — reverted genome to parent")

                    # --- CALLBACK: on_between_candidates ---
                    # Process max 1 pending evaluation request between candidates
                    if _cb_between:
                        try:
                            _bc_stage = {
                                "stage": current_meta_stage if meta_mode else incremental_stage,
                                "total_stages": total_stages,
                                "generation": gen,
                                "candidate_num": _cand_num,
                                "total_candidates": len(invalid_ind),
                                "total_candidates_evaluated": int(self.total_eval_counter),
                                "fitness": float(fit[0]) if fit else None,
                                "val_mae": getattr(ind, 'val_mae', None),
                                "train_mae": getattr(ind, 'train_mae', None),
                                "val_naive_mae": getattr(ind, 'naive_mae', None),
                                "train_naive_mae": getattr(ind, 'train_naive_mae', None),
                                "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far != float('inf') else None,
                                "candidate_params": getattr(ind, 'hyper_dict', None),
                                "model_summary": getattr(ind, 'model_summary', None),
                            }
                            _between_result = _cb_between(gen, _cand_num, _bc_stage)
                            # Stage-sync: network signalled stage advance between candidates
                            if isinstance(_between_result, dict) and _between_result.get("_force_stage_advance"):
                                print(f"  [STAGE SYNC] Network signalled stage advance between candidates ({_cand_num}/{len(invalid_ind)} evaluated)")
                                _force_advance_flag = True
                                break
                        except Exception as _cb_err:
                            print(f"  [EVAL SERVICE] Between-candidates callback error: {_cb_err}")

                # Stage-sync: if force advance was signalled mid-generation, stop immediately
                if _force_advance_flag:
                    population[:] = offspring  # Keep partially evaluated offspring
                    print(f"  [STAGE SYNC] Breaking generation loop for stage transition")
                    break

                # Replace the old population by the offspring
                population[:] = offspring

                # Update HallOfFame
                hof.update(population)

                # Check for improvement
                current_best = hof[0].fitness.values[0]
                print(f"  Best Val Loss: {current_best}")

                # CRITICAL FIX: Do NOT update best_fitness_so_far here!
                # It's already updated correctly in lines 531-538 when evaluating each candidate.
                # Updating it here from hof[0] causes data inconsistency because hof[0] 
                # might be a different candidate than the one that achieved best_fitness_so_far.
                # For early-stopping, detect improvements relative to the start of this generation.
                if current_best < best_at_gen_start:
                    no_improve_counter = 0
                    self.patience_counter = 0
                    print(f"  New best found!")
                else:
                    no_improve_counter += 1
                    self.patience_counter = no_improve_counter
                    print(f"  No improvement for {no_improve_counter} generations.")

                # --- Save Statistics and Champion Parameters ---
                gen_end_time = time.time()
                elapsed_time = gen_end_time - start_opt
                gen_duration = gen_end_time - gen_start_time

                # Calculate average MAE for this generation
                valid_fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.valid and ind.fitness.values[0] != float("inf")]
                avg_mae = sum(valid_fitnesses) / len(valid_fitnesses) if valid_fitnesses else float("inf")

                # Best individual in this generation (by VALIDATION fitness = Val MAE on max horizon)
                best_ind_gen = tools.selBest(population, 1)[0]
                best_val_mae_gen = float(best_ind_gen.fitness.values[0]) if best_ind_gen.fitness.valid else float("inf")
                best_naive_mae_gen = getattr(best_ind_gen, "naive_mae", None)

                # Global champion so far (HallOfFame), also by FITNESS
                champion_fitness_global = float(hof[0].fitness.values[0]) if hof and hof[0].fitness.valid else float("inf")

                # FIX: ALL champion metrics must come from self.best_*_so_far to match best_fitness_so_far
                # Using hof[0] mixes data from different candidates because HoF tracks current best, 
                # but self.best_fitness_so_far tracks absolute best ever seen
                champion_val_mae_global = self.best_val_mae_so_far
                champion_naive_mae_global = self.best_naive_mae_so_far
                champion_test_mae_global = self.best_test_mae_so_far
                champion_test_naive_mae_global = self.best_test_naive_mae_so_far
                champion_train_mae_global = self.best_train_mae_so_far
                champion_train_naive_mae_global = self.best_train_naive_mae_so_far

                stats_history.append({
                    "generation": self.current_gen,
                    "duration": gen_duration,
                    # NOTE: avg is over population FITNESS values (avg of train/val deltas from naive)
                    "avg_fitness": avg_mae,
                    # Actual validation MAE from best of generation
                    "best_validation_mae_gen": getattr(best_ind_gen, "val_mae", None) if best_ind_gen else None,
                    "best_fitness_gen": best_val_mae_gen,  # This is actually fitness, not val_mae
                    "champion_fitness_global": champion_fitness_global,
                    "champion_validation_mae_global": champion_val_mae_global,
                    "best_validation_naive_mae_gen": best_naive_mae_gen,
                    "champion_validation_naive_mae_global": champion_naive_mae_global,
                })

                avg_time_per_epoch = sum(s["duration"] for s in stats_history) / len(stats_history)
                total_candidates = int(self.total_eval_counter)

                stats_data = {
                    "total_time_elapsed": elapsed_time,
                    "average_time_per_epoch": avg_time_per_epoch,
                    "candidates_evaluated_so_far": total_candidates,
                    # CORRECT LABELS: Fitness vs Actual Metrics
                    "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far is not None else None,
                    "champion_validation_mae": float(self.best_val_mae_so_far) if self.best_val_mae_so_far is not None else None,
                    "champion_validation_naive_mae": champion_naive_mae_global,
                    "champion_test_mae": champion_test_mae_global,
                    "champion_test_naive_mae": champion_test_naive_mae_global,
                    "champion_train_mae": champion_train_mae_global,
                    "champion_train_naive_mae": champion_train_naive_mae_global,
                    # Per-epoch lists (correctly labeled)
                    "average_fitness_per_epoch": [s["avg_fitness"] for s in stats_history],
                    "champion_fitness_per_epoch": [s["champion_fitness_global"] for s in stats_history],
                    "champion_validation_mae_per_epoch": [s["champion_validation_mae_global"] for s in stats_history],
                    "best_fitness_per_epoch": [s["best_fitness_gen"] for s in stats_history],
                    "best_validation_mae_per_epoch": [s["best_validation_mae_gen"] for s in stats_history],
                    "history": stats_history
                }

                # Save Statistics
                stats_file = config.get("optimization_statistics", "optimization_stats.json")
                resolved_stats_file = _resolve_repo_path(stats_file) or stats_file
                try:
                    _atomic_json_dump(resolved_stats_file, stats_data)
                    print(f"  Statistics saved to {resolved_stats_file}")
                except Exception as e:
                    print(f"  Failed to save statistics: {e}")

                # REMOVED: Duplicate parameter saving that could overwrite correct champion
                # Parameters are saved at line 543 when new champion is found during evaluation
                # Saving again here from hof[0] can overwrite with wrong candidate if hof[0] != best_fitness_so_far

                # --- Save Resume State ---
                if resume_path:
                    _opt_state = {
                        "best_fitness_so_far": float(self.best_fitness_so_far) if self.best_fitness_so_far != float('inf') else None,
                        "best_val_mae_so_far": float(self.best_val_mae_so_far) if self.best_val_mae_so_far is not None else None,
                        "best_naive_mae_so_far": float(self.best_naive_mae_so_far) if self.best_naive_mae_so_far is not None else None,
                        "best_test_mae_so_far": float(self.best_test_mae_so_far) if self.best_test_mae_so_far is not None else None,
                        "best_test_naive_mae_so_far": float(self.best_test_naive_mae_so_far) if self.best_test_naive_mae_so_far is not None else None,
                        "best_train_mae_so_far": float(self.best_train_mae_so_far) if self.best_train_mae_so_far is not None else None,
                        "best_train_naive_mae_so_far": float(self.best_train_naive_mae_so_far) if self.best_train_naive_mae_so_far is not None else None,
                        "total_eval_counter": int(self.total_eval_counter),
                        "patience_counter": int(self.patience_counter),
                    }
                    save_resume_checkpoint(
                        resume_path, 
                        gen, 
                        population,
                        current_stage=current_meta_stage,
                        active_parameters=hyper_keys,
                        meta_mode=meta_mode,
                        optimizer_state=_opt_state
                    )

                # --- CALLBACK: on_generation_end ---
                _cb_gen_end = _opt_callbacks.get("on_generation_end")
                if _cb_gen_end:
                    try:
                        _gen_end_info = {
                            "stage": current_meta_stage if meta_mode else incremental_stage,
                            "total_stages": total_stages,
                            "meta_mode": meta_mode,
                            "generation": gen,
                            "total_candidates_evaluated": int(self.total_eval_counter),
                            "population_size": population_size,
                            "no_improve_counter": no_improve_counter,
                            "patience": patience,
                            "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far is not None else None,
                            "champion_val_mae": float(self.best_val_mae_so_far) if self.best_val_mae_so_far is not None else None,
                            "champion_naive_mae": float(self.best_naive_mae_so_far) if self.best_naive_mae_so_far is not None else None,
                            "champion_test_mae": float(self.best_test_mae_so_far) if self.best_test_mae_so_far is not None else None,
                            "champion_train_mae": float(self.best_train_mae_so_far) if self.best_train_mae_so_far is not None else None,
                            "champion_train_naive_mae": float(self.best_train_naive_mae_so_far) if self.best_train_naive_mae_so_far is not None else None,
                            "avg_fitness": avg_mae,
                            "best_fitness_gen": best_val_mae_gen,
                        }
                        _cb_gen_end(population, hof, hyper_keys, gen, _gen_end_info, stats_data)
                    except Exception as _cb_err:
                        print(f"  [DOIN] Generation end callback error: {_cb_err}")

                if no_improve_counter >= patience:
                    print(f"Early stopping triggered after {gen + 1} generations.")
                    break

            # --- end of for gen loop ---
            end_opt = time.time()
            print(f"Optimization stage completed in {end_opt - start_opt:.2f} seconds.")

            # --- INCREMENTAL: Check if we need to add more parameters ---
            from .modules.incremental_logic import should_add_more_parameters, get_next_parameter_batch
            from .modules.genome_operations import expand_population_with_new_params

            if should_add_more_parameters(hyper_keys, all_param_keys, incremental_enabled):
                # --- CALLBACK: on_stage_end (broadcast to peers for stage sync) ---
                _completed_stage = current_meta_stage if meta_mode else incremental_stage
                _cb_stage_end = _opt_callbacks.get("on_stage_end")
                if _cb_stage_end:
                    try:
                        _stage_end_metrics = {
                            "champion_fitness": float(self.best_fitness_so_far) if self.best_fitness_so_far != float('inf') else None,
                            "champion_val_mae": float(self.best_val_mae_so_far) if self.best_val_mae_so_far is not None else None,
                            "champion_naive_mae": float(self.best_naive_mae_so_far) if self.best_naive_mae_so_far is not None else None,
                            "champion_train_mae": float(self.best_train_mae_so_far) if self.best_train_mae_so_far is not None else None,
                            "champion_train_naive_mae": float(self.best_train_naive_mae_so_far) if self.best_train_naive_mae_so_far is not None else None,
                        }
                        _cb_stage_end(
                            _completed_stage, total_stages,
                            self.best_params_so_far, float(self.best_fitness_so_far),
                            _stage_end_metrics,
                        )
                    except Exception as _cb_err:
                        print(f"  [STAGE END] Callback error: {_cb_err}")

                # Get next batch of parameters (meta-mode aware)
                new_params, hyper_keys = get_next_parameter_batch(
                    hyper_keys, 
                    all_param_keys, 
                    increment_size,
                    current_stage=current_meta_stage,
                    meta_mode=meta_mode,
                    config=config
                )

                # Update meta stage counter if in meta mode
                if meta_mode:
                    current_meta_stage += 1

                print(f"\n[INCREMENTAL] Stage {incremental_stage} complete!")
                print(f"[INCREMENTAL] Adding {len(new_params)} new parameters: {new_params}")
                print(f"[INCREMENTAL] Total active parameters now: {len(hyper_keys)}")

                # Expand entire population with new parameters
                expanded_count = expand_population_with_new_params(population, new_params, full_bounds, config)
                print(f"[INCREMENTAL] Expanded {expanded_count} individuals in population")

                # CRITICAL: Clear HallOfFame after expansion!
                # Old HoF individuals have fewer genes than expanded population.
                # If we keep them, hof[0] at final result extraction may have wrong genome size → IndexError.
                hof.clear()
                print(f"[INCREMENTAL] Cleared HallOfFame (old individuals had {len(hyper_keys) - len(new_params)} genes, new have {len(hyper_keys)})")

                # STAGE SYNC: Invalidate ALL fitness values so the entire population
                # is re-evaluated with the new expanded parameter set.
                # This ensures all individuals are scored with the new parameters active.
                _invalidated_count = 0
                for ind in population:
                    if ind.fitness.valid:
                        del ind.fitness.values
                        _invalidated_count += 1
                print(f"[INCREMENTAL] Invalidated {_invalidated_count} fitness values for full re-evaluation")

                # Update bounds and param_types for expanded genome
                bounds = {k: full_bounds[k] for k in hyper_keys}
                lower_bounds = [full_bounds[k][0] for k in hyper_keys]
                upper_bounds = [full_bounds[k][1] for k in hyper_keys]
                param_types = {}
                for key in hyper_keys:
                    low, up = full_bounds[key]
                    param_types[key] = 'int' if isinstance(low, int) and isinstance(up, int) else 'float'

                # Reset optimization tracking for next stage
                self.best_fitness_so_far = float("inf")
                self.best_val_mae_so_far = None
                self.best_naive_mae_so_far = None
                self.best_test_mae_so_far = None
                self.best_test_naive_mae_so_far = None
                self.best_train_mae_so_far = None
                self.best_train_naive_mae_so_far = None
                self.patience_counter = 0
                self.best_at_gen_start = float("inf")

                incremental_stage += 1
                print(f"[INCREMENTAL] Starting stage {incremental_stage}...\n")
                # Continue to next optimization stage (while True loop)
                continue
            else:
                # All parameters optimized - exit while True loop
                if incremental_enabled:
                    print(f"\n[INCREMENTAL] All {len(all_param_keys)} parameters optimized!")
                break

        # Seleccionar el mejor individuo (outside while loop)
        best_ind = hof[0]
        best_hyper = {}
        for i, key in enumerate(hyper_keys):
            value = best_ind[i]
            ptype = param_types[key]
            
            if key == "use_log1p_features":
                val_int = int(round(value))
                best_hyper[key] = ["typical_price"] if val_int == 1 else None
            elif key == "positional_encoding":
                val_int = int(round(value))
                best_hyper[key] = bool(val_int)
            elif key == "activation":
                act_idx = int(round(value))
                act_idx = max(0, min(act_idx, len(ACTIVATION_INDEX_TO_NAME) - 1))
                best_hyper[key] = ACTIVATION_INDEX_TO_NAME[act_idx]
            elif ptype == 'int':
                best_hyper[key] = int(round(value))
            else:
                best_hyper[key] = value
                    
        print(f"Best hyperparameters found: {best_hyper}")
        return best_hyper

# Debugging usage example (cuando se ejecuta el plugin directamente)
if __name__ == "__main__":
    optimizer_plugin = Plugin()
    test_config = {
        "plugin": "ann",
        "x_train_file": "data/train.csv",
        "x_validation_file": "data/val.csv",
        "x_test_file": "data/test.csv",
        "window_size": 24,
        "time_horizon": 1,
        "batch_size": 32,
        "epochs": 10,
        "threshold_error": 0.001
    }
    # Cargar instancias de Ioin y Preprocessor Plugins.
    from app.plugin_loader import load_plugin
    predictor_class, _ = load_plugin('ioin.plugins', test_config.get('plugin', 'default_predictor'))
    predictor_plugin = predictor_class()
    predictor_plugin.set_params(**test_config)
    from app.plugin_loader import load_plugin as load_preprocessor_plugin
    preprocessor_class, _ = load_preprocessor_plugin('preprocessor.plugins', 'default_preprocessor')
    preprocessor_plugin = preprocessor_class()
    preprocessor_plugin.set_params(**test_config)
    best_params = optimizer_plugin.optimize(predictor_plugin, preprocessor_plugin, test_config)
    print("Optimized parameters:", best_params)
