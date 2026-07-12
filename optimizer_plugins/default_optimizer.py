"""
default_optimizer.py — DEAP GA over agent hyperparameters.

Uses the `hparam_schema()` of an agent plugin (list of
(name, low, high, type)) to build chromosomes, then evaluates each
candidate by running a short `train` + `evaluate` cycle through the
pipeline plugin and scoring via the configured metric or agent.fitness(summary, config).
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import random
import tempfile as _tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from deap import base, creator, tools


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (_repo_root() / path).resolve()


def _atomic_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resume_contract_hash(config: Dict[str, Any], schema) -> str:
    keys = (
        "env_plugin", "agent_plugin", "pipeline_plugin", "optimizer_plugin",
        "input_data_file", "validation_data_file", "asset", "timeframe",
        "train_start", "train_end", "validation_start", "validation_end",
        "selection_metric", "optimization_metric", "metric_schema",
        "optimization_stages", "hyperparameter_bounds", "ga_population",
        "ga_cxpb", "ga_mutpb", "ga_seed", "risk_penalty_lambda",
        "l1_generalization_gap_penalty_beta",
    )
    payload = {
        "parameter_schema": [list(item) for item in schema],
        "config": {key: config.get(key) for key in keys},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _swap_split_path(path: str, split: str) -> str:
    """Swap d4 → d{split} in a dataset path. Preserves _norm suffix."""
    if not path:
        return path
    base, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)
    # stem is e.g. "d4" or "d4_norm"
    if stem.startswith("d4"):
        new_stem = split + stem[2:]
        return os.path.join(base, new_stem + ext)
    return path


_CREATOR_READY = False


def _ensure_creator() -> None:
    global _CREATOR_READY
    if _CREATOR_READY:
        return
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    _CREATOR_READY = True


class Plugin:
    plugin_params: Dict[str, Any] = {
        "ga_population": 8,
        "ga_generations": 4,
        "ga_cxpb": 0.5,
        "ga_mutpb": 0.2,
        "ga_eval_timesteps": 2_000,
        "ga_seed": 0,
        "optimization_patience": 3,
        "optimization_capture_model_artifact": False,
        "optimization_require_model_artifact": False,
    }

    plugin_debug_vars = [
        "ga_population", "ga_generations", "ga_cxpb", "ga_mutpb",
        "ga_eval_timesteps", "ga_seed", "optimization_patience",
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]) -> None:
        debug_info.update(self.get_debug_info())

    # ------------------------------------------------------------------
    def optimize(
        self,
        *,
        env_plugin,
        agent_plugin,
        pipeline_plugin,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        schema = self._effective_schema(agent_plugin.hparam_schema(), config)
        if not schema:
            raise ValueError("agent plugin exposes no hparam_schema; cannot optimize")

        rng = random.Random(int(config.get("ga_seed", self.params["ga_seed"])))
        _ensure_creator()
        n_pop = int(config.get("ga_population", self.params["ga_population"]))
        if n_pop < 1:
            raise ValueError("ga_population must be at least 1")
        cxpb = float(config.get("ga_cxpb", self.params["ga_cxpb"]))
        mutpb = float(config.get("ga_mutpb", self.params["ga_mutpb"]))
        callbacks = config.get("optimization_callbacks") or {}
        stages = self._build_stage_schedule(schema, config)
        base_params = self._initial_params(schema, config)
        resume = self._load_resume(config, schema)
        start_stage = 0
        start_generation = 0
        global_best = None
        global_best_fitness = float("-inf")
        total_evaluations = 0
        stage_summaries: list[dict[str, Any]] = []

        if resume:
            base_params.update(resume["best_params"])
            global_best_fitness = float(resume["best_fitness"])
            requested_stage = int(resume.get("stage_index", 0))
            if requested_stage >= len(stages):
                return self._completed_resume_result(resume)
            else:
                start_stage = requested_stage
                start_generation = int(resume.get("generation_in_stage", -1)) + 1

        toolbox = base.Toolbox()
        toolbox.register(
            "evaluate", self._evaluate, schema=schema, env_plugin=env_plugin,
            agent_plugin=agent_plugin, pipeline_plugin=pipeline_plugin, config=config,
        )
        toolbox.register(
            "select", self._select_tournament, tournsize=min(3, n_pop), rng=rng
        )

        for stage_index, stage in enumerate(stages):
            if stage_index < start_stage:
                continue
            generation_start = start_generation if stage_index == start_stage else 0
            stage_info = {
                "stage": stage_index + 1,
                "total_stages": len(stages),
                "stage_name": stage["name"],
                "active_params": list(stage["active_params"]),
                "frozen_params": list(stage["frozen_params"]),
                "n_generations_stage": stage["generations"],
                "n_generations_total": sum(item["generations"] for item in stages),
            }
            _call_callback(callbacks, "on_stage_start", stage_index + 1, len(stages))
            active_indices = {
                index for index, item in enumerate(schema)
                if item[0] in set(stage["active_params"])
            }
            population = self._build_stage_population(
                schema, base_params, active_indices, n_pop, rng
            )
            if global_best_fitness > float("-inf"):
                seeded = self._encode(base_params, schema)
                seeded.fitness.values = (global_best_fitness,)
                seeded.evaluation_metrics = dict(resume.get("best_metrics", {})) if resume else {}
                self._replace_worst(population, seeded)

            patience = int(stage["patience"])
            no_improve = 0
            stage_best_fitness = float("-inf")
            completed_generation = generation_start - 1
            for generation in range(generation_start, int(stage["generations"]) + 1):
                if generation == generation_start:
                    candidates = population
                else:
                    incoming = _call_callback(callbacks, "network_champion_provider")
                    if isinstance(incoming, dict):
                        self._replace_worst(population, self._encode(incoming, schema))
                    if bool(_call_callback(callbacks, "stage_advance_requested")):
                        break
                    candidates = list(map(toolbox.clone, toolbox.select(population, len(population))))
                    for first, second in zip(candidates[::2], candidates[1::2]):
                        if rng.random() < cxpb:
                            self._mate_active(first, second, schema, active_indices, rng)
                            self._invalidate(first)
                            self._invalidate(second)
                    for candidate in candidates:
                        if rng.random() < mutpb:
                            self._mutate_active(candidate, schema, active_indices, rng, indpb=0.3)
                            self._invalidate(candidate)

                evaluated = self._evaluate_population(
                    candidates, toolbox, schema=schema, callbacks=callbacks,
                    generation=generation, stage_info=stage_info, config=config,
                    evaluation_offset=total_evaluations,
                )
                total_evaluations += evaluated
                population[:] = candidates
                stage_best = tools.selBest(population, 1)[0]
                current = float(stage_best.fitness.values[0])
                completed_generation = generation
                if current > stage_best_fitness + 1e-9:
                    stage_best_fitness = current
                    no_improve = 0
                else:
                    no_improve += 1
                if current > global_best_fitness + 1e-9:
                    global_best = toolbox.clone(stage_best)
                    global_best_fitness = current
                    base_params = self._decode(global_best, schema)
                    self._persist_champion_model(
                        config, getattr(global_best, "evaluation_metrics", {})
                    )
                    self._notify_champion(
                        callbacks, base_params, current, generation,
                        getattr(global_best, "evaluation_metrics", {}), stage_info,
                    )
                self._write_resume(
                    config, schema, stage_index, generation, base_params,
                    global_best_fitness,
                    getattr(global_best, "evaluation_metrics", {}) if global_best else {},
                )
                _call_callback(
                    callbacks, "on_generation_end", population, [],
                    list(stage["active_params"]), generation, stage_info,
                    {"best_fitness": current, "global_best_fitness": global_best_fitness},
                )
                if not config.get("quiet_mode"):
                    print(
                        f"[optimizer] stage {stage_index + 1}/{len(stages)} "
                        f"{stage['name']} gen {generation} best={current:.6f} "
                        f"global={global_best_fitness:.6f} patience={no_improve}/{patience}"
                    )
                if no_improve >= patience:
                    break

            if global_best is None:
                global_best = toolbox.clone(tools.selBest(population, 1)[0])
                global_best_fitness = float(global_best.fitness.values[0])
                base_params = self._decode(global_best, schema)
            stage_summary = {
                "stage_index": stage_index,
                "stage_name": stage["name"],
                "active_params": list(stage["active_params"]),
                "completed_generation": completed_generation,
                "stage_best_fitness": stage_best_fitness,
                "global_best_fitness": global_best_fitness,
            }
            stage_summaries.append(stage_summary)
            _call_callback(
                callbacks, "on_stage_end", stage_index + 1, len(stages),
                base_params, global_best_fitness,
                {"fitness": global_best_fitness, **dict(getattr(global_best, "evaluation_metrics", {}))},
            )
            self._write_resume(
                config, schema, stage_index + 1, -1, base_params,
                global_best_fitness, getattr(global_best, "evaluation_metrics", {}),
            )
            resume = None
            start_generation = 0

        if global_best is None:
            raise RuntimeError("optimizer completed without evaluating a candidate")
        best_params = self._decode(global_best, schema)
        best_metrics = dict(getattr(global_best, "evaluation_metrics", {}))
        best_params["_best_fitness"] = global_best_fitness
        best_model_b64 = best_metrics.pop("_model_b64", None)
        if best_model_b64:
            best_params["_best_model_b64"] = best_model_b64
        best_params["_best_metrics"] = best_metrics
        self._write_final_outputs(
            config, best_params, global_best_fitness, best_metrics,
            stage_summaries, total_evaluations,
        )
        return best_params

    @staticmethod
    def _effective_schema(raw_schema, config: Dict[str, Any]):
        schema: List[Tuple[str, float, float, str]] = list(raw_schema)
        declared = config.get("hyperparameter_bounds")
        if not declared:
            return schema
        if not isinstance(declared, dict):
            raise ValueError("hyperparameter_bounds must be an object")
        hard = {name: (float(low), float(high), kind) for name, low, high, kind in schema}
        unknown = sorted(set(declared) - set(hard))
        if unknown:
            raise ValueError(f"hyperparameter_bounds has unknown params: {unknown}")
        effective = []
        for name, hard_low, hard_high, kind in schema:
            bounds = declared.get(name, [hard_low, hard_high])
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"hyperparameter_bounds[{name!r}] must be [low, high]")
            low, high = float(bounds[0]), float(bounds[1])
            if low > high:
                raise ValueError(f"hyperparameter_bounds[{name!r}] has low > high")
            if low < hard_low or high > hard_high:
                raise ValueError(
                    f"hyperparameter_bounds[{name!r}]={bounds!r} exceeds "
                    f"agent safety bounds [{hard_low}, {hard_high}]"
                )
            effective.append((name, low, high, kind))
        return effective

    @staticmethod
    def _select_tournament(population, k: int, *, tournsize: int, rng: random.Random):
        selected = []
        for _ in range(k):
            aspirants = [rng.choice(population) for _ in range(tournsize)]
            selected.append(max(aspirants, key=lambda item: item.fitness.values[0]))
        return selected

    @staticmethod
    def _completed_resume_result(resume: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(resume["best_params"])
        metrics = dict(resume.get("best_metrics") or {})
        encoded = metrics.pop("_model_b64", None)
        result["_best_fitness"] = float(resume["best_fitness"])
        result["_best_metrics"] = metrics
        if encoded:
            result["_best_model_b64"] = encoded
        return result

    def _build_stage_schedule(self, schema, config: Dict[str, Any]) -> list[dict[str, Any]]:
        names = [item[0] for item in schema]
        raw = config.get("optimization_stages")
        if not raw:
            raw = [{
                "name": "all",
                "params": "all",
                "generations": int(config.get("ga_generations", self.params["ga_generations"])),
            }]
        stages = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"optimization_stages[{index}] must be an object")
            requested = item.get("params", "all")
            active = names if requested == "all" else list(requested)
            unknown = sorted(set(active) - set(names))
            if unknown:
                raise ValueError(
                    f"optimization stage {item.get('name', index)!r} has unknown params: {unknown}"
                )
            generations = int(item.get("generations", 1))
            if generations < 0:
                raise ValueError("stage generations cannot be negative")
            stages.append({
                "name": str(item.get("name") or f"stage_{index + 1}"),
                "active_params": active,
                "frozen_params": [name for name in names if name not in active],
                "generations": generations,
                "patience": int(item.get(
                    "patience", config.get("optimization_patience", self.params["optimization_patience"])
                )),
            })
        return stages

    def _initial_params(self, schema, config: Dict[str, Any]) -> dict[str, Any]:
        params = {}
        for name, low, high, kind in schema:
            value = config.get(name, (float(low) + float(high)) / 2.0)
            value = max(float(low), min(float(high), float(value)))
            params[name] = int(round(value)) if kind == "int" else value
        initial = config.get("initial_candidate_params")
        if isinstance(initial, dict):
            params.update({key: value for key, value in initial.items() if key in params})
        return self._decode(self._encode(params, schema), schema)

    def _build_stage_population(self, schema, baseline, active_indices, size, rng):
        population = [self._encode(baseline, schema)]
        for _ in range(1, size):
            values = list(self._encode(baseline, schema))
            for index in active_indices:
                _name, low, high, kind = schema[index]
                values[index] = (
                    float(rng.randint(int(low), int(high)))
                    if kind == "int" else rng.uniform(float(low), float(high))
                )
            population.append(creator.Individual(values))
        return population

    @staticmethod
    def _invalidate(individual) -> None:
        if individual.fitness.valid:
            del individual.fitness.values

    def _mate_active(self, first, second, schema, active_indices, rng, alpha=0.5):
        for index in active_indices:
            gamma = (1.0 + 2.0 * alpha) * rng.random() - alpha
            x1, x2 = first[index], second[index]
            first[index] = (1.0 - gamma) * x1 + gamma * x2
            second[index] = gamma * x1 + (1.0 - gamma) * x2
            _name, low, high, _kind = schema[index]
            first[index] = max(float(low), min(float(high), first[index]))
            second[index] = max(float(low), min(float(high), second[index]))
        return first, second

    def _mutate_active(self, individual, schema, active_indices, rng, indpb):
        for index in active_indices:
            if rng.random() >= indpb:
                continue
            _name, low, high, kind = schema[index]
            individual[index] = (
                float(rng.randint(int(low), int(high)))
                if kind == "int" else rng.uniform(float(low), float(high))
            )
        return (individual,)

    # ------------------------------------------------------------------
    def _make_individual(self, schema: Sequence[Tuple[str, float, float, str]], rng: random.Random):
        values = []
        for _name, low, high, kind in schema:
            if kind == "int":
                values.append(float(rng.randint(int(low), int(high))))
            else:
                values.append(rng.uniform(float(low), float(high)))
        return creator.Individual(values)

    def _mutate(self, individual, schema, rng: random.Random, indpb: float):
        for i, (_name, low, high, kind) in enumerate(schema):
            if rng.random() < indpb:
                if kind == "int":
                    individual[i] = float(rng.randint(int(low), int(high)))
                else:
                    individual[i] = rng.uniform(float(low), float(high))
        return (individual,)

    def _decode(self, individual, schema: Sequence[Tuple[str, float, float, str]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for (name, low, high, kind), raw in zip(schema, individual):
            if kind == "int":
                val = int(round(float(raw)))
                val = max(int(low), min(int(high), val))
                out[name] = val
            else:
                val = float(raw)
                val = max(float(low), min(float(high), val))
                out[name] = val
        return out

    def _encode(self, params: Dict[str, Any], schema: Sequence[Tuple[str, float, float, str]]) -> Any:
        values = []
        for name, low, high, kind in schema:
            value = params.get(name, low)
            value = max(float(low), min(float(high), float(value)))
            values.append(float(round(value)) if kind == "int" else value)
        return creator.Individual(values)

    def _evaluate_population(
        self, population, toolbox, *, schema, callbacks, generation: int,
        stage_info: Dict[str, Any], config: Dict[str, Any], evaluation_offset: int,
    ) -> int:
        evaluated = 0
        for candidate_index, individual in enumerate(population):
            if individual.fitness.valid:
                continue
            evaluation = toolbox.evaluate(individual)
            if isinstance(evaluation, tuple) and len(evaluation) == 2:
                fitness = float(evaluation[0])
                metrics = dict(evaluation[1] or {})
            else:
                fitness = float(evaluation)
                metrics = {}
            individual.fitness.values = (fitness,)
            individual.evaluation_metrics = metrics
            evaluated += 1
            public_metrics = {
                key: value for key, value in metrics.items() if key != "_model_b64"
            }
            candidate_record = {
                "parameters": self._decode(individual, schema),
                "fitness": fitness,
                "generation": generation,
                "candidate_index": candidate_index,
                "total_evaluation": evaluation_offset + evaluated,
                "stage_index": int(stage_info["stage"]) - 1,
                "stage_name": stage_info["stage_name"],
                "metrics": public_metrics,
                **public_metrics,
            }
            self._append_candidate_history(config, candidate_record)
            _call_callback(
                callbacks, "on_candidate_evaluated",
                candidate_record,
            )
        return evaluated

    @staticmethod
    def _replace_worst(population, candidate) -> None:
        if not population:
            return
        worst_index = min(
            range(len(population)),
            key=lambda index: population[index].fitness.values[0]
            if population[index].fitness.valid else float("-inf"),
        )
        population[worst_index] = candidate

    @staticmethod
    def _notify_champion(
        callbacks, params, fitness: float, generation: int, metrics: Dict[str, Any],
        stage_info: Dict[str, Any],
    ) -> None:
        _call_callback(
            callbacks, "on_new_champion", params, fitness,
            {"fitness": fitness, **dict(metrics)},
            generation,
            stage_info,
        )

    def _load_resume(self, config: Dict[str, Any], schema) -> Dict[str, Any] | None:
        if not bool(config.get("optimization_resume", False)):
            return None
        path = _resolve_repo_path(config.get("optimization_resume_file"))
        if path is None or not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "agent_multi.optimization_resume.v1":
            raise ValueError(f"unsupported optimization resume schema: {path}")
        expected = [item[0] for item in schema]
        if payload.get("parameter_names") != expected:
            raise ValueError("optimization resume parameter schema mismatch")
        expected_contract_hash = _resume_contract_hash(config, schema)
        stored_contract_hash = payload.get("resume_contract_hash")
        if stored_contract_hash and expected_contract_hash != stored_contract_hash:
            raise ValueError("optimization resume contract hash mismatch")
        model_path = _resolve_repo_path(config.get("optimization_champion_model_file"))
        if model_path and model_path.is_file():
            model_bytes = model_path.read_bytes()
            expected_model_hash = payload.get("model_artifact_sha256")
            actual_model_hash = hashlib.sha256(model_bytes).hexdigest()
            if expected_model_hash and expected_model_hash != actual_model_hash:
                raise ValueError("optimization resume champion model hash mismatch")
            payload.setdefault("best_metrics", {})["_model_b64"] = (
                base64.b64encode(model_bytes).decode("ascii")
            )
        return payload

    def _write_resume(
        self, config, schema, stage_index, generation, best_params,
        best_fitness, best_metrics,
    ) -> None:
        path = _resolve_repo_path(config.get("optimization_resume_file"))
        if path is None:
            return
        _atomic_json_dump(path, {
            "schema_version": "agent_multi.optimization_resume.v1",
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "canonical_config_hash": config.get("canonical_config_hash"),
            "resume_contract_hash": _resume_contract_hash(config, schema),
            "parameter_names": [item[0] for item in schema],
            "stage_index": stage_index,
            "generation_in_stage": generation,
            "best_fitness": best_fitness,
            "best_params": best_params,
            "best_metrics": {
                key: value for key, value in dict(best_metrics or {}).items()
                if key != "_model_b64"
            },
            "model_artifact_sha256": dict(best_metrics or {}).get("model_artifact_sha256"),
            "model_artifact_path": str(
                _resolve_repo_path(config.get("optimization_champion_model_file")) or ""
            ),
        })

    def _persist_champion_model(self, config, metrics: Dict[str, Any]) -> None:
        encoded = metrics.get("_model_b64")
        path = _resolve_repo_path(config.get("optimization_champion_model_file"))
        if not encoded or path is None:
            return
        model_bytes = base64.b64decode(encoded, validate=True)
        expected = metrics.get("model_artifact_sha256")
        actual = hashlib.sha256(model_bytes).hexdigest()
        if expected and expected != actual:
            raise ValueError("champion model hash differs from metric evidence")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(model_bytes)
        temporary.replace(path)

    def _append_candidate_history(self, config, record: Dict[str, Any]) -> None:
        path = _resolve_repo_path(config.get("optimization_candidate_history"))
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "timestamp_utc", "total_evaluation", "stage_index", "stage_name",
            "generation", "candidate_index", "fitness", "parameters_json",
            "metrics_json", "model_artifact_sha256",
        ]
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_evaluation": record["total_evaluation"],
            "stage_index": record["stage_index"],
            "stage_name": record["stage_name"],
            "generation": record["generation"],
            "candidate_index": record["candidate_index"],
            "fitness": record["fitness"],
            "parameters_json": json.dumps(record["parameters"], sort_keys=True),
            "metrics_json": json.dumps(record["metrics"], sort_keys=True, default=str),
            "model_artifact_sha256": record["metrics"].get("model_artifact_sha256"),
        }
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _write_final_outputs(
        self, config, best_params, best_fitness, best_metrics,
        stage_summaries, total_evaluations,
    ) -> None:
        public_params = {key: value for key, value in best_params.items() if not key.startswith("_")}
        public_metrics = {
            key: value for key, value in best_metrics.items() if key != "_model_b64"
        }
        params_path = _resolve_repo_path(config.get("optimization_parameters_file"))
        if params_path:
            _atomic_json_dump(params_path, {
                "schema_version": "agent_multi.optimization_parameters.v1",
                "canonical_config_hash": config.get("canonical_config_hash"),
                "fitness": best_fitness,
                "parameters": public_params,
                "metrics": public_metrics,
            })
        stats_path = _resolve_repo_path(config.get("optimization_statistics"))
        if stats_path:
            _atomic_json_dump(stats_path, {
                "schema_version": "agent_multi.optimization_statistics.v1",
                "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "canonical_config_hash": config.get("canonical_config_hash"),
                "best_fitness": best_fitness,
                "total_evaluations": total_evaluations,
                "stages": stage_summaries,
                "metric_schema": config.get("metric_schema"),
                "optimization_metric": config.get("optimization_metric"),
            })

    def _evaluate(
        self,
        individual,
        *,
        schema,
        env_plugin,
        agent_plugin,
        pipeline_plugin,
        config: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        candidate_params = self._decode(individual, schema)
        run_config = dict(config)
        run_config.update(candidate_params)
        run_config["total_timesteps"] = int(
            config.get("ga_eval_timesteps", self.params["ga_eval_timesteps"])
        )
        run_config["load_model"] = None
        run_config["quiet_mode"] = True
        run_config["write_results_sidecar"] = False

        # P2c.3 — GA fitness split: "train" scores on the d4 train env
        # (legacy); "val" trains on d4 then evaluates on d5 with frozen weights.
        split = str(config.get("ga_fitness_split", "train")).lower()
        capture_model = bool(config.get(
            "optimization_capture_model_artifact",
            self.params["optimization_capture_model_artifact"],
        ))
        require_model = bool(config.get(
            "optimization_require_model_artifact",
            self.params["optimization_require_model_artifact"],
        ))
        tmp_model: str | None = None
        if split == "val" or capture_model:
            tmp_model = str(_tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name)
            run_config["save_model"] = tmp_model
        else:
            run_config["save_model"] = None

        agent_plugin.set_params(**run_config)
        try:
            summary = pipeline_plugin.run_pipeline(
                config=run_config,
                env_plugin=env_plugin,
                agent_plugin=agent_plugin,
                mode="train",
            )
        except Exception as exc:
            print(f"[optimizer] candidate failed: {exc}")
            if tmp_model:
                _safe_unlink(tmp_model)
            return -1e9, {"evaluation_error": str(exc)}

        if split == "val" and tmp_model is not None:
            val_config = dict(run_config)
            val_config["input_data_file"] = (
                run_config.get("validation_data_file")
                or _swap_split_path(run_config.get("input_data_file", ""), "d5")
            )
            val_config["load_model"] = tmp_model
            val_config["save_model"] = None
            try:
                summary = pipeline_plugin.run_pipeline(
                    config=val_config,
                    env_plugin=env_plugin,
                    agent_plugin=agent_plugin,
                    mode="inference",
                )
            except Exception as exc:
                print(f"[optimizer] val eval failed: {exc}")
                _safe_unlink(tmp_model)
                return -1e9, {"evaluation_error": str(exc)}

        from app.metrics import compute_optimization_fitness

        fitness = compute_optimization_fitness(summary, run_config, agent_plugin)
        metrics = _metric_payload(summary)
        metrics["fitness"] = float(fitness)
        metrics["optimization_metric"] = str(
            run_config.get("optimization_metric")
            or run_config.get("metric_type")
            or "agent_fitness"
        )
        if capture_model and tmp_model:
            model_path = os.path.abspath(tmp_model)
            try:
                model_bytes = Path(model_path).read_bytes()
            except OSError as exc:
                if require_model:
                    _safe_unlink(tmp_model)
                    return -1e9, {"evaluation_error": f"model artifact missing: {exc}"}
            else:
                metrics["_model_b64"] = base64.b64encode(model_bytes).decode("ascii")
                metrics["model_artifact_sha256"] = hashlib.sha256(model_bytes).hexdigest()
                metrics["model_artifact_bytes"] = len(model_bytes)
                metrics["model_artifact_format"] = "stable_baselines3_zip"
        if tmp_model:
            _safe_unlink(tmp_model)
        if not config.get("quiet_mode"):
            print(
                f"[optimizer] candidate {candidate_params} "
                f"split={split} → fitness={fitness:.6f}"
            )
        return float(fitness), metrics


def _metric_payload(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe metric vector without dropping useful nested evidence."""

    def convert(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        try:
            return value.item()
        except (AttributeError, ValueError):
            return str(value)

    return {str(key): convert(value) for key, value in summary.items()}


def _call_callback(callbacks: Dict[str, Any], name: str, *args):
    callback = callbacks.get(name)
    if not callable(callback):
        return None
    return callback(*args)
