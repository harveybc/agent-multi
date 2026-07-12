"""
default_optimizer.py — DEAP GA over agent hyperparameters.

Uses the `hparam_schema()` of an agent plugin (list of
(name, low, high, type)) to build chromosomes, then evaluates each
candidate by running a short `train` + `evaluate` cycle through the
pipeline plugin and scoring via the configured metric or agent.fitness(summary, config).
"""
from __future__ import annotations

import base64
import hashlib
import os
import random
import tempfile as _tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from deap import base, creator, tools


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
        schema: List[Tuple[str, float, float, str]] = list(agent_plugin.hparam_schema())
        if not schema:
            raise ValueError("agent plugin exposes no hparam_schema; cannot optimize")

        rng = random.Random(int(config.get("ga_seed", self.params["ga_seed"])))
        _ensure_creator()

        toolbox = base.Toolbox()
        toolbox.register("individual", self._make_individual, schema=schema, rng=rng)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register(
            "evaluate",
            self._evaluate,
            schema=schema,
            env_plugin=env_plugin,
            agent_plugin=agent_plugin,
            pipeline_plugin=pipeline_plugin,
            config=config,
        )
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", self._mutate, schema=schema, rng=rng, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=3)

        n_pop = int(config.get("ga_population", self.params["ga_population"]))
        n_gen = int(config.get("ga_generations", self.params["ga_generations"]))
        cxpb = float(config.get("ga_cxpb", self.params["ga_cxpb"]))
        mutpb = float(config.get("ga_mutpb", self.params["ga_mutpb"]))

        callbacks = config.get("optimization_callbacks") or {}
        _call_callback(callbacks, "on_stage_start", 1, 1)

        population = toolbox.population(n=n_pop)
        initial = config.get("initial_candidate_params")
        if isinstance(initial, dict):
            self._replace_worst(population, self._encode(initial, schema))
        self._evaluate_population(
            population, toolbox, schema=schema, callbacks=callbacks, generation=0,
        )

        best = tools.selBest(population, 1)[0]
        best_fitness_so_far = float(best.fitness.values[0])
        self._notify_champion(
            callbacks,
            self._decode(best, schema),
            best_fitness_so_far,
            0,
            getattr(best, "evaluation_metrics", {}),
        )
        patience = int(config.get(
            "optimization_patience", self.params["optimization_patience"]
        ))
        no_improve = 0
        if not config.get("quiet_mode"):
            print(
                f"[optimizer] gen 0 best fitness = {best_fitness_so_far:.6f} "
                f"| L2 patience 0/{patience}"
            )

        for gen in range(1, n_gen + 1):
            incoming = _call_callback(callbacks, "network_champion_provider")
            if isinstance(incoming, dict):
                self._replace_worst(population, self._encode(incoming, schema))
            if bool(_call_callback(callbacks, "stage_advance_requested")):
                break
            offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if rng.random() < cxpb:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values
            for mutant in offspring:
                if rng.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            self._evaluate_population(
                offspring, toolbox, schema=schema, callbacks=callbacks, generation=gen,
            )
            population[:] = offspring
            best = tools.selBest(population, 1)[0]
            current = float(best.fitness.values[0])
            if current > best_fitness_so_far + 1e-9:
                best_fitness_so_far = current
                no_improve = 0
                self._notify_champion(
                    callbacks,
                    self._decode(best, schema),
                    current,
                    gen,
                    getattr(best, "evaluation_metrics", {}),
                )
            else:
                no_improve += 1
            if not config.get("quiet_mode"):
                print(
                    f"[optimizer] gen {gen} best fitness = {current:.6f} "
                    f"| global best = {best_fitness_so_far:.6f} "
                    f"| L2 patience {no_improve}/{patience}"
                )
            if no_improve >= patience:
                if not config.get("quiet_mode"):
                    print(
                        f"[optimizer] L2 EARLY STOP at gen {gen} "
                        f"(no improvement for {no_improve} gens, patience={patience})"
                    )
                break

            _call_callback(
                callbacks, "on_generation_end", population, [],
                [name for name, *_ in schema], gen,
                {"stage": 1, "total_stages": 1},
                {"best_fitness": current, "global_best_fitness": best_fitness_so_far},
            )

        best_params = self._decode(best, schema)
        best_metrics = dict(getattr(best, "evaluation_metrics", {}))
        best_params["_best_fitness"] = float(best.fitness.values[0])
        best_model_b64 = best_metrics.pop("_model_b64", None)
        if best_model_b64:
            best_params["_best_model_b64"] = best_model_b64
        best_params["_best_metrics"] = best_metrics
        _call_callback(
            callbacks, "on_stage_end", 1, 1, self._decode(best, schema),
            float(best.fitness.values[0]),
            {"fitness": float(best.fitness.values[0]), **dict(getattr(best, "evaluation_metrics", {}))},
        )
        return best_params

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

    def _evaluate_population(self, population, toolbox, *, schema, callbacks, generation: int) -> None:
        for individual in population:
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
            public_metrics = {
                key: value for key, value in metrics.items() if key != "_model_b64"
            }
            _call_callback(
                callbacks, "on_candidate_evaluated",
                {
                    "parameters": self._decode(individual, schema),
                    "fitness": fitness,
                    "generation": generation,
                    "metrics": public_metrics,
                    **public_metrics,
                },
            )

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
        callbacks, params, fitness: float, generation: int, metrics: Dict[str, Any]
    ) -> None:
        _call_callback(
            callbacks, "on_new_champion", params, fitness,
            {"fitness": fitness, **dict(metrics)},
            generation,
            {"stage": 1, "total_stages": 1},
        )

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
            val_config["input_data_file"] = _swap_split_path(
                run_config.get("input_data_file", ""), "d5"
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
