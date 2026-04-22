"""
default_optimizer.py — DEAP GA over agent hyperparameters.

Uses the `hparam_schema()` of an agent plugin (list of
(name, low, high, type)) to build chromosomes, then evaluates each
candidate by running a short `train` + `evaluate` cycle through the
pipeline plugin and scoring via agent.fitness(summary, config).
"""
from __future__ import annotations

import os
import random
import tempfile as _tempfile
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
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

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

        population = toolbox.population(n=n_pop)
        for ind in population:
            ind.fitness.values = (toolbox.evaluate(ind),)

        best = tools.selBest(population, 1)[0]
        if not config.get("quiet_mode"):
            print(f"[optimizer] gen 0 best fitness = {best.fitness.values[0]:.6f}")

        for gen in range(1, n_gen + 1):
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
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = (toolbox.evaluate(ind),)
            population[:] = offspring
            best = tools.selBest(population, 1)[0]
            if not config.get("quiet_mode"):
                print(f"[optimizer] gen {gen} best fitness = {best.fitness.values[0]:.6f}")

        best_params = self._decode(best, schema)
        best_params["_best_fitness"] = float(best.fitness.values[0])
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

    def _evaluate(
        self,
        individual,
        *,
        schema,
        env_plugin,
        agent_plugin,
        pipeline_plugin,
        config: Dict[str, Any],
    ) -> float:
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
        tmp_model: str | None = None
        if split == "val":
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
            return -1e9

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
                return -1e9
            finally:
                _safe_unlink(tmp_model)

        fitness = agent_plugin.fitness(summary, run_config)
        if not config.get("quiet_mode"):
            print(
                f"[optimizer] candidate {candidate_params} "
                f"split={split} → fitness={fitness:.6f}"
            )
        return float(fitness)
