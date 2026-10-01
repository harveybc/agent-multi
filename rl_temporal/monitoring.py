"""Heartbeat, hard limits and validation-episode early stopping for SB3 ``learn()``.

``rl_pipeline_with_validation`` already runs an epoch loop with validation
early stopping and best-checkpoint restore; these callbacks make the same
mechanisms available inside one ``learn()`` call (cost pilots, direct runs)
and testable in-process. Selection reads validation episodes only: a training
episode's reward is never a held-out number.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback

HEARTBEAT_SCHEMA = "rl_temporal.heartbeat.v1"
MAX_HEARTBEAT_INTERVAL_S = 60.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HeartbeatCallback(BaseCallback):
    """Atomic JSON heartbeat at most ``interval_s`` apart (never above 60 s)."""

    def __init__(self, path, *, interval_s: float = 30.0, extra: Optional[Dict[str, Any]] = None):
        super().__init__(verbose=0)
        if not (0.0 <= float(interval_s) <= MAX_HEARTBEAT_INTERVAL_S):
            raise ValueError(f"heartbeat interval must be within [0, {MAX_HEARTBEAT_INTERVAL_S:.0f}] s; got {interval_s}")
        self.path = Path(path)
        self.interval_s = float(interval_s)
        self.extra = dict(extra or {})
        self._started = time.monotonic()
        self._last = -1e9
        self.writes = 0

    def _write(self, phase: str) -> None:
        doc = {"schema": HEARTBEAT_SCHEMA, "phase": phase, "written_at": _now(), "pid": os.getpid(),
               "num_timesteps": int(getattr(self.model, "num_timesteps", 0) or 0),
               "elapsed_s": time.monotonic() - self._started, "interval_s": self.interval_s, **self.extra}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".hb-")
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, sort_keys=True)
        os.replace(tmp, self.path)
        self._last = time.monotonic()
        self.writes += 1

    def _on_training_start(self) -> None:
        self._write("training_started")

    def _on_step(self) -> bool:
        if time.monotonic() - self._last >= self.interval_s:
            self._write("training")
        return True

    def _on_training_end(self) -> None:
        self._write("training_ended")


class HardLimitCallback(BaseCallback):
    """Stop ``learn()`` at a timestep or wall-clock ceiling; records why."""

    def __init__(self, *, max_timesteps: Optional[int] = None, max_wall_s: Optional[float] = None):
        super().__init__(verbose=0)
        if max_timesteps is None and max_wall_s is None:
            raise ValueError("declare at least one hard limit")
        self.max_timesteps = None if max_timesteps is None else int(max_timesteps)
        self.max_wall_s = None if max_wall_s is None else float(max_wall_s)
        self.stop_reason: Optional[str] = None
        self._started = time.monotonic()

    def _on_training_start(self) -> None:
        self._started = time.monotonic()

    def _on_step(self) -> bool:
        if self.max_timesteps is not None and self.model.num_timesteps >= self.max_timesteps:
            self.stop_reason = "max_timesteps"
            return False
        if self.max_wall_s is not None and time.monotonic() - self._started >= self.max_wall_s:
            self.stop_reason = "max_wall_s"
            return False
        return True


class ValidationEpisodeEarlyStopping(BaseCallback):
    """Evaluate on chronological validation episodes every ``eval_every_steps``,
    keep the best policy on disk, stop after ``patience`` non-improving
    evaluations and restore the best on request."""

    def __init__(self, *, evaluate_fn: Callable[[Any], Dict[str, Any]], eval_every_steps: int,
                 patience: int, best_path, min_delta: float = 0.0, selection_key: str = "selection_value",
                 max_evaluations: Optional[int] = None):
        super().__init__(verbose=0)
        if int(eval_every_steps) <= 0 or int(patience) <= 0:
            raise ValueError("eval_every_steps and patience must be positive")
        self.evaluate_fn = evaluate_fn
        self.eval_every_steps = int(eval_every_steps)
        self.patience = int(patience)
        self.best_path = Path(best_path)
        self.min_delta = float(min_delta)
        self.selection_key = selection_key
        self.max_evaluations = max_evaluations
        self.history: List[Dict[str, Any]] = []
        self.best_value = float("-inf")
        self.best_index: Optional[int] = None
        self.no_improve = 0
        self.stopped_early = False
        self.stop_reason: Optional[str] = None
        self._last_eval_step = 0

    @property
    def evaluations(self) -> int:
        return len(self.history)

    def _evaluate(self) -> bool:
        result = dict(self.evaluate_fn(self.model))
        value = float(result[self.selection_key])
        entry = {"index": len(self.history), "num_timesteps": int(self.model.num_timesteps),
                 "selection_value": value, "source": "validation_episode", "at": _now(), **result}
        self.history.append(entry)
        if value > self.best_value + self.min_delta:
            self.best_value, self.best_index, self.no_improve = value, entry["index"], 0
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_path)
            entry["saved_best"] = True
        else:
            self.no_improve += 1
            entry["saved_best"] = False
        if self.no_improve >= self.patience:
            self.stopped_early, self.stop_reason = True, "patience"
            return False
        if self.max_evaluations is not None and len(self.history) >= self.max_evaluations:
            self.stop_reason = "max_evaluations"
            return False
        return True

    def _on_step(self) -> bool:
        if self.model.num_timesteps - self._last_eval_step >= self.eval_every_steps:
            self._last_eval_step = self.model.num_timesteps
            return self._evaluate()
        return True

    def restore_best(self, model) -> Dict[str, Any]:
        if self.best_index is None:
            raise RuntimeError("no best policy was recorded")
        model.set_parameters(str(self.best_path), exact_match=True, device=model.device)
        return {"restored_from": str(self.best_path), "best_index": self.best_index, "best_value": self.best_value}

    def record(self) -> Dict[str, Any]:
        return {"selection_source": "validation_episode", "train_episode_reward_is_selection": False,
                "eval_every_steps": self.eval_every_steps, "patience": self.patience, "min_delta": self.min_delta,
                "evaluations": self.evaluations, "best_index": self.best_index, "best_value": self.best_value,
                "stopped_early": self.stopped_early, "stop_reason": self.stop_reason, "history": list(self.history)}


# ----------------------------------------------------------------------------
# chronological validation episodes
# ----------------------------------------------------------------------------

class _ValidationFactory:
    def __init__(self, cfg: Dict[str, Any], description: Dict[str, Any], sliced_csv: Path, tmpdir):
        self._cfg, self.description, self._csv, self._tmpdir = cfg, description, sliced_csv, tmpdir

    def __call__(self):
        from pipeline_plugins._nested_splits import ContextPrefixWrapper
        from env_plugins.gym_fx_env import Plugin as EnvPlugin

        cfg = dict(self._cfg)
        cfg.update(input_data_file=str(self._csv), max_rows=None, env_mode="validation")
        env = EnvPlugin().make_env(cfg)
        wrapped = ContextPrefixWrapper(env, self.description["context_rows"])
        wrapped.unwrapped_env = env
        return wrapped


def chronological_validation_factory(cfg: Dict[str, Any], *, train_rows, validation_rows):
    """Validation episodes over ``validation_rows`` with only the scaler/window
    context before them (forced hold, no account mutation) and nothing after."""
    import pandas as pd

    t0, t1 = int(train_rows[0]), int(train_rows[1])
    v0, v1 = int(validation_rows[0]), int(validation_rows[1])
    if not (t0 <= t1 <= v0 < v1):
        raise ValueError("validation rows must follow training rows chronologically")
    context = int(cfg.get("window_size", 32)) + int(cfg.get("feature_scaling_window", 256))
    start = max(0, v0 - context)
    frame = pd.read_csv(cfg["input_data_file"], header=0 if cfg.get("headers", True) else None)
    sliced = frame.iloc[start:v1]
    tmpdir = tempfile.TemporaryDirectory(prefix="rl-temporal-validation-")
    path = Path(tmpdir.name) / "validation.csv"
    sliced.to_csv(path, index=False)
    description = {"train_rows": [t0, t1], "validation_rows": [v0, v1], "context_rows": v0 - start,
                   "first_decision_row": v0, "rows_written": int(len(sliced)), "source_csv": cfg["input_data_file"]}
    return _ValidationFactory(cfg, description, path, tmpdir)


def make_validation_evaluator(factory, *, episodes: int = 1, deterministic: bool = True,
                              selection_metric: str = "net_return") -> Callable[[Any], Dict[str, Any]]:
    """Run deterministic policy episodes on the factory's env and summarize them."""
    from .reconciliation import reconcile_episode

    def evaluate(model) -> Dict[str, Any]:
        records = []
        for _ in range(int(episodes)):
            env = factory()
            try:
                from gymnasium.wrappers import FlattenObservation

                flat = FlattenObservation(env)

                def policy(step, info, _m=model, _f=flat):
                    obs = policy.last_obs
                    action, _ = _m.predict(obs, deterministic=deterministic)
                    return action

                rec = reconcile_episode(flat, actions=policy, policy_sees_obs=True)
                records.append(rec)
            finally:
                env.close()
        mean = lambda k: sum(float(r[k]) for r in records) / len(records)
        return {"selection_value": mean(selection_metric), "validation_episodes": len(records),
                "net_return": mean("net_return"), "max_drawdown_fraction": mean("max_drawdown_fraction"),
                "trades_closed": mean("trades_closed"), "turnover_units": mean("turnover_units"),
                "exposure_fraction": mean("exposure_fraction"), "episodes": records}

    return evaluate
