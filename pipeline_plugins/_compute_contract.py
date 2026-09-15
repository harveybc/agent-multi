"""The compute contract: what was asked for, what was allowed, and what actually ran.

S1 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_COUNTERS_ARCHIVE_AND_TERMS_2026_09_15.md`.

The defect this comes from is a real one Musashi measured: `_observed_work` published
`_n_updates` as "gradient updates". On the installed Stable-Baselines3 that is false for PPO —
`PPO.train` increments it once per **epoch**, outside the minibatch loop, while the optimizer
steps inside it. A counter does not acquire a meaning by sharing a name across algorithms.

Every quantity here is one of three things, and they are never merged:

  declared    what the operator asked for: a training target, a hard transition ceiling, an
              evaluation ceiling. A target is not a ceiling;
  resolved    what the runtime actually built: rollout steps per environment, environment
              count, batch size, optimization epochs, the algorithm and the library version;
  measured    what the run did: transitions collected, rollouts completed, epochs completed
              and optimizer calls **counted by instrumenting the optimizer**, never inferred.

Two traps are handled explicitly rather than assumed away.

* **A resumed model resets its step counter.** `learn()` defaults to `reset_num_timesteps=True`,
  so a model that had already run 64 transitions goes 64 -> 64: the naive delta says the run
  did NO work. `_n_updates` is not reset and keeps accumulating. Measured on SB3 2.9.0.
* **A partial rollout is not training.** On-policy algorithms optimize only on whole rollouts.
  A run stopped inside its first rollout has collected transitions and learned nothing, and it
  is recorded as such instead of being read as a short training run.

The meaning of `_n_updates` per algorithm was measured on SB3 2.9.0, not read from a docstring:
see `tests/test_compute_contract.py::test_the_meaning_table_matches_the_installed_library`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

SCHEMA = "compute_contract.v1"

#: What a delta of `_n_updates` means for each algorithm. Measured, and deliberately partial:
#: an algorithm that is not listed reports the raw delta with meaning UNKNOWN and makes NO
#: epoch or gradient-update claim at all.
N_UPDATES_MEANING = {
    "PPO": "optimization_epochs",   # rollouts x n_epochs; optimizer steps once per minibatch
    "A2C": "gradient_updates",      # one full-batch update per rollout
    "DQN": "gradient_updates",
    "SAC": "gradient_updates",
    "TD3": "gradient_updates",
    "DDPG": "gradient_updates",
}

#: Algorithms that collect whole rollouts before optimizing, so their minimum spend per
#: training call is `n_steps * n_envs` transitions and cannot be subdivided.
ON_POLICY = ("PPO", "A2C")


class ComputeContractRefusal(RuntimeError):
    """A declared limit that the resolved settings cannot honour. Raised BEFORE stepping."""


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return int(value)


def _environment_count(model) -> Optional[int]:
    env = getattr(model, "env", None)
    for holder, name in ((model, "n_envs"), (env, "num_envs")):
        found = _int(getattr(holder, name, None))
        if found:
            return found
    return None


class ComputeMeter:
    """Measures one training call against the limits declared for it.

    Usage is deliberately three explicit steps, because each answers a different question:
    `refuse_if_infeasible()` before anything is built or stepped, `start()` immediately before
    `learn()`, `stop()` immediately after, and `record()` to publish.
    """

    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.algorithm = type(model).__name__
        self.unavailable: list[str] = []
        self._restore = None
        self._optimizer_calls = 0
        self._started = False
        self._before: Dict[str, Optional[int]] = {}
        self._after: Dict[str, Optional[int]] = {}

        self.requested = _int(config.get("total_timesteps"))
        self.cap = _int(config.get("training_transition_cap"))
        self.evaluation_cap = _int(config.get("evaluation_transition_cap"))

        self.n_steps = _int(getattr(model, "n_steps", None))
        self.n_envs = _environment_count(model)
        self.batch_size = _int(getattr(model, "batch_size", None))
        self.n_epochs = _int(getattr(model, "n_epochs", None))

        try:
            import stable_baselines3

            self.library_version = str(stable_baselines3.__version__)
        except Exception:  # pragma: no cover - the library is required to have a model at all
            self.library_version = None
            self.unavailable.append("library_version")

    # -- limits ---------------------------------------------------------------
    @property
    def minimum_training_transitions(self) -> Optional[int]:
        """The smallest amount this algorithm can spend before it can optimize at all."""
        if self.algorithm not in ON_POLICY:
            return None
        if not self.n_steps or not self.n_envs:
            return None
        return self.n_steps * self.n_envs

    def refuse_if_infeasible(self) -> None:
        """Refuse a configuration that cannot honour its ceiling. Never widens the ceiling."""
        if self.cap is None:
            return
        if self.cap <= 0:
            raise ComputeContractRefusal(
                f"training_transition_cap must be positive, got {self.cap}")
        if self.requested is not None and self.requested > self.cap:
            raise ComputeContractRefusal(
                f"requested {self.requested} training transitions against a hard cap of "
                f"{self.cap}: the target may not exceed the ceiling")
        minimum = self.minimum_training_transitions
        if minimum is not None and minimum > self.cap:
            raise ComputeContractRefusal(
                f"{self.algorithm} collects whole rollouts of n_steps={self.n_steps} x "
                f"n_envs={self.n_envs} = {minimum} transitions, which cannot fit a hard cap "
                f"of {self.cap}; lower n_steps or raise the declared cap deliberately")

    # -- measurement ----------------------------------------------------------
    def _snapshot(self) -> Dict[str, Optional[int]]:
        return {"num_timesteps": _int(getattr(self.model, "num_timesteps", None)),
                "n_updates": _int(getattr(self.model, "_n_updates", None))}

    def start(self) -> "ComputeMeter":
        """Instrument the real optimizer and snapshot the lifetime counters."""
        self._before = self._snapshot()
        if self._before["num_timesteps"] is None:
            self.unavailable.append("training_transitions_observed")
        if self._before["n_updates"] is None:
            self.unavailable.append("sb3_n_updates_delta")
        optimizer = getattr(getattr(self.model, "policy", None), "optimizer", None)
        step = getattr(optimizer, "step", None)
        if not callable(step):
            self.unavailable.append("optimizer_step_calls")
        else:
            original = step

            def counted(*args, **kwargs):
                self._optimizer_calls += 1
                return original(*args, **kwargs)

            optimizer.step = counted
            self._restore = (optimizer, original)
        self._started = True
        return self

    def stop(self) -> "ComputeMeter":
        self._after = self._snapshot()
        if self._restore is not None:
            optimizer, original = self._restore
            # The wrapper was set as an INSTANCE attribute over the class's bound method;
            # removing it restores the original lookup instead of pinning a bound method that
            # would outlive this meter and keep counting into it.
            try:
                del optimizer.step
            except AttributeError:
                optimizer.step = original
            self._restore = None
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- derived quantities ---------------------------------------------------
    @property
    def timestep_counter_was_reset(self) -> bool:
        """`learn(reset_num_timesteps=True)` — the default — zeroes a resumed model's counter.

        Detected rather than assumed: if the counter did not advance past where it started,
        it was reset, and the naive delta would report a real training call as no work.
        """
        before, after = self._before.get("num_timesteps"), self._after.get("num_timesteps")
        return before is not None and after is not None and before > 0 and after <= before

    @property
    def training_transitions_observed(self) -> Optional[int]:
        before, after = self._before.get("num_timesteps"), self._after.get("num_timesteps")
        if after is None:
            return None
        if before is None:
            return None
        return after if self.timestep_counter_was_reset else after - before

    @property
    def n_updates_delta(self) -> Optional[int]:
        before, after = self._before.get("n_updates"), self._after.get("n_updates")
        if before is None or after is None:
            return None
        return after - before

    @property
    def collected_rollouts(self) -> Optional[int]:
        """Whole rollouts only. A partial rollout counts as zero, because it never optimized."""
        minimum = self.minimum_training_transitions
        observed = self.training_transitions_observed
        if minimum is None or observed is None:
            return None
        return observed // minimum

    @property
    def partial_rollout(self) -> Optional[bool]:
        minimum = self.minimum_training_transitions
        observed = self.training_transitions_observed
        if minimum is None or observed is None:
            return None
        return observed % minimum != 0 or observed < minimum

    # -- publication ----------------------------------------------------------
    def record(self, *, evaluation_transitions: Optional[int] = None) -> Dict[str, Any]:
        """The counters, each named for what it is, with what could not be measured listed."""
        if not self._started:
            raise RuntimeError("record() before start(): nothing was measured")
        meaning = N_UPDATES_MEANING.get(self.algorithm, "UNKNOWN")
        delta = self.n_updates_delta
        observed = self.training_transitions_observed

        body: Dict[str, Any] = {
            "schema": SCHEMA,
            "algorithm": self.algorithm,
            "library": "stable_baselines3",
            "library_version": self.library_version,
            # declared
            "requested_training_transitions": self.requested,
            "training_transition_cap": self.cap,
            "evaluation_transition_cap": self.evaluation_cap,
            # resolved
            "rollout_steps_per_environment": self.n_steps,
            "environment_count": self.n_envs,
            "batch_size": self.batch_size,
            "optimization_epochs_configured": self.n_epochs,
            "minimum_training_transitions": self.minimum_training_transitions,
            # measured
            "training_transitions_observed": observed,
            "collected_rollouts": self.collected_rollouts,
            "partial_rollout": self.partial_rollout,
            "optimizer_step_calls": (None if "optimizer_step_calls" in self.unavailable
                                     else self._optimizer_calls),
            "sb3_n_updates_delta": delta,
            "sb3_n_updates_meaning": meaning,
            "model_timesteps_before": self._before.get("num_timesteps"),
            "model_timesteps_after": self._after.get("num_timesteps"),
            "timestep_counter_was_reset": self.timestep_counter_was_reset,
            "evaluation_transitions": evaluation_transitions,
        }

        # The epoch claim is made ONLY where the library was measured to mean epochs.
        body["optimization_epochs_completed"] = delta if meaning == "optimization_epochs" else None
        body["gradient_updates"] = delta if meaning == "gradient_updates" else None

        unavailable = list(dict.fromkeys(self.unavailable))
        if meaning == "UNKNOWN":
            unavailable.append("optimization_epochs_completed")
        if evaluation_transitions is None:
            unavailable.append("evaluation_transitions")
        body["counters_unavailable"] = unavailable

        body["cap_respected"] = (None if self.cap is None or observed is None
                                 else observed <= self.cap)
        if self.evaluation_cap is not None and evaluation_transitions is not None:
            body["evaluation_cap_respected"] = evaluation_transitions <= self.evaluation_cap
        return body
