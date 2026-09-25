"""Typed first-layer liveness / constant-policy diagnostic for the actor.

AUD-P1LR-20260815-235.  Replaying real ``inner_validation`` observations
against the sealed P1LR artifacts showed the SAC actor's FIRST hidden
layer dead:

  * phase-1 handoff (the artifact selection actually picked): 21 of 256
    live first-layer units, mean pre-activation -63.8;
  * phase-2 terminal: 0 of 256 live units, one constant action
    ``-0.001271`` reproducible from weights alone as
    ``tanh(W_mu @ ReLU(b_latent2) + b_mu)``.

A ReLU has exactly zero gradient on observations for which it never fires.
The sealed P1LR terminal was independently replayed over the full approved
fit and inner-validation intervals and did not fire on either, which explains
why more epochs and a different learning rate could not recover that artifact.
The online detector below is deliberately described as a bounded probe, not
as a proof over observations it did not inspect. The driver of the observed
collapse is an observation-contract defect: ``include_price_window=true``
injects 64 UNNORMALIZED
dimensions (raw ETH prices, mean |value| ~1742, plus their raw diffs)
into an otherwise rolling-z-scored, +-10-clipped 2724-dim observation.
Those 64 dimensions dominate layer 1: the sign of a unit's
pre-activation is fixed by the (always positive) price level, so roughly
half the units are dead for EVERY observation from initialisation
onward, and the surviving half saturate ``tanh`` into a constant action.

The cost of NOT typing this was 80 silent epochs per cell across three
experiments that measured nothing.  This module makes the state a FACT
recorded at every checkpoint instead of a post-mortem: it is pure
measurement (no training-path side effects), it never invents a
behaviour floor for the constant-policy call (the tolerance is derived
from dtype precision at the observed action scale, the same derivation
``rl_pipeline_with_solvency_curriculum._dtype_constant_classification``
uses), and the refusal is OPT-IN so an operator can still watch a dead
run on purpose.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

SCHEMA = "agent_multi.actor_liveness.v1"

#: Typed classifications.  ``ACTOR_UNMEASURED`` is a first-class outcome:
#: a probe that could not read the actor or had no observations must say
#: so rather than silently reporting a healthy default.
ALIVE = "ACTOR_ALIVE"
DEGRADED = "ACTOR_FIRST_LAYER_DEGRADED"
DEAD = "ACTOR_FIRST_LAYER_DEAD"
CONSTANT = "ACTOR_CONSTANT_POLICY"
UNMEASURED = "ACTOR_UNMEASURED"

CLASSIFICATIONS = (ALIVE, DEGRADED, DEAD, CONSTANT, UNMEASURED)

#: ``ACTOR_FIRST_LAYER_DEAD`` refuses by default after the declared probe
#: sees no activation. This is a conservative campaign policy, not a claim
#: that a bounded sample exhausts every observation the learner may see.
DEFAULT_REFUSABLE = (DEAD,)

#: ``ACTOR_CONSTANT_POLICY`` can be transient in a healthy net early in
#: training (a saturated tanh that later unsaturates), so it is recorded
#: on every checkpoint but only refuses when a campaign asks it to.
OPTIONAL_REFUSABLE = (CONSTANT,)

REFUSABLE = DEFAULT_REFUSABLE + OPTIONAL_REFUSABLE

#: Below this share of live first-layer units the actor is typed
#: DEGRADED.  The default is deliberately generous: the observed P1LR
#: phase-1 handoff sat at 21/256 = 0.082, so a 0.10 floor types that
#: artifact at the checkpoint that produced it, not at epoch 80.
DEFAULT_MIN_LIVE_UNIT_FRACTION = 0.10

#: Default size of the strided observation sample the probe measures on.
DEFAULT_PROBE_OBSERVATIONS = 256

REFUSAL_OUTCOME = "REFUSED_DEAD_ACTOR"


class DeadActorRefusal(RuntimeError):
    """Opt-in refusal: the actor cannot learn from its observations."""

    def __init__(self, facts: Mapping[str, Any]) -> None:
        self.facts = dict(facts)
        self.classification = str(facts.get("classification"))
        super().__init__(
            f"{REFUSAL_OUTCOME}: {self.classification} at "
            f"epoch {facts.get('epoch')} on split "
            f"{facts.get('split')!r} — {facts.get('reason')}")


# ---------------------------------------------------------------------------
# capturing a REAL observation batch without a second rollout
# ---------------------------------------------------------------------------

class StridedObservationSampler:
    """Uniform, bounded sample of the observations a rollout acted on.

    The split length is not known in advance, so the sampler keeps every
    ``stride``-th observation and, whenever the buffer would exceed the
    cap, discards every second entry and doubles the stride.  The result
    covers the whole split (never just its opening rows), never exceeds
    ``2 * capacity`` rows of memory, and is fully deterministic.

    ``capacity <= 0`` disables capture entirely, so the probe can be
    turned off without branching at the call site.
    """

    __slots__ = ("capacity", "stride", "_index", "_rows", "_unsupported")

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity or 0)
        self.stride = 1
        self._index = 0
        self._rows: list = []
        self._unsupported = False

    @property
    def enabled(self) -> bool:
        return self.capacity > 0 and not self._unsupported

    def offer(self, observation: Any) -> None:
        if not self.enabled:
            return
        if not isinstance(observation, np.ndarray):
            # Dict observations never reach a flat first layer; say so
            # once rather than guessing a flattening order.
            self._unsupported = True
            self._rows = []
            return
        index = self._index
        self._index += 1
        if index % self.stride:
            return
        self._rows.append(np.asarray(observation, dtype=np.float32).ravel())
        if len(self._rows) > 2 * self.capacity:
            self._rows = self._rows[::2]
            self.stride *= 2

    def batch(self) -> np.ndarray | None:
        if not self._rows:
            return None
        rows = self._rows
        if len(rows) > self.capacity:
            # Preserve the whole observed interval. Taking only the first
            # ``capacity`` rows silently discarded the tail whenever the
            # buffer ended between compactions.
            positions = np.linspace(
                0, len(rows) - 1, num=self.capacity, dtype=np.int64)
            rows = [rows[int(index)] for index in positions]
        widths = {row.shape[0] for row in rows}
        if len(widths) != 1:
            return None
        return np.stack(rows)


def combine_observation_batches(*batches: Any) -> np.ndarray | None:
    """Combine compatible fit/validation probes without guessing a layout."""
    arrays = []
    for batch in batches:
        if batch is None:
            return None
        array = np.asarray(batch, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] == 0:
            return None
        arrays.append(array)
    if not arrays or len({array.shape[1] for array in arrays}) != 1:
        return None
    return np.concatenate(arrays, axis=0)


# ---------------------------------------------------------------------------
# reading the actor's first hidden layer
# ---------------------------------------------------------------------------

def first_layer_parameters(model: Any) -> tuple:
    """``(weight, bias, module_path)`` of the actor's FIRST linear layer.

    Generic on purpose: the first 2-D ``weight`` reached in module
    registration order is the layer the observation hits.  For a
    Stable-Baselines3 SAC actor that is ``actor.latent_pi[0]``; the
    parameterless ``FlattenExtractor`` in front of it is skipped for
    free.  Raises ``LookupError`` when no such layer exists, so a caller
    reports ``ACTOR_UNMEASURED`` instead of guessing.
    """
    actor = getattr(getattr(model, "policy", None), "actor", None)
    if actor is None:
        actor = getattr(model, "actor", None)
    if actor is None:
        actor = getattr(model, "policy", None)
    if actor is None:
        raise LookupError("model exposes no actor/policy to measure")
    modules = getattr(actor, "named_modules", None)
    if not callable(modules):
        raise LookupError("actor exposes no torch module tree")
    for name, module in modules():
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            continue
        bias = getattr(module, "bias", None)
        weight_array = np.asarray(weight.detach().cpu().numpy(),
                                  dtype=np.float64)
        if bias is None:
            bias_array = np.zeros(weight_array.shape[0], dtype=np.float64)
        else:
            bias_array = np.asarray(bias.detach().cpu().numpy(),
                                    dtype=np.float64)
        return weight_array, bias_array, str(name or "<root>")
    raise LookupError("actor exposes no 2-D linear weight")


def first_layer_liveness(weight: np.ndarray, bias: np.ndarray,
                         observations: np.ndarray) -> Dict[str, Any]:
    """Measure the first layer on a REAL observation batch.

    A unit is LIVE when its ReLU fires on at least one observation in the
    batch; it is sample-dead when it fires on none. It has exactly zero
    gradient for those observations, but a bounded sample alone cannot prove
    that no unseen training observation could activate it. ``varying`` is the
    stricter, more informative count: units that are live AND whose
    output actually changes with the observation.  A unit that is live on
    every row with a constant value carries no information either.
    """
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations.reshape(1, -1)
    if observations.ndim != 2:
        raise ValueError(
            f"observations must be 2-D (n, dim); got shape "
            f"{observations.shape}")
    if observations.shape[1] != weight.shape[1]:
        raise ValueError(
            f"observation dim {observations.shape[1]} != first-layer input "
            f"dim {weight.shape[1]} — the probe batch does not belong to "
            "this actor")

    pre = observations @ weight.T + bias
    fired = pre > 0.0
    live_mask = fired.any(axis=0)
    always_mask = fired.all(axis=0)
    activation = np.maximum(pre, 0.0)
    spread = activation.max(axis=0) - activation.min(axis=0)
    varying_mask = live_mask & (spread > 0.0)

    units = int(weight.shape[0])
    live = int(live_mask.sum())
    varying = int(varying_mask.sum())
    return {
        "observation_count": int(observations.shape[0]),
        "observation_dim": int(observations.shape[1]),
        "first_layer_units": units,
        "live_unit_count": live,
        "live_unit_fraction": float(live / units) if units else 0.0,
        "dead_unit_count": int(units - live),
        "dead_unit_fraction": (float((units - live) / units)
                               if units else 0.0),
        "always_live_unit_count": int(always_mask.sum()),
        "varying_unit_count": varying,
        "varying_unit_fraction": float(varying / units) if units else 0.0,
        "preactivation_mean": float(pre.mean()),
        "preactivation_min": float(pre.min()),
        "preactivation_max": float(pre.max()),
        "preactivation_unit_mean_median": float(
            np.median(pre.mean(axis=0))),
        "observation_abs_max": float(np.abs(observations).max()),
        "live_rule": "a unit is live iff its ReLU fires on >=1 observation",
    }


# ---------------------------------------------------------------------------
# constant-policy evidence
# ---------------------------------------------------------------------------

def constant_action_facts(actions: Sequence[float] | np.ndarray | None,
                          ) -> Dict[str, Any] | None:
    """Exact/near-constant facts from dtype precision only.

    Same derivation as the phase-1 handoff evidence
    (``finfo(dtype).eps * max(1.0, max|a|)``): no invented activity
    floor, and ``exact_constant`` means exactly one bitwise-unique value.
    """
    if actions is None:
        return None
    values = np.asarray(actions)
    if values.size == 0:
        return None
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    eps = float(np.finfo(values.dtype).eps)
    observed_max_abs = float(np.max(np.abs(values)))
    tolerance = eps * max(1.0, observed_max_abs)
    peak_to_peak = float(np.max(values) - np.min(values))
    unique_exact = int(np.unique(values).size)
    return {
        "dtype": str(values.dtype),
        "dtype_eps": eps,
        "observed_max_abs": observed_max_abs,
        "near_constant_tolerance": tolerance,
        "tolerance_derivation":
            "finfo(dtype).eps * max(1.0, observed_max_abs)",
        "unique_action_count_exact": unique_exact,
        "peak_to_peak": peak_to_peak,
        "action_std": float(np.std(values.astype(np.float64))),
        "exact_constant": bool(unique_exact == 1),
        "near_constant": bool(peak_to_peak <= tolerance),
    }


def _constant_policy(action_raw_std: Any,
                     action_facts: Mapping[str, Any] | None) -> tuple:
    """``(is_constant, evidence_source)`` — exact zero spread only."""
    if action_facts is not None:
        if bool(action_facts.get("exact_constant")):
            return True, "probe_actions_exact_constant"
        if bool(action_facts.get("near_constant")):
            return True, "probe_actions_near_constant_at_dtype_precision"
        return False, "probe_actions_vary"
    if action_raw_std is None:
        return False, "no_action_evidence"
    try:
        value = float(action_raw_std)
    except (TypeError, ValueError):
        return False, "no_action_evidence"
    if value != value:                                    # NaN
        return False, "no_action_evidence"
    if value == 0.0:
        return True, "split_action_raw_std_is_exactly_zero"
    return False, "split_action_raw_std_is_nonzero"


# ---------------------------------------------------------------------------
# the typed record
# ---------------------------------------------------------------------------

def unmeasured_facts(*, reason: str, epoch: Any = None,
                     split: str = "", phase: str = "") -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "epoch": epoch,
        "phase": str(phase),
        "split": str(split),
        "measured": False,
        "classification": UNMEASURED,
        "reason": str(reason),
        "live_unit_fraction": None,
        "live_unit_count": None,
        "first_layer_units": None,
        "dead_unit_fraction": None,
        "varying_unit_fraction": None,
        "preactivation_mean": None,
        "action_raw_std": None,
        "constant_policy": None,
    }


def actor_liveness_facts(
    *,
    model: Any = None,
    observations: Any = None,
    action_raw_std: Any = None,
    actions: Any = None,
    epoch: Any = None,
    split: str = "",
    phase: str = "",
    min_live_unit_fraction: float = DEFAULT_MIN_LIVE_UNIT_FRACTION,
    weights: tuple | None = None,
) -> Dict[str, Any]:
    """One typed liveness record for ONE checkpoint.

    Never raises for a measurement problem: an unreadable actor or an
    empty probe batch is typed ``ACTOR_UNMEASURED`` with a reason, so the
    training path can record the fact without a new failure mode.
    """
    if observations is None or np.asarray(observations).size == 0:
        return unmeasured_facts(
            reason="no real observation batch was captured for this "
                   "checkpoint",
            epoch=epoch, split=split, phase=phase)
    try:
        if weights is not None:
            weight, bias, module_path = weights
        else:
            weight, bias, module_path = first_layer_parameters(model)
        layer = first_layer_liveness(weight, bias, observations)
    except (LookupError, ValueError) as exc:
        return unmeasured_facts(
            reason=f"{type(exc).__name__}: {exc}",
            epoch=epoch, split=split, phase=phase)

    action_facts = constant_action_facts(actions)
    constant, constant_source = _constant_policy(action_raw_std,
                                                 action_facts)
    floor = float(min_live_unit_fraction)
    live_fraction = float(layer["live_unit_fraction"])

    if live_fraction <= 0.0:
        classification = DEAD
        reason = (
            f"0 of {layer['first_layer_units']} first-layer units fire on "
            f"any of {layer['observation_count']} real observations "
            f"(mean pre-activation {layer['preactivation_mean']:.4g}); "
            "the first layer has zero gradient on every probed observation "
            "and the campaign's declared fail-closed policy refuses further "
            "updates")
    elif constant:
        classification = CONSTANT
        reason = (
            "the policy is a constant function of the observation "
            f"({constant_source}); {layer['live_unit_count']} of "
            f"{layer['first_layer_units']} first-layer units are live but "
            "the emitted action does not depend on the input")
    elif live_fraction < floor:
        classification = DEGRADED
        reason = (
            f"only {layer['live_unit_count']} of "
            f"{layer['first_layer_units']} first-layer units "
            f"({live_fraction:.4f}) fire on any real observation, below "
            f"the declared floor {floor:.4f}; those units have zero "
            "gradient on the declared probe")
    else:
        classification = ALIVE
        reason = (
            f"{layer['live_unit_count']} of {layer['first_layer_units']} "
            f"first-layer units live ({live_fraction:.4f}), "
            f"{layer['varying_unit_count']} of them vary with the "
            "observation")

    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "epoch": epoch,
        "phase": str(phase),
        "split": str(split),
        "measured": True,
        "first_layer_module": module_path,
        "min_live_unit_fraction": floor,
        "action_raw_std": (None if action_raw_std is None
                           else float(action_raw_std)),
        "constant_policy": bool(constant),
        "constant_policy_evidence": constant_source,
        "action_constant_facts": action_facts,
        "classification": classification,
        "reason": reason,
    }
    record.update(layer)
    return record


def assert_actor_alive(facts: Mapping[str, Any], *,
                       refuse_dead: bool = True,
                       refuse_constant: bool = False) -> Mapping[str, Any]:
    """Mechanical refusal — no flag to remember, no approval to obtain.

    ``refuse_dead`` defaults to TRUE. A first layer with zero live units has
    zero gradient on the declared probe; refusal is a conservative campaign
    policy and the record retains the exact probe size and split.
    ``refuse_constant`` defaults to false because a constant action can
    be a transient saturation in an otherwise healthy actor.
    """
    refusable = set()
    if refuse_dead:
        refusable.update(DEFAULT_REFUSABLE)
    if refuse_constant:
        refusable.update(OPTIONAL_REFUSABLE)
    if str(facts.get("classification")) in refusable:
        raise DeadActorRefusal(facts)
    return facts


def liveness_summary_line(facts: Mapping[str, Any]) -> str:
    """One-line operator summary for the per-epoch log."""
    if not facts.get("measured"):
        return f"actor {facts.get('classification')}: {facts.get('reason')}"
    return (
        f"actor {facts.get('classification')} "
        f"live={facts.get('live_unit_count')}/"
        f"{facts.get('first_layer_units')} "
        f"({float(facts.get('live_unit_fraction') or 0.0):.3f}) "
        f"varying={facts.get('varying_unit_count')} "
        f"pre_mean={float(facts.get('preactivation_mean') or 0.0):+.4g} "
        f"action_std={facts.get('action_raw_std')} "
        f"constant={facts.get('constant_policy')}")
