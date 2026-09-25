"""AUD-P1LR-20260815-235: a dead actor is typed at the checkpoint.

The P1LR phase-1 handoff carried 21 of 256 live first-layer units and the
phase-2 terminal carried 0, emitting one constant action.  Nothing in the
training path said so, so the state surfaced only after an 80-epoch grind
that ended in an anchor fallback.  These tests hold the typed diagnostic
and its mechanical refusal in place.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline_plugins import _actor_liveness as liveness


class _Linear:
    """Minimal stand-in for a torch Linear exposing weight/bias."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = _Tensor(weight)
        self.bias = _Tensor(bias)


class _Tensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array, dtype=np.float64)
        self.ndim = self._array.ndim

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _Actor:
    def __init__(self, layer: _Linear) -> None:
        self._layer = layer

    def named_modules(self):
        yield "", self
        yield "latent_pi.0", self._layer


class _Policy:
    def __init__(self, actor: _Actor) -> None:
        self.actor = actor


class _Model:
    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.policy = _Policy(_Actor(_Linear(weight, bias)))


def _observations(rows: int = 8, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(rows, dim))


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def test_all_units_dead_is_typed_and_refuses_by_default():
    dim, units = 4, 6
    model = _Model(np.zeros((units, dim)), np.full(units, -1.0))
    facts = liveness.actor_liveness_facts(
        model=model, observations=_observations(dim=dim), epoch=1,
        action_raw_std=0.0)

    assert facts["classification"] == liveness.DEAD
    assert facts["live_unit_count"] == 0
    assert facts["live_unit_fraction"] == 0.0
    assert facts["dead_unit_count"] == units
    assert "zero gradient" in facts["reason"]

    with pytest.raises(liveness.DeadActorRefusal):
        liveness.assert_actor_alive(facts)


def test_a_live_varying_actor_is_alive_and_never_refuses():
    dim, units = 4, 6
    rng = np.random.default_rng(1)
    model = _Model(rng.normal(size=(units, dim)), np.zeros(units))
    facts = liveness.actor_liveness_facts(
        model=model, observations=_observations(dim=dim), epoch=1,
        action_raw_std=0.31)

    assert facts["classification"] == liveness.ALIVE
    assert facts["live_unit_fraction"] > 0.5
    assert facts["varying_unit_count"] >= 1
    assert liveness.assert_actor_alive(facts) is facts


def test_constant_policy_is_typed_but_does_not_refuse_by_default():
    """A constant action can be transient saturation in a healthy net."""
    dim, units = 4, 6
    rng = np.random.default_rng(2)
    model = _Model(rng.normal(size=(units, dim)), np.zeros(units))
    facts = liveness.actor_liveness_facts(
        model=model, observations=_observations(dim=dim), epoch=1,
        action_raw_std=0.0)

    assert facts["classification"] == liveness.CONSTANT
    assert facts["constant_policy"] is True
    assert facts["constant_policy_evidence"] == (
        "split_action_raw_std_is_exactly_zero")

    assert liveness.assert_actor_alive(facts) is facts
    with pytest.raises(liveness.DeadActorRefusal):
        liveness.assert_actor_alive(facts, refuse_constant=True)


def test_a_dead_first_layer_outranks_a_constant_action():
    dim, units = 4, 6
    model = _Model(np.zeros((units, dim)), np.full(units, -1.0))
    facts = liveness.actor_liveness_facts(
        model=model, observations=_observations(dim=dim), action_raw_std=0.0)
    assert facts["classification"] == liveness.DEAD


def test_degraded_fires_at_the_observed_p1lr_handoff_ratio():
    """21 of 256 live units — the artifact selection actually promoted."""
    dim, units = 4, 256
    weight = np.zeros((units, dim))
    bias = np.full(units, -1.0)
    weight[:21] = 1.0
    bias[:21] = 1.0
    facts = liveness.actor_liveness_facts(
        model=_Model(weight, bias), observations=np.ones((8, dim)),
        action_raw_std=0.02, epoch=1)
    assert facts["classification"] == liveness.DEGRADED
    assert facts["live_unit_count"] == 21
    assert facts["live_unit_fraction"] == pytest.approx(21 / 256)


def test_missing_observations_are_unmeasured_not_healthy():
    facts = liveness.actor_liveness_facts(model=None, observations=None)
    assert facts["classification"] == liveness.UNMEASURED
    assert facts["measured"] is False
    assert liveness.assert_actor_alive(facts) is facts


def test_an_unreadable_actor_is_unmeasured():
    facts = liveness.actor_liveness_facts(
        model=object(), observations=_observations())
    assert facts["classification"] == liveness.UNMEASURED


def test_a_mismatched_observation_batch_is_unmeasured_not_a_crash():
    model = _Model(np.ones((3, 4)), np.zeros(3))
    facts = liveness.actor_liveness_facts(
        model=model, observations=np.ones((5, 9)))
    assert facts["classification"] == liveness.UNMEASURED
    assert "does not belong to this actor" in facts["reason"]


# --------------------------------------------------------------------------
# constant-action evidence derives its tolerance from dtype precision only
# --------------------------------------------------------------------------

def test_exact_constant_actions_are_detected_without_an_invented_floor():
    facts = liveness.constant_action_facts(
        np.full(64, -0.001271, dtype=np.float32))
    assert facts["exact_constant"] is True
    assert facts["unique_action_count_exact"] == 1
    assert facts["near_constant_tolerance"] == pytest.approx(
        float(np.finfo(np.float32).eps))


def test_varying_actions_are_not_constant():
    facts = liveness.constant_action_facts(
        np.linspace(-0.5, 0.5, 64, dtype=np.float32))
    assert facts["exact_constant"] is False
    assert facts["near_constant"] is False


# --------------------------------------------------------------------------
# the probe batch
# --------------------------------------------------------------------------

def test_sampler_is_bounded_and_covers_the_whole_split():
    sampler = liveness.StridedObservationSampler(16)
    for index in range(2190):
        sampler.offer(np.full(3, float(index), dtype=np.float32))
    batch = sampler.batch()
    assert batch is not None
    assert batch.shape[0] <= 16
    # coverage: the sample must not be only the opening rows
    assert float(batch[0][0]) < 100.0
    assert float(batch[-1][0]) > 2100.0


def test_combined_probe_requires_both_compatible_sides():
    fit = np.zeros((3, 4), dtype=np.float32)
    validation = np.ones((2, 4), dtype=np.float32)
    combined = liveness.combine_observation_batches(fit, validation)
    assert combined is not None
    assert combined.shape == (5, 4)
    assert liveness.combine_observation_batches(fit, None) is None
    assert liveness.combine_observation_batches(
        fit, np.ones((2, 5), dtype=np.float32)) is None


def test_sampler_disabled_at_zero_capacity():
    sampler = liveness.StridedObservationSampler(0)
    sampler.offer(np.zeros(3, dtype=np.float32))
    assert sampler.enabled is False
    assert sampler.batch() is None


def test_sampler_refuses_to_guess_a_dict_observation_layout():
    sampler = liveness.StridedObservationSampler(8)
    sampler.offer({"features": np.zeros(3)})
    assert sampler.batch() is None
