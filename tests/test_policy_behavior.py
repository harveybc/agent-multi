"""WP5 acceptance tests 1-3 for finding AUD-F1-20260817-277.

The ordered shapes, verbatim:

1. a constant ``+0.0008`` at threshold zero is classified
   constant-directional, not state-responsive;
2. the same trace at threshold ``0.1`` is constant-HOLD;
3. varying sub-threshold actions are distinguished from exact/near
   constants.

The reproduced constants come from the live identity ``f9379f596e80fda4``
(Musashi's independent reproduction, 2026-08-17 15:48):
seed404 easy monitor ``+0.00083673`` produced 114 trades and −3.05%;
seed303 easy monitor ``−0.00037974`` produced 85 trades and +4.38%.
Both must classify as constant-directional — the counterexample that
kills "trades imply learning".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _policy_behavior as pb  # noqa: E402

SEED404_EASY_CONSTANT = 0.00083673
SEED303_EASY_CONSTANT = -0.00037974
SEED101_NORMAL_CONSTANT = -0.00100470


def constant(value: float, n: int = 2190) -> list[float]:
    return [value] * n


class TestOrderedShape1And2:
    """The same series is a different behavior under a different
    adapter threshold — and neither reading is state-responsive."""

    def test_constant_at_threshold_zero_is_directional_not_responsive(
            self):
        result = pb.classify_policy_behavior(
            constant(0.0008), threshold=0.0)
        assert result["classification"] == \
            pb.CONSTANT_DIRECTIONAL_EXPOSURE
        assert result["promotable_as_learned_activity"] is False
        assert result["threshold_counterfactuals"]["0"]["long"] == 2190
        assert result["threshold_counterfactuals"]["0"]["hold"] == 0

    def test_same_trace_at_threshold_0_1_is_constant_hold(self):
        result = pb.classify_policy_behavior(
            constant(0.0008), threshold=0.1)
        assert result["classification"] == pb.CONSTANT_HOLD
        assert result["promotable_as_learned_activity"] is False
        assert result["threshold_crossings"] == 0
        assert result["threshold_counterfactuals"]["0.1"]["hold"] == 2190

    @pytest.mark.parametrize("value,expected_side", [
        (SEED404_EASY_CONSTANT, "long"),
        (SEED303_EASY_CONSTANT, "short"),
        (SEED101_NORMAL_CONSTANT, "short"),
    ])
    def test_reproduced_live_constants_are_never_learned_activity(
            self, value, expected_side):
        at_zero = pb.classify_policy_behavior(
            constant(value), threshold=0.0)
        assert at_zero["classification"] == \
            pb.CONSTANT_DIRECTIONAL_EXPOSURE
        assert at_zero["promotable_as_learned_activity"] is False
        assert at_zero["threshold_counterfactuals"]["0"][
            expected_side] == 2190
        at_normal = pb.classify_policy_behavior(
            constant(value), threshold=0.1)
        assert at_normal["classification"] == pb.CONSTANT_HOLD
        assert at_normal["promotable_as_learned_activity"] is False

    def test_directional_constant_reason_names_the_adapter(self):
        result = pb.classify_policy_behavior(
            constant(SEED404_EASY_CONSTANT), threshold=0.0)
        reason = " ".join(result["reasons"])
        assert "adapter" in reason and "market path" in reason
        assert "NOT from state-conditioned learning" in reason


class TestOrderedShape3:
    """Varying sub-threshold actions must be distinguished from exact
    and near constants — exact float equality alone is insufficient."""

    def test_varying_below_threshold_is_responsive_below_threshold(self):
        actions = [0.001 + 0.02 * ((i % 7) - 3) for i in range(2190)]
        assert max(abs(a) for a in actions) < 0.1
        result = pb.classify_policy_behavior(actions, threshold=0.1)
        assert result["classification"] == \
            pb.STATE_RESPONSIVE_BELOW_THRESHOLD
        assert result["promotable_as_learned_activity"] is False
        assert result["threshold_crossings"] == 0
        assert result["deterministic"]["unique_count"] > 1

    def test_near_constant_jitter_is_constant_not_responsive(self):
        # Float-distinct but behaviorally constant: 1e-12 jitter must
        # NOT read as a responsive policy, which is exactly why exact
        # equality is insufficient.
        actions = [SEED404_EASY_CONSTANT + 1e-12 * (i % 3)
                   for i in range(2190)]
        assert len(set(actions)) > 1
        result = pb.classify_policy_behavior(actions, threshold=0.0)
        assert result["classification"] == \
            pb.CONSTANT_DIRECTIONAL_EXPOSURE

    def test_tolerance_is_declared_in_the_result(self):
        result = pb.classify_policy_behavior(
            constant(0.0008), threshold=0.1, tolerance=1e-9)
        assert result["constancy_tolerance"] == 1e-9

    def test_variation_just_above_tolerance_is_responsive(self):
        tol = 1e-6
        actions = [0.05, 0.05 + 10 * tol] * 1095
        result = pb.classify_policy_behavior(
            actions, threshold=0.1, tolerance=tol)
        assert result["classification"] == \
            pb.STATE_RESPONSIVE_BELOW_THRESHOLD

    def test_varying_numbers_with_constant_mapped_decision_is_constant(
            self):
        # The self-caught defect: at threshold 0 EVERY non-zero action
        # "crosses", so crossings alone made ACTIVE trivially reachable.
        # A policy whose numbers vary while its mapped decision never
        # changes is behaviorally constant.
        actions = [0.30 + 0.01 * (i % 5) for i in range(2190)]
        result = pb.classify_policy_behavior(actions, threshold=0.0)
        assert result["classification"] == \
            pb.CONSTANT_DIRECTIONAL_EXPOSURE
        assert result["promotable_as_learned_activity"] is False
        assert "MAPPED decision never changes" in " ".join(
            result["reasons"])

    def test_active_requires_the_mapped_decision_to_change(self):
        actions = [0.5 if (i // 100) % 2 else -0.5 for i in range(2190)]
        result = pb.classify_policy_behavior(actions, threshold=0.0)
        assert result["classification"] == pb.STATE_RESPONSIVE_ACTIVE
        assert "MAPPED decision changes" in " ".join(result["reasons"])

    def test_responsive_and_crossing_is_the_only_promotable_class(self):
        actions = [0.5 if i % 2 else -0.5 for i in range(2190)]
        result = pb.classify_policy_behavior(actions, threshold=0.1)
        assert result["classification"] == pb.STATE_RESPONSIVE_ACTIVE
        assert result["promotable_as_learned_activity"] is True
        promotable = {
            c for c in pb.CLASSIFICATIONS
            if c in pb.PROMOTABLE_AS_LEARNED_ACTIVITY}
        assert promotable == {pb.STATE_RESPONSIVE_ACTIVE}


class TestStochasticAndUnavailable:
    def test_stochastic_only_activity_is_named(self):
        result = pb.classify_policy_behavior(
            constant(0.0008), threshold=0.1,
            stochastic_actions=[0.4, -0.6, 0.05, 0.9])
        assert result["classification"] == pb.STOCHASTIC_ONLY_ACTIVITY
        assert result["promotable_as_learned_activity"] is False
        # 0.4, -0.6 and 0.9 cross; 0.05 does not
        assert result["stochastic"]["threshold_crossings"] == 3

    @pytest.mark.parametrize("actions", [None, [], [float("nan")],
                                         ["x", None]])
    def test_unmeasurable_is_typed_not_raised(self, actions):
        result = pb.classify_policy_behavior(actions, threshold=0.1)
        assert result["classification"] == pb.UNAVAILABLE
        assert result["promotable_as_learned_activity"] is False

    def test_negative_tolerance_is_a_request_error(self):
        with pytest.raises(pb.PolicyBehaviorError):
            pb.classify_policy_behavior([0.1], threshold=0.1,
                                        tolerance=-1.0)


class TestAdapterFidelity:
    """The classifier's mapping must equal the environment's."""

    @pytest.mark.parametrize("value,threshold,expected", [
        (0.0, 0.0, pb.HOLD),
        (1e-12, 0.0, pb.LONG),
        (-1e-12, 0.0, pb.SHORT),
        (0.1, 0.1, pb.LONG),
        (0.0999, 0.1, pb.HOLD),
        (-0.1, 0.1, pb.SHORT),
    ])
    def test_map_action_matches_env_coerce_action(self, value,
                                                  threshold, expected):
        assert pb.map_action(value, threshold) == expected

    def test_counterfactuals_cover_the_ordered_thresholds(self):
        result = pb.classify_policy_behavior(
            constant(0.02), threshold=0.1)
        assert set(result["threshold_counterfactuals"]) == {
            "0", "0.001", "0.01", "0.05", "0.1"}
        # 0.02 is directional below 0.05 and HOLD at/above it
        cf = result["threshold_counterfactuals"]
        assert cf["0.01"]["long"] == 2190
        assert cf["0.05"]["hold"] == 2190
