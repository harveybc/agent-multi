"""WP1 adversarial suite (order 2026-08-18): the one typed activity
authority. Each ordered requirement is pinned by name."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _activity_authority as aa  # noqa: E402
from pipeline_plugins.rl_pipeline_with_validation import (  # noqa: E402
    _early_stop_composite,
)


def _summary(trades, total_return=0.01):
    return {"trades_total": trades, "total_return": total_return,
            "mean_weekly_return": total_return / 52,
            "max_drawdown_fraction": 0.01}


class TestOrderedRequirements:
    def test_zero_trades_on_either_role_is_ineligible(self):
        for kwargs in (
                dict(train_monitor_trades=0, inner_validation_trades=5),
                dict(train_monitor_trades=5, inner_validation_trades=0)):
            result = aa.evaluate_activity(**kwargs)
            assert result["eligible"] is False
            assert result["selection_score_permitted"] is False

    @pytest.mark.parametrize("bad", [None, "x", float("nan"), -3, {}])
    def test_missing_or_malformed_is_unavailable_and_ineligible(
            self, bad):
        result = aa.evaluate_activity(train_monitor_trades=bad,
                                      inner_validation_trades=4)
        assert result["eligible"] is False
        assert result["train_monitor_trades_available"] is False
        assert result["train_monitor_trades"] is None
        assert "TRADES_UNAVAILABLE_TRAIN_MONITOR" in \
            result["reason_codes"]

    def test_negative_return_is_not_an_activity_failure(self):
        # Structural: the authority signature carries NO return at all,
        # so a losing-but-active candidate is eligible by construction.
        import inspect
        params = inspect.signature(aa.evaluate_activity).parameters
        assert not any("return" in name or "profit" in name
                       or "sharpe" in name for name in params)
        result = aa.evaluate_activity(train_monitor_trades=3,
                                      inner_validation_trades=1)
        assert result["eligible"] is True

    def test_ineligible_carries_no_comparable_score(self):
        result = aa.evaluate_activity(train_monitor_trades=0,
                                      inner_validation_trades=0)
        assert not any(
            isinstance(v, float) and k not in (
                "exposure_fraction",) for k, v in result.items()), \
            "an authority result must expose no float score"
        with pytest.raises(aa.IneligibleCandidateError,
                           match="NO_SELECTION_SCORE"):
            aa.require_rankable(result)

    def test_no_numeric_sentinel_in_the_pipeline_composite(self):
        # The historical sentinel: composite = raw - 1_000_000 kept an
        # ineligible epoch RANKABLE. Now composite is None.
        composite, raw, gate, *_ = _early_stop_composite(
            _summary(0), _summary(0),
            min_trades=1, no_trade_penalty=1_000_000.0)
        assert gate is False
        assert composite is None
        assert isinstance(raw, float)  # evidence, not a selection score

    def test_eligible_epoch_keeps_its_raw_composite(self):
        composite, raw, gate, *_ = _early_stop_composite(
            _summary(5, total_return=-0.02),
            _summary(2, total_return=-0.01),
            min_trades=1, no_trade_penalty=1_000_000.0)
        assert gate is True
        assert composite == raw  # negative return, still eligible

    def test_trade_count_never_in_fitness(self):
        # The authority exposes counts and eligibility, never a score
        # combining them: no field multiplies or adds a count.
        result = aa.evaluate_activity(train_monitor_trades=7,
                                      inner_validation_trades=3)
        # every numeric field is a raw count or bounded fraction —
        # nothing combines a count with anything else
        numeric = {k: v for k, v in result.items()
                   if isinstance(v, (int, float))
                   and not isinstance(v, bool)}
        assert set(numeric) <= {"train_monitor_trades",
                                "inner_validation_trades",
                                "active_weeks", "exposure_fraction",
                                "floor"}
        assert result["train_monitor_trades"] == 7

    def test_explicit_zero_floor_refuses_typed(self):
        with pytest.raises(aa.ActivityAuthorityError,
                           match="CONTRADICTORY_ACTIVITY_FLOOR"):
            aa.resolve_floor({"selection_min_trades": 0})

    def test_absent_floor_is_strict_nonzero_not_stronger(self):
        # "Do not invent a stronger minimum count."
        assert aa.resolve_floor(None) == 1
        assert aa.resolve_floor({}) == 1
        assert aa.STRICT_NONZERO_FLOOR == 1

    def test_explicit_higher_floor_is_honored(self):
        assert aa.resolve_floor({"selection_min_trades": 12}) == 12

    def test_calibrated_floor_is_pending_not_invented(self):
        result = aa.evaluate_activity(train_monitor_trades=1,
                                      inner_validation_trades=1)
        assert result["calibrated_floor"] == "pending_wp2_evidence"
        assert result["threshold_contract_id"] == \
            aa.THRESHOLD_CONTRACT_ID


class TestConsumersShareTheAuthority:
    CONSUMERS = {
        "stopping+checkpoint":
            REPO / "pipeline_plugins/rl_pipeline_with_validation.py",
        "selection":
            REPO / "pipeline_plugins/_lexicographic_selection.py",
        "handoff+aggregation":
            REPO / "tools/p1_difficulty_lr_factorial.py",
        "promotion":
            REPO / "examples/scripts/"
                   "materialize_phase_1_promotion_candidates.py",
    }

    @pytest.mark.parametrize("name", sorted(CONSUMERS))
    def test_consumer_imports_the_authority(self, name):
        text = self.CONSUMERS[name].read_text()
        assert "_activity_authority" in text, (
            f"{name} does not consume the shared authority")

    def test_the_sentinel_subtraction_is_gone(self):
        text = self.CONSUMERS["stopping+checkpoint"].read_text()
        # the EXECUTABLE sentinel pattern, not its mention in comments
        assert "composite = raw if trade_gate_passed else raw" \
            not in text
        assert "composite = raw if trade_gate_passed else None" in text
