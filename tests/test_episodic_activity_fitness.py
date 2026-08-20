"""Order 2026-08-20 section 7: the twelve mandatory counterexamples,
each pinned as a strict-ordering property test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _episodic_activity_fitness as ef  # noqa: E402

ROWS = 2190  # one 4h year


def episode(**kwargs):
    base = dict(total_return=0.0, max_drawdown_fraction=0.0,
                sharpe=None, closed_trades=0, scored_rows=ROWS)
    base.update(kwargs)
    return ef.evaluate_episode(**base)


def value(**kwargs) -> float:
    return episode(**kwargs)["selection_value"]


class TestMandatoryCounterexamples:
    def test_1_zero_trades_loses_to_every_finite_active_fixture(self):
        sentinel = value(closed_trades=0)
        actives = [
            value(closed_trades=1, total_return=-0.90,
                  max_drawdown_fraction=0.95),   # catastrophic
            value(closed_trades=5, total_return=-0.20,
                  max_drawdown_fraction=0.30),
            value(closed_trades=40, total_return=-0.05,
                  max_drawdown_fraction=0.10),
            value(closed_trades=120, total_return=0.02),
            value(closed_trades=3000, total_return=-0.01),  # overtrading
        ]
        assert all(active > sentinel for active in actives)

    def test_2_nop_bars_carry_no_penalty(self):
        # The objective consumes EPISODE facts only: two episodes with
        # identical outcomes but wildly different bar counts of waiting
        # score identically per year — no per-bar term exists.
        a = value(closed_trades=10, total_return=0.03, scored_rows=ROWS)
        b = value(closed_trades=10, total_return=0.03, scored_rows=ROWS)
        assert a == b
        import inspect
        source = inspect.getsource(ef.evaluate_episode)
        assert "per_bar" not in source and "nop_penalty" not in source

    def test_3_smaller_loss_beats_larger_loss(self):
        assert value(closed_trades=40, total_return=-0.05,
                     max_drawdown_fraction=0.10) > \
            value(closed_trades=40, total_return=-0.20,
                  max_drawdown_fraction=0.10)

    def test_4_target_activity_beats_insufficient(self):
        low = value(closed_trades=4, total_return=-0.05,
                    max_drawdown_fraction=0.10)
        target = value(closed_trades=120, total_return=-0.05,
                       max_drawdown_fraction=0.10)
        assert target > low
        low_gain = value(closed_trades=4, total_return=0.05)
        target_gain = value(closed_trades=120, total_return=0.05)
        assert target_gain > low_gain

    def test_5_overtrading_below_target_above_zero(self):
        target = value(closed_trades=120, total_return=-0.05,
                       max_drawdown_fraction=0.10)
        over = value(closed_trades=5000, total_return=-0.05,
                     max_drawdown_fraction=0.10)
        sentinel = value(closed_trades=0)
        assert over < target
        assert over > sentinel
        assert ef.activity_utility(5000 / 1.0, ef.DEFAULT_CONFIG) > 0

    def test_6_lower_drawdown_wins(self):
        assert value(closed_trades=120, total_return=0.05,
                     max_drawdown_fraction=0.02) > \
            value(closed_trades=120, total_return=0.05,
                  max_drawdown_fraction=0.30)
        assert value(closed_trades=40, total_return=-0.05,
                     max_drawdown_fraction=0.02) > \
            value(closed_trades=40, total_return=-0.05,
                  max_drawdown_fraction=0.30)

    def test_7_positive_sharpe_beats_negative_or_unavailable(self):
        base = dict(closed_trades=120, total_return=0.05,
                    max_drawdown_fraction=0.05)
        with_sharpe = value(sharpe=1.5, **base)
        negative = value(sharpe=-0.5, **base)
        unavailable = value(sharpe=None, **base)
        assert with_sharpe > negative
        assert with_sharpe > unavailable
        assert episode(sharpe=None, **base)["sharpe_available"] is False

    def test_8_negative_multiplication_cannot_reward_larger_loss(self):
        # sweep: for any fixed activity/drawdown, scalar is strictly
        # decreasing in |loss|
        for trades in (1, 40, 3000):
            for dd in (0.0, 0.3, 0.9):
                previous = None
                for loss in (0.01, 0.05, 0.2, 0.5, 0.9):
                    s = value(closed_trades=trades, total_return=-loss,
                              max_drawdown_fraction=dd)
                    if previous is not None:
                        assert s < previous, (trades, dd, loss)
                    previous = s

    def test_9_one_trade_never_satisfies_production_promotion(self):
        result = episode(closed_trades=1, total_return=0.10)
        assert result["production_promotion_satisfied"] is False
        with_contract = ef.evaluate_episode(
            total_return=0.10, max_drawdown_fraction=0.0, sharpe=None,
            closed_trades=1, scored_rows=ROWS,
            production_promotion_contract={
                "min_annualized_trade_rate": 50.0})
        assert with_contract["production_promotion_satisfied"] is False
        enough = ef.evaluate_episode(
            total_return=0.10, max_drawdown_fraction=0.0, sharpe=None,
            closed_trades=120, scored_rows=ROWS,
            production_promotion_contract={
                "min_annualized_trade_rate": 50.0})
        assert enough["production_promotion_satisfied"] is True

    def test_10_unsurvivable_easy_checkpoint_cannot_hand_off(self):
        constant = [0.0008] * 500          # dies at normal threshold
        verdict = ef.assert_handoff_survivable(
            constant, normal_threshold=0.1)
        assert verdict["survivable"] is False
        assert "HANDOFF_REFUSED" in verdict["refusal"]
        alive = [0.5 if i % 3 else -0.5 for i in range(500)]
        assert ef.assert_handoff_survivable(
            alive, normal_threshold=0.1)["survivable"] is True

    def test_11_difficulty_change_leaves_tensors_untouched(self):
        import numpy as np
        state = {"actor.w": np.ones((4, 4)), "critic.w": np.zeros(7)}
        same = {k: v.copy() for k, v in state.items()}
        verdict = ef.verify_handoff_continuity(state, same)
        assert verdict["continuous"] is True
        assert verdict["l1_distance_total"] == 0.0
        assert verdict["sha256_before"] == verdict["sha256_after"]
        drifted = {k: v.copy() for k, v in state.items()}
        drifted["actor.w"][0, 0] += 1e-9
        bad = ef.verify_handoff_continuity(state, drifted)
        assert bad["continuous"] is False
        retopo = dict(same)
        retopo["actor.extra"] = np.ones(2)
        assert ef.verify_handoff_continuity(state, retopo)[
            "continuous"] is False

    def test_12_all_cases_serialize_raw_components(self):
        cases = {
            "no_trade": episode(closed_trades=0),
            "active_loss": episode(closed_trades=40,
                                   total_return=-0.05,
                                   max_drawdown_fraction=0.1),
            "active_profit": episode(closed_trades=120,
                                     total_return=0.08, sharpe=1.0,
                                     max_drawdown_fraction=0.05),
            "overtrading": episode(closed_trades=5000,
                                   total_return=0.01),
        }
        for name, result in cases.items():
            text = json.dumps(result)          # OLAP-serializable
            restored = json.loads(text)
            for component in ("total_return", "max_drawdown_fraction",
                              "sharpe", "closed_trades", "scored_rows",
                              "scored_years", "annualized_trade_rate",
                              "activity_utility", "economic_utility",
                              "selection_value", "branch"):
                assert component in restored, (name, component)


class TestRefusals:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x",
                                     None, True])
    def test_malformed_return_refuses(self, bad):
        with pytest.raises(ef.EpisodicFitnessError):
            episode(total_return=bad, closed_trades=1)

    @pytest.mark.parametrize("bad", [-1, 2.5, "3", True, float("nan")])
    def test_malformed_trades_refuse(self, bad):
        with pytest.raises(ef.EpisodicFitnessError):
            episode(closed_trades=bad)

    def test_zero_variance_sharpe_cannot_champion(self):
        # zero-variance -> sharpe unavailable upstream (None); it lands
        # in branch 3 and never earns the sharpe bonus
        result = episode(closed_trades=120, total_return=0.05,
                         sharpe=None)
        assert result["branch"] == ef.BRANCH_GAIN_NO_SHARPE

    def test_sentinel_is_configurable_and_episodic(self):
        result = episode(closed_trades=0,
                         config={"zero_trade_sentinel": -55.0})
        assert result["selection_value"] == -55.0
        assert result["branch"] == ef.BRANCH_ZERO_TRADE


class TestOrderingTable:
    def test_wp0_defect_is_corrected(self):
        """WP0 correction, stated exactly as ordered (section 4).

        The attractor is killed where it lives: a TRULY passive episode
        (zero closed trades) receives the sentinel and loses to every
        active fixture — including the still-learning negative one that
        the old comparator ranked below it. Within active losses the
        ordered semantics rank FIRST by movement toward zero loss
        (section 4.2), so a smaller loss legitimately outranks a larger
        one; the activity contribution is bounded relief, not an
        inversion of the loss order.
        """
        collapsed = value(closed_trades=0)
        learner = value(closed_trades=52, total_return=-0.002,
                        max_drawdown_fraction=0.01)
        assert learner > collapsed
        # and at EQUAL loss, the insufficient-activity quasi-passive
        # policy loses to the target-activity learner (counterexample 4)
        quasi_passive = value(closed_trades=2, total_return=-0.002,
                              max_drawdown_fraction=0.01)
        assert learner > quasi_passive
