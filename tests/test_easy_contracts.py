"""Order 2026-08-21 §2 hierarchy + §3 adversarial cases."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _easy_contracts as ec  # noqa: E402

CAL = {"activity_plateau_low_rate": 50.0,
       "activity_plateau_high_rate": 300.0}


def fitness(trades, val_ret, val_dd=0.05, tt_ret=0.0, rows=2190):
    return ec.easy_doin_candidate_fitness(
        closed_trades=trades, scored_rows=rows,
        validation_return=val_ret, validation_drawdown=val_dd,
        train_tail_return=tt_ret, activity_config=CAL)


class TestHierarchy:
    def test_zero_trades_loses_to_every_finite_active_learner(self):
        zero = fitness(0, 0.0)
        for case in (fitness(1, -0.90, 0.95), fitness(20, -0.05),
                     fitness(120, 0.02), fitness(5000, -0.01)):
            assert case["lex_key"] > zero["lex_key"]
        assert zero["eligible"] is False
        assert zero["reason"] == "ZERO_TRADES"

    def test_activity_band_orders_materially_different_levels(self):
        low = fitness(4, 0.05)          # below band
        target = fitness(120, 0.01)     # in band, WORSE economics
        assert target["lex_key"] > low["lex_key"]

    def test_economics_orders_within_comparable_activity(self):
        a = fitness(120, 0.05)
        b = fitness(150, 0.01)          # same band, worse economics
        assert a["lex_key"] > b["lex_key"]

    def test_gap_is_bounded_tiebreak_never_reverses(self):
        # equal economics, unequal generalization gap -> gap decides
        small_gap = fitness(120, 0.05, tt_ret=0.06)
        big_gap = fitness(120, 0.05, tt_ret=0.90)
        assert small_gap["lex_key"] > big_gap["lex_key"]
        # but a HUGE gap can never beat better economics or activity
        better_econ = fitness(120, 0.06, tt_ret=5.0)
        assert better_econ["lex_key"] > small_gap["lex_key"]
        assert big_gap["components"]["gap_bounded"] <= 1.0

    def test_catastrophic_loss_monotonically_worse(self):
        assert fitness(120, -0.05)["lex_key"] > \
            fitness(120, -0.50)["lex_key"] > \
            fitness(120, -0.95)["lex_key"]

    def test_equal_activity_unequal_risk(self):
        assert fitness(120, 0.05, val_dd=0.02)["lex_key"] > \
            fitness(120, 0.05, val_dd=0.30)["lex_key"]

    def test_one_losing_trade_beats_zero(self):
        assert fitness(1, -0.02)["lex_key"] > fitness(0, 0.0)["lex_key"]

    def test_overtrading_below_target_above_zero(self):
        over = fitness(5000, 0.02)
        target = fitness(120, 0.02)
        zero = fitness(0, 0.0)
        assert zero["lex_key"] < over["lex_key"] < target["lex_key"]


class TestSeparation:
    def test_contracts_are_independently_named(self):
        m = ec.easy_checkpoint_monitor(
            train_tail_return=0.01, validation_return=0.01,
            train_tail_drawdown=0.0, validation_drawdown=0.0)
        f = fitness(10, 0.01)
        ec.assert_distinct_contracts(m, f)
        assert m["contract_id"] != f["contract_id"]

    def test_shared_identity_refuses(self):
        with pytest.raises(ec.EasyContractError,
                           match="SHARED_CONTRACT_IDENTITY"):
            ec.assert_distinct_contracts({"contract_id": "x"},
                                         {"contract_id": "x"})

    @pytest.mark.parametrize("fn,kwargs", [
        (ec.easy_checkpoint_monitor,
         dict(train_tail_return=0.1, validation_return=0.1,
              train_tail_drawdown=0.0, validation_drawdown=0.0,
              test_sharpe=0.5)),
    ])
    def test_test_facts_refuse(self, fn, kwargs):
        with pytest.raises(ec.EasyContractError,
                           match="REFUSED_TEST_FACT"):
            fn(**kwargs)

    def test_fitness_refuses_test_facts_too(self):
        with pytest.raises(ec.EasyContractError,
                           match="REFUSED_TEST_FACT"):
            fitness_kwargs = dict(
                closed_trades=10, scored_rows=2190,
                validation_return=0.01, validation_drawdown=0.0,
                train_tail_return=0.0, activity_config=CAL,
                test_return=0.5)
            ec.easy_doin_candidate_fitness(**fitness_kwargs)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x",
                                     True, None])
    def test_non_finite_facts_refuse(self, bad):
        with pytest.raises(ec.EasyContractError):
            fitness(10, bad)
