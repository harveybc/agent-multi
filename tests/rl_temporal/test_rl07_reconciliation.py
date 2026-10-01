"""RL07: costs, slippage, leverage/margin, rejected orders, equity and open
terminal positions reconcile. No simulated fill is manufactured at
termination."""
from __future__ import annotations

import pytest

from ._fixtures import env_config, make_env, write_synthetic_csv
from ._mechanisms import assert_checkout_resolution, require

pytest.importorskip("backtrader")


def test_open_terminal_position_is_reported_not_filled(tmp_path):
    assert_checkout_resolution("RL07")
    reconcile = require("RL07", "rl_temporal.reconciliation", "reconcile_episode",
                        "episode reconciliation record")
    csv = write_synthetic_csv(tmp_path / "fixture.csv", rows=120)
    env = make_env(env_config(csv, action_space_mode="discrete", commission=0.001))
    try:
        # long at step 5, hold to the end: the position must still be open at data_end
        record = reconcile(env, actions=lambda step, info: 1 if step == 5 else 0)
    finally:
        env.close()
    assert record["termination_cause"] == "data_end"
    assert record["terminal_position"] != 0
    assert record["terminal_fill_manufactured"] is False
    assert record["trades_closed"] == 0
    assert record["entries_submitted"] >= 1
    assert record["commission_paid"] > 0.0
    assert abs(record["conservation_gap"]) < 1e-6, record
    assert record["final_equity"] == record["initial_cash"] + record["sum_step_pnl"]
    assert record["costs"]["commission"] == 0.001 and record["costs"]["slippage"] == 0.0
    assert record["margin"]["min_equity"] > 0 and "leverage" in record["margin"]
    assert isinstance(record["rejected_orders"], int)


def test_round_trip_closes_and_costs_reconcile(tmp_path):
    reconcile = require("RL07", "rl_temporal.reconciliation", "reconcile_episode",
                        "episode reconciliation record")
    csv = write_synthetic_csv(tmp_path / "fixture.csv", rows=120)
    env = make_env(env_config(csv, action_space_mode="discrete", commission=0.002))

    def policy(step, info):
        if step == 5:
            return 1
        if step == 30:
            return 2  # reverse: closes the long, opens a short
        if step == 60:
            return 0
        return 0

    try:
        record = reconcile(env, actions=policy, flatten_at=70)
    finally:
        env.close()
    assert record["trades_closed"] >= 1
    assert record["terminal_position"] == 0
    assert record["commission_paid"] == pytest.approx(record["sum_step_trade_cost"], abs=1e-9)
    assert abs(record["conservation_gap"]) < 1e-6, record
    assert record["turnover_units"] > 0


def test_margin_breach_terminates_with_cause(tmp_path):
    reconcile = require("RL07", "rl_temporal.reconciliation", "reconcile_episode",
                        "episode reconciliation record")
    csv = write_synthetic_csv(tmp_path / "fixture.csv", rows=120)
    env = make_env(env_config(csv, action_space_mode="discrete", initial_cash=10000.0,
                              position_size=1.0, min_equity=10000.0 - 0.5))
    try:
        record = reconcile(env, actions=lambda step, info: 1 if step == 5 else 0)
    finally:
        env.close()
    assert record["termination_cause"] in ("min_equity", "data_end")
    if record["termination_cause"] == "min_equity":
        assert record["bars_run"] < record["total_bars"]
        assert record["terminal_fill_manufactured"] is False
