"""WP2.1 (continuation order 2026-08-20): REAL-environment NOP and
reward evidence — the executing gym-fx env over real ETH 4h bars, not a
standalone arithmetic fixture."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _episodic_activity_fitness as ef  # noqa: E402
from pipeline_plugins.rl_pipeline_with_validation import (  # noqa: E402
    _load_env_plugin,
)

DATA = Path("/home/harveybc/Documents/GitHub/predictor/examples/data/"
            "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
CAL = {"activity_plateau_low_rate": 50.0,
       "activity_plateau_high_rate": 300.0}


@pytest.fixture(scope="module")
def eth_slice(tmp_path_factory):
    if not DATA.is_file():
        pytest.skip("real ETH csv not present")
    sliced = tmp_path_factory.mktemp("eth") / "eth_600.csv"
    with DATA.open() as src, sliced.open("w") as dst:
        for i, line in enumerate(src):
            if i > 600:
                break
            dst.write(line)
    return sliced


def make_env(eth_slice, threshold=0.1):
    cfg = {"input_data_file": str(eth_slice), "window_size": 32,
           "initial_cash": 10000.0, "action_space_mode": "continuous",
           "continuous_action_threshold": threshold,
           "solvency_mode": "normal_realistic", "max_steps": 500}
    plug = _load_env_plugin("gym_fx_env", cfg)
    return plug.make_env(cfg)


def run_policy(env, actions_fn, bars=400):
    obs, _ = env.reset(seed=1)
    total, neg, rewards, equity_series = 0.0, 0, [], []
    info = {}
    for i in range(bars):
        obs, r, term, trunc, info = env.step([actions_fn(i)])
        total += r
        rewards.append(r)
        equity_series.append(float(info.get("equity", 10000.0)
                                   or 10000.0))
        if r < 0:
            neg += 1
        if term or trunc:
            break
    trades = int(info.get("trades", info.get("trades_total", 0)) or 0)
    # correction 1 (2026-08-20 20:40): drawdown is DERIVED from the
    # recorded equity series of the trajectory, never hard-coded.
    peak, max_dd = equity_series[0], 0.0
    for value in equity_series:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    return {"steps": i + 1, "total_reward": total, "negative_steps": neg,
            "trades": trades, "equity": equity_series[-1],
            "equity_series": equity_series,
            "max_drawdown_fraction": max_dd, "rewards": rewards}


class TestRealEnvironmentNop:
    def test_intraperiod_nop_is_legal_and_free(self, eth_slice):
        """Waiting bar-by-bar earns exactly zero — no per-bar penalty
        exists anywhere in the executing reward path."""
        result = run_policy(make_env(eth_slice), lambda i: 0.0)
        assert result["steps"] >= 400
        assert result["trades"] == 0
        assert result["total_reward"] == 0.0
        assert result["negative_steps"] == 0
        assert all(r == 0.0 for r in result["rewards"])

    def test_zero_trades_penalized_only_at_episode_completion(
            self, eth_slice):
        """The sentinel exists ONLY in the episodic evaluation of the
        finished trajectory, never inside the env steps."""
        result = run_policy(make_env(eth_slice), lambda i: 0.0)
        episode = ef.evaluate_episode(
            total_return=result["equity"] / 10000.0 - 1.0,
            max_drawdown_fraction=0.0, sharpe=None,
            closed_trades=result["trades"],
            scored_rows=result["steps"], config=dict(CAL))
        assert episode["branch"] == ef.BRANCH_ZERO_TRADE
        assert episode["selection_value"] == -100.0

    def test_active_losing_learner_outranks_terminal_inactivity(
            self, eth_slice):
        """A real active trajectory (crossing the normal threshold,
        paying real costs) must outrank the finished NOP episode —
        without making catastrophic loss attractive."""
        nop = run_policy(make_env(eth_slice), lambda i: 0.0)
        active = run_policy(
            make_env(eth_slice),
            lambda i: 0.5 if (i // 40) % 2 else -0.5)
        assert active["trades"] > 0
        def episode(run):
            return ef.evaluate_episode(
                total_return=run["equity"] / 10000.0 - 1.0,
                max_drawdown_fraction=run["max_drawdown_fraction"],
                sharpe=None, closed_trades=run["trades"],
                scored_rows=run["steps"], config=dict(CAL))
        nop_ep = episode(nop)
        active_ep = episode(active)
        assert active["max_drawdown_fraction"] >= 0.0  # measured
        assert active_ep["selection_value"] > \
            nop_ep["selection_value"]
        # SYNTHETIC catastrophic counterexample (labelled synthetic:
        # not derived from this trajectory)
        catastrophic = ef.evaluate_episode(
            total_return=-0.95, max_drawdown_fraction=0.95,
            sharpe=None, closed_trades=active["trades"],
            scored_rows=active["steps"], config=dict(CAL))
        assert nop_ep["selection_value"] < \
            catastrophic["selection_value"] < \
            active_ep["selection_value"]
