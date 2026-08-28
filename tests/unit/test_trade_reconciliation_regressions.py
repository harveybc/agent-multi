"""Runtime order 2026-08-28 §2: the gamma reconciliation defect
(closed_trades_cumulative > trades_total) reproduced and corrected
THROUGH THE EXECUTING PIPELINE ROUTE (_load_env_plugin + the strict
reconcile primitive) — the exact PRE counterexample as a permanent
regression."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _return_trace as trace_mod  # noqa: E402
from pipeline_plugins.rl_pipeline_with_validation import (  # noqa: E402
    _load_env_plugin)


def _entry_bar_settlement_env(tmp_path):
    """26 flat bars; every bar from 8 pierces the 5% stop intrabar, so
    every long entry settles ON its own entry bar by direct broker
    accounting — the closure backtrader's analyzer never sees."""
    n = 26
    closes = [100.0] * n
    lows = [c * 0.9995 for c in closes]
    for i in range(8, n):
        lows[i] = 94.0
    frame = pd.DataFrame({
        "DATE_TIME": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "OPEN": closes, "HIGH": [c * 1.0005 for c in closes],
        "LOW": lows, "CLOSE": closes, "VOLUME": 1000.0,
        "feat": np.linspace(0, 1, n)})
    csv = tmp_path / "entry_bar_settlements.csv"
    frame.to_csv(csv, index=False)
    cfg = {
        "input_data_file": str(csv), "date_column": "DATE_TIME",
        "price_column": "CLOSE", "feature_columns": ["feat"],
        "feature_binary_columns": [], "window_size": 4,
        "initial_cash": 10000.0, "position_size": 1.0,
        "min_equity": 0.0, "env_mode": "training",
        "commission": 0.0, "leverage": 1.0,
        "action_space_mode": "continuous",
        "continuous_action_threshold": 0.0,
        "strategy_plugin": "shared_execution_envelope",
        "execution_envelope": {"envelope_mode": "fixed_fraction",
                               "sl_fraction": 0.05,
                               "tp_fraction": 0.10,
                               "leverage_cap": 1.0},
        "data_feed_plugin": "default_data_feed",
        "broker_plugin": "default_broker",
        "preprocessor_plugin": "default_preprocessor",
        "reward_plugin": "pnl_reward",
        "metrics_plugin": "default_metrics", "headers": True,
    }
    return _load_env_plugin("gym_fx_env", cfg).make_env(cfg)


def _drive(env):
    env.reset(seed=7)
    actions = [0.0] * 30
    for step in (4, 7, 10, 13):
        actions[step] = 1.0
    rows, done = [], False
    for a in actions:
        if done:
            break
        _o, _r, term, trunc, info = env.step([float(a)])
        done = bool(term or trunc)
        rows.append({"closed_trades_cumulative": info.get("trades"),
                     "position": info.get("position")})
    return rows, env.summary()


class TestGammaReconciliationCounterexample:
    def test_stream_and_summary_agree_exactly(self, tmp_path):
        rows, summary = _drive(_entry_bar_settlement_env(tmp_path))
        final = rows[-1]["closed_trades_cumulative"]
        assert final >= 3  # multiple settlement cycles occurred
        assert summary["trades_total"] == final
        # the strict primitive that refused on gamma now reconciles
        # with ZERO settlement — the counts derive from one stream
        recon = trace_mod.reconcile_trace_trades(
            rows, summary["trades_total"],
            terminal_open_positions=(
                1 if rows[-1]["position"] else 0))
        assert recon["terminal_settlement_trades"] == 0
        assert recon["final_cumulative"] == final

    def test_divergent_totals_still_refuse(self, tmp_path):
        """The guard must keep BITING: a foreign total below the trace
        counter is incoherent accounting, never settled."""
        rows, summary = _drive(_entry_bar_settlement_env(tmp_path))
        with pytest.raises(trace_mod.TraceReconciliationError,
                           match="exceeds"):
            trace_mod.reconcile_trace_trades(
                rows, summary["trades_total"] - 1,
                terminal_open_positions=0)

    def test_source_breakdown_is_conserved(self, tmp_path):
        _rows, summary = _drive(_entry_bar_settlement_env(tmp_path))
        sources = summary["closed_trades_by_source"]
        assert sum(sources.values()) == summary["trades_total"]
        assert sources.get("envelope_direct_settlement", 0) >= 1
        # the analyzer's undercount is preserved as a diagnostic
        assert summary["analyzer_trades_total"] <= \
            summary["trades_total"]
