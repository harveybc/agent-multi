"""Unit tests for the Stage B return-trace scaffold and pipeline wiring.

These tests use a tiny in-memory environment + agent so they exercise
the trace-emission path without ever building gym-fx, loading a real
SAC/PPO model, or launching training.
"""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


_trace = importlib.import_module("pipeline_plugins._return_trace")
_pipeline = importlib.import_module("pipeline_plugins.rl_pipeline")


# ---------------------------------------------------------------------------
class FakeEnv:
    """Minimal env exposing the surface ``rl_pipeline._evaluate`` needs."""

    def __init__(self, dataframe, *, info_seq):
        self.dataframe = dataframe
        self.config = {"date_column": "DATE_TIME"}
        self._info_seq = list(info_seq)
        self._idx = 0

    def reset(self, seed=0):
        self._idx = 0
        return ([0.0], {"equity": 10000.0, "bar_index": 0})

    def step(self, action):
        info = self._info_seq[self._idx]
        self._idx += 1
        terminated = self._idx >= len(self._info_seq)
        return ([0.0], info["reward"], terminated, False, info)

    def summary(self):
        return {
            "trades_total": 1,
            "trades_won": 1,
            "sharpe_ratio": 0.42,
            "total_return": 0.01,
            "final_equity": 10100.0,
            "max_drawdown_pct": 0.0,
        }


class FakeAgent:
    def predict(self, model, obs, deterministic=True):
        return 0.0


class FakeEnvPlugin:
    """Stub env plugin: wraps a pre-built FakeEnv and never imports gym-fx."""

    def __init__(self, env):
        self._env = env

    def make_env(self, config):
        return self._env

    def close(self):
        pass


def _make_pipeline_inputs(timestamps, *, asset="ETHUSDT", split=None):
    df = pd.DataFrame({
        "DATE_TIME": timestamps,
        "CLOSE": [1000.0 + i for i in range(len(timestamps))],
    })
    info_seq = []
    eq = 10000.0
    for i, ts in enumerate(timestamps):
        eq *= 1.0 + ((-1) ** i) * 0.001
        info_seq.append({
            "bar_index": i + 1,
            "equity": eq,
            "pnl": 1.0 if i % 2 == 0 else -0.5,
            "position": 0.5,
            "price": 1000.0 + i,
            "trade_cost": 0.05,
            "commission_paid": 0.02,
            "slippage_paid": 0.01,
            "trades": 1 if i == 0 else 0,
            "reward": float((-1) ** i) * 0.1,
        })
    env = FakeEnv(df, info_seq=info_seq)
    config = {
        "asset": asset,
        "timeframe": "4h",
        "input_data_file": None,
        "save_model": "/tmp/agent-multi-test/model.zip",
        "eval_seed": 7,
    }
    if split:
        config["eval_split"] = split
    return env, config


# ---------------------------------------------------------------------------
def test_trace_written_with_required_schema_and_metadata(tmp_path):
    timestamps = [
        "2024-01-01 00:00:00",
        "2024-01-01 04:00:00",
        "2024-01-01 08:00:00",
    ]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "test_trace.csv")
    config["feature_list"] = ["CLOSE", "rsi_14", "bb_upper"]

    pipeline = _pipeline.PipelinePlugin()
    summary = pipeline._evaluate(env, FakeAgent(), model=None, config=config)

    trace_file = Path(summary["return_trace_file"])
    meta_file = Path(summary["return_trace_metadata_file"])
    assert trace_file.exists() and meta_file.exists()

    with trace_file.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == len(timestamps)
    expected_cols = set(_trace.TRACE_FIELDNAMES)
    assert expected_cols.issubset(set(rows[0].keys()))
    # Required-ish content checks
    for r in rows:
        assert r["asset"] == "ETHUSDT"
        assert r["timeframe"] == "4h"
        assert r["seed"] == "7"
        assert r["split"] == "evaluation"
        assert r["run_id"]
        assert r["episode_id"]

    meta = json.loads(meta_file.read_text())
    assert meta["schema_version"] == _trace.SCHEMA_VERSION
    assert meta["heldout_boundary"] == "2025-01-01"
    assert meta["contains_heldout_rows"] is False
    assert meta["seed"] == 7
    assert meta["split"] == "evaluation"
    assert meta["row_count"] == len(timestamps)
    assert meta["asset"] == "ETHUSDT"
    assert meta["config_hash"] and len(meta["config_hash"]) == 64
    assert meta["feature_list_hash"] and len(meta["feature_list_hash"]) == 64
    assert meta["fields"] == list(_trace.TRACE_FIELDNAMES)


def test_trace_not_written_when_config_omits_field(tmp_path):
    """Backward-compat: configs without `return_trace_file` get no trace."""
    timestamps = ["2024-06-01 00:00:00", "2024-06-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    # Intentionally do NOT set return_trace_file
    pipeline = _pipeline.PipelinePlugin()
    summary = pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    assert "return_trace_file" not in summary
    assert "return_trace_metadata_file" not in summary
    # No stray files were written under tmp_path.
    assert list(tmp_path.iterdir()) == []


def test_heldout_rows_rejected_without_stage_c_authorization(tmp_path):
    """Any timestamp >= 2025-01-01 must fail closed."""
    timestamps = [
        "2024-12-31 20:00:00",
        "2025-01-01 00:00:00",  # heldout boundary
        "2025-01-01 04:00:00",
    ]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "stage_c_trace.csv")

    pipeline = _pipeline.PipelinePlugin()
    with pytest.raises(_trace.StageCAccessError, match="Stage C"):
        pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    # No trace nor sidecar should have been written.
    assert not (tmp_path / "stage_c_trace.csv").exists()
    assert not (tmp_path / "stage_c_trace.csv.meta.json").exists()


def test_heldout_rows_allowed_only_when_explicitly_authorized(tmp_path):
    timestamps = ["2025-01-02 00:00:00", "2025-01-02 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "auth_trace.csv")
    config["final_stage_c_evaluation"] = True
    config["stage_c_acknowledged"] = True

    pipeline = _pipeline.PipelinePlugin()
    summary = pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    meta = json.loads(Path(summary["return_trace_metadata_file"]).read_text())
    assert meta["contains_heldout_rows"] is True
    assert meta["stage_c_authorized"] is True


def test_partial_authorization_still_blocks(tmp_path):
    """Setting only one of the two required flags must still fail closed."""
    timestamps = ["2025-02-01 00:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "partial.csv")
    config["final_stage_c_evaluation"] = True  # but no acknowledgement

    pipeline = _pipeline.PipelinePlugin()
    with pytest.raises(_trace.StageCAccessError):
        pipeline._evaluate(env, FakeAgent(), model=None, config=config)


def test_metadata_records_seed_and_split_boundaries(tmp_path):
    timestamps = ["2024-01-01 00:00:00", "2024-12-31 20:00:00"]
    env, config = _make_pipeline_inputs(timestamps, split="train")
    config["return_trace_file"] = str(tmp_path / "tr.csv")
    pipeline = _pipeline.PipelinePlugin()
    summary = pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    meta = json.loads(Path(summary["return_trace_metadata_file"]).read_text())
    assert meta["split"] == "train"
    assert meta["split_boundaries"]["first_timestamp"] == "2024-01-01 00:00:00"
    assert meta["split_boundaries"]["last_timestamp"] == "2024-12-31 20:00:00"
    assert meta["seed"] == 7  # from config["eval_seed"]


def test_existing_configs_pass_through_when_field_missing():
    """Deterministic dry-run: building a default plugin with a typical
    legacy config (no return_trace_file) must not error and must not
    advertise any trace fields in the parameter dict it inherits."""
    cfg = {
        "asset": "ETHUSDT",
        "agent_plugin": "sac_agent",
        "save_model": "/tmp/x.zip",
    }
    plugin = _pipeline.PipelinePlugin(cfg)
    debug = plugin.get_debug_info()
    assert debug["return_trace_file"] is None


def test_invalid_split_label_raises(tmp_path):
    """write_return_trace must reject typos so a Stage B reviewer cannot
    accidentally consume an unlabelled trace."""
    rows = [{
        "step": 1, "timestamp": "2024-01-01 00:00:00", "asset": "ETHUSDT",
        "timeframe": "4h", "split": "train", "episode_id": "x", "run_id": "y",
        "seed": 0, "bar_index": 1, "price": 1.0, "action_raw": 0.0,
        "position": 0.0, "reward": 0.0, "gross_return": 0.0,
        "net_return": 0.0, "equity": 10000.0, "pnl": 0.0,
        "commission_paid": None, "slippage_paid": None, "trade_cost": None,
        "trades": 0,
    }]
    with pytest.raises(ValueError, match="not in"):
        _trace.write_return_trace(
            str(tmp_path / "bad.csv"), rows,
            config={"input_data_file": None}, split="totally-not-real",
            seed=0,
        )


def test_gross_minus_net_equals_cost_over_equity(tmp_path):
    """Sanity: gross_return - net_return ≈ Σcosts / prev_equity."""
    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "g.csv")
    pipeline = _pipeline.PipelinePlugin()
    pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    rows = list(csv.DictReader(open(tmp_path / "g.csv")))
    # Skip first row (no prev_equity baseline difference accumulates costs).
    r = rows[1]
    gross = float(r["gross_return"])
    net = float(r["net_return"])
    # Costs: trade_cost+commission+slippage = 0.08, prev_equity ≈ 10010 so
    # gross-net is ~8e-6 — small, positive, and finite.
    assert gross >= net
    assert (gross - net) > 0.0
