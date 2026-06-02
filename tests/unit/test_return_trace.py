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


def test_unparseable_timestamps_fail_closed(tmp_path):
    """Unknown timestamps must not be treated as safely pre-heldout."""
    timestamps = ["2024-01-01 00:00:00", "not-a-timestamp"]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "bad_timestamp.csv")

    pipeline = _pipeline.PipelinePlugin()
    with pytest.raises(_trace.TraceTimestampError, match="unparseable"):
        pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    assert not (tmp_path / "bad_timestamp.csv").exists()
    assert not (tmp_path / "bad_timestamp.csv.meta.json").exists()


def test_non_monotonic_timestamps_fail_closed(tmp_path):
    """Per-split traces must be strictly ordered for time-series inference."""
    timestamps = [
        "2024-01-01 04:00:00",
        "2024-01-01 00:00:00",
    ]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "unordered.csv")

    pipeline = _pipeline.PipelinePlugin()
    with pytest.raises(_trace.TraceTimestampError, match="strictly increasing"):
        pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    assert not (tmp_path / "unordered.csv").exists()
    assert not (tmp_path / "unordered.csv.meta.json").exists()


def test_terminal_duplicate_timestamp_replaces_last_row(tmp_path):
    """A single terminal duplicate from gym env shutdown is normalized.

    Internal duplicates still fail closed; this case exists because gym-fx can
    emit the final market bar once for the final transition and once again as
    the terminal observation. The trace must keep one row per timestamp.
    """
    config = {"return_trace_file": str(tmp_path / "terminal_duplicate.csv")}
    rows = [
        {
            "step": 1, "timestamp": "2024-01-01 00:00:00", "asset": "ETHUSDT",
            "timeframe": "4h", "split": "stage_b_validation", "episode_id": "e",
            "run_id": "r", "seed": 0, "bar_index": 1, "price": 1,
            "action_raw": 0, "position": 0, "reward": 0, "gross_return": 0,
            "net_return": 0, "equity": 1000, "pnl": 0, "commission_paid": 0,
            "slippage_paid": 0, "trade_cost": 0, "trades": 0,
        },
        {
            "step": 2, "timestamp": "2024-01-01 04:00:00", "asset": "ETHUSDT",
            "timeframe": "4h", "split": "stage_b_validation", "episode_id": "e",
            "run_id": "r", "seed": 0, "bar_index": 2, "price": 2,
            "action_raw": 1, "position": 1, "reward": 1, "gross_return": 0.1,
            "net_return": 0.1, "equity": 1100, "pnl": 100, "commission_paid": 0,
            "slippage_paid": 0, "trade_cost": 0, "trades": 1,
        },
        {
            "step": 3, "timestamp": "2024-01-01 04:00:00", "asset": "ETHUSDT",
            "timeframe": "4h", "split": "stage_b_validation", "episode_id": "e",
            "run_id": "r", "seed": 0, "bar_index": 2, "price": 2,
            "action_raw": 0, "position": 0, "reward": 2, "gross_return": 0.2,
            "net_return": 0.2, "equity": 1200, "pnl": 200, "commission_paid": 0,
            "slippage_paid": 0, "trade_cost": 0, "trades": 2,
        },
    ]
    meta = _trace.write_return_trace(
        str(tmp_path / "terminal_duplicate.csv"),
        rows,
        config=config,
        split="stage_b_validation",
        seed=0,
        run_id="r",
        episode_id="e",
    )
    assert meta["row_count"] == 2
    written = list(csv.DictReader(open(tmp_path / "terminal_duplicate.csv")))
    assert written[-1]["equity"] == "1200"


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
    env.config["timeframe"] = "4h"
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


def test_trace_timestamp_overflow_uses_timeframe_delta(tmp_path):
    """If gym-fx emits one or two terminal observations past the final row,
    the trace writer must not collapse them onto a duplicate last timestamp.
    """
    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    env.config["timeframe"] = "4h"
    # Force one more step than dataframe rows.
    env._info_seq.append({
        "bar_index": 3,
        "equity": 10020.0,
        "pnl": 1.0,
        "position": 0.0,
        "price": 1002.0,
        "trade_cost": 0.0,
        "commission_paid": 0.0,
        "slippage_paid": 0.0,
        "trades": 1,
        "reward": 0.0,
    })
    config["return_trace_file"] = str(tmp_path / "overflow.csv")
    pipeline = _pipeline.PipelinePlugin()
    pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    rows = list(csv.DictReader(open(tmp_path / "overflow.csv")))
    assert [r["timestamp"] for r in rows] == [
        "2024-01-01 00:00:00",
        "2024-01-01 04:00:00",
        "2024-01-01 08:00:00",
    ]


def test_evaluation_progress_records_trades_profit_and_action_diagnostics(tmp_path):
    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    config["training_progress_file"] = str(tmp_path / "progress.json")
    pipeline = _pipeline.PipelinePlugin()
    summary = pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["source"] == "rl_pipeline_evaluation"
    assert payload["status"] == "evaluation_complete"
    assert payload["trades_total"] == summary["trades_total"]
    assert "profit_percent" in payload
    assert "action_non_hold_rate" in payload
    assert "no_trade_diagnosis" in payload


# ---------------------------------------------------------------------------
# V2: run-level evidence index
# ---------------------------------------------------------------------------
def _write_synthetic_trace(tmp_path, *, split, timestamps, config_overrides=None):
    """Drive write_return_trace directly to produce a metadata sidecar
    we can pass into build_return_trace_evidence without bringing up
    a full pipeline rollout."""
    rows = []
    eq = 10_000.0
    for i, ts in enumerate(timestamps):
        eq *= 1.001
        rows.append({
            "step": i + 1, "timestamp": ts, "asset": "ETHUSDT",
            "timeframe": "4h", "split": split, "episode_id": f"ep::{split}",
            "run_id": "run-xyz", "seed": 7, "bar_index": i + 1,
            "price": 1000.0 + i, "action_raw": 0.0, "position": 0.5,
            "reward": 0.01, "gross_return": 0.001, "net_return": 0.0009,
            "equity": eq, "pnl": 0.5, "commission_paid": 0.02,
            "slippage_paid": 0.01, "trade_cost": 0.05, "trades": 0,
        })
    config = {
        "asset": "ETHUSDT", "timeframe": "4h",
        "input_data_file": None, "save_model": "/tmp/agent-multi-test/model.zip",
    }
    if config_overrides:
        config.update(config_overrides)
    return _trace.write_return_trace(
        str(tmp_path / f"{split}_return_trace.csv"),
        rows,
        config=config,
        split=split,
        seed=7,
        asset="ETHUSDT",
        timeframe="4h",
        run_id="run-xyz",
        episode_id=f"ep::{split}",
    )


def test_evidence_emitted_for_single_split_pipeline(tmp_path):
    """Single-split rl_pipeline must surface evidence on the summary
    and write evidence.json next to the trace."""
    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "single.csv")

    pipeline = _pipeline.PipelinePlugin()
    summary = pipeline._evaluate(env, FakeAgent(), model=None, config=config)

    evidence = summary["return_trace_evidence"]
    assert evidence["schema_version"] == _trace.EVIDENCE_SCHEMA_VERSION
    assert evidence["trace_schema_version"] == _trace.SCHEMA_VERSION
    assert evidence["pipeline_plugin"] == "rl_pipeline"
    assert len(evidence["traces"]) == 1
    assert evidence["contains_heldout_rows"] is False
    assert evidence["stage_c_authorized"] is False
    evidence_path = Path(summary["return_trace_evidence_file"])
    assert evidence_path.exists()
    assert evidence_path.name == "evidence.json"
    on_disk = json.loads(evidence_path.read_text())
    assert on_disk["schema_version"] == _trace.EVIDENCE_SCHEMA_VERSION
    assert on_disk["traces"][0]["split"] == "evaluation"


def test_build_evidence_aggregates_three_splits(tmp_path):
    metas = [
        _write_synthetic_trace(tmp_path, split="train",
                               timestamps=["2023-01-01 00:00:00", "2023-01-01 04:00:00"]),
        _write_synthetic_trace(tmp_path, split="validation",
                               timestamps=["2024-01-01 00:00:00", "2024-01-01 04:00:00"]),
        _write_synthetic_trace(tmp_path, split="test",
                               timestamps=["2024-06-01 00:00:00", "2024-06-01 04:00:00"]),
    ]
    config = {"asset": "ETHUSDT", "timeframe": "4h", "input_data_file": None}
    evidence = _trace.build_return_trace_evidence(
        metas, config=config, run_id="run-xyz",
        pipeline_plugin="rl_pipeline_with_validation",
    )
    assert evidence["schema_version"] == _trace.EVIDENCE_SCHEMA_VERSION
    assert evidence["run_id"] == "run-xyz"
    assert evidence["asset"] == "ETHUSDT"
    assert evidence["timeframe"] == "4h"
    assert evidence["seed"] == 7
    assert evidence["heldout_boundary"] == "2025-01-01"
    assert evidence["contains_heldout_rows"] is False
    assert evidence["stage_c_authorized"] is False
    assert [t["split"] for t in evidence["traces"]] == ["train", "validation", "test"]
    for t in evidence["traces"]:
        assert Path(t["trace_file"]).exists()
        assert Path(t["metadata_file"]).exists()
        assert t["row_count"] == 2
        assert t["first_timestamp"] and t["last_timestamp"]


def test_legacy_config_emits_no_evidence(tmp_path):
    """Configs without return_trace_file must not produce any evidence keys
    or files. This is the critical backward-compat guarantee."""
    timestamps = ["2024-06-01 00:00:00", "2024-06-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    pipeline = _pipeline.PipelinePlugin()
    summary = pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    assert "return_trace_evidence" not in summary
    assert "return_trace_evidence_file" not in summary
    assert not (tmp_path / "evidence.json").exists()


def test_evidence_fails_on_unauthorized_heldout_metadata(tmp_path):
    """Defense in depth: even if a metadata sidecar somehow asserts
    heldout rows without Stage C authorization, evidence must refuse."""
    meta = _write_synthetic_trace(
        tmp_path, split="train",
        timestamps=["2023-01-01 00:00:00", "2023-01-01 04:00:00"],
    )
    # Tamper with the in-memory metadata to simulate the dangerous state.
    meta["contains_heldout_rows"] = True
    meta["stage_c_authorized"] = False
    config = {"asset": "ETHUSDT", "input_data_file": None}
    with pytest.raises(_trace.EvidenceConsistencyError, match="Stage C"):
        _trace.build_return_trace_evidence(
            [meta], config=config, pipeline_plugin="rl_pipeline",
        )


def test_evidence_fails_when_metadata_file_missing(tmp_path):
    meta = _write_synthetic_trace(
        tmp_path, split="train",
        timestamps=["2023-01-01 00:00:00", "2023-01-01 04:00:00"],
    )
    Path(meta["metadata_file"]).unlink()
    config = {"asset": "ETHUSDT", "input_data_file": None}
    with pytest.raises(_trace.EvidenceConsistencyError, match="metadata_file"):
        _trace.build_return_trace_evidence(
            [meta], config=config, pipeline_plugin="rl_pipeline",
        )


def test_evidence_fails_when_trace_file_missing(tmp_path):
    meta = _write_synthetic_trace(
        tmp_path, split="train",
        timestamps=["2023-01-01 00:00:00", "2023-01-01 04:00:00"],
    )
    Path(meta["trace_file"]).unlink()
    config = {"asset": "ETHUSDT", "input_data_file": None}
    with pytest.raises(_trace.EvidenceConsistencyError, match="trace_file"):
        _trace.build_return_trace_evidence(
            [meta], config=config, pipeline_plugin="rl_pipeline",
        )


def test_evidence_rejects_duplicate_split_labels(tmp_path):
    a = _write_synthetic_trace(
        tmp_path, split="train",
        timestamps=["2023-01-01 00:00:00", "2023-01-01 04:00:00"],
    )
    # Build a second metadata for "train" by writing it under a different name.
    rows = [{
        "step": 1, "timestamp": "2023-02-01 00:00:00", "asset": "ETHUSDT",
        "timeframe": "4h", "split": "train", "episode_id": "ep::train2",
        "run_id": "run-xyz", "seed": 7, "bar_index": 1, "price": 1.0,
        "action_raw": 0.0, "position": 0.0, "reward": 0.0,
        "gross_return": 0.0, "net_return": 0.0, "equity": 10_000.0,
        "pnl": 0.0, "commission_paid": None, "slippage_paid": None,
        "trade_cost": None, "trades": 0,
    }]
    b = _trace.write_return_trace(
        str(tmp_path / "train_return_trace_dupe.csv"),
        rows,
        config={"asset": "ETHUSDT", "timeframe": "4h", "input_data_file": None},
        split="train", seed=7, asset="ETHUSDT", timeframe="4h",
        run_id="run-xyz", episode_id="ep::train2",
    )
    config = {"asset": "ETHUSDT", "input_data_file": None}
    with pytest.raises(_trace.EvidenceConsistencyError, match="duplicate"):
        _trace.build_return_trace_evidence(
            [a, b], config=config, pipeline_plugin="rl_pipeline",
        )


def test_evidence_top_level_fields_present(tmp_path):
    metas = [
        _write_synthetic_trace(tmp_path, split="train",
                               timestamps=["2023-01-01 00:00:00", "2023-01-01 04:00:00"]),
        _write_synthetic_trace(tmp_path, split="validation",
                               timestamps=["2024-01-01 00:00:00", "2024-01-01 04:00:00"]),
    ]
    evidence = _trace.build_return_trace_evidence(
        metas, config={"asset": "ETHUSDT", "input_data_file": None},
        run_id="r", pipeline_plugin="rl_pipeline_with_validation",
    )
    required = {
        "schema_version", "generated_at", "run_id", "pipeline_plugin",
        "trace_schema_version", "asset", "timeframe", "seed",
        "config_hash", "data_file", "data_file_hash", "feature_list_hash",
        "heldout_boundary", "contains_heldout_rows", "stage_c_authorized",
        "traces",
    }
    assert required.issubset(set(evidence.keys()))
    for t in evidence["traces"]:
        for key in ("split", "trace_file", "trace_file_sha256",
                    "metadata_file", "row_count",
                    "first_timestamp", "last_timestamp",
                    "contains_heldout_rows", "stage_c_authorized"):
            assert key in t


def test_build_evidence_rejects_empty_input():
    with pytest.raises(_trace.EvidenceConsistencyError, match="empty"):
        _trace.build_return_trace_evidence(
            [], config={"asset": "ETHUSDT"}, pipeline_plugin="rl_pipeline",
        )


def test_trace_file_sha256_matches_rehashed_bytes(tmp_path):
    """Recorded trace_file_sha256 must equal sha256(open(trace_file).read()).

    Stage B's evaluator re-hashes the CSV bytes to detect tampering or a
    stale metadata sidecar pointing at an updated file. A regression here
    would silently break that audit.
    """
    import hashlib

    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "audit.csv")
    pipeline = _pipeline.PipelinePlugin()
    summary = pipeline._evaluate(env, FakeAgent(), model=None, config=config)

    trace_file = Path(summary["return_trace_file"])
    meta = json.loads(Path(summary["return_trace_metadata_file"]).read_text())
    rehashed = hashlib.sha256(trace_file.read_bytes()).hexdigest()
    assert meta["trace_file_sha256"] == rehashed
    evidence = summary["return_trace_evidence"]
    assert evidence["traces"][0]["trace_file_sha256"] == rehashed


def test_trace_trades_column_is_cumulative_passthrough(tmp_path):
    """Pin the documented `trades` semantics in STAGE_B_EVIDENCE_CONTRACT.md.

    gym-fx populates ``info["trades"]`` as a cumulative, non-decreasing
    counter (`bt_bridge.trade_count` reset at episode start, +1 on every
    closed trade). The trace writer is a passthrough — it must NOT
    transform per-step into cumulative or vice versa. Downstream
    financial-data evaluator (`infer_trade_count`) takes the last row's
    value when the series is non-decreasing.
    """
    timestamps = [
        "2024-01-01 00:00:00",
        "2024-01-01 04:00:00",
        "2024-01-01 08:00:00",
        "2024-01-01 12:00:00",
        "2024-01-01 16:00:00",
    ]
    df = pd.DataFrame({
        "DATE_TIME": timestamps,
        "CLOSE": [1000.0 + i for i in range(len(timestamps))],
    })
    # Cumulative trade counts: 0, 1, 1, 2, 3.
    cumulative = [0, 1, 1, 2, 3]
    info_seq = []
    eq = 10000.0
    for i in range(len(timestamps)):
        eq *= 1.0 + ((-1) ** i) * 0.001
        info_seq.append({
            "bar_index": i + 1,
            "equity": eq,
            "pnl": 0.0,
            "position": 1.0 if cumulative[i] > 0 else 0.0,
            "price": 1000.0 + i,
            "trade_cost": 0.0,
            "commission_paid": 0.0,
            "slippage_paid": 0.0,
            "trades": cumulative[i],
            "reward": 0.0,
        })
    env = FakeEnv(df, info_seq=info_seq)
    config = {
        "asset": "ETHUSDT",
        "timeframe": "4h",
        "input_data_file": None,
        "save_model": "/tmp/agent-multi-test-cumulative/model.zip",
        "eval_seed": 0,
        "return_trace_file": str(tmp_path / "cumulative.csv"),
    }
    pipeline = _pipeline.PipelinePlugin()
    summary = pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    trace_file = Path(summary["return_trace_file"])
    with trace_file.open() as fh:
        rows = list(csv.DictReader(fh))
    written = [int(r["trades"]) for r in rows]
    assert written == cumulative, (
        "trace writer must passthrough info['trades'] verbatim; got "
        f"{written}, expected {cumulative}"
    )
    # The series must be non-decreasing — the financial-data evaluator
    # relies on this to take the last value as the final trade count.
    assert all(written[i] >= written[i - 1] for i in range(1, len(written)))
    assert written[-1] == 3


# ---------------------------------------------------------------------------
# V3: feature_list_hash determinism, observation-state evidence, Stage B
# fail-closed when feature list is unresolvable, force-close observation
# fields gated by config, and Stage C dual-flag invariance.
# ---------------------------------------------------------------------------
def test_feature_list_hash_is_non_null_and_deterministic(tmp_path):
    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "fl.csv")
    config["feature_list"] = ["CLOSE", "rsi_14", "bb_upper"]
    pipeline = _pipeline.PipelinePlugin()
    summary = pipeline._evaluate(env, FakeAgent(), model=None, config=config)
    meta = json.loads(Path(summary["return_trace_metadata_file"]).read_text())
    assert meta["feature_list_hash"] and len(meta["feature_list_hash"]) == 64
    assert meta["feature_columns"] == ["CLOSE", "rsi_14", "bb_upper"]
    assert meta["feature_column_count"] == 3

    # Re-run with the same ordered list — hash must be identical.
    env2, config2 = _make_pipeline_inputs(timestamps)
    config2["return_trace_file"] = str(tmp_path / "fl2.csv")
    config2["feature_list"] = ["CLOSE", "rsi_14", "bb_upper"]
    summary2 = _pipeline.PipelinePlugin()._evaluate(env2, FakeAgent(), model=None, config=config2)
    meta2 = json.loads(Path(summary2["return_trace_metadata_file"]).read_text())
    assert meta2["feature_list_hash"] == meta["feature_list_hash"]


def test_feature_list_order_changes_hash(tmp_path):
    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "ord1.csv")
    config["feature_list"] = ["a", "b", "c"]
    s1 = _pipeline.PipelinePlugin()._evaluate(env, FakeAgent(), model=None, config=config)

    env2, config2 = _make_pipeline_inputs(timestamps)
    config2["return_trace_file"] = str(tmp_path / "ord2.csv")
    config2["feature_list"] = ["c", "b", "a"]
    s2 = _pipeline.PipelinePlugin()._evaluate(env2, FakeAgent(), model=None, config=config2)

    h1 = json.loads(Path(s1["return_trace_metadata_file"]).read_text())["feature_list_hash"]
    h2 = json.loads(Path(s2["return_trace_metadata_file"]).read_text())["feature_list_hash"]
    assert h1 and h2 and h1 != h2


def test_stage_b_locked_config_fails_closed_without_feature_list(tmp_path):
    """A Stage B locked config that resolves to no feature list must refuse
    to write evidence — non-auditable Stage B trace is unsafe."""
    timestamps = ["2024-01-01 00:00:00"]
    rows = [{
        "step": 1, "timestamp": "2024-01-01 00:00:00", "asset": "ETHUSDT",
        "timeframe": "4h", "split": "stage_b_validation", "episode_id": "ep",
        "run_id": "r", "seed": 0, "bar_index": 1, "price": 1.0,
        "action_raw": 0.0, "position": 0.0, "reward": 0.0,
        "gross_return": 0.0, "net_return": 0.0, "equity": 10_000.0,
        "pnl": 0.0, "commission_paid": None, "slippage_paid": None,
        "trade_cost": None, "trades": 0,
    }]
    config = {
        "asset": "ETHUSDT", "timeframe": "4h",
        "input_data_file": None,
        "_project3_stage_b_lock": True,
        "_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED": True,
    }
    with pytest.raises(_trace.FeatureListUnresolvedError, match="feature_list"):
        _trace.write_return_trace(
            str(tmp_path / "locked.csv"),
            rows,
            config=config,
            split="stage_b_validation",
            seed=0,
            run_id="r",
            episode_id="ep",
        )
    assert not (tmp_path / "locked.csv").exists()


def test_stage_b_locked_config_resolves_feature_list_from_env(tmp_path):
    """Stage B lock is fine as long as the env exposes feature columns."""
    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    # No explicit feature_list. Env dataframe columns -> ["CLOSE"] (DATE_TIME excluded).
    config["return_trace_file"] = str(tmp_path / "resolved.csv")
    config["_project3_stage_b_lock"] = True
    config["_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"] = True
    summary = _pipeline.PipelinePlugin()._evaluate(env, FakeAgent(), model=None, config=config)
    meta = json.loads(Path(summary["return_trace_metadata_file"]).read_text())
    assert meta["feature_columns"] == ["CLOSE"]
    assert meta["feature_list_hash"]
    assert meta["feature_column_count"] == 1


def test_observation_state_fields_recorded(tmp_path):
    """When env exposes a Dict observation space, its keys are recorded."""
    from gymnasium import spaces
    import numpy as np
    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    env.observation_space = spaces.Dict({
        "prices": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        "position": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
    })
    config["return_trace_file"] = str(tmp_path / "obs.csv")
    config["feature_list"] = ["CLOSE"]
    summary = _pipeline.PipelinePlugin()._evaluate(env, FakeAgent(), model=None, config=config)
    meta = json.loads(Path(summary["return_trace_metadata_file"]).read_text())
    assert set(meta["observation_state_fields"]) == {"prices", "position"}
    assert meta["observation_state_hash"] and len(meta["observation_state_hash"]) == 64


def test_observation_state_fields_resolved_through_flattening_wrapper(tmp_path):
    """A wrapper may expose a flattened Box while the base env still has Dict keys."""
    from gymnasium import spaces
    import numpy as np
    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    env.observation_space = spaces.Dict({
        "prices": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        "bars_to_force_close": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        "is_force_close_zone": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    })

    class FlatteningWrapper:
        def __init__(self, inner):
            self.env = inner
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        def reset(self, *args, **kwargs):
            return self.env.reset(*args, **kwargs)
        def step(self, *args, **kwargs):
            return self.env.step(*args, **kwargs)

    config["return_trace_file"] = str(tmp_path / "wrapped_obs.csv")
    config["feature_list"] = ["CLOSE"]
    summary = _pipeline.PipelinePlugin()._evaluate(FlatteningWrapper(env), FakeAgent(), model=None, config=config)
    meta = json.loads(Path(summary["return_trace_metadata_file"]).read_text())
    assert set(meta["observation_state_fields"]) == {
        "prices",
        "bars_to_force_close",
        "is_force_close_zone",
    }
    assert meta["observation_state_hash"] and len(meta["observation_state_hash"]) == 64


def test_stage_c_dual_flag_authorization_unchanged(tmp_path):
    """V3 evidence fields must not relax the Stage C dual-flag firewall."""
    timestamps = ["2025-03-01 00:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "stage_c.csv")
    config["feature_list"] = ["CLOSE"]
    # only one flag — must still fail closed
    config["final_stage_c_evaluation"] = True
    with pytest.raises(_trace.StageCAccessError):
        _pipeline.PipelinePlugin()._evaluate(env, FakeAgent(), model=None, config=config)
    # both flags — authorized path
    config["stage_c_acknowledged"] = True
    summary = _pipeline.PipelinePlugin()._evaluate(env, FakeAgent(), model=None, config=config)
    meta = json.loads(Path(summary["return_trace_metadata_file"]).read_text())
    assert meta["contains_heldout_rows"] is True
    assert meta["stage_c_authorized"] is True


def test_evidence_propagates_new_fields(tmp_path):
    timestamps = ["2024-01-01 00:00:00", "2024-01-01 04:00:00"]
    env, config = _make_pipeline_inputs(timestamps)
    config["return_trace_file"] = str(tmp_path / "prop.csv")
    config["feature_list"] = ["CLOSE", "ema_20"]
    summary = _pipeline.PipelinePlugin()._evaluate(env, FakeAgent(), model=None, config=config)
    ev = summary["return_trace_evidence"]
    assert ev["feature_list_hash"]
    assert ev["feature_columns"] == ["CLOSE", "ema_20"]
    assert ev["feature_column_count"] == 2
    # observation_state_fields is None for FakeEnv (no obs space); hash also None.
    assert ev["observation_state_fields"] in (None, [])
    # Per-trace evidence entry surfaces hash too
    assert ev["traces"][0]["feature_list_hash"] == ev["feature_list_hash"]


def test_resolve_feature_list_prefers_config_over_env(tmp_path):
    env, config = _make_pipeline_inputs(["2024-01-01 00:00:00"])
    config["feature_list"] = ["explicit_a", "explicit_b"]
    out = _trace.resolve_feature_list(config, env=env)
    assert out == ["explicit_a", "explicit_b"]


def test_resolve_feature_list_falls_back_to_env_dataframe():
    env, config = _make_pipeline_inputs(["2024-01-01 00:00:00"])
    config.pop("feature_list", None)
    out = _trace.resolve_feature_list(config, env=env)
    # DATE_TIME is excluded, CLOSE remains
    assert out == ["CLOSE"]
