from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline_plugins.rl_pipeline_with_validation import (
    PipelinePlugin,
    _activity_stop_disposition,
    _early_stop_composite,
    _resolve_l1_min_checkpoint_timesteps,
    _set_env_training_progress,
    _training_progress_for_epoch,
    _update_l1_checkpoint_state,
    _verify_artifact_sha256,
)


def test_activity_stop_preserves_the_truth_about_an_earlier_checkpoint():
    code, detail = _activity_stop_disposition(
        best_checkpoint_saved=True,
        streak=40,
        start_epoch=40,
        budget=40,
    )
    assert code == "activity_stop_after_best_checkpoint"
    assert "previously saved activity-eligible checkpoint" in detail
    assert "never passed" not in detail


def test_activity_stop_names_a_genuinely_missing_checkpoint():
    code, detail = _activity_stop_disposition(
        best_checkpoint_saved=False,
        streak=40,
        start_epoch=40,
        budget=40,
    )
    assert code == "activity_stop_no_eligible_checkpoint"
    assert "trade gate never passed" in detail


def test_training_progress_reaches_wrapped_env() -> None:
    class Base:
        progress = None

        def set_training_progress(self, value):
            self.progress = value

    class Wrapper:
        def __init__(self, env):
            self.env = env

    base = Base()
    assert _set_env_training_progress(Wrapper(Wrapper(base)), 1.4)
    assert base.progress == 1.0


def test_curriculum_progress_reaches_stress_before_hard_epoch_cap() -> None:
    assert _training_progress_for_epoch(
        1,
        max_epochs=2_000,
        curriculum_epochs=100,
    ) == 0.0
    assert _training_progress_for_epoch(
        81,
        max_epochs=2_000,
        curriculum_epochs=100,
    ) > 0.8
    assert _training_progress_for_epoch(
        101,
        max_epochs=2_000,
        curriculum_epochs=100,
    ) == 1.0


def test_warm_start_artifact_hash_fails_closed(tmp_path) -> None:
    artifact = tmp_path / "policy.zip"
    artifact.write_bytes(b"champion")

    actual = _verify_artifact_sha256(artifact, None)
    assert _verify_artifact_sha256(artifact, actual) == actual
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _verify_artifact_sha256(artifact, "0" * 64)


def test_day_based_micro_split_uses_small_windows(tmp_path):
    data = tmp_path / "tiny_window.csv"
    dates = pd.date_range("2024-09-01 00:00:00", "2024-10-15 20:00:00", freq="4h")
    data.write_text(
        "DATE_TIME,CLOSE,f1\n"
        + "\n".join(f"{ts},100,0.1" for ts in dates)
        + "\n",
        encoding="utf-8",
    )
    plugin = PipelinePlugin(
        {
            "input_data_file": str(data),
            "date_column": "DATE_TIME",
            "split_anchor": "end",
            "train_days": 14,
            "val_days": 7,
            "test_days": 7,
            "min_split_rows": 30,
            "quiet_mode": True,
        }
    )

    paths = plugin._split_csv(plugin.params | {"input_data_file": str(data), "date_column": "DATE_TIME"})

    try:
        counts = {name: len(pd.read_csv(path)) for name, path in paths.items()}
    finally:
        plugin._tempdir.cleanup()

    assert counts["train"] >= 30
    assert counts["train_tail"] >= 30
    assert counts["val"] >= 30
    assert counts["test"] >= 30
    assert counts["train"] < 120
    assert counts["train_tail"] < 80
    assert counts["val"] < 80
    assert counts["test"] < 80


def test_explicit_weekly_split_windows_are_used_exactly(tmp_path):
    data = tmp_path / "weekly_window.csv"
    dates = pd.date_range("2020-01-01 00:00:00", "2020-03-01 20:00:00", freq="4h")
    data.write_text(
        "DATE_TIME,CLOSE,f1\n"
        + "\n".join(f"{ts},100,0.1" for ts in dates)
        + "\n",
        encoding="utf-8",
    )
    plugin = PipelinePlugin(
        {
            "input_data_file": str(data),
            "date_column": "DATE_TIME",
            "train_start": "2020-01-01 00:00:00",
            "train_end": "2020-02-01 00:00:00",
            "validation_start": "2020-02-01 00:00:00",
            "validation_end": "2020-02-08 00:00:00",
            "test_start": "2020-02-08 00:00:00",
            "test_end": "2020-02-15 00:00:00",
            "min_split_rows": 30,
            "quiet_mode": True,
        }
    )

    paths = plugin._split_csv(plugin.params | {"input_data_file": str(data), "date_column": "DATE_TIME"})

    try:
        parts = {name: pd.read_csv(path) for name, path in paths.items()}
    finally:
        plugin._tempdir.cleanup()

    assert len(parts["train"]) == 31 * 6
    assert len(parts["train_tail"]) == 7 * 6
    assert len(parts["val"]) == 7 * 6
    assert len(parts["test"]) == 7 * 6
    assert parts["train"]["DATE_TIME"].iloc[0] == "2020-01-01 00:00:00"
    assert parts["train_tail"]["DATE_TIME"].iloc[0] == "2020-01-25 00:00:00"
    assert parts["val"]["DATE_TIME"].iloc[0] == "2020-02-01 00:00:00"
    assert parts["test"]["DATE_TIME"].iloc[0] == "2020-02-08 00:00:00"


def test_explicit_split_uses_configurable_train_tail_window(tmp_path):
    data = tmp_path / "weekly_window.csv"
    dates = pd.date_range("2020-01-01 00:00:00", "2020-03-01 20:00:00", freq="4h")
    data.write_text(
        "DATE_TIME,CLOSE,f1\n"
        + "\n".join(f"{ts},100,0.1" for ts in dates)
        + "\n",
        encoding="utf-8",
    )
    plugin = PipelinePlugin(
        {
            "input_data_file": str(data),
            "date_column": "DATE_TIME",
            "train_start": "2020-01-01 00:00:00",
            "train_end": "2020-02-01 00:00:00",
            "validation_start": "2020-02-01 00:00:00",
            "validation_end": "2020-02-15 00:00:00",
            "test_start": "2020-02-15 00:00:00",
            "test_end": "2020-02-29 00:00:00",
            "early_stop_train_tail_days": 14,
            "min_split_rows": 30,
            "quiet_mode": True,
        }
    )

    paths = plugin._split_csv(plugin.params | {"input_data_file": str(data), "date_column": "DATE_TIME"})

    try:
        parts = {name: pd.read_csv(path) for name, path in paths.items()}
    finally:
        plugin._tempdir.cleanup()

    assert len(parts["train_tail"]) == 14 * 6
    assert len(parts["val"]) == 14 * 6
    assert len(parts["test"]) == 14 * 6
    assert parts["train_tail"]["DATE_TIME"].iloc[0] == "2020-01-18 00:00:00"


def test_train_tail_expands_to_support_configured_observation_window(tmp_path):
    data = tmp_path / "weekly_window.csv"
    dates = pd.date_range("2020-01-01 00:00:00", "2020-03-01 20:00:00", freq="4h")
    data.write_text(
        "DATE_TIME,CLOSE,f1\n"
        + "\n".join(f"{ts},100,0.1" for ts in dates)
        + "\n",
        encoding="utf-8",
    )
    plugin = PipelinePlugin(
        {
            "input_data_file": str(data),
            "date_column": "DATE_TIME",
            "train_start": "2020-01-01 00:00:00",
            "train_end": "2020-02-01 00:00:00",
            "validation_start": "2020-02-01 00:00:00",
            "validation_end": "2020-02-15 00:00:00",
            "test_start": "2020-02-15 00:00:00",
            "test_end": "2020-02-29 00:00:00",
            "early_stop_train_tail_days": 2,
            "window_size": 42,
            "min_split_rows": 30,
            "quiet_mode": True,
        }
    )

    paths = plugin._split_csv(
        plugin.params
        | {
            "input_data_file": str(data),
            "date_column": "DATE_TIME",
            "window_size": 42,
        }
    )

    try:
        parts = {name: pd.read_csv(path) for name, path in paths.items()}
    finally:
        plugin._tempdir.cleanup()

    assert len(parts["train_tail"]) == 44
    assert parts["train_tail"]["DATE_TIME"].iloc[-1] == "2020-01-31 20:00:00"


def test_early_stop_composite_penalizes_validation_no_trade():
    composite, raw, passed, train_tail_ret, val_ret, train_trades, val_trades = _early_stop_composite(
        {"total_return": 3.5, "trades_total": 2},
        {"total_return": 0.0, "trades_total": 0},
        min_trades=1,
        no_trade_penalty=1_000_000.0,
    )

    assert raw == 0.875
    # WP1 (order 2026-08-18): an ineligible epoch carries NO comparable
    # selection score — None, never the historical raw-minus-1e6
    # sentinel that kept it rankable.
    assert composite is None
    assert passed is False
    assert train_tail_ret == 3.5
    assert val_ret == 0.0
    assert train_trades == 2
    assert val_trades == 0


def test_early_stop_composite_uses_split_specific_trade_minimums():
    composite, raw, passed, *_ = _early_stop_composite(
        {"total_return": 0.01, "trades_total": 1},
        {"total_return": 0.02, "trades_total": 1},
        min_train_tail_trades=1,
        min_validation_trades=12,
        no_trade_penalty=1_000_000.0,
    )

    assert passed is False
    assert composite is None  # WP1: no numeric sentinel


def test_l1_checkpoint_defaults_to_after_off_policy_learning_starts():
    assert _resolve_l1_min_checkpoint_timesteps({"learning_starts": 5_000}) == 5_001
    assert _resolve_l1_min_checkpoint_timesteps({}) == 0
    assert _resolve_l1_min_checkpoint_timesteps(
        {"learning_starts": 5_000, "l1_min_checkpoint_timesteps": 8_000}
    ) == 8_000


def test_l1_warmup_epoch_cannot_become_best_or_consume_patience():
    best, no_improve, improved = _update_l1_checkpoint_state(
        composite=0.25,
        best_composite=float("-inf"),
        no_improve=0,
        min_delta=0.0001,
        eligible=False,
    )

    assert best == float("-inf")
    assert no_improve == 0
    assert improved is False

    best, no_improve, improved = _update_l1_checkpoint_state(
        composite=0.20,
        best_composite=best,
        no_improve=no_improve,
        min_delta=0.0001,
        eligible=True,
    )
    assert best == 0.20
    assert no_improve == 0
    assert improved is True


def test_l1_checkpoint_can_improve_before_patience_counter_starts():
    best, no_improve, improved = _update_l1_checkpoint_state(
        composite=0.25,
        best_composite=0.2,
        no_improve=0,
        min_delta=0.01,
        eligible=True,
        patience_eligible=False,
    )
    assert best == pytest.approx(0.25)
    assert no_improve == 0
    assert improved is True

    best, no_improve, improved = _update_l1_checkpoint_state(
        composite=0.24,
        best_composite=best,
        no_improve=no_improve,
        min_delta=0.01,
        eligible=True,
        patience_eligible=False,
    )
    assert best == pytest.approx(0.25)
    assert no_improve == 0
    assert improved is False
