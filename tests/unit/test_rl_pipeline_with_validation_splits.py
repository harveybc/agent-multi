from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline_plugins.rl_pipeline_with_validation import PipelinePlugin, _early_stop_composite


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


def test_early_stop_composite_penalizes_validation_no_trade():
    composite, raw, passed, train_tail_ret, val_ret, train_trades, val_trades = _early_stop_composite(
        {"total_return": 3.5, "trades_total": 2},
        {"total_return": 0.0, "trades_total": 0},
        min_trades=1,
        no_trade_penalty=1_000_000.0,
    )

    assert raw == 1.75
    assert composite < -999_000
    assert passed is False
    assert train_tail_ret == 3.5
    assert val_ret == 0.0
    assert train_trades == 2
    assert val_trades == 0
