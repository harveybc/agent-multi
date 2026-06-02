from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline_plugins.rl_pipeline_with_validation import PipelinePlugin


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
    assert counts["val"] >= 30
    assert counts["test"] >= 30
    assert counts["train"] < 120
    assert counts["val"] < 80
    assert counts["test"] < 80
