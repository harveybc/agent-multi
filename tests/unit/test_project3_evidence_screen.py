from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from project3_evidence_screen import feature_proxy_screen  # noqa: E402


def test_feature_screen_keeps_selection_train_only_and_labels_metrics(tmp_path: Path) -> None:
    timestamps = pd.date_range("2021-01-01", "2023-12-31 23:00", freq="6h", tz="UTC")
    rng = np.random.default_rng(1701)
    signal = rng.normal(size=len(timestamps))
    returns = 0.002 * np.roll(signal, 1) + rng.normal(scale=0.003, size=len(timestamps))
    close = 100.0 * np.cumprod(1.0 + returns)
    frame = pd.DataFrame(
        {
            "DATE_TIME": timestamps,
            "OPEN": close,
            "HIGH": close * 1.001,
            "LOW": close * 0.999,
            "CLOSE": close,
            "VOLUME": 1000.0 + rng.normal(size=len(timestamps)),
            "signal": signal,
            "noise": rng.normal(size=len(timestamps)),
        }
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    result = feature_proxy_screen(
        {
            "asset": "SYNTH",
            "timeframe": "6h",
            "base_feature_bundle": "test",
            "input_data_file": str(path),
            "data_root": str(tmp_path),
            "external_context_bundle": "none",
            "external_context_lag_hours": 0,
            "train_start": "2021-01-01T00:00:00",
            "train_end": "2021-12-31T23:59:59",
            "validation_start": "2022-01-01T00:00:00",
            "validation_end": "2022-12-31T23:59:59",
            "test_start": "2023-01-01T00:00:00",
            "test_end": "2023-12-31T23:59:59",
            "target_horizon_hours": 6,
            "feature_selection_method": "rank_ic_topk",
            "feature_budget": 2,
            "preprocessing_mode": "rolling_zscore",
            "scaling_history_hours": 168,
            "clip_value": 10,
            "context_hours": 72,
            "context_representation": "summary",
            "ridge_alpha": 1.0,
            "action_threshold_quantile": 0.65,
            "transaction_cost_fraction": 0.0001,
            "risk_penalty_lambda": 1.0,
            "minimum_split_rows": 100,
        }
    )
    assert result["summary"]["selection_uses_validation"] is False
    assert result["summary"]["selection_uses_test"] is False
    keys = {
        (row["metric_name"], row["split"], row["unit"], row["horizon"])
        for row in result["metric_rows"]
    }
    assert ("mean_weekly_return", "validation", "fraction", "week") in keys
    assert ("annualized_return", "validation", "fraction", "year") in keys
    assert ("mean_weekly_rap", "test", "fraction", "week") in keys
    assert ("annual_rap", "test", "fraction", "year") in keys
