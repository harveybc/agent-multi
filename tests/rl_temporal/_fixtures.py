"""Synthetic fixtures for the RL temporal-representation mechanism tests.

Everything here proves MECHANISMS. Nothing here says anything about trading
quality: the price is a seeded random walk and the features are causal
functions of past closes plus noise.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

FEATURES: List[str] = ["f_ret1", "f_ret4", "f_vol8"]
WINDOW = 24
SAMPLE_HOURS = 1


def write_synthetic_csv(path: Path, rows: int = 400, seed: int = 0,
                        start: str = "2020-01-01T00:00:00") -> Path:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=rows))
    close = np.maximum(close, 1.0)
    frame = pd.DataFrame({
        "DATE_TIME": pd.date_range(start, periods=rows, freq="h").strftime("%Y-%m-%d %H:%M:%S"),
        "OPEN": close, "HIGH": close + 0.2, "LOW": close - 0.2, "CLOSE": close,
        "VOLUME": rng.integers(100, 1000, size=rows).astype(float),
    })
    ret1 = pd.Series(close).pct_change(1).fillna(0.0)
    ret4 = pd.Series(close).pct_change(4).fillna(0.0)
    vol8 = ret1.rolling(8, min_periods=1).std().fillna(0.0)
    frame["f_ret1"] = ret1.to_numpy() + rng.normal(0, 1e-4, size=rows)
    frame["f_ret4"] = ret4.to_numpy() + rng.normal(0, 1e-4, size=rows)
    frame["f_vol8"] = vol8.to_numpy() + rng.normal(0, 1e-4, size=rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(path: Path, csv_path: Path, *, status: str = "FROZEN",
                   features: List[str] | None = None, train_rows: int = 300) -> Path:
    """A selected_feature_manifest.v1 shaped like lane B's ETH 4h draft."""
    features = list(features or FEATURES)
    doc: Dict[str, Any] = {
        "schema": "selected_feature_manifest.v1",
        "version": "fixture",
        "status": status,
        "task": {"dataset_id": "fixture.synthetic_random_walk.v1", "asset": "FIXTURE",
                 "bar": "1h", "decision_time": "after bar t closes"},
        "split": {"rule": "fixture", "train_rows": [0, train_rows],
                  "validation_rows": [train_rows, train_rows + 50],
                  "test_rows": [train_rows + 50, 400]},
        "resources": {"model_ready_view": {"path": str(csv_path), "sha256": sha256_file(csv_path),
                                           "governance": "FIXTURE file"}},
        "variants": {"A_all_admissible_control": {"features": features, "count": len(features)}},
        "selected_variant": "A_all_admissible_control",
    }
    path = Path(path)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def env_config(csv_path: Path, *, action_space_mode: str = "discrete",
               features: List[str] | None = None, max_rows: int | None = None,
               **overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "env_mode": "training",
        "input_data_file": str(csv_path),
        "date_column": "DATE_TIME", "price_column": "CLOSE", "headers": True,
        "max_rows": max_rows,
        "window_size": WINDOW,
        "initial_cash": 10000.0, "position_size": 1.0,
        "commission": 0.001, "slippage": 0.0,
        "data_feed_plugin": "default_data_feed", "broker_plugin": "default_broker",
        "strategy_plugin": "default_strategy",
        "preprocessor_plugin": "feature_window_preprocessor",
        "feature_columns": list(features or FEATURES),
        "feature_scaling": "rolling_zscore", "feature_scaling_window": 64, "feature_clip": 10.0,
        "include_price_window": False, "include_agent_state": True,
        "reward_plugin": "pnl_reward", "metrics_plugin": "trading_metrics",
        "action_space_mode": action_space_mode,
        "continuous_action_threshold": 0.33,
        "train_seed": 0,
    }
    cfg.update(overrides)
    return cfg


def make_env(cfg: Dict[str, Any]):
    from env_plugins.gym_fx_env import Plugin as EnvPlugin

    return EnvPlugin().make_env(dict(cfg))


def flat_env(cfg: Dict[str, Any]):
    from gymnasium.wrappers import FlattenObservation

    return FlattenObservation(make_env(cfg))


def modular_config(features: List[str] | None = None) -> Dict[str, Any]:
    features = list(features or FEATURES)
    return {
        "schema": "predictor.modular.v1", "window": WINDOW, "sample_hours": SAMPLE_HOURS,
        "feature_names": features,
        "branches": [{"name": f"branch_{i}", "features": [f]} for i, f in enumerate(features)],
        "core": {"params": {"d_model": 16, "heads": 2, "blocks": 1, "ff_dim": 32,
                            "stage_channels": [12, 10, 8], "time_factors": [2, 2, 1]}},
        "output_steps": 6, "output_channels": 8,
    }


if __name__ == "__main__":  # python -m tests.rl_temporal._fixtures > fixture_modular_config.json
    print(json.dumps(modular_config(), indent=2, sort_keys=True))
