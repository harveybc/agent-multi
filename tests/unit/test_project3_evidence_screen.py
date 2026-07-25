from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from project3_evidence_screen import (  # noqa: E402
    _evaluate_split,
    _fit_proxy_model,
    _load_external_frame,
    _purged_split_mask,
    _target_series,
    FEATURE_PROXY_PROTOCOL,
    add_configured_transform_features,
    execute,
    feature_proxy_screen,
    merge_cross_asset_context,
    source_files,
)


@pytest.mark.parametrize(
    "family",
    ("ridge", "elastic_net", "pca_ridge", "hist_gradient_boosting", "mlp"),
)
def test_proxy_model_families_are_executable_and_deterministic(family: str) -> None:
    rng = np.random.default_rng(1701)
    train_x = pd.DataFrame(rng.normal(size=(220, 12)))
    train_y = pd.Series(
        0.4 * train_x.iloc[:, 0] - 0.2 * train_x.iloc[:, 1]
        + rng.normal(scale=0.05, size=len(train_x))
    )
    eval_x = pd.DataFrame(rng.normal(size=(40, 12)))
    kwargs = {
        "family": family,
        "alpha": 0.1,
        "latent_dimension": 8,
        "random_seed": 1701,
        "max_train_rows": 1000,
    }
    _, first, metadata = _fit_proxy_model(train_x, train_y, eval_x, **kwargs)
    _, second, _ = _fit_proxy_model(train_x, train_y, eval_x, **kwargs)
    assert len(first) == len(eval_x)
    assert np.isfinite(first).all()
    assert np.allclose(first, second)
    assert metadata["proxy_model_family"] == family


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


def test_cross_asset_context_is_causal_and_target_is_configurable(tmp_path: Path) -> None:
    timestamps = pd.date_range("2021-01-01", periods=80, freq="1h", tz="UTC")
    root = (
        tmp_path
        / "experiments"
        / "stage_a_screening"
        / "inputs"
        / "btcusdt"
        / "1h"
        / "baseline_12"
    )
    root.mkdir(parents=True)
    reference = pd.DataFrame(
        {
            "DATE_TIME": timestamps,
            "CLOSE": np.linspace(100.0, 120.0, len(timestamps)),
            "VOLUME": np.linspace(1000.0, 1100.0, len(timestamps)),
        }
    )
    reference.to_csv(root / "train.csv", index=False)
    target = pd.DataFrame(
        {
            "DATE_TIME": timestamps,
            "OPEN": np.linspace(10.0, 12.0, len(timestamps)),
            "HIGH": np.linspace(10.1, 12.1, len(timestamps)),
            "LOW": np.linspace(9.9, 11.9, len(timestamps)),
            "CLOSE": np.linspace(10.0, 12.0, len(timestamps)),
            "VOLUME": 100.0,
        }
    )
    merged, meta = merge_cross_asset_context(
        target,
        {
            "asset": "ETHUSDT",
            "timeframe": "1h",
            "data_root": str(tmp_path),
            "cross_asset_reference_set": "btc_eth",
        },
    )
    assert meta["cross_asset_source_count"] == 1
    assert meta["cross_asset_feature_count"] == 4
    assert merged.filter(like="cross_asset__btcusdt").shape[1] == 4
    forward = _target_series(
        target,
        horizon_rows=4,
        definition="forward_return",
        transaction_cost=0.001,
        risk_penalty_lambda=1.0,
    )
    cost_adjusted = _target_series(
        target,
        horizon_rows=4,
        definition="cost_adjusted_forward_return",
        transaction_cost=0.001,
        risk_penalty_lambda=1.0,
    )
    assert cost_adjusted.abs().dropna().mean() < forward.abs().dropna().mean()


def test_feature_screen_supports_registered_preprocessing_and_selectors(tmp_path: Path) -> None:
    timestamps = pd.date_range("2021-01-01", "2023-12-31 18:00", freq="6h", tz="UTC")
    rng = np.random.default_rng(12)
    signal = rng.normal(size=len(timestamps))
    close = 100.0 * np.cumprod(1.0 + 0.001 * np.roll(signal, 1) + rng.normal(0, 0.002, len(signal)))
    frame = pd.DataFrame(
        {
            "DATE_TIME": timestamps,
            "OPEN": close,
            "HIGH": close * 1.002,
            "LOW": close * 0.998,
            "CLOSE": close,
            "VOLUME": 1000.0 + np.abs(rng.normal(size=len(signal))),
            "signal": signal,
            "signal_copy": signal + rng.normal(0, 0.001, len(signal)),
            "noise": rng.normal(size=len(signal)),
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
            "cross_asset_reference_set": "none",
            "train_start": "2021-01-01T00:00:00",
            "train_end": "2021-12-31T23:59:59",
            "validation_start": "2022-01-01T00:00:00",
            "validation_end": "2022-12-31T23:59:59",
            "test_start": "2023-01-01T00:00:00",
            "test_end": "2023-12-31T23:59:59",
            "target_horizon_hours": 12,
            "target_definition": "future_rap",
            "feature_selection_method": "redundancy_stability_topk",
            "feature_budget": 2,
            "redundancy_threshold": 0.90,
            "stability_folds": 3,
            "preprocessing_mode": "rolling_rank_gaussian",
            "scaling_history_hours": 168,
            "clip_value": None,
            "log_transform_positive_features": True,
            "context_hours": 72,
            "context_representation": "multiscale_sequence",
            "ridge_alpha": 1.0,
            "action_threshold_quantile": 0.65,
            "transaction_cost_fraction": 0.0001,
            "risk_penalty_lambda": 1.0,
            "minimum_split_rows": 100,
        }
    )
    assert result["summary"]["selected_features"]
    assert result["summary"]["test"]["evaluation_weeks"] >= 51


def test_proxy_equity_uses_one_bar_realized_returns_not_overlapping_target() -> None:
    frame = pd.DataFrame(
        {
            "DATE_TIME": pd.date_range("2023-01-01", periods=24 * 14, freq="1h", tz="UTC"),
            "target_return": 0.50,
            "realized_return": 0.0,
        }
    )
    _, summary = _evaluate_split(
        "test",
        frame,
        np.ones(len(frame)),
        threshold=0.1,
        transaction_cost=0.001,
        risk_penalty_lambda=1.0,
    )
    assert summary["total_return"] == pytest.approx(-0.001)


def test_split_mask_purges_targets_that_cross_the_split_boundary() -> None:
    timestamps = pd.Series(
        pd.date_range("2022-12-31 16:00", "2023-01-01 04:00", freq="4h", tz="UTC")
    )
    mask = _purged_split_mask(
        timestamps,
        start=pd.Timestamp("2022-01-01T00:00:00Z"),
        end=pd.Timestamp("2022-12-31T23:59:59Z"),
        target_horizon_hours=4,
    )
    assert timestamps[mask].tolist() == [
        pd.Timestamp("2022-12-31T16:00:00Z"),
    ]


def test_empty_learned_bundle_is_recorded_as_blocked_not_worker_failure(tmp_path: Path) -> None:
    timestamps = pd.date_range("2021-01-01", periods=20, freq="1h", tz="UTC")
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "DATE_TIME": timestamps,
            "OPEN": 1.0,
            "HIGH": 1.0,
            "LOW": 1.0,
            "CLOSE": 1.0,
            "VOLUME": 1.0,
        }
    ).to_csv(path, index=False)
    result = execute(
        "feature_proxy_screen",
        {
            "asset": "SYNTH",
            "timeframe": "1h",
            "base_feature_bundle": "learned_cnn",
            "input_data_file": str(path),
            "data_root": str(tmp_path),
            "target_horizon_hours": 1,
        },
    )
    assert result["summary"]["screen_status"] == "blocked_no_numeric_nonleaking_features"
    assert result["metric_rows"][0]["value"] == 0.0
    assert result["evaluation_protocol_id"] == FEATURE_PROXY_PROTOCOL


def test_all_non_cryptoquant_bundle_uses_every_materialized_source(tmp_path: Path) -> None:
    root = tmp_path / "features" / "cross_source_features" / "1h"
    root.mkdir(parents=True)
    included = (
        "macro_economic__fred__rates__dff__observations.parquet",
        "economic_calendar__release_actuals__provider__announcements.parquet",
        "alternative_data__sentiment__provider__observations.parquet",
    )
    for name in included:
        (root / name).touch()
    (root / "alternative_data__cryptoquant__btc__observations.parquet").touch()
    files = source_files(
        {
            "asset": "BTCUSDT",
            "timeframe": "1h",
            "data_root": str(tmp_path),
            "external_context_bundle": "all_non_cryptoquant",
        }
    )
    assert [path.name for path in files] == sorted(included)


def test_external_publication_lag_uses_elapsed_time_not_source_rows(tmp_path: Path) -> None:
    path = tmp_path / "monthly.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-01-01T00:00:00Z", "2023-02-01T00:00:00Z"]
            ),
            "value": [1.0, 2.0],
        }
    ).to_parquet(path, index=False)
    loaded = _load_external_frame(
        path,
        start=pd.Timestamp("2023-01-01T00:00:00Z"),
        end=pd.Timestamp("2023-03-01T00:00:00Z"),
        lag_hours=24.0,
    )
    assert loaded["DATE_TIME"].tolist() == [
        pd.Timestamp("2023-01-02T00:00:00Z"),
        pd.Timestamp("2023-02-02T00:00:00Z"),
    ]


@pytest.mark.parametrize(
    ("patch", "family"),
    [
        (
            {
                "wavelet_family": "causal_multiscale_rolling",
                "transform_input_signal": "log_return",
                "wavelet_levels": [1, 2, 3],
            },
            "wavelet",
        ),
        (
            {"hilbert_input_signal": "volatility", "hilbert_window_hours": 72},
            "hilbert",
        ),
        (
            {
                "multitaper_input_signal": "log_return",
                "multitaper_window_hours": 72,
                "multitaper_time_bandwidth": 2.5,
                "multitaper_taper_count": 3,
            },
            "multitaper",
        ),
        ({"emd_input_signal": "detrended_close"}, "emd_proxy"),
        (
            {
                "fracdiff_input_signal": "log_close",
                "fracdiff_d": 0.6,
                "fracdiff_weight_threshold": 0.0001,
                "fracdiff_max_history_hours": 256,
            },
            "fracdiff",
        ),
    ],
)
def test_configured_transform_families_are_causal(
    patch: dict,
    family: str,
) -> None:
    count = 900
    rng = np.random.default_rng(91)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0001, 0.003, count))
    frame = pd.DataFrame(
        {
            "DATE_TIME": pd.date_range("2023-01-01", periods=count, freq="1h", tz="UTC"),
            "OPEN": close,
            "HIGH": close * 1.002,
            "LOW": close * 0.998,
            "CLOSE": close,
            "VOLUME": 1000.0 + np.abs(rng.normal(size=count)),
        }
    )
    config = {"timeframe": "1h", **patch}
    transformed, metadata = add_configured_transform_features(frame, config)
    modified = frame.copy()
    modified.loc[700:, "CLOSE"] *= 3.0
    modified.loc[700:, "VOLUME"] *= 7.0
    transformed_modified, _ = add_configured_transform_features(modified, config)
    generated = metadata["configured_transform_features"]
    assert family in metadata["configured_transform_families"]
    assert generated
    pd.testing.assert_frame_equal(
        transformed.loc[:699, generated],
        transformed_modified.loc[:699, generated],
    )
