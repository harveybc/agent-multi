#!/usr/bin/env python3
"""Leakage-aware CPU evidence screens for Project 3 data contracts."""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.signal.windows import dpss
from scipy.stats import norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neural_network import MLPRegressor

from project3_evidence_metrics import METRIC_SCHEMA, canonical_trading_metrics


CORE_COLUMNS = {"DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"}
LEAK_TOKENS = ("target", "label", "future", "fwd_", "next_", "prediction")
FEATURE_PROXY_PROTOCOL = "project3.feature_proxy.one_bar_execution.v3"
EVIDENCE_EXECUTOR_VERSION = "project3.evidence.executor.v3"
FEATURE_PROXY_PROTOCOL_HASH = hashlib.sha256(
    (
        "target may span configured horizon; positions are recomputed per bar; "
        "equity uses next-bar realized return exactly once; transaction cost "
        "is charged on absolute position turnover; train-only feature selection; "
        "each split purges decisions whose target horizon crosses its end"
    ).encode("utf-8")
).hexdigest()

SCREEN_DEFAULTS = {
    "external_context_bundle": "none",
    "external_context_lag_hours": 0,
    "missing_value_policy": "causal_ffill",
    "max_staleness_hours": None,
    "cross_asset_reference_set": "none",
    "cross_asset_volatility_window_hours": 24,
    "target_definition": "forward_return",
    "target_barrier_volatility_window_hours": 24,
    "feature_selection_method": "rank_ic_topk",
    "feature_budget": 32,
    "redundancy_threshold": 0.95,
    "stability_folds": 5,
    "selection_regime_volatility_window_hours": 24,
    "preprocessing_mode": "rolling_zscore",
    "scaling_history_hours": 168,
    "clip_value": 10,
    "log_transform_positive_features": False,
    "transform_volatility_window_hours": 24,
    "transform_detrend_window_hours": 168,
    "transform_sample_interval_hours": 24,
    "transform_input_signal": "close",
    "wavelet_family": "off",
    "wavelet_base_scale_hours": 8,
    "wavelet_levels": [1, 2, 3, 4],
    "hilbert_input_signal": "off",
    "hilbert_window_hours": 168,
    "multitaper_input_signal": "off",
    "multitaper_window_hours": 168,
    "multitaper_time_bandwidth": 3.5,
    "multitaper_taper_count": 5,
    "emd_input_signal": "off",
    "emd_backend": "causal_rolling_proxy",
    "emd_window_hours": [8, 32, 128],
    "fracdiff_input_signal": "off",
    "fracdiff_d": 0.4,
    "fracdiff_weight_threshold": 0.0001,
    "fracdiff_max_history_hours": 720,
    "context_hours": 168,
    "context_representation": "summary",
    "ridge_alpha": 1.0,
    "proxy_model_family": "ridge",
    "proxy_latent_dimension": 32,
    "proxy_random_seed": 1701,
    "proxy_max_train_rows": 25000,
    "action_threshold_quantile": 0.65,
    "transaction_cost_fraction": 0.0005,
    "risk_penalty_lambda": 1.0,
    "minimum_split_rows": 200,
}

RESOLVED_PARAMETER_KEYS = (
    "asset",
    "timeframe",
    "base_feature_bundle",
    "external_context_bundle",
    "external_context_lag_hours",
    "missing_value_policy",
    "max_staleness_hours",
    "cross_asset_reference_set",
    "cross_asset_volatility_window_hours",
    "target_horizon_hours",
    "target_definition",
    "target_barrier_volatility_window_hours",
    "feature_selection_method",
    "feature_budget",
    "redundancy_threshold",
    "stability_folds",
    "selection_regime_volatility_window_hours",
    "preprocessing_mode",
    "scaling_history_hours",
    "clip_value",
    "log_transform_positive_features",
    "transform_volatility_window_hours",
    "transform_detrend_window_hours",
    "transform_sample_interval_hours",
    "transform_input_signal",
    "wavelet_family",
    "wavelet_base_scale_hours",
    "wavelet_levels",
    "hilbert_input_signal",
    "hilbert_window_hours",
    "multitaper_input_signal",
    "multitaper_window_hours",
    "multitaper_time_bandwidth",
    "multitaper_taper_count",
    "emd_input_signal",
    "emd_backend",
    "emd_window_hours",
    "fracdiff_input_signal",
    "fracdiff_d",
    "fracdiff_weight_threshold",
    "fracdiff_max_history_hours",
    "context_hours",
    "context_representation",
    "ridge_alpha",
    "proxy_model_family",
    "proxy_latent_dimension",
    "proxy_random_seed",
    "proxy_max_train_rows",
    "action_threshold_quantile",
    "transaction_cost_fraction",
    "risk_penalty_lambda",
    "minimum_split_rows",
)

SOURCE_PATTERNS = {
    "macro_core": [
        "macro_economic__fred__rates__dff__observations.parquet",
        "macro_economic__fred__rates__dgs2__observations.parquet",
        "macro_economic__fred__rates__dgs10__observations.parquet",
        "macro_economic__fred__rates__dgs30__observations.parquet",
        "macro_economic__fred__rates__t10y2y__observations.parquet",
        "macro_economic__fred__rates__t10y3m__observations.parquet",
        "macro_economic__fred__credit__bamlh0a0hym2__observations.parquet",
        "macro_economic__fred__employment__icsa__observations.parquet",
        "macro_economic__fred__employment__payems__observations.parquet",
        "macro_economic__fred__employment__unrate__observations.parquet",
        "macro_economic__fred__inflation__cpiaucsl__observations.parquet",
        "macro_economic__fred__inflation__pcepilfe__observations.parquet",
        "macro_economic__fred__inflation_expectations__t10yie__observations.parquet",
        "macro_economic__fred__money__m2sl__observations.parquet",
        "macro_economic__fred__money__walcl__observations.parquet",
        "macro_economic__fred__fx_indices__dtwexbgs__observations.parquet",
        "macro_economic__fred__recession__usrec__observations.parquet",
        "macro_economic__fred__stress__anfci__observations.parquet",
        "macro_economic__fred__stress__nfci__observations.parquet",
        "macro_economic__fred__stress__vixcls__observations.parquet",
    ],
    "market_core": [
        "market_data__equities__us_indices__gspc__daily.parquet",
        "market_data__equities__us_indices__vix__daily.parquet",
        "market_data__equities__etfs__qqq__daily.parquet",
        "market_data__equities__etfs__tlt__daily.parquet",
        "market_data__equities__etfs__tip__daily.parquet",
        "market_data__commodities__energy__cl_f__daily.parquet",
        "market_data__commodities__precious_metals__gc_f__daily.parquet",
    ],
    "funding_core": ["market_data__crypto__funding_rates__*__funding_rates.parquet"],
    "economic_calendar": [
        "economic_calendar__scheduled_events__fxmacrodata__release_calendar.parquet",
        "economic_calendar__release_actuals__fxmacrodata__announcements.parquet",
    ],
}

CROSS_ASSET_SETS = {
    "btc_eth": ("btcusdt", "ethusdt"),
    "crypto_leaders": ("btcusdt", "ethusdt", "solusdt"),
    "fx_leaders": ("eurusd", "usdjpy", "gbpusd"),
    "portfolio_candidates": (
        "solusdt",
        "btcusdt",
        "adausdt",
        "eurusd",
        "dogeusdt",
        "audusd",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _timeframe_hours(timeframe: str) -> float:
    text = timeframe.strip().lower()
    if text.endswith("m"):
        return float(text[:-1]) / 60.0
    if text.endswith("h"):
        return float(text[:-1])
    if text.endswith("d"):
        return float(text[:-1]) * 24.0
    raise ValueError(f"unsupported timeframe: {timeframe}")


def _purged_split_mask(
    timestamps: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_horizon_hours: float,
) -> pd.Series:
    horizon = pd.Timedelta(hours=float(target_horizon_hours))
    return (timestamps >= start) & (timestamps <= end - horizon)


def _metric(
    name: str,
    value: float | None,
    *,
    unit: str,
    horizon: str,
    aggregation: str,
    split: str,
) -> dict[str, Any]:
    return {
        "metric_schema": METRIC_SCHEMA,
        "metric_name": name,
        "value": value,
        "unit": unit,
        "horizon": horizon,
        "aggregation": aggregation,
        "split": split,
    }


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in CORE_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
        and not any(token in column.lower() for token in LEAK_TOKENS)
    ]


def _load_base(config: dict[str, Any]) -> pd.DataFrame:
    path = Path(str(config["input_data_file"]))
    if not path.exists():
        raise FileNotFoundError(str(path))
    frame = pd.read_csv(path)
    if "DATE_TIME" not in frame or "CLOSE" not in frame:
        raise ValueError(f"{path} must contain DATE_TIME and CLOSE")
    frame["DATE_TIME"] = pd.to_datetime(frame["DATE_TIME"], utc=True, errors="raise")
    frame = frame.sort_values("DATE_TIME").drop_duplicates("DATE_TIME", keep="last")
    return frame


def _glob_patterns(root: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted(set(paths))


def _onchain_patterns(asset: str) -> list[str]:
    token = asset.lower().replace("_perp", "").replace("usdt", "")
    symbols = sorted({token, "btc", "eth"})
    patterns: list[str] = []
    for symbol in symbols:
        patterns.extend(
            [
                f"alternative_data__onchain_other__coinmetrics_community__{symbol}__*.parquet",
                f"alternative_data__onchain_{symbol}__coinmetrics_community__{symbol}__*.parquet",
            ]
        )
    return patterns


def source_files(config: dict[str, Any]) -> list[Path]:
    bundle = str(config.get("external_context_bundle") or "none")
    if bundle == "none":
        return []
    root = (
        Path(str(config["data_root"]))
        / "features"
        / "cross_source_features"
        / str(config["timeframe"])
    )
    if bundle == "all_non_cryptoquant":
        return sorted(
            path
            for path in root.glob("*.parquet")
            if "cryptoquant" not in path.name.lower()
        )
    patterns: list[str] = []
    if bundle in {"macro_core", "macro_market"}:
        patterns.extend(SOURCE_PATTERNS["macro_core"])
    if bundle in {"market_core", "macro_market", "crypto_context"}:
        patterns.extend(SOURCE_PATTERNS["market_core"])
    if bundle in {"onchain_asset", "crypto_context"}:
        patterns.extend(_onchain_patterns(str(config["asset"])))
    if bundle in {"funding_core", "crypto_context"}:
        patterns.extend(SOURCE_PATTERNS["funding_core"])
    if bundle == "economic_calendar":
        patterns.extend(SOURCE_PATTERNS["economic_calendar"])
    files = _glob_patterns(root, patterns)
    return [path for path in files if "cryptoquant" not in path.name.lower()]


def cross_asset_files(config: dict[str, Any]) -> list[Path]:
    reference_set = str(config.get("cross_asset_reference_set") or "none")
    if reference_set == "none":
        return []
    references = CROSS_ASSET_SETS.get(reference_set)
    if references is None:
        raise ValueError(f"unsupported cross_asset_reference_set: {reference_set}")
    data_root = Path(str(config["data_root"]))
    timeframe = str(config["timeframe"])
    target = str(config["asset"]).lower().replace("_perp", "")
    files = []
    for asset in references:
        if asset == target:
            continue
        path = (
            data_root
            / "experiments"
            / "stage_a_screening"
            / "inputs"
            / asset
            / timeframe
            / "baseline_12"
            / "train.csv"
        )
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def required_source_files(config: dict[str, Any]) -> list[Path]:
    return sorted(set([*source_files(config), *cross_asset_files(config)]))


def _load_external_frame(
    path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lag_hours: float,
) -> pd.DataFrame:
    lag = pd.Timedelta(hours=float(lag_hours))
    try:
        source = pd.read_parquet(
            path,
            filters=[
                ("timestamp", ">=", (start - lag).to_pydatetime()),
                ("timestamp", "<=", (end - lag).to_pydatetime()),
            ],
        )
    except Exception:
        source = pd.read_parquet(path)
    if "timestamp" not in source:
        raise ValueError(f"missing timestamp: {path}")
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    if lag > pd.Timedelta(0):
        source["timestamp"] = source["timestamp"] + lag
    source = source.dropna(subset=["timestamp"]).sort_values("timestamp")
    source = source[(source["timestamp"] >= start) & (source["timestamp"] <= end)]
    numeric = [
        column
        for column in source.columns
        if column != "timestamp" and pd.api.types.is_numeric_dtype(source[column])
    ]
    if not numeric:
        return pd.DataFrame({"DATE_TIME": source["timestamp"]})
    prefix = path.stem.replace("__observations", "").replace("__daily", "")
    result_columns: dict[str, pd.Series] = {
        "DATE_TIME": source["timestamp"].reset_index(drop=True)
    }
    for column in numeric:
        values = pd.to_numeric(source[column], errors="coerce").reset_index(drop=True)
        name = f"external__{prefix}__{column}"
        result_columns[name] = values
        result_columns[f"{name}__change"] = values.diff()
    result = pd.DataFrame(result_columns)
    return result.drop_duplicates("DATE_TIME", keep="last")


def merge_external_context(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = source_files(config)
    if not files:
        return frame, {
            "external_source_count": 0,
            "external_feature_count": 0,
            "external_source_files": [],
            "external_coverage_fraction": 0.0,
        }
    timeframe_hours = _timeframe_hours(str(config["timeframe"]))
    lag_hours = float(config.get("external_context_lag_hours") or 0.0)
    start = frame["DATE_TIME"].min()
    end = frame["DATE_TIME"].max()
    before = set(frame.columns)
    base = frame.reset_index(drop=True)
    base_index = pd.DatetimeIndex(base["DATE_TIME"])
    aligned_contexts: list[pd.DataFrame] = []
    for path in files:
        context = _load_external_frame(path, start=start, end=end, lag_hours=lag_hours)
        if len(context.columns) <= 1:
            continue
        context = context.set_index("DATE_TIME")
        context = context[~context.index.duplicated(keep="last")]
        aligned = context.reindex(base_index)
        aligned.index = base.index
        aligned_contexts.append(aligned)
    if aligned_contexts:
        external = pd.concat(aligned_contexts, axis=1, copy=False)
        if external.columns.duplicated().any():
            external = external.loc[:, ~external.columns.duplicated(keep="last")]
        merged = pd.concat([base, external], axis=1, copy=False).copy()
        del external
    else:
        merged = base.copy()
    aligned_contexts.clear()
    gc.collect()
    external_columns = sorted(set(merged.columns) - before)
    if external_columns:
        missing_policy = str(config.get("missing_value_policy") or "causal_ffill")
        timeframe_hours = _timeframe_hours(str(config["timeframe"]))
        max_staleness_hours = config.get("max_staleness_hours")
        ffill_limit = None
        if max_staleness_hours is not None:
            ffill_limit = max(1, int(math.ceil(float(max_staleness_hours) / timeframe_hours)))
        if missing_policy in {"causal_ffill", "causal_ffill_plus_missing_indicator"}:
            missing_before = merged[external_columns].isna()
            merged[external_columns] = merged[external_columns].ffill(limit=ffill_limit)
            if missing_policy == "causal_ffill_plus_missing_indicator":
                indicators = missing_before.astype(float)
                indicators.columns = [
                    f"{column}__was_missing" for column in external_columns
                ]
                merged = pd.concat([merged, indicators], axis=1, copy=False).copy()
        elif missing_policy == "train_median":
            train_end = pd.Timestamp(str(config["train_end"]), tz="UTC")
            train_mask = merged["DATE_TIME"] <= train_end
            medians = merged.loc[train_mask, external_columns].median()
            merged[external_columns] = merged[external_columns].fillna(medians)
        else:
            raise ValueError(f"unsupported missing_value_policy: {missing_policy}")
        coverage = float(merged[external_columns].notna().mean().mean())
    else:
        coverage = 0.0
    return merged, {
        "external_source_count": len(files),
        "external_feature_count": len(external_columns),
        "external_source_files": [str(path) for path in files],
        "external_coverage_fraction": coverage,
        "external_context_lag_hours": lag_hours,
    }


def merge_cross_asset_context(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = cross_asset_files(config)
    if not files:
        return frame, {
            "cross_asset_source_count": 0,
            "cross_asset_feature_count": 0,
            "cross_asset_source_files": [],
            "cross_asset_coverage_fraction": 0.0,
        }
    merged = frame
    generated: list[str] = []
    timeframe_hours = _timeframe_hours(str(config["timeframe"]))
    volatility_window_hours = float(
        config.get("cross_asset_volatility_window_hours") or 24
    )
    volatility_rows = max(
        4,
        int(round(volatility_window_hours / timeframe_hours)),
    )
    for path in files:
        source = pd.read_csv(
            path,
            usecols=lambda column: column in {"DATE_TIME", "CLOSE", "VOLUME"},
        )
        source["DATE_TIME"] = pd.to_datetime(source["DATE_TIME"], utc=True, errors="coerce")
        source = source.dropna(subset=["DATE_TIME"]).sort_values("DATE_TIME")
        prefix = path.parents[2].name
        close = pd.to_numeric(source["CLOSE"], errors="coerce")
        context = pd.DataFrame(
            {
                "DATE_TIME": source["DATE_TIME"],
                f"cross_asset__{prefix}__return_1": close.pct_change(),
                f"cross_asset__{prefix}__log_return_1": np.log(close.clip(lower=1e-12)).diff(),
                (
                    f"cross_asset__{prefix}__volatility_"
                    f"{volatility_window_hours:g}h"
                ): close.pct_change().rolling(
                    volatility_rows,
                    min_periods=max(4, volatility_rows // 3),
                ).std(),
            }
        )
        if "VOLUME" in source:
            volume = pd.to_numeric(source["VOLUME"], errors="coerce")
            context[f"cross_asset__{prefix}__volume_change"] = volume.pct_change()
        generated.extend(column for column in context if column != "DATE_TIME")
        merged = merged.merge(
            context.drop_duplicates("DATE_TIME", keep="last"),
            on="DATE_TIME",
            how="left",
        )
    generated = sorted(set(generated))
    merged[generated] = merged[generated].ffill()
    return merged, {
        "cross_asset_source_count": len(files),
        "cross_asset_feature_count": len(generated),
        "cross_asset_source_files": [str(path) for path in files],
        "cross_asset_coverage_fraction": float(merged[generated].notna().mean().mean()),
        "cross_asset_volatility_window_hours": volatility_window_hours,
    }


def _transform_signal(
    frame: pd.DataFrame,
    name: str,
    *,
    timeframe_hours: float,
    volatility_window_hours: float,
    detrend_window_hours: float,
) -> pd.Series:
    close = pd.to_numeric(frame["CLOSE"], errors="coerce")
    if name == "close":
        return close
    if name == "log_close":
        return np.log(close.clip(lower=1e-12))
    if name == "return":
        return close.pct_change()
    if name == "log_return":
        return np.log(close.clip(lower=1e-12)).diff()
    if name == "volume":
        return pd.to_numeric(frame["VOLUME"], errors="coerce")
    if name == "volume_change":
        return pd.to_numeric(frame["VOLUME"], errors="coerce").pct_change()
    if name == "volatility":
        window_rows = max(4, int(round(volatility_window_hours / timeframe_hours)))
        return close.pct_change().rolling(
            window_rows,
            min_periods=max(4, window_rows // 3),
        ).std()
    if name == "spread_proxy":
        high = pd.to_numeric(frame["HIGH"], errors="coerce")
        low = pd.to_numeric(frame["LOW"], errors="coerce")
        return (high - low) / close.replace(0.0, np.nan)
    if name == "detrended_close":
        window_rows = max(4, int(round(detrend_window_hours / timeframe_hours)))
        return close - close.rolling(
            window_rows,
            min_periods=max(4, window_rows // 3),
        ).mean()
    raise ValueError(f"unsupported transform input signal: {name}")


def _wavelet_proxy_features(
    signal: pd.Series,
    *,
    signal_name: str,
    levels: list[int],
    timeframe_hours: float,
    base_scale_hours: float,
) -> pd.DataFrame:
    valid_levels = sorted({int(level) for level in levels if 1 <= int(level) <= 8})
    if not valid_levels:
        raise ValueError("wavelet_levels must contain at least one level in [1, 8]")
    prefix = f"transform__wavelet__{signal_name}"
    approximations: dict[int, pd.Series] = {}
    output: dict[str, pd.Series] = {}
    previous = signal
    energy_columns: list[str] = []
    for level in valid_levels:
        window = max(
            2,
            int(round(float(base_scale_hours) * (2**level) / timeframe_hours)),
        )
        approximation = signal.rolling(window, min_periods=window).mean()
        detail = previous - approximation
        approximations[level] = approximation
        output[f"{prefix}__approx_l{level}"] = approximation
        output[f"{prefix}__detail_l{level}"] = detail
        energy_name = f"{prefix}__energy_l{level}"
        output[energy_name] = detail.pow(2).rolling(
            max(
                4,
                int(
                    round(
                        float(base_scale_hours)
                        * (2 ** max(valid_levels))
                        / timeframe_hours
                    )
                ),
            ),
            min_periods=max(4, window // 3),
        ).mean()
        energy_columns.append(energy_name)
        previous = approximation
    result = pd.DataFrame(output, index=signal.index)
    total_energy = result[energy_columns].sum(axis=1).replace(0.0, np.nan)
    relative_columns = []
    for level, energy_name in zip(valid_levels, energy_columns):
        name = f"{prefix}__relative_energy_l{level}"
        result[name] = result[energy_name] / total_energy
        relative_columns.append(name)
    probabilities = result[relative_columns].clip(lower=0.0)
    result[f"{prefix}__entropy"] = -(
        probabilities * np.log(probabilities + 1e-12)
    ).sum(axis=1)
    return result


def _hilbert_features(
    signal: pd.Series,
    *,
    signal_name: str,
    window_rows: int,
    step: int,
) -> pd.DataFrame:
    values = signal.reset_index(drop=True)
    output = pd.DataFrame(
        {
            f"transform__hilbert__{signal_name}__amplitude": np.nan,
            f"transform__hilbert__{signal_name}__phase": np.nan,
            f"transform__hilbert__{signal_name}__instantaneous_frequency": np.nan,
        },
        index=signal.index,
    )
    for end in range(window_rows, len(values) + 1, max(1, step)):
        segment = values.iloc[end - window_rows : end].interpolate(
            limit_direction="both"
        ).to_numpy(dtype=float)
        if not np.isfinite(segment).all():
            continue
        analytic = hilbert(segment)
        phase = np.unwrap(np.angle(analytic))
        target_index = output.index[end - 1]
        output.at[
            target_index, f"transform__hilbert__{signal_name}__amplitude"
        ] = float(np.abs(analytic[-1]))
        output.at[
            target_index, f"transform__hilbert__{signal_name}__phase"
        ] = float(phase[-1])
        output.at[
            target_index,
            f"transform__hilbert__{signal_name}__instantaneous_frequency",
        ] = float(phase[-1] - phase[-2])
    return output.ffill()


def _multitaper_features(
    signal: pd.Series,
    *,
    signal_name: str,
    window_rows: int,
    step: int,
    time_bandwidth: float,
    taper_count: int,
) -> pd.DataFrame:
    values = signal.reset_index(drop=True)
    names = {
        "entropy": f"transform__multitaper__{signal_name}__spectral_entropy",
        "dominant": f"transform__multitaper__{signal_name}__dominant_frequency",
        "centroid": f"transform__multitaper__{signal_name}__spectral_centroid",
    }
    output = pd.DataFrame({name: np.nan for name in names.values()}, index=signal.index)
    tapers = dpss(
        window_rows,
        NW=float(time_bandwidth),
        Kmax=int(taper_count),
        sym=False,
    )
    for end in range(window_rows, len(values) + 1, max(1, step)):
        segment = values.iloc[end - window_rows : end].interpolate(
            limit_direction="both"
        ).to_numpy(dtype=float)
        if not np.isfinite(segment).all():
            continue
        centered = segment - float(np.mean(segment))
        spectrum = np.fft.rfft(tapers * centered, axis=1)
        power = np.mean(np.abs(spectrum) ** 2, axis=0)
        total = float(power.sum())
        if total <= 0.0 or not np.isfinite(total):
            continue
        frequencies = np.fft.rfftfreq(window_rows, d=1.0)
        normalized = power / total
        target_index = output.index[end - 1]
        output.at[target_index, names["entropy"]] = float(
            -(normalized * np.log(normalized + 1e-12)).sum()
        )
        output.at[target_index, names["dominant"]] = float(
            frequencies[int(np.argmax(power))]
        )
        output.at[target_index, names["centroid"]] = float(
            (frequencies * power).sum() / total
        )
    return output.ffill()


def _emd_proxy_features(
    signal: pd.Series,
    *,
    signal_name: str,
    window_rows: list[int],
) -> pd.DataFrame:
    if len(window_rows) != 3 or sorted(window_rows) != window_rows:
        raise ValueError("emd_window_hours must define three increasing windows")
    short, medium, long = window_rows
    mean_short = signal.rolling(short, min_periods=short).mean()
    mean_medium = signal.rolling(medium, min_periods=medium).mean()
    mean_long = signal.rolling(long, min_periods=long).mean()
    prefix = f"transform__emd_proxy__{signal_name}"
    return pd.DataFrame(
        {
            f"{prefix}__imf_1": signal - mean_short,
            f"{prefix}__imf_2": mean_short - mean_medium,
            f"{prefix}__imf_3": mean_medium - mean_long,
            f"{prefix}__residue": mean_long,
        },
        index=signal.index,
    )


def _fractional_difference(
    signal: pd.Series,
    *,
    signal_name: str,
    d: float,
    threshold: float = 1e-4,
    max_size: int = 256,
) -> pd.DataFrame:
    weights = [1.0]
    for k in range(1, max_size):
        weight = -weights[-1] * (float(d) - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
    values = signal.ffill().to_numpy(dtype=float)
    transformed = np.convolve(values, np.asarray(weights), mode="full")[: len(values)]
    transformed[: len(weights) - 1] = np.nan
    return pd.DataFrame(
        {
            f"transform__fracdiff__{signal_name}__d_{float(d):.2f}": transformed
        },
        index=signal.index,
    )


def add_configured_transform_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    generated: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {"configured_transform_families": []}
    timeframe = str(config["timeframe"])
    timeframe_hours = _timeframe_hours(timeframe)
    volatility_window_hours = float(
        config.get("transform_volatility_window_hours") or 24
    )
    detrend_window_hours = float(config.get("transform_detrend_window_hours") or 168)
    sample_step = max(
        1,
        int(
            round(
                float(config.get("transform_sample_interval_hours") or 24)
                / timeframe_hours
            )
        ),
    )

    def input_signal(name: str) -> pd.Series:
        return _transform_signal(
            frame,
            name,
            timeframe_hours=timeframe_hours,
            volatility_window_hours=volatility_window_hours,
            detrend_window_hours=detrend_window_hours,
        )

    wavelet_family = str(config.get("wavelet_family") or "off")
    if wavelet_family != "off":
        if wavelet_family != "causal_multiscale_rolling":
            raise ValueError(f"unsupported wavelet_family: {wavelet_family}")
        signal_name = str(config.get("transform_input_signal") or "close")
        levels = [int(value) for value in config.get("wavelet_levels") or [1, 2, 3, 4]]
        generated.append(
            _wavelet_proxy_features(
                input_signal(signal_name),
                signal_name=signal_name,
                levels=levels,
                timeframe_hours=timeframe_hours,
                base_scale_hours=float(config.get("wavelet_base_scale_hours") or 8),
            )
        )
        metadata["configured_transform_families"].append("wavelet")

    hilbert_signal = str(config.get("hilbert_input_signal") or "off")
    if hilbert_signal != "off":
        window_rows = max(
            32,
            int(round(float(config.get("hilbert_window_hours") or 168) / timeframe_hours)),
        )
        generated.append(
            _hilbert_features(
                input_signal(hilbert_signal),
                signal_name=hilbert_signal,
                window_rows=window_rows,
                step=sample_step,
            )
        )
        metadata["configured_transform_families"].append("hilbert")

    multitaper_signal = str(config.get("multitaper_input_signal") or "off")
    if multitaper_signal != "off":
        window_rows = max(
            32,
            int(
                round(
                    float(config.get("multitaper_window_hours") or 168)
                    / timeframe_hours
                )
            ),
        )
        generated.append(
            _multitaper_features(
                input_signal(multitaper_signal),
                signal_name=multitaper_signal,
                window_rows=window_rows,
                step=sample_step,
                time_bandwidth=float(
                    config.get("multitaper_time_bandwidth") or 3.5
                ),
                taper_count=int(config.get("multitaper_taper_count") or 5),
            )
        )
        metadata["configured_transform_families"].append("multitaper")

    emd_signal = str(config.get("emd_input_signal") or "off")
    if emd_signal != "off":
        backend = str(config.get("emd_backend") or "causal_rolling_proxy")
        if backend != "causal_rolling_proxy":
            raise ValueError(f"unsupported emd_backend: {backend}")
        generated.append(
            _emd_proxy_features(
                input_signal(emd_signal),
                signal_name=emd_signal,
                window_rows=[
                    max(2, int(round(float(hours) / timeframe_hours)))
                    for hours in config.get("emd_window_hours") or [8, 32, 128]
                ],
            )
        )
        metadata["configured_transform_families"].append("emd_proxy")

    fracdiff_signal = str(config.get("fracdiff_input_signal") or "off")
    if fracdiff_signal != "off":
        generated.append(
            _fractional_difference(
                input_signal(fracdiff_signal),
                signal_name=fracdiff_signal,
                d=float(config.get("fracdiff_d") or 0.4),
                threshold=float(
                    config.get("fracdiff_weight_threshold") or 0.0001
                ),
                max_size=max(
                    2,
                    int(
                        round(
                            float(
                                config.get("fracdiff_max_history_hours") or 720
                            )
                            / timeframe_hours
                        )
                    ),
                ),
            )
        )
        metadata["configured_transform_families"].append("fracdiff")

    if not generated:
        metadata["configured_transform_feature_count"] = 0
        return frame, metadata
    transformed = pd.concat([frame, *generated], axis=1)
    generated_columns = sorted(set(transformed.columns) - set(frame.columns))
    metadata["configured_transform_feature_count"] = len(generated_columns)
    metadata["configured_transform_features"] = generated_columns
    return transformed, metadata


def _causal_scale(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    mode: str,
    window_rows: int,
    clip: float | None,
) -> pd.DataFrame:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    if mode == "none":
        scaled = values.copy()
    elif mode == "rolling_zscore":
        min_periods = max(8, min(window_rows // 4, window_rows))
        center = values.rolling(window_rows, min_periods=min_periods).mean().shift(1)
        spread = values.rolling(window_rows, min_periods=min_periods).std().shift(1)
        scaled = (values - center) / spread.replace(0.0, np.nan)
    elif mode == "expanding_zscore":
        center = values.expanding(min_periods=16).mean().shift(1)
        spread = values.expanding(min_periods=16).std().shift(1)
        scaled = (values - center) / spread.replace(0.0, np.nan)
    elif mode == "rolling_robust":
        min_periods = max(8, min(window_rows // 4, window_rows))
        center = values.rolling(window_rows, min_periods=min_periods).median().shift(1)
        deviation = (values - center).abs()
        spread = deviation.rolling(window_rows, min_periods=min_periods).median().shift(1)
        scaled = (values - center) / (1.4826 * spread.replace(0.0, np.nan))
    elif mode == "rolling_rank_gaussian":
        min_periods = max(8, min(window_rows // 4, window_rows))
        percentile = values.rolling(
            window_rows,
            min_periods=min_periods,
        ).rank(pct=True).shift(1)
        clipped = percentile.clip(1e-4, 1.0 - 1e-4)
        scaled = pd.DataFrame(
            norm.ppf(clipped.to_numpy(dtype=float)),
            index=values.index,
            columns=values.columns,
        )
    else:
        raise ValueError(f"unsupported preprocessing mode: {mode}")
    if clip is not None and float(clip) > 0.0:
        scaled = scaled.clip(-float(clip), float(clip))
    return scaled.replace([np.inf, -np.inf], np.nan)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 40 or aligned.iloc[:, 0].nunique() < 2:
        return 0.0
    return _safe_float(spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1]).statistic)


def _select_features(
    train: pd.DataFrame,
    columns: list[str],
    target: str,
    *,
    method: str,
    budget: int,
    redundancy_threshold: float = 0.95,
    stability_folds: int = 5,
    timeframe_hours: float = 1.0,
    regime_volatility_window_hours: float = 24.0,
) -> tuple[list[str], dict[str, float]]:
    valid = [
        column
        for column in columns
        if train[column].notna().mean() >= 0.75 and train[column].nunique(dropna=True) > 2
    ]
    scores: dict[str, float] = {}
    if method == "rank_ic_topk":
        scores = {column: abs(_spearman(train[column], train[target])) for column in valid}
    elif method == "mutual_info_topk":
        sample = train[[*valid, target]].replace([np.inf, -np.inf], np.nan)
        medians = sample[valid].median()
        x = sample[valid].fillna(medians)
        y = sample[target].fillna(0.0)
        if len(x) > 5000:
            indices = np.linspace(0, len(x) - 1, 5000).astype(int)
            x = x.iloc[indices]
            y = y.iloc[indices]
        values = mutual_info_regression(x, y, random_state=1701, n_neighbors=5)
        scores = {column: _safe_float(value) for column, value in zip(valid, values)}
    elif method == "redundancy_stability_topk":
        folds = np.array_split(np.arange(len(train)), max(2, int(stability_folds)))
        for column in valid:
            fold_scores = [
                abs(_spearman(train.iloc[index][column], train.iloc[index][target]))
                for index in folds
                if len(index) >= 40
            ]
            scores[column] = (
                float(np.mean(fold_scores) - np.std(fold_scores))
                if fold_scores
                else 0.0
            )
    elif method == "regime_conditioned_topk":
        regime_rows = max(
            4,
            int(round(float(regime_volatility_window_hours) / timeframe_hours)),
        )
        target_volatility = train[target].rolling(
            regime_rows,
            min_periods=max(4, regime_rows // 3),
        ).std()
        try:
            regimes = pd.qcut(target_volatility.rank(method="first"), 3, labels=False)
        except ValueError:
            regimes = pd.Series(0, index=train.index)
        for column in valid:
            regime_scores = [
                abs(_spearman(train.loc[regimes == regime, column], train.loc[regimes == regime, target]))
                for regime in sorted(regimes.dropna().unique())
            ]
            scores[column] = float(np.mean(regime_scores)) if regime_scores else 0.0
    elif method == "sparse_mask":
        sample = train[[*valid, target]].replace([np.inf, -np.inf], np.nan)
        medians = sample[valid].median()
        x = sample[valid].fillna(medians)
        y = sample[target].fillna(0.0)
        if len(x) > 10_000:
            indices = np.linspace(0, len(x) - 1, 10_000).astype(int)
            x = x.iloc[indices]
            y = y.iloc[indices]
        means = x.mean()
        stds = x.std().replace(0.0, 1.0)
        model = Lasso(alpha=1e-4, max_iter=5000, random_state=1701)
        model.fit((x - means) / stds, y)
        scores = {
            column: abs(_safe_float(coefficient))
            for column, coefficient in zip(valid, model.coef_)
        }
    else:
        raise ValueError(f"unsupported feature selection method: {method}")
    ranked = sorted(valid, key=lambda column: (scores.get(column, 0.0), column), reverse=True)
    if method != "redundancy_stability_topk":
        return ranked[: max(1, int(budget))], scores
    selected: list[str] = []
    for column in ranked:
        if len(selected) >= max(1, int(budget)):
            break
        if not selected:
            selected.append(column)
            continue
        correlations = train[[column, *selected]].corr(method="spearman").loc[column, selected]
        if correlations.abs().max() < float(redundancy_threshold):
            selected.append(column)
    return selected, scores


def _context_features(
    scaled: pd.DataFrame,
    selected: list[str],
    *,
    representation: str,
    context_rows: int,
) -> pd.DataFrame:
    current = scaled[selected].copy()
    current.columns = [f"{column}__last" for column in selected]
    if representation == "last":
        return current
    if representation == "summary":
        rolling = scaled[selected].rolling(context_rows, min_periods=max(4, context_rows // 4))
        mean = rolling.mean().shift(1)
        mean.columns = [f"{column}__mean" for column in selected]
        std = rolling.std().shift(1)
        std.columns = [f"{column}__std" for column in selected]
        delta = scaled[selected] - scaled[selected].shift(context_rows)
        delta.columns = [f"{column}__delta" for column in selected]
        return pd.concat([current, mean, std, delta], axis=1)
    if representation == "sparse_lags":
        offsets = sorted({0, max(1, context_rows // 4), max(1, context_rows // 2), context_rows - 1})
        pieces = []
        for offset in offsets:
            lag = scaled[selected].shift(offset)
            lag.columns = [f"{column}__lag_{offset}" for column in selected]
            pieces.append(lag)
        return pd.concat(pieces, axis=1)
    if representation == "raw_sequence":
        offsets = np.unique(
            np.linspace(0, max(1, context_rows - 1), min(context_rows, 32)).astype(int)
        )
        pieces = []
        for offset in offsets:
            lag = scaled[selected].shift(int(offset))
            lag.columns = [f"{column}__lag_{int(offset)}" for column in selected]
            pieces.append(lag)
        return pd.concat(pieces, axis=1)
    if representation == "multiscale_sequence":
        scales = sorted(
            {
                max(2, context_rows // 16),
                max(2, context_rows // 8),
                max(2, context_rows // 4),
                max(2, context_rows // 2),
                context_rows,
            }
        )
        pieces = [current]
        for scale in scales:
            rolling = scaled[selected].rolling(
                scale,
                min_periods=min(scale, max(4, scale // 4)),
            )
            mean = rolling.mean().shift(1)
            mean.columns = [f"{column}__mean_{scale}" for column in selected]
            std = rolling.std().shift(1)
            std.columns = [f"{column}__std_{scale}" for column in selected]
            pieces.extend([mean, std])
        return pd.concat(pieces, axis=1)
    raise ValueError(f"unsupported context representation: {representation}")


def _target_series(
    frame: pd.DataFrame,
    *,
    horizon_rows: int,
    definition: str,
    transaction_cost: float,
    risk_penalty_lambda: float,
    timeframe_hours: float = 1.0,
    barrier_volatility_window_hours: float = 24.0,
) -> pd.Series:
    close = pd.to_numeric(frame["CLOSE"], errors="coerce")
    forward = close.shift(-horizon_rows) / close - 1.0
    if definition == "forward_return":
        return forward
    if definition == "cost_adjusted_forward_return":
        net_magnitude = (forward.abs() - 2.0 * float(transaction_cost)).clip(lower=0.0)
        return np.sign(forward) * net_magnitude
    high = pd.to_numeric(frame["HIGH"], errors="coerce")
    low = pd.to_numeric(frame["LOW"], errors="coerce")
    future_high = (
        high.shift(-1).iloc[::-1].rolling(horizon_rows, min_periods=horizon_rows).max().iloc[::-1]
    )
    future_low = (
        low.shift(-1).iloc[::-1].rolling(horizon_rows, min_periods=horizon_rows).min().iloc[::-1]
    )
    barrier_volatility_rows = max(
        horizon_rows,
        4,
        int(round(float(barrier_volatility_window_hours) / timeframe_hours)),
    )
    past_volatility = close.pct_change().rolling(
        barrier_volatility_rows,
        min_periods=max(4, barrier_volatility_rows // 3),
    ).std().shift(1)
    barrier = (past_volatility * math.sqrt(horizon_rows)).clip(lower=2.0 * transaction_cost)
    upper_excursion = future_high / close - 1.0
    lower_excursion = future_low / close - 1.0
    if definition == "triple_barrier":
        return pd.Series(
            np.where(
                upper_excursion >= barrier,
                1.0,
                np.where(lower_excursion <= -barrier, -1.0, np.sign(forward)),
            ),
            index=frame.index,
        )
    if definition == "future_rap":
        adverse = np.where(
            forward >= 0.0,
            np.maximum(0.0, -lower_excursion),
            np.maximum(0.0, upper_excursion),
        )
        return forward - np.sign(forward) * float(risk_penalty_lambda) * adverse
    raise ValueError(f"unsupported target_definition: {definition}")


def _fit_proxy_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    eval_x: pd.DataFrame,
    *,
    family: str,
    alpha: float,
    latent_dimension: int,
    random_seed: int,
    max_train_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    medians = train_x.median().fillna(0.0)
    x_train = train_x.fillna(medians).to_numpy(dtype=np.float64)
    x_eval = eval_x.fillna(medians).to_numpy(dtype=np.float64)
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales[scales < 1e-12] = 1.0
    x_train = (x_train - means) / scales
    x_eval = (x_eval - means) / scales
    y_train = train_y.to_numpy(dtype=float)
    if len(x_train) > max(1000, int(max_train_rows)):
        indices = np.linspace(
            0,
            len(x_train) - 1,
            max(1000, int(max_train_rows)),
        ).astype(int)
        x_fit = x_train[indices]
        y_fit = y_train[indices]
    else:
        x_fit = x_train
        y_fit = y_train
    model_family = str(family or "ridge")
    metadata: dict[str, float | int | str] = {
        "proxy_model_family": model_family,
        "proxy_fit_rows": int(len(x_fit)),
        "proxy_input_dimension": int(x_fit.shape[1]),
    }
    if model_family == "ridge":
        model = Ridge(alpha=float(alpha))
    elif model_family == "pca_ridge":
        components = max(
            1,
            min(int(latent_dimension), x_fit.shape[1], len(x_fit) - 1),
        )
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=int(random_seed),
        )
        x_fit = pca.fit_transform(x_fit)
        x_train = pca.transform(x_train)
        x_eval = pca.transform(x_eval)
        model = Ridge(alpha=float(alpha))
        metadata["proxy_latent_dimension"] = components
        metadata["proxy_explained_variance"] = float(
            pca.explained_variance_ratio_.sum()
        )
    elif model_family == "elastic_net":
        model = ElasticNet(
            alpha=max(1e-8, float(alpha)),
            l1_ratio=0.5,
            max_iter=5000,
            random_state=int(random_seed),
        )
    elif model_family == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=max(7, min(63, int(latent_dimension))),
            l2_regularization=max(0.0, float(alpha)),
            random_state=int(random_seed),
        )
    elif model_family == "mlp":
        width = max(8, int(latent_dimension))
        model = MLPRegressor(
            hidden_layer_sizes=(width, width),
            activation="relu",
            alpha=max(1e-8, float(alpha)),
            batch_size=min(512, max(32, len(x_fit) // 20)),
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            random_state=int(random_seed),
        )
    else:
        raise ValueError(f"unsupported proxy_model_family: {model_family}")
    model.fit(x_fit, y_fit)
    coefficients = getattr(model, "coef_", np.asarray([], dtype=float))
    metadata.update(
        {
            "ridge_alpha": float(alpha),
            "coefficient_l1": float(np.abs(coefficients).sum()),
            "coefficient_nonzero": int(
                (np.abs(coefficients) > 1e-12).sum()
            ),
        }
    )
    return model.predict(x_train), model.predict(x_eval), {
        **metadata,
    }


def _strategy_equity(
    timestamps: pd.Series,
    predictions: np.ndarray,
    forward_returns: pd.Series,
    *,
    threshold: float,
    transaction_cost: float,
    initial_equity: float = 1.0,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    position = np.where(predictions > threshold, 1.0, np.where(predictions < -threshold, -1.0, 0.0))
    turnover = np.abs(np.diff(position, prepend=0.0))
    net = position * forward_returns.to_numpy(dtype=float) - turnover * float(transaction_cost)
    equity = float(initial_equity) * np.cumprod(1.0 + np.clip(net, -0.999, None))
    return pd.Series(equity, index=timestamps.index), position, turnover


def _evaluate_split(
    split_name: str,
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    threshold: float,
    transaction_cost: float,
    risk_penalty_lambda: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    equity, position, turnover = _strategy_equity(
        frame["DATE_TIME"],
        predictions,
        frame["realized_return"],
        threshold=threshold,
        transaction_cost=transaction_cost,
    )
    # Add an initial timestamp one bar before the first evaluated return.
    step = frame["DATE_TIME"].iloc[1] - frame["DATE_TIME"].iloc[0] if len(frame) > 1 else pd.Timedelta(hours=1)
    canonical = canonical_trading_metrics(
        timestamps=pd.concat(
            [pd.Series([frame["DATE_TIME"].iloc[0] - step]), frame["DATE_TIME"].reset_index(drop=True)],
            ignore_index=True,
        ),
        equity=np.concatenate(([1.0], equity.to_numpy())),
        risk_penalty_lambda=risk_penalty_lambda,
    )
    rows = [{**row, "split": split_name} for row in canonical["metrics"]]
    ic = _spearman(pd.Series(predictions, index=frame.index), frame["target_return"])
    direction = float(np.mean(np.sign(predictions) == np.sign(frame["target_return"].to_numpy(dtype=float))))
    exposure = float(np.mean(position != 0.0))
    rows.extend(
        [
            _metric("prediction_rank_ic", ic, unit="dimensionless", horizon="target", aggregation="spearman", split=split_name),
            _metric("directional_accuracy", direction, unit="fraction", horizon="target", aggregation="mean", split=split_name),
            _metric("exposure", exposure, unit="fraction", horizon="evaluation_period", aggregation="mean", split=split_name),
            _metric("turnover_events", float(np.count_nonzero(turnover)), unit="count", horizon="evaluation_period", aggregation="count", split=split_name),
        ]
    )
    summary = {
        **canonical["canonical"],
        "prediction_rank_ic": ic,
        "directional_accuracy": direction,
        "exposure": exposure,
        "turnover_events": int(np.count_nonzero(turnover)),
    }
    return rows, summary


def data_contract_audit(config: dict[str, Any]) -> dict[str, Any]:
    frame = _load_base(config)
    enriched, source_meta = merge_external_context(frame, config)
    enriched, cross_asset_meta = merge_cross_asset_context(enriched, config)
    features = _feature_columns(enriched)
    start = enriched["DATE_TIME"].min()
    end = enriched["DATE_TIME"].max()
    duplicate_rows = int(enriched["DATE_TIME"].duplicated().sum())
    monotonic = bool(enriched["DATE_TIME"].is_monotonic_increasing)
    source_coverage = float(source_meta["external_coverage_fraction"])
    metric_rows = [
        _metric("row_count", float(len(enriched)), unit="count", horizon="dataset", aggregation="count", split="audit"),
        _metric("feature_count", float(len(features)), unit="count", horizon="dataset", aggregation="count", split="audit"),
        _metric("external_source_count", float(source_meta["external_source_count"]), unit="count", horizon="dataset", aggregation="count", split="audit"),
        _metric("external_feature_count", float(source_meta["external_feature_count"]), unit="count", horizon="dataset", aggregation="count", split="audit"),
        _metric("external_coverage", source_coverage, unit="fraction", horizon="dataset", aggregation="mean", split="audit"),
        _metric("cross_asset_source_count", float(cross_asset_meta["cross_asset_source_count"]), unit="count", horizon="dataset", aggregation="count", split="audit"),
        _metric("cross_asset_feature_count", float(cross_asset_meta["cross_asset_feature_count"]), unit="count", horizon="dataset", aggregation="count", split="audit"),
        _metric("cross_asset_coverage", float(cross_asset_meta["cross_asset_coverage_fraction"]), unit="fraction", horizon="dataset", aggregation="mean", split="audit"),
        _metric("duplicate_timestamp_rows", float(duplicate_rows), unit="count", horizon="dataset", aggregation="count", split="audit"),
    ]
    return {
        "task_type": "data_contract_audit",
        "metric_rows": metric_rows,
        "summary": {
            "rows": len(enriched),
            "feature_count": len(features),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timestamps_monotonic": monotonic,
            "duplicate_timestamp_rows": duplicate_rows,
            **source_meta,
            **cross_asset_meta,
            "cryptoquant_excluded": True,
        },
        "artifacts": [
            {
                "artifact_type": "input_dataset",
                "path": str(config["input_data_file"]),
                "sha256": _sha256(Path(str(config["input_data_file"]))),
                "size_bytes": Path(str(config["input_data_file"])).stat().st_size,
            }
        ],
    }


def feature_proxy_screen(config: dict[str, Any]) -> dict[str, Any]:
    upstream_protocol_id = config.get("upstream_evaluation_protocol_id")
    upstream_protocol_hash = config.get("upstream_evaluation_protocol_hash")
    if upstream_protocol_id and upstream_protocol_id != FEATURE_PROXY_PROTOCOL:
        raise ValueError(
            "upstream evaluation protocol does not match current executor: "
            f"{upstream_protocol_id} != {FEATURE_PROXY_PROTOCOL}"
        )
    if upstream_protocol_hash and upstream_protocol_hash != FEATURE_PROXY_PROTOCOL_HASH:
        raise ValueError(
            "upstream evaluation protocol hash does not match current executor"
        )
    frame = _load_base(config)
    frame, source_meta = merge_external_context(frame, config)
    frame, cross_asset_meta = merge_cross_asset_context(frame, config)
    frame, transform_meta = add_configured_transform_features(frame, config)
    features = _feature_columns(frame)
    if not features:
        path = Path(str(config["input_data_file"]))
        return {
            "task_type": "feature_proxy_screen",
            "evaluation_protocol_id": FEATURE_PROXY_PROTOCOL,
            "evaluation_protocol_hash": FEATURE_PROXY_PROTOCOL_HASH,
            "metric_rows": [
                _metric(
                    "selected_feature_count",
                    0.0,
                    unit="count",
                    horizon="dataset",
                    aggregation="count",
                    split="train",
                )
            ],
            "summary": {
                "screen_status": "blocked_no_numeric_nonleaking_features",
                "selected_features": [],
                "selection_uses_validation": False,
                "selection_uses_test": False,
                **source_meta,
                **cross_asset_meta,
                **transform_meta,
                "cryptoquant_excluded": True,
            },
            "artifacts": [
                {
                    "artifact_type": "input_dataset",
                    "path": str(path),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                    "metadata": {
                        "screen_status": "blocked_no_numeric_nonleaking_features"
                    },
                }
            ],
        }
    timeframe_hours = _timeframe_hours(str(config["timeframe"]))
    horizon_rows = max(1, int(round(float(config["target_horizon_hours"]) / timeframe_hours)))
    transaction_cost = float(config.get("transaction_cost_fraction") or 0.0005)
    risk_lambda = float(config.get("risk_penalty_lambda") or 1.0)
    frame["target_return"] = _target_series(
        frame,
        horizon_rows=horizon_rows,
        definition=str(config.get("target_definition") or "forward_return"),
        transaction_cost=transaction_cost,
        risk_penalty_lambda=risk_lambda,
        timeframe_hours=timeframe_hours,
        barrier_volatility_window_hours=float(
            config.get("target_barrier_volatility_window_hours") or 24
        ),
    )
    close = pd.to_numeric(frame["CLOSE"], errors="coerce")
    frame["realized_return"] = close.shift(-1) / close - 1.0
    frame = frame.replace([np.inf, -np.inf], np.nan)

    train_start = pd.Timestamp(str(config["train_start"]), tz="UTC")
    train_end = pd.Timestamp(str(config["train_end"]), tz="UTC")
    validation_start = pd.Timestamp(str(config["validation_start"]), tz="UTC")
    validation_end = pd.Timestamp(str(config["validation_end"]), tz="UTC")
    test_start = pd.Timestamp(str(config["test_start"]), tz="UTC")
    test_end = pd.Timestamp(str(config["test_end"]), tz="UTC")
    in_scope = frame[
        (frame["DATE_TIME"] >= train_start)
        & (frame["DATE_TIME"] <= test_end)
    ].copy()
    del frame
    gc.collect()
    if in_scope.empty:
        raise ValueError("dataset has no rows in configured experiment period")

    scaling_rows = max(
        2,
        int(round(float(config["scaling_history_hours"]) / timeframe_hours)),
    )
    feature_frame = in_scope[features].copy()
    if bool(config.get("log_transform_positive_features", False)):
        for column in features:
            values = pd.to_numeric(feature_frame[column], errors="coerce")
            if values.dropna().empty or float(values.min(skipna=True)) < 0.0:
                continue
            feature_frame[column] = np.log1p(values)
    target = in_scope["target_return"]
    train_mask = _purged_split_mask(
        in_scope["DATE_TIME"],
        start=train_start,
        end=train_end,
        target_horizon_hours=float(config["target_horizon_hours"]),
    )
    validation_mask = _purged_split_mask(
        in_scope["DATE_TIME"],
        start=validation_start,
        end=validation_end,
        target_horizon_hours=float(config["target_horizon_hours"]),
    )
    test_mask = _purged_split_mask(
        in_scope["DATE_TIME"],
        start=test_start,
        end=test_end,
        target_horizon_hours=float(config["target_horizon_hours"]),
    )
    selection_scaled = _causal_scale(
        feature_frame.loc[train_mask],
        features,
        mode=str(config["preprocessing_mode"]),
        window_rows=scaling_rows,
        clip=config.get("clip_value"),
    )
    selection_frame = selection_scaled.copy()
    selection_frame["target_return"] = target.loc[train_mask]
    upstream_selected = [
        str(column) for column in config.get("upstream_selected_features") or []
    ]
    if upstream_selected:
        missing = sorted(set(upstream_selected) - set(features))
        if missing:
            raise ValueError(
                "upstream_selected_features missing from materialized input: "
                + ", ".join(missing)
            )
        selected = upstream_selected
        feature_scores = {
            name: _spearman(selection_frame[name], selection_frame["target_return"])
            for name in selected
        }
        selection_source = "frozen_upstream_contract"
    else:
        selected, feature_scores = _select_features(
            selection_frame,
            features,
            "target_return",
            method=str(config["feature_selection_method"]),
            budget=int(config["feature_budget"]),
            redundancy_threshold=float(config.get("redundancy_threshold") or 0.95),
            stability_folds=int(config.get("stability_folds") or 5),
            timeframe_hours=timeframe_hours,
            regime_volatility_window_hours=float(
                config.get("selection_regime_volatility_window_hours") or 24
            ),
        )
        selection_source = "train_only_selector"
    scaled = _causal_scale(
        feature_frame[selected],
        selected,
        mode=str(config["preprocessing_mode"]),
        window_rows=scaling_rows,
        clip=config.get("clip_value"),
    )
    del selection_scaled
    del feature_frame
    del selection_frame
    gc.collect()
    context_rows = max(2, int(round(float(config["context_hours"]) / timeframe_hours)))
    observations = _context_features(
        scaled,
        selected,
        representation=str(config["context_representation"]),
        context_rows=context_rows,
    )
    model_frame = pd.concat(
        [in_scope[["DATE_TIME", "target_return", "realized_return"]], observations],
        axis=1,
    ).dropna(subset=["target_return", "realized_return"])
    observation_columns = list(observations.columns)
    train = model_frame.loc[train_mask.reindex(model_frame.index, fill_value=False)].copy()
    validation = model_frame.loc[validation_mask.reindex(model_frame.index, fill_value=False)].copy()
    test = model_frame.loc[test_mask.reindex(model_frame.index, fill_value=False)].copy()
    minimum_rows = int(config.get("minimum_split_rows") or 200)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        if len(split) < minimum_rows:
            raise ValueError(f"{name} has {len(split)} rows; need {minimum_rows}")

    model_kwargs = {
        "family": str(config.get("proxy_model_family") or "ridge"),
        "alpha": float(config.get("ridge_alpha") or 1.0),
        "latent_dimension": int(config.get("proxy_latent_dimension") or 32),
        "random_seed": int(config.get("proxy_random_seed") or 1701),
        "max_train_rows": int(config.get("proxy_max_train_rows") or 25000),
    }
    train_pred, validation_pred, model_meta = _fit_proxy_model(
        train[observation_columns],
        train["target_return"],
        validation[observation_columns],
        **model_kwargs,
    )
    _, test_pred, _ = _fit_proxy_model(
        train[observation_columns],
        train["target_return"],
        test[observation_columns],
        **model_kwargs,
    )
    threshold_quantile = float(config.get("action_threshold_quantile") or 0.65)
    threshold = float(np.quantile(np.abs(train_pred), threshold_quantile))
    validation_rows, validation_summary = _evaluate_split(
        "validation",
        validation,
        validation_pred,
        threshold=threshold,
        transaction_cost=transaction_cost,
        risk_penalty_lambda=risk_lambda,
    )
    test_rows, test_summary = _evaluate_split(
        "test",
        test,
        test_pred,
        threshold=threshold,
        transaction_cost=transaction_cost,
        risk_penalty_lambda=risk_lambda,
    )
    optimization_score = _safe_float(validation_summary.get("annual_rap"))
    metric_rows = [
        *validation_rows,
        *test_rows,
        _metric(
            "optimization_score",
            optimization_score,
            unit="dimensionless",
            horizon="experiment",
            aggregation="selection_objective",
            split="validation",
        ),
        _metric("selected_feature_count", float(len(selected)), unit="count", horizon="dataset", aggregation="count", split="train"),
        _metric("observation_dimension", float(len(observation_columns)), unit="count", horizon="observation", aggregation="count", split="train"),
        _metric("external_coverage", float(source_meta["external_coverage_fraction"]), unit="fraction", horizon="dataset", aggregation="mean", split="train"),
        _metric("cross_asset_coverage", float(cross_asset_meta["cross_asset_coverage_fraction"]), unit="fraction", horizon="dataset", aggregation="mean", split="train"),
    ]
    return {
        "task_type": "feature_proxy_screen",
        "evaluation_protocol_id": FEATURE_PROXY_PROTOCOL,
        "evaluation_protocol_hash": FEATURE_PROXY_PROTOCOL_HASH,
        "metric_rows": metric_rows,
        "summary": {
            "evaluation_protocol_id": FEATURE_PROXY_PROTOCOL,
            "evaluation_protocol_hash": FEATURE_PROXY_PROTOCOL_HASH,
            "optimization_score_dimensionless": optimization_score,
            "validation": validation_summary,
            "test": test_summary,
            "selected_features": selected,
            "selected_feature_scores": {name: feature_scores[name] for name in selected},
            "selection_source": selection_source,
            "observation_columns": observation_columns,
            "action_threshold": threshold,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            **model_meta,
            **source_meta,
            **cross_asset_meta,
            **transform_meta,
            "cryptoquant_excluded": True,
            "selection_uses_validation": False,
            "selection_uses_test": False,
        },
        "artifacts": [
            {
                "artifact_type": "input_dataset",
                "path": str(config["input_data_file"]),
                "sha256": _sha256(Path(str(config["input_data_file"]))),
                "size_bytes": Path(str(config["input_data_file"])).stat().st_size,
            }
        ],
    }


def execute(task_type: str, config: dict[str, Any]) -> dict[str, Any]:
    resolved = {**SCREEN_DEFAULTS, **config}
    if task_type == "data_contract_audit":
        result = data_contract_audit(resolved)
    elif task_type == "feature_proxy_screen":
        result = feature_proxy_screen(resolved)
    else:
        raise ValueError(f"unsupported task_type: {task_type}")
    result["resolved_parameters"] = {
        key: resolved.get(key)
        for key in RESOLVED_PARAMETER_KEYS
        if key in resolved
    }
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = execute(args.task_type, config)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": args.output}, sort_keys=True))


if __name__ == "__main__":
    main()
