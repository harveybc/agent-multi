#!/usr/bin/env python3
"""Leakage-aware CPU evidence screens for Project 3 data contracts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso, Ridge

from project3_evidence_metrics import METRIC_SCHEMA, canonical_trading_metrics


CORE_COLUMNS = {"DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"}
LEAK_TOKENS = ("target", "label", "future", "fwd_", "next_", "prediction")
FEATURE_PROXY_PROTOCOL = "project3.feature_proxy.one_bar_execution.v2"
FEATURE_PROXY_PROTOCOL_HASH = hashlib.sha256(
    (
        "target may span configured horizon; positions are recomputed per bar; "
        "equity uses next-bar realized return exactly once; transaction cost "
        "is charged on absolute position turnover; train-only feature selection"
    ).encode("utf-8")
).hexdigest()

SCREEN_DEFAULTS = {
    "external_context_bundle": "none",
    "external_context_lag_hours": 0,
    "missing_value_policy": "causal_ffill",
    "max_staleness_hours": None,
    "cross_asset_reference_set": "none",
    "target_definition": "forward_return",
    "feature_selection_method": "rank_ic_topk",
    "feature_budget": 32,
    "redundancy_threshold": 0.95,
    "stability_folds": 5,
    "preprocessing_mode": "rolling_zscore",
    "scaling_history_hours": 168,
    "clip_value": 10,
    "log_transform_positive_features": False,
    "context_hours": 168,
    "context_representation": "summary",
    "ridge_alpha": 1.0,
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
    "target_horizon_hours",
    "target_definition",
    "feature_selection_method",
    "feature_budget",
    "redundancy_threshold",
    "stability_folds",
    "preprocessing_mode",
    "scaling_history_hours",
    "clip_value",
    "log_transform_positive_features",
    "context_hours",
    "context_representation",
    "ridge_alpha",
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
    patterns: list[str] = []
    if bundle in {"macro_core", "macro_market", "all_non_cryptoquant"}:
        patterns.extend(SOURCE_PATTERNS["macro_core"])
    if bundle in {"market_core", "macro_market", "crypto_context", "all_non_cryptoquant"}:
        patterns.extend(SOURCE_PATTERNS["market_core"])
    if bundle in {"onchain_asset", "crypto_context", "all_non_cryptoquant"}:
        patterns.extend(_onchain_patterns(str(config["asset"])))
    if bundle in {"funding_core", "crypto_context", "all_non_cryptoquant"}:
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
    lag_rows: int,
) -> pd.DataFrame:
    try:
        source = pd.read_parquet(
            path,
            filters=[
                ("timestamp", ">=", start.to_pydatetime()),
                ("timestamp", "<=", end.to_pydatetime()),
            ],
        )
    except Exception:
        source = pd.read_parquet(path)
    if "timestamp" not in source:
        raise ValueError(f"missing timestamp: {path}")
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
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
    result = pd.DataFrame({"DATE_TIME": source["timestamp"]})
    for column in numeric:
        values = pd.to_numeric(source[column], errors="coerce")
        if lag_rows > 0:
            values = values.shift(lag_rows)
        name = f"external__{prefix}__{column}"
        result[name] = values
        result[f"{name}__change"] = values.diff()
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
    lag_rows = int(math.ceil(lag_hours / timeframe_hours))
    start = frame["DATE_TIME"].min()
    end = frame["DATE_TIME"].max()
    before = set(frame.columns)
    merged = frame
    for path in files:
        context = _load_external_frame(path, start=start, end=end, lag_rows=lag_rows)
        if len(context.columns) <= 1:
            continue
        merged = merged.merge(context, on="DATE_TIME", how="left")
    merged = merged.copy()
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
                for column in external_columns:
                    merged[f"{column}__was_missing"] = missing_before[column].astype(float)
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
                f"cross_asset__{prefix}__volatility_24": close.pct_change().rolling(24).std(),
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
    }


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
        target_volatility = train[target].rolling(24, min_periods=8).std()
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
    past_volatility = close.pct_change().rolling(max(24, horizon_rows), min_periods=8).std().shift(1)
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


def _fit_ridge(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    eval_x: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    medians = train_x.median().fillna(0.0)
    x_train = train_x.fillna(medians).to_numpy(dtype=np.float64)
    x_eval = eval_x.fillna(medians).to_numpy(dtype=np.float64)
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales[scales < 1e-12] = 1.0
    x_train = (x_train - means) / scales
    x_eval = (x_eval - means) / scales
    model = Ridge(alpha=float(alpha))
    model.fit(x_train, train_y.to_numpy(dtype=float))
    return model.predict(x_train), model.predict(x_eval), {
        "ridge_alpha": float(alpha),
        "coefficient_l1": float(np.abs(model.coef_).sum()),
        "coefficient_nonzero": int((np.abs(model.coef_) > 1e-12).sum()),
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
    frame = _load_base(config)
    frame, source_meta = merge_external_context(frame, config)
    frame, cross_asset_meta = merge_cross_asset_context(frame, config)
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
    scaled = _causal_scale(
        feature_frame,
        features,
        mode=str(config["preprocessing_mode"]),
        window_rows=scaling_rows,
        clip=config.get("clip_value"),
    )
    target = in_scope["target_return"]
    train_mask = (in_scope["DATE_TIME"] >= train_start) & (in_scope["DATE_TIME"] <= train_end)
    validation_mask = (in_scope["DATE_TIME"] >= validation_start) & (in_scope["DATE_TIME"] <= validation_end)
    test_mask = (in_scope["DATE_TIME"] >= test_start) & (in_scope["DATE_TIME"] <= test_end)
    selection_frame = scaled.loc[train_mask].copy()
    selection_frame["target_return"] = target.loc[train_mask]
    selected, feature_scores = _select_features(
        selection_frame,
        features,
        "target_return",
        method=str(config["feature_selection_method"]),
        budget=int(config["feature_budget"]),
        redundancy_threshold=float(config.get("redundancy_threshold") or 0.95),
        stability_folds=int(config.get("stability_folds") or 5),
    )
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

    train_pred, validation_pred, model_meta = _fit_ridge(
        train[observation_columns],
        train["target_return"],
        validation[observation_columns],
        alpha=float(config.get("ridge_alpha") or 1.0),
    )
    _, test_pred, _ = _fit_ridge(
        train[observation_columns],
        train["target_return"],
        test[observation_columns],
        alpha=float(config.get("ridge_alpha") or 1.0),
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
            "observation_columns": observation_columns,
            "action_threshold": threshold,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            **model_meta,
            **source_meta,
            **cross_asset_meta,
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
