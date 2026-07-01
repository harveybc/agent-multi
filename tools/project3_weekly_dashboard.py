#!/usr/bin/env python3
"""AdminLTE dashboard for the Project 3 weekly walk-forward pool."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ORACLE_DEFAULT_FULL_SPREAD = 0.0004
ORACLE_COST_STRESS_MULTIPLIER = 2.0
CHART_FOCUS_START_LABEL = "2026-06-09 12:00:00Z"
CHART_MAX_POINTS = 1200

_BASELINE_CACHE: dict[tuple, dict[str, float | None]] = {}
_PRICE_SERIES_CACHE: dict[tuple, list[tuple[datetime, float, float, float]]] = {}
_RESULT_FILE_CACHE: dict[tuple[str, int], dict] = {}

CDT_ANNUAL_BASELINE_RETURN = 0.12
CDT_ANNUAL_STRETCH_RETURN = 0.126
CDT_WEEKLY_BASELINE_RETURN = (1.0 + CDT_ANNUAL_BASELINE_RETURN) ** (1.0 / 52.0) - 1.0
CDT_WEEKLY_STRETCH_RETURN = (1.0 + CDT_ANNUAL_STRETCH_RETURN) ** (1.0 / 52.0) - 1.0


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(row) for row in conn.execute(sql, params)]


def _progress_for_run(run_dir: str | None) -> dict:
    if not run_dir:
        return {}
    path = Path(run_dir) / "training_progress.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_utc(value: object) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h {minutes % 60}m"


def _heartbeat_freshness(age_seconds: float | None, status: str | None) -> str:
    if status == "stale":
        return "stale"
    if age_seconds is None:
        return "unknown"
    if age_seconds >= 600:
        return "stale"
    if age_seconds >= 180:
        return "warn"
    return "fresh"


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_or_empty(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_result_file_from_summary(summary: dict) -> dict:
    results_file = summary.get("results_file")
    if not results_file:
        return {}
    path = Path(str(results_file))
    if not path.exists():
        return {}
    try:
        stat = path.stat()
    except OSError:
        return {}
    cache_key = (str(path), int(stat.st_mtime_ns))
    cached = _RESULT_FILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload = {"splits": payload.get("splits") if isinstance(payload.get("splits"), dict) else {}}
    _RESULT_FILE_CACHE[cache_key] = payload
    if len(_RESULT_FILE_CACHE) > 1024:
        for old_key in list(_RESULT_FILE_CACHE)[:256]:
            _RESULT_FILE_CACHE.pop(old_key, None)
    return payload


def _split_summary_metric(
    summary: dict,
    raw_result: dict,
    *,
    split: str,
    summary_key: str,
    metric_key: str,
) -> float | None:
    value = _float_or_none(summary.get(summary_key))
    if value is not None:
        return value
    split_summary = ((raw_result.get("splits") or {}).get(split) or {})
    if isinstance(split_summary, dict):
        return _float_or_none(split_summary.get(metric_key))
    return None


def _split_drawdown_fraction(
    summary: dict,
    raw_result: dict,
    *,
    split: str,
    summary_prefix: str,
) -> float | None:
    value = _float_or_none(summary.get(f"{summary_prefix}_max_drawdown_fraction"))
    if value is not None:
        return max(0.0, value)
    pct = _float_or_none(summary.get(f"{summary_prefix}_max_drawdown_pct"))
    if pct is None:
        split_summary = ((raw_result.get("splits") or {}).get(split) or {})
        if isinstance(split_summary, dict):
            pct = _float_or_none(split_summary.get("max_drawdown_pct"))
            if pct is None:
                fraction = _float_or_none(split_summary.get("max_drawdown"))
                if fraction is not None:
                    return abs(fraction)
    if pct is not None:
        return max(0.0, pct / 100.0)
    return None


def _risk_adjusted_return_value(total_return: float | None, drawdown: float | None, risk_lambda: float) -> float | None:
    if total_return is None:
        return None
    return float(total_return) - float(risk_lambda) * float(drawdown or 0.0)


def _drawdown_fraction_from_result(result: dict, prefix: str) -> float | None:
    value = _float_or_none(result.get(f"{prefix}_max_drawdown_fraction"))
    if value is not None:
        return value
    pct = _float_or_none(result.get(f"{prefix}_max_drawdown_pct"))
    if pct is not None:
        return max(0.0, pct / 100.0)
    return None


def _fmt_metric(value: object) -> str:
    metric = _float_or_none(value)
    if metric is None:
        return ""
    return f"{metric:+.6f}"


def _fmt_percent(value: object, digits: int = 3) -> str:
    metric = _float_or_none(value)
    if metric is None:
        return ""
    return f"{100.0 * metric:+.{digits}f}%"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _config_costs(config_path: str | None) -> tuple[float, float, float, int]:
    commission = 0.0002
    slippage = 0.0
    full_spread = ORACLE_DEFAULT_FULL_SPREAD
    tail_days = 7
    if not config_path:
        return commission, slippage, full_spread, tail_days
    path = Path(config_path)
    if not path.exists():
        return commission, slippage, full_spread, tail_days
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return commission, slippage, full_spread, tail_days
    commission = _float_or_none(config.get("commission")) or commission
    slippage = _float_or_none(config.get("slippage")) or slippage
    full_spread = (
        _float_or_none(
            config.get("oracle_full_spread")
            or config.get("full_spread")
            or config.get("spread")
            or config.get("spread_perc")
            or config.get("bid_ask_spread")
        )
        or full_spread
    )
    try:
        tail_days = int(config.get("early_stop_train_tail_days") or tail_days)
    except (TypeError, ValueError):
        tail_days = 7
    return commission, slippage, full_spread, max(1, tail_days)


def _oracle_execution_config(config_path: str | None) -> dict[str, float | str | None]:
    config: dict = {}
    if config_path:
        path = Path(config_path)
        if path.exists():
            try:
                config = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
    commission, slippage, full_spread, tail_days = _config_costs(config_path)
    return {
        "initial_cash": _float_or_none(config.get("initial_cash")) or 10000.0,
        "position_size": _float_or_none(config.get("position_size")) or 1.0,
        "rel_volume": _float_or_none(config.get("rel_volume")),
        "leverage": _float_or_none(config.get("leverage")) or 1.0,
        "min_order_volume": _float_or_none(config.get("min_order_volume")) or 0.0,
        "max_order_volume": _float_or_none(config.get("max_order_volume")) or 1e12,
        "size_mode": str(config.get("size_mode") or "fx_units").lower(),
        "commission": commission,
        "slippage": slippage,
        "full_spread": full_spread,
        "tail_days": float(tail_days),
    }


def _oracle_order_size(cash: float, fill_price: float, execution: dict[str, float | str | None]) -> float:
    rel_volume = execution.get("rel_volume")
    if rel_volume is None:
        return max(0.0, float(execution["position_size"] or 0.0))
    leverage = float(execution["leverage"] or 1.0)
    raw: float
    if str(execution.get("size_mode") or "fx_units").lower() == "notional":
        raw = cash * float(rel_volume) * leverage / fill_price if fill_price > 0 else 0.0
    else:
        raw = cash * float(rel_volume) * leverage
    min_volume = float(execution["min_order_volume"] or 0.0)
    max_volume = float(execution["max_order_volume"] or 1e12)
    return max(min_volume, min(raw, max_volume))


def _oracle_fill_prices(
    entry_price: float,
    exit_price: float,
    direction: str,
    price_cost_rate: float,
) -> tuple[float, float]:
    if direction == "long":
        return entry_price * (1.0 + price_cost_rate), exit_price * (1.0 - price_cost_rate)
    return entry_price * (1.0 - price_cost_rate), exit_price * (1.0 + price_cost_rate)


def _apply_oracle_trade(
    cash: float,
    entry_price: float,
    exit_price: float,
    direction: str,
    execution: dict[str, float | str | None],
    commission_rate: float,
    price_cost_rate: float,
) -> tuple[float, bool]:
    entry_fill, exit_fill = _oracle_fill_prices(entry_price, exit_price, direction, price_cost_rate)
    if entry_fill <= 0 or exit_fill <= 0:
        return cash, False
    size = _oracle_order_size(cash, entry_fill, execution)
    if size <= 0:
        return cash, False
    entry_notional = size * entry_fill
    exit_notional = size * exit_fill
    if direction == "long":
        pnl = size * (exit_fill - entry_fill)
    else:
        pnl = size * (entry_fill - exit_fill)
    cost = commission_rate * (entry_notional + exit_notional)
    return cash + pnl - cost, True


def _price_series(data_path: Path, data_stamp: tuple[int, int]) -> list[tuple[datetime, float, float, float]]:
    cache_key = (str(data_path), data_stamp)
    cached = _PRICE_SERIES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    series: list[tuple[datetime, float, float, float]] = []
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stamp = _parse_datetime(row.get("DATE_TIME"))
            close = _float_or_none(row.get("CLOSE"))
            high = _float_or_none(row.get("HIGH")) or close
            low = _float_or_none(row.get("LOW")) or close
            if stamp is None or close is None or close <= 0:
                continue
            if high is None or low is None or high <= 0 or low <= 0:
                continue
            series.append((stamp, max(high, low, close), min(high, low, close), close))
    _PRICE_SERIES_CACHE[cache_key] = series
    return series


def _minimum_zigzag_bars(series: list[tuple[datetime, float, float, float]]) -> int:
    deltas = [
        (next_stamp - previous_stamp).total_seconds()
        for (previous_stamp, *_), (next_stamp, *_) in zip(series, series[1:])
        if next_stamp > previous_stamp
    ]
    median_delta = _median(deltas)
    if not median_delta or median_delta <= 0:
        return 2
    return max(1, round((12 * 60 * 60) / median_delta))


def _zigzag_pivots(
    bars: list[tuple[datetime, float, float, float]],
    reversal_threshold: float,
    depth_bars: int,
    backstep_bars: int,
) -> list[tuple[int, str, float]]:
    if len(bars) < 2:
        return []

    highs = [high for _, high, _, _ in bars]
    lows = [low for _, _, low, _ in bars]
    candidates: list[tuple[int, str, float]] = []
    for idx, (_, high, low, _) in enumerate(bars):
        left = max(0, idx - depth_bars)
        right = min(len(bars), idx + depth_bars + 1)
        if low <= min(lows[left:right]):
            candidates.append((idx, "valley", low))
        if high >= max(highs[left:right]):
            candidates.append((idx, "peak", high))

    pivots: list[tuple[int, str, float]] = []
    for candidate in sorted(candidates, key=lambda item: (item[0], 0 if item[1] == "valley" else 1)):
        idx, kind, price = candidate
        if not pivots:
            pivots.append(candidate)
            continue
        last_idx, last_kind, last_price = pivots[-1]
        if kind == last_kind:
            replace = (kind == "valley" and price < last_price) or (kind == "peak" and price > last_price)
            if replace:
                pivots[-1] = candidate
            continue
        if idx - last_idx < backstep_bars:
            continue
        move = abs(price - last_price) / last_price if last_price > 0 else 0.0
        if move >= reversal_threshold:
            pivots.append(candidate)
    return pivots


def _zigzag_threshold_candidates(
    bars: list[tuple[datetime, float, float, float]],
    per_side_cost: float,
) -> list[float]:
    closes = [close for _, _, _, close in bars]
    intrabar_ranges = [
        (high - low) / close
        for _, high, low, close in bars
        if close > 0 and high >= low
    ]
    abs_returns = [
        abs(next_close - previous_close) / previous_close
        for previous_close, next_close in zip(closes, closes[1:])
        if previous_close > 0 and next_close > 0
    ]
    median_abs_return = _median(abs_returns) or 0.0
    median_intrabar_range = _median(intrabar_ranges) or 0.0
    price_range = (max(closes) - min(closes)) / closes[0] if closes and closes[0] > 0 else 0.0
    round_trip_cost = 2.0 * per_side_cost
    base_threshold = max(2.0 * round_trip_cost, 3.0 * median_abs_return, median_intrabar_range, 0.0015)
    grid = {
        base_threshold * multiplier
        for multiplier in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    }
    grid.update({round_trip_cost * multiplier for multiplier in (4.0, 6.0, 8.0, 12.0)})
    grid.update({0.0025, 0.0035, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05})
    upper_bound = max(0.001, min(0.20, max(price_range, base_threshold) * 0.9))
    return sorted(threshold for threshold in grid if 0 < threshold <= upper_bound)


def _score_zigzag_oracle(
    bars: list[tuple[datetime, float, float, float]],
    reversal_threshold: float,
    execution: dict[str, float | str | None],
    depth_bars: int,
    backstep_bars: int,
    min_profit_multiple: float,
) -> dict[str, float | int]:
    pivots = _zigzag_pivots(bars, reversal_threshold, depth_bars, backstep_bars)
    if len(pivots) < 2:
        return {"ideal": 0.0, "anti": 0.0, "cycles": 0, "trades": 0}

    initial_cash = float(execution["initial_cash"] or 10000.0)
    ideal_cash = initial_cash
    anti_cash = initial_cash
    cycles = 0
    trades = 0
    commission_rate = ORACLE_COST_STRESS_MULTIPLIER * float(execution["commission"] or 0.0)
    price_cost_rate = ORACLE_COST_STRESS_MULTIPLIER * (
        float(execution["slippage"] or 0.0) + 0.5 * float(execution["full_spread"] or 0.0)
    )
    round_trip_cost = 2.0 * (commission_rate + price_cost_rate)
    legs = list(zip(pivots, pivots[1:]))
    last_idx, last_kind, last_price = pivots[-1]
    final_idx = len(bars) - 1
    final_close = bars[-1][3]
    if final_idx > last_idx:
        final_kind = "peak" if last_kind == "valley" else "valley"
        legs.append((pivots[-1], (final_idx, final_kind, final_close)))

    for (_, entry_kind, entry_price), (_, exit_kind, exit_price) in legs:
        if entry_price <= 0 or exit_price <= 0:
            continue
        if entry_kind == "valley" and exit_kind == "peak":
            gross_move = (exit_price - entry_price) / entry_price
        elif entry_kind == "peak" and exit_kind == "valley":
            gross_move = (entry_price - exit_price) / entry_price
        else:
            continue
        if gross_move <= 0:
            continue
        if gross_move < round_trip_cost * min_profit_multiple:
            continue
        direction = "long" if entry_kind == "valley" else "short"
        anti_direction = "short" if direction == "long" else "long"
        next_ideal_cash, ideal_filled = _apply_oracle_trade(
            ideal_cash,
            entry_price,
            exit_price,
            direction,
            execution,
            commission_rate,
            price_cost_rate,
        )
        if not ideal_filled or next_ideal_cash <= ideal_cash:
            continue
        next_anti_cash, anti_filled = _apply_oracle_trade(
            anti_cash,
            entry_price,
            exit_price,
            anti_direction,
            execution,
            commission_rate,
            price_cost_rate,
        )
        ideal_cash = next_ideal_cash
        if anti_filled:
            anti_cash = next_anti_cash
        cycles += 1
        trades += 2
    return {
        "ideal": (ideal_cash - initial_cash) / initial_cash,
        "anti": (anti_cash - initial_cash) / initial_cash,
        "cycles": cycles,
        "trades": trades,
    }


def _feature_summary(config_json: str | None, feature_count: int | None) -> dict:
    try:
        config = json.loads(config_json or "{}")
    except json.JSONDecodeError:
        config = {}
    features = config.get("feature_columns") or []
    if not isinstance(features, list):
        features = []
    count = int(feature_count or len(features))
    preview = [str(name) for name in features[:24]]
    extra = max(0, len(features) - len(preview))
    return {
        "count": count,
        "preview": preview,
        "extra": extra,
        "text": ", ".join(preview) + (f", +{extra} more" if extra else ""),
    }


def _avg_pair(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def _trim_chart_for_focus(chart: dict) -> dict:
    labels = list(chart.get("labels") or [])
    start_idx = next(
        (idx for idx, label in enumerate(labels) if str(label) >= CHART_FOCUS_START_LABEL),
        0,
    )

    def trim_group(group: dict, trim_start: int) -> dict:
        trimmed: dict = {}
        for key, value in group.items():
            trimmed[key] = value[trim_start:] if isinstance(value, list) else value
        return trimmed

    def tail_group(group: dict) -> dict:
        trimmed: dict = {}
        for key, value in group.items():
            trimmed[key] = value[-CHART_MAX_POINTS:] if isinstance(value, list) else value
        return trimmed

    chart["labels"] = labels[start_idx:]
    chart["model"] = trim_group(dict(chart.get("model") or {}), start_idx)
    chart["oracle"] = trim_group(dict(chart.get("oracle") or {}), start_idx)
    if len(chart["labels"]) > CHART_MAX_POINTS:
        chart["trimmed_points"] = start_idx + len(chart["labels"]) - CHART_MAX_POINTS
        chart["labels"] = chart["labels"][-CHART_MAX_POINTS:]
        chart["model"] = tail_group(chart["model"])
        chart["oracle"] = tail_group(chart["oracle"])
        chart["point_limit"] = CHART_MAX_POINTS
    else:
        chart["trimmed_points"] = start_idx
    chart["focus_start"] = CHART_FOCUS_START_LABEL
    return chart


def _oracle_baseline_for_window(
    input_data_file: str | None,
    window_start: str | datetime | None,
    window_end: str | datetime | None,
    config_path: str | None,
) -> dict[str, float | None]:
    if not input_data_file or not window_start or not window_end:
        return {"ideal": None, "anti": None, "cycles": None, "threshold": None}
    data_path = Path(input_data_file)
    start = window_start if isinstance(window_start, datetime) else _parse_datetime(window_start)
    end = window_end if isinstance(window_end, datetime) else _parse_datetime(window_end)
    if not data_path.exists() or start is None or end is None or end <= start:
        return {"ideal": None, "anti": None, "cycles": None, "threshold": None}

    execution = _oracle_execution_config(config_path)
    try:
        stat = data_path.stat()
        data_stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        data_stamp = (0, 0)

    commission_rate = ORACLE_COST_STRESS_MULTIPLIER * float(execution["commission"] or 0.0)
    price_cost_rate = ORACLE_COST_STRESS_MULTIPLIER * (
        float(execution["slippage"] or 0.0) + 0.5 * float(execution["full_spread"] or 0.0)
    )
    per_side_cost = max(0.0, commission_rate + price_cost_rate)
    window_series = [bar for bar in _price_series(data_path, data_stamp) if start <= bar[0] <= end]
    min_pivot_bars = _minimum_zigzag_bars(window_series)
    backstep_bars = max(1, min_pivot_bars // 3)
    cache_key = (
        "mt4_style_zigzag_oracle_v3_exact_sizing",
        str(data_path),
        data_stamp,
        start.isoformat(),
        end.isoformat(),
        per_side_cost,
        min_pivot_bars,
        backstep_bars,
        execution.get("initial_cash"),
        execution.get("position_size"),
        execution.get("rel_volume"),
        execution.get("leverage"),
        execution.get("min_order_volume"),
        execution.get("max_order_volume"),
        execution.get("size_mode"),
    )
    cached = _BASELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if len(window_series) < 2:
        baseline = {"ideal": None, "anti": None, "cycles": None, "threshold": None}
        _BASELINE_CACHE[cache_key] = baseline
        return baseline

    best_score = {"ideal": 0.0, "anti": 0.0, "cycles": 0, "trades": 0}
    best_threshold = None
    best_profit_multiple = None
    best_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    for threshold in _zigzag_threshold_candidates(window_series, per_side_cost):
        for min_profit_multiple in (2.0, 3.0, 5.0, 10.0):
            score = _score_zigzag_oracle(
                window_series,
                threshold,
                execution,
                min_pivot_bars,
                backstep_bars,
                min_profit_multiple,
            )
            score_key = (
                float(score["ideal"]),
                min_profit_multiple,
                threshold,
                -float(score["cycles"]),
            )
            if score_key <= best_key:
                continue
            best_score = score
            best_threshold = threshold
            best_profit_multiple = min_profit_multiple
            best_key = score_key

    baseline = {
        "ideal": float(best_score["ideal"]),
        "anti": float(best_score["anti"]),
        "cycles": float(best_score["cycles"]),
        "trades": float(best_score["trades"]),
        "threshold": best_threshold,
        "min_bars": float(min_pivot_bars),
        "backstep_bars": float(backstep_bars),
        "cost_per_side": float(per_side_cost),
        "round_trip_cost": float(2.0 * per_side_cost),
        "profit_multiple": best_profit_multiple,
    }
    _BASELINE_CACHE[cache_key] = baseline
    return baseline


def _performance_payload(conn: sqlite3.Connection) -> dict:
    rows = _rows(
        conn,
        """
        SELECT s.external_id AS subjob_id, s.completed_at, s.result_json,
               s.train_end, s.validation_start, s.validation_end,
               s.test_start, s.test_end, s.config_path,
               j.external_id AS job_id, j.candidate_id, j.asset, j.timeframe,
               j.model_family, j.train_years, j.training_policy, j.feature_count,
               j.input_data_file, j.config_json
        FROM subjobs s
        JOIN jobs j ON j.id = s.job_id
        WHERE s.status='done' AND s.result_json IS NOT NULL
        ORDER BY COALESCE(s.completed_at, s.updated_at), s.id
        """,
    )

    chart_labels: list[str] = []
    champion_train: list[float | None] = []
    champion_validation: list[float | None] = []
    champion_test: list[float | None] = []
    champion_composite: list[float | None] = []
    champion_oracle_train: list[float | None] = []
    champion_oracle_validation: list[float | None] = []
    champion_oracle_test: list[float | None] = []
    champion_oracle_composite: list[float | None] = []
    champion_anti_composite: list[float | None] = []
    min_samples = 5
    job_values: dict[str, dict] = {}
    focus_start_dt = _parse_datetime(CHART_FOCUS_START_LABEL)

    for row in rows:
        try:
            result = json.loads(row.get("result_json") or "{}")
        except json.JSONDecodeError:
            result = {}
        raw_result = _load_result_file_from_summary(result)
        job_config = _json_or_empty(row.get("config_json"))
        hparams = job_config.get("hyperparameters") or {}
        if not isinstance(hparams, dict):
            hparams = {}
        risk_lambda = (
            _float_or_none(result.get("risk_penalty_lambda"))
            or _float_or_none(hparams.get("risk_penalty_lambda"))
            or _float_or_none(hparams.get("penalty_lambda"))
            or _float_or_none(job_config.get("risk_penalty_lambda"))
            or 1.0
        )
        rel_volume = (
            _float_or_none(result.get("rel_volume"))
            or _float_or_none(hparams.get("rel_volume"))
            or _float_or_none(job_config.get("rel_volume"))
            or _float_or_none(job_config.get("risk_sizing_rel_volume"))
        )
        k_sl = (
            _float_or_none(result.get("k_sl"))
            or _float_or_none(hparams.get("k_sl"))
            or _float_or_none(job_config.get("k_sl"))
        )
        k_tp = (
            _float_or_none(result.get("k_tp"))
            or _float_or_none(hparams.get("k_tp"))
            or _float_or_none(job_config.get("k_tp"))
        )
        reward_risk_ratio = _float_or_none(result.get("reward_risk_ratio"))
        if reward_risk_ratio is None and k_sl and k_tp is not None:
            reward_risk_ratio = (k_tp / k_sl) if k_sl > 0 else None
        business_risk_fraction = _float_or_none(result.get("business_risk_fraction"))
        sltp_risk_mode = str(
            result.get("sltp_risk_mode")
            or hparams.get("sltp_risk_mode")
            or job_config.get("sltp_risk_mode")
            or "fixed_atr"
        )
        sltp_profile_tag = str(
            result.get("sltp_profile_tag")
            or hparams.get("sltp_profile_tag")
            or job_config.get("sltp_profile_tag")
            or ""
        )
        atr_period = (
            _float_or_none(result.get("atr_period"))
            or _float_or_none(hparams.get("atr_period"))
            or _float_or_none(job_config.get("atr_period"))
        )
        reward_plugin = str(hparams.get("reward_plugin") or job_config.get("reward_plugin") or "")
        selection_metric = str(result.get("selection_metric") or hparams.get("selection_metric") or "total_return")
        train = _split_summary_metric(
            result,
            raw_result,
            split="train_tail",
            summary_key="train_tail_total_return",
            metric_key="total_return",
        )
        validation = _split_summary_metric(
            result,
            raw_result,
            split="validation",
            summary_key="validation_total_return",
            metric_key="total_return",
        )
        test = _split_summary_metric(
            result,
            raw_result,
            split="test",
            summary_key="test_total_return",
            metric_key="total_return",
        )
        validation_sharpe = _float_or_none(result.get("validation_sharpe"))
        test_sharpe = _float_or_none(result.get("test_sharpe"))
        train_dd = _split_drawdown_fraction(
            result, raw_result, split="train_tail", summary_prefix="train_tail"
        )
        validation_dd = _split_drawdown_fraction(
            result, raw_result, split="validation", summary_prefix="validation"
        )
        test_dd = _split_drawdown_fraction(
            result, raw_result, split="test", summary_prefix="test"
        )
        train_rap = _float_or_none(result.get("train_tail_risk_adjusted_total_return"))
        if train_rap is None:
            train_rap = _risk_adjusted_return_value(train, train_dd, risk_lambda)
        validation_rap = _float_or_none(result.get("validation_risk_adjusted_total_return"))
        if validation_rap is None:
            validation_rap = _risk_adjusted_return_value(validation, validation_dd, risk_lambda)
        test_rap = _float_or_none(result.get("test_risk_adjusted_total_return"))
        if test_rap is None:
            test_rap = _risk_adjusted_return_value(test, test_dd, risk_lambda)
        risk_composite = _float_or_none(result.get("train_validation_risk_adjusted_composite_score"))
        composite = _float_or_none(result.get("train_validation_composite_score"))
        if composite is None and train is not None and validation is not None:
            composite = (train + validation) / 2
        if risk_composite is None and train_rap is not None and validation_rap is not None:
            risk_composite = (train_rap + validation_rap) / 2
        l1_score = _float_or_none(result.get("train_validation_l1_score"))
        if l1_score is None:
            l1_score = _float_or_none(result.get("train_validation_selection_score"))
        if l1_score is None:
            l1_score = risk_composite
        if l1_score is None:
            l1_score = composite
        completed = str(row.get("completed_at") or "")
        label = completed.replace("T", " ").replace("+00:00", "Z")
        completed_dt = _parse_datetime(completed)
        compute_oracle = bool(
            completed_dt is not None
            and focus_start_dt is not None
            and completed_dt >= focus_start_dt
        )
        _commission, _slippage, _full_spread, train_tail_days = _config_costs(row.get("config_path"))
        train_end = _parse_datetime(row.get("train_end"))
        train_tail_start = train_end - timedelta(days=train_tail_days) if train_end else None
        train_baseline = {"ideal": None, "anti": None, "cycles": None, "threshold": None}
        validation_baseline = {"ideal": None, "anti": None}
        test_baseline = {"ideal": None, "anti": None}
        ideal = train_baseline.get("ideal")
        anti = train_baseline.get("anti")
        oracle_validation = validation_baseline.get("ideal")
        oracle_test = test_baseline.get("ideal")
        oracle_composite = _avg_pair(ideal, oracle_validation)
        anti_composite = _avg_pair(anti, validation_baseline.get("anti"))
        oracle_cycles = train_baseline.get("cycles")
        oracle_threshold = train_baseline.get("threshold")
        oracle_min_bars = train_baseline.get("min_bars")
        oracle_backstep_bars = train_baseline.get("backstep_bars")
        oracle_round_trip_cost = train_baseline.get("round_trip_cost")
        oracle_profit_multiple = train_baseline.get("profit_multiple")

        job_id = str(row["job_id"])
        job = job_values.setdefault(
            job_id,
            {
                "job_id": job_id,
                "candidate_id": row.get("candidate_id"),
                "asset": row.get("asset"),
                "timeframe": row.get("timeframe"),
                "model_family": row.get("model_family"),
                "train_years": row.get("train_years"),
                "training_policy": row.get("training_policy"),
                "feature_count": row.get("feature_count"),
                "input_data_file": row.get("input_data_file"),
                "feature_summary": _feature_summary(row.get("config_json"), row.get("feature_count")),
                "reward_plugin": reward_plugin,
                "selection_metric": selection_metric,
                "risk_penalty_lambda": risk_lambda,
                "rel_volume": rel_volume,
                "business_risk_fraction": business_risk_fraction,
                "sltp_risk_mode": sltp_risk_mode,
                "sltp_profile_tag": sltp_profile_tag,
                "atr_period": atr_period,
                "k_sl": k_sl,
                "k_tp": k_tp,
                "reward_risk_ratio": reward_risk_ratio,
                "n": 0,
                "train": [],
                "validation": [],
                "test": [],
                "composite": [],
                "validation_sharpe": [],
                "test_sharpe": [],
                "train_rap": [],
                "validation_rap": [],
                "test_rap": [],
                "risk_composite": [],
                "l1_score": [],
                "train_drawdown": [],
                "validation_drawdown": [],
                "test_drawdown": [],
                "test_trades": [],
                "ideal": [],
                "anti": [],
                "oracle_validation": [],
                "oracle_test": [],
                "oracle_composite": [],
                "anti_composite": [],
                "oracle_cycles": [],
                "oracle_threshold": [],
                "oracle_min_bars": [],
                "oracle_backstep_bars": [],
                "oracle_round_trip_cost": [],
                "oracle_profit_multiple": [],
                "oracle_rows": [],
            },
        )
        if compute_oracle:
            job["oracle_rows"].append(
                {
                    "input_data_file": row.get("input_data_file"),
                    "train_tail_start": train_tail_start,
                    "train_end": train_end,
                    "validation_start": row.get("validation_start"),
                    "validation_end": row.get("validation_end"),
                    "config_path": row.get("config_path"),
                }
            )
        job["n"] += 1
        if train is not None:
            job["train"].append(train)
        if validation is not None:
            job["validation"].append(validation)
        if test is not None:
            job["test"].append(test)
        if composite is not None:
            job["composite"].append(composite)
        if validation_sharpe is not None:
            job["validation_sharpe"].append(validation_sharpe)
        if test_sharpe is not None:
            job["test_sharpe"].append(test_sharpe)
        if train_rap is not None:
            job["train_rap"].append(train_rap)
        if validation_rap is not None:
            job["validation_rap"].append(validation_rap)
        if test_rap is not None:
            job["test_rap"].append(test_rap)
        if risk_composite is not None:
            job["risk_composite"].append(risk_composite)
        if l1_score is not None:
            job["l1_score"].append(l1_score)
        if train_dd is not None:
            job["train_drawdown"].append(train_dd)
        if validation_dd is not None:
            job["validation_drawdown"].append(validation_dd)
        if test_dd is not None:
            job["test_drawdown"].append(test_dd)
        if ideal is not None:
            job["ideal"].append(ideal)
        if anti is not None:
            job["anti"].append(anti)
        if oracle_validation is not None:
            job["oracle_validation"].append(float(oracle_validation))
        if oracle_test is not None:
            job["oracle_test"].append(float(oracle_test))
        if oracle_composite is not None:
            job["oracle_composite"].append(float(oracle_composite))
        if anti_composite is not None:
            job["anti_composite"].append(float(anti_composite))
        if oracle_cycles is not None:
            job["oracle_cycles"].append(float(oracle_cycles))
        if oracle_threshold is not None:
            job["oracle_threshold"].append(float(oracle_threshold))
        if oracle_min_bars is not None:
            job["oracle_min_bars"].append(float(oracle_min_bars))
        if oracle_backstep_bars is not None:
            job["oracle_backstep_bars"].append(float(oracle_backstep_bars))
        if oracle_round_trip_cost is not None:
            job["oracle_round_trip_cost"].append(float(oracle_round_trip_cost))
        if oracle_profit_multiple is not None:
            job["oracle_profit_multiple"].append(float(oracle_profit_multiple))
        trades = _float_or_none(result.get("test_trades_total"))
        if trades is not None:
            job["test_trades"].append(trades)

        chart_labels.append(label)
        eligible_running = [
            candidate
            for candidate in job_values.values()
            if candidate["n"] >= min_samples and candidate["l1_score"]
        ]
        champion = max(
            eligible_running,
            key=lambda candidate: _mean(candidate["l1_score"]) or float("-inf"),
            default=None,
        )
        if champion is None:
            champion_train.append(None)
            champion_validation.append(None)
            champion_test.append(None)
            champion_composite.append(None)
            champion_oracle_train.append(None)
            champion_oracle_validation.append(None)
            champion_oracle_test.append(None)
            champion_oracle_composite.append(None)
            champion_anti_composite.append(None)
        else:
            champion_train.append(_mean(champion["train"]))
            champion_validation.append(_mean(champion["validation"]))
            champion_test.append(_mean(champion["test"]))
            champion_composite.append(_mean(champion["composite"]))
            champion_oracle_train.append(_mean(champion["ideal"]))
            champion_oracle_validation.append(_mean(champion["oracle_validation"]))
            champion_oracle_test.append(_mean(champion["oracle_test"]))
            champion_oracle_composite.append(_mean(champion["oracle_composite"]))
            champion_anti_composite.append(_mean(champion["anti_composite"]))

    summaries = []
    for job in job_values.values():
        summary = dict(job)
        summary["train_avg"] = _mean(job["train"])
        summary["validation_avg"] = _mean(job["validation"])
        summary["test_avg"] = _mean(job["test"])
        summary["composite_avg"] = _mean(job["composite"])
        summary["validation_sharpe_avg"] = _mean(job["validation_sharpe"])
        summary["test_sharpe_avg"] = _mean(job["test_sharpe"])
        summary["train_rap_avg"] = _mean(job["train_rap"])
        summary["validation_rap_avg"] = _mean(job["validation_rap"])
        summary["test_rap_avg"] = _mean(job["test_rap"])
        summary["risk_composite_avg"] = _mean(job["risk_composite"])
        summary["l1_score_avg"] = _mean(job["l1_score"])
        summary["train_drawdown_avg"] = _mean(job["train_drawdown"])
        summary["validation_drawdown_avg"] = _mean(job["validation_drawdown"])
        summary["test_drawdown_avg"] = _mean(job["test_drawdown"])
        summary["test_trades_avg"] = _mean(job["test_trades"])
        summary["ideal_avg"] = _mean(job["ideal"])
        summary["anti_avg"] = _mean(job["anti"])
        summary["oracle_validation_avg"] = _mean(job["oracle_validation"])
        summary["oracle_test_avg"] = _mean(job["oracle_test"])
        summary["oracle_composite_avg"] = _mean(job["oracle_composite"])
        summary["anti_composite_avg"] = _mean(job["anti_composite"])
        summary["oracle_cycles_avg"] = _mean(job["oracle_cycles"])
        summary["oracle_threshold_avg"] = _mean(job["oracle_threshold"])
        summary["oracle_min_bars_avg"] = _mean(job["oracle_min_bars"])
        summary["oracle_backstep_bars_avg"] = _mean(job["oracle_backstep_bars"])
        summary["oracle_round_trip_cost_avg"] = _mean(job["oracle_round_trip_cost"])
        summary["oracle_profit_multiple_avg"] = _mean(job["oracle_profit_multiple"])
        summaries.append(summary)

    compact_summaries = [_compact_job_summary(row) for row in summaries]
    eligible_composite = [
        row for row in compact_summaries if row["n"] >= min_samples and row["l1_score_avg"] is not None
    ]
    best_composite_job = max(eligible_composite, key=lambda row: row["l1_score_avg"], default=None)
    risk_profit_points = []
    for row in compact_summaries:
        if row["n"] < min_samples:
            continue
        test_avg = _float_or_none(row.get("test_avg"))
        test_drawdown = _float_or_none(row.get("test_drawdown_avg"))
        if test_avg is None or test_drawdown is None:
            continue
        risk_profit_points.append(
            {
                "x": max(0.0, test_drawdown),
                "y": test_avg,
                "label": str(row.get("job_id") or row.get("candidate_id") or ""),
                "asset": row.get("asset"),
                "timeframe": row.get("timeframe"),
                "model": row.get("model_family"),
                "policy": row.get("training_policy"),
                "samples": row.get("n"),
                "composite": row.get("composite_avg"),
                "l1_score": row.get("l1_score_avg"),
                "risk_composite": row.get("risk_composite_avg"),
                "test_rap": row.get("test_rap_avg"),
                "risk_lambda": row.get("risk_penalty_lambda"),
                "rel_volume": row.get("rel_volume"),
                "business_risk_fraction": row.get("business_risk_fraction"),
                "sltp_risk_mode": row.get("sltp_risk_mode"),
                "sltp_profile_tag": row.get("sltp_profile_tag"),
                "atr_period": row.get("atr_period"),
                "k_sl": row.get("k_sl"),
                "k_tp": row.get("k_tp"),
                "reward_risk_ratio": row.get("reward_risk_ratio"),
                "selection_metric": row.get("selection_metric"),
                "reward_plugin": row.get("reward_plugin"),
            }
        )
    risk_profit_points.sort(
        key=lambda item: (
            item.get("risk_composite") is not None,
            float(item.get("l1_score") or item.get("risk_composite") or item.get("composite") or float("-inf")),
            float(item.get("risk_composite") or item.get("composite") or float("-inf")),
        ),
        reverse=True,
    )

    chart = _trim_chart_for_focus(
        {
            "labels": chart_labels,
            "model": {
                "train": champion_train,
                "validation": champion_validation,
                "test": champion_test,
                "composite": champion_composite,
            },
            "oracle": {
                "train": champion_oracle_train,
                "validation": champion_oracle_validation,
                "test": champion_oracle_test,
                "composite": champion_oracle_composite,
                "anti_composite": champion_anti_composite,
            },
            "risk_profit": {
                "points": risk_profit_points[:400],
                "x_label": "Average weekly test max drawdown fraction",
                "y_label": "Average weekly test net total_return",
                "cdt_weekly_return": CDT_WEEKLY_BASELINE_RETURN,
                "cdt_weekly_stretch_return": CDT_WEEKLY_STRETCH_RETURN,
                "cdt_note": (
                    "Colombia CDT reference: 12.0% and 12.6% effective annual returns "
                    "converted to weekly geometric returns; drawdown baseline is shown "
                    "near zero for held-to-maturity comparison."
                ),
            },
        },
    )
    if best_composite_job:
        raw_best = job_values.get(str(best_composite_job.get("job_id")))
        oracle_composites: list[float] = []
        anti_composites: list[float] = []
        oracle_rows = list((raw_best or {}).get("oracle_rows") or [])[-64:]
        for oracle_row in oracle_rows:
            train_baseline = _oracle_baseline_for_window(
                oracle_row.get("input_data_file"),
                oracle_row.get("train_tail_start"),
                oracle_row.get("train_end"),
                oracle_row.get("config_path"),
            )
            validation_baseline = _oracle_baseline_for_window(
                oracle_row.get("input_data_file"),
                oracle_row.get("validation_start"),
                oracle_row.get("validation_end"),
                oracle_row.get("config_path"),
            )
            oracle_composite = _avg_pair(train_baseline.get("ideal"), validation_baseline.get("ideal"))
            anti_composite = _avg_pair(train_baseline.get("anti"), validation_baseline.get("anti"))
            if oracle_composite is not None:
                oracle_composites.append(float(oracle_composite))
            if anti_composite is not None:
                anti_composites.append(float(anti_composite))
        if chart["labels"]:
            if oracle_composites:
                chart["oracle"]["composite"] = [_mean(oracle_composites)] * len(chart["labels"])
            if anti_composites:
                chart["oracle"]["anti_composite"] = [_mean(anti_composites)] * len(chart["labels"])

    return {
        "chart": chart,
        "best_composite_job": best_composite_job,
        "job_summaries": sorted(
            compact_summaries,
            key=lambda row: (row["l1_score_avg"] is not None, row["l1_score_avg"] or float("-inf")),
            reverse=True,
        )[:20],
        "min_samples": min_samples,
    }


def _annual_protocol_payload(conn: sqlite3.Connection) -> dict:
    try:
        rows = _rows(
            conn,
            """
            SELECT metric_block, candidate_id, asset, timeframe, model_family,
                   train_years, training_policy, experiment_phase, metric_year,
                   unique_weeks, mean_weekly_return, sum_weekly_return, annual_return,
                   observed_return, projected_annual_return_52w,
                   mean_weekly_drawdown, sum_weekly_drawdown, mean_weekly_rap,
                   sum_weekly_rap, annual_rap, worst_weekly_rap, best_weekly_rap,
                   observed_rap, projected_annual_rap_52w,
                   mean_weekly_l1_score, mean_weekly_l1_gap,
                   rel_volume, business_risk_fraction, sltp_risk_mode, k_sl, k_tp,
                   risk_penalty_lambda, first_week, last_week, has_near_full_year_coverage
            FROM weekly_result_full_year_protocol_olap
            WHERE has_near_full_year_coverage = 1
            ORDER BY metric_block, annual_rap DESC
            LIMIT 40
            """,
        )
    except sqlite3.OperationalError:
        rows = []
    return {"rows": rows}


def _portfolio_payload(conn: sqlite3.Connection) -> dict:
    try:
        rows = _rows(
            conn,
            """
            SELECT run_id, generated_at, config_json, summary_json
            FROM portfolio_runs
            ORDER BY generated_at DESC, run_id
            """,
        )
    except sqlite3.OperationalError:
        return {"runs": [], "best": None, "chart": {"labels": [], "mean_weekly_return": [], "cumulative_return": []}}

    parsed_runs: list[dict] = []
    for row in rows:
        try:
            config = json.loads(row.get("config_json") or "{}")
        except json.JSONDecodeError:
            config = {}
        try:
            summary = json.loads(row.get("summary_json") or "{}")
        except json.JSONDecodeError:
            summary = {}
        mean_weekly = _float_or_none(summary.get("mean_weekly_return"))
        parsed_runs.append(
            {
                "run_id": row.get("run_id"),
                "generated_at": row.get("generated_at"),
                "method": config.get("method") or summary.get("method"),
                "weeks": summary.get("weeks"),
                "allocations": summary.get("allocations"),
                "inactive_decisions": summary.get("inactive_decisions"),
                "mean_weekly_return": mean_weekly,
                "mean_weekly_gross_return": _float_or_none(summary.get("mean_weekly_gross_return")),
                "mean_rebalance_cost": _float_or_none(summary.get("mean_rebalance_cost")),
                "mean_turnover": _float_or_none(summary.get("mean_turnover")),
                "cumulative_return": _float_or_none(summary.get("cumulative_return")),
                "max_drawdown": _float_or_none(summary.get("max_drawdown")),
                "cvar_20_weekly": _float_or_none(summary.get("cvar_20_weekly")),
                "sharpe_like_weekly": _float_or_none(summary.get("sharpe_like_weekly")),
                "max_assets": config.get("max_assets"),
                "max_weight": config.get("max_weight"),
                "max_asset_weight": config.get("max_asset_weight"),
                "max_cluster_weight": config.get("max_cluster_weight"),
                "turnover_penalty_bps": config.get("turnover_penalty_bps"),
                "portfolio_vol_target": config.get("portfolio_vol_target"),
                "min_observations": config.get("min_observations"),
                "max_drawdown_threshold": config.get("max_drawdown_threshold"),
            }
        )

    ranked = sorted(
        [run for run in parsed_runs if run["mean_weekly_return"] is not None],
        key=lambda run: float(run["mean_weekly_return"]),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    top_chart = ranked[:10]
    return {
        "runs": ranked[:20],
        "latest_runs": parsed_runs[:10],
        "best": best,
        "chart": {
            "labels": [str(run.get("method") or run.get("run_id") or "") for run in top_chart],
            "mean_weekly_return": [run.get("mean_weekly_return") for run in top_chart],
            "cumulative_return": [run.get("cumulative_return") for run in top_chart],
            "max_drawdown": [run.get("max_drawdown") for run in top_chart],
        },
    }


def _compact_job_summary(row: dict) -> dict:
    series_keys = {
        "train",
        "validation",
        "test",
        "composite",
        "validation_sharpe",
        "test_sharpe",
        "train_rap",
        "validation_rap",
        "test_rap",
        "risk_composite",
        "l1_score",
        "train_drawdown",
        "validation_drawdown",
        "test_drawdown",
        "test_trades",
        "ideal",
        "anti",
        "oracle_validation",
        "oracle_test",
        "oracle_composite",
        "anti_composite",
        "oracle_cycles",
        "oracle_threshold",
        "oracle_min_bars",
        "oracle_backstep_bars",
        "oracle_round_trip_cost",
        "oracle_profit_multiple",
        "oracle_rows",
    }
    return {key: value for key, value in row.items() if key not in series_keys}


def _status_payload(db_path: Path) -> dict:
    conn = _connect(db_path)
    try:
        counts = {row["status"]: row["n"] for row in conn.execute("SELECT status, COUNT(*) AS n FROM subjobs GROUP BY status")}
        jobs = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
        total = sum(counts.values())
        done = counts.get("done", 0)
        pending = counts.get("pending", 0)
        running = counts.get("running", 0)
        failed = counts.get("failed", 0)
        machines = _rows(conn, "SELECT * FROM machine_heartbeats ORDER BY machine_id")
        now_utc = datetime.now(timezone.utc)
        for machine in machines:
            heartbeat_dt = _parse_utc(machine.get("heartbeat_at"))
            age_seconds = None
            if heartbeat_dt is not None:
                age_seconds = (now_utc - heartbeat_dt).total_seconds()
            machine["heartbeat_age_seconds"] = age_seconds
            machine["heartbeat_age"] = _fmt_age(age_seconds)
            machine["freshness"] = _heartbeat_freshness(age_seconds, str(machine.get("status") or ""))
        active = _rows(
            conn,
            """
            SELECT s.external_id, s.weekly_anchor_id, s.claimed_by, s.heartbeat_at,
                   s.train_start, s.train_end, s.validation_start, s.validation_end,
                   s.test_start, s.test_end, s.train_rows, s.validation_rows, s.test_rows,
                   s.depends_on_subjob_id, s.warm_start_parent_subjob_id,
                   s.run_dir, s.config_path,
                   j.candidate_id, j.asset, j.timeframe, j.model_family, j.train_years,
                   j.training_policy, j.feature_count
            FROM subjobs s
            JOIN jobs j ON j.id = s.job_id
            WHERE s.status='running'
            ORDER BY s.claimed_at
            """,
        )
        for row in active:
            progress = _progress_for_run(row.get("run_dir"))
            row["progress_pct"] = progress.get("progress_pct", progress.get("progress_percent"))
            row["progress_steps"] = progress.get("num_timesteps", progress.get("current_step"))
            row["progress_total"] = progress.get("total_timesteps")
            row["progress_return"] = progress.get("total_return")
            row["progress_trades"] = progress.get("trades_total_cumulative", progress.get("trades_total"))
            row["progress_updated_at"] = progress.get("updated_at_utc")
        next_pending = _rows(
            conn,
            """
            SELECT s.external_id, s.weekly_anchor_id, s.priority, s.train_start, s.validation_start,
                   s.test_start, s.depends_on_subjob_id, s.warm_start_parent_subjob_id,
                   j.asset, j.timeframe, j.model_family, j.train_years, j.training_policy, j.feature_count
            FROM subjobs s
            JOIN jobs j ON j.id = s.job_id
            WHERE s.status='pending'
            ORDER BY s.priority, s.id
            LIMIT 20
            """,
        )
        recent = _rows(
            conn,
            """
            SELECT s.external_id, s.status, s.completed_at, s.error, s.result_json,
                   j.candidate_id, j.asset, j.timeframe, j.model_family, j.train_years
            FROM subjobs s
            JOIN jobs j ON j.id = s.job_id
            WHERE s.status IN ('done','failed')
            ORDER BY COALESCE(s.completed_at, s.updated_at) DESC
            LIMIT 30
            """,
        )
        best = []
        for row in _rows(
            conn,
            """
            SELECT s.external_id, s.result_json, j.candidate_id, j.asset, j.timeframe,
                   j.model_family, j.train_years, j.training_policy, j.feature_count
            FROM subjobs s
            JOIN jobs j ON j.id = s.job_id
            WHERE s.status='done' AND s.result_json IS NOT NULL
            """
        ):
            try:
                result = json.loads(row.get("result_json") or "{}")
            except json.JSONDecodeError:
                result = {}
            score = result.get("score")
            if score is not None:
                row["score"] = score
                row["metrics"] = result
                best.append(row)
        best.sort(key=lambda r: float(r["score"]), reverse=True)
        performance = _performance_payload(conn)
        portfolio = _portfolio_payload(conn)
        annual_protocol = _annual_protocol_payload(conn)
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "db_path": str(db_path),
            "jobs": jobs,
            "counts": {"total": total, "pending": pending, "running": running, "done": done, "failed": failed},
            "machines": machines,
            "active": active,
            "next_pending": next_pending,
            "recent": recent,
            "best": best[:20],
            "performance": performance,
            "portfolio": portfolio,
            "annual_protocol": annual_protocol,
        }
    finally:
        conn.close()


def _badge(status: str) -> str:
    cls = {
        "pending": "secondary",
        "running": "primary",
        "done": "success",
        "failed": "danger",
        "idle": "info",
        "fresh": "success",
        "warn": "warning",
        "stale": "danger",
        "unknown": "secondary",
    }.get(status, "dark")
    return f'<span class="badge badge-{cls}">{html.escape(status)}</span>'


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f'<table class="table table-sm table-striped table-hover"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _best_job_card(title: str, job: dict | None, accent: str, primary_metric: str) -> str:
    if not job:
        return f"""
        <div class="card card-outline card-{accent}">
          <div class="card-header"><h3 class="card-title">{html.escape(title)}</h3></div>
          <div class="card-body"><p class="text-muted mb-0">No completed result with enough samples yet.</p></div>
        </div>
        """
    feature_summary = job.get("feature_summary") or {}
    features = feature_summary.get("text") or ""
    return f"""
    <div class="card card-outline card-{accent}">
      <div class="card-header"><h3 class="card-title">{html.escape(title)}</h3></div>
      <div class="card-body">
        <h4 class="mb-1">{html.escape(_fmt_metric(job.get(primary_metric)))}</h4>
        <p class="text-muted mb-2">{html.escape(str(job.get("job_id") or ""))}</p>
        <dl class="row mb-0 small">
          <dt class="col-5">L1 score</dt><dd class="col-7 text-success">{html.escape(_fmt_metric(job.get("l1_score_avg")))}</dd>
          <dt class="col-5">Composite</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("composite_avg")))}</dd>
          <dt class="col-5">Train tail</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("train_avg")))}</dd>
          <dt class="col-5">Validation</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("validation_avg")))}</dd>
          <dt class="col-5">Test</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("test_avg")))}</dd>
          <dt class="col-5">RAP comp</dt><dd class="col-7 text-info">{html.escape(_fmt_metric(job.get("risk_composite_avg")))}</dd>
          <dt class="col-5">Train RAP</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("train_rap_avg")))}</dd>
          <dt class="col-5">Val RAP</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("validation_rap_avg")))}</dd>
          <dt class="col-5">Test RAP</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("test_rap_avg")))}</dd>
          <dt class="col-5">Val DD</dt><dd class="col-7 text-warning">{html.escape(_fmt_percent(job.get("validation_drawdown_avg")))}</dd>
          <dt class="col-5">Test DD</dt><dd class="col-7 text-warning">{html.escape(_fmt_percent(job.get("test_drawdown_avg")))}</dd>
          <dt class="col-5">rel_volume</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("rel_volume")))}</dd>
          <dt class="col-5">Risk fraction</dt><dd class="col-7">{html.escape(_fmt_percent(job.get("business_risk_fraction")))}</dd>
          <dt class="col-5">SL/TP</dt><dd class="col-7">{html.escape(str(job.get("sltp_risk_mode") or ""))}</dd>
          <dt class="col-5">ATR k</dt><dd class="col-7">SL {html.escape(_fmt_metric(job.get("k_sl")))} / TP {html.escape(_fmt_metric(job.get("k_tp")))}</dd>
          <dt class="col-5">RR</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("reward_risk_ratio")))}</dd>
          <dt class="col-5">Val Sharpe</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("validation_sharpe_avg")))}</dd>
          <dt class="col-5">Test Sharpe</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("test_sharpe_avg")))}</dd>
          <dt class="col-5">Oracle train</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("ideal_avg")))}</dd>
          <dt class="col-5">Oracle val</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("oracle_validation_avg")))}</dd>
          <dt class="col-5">Oracle test</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("oracle_test_avg")))}</dd>
          <dt class="col-5">Oracle comp</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("oracle_composite_avg")))}</dd>
          <dt class="col-5">Anti comp</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("anti_composite_avg")))}</dd>
          <dt class="col-5">Oracle cycles</dt><dd class="col-7">{html.escape('' if job.get("oracle_cycles_avg") is None else f'{float(job["oracle_cycles_avg"]):.1f}')}</dd>
          <dt class="col-5">Oracle threshold</dt><dd class="col-7">{html.escape('' if job.get("oracle_threshold_avg") is None else f'{100.0 * float(job["oracle_threshold_avg"]):.3f}%')}</dd>
          <dt class="col-5">Oracle depth</dt><dd class="col-7">{html.escape('' if job.get("oracle_min_bars_avg") is None else f'{float(job["oracle_min_bars_avg"]):.1f} bars')}</dd>
          <dt class="col-5">Oracle backstep</dt><dd class="col-7">{html.escape('' if job.get("oracle_backstep_bars_avg") is None else f'{float(job["oracle_backstep_bars_avg"]):.1f} bars')}</dd>
          <dt class="col-5">Oracle min edge</dt><dd class="col-7">{html.escape('' if job.get("oracle_profit_multiple_avg") is None else f'{float(job["oracle_profit_multiple_avg"]):.1f}x cost')}</dd>
          <dt class="col-5">Oracle cost RT</dt><dd class="col-7">{html.escape('' if job.get("oracle_round_trip_cost_avg") is None else f'{100.0 * float(job["oracle_round_trip_cost_avg"]):.3f}%')}</dd>
          <dt class="col-5">Test trades</dt><dd class="col-7">{html.escape('' if job.get("test_trades_avg") is None else f'{float(job["test_trades_avg"]):.1f}')}</dd>
          <dt class="col-5">Samples</dt><dd class="col-7">{html.escape(str(job.get("n") or ""))}</dd>
          <dt class="col-5">Asset</dt><dd class="col-7">{html.escape(str(job.get("asset") or ""))} {html.escape(str(job.get("timeframe") or ""))}</dd>
          <dt class="col-5">Model</dt><dd class="col-7">{html.escape(str(job.get("model_family") or ""))}</dd>
          <dt class="col-5">Policy</dt><dd class="col-7">{html.escape(str(job.get("training_policy") or ""))}</dd>
          <dt class="col-5">Train years</dt><dd class="col-7">{html.escape(str(job.get("train_years") or ""))}</dd>
          <dt class="col-5">Features</dt><dd class="col-7">{html.escape(str(feature_summary.get("count") or job.get("feature_count") or ""))}</dd>
        </dl>
        <p class="small text-muted mt-2 mb-0" title="{html.escape(features)}">{html.escape(features[:520])}{'...' if len(features) > 520 else ''}</p>
        <p class="small text-muted mt-2 mb-0">{html.escape(str(job.get("input_data_file") or ""))}</p>
      </div>
    </div>
    """


def _page(payload: dict) -> bytes:
    counts = payload["counts"]
    performance = payload.get("performance") or {}
    portfolio = payload.get("portfolio") or {}
    annual_protocol = payload.get("annual_protocol") or {}
    chart_payload = performance.get("chart") or {
        "labels": [],
        "model": {
            "train": [],
            "validation": [],
            "test": [],
            "composite": [],
        },
        "oracle": {
            "train": [],
            "validation": [],
            "test": [],
            "composite": [],
            "anti_composite": [],
        },
        "risk_profit": {"points": []},
    }
    chart_json = json.dumps(chart_payload)
    portfolio_chart_json = json.dumps(portfolio.get("chart") or {"labels": [], "mean_weekly_return": [], "cumulative_return": [], "max_drawdown": []})
    focus_start = html.escape(str(chart_payload.get("focus_start") or ""))
    trimmed_points = int(chart_payload.get("trimmed_points") or 0)
    focus_badge = (
        f'<span class="badge badge-secondary ml-1">chart focus from {focus_start}; '
        f'{trimmed_points} legacy points hidden</span>'
        if focus_start and trimmed_points
        else ""
    )
    cards = [
        ("Jobs", payload["jobs"], "info"),
        ("Pending", counts["pending"], "secondary"),
        ("Running", counts["running"], "primary"),
        ("Done", counts["done"], "success"),
        ("Deferred", counts.get("deferred", 0), "warning"),
        ("Failed", counts["failed"], "danger"),
    ]
    card_html = "".join(
        f"""
        <div class="col-6 col-lg">
          <div class="info-box bg-{color} top-metric-box">
            <span class="info-box-icon"><i class="fas fa-layer-group"></i></span>
            <div class="info-box-content"><span class="info-box-text">{label}</span><span class="info-box-number">{value}</span></div>
          </div>
        </div>
        """
        for label, value, color in cards
    )
    machine_rows = [
        [
            html.escape(str(m["machine_id"])),
            _badge(str(m["status"])),
            _badge(str(m.get("freshness") or "unknown")),
            html.escape(str(m.get("heartbeat_age") or "")),
            html.escape(str(m.get("current_subjob_id") or "")),
            html.escape(str(m.get("heartbeat_at") or "")),
            html.escape(str(m.get("gpu_summary") or "")),
            html.escape(str(m.get("message") or "")),
        ]
        for m in payload["machines"]
    ]
    active_rows = [
        [
            html.escape(str(r["external_id"])),
            html.escape(str(r["claimed_by"] or "")),
            html.escape(f"{r['asset']} {r['timeframe']}"),
            html.escape(str(r["model_family"])),
            html.escape(str(r["train_years"])),
            html.escape(
                ""
                if r.get("progress_pct") is None
                else f"{float(r['progress_pct']):.1f}% ({r.get('progress_steps')}/{r.get('progress_total')})"
            ),
            html.escape(
                ""
                if r.get("progress_return") is None
                else f"{float(r['progress_return']):+.5f}"
            ),
            html.escape(str(r.get("progress_trades") or "")),
            html.escape(str(r["train_rows"] or "")),
            html.escape(f"{r['train_start']} -> {r['train_end']}"),
            html.escape(f"{r['validation_start']} -> {r['validation_end']}"),
            html.escape(f"{r['test_start']} -> {r['test_end']}"),
        ]
        for r in payload["active"]
    ]
    pending_rows = [
        [
            html.escape(str(r["external_id"])),
            html.escape(str(r["priority"])),
            html.escape(f"{r['asset']} {r['timeframe']}"),
            html.escape(str(r["model_family"])),
            html.escape(str(r["train_years"])),
            html.escape(str(r["training_policy"])),
            html.escape(str(r.get("depends_on_subjob_id") or "")),
            html.escape(str(r["feature_count"])),
            html.escape(str(r["weekly_anchor_id"])),
        ]
        for r in payload["next_pending"]
    ]
    best_rows = [
        [
            html.escape(str(r["external_id"])),
            html.escape(str(r["score"])),
            html.escape(f"{r['asset']} {r['timeframe']}"),
            html.escape(str(r["model_family"])),
            html.escape(str(r["train_years"])),
            html.escape(str(r["training_policy"])),
        ]
        for r in payload["best"]
    ]
    recent_rows = []
    for r in payload["recent"]:
        err = r.get("error") or ""
        recent_rows.append(
            [
                html.escape(str(r["external_id"])),
                _badge(str(r["status"])),
                html.escape(f"{r['asset']} {r['timeframe']}"),
                html.escape(str(r["train_years"])),
                html.escape(str(r.get("completed_at") or "")),
                html.escape(str(err)[:180]),
            ]
        )
    portfolio_best = portfolio.get("best") or {}
    portfolio_best_html = (
        f"""
        <dl class="row mb-0 small">
          <dt class="col-5">Method</dt><dd class="col-7">{html.escape(str(portfolio_best.get("method") or ""))}</dd>
          <dt class="col-5">Mean weekly net</dt><dd class="col-7 text-success">{html.escape(_fmt_percent(portfolio_best.get("mean_weekly_return")))}</dd>
          <dt class="col-5">Cumulative</dt><dd class="col-7">{html.escape(_fmt_percent(portfolio_best.get("cumulative_return")))}</dd>
          <dt class="col-5">Max drawdown</dt><dd class="col-7 text-warning">{html.escape(_fmt_percent(portfolio_best.get("max_drawdown")))}</dd>
          <dt class="col-5">CVaR20 weekly</dt><dd class="col-7">{html.escape(_fmt_percent(portfolio_best.get("cvar_20_weekly")))}</dd>
          <dt class="col-5">Weeks</dt><dd class="col-7">{html.escape(str(portfolio_best.get("weeks") or ""))}</dd>
          <dt class="col-5">Allocations</dt><dd class="col-7">{html.escape(str(portfolio_best.get("allocations") or ""))}</dd>
          <dt class="col-5">Inactive</dt><dd class="col-7">{html.escape(str(portfolio_best.get("inactive_decisions") or ""))}</dd>
          <dt class="col-5">Turnover</dt><dd class="col-7">{html.escape(_fmt_metric(portfolio_best.get("mean_turnover")))}</dd>
          <dt class="col-5">Mean cost</dt><dd class="col-7">{html.escape(_fmt_percent(portfolio_best.get("mean_rebalance_cost")))}</dd>
          <dt class="col-5">Run</dt><dd class="col-7 text-break">{html.escape(str(portfolio_best.get("run_id") or ""))}</dd>
        </dl>
        """
        if portfolio_best
        else '<p class="text-muted mb-0">No portfolio supervisor runs yet.</p>'
    )
    portfolio_rows = [
        [
            html.escape(str(r.get("method") or "")),
            html.escape(str(r.get("run_id") or "")),
            html.escape(str(r.get("weeks") or "")),
            html.escape(_fmt_percent(r.get("mean_weekly_return"))),
            html.escape(_fmt_percent(r.get("cumulative_return"))),
            html.escape(_fmt_percent(r.get("max_drawdown"))),
            html.escape(_fmt_percent(r.get("cvar_20_weekly"))),
            html.escape(_fmt_metric(r.get("mean_turnover"))),
            html.escape(_fmt_percent(r.get("mean_rebalance_cost"))),
            html.escape(str(r.get("generated_at") or "")),
        ]
        for r in (portfolio.get("runs") or [])[:12]
    ]
    annual_rows = [
        [
            html.escape(str(r.get("metric_block") or "")),
            html.escape(str(r.get("candidate_id") or "")),
            html.escape(f"{r.get('asset') or ''} {r.get('timeframe') or ''}"),
            html.escape(str(r.get("metric_year") or "")),
            html.escape(str(r.get("unique_weeks") or "")),
            html.escape(_fmt_percent(r.get("mean_weekly_return"))),
            html.escape(_fmt_percent(r.get("observed_return"))),
            html.escape(_fmt_percent(r.get("projected_annual_return_52w"))),
            html.escape(_fmt_percent(r.get("annual_return"))),
            html.escape(_fmt_percent(r.get("mean_weekly_rap"))),
            html.escape(_fmt_percent(r.get("observed_rap"))),
            html.escape(_fmt_percent(r.get("projected_annual_rap_52w"))),
            html.escape(_fmt_percent(r.get("annual_rap"))),
            html.escape(_fmt_percent(r.get("mean_weekly_drawdown"))),
            html.escape(_fmt_metric(r.get("mean_weekly_l1_score"))),
            html.escape(_fmt_metric(r.get("mean_weekly_l1_gap"))),
            html.escape(_fmt_metric(r.get("rel_volume"))),
            html.escape(str(r.get("sltp_risk_mode") or "")),
        ]
        for r in (annual_protocol.get("rows") or [])[:20]
    ]
    html_page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project 3 Weekly Pool</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/admin-lte@3.2/dist/css/adminlte.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@5.15.4/css/all.min.css">
  <style>
    :root {{
      --p3-bg: #070b10;
      --p3-panel: #101820;
      --p3-panel-2: #121d28;
      --p3-border: rgba(148, 163, 184, 0.34);
      --p3-text: #d7e0ea;
      --p3-muted: #91a0af;
      --p3-heading: #f8fafc;
    }}
    body.dark-mode {{
      background: var(--p3-bg);
      color: var(--p3-text);
    }}
    .dark-mode .content-wrapper {{
      background: radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 34rem), var(--p3-bg);
      color: var(--p3-text);
    }}
    .main-header.navbar {{
      background: #0d141c !important;
      border-bottom: 1px solid var(--p3-border);
      color: var(--p3-text);
    }}
    .main-header .navbar-brand,
    .main-header .navbar-text,
    .main-header label {{
      color: var(--p3-text) !important;
    }}
    .main-sidebar {{
      background: #070b10 !important;
      border-right: 1px solid var(--p3-border);
    }}
    .brand-link {{
      border-bottom: 1px solid var(--p3-border) !important;
    }}
    .content-header {{
      padding-bottom: 0.35rem;
    }}
    .content-header h1 {{
      color: var(--p3-heading);
      font-weight: 650;
      letter-spacing: 0;
    }}
    .content-header .project-subtitle {{
      color: var(--p3-muted);
      max-width: 76rem;
      line-height: 1.45;
    }}
    .db-pill {{
      display: inline-block;
      max-width: 100%;
      margin-top: .45rem;
      padding: .35rem .65rem;
      border: 1px solid var(--p3-border);
      border-radius: .45rem;
      background: rgba(15, 23, 42, 0.78);
      color: #b9c6d3;
      font-size: .82rem;
      word-break: break-all;
    }}
    .dark-mode .card {{
      background: linear-gradient(180deg, rgba(20, 30, 41, 0.98), rgba(13, 20, 28, 0.98));
      border: 1px solid var(--p3-border);
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.34);
    }}
    .dark-mode .card-header {{
      border-bottom: 1px solid var(--p3-border);
      color: var(--p3-heading);
      background: rgba(255, 255, 255, 0.018);
    }}
    .dark-mode .card-title {{
      color: var(--p3-heading);
      font-weight: 600;
    }}
    .top-metric-box {{
      min-height: 90px;
      border: 1px solid rgba(255, 255, 255, 0.22);
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.28);
      border-radius: .55rem;
    }}
    .top-metric-box .info-box-number {{
      font-size: 1.55rem;
      line-height: 1.1;
    }}
    .dark-mode .table {{
      color: var(--p3-text);
    }}
    .dark-mode .table thead th {{
      border-bottom: 1px solid var(--p3-border);
      color: #eef4fb;
      background: rgba(255, 255, 255, 0.035);
    }}
    .dark-mode .table td,
    .dark-mode .table th {{
      border-top: 1px solid rgba(148, 163, 184, 0.18);
    }}
    .dark-mode .table-striped tbody tr:nth-of-type(odd) {{
      background-color: rgba(255, 255, 255, 0.026);
    }}
    .dark-mode .table-hover tbody tr:hover {{
      color: #ffffff;
      background-color: rgba(59, 130, 246, 0.16);
    }}
    .dark-mode .text-muted {{
      color: var(--p3-muted) !important;
    }}
    .refresh-control {{
      min-width: 112px;
      background: #111827;
      border: 1px solid var(--p3-border);
      color: var(--p3-text);
    }}
    .refresh-control:focus {{
      background: #111827;
      color: var(--p3-text);
      border-color: #60a5fa;
      box-shadow: 0 0 0 .15rem rgba(96, 165, 250, .25);
    }}
    .refresh-status {{
      color: var(--p3-muted);
      font-size: .86rem;
      margin-left: .5rem;
      white-space: nowrap;
    }}
    .performance-chart-card .card-body {{ min-height: 0; }}
    .chart-canvas-wrap {{
      position: relative;
      width: 100%;
      height: 430px;
      max-height: 430px;
      overflow: hidden;
    }}
	    #performanceChart {{
	      display: block;
	      width: 100% !important;
	      height: 100% !important;
      min-height: 0 !important;
	      max-height: 430px !important;
	    }}
	    #portfolioChart {{
	      display: block;
	      width: 100% !important;
	      height: 100% !important;
	      min-height: 0 !important;
	      max-height: 280px !important;
	    }}
	    #riskProfitChart {{
	      display: block;
	      width: 100% !important;
	      height: 100% !important;
	      min-height: 0 !important;
	      max-height: 340px !important;
	    }}
	    .portfolio-chart-wrap {{
	      position: relative;
	      width: 100%;
	      height: 280px;
	      max-height: 280px;
	      overflow: hidden;
	    }}
	    .risk-profit-chart-wrap {{
	      position: relative;
	      width: 100%;
	      height: 340px;
	      max-height: 340px;
	      overflow: hidden;
	    }}
	    .chart-explainer {{
	      margin: 0 0 .85rem 0;
      padding: .65rem .8rem;
      border: 1px solid rgba(96, 165, 250, 0.26);
      border-radius: .45rem;
      background: rgba(15, 23, 42, 0.55);
      color: #cbd5e1;
      font-size: .88rem;
      line-height: 1.4;
    }}
    .best-side-card .card-body {{ max-height: 560px; overflow-y: auto; }}
  </style>
</head>
<body class="hold-transition sidebar-mini layout-fixed dark-mode">
<div class="wrapper">
  <nav class="main-header navbar navbar-expand navbar-dark">
    <span class="navbar-brand">Project 3 Weekly Walk-Forward Pool</span>
    <div class="ml-auto d-flex align-items-center">
      <span class="navbar-text mr-3">Generated {html.escape(payload["generated_at"])}</span>
      <label class="mb-0 mr-2 small" for="refreshInterval">Refresh</label>
      <select id="refreshInterval" class="custom-select custom-select-sm refresh-control">
        <option value="300000">5 min</option>
        <option value="600000">10 min</option>
        <option value="1800000">30 min</option>
        <option value="3600000">1 h</option>
      </select>
      <span id="refreshStatus" class="refresh-status"></span>
    </div>
  </nav>
  <aside class="main-sidebar sidebar-dark-primary elevation-4">
    <span class="brand-link"><span class="brand-text font-weight-light ml-3">Project 3</span></span>
    <div class="sidebar"><nav class="mt-2"><ul class="nav nav-pills nav-sidebar flex-column">
      <li class="nav-item"><a class="nav-link active" href="/"><i class="nav-icon fas fa-tachometer-alt"></i><p>Pool Status</p></a></li>
      <li class="nav-item"><a class="nav-link" href="/api/status"><i class="nav-icon fas fa-code"></i><p>JSON API</p></a></li>
    </ul></nav></div>
  </aside>
  <div class="content-wrapper">
    <section class="content-header"><div class="container-fluid">
      <h1>Weekly Retrained Portfolio Experiments</h1>
      <p class="project-subtitle mb-0">Live orchestration view for weekly walk-forward SAC experiments. Current ranking uses L1 when present: mean(train-tail score, validation score) minus a generalization-gap penalty, with RAP score used by risk phases. Same-week test remains report-only.</p>
      <span class="db-pill"><i class="fas fa-database mr-1"></i>{html.escape(payload["db_path"])}</span>
    </div></section>
    <section class="content"><div class="container-fluid">
      <div class="row">{card_html}</div>
	      <div class="row">
	        <div class="col-lg-9">
          <div class="card performance-chart-card">
            <div class="card-header">
              <h3 class="card-title">Running Best Candidate Average Weekly Returns vs L1 Baselines</h3>
              <div class="card-tools"><span class="badge badge-info">Each point is the best candidate by average L1 score across completed weekly subjobs</span>{focus_badge}</div>
            </div>
            <div class="card-body">
              <p class="chart-explainer">
                Each point is recalculated after a weekly subjob finishes. The chart selects the candidate with the highest average L1 score over completed weekly retrain/fine-tune subjobs, then plots that candidate's average weekly train-tail return, average weekly validation return, average weekly test return, and legacy return composite. Train-tail is the last metric window of training; validation is the next window; test is report-only. L1 and RAP values are shown in the side card and annual table.
              </p>
              <div class="chart-canvas-wrap">
                <canvas id="performanceChart"></canvas>
              </div>
            </div>
          </div>
        </div>
	        <div class="col-lg-3 best-side-card">
	          {_best_job_card("Best L1/RAP So Far", performance.get("best_composite_job"), "success", "l1_score_avg")}
	        </div>
	      </div>
	      <div class="row">
	        <div class="col-12">
	          <div class="card">
	            <div class="card-header">
	              <h3 class="card-title">Full-Year Validation/Test Protocol</h3>
	              <div class="card-tools"><span class="badge badge-info">near-full-year coverage only; observed is covered-week sum, projected is 52x mean weekly</span></div>
	            </div>
	            <div class="card-body">
	              {_render_table(["Block", "Candidate", "Asset", "Year", "Weeks", "Mean weekly return", "Observed return", "Projected return 52w", "Legacy annual return", "Mean weekly RAP", "Observed RAP", "Projected RAP 52w", "Legacy annual RAP", "Mean weekly DD", "Mean L1", "Mean gap", "rel_volume", "SL/TP"], annual_rows)}
	            </div>
	          </div>
	        </div>
	      </div>
	      <div class="row">
	        <div class="col-12">
	          <div class="card">
	            <div class="card-header">
	              <h3 class="card-title">Risk vs Profit Map</h3>
	              <div class="card-tools"><span class="badge badge-info">test return vs test drawdown, averaged across completed weekly subjobs</span></div>
	            </div>
	            <div class="card-body">
	              <p class="chart-explainer">
	                Each point is one completed candidate with enough weekly walk-forward samples. X is average weekly test max drawdown fraction; Y is average weekly test net total_return. The dotted CDT lines are Colombia fixed-income references converted from annual effective return to weekly geometric return, with near-zero drawdown as a held-to-maturity reference. Selection still does not use same-week test.
	              </p>
	              <div class="risk-profit-chart-wrap"><canvas id="riskProfitChart"></canvas></div>
	            </div>
	          </div>
	        </div>
	      </div>
	      <div class="row">
	        <div class="col-lg-4">
	          <div class="card card-outline card-info">
	            <div class="card-header"><h3 class="card-title">Best Portfolio v2 Supervisor</h3></div>
	            <div class="card-body">{portfolio_best_html}</div>
	          </div>
	        </div>
	        <div class="col-lg-8">
	          <div class="card">
	            <div class="card-header">
	              <h3 class="card-title">Portfolio v2 Method Comparison</h3>
	              <div class="card-tools"><span class="badge badge-info">net mean weekly return; drawdown shown negative</span></div>
	            </div>
	            <div class="card-body">
	              <p class="chart-explainer">
	                Portfolio runs are computed above the per-asset agents. Weights use only known train-tail, validation, prior weekly returns, activation/no-trade gates, caps, and configured cost assumptions. Same-week test is recorded only after the allocation decision.
	              </p>
	              <div class="portfolio-chart-wrap"><canvas id="portfolioChart"></canvas></div>
	            </div>
	          </div>
	        </div>
	      </div>
	      <div class="card"><div class="card-header"><h3 class="card-title">Portfolio v2 Ranked Runs</h3></div><div class="card-body">
	        {_render_table(["Method", "Run", "Weeks", "Mean Weekly Net", "Cumulative", "Max DD", "CVaR20", "Turnover", "Mean Cost", "Generated"], portfolio_rows)}
	      </div></div>
	      <div class="card"><div class="card-header"><h3 class="card-title">Machines</h3></div><div class="card-body">
	        {_render_table(["Machine", "Status", "Freshness", "Age", "Subjob", "Heartbeat", "GPU", "Message"], machine_rows)}
      </div></div>
      <div class="card"><div class="card-header"><h3 class="card-title">Active Subjobs</h3></div><div class="card-body">
        {_render_table(["Subjob", "Machine", "Asset", "Model", "Train Years", "Progress", "Return", "Trades", "Train Rows", "Train Window", "Validation Window", "Test Window"], active_rows)}
      </div></div>
      <div class="card"><div class="card-header"><h3 class="card-title">Next Pending</h3></div><div class="card-body">
        {_render_table(["Subjob", "Priority", "Asset", "Model", "Train Years", "Policy", "Depends On", "Features", "Anchor"], pending_rows)}
      </div></div>
      <div class="card"><div class="card-header"><h3 class="card-title">Best Completed Results</h3></div><div class="card-body">
        {_render_table(["Subjob", "Score", "Asset", "Model", "Train Years", "Policy"], best_rows)}
      </div></div>
      <div class="card"><div class="card-header"><h3 class="card-title">Recent Finished Subjobs</h3></div><div class="card-body">
        {_render_table(["Subjob", "Status", "Asset", "Train Years", "Completed", "Error"], recent_rows)}
      </div></div>
    </div></section>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/admin-lte@3.2/dist/js/adminlte.min.js"></script>
<script>
		  const performanceData = {chart_json};
		  const portfolioData = {portfolio_chart_json};
		  const labels = performanceData.labels || [];
		  const modelData = performanceData.model || {{}};
		  const oracleData = performanceData.oracle || {{}};
		  const riskProfitData = performanceData.risk_profit || {{ points: [] }};
  const refreshSelect = document.getElementById("refreshInterval");
  const refreshStatus = document.getElementById("refreshStatus");
  const refreshOptions = ["300000", "600000", "1800000", "3600000"];
  let refreshTimer = null;
  const refreshLabel = (ms) => {{
    const minutes = Number(ms) / 60000;
    return minutes >= 60 ? `${{minutes / 60}} h` : `${{minutes}} min`;
  }};
  const scheduleRefresh = () => {{
    if (!refreshSelect) return;
    const selected = refreshOptions.includes(refreshSelect.value) ? refreshSelect.value : "300000";
    refreshSelect.value = selected;
    localStorage.setItem("project3DashboardRefreshMs", selected);
    if (refreshTimer) window.clearTimeout(refreshTimer);
    refreshTimer = window.setTimeout(() => window.location.reload(), Number(selected));
    if (refreshStatus) refreshStatus.textContent = `next in ${{refreshLabel(selected)}}`;
  }};
  if (refreshSelect) {{
    const storedRefresh = localStorage.getItem("project3DashboardRefreshMs");
    refreshSelect.value = refreshOptions.includes(storedRefresh) ? storedRefresh : "300000";
    refreshSelect.addEventListener("change", scheduleRefresh);
    scheduleRefresh();
  }}
  const compactLabel = (value) => {{
    if (!value) return "";
    return String(value).replace("+00:00", "Z").replace("T", " ");
  }};
  const ctx = document.getElementById("performanceChart");
		  if (ctx) {{
		    new Chart(ctx, {{
      type: "line",
      data: {{
        labels,
        datasets: [
          {{
	            label: "Champion composite",
	            data: modelData.composite || [],
            borderColor: "#28a745",
            backgroundColor: "rgba(40, 167, 69, 0.08)",
            tension: 0.18,
            borderWidth: 2,
            pointRadius: 0,
            spanGaps: true
          }},
          {{
	            label: "Champion train tail",
	            data: modelData.train || [],
            borderColor: "#007bff",
            backgroundColor: "rgba(0, 123, 255, 0.08)",
            tension: 0.18,
            borderWidth: 2,
            pointRadius: 0,
            spanGaps: true
          }},
          {{
	            label: "Champion validation",
	            data: modelData.validation || [],
            borderColor: "#ffc107",
            backgroundColor: "rgba(255, 193, 7, 0.08)",
            tension: 0.18,
            borderWidth: 2,
            pointRadius: 0,
            spanGaps: true
          }},
          {{
	            label: "Champion test",
	            data: modelData.test || [],
            borderColor: "#ff4d8d",
            backgroundColor: "rgba(255, 77, 141, 0.10)",
            tension: 0.18,
            borderWidth: 3,
            pointRadius: 0,
            spanGaps: true
	          }},
	          {{
		            label: "Avg ZigZag oracle composite",
		            data: oracleData.composite || [],
	            borderColor: "#7dd3fc",
	            backgroundColor: "rgba(125, 211, 252, 0.06)",
	            borderDash: [8, 5],
	            tension: 0,
	            borderWidth: 2,
	            pointRadius: 0,
	            spanGaps: true
	          }},
	          {{
		            label: "Avg ZigZag anti-oracle composite",
		            data: oracleData.anti_composite || [],
	            borderColor: "#ef4444",
	            backgroundColor: "rgba(239, 68, 68, 0.05)",
	            borderDash: [8, 5],
	            tension: 0,
	            borderWidth: 2,
	            pointRadius: 0,
	            spanGaps: true
	          }}
	        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {{ mode: "index", intersect: false }},
        plugins: {{
          legend: {{
            position: "top",
            labels: {{ color: "#d7e0ea", boxWidth: 14, usePointStyle: true }}
          }},
          tooltip: {{
            backgroundColor: "rgba(9, 14, 21, 0.94)",
            titleColor: "#f8fafc",
            bodyColor: "#d7e0ea",
            borderColor: "rgba(148, 163, 184, 0.45)",
            borderWidth: 1,
            callbacks: {{
              title: (items) => compactLabel(items[0]?.label),
	              label: (item) => `${{item.dataset.label}} net total_return: ${{Number(item.parsed.y).toFixed(6)}}`
            }}
          }}
        }},
        scales: {{
          x: {{
            grid: {{ color: "rgba(148, 163, 184, 0.12)" }},
            ticks: {{
              color: "#9fb0c0",
              maxTicksLimit: 10,
              callback: function(value) {{ return compactLabel(this.getLabelForValue(value)).slice(0, 16); }}
            }}
          }},
		          y: {{
		            grid: {{ color: "rgba(148, 163, 184, 0.16)" }},
		            ticks: {{ color: "#9fb0c0", callback: (value) => Number(value).toFixed(4) }},
			            title: {{ display: true, text: "Average weekly net total_return across completed weekly subjobs", color: "#d7e0ea" }}
		          }}
		        }}
	      }}
		    }});
		  }}
	  const riskProfitCtx = document.getElementById("riskProfitChart");
	  if (riskProfitCtx) {{
	    const riskPoints = (riskProfitData.points || []).map((point) => ({{
	      x: Number(point.x || 0),
	      y: Number(point.y || 0),
	      label: point.label || "",
	      asset: point.asset || "",
	      timeframe: point.timeframe || "",
	      model: point.model || "",
	      policy: point.policy || "",
	      samples: point.samples || "",
	      composite: point.composite,
	      riskComposite: point.risk_composite,
	      testRap: point.test_rap,
	      riskLambda: point.risk_lambda,
	      relVolume: point.rel_volume,
	      businessRisk: point.business_risk_fraction,
	      sltpMode: point.sltp_risk_mode || "",
	      sltpProfile: point.sltp_profile_tag || "",
	      kSl: point.k_sl,
	      kTp: point.k_tp,
	      rewardRiskRatio: point.reward_risk_ratio,
	      rewardPlugin: point.reward_plugin || "",
	      selectionMetric: point.selection_metric || ""
	    }}));
	    const maxRisk = Math.max(0.01, ...riskPoints.map((p) => Number(p.x || 0))) * 1.08;
	    const cdt = Number(riskProfitData.cdt_weekly_return || 0);
	    const cdtStretch = Number(riskProfitData.cdt_weekly_stretch_return || 0);
	    new Chart(riskProfitCtx, {{
	      type: "scatter",
	      data: {{
	        datasets: [
	          {{
	            label: "Candidates",
	            data: riskPoints,
	            parsing: false,
	            backgroundColor: "rgba(34, 197, 94, 0.58)",
	            borderColor: "#22c55e",
	            pointRadius: 4,
	            pointHoverRadius: 7
	          }},
	          {{
	            label: "CDT 12.0% EA weekly",
	            type: "line",
	            data: [{{x: 0, y: cdt}}, {{x: maxRisk, y: cdt}}],
	            borderColor: "#facc15",
	            backgroundColor: "rgba(250, 204, 21, 0.10)",
	            borderDash: [8, 5],
	            borderWidth: 2,
	            pointRadius: 0,
	            parsing: false
	          }},
	          {{
	            label: "CDT 12.6% EA weekly",
	            type: "line",
	            data: [{{x: 0, y: cdtStretch}}, {{x: maxRisk, y: cdtStretch}}],
	            borderColor: "#fb923c",
	            backgroundColor: "rgba(251, 146, 60, 0.10)",
	            borderDash: [4, 4],
	            borderWidth: 2,
	            pointRadius: 0,
	            parsing: false
	          }}
	        ]
	      }},
	      options: {{
	        responsive: true,
	        maintainAspectRatio: false,
	        animation: false,
	        plugins: {{
	          legend: {{
	            position: "top",
	            labels: {{ color: "#d7e0ea", boxWidth: 14, usePointStyle: true }}
	          }},
	          tooltip: {{
	            backgroundColor: "rgba(9, 14, 21, 0.94)",
	            titleColor: "#f8fafc",
	            bodyColor: "#d7e0ea",
	            borderColor: "rgba(148, 163, 184, 0.45)",
	            borderWidth: 1,
	            callbacks: {{
	              title: (items) => items[0]?.raw?.label || items[0]?.dataset?.label || "",
	              label: (item) => {{
	                const p = item.raw || {{}};
	                if (!p.label) return `${{item.dataset.label}}: ${{Number(item.parsed.y).toFixed(6)}}`;
	                return [
	                  `${{p.asset}} ${{p.timeframe}} ${{p.model}}`,
	                  `test return=${{Number(p.y).toFixed(6)}} drawdown=${{(100 * Number(p.x)).toFixed(3)}}%`,
	                  `RAP comp=${{p.riskComposite == null ? "" : Number(p.riskComposite).toFixed(6)}} test RAP=${{p.testRap == null ? "" : Number(p.testRap).toFixed(6)}}`,
	                  `lambda=${{p.riskLambda ?? ""}} rel_volume=${{p.relVolume ?? ""}} risk=${{p.businessRisk == null ? "" : (100 * Number(p.businessRisk)).toFixed(1) + "%"}} samples=${{p.samples}}`,
	                  `SL/TP=${{p.sltpMode}} ${{p.sltpProfile}} k_sl=${{p.kSl ?? ""}} k_tp=${{p.kTp ?? ""}} rr=${{p.rewardRiskRatio ?? ""}}`,
	                  `${{p.rewardPlugin}} / ${{p.selectionMetric}}`
	                ];
	              }}
	            }}
	          }}
	        }},
	        scales: {{
	          x: {{
	            min: 0,
	            grid: {{ color: "rgba(148, 163, 184, 0.12)" }},
	            ticks: {{ color: "#9fb0c0", callback: (value) => `${{(100 * Number(value)).toFixed(2)}}%` }},
	            title: {{ display: true, text: "Average weekly test max drawdown", color: "#d7e0ea" }}
	          }},
	          y: {{
	            grid: {{ color: "rgba(148, 163, 184, 0.16)" }},
	            ticks: {{ color: "#9fb0c0", callback: (value) => `${{(100 * Number(value)).toFixed(2)}}%` }},
	            title: {{ display: true, text: "Average weekly test net total_return", color: "#d7e0ea" }}
	          }}
	        }}
	      }}
	    }});
	  }}
	  const portfolioCtx = document.getElementById("portfolioChart");
	  if (portfolioCtx) {{
	    new Chart(portfolioCtx, {{
	      type: "bar",
	      data: {{
	        labels: portfolioData.labels || [],
	        datasets: [
	          {{
	            label: "Mean weekly net return",
	            data: portfolioData.mean_weekly_return || [],
	            backgroundColor: "rgba(34, 197, 94, 0.48)",
	            borderColor: "#22c55e",
	            borderWidth: 1
	          }},
	          {{
	            label: "Cumulative return",
	            data: portfolioData.cumulative_return || [],
	            backgroundColor: "rgba(59, 130, 246, 0.34)",
	            borderColor: "#60a5fa",
	            borderWidth: 1
	          }},
	          {{
	            label: "Max drawdown",
	            data: portfolioData.max_drawdown || [],
	            backgroundColor: "rgba(239, 68, 68, 0.38)",
	            borderColor: "#ef4444",
	            borderWidth: 1
	          }}
	        ]
	      }},
	      options: {{
	        responsive: true,
	        maintainAspectRatio: false,
	        animation: false,
	        plugins: {{
	          legend: {{
	            position: "top",
	            labels: {{ color: "#d7e0ea", boxWidth: 14, usePointStyle: true }}
	          }},
	          tooltip: {{
	            backgroundColor: "rgba(9, 14, 21, 0.94)",
	            titleColor: "#f8fafc",
	            bodyColor: "#d7e0ea",
	            borderColor: "rgba(148, 163, 184, 0.45)",
	            borderWidth: 1,
	            callbacks: {{
	              label: (item) => `${{item.dataset.label}}: ${{(100 * Number(item.parsed.y)).toFixed(3)}}%`
	            }}
	          }}
	        }},
	        scales: {{
	          x: {{
	            grid: {{ color: "rgba(148, 163, 184, 0.12)" }},
	            ticks: {{ color: "#9fb0c0", maxRotation: 35, minRotation: 0 }}
	          }},
	          y: {{
	            grid: {{ color: "rgba(148, 163, 184, 0.16)" }},
	            ticks: {{ color: "#9fb0c0", callback: (value) => `${{(100 * Number(value)).toFixed(2)}}%` }},
	            title: {{ display: true, text: "Portfolio return fraction", color: "#d7e0ea" }}
	          }}
	        }}
	      }}
	    }});
	  }}
		</script>
</body>
</html>"""
    return html_page.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    db_path: Path

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            payload = _status_payload(self.db_path)
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/portfolio":
            conn = _connect(self.db_path)
            try:
                payload = {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "db_path": str(self.db_path),
                    "portfolio": _portfolio_payload(conn),
                }
            finally:
                conn.close()
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path in ("/", "/index.html"):
            body = _page(_status_payload(self.db_path))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    Handler.db_path = Path(args.db)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Project 3 dashboard listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
