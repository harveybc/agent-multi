#!/usr/bin/env python3
"""AdminLTE dashboard for the Project 3 weekly walk-forward pool."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ORACLE_DEFAULT_FULL_SPREAD = 0.0004
ORACLE_COST_STRESS_MULTIPLIER = 2.0
CHART_FOCUS_START_LABEL = "2026-06-09 12:00:00Z"

_BASELINE_CACHE: dict[tuple, dict[str, float | None]] = {}
_PRICE_SERIES_CACHE: dict[tuple, list[tuple[datetime, float, float, float]]] = {}


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


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_metric(value: object) -> str:
    metric = _float_or_none(value)
    if metric is None:
        return ""
    return f"{metric:+.6f}"


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
    if start_idx <= 0:
        chart["focus_start"] = CHART_FOCUS_START_LABEL
        chart["trimmed_points"] = 0
        return chart

    def trim_group(group: dict) -> dict:
        trimmed: dict = {}
        for key, value in group.items():
            trimmed[key] = value[start_idx:] if isinstance(value, list) else value
        return trimmed

    chart["labels"] = labels[start_idx:]
    chart["model"] = trim_group(dict(chart.get("model") or {}))
    chart["oracle"] = trim_group(dict(chart.get("oracle") or {}))
    chart["focus_start"] = CHART_FOCUS_START_LABEL
    chart["trimmed_points"] = start_idx
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

    for row in rows:
        try:
            result = json.loads(row.get("result_json") or "{}")
        except json.JSONDecodeError:
            result = {}
        train = _float_or_none(result.get("train_tail_total_return"))
        validation = _float_or_none(result.get("validation_total_return"))
        test = _float_or_none(result.get("test_total_return"))
        validation_sharpe = _float_or_none(result.get("validation_sharpe"))
        test_sharpe = _float_or_none(result.get("test_sharpe"))
        composite = _float_or_none(result.get("train_validation_composite_score"))
        if composite is None and train is not None and validation is not None:
            composite = (train + validation) / 2
        _commission, _slippage, _full_spread, train_tail_days = _config_costs(row.get("config_path"))
        train_end = _parse_datetime(row.get("train_end"))
        train_tail_start = train_end - timedelta(days=train_tail_days) if train_end else None
        train_baseline = _oracle_baseline_for_window(
            row.get("input_data_file"),
            train_tail_start,
            train_end,
            row.get("config_path"),
        )
        validation_baseline = _oracle_baseline_for_window(
            row.get("input_data_file"),
            row.get("validation_start"),
            row.get("validation_end"),
            row.get("config_path"),
        )
        test_baseline = _oracle_baseline_for_window(
            row.get("input_data_file"),
            row.get("test_start"),
            row.get("test_end"),
            row.get("config_path"),
        )
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

        completed = str(row.get("completed_at") or "")
        label = completed.replace("T", " ").replace("+00:00", "Z")

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
                "n": 0,
                "train": [],
                "validation": [],
                "test": [],
                "composite": [],
                "validation_sharpe": [],
                "test_sharpe": [],
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
            },
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
            if candidate["n"] >= min_samples and candidate["composite"]
        ]
        champion = max(
            eligible_running,
            key=lambda candidate: _mean(candidate["composite"]) or float("-inf"),
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

    eligible_composite = [row for row in summaries if row["n"] >= min_samples and row["composite_avg"] is not None]
    best_composite_job = max(eligible_composite, key=lambda row: row["composite_avg"], default=None)

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
        },
    )

    return {
        "chart": chart,
        "best_composite_job": best_composite_job,
        "job_summaries": sorted(
            summaries,
            key=lambda row: (row["composite_avg"] is not None, row["composite_avg"] or float("-inf")),
            reverse=True,
        )[:20],
        "min_samples": min_samples,
    }


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
          <dt class="col-5">Composite</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("composite_avg")))}</dd>
          <dt class="col-5">Train tail</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("train_avg")))}</dd>
          <dt class="col-5">Validation</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("validation_avg")))}</dd>
          <dt class="col-5">Test</dt><dd class="col-7">{html.escape(_fmt_metric(job.get("test_avg")))}</dd>
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
    }
    chart_json = json.dumps(chart_payload)
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
      <p class="project-subtitle mb-0">Live orchestration view for weekly walk-forward SAC experiments. Current ranking uses net total_return only: avg(train-tail total_return, validation total_return) with a trade gate; Sharpe/risk metrics are tracked separately and are not the plotted composite line.</p>
      <span class="db-pill"><i class="fas fa-database mr-1"></i>{html.escape(payload["db_path"])}</span>
    </div></section>
    <section class="content"><div class="container-fluid">
      <div class="row">{card_html}</div>
      <div class="row">
        <div class="col-lg-9">
          <div class="card performance-chart-card">
            <div class="card-header">
              <h3 class="card-title">Running Best Candidate Average Weekly Returns vs Composite Baselines</h3>
              <div class="card-tools"><span class="badge badge-info">Each point is the best candidate by average composite across its completed weekly subjobs</span>{focus_badge}</div>
            </div>
            <div class="card-body">
              <p class="chart-explainer">
                Each point is recalculated after a weekly subjob finishes. The chart selects the candidate with the highest average composite over its completed weekly retrain/fine-tune subjobs, then plots that candidate's average weekly train-tail return, average weekly validation return, average weekly test return, and average composite. Train-tail is the last 7 days of each training window; validation is the next 7 days; test is the following 7 days. Composite is avg(train-tail return, validation return) per week, then averaged across the candidate's completed weeks. The ZigZag oracle and anti-oracle lines are averaged composite baselines for that same currently best candidate.
              </p>
              <div class="chart-canvas-wrap">
                <canvas id="performanceChart"></canvas>
              </div>
            </div>
          </div>
        </div>
        <div class="col-lg-3 best-side-card">
          {_best_job_card("Best Composite So Far", performance.get("best_composite_job"), "success", "composite_avg")}
        </div>
      </div>
      <div class="card"><div class="card-header"><h3 class="card-title">Machines</h3></div><div class="card-body">
        {_render_table(["Machine", "Status", "Subjob", "Heartbeat", "GPU", "Message"], machine_rows)}
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
	  const labels = performanceData.labels || [];
	  const modelData = performanceData.model || {{}};
	  const oracleData = performanceData.oracle || {{}};
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
