#!/usr/bin/env python3
"""Build train-only oracle/anti-oracle behavior labels for Project 3 jobs.

This is the first step of the oracle-behavior pretraining lane: it does not
train a model and it never labels validation/test rows.  Labels are generated
from the same pessimistic MT4-style zig-zag oracle used by the dashboard.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from project3_weekly_dashboard import (
    ORACLE_COST_STRESS_MULTIPLIER,
    _apply_oracle_trade,
    _float_or_none,
    _minimum_zigzag_bars,
    _oracle_execution_config,
    _parse_datetime,
    _price_series,
    _score_zigzag_oracle,
    _zigzag_pivots,
    _zigzag_threshold_candidates,
)


SCHEMA_VERSION = "project3_oracle_behavior_labels_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _result_metric(result_json: str | None, key: str) -> float | None:
    try:
        result = json.loads(result_json or "{}")
    except json.JSONDecodeError:
        return None
    value = _float_or_none(result.get(key))
    if value is not None:
        return value
    if key == "train_validation_composite_score":
        train = _float_or_none(result.get("train_tail_total_return"))
        validation = _float_or_none(result.get("validation_total_return"))
        if train is not None and validation is not None:
            return (train + validation) / 2.0
    return None


def _top_jobs(conn: sqlite3.Connection, top_n: int, min_completed_weeks: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT j.external_id AS job_id, j.candidate_id, j.asset, j.timeframe,
               j.model_family, j.train_years, j.training_policy,
               j.input_data_file, j.feature_count, j.config_json,
               s.result_json
        FROM jobs j
        JOIN subjobs s ON s.job_id = j.id
        WHERE s.status='done' AND s.result_json IS NOT NULL
        ORDER BY j.external_id, s.completed_at, s.id
        """
    ).fetchall()
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        job = grouped.setdefault(
            row["job_id"],
            {
                "job_id": row["job_id"],
                "candidate_id": row["candidate_id"],
                "asset": row["asset"],
                "timeframe": row["timeframe"],
                "model_family": row["model_family"],
                "train_years": row["train_years"],
                "training_policy": row["training_policy"],
                "input_data_file": row["input_data_file"],
                "feature_count": row["feature_count"],
                "config_json": row["config_json"],
                "composite": [],
                "train": [],
                "validation": [],
                "test": [],
            },
        )
        for metric_key, bucket in (
            ("train_validation_composite_score", "composite"),
            ("train_tail_total_return", "train"),
            ("validation_total_return", "validation"),
            ("test_total_return", "test"),
        ):
            metric = _result_metric(row["result_json"], metric_key)
            if metric is not None:
                job[bucket].append(metric)

    eligible = []
    for job in grouped.values():
        job["n"] = len(job["composite"])
        job["composite_avg"] = _mean(job["composite"])
        job["train_avg"] = _mean(job["train"])
        job["validation_avg"] = _mean(job["validation"])
        job["test_avg"] = _mean(job["test"])
        if job["n"] >= min_completed_weeks and job["composite_avg"] is not None:
            eligible.append(job)
    return sorted(eligible, key=lambda item: item["composite_avg"], reverse=True)[:top_n]


def _choose_oracle(
    bars: list[tuple[datetime, float, float, float]],
    config_path: str | None,
) -> dict[str, Any]:
    execution = _oracle_execution_config(config_path)
    commission_rate = ORACLE_COST_STRESS_MULTIPLIER * float(execution["commission"] or 0.0)
    price_cost_rate = ORACLE_COST_STRESS_MULTIPLIER * (
        float(execution["slippage"] or 0.0) + 0.5 * float(execution["full_spread"] or 0.0)
    )
    per_side_cost = max(0.0, commission_rate + price_cost_rate)
    depth_bars = _minimum_zigzag_bars(bars)
    backstep_bars = max(1, depth_bars // 3)
    best_score: dict[str, float | int] = {"ideal": 0.0, "anti": 0.0, "cycles": 0, "trades": 0}
    best_threshold: float | None = None
    best_profit_multiple: float | None = None
    best_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))

    for threshold in _zigzag_threshold_candidates(bars, per_side_cost):
        for min_profit_multiple in (2.0, 3.0, 5.0, 10.0):
            score = _score_zigzag_oracle(
                bars,
                threshold,
                execution,
                depth_bars,
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

    return {
        "execution": execution,
        "commission_rate": commission_rate,
        "price_cost_rate": price_cost_rate,
        "per_side_cost": per_side_cost,
        "round_trip_cost": 2.0 * per_side_cost,
        "depth_bars": depth_bars,
        "backstep_bars": backstep_bars,
        "threshold": best_threshold,
        "profit_multiple": best_profit_multiple,
        "score": best_score,
    }


def _label_segments(
    bars: list[tuple[datetime, float, float, float]],
    oracle: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = [
        {
            "oracle_action": 0,
            "anti_oracle_action": 0,
            "oracle_confidence": 0.0,
            "oracle_segment_id": "",
        }
        for _ in bars
    ]
    threshold = oracle.get("threshold")
    profit_multiple = oracle.get("profit_multiple")
    if threshold is None or profit_multiple is None or len(bars) < 2:
        return labels, {"segments": 0, "long_segments": 0, "short_segments": 0}

    pivots = _zigzag_pivots(
        bars,
        float(threshold),
        int(oracle["depth_bars"]),
        int(oracle["backstep_bars"]),
    )
    if len(pivots) < 2:
        return labels, {"segments": 0, "long_segments": 0, "short_segments": 0}

    execution = oracle["execution"]
    initial_cash = float(execution["initial_cash"] or 10000.0)
    cash = initial_cash
    commission_rate = float(oracle["commission_rate"])
    price_cost_rate = float(oracle["price_cost_rate"])
    round_trip_cost = float(oracle["round_trip_cost"])
    required_move = round_trip_cost * float(profit_multiple)
    segments = 0
    long_segments = 0
    short_segments = 0
    legs = list(zip(pivots, pivots[1:]))
    final_idx = len(bars) - 1
    last_idx, last_kind, _last_price = pivots[-1]
    if final_idx > last_idx:
        final_kind = "peak" if last_kind == "valley" else "valley"
        legs.append((pivots[-1], (final_idx, final_kind, bars[-1][3])))

    for (entry_idx, entry_kind, entry_price), (exit_idx, exit_kind, exit_price) in legs:
        if entry_price <= 0 or exit_price <= 0 or exit_idx <= entry_idx:
            continue
        if entry_kind == "valley" and exit_kind == "peak":
            gross_move = (exit_price - entry_price) / entry_price
            action = 1
        elif entry_kind == "peak" and exit_kind == "valley":
            gross_move = (entry_price - exit_price) / entry_price
            action = -1
        else:
            continue
        if gross_move < required_move:
            continue
        direction = "long" if action > 0 else "short"
        next_cash, filled = _apply_oracle_trade(
            cash,
            entry_price,
            exit_price,
            direction,
            execution,
            commission_rate,
            price_cost_rate,
        )
        if not filled or next_cash <= cash:
            continue
        cash = next_cash
        segments += 1
        long_segments += 1 if action > 0 else 0
        short_segments += 1 if action < 0 else 0
        confidence = gross_move / required_move if required_move > 0 else 1.0
        segment_id = f"seg_{segments:04d}_{direction}_{entry_idx}_{exit_idx}"
        for idx in range(entry_idx, exit_idx):
            labels[idx]["oracle_action"] = action
            labels[idx]["anti_oracle_action"] = -action
            labels[idx]["oracle_confidence"] = round(float(confidence), 8)
            labels[idx]["oracle_segment_id"] = segment_id

    return labels, {
        "segments": segments,
        "long_segments": long_segments,
        "short_segments": short_segments,
        "labeled_rows": sum(1 for row in labels if row["oracle_action"] != 0),
        "hold_rows": sum(1 for row in labels if row["oracle_action"] == 0),
        "realized_ideal_return": (cash - initial_cash) / initial_cash,
    }


def _read_train_rows(
    data_path: Path,
    train_start: str,
    train_end: str,
) -> tuple[list[dict[str, str]], list[tuple[datetime, float, float, float]]]:
    start = _parse_datetime(train_start)
    end = _parse_datetime(train_end)
    if start is None or end is None or end <= start:
        raise ValueError(f"invalid train window: {train_start} -> {train_end}")
    rows: list[dict[str, str]] = []
    bars: list[tuple[datetime, float, float, float]] = []
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stamp = _parse_datetime(row.get("DATE_TIME"))
            if stamp is None or not (start <= stamp < end):
                continue
            close = _float_or_none(row.get("CLOSE"))
            high = _float_or_none(row.get("HIGH")) or close
            low = _float_or_none(row.get("LOW")) or close
            if close is None or high is None or low is None or close <= 0 or high <= 0 or low <= 0:
                continue
            rows.append(row)
            bars.append((stamp, max(high, low, close), min(high, low, close), close))
    return rows, bars


def _write_labels(
    output_path: Path,
    source_rows: list[dict[str, str]],
    labels: list[dict[str, Any]],
    oracle: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "DATE_TIME",
        "oracle_action",
        "anti_oracle_action",
        "oracle_confidence",
        "oracle_segment_id",
        "oracle_threshold",
        "oracle_profit_multiple",
        "oracle_round_trip_cost",
        "oracle_depth_bars",
        "oracle_backstep_bars",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source, label in zip(source_rows, labels):
            writer.writerow(
                {
                    "DATE_TIME": source.get("DATE_TIME"),
                    "oracle_action": label["oracle_action"],
                    "anti_oracle_action": label["anti_oracle_action"],
                    "oracle_confidence": label["oracle_confidence"],
                    "oracle_segment_id": label["oracle_segment_id"],
                    "oracle_threshold": oracle.get("threshold"),
                    "oracle_profit_multiple": oracle.get("profit_multiple"),
                    "oracle_round_trip_cost": oracle.get("round_trip_cost"),
                    "oracle_depth_bars": oracle.get("depth_bars"),
                    "oracle_backstep_bars": oracle.get("backstep_bars"),
                }
            )


def _subjobs_for_job(conn: sqlite3.Connection, job_id: str, limit: int | None) -> list[sqlite3.Row]:
    sql = """
        SELECT s.*, j.external_id AS job_external_id, j.asset, j.timeframe,
               j.input_data_file, j.config_json
        FROM subjobs s
        JOIN jobs j ON j.id = s.job_id
        WHERE j.external_id = ? AND s.status='done'
        ORDER BY s.weekly_anchor_id, s.id
    """
    rows = conn.execute(sql, (job_id,)).fetchall()
    if limit and limit > 0:
        return rows[:limit]
    return rows


def build_labels(
    db_path: Path,
    output_dir: Path,
    top_n: int,
    min_completed_weeks: int,
    max_subjobs_per_job: int | None,
) -> dict[str, Any]:
    conn = _connect(db_path)
    now = _utc_now()
    top_jobs = _top_jobs(conn, top_n=top_n, min_completed_weeks=min_completed_weeks)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "db_path": str(db_path),
        "output_dir": str(output_dir),
        "stage_c_access": "DENIED",
        "training_launched": False,
        "selection_rule": "top jobs by mean train_validation_composite_score; labels are train-window only",
        "top_n": top_n,
        "min_completed_weeks": min_completed_weeks,
        "jobs": [],
    }
    with conn:
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (
                "oracle_label_batch_started",
                "oracle_behavior_pretraining",
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "top_n": top_n,
                        "min_completed_weeks": min_completed_weeks,
                        "output_dir": str(output_dir),
                    },
                    sort_keys=True,
                ),
                now,
            ),
        )

    for job in top_jobs:
        job_dir = output_dir / job["job_id"]
        subjob_summaries = []
        for subjob in _subjobs_for_job(conn, job["job_id"], max_subjobs_per_job):
            data_path = Path(subjob["input_data_file"])
            source_rows, bars = _read_train_rows(data_path, subjob["train_start"], subjob["train_end"])
            oracle = _choose_oracle(bars, subjob["config_path"])
            labels, label_summary = _label_segments(bars, oracle)
            label_path = job_dir / f"{subjob['external_id']}_oracle_behavior_labels.csv"
            _write_labels(label_path, source_rows, labels, oracle)
            subjob_summaries.append(
                {
                    "subjob_id": subjob["external_id"],
                    "weekly_anchor_id": subjob["weekly_anchor_id"],
                    "train_start": subjob["train_start"],
                    "train_end": subjob["train_end"],
                    "validation_start": subjob["validation_start"],
                    "validation_end": subjob["validation_end"],
                    "test_start": subjob["test_start"],
                    "test_end": subjob["test_end"],
                    "train_rows": len(source_rows),
                    "label_path": str(label_path),
                    "oracle_return": oracle["score"]["ideal"],
                    "anti_oracle_return": oracle["score"]["anti"],
                    "oracle_cycles": oracle["score"]["cycles"],
                    "oracle_trades": oracle["score"]["trades"],
                    "oracle_threshold": oracle.get("threshold"),
                    "oracle_profit_multiple": oracle.get("profit_multiple"),
                    "oracle_round_trip_cost": oracle.get("round_trip_cost"),
                    **label_summary,
                }
            )
        job_summary = {
            **{key: job[key] for key in (
                "job_id",
                "candidate_id",
                "asset",
                "timeframe",
                "model_family",
                "train_years",
                "training_policy",
                "input_data_file",
                "feature_count",
                "n",
                "train_avg",
                "validation_avg",
                "test_avg",
                "composite_avg",
            )},
            "labels_dir": str(job_dir),
            "subjobs": subjob_summaries,
        }
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "oracle_behavior_label_manifest.json").write_text(
            json.dumps(job_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest["jobs"].append(job_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "oracle_behavior_label_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with conn:
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (
                "oracle_label_batch_completed",
                "oracle_behavior_pretraining",
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "jobs": len(manifest["jobs"]),
                        "subjobs": sum(len(job["subjobs"]) for job in manifest["jobs"]),
                        "manifest_path": str(manifest_path),
                    },
                    sort_keys=True,
                ),
                _utc_now(),
            ),
        )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(
            "/home/harveybc/Documents/GitHub/financial-data/experiments/weekly_walkforward_pool/project3_weekly_pool.sqlite"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/harveybc/Documents/GitHub/financial-data/experiments/oracle_behavior_pretraining/labels"
        ),
    )
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-completed-weeks", type=int, default=5)
    parser.add_argument(
        "--max-subjobs-per-job",
        type=int,
        default=0,
        help="0 means all completed subjobs for each selected job.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_labels(
        db_path=args.db,
        output_dir=args.output_dir,
        top_n=max(1, args.top_n),
        min_completed_weeks=max(1, args.min_completed_weeks),
        max_subjobs_per_job=args.max_subjobs_per_job if args.max_subjobs_per_job > 0 else None,
    )
    print(json.dumps(
        {
            "manifest_path": manifest["manifest_path"],
            "jobs": len(manifest["jobs"]),
            "subjobs": sum(len(job["subjobs"]) for job in manifest["jobs"]),
            "training_launched": manifest["training_launched"],
            "stage_c_access": manifest["stage_c_access"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
