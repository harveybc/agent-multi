"""Local-first, weekly-retrained protected promotion evaluation utilities."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


PROMOTION_SCHEMA_VERSION = "agent_multi.weekly_promotion.v1"
WEEKLY_OLAP_SCHEMA_VERSION = "agent_multi.weekly_promotion_olap.v1"
_PARAMETER_SECTIONS = {
    "continuous_action_threshold": "asset_policy",
    "learning_rate": "training",
    "batch_size": "training",
    "gamma": "training",
    "tau": "training",
    "train_freq": "training",
    "gradient_steps": "training",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=None)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class WeekWindow:
    index: int
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime

    @property
    def label(self) -> str:
        return self.test_start.date().isoformat()

    def as_dict(self) -> dict[str, Any]:
        return {
            "week_index": self.index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def build_week_windows(
    *,
    train_start: str,
    protected_test_start: str,
    protected_test_end: str,
    validation_days: int,
) -> list[WeekWindow]:
    """Build Monday-aligned weeks with no training/validation future leakage."""
    if validation_days < 7:
        raise ValueError("validation_days must be at least 7")
    train_begin = _parse_timestamp(train_start)
    test_begin = _parse_timestamp(protected_test_start)
    # Configurations conventionally use 23:59:59. Treat that calendar date as
    # fully available while retaining an exclusive interval internally.
    test_end_exclusive = _parse_timestamp(protected_test_end).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)
    first_monday = test_begin + timedelta(days=(7 - test_begin.weekday()) % 7)
    windows: list[WeekWindow] = []
    current = first_monday
    while current + timedelta(days=7) <= test_end_exclusive:
        validation_start = current - timedelta(days=validation_days)
        if validation_start <= train_begin:
            raise ValueError(
                "Not enough pre-test history for the requested rolling validation window"
            )
        windows.append(
            WeekWindow(
                index=len(windows) + 1,
                train_start=train_begin,
                train_end=validation_start,
                validation_start=validation_start,
                validation_end=current,
                test_start=current,
                test_end=current + timedelta(days=7),
            )
        )
        current += timedelta(days=7)
    if not windows:
        raise ValueError("Protected test period contains no complete Monday-aligned weeks")
    return windows


def _candidate_by_rank(manifest: dict[str, Any], rank: int) -> dict[str, Any]:
    for candidate in manifest.get("candidates", []):
        if candidate.get("promotion_rank") == rank:
            return candidate
    raise ValueError(f"Promotion candidate rank {rank} is not present in manifest")


def _apply_candidate_parameters(config: dict[str, Any], parameters: dict[str, Any]) -> None:
    unknown = sorted(set(parameters) - set(_PARAMETER_SECTIONS))
    if unknown:
        raise ValueError("Unsupported frozen candidate parameters: " + ", ".join(unknown))
    for name, value in parameters.items():
        config.setdefault(_PARAMETER_SECTIONS[name], {})[name] = value


def build_week_config(
    *,
    base_config: dict[str, Any],
    candidate: dict[str, Any],
    window: WeekWindow,
    output_root: Path,
    min_test_rows: int,
) -> dict[str, Any]:
    """Build one canonical, frozen-recipe weekly promotion configuration."""
    config = copy.deepcopy(base_config)
    candidate_id = str(candidate["candidate_id"])
    week_root = output_root / "weeks" / window.label
    _apply_candidate_parameters(config, dict(candidate["parameters"]))

    experiment = config.setdefault("experiment", {})
    experiment["name"] = f"{experiment.get('name', 'asset_policy')}_promotion_{candidate_id[-12:]}_{window.label}"
    experiment["mode"] = "train"
    experiment["quiet_mode"] = True
    experiment["role"] = "weekly_retrained_protected_promotion_evaluation"
    legacy_flat = experiment.setdefault("legacy_flat", {})
    legacy_flat["return_trace_dir"] = str(week_root / "return_traces")

    data = config.setdefault("data", {})
    data.update(
        {
            "train_start": window.train_start.isoformat(),
            "train_end": window.train_end.isoformat(),
            "validation_start": window.validation_start.isoformat(),
            "validation_end": window.validation_end.isoformat(),
            "test_start": window.test_start.isoformat(),
            "test_end": window.test_end.isoformat(),
            "min_split_rows": int(min_test_rows),
        }
    )
    config.setdefault("walk_forward", {}).update(
        {
            "enabled": True,
            "weekly_retraining": True,
            "selection_uses_test": False,
            "promotion_requires_weekly_walk_forward": True,
            "target_week": window.label,
        }
    )
    config.setdefault("training", {}).update(
        {
            "evaluate_test_split": True,
        }
    )
    config.setdefault("optimization", {}).update(
        {
            "enabled": False,
            "optimization_resume": False,
        }
    )
    config.setdefault("artifacts", {}).update(
        {
            "artifact_root": str(week_root),
            "save_model": str(week_root / "policy.zip"),
            "results_file": str(week_root / "results.json"),
            "resolved_config_file": str(week_root / "resolved_config.json"),
            "config_manifest_file": str(week_root / "config_manifest.json"),
            "optimizer_output_file": str(week_root / "optimizer_output.json"),
        }
    )
    config.setdefault("deployment", {}).update(
        {
            "channel": "promotion_evaluation_only",
            "lifecycle": "not_for_live_orders",
            "protected_test_week": window.label,
        }
    )
    config.setdefault("olap", {}).update(
        {
            "enabled": True,
            "metric_schema": "trading.metrics.v1",
            "promotion_schema": WEEKLY_OLAP_SCHEMA_VERSION,
        }
    )
    return config


def weekly_result_from_pipeline(
    *,
    run_id: str,
    candidate_id: str,
    window: WeekWindow,
    config_path: Path,
    results_path: Path,
    model_path: Path,
    keep_weekly_models: bool,
) -> dict[str, Any]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    test = payload.get("splits", {}).get("test")
    if not isinstance(test, dict):
        raise ValueError(f"Missing protected test split in {results_path}")
    total_return = _finite(test.get("total_return"), field="test.total_return")
    drawdown = _finite(
        test.get("max_drawdown_fraction", test.get("max_drawdown_pct", 0.0) / 100.0),
        field="test.max_drawdown_fraction",
    )
    rap = _finite(
        test.get("risk_adjusted_total_return", total_return - drawdown),
        field="test.risk_adjusted_total_return",
    )
    artifact_hash = _sha256_file(model_path) if model_path.exists() else None
    if model_path.exists() and not keep_weekly_models:
        model_path.unlink()
    return {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "week_start": window.test_start.isoformat(),
        "week_end": window.test_end.isoformat(),
        "week_index": window.index,
        "total_return": total_return,
        "risk_adjusted_total_return": rap,
        "max_drawdown_fraction": drawdown,
        "trades_total": int(_finite(test.get("trades_total"), field="test.trades_total")),
        "sharpe_ratio": test.get("sharpe_ratio"),
        "final_equity": test.get("final_equity"),
        "pipeline_results_file": str(results_path),
        "week_config_file": str(config_path),
        "model_artifact_sha256": artifact_hash,
        "model_retained": bool(model_path.exists()),
        "return_trace_file": test.get("return_trace_file"),
    }


def aggregate_weekly_results(rows: Iterable[dict[str, Any]], *, expected_weeks: int) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: item["week_start"])
    if not ordered:
        raise ValueError("Cannot aggregate an empty weekly promotion result")
    weekly_returns = [float(item["total_return"]) for item in ordered]
    weekly_rap = [float(item["risk_adjusted_total_return"]) for item in ordered]
    weekly_drawdowns = [float(item["max_drawdown_fraction"]) for item in ordered]
    equity = 1.0
    peak = 1.0
    endpoint_drawdown = 0.0
    for weekly_return in weekly_returns:
        equity *= 1.0 + weekly_return
        peak = max(peak, equity)
        endpoint_drawdown = max(endpoint_drawdown, 1.0 - equity / peak)
    annual_drawdown = max(max(weekly_drawdowns), endpoint_drawdown)
    annual_return = equity - 1.0
    return {
        "coverage_weeks": len(ordered),
        "expected_weeks": expected_weeks,
        "complete_coverage": len(ordered) == expected_weeks,
        "mean_weekly_return": sum(weekly_returns) / len(weekly_returns),
        "annual_return": annual_return,
        "mean_weekly_rap": sum(weekly_rap) / len(weekly_rap),
        "annual_rap": annual_return - annual_drawdown,
        "mean_weekly_drawdown": sum(weekly_drawdowns) / len(weekly_drawdowns),
        "annual_max_drawdown_fraction": annual_drawdown,
        "annual_drawdown_method": "max_of_per_week_intraperiod_and_compounded_week_end_drawdown",
        "trades_total": sum(int(item["trades_total"]) for item in ordered),
    }


def init_weekly_olap(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS promotion_run_olap (
            run_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            candidate_manifest_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            summary_json TEXT
        );
        CREATE TABLE IF NOT EXISTS promotion_weekly_result_olap (
            run_id TEXT NOT NULL,
            week_start TEXT NOT NULL,
            week_end TEXT NOT NULL,
            week_index INTEGER NOT NULL,
            total_return REAL NOT NULL,
            risk_adjusted_total_return REAL NOT NULL,
            max_drawdown_fraction REAL NOT NULL,
            trades_total INTEGER NOT NULL,
            sharpe_ratio REAL,
            final_equity REAL,
            result_json TEXT NOT NULL,
            PRIMARY KEY (run_id, week_start),
            FOREIGN KEY (run_id) REFERENCES promotion_run_olap(run_id)
        );
        """
    )
    return connection


def upsert_weekly_result(connection: sqlite3.Connection, row: dict[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO promotion_weekly_result_olap (
            run_id, week_start, week_end, week_index, total_return,
            risk_adjusted_total_return, max_drawdown_fraction, trades_total,
            sharpe_ratio, final_equity, result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, week_start) DO UPDATE SET
            week_end=excluded.week_end,
            week_index=excluded.week_index,
            total_return=excluded.total_return,
            risk_adjusted_total_return=excluded.risk_adjusted_total_return,
            max_drawdown_fraction=excluded.max_drawdown_fraction,
            trades_total=excluded.trades_total,
            sharpe_ratio=excluded.sharpe_ratio,
            final_equity=excluded.final_equity,
            result_json=excluded.result_json
        """,
        (
            row["run_id"], row["week_start"], row["week_end"], row["week_index"],
            row["total_return"], row["risk_adjusted_total_return"],
            row["max_drawdown_fraction"], row["trades_total"], row["sharpe_ratio"],
            row["final_equity"], json.dumps(row, sort_keys=True),
        ),
    )
    connection.commit()


def run_week_subprocess(
    *,
    python_bin: str,
    repository_root: Path,
    config_path: Path,
    runtime_overlay: Path,
    log_path: Path,
) -> None:
    command = [
        python_bin,
        "-m",
        "app.main",
        "--load_config",
        str(config_path),
        "--runtime_overlay",
        str(runtime_overlay),
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(f"weekly promotion subprocess failed rc={completed.returncode}; log={log_path}")


def now_utc() -> str:
    return datetime.now(UTC).isoformat()
