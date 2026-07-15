#!/usr/bin/env python3
"""Consolidate disjoint weekly-promotion shards into one auditable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.weekly_promotion import (
    PROMOTION_SCHEMA_VERSION,
    _atomic_write_json,
    aggregate_weekly_results,
    init_weekly_olap,
    now_utc,
    upsert_weekly_result,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _local_trace_path(summary_path: Path, row: dict[str, Any]) -> Path:
    configured = row.get("return_trace_file")
    if configured:
        direct = Path(str(configured))
        if direct.is_file():
            return direct
    week_label = str(row["week_start"])[:10]
    derived = summary_path.parent / "weeks" / week_label / "return_traces" / "test_return_trace.csv"
    if not derived.is_file():
        raise FileNotFoundError(
            f"Missing copied return trace for {week_label}; expected {derived}"
        )
    return derived


def _same_evidence(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = (
        "candidate_id",
        "week_start",
        "week_end",
        "total_return",
        "risk_adjusted_total_return",
        "max_drawdown_fraction",
        "trades_total",
        "model_artifact_sha256",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-summary",
        type=Path,
        action="append",
        required=True,
        help="One copied shard promotion_summary.json; specify once per shard.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-weeks", type=int, default=48)
    args = parser.parse_args()
    if args.expected_weeks < 1:
        parser.error("--expected-weeks must be at least 1")

    summaries = [(path, _read_json(path)) for path in args.shard_summary]
    candidate_ids = {str(summary["candidate_id"]) for _, summary in summaries}
    run_ids = {str(summary["run_id"]) for _, summary in summaries}
    if len(candidate_ids) != 1 or len(run_ids) != 1:
        raise ValueError("All shards must belong to exactly one candidate and one promotion run")

    rows_by_week: dict[str, dict[str, Any]] = {}
    for summary_path, summary in summaries:
        protected = summary.get("protected_test", {})
        if int(protected.get("protocol_weeks", args.expected_weeks)) != args.expected_weeks:
            raise ValueError(f"Protocol-week mismatch in {summary_path}")
        for original in summary.get("weekly_results", []):
            if not isinstance(original, dict):
                raise ValueError(f"Invalid weekly result in {summary_path}")
            row = dict(original)
            row["return_trace_file"] = str(_local_trace_path(summary_path, row))
            key = str(row["week_start"])
            previous = rows_by_week.get(key)
            if previous is not None:
                if not _same_evidence(previous, row):
                    raise ValueError(f"Conflicting duplicate evidence for week {key}")
                continue
            rows_by_week[key] = row

    rows = [rows_by_week[key] for key in sorted(rows_by_week)]
    aggregate = aggregate_weekly_results(rows, expected_weeks=args.expected_weeks)
    trace_complete = (
        aggregate["annual_drawdown_method"]
        == "observed_concatenated_intraperiod_equity_trace"
    )
    complete = bool(aggregate["complete_coverage"] and trace_complete)
    blockers = (
        ["RELEASE_VALIDATION_NOT_RUN", "COMPONENT_COMPATIBILITY_NOT_RUN"]
        if complete
        else ["INCOMPLETE_PROTECTED_TEST_COVERAGE"]
    )
    run_id = next(iter(run_ids))
    output_root = args.output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_id": next(iter(candidate_ids)),
        "status": "protected_test_complete" if complete else "partial",
        "protected_test": {
            "opened": True,
            "protocol_weeks": args.expected_weeks,
            "coverage_weeks": len(rows),
        },
        "aggregate": aggregate,
        "weekly_results": rows,
        "source_shards": [str(path.resolve()) for path, _ in summaries],
        "promotion_blockers": blockers,
        "promotion_eligible": False,
        "olap_db": str(output_root / "promotion_olap.sqlite"),
    }
    _atomic_write_json(output_root / "promotion_summary.json", summary)

    connection = init_weekly_olap(output_root / "promotion_olap.sqlite")
    connection.execute(
        """
        INSERT INTO promotion_run_olap (
            run_id, schema_version, candidate_id, candidate_manifest_hash, created_at, status, summary_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, summary_json=excluded.summary_json
        """,
        (
            run_id,
            PROMOTION_SCHEMA_VERSION,
            summary["candidate_id"],
            "consolidated_from_shards",
            now_utc(),
            summary["status"],
            json.dumps(summary, sort_keys=True),
        ),
    )
    for row in rows:
        upsert_weekly_result(connection, row)
    connection.commit()
    connection.close()
    print(json.dumps({"summary": str(output_root / "promotion_summary.json"), "status": summary["status"], "aggregate": aggregate}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
