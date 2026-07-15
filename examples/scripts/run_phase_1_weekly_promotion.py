#!/usr/bin/env python3
"""Run one frozen Phase 1 recipe through its full weekly-retrained test year."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.weekly_promotion import (
    PROMOTION_SCHEMA_VERSION,
    _atomic_write_json,
    _candidate_by_rank,
    _sha256,
    aggregate_weekly_results,
    build_week_config,
    build_week_windows,
    init_weekly_olap,
    now_utc,
    run_week_subprocess,
    select_execution_week_windows,
    select_protocol_week_windows,
    upsert_weekly_result,
    weekly_result_from_pipeline,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-rank", type=int, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--runtime-overlay", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-days", type=int, default=182)
    parser.add_argument("--min-test-rows", type=int, default=36)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--protocol-weeks",
        type=int,
        default=48,
        help="Fixed contiguous promotion horizon; results must not be called a calendar-year result.",
    )
    parser.add_argument(
        "--week-offset",
        type=int,
        default=0,
        help="Zero-based offset within the frozen protocol horizon for distributed shards.",
    )
    parser.add_argument(
        "--max-weeks",
        type=int,
        default=0,
        help="Shard length; 0 means all remaining protocol weeks from --week-offset.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-weekly-models", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    candidate_manifest = _read_json(args.candidate_manifest)
    protected = candidate_manifest.get("selection_scope", {}).get("protected_test", {})
    if protected.get("status") != "not_opened":
        raise ValueError("candidate manifest is not eligible to open a first protected evaluation")
    candidate = _candidate_by_rank(candidate_manifest, args.candidate_rank)
    base_config = _read_json(args.base_config)
    data = base_config.get("data", {})
    all_windows = build_week_windows(
        train_start=data["train_start"],
        protected_test_start=data["test_start"],
        protected_test_end=data["test_end"],
        validation_days=args.validation_days,
    )
    protocol_windows = select_protocol_week_windows(
        all_windows,
        protocol_weeks=args.protocol_weeks,
    )
    windows = select_execution_week_windows(
        protocol_windows,
        week_offset=args.week_offset,
        max_weeks=args.max_weeks,
    )

    run_id = f"{candidate['candidate_id'].split(':', 1)[-1][:16]}_weekly_retrained_2023"
    run_root = args.output_root / run_id
    plan_path = run_root / "promotion_plan.json"
    summary_path = run_root / "promotion_summary.json"
    olap_path = run_root / "promotion_olap.sqlite"
    plan = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": now_utc(),
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "candidate_manifest_hash": _sha256(candidate_manifest),
        "candidate": candidate,
        "base_config": str(args.base_config.resolve()),
        "runtime_overlay": str(args.runtime_overlay.resolve()),
        "protocol_weeks": args.protocol_weeks,
        "protocol_window_start": protocol_windows[0].label,
        "protocol_window_end": protocol_windows[-1].label,
        "execution_week_offset": args.week_offset,
        "execution_weeks": len(windows),
        "validation_days": args.validation_days,
        "min_test_rows": args.min_test_rows,
        "protocol_windows": [window.as_dict() for window in protocol_windows],
        "execution_windows": [window.as_dict() for window in windows],
        "protected_test_rule": (
            "Each target week is opened only after its frozen recipe trains with data ending before that week. "
            "A complete promotion result requires every frozen protocol week."
        ),
    }
    _atomic_write_json(plan_path, plan)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "plan": str(plan_path),
                    "protocol_weeks": len(protocol_windows),
                    "execution_weeks": len(windows),
                    "week_offset": args.week_offset,
                },
                indent=2,
            )
        )
        return 0

    connection = init_weekly_olap(olap_path)
    connection.execute(
        """
        INSERT INTO promotion_run_olap (
            run_id, schema_version, candidate_id, candidate_manifest_hash, created_at, status
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET status=excluded.status
        """,
        (run_id, PROMOTION_SCHEMA_VERSION, candidate["candidate_id"], _sha256(candidate_manifest), now_utc(), "running"),
    )
    connection.commit()
    rows: list[dict] = []
    failures: list[dict] = []
    for window in windows:
        week_root = run_root / "weeks" / window.label
        config = build_week_config(
            base_config=base_config,
            candidate=candidate,
            window=window,
            output_root=run_root,
            min_test_rows=args.min_test_rows,
        )
        config_path = week_root / "config.json"
        _atomic_write_json(config_path, config)
        results_path = Path(config["artifacts"]["results_file"])
        model_path = Path(config["artifacts"]["save_model"])
        try:
            if not results_path.exists():
                week_root.mkdir(parents=True, exist_ok=True)
                run_week_subprocess(
                    python_bin=args.python_bin,
                    repository_root=REPOSITORY_ROOT,
                    config_path=config_path,
                    runtime_overlay=args.runtime_overlay,
                    log_path=week_root / "run.log",
                )
            row = weekly_result_from_pipeline(
                run_id=run_id,
                candidate_id=candidate["candidate_id"],
                window=window,
                config_path=config_path,
                results_path=results_path,
                model_path=model_path,
                keep_weekly_models=args.keep_weekly_models,
            )
            # The durable evidence is the frozen config, structured result,
            # return trace, artifact hash, and local OLAP row. Successful
            # subprocess stdout is redundant and can become very large over
            # a 52-week promotion run.
            (week_root / "run.log").unlink(missing_ok=True)
            rows.append(row)
            upsert_weekly_result(connection, row)
            _atomic_write_json(
                summary_path,
                {
                    "schema_version": PROMOTION_SCHEMA_VERSION,
                    "run_id": run_id,
                    "status": "running",
                    "completed_weeks": len(rows),
                    "expected_weeks": len(windows),
                    "weekly_results": rows,
                    "failures": failures,
                },
            )
        except Exception as exc:
            failure = {"week": window.label, "error": str(exc)}
            failures.append(failure)
            if not args.continue_on_error:
                break

    aggregate = (
        aggregate_weekly_results(rows, expected_weeks=len(protocol_windows)) if rows else None
    )
    trace_complete = bool(
        aggregate
        and aggregate.get("annual_drawdown_method")
        == "observed_concatenated_intraperiod_equity_trace"
    )
    complete = bool(aggregate and aggregate["complete_coverage"] and not failures and trace_complete)
    if complete:
        promotion_blockers = ["RELEASE_VALIDATION_NOT_RUN", "COMPONENT_COMPATIBILITY_NOT_RUN"]
    elif aggregate and aggregate["complete_coverage"] and not failures:
        promotion_blockers = ["MISSING_RETURN_TRACE_EVIDENCE"]
    else:
        promotion_blockers = ["INCOMPLETE_PROTECTED_TEST_COVERAGE"]
    summary = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_id": candidate["candidate_id"],
        "status": "protected_test_complete" if complete else "partial",
        "protected_test": {
            "opened": True,
            "period": protected.get("period"),
            "protocol_weeks": len(protocol_windows),
            "execution_weeks": len(windows),
            "week_offset": args.week_offset,
        },
        "aggregate": aggregate,
        "weekly_results": rows,
        "failures": failures,
        "promotion_blockers": promotion_blockers,
        "promotion_eligible": False,
        "olap_db": str(olap_path),
    }
    _atomic_write_json(summary_path, summary)
    connection.execute(
        "UPDATE promotion_run_olap SET status=?, summary_json=? WHERE run_id=?",
        (summary["status"], json.dumps(summary, sort_keys=True), run_id),
    )
    connection.commit()
    connection.close()
    print(json.dumps({"summary": str(summary_path), "status": summary["status"], "aggregate": aggregate}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
