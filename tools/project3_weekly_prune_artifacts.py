#!/usr/bin/env python3
"""Prune reproducible Project 3 weekly-pool run artifacts.

The OLAP database stores the metrics and artifact metadata. This tool frees disk
by deleting bulky reproducible files while preserving models, results, configs,
and evidence. It skips currently running subjobs by default.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB = Path(
    "/home/harveybc/Documents/GitHub/financial-data/experiments/"
    "weekly_walkforward_pool/project3_weekly_pool.sqlite"
)
DEFAULT_RUNS_ROOT = Path(
    "/home/harveybc/Documents/GitHub/agent-multi/experiments/weekly_walkforward_pool/runs"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_active_run_dirs(db: Path) -> set[Path]:
    if not db.exists():
        return set()
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT run_dir FROM subjobs WHERE status='running' AND run_dir IS NOT NULL").fetchall()
    finally:
        conn.close()
    return {Path(str(row[0])).resolve() for row in rows}


def candidates_for_run_dir(run_dir: Path, args: argparse.Namespace) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if args.return_traces:
        trace_root = run_dir / "return_traces"
        if trace_root.exists():
            for path in trace_root.rglob("*.csv"):
                if path.is_file():
                    out.append(("return_trace_csv", path))
    if args.context_embedding_csv:
        path = run_dir / "context_embedding" / "input_with_context_embedding.csv"
        if path.is_file():
            out.append(("context_embedding_csv", path))
    if args.logs:
        for path in run_dir.glob("subprocess_stdout*.log"):
            if path.is_file():
                out.append(("stdout_log", path))
    if args.training_progress:
        path = run_dir / "training_progress.json"
        if path.is_file():
            out.append(("training_progress_json", path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--return-traces", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--context-embedding-csv", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--logs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--training-progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-active", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-run-dir", action="append", type=Path, default=[])
    args = ap.parse_args()

    active_dirs = load_active_run_dirs(args.db) if args.skip_active else set()
    active_dirs.update(path.resolve() for path in args.skip_run_dir)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    skipped_active = 0
    deleted = 0
    errors: list[dict[str, str]] = []

    for run_dir in sorted(args.runs_root.iterdir()) if args.runs_root.exists() else []:
        if not run_dir.is_dir():
            continue
        resolved = run_dir.resolve()
        if resolved in active_dirs:
            skipped_active += 1
            continue
        for kind, path in candidates_for_run_dir(run_dir, args):
            try:
                size = path.stat().st_size
            except OSError as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            stats[kind]["files"] += 1
            stats[kind]["bytes"] += size
            if args.execute:
                try:
                    path.unlink()
                    deleted += 1
                except OSError as exc:
                    errors.append({"path": str(path), "error": str(exc)})

    payload = {
        "generated_at": utc_now(),
        "mode": "execute" if args.execute else "dry_run",
        "runs_root": str(args.runs_root),
        "skip_active": bool(args.skip_active),
        "skipped_active_run_dirs": skipped_active,
        "deleted_files": deleted,
        "stats": {
            kind: {
                "files": item["files"],
                "bytes": item["bytes"],
                "gib": round(item["bytes"] / 1024**3, 4),
            }
            for kind, item in sorted(stats.items())
        },
        "total_gib": round(sum(item["bytes"] for item in stats.values()) / 1024**3, 4),
        "errors": errors[:20],
        "error_count": len(errors),
        "preserved_by_design": [
            "policy.zip",
            "results.json",
            "config_out.json",
            "evidence.json",
            "context_embedding_manifest.json",
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
