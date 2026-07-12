#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "selection_role",
    "asset",
    "timeframe",
    "horizon_bucket",
    "model_family",
    "experiment_phase",
    "input_family",
    "unique_test_weeks",
    "has_near_full_year_coverage",
    "mean_weekly_return_pct",
    "annual_return_pct",
    "mean_weekly_rap_pct",
    "annual_rap_pct",
    "mean_weekly_drawdown_pct",
    "mean_weekly_test_trades",
    "rel_volume",
    "sltp_risk_mode",
    "k_sl",
    "k_tp",
    "risk_penalty_lambda",
    "train_years",
    "training_policy",
    "candidate_id",
    "olap_profile_key",
}

NUMERIC_FIELDS = {
    "mean_weekly_return_pct",
    "annual_return_pct",
    "mean_weekly_rap_pct",
    "annual_rap_pct",
    "mean_weekly_drawdown_pct",
    "mean_weekly_test_trades",
    "rel_volume",
    "k_sl",
    "k_tp",
    "risk_penalty_lambda",
}

FX_QUOTES = ("USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot resolve source repository commit: {repo}") from exc


def canonical_asset_id(symbol: str) -> str:
    upper = symbol.strip().upper()
    if upper.endswith("USDT") and len(upper) > 4:
        return f"crypto:{upper[:-4]}/USDT"
    if len(upper) == 6 and upper[-3:] in FX_QUOTES:
        return f"fx:{upper[:3]}/{upper[3:]}"
    return f"research:{upper}"


def _relative_to_repo(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence file is outside source repository: {path}") from exc


def _parse_float(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    return None if raw == "" else float(raw)


def _evidence_file(path: Path, repo: Path) -> dict[str, Any]:
    return {
        "path": _relative_to_repo(path, repo),
        "content_hash": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_manifest(
    *,
    shortlist_path: Path,
    selection_report_path: Path,
    proposed_selection_path: Path,
    source_repo: Path,
    generated_at: str,
    commit: str | None = None,
) -> dict[str, Any]:
    timestamp = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")

    with shortlist_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError("shortlist is missing columns: " + ", ".join(missing))
        rows = list(reader)

    candidates: list[dict[str, Any]] = []
    for row in rows:
        symbol = row["asset"].strip().upper()
        asset_id = canonical_asset_id(symbol)
        timeframe = row["timeframe"].strip()
        data_profile = row["input_family"].strip()
        policy_role = row["model_family"].strip()
        weeks = int(row["unique_test_weeks"])
        near_full = row["has_near_full_year_coverage"].strip() in {"1", "true", "True"}
        metrics = {key.removesuffix("_pct") + "_percent": _parse_float(row, key) for key in NUMERIC_FIELDS if key.endswith("_pct")}
        metrics["mean_weekly_test_trades"] = _parse_float(row, "mean_weekly_test_trades")
        candidates.append(
            {
                "selection_role": row["selection_role"].strip(),
                "asset_id": asset_id,
                "research_symbol": symbol,
                "timeframe": timeframe,
                "cell_id": f"{asset_id}@{timeframe}:{data_profile}:{policy_role}",
                "horizon_bucket": row["horizon_bucket"].strip(),
                "model_family": policy_role,
                "experiment_phase": row["experiment_phase"].strip(),
                "input_family": data_profile,
                "evidence_scope": "near_full_year" if near_full else "partial",
                "coverage_weeks": weeks,
                "metrics": metrics,
                "risk_profile": {
                    "rel_volume": _parse_float(row, "rel_volume"),
                    "sltp_risk_mode": row["sltp_risk_mode"].strip(),
                    "k_sl": _parse_float(row, "k_sl"),
                    "k_tp": _parse_float(row, "k_tp"),
                    "risk_penalty_lambda": _parse_float(row, "risk_penalty_lambda"),
                },
                "training": {
                    "train_years": int(row["train_years"]),
                    "policy": row["training_policy"].strip(),
                },
                "candidate_id": row["candidate_id"].strip(),
                "olap_profile_key": row["olap_profile_key"].strip(),
            }
        )

    if not candidates:
        raise ValueError("shortlist contains no candidates")
    identities = [candidate["candidate_id"] for candidate in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError("shortlist contains duplicate candidate_id values")

    return {
        "schema_version": "project3_shortlist_import.v1",
        "manifest_id": "project3-doin-shortlist-2026-07-10",
        "generated_at": timestamp.isoformat(),
        "source_repository": {
            "name": "financial-data",
            "commit": commit or source_commit(source_repo),
        },
        "evidence_files": [
            _evidence_file(shortlist_path, source_repo),
            _evidence_file(proposed_selection_path, source_repo),
            _evidence_file(selection_report_path, source_repo),
        ],
        "selection_policy": {
            "use": "research_seed_only",
            "protected_test_rule": "Imported test evidence may seed the search universe but cannot select, early-stop, optimize or promote future candidates.",
            "partial_evidence_rule": "Partial candidates retain exact observed week coverage and cannot carry annual labels.",
            "portfolio_requirement": "Eligibility requires comparable full validation coverage and marginal diversification evidence.",
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the Project 3 DOIN shortlist.")
    parser.add_argument("--shortlist", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--proposed-selection", required=True, type=Path)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        shortlist_path=args.shortlist,
        selection_report_path=args.selection_report,
        proposed_selection_path=args.proposed_selection,
        source_repo=args.source_repo,
        generated_at=args.generated_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"wrote {args.output} ({manifest['candidate_count']} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
