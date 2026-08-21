#!/usr/bin/env python3
"""Correction 2 (order 2026-08-20 20:40): deterministic generator and
validator for WP2_ACTIVITY_PLATEAU_SENSITIVITY_DATASET_2026_08_20.json.

A committed JSON without its derivation path is insufficient. This tool
IS the derivation path:

- ``--validate``: discover every referenced trace, verify its SHA-256
  byte-for-byte, verify its split role (outer/sealed refused), recompute
  its annualized rate, recompute the quantiles and every candidate
  score, and compare against the artifact under the DECLARED canonical
  comparison (sorted-key JSON; rates to 2 decimals, scores to 4 — the
  same rounding the artifact was generated with). Rows referencing
  another host are typed REMOTE_UNVERIFIABLE_ON_THIS_HOST, never
  silently passed: full validation is the union of per-host runs.
- ``--regenerate``: rebuild this host's rows from filesystem discovery
  with the exact measurement rule (final value of the trades column /
  scored years at 2,190 bars/year).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _episodic_activity_fitness as ef  # noqa: E402

BARS_PER_YEAR = 2190.0
FORBIDDEN = ("outer", "sealed")
FIXTURES = {"quasi_passive_1t": (1, -0.0001, 0.0),
            "low_20t": (20, -0.05, 0.1),
            "median_85t": (85, -0.05, 0.1),
            "q75_114t": (114, -0.05, 0.1),
            "max_381t": (381, -0.05, 0.1),
            "over_2000t": (2000, -0.05, 0.1)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(path: Path) -> dict | None:
    """The exact measurement rule used at generation time."""
    low = str(path).lower()
    if any(marker in low for marker in FORBIDDEN):
        return None
    try:
        rows = list(csv.DictReader(path.open()))
    except Exception:
        return None
    if len(rows) < 50:
        return None
    split = rows[0].get("split", "")
    if any(marker in split for marker in FORBIDDEN):
        return None
    try:
        trades = float(rows[-1].get("trades") or 0)
    except Exception:
        return None
    return {"split": split, "rows": len(rows), "closed_trades": trades,
            "annualized_rate": round(
                trades / (len(rows) / BARS_PER_YEAR), 2)}


def discover(base: str) -> list[dict]:
    out = []
    for root in glob.glob(base + "/p1_difficulty_lr_factorial_*"):
        for pat in ("*/seed*/*/attempt-*/return_traces/"
                    "train_epoch_return_trace.csv",
                    "*/seed*/*/attempt-*/return_traces/"
                    "train_tail_epoch_return_trace.csv",
                    "*/seed*/*/attempt-*/return_traces/"
                    "evaluation_return_trace.csv"):
            for name in glob.glob(os.path.join(root, pat)):
                path = Path(name)
                fact = measure(path)
                if fact is None:
                    continue
                out.append({"host": socket.gethostname(),
                            "trace": name.replace(base + "/", ""),
                            "sha256": _sha(path), **fact})
    return out


def candidate_scores(plateau: list[float]) -> dict:
    cfg = {"activity_plateau_low_rate": plateau[0],
           "activity_plateau_high_rate": plateau[1]}
    return {fx: round(ef.evaluate_episode(
        total_return=r, max_drawdown_fraction=dd, sharpe=None,
        closed_trades=t, scored_rows=2190,
        config=cfg)["selection_value"], 4)
        for fx, (t, r, dd) in FIXTURES.items()}


def quantiles(rates: list[float]) -> dict:
    active = sorted(r for r in rates if r > 0)
    n = len(active)
    q = lambda p: active[min(n - 1, int(p * (n - 1)))]  # noqa: E731
    return {"min": active[0], "q25": q(.25), "q50": q(.5),
            "q75": q(.75), "q90": q(.9), "max": active[-1],
            "units": "closed trades per 2190-bar year"}


def validate(artifact_path: Path, base: str) -> dict:
    artifact = json.loads(artifact_path.read_text())
    host = socket.gethostname()
    verified, remote, problems = 0, 0, []
    for row in artifact["source_traces"]["per_trace_references"]:
        if row["host"] != host:
            remote += 1
            continue
        path = Path(base) / row["trace"]
        if not path.is_file():
            problems.append(f"MISSING: {row['trace']}")
            continue
        if _sha(path) != row["sha256"]:
            problems.append(f"SHA_MISMATCH: {row['trace']}")
            continue
        fact = measure(path)
        if fact is None:
            problems.append(f"ROLE_OR_SHAPE_REFUSED: {row['trace']}")
            continue
        if fact["annualized_rate"] != row["annualized_rate"]:
            problems.append(
                f"RATE_MISMATCH: {row['trace']} "
                f"{fact['annualized_rate']} != {row['annualized_rate']}")
            continue
        verified += 1
    stored_rates = [r["annualized_rate"]
                    for r in artifact["source_traces"]
                    ["per_trace_references"]]
    recomputed_q = quantiles(stored_rates)
    q_match = recomputed_q == artifact["measured_annualized_rates"]
    score_problems = []
    for name, spec in artifact["candidates"].items():
        recomputed = candidate_scores(spec["plateau"])
        if recomputed != spec["scores"]:
            score_problems.append(name)
    return {
        "schema": "agent_multi.wp2_dataset_validation.v1",
        "host": host,
        "canonical_comparison": ("sorted-key JSON equality; rates "
                                 "rounded to 2 decimals, scores to 4 — "
                                 "the generation rounding"),
        "rows_verified_on_this_host": verified,
        "rows_remote_unverifiable_on_this_host": remote,
        "row_problems": problems,
        "quantiles_reproduced": q_match,
        "candidate_scores_reproduced": not score_problems,
        "candidate_score_problems": score_problems,
        "valid_for_this_host": (not problems and q_match
                                and not score_problems),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=REPO /
                        "docs/audits/evidence/"
                        "WP2_ACTIVITY_PLATEAU_SENSITIVITY_DATASET_"
                        "2026_08_20.json")
    parser.add_argument("--base", default=os.path.expanduser(
        "~/.local/share/agent-multi"))
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--merge", nargs="+", type=Path, default=None,
                        help="WP2-G: per-host regenerated fragments -> "
                             "one canonical global artifact, compared "
                             "against --artifact without trusting any "
                             "stored rate")
    args = parser.parse_args()
    if args.merge:
        rows = []
        for fragment in args.merge:
            rows.extend(json.loads(fragment.read_text()))
        rows.sort(key=lambda r: (r["host"], r["trace"]))
        rates = [r["annualized_rate"] for r in rows]
        rebuilt_q = quantiles(rates)
        artifact = json.loads(args.artifact.read_text())
        stored_rows = sorted(
            artifact["source_traces"]["per_trace_references"],
            key=lambda r: (r["host"], r["trace"]))
        rebuilt_scores = {name: candidate_scores(spec["plateau"])
                          for name, spec in
                          artifact["candidates"].items()}
        stored_scores = {name: spec["scores"] for name, spec in
                         artifact["candidates"].items()}
        report = {
            "schema": "agent_multi.wp2_global_merge.v1",
            "fragments": [str(f) for f in args.merge],
            "rows_regenerated": len(rows),
            "rows_match_artifact": rows == stored_rows,
            "quantiles_match":
                rebuilt_q == artifact["measured_annualized_rates"],
            "candidate_scores_match": rebuilt_scores == stored_scores,
        }
        report["global_artifact_reproduced"] = all(
            report[k] for k in ("rows_match_artifact",
                                "quantiles_match",
                                "candidate_scores_match"))
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0 if report["global_artifact_reproduced"] else 1
    if args.regenerate:
        print(json.dumps(discover(args.base), indent=1))
        return 0
    report = validate(args.artifact, args.base)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["valid_for_this_host"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
