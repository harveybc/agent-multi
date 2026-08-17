#!/usr/bin/env python3
"""WP0 (finding AUD-F1-20260817-277): read-only policy-behavior sidecar.

Measures the running identity WITHOUT touching it. This tool opens
return traces and checkpoints for reading only; it never writes into an
experiment output root, never loads sealed 2025, and never signals a
worker. Its own output goes to a separate diagnostic root.

Per WP0 it persists, for every available role trace:

- deterministic raw action min/max/mean/std, quantiles, unique count;
- sign changes, threshold crossings, mapped action proportions and
  mapped-action changes under thresholds 0, 0.001, 0.01, 0.05 and 0.1;
- trades, exposure fraction, return, drawdown and costs;
- the WP1 typed classification;
- and the custody binding: source trace sha256, model sha256 where a
  checkpoint exists, role, seed, cell, experiment identity and host.

Roles are restricted to train_monitor and inner_validation evidence
(plus phase-1 easy monitor traces, which are the decisive
counterexamples). Outer validation stays one-shot and sealed 2025 is
refused by name.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _policy_behavior as pb  # noqa: E402

SCHEMA = "agent_multi.p1lr_policy_behavior_sidecar.v1"

#: Refused by name — a diagnostic may never open the sealed year.
SEALED_MARKERS = ("sealed_test", "sealed", "2025")
#: One-shot: outer validation is not a diagnostic surface.
ONE_SHOT_MARKERS = ("outer_validation",)


class SidecarRefusal(RuntimeError):
    """Typed refusal; the sidecar fails closed rather than reading a
    role it must not touch."""


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_role_allowed(name: str) -> None:
    low = name.lower()
    for marker in SEALED_MARKERS:
        if marker in low:
            raise SidecarRefusal(
                f"REFUSED_SEALED_ROLE: {name} matches sealed marker "
                f"{marker!r}; the sealed year is never opened by a "
                "diagnostic")
    for marker in ONE_SHOT_MARKERS:
        if marker in low:
            raise SidecarRefusal(
                f"REFUSED_ONE_SHOT_ROLE: {name} is outer-validation "
                "evidence and stays one-shot")


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak, worst = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0.0:
            worst = max(worst, (peak - value) / peak)
    return worst


def _f(row: dict, key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def measure_trace(path: Path, *, threshold: float,
                  tolerance: float) -> dict:
    """One trace -> one fully bound measurement row."""
    _assert_role_allowed(path.name)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SidecarRefusal(f"REFUSED_EMPTY_TRACE: {path}")

    split = str(rows[0].get("split", "")) or path.stem
    _assert_role_allowed(split)

    actions = [_f(r, "action_raw") for r in rows]
    actions = [a for a in actions if a is not None]
    positions = [_f(r, "position") for r in rows]
    equity = [e for e in (_f(r, "equity") for r in rows) if e is not None]
    trades = [t for t in (_f(r, "trades") for r in rows) if t is not None]
    costs = [c for c in (_f(r, "trade_cost") for r in rows)
             if c is not None]

    behavior = pb.classify_policy_behavior(
        actions, threshold=threshold, tolerance=tolerance,
        source={"trace": path.name, "split": split})

    exposed = sum(1 for p in positions if p not in (None, 0.0))
    return {
        "split": split,
        "scored_rows": len(rows),
        "first_timestamp": rows[0].get("timestamp"),
        "last_timestamp": rows[-1].get("timestamp"),
        "behavior": behavior,
        "economics": {
            "trades": max(trades) if trades else None,
            "exposure_fraction": (exposed / len(rows)) if rows else None,
            "initial_equity": equity[0] if equity else None,
            "final_equity": equity[-1] if equity else None,
            "total_return": ((equity[-1] / equity[0] - 1.0)
                             if len(equity) > 1 and equity[0] else None),
            "max_drawdown": _max_drawdown(equity),
            "total_cost": math.fsum(costs) if costs else 0.0,
        },
        "custody": {
            "trace_file": str(path),
            "trace_sha256": _sha_file(path),
        },
    }


def scan_cell(cell_dir: Path, *, threshold: float,
              tolerance: float) -> dict:
    """Every readable trace of one cell, plus checkpoint custody."""
    parts = cell_dir.parts
    seed = next((p for p in parts if p.startswith("seed")), None)
    traces = sorted(cell_dir.glob("**/return_traces/*return_trace.csv"))
    measurements, refusals = [], []
    for trace in traces:
        try:
            measurements.append(
                measure_trace(trace, threshold=threshold,
                              tolerance=tolerance))
        except SidecarRefusal as exc:
            refusals.append({"trace": str(trace), "refusal": str(exc)})
    checkpoints = [
        {"file": str(zip_path), "sha256": _sha_file(zip_path)}
        for zip_path in sorted(cell_dir.glob("**/*.zip"))
    ]
    return {
        "seed": seed,
        "cell": cell_dir.name,
        "cell_dir": str(cell_dir),
        "checkpoints": checkpoints,
        "measurements": measurements,
        "refusals": refusals,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WP0 read-only policy-behavior sidecar "
                    "(finding AUD-F1-20260817-277). Reads traces and "
                    "checkpoints; writes only to --out.")
    parser.add_argument("--identity-root", required=True, type=Path,
                        help="experiment identity root to READ")
    parser.add_argument("--out", required=True, type=Path,
                        help="diagnostic JSON to write (never inside "
                             "an experiment output root)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="adapter threshold the normal phase runs "
                             "under (default 0.1)")
    parser.add_argument("--tolerance", type=float,
                        default=pb.DEFAULT_CONSTANCY_TOLERANCE)
    args = parser.parse_args(argv)

    root = args.identity_root.expanduser()
    if not root.is_dir():
        print(json.dumps({"outcome": "REFUSED_NO_IDENTITY_ROOT",
                          "root": str(root)}))
        return 2
    out = args.out.expanduser()
    if str(out.resolve()).startswith(str(root.resolve())):
        print(json.dumps({
            "outcome": "REFUSED_OUTPUT_INSIDE_IDENTITY",
            "detail": "the sidecar never writes into the identity it "
                      "measures"}))
        return 2

    cells = sorted(
        {trace.parent.parent.parent
         for trace in root.glob("seed*/*/**/return_traces/*.csv")}
        | {zip_path.parent.parent
           for zip_path in root.glob("seed*/*/**/*.zip")})
    report = {
        "schema": SCHEMA,
        "experiment_identity": root.name,
        "identity_root": str(root),
        "hostname": socket.gethostname(),
        "threshold": float(args.threshold),
        "constancy_tolerance": float(args.tolerance),
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "cells": [scan_cell(cell, threshold=args.threshold,
                            tolerance=args.tolerance)
                  for cell in cells],
    }
    counts: dict[str, int] = {}
    for cell in report["cells"]:
        for measurement in cell["measurements"]:
            key = measurement["behavior"]["classification"]
            counts[key] = counts.get(key, 0) + 1
    report["classification_counts"] = counts
    report["any_promotable_as_learned_activity"] = any(
        m["behavior"]["promotable_as_learned_activity"]
        for c in report["cells"] for m in c["measurements"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "outcome": "COLLECTED",
        "out": str(out),
        "cells": len(report["cells"]),
        "classification_counts": counts,
        "any_promotable_as_learned_activity":
            report["any_promotable_as_learned_activity"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
