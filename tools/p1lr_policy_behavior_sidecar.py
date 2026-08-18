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
    """Refuse a forbidden role. ``name`` may be a resolved path, a
    manifest role or a split label; EVERY path component is checked,
    because a sealed year can hide in a parent directory."""
    text = str(name)
    parts = [text.lower()] + [p.lower() for p in Path(text).parts]
    for marker in SEALED_MARKERS:
        if any(marker in part for part in parts):
            raise SidecarRefusal(
                f"REFUSED_SEALED_ROLE: {text} matches sealed marker "
                f"{marker!r} in its resolved path or role; the sealed "
                "year is never opened by a diagnostic")
    for marker in ONE_SHOT_MARKERS:
        if any(marker in part for part in parts):
            raise SidecarRefusal(
                f"REFUSED_ONE_SHOT_ROLE: {text} is outer-validation "
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


def _resolve_role(path: Path) -> dict:
    """Role authority is the trace ``.meta.json`` and the nested split
    manifest — never the free-text CSV column or the filename."""
    out: dict = {"role": None, "role_source": None,
                 "data_file": None, "data_file_sha256": None,
                 "config_sha256": None, "experiment_identity": None,
                 "observation_contract_sha256": None}
    meta_path = Path(str(path) + ".meta.json")
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except ValueError as error:
            raise SidecarRefusal(
                f"REFUSED_UNREADABLE_META: {meta_path}: {error}")
        out["role"] = meta.get("nested_role") or meta.get("split")
        out["role_source"] = str(meta_path)
        out["meta_sha256"] = _sha_file(meta_path)
        out["data_file"] = meta.get("input_data_file") or meta.get("data_file")
        out["data_file_sha256"] = meta.get("data_file_sha256")
        out["config_sha256"] = (meta.get("_run_config_hash")
                                or meta.get("config_sha256"))
        out["observation_contract_sha256"] = meta.get(
            "observation_contract_sha256")
    manifest = None
    for parent in path.parents:
        candidate = parent / "nested_splits" / "nested_split_manifest.json"
        if candidate.is_file():
            manifest = candidate
            break
    if manifest is not None:
        try:
            data = json.loads(manifest.read_text())
        except ValueError as error:
            raise SidecarRefusal(
                f"REFUSED_UNREADABLE_MANIFEST: {manifest}: {error}")
        out["manifest_file"] = str(manifest)
        out["manifest_sha256"] = _sha_file(manifest)
        out["experiment_identity"] = data.get("experiment_identity")
        roles = data.get("roles") or data.get("role_facts") or {}
        if out["role"] and isinstance(roles, dict):
            entry = roles.get(out["role"]) or {}
            out["data_file"] = out["data_file"] or entry.get("csv")
            out["data_file_sha256"] = (out["data_file_sha256"]
                                       or entry.get("csv_sha256"))
    if not out["role"]:
        raise SidecarRefusal(
            f"REFUSED_UNRESOLVED_ROLE: {path} has no trace meta and no "
            "nested split manifest; a free-text CSV column and a "
            "filename are not role authority")
    return out


def _required_float(row: dict, key: str, index: int) -> float:
    """One complete, schema-valid value per scored row. A malformed
    value REFUSES the measurement; it is never dropped."""
    raw = row.get(key)
    if raw is None or str(raw).strip() == "":
        raise SidecarRefusal(
            f"REFUSED_MISSING_VALUE: column {key!r} is empty at row "
            f"{index}; a missing value makes the metric unavailable, "
            "it is never treated as zero")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise SidecarRefusal(
            f"REFUSED_MALFORMED_VALUE: column {key!r} = {raw!r} at row "
            f"{index} is not a number")
    if not math.isfinite(value):
        raise SidecarRefusal(
            f"REFUSED_NONFINITE_VALUE: column {key!r} = {raw!r} at row "
            f"{index}")
    return value


def _optional_metric(rows: list[dict], key: str) -> list[float] | None:
    """A metric is available only when EVERY scored row carries a
    complete valid value; otherwise the metric is unavailable (None) —
    never partially summed."""
    out: list[float] = []
    for index, row in enumerate(rows):
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        out.append(value)
    return out


def _stable_read(path: Path, attempts: int = 3) -> tuple[list[dict], str]:
    """Hash before and after the read; refuse if the live file moved."""
    last: str | None = None
    for _ in range(max(1, attempts)):
        before = _sha_file(path)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        after = _sha_file(path)
        if before == after:
            return rows, after
        last = f"{before}->{after}"
    raise SidecarRefusal(
        f"REFUSED_UNSTABLE_SOURCE: {path} changed while being read "
        f"({last}); a moving file cannot be measured")


def measure_trace(path: Path, *, threshold: float,
                  tolerance: float,
                  model_sha256: str | None = None,
                  model_file: str | None = None,
                  code_revision: str | None = None) -> dict:
    """One trace -> one fully bound measurement row."""
    path = Path(path).resolve()
    _assert_role_allowed(path)                      # full resolved path
    resolved = _resolve_role(path)
    _assert_role_allowed(str(resolved["role"]))     # manifest role

    rows, trace_sha = _stable_read(path)
    if not rows:
        raise SidecarRefusal(f"REFUSED_EMPTY_TRACE: {path}")

    actions = [_required_float(row, "action_raw", i)
               for i, row in enumerate(rows)]
    positions = [_required_float(row, "position", i)
                 for i, row in enumerate(rows)]
    equity = _optional_metric(rows, "equity")
    trades = _optional_metric(rows, "trades")
    costs = _optional_metric(rows, "trade_cost")

    behavior = pb.classify_policy_behavior(
        actions, threshold=threshold, tolerance=tolerance,
        source={"trace": path.name, "role": resolved["role"]})

    exposed = sum(1 for p in positions if p != 0.0)
    return {
        "role": resolved["role"],
        "scored_rows": len(rows),
        "first_timestamp": rows[0].get("timestamp"),
        "last_timestamp": rows[-1].get("timestamp"),
        "behavior": behavior,
        "promotable": bool(model_sha256) and behavior[
            "promotable_as_learned_activity"],
        "promotable_note": (
            None if model_sha256 else
            "no load-tested model checkpoint was bound to this "
            "measurement; the result is NON-PROMOTABLE regardless of "
            "its behavior class"),
        "economics": {
            "trades": (max(trades) if trades else None),
            "exposure_fraction": exposed / len(rows),
            "initial_equity": (equity[0] if equity else None),
            "final_equity": (equity[-1] if equity else None),
            "total_return": ((equity[-1] / equity[0] - 1.0)
                             if equity and equity[0] else None),
            "max_drawdown": (_max_drawdown(equity) if equity else None),
            "total_cost": (math.fsum(costs) if costs is not None else None),
        },
        "custody": {
            "trace_file": str(path),
            "trace_sha256": trace_sha,
            "meta_sha256": resolved.get("meta_sha256"),
            "manifest_file": resolved.get("manifest_file"),
            "manifest_sha256": resolved.get("manifest_sha256"),
            "role": resolved["role"],
            "role_source": resolved["role_source"],
            "data_file": resolved["data_file"],
            "data_file_sha256": resolved["data_file_sha256"],
            "config_sha256": resolved["config_sha256"],
            "experiment_identity": resolved["experiment_identity"],
            "observation_contract_sha256":
                resolved["observation_contract_sha256"],
            "model_file": model_file,
            "model_sha256": model_sha256,
            "code_revision": code_revision,
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
