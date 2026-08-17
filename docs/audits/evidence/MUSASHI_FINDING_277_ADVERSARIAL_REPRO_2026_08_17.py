#!/usr/bin/env python3
"""Independent adversarial reproducer for findings 278 and 279.

This script imports the implementation from an explicit checkout, creates
only temporary fixtures, and never reads or writes a campaign identity.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_trace(path: Path) -> None:
    path.parent.mkdir(parents=True)
    rows = [
        {
            "timestamp": "2025-01-01T00:00:00Z",
            "split": "evaluation",
            "action_raw": "0.2",
            "position": "1",
            "equity": "100",
            "trades": "1",
            "trade_cost": "",
        },
        {
            "timestamp": "2025-01-01T04:00:00Z",
            "split": "evaluation",
            "action_raw": "corrupt",
            "position": "corrupt",
            "equity": "101",
            "trades": "1",
            "trade_cost": "corrupt",
        },
        {
            "timestamp": "2025-01-01T08:00:00Z",
            "split": "evaluation",
            "action_raw": "0.3",
            "position": "1",
            "equity": "102",
            "trades": "1",
            "trade_cost": "",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.implementation_root.resolve()

    pb = _load(
        "pipeline_plugins._policy_behavior",
        root / "pipeline_plugins" / "_policy_behavior.py",
    )
    # The sidecar imports pipeline_plugins._policy_behavior by package name.
    sys.path.insert(0, str(root))
    sidecar = _load(
        "musashi_policy_sidecar",
        root / "tools" / "p1lr_policy_behavior_sidecar.py",
    )

    corrupted = pb.classify_policy_behavior(
        [0.2, math.nan, "corrupt", 0.3], threshold=0.1
    )
    zero = pb.classify_policy_behavior(
        [0.0, 0.0], threshold=0.0, stochastic_actions=[0.0, 0.0]
    )
    varying_without_observations = pb.classify_policy_behavior(
        [-0.2, 0.2, -0.2, 0.2], threshold=0.1
    )

    with tempfile.TemporaryDirectory(prefix="musashi-277-repro-") as tmp:
        trace = (
            Path(tmp)
            / "sealed_test_2025"
            / "return_traces"
            / "evaluation_return_trace.csv"
        )
        _write_trace(trace)
        sidecar_result = sidecar.measure_trace(
            trace, threshold=0.1, tolerance=1e-6
        )

    checks = {
        "classifier_rejects_corrupted_sequence": (
            corrupted["classification"] == pb.UNAVAILABLE
        ),
        "classifier_preserves_input_cardinality": (
            corrupted["deterministic"].get("count") == 4
        ),
        "zero_at_zero_threshold_is_not_a_crossing": (
            zero["threshold_crossings"] == 0
            and zero["stochastic"]["threshold_crossings"] == 0
        ),
        "zero_policy_is_constant_hold": (
            zero["classification"] == pb.CONSTANT_HOLD
        ),
        "state_responsive_requires_observation_evidence": (
            varying_without_observations["classification"]
            != pb.STATE_RESPONSIVE_ACTIVE
        ),
        "sealed_parent_path_is_refused": False,
        "corrupt_trace_row_is_refused": False,
        "missing_cost_is_unavailable": (
            sidecar_result["economics"]["total_cost"] is None
        ),
        "measurement_binds_model_and_role": all(
            key in sidecar_result["custody"]
            for key in (
                "model_sha256",
                "experiment_identity",
                "role",
                "config_sha256",
            )
        ),
    }
    # Reaching this point proves the sidecar accepted the sealed-parent fixture
    # and silently skipped its malformed values.
    checks["sealed_parent_path_is_refused"] = False
    checks["corrupt_trace_row_is_refused"] = False

    report = {
        "schema": "agent_multi.musashi_finding_277_adversarial_repro.v1",
        "implementation_root": str(root),
        "checks": checks,
        "all_acceptance_checks_pass": all(checks.values()),
        "observed": {
            "corrupted_sequence": corrupted,
            "zero_threshold_sequence": zero,
            "varying_actions_without_observations": varying_without_observations,
            "sealed_corrupt_trace_measurement": sidecar_result,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["all_acceptance_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
