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


def _write_trace(path: Path, *, corrupt: bool = True) -> None:
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
            "action_raw": "corrupt" if corrupt else "0.25",
            "position": "corrupt" if corrupt else "1",
            "equity": "101",
            "trades": "1",
            "trade_cost": "corrupt" if corrupt else "",
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


def _write_role_authority(trace: Path, identity: str = "fixture-identity") -> None:
    role_root = trace.parent.parent
    split_dir = role_root / "nested_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    data_file = split_dir / "train_monitor.csv"
    data_file.write_text("timestamp,value\n2025-01-01,1\n")
    manifest = {
        "experiment_identity": identity,
        "roles": {
            "train_monitor": {
                "csv": str(data_file),
                "csv_sha256": "fixture-data-sha256",
            }
        },
    }
    (split_dir / "nested_split_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    meta = {
        "nested_role": "train_monitor",
        "data_file": str(data_file),
        "data_file_sha256": "fixture-data-sha256",
        "config_sha256": "fixture-config-sha256",
        "observation_contract_sha256": "fixture-observation-sha256",
    }
    Path(str(trace) + ".meta.json").write_text(
        json.dumps(meta, sort_keys=True) + "\n"
    )


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

    sidecar_observed: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="musashi-277-repro-") as tmp:
        sealed_trace = (
            Path(tmp)
            / "sealed_test_2025"
            / "return_traces"
            / "evaluation_return_trace.csv"
        )
        _write_trace(sealed_trace, corrupt=False)
        try:
            sidecar.measure_trace(sealed_trace, threshold=0.1, tolerance=1e-6)
            sealed_refused = False
        except sidecar.SidecarRefusal as error:
            sealed_refused = True
            sidecar_observed["sealed_refusal"] = str(error)

        corrupt_trace = (
            Path(tmp)
            / "allowed_corrupt"
            / "return_traces"
            / "evaluation_return_trace.csv"
        )
        _write_trace(corrupt_trace, corrupt=True)
        _write_role_authority(corrupt_trace)
        try:
            sidecar.measure_trace(corrupt_trace, threshold=0.1, tolerance=1e-6)
            corrupt_refused = False
        except sidecar.SidecarRefusal as error:
            corrupt_refused = True
            sidecar_observed["corrupt_refusal"] = str(error)

        valid_trace = (
            Path(tmp)
            / "allowed_valid"
            / "return_traces"
            / "evaluation_return_trace.csv"
        )
        _write_trace(valid_trace, corrupt=False)
        _write_role_authority(valid_trace)
        try:
            valid_result = sidecar.measure_trace(
                valid_trace,
                threshold=0.1,
                tolerance=1e-6,
                model_sha256="fixture-model-sha256",
                model_file="fixture-model.zip",
                code_revision="fixture-code-revision",
            )
        except TypeError:
            # The pre-correction implementation did not accept model custody.
            valid_result = sidecar.measure_trace(
                valid_trace, threshold=0.1, tolerance=1e-6
            )
        sidecar_observed["valid_measurement"] = valid_result

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
        "sealed_parent_path_is_refused": sealed_refused,
        "corrupt_trace_row_is_refused": corrupt_refused,
        "missing_cost_is_unavailable": (
            valid_result["economics"]["total_cost"] is None
        ),
        "measurement_binds_model_and_role": all(
            valid_result["custody"].get(key)
            for key in (
                "model_sha256",
                "experiment_identity",
                "role",
                "config_sha256",
            )
        ),
    }
    report = {
        "schema": "agent_multi.musashi_finding_277_adversarial_repro.v1",
        "implementation_root": str(root),
        "checks": checks,
        "all_acceptance_checks_pass": all(checks.values()),
        "observed": {
            "corrupted_sequence": corrupted,
            "zero_threshold_sequence": zero,
            "varying_actions_without_observations": varying_without_observations,
            "sidecar": sidecar_observed,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["all_acceptance_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
