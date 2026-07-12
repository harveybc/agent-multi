#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_plugins.project3_sac_actor_critic_agent import Plugin as SacPlugin
from app.canonical_config import load_json_object, resolve_config
from app.config import DEFAULT_VALUES
from app.runtime_overlay import resolve_runtime_overlay
from optimizer_plugins.default_optimizer import Plugin as OptimizerPlugin


CONFIG_ROOT = ROOT / "examples/config/phase_1_asset_policy"
BASE_CONFIG = CONFIG_ROOT / "phase_1_asset_policy_solusdt_4h_sac_config.json"
OPTIMIZATION_CONFIG = (
    CONFIG_ROOT
    / "optimization/phase_1_asset_policy_solusdt_4h_sac_optimization_config.json"
)
SMOKE_CONFIG = (
    CONFIG_ROOT
    / "optimization/phase_1_asset_policy_solusdt_4h_sac_smoke_optimization_config.json"
)
INFERENCE_CONFIG = (
    CONFIG_ROOT
    / "inference/phase_1_asset_policy_solusdt_4h_sac_inference_config.json"
)
DATA_MANIFEST = ROOT / "examples/data/phase_1_asset_policy/solusdt_4h_dataset_manifest.json"


def _resolve(*, base: Path | None, profile: Path, overlay: Path):
    resolution = resolve_config(
        DEFAULT_VALUES,
        base_profile=load_json_object(base) if base else None,
        file_config=load_json_object(profile),
        source_descriptors={
            "base_profile": str(base) if base else "",
            "file_config": str(profile),
        },
    )
    overlay_payload = load_json_object(overlay)
    runtime = resolve_runtime_overlay(
        resolution.runtime,
        overlay_payload=overlay_payload,
        overlay_base_dir=overlay.parent,
        expected_repositories=resolution.canonical.code.get("repositories", {}),
    )
    return resolution, runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_data(runtime: dict, manifest: dict) -> None:
    source = Path(runtime["input_data_file"])
    if not source.is_file():
        raise SystemExit(f"dataset does not exist: {source}")
    actual_hash = _sha256(source)
    if actual_hash != manifest["sha256"]:
        raise SystemExit(
            f"dataset hash mismatch: expected {manifest['sha256']}, got {actual_hash}"
        )
    rows = sum(1 for _ in source.open("rb")) - 1
    if rows != int(manifest["row_count"]):
        raise SystemExit(
            f"dataset row count mismatch: expected {manifest['row_count']}, got {rows}"
        )


def _validate_optimizer(runtime: dict) -> None:
    schema = OptimizerPlugin._effective_schema(SacPlugin().hparam_schema(), runtime)
    names = [item[0] for item in schema]
    stages = runtime.get("optimization_stages") or []
    if not stages:
        raise SystemExit("optimization_stages cannot be empty")
    for stage in stages:
        params = stage.get("params")
        if params == "all":
            continue
        unknown = sorted(set(params or []) - set(names))
        if unknown:
            raise SystemExit(f"stage {stage.get('name')} has unknown params: {unknown}")
    if runtime.get("optimization_metric") != "train_validation_l1_score":
        raise SystemExit("phase 1 must optimize train_validation_l1_score")
    if runtime.get("selection_metric") != "risk_adjusted_return":
        raise SystemExit("L1 selection_metric must be risk_adjusted_return")
    if bool(runtime.get("evaluate_test_split", True)):
        raise SystemExit("protected test evaluation must be disabled during optimization")
    if bool(runtime.get("selection_uses_test", True)):
        raise SystemExit("test data cannot participate in selection")
    if not bool(runtime.get("optimization_capture_model_artifact")):
        raise SystemExit("optimizer must capture the exact champion checkpoint")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-overlay",
        default=str(ROOT / "configs/runtime/omega.json"),
        help="Machine trading_runtime_overlay.v1 JSON used for path resolution.",
    )
    args = parser.parse_args()
    overlay = Path(args.runtime_overlay).expanduser().resolve()
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))

    base, base_runtime = _resolve(base=None, profile=BASE_CONFIG, overlay=overlay)
    full, full_runtime = _resolve(
        base=None, profile=OPTIMIZATION_CONFIG, overlay=overlay
    )
    smoke, smoke_runtime = _resolve(
        base=BASE_CONFIG, profile=SMOKE_CONFIG, overlay=overlay
    )
    inference, inference_runtime = _resolve(
        base=BASE_CONFIG, profile=INFERENCE_CONFIG, overlay=overlay
    )

    _validate_data(full_runtime.runtime, manifest)
    _validate_optimizer(full_runtime.runtime)
    _validate_optimizer(smoke_runtime.runtime)
    if inference_runtime.runtime.get("use_optimizer"):
        raise SystemExit("inference config cannot enable optimization")
    if inference_runtime.runtime.get("pipeline_plugin") != "rl_pipeline":
        raise SystemExit("inference config must use the inference-safe rl_pipeline")

    report = {
        "schema_version": "agent_multi.phase_config_validation.v1",
        "runtime_machine_id": full_runtime.manifest["machine_id"],
        "dataset_sha256": manifest["sha256"],
        "config_hashes": {
            "baseline": base.canonical.canonical_hash,
            "optimization": full.canonical.canonical_hash,
            "smoke": smoke.canonical.canonical_hash,
            "inference": inference.canonical.canonical_hash,
        },
        "optimization_stages": [
            stage["name"] for stage in full_runtime.runtime["optimization_stages"]
        ],
        "test_evaluated_during_optimization": False,
        "status": "valid",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
