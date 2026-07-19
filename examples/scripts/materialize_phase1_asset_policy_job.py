#!/usr/bin/env python3
"""Materialize a Phase 1 asset-policy job and its immutable dataset manifest."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _dataset_evidence(path: Path, date_column: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        date_index = header.index(date_column)
        binary_candidates = {
            index: True for index, name in enumerate(header) if name != date_column
        }

        def update_binary_candidates(row: list[str]) -> None:
            for index in tuple(binary_candidates):
                if not binary_candidates[index]:
                    continue
                try:
                    value = float(row[index])
                except (TypeError, ValueError, IndexError):
                    binary_candidates[index] = False
                    continue
                if not math.isfinite(value) or value not in {-1.0, 0.0, 1.0}:
                    binary_candidates[index] = False

        first = next(reader)
        last = first
        rows = 1
        update_binary_candidates(first)
        for row in reader:
            rows += 1
            last = row
            update_binary_candidates(row)
        feature_columns = [name for name in header if name != date_column]
        return {
            "sha256": digest.hexdigest(),
            "row_count": rows,
            "column_count": len(header),
            "first_timestamp": first[date_index].replace(" ", "T"),
            "last_timestamp": last[date_index].replace(" ", "T"),
            "columns": header,
            "feature_columns": feature_columns,
            "feature_binary_columns": [
                header[index]
                for index, is_binary in binary_candidates.items()
                if is_binary
            ],
        }


def materialize(
    *,
    base_config: Path,
    output_config: Path,
    output_manifest: Path,
    dataset: Path,
    dataset_source_path: str,
    asset: str,
    timeframe: str,
    data_profile: str,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    risk_penalty_lambda: float,
    k_sl: float,
    k_tp: float,
    rel_volume: float,
) -> tuple[Path, Path]:
    config = copy.deepcopy(_load(base_config))
    asset_upper = asset.upper()
    asset_lower = asset.lower()
    slug = f"{asset_lower}_{timeframe}_sac"
    profile_slug = data_profile.lower()
    experiment_name = f"phase_1_asset_policy_{slug}_optimization"
    manifest_name = output_manifest.name

    config["experiment"]["name"] = experiment_name
    data = config["data"]
    data.update({
        "asset": asset_upper,
        "timeframe": timeframe,
        "input_data_file": f"${{DATA_ROOT}}/{dataset_source_path}",
        "dataset_manifest_file": (
            f"${{REPO_ROOT}}/agent-multi/examples/data/phase_1_asset_policy/{manifest_name}"
        ),
        "features_preset": profile_slug,
        "data_profile": profile_slug,
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": test_end,
    })
    config["risk"].update({
        "rel_volume": rel_volume,
        "k_sl": k_sl,
        "k_tp": k_tp,
    })
    config["training"]["risk_penalty_lambda"] = risk_penalty_lambda
    artifact_root = f"${{ARTIFACT_ROOT}}/phase_1_asset_policy/optimization/{slug}"
    optimization = config["optimization"]
    optimization.update({
        "optimization_statistics": f"{artifact_root}/optimization_stats.json",
        "optimization_parameters_file": f"{artifact_root}/optimization_parameters.json",
        "optimization_resume_file": f"{artifact_root}/optimization_resume.json",
        "optimization_candidate_history": f"{artifact_root}/optimization_candidate_history.csv",
        "optimization_champion_model_file": f"{artifact_root}/champion_policy.zip",
    })
    config["artifacts"].update({
        "artifact_root": artifact_root,
        "save_model": f"{artifact_root}/final_policy.zip",
        "results_file": f"{artifact_root}/results.json",
        "resolved_config_file": f"{artifact_root}/resolved_config.json",
        "config_manifest_file": f"{artifact_root}/config_manifest.json",
        "optimizer_output_file": f"{artifact_root}/optimizer_output.json",
    })

    evidence = _dataset_evidence(dataset, str(data["date_column"]))
    config.setdefault("environment", {}).update({
        "preprocessor_plugin": "feature_window_preprocessor",
        "feature_columns": evidence["feature_columns"],
        "feature_binary_columns": evidence["feature_binary_columns"],
        "feature_scaling": "rolling_zscore",
        "feature_scaling_window": 256,
        "feature_clip": 10.0,
        "include_price_window": False,
        "include_agent_state": True,
        "require_feature_aware_preprocessor": True,
    })
    config["optimization"].update({
        "optimization_reject_action_collapse": True,
        "optimization_action_collapse_min_steps": 64,
        "optimization_min_action_std": 1e-5,
        "optimization_max_dominant_action_rate": 0.999,
        "optimization_action_collapse_splits": ["train_tail", "validation"],
    })
    manifest = {
        "schema_version": "agent_multi.dataset_manifest.v1",
        "dataset_id": (
            f"{asset_lower}_{timeframe}_{profile_slug}_"
            f"{evidence['first_timestamp'][:4]}_{evidence['last_timestamp'][:4]}"
        ),
        "asset": asset_upper,
        "timeframe": timeframe,
        "source_repository": "financial-data",
        "source_path": dataset_source_path,
        **evidence,
        "date_column": data["date_column"],
        "data_profile": profile_slug,
        "splits": {
            "train": [train_start, train_end],
            "validation": [validation_start, validation_end],
            "test": [test_start, test_end],
        },
        "selection_uses_test": False,
    }
    _write(output_config, config)
    _write(output_manifest, manifest)
    return output_config, output_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-source-path", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--data-profile", required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-start", default="2022-01-01T00:00:00")
    parser.add_argument("--validation-end", default="2022-12-31T23:59:59")
    parser.add_argument("--test-start", default="2023-01-01T00:00:00")
    parser.add_argument("--test-end", default="2023-12-31T23:59:59")
    parser.add_argument("--risk-penalty-lambda", type=float, required=True)
    parser.add_argument("--k-sl", type=float, required=True)
    parser.add_argument("--k-tp", type=float, required=True)
    parser.add_argument("--rel-volume", type=float, required=True)
    args = parser.parse_args()
    outputs = materialize(
        base_config=args.base_config.resolve(),
        output_config=args.output_config.resolve(),
        output_manifest=args.output_manifest.resolve(),
        dataset=args.dataset.resolve(),
        dataset_source_path=args.dataset_source_path,
        asset=args.asset,
        timeframe=args.timeframe,
        data_profile=args.data_profile,
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
        risk_penalty_lambda=args.risk_penalty_lambda,
        k_sl=args.k_sl,
        k_tp=args.k_tp,
        rel_volume=args.rel_volume,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
