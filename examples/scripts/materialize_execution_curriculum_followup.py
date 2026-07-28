#!/usr/bin/env python3
"""Materialize a cost-curriculum follow-up from an archived asset champion."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from env_plugins.execution_cost_curriculum import load_curriculum


DEFAULT_SOURCE_RUNTIME_PATH = (
    "${ARTIFACT_ROOT}/full_genome/usdcad_4h/champion_policy.zip"
)
DEFAULT_CURRICULUM_RUNTIME_PATH = (
    "${REPO_ROOT}/agent-multi/examples/config/execution_curriculum/"
    "project3_execution_cost_curriculum_v1.json"
)
ROBUST_SCENARIOS = [
    "easy_upper",
    "nominal_low",
    "nominal_reference",
    "nominal_high",
    "stress_severe",
]
ACTIVE_GENES = [
    "learning_rate_gene",
    "gamma_gene",
    "tau_gene",
    "train_freq_gene",
    "gradient_steps_gene",
    "entropy_gene",
    "action_threshold_gene",
    "relative_volume_gene",
    "stop_loss_atr_gene",
    "take_profit_atr_gene",
]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _champion_parameters(
    base: dict[str, Any],
    source: Path | None,
) -> tuple[dict[str, Any], str | None]:
    initial = copy.deepcopy(
        base.get("optimization", {}).get("initial_candidate_decoded", {})
    )
    if source is None:
        return initial, None
    payload = _load(source)
    parameters = (
        payload.get("decoded_parameters")
        or payload.get("parameters_decoded")
        or payload.get("parameters")
    )
    if not isinstance(parameters, dict):
        raise ValueError(
            "champion parameters must contain decoded parameters at "
            "'decoded_parameters', 'parameters_decoded', or 'parameters'"
        )
    unknown = sorted(set(parameters) - set(initial))
    if unknown:
        parameters = {
            key: value for key, value in parameters.items() if key in initial
        }
    for key in initial:
        if key in parameters:
            initial[key] = copy.deepcopy(parameters[key])
    return initial, _sha256(source)


def materialize(
    *,
    base_config: Path,
    curriculum_config: Path,
    output_config: Path,
    source_model_runtime_path: str,
    source_model_file: Path | None,
    source_parameters_file: Path | None,
    template: bool,
) -> Path:
    if not template and (source_model_file is None or source_parameters_file is None):
        raise ValueError(
            "launchable materialization requires source model and parameters files"
        )
    if source_model_file is not None and not source_model_file.is_file():
        raise FileNotFoundError(source_model_file)
    if source_parameters_file is not None and not source_parameters_file.is_file():
        raise FileNotFoundError(source_parameters_file)

    config = copy.deepcopy(_load(base_config))
    if config.get("schema_version") != "trading_experiment.v1":
        raise ValueError("base config must use trading_experiment.v1")
    curriculum = load_curriculum(
        str(curriculum_config),
        base_dir=Path.cwd(),
    )
    initial_candidate, parameters_sha = _champion_parameters(
        config,
        source_parameters_file,
    )
    model_sha = _sha256(source_model_file) if source_model_file else None

    config["experiment"].update(
        {
            "name": "phase_1_asset_policy_usdcad_4h_execution_curriculum_v1",
            "source_champion": {
                "campaign_id": "usdcad-4h-full-genome-sac-shared-v1",
                "model_runtime_path": source_model_runtime_path,
                "model_sha256": model_sha,
                "parameters_sha256": parameters_sha,
            },
            "materialization_status": (
                "template_waiting_for_final_archive"
                if template
                else "launchable_verified_source_artifacts"
            ),
        }
    )
    config["code"].setdefault("contract_versions", {}).update(
        {
            "execution_cost_curriculum": "execution_cost_curriculum.v1",
            "execution_router": "adaptive_order_router.v1",
            "robust_metrics": "trading.execution_robust.v1",
        }
    )
    curriculum_fields = {
        "execution_cost_curriculum": DEFAULT_CURRICULUM_RUNTIME_PATH,
        "execution_cost_curriculum_fingerprint": curriculum.fingerprint,
    }
    config["environment"].update(
        {
            **curriculum_fields,
            "execution_cost_observation_enabled": True,
        }
    )
    config["training"].update(
        {
            **curriculum_fields,
            "pipeline_plugin": "rl_pipeline_with_execution_curriculum",
            "selection_metric": "robust_weekly_rap_fitness",
            "warm_start_model": source_model_runtime_path,
            "warm_start_model_sha256": model_sha,
            "warm_start_expand_observation_space": True,
            "execution_cost_curriculum_epochs": 100,
            "robust_validation_scenarios": ROBUST_SCENARIOS,
            "robust_fitness_config": {
                "lower_tail_fraction": 0.25,
                "downside_penalty_weight": 1.0,
                "dispersion_penalty_weight": 0.5,
                "annualization_weeks": 52.0,
            },
            "evaluate_test_split": False,
        }
    )

    artifact_root = "${ARTIFACT_ROOT}/execution_curriculum/usdcad_4h"
    config["artifacts"].update(
        {
            "artifact_root": artifact_root,
            "save_model": f"{artifact_root}/final_policy.zip",
            "results_file": f"{artifact_root}/results.json",
            "resolved_config_file": f"{artifact_root}/resolved_config.json",
            "config_manifest_file": f"{artifact_root}/config_manifest.json",
            "optimizer_output_file": f"{artifact_root}/optimizer_output.json",
            "return_trace_dir": f"{artifact_root}/return_traces",
        }
    )
    config["optimization"].update(
        {
            "enabled": not template,
            "metric": "robust_weekly_rap_fitness",
            "metric_schema": "trading.execution_robust.v1",
            "ga_fitness_split": "train",
            "initial_candidate_decoded": initial_candidate,
            "optimization_statistics": f"{artifact_root}/optimization_stats.json",
            "optimization_parameters_file": (
                f"{artifact_root}/optimization_parameters.json"
            ),
            "optimization_resume_file": f"{artifact_root}/optimization_resume.json",
            "optimization_candidate_history": (
                f"{artifact_root}/candidate_history.csv"
            ),
            "optimization_champion_model_file": (
                f"{artifact_root}/champion_policy.zip"
            ),
            "optimization_resume": False,
            "optimization_stages": [
                {
                    "name": "cost_adaptation",
                    "params": ACTIVE_GENES[:6],
                    "generations": 6,
                    "patience": 4,
                },
                {
                    "name": "execution_risk",
                    "params": ACTIVE_GENES[6:],
                    "generations": 4,
                    "patience": 3,
                },
                {
                    "name": "bounded_joint_refinement",
                    "params": ACTIVE_GENES,
                    "generations": 6,
                    "patience": 5,
                },
            ],
        }
    )
    config["deployment"].update(
        {
            "lifecycle": (
                "materialization_template"
                if template
                else "research_component"
            ),
            "promotion_gate": (
                "robust_validation_and_source_artifact_hash_required"
            ),
        }
    )
    _write(output_config, config)
    return output_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--curriculum-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument(
        "--source-model-runtime-path",
        default=DEFAULT_SOURCE_RUNTIME_PATH,
    )
    parser.add_argument("--source-model-file", type=Path)
    parser.add_argument("--source-parameters-file", type=Path)
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()
    output = materialize(
        base_config=args.base_config.resolve(),
        curriculum_config=args.curriculum_config.resolve(),
        output_config=args.output_config.resolve(),
        source_model_runtime_path=args.source_model_runtime_path,
        source_model_file=(
            args.source_model_file.resolve()
            if args.source_model_file is not None
            else None
        ),
        source_parameters_file=(
            args.source_parameters_file.resolve()
            if args.source_parameters_file is not None
            else None
        ),
        template=args.template,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
