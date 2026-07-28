from __future__ import annotations

import hashlib
import json
from pathlib import Path

from examples.scripts.materialize_execution_curriculum_followup import (
    ACTIVE_GENES,
    materialize,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "examples/config/phase_1_asset_policy/optimization"
    / "phase_1_asset_policy_usdcad_4h_full_genome_v1.json"
)
CURRICULUM = (
    ROOT
    / "examples/config/execution_curriculum"
    / "project3_execution_cost_curriculum_v1.json"
)


def test_launchable_followup_requires_and_hashes_champion_artifacts(
    tmp_path: Path,
) -> None:
    model = tmp_path / "champion_policy.zip"
    model.write_bytes(b"verified champion")
    parameters = tmp_path / "optimization_parameters.json"
    parameters.write_text(
        json.dumps(
            {
                "parameters": {
                    "learning_rate_gene": 0.000123,
                    "action_threshold_gene": 0.22,
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "followup.json"

    materialize(
        base_config=BASE,
        curriculum_config=CURRICULUM,
        output_config=output,
        source_model_runtime_path="${ARTIFACT_ROOT}/source/champion_policy.zip",
        source_model_file=model,
        source_parameters_file=parameters,
        template=False,
    )
    config = json.loads(output.read_text(encoding="utf-8"))

    assert config["optimization"]["enabled"] is True
    assert (
        config["training"]["pipeline_plugin"]
        == "rl_pipeline_with_execution_curriculum"
    )
    assert (
        config["optimization"]["metric"]
        == "robust_weekly_rap_fitness"
    )
    assert config["training"]["execution_cost_curriculum_epochs"] == 100
    assert config["optimization"]["initial_candidate_decoded"][
        "learning_rate_gene"
    ] == 0.000123
    assert config["experiment"]["source_champion"]["model_sha256"] == (
        hashlib.sha256(b"verified champion").hexdigest()
    )
    assert config["training"]["warm_start_model_sha256"] == (
        hashlib.sha256(b"verified champion").hexdigest()
    )
    configured_genes = {
        gene
        for stage in config["optimization"]["optimization_stages"]
        for gene in stage["params"]
    }
    assert configured_genes == set(ACTIVE_GENES)
    assert not any(name.startswith("feature_group__") for name in configured_genes)


def test_launchable_followup_fails_without_final_artifacts(tmp_path: Path) -> None:
    try:
        materialize(
            base_config=BASE,
            curriculum_config=CURRICULUM,
            output_config=tmp_path / "followup.json",
            source_model_runtime_path="${ARTIFACT_ROOT}/source/champion_policy.zip",
            source_model_file=None,
            source_parameters_file=None,
            template=False,
        )
    except ValueError as exc:
        assert "requires source model and parameters" in str(exc)
    else:
        raise AssertionError("launchable materialization must fail closed")
