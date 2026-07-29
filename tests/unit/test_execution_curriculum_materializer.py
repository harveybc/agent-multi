from __future__ import annotations

import hashlib
import json
from pathlib import Path

from examples.scripts.materialize_execution_curriculum_followup import (
    ACTIVE_GENES,
    materialize,
)
from examples.scripts.materialize_execution_curriculum_campaign import (
    materialize_campaign,
)
from examples.scripts.materialize_protected_execution_v2_configs import (
    build_curriculum,
    build_easy,
)
from app.campaign_supervisor import _domain_semantic_hash


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
OPTIMIZATION_CONFIG_ROOT = (
    ROOT / "examples/config/phase_1_asset_policy/optimization"
)
PROTECTED_EASY = (
    OPTIMIZATION_CONFIG_ROOT
    / "phase_1_asset_policy_usdcad_4h_protected_easy_v2.json"
)
PROTECTED_CURRICULUM = (
    OPTIMIZATION_CONFIG_ROOT
    / "phase_1_asset_policy_usdcad_4h_protected_curriculum_template_v2.json"
)
PROTECTED_NODE_TEMPLATES = (
    ROOT.parent
    / "doin-node/examples/trading"
    / "phase_1_asset_policy_usdcad_4h_protected_easy_v2"
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
                "decoded_parameters": {
                    "learning_rate_gene": 0.000123,
                    "action_threshold_gene": 0.22,
                    "_repairs": ["must not enter the chromosome"],
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


def test_followup_prefers_decoded_blockchain_parameters(tmp_path: Path) -> None:
    model = tmp_path / "champion_policy.zip"
    model.write_bytes(b"verified champion")
    parameters = tmp_path / "optimization_parameters.json"
    parameters.write_text(
        json.dumps(
            {
                "parameters": {
                    "preprocessing_mode": 1,
                    "net_architecture": 1,
                },
                "decoded_parameters": {
                    "preprocessing_mode": "rolling_zscore",
                    "net_architecture": "256x256",
                },
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
    initial = config["optimization"]["initial_candidate_decoded"]
    assert initial["preprocessing_mode"] == "rolling_zscore"
    assert initial["net_architecture"] == "256x256"


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


def test_campaign_materialization_builds_one_semantic_domain_for_all_workers(
    tmp_path: Path,
) -> None:
    model = tmp_path / "champion_policy.zip"
    model.write_bytes(b"verified champion")
    parameters = tmp_path / "champion_parameters.json"
    parameters.write_text(
        json.dumps(
            {
                "decoded_parameters": {
                    "preprocessing_mode": "rolling_zscore",
                    "net_architecture": "256x256",
                    "learning_rate_gene": 0.000123,
                }
            }
        ),
        encoding="utf-8",
    )

    result = materialize_campaign(
        agent_root=ROOT,
        doin_root=ROOT.parent / "doin-node",
        source_model_file=model,
        source_parameters_file=parameters,
        output_root=tmp_path / "generated",
        domain_id="curriculum-test-domain",
        campaign_slug="curriculum-test",
    )

    canonical = json.loads(Path(result["canonical_config"]).read_text())
    assert canonical["optimization"]["enabled"] is True
    assert canonical["experiment"]["source_champion"]["model_sha256"]
    node_configs = [
        json.loads(Path(item["path"]).read_text())
        for item in result["node_configs"].values()
    ]
    assert len(node_configs) == 4
    assert {
        config["domains"][0]["domain_id"] for config in node_configs
    } == {"curriculum-test-domain"}
    assert len({_domain_semantic_hash(config) for config in node_configs}) == 1


def test_protected_v2_configs_are_reproducible_from_versioned_sources() -> None:
    easy_source = json.loads(BASE.read_text(encoding="utf-8"))
    curriculum_source = json.loads(
        (
            OPTIMIZATION_CONFIG_ROOT
            / "phase_1_asset_policy_usdcad_4h_execution_curriculum_template_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert build_easy(easy_source) == json.loads(
        PROTECTED_EASY.read_text(encoding="utf-8")
    )
    assert build_curriculum(curriculum_source) == json.loads(
        PROTECTED_CURRICULUM.read_text(encoding="utf-8")
    )


def test_protected_campaign_transition_uses_v2_template_and_runtime_artifact(
    tmp_path: Path,
) -> None:
    model = tmp_path / "champion_policy.zip"
    model.write_bytes(b"protected champion")
    parameters = tmp_path / "champion_parameters.json"
    parameters.write_text(
        json.dumps(
            {
                "decoded_parameters": {
                    "entry_order_mode_gene": "limit",
                    "limit_offset_atr_gene": 0.07,
                }
            }
        ),
        encoding="utf-8",
    )
    runtime_model = "${ARTIFACT_ROOT}/protected_easy/usdcad_4h/champion_policy.zip"

    result = materialize_campaign(
        agent_root=ROOT,
        doin_root=ROOT.parent / "doin-node",
        source_model_file=model,
        source_parameters_file=parameters,
        output_root=tmp_path / "generated",
        domain_id="protected-curriculum-test-domain",
        campaign_slug="protected-curriculum-test",
        base_config=PROTECTED_CURRICULUM,
        node_template_dir=PROTECTED_NODE_TEMPLATES,
        source_model_runtime_path=runtime_model,
    )

    canonical = json.loads(Path(result["canonical_config"]).read_text())
    assert canonical["environment"]["require_protected_entries"] is True
    assert canonical["training"]["warm_start_model"] == runtime_model
    assert canonical["optimization"]["initial_candidate_decoded"][
        "entry_order_mode_gene"
    ] == "limit"
    nodes = [
        json.loads(Path(item["path"]).read_text())
        for item in result["node_configs"].values()
    ]
    assert len(nodes) == 4
    assert {
        node["domains"][0]["domain_id"] for node in nodes
    } == {"protected-curriculum-test-domain"}
    assert len({_domain_semantic_hash(node) for node in nodes}) == 1
