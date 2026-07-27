from __future__ import annotations

import json
from pathlib import Path

from examples.scripts.materialize_doin_campaign_nodes import materialize


def test_materializer_keeps_machine_runtime_and_unifies_domain(tmp_path: Path):
    templates = tmp_path / "templates"
    templates.mkdir()
    for label, port, overlay in (("omega", 8470, "omega.json"), ("gamma-5090", 8471, "gamma.json")):
        value = {
            "node_label": label,
            "port": port,
            "data_dir": "old",
            "bootstrap_peers": ["peer:8470"],
            "domains": [{
                "domain_id": "old",
                "optimization_config": {
                    "agent_multi_root": "/repo/agent-multi",
                    "runtime_overlay": overlay,
                },
            }],
        }
        (templates / f"{label}_node.json").write_text(json.dumps(value))
    canonical = {
        "environment": {
            "plugin": "gym_fx_env",
            "preprocessor_plugin": "feature_window_preprocessor",
        },
        "training": {
            "learning_rate": 0.001,
            "pipeline_plugin": "rl_pipeline_with_validation",
        },
        "asset_policy": {
            "plugin": "project3_sac_actor_critic_agent",
            "continuous_action_threshold": 0.2,
        },
        "optimization": {
            "plugin": "default_optimizer",
            "metric": "rap",
            "metric_schema": "trading.metrics.v1",
            "higher_is_better": True,
            "ga_population": 12,
            "ga_seed": 7,
            "optimization_patience": 3,
            "optimization_stages": [{"name": "all", "params": "all", "generations": 2}],
            "hyperparameter_bounds": {
                "learning_rate": [0.0001, 0.01],
                "continuous_action_threshold": [0.1, 0.4],
            },
        },
    }
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(json.dumps(canonical))
    output = tmp_path / "output"
    paths = materialize(
        template_dir=templates,
        output_dir=output,
        canonical_config=canonical_path,
        load_config="examples/config/job.json",
        domain_id="new-domain",
        campaign_slug="job-v1",
    )
    assert len(paths) == 2
    omega = json.loads((output / "omega_node.json").read_text())
    gamma = json.loads((output / "gamma-5090_node.json").read_text())
    assert omega["port"] == 8470
    assert gamma["port"] == 8471
    assert omega["shared_initialize_before_peers"] is True
    assert gamma["shared_initialize_before_peers"] is True
    assert omega["domains"][0]["optimization_config"]["runtime_overlay"] == "omega.json"
    assert gamma["domains"][0]["optimization_config"]["runtime_overlay"] == "gamma.json"
    assert omega["domains"][0]["domain_id"] == "new-domain"
    assert omega["domains"][0]["optimization_config"]["initial_candidate_params"] == {
        "learning_rate": 0.001,
        "continuous_action_threshold": 0.2,
    }
    assert omega["domains"][0]["optimization_config"]["preprocessor_plugin"] == (
        "feature_window_preprocessor"
    )
    assert omega["domains"][0]["optimization_config"]["pipeline_plugin"] == (
        "rl_pipeline_with_validation"
    )


def test_materializer_uses_optimizer_encoding_for_mixed_genome(tmp_path: Path):
    templates = tmp_path / "templates"
    templates.mkdir()
    template = {
        "node_label": "omega",
        "domains": [{
            "domain_id": "old",
            "optimization_config": {
                "agent_multi_root": "/repo/agent-multi",
            },
        }],
    }
    (templates / "omega_node.json").write_text(json.dumps(template))
    canonical = {
        "environment": {"plugin": "gym_fx_env"},
        "training": {"pipeline_plugin": "rl_pipeline_with_validation"},
        "asset_policy": {"plugin": "project3_sac_actor_critic_agent"},
        "optimization": {
            "enabled": True,
            "plugin": "project3_full_genome_optimizer",
            "metric": "train_validation_l1_score",
            "metric_schema": "trading.metrics.v1",
            "higher_is_better": True,
            "ga_population": 4,
            "ga_seed": 17,
            "optimization_patience": 2,
            "optimization_stages": [{
                "name": "all",
                "params": "all",
                "generations": 2,
            }],
            "mixed_genome_schema": [
                {
                    "name": "mode",
                    "kind": "categorical",
                    "choices": ["none", "rolling"],
                    "target": "feature_scaling",
                },
                {
                    "name": "learning_rate_gene",
                    "kind": "log_float",
                    "low": 0.00001,
                    "high": 0.001,
                    "target": "learning_rate",
                },
                {
                    "name": "enabled",
                    "kind": "boolean",
                    "target": "_enabled",
                },
            ],
            "initial_candidate_decoded": {
                "mode": "rolling",
                "learning_rate_gene": 0.0001,
                "enabled": True,
            },
        },
    }
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(json.dumps(canonical))
    output = tmp_path / "output"

    materialize(
        template_dir=templates,
        output_dir=output,
        canonical_config=canonical_path,
        load_config="examples/config/job.json",
        domain_id="mixed-domain",
        campaign_slug="mixed-v1",
    )

    node = json.loads((output / "omega_node.json").read_text())
    domain = node["domains"][0]
    config = domain["optimization_config"]
    assert domain["param_bounds"] == {
        "mode": [0.0, 1.0],
        "learning_rate_gene": [-5.0, -3.0],
        "enabled": [0.0, 1.0],
    }
    assert config["initial_candidate_params"] == {
        "mode": 1,
        "learning_rate_gene": -4.0,
        "enabled": 1,
    }
    assert config["hyperparameter_bounds"] == domain["param_bounds"]
    assert config["mixed_genome_schema_version"] == (
        "agent_multi.project3_full_genome.v1"
    )
    assert len(config["mixed_genome_schema_hash"]) == 64
    assert domain["resource_limits"]["max_epochs"] == 2_000
    assert domain["resource_limits"]["max_batch_size"] == 512
    assert domain["resource_limits"]["max_training_seconds"] == 604_800
