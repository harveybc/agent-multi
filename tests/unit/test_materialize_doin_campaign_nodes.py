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
        "training": {"learning_rate": 0.001},
        "asset_policy": {"continuous_action_threshold": 0.2},
        "optimization": {
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
    assert omega["shared_min_peers"] == 0
    assert "shared_min_peers" not in gamma
    assert omega["domains"][0]["optimization_config"]["runtime_overlay"] == "omega.json"
    assert gamma["domains"][0]["optimization_config"]["runtime_overlay"] == "gamma.json"
    assert omega["domains"][0]["domain_id"] == "new-domain"
    assert omega["domains"][0]["optimization_config"]["initial_candidate_params"] == {
        "learning_rate": 0.001,
        "continuous_action_threshold": 0.2,
    }
