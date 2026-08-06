from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.campaign_supervisor import _domain_semantic_hash
from tools.materialize_eth_anchored_campaign import (
    ANCHOR_MODEL,
    build_config,
)


ROOT = Path(__file__).resolve().parents[2]
DOIN_ROOT = ROOT.parent / "doin-node"


def test_anchored_config_keeps_loadable_observation_and_model_shape():
    config = build_config(smoke=True)
    environment = config["environment"]
    training = config["training"]
    optimization = config["optimization"]

    assert environment["window_size"] == 32
    assert len(environment["feature_columns"]) == 83
    assert environment["feature_scaling"] == "rolling_zscore"
    assert environment["feature_scaling_window"] == 256
    assert environment["include_price_window"] is True
    assert training["net_arch"] == [256, 256]
    assert training["warm_start_model"] == str(ANCHOR_MODEL)
    assert training["warm_start_model_sha256"] == hashlib.sha256(
        ANCHOR_MODEL.read_bytes()
    ).hexdigest()
    assert optimization["optimization_min_trades_by_split"] == {
        "train_tail": 0,
        "validation": 12,
    }


def test_anchored_genome_does_not_mutate_observation_or_network_shape():
    config = build_config(smoke=False)
    names = {
        gene["name"] for gene in config["optimization"]["mixed_genome_schema"]
    }
    assert not names.intersection({
        "preprocessing_mode",
        "scaling_history_bars",
        "feature_clip",
        "context_hours",
        "net_architecture",
    })
    assert config["optimization"]["ga_population"] == 20
    assert sum(
        stage["generations"]
        for stage in config["optimization"]["optimization_stages"]
    ) == 18


def test_all_four_workers_share_one_semantic_domain_per_job():
    plan = json.loads((
        ROOT
        / "examples/campaigns/phase_2_eth_anchored_fleet_v1/campaign_plan.json"
    ).read_text(encoding="utf-8"))
    assert [job["ordinal"] for job in plan["jobs"]] == [0, 1]
    for job in plan["jobs"]:
        hashes = set()
        for relative in job["worker_configs"].values():
            config = json.loads((DOIN_ROOT / relative).read_text(encoding="utf-8"))
            hashes.add(_domain_semantic_hash(config))
        assert hashes == {job["domain_semantic_hash"]}
