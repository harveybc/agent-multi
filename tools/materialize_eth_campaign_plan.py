#!/usr/bin/env python3
"""Materialize the phase-2 ETH paired-arm campaign plan (ETH order WP-E).

One immutable two-job plan on the SAME single swarm/blockchain:
ordinal 0 = ETH-EN (stage-integrated easy->normal solvency curriculum),
ordinal 1 = ETH-N (normal-only control). Identical data, seed, genome
schema, shared population size and stage budget; the ONLY differences
are the pipeline plugin and artifact roots. No parallel chain, no idle
interval: the supervisor starts ordinal 1 at ordinal 0's completion
boundary.

The three node profiles pin expected_plan_hash to the canonical-JSON
sha256 of the emitted plan, exactly as app.campaign_supervisor computes
it. Running twice produces byte-identical files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "examples/campaigns/phase_2_eth_curriculum_fleet_v1"
PLAN_ID = "phase-2-eth-curriculum-fleet-v1"
STATE_ROOT = "~/.local/state/agent-multi/doin-campaigns/" + PLAN_ID
ARTIFACT_ROOT = "/home/harveybc/.local/share/agent-multi/eth_curriculum"

ARMS = {
    "en": {
        "job_id": "eth-4h-en-curriculum-sac-shared-v1",
        "domain_id": "trading-asset-policy-eth-4h-en-v1",
        "purpose": "eth_stage_integrated_easy_normal_curriculum_optimization",
        "domain_semantic_hash": (
            "3a90e61d2381ec201c71832f66089559c1196d4d45fd8841b9f103029fd5"
            "995e"),
        "config_dir": "examples/trading/phase_2_eth_en_curriculum_v1",
    },
    "n": {
        "job_id": "eth-4h-n-normal-sac-shared-v1",
        "domain_id": "trading-asset-policy-eth-4h-n-v1",
        "purpose": "eth_normal_only_control_optimization",
        "domain_semantic_hash": (
            "6459f2c45984de7f6674e02fad0b53df6e0516e53e3c7a0ebef3a3ee41b6"
            "8efb"),
        "config_dir": "examples/trading/phase_2_eth_n_normal_v1",
    },
}

PARTICIPANTS = [
    {"node_id": "omega", "supervisor_url": "http://100.99.54.79:8795",
     "workers": ["omega"]},
    {"node_id": "dragon", "supervisor_url": "http://100.110.215.85:8795",
     "workers": ["dragon"]},
    {"node_id": "gamma", "supervisor_url": "http://100.107.204.49:8795",
     "workers": ["gamma-5070ti", "gamma-5090"]},
]

WORKER_ENV = {
    "omega": {"CUDA_VISIBLE_DEVICES": "0"},
    "dragon": {"CUDA_VISIBLE_DEVICES": "0"},
    "gamma-5070ti": {
        "CUDA_VISIBLE_DEVICES":
            "GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519"},
    "gamma-5090": {
        "CUDA_VISIBLE_DEVICES":
            "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"},
}
NODE_WORKERS = {"omega": ["omega"], "dragon": ["dragon"],
                "gamma": ["gamma-5070ti", "gamma-5090"]}
NODE_FILE = {"omega": "omega_node.json", "dragon": "dragon_node.json",
             "gamma-5070ti": "gamma_5070ti_node.json",
             "gamma-5090": "gamma_5090_node.json"}
PYTHON = "/home/harveybc/anaconda3/envs/trading-stack/bin/python"
DOIN_ROOT = "/home/harveybc/Documents/GitHub/doin-node"


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str)


def build_plan() -> dict:
    jobs = []
    for ordinal, arm in enumerate(("en", "n")):
        spec = ARMS[arm]
        root = f"{ARTIFACT_ROOT}/{arm}"
        jobs.append({
            "ordinal": ordinal,
            "job_id": spec["job_id"],
            "domain_id": spec["domain_id"],
            "purpose": spec["purpose"],
            "higher_is_better": True,
            "domain_semantic_hash": spec["domain_semantic_hash"],
            "artifact_handoff": {
                "elite_count": 5,
                "model_path": f"{root}/champion_policy.zip",
                "parameters_path": f"{root}/champion_parameters.json",
                "manifest_path": f"{root}/champion_manifest.json",
                "elite_manifest_path": f"{root}/elite_manifest.json",
            },
            "worker_configs": {
                worker: f"{spec['config_dir']}/{NODE_FILE[worker]}"
                for worker in NODE_FILE
            },
        })
    return {
        "schema_version": "agent_multi.doin_campaign_plan.v1",
        "plan_id": PLAN_ID,
        "$doc": (
            "Immutable two-job paired ETH campaign on ONE swarm/"
            "blockchain: ordinal 0 ETH-EN (stage-integrated easy->"
            "normal solvency curriculum), ordinal 1 ETH-N (normal-only"
            " control) queued next with no parallel chain. Arms share"
            " data (sha256 1b447c66e68495e826c53e2ab2b08ecd3922c8fdc7"
            "35747628f8d0435ebe440f), ga_seed 2703, genome schema,"
            " shared population and stage budget; they differ ONLY in"
            " pipeline plugin and artifact roots. Selection is"
            " lexicographic_weekly_v1 on validation; the"
            " train_validation_l1_score proxy is prohibited in"
            " owner-facing reports."),
        "participants": PARTICIPANTS,
        "jobs": jobs,
    }


def build_profile(node_id: str, plan_hash: str) -> dict:
    return {
        "schema_version": "agent_multi.doin_campaign_profile.v1",
        "node_id": node_id,
        "plan_file": "campaign_plan.json",
        "expected_plan_hash": plan_hash,
        "state_dir": f"{STATE_ROOT}/{node_id}",
        "listen_host": "0.0.0.0",
        "listen_port": 8795,
        "poll_seconds": 5,
        "peer_timeout_seconds": 10,
        "convergence_stability_seconds": 20,
        "stop_timeout_seconds": 30,
        "worker_restart_limit": 5,
        "workers": {
            worker: {
                "doin_node_root": DOIN_ROOT,
                "python": PYTHON,
                "log_level": "INFO",
                "environment": WORKER_ENV[worker],
            }
            for worker in NODE_WORKERS[node_id]
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    plan_path = OUT_DIR / "campaign_plan.json"
    plan_path.write_text(json.dumps(plan, indent=1, sort_keys=False)
                         + "\n", encoding="utf-8")
    plan_hash = hashlib.sha256(
        _canonical_json(plan).encode("utf-8")).hexdigest()
    print(f"campaign_plan.json: plan_hash {plan_hash}")
    for node_id in NODE_WORKERS:
        profile = build_profile(node_id, plan_hash)
        path = OUT_DIR / f"{node_id}_profile.json"
        path.write_text(json.dumps(profile, indent=1, sort_keys=False)
                        + "\n", encoding="utf-8")
        print(f"{path.name}: expected_plan_hash pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
