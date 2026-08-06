#!/usr/bin/env python3
"""Materialize the WP7 smoke campaign (correction order §8).

New plan/domain/genesis — never the contaminated chain. One ETH-EN
smoke domain: population 4, ONE stage, ONE generation, tiny per-epoch
budget, corrected comparison contract. Derives from the corrected
phase_2_eth_en_v2 arm config; runs the same fail-closed token scan and
runtime metric assertions.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.materialize_eth_curriculum_configs import (  # noqa: E402
    FOREIGN_TOKENS, _flatten)

DOIN = Path("/home/harveybc/Documents/GitHub/doin-node")
BASE = (REPO / "examples/config/phase_2_eth_curriculum/optimization/"
        "phase_2_eth_en_v2.json")
OUT_CONFIG = (REPO / "examples/config/phase_2_eth_curriculum/optimization/"
              "phase_2_eth_en_smoke_v1.json")
DOMAIN_ID = "trading-asset-policy-eth-4h-en-smoke-v1"
PLAN_ID = "phase-2-eth-smoke-v1"
CAMPAIGN_DIR = REPO / "examples/campaigns/phase_2_eth_smoke_v1"
NODE_DIR = DOIN / "examples/trading/phase_2_eth_en_smoke_v1"
SMOKE_ROOT = "${ARTIFACT_ROOT}/eth_smoke_v1/en"

PARTICIPANTS = [
    {"node_id": "omega", "supervisor_url": "http://192.0.2.10:8795",
     "workers": ["omega"]},
    {"node_id": "dragon", "supervisor_url": "http://192.0.2.11:8795",
     "workers": ["dragon"]},
    {"node_id": "gamma", "supervisor_url": "http://192.0.2.12:8795",
     "workers": ["gamma-5070ti", "gamma-5090"]},
]
WORKER_ENV = {
    "omega": {"CUDA_VISIBLE_DEVICES": "0"},
    "dragon": {"CUDA_VISIBLE_DEVICES": "0"},
    "gamma-5070ti": {"CUDA_VISIBLE_DEVICES":
                     "GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519"},
    "gamma-5090": {"CUDA_VISIBLE_DEVICES":
                   "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"},
}
NODE_WORKERS = {"omega": ["omega"], "dragon": ["dragon"],
                "gamma": ["gamma-5070ti", "gamma-5090"]}
NODE_FILE = {"omega": "omega_node.json", "dragon": "dragon_node.json",
             "gamma-5070ti": "gamma_5070ti_node.json",
             "gamma-5090": "gamma_5090_node.json"}
PYTHON = "/home/harveybc/anaconda3/envs/trading-stack/bin/python"


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str)


def build_smoke_config() -> dict:
    config = json.loads(BASE.read_text())
    config["experiment"]["name"] = "phase_2_eth_en_smoke_v1"
    config["experiment"]["description"] = (
        "WP7 smoke: 4-candidate, one-stage, one-generation ETH-EN"
        " mechanism proof — never a champion claim")
    opt = config["optimization"]
    opt["ga_population"] = 4
    opt["shared_population_size"] = 4
    genes = [g["name"] for g in opt["mixed_genome_schema"]]
    opt["optimization_stages"] = [
        {"name": "smoke_all_params", "generations": 1, "params": genes,
         "patience": 1},
    ]
    for key in ("optimization_candidate_history",
                "optimization_champion_model_file",
                "optimization_parameters_file",
                "optimization_resume_file",
                "optimization_statistics"):
        opt[key] = SMOKE_ROOT + "/" + str(opt[key]).rsplit("/", 1)[-1]
    training = config["training"]
    training["epoch_timesteps"] = 2000
    training["max_epochs"] = 2
    training["l1_patience"] = 2
    training["l1_min_checkpoint_timesteps"] = 1
    training["execution_cost_curriculum_epochs"] = 2
    training["easy_max_epochs"] = 1
    training["easy_patience"] = 1
    for key, value in list((config.get("artifacts") or {}).items()):
        if isinstance(value, str) and value.startswith("${ARTIFACT_ROOT}"):
            config["artifacts"][key] = (
                SMOKE_ROOT + "/" + value.rsplit("/", 1)[-1])

    flat = _flatten(config)
    for path_key, value in flat.items():
        if isinstance(value, str):
            lowered = value.lower()
            for token in FOREIGN_TOKENS:
                if token in lowered:
                    raise SystemExit(
                        f"foreign token {token!r} at {path_key}")
    from app.canonical_config import resolve_config
    from app.metrics import compute_optimization_fitness
    runtime = resolve_config({}, file_config=config).runtime
    assert runtime.get("selection_metric") == "lexicographic_weekly_v1"
    assert runtime.get("optimization_metric") == "lexicographic_weekly_v1"
    probe = {"mean_weekly_return": 0.001, "max_drawdown_fraction": 0.1,
             "total_return": 0.05,
             "trades_total": int(runtime.get("selection_min_trades", 0)) + 1}
    assert compute_optimization_fitness(probe, runtime, object()) > 0
    return config


def main() -> int:
    config = build_smoke_config()
    text = json.dumps(config, indent=1, sort_keys=True) + "\n"
    OUT_CONFIG.write_text(text, encoding="utf-8")
    print(f"{OUT_CONFIG.name}: sha256"
          f" {hashlib.sha256(text.encode()).hexdigest()}")

    subprocess.run([
        PYTHON, str(REPO / "examples/scripts/materialize_doin_campaign_nodes.py"),
        "--template-dir", str(DOIN / "examples/trading/"
                              "phase_1_asset_policy_usdcad_4h_protected_easy_v2"),
        "--output-dir", str(NODE_DIR),
        "--canonical-config", str(OUT_CONFIG.relative_to(REPO)),
        "--load-config", str(OUT_CONFIG),
        "--domain-id", DOMAIN_ID,
        "--campaign-slug", "eth-en-smoke-v1",
    ], check=True, cwd=REPO)

    from app.campaign_supervisor import _domain_semantic_hash
    hashes = set()
    for node_file in NODE_FILE.values():
        node = json.loads((NODE_DIR / node_file).read_text())
        hashes.add(_domain_semantic_hash(node))
    assert len(hashes) == 1, hashes
    semantic = hashes.pop()
    print(f"domain semantic hash: {semantic}")

    plan = {
        "schema_version": "agent_multi.doin_campaign_plan.v1",
        "plan_id": PLAN_ID,
        "$doc": ("WP7 smoke campaign: fresh genesis, one ETH-EN smoke"
                 " job, population 4, one generation. Never the"
                 " contaminated chain; full swarm remains disabled"
                 " until Musashi accepts this packet."),
        "participants": PARTICIPANTS,
        "jobs": [{
            "ordinal": 0,
            "job_id": "eth-4h-en-smoke-v1",
            "domain_id": DOMAIN_ID,
            "purpose": "wp7_smoke_mechanism_proof",
            "higher_is_better": True,
            "domain_semantic_hash": semantic,
            "artifact_handoff": {
                "elite_count": 1,
                "model_path": "/home/harveybc/.local/share/agent-multi/eth_smoke_v1/en/champion_policy.zip",
                "parameters_path": "/home/harveybc/.local/share/agent-multi/eth_smoke_v1/en/champion_parameters.json",
                "manifest_path": "/home/harveybc/.local/share/agent-multi/eth_smoke_v1/en/champion_manifest.json",
                "elite_manifest_path": "/home/harveybc/.local/share/agent-multi/eth_smoke_v1/en/elite_manifest.json",
            },
            "worker_configs": {
                worker: f"examples/trading/phase_2_eth_en_smoke_v1/{name}"
                for worker, name in NODE_FILE.items()
            },
        }],
    }
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    (CAMPAIGN_DIR / "campaign_plan.json").write_text(
        json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    plan_hash = hashlib.sha256(
        _canonical_json(plan).encode()).hexdigest()
    print(f"plan_hash: {plan_hash}")
    for node_id, workers in NODE_WORKERS.items():
        profile = {
            "schema_version": "agent_multi.doin_campaign_profile.v1",
            "node_id": node_id,
            "plan_file": "campaign_plan.json",
            "expected_plan_hash": plan_hash,
            "state_dir": ("~/.local/state/agent-multi/doin-campaigns/"
                          f"{PLAN_ID}/{node_id}"),
            "listen_host": "0.0.0.0",
            "listen_port": 8795,
            "poll_seconds": 5,
            "peer_timeout_seconds": 10,
            "convergence_stability_seconds": 20,
            "stop_timeout_seconds": 30,
            "worker_restart_limit": 3,
            "workers": {
                worker: {
                    "doin_node_root": str(DOIN),
                    "python": PYTHON,
                    "log_level": "INFO",
                    "environment": WORKER_ENV[worker],
                } for worker in workers
            },
        }
        (CAMPAIGN_DIR / f"{node_id}_profile.json").write_text(
            json.dumps(profile, indent=1) + "\n", encoding="utf-8")
        print(f"{node_id}_profile.json pinned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
