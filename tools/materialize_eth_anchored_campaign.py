#!/usr/bin/env python3
"""Materialize the ETH current-stack anchored smoke and full DOIN campaign."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.campaign_supervisor import _domain_semantic_hash
from app.canonical_config import resolve_config
from app.config import DEFAULT_VALUES
from examples.scripts.materialize_doin_campaign_nodes import materialize

DOIN_REPO = REPO.parent / "doin-node"
SOURCE_CONFIG = (
    REPO
    / "examples/config/phase_2_eth_curriculum/optimization/phase_2_eth_en_v2.json"
)
ANCHOR_SOURCE_CONFIG = (
    REPO
    / "examples/results/project3_ethusdt_4h_sac_train_val_test_v2/config_out.json"
)
ANCHOR_MODEL = REPO / "examples/models/eth_4h_sac_current_stack_anchor_v1.zip"
ANCHOR_MANIFEST = ANCHOR_MODEL.with_suffix(".manifest.json")
CONFIG_DIR = REPO / "examples/config/phase_2_eth_anchored/optimization"
CAMPAIGN_DIR = REPO / "examples/campaigns/phase_2_eth_anchored_fleet_v1"
RECOVERY_CAMPAIGN_DIR = (
    REPO / "examples/campaigns/phase_2_eth_anchored_full_fleet_v2"
)
TEMPLATE_DIR = DOIN_REPO / "examples/trading/phase_2_eth_en_curriculum_v1"
PLAN_ID = "phase-2-eth-anchored-fleet-v1"
RECOVERY_PLAN_ID = "phase-2-eth-anchored-full-fleet-v2"
ARTIFACT_ROOT = "/home/harveybc/.local/share/agent-multi/eth_anchored"
PYTHON = "/home/harveybc/anaconda3/envs/trading-stack/bin/python"
DOIN_ROOT = "/home/harveybc/Documents/GitHub/doin-node"

PARTICIPANTS = [
    {
        "node_id": "omega",
        "supervisor_url": "http://100.99.54.79:8795",
        "workers": ["omega"],
    },
    {
        "node_id": "dragon",
        "supervisor_url": "http://100.110.215.85:8795",
        "workers": ["dragon"],
    },
    {
        "node_id": "gamma",
        "supervisor_url": "http://100.107.204.49:8795",
        "workers": ["gamma-5070ti", "gamma-5090"],
    },
]
NODE_WORKERS = {
    "omega": ["omega"],
    "dragon": ["dragon"],
    "gamma": ["gamma-5070ti", "gamma-5090"],
}
NODE_FILE = {
    "omega": "omega_node.json",
    "dragon": "dragon_node.json",
    "gamma-5070ti": "gamma_5070ti_node.json",
    "gamma-5090": "gamma_5090_node.json",
}
WORKER_ENV = {
    "omega": {"CUDA_VISIBLE_DEVICES": "0"},
    "dragon": {"CUDA_VISIBLE_DEVICES": "0"},
    "gamma-5070ti": {
        "CUDA_VISIBLE_DEVICES": "GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519"
    },
    "gamma-5090": {
        "CUDA_VISIBLE_DEVICES": "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"
    },
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _anchored_schema(source_schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = {
        "learning_rate_gene",
        "batch_size_gene",
        "buffer_size_gene",
        "learning_starts_gene",
        "gamma_gene",
        "tau_gene",
        "train_freq_gene",
        "gradient_steps_gene",
        "entropy_gene",
        "action_threshold_gene",
        "relative_volume_gene",
        "stop_loss_atr_gene",
        "take_profit_atr_gene",
        "entry_order_mode_gene",
        "market_urgency_threshold_gene",
        "market_max_spread_bps_gene",
        "stop_breakout_threshold_gene",
        "limit_offset_atr_gene",
        "stop_offset_atr_gene",
    }
    schema = [copy.deepcopy(gene) for gene in source_schema
              if gene.get("name") in names]
    for gene in schema:
        if gene["name"] == "learning_starts_gene":
            gene["choices"] = [1_000, 2_000, 5_000, 10_000]
        elif gene["name"] == "entropy_gene":
            gene["choices"] = ["auto", 0.01, 0.05, 0.1, 0.2]
    return schema


def build_config(
    *, smoke: bool, revision: int = 1, artifact_arm: str | None = None
) -> dict[str, Any]:
    if not ANCHOR_MODEL.exists():
        raise FileNotFoundError(f"tracked ETH anchor is missing: {ANCHOR_MODEL}")
    source = _load(SOURCE_CONFIG)
    anchor = _load(ANCHOR_SOURCE_CONFIG)
    anchor_hash = _sha256(ANCHOR_MODEL)
    config = copy.deepcopy(source)
    arm = "smoke" if smoke else "full"
    arm_tag = artifact_arm or arm
    experiment = config["experiment"]
    experiment.update({
        "name": f"phase_2_eth_anchored_{arm}_v{revision}",
        "curriculum_arm": "anchored_easy_normal",
        "role": "champion_anchored_model_execution_optimization",
        "description": (
            "ETH easy->normal optimization warm-started from the verified "
            "current-stack active policy; observation and network shape fixed"
        ),
    })
    # The current-stack anchor's 2724-element observation includes its legacy
    # raw-price window. Pin that exact shape until a separately trained
    # feature-only successor exists; silently dropping the 32 prices would
    # make the artifact unloadable.
    experiment.setdefault("legacy_flat", {})["include_price_window"] = True

    environment = config["environment"]
    environment.update({
        "window_size": int(anchor["window_size"]),
        "feature_scaling": anchor["feature_scaling"],
        "feature_scaling_window": int(anchor["feature_scaling_window"]),
        # The legacy config records null while the current feature-window
        # preprocessor requires an explicit finite clip.  The compatible v2
        # artifact was generated under the v2 contract's value of 10.
        "feature_clip": 10,
        "feature_columns": list(anchor["feature_columns"]),
        "feature_binary_columns": list(anchor["feature_binary_columns"]),
        "include_agent_state": bool(anchor["include_agent_state"]),
        "include_price_window": bool(anchor["include_price_window"]),
        "require_feature_aware_preprocessor": False,
        "continuous_action_threshold": float(
            anchor["continuous_action_threshold"]
        ),
    })
    config["data"]["feature_list"] = list(anchor["feature_columns"])
    config["asset_policy"].update({
        "window_size": int(anchor["window_size"]),
        "continuous_action_threshold": float(
            anchor["continuous_action_threshold"]
        ),
    })

    training = config["training"]
    training.update({
        "warm_start_model": str(ANCHOR_MODEL),
        "warm_start_model_sha256": anchor_hash,
        "warm_start_expand_observation_space": False,
        "warm_start_baseline_checkpoint_enabled": True,
        "net_arch": list(anchor["net_arch"]),
        "learning_rate": float(anchor["learning_rate"]),
        "batch_size": int(anchor["batch_size"]),
        "buffer_size": int(anchor["buffer_size"]),
        "learning_starts": int(anchor["learning_starts"]),
        "gamma": float(anchor["gamma"]),
        "tau": float(anchor["tau"]),
        "train_freq": int(anchor["train_freq"]),
        "gradient_steps": int(anchor["gradient_steps"]),
        "ent_coef": anchor["ent_coef"],
        "use_sde": bool(anchor["use_sde"]),
        "easy_min_trades": 12,
        "early_stop_min_trades": 0,
        "early_stop_min_train_tail_trades": 0,
        "early_stop_min_validation_trades": 12,
        "evaluate_test_split": False,
    })
    config["risk"].update({
        "rel_volume": float(anchor["rel_volume"]),
        "k_sl": float(anchor["k_sl"]),
        "k_tp": float(anchor["k_tp"]),
    })

    optimization = config["optimization"]
    schema = _anchored_schema(optimization["mixed_genome_schema"])
    initial = {
        "learning_rate_gene": float(anchor["learning_rate"]),
        "batch_size_gene": int(anchor["batch_size"]),
        "buffer_size_gene": int(anchor["buffer_size"]),
        "learning_starts_gene": int(anchor["learning_starts"]),
        "gamma_gene": float(anchor["gamma"]),
        "tau_gene": float(anchor["tau"]),
        "train_freq_gene": int(anchor["train_freq"]),
        "gradient_steps_gene": int(anchor["gradient_steps"]),
        "entropy_gene": anchor["ent_coef"],
        "action_threshold_gene": float(anchor["continuous_action_threshold"]),
        "relative_volume_gene": float(anchor["rel_volume"]),
        "stop_loss_atr_gene": float(anchor["k_sl"]),
        "take_profit_atr_gene": float(anchor["k_tp"]),
        "entry_order_mode_gene": environment["entry_order_mode"],
        "market_urgency_threshold_gene": environment[
            "market_urgency_threshold"
        ],
        "market_max_spread_bps_gene": environment["market_max_spread_bps"],
        "stop_breakout_threshold_gene": environment[
            "stop_breakout_threshold"
        ],
        "limit_offset_atr_gene": environment["limit_offset_atr_multiple"],
        "stop_offset_atr_gene": environment["stop_offset_atr_multiple"],
    }
    model_params = [
        "learning_rate_gene", "batch_size_gene", "buffer_size_gene",
        "learning_starts_gene", "gamma_gene", "tau_gene",
        "train_freq_gene", "gradient_steps_gene", "entropy_gene",
    ]
    execution_params = [
        "action_threshold_gene", "relative_volume_gene",
        "stop_loss_atr_gene", "take_profit_atr_gene",
        "entry_order_mode_gene", "market_urgency_threshold_gene",
        "market_max_spread_bps_gene", "stop_breakout_threshold_gene",
        "limit_offset_atr_gene", "stop_offset_atr_gene",
    ]
    optimization.update({
        "mixed_genome_schema": schema,
        "initial_candidate_decoded": initial,
        "optimization_min_trades_by_split": {
            "train_tail": 0,
            "validation": 12,
        },
        "optimization_action_collapse_splits": ["validation"],
        "mixed_genome_fixed_observation_contract": {
            "anchor_sha256": anchor_hash,
            "window_size": environment["window_size"],
            "feature_count": len(environment["feature_columns"]),
            "feature_scaling": environment["feature_scaling"],
            "feature_scaling_window": environment["feature_scaling_window"],
            "net_arch": training["net_arch"],
        },
    })
    for key in (
        "mixed_genome_feature_groups",
        "mixed_genome_required_feature_group",
        "mixed_genome_repair_rules",
        "mixed_genome_dropped_groups_eth",
    ):
        optimization.pop(key, None)

    if smoke:
        training.update({
            "epoch_timesteps": 1_000,
            "max_epochs": 2,
            "l1_patience": 2,
            "l1_patience_start_epoch": 1,
            "l1_min_checkpoint_timesteps": 1,
            "easy_epoch_timesteps": 1_000,
            "easy_max_epochs": 1,
            "easy_patience": 1,
            "total_timesteps": 2_000,
        })
        optimization.update({
            "ga_population": 4,
            "ga_generations": 1,
            "ga_eval_timesteps": 2_000,
            "optimization_patience": 1,
            "optimization_stages": [{
                "name": "anchored_smoke",
                "params": "all",
                "generations": 1,
                "patience": 1,
            }],
        })
    else:
        optimization.update({
            "ga_population": 20,
            "ga_generations": 18,
            "optimization_stages": [
                {
                    "name": "model_training",
                    "params": model_params,
                    "generations": 8,
                    "patience": 5,
                },
                {
                    "name": "execution_risk",
                    "params": execution_params,
                    "generations": 4,
                    "patience": 3,
                },
                {
                    "name": "joint_refinement",
                    "params": "all",
                    "generations": 6,
                    "patience": 5,
                },
            ],
        })

    artifact_root = f"{ARTIFACT_ROOT}/{arm_tag}"
    config["artifacts"].update({
        "artifact_root": artifact_root,
        "optimizer_output_file": f"{artifact_root}/optimizer_output.json",
        "results_file": f"{artifact_root}/results.json",
        "resolved_config_file": f"{artifact_root}/resolved_config.json",
        "config_manifest_file": f"{artifact_root}/config_manifest.json",
        "return_trace_dir": f"{artifact_root}/return_traces",
        "save_model": f"{artifact_root}/final_policy.zip",
    })
    optimization.update({
        "optimization_candidate_history": f"{artifact_root}/candidate_history.csv",
        "optimization_champion_model_file": f"{artifact_root}/champion_policy.zip",
        "optimization_parameters_file": f"{artifact_root}/champion_parameters.json",
        "optimization_resume_file": f"{artifact_root}/optimization_resume.json",
        "optimization_statistics": f"{artifact_root}/optimization_stats.json",
    })

    runtime = resolve_config(
        DEFAULT_VALUES,
        file_config=config,
    ).runtime
    if runtime.get("warm_start_model") != str(ANCHOR_MODEL):
        raise ValueError("canonical resolver lost the ETH warm-start path")
    if runtime.get("window_size") != int(anchor["window_size"]):
        raise ValueError("canonical resolver changed the anchored window")
    if runtime.get("include_price_window") is not True:
        raise ValueError("canonical resolver changed the anchored price window")
    return config


def _build_profile(
    node_id: str, plan_hash: str, *, plan_id: str = PLAN_ID
) -> dict[str, Any]:
    return {
        "schema_version": "agent_multi.doin_campaign_profile.v1",
        "node_id": node_id,
        "plan_file": "campaign_plan.json",
        "expected_plan_hash": plan_hash,
        "state_dir": (
            f"~/.local/state/agent-multi/doin-campaigns/{plan_id}/{node_id}"
        ),
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


def materialize_all() -> dict[str, Any]:
    manifest = _load(ANCHOR_MANIFEST)
    if manifest["artifact_sha256"] != _sha256(ANCHOR_MODEL):
        raise ValueError("ETH anchor manifest hash does not match artifact")

    jobs = []
    for ordinal, smoke in enumerate((True, False)):
        arm = "smoke" if smoke else "full"
        config = build_config(smoke=smoke)
        config_path = CONFIG_DIR / f"phase_2_eth_anchored_{arm}_v1.json"
        _write(config_path, config)
        domain_id = f"trading-asset-policy-eth-4h-anchored-{arm}-v1"
        node_dir_name = f"phase_2_eth_anchored_{arm}_v1"
        node_dir = DOIN_REPO / "examples/trading" / node_dir_name
        load_config = str(config_path)
        materialize(
            template_dir=TEMPLATE_DIR,
            output_dir=node_dir,
            canonical_config=config_path,
            load_config=load_config,
            domain_id=domain_id,
            campaign_slug=f"eth-anchored-{arm}-v1",
        )
        omega_config = _load(node_dir / "omega_node.json")
        semantic_hash = _domain_semantic_hash(omega_config)
        artifact_root = f"{ARTIFACT_ROOT}/{arm}"
        jobs.append({
            "ordinal": ordinal,
            "job_id": f"eth-4h-anchored-{arm}-sac-shared-v1",
            "domain_id": domain_id,
            "purpose": (
                "four_worker_single_chain_convergence_and_activity_smoke"
                if smoke
                else "eth_anchored_easy_normal_model_execution_optimization"
            ),
            "higher_is_better": True,
            "domain_semantic_hash": semantic_hash,
            "artifact_handoff": {
                "elite_count": 5,
                "model_path": f"{artifact_root}/champion_policy.zip",
                "parameters_path": f"{artifact_root}/champion_parameters.json",
                "manifest_path": f"{artifact_root}/champion_manifest.json",
                "elite_manifest_path": f"{artifact_root}/elite_manifest.json",
            },
            "worker_configs": {
                worker: f"examples/trading/{node_dir_name}/{NODE_FILE[worker]}"
                for worker in NODE_FILE
            },
        })

    plan = {
        "schema_version": "agent_multi.doin_campaign_plan.v1",
        "plan_id": PLAN_ID,
        "$doc": (
            "Sequential ETH campaign. Each job is one shared-population DOIN "
            "chain used by all four workers. Ordinal 0 proves four-worker "
            "convergence and normal activity with four candidates; ordinal 1 "
            "starts automatically after completion and performs the full "
            "champion-anchored easy-to-normal optimization. No job may run as "
            "an independent per-machine chain."
        ),
        "participants": PARTICIPANTS,
        "jobs": jobs,
    }
    _write(CAMPAIGN_DIR / "campaign_plan.json", plan)
    plan_hash = hashlib.sha256(
        _canonical_json(plan).encode("utf-8")
    ).hexdigest()
    for node_id in NODE_WORKERS:
        _write(
            CAMPAIGN_DIR / f"{node_id}_profile.json",
            _build_profile(node_id, plan_hash),
        )
    return {"plan_hash": plan_hash, "jobs": jobs}


def materialize_recovery_full_v2() -> dict[str, Any]:
    """Create a fresh full domain after rejecting the incompatible v1 run."""
    manifest = _load(ANCHOR_MANIFEST)
    if manifest["artifact_sha256"] != _sha256(ANCHOR_MODEL):
        raise ValueError("ETH anchor manifest hash does not match artifact")

    config = build_config(
        smoke=False,
        revision=2,
        artifact_arm="full_v2",
    )
    config_path = CONFIG_DIR / "phase_2_eth_anchored_full_v2.json"
    _write(config_path, config)
    domain_id = "trading-asset-policy-eth-4h-anchored-full-v2"
    node_dir_name = "phase_2_eth_anchored_full_v2"
    node_dir = DOIN_REPO / "examples/trading" / node_dir_name
    materialize(
        template_dir=TEMPLATE_DIR,
        output_dir=node_dir,
        canonical_config=config_path,
        load_config=str(config_path),
        domain_id=domain_id,
        campaign_slug="eth-anchored-full-v2",
    )
    semantic_hash = _domain_semantic_hash(_load(node_dir / "omega_node.json"))
    artifact_root = f"{ARTIFACT_ROOT}/full_v2"
    job = {
        "ordinal": 0,
        "job_id": "eth-4h-anchored-full-sac-shared-v2",
        "domain_id": domain_id,
        "purpose": (
            "eth_anchored_easy_normal_optimization_with_weight_only_"
            "warm_start"
        ),
        "higher_is_better": True,
        "domain_semantic_hash": semantic_hash,
        "artifact_handoff": {
            "elite_count": 5,
            "model_path": f"{artifact_root}/champion_policy.zip",
            "parameters_path": f"{artifact_root}/champion_parameters.json",
            "manifest_path": f"{artifact_root}/champion_manifest.json",
            "elite_manifest_path": f"{artifact_root}/elite_manifest.json",
        },
        "worker_configs": {
            worker: f"examples/trading/{node_dir_name}/{NODE_FILE[worker]}"
            for worker in NODE_FILE
        },
    }
    plan = {
        "schema_version": "agent_multi.doin_campaign_plan.v1",
        "plan_id": RECOVERY_PLAN_ID,
        "$doc": (
            "Fresh full ETH domain after the full-v1 entropy-mode load defect. "
            "All four workers share this one population and chain. The valid "
            "v1 smoke and rejected full-v1 chain remain immutable evidence."
        ),
        "participants": PARTICIPANTS,
        "jobs": [job],
    }
    _write(RECOVERY_CAMPAIGN_DIR / "campaign_plan.json", plan)
    plan_hash = hashlib.sha256(
        _canonical_json(plan).encode("utf-8")
    ).hexdigest()
    for node_id in NODE_WORKERS:
        _write(
            RECOVERY_CAMPAIGN_DIR / f"{node_id}_profile.json",
            _build_profile(
                node_id,
                plan_hash,
                plan_id=RECOVERY_PLAN_ID,
            ),
        )
    return {"plan_hash": plan_hash, "jobs": [job]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-v2-only",
        action="store_true",
        help="materialize only the clean full-v2 recovery domain",
    )
    args = parser.parse_args()
    result = (
        materialize_recovery_full_v2()
        if args.full_v2_only
        else materialize_all()
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
