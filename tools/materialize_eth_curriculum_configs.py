#!/usr/bin/env python3
"""Materialize the paired ETH optimization configs (ETH order WP-E).

Deterministic transform: the working USDCAD v2 full-genome contract is
the structural template; data/environment facts come from the proven ETH
SAC v2 run config; the split boundaries and dataset hash are the order's
frozen contract; selection is the transparent lexicographic contract
(§9). Both arms share EVERYTHING except artifact roots and the arm
description — the pipeline plugin difference (curriculum vs normal-only)
lives in the DOIN node configs, not here.

Running twice produces byte-identical files; sha256 hashes print for the
evidence packet.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = (REPO / "examples/config/phase_1_asset_policy/optimization/"
            "phase_1_asset_policy_usdcad_4h_protected_easy_v2.json")
ETH_BASE = (REPO / "examples/results/"
            "project3_ethusdt_4h_sac_train_val_test_v2/config_out.json")
OUT_DIR = REPO / "examples/config/phase_2_eth_curriculum/optimization"
DATA_FILE = ("/home/harveybc/Documents/GitHub/predictor/examples/data/"
             "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
DATA_SHA256 = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435"
               "ebe440f")
GA_SEED = 2703

SPLITS = {
    "train_start": "2017-09-28T04:00:00",
    "train_end": "2023-12-31T23:59:59",
    "validation_start": "2024-01-01T00:00:00",
    "validation_end": "2024-12-31T23:59:59",
    "test_start": "2025-01-01T00:00:00",
    "test_end": "2025-12-31T23:59:59",
}


def build(arm: str) -> dict:
    template = json.loads(TEMPLATE.read_text())
    eth = json.loads(ETH_BASE.read_text())
    config = copy.deepcopy(template)

    features = list(eth["feature_columns"])
    binary = list(eth["feature_binary_columns"])
    available = set(features) | set(binary)

    config["experiment"]["description"] = (
        "ETH-EN: stage-integrated easy->normal solvency curriculum arm"
        if arm == "en" else
        "ETH-N: normal-only control arm (identical data/seed/population/"
        "budget)")
    config["experiment"]["curriculum_arm"] = arm

    data = config["data"]
    data["asset"] = "ETHUSD"
    data["input_data_file"] = DATA_FILE
    data["input_data_sha256"] = DATA_SHA256
    data["date_column"] = "DATE_TIME"
    data["feature_list"] = features
    data["feature_binary_columns"] = binary
    data["dataset_manifest_file"] = DATA_FILE.replace(
        ".csv", ".manifest.json")
    data["data_profile"] = "eth_tech_stat_model_ready"
    data.update(SPLITS)

    env = config["environment"]
    env["feature_columns"] = features
    env["feature_binary_columns"] = binary
    for key in ("commission", "leverage", "position_size", "rel_volume",
                "continuous_action_threshold", "feature_scaling",
                "feature_scaling_window", "atr_period"):
        if key in eth:
            env[key] = eth[key]
    env.pop("execution_difficulty", None)
    env["solvency_mode"] = "normal_realistic"   # training arm overrides
    env["env_mode"] = "training"

    risk = config.get("risk", {})
    for key in ("commission", "atr_period", "k_sl", "k_tp",
                "initial_cash"):
        if key in eth:
            risk[key] = eth[key]

    objectives = config["objectives"]
    objectives["selection_metric"] = "lexicographic_weekly_v1"
    objectives["primary_report_metrics"] = [
        "mean_weekly_return", "max_drawdown_fraction", "total_return",
        "annualized_return", "trades_total",
    ]
    objectives["proxy_metrics_prohibited_in_owner_reports"] = [
        "train_validation_l1_score",
    ]

    optimization = config["optimization"]
    optimization["metric"] = "lexicographic_weekly_v1"
    optimization["optimization_metric"] = "lexicographic_weekly_v1"
    optimization["ga_seed"] = GA_SEED
    optimization.pop("e4_baseline_job_id", None)
    optimization.pop("e4_baseline_validation_annual_rap", None)

    # ETH genome revision: prune feature groups to features the ETH
    # dataset actually carries. A group with no ETH members is REMOVED
    # together with its gene — recorded explicitly, never silent. Both
    # arms share this identical revised schema, which is the pairing
    # requirement; the required feature group must survive.
    groups = optimization.get("mixed_genome_feature_groups", {})
    pruned = {}
    dropped_groups = []
    for group, members in groups.items():
        kept = [m for m in members if m in available]
        if kept:
            pruned[group] = kept
        else:
            dropped_groups.append(group)
    required = optimization.get("mixed_genome_required_feature_group")
    if required and required not in pruned:
        raise SystemExit(
            f"required feature group {required!r} has no ETH members —"
            " refusing")
    optimization["mixed_genome_feature_groups"] = pruned
    optimization["mixed_genome_dropped_groups_eth"] = sorted(
        dropped_groups)
    dropped_genes = {f"feature_group__{g}" for g in dropped_groups}
    schema = optimization.get("mixed_genome_schema")
    if isinstance(schema, list):
        optimization["mixed_genome_schema"] = [
            gene for gene in schema
            if gene.get("name") not in dropped_genes]
    for section in (optimization.get("initial_candidate_decoded"),):
        if isinstance(section, dict):
            for gene in dropped_genes:
                section.pop(gene, None)
    stages = optimization.get("optimization_stages")
    if isinstance(stages, list):
        for stage in stages:
            params = stage.get("params")
            if isinstance(params, list):
                stage["params"] = [
                    p for p in params if p not in dropped_genes]

    config["experiment"]["name"] = f"phase_2_eth_{arm}_curriculum_v2"
    # Dead template baggage; carries foreign sample paths (fails the
    # WP1 token scan) and describes no runtime behaviour.
    config["experiment"].pop("legacy_flat", None)

    # AUD-F1-20260805-110: every optimizer output lives below the ETH
    # arm root; nothing may reference the USDCAD namespace.
    arm_root = f"${{ARTIFACT_ROOT}}/eth_curriculum_v2/{arm}"
    for key in ("optimization_candidate_history",
                "optimization_champion_model_file",
                "optimization_parameters_file",
                "optimization_resume_file",
                "optimization_statistics"):
        value = optimization.get(key)
        if isinstance(value, str):
            optimization[key] = (
                arm_root + "/" + value.rsplit("/", 1)[-1])

    # AUD-F1-20260805-113: preprocessing_mode 'none' is forbidden while
    # the observation contract requires a feature-aware preprocessor and
    # no content-hashed precomputed causal feature contract exists.
    for gene in optimization.get("mixed_genome_schema", []):
        if gene.get("name") == "preprocessing_mode":
            gene["choices"] = [
                c for c in gene["choices"] if c != "none"]
    optimization["mixed_genome_repair_rules"] = [
        {"rule": "forbid_value", "gene": "preprocessing_mode",
         "value": "none",
         "reason": ("feature-aware observation contract has no"
                    " precomputed causal feature contract"),
         "repair": "resample_categorical"},
    ]

    training = config["training"]
    training["selection_metric"] = "lexicographic_weekly_v1"
    training["selection_min_trades"] = int(
        (optimization.get("optimization_min_trades_by_split") or {})
        .get("validation", 12))
    for key in ("batch_size", "buffer_size", "gamma", "gradient_steps",
                "ent_coef", "epoch_timesteps", "device"):
        if key in eth:
            training[key] = eth[key]
    if arm == "en":
        training["solvency_curriculum_enabled"] = True
        training["easy_max_epochs"] = 4
        training["easy_patience"] = 2
        training["pipeline_plugin"] = "rl_pipeline_with_solvency_curriculum"
    else:
        training["solvency_curriculum_enabled"] = False
        training["pipeline_plugin"] = "rl_pipeline_with_validation"
    optimization["plugin"] = "project3_full_genome_optimizer"

    # Canonical runtime keys must agree across sections: the action
    # threshold appears in both asset_policy and environment.
    asset_policy = config.get("asset_policy", {})
    if "continuous_action_threshold" in eth:
        asset_policy["continuous_action_threshold"] = (
            eth["continuous_action_threshold"])

    root = f"${{ARTIFACT_ROOT}}/eth_curriculum_v2/{arm}"
    artifacts = config.get("artifacts", {})
    for key, value in list(artifacts.items()):
        if isinstance(value, str):
            artifacts[key] = value.replace(
                "${ARTIFACT_ROOT}/protected_easy/usdcad_4h", root)
    config["artifacts"] = artifacts
    return config


FOREIGN_TOKENS = ("usdcad", "eurusd", "gbpusd", "audusd", "usdjpy",
                  "btcusdt", "solusdt", "adausdt", "protected_easy")

ALLOWED_ARM_DIFF_PREFIXES = (
    "/artifacts/", "/experiment/curriculum_arm", "/experiment/name",
    "/experiment/description", "/optimization/optimization_candidate_history",
    "/optimization/optimization_champion_model_file",
    "/optimization/optimization_parameters_file",
    "/optimization/optimization_resume_file",
    "/optimization/optimization_statistics",
    "/training/solvency_curriculum_enabled", "/training/easy_max_epochs",
    "/training/easy_patience", "/training/pipeline_plugin",
)


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}/{key}"))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, sort_keys=True)
    else:
        out[prefix] = obj
    return out


def validate(config: dict, arm: str) -> None:
    """WP1 fail-closed assertions (AUD-F1-20260805-108/110/113)."""
    arm_root = f"${{ARTIFACT_ROOT}}/eth_curriculum_v2/{arm}"
    flat = _flatten(config)

    for path_key, value in flat.items():
        if not isinstance(value, str):
            continue
        lowered = value.lower()
        for token in FOREIGN_TOKENS:
            if token in lowered:
                raise SystemExit(
                    f"foreign asset token {token!r} at {path_key}:"
                    f" {value!r}")

    import sys
    sys.path.insert(0, str(REPO))
    from app.canonical_config import resolve_config
    from app.metrics import compute_optimization_fitness
    runtime = resolve_config({}, file_config=config).runtime
    for key in ("selection_metric", "optimization_metric"):
        if runtime.get(key) != "lexicographic_weekly_v1":
            raise SystemExit(
                f"runtime {key}={runtime.get(key)!r} !="
                " lexicographic_weekly_v1")
    probe = {"mean_weekly_return": 0.001, "max_drawdown_fraction": 0.1,
             "total_return": 0.05,
             "trades_total": int(runtime.get("selection_min_trades", 0))
             + 1}
    fitness = compute_optimization_fitness(probe, runtime, object())
    if not fitness > 0:
        raise SystemExit("configured objective did not resolve")

    optimization = config["optimization"]
    for key in ("optimization_candidate_history",
                "optimization_champion_model_file",
                "optimization_parameters_file",
                "optimization_resume_file",
                "optimization_statistics"):
        if not str(optimization.get(key, "")).startswith(arm_root):
            raise SystemExit(
                f"{key} escapes the {arm} arm root:"
                f" {optimization.get(key)!r}")
    # save_config/save_log are per-candidate working sidecars written
    # inside each candidate's own run directory; a shared arm-root path
    # would collide concurrent candidates. Everything else must live
    # under the arm root.
    per_candidate_sidecars = {"save_config", "save_log"}
    for key, value in (config.get("artifacts") or {}).items():
        if not isinstance(value, str):
            continue
        if key in per_candidate_sidecars and value.startswith("./"):
            continue
        if not value.startswith(arm_root):
            raise SystemExit(
                f"artifacts.{key} escapes the {arm} arm root: {value!r}")
    if config["experiment"]["name"] != f"phase_2_eth_{arm}_curriculum_v2":
        raise SystemExit("experiment identity is not the ETH arm")
    if config["data"]["asset"] != "ETHUSD":
        raise SystemExit("data.asset is not ETHUSD")

    for gene in optimization.get("mixed_genome_schema", []):
        if gene.get("name") == "preprocessing_mode" and                 "none" in gene.get("choices", []):
            raise SystemExit("preprocessing_mode still offers 'none'")
    if not optimization.get("mixed_genome_repair_rules"):
        raise SystemExit("mixed_genome_repair_rules is empty")
    from optimizer_plugins.project3_full_genome_optimizer import (
        Plugin as _GenomePlugin)
    _GenomePlugin.validate_repair_rules(
        optimization["mixed_genome_repair_rules"], config)


def check_arm_pairing(en: dict, n: dict) -> None:
    flat_en, flat_n = _flatten(en), _flatten(n)
    diffs = [key for key in sorted(set(flat_en) | set(flat_n))
             if flat_en.get(key) != flat_n.get(key)]
    illegal = [key for key in diffs
               if not any(key.startswith(p)
                          for p in ALLOWED_ARM_DIFF_PREFIXES)]
    if illegal:
        raise SystemExit(f"arms differ outside declared identity: {illegal}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = {}
    for arm in ("en", "n"):
        config = build(arm)
        validate(config, arm)
        configs[arm] = config
        path = OUT_DIR / f"phase_2_eth_{arm}_v2.json"
        text = json.dumps(config, indent=1, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode()).hexdigest()
        print(f"{path.name}: sha256 {digest}")
    check_arm_pairing(configs["en"], configs["n"])
    print("arm pairing verified: differences limited to declared identity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
