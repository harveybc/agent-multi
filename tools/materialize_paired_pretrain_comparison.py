#!/usr/bin/env python3
"""WP3 — paired random-vs-pretrained comparison harness, MATERIALIZED
AND NEVER LAUNCHED (post-transfer order 2026-08-27).

Emits the full paired design: identical strong architecture, data
roles, seeds, optimizer, SAC budget, execution envelope, costs,
stopping and evaluation across three arms — random-init control,
pretrained-frozen and pretrained-finetuned (the CPU identifiability
screen was NOT run, so BOTH treatment arms are retained prospectively,
as the order allows only prospective elimination). Four seeds with
counterbalanced arm order, one declared primary endpoint, a registered
trial ledger and predeclared refusals/INCONCLUSIVE.

Runtime/economic results cannot be inferred from pretraining losses —
this tool only binds identities and writes the design; it launches
NOTHING and no GPU authority exists until Musashi dispatches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    sha256_file, sha256_obj)
from agent_plugins.grouped_architecture import (  # noqa: E402
    snapshot_effective_config)

STRONG_CONFIG = ("examples/config/"
                 "project3_ethusdt_4h_sac_grouped_strong_v1.json")
FULL5_CONTRACT = ("examples/config/"
                  "pretrain_contract_eth_h4_o2022_full5_v1.json")
COST_MANIFEST = "examples/config/cost_manifest_eth_h4_v2.json"
SEEDS = (101, 202, 303, 404)
ARMS = ("control_random_init", "pretrained_frozen",
        "pretrained_finetuned")
# counterbalanced: each arm appears in every position across seeds
ARM_ORDER = {101: ("control_random_init", "pretrained_frozen",
                   "pretrained_finetuned"),
             202: ("pretrained_frozen", "pretrained_finetuned",
                   "control_random_init"),
             303: ("pretrained_finetuned", "control_random_init",
                   "pretrained_frozen"),
             404: ("control_random_init", "pretrained_finetuned",
                   "pretrained_frozen")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",
                        default=str(REPO / "docs/audits/evidence/"
                                    "PAIRED_PRETRAIN_COMPARISON_DESIGN_"
                                    "2026_08_27.json"))
    parser.add_argument("--pretrain-dir", required=True,
                        help="SEALED five-objective generation "
                             "(DATA-SOTA-367: prose placeholders "
                             "refuse; every binding is a digest)")
    args = parser.parse_args()

    # DATA-SOTA-367: the paired genesis binds a REAL sealed generation
    import json as _json

    from agent_plugins.branch_pretraining import load_generation
    pretrain_dir = Path(args.pretrain_dir)
    _ckpt, generation_manifest, generation_number = load_generation(
        pretrain_dir)
    if not generation_manifest.get("completed"):
        raise SystemExit("REFUSED: the pretraining generation is not "
                         "completed (DATA-SOTA-367)")
    seal = _json.loads((pretrain_dir / "generation.json").read_text())
    identity = generation_manifest["identity"]
    artifacts = generation_manifest["artifacts"]
    if not artifacts:
        raise SystemExit("REFUSED: no per-family artifacts in the "
                         "sealed generation")
    expected_contract_sha = None
    try:
        expected_contract_sha = sha256_file(REPO / FULL5_CONTRACT)
    except OSError:
        pass
    if identity.get("contract_sha256") != expected_contract_sha:
        raise SystemExit("REFUSED: the sealed generation was not "
                         "trained under the committed five-objective "
                         "contract (digest mismatch; DATA-SOTA-367)")

    snapshot = snapshot_effective_config(REPO / STRONG_CONFIG)
    shared = {
        "strong_config": STRONG_CONFIG,
        "strong_config_sha256": snapshot["config_sha256"],
        "architecture_digest":
            snapshot["materialized"]["architecture_digest"],
        "expected_output_dim":
            snapshot["materialized"]["expected_output_dim"],
        "pretrain_contract": FULL5_CONTRACT,
        "pretrain_contract_sha256": sha256_file(REPO / FULL5_CONTRACT),
        "pretrain_generation": {
            "generation_number": generation_number,
            "seal_manifest_sha256": seal["manifest_sha256"],
            "seal_checkpoint_sha256": seal["checkpoint_sha256"],
            "identity_contract_sha256": identity["contract_sha256"],
            "identity_data_sha256": identity["data_sha256"],
            "preprocessing_config_digest":
                identity["preprocessing_config_digest"],
            "per_family_encoder_digests": {
                family: entry["encoder_sha256"]
                for family, entry in artifacts.items()},
            "eligibility": ("PAIRED_SCREEN_ONLY — pending Musashi's "
                            "acceptance of the C1-C5 return; no other "
                            "use")},
        "data_roles": {"origin": "o2022",
                       "fit_end": "2021-12-31T20:00:00",
                       "scored_year": "2022 (trial ledger only)",
                       "development_outer_2024": "UNTOUCHED",
                       "sealed_2025": "STRUCTURALLY_UNAVAILABLE"},
        "observation_authority": {
            "require_observation_declaration": True,
            "feature_columns_sha256":
                snapshot["materialized"]["feature_columns_sha256"],
            "window_size": 32},
        "execution_envelope": {
            "source": ("ALPACA-frozen o2022 envelope from the B4 "
                       "causal calibration (ATR 3.0/6.0), "
                       "shared_execution_envelope plugin"),
            "cost_manifest": COST_MANIFEST,
            "cost_contract": "alpaca_ethusd ~30.5bp/side (sole "
                             "G1-eligible economy)",
            "headroom": 0.0071},
        "sac": {"learning_rate": 3e-4, "policy": "MultiInputPolicy",
                "budget_total_timesteps": 260000,
                "stopping": "early patience 40 evaluations on the "
                            "in-trial monitor; no monitor-based "
                            "weight/architecture selection",
                "train_seed": "per arm-seed cell"},
        "evaluation": {
            "primary_endpoint": ("risk-adjusted scored-2022 return: "
                                 "mean per-bar log return / std of "
                                 "per-bar log returns (annualized "
                                 "sqrt(2190)), computed ONCE per "
                                 "arm-seed through the trial ledger"),
            "paired_effect": "mean over seeds of (treatment - control)",
            "dispersion": "IQR of the paired differences across seeds",
        },
    }
    predeclared = {
        "minimum_activity": ">= 12 position changes in the scored "
                            "year, else the cell is "
                            "ACTIVITY_REFUSED (excluded, reported)",
        "constant_policy_refusal": "std of raw actions over the "
                                   "scored year < 1e-4 refuses the "
                                   "cell",
        "dead_actor_refusal": "finding-235 family: zero action "
                              "variance across distinct observations "
                              "refuses the cell",
        "inconclusive": ("|paired effect| < 0.5 * std of paired "
                         "differences, OR >= 2 seeds refused in any "
                         "arm -> outcome INCONCLUSIVE; no reseeding, "
                         "no post-hoc endpoint change"),
        "no_selection": "no outer-2024/sealed-2025 access; no "
                        "monitor-driven weight choice; endpoints "
                        "frozen by this design before any dispatch",
    }
    arm_mechanisms = {
        "control_random_init": {"encoder_init": "random (seeded)",
                                "temporal_branches_trainable": True},
        "pretrained_frozen": {
            "encoder_init": "load_pretrained_branches (encoder-only, "
                            "family-digest bound, custody route)",
            "temporal_branches_trainable": False},
        "pretrained_finetuned": {
            "encoder_init": "load_pretrained_branches (encoder-only, "
                            "family-digest bound, custody route)",
            "temporal_branches_trainable": True},
    }
    trials = []
    for seed in SEEDS:
        for position, arm in enumerate(ARM_ORDER[seed]):
            genesis = {"schema": "agent_multi.paired_trial_genesis.v1",
                       "arm": arm, "seed": seed,
                       "execution_position": position,
                       "mechanism": arm_mechanisms[arm],
                       "shared_bindings_sha256": sha256_obj(shared)}
            trials.append({"trial_id": f"{arm}_s{seed}",
                           "genesis": genesis,
                           "genesis_sha256": sha256_obj(genesis)})
    design = {
        "schema": "agent_multi.paired_pretrain_comparison_design.v1",
        "order": "post-transfer objectives order 2026-08-27 WP3",
        "status": "MATERIALIZED_NOT_LAUNCHED — no GPU authority; "
                  "Musashi dispatches the smallest informative screen",
        "identifiability_screen": ("NOT RUN — both treatment arms "
                                   "retained prospectively (the order "
                                   "permits only prospective "
                                   "elimination)"),
        "shared_bindings": shared,
        "predeclared": predeclared,
        "arms": arm_mechanisms,
        "seeds": list(SEEDS),
        "arm_order_counterbalanced": {str(k): list(v)
                                      for k, v in ARM_ORDER.items()},
        "trial_ledger": trials,
        "gpu_estimate": {
            "runs": 12,
            "basis": "P1 lineage: ~7-9 h per 2e4x2e3-step phase on "
                     "one RTX-class GPU; 260k-step budget per cell",
            "estimated_hours_per_cell": "6-10",
            "estimated_total_gpu_hours": "72-120 (sequential on one "
                                         "GPU: 3-5 days; two GPUs: "
                                         "half)"},
        "proposed_gpu_command_NOT_LAUNCHED": (
            "CUDA_VISIBLE_DEVICES=<assigned> PYTHONPATH=. python "
            "tools/dispatch_paired_pretrain_comparison.py "
            "--design docs/audits/evidence/"
            "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json "
            "--pretrain-dir <accepted five-objective generation> "
            "--seed <seed> --arm <arm>   # driver to be implemented "
            "ONLY under Musashi's GPU dispatch"),
    }
    output = Path(args.output)
    output.write_text(json.dumps(design, indent=1))
    print(json.dumps({"trials": len(trials),
                      "architecture_digest":
                          shared["architecture_digest"][:16],
                      "status": design["status"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
