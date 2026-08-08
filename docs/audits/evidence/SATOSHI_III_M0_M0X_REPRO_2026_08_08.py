#!/usr/bin/env python3
"""Independent, read-only reproduction for the M0/M1/M0-X audit.

The script performs no training, broker call, network request, or mutation of
campaign state. A temporary SAC archive is written only to demonstrate that a
ZIP digest is not a policy-weight digest.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
import tempfile
import warnings
from pathlib import Path

import torch
from stable_baselines3 import SAC

warnings.filterwarnings(
    "ignore",
    message="This system does not have apparently enough memory to store.*",
)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools import aggregate_eth_sac_inner_curriculum as aggregate  # noqa: E402
from tools import eth_curriculum_decision_experiment as d1  # noqa: E402
from tools import eth_sac_inner_curriculum_screen as screen  # noqa: E402


RESULT_ROOT = (
    Path.home()
    / ".local/share/agent-multi/eth_sac_inner_curriculum_m0_20260807_v1"
)
CFG = REPO / "examples/config/phase_3_eth_sac_dynamics"
EASY_ARMS = ("E1_N1_LR1", "E1_N1_LR03", "E1_N1_LR01")
SEEDS = (101, 202, 303, 404)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_distance(left: Path, right: Path) -> dict:
    left_state = SAC.load(str(left), device="cpu").policy.state_dict()
    right_state = SAC.load(str(right), device="cpu").policy.state_dict()
    if left_state.keys() != right_state.keys():
        return {"same_keys": False, "changed_tensors": None}
    changed = 0
    max_abs = 0.0
    sum_abs = 0.0
    for key in left_state:
        delta = (left_state[key].detach().cpu()
                 - right_state[key].detach().cpu()).abs()
        if not torch.equal(left_state[key], right_state[key]):
            changed += 1
        if delta.numel():
            max_abs = max(max_abs, float(delta.max()))
            sum_abs += float(delta.sum())
    return {
        "same_keys": True,
        "tensor_count": len(left_state),
        "changed_tensors": changed,
        "max_abs_delta": max_abs,
        "sum_abs_delta": sum_abs,
        "weights_identical": changed == 0,
    }


def easy_handoff_evidence(contract: dict) -> list[dict]:
    rows = []
    for seed in SEEDS:
        anchor = Path(contract["anchors"][str(seed)]["path"]).expanduser()
        for arm in EASY_ARMS:
            arm_dir = RESULT_ROOT / f"seed{seed}" / arm
            meta = json.loads(
                (arm_dir / "model.post_easy.zip.meta.json").read_text())
            record = json.loads((arm_dir / "m0_arm_record.json").read_text())
            post_easy = Path(meta["artifact"])
            trained_rows = [
                item for item in meta.get("history", [])
                if item.get("checkpoint_source") == "easy_training_epoch"
            ]
            rows.append({
                "seed": seed,
                "arm": arm,
                "best_easy_epoch": meta.get("best_easy_epoch"),
                "easy_budget_epochs": meta.get("easy_budget_epochs"),
                "trained_easy_activity_eligible": [
                    item.get("easy_activity_eligible") for item in trained_rows
                ],
                "trained_normal_handoff_eligible": [
                    item.get("normal_handoff_eligible") for item in trained_rows
                ],
                "recorded_post_easy_activity": (
                    record.get("decision_facts", {}).get("post_easy_activity")
                ),
                "post_easy_vs_anchor": policy_distance(post_easy, anchor),
            })
    return rows


def terminal_weight_evidence(contract: dict) -> list[dict]:
    rows = []
    for seed in SEEDS:
        anchor = Path(contract["anchors"][str(seed)]["path"]).expanduser()
        for arm in aggregate.ARMS:
            record = json.loads(
                (RESULT_ROOT / f"seed{seed}" / arm
                 / "m0_arm_record.json").read_text())
            terminal = Path(record["terminal_evaluation"]["artifact_path"])
            rows.append({
                "seed": seed,
                "arm": arm,
                "terminal_vs_anchor": policy_distance(terminal, anchor),
            })
    return rows


def archive_sha_counterexample(anchor: Path) -> dict:
    model = SAC.load(str(anchor), device="cpu")
    with tempfile.TemporaryDirectory(prefix="m0-audit-") as temp_dir:
        resaved = Path(temp_dir) / "unchanged-policy.zip"
        model.save(str(resaved))
        return {
            "original_sha256": sha256(anchor),
            "resaved_sha256": sha256(resaved),
            "archive_sha_changed": sha256(anchor) != sha256(resaved),
            "policy_distance": policy_distance(anchor, resaved),
        }


def contract_evidence() -> dict:
    m1_m03_path = CFG / "m1_factorial_contract_M03.json"
    m1_m01_path = CFG / "m1_factorial_contract_M01.json"
    m0x_path = CFG / "m0x_usdcad_contract_M03.json"
    m1_m03 = json.loads(m1_m03_path.read_text())
    m1_m01 = json.loads(m1_m01_path.read_text())
    m0x = json.loads(m0x_path.read_text())

    malformed = copy.deepcopy(m1_m03)
    malformed["arms"] = {"N14_M10": malformed["arms"]["N14_M10"]}
    malformed_accepted = True
    try:
        screen.validate_contract_v2(malformed)
    except (KeyError, TypeError, ValueError):
        malformed_accepted = False

    mismatched = copy.deepcopy(m1_m03)
    mismatched["winner_multiplier"] = 0.1
    mismatched_accepted = True
    try:
        screen.validate_contract_v2(mismatched)
    except (KeyError, TypeError, ValueError):
        mismatched_accepted = False

    common_anchor = Path(m1_m03["anchors"]["101"]["path"]).expanduser()
    shared_execution_id = d1._execution_id(
        "M0_N14_M10", 101, common_anchor,
        epoch_timesteps=m1_m03["epoch_timesteps"])

    return {
        "m0x_declared_asset": m0x["asset"],
        "runner_base_data_file": d1.DATA_FILE,
        "runner_base_contract": str(d1.ETH_BASE),
        "runner_base_asset_is_eth": "eth" in d1.DATA_FILE.lower(),
        "run_m0_arm_calls_eth_base_config": (
            "d1._base_config" in inspect.getsource(screen.run_m0_arm)
        ),
        "m1_variant_output_roots_equal": (
            m1_m03["output_root"] == m1_m01["output_root"]
        ),
        "execution_identity_has_no_contract_argument": (
            "contract" not in inspect.signature(d1._execution_id).parameters
        ),
        "common_arm_execution_id_both_variants": shared_execution_id,
        "malformed_one_cell_factorial_accepted": malformed_accepted,
        "winner_multiplier_arm_mismatch_accepted": mismatched_accepted,
        "aggregator_contract_path": str(aggregate.CONTRACT_PATH),
        "aggregator_arms": list(aggregate.ARMS),
        "aggregator_is_m0_v1_only": (
            aggregate.CONTRACT_PATH.name == "m0_contract.json"
            and set(aggregate.ARMS) == set(screen.M0_ARM_ORDER)
        ),
    }


def main() -> int:
    contract = screen.load_contract()
    handoffs = easy_handoff_evidence(contract)
    terminals = terminal_weight_evidence(contract)
    anchor = Path(contract["anchors"]["101"]["path"]).expanduser()
    sha_counterexample = archive_sha_counterexample(anchor)
    contracts = contract_evidence()

    reproduced = {
        "all_12_easy_handoffs_are_epoch_zero": all(
            row["best_easy_epoch"] == 0 for row in handoffs),
        "all_12_easy_handoffs_equal_anchor_weights": all(
            row["post_easy_vs_anchor"]["weights_identical"]
            for row in handoffs),
        "all_12_records_omit_post_easy_activity": all(
            row["recorded_post_easy_activity"] is None for row in handoffs),
        "all_16_terminal_policies_really_changed": all(
            row["terminal_vs_anchor"]["changed_tensors"] > 0
            for row in terminals),
        "archive_sha_can_change_without_policy_change": (
            sha_counterexample["archive_sha_changed"]
            and sha_counterexample["policy_distance"]["weights_identical"]
        ),
        "m0x_runner_is_still_eth_bound": (
            contracts["m0x_declared_asset"] == "USDCAD"
            and contracts["runner_base_asset_is_eth"]
            and contracts["run_m0_arm_calls_eth_base_config"]
        ),
        "v2_factorial_shape_is_not_enforced": (
            contracts["malformed_one_cell_factorial_accepted"]),
        "v2_winner_binding_is_not_enforced": (
            contracts["winner_multiplier_arm_mismatch_accepted"]),
        "v2_aggregator_is_absent": contracts["aggregator_is_m0_v1_only"],
    }
    packet = {
        "schema": "agent_multi.audit.m0_m0x_reproduction.v1",
        "audited_head": "99bb7fff9c78999fee6ed9b5d5060a7860d61dae",
        "network_used": False,
        "training_started": False,
        "runtime_mutated": False,
        "reproduced": reproduced,
        "easy_handoffs": handoffs,
        "terminal_weights": terminals,
        "archive_sha_counterexample": sha_counterexample,
        "contract_evidence": contracts,
    }
    print(json.dumps(packet, indent=1, sort_keys=True))
    return 0 if all(reproduced.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
