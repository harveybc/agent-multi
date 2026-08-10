#!/usr/bin/env python3
"""Independent reproduction for the L1 round-2 acceptance packet.

This reads the published smoke seal and the deployed, non-secret systemd
configuration. It never starts training and never mutates runtime state.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as system_config  # noqa: E402
from tools import aggregate_l1_factorial as aggregator  # noqa: E402
from tools import collect_l1_factorial as collector  # noqa: E402

COLLECTION = Path(
    "/home/harveybc/.local/share/agent-multi/"
    "l1_smoke_collection_13bfdb1a_v2"
)
EXPERIMENT = "13bfdb1a89fe24ec"
RESTART_ROOT = (
    "/home/harveybc/.local/share/agent-multi/"
    "l1_matched_factorial_20260809_v1/restart_proof_20260809"
)


def ssh(host: str, command: str) -> str:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         host, command],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return proc.stdout.strip()


def main() -> int:
    manifest_path = COLLECTION / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sealed = COLLECTION / "sealed" / EXPERIMENT
    published_digest = manifest["collection_tree_digest"]
    current_digest = collector.tree_digest(sealed)

    system_manifest = system_config.load_system_manifest(
        REPO / "examples/config/phase_3_eth_sac_dynamics/systems/"
        "ethusdt_4h_l1_system_v2.json"
    )
    bindings = system_manifest["costs"]["config_bindings"]
    validate_source = inspect.getsource(
        system_config.validate_normal_contract
    )
    aggregate_main_source = inspect.getsource(aggregator.main)
    launcher_source = (REPO / "tools/l1_fleet_launcher.py").read_text()

    gamma_env_303 = ssh(
        "gamma",
        "cat /home/harveybc/.config/agent-multi/"
        "l1-factorial@303.env",
    )
    gamma_env_404 = ssh(
        "gamma",
        "cat /home/harveybc/.config/agent-multi/"
        "l1-factorial@404.env",
    )
    failed_attempts = ssh(
        "dragon",
        f"find {RESTART_ROOT}/out/39af1c3325406189 -type d "
        "-name 'attempt-*' -printf '%P\\n' | sort",
    ).splitlines()
    successful_attempts = ssh(
        "dragon",
        f"find {RESTART_ROOT}/out/fcdf62cdf65a8577 -type d "
        "-name 'attempt-*' -printf '%P\\n' | sort",
    ).splitlines()
    journal = ssh(
        "dragon",
        f"cat {RESTART_ROOT}/journal_excerpt.txt",
    )

    record_path = next(sealed.rglob("l1_cell_record.json"))
    record = json.loads(record_path.read_text())
    phase1_meta = (record.get("curriculum") or {}).get("post_easy") or {}

    facts = {
        "schema": "agent_multi.audit.l1_round2_acceptance_repro.v1",
        "runtime_mutated": False,
        "counterexamples": {
            "sealed_tree_changed_after_published_digest": {
                "published_digest": published_digest,
                "current_digest": current_digest,
                "replica_digest": manifest["replica"][
                    "replica_tree_digest"
                ],
                "aggregation_inside_seal": (
                    sealed / "aggregation/l1_factorial_aggregation.json"
                ).is_file(),
                "reproduced": current_digest != published_digest,
            },
            "direct_aggregator_bypasses_collection_replica_gate": {
                "main_accepts_collection_manifest": (
                    "collection-manifest" in aggregate_main_source
                ),
                "main_calls_aggregate_directly": (
                    "aggregate(root, args.experiment_id" in
                    aggregate_main_source
                ),
                "reproduced": (
                    "collection-manifest" not in aggregate_main_source
                    and "aggregate(root, args.experiment_id" in
                    aggregate_main_source
                ),
            },
            "assigned_gpu_is_visible_but_not_bound": {
                "gamma_303_env": gamma_env_303,
                "gamma_404_env": gamma_env_404,
                "launcher_requires_cuda_visible_devices": (
                    "os.environ[\"CUDA_VISIBLE_DEVICES\"]" in
                    launcher_source
                ),
                "reproduced": (
                    "CUDA_VISIBLE_DEVICES" not in gamma_env_303
                    or "CUDA_VISIBLE_DEVICES" not in gamma_env_404
                ),
            },
            "restart_proof_did_not_create_new_failed_attempts": {
                "restart_counter_21_observed": (
                    "restart counter is at 21" in journal
                ),
                "failed_identity_attempt_directories": failed_attempts,
                "successful_identity_attempt_directories":
                    successful_attempts,
                "reproduced": (
                    "restart counter is at 21" in journal
                    and len(failed_attempts) == 1
                    and "attempt-partialdead-01" in failed_attempts[0]
                ),
            },
            "financing_treatment_is_not_explicit": {
                "financing_enabled_binding": bindings.get(
                    "financing_enabled"
                ),
                "validator_checks_financing": (
                    "financing" in validate_source
                ),
                "reproduced": (
                    "financing_enabled" not in bindings
                    and "financing" not in validate_source
                ),
            },
            "phase1_realized_epochs_counts_baseline": {
                "record": str(record_path),
                "phase1_requested_epochs": record.get(
                    "phase1_requested_epochs"
                ),
                "phase1_realized_epochs": record.get(
                    "phase1_realized_epochs"
                ),
                "easy_budget_epochs": phase1_meta.get(
                    "easy_budget_epochs"
                ),
                "easy_epochs_run": phase1_meta.get("easy_epochs_run"),
                "phase1_gradient_updates": phase1_meta.get(
                    "phase1_gradient_updates"
                ),
                "history_epochs": [
                    row.get("epoch")
                    for row in phase1_meta.get("history", [])
                ],
                "reproduced": (
                    record.get("phase1_requested_epochs") == 1
                    and record.get("phase1_realized_epochs") == 2
                    and [row.get("epoch") for row in
                         phase1_meta.get("history", [])] == [0, 1]
                ),
            },
        },
    }
    print(json.dumps(facts, indent=2, sort_keys=True))
    return 1 if any(
        item["reproduced"]
        for item in facts["counterexamples"].values()
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
