#!/usr/bin/env python3
"""Independent, socket-free counterexamples for the 209-220 return packet.

This script does not mutate runtime state. It checks the committed README-link
evidence and proves that the ladder collector can seal and publish a collection
whose four arm records contain no terminal artifact at all.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tools import collect_l1_factorial as factorial  # noqa: E402
from tools import m0_l1_ladder_collect as ladder  # noqa: E402


ARMS = (
    "D0_M0_EXACT",
    "D2_BOUNDARY_ONLY",
    "D3_COST_PROTECTION",
    "D4_FULL_L1",
)
IDENTITY = "musashi220counterexample"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=False
    )


def _record(arm: str) -> dict:
    active = arm == "D0_M0_EXACT"
    validation = {
        "action_raw_mean": 0.2 if active else 0.0,
        "action_raw_std": 0.5 if active else 0.0,
        "action_non_hold_rate": 1.0 if active else 0.0,
        "trades_total": 1 if active else 0,
        "execution_diagnostics": {
            "protected_market_entries": 1 if active else 0
        },
    }
    return {
        "arm": arm,
        "diagnostic_identity": IDENTITY,
        "arm_identity": f"arm-{arm}",
        "contract_sha256": "c" * 64,
        "stop_reason": "fixture",
        "gradient_updates_total": 1,
        "m0_activity_facts": {"activity_survived_normal": active},
        "curriculum": {
            "post_easy": {
                "phase1_handoff_semantics": (
                    "m0_epoch0_eligible_v3"
                    if active
                    else "l1_trained_epoch_v4"
                ),
                "best_easy_epoch": 0 if active else 1,
            }
        },
        "terminal_evaluation_as_run": {
            "splits_raw": {"validation": validation}
        },
        # Deliberately no best_model_* or terminal_model_* fields.
    }


def collector_counterexample(root: Path) -> dict:
    source = root / "source"
    for arm in ARMS:
        arm_dir = source / IDENTITY / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / ladder.RECORD_NAME).write_text(
            json.dumps(_record(arm))
        )
    (source / IDENTITY / "D0_M0_EXACT" / ladder.D1_RECORD_NAME).write_text(
        json.dumps(
            {
                "arm": "D1_EVALUATOR_ONLY",
                "label_under_m0_definition": "active",
                "label_under_l1_definition": "active",
                "labels_agree": True,
            }
        )
    )

    def fetch(_host: str, remote: Path, stage: Path) -> None:
        shutil.copytree(remote, stage)

    def replicate(_host: str, sealed: Path, replica: Path) -> None:
        shutil.copytree(sealed, replica)

    def verify(_host: str, replica: Path, expectations: list[dict]) -> dict:
        return {
            "tree_digest": factorial.tree_digest(replica),
            "verifier_version": "musashi.counterexample.v1",
            "terminals": [],
            "expectation_count": len(expectations),
        }

    collection = root / "collection"
    replica = root / "replica"
    contract = {
        "output_root": str(source),
        "seed": 101,
        "assignments": {
            arm: {"hostname": "fixture-host"} for arm in ARMS
        },
    }
    manifest = ladder.collect(
        contract=contract,
        diagnostic_identity=IDENTITY,
        collection_root=collection,
        fetch_fn=fetch,
        replica_host="fixture-replica",
        replica_root=replica,
        replicate_fn=replicate,
        replica_verify_fn=verify,
    )
    table = ladder.publish_contrast_table(
        collection_root=collection,
        diagnostic_identity=IDENTITY,
        out_json=root / "published.json",
    )
    return {
        "manifest_outcome": manifest.get("outcome"),
        "published_outcome": table.get("outcome"),
        "replica_terminal_expectations": (
            manifest.get("replica", {}).get("proof", {}).get(
                "expectation_count"
            )
        ),
        "replica_terminals_loaded": len(
            manifest.get("replica", {}).get("proof", {}).get(
                "terminals", []
            )
        ),
        "missing_terminal_artifacts_were_accepted": (
            manifest.get("outcome") == "COLLECTION_SEALED"
            and table.get("outcome") == "TABLE_PUBLISHED"
        ),
    }


def readme_checker_counterexample() -> dict:
    evidence = json.loads(
        (
            REPO
            / "docs/audits/evidence/README_LINK_RESOLUTION_CHECK_2026_08_10.json"
        ).read_text()
    )["after"]
    errors = [
        row
        for row in evidence["repositories"]
        if row.get("error")
    ]
    agent_multi = REPO
    links = (
        "pipeline_plugins/_nested_splits.py",
        "pipeline_plugins/_paired_generalization.py",
    )
    missing = []
    for target in links:
        check = _run(
            "git", "cat-file", "-e", f"origin/master:{target}",
            cwd=agent_multi,
        )
        if check.returncode != 0:
            missing.append(target)
    return {
        "recorded_broken_total": evidence["broken_relative_total"],
        "recorded_repository_errors": errors,
        "default_branch_links_missing": missing,
        "false_zero_reproduced": bool(errors or missing)
        and evidence["broken_relative_total"] == 0,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="musashi-209-220-") as tmp:
        result = {
            "schema": "agent_multi.musashi_209_220_repro.v1",
            "collector_counterexample": collector_counterexample(Path(tmp)),
            "readme_checker_counterexample": readme_checker_counterexample(),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
