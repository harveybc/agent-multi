#!/usr/bin/env python3
"""Independent reproducer for the post-outage WP2/WP4 audit.

This script never opens a socket, trains a model, modifies a sealed collection,
or imports a brokerage adapter. It proves the execution-contract mismatch in
the materialized P1 factorial and the unenforced replica gate in the screen
verdict using temporary files only.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _nested_splits
from tools import p1_difficulty_lr_factorial as factorial


def _fake_records(contract: dict, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    experiment_identity = "audit-fixture-one-coherent-experiment"
    for seed in factorial.SEEDS:
        for cell in contract["cell_order"][str(seed)]:
            terminal = root / f"seed{seed}-{cell}.zip"
            terminal.write_bytes(f"{seed}:{cell}".encode("ascii"))
            records[(seed, cell)] = {
                "schema": factorial.RECORD_SCHEMA,
                "experiment_identity": experiment_identity,
                "cell_identity": f"audit-{seed}-{cell}",
                "contract_sha256": contract["_contract_sha256"],
                "seed": seed,
                "cell": cell,
                "terminal_model_path": str(terminal),
                "terminal_model_sha256": factorial._sha_file(terminal),
                "handoff_viability": {
                    "selected": {
                        "handoff_viability": "VIABLE",
                        "trained_treatment": True,
                    }
                },
            }
    return records


def reproduce() -> dict:
    contract = factorial.load_contract()
    bindings = factorial.load_bindings()
    with tempfile.TemporaryDirectory(prefix="musashi-224-230-") as temp:
        root = Path(temp)
        materialized = {}
        for seed in factorial.SEEDS:
            for cell in contract["cell_order"][str(seed)]:
                config = factorial.materialize_cell_config(
                    contract, bindings, seed, cell,
                    root / "materialized" / str(seed) / cell,
                )
                materialized[f"{seed}:{cell}"] = {
                    "nested_split_contract": config.get(
                        "nested_split_contract"),
                    "selection_metric": config.get("selection_metric"),
                    "train_start": config.get("train_start"),
                    "train_end": config.get("train_end"),
                    "validation_start": config.get("validation_start"),
                    "validation_end": config.get("validation_end"),
                    "evaluate_test_split": config.get(
                        "evaluate_test_split"),
                }

        nested_contract = _nested_splits.load_contract(
            Path("examples/config/phase_3_eth_sac_dynamics/splits/"
                 "eth_nested_split_contract_v1.json"))
        nested_manifest = _nested_splits.materialize_nested_splits(
            nested_contract, root / "nested", mode="l1")
        role_facts = {
            role: {
                "status": facts.get("status"),
                "scored_rows": facts.get("scored_rows"),
                "context_rows": facts.get("context_rows"),
                "csv_sha256": facts.get("csv_sha256"),
            }
            for role, facts in nested_manifest["roles"].items()
        }

        fake_records = _fake_records(contract, root / "records")
        verdict, exit_code = factorial.screen_verdict(
            contract, records=fake_records)

    first = next(iter(materialized.values()))
    return {
        "schema": "agent_multi.musashi_post_outage_224_230_repro.v1",
        "network_used": False,
        "training_used": False,
        "sealed_collection_mutated": False,
        "factorial_contract_sha256": contract["_contract_sha256"],
        "factorial_materialized_cells": materialized,
        "materialized_execution_mismatch": {
            "nested_split_contract_missing": (
                first["nested_split_contract"] is None),
            "selection_metric_observed": first["selection_metric"],
            "selection_metric_required":
                "paired_generalization_weekly_v1",
            "legacy_validation_role_observed": {
                "start": first["validation_start"],
                "end": first["validation_end"],
            },
            "nested_inner_validation_required": {
                "start": nested_contract["roles"]["inner_validation"][
                    "start"],
                "end": nested_contract["roles"]["inner_validation"][
                    "end"],
            },
        },
        "nested_role_facts": role_facts,
        "replica_gate_counterexample": {
            "input_replica_proofs": 0,
            "screen_outcome": verdict["outcome"],
            "screen_exit_code": exit_code,
            "replica_gate_value": verdict["gates"][
                "replica_terminal_loads"],
            "screen_declared_eligible_without_replica_proof": (
                verdict["outcome"] == "SCREEN_VIABLE_REGION"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = reproduce()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "nested_split_contract_missing": payload[
            "materialized_execution_mismatch"][
                "nested_split_contract_missing"],
        "screen_declared_eligible_without_replica_proof": payload[
            "replica_gate_counterexample"][
                "screen_declared_eligible_without_replica_proof"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
