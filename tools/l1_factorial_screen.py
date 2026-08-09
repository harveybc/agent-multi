#!/usr/bin/env python3
"""L1 matched factorial runner — one seed, the four v3 cells.

Doc 38 §5.3 / continuation order. Every cell traverses the SAME
two-phase path (mode-aware phase 1 in the solvency pipeline, matched
boundary via load_for_training, paired stopping, nested chronology);
only phase-1 dynamics and the phase-2 LR multiplier differ. Smoke runs
carry evidence_class=mechanics_smoke and can never enter aggregation.

One experiment identity binds contract sha + nested split contract sha
+ code revisions + seed + cell + anchor; records land under
<output_root>/<experiment_id>/seed<seed>/<cell>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import eth_curriculum_decision_experiment as d1  # noqa: E402

CONTRACT_PATH = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                 "l1_factorial_contract_v3.json")
SCHEMA = "agent_multi.l1_factorial_cell_record.v1"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    contract = json.loads(path.read_text())
    if contract.get("schema") != "agent_multi.l1_factorial_contract.v3":
        raise ValueError("unknown l1 factorial contract schema")
    contract["_contract_sha256"] = _sha_file(path)
    return contract


def experiment_identity(contract: dict, smoke: bool) -> str:
    payload = {
        "contract": contract["_contract_sha256"],
        "nested_split_contract": _sha_file(
            REPO / contract["nested_split_contract"]),
        "code": {r: d1._git_rev(r) for r in ("agent-multi", "gym-fx")},
        "profile": "smoke" if smoke else "decision",
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True).encode()).hexdigest()[:16]


def run_cell(cell: str, seed: int, *, contract: dict, smoke: bool,
             agent_name: str = "sac_agent") -> dict:
    from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
        PipelinePlugin as CurriculumPipeline)

    spec = contract["cells"][cell]
    budget = contract["smoke_budget" if smoke else "decision_budget"]
    stopping = contract["stopping"]
    anchor_entry = contract["anchors"][str(seed)]
    anchor = Path(anchor_entry["path"]).expanduser()
    actual = d1._sha(anchor)
    if actual != anchor_entry["sha256"]:
        raise RuntimeError(f"anchor hash mismatch for seed {seed}")

    exp_id = experiment_identity(contract, smoke)
    out_root = Path(contract["output_root"]).expanduser()
    out_dir = out_root / exp_id / f"seed{seed}" / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    config = d1._base_config(out_dir, cell, seed,
                             epoch_timesteps=int(
                                 budget.get("epoch_timesteps", 20000)))
    config.update({
        "nested_split_contract": str(REPO / contract["nested_split_contract"]),
        "nested_split_dir": str(out_dir / "nested_splits"),
        "selection_metric": contract["selection_metric"],
        "l1_gap_penalty_beta": contract["l1_gap_penalty_beta"],
        "phase1_mode": spec["phase1_mode"],
        "easy_max_epochs": int(budget["phase1_epochs"]),
        "easy_patience": 10_000,
        "max_epochs": int(budget["phase2_max_epochs"]),
        "l1_patience": int(stopping["l1_patience"]),
        "l1_patience_start_epoch": int(
            stopping["l1_patience_start_epoch"]),
        "total_max_passes": int(stopping["total_max_passes"]),
        "phase1_max_fraction": float(stopping["phase1_max_fraction"]),
        "normal_phase_min_passes": (
            1 if smoke else int(stopping["normal_phase_min_passes"])),
        "learning_rate": (float(contract["baseline_learning_rate"])
                          * float(spec["phase2_lr_multiplier"])),
        "phase1_learning_rate": float(contract["baseline_learning_rate"]),
        "easy_learning_rate": float(contract["baseline_learning_rate"]),
        "warm_start_model": str(anchor),
        "warm_start_model_sha256": actual,
        "evaluate_test_split": False,
        "learning_starts": int(budget.get(
            "learning_starts", 1000)),
        "execution_cost_curriculum_epochs": max(
            2, int(budget["phase2_max_epochs"])),
    })
    agent = d1._agent_plugin(agent_name)
    code_before = {r: d1._git_rev(r) for r in ("agent-multi", "gym-fx")}
    started = datetime.now(timezone.utc)
    pipeline = CurriculumPipeline(config)
    result = pipeline.run_pipeline(config=config, env_plugin=None,
                                   agent_plugin=agent, mode="train")
    finished = datetime.now(timezone.utc)
    code_after = {r: d1._git_rev(r) for r in ("agent-multi", "gym-fx")}
    if code_before != code_after:
        raise RuntimeError("code revisions moved during the cell")

    record = {
        "schema": SCHEMA,
        "evidence_class": ("mechanics_smoke" if smoke else "decision_run"),
        "decision_eligible": not smoke,
        "performance_aggregate_eligible": not smoke,
        "experiment_id": exp_id,
        "cell": cell,
        "seed": seed,
        "contract_sha256": contract["_contract_sha256"],
        "phase1_mode": spec["phase1_mode"],
        "phase2_lr_multiplier": spec["phase2_lr_multiplier"],
        "anchor_sha256": actual,
        "code_revisions": code_before,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "nested_split_manifest": config.get("nested_split_manifest"),
        "curriculum": result.get("curriculum"),
        "best_model_path": result.get("best_model_path"),
        "terminal_model_path": result.get("terminal_model_path"),
        "history_len": len(result.get("history") or []),
        "boundary_transfer_evidence": result.get(
            "warm_start_transfer_evidence"),
    }
    (out_dir / "l1_cell_record.json").write_text(
        json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    contract = load_contract()
    cells = args.cells or list(contract["cells"])
    results = {}
    for cell in cells:
        record = run_cell(cell, args.seed, contract=contract,
                          smoke=args.smoke)
        results[cell] = {
            "evidence_class": record["evidence_class"],
            "history_len": record["history_len"],
            "terminal": bool(record["terminal_model_path"]),
        }
        print(json.dumps({"cell": cell, "seed": args.seed,
                          "experiment_id": record["experiment_id"],
                          **results[cell]}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
