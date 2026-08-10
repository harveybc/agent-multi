#!/usr/bin/env python3
"""L1 matched factorial runner v2 — one seed, the four v3 cells.

Correction order 2026-08-09 (findings 178-187): every cell config is
materialized exclusively THROUGH the frozen system manifest
(`materialize_system_config`), never the legacy ETH `_base_config`.
The execution identity binds contract + system manifest + nested split
+ the ACTUAL executing source trees (full commit plus dirty/untracked
digest, resolved from this file's own checkout — never a hard-coded
sibling). Records are published atomically (fsync + replace) into
content-addressed attempt directories; a complete hash-valid record is
reused as ALREADY_COMPLETE, a partial directory is recovered into a new
attempt, and a moved source tree or bound dependency fails the cell
closed. Smoke runs carry evidence_class=mechanics_smoke and can never
enter aggregation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid  # noqa: E402

CONTRACT_PATH = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                 "l1_factorial_contract_v3.json")
SYSTEM_MANIFEST_PATH = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                        "systems/ethusdt_4h_l1_system_v2.json")
SCHEMA = "agent_multi.l1_factorial_cell_record.v2"
GYM_FX_ROOT = Path("/home/harveybc/Documents/GitHub/gym-fx")


def _sha_file(path: Path) -> str:
    return sysid.sha_file(path)


def _agent_plugin(name: str):
    from importlib.metadata import entry_points

    ep = next(e for e in entry_points().select(group="agent.plugins")
              if e.name == name)
    return ep.load()()


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    contract = json.loads(path.read_text())
    if contract.get("schema") != "agent_multi.l1_factorial_contract.v3":
        raise ValueError("unknown l1 factorial contract schema")
    contract["_contract_sha256"] = _sha_file(path)
    return contract


def load_system_manifest(path: Path = SYSTEM_MANIFEST_PATH) -> dict:
    return sysid.load_system_manifest(path)


def source_identities() -> dict:
    """Actual executing trees: this script's own checkout + gym-fx."""
    return {
        "agent-multi": sysid.source_tree_identity(
            sysid.resolve_repo_root(Path(__file__))),
        "gym-fx": sysid.source_tree_identity(GYM_FX_ROOT),
    }


def experiment_identity(contract: dict, manifest: dict, smoke: bool,
                        sources: dict | None = None) -> str:
    sources = sources or source_identities()
    payload = {
        "contract": contract["_contract_sha256"],
        "system_manifest": manifest["_manifest_sha256"],
        "nested_split_contract": _sha_file(
            REPO / contract["nested_split_contract"]),
        "code": {name: {"commit": s["commit"],
                        "dirty_untracked_digest":
                            s["dirty_untracked_digest"]}
                 for name, s in sorted(sources.items())},
        "profile": "smoke" if smoke else "decision",
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True).encode()).hexdigest()[:16]


def cell_identity(exp_id: str, seed: int, cell: str,
                  contract: dict) -> str:
    spec = contract["cells"][cell]
    payload = {
        "experiment_id": exp_id,
        "seed": seed,
        "cell": cell,
        "factors": {"phase1_mode": spec["phase1_mode"],
                    "phase2_lr_multiplier": spec["phase2_lr_multiplier"]},
        "anchor_sha256": contract["anchors"][str(seed)]["sha256"],
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True).encode()).hexdigest()[:16]


def atomic_write_json(path: Path, payload: dict) -> None:
    """fsync + replace: the record appears only complete, never partial."""
    text = json.dumps(payload, indent=1, sort_keys=True,
                      default=str) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def record_is_complete(record_path: Path, expected_cell_id: str) -> bool:
    """A reusable record hash-validates and its artifacts exist."""
    try:
        record = json.loads(record_path.read_text())
    except Exception:
        return False
    if record.get("schema") != SCHEMA:
        return False
    if record.get("cell_identity") != expected_cell_id:
        return False
    for key in ("terminal_model_path", "terminal_model_sha256"):
        if not record.get(key):
            return False
    terminal = Path(record["terminal_model_path"])
    if not terminal.is_file():
        return False
    if _sha_file(terminal) != record["terminal_model_sha256"]:
        return False
    return True


def _next_attempt_dir(cell_dir: Path, cell_id: str) -> Path:
    """Content-addressed attempt directory; existing partials are never
    overwritten — each recovery lands in a fresh attempt."""
    n = 0
    while True:
        n += 1
        attempt = cell_dir / f"attempt-{cell_id}-{n:02d}"
        if not attempt.exists():
            attempt.mkdir(parents=True)
            return attempt


def run_cell(cell: str, seed: int, *, contract: dict, manifest: dict,
             smoke: bool, agent_name: str | None = None) -> dict:
    from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
        PipelinePlugin as CurriculumPipeline)

    # Finding 191: the manifest names must equal the classes that
    # execute. The agent comes FROM the manifest; the intentionally
    # varying curriculum wrapper is bound explicitly and asserted.
    plugins = manifest.get("plugins") or {}
    if agent_name is None:
        agent_name = plugins.get("agent_plugin")
    if not agent_name or agent_name != plugins.get("agent_plugin"):
        raise RuntimeError(
            f"agent plugin {agent_name!r} does not equal the manifest "
            f"binding {plugins.get('agent_plugin')!r}")
    if plugins.get("curriculum_pipeline_plugin") != \
            "rl_pipeline_with_solvency_curriculum":
        raise RuntimeError(
            "manifest curriculum_pipeline_plugin does not name the "
            "executing curriculum wrapper "
            "rl_pipeline_with_solvency_curriculum")

    spec = contract["cells"][cell]
    budget = contract["smoke_budget" if smoke else "decision_budget"]
    stopping = contract["stopping"]
    anchor_entry = contract["anchors"][str(seed)]
    anchor = Path(anchor_entry["path"]).expanduser()
    actual = _sha_file(anchor)
    if actual != anchor_entry["sha256"]:
        raise RuntimeError(f"anchor hash mismatch for seed {seed}")

    sources_before = source_identities()
    exp_id = experiment_identity(contract, manifest, smoke,
                                 sources=sources_before)
    cell_id = cell_identity(exp_id, seed, cell, contract)
    out_root = Path(contract["output_root"]).expanduser()
    cell_dir = out_root / exp_id / f"seed{seed}" / cell
    cell_dir.mkdir(parents=True, exist_ok=True)

    record_path = cell_dir / "l1_cell_record.json"
    if record_path.exists():
        if record_is_complete(record_path, cell_id):
            record = json.loads(record_path.read_text())
            record["_reuse"] = "ALREADY_COMPLETE"
            return record
        raise RuntimeError(
            f"cell {cell} seed {seed}: existing record fails validation; "
            "refusing to overwrite — recover it explicitly")

    out_dir = _next_attempt_dir(cell_dir, cell_id)

    config = sysid.materialize_system_config(
        contract, manifest, cell, seed, out_dir)
    identity = config.pop("_identity")
    epoch_timesteps = int(budget.get("epoch_timesteps", 20000))
    config.update({
        "epoch_timesteps": epoch_timesteps,
        "nested_split_contract": str(
            REPO / contract["nested_split_contract"]),
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
        "l1_activity_patience": int(
            stopping.get("l1_activity_patience", 40)),
        "l1_activity_patience_start_epoch": int(
            stopping.get("l1_activity_patience_start_epoch", 40)),
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
        "learning_starts": int(budget.get("learning_starts", 1000)),
        "execution_cost_curriculum_epochs": max(
            2, int(budget["phase2_max_epochs"])),
        "inactive_terminal_is_typed_result": True,
        "selection_min_trades": 0,
    })
    resolved_sha = sysid.resolved_config_sha256(config)

    agent = _agent_plugin(agent_name)
    started = datetime.now(timezone.utc)
    pipeline = CurriculumPipeline(config)
    result = pipeline.run_pipeline(config=config, env_plugin=None,
                                   agent_plugin=agent, mode="train")
    finished = datetime.now(timezone.utc)
    sources_after = source_identities()
    for name in sources_before:
        sysid.assert_source_identity_unmoved(
            sources_before[name], sources_after[name])

    terminal_path = result.get("terminal_model_path")
    if not terminal_path or not Path(terminal_path).is_file():
        raise RuntimeError(
            "cell finished without a terminal artifact — invalid, "
            "record refused")
    terminal_sha = _sha_file(Path(terminal_path))
    terminal_tensor_sha = _terminal_tensor_sha(terminal_path)

    history = result.get("history") or []
    record = {
        "schema": SCHEMA,
        "evidence_class": ("mechanics_smoke" if smoke else "decision_run"),
        "decision_eligible": not smoke,
        "performance_aggregate_eligible": not smoke,
        "experiment_id": exp_id,
        "cell_identity": cell_id,
        "attempt_dir": str(out_dir),
        "cell": cell,
        "seed": seed,
        "contract_sha256": contract["_contract_sha256"],
        "system_manifest_sha256": identity["system_manifest_sha256"],
        "nested_split_contract_sha256": identity[
            "nested_split_contract_sha256"],
        "data_sha256": identity["data_sha256"],
        "data_rows": identity["data_rows"],
        "data_time_bounds": identity["data_time_bounds"],
        "resolved_config_sha256": resolved_sha,
        "observation_manifest_sha256": identity["observation"][
            "observation_manifest_sha256"],
        "observation_flattened_shape": identity["observation"][
            "flattened_shape"],
        "asset": identity["asset_label"],
        "env_asset": identity["env_asset"],
        "metric_schema": identity["metric_schema"],
        "initial_cash": identity["initial_cash"],
        "cost_contract": identity["cost_contract"],
        "phase1_mode": spec["phase1_mode"],
        "phase2_lr_multiplier": spec["phase2_lr_multiplier"],
        "anchor_sha256": actual,
        "anchor_policy_tensor_sha256": identity[
            "anchor_policy_tensor_sha256"],
        "phase1_requested_epochs": int(budget["phase1_epochs"]),
        "phase2_requested_epochs": int(budget["phase2_max_epochs"]),
        "phase1_realized_epochs": (
            (result.get("curriculum") or {}).get("post_easy", {})
            .get("easy_epochs_run")),
        "phase2_realized_epochs": len(history),
        "stop_reason": result.get("stop_reason"),
        "termination_cause": result.get("termination_cause"),
        "activity_stopped_without_eligible_checkpoint": bool(
            result.get("activity_stopped_without_eligible_checkpoint",
                       False)),
        "history_len": len(history),
        "subject_code_identity": sources_before,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "nested_split_manifest": config.get("nested_split_manifest"),
        "curriculum": result.get("curriculum"),
        "best_model_path": result.get("best_model_path"),
        "terminal_model_path": str(Path(terminal_path).resolve()),
        "terminal_model_sha256": terminal_sha,
        "terminal_policy_tensor_sha256": terminal_tensor_sha,
        "boundary_transfer_evidence": result.get(
            "warm_start_transfer_evidence"),
    }
    atomic_write_json(record_path, record)
    return record


def _terminal_tensor_sha(terminal_path: str) -> str:
    from stable_baselines3 import SAC

    from agent_plugins.sac_agent import _policy_tensor_hash

    model = SAC.load(str(terminal_path), device="cpu")
    try:
        return _policy_tensor_hash(model.policy)
    finally:
        del model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    contract = load_contract()
    manifest = load_system_manifest()
    cells = args.cells or list(contract["cells"])
    for cell in cells:
        record = run_cell(cell, args.seed, contract=contract,
                          manifest=manifest, smoke=args.smoke)
        print(json.dumps({
            "cell": cell, "seed": args.seed,
            "experiment_id": record["experiment_id"],
            "cell_identity": record.get("cell_identity"),
            "reuse": record.get("_reuse"),
            "evidence_class": record["evidence_class"],
            "stop_reason": record.get("stop_reason"),
            "history_len": record.get("history_len"),
            "terminal": bool(record.get("terminal_model_path")),
        }, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
