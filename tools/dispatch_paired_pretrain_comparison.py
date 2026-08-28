#!/usr/bin/env python3
"""Paired SAC comparison dispatch driver (C3, SAC driver correction
order 2026-08-28). The REAL execution path uses the accepted nested
trainer — pipeline_plugins/rl_pipeline_with_validation.PipelinePlugin
with the sac_agent grouped strong route and gym_fx_env — never a
second trainer.

Modes:
* default (no execution flag): verifies the design digest, the
  candidate generation seal + quarantine status + per-family encoder
  digests, the strong-config snapshot identity, and materializes the
  per-cell SAC genesis config for (--seed, --arm) WITHOUT constructing
  any model or env; prints the cell identity packet. NOT_LAUNCHED.
* ``--execute-cpu-dry-run``: runs the cell end to end on CPU under a
  BOUNDED disclosed budget (C4 acceptance evidence). Refuses if CUDA
  is visible — the dry run must be structurally unable to touch a GPU.
* ``--execute``: the full-budget run. REFUSED unless
  ``--gpu-authorized-by-musashi <dispatch-doc>`` names an existing
  written dispatch AND CUDA is visible. Musashi's acceptance of the C5
  packet is that document (order 2026-08-28 C5).

Every execution attempt is NON-RESUMABLE: each invocation mints a new
attempt identity (custody dispatch key) and cold-starts; no resume
path exists in this driver (order C3 — resumption would require a
proven full replay/optimizer/RNG restore, which no artifact provides).
No venue socket: the cell config is refused if any live-credential key
is present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    load_generation, sha256_file, sha256_obj)
from agent_plugins.dispatch_custody import (  # noqa: E402
    DispatchLedger, dispatch_key)
from agent_plugins.grouped_architecture import (  # noqa: E402
    snapshot_effective_config)

DESIGN_PATH = (REPO / "docs/audits/evidence/"
               "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json")
REGISTER_PATH = (REPO / "docs/audits/evidence/"
                 "GENERATION_QUARANTINE_REGISTER.json")
NESTED_CONTRACT = ("examples/config/phase_3_eth_sac_dynamics/splits/"
                   "eth_nested_split_contract_o2022_paired_v1.json")
ENVELOPE_CALIBRATION = ("docs/audits/evidence/"
                        "screen_b_rule_arms_v3_20260826/"
                        "ENVELOPE_CALIBRATION_o2022.json")
COST_MANIFEST = ("examples/config/phase_3_eth_sac_dynamics/"
                 "cost_manifest_eth_h4_v2.json")
# assessment cadence: 100 assessments cover the 260k budget; patience
# 40 (design binding) can therefore actually fire
EPOCH_TIMESTEPS = 2600
MAX_EPOCHS = 100
L1_PATIENCE = 40
# venue-credential key fragments whose presence refuses the cell —
# this driver must be structurally unable to open a venue socket
FORBIDDEN_KEY_FRAGMENTS = ("api_key", "api_secret", "mt5_login",
                           "mt5_password", "mt5_server", "account_id",
                           "username", "password", "remote_log")


class DispatchRefused(RuntimeError):
    pass


def verify_cell(design: dict, pretrain_dir: Path, seed: int,
                arm: str) -> dict:
    if arm not in design["arms"]:
        raise DispatchRefused(f"unknown arm {arm!r}")
    if seed not in design["seeds"]:
        raise DispatchRefused(f"seed {seed} not in the design")
    _ckpt, manifest, generation = load_generation(pretrain_dir)
    seal = json.loads((pretrain_dir / "generation.json").read_text())
    if REGISTER_PATH.exists():
        register = json.loads(REGISTER_PATH.read_text())
        if (register.get("entries") or {}).get(seal["manifest_sha256"]):
            raise DispatchRefused(
                "candidate generation is QUARANTINED — dispatch "
                "refused")
    bound = design["shared_bindings"]["pretrain_generation"]
    if seal["manifest_sha256"] != bound["seal_manifest_sha256"]:
        raise DispatchRefused(
            "generation seal differs from the design binding — the "
            "design must be regenerated from the real seal")
    for family, digest in bound["per_family_encoder_digests"].items():
        actual = sha256_file(
            pretrain_dir / f"branch_{family}_encoder.pt")
        if actual != digest:
            raise DispatchRefused(
                f"encoder digest drift for {family}")
    snapshot = snapshot_effective_config(
        REPO / design["shared_bindings"]["strong_config"])
    if snapshot["materialized"]["architecture_digest"] != \
            design["shared_bindings"]["architecture_digest"]:
        raise DispatchRefused("strong-config architecture drift")
    trial = next(t for t in design["trial_ledger"]
                 if t["genesis"]["seed"] == seed
                 and t["genesis"]["arm"] == arm)
    cell = {
        "schema": "agent_multi.paired_sac_cell_genesis.v1",
        "trial_id": trial["trial_id"],
        "genesis_sha256": trial["genesis_sha256"],
        "arm": arm, "seed": seed,
        "mechanism": design["arms"][arm],
        "strong_config": design["shared_bindings"]["strong_config"],
        "architecture_digest":
            design["shared_bindings"]["architecture_digest"],
        "pretrain_generation_seal": seal["manifest_sha256"],
        "sac": design["shared_bindings"]["sac"],
        "envelope": design["shared_bindings"]["execution_envelope"],
        "data_roles": design["shared_bindings"]["data_roles"],
        "evaluation": design["shared_bindings"]["evaluation"],
        "predeclared_refusals": design["predeclared"],
        "status": "MATERIALIZED_NOT_LAUNCHED",
    }
    cell["cell_sha256"] = sha256_obj(cell)
    return cell


def frozen_o2022_envelope() -> dict:
    """The B4-calibrated, pre-2022-frozen ATR envelope (design
    binding); digest-verified against the calibration evidence."""
    calibration = json.loads(
        (REPO / ENVELOPE_CALIBRATION).read_text())
    if not calibration.get("frozen_before_score_year"):
        raise DispatchRefused(
            "o2022 envelope calibration is not frozen before the "
            "scored year")
    geometry = calibration["frozen_geometry"]
    expected = calibration["frozen_envelope_sha256"]
    import hashlib
    actual = hashlib.sha256(json.dumps(
        geometry, sort_keys=True, default=str).encode()).hexdigest()
    if actual != expected:
        raise DispatchRefused(
            f"frozen envelope digest drift: {actual[:12]} != "
            f"{expected[:12]}")
    if not (geometry.get("atr_sl_mult") == 3.0
            and geometry.get("atr_tp_mult") == 6.0):
        raise DispatchRefused(
            "frozen o2022 geometry is not the design-bound ATR "
            "3.0/6.0 envelope")
    return geometry


def build_cell_config(design: dict, cell: dict, pretrain_dir: Path,
                      output_root: Path, *, device: str,
                      dry_run_budget: dict | None = None) -> dict:
    """The FULL runtime config of one cell — pure and importable so the
    adversarial identity tests can prove that, for one seed, the
    resolved configs of the two arms differ ONLY in the initialization
    keys (order C4)."""
    snapshot = snapshot_effective_config(
        REPO / design["shared_bindings"]["strong_config"])
    cfg = dict(snapshot["env_config"])
    shared = design["shared_bindings"]
    envelope = frozen_o2022_envelope()
    cost_manifest = json.loads((REPO / COST_MANIFEST).read_text())
    cell_dir = output_root / cell["trial_id"]
    cfg.update({
        # accepted nested trainer + typed split roles
        "pipeline_plugin": "rl_pipeline_with_validation",
        "nested_split_contract": str(REPO / NESTED_CONTRACT),
        "nested_split_mode": "l1",
        "nested_split_dir": str(cell_dir / "splits"),
        "selection_metric": "paired_generalization_weekly_v1",
        # design-bound SAC facts, identical in both arms
        "learning_rate": float(shared["sac"]["learning_rate"]),
        "total_timesteps": int(
            shared["sac"]["budget_total_timesteps"]),
        "epoch_timesteps": EPOCH_TIMESTEPS,
        "max_epochs": MAX_EPOCHS,
        "l1_patience": L1_PATIENCE,
        "train_seed": int(cell["seed"]),
        "eval_seed": int(cell["seed"]),
        "device": device,
        # frozen pre-2022 execution envelope + ALPACA cost contract
        "strategy_plugin": "shared_execution_envelope",
        "execution_envelope": dict(envelope),
        **cost_manifest["alpaca_ethusd"]["env_binding"],
        # observation authority (findings 235/327 semantics): the
        # feature-aware contract is DECLARED, never absent
        "require_observation_declaration": True,
        "require_feature_aware_preprocessor": True,
        # finding 235: the unscaled raw price window is exactly what
        # killed the P1LR actor; the feature-aware contract forbids it
        # and the grouped extractor never consumes it
        "include_price_window": False,
        "observation_contract": {
            "require_feature_aware_preprocessor": True,
            "preprocessor_plugin": "feature_window_preprocessor",
            "feature_scaling": cfg["feature_scaling"],
            "feature_scaling_window": cfg["feature_scaling_window"],
            "include_price_window": False,
            "include_agent_state": cfg["include_agent_state"],
            "agent_state_contract": cfg["agent_state_contract"],
            "window_size": cfg["window_size"],
            "feature_columns_sha256":
                snapshot["materialized"]["feature_columns_sha256"],
        },
        # predeclared refusals are typed outcomes, not crashes
        "refuse_dead_actor": True,
        "refuse_constant_policy_actor": True,
        "inactive_terminal_is_typed_result": True,
        "evaluate_test_split": False,
        "quiet_mode": True,
        "save_model": str(cell_dir / "cell_model.zip"),
        "results_file": str(cell_dir / "cell_results.csv"),
        "save_config": str(cell_dir / "cell_effective_config.json"),
    })
    if cell["arm"] == "pretrained_finetuned":
        cfg["pretrained_branch_generation_dir"] = str(pretrain_dir)
        cfg["pretrained_branch_expected_seal"] = \
            cell["pretrain_generation_seal"]
    elif cell["arm"] != "control_random_init":
        raise DispatchRefused(f"unknown arm {cell['arm']!r}")
    if dry_run_budget:
        cfg.update(dict(dry_run_budget))
        cfg["dry_run_budget_disclosed"] = dict(dry_run_budget)
    assert_no_venue_keys(cfg)
    cfg["_snapshot_config_sha256"] = snapshot["config_sha256"]
    return cfg


def assert_no_venue_keys(cfg: dict) -> None:
    for key in cfg:
        lowered = str(key).lower()
        if any(fragment in lowered
               for fragment in FORBIDDEN_KEY_FRAGMENTS):
            raise DispatchRefused(
                f"venue-credential key {key!r} present in the cell "
                "config — this driver never opens a venue socket")


DRY_RUN_BUDGET = {
    # bounded CPU acceptance budget (C4) — every value disclosed in
    # the cell record; the full run drops this dict entirely
    "total_timesteps": 2600,
    "epoch_timesteps": 1300,
    "max_epochs": 2,
    "l1_patience": 2,
    "learning_starts": 128,
    "buffer_size": 4000,
    "batch_size": 64,
}


def execute_cell(design: dict, cell: dict, pretrain_dir: Path,
                 output_root: Path, *, device: str,
                 dry_run: bool) -> dict:
    """Run ONE cell through the accepted nested trainer under custody.
    Every attempt is a NEW identity (non-resumable by construction)."""
    from app.plugin_loader import load_plugin

    cfg = build_cell_config(
        design, cell, pretrain_dir, output_root, device=device,
        dry_run_budget=DRY_RUN_BUDGET if dry_run else None)
    attempt_nonce = os.urandom(8).hex()
    snapshot_sha = cfg.pop("_snapshot_config_sha256")
    key = dispatch_key(
        dispatch_id=(f"paired_sac_{cell['trial_id']}"
                     f"_attempt_{attempt_nonce}"),
        generation_digest=cell["pretrain_generation_seal"],
        architecture_digest=cell["architecture_digest"],
        config_snapshot_digest=snapshot_sha,
        data_digest=sha256_file(Path(cfg["input_data_file"])),
        code_identity={"driver": sha256_file(Path(__file__))})
    cell_dir = output_root / cell["trial_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = cell_dir / f"cell_record_{attempt_nonce}.json"
    ledger = DispatchLedger()
    ledger.reserve(key, identity={
        "dispatch_id": f"paired_sac_{cell['trial_id']}",
        "attempt_nonce": attempt_nonce,
        "trial_id": cell["trial_id"],
        "genesis_sha256": cell["genesis_sha256"],
        "cell_sha256": cell["cell_sha256"],
        "arm": cell["arm"], "seed": cell["seed"],
        "mode": "cpu_dry_run" if dry_run else "full_budget",
    }, output_path=evidence_path)
    ledger.transition(key, "running")
    import time
    wall_start = time.perf_counter()
    try:
        agent_cls, _ = load_plugin("agent.plugins", "sac_agent")
        pipeline_cls, _ = load_plugin(
            "pipeline.plugins", "rl_pipeline_with_validation")
        agent_plugin = agent_cls(cfg)
        pipeline = pipeline_cls(cfg)
        ledger.mark_forward_started(key)
        final = pipeline.run_pipeline(
            config=cfg, env_plugin=None, agent_plugin=agent_plugin,
            mode="train")
        record = {
            "schema": "agent_multi.paired_sac_cell_record.v1",
            "cell": cell,
            "attempt_nonce": attempt_nonce,
            "mode": "cpu_dry_run" if dry_run else "full_budget",
            "dry_run_budget": (dict(DRY_RUN_BUDGET) if dry_run
                               else None),
            "custody_key": key,
            "wall_seconds": round(time.perf_counter() - wall_start, 1),
            "resolved_overrides": {
                k: cfg[k] for k in sorted(cfg)
                if k not in ("feature_columns",
                             "feature_binary_columns")},
            "pretrained_branch_transfer_evidence":
                final.get("pretrained_branch_transfer_evidence"),
            "history": final.get("history"),
            "actor_liveness_history":
                final.get("actor_liveness_history"),
            "final": {k: final.get(k) for k in (
                "splits", "summary_table", "selection_metric",
                "train_validation_l1_score",
                "risk_adjusted_total_return", "sharpe_ratio",
                "max_drawdown_fraction",
                "best_composite", "stop_reason", "termination_cause",
                "artifacts", "best_model_path",
                "terminal_model_path", "replay_disposition",
                "observation_contract",
                "activity_stopped_without_eligible_checkpoint")},
            "gradient_updates_derived": {
                "note": "DERIVED: terminal num_timesteps - "
                        "learning_starts at train_freq=1, "
                        "gradient_steps=1",
                "terminal_num_timesteps": (final.get("artifacts") or
                                           {}).get("terminal", {}).get(
                                               "num_timesteps"),
                "learning_starts": cfg.get("learning_starts")},
        }
        evidence_path.write_text(json.dumps(record, indent=1,
                                            default=str))
        ledger.complete(
            key, evidence_path,
            expected_schema="agent_multi.paired_sac_cell_record.v1",
            run_id=key[:16],
            dispatch_id=f"paired_sac_{cell['trial_id']}")
        return record
    except BaseException as exc:
        try:
            ledger.transition(key, "interrupted", {
                "interruption": f"{type(exc).__name__}: {exc}",
                "non_resumable": "order C3 — a new attempt identity "
                                 "is required"})
        except Exception:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-root", default=None,
                        help="cell output root (required for "
                             "execution modes)")
    parser.add_argument("--execute-cpu-dry-run", action="store_true",
                        help="bounded CPU acceptance run (C4); "
                             "refuses if CUDA is visible")
    parser.add_argument("--execute", action="store_true",
                        help="full-budget run; requires "
                             "--gpu-authorized-by-musashi and CUDA")
    parser.add_argument("--gpu-authorized-by-musashi", default=None,
                        metavar="DISPATCH_DOC")
    args = parser.parse_args()
    design = json.loads(DESIGN_PATH.read_text())
    cell = verify_cell(design, Path(args.pretrain_dir), args.seed,
                       args.arm)
    print(json.dumps(cell, indent=1))
    if not (args.execute or args.execute_cpu_dry_run):
        print("NOT_LAUNCHED: verification only — execution requires "
              "--execute-cpu-dry-run (CPU) or --execute (GPU, "
              "Musashi-authorized)", file=sys.stderr)
        return 0
    if args.execute and args.execute_cpu_dry_run:
        raise DispatchRefused("choose ONE execution mode")
    if not args.output_root:
        raise DispatchRefused("execution requires --output-root")
    import torch
    if args.execute_cpu_dry_run:
        if torch.cuda.is_available():
            raise DispatchRefused(
                "CUDA is visible — the CPU dry run must be "
                "structurally unable to touch a GPU (run under "
                "CUDA_VISIBLE_DEVICES=\"\")")
        record = execute_cell(design, cell, Path(args.pretrain_dir),
                              Path(args.output_root), device="cpu",
                              dry_run=True)
        print(json.dumps({"status": "CPU_DRY_RUN_COMPLETED",
                          "custody_key": record["custody_key"][:16],
                          "stop_reason":
                              record["final"]["stop_reason"]},
                         indent=1))
        return 0
    if args.gpu_authorized_by_musashi is None:
        raise DispatchRefused(
            "full-budget execution requires "
            "--gpu-authorized-by-musashi <dispatch-doc> naming "
            "Musashi's written acceptance (order C5)")
    dispatch_doc = Path(args.gpu_authorized_by_musashi)
    if not dispatch_doc.is_file():
        raise DispatchRefused(
            "authorization document does not exist — refused")
    if not torch.cuda.is_available():
        raise DispatchRefused(
            "no CUDA visibility — the operator has not granted a GPU")
    record = execute_cell(design, cell, Path(args.pretrain_dir),
                          Path(args.output_root), device="cuda",
                          dry_run=False)
    print(json.dumps({"status": "CELL_COMPLETED",
                      "custody_key": record["custody_key"][:16]},
                     indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DispatchRefused as exc:
        print(f"DISPATCH REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
