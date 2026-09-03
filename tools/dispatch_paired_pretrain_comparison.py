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

Execution attempts are OBSERVABLE and RESUMABLE (Musashi correction 3,
2026-09-03, superseding the C3-era non-resumable rule): a fresh
dispatch mints a new attempt identity (custody dispatch key); an
INTERRUPTED attempt resumes THE SAME identity via
``--resume-attempt <dir>``, restoring model, optimizers, replay
buffer, counters, RNG, patience and evaluation histories exactly from
the attempt's resume bundle; ``--status-attempt <dir>`` reads the
machine-readable heartbeat without touching any process.
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
from agent_plugins.dispatch_authorization import (  # noqa: E402
    AuthorizationRefused, bounded_extractor_forward,
    cudnn_micro_preflight, executable_manifest,
    executable_manifest_digest, resolve_required_entry_points,
    verify_authorization, verify_device_binding,
    verify_worktree_identity)
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
FLEET_PLAN = (REPO / "docs/audits/evidence/"
              "PAIRED_SAC_FLEET_PLAN_2026_08_28.json")
MANIFEST_DIR = (REPO / "docs/audits/evidence/"
                "paired_sac_launch_manifests_20260828")
CAMPAIGN_ID = "paired_pretrain_sac_eth_o2022_20260828"
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
                      dry_run_budget: dict | None = None,
                      attempt_nonce: str | None = None) -> dict:
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
    # DATA-SOTA-378: every artifact of an attempt lives in its own
    # exclusive attempt directory; attempts never share a path
    cell_dir = output_root / cell["trial_id"]
    if attempt_nonce is not None:
        cell_dir = cell_dir / f"attempt_{attempt_nonce}"
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
    # Musashi correction 3 (2026-09-03): observable resumable cell
    # runtime — heartbeat/status/ETA + periodic exact-resume bundle
    cfg["cell_runtime_dir"] = str(cell_dir / "runtime")
    cfg["resume_checkpoint_every_epochs"] = 5
    # Musashi correction 4 (2026-09-03): the ACCEPTED executing
    # budget guard rides in EVERY cell — steps, updates, wall clock
    # and external stop-file, enforced inside the trainer
    _total = int(cfg["total_timesteps"])
    cfg["budget_max_env_steps"] = _total
    cfg["budget_max_updates"] = _total
    cfg["budget_max_wall_seconds"] = float(
        shared["sac"].get("cell_wall_ceiling_seconds", 43200.0))
    cfg["budget_stop_file"] = str(cell_dir / "STOP")
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


def make_attempt_dir(output_root: Path, trial_id: str,
                     attempt_nonce: str) -> Path:
    """DATA-SOTA-378: exclusive creation BEFORE pipeline construction.
    Pre-existing, non-empty or symlinked destinations refuse; the
    parent directory is fsynced so the attempt boundary is durable."""
    output_root = Path(output_root)
    trial_dir = output_root / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    for probe in (output_root, trial_dir):
        if probe.is_symlink():
            raise DispatchRefused(
                f"{probe} is a symlink — attempt isolation refuses "
                "(DATA-SOTA-378)")
    attempt_dir = trial_dir / f"attempt_{attempt_nonce}"
    if attempt_dir.is_symlink() or attempt_dir.exists():
        raise DispatchRefused(
            f"attempt directory {attempt_dir.name} already exists — "
            "a fresh attempt NEVER reuses or overwrites a prior "
            "attempt's paths (DATA-SOTA-378)")
    os.mkdir(attempt_dir)
    fd = os.open(trial_dir, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return attempt_dir


def attempt_inventory(attempt_dir: Path,
                      exclude: set[str] | None = None) -> dict:
    """Digest inventory of every file the attempt produced — bound
    into the terminal evidence (DATA-SOTA-378)."""
    exclude = exclude or set()
    inventory = {}
    for path in sorted(Path(attempt_dir).rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(attempt_dir))
            if rel not in exclude:
                inventory[rel] = sha256_file(path)
    return inventory


def verify_slot_binding(cell: dict, logical_slot: str) -> dict:
    """DATA-SOTA-380: the logical slot is a REQUIRED, verified input —
    wrong slot, seed, arm or within-slot position refuses; the cell's
    launch manifest must bind the same genesis."""
    plan = json.loads(FLEET_PLAN.read_text())
    assignment = next((a for a in plan["assignments"]
                       if a["slot"] == logical_slot), None)
    if assignment is None:
        raise DispatchRefused(
            f"unknown logical slot {logical_slot!r} (DATA-SOTA-380)")
    if int(assignment["seed"]) != int(cell["seed"]):
        raise DispatchRefused(
            f"slot {logical_slot} carries seed {assignment['seed']}, "
            f"not {cell['seed']} — wrong slot for this cell "
            "(DATA-SOTA-380)")
    if cell["trial_id"] not in assignment["cells_in_order"]:
        raise DispatchRefused(
            f"{cell['trial_id']} is not assigned to {logical_slot} "
            "(DATA-SOTA-380)")
    position = assignment["cells_in_order"].index(cell["trial_id"])
    manifest_path = MANIFEST_DIR / f"launch_{cell['trial_id']}.json"
    if not manifest_path.is_file():
        raise DispatchRefused(
            f"launch manifest absent for {cell['trial_id']}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["gpu_logical_slot"] != logical_slot:
        raise DispatchRefused(
            "launch manifest binds slot "
            f"{manifest['gpu_logical_slot']!r}, not {logical_slot!r} "
            "(DATA-SOTA-380)")
    if int(manifest["within_slot_position"]) != position:
        raise DispatchRefused(
            "within-slot execution position mismatch between fleet "
            "plan and launch manifest (DATA-SOTA-380)")
    genesis = manifest["cell_genesis"]
    if genesis["cell_sha256"] != cell["cell_sha256"] or             genesis["genesis_sha256"] != cell["genesis_sha256"]:
        raise DispatchRefused(
            "launch-manifest genesis differs from the verified cell "
            "— stale manifest refuses (DATA-SOTA-380)")
    if genesis["arm"] != cell["arm"] or             int(genesis["seed"]) != int(cell["seed"]):
        raise DispatchRefused(
            "launch-manifest arm/seed mismatch (DATA-SOTA-380)")
    return {"manifest": manifest,
            "manifest_sha256": sha256_file(manifest_path),
            "within_slot_position": position}


def verify_executable_identity(manifest: dict) -> dict:
    """DATA-SOTA-379: the executing file set must equal the canonical
    allowlist bound in the launch manifest — computed from bytes NOW,
    never from a sidecar."""
    actual = executable_manifest(REPO)
    expected = manifest.get("executable_allowlist_sha256")
    if not expected:
        raise DispatchRefused(
            "launch manifest carries no executable allowlist — "
            "refused (DATA-SOTA-379)")
    drift = {name: (expected.get(name, "<absent>")[:12],
                    actual.get(name, "<absent>")[:12])
             for name in set(expected) | set(actual)
             if expected.get(name) != actual.get(name)}
    if drift:
        raise DispatchRefused(
            f"executable identity drift vs the launch manifest: "
            f"{drift} — a modified executable never runs under an "
            "accepted genesis (DATA-SOTA-379)")
    return {"executable_allowlist_sha256": actual,
            "executable_allowlist_digest":
                executable_manifest_digest(actual)}


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
                 output_root: Path, *, device: str, dry_run: bool,
                 logical_slot: str,
                 slot_binding: dict | None = None,
                 identity_proof: dict | None = None,
                 authorization_sha256: str | None = None,
                 environment_preflight: dict | None = None,
                 resume_attempt_dir: Path | None = None,
                 scientific_gate_path: Path | None = None) -> dict:
    """Run ONE cell through the accepted nested trainer under custody.

    Musashi correction 3 (2026-09-03): cells are OBSERVABLE and
    RESUMABLE. A fresh dispatch mints a new attempt identity (nonce
    FIRST, every artifact inside the exclusive attempt directory,
    DATA-SOTA-378); an interrupted attempt resumes THE SAME identity
    through the evidenced custody door (interrupted -> running),
    restoring model, optimizers, replay buffer, counters, RNG,
    patience and the evaluation histories exactly from the attempt's
    resume bundle. Heartbeat/status/ETA in runtime/status.json."""
    from app.plugin_loader import load_plugin

    if resume_attempt_dir is not None:
        attempt_dir = Path(resume_attempt_dir)
        name = attempt_dir.name
        if not name.startswith("attempt_"):
            raise DispatchRefused(
                f"{attempt_dir} is not an attempt directory")
        attempt_nonce = name[len("attempt_"):]
        if not (attempt_dir / "custody_key.json").exists():
            raise DispatchRefused(
                "resume refused: the attempt carries no "
                "custody_key.json")
    else:
        attempt_nonce = os.urandom(8).hex()
        attempt_dir = make_attempt_dir(output_root, cell["trial_id"],
                                       attempt_nonce)
    cfg = build_cell_config(
        design, cell, pretrain_dir, output_root, device=device,
        dry_run_budget=DRY_RUN_BUDGET if dry_run else None,
        attempt_nonce=attempt_nonce)
    snapshot_sha = cfg.pop("_snapshot_config_sha256")
    # DATA-SOTA-379: hash the executable tree at preflight...
    executables_preflight = executable_manifest(REPO)
    key = dispatch_key(
        dispatch_id=(f"paired_sac_{cell['trial_id']}"
                     f"_attempt_{attempt_nonce}"),
        generation_digest=cell["pretrain_generation_seal"],
        architecture_digest=cell["architecture_digest"],
        config_snapshot_digest=snapshot_sha,
        data_digest=sha256_file(Path(cfg["input_data_file"])),
        code_identity={
            "executable_allowlist_digest":
                executable_manifest_digest(executables_preflight)})
    evidence_path = attempt_dir / f"cell_record_{attempt_nonce}.json"
    ledger = DispatchLedger()
    if resume_attempt_dir is not None:
        # correction 3: same identity, evidenced resume door
        stored = json.loads(
            (attempt_dir / "custody_key.json").read_text())
        if stored["custody_key"] != key:
            raise DispatchRefused(
                "resume refused: the recomputed dispatch key does "
                "not equal the stored one — design/cell/config/code "
                "identity drifted since the interrupted attempt")
        resume_state = (attempt_dir / "runtime" /
                        "resume_state.json")
        if not resume_state.exists():
            raise DispatchRefused(
                "resume refused: no resume bundle in the attempt's "
                "runtime directory")
        saved = json.loads(resume_state.read_text())
        ledger.resume(key, resume_evidence={
            "resume_state_sha256": sha256_file(resume_state),
            "resumed_from_epoch": saved["epoch"],
            "resumed_at_wall": __import__("time").time()})
        cfg["resume_from_cell_runtime"] = True
    else:
        ledger.reserve(key, identity={
            "dispatch_id": f"paired_sac_{cell['trial_id']}",
            "attempt_nonce": attempt_nonce,
            "trial_id": cell["trial_id"],
            "genesis_sha256": cell["genesis_sha256"],
            "cell_sha256": cell["cell_sha256"],
            "arm": cell["arm"], "seed": cell["seed"],
            "logical_slot": logical_slot,
            "executable_allowlist_digest":
                executable_manifest_digest(executables_preflight),
            "worktree_identity": identity_proof,
            "authorization_sha256": authorization_sha256,
            "mode": "cpu_dry_run" if dry_run else "full_budget",
        }, output_path=evidence_path)
        ledger.transition(key, "running")
        (attempt_dir / "custody_key.json").write_text(json.dumps(
            {"custody_key": key,
             "trial_id": cell["trial_id"],
             "attempt_nonce": attempt_nonce}, indent=1))
    import time
    wall_start = time.perf_counter()
    try:
        # ...and again immediately before construction: any drift
        # between preflight and model construction refuses
        executables_now = executable_manifest(REPO)
        if executables_now != executables_preflight:
            raise DispatchRefused(
                "executable tree changed between preflight and model "
                "construction — refused (DATA-SOTA-379)")
        # C3 (order @1649e7c0 §4.4): the dispatcher independently
        # RE-DERIVES the scientific gate immediately before any
        # CUDA/model/environment construction — a gate accepted at
        # argument parsing can never be swapped afterwards
        if scientific_gate_path is not None:
            from tools.sac_scientific_gate import (
                verify_gate_for_dispatch)
            try:
                verify_gate_for_dispatch(Path(scientific_gate_path))
            except SystemExit as exc:
                raise DispatchRefused(
                    f"pre-construction gate re-derivation: {exc}")
        device_class = None
        if device == "cuda":
            import torch
            if torch.cuda.device_count() != 1:
                raise DispatchRefused(
                    f"{torch.cuda.device_count()} CUDA devices "
                    "visible — exactly ONE device per cell process "
                    "(DATA-SOTA-380)")
            device_class = torch.cuda.get_device_name(0)
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
            "environment_preflight": environment_preflight,
            "logical_slot": logical_slot,
            "device_class_sanitized": device_class,
            "slot_binding": ({k: slot_binding[k] for k in
                              ("manifest_sha256",
                               "within_slot_position")}
                             if slot_binding else None),
            "worktree_identity": identity_proof,
            "authorization_sha256": authorization_sha256,
            "executable_allowlist_sha256": executables_preflight,
            "executable_allowlist_digest":
                executable_manifest_digest(executables_preflight),
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
        # DATA-SOTA-378: bind the attempt directory's digest
        # inventory into the terminal evidence (the record itself is
        # excluded by construction — it is being written)
        record["attempt_directory_inventory"] = attempt_inventory(
            attempt_dir, exclude={evidence_path.name})
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
                "resumable": "Musashi correction 3 (2026-09-03): "
                             "resume THE SAME attempt via "
                             "--resume-attempt <dir> — exact state "
                             "from runtime/resume_state.json"})
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
                        metavar="AUTHORIZATION_ARTIFACT",
                        help="typed agent_multi.paired_sac_dispatch_"
                             "authorization.v1 artifact published by "
                             "Musashi — content-verified field by "
                             "field (DATA-SOTA-377)")
    parser.add_argument("--logical-slot", default=None,
                        metavar="gpu_slot_N",
                        help="REQUIRED for execution modes: the fleet "
                             "plan's logical slot for this cell "
                             "(DATA-SOTA-380)")
    parser.add_argument("--scientific-gate", default=None,
                        metavar="GATE_ARTIFACT",
                        help="REQUIRED for execution modes (Musashi "
                             "correction 2): a SAC_GATE_PASS "
                             "artifact from tools/sac_scientific_"
                             "gate.py bound to the observable-"
                             "runtime screen report")
    parser.add_argument("--resume-attempt", default=None,
                        metavar="ATTEMPT_DIR",
                        help="resume an INTERRUPTED attempt exactly "
                             "(same custody identity; model, "
                             "optimizers, replay, counters, RNG, "
                             "patience, evaluations restored)")
    parser.add_argument("--status-attempt", default=None,
                        metavar="ATTEMPT_DIR",
                        help="print the machine-readable runtime "
                             "status of an attempt and exit — no "
                             "process attachment")
    args = parser.parse_args()
    if args.status_attempt:
        adir = Path(args.status_attempt)
        out = {"attempt_dir": str(adir)}
        sp = adir / "runtime" / "status.json"
        out["runtime_status"] = (json.loads(sp.read_text())
                                 if sp.exists() else None)
        kp = adir / "custody_key.json"
        if kp.exists():
            stored = json.loads(kp.read_text())
            out["custody_key"] = stored["custody_key"][:16]
            rec = DispatchLedger().read(stored["custody_key"])
            out["custody_state"] = (rec or {}).get("state")
            out["resume_history"] = (rec or {}).get(
                "resume_history")
        print(json.dumps(out, indent=1, default=str))
        return 0
    design = json.loads(DESIGN_PATH.read_text())
    cell = verify_cell(design, Path(args.pretrain_dir), args.seed,
                       args.arm)
    print(json.dumps(cell, indent=1))
    if not (args.execute or args.execute_cpu_dry_run):
        if args.logical_slot:
            binding = verify_slot_binding(cell, args.logical_slot)
            identity = verify_executable_identity(
                binding["manifest"])
            print(json.dumps({
                "verification_only": True,
                "logical_slot": args.logical_slot,
                "within_slot_position":
                    binding["within_slot_position"],
                "launch_manifest_sha256":
                    binding["manifest_sha256"],
                "executable_allowlist_digest":
                    identity["executable_allowlist_digest"],
            }, indent=1))
        print("NOT_LAUNCHED: verification only — execution requires "
              "--execute-cpu-dry-run (CPU) or --execute (GPU, "
              "Musashi-authorized)", file=sys.stderr)
        return 0
    if args.execute and args.execute_cpu_dry_run:
        raise DispatchRefused("choose ONE execution mode")
    if not args.output_root:
        raise DispatchRefused("execution requires --output-root")
    if not args.logical_slot:
        raise DispatchRefused(
            "execution requires --logical-slot (DATA-SOTA-380)")
    # Musashi correction 2 (2026-09-03): the SCIENTIFIC GATE comes
    # before any SAC spend — the fused representation must have
    # demonstrated branch-signal conservation in the observable
    # screen, or the eight cells are NOT launched.
    if not args.scientific_gate:
        raise DispatchRefused(
            "execution requires --scientific-gate <artifact> "
            "(Musashi correction 2): no fusion evidence, no SAC")
    from tools.sac_scientific_gate import verify_gate_for_dispatch
    try:
        gate = verify_gate_for_dispatch(Path(args.scientific_gate))
    except SystemExit as exc:
        raise DispatchRefused(str(exc))
    print(json.dumps({"scientific_gate": gate["gate"],
                      "advancing_fusion_variants":
                          gate["advancing_fusion_variants"]},
                     indent=1))
    slot_binding = verify_slot_binding(cell, args.logical_slot)
    import torch
    if args.execute_cpu_dry_run:
        if torch.cuda.is_available():
            raise DispatchRefused(
                "CUDA is visible — the CPU dry run must be "
                "structurally unable to touch a GPU (run under "
                "CUDA_VISIBLE_DEVICES=\"\")")
        # DATA-SOTA-382: the EXECUTING environment must prove its
        # entry-point metadata and run a bounded forward on the
        # selected device BEFORE any attempt exists
        preflight = {
            **resolve_required_entry_points(REPO),
            "bounded_forward": bounded_extractor_forward(REPO, "cpu"),
        }
        record = execute_cell(design, cell, Path(args.pretrain_dir),
                              Path(args.output_root), device="cpu",
                              dry_run=True,
                              logical_slot=args.logical_slot,
                              slot_binding=slot_binding,
                              environment_preflight=preflight,
                              resume_attempt_dir=(
                                  Path(args.resume_attempt)
                                  if args.resume_attempt else None),
                              scientific_gate_path=Path(
                                  args.scientific_gate))
        print(json.dumps({"status": "CPU_DRY_RUN_COMPLETED",
                          "custody_key": record["custody_key"][:16],
                          "stop_reason":
                              record["final"]["stop_reason"]},
                         indent=1))
        return 0
    if args.gpu_authorized_by_musashi is None:
        raise DispatchRefused(
            "full-budget execution requires "
            "--gpu-authorized-by-musashi <authorization-artifact> — "
            "the typed artifact Musashi publishes after reproducing "
            "the return (order H5)")
    # DATA-SOTA-377: typed, content-bound authorization — verified
    # field by field BEFORE any CUDA probe or model construction
    manifest_digests = {
        trial["trial_id"]: sha256_file(
            MANIFEST_DIR / f"launch_{trial['trial_id']}.json")
        for trial in design["trial_ledger"]}
    identity = verify_executable_identity(slot_binding["manifest"])
    authorization = verify_authorization(
        Path(args.gpu_authorized_by_musashi),
        campaign_id=CAMPAIGN_ID,
        trial_ids=[t["trial_id"] for t in design["trial_ledger"]],
        paired_design_sha256=sha256_file(DESIGN_PATH),
        candidate_seal_manifest_sha256=cell[
            "pretrain_generation_seal"],
        launch_manifest_sha256=manifest_digests,
        executable_allowlist_sha256=identity[
            "executable_allowlist_digest"])
    # DATA-SOTA-379: exact HEAD + clean tree, equal to the
    # authorization's reviewed commit
    identity_proof = verify_worktree_identity(
        REPO, expected_commit=authorization[
            "reviewed_correction_commit"])
    if not torch.cuda.is_available():
        raise DispatchRefused(
            "no CUDA visibility — the operator has not granted a GPU")
    # DATA-SOTA-383: physical binding + cuDNN micro-preflight BEFORE
    # any custody reservation — failure refuses without spending an
    # attempt identity
    device_binding = verify_device_binding(args.logical_slot)
    micro = cudnn_micro_preflight("cuda")
    # DATA-SOTA-382: entry-point metadata + bounded forward on the
    # BOUND device, still before the attempt exists
    preflight = {
        **resolve_required_entry_points(REPO),
        "bounded_forward": bounded_extractor_forward(REPO, "cuda"),
        "device_binding": device_binding,
        "cudnn_micro_preflight": micro,
    }
    record = execute_cell(
        design, cell, Path(args.pretrain_dir),
        Path(args.output_root), device="cuda", dry_run=False,
        logical_slot=args.logical_slot, slot_binding=slot_binding,
        identity_proof=identity_proof,
        authorization_sha256=sha256_file(
            Path(args.gpu_authorized_by_musashi)),
        environment_preflight=preflight,
        resume_attempt_dir=(Path(args.resume_attempt)
                            if args.resume_attempt else None),
        scientific_gate_path=Path(args.scientific_gate))
    print(json.dumps({"status": "CELL_COMPLETED",
                      "custody_key": record["custody_key"][:16],
                      "logical_slot": args.logical_slot},
                     indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DispatchRefused, AuthorizationRefused) as exc:
        print(f"DISPATCH REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
