"""Project 3 Phase 4 dry-run protocol comparator scaffold.

Builds the four mandatory comparison arms (A/B/C/D) for the
synthetic-pretraining ablation as **locked, non-runnable templates**
plus a manifest with hashes of the source artifacts. It is strictly a
validation + scaffolding tool: it never invokes ``agent-multi`` and
never writes anywhere outside ``--out-dir``.

Design (see Phase 4 §4.3):

* Arm A — ``real_only_standard``: 1:1 copy of the Stage A reference
  config; promotion eligible.
* Arm B — ``real_only_compute_matched``: same as Arm A but with a
  training budget equal to Arm D's pretrain + finetune total. The
  *only* permitted delta vs Arm A is ``total_timesteps``. Promotion
  eligible.
* Arm C — ``synthetic_only_diagnostic``: trains exclusively on the
  augmented synthetic panel for the train window. Validation/test
  windows still consume the real panel. Always
  non-promotion-eligible.
* Arm D — ``synthetic_pretrain_then_real_finetune``: pretrain on the
  augmented panel, then load the resulting policy and fine-tune on
  the real panel. SAC hyperparameters MUST match the reference
  exactly. Promotion eligible (Phase 4 primary candidate).

The validator is fail-closed; refer to ``DryRunValidator`` for the
full rule set. No training is launched.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
SCHEMA_VERSION = "1.0.0"
PROJECT3_HELDOUT_START = "2025-01-01"

# Fields that must be byte-identical across all four arms.
REQUIRED_EQUAL_FIELDS = (
    "asset",
    "env_plugin",
    "agent_plugin",
    "strategy_plugin",
    "reward_plugin",
    "metrics_plugin",
    "broker_plugin",
    "data_feed_plugin",
    "preprocessor_plugin",
    "pipeline_plugin",
    "optimizer_plugin",
    "features_preset",
    "action_space_mode",
    "continuous_action_threshold",
    "window_size",
    "atr_period",
    "k_sl",
    "k_tp",
    "commission",
    "slippage",
    "leverage",
    "position_size",
    "rel_volume",
    "size_mode",
    "min_order_volume",
    "max_order_volume",
    "initial_cash",
    "price_column",
    "date_column",
    # SAC hyperparameters — Arm D must NOT change these.
    "learning_rate",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "tau",
    "gamma",
    "train_freq",
    "gradient_steps",
    "ent_coef",
    "use_sde",
)

# Fields the arms are *allowed* to differ on. Any other delta fails.
ALLOWED_DIFFERENT_FIELDS = (
    "input_data_file",
    "total_timesteps",
    "_protocol_lock",
    "_phase4_arm",
    "_arm_pretrain",
    "_arm_finetune",
    "save_model",
    "save_config",
    "results_file",
    "train_seed",
    "eval_seed",
    "mode",
    "env_mode",
    "quiet_mode",
    "agent_verbose",
)

# Cost scenarios — kept identical across arms; different scenarios are
# evaluated downstream, not by this scaffold.
COST_SCENARIOS = ("base", "pessimistic")


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_synthetic(path: str) -> bool:
    p = (path or "").lower()
    if not p:
        return False
    return (
        "synthetic" in p
        or "/synthetic-datagen/" in p
        or "augmented_regime_residual" in p
        or "augmented_stationary_bootstrap" in p
        or "augmented_synthetic" in p
        or "_synthetic" in os.path.basename(p)
    )


# ---------------------------------------------------------------------------
@dataclass
class ArmSpec:
    """Concrete description of one comparison arm."""

    name: str
    role: str  # "real_only_standard" | ... | "synthetic_pretrain_then_real_finetune"
    promotion_eligible: bool
    description: str
    config_filename: str
    execution_status: str
    multi_phase: bool = False
    # Filled in when the config is materialized.
    output_path: Optional[str] = None
    extra_keys_for_validator: Dict[str, Any] = field(default_factory=dict)


ARM_SPECS: Tuple[ArmSpec, ...] = (
    ArmSpec(
        name="arm_a",
        role="real_only_standard",
        promotion_eligible=True,
        description=(
            "Reference real-only SAC training. Identical to the Stage A "
            "reference except for run-output paths and seeds."
        ),
        config_filename="arm_a_real_only_standard.json",
        execution_status="RUNNABLE_AFTER_STAGE_B_APPROVAL",
    ),
    ArmSpec(
        name="arm_b",
        role="real_only_compute_matched",
        promotion_eligible=True,
        description=(
            "Real-only SAC with training budget compute-matched to "
            "Arm D's pretrain + finetune total. Only `total_timesteps` "
            "differs from Arm A."
        ),
        config_filename="arm_b_real_only_compute_matched.json",
        execution_status="RUNNABLE_AFTER_STAGE_B_APPROVAL",
    ),
    ArmSpec(
        name="arm_c",
        role="synthetic_only_diagnostic",
        promotion_eligible=False,
        description=(
            "Synthetic-only training. Diagnostic only; never promotion "
            "eligible. Validation and test windows still consume the "
            "real panel."
        ),
        config_filename="arm_c_synthetic_only_diagnostic.json",
        execution_status="DIAGNOSTIC_ONLY_NEVER_PROMOTE",
    ),
    ArmSpec(
        name="arm_d",
        role="synthetic_pretrain_then_real_finetune",
        promotion_eligible=True,
        description=(
            "Two-phase: pretrain on augmented synthetic panel, then "
            "fine-tune on real panel. SAC hyperparameters identical to "
            "the reference. PRIMARY Phase 4 comparison vs Arm B."
        ),
        config_filename="arm_d_synthetic_pretrain_then_real_finetune.json",
        execution_status="TEMPLATE_ONLY_MULTI_PHASE_RUNNER_NOT_IMPLEMENTED",
        multi_phase=True,
    ),
)


# ---------------------------------------------------------------------------
def _strip_runtime_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Remove run-specific output paths from a base config copy."""
    out = copy.deepcopy(cfg)
    for k in ("save_model", "save_config", "results_file"):
        out.pop(k, None)
    return out


def _make_protocol_lock(
    arm_role: str,
    promotion_eligible: bool,
    protocol_packet_path: str,
    protocol_packet_hash: str,
    multi_phase: bool,
) -> Dict[str, Any]:
    if arm_role == "synthetic_only_diagnostic":
        execution_status = "DIAGNOSTIC_ONLY_NEVER_PROMOTE"
    elif multi_phase:
        execution_status = "TEMPLATE_ONLY_MULTI_PHASE_RUNNER_NOT_IMPLEMENTED"
    else:
        execution_status = "RUNNABLE_AFTER_STAGE_B_APPROVAL"
    return {
        "_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED": True,
        "phase": "phase_4",
        "arm_role": arm_role,
        "promotion_eligible": bool(promotion_eligible),
        "reason": (
            "Project 3 Phase 4 dry-run scaffold. Stage B approval must "
            "be recorded in SYNTHETIC_LEDGER.csv before any of the "
            "four arms is executed."
        ),
        "protocol_packet": protocol_packet_path,
        "protocol_packet_sha256": protocol_packet_hash,
        "compare_plan": (
            "synthetic-datagen/experiments/synthetic_data/"
            "project3_eth_4h/COMPARE_PLAN.md"
        ),
        "execution_status": execution_status,
        "diagnostic_only": (arm_role == "synthetic_only_diagnostic"),
    }


def _resolve_synthetic_inputs(packet: Dict[str, Any]) -> Tuple[str, str]:
    """Extract (synthetic_train_csv, synthetic_only_train_csv) from packet."""
    out = packet["output_files"]
    augmented = out["augmented_tech_stat_csv"]["path"]
    syn_only = out.get("synthetic_only_tech_stat_csv", {}).get("path", augmented)
    return augmented, syn_only


def _resolve_real_input(reference_cfg: Dict[str, Any]) -> str:
    """Resolve the real Project 3 panel from either a real or locked Phase 4 config.

    Some callers pass the locked synthetic-pretraining template as the
    reference config because it carries the exact SAC/agent-multi settings.
    In that case the top-level ``input_data_file`` is intentionally synthetic,
    while the real fine-tune panel is recorded in the protocol lock. The arm
    scaffold must use that real panel for Arms A/B, Arm C validation/test, and
    Arm D fine-tuning.
    """
    top_level = str(reference_cfg.get("input_data_file", ""))
    if top_level and not _looks_like_synthetic(top_level):
        return top_level

    lock = reference_cfg.get("_protocol_lock") or {}
    finetune = lock.get("finetune") or {}
    lock_real_input = str(finetune.get("input_data_file", ""))
    if lock_real_input and not _looks_like_synthetic(lock_real_input):
        return lock_real_input

    raise ValueError(
        "Could not resolve a real input_data_file from reference config. "
        "Pass a real Stage A/agent-multi config or a locked Phase 4 template "
        "with _protocol_lock.finetune.input_data_file pointing to the real panel."
    )


# ---------------------------------------------------------------------------
def build_arm_configs(
    reference_cfg: Dict[str, Any],
    protocol_packet: Dict[str, Any],
    *,
    seeds: List[int],
    protocol_packet_path: str,
    protocol_packet_hash: str,
    reference_config_path: str,
) -> Dict[str, Dict[str, Any]]:
    """Construct in-memory Arm A/B/C/D configs (one per arm, not per seed).

    Per-seed expansion is the responsibility of the downstream sweep
    runner. Each arm config carries the canonical seed list inside the
    ``_phase4_arm`` block so the validator can confirm parity.
    """
    base = _strip_runtime_paths(reference_cfg)
    augmented_csv, syn_only_csv = _resolve_synthetic_inputs(protocol_packet)
    real_input = _resolve_real_input(reference_cfg)
    ref_total = int(reference_cfg.get("total_timesteps", 0))
    pretrain_total = ref_total  # mirror reference budget for pretrain
    finetune_total = ref_total  # mirror reference budget for finetune
    arm_d_total = pretrain_total + finetune_total
    arm_b_total = arm_d_total  # compute-match exactly

    def _phase4_block(arm: ArmSpec, total: int) -> Dict[str, Any]:
        return {
            "scaffold_version": SCHEMA_VERSION,
            "arm_name": arm.name,
            "arm_role": arm.role,
            "promotion_eligible": arm.promotion_eligible,
            "description": arm.description,
            "primary_comparison_partner": "arm_b" if arm.name == "arm_d"
            else ("arm_d" if arm.name == "arm_b" else None),
            "seeds": list(seeds),
            "cost_scenarios": list(COST_SCENARIOS),
            "training_launched": False,
            "execution_status": arm.execution_status,
            "compute_budget_total_timesteps": int(total),
        }

    configs: Dict[str, Dict[str, Any]] = {}

    # ---- Arm A: identical to reference (real-only standard).
    a_cfg = copy.deepcopy(base)
    a_cfg["input_data_file"] = real_input
    a_cfg["total_timesteps"] = ref_total
    a_cfg["_phase4_arm"] = _phase4_block(ARM_SPECS[0], ref_total)
    a_cfg["_protocol_lock"] = _make_protocol_lock(
        ARM_SPECS[0].role, ARM_SPECS[0].promotion_eligible,
        protocol_packet_path, protocol_packet_hash, False,
    )
    configs["arm_a"] = a_cfg

    # ---- Arm B: real-only, compute-matched to Arm D total.
    b_cfg = copy.deepcopy(base)
    b_cfg["input_data_file"] = real_input
    b_cfg["total_timesteps"] = arm_b_total
    b_cfg["_phase4_arm"] = {
        **_phase4_block(ARM_SPECS[1], arm_b_total),
        "compute_match": {
            "matched_to_arm": "arm_d",
            "components": {
                "pretrain_total_timesteps": pretrain_total,
                "finetune_total_timesteps": finetune_total,
            },
            "total": arm_b_total,
            "rule": "Arm B total_timesteps = pretrain_total + finetune_total",
        },
    }
    b_cfg["_protocol_lock"] = _make_protocol_lock(
        ARM_SPECS[1].role, ARM_SPECS[1].promotion_eligible,
        protocol_packet_path, protocol_packet_hash, False,
    )
    configs["arm_b"] = b_cfg

    # ---- Arm C: synthetic-only diagnostic.
    c_cfg = copy.deepcopy(base)
    c_cfg["input_data_file"] = syn_only_csv  # train ONLY on synthetic span
    c_cfg["total_timesteps"] = ref_total
    c_cfg["_phase4_arm"] = {
        **_phase4_block(ARM_SPECS[2], ref_total),
        "validation_input_data_file": real_input,
        "test_input_data_file": real_input,
        "warning": (
            "Synthetic-only training is diagnostic. The resulting policy "
            "MUST NOT be used as Project 3 evidence."
        ),
    }
    c_cfg["_protocol_lock"] = _make_protocol_lock(
        ARM_SPECS[2].role, ARM_SPECS[2].promotion_eligible,
        protocol_packet_path, protocol_packet_hash, False,
    )
    configs["arm_c"] = c_cfg

    # ---- Arm D: synthetic pretrain → real finetune.
    d_cfg = copy.deepcopy(base)
    # Top-level fields stay identical to reference (real input, real budget)
    # so the validator can confirm SAC hyperparameter parity. The actual
    # multi-phase plan lives in the _arm_pretrain / _arm_finetune blocks.
    d_cfg["input_data_file"] = real_input
    d_cfg["total_timesteps"] = ref_total
    d_cfg["_phase4_arm"] = _phase4_block(ARM_SPECS[3], arm_d_total)
    d_cfg["_arm_pretrain"] = {
        "input_data_file": augmented_csv,
        "total_timesteps": pretrain_total,
        "save_pretrained_policy": (
            "./examples/results/project3_phase4_arm_d/pretrain/policy.zip"
        ),
        "comment": (
            "Pretrain on augmented synthetic panel for the train span. "
            "Validation and test windows must remain real."
        ),
    }
    d_cfg["_arm_finetune"] = {
        "input_data_file": real_input,
        "total_timesteps": finetune_total,
        "load_pretrained_policy": (
            "./examples/results/project3_phase4_arm_d/pretrain/policy.zip"
        ),
        "save_model": (
            "./examples/results/project3_phase4_arm_d/finetune/policy.zip"
        ),
    }
    d_cfg["_protocol_lock"] = _make_protocol_lock(
        ARM_SPECS[3].role, ARM_SPECS[3].promotion_eligible,
        protocol_packet_path, protocol_packet_hash, True,
    )
    configs["arm_d"] = d_cfg

    return configs


# ---------------------------------------------------------------------------
class DryRunValidator:
    """Fail-closed validator for the four Phase 4 arms.

    Every check appends to ``self.errors``; ``raise_if_invalid`` raises
    ``RuntimeError`` if the list is non-empty.
    """

    def __init__(
        self,
        reference_cfg: Dict[str, Any],
        protocol_packet: Dict[str, Any],
        configs: Dict[str, Dict[str, Any]],
    ) -> None:
        self.reference_cfg = reference_cfg
        self.protocol_packet = protocol_packet
        self.configs = configs
        self.errors: List[str] = []
        self.warnings: List[str] = []

    # ---- individual checks -------------------------------------------------
    def check_protocol_packet(self) -> None:
        pkt = self.protocol_packet
        if not pkt.get("project3_valid_for_training"):
            self.errors.append(
                "protocol packet project3_valid_for_training is not true"
            )
        status = pkt.get("stage_b_status")
        if status not in ("PENDING_APPROVAL", "APPROVED"):
            self.errors.append(
                f"protocol packet stage_b_status='{status}' not in "
                "{PENDING_APPROVAL, APPROVED}"
            )
        # Heldout boundary metadata.
        wnd = pkt.get("windows", {})
        if str(wnd.get("heldout_boundary", ""))[:10] != PROJECT3_HELDOUT_START:
            self.errors.append(
                f"protocol packet heldout_boundary != {PROJECT3_HELDOUT_START}"
            )

    def check_locks_present(self) -> None:
        for name, cfg in self.configs.items():
            lock = cfg.get("_protocol_lock")
            if not isinstance(lock, dict):
                self.errors.append(f"{name}: missing _protocol_lock block")
                continue
            if not lock.get("_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"):
                self.errors.append(
                    f"{name}: _protocol_lock not asserting Stage B block"
                )

    def check_seed_sweep_blocked(self) -> None:
        """Mirror tools/seed_sweep.py::_locked_protocol_reason exactly."""
        for name, cfg in self.configs.items():
            lock = cfg.get("_protocol_lock")
            if not isinstance(lock, dict):
                self.errors.append(f"{name}: would not be blocked (no lock)")
                continue
            if not lock.get("_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"):
                self.errors.append(
                    f"{name}: would not be blocked by seed_sweep "
                    "(Stage B flag false)"
                )

    def check_required_equal_fields(self) -> None:
        ref = self.reference_cfg
        for name, cfg in self.configs.items():
            for k in REQUIRED_EQUAL_FIELDS:
                if k not in ref:
                    continue
                if cfg.get(k) != ref[k]:
                    self.errors.append(
                        f"{name}.{k}={cfg.get(k)!r} differs from "
                        f"reference {ref[k]!r} (must be identical)"
                    )

    def check_arm_d_sac_hyperparams_unchanged(self) -> None:
        """Arm D specifically must not move SAC hyperparameters."""
        ref = self.reference_cfg
        d = self.configs.get("arm_d", {})
        sac_keys = (
            "learning_rate", "buffer_size", "learning_starts", "batch_size",
            "tau", "gamma", "train_freq", "gradient_steps", "ent_coef",
            "use_sde", "agent_plugin",
        )
        for k in sac_keys:
            if k in ref and d.get(k) != ref[k]:
                self.errors.append(
                    f"arm_d changes SAC hyperparameter {k}: "
                    f"{d.get(k)!r} != reference {ref[k]!r}"
                )

    def check_unexpected_field_diffs(self) -> None:
        """No arm may differ from the reference outside the allow-list."""
        ref = self.reference_cfg
        ref_keys = set(ref.keys())
        for name, cfg in self.configs.items():
            cfg_keys = set(cfg.keys())
            for k in cfg_keys | ref_keys:
                if k in REQUIRED_EQUAL_FIELDS or k in ALLOWED_DIFFERENT_FIELDS:
                    continue
                if cfg.get(k) != ref.get(k):
                    self.errors.append(
                        f"{name}.{k} differs from reference and is not in "
                        "the allowed-difference list"
                    )

    def check_arm_c_not_promotion_eligible(self) -> None:
        c_lock = self.configs.get("arm_c", {}).get("_protocol_lock", {})
        if c_lock.get("promotion_eligible"):
            self.errors.append(
                "arm_c is marked promotion_eligible (must always be False)"
            )
        if not c_lock.get("diagnostic_only"):
            self.errors.append("arm_c lock missing diagnostic_only=true")
        c_arm = self.configs.get("arm_c", {}).get("_phase4_arm", {})
        if c_arm.get("promotion_eligible"):
            self.errors.append(
                "arm_c._phase4_arm.promotion_eligible must be false"
            )

    def check_validation_test_real_only(self) -> None:
        """No arm may declare a synthetic CSV for validation/test."""
        for name, cfg in self.configs.items():
            arm_block = cfg.get("_phase4_arm", {}) or {}
            for key in ("validation_input_data_file", "test_input_data_file"):
                p = arm_block.get(key)
                if p and _looks_like_synthetic(p):
                    self.errors.append(
                        f"{name}._phase4_arm.{key} points to synthetic data: {p}"
                    )

    def check_no_stage_c_references(self) -> None:
        for name, cfg in self.configs.items():
            for k in ("input_data_file",):
                p = str(cfg.get(k, ""))
                if "2025" in p and "_2025-01" in p:
                    self.errors.append(
                        f"{name}.{k} references Stage C path: {p}"
                    )
            for sub in ("_arm_pretrain", "_arm_finetune"):
                blob = cfg.get(sub) or {}
                p = str(blob.get("input_data_file", ""))
                if "2025" in p and "_2025-01" in p:
                    self.errors.append(
                        f"{name}.{sub}.input_data_file references "
                        f"Stage C path: {p}"
                    )

    def check_arm_d_pretrain_uses_synthetic(self) -> None:
        d = self.configs.get("arm_d", {})
        pre = d.get("_arm_pretrain", {})
        if not _looks_like_synthetic(pre.get("input_data_file", "")):
            self.errors.append(
                "arm_d._arm_pretrain.input_data_file must point to the "
                "synthetic-augmented panel from the protocol packet"
            )
        fine = d.get("_arm_finetune", {})
        if _looks_like_synthetic(fine.get("input_data_file", "")):
            self.errors.append(
                "arm_d._arm_finetune.input_data_file must be the real "
                "panel, not synthetic"
            )

    def check_arm_b_compute_match(self) -> None:
        b = self.configs.get("arm_b", {})
        d = self.configs.get("arm_d", {})
        cm = b.get("_phase4_arm", {}).get("compute_match")
        if not cm:
            self.errors.append("arm_b missing compute_match block")
            return
        comp = cm.get("components", {})
        d_pre = d.get("_arm_pretrain", {}).get("total_timesteps")
        d_fine = d.get("_arm_finetune", {}).get("total_timesteps")
        if comp.get("pretrain_total_timesteps") != d_pre:
            self.errors.append(
                "arm_b.compute_match.pretrain mismatch with arm_d._arm_pretrain"
            )
        if comp.get("finetune_total_timesteps") != d_fine:
            self.errors.append(
                "arm_b.compute_match.finetune mismatch with arm_d._arm_finetune"
            )
        expected_total = (d_pre or 0) + (d_fine or 0)
        if int(b.get("total_timesteps", 0)) != expected_total:
            self.errors.append(
                f"arm_b.total_timesteps={b.get('total_timesteps')} != "
                f"pretrain+finetune={expected_total}"
            )

    # ---- driver ------------------------------------------------------------
    def run_all(self) -> None:
        self.check_protocol_packet()
        self.check_locks_present()
        self.check_seed_sweep_blocked()
        self.check_required_equal_fields()
        self.check_arm_d_sac_hyperparams_unchanged()
        self.check_unexpected_field_diffs()
        self.check_arm_c_not_promotion_eligible()
        self.check_validation_test_real_only()
        self.check_no_stage_c_references()
        self.check_arm_d_pretrain_uses_synthetic()
        self.check_arm_b_compute_match()

    def raise_if_invalid(self) -> None:
        if self.errors:
            msg = "Phase 4 dry-run validator FAILED:\n  - " + "\n  - ".join(
                self.errors
            )
            raise RuntimeError(msg)


# ---------------------------------------------------------------------------
def build_manifest(
    *,
    reference_config_path: str,
    reference_config_hash: str,
    protocol_packet_path: str,
    protocol_packet_hash: str,
    seeds: List[int],
    arm_paths: Dict[str, str],
    protocol_packet: Dict[str, Any],
) -> Dict[str, Any]:
    arms_meta = []
    for spec in ARM_SPECS:
        arms_meta.append({
            "name": spec.name,
            "role": spec.role,
            "promotion_eligible": spec.promotion_eligible,
            "description": spec.description,
            "config_path": arm_paths.get(spec.name),
            "execution_status": spec.execution_status,
            "multi_phase": spec.multi_phase,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utcnow_iso(),
        "stage_b_status_required": "PENDING_APPROVAL or APPROVED",
        "project3_heldout_start": PROJECT3_HELDOUT_START,
        "stage_c_access": "DENIED — Stage C must not be touched",
        "reference_config_path": reference_config_path,
        "reference_config_hash": reference_config_hash,
        "protocol_packet_path": protocol_packet_path,
        "protocol_packet_hash": protocol_packet_hash,
        "protocol_packet_stage_b_status": protocol_packet.get("stage_b_status"),
        "protocol_packet_family_id": protocol_packet.get(
            "generator", {}).get("family_id"),
        "protocol_packet_family_revision": protocol_packet.get(
            "generator", {}).get("family_revision"),
        "arms": arms_meta,
        "primary_comparison": {
            "treatment": "arm_d",
            "control": "arm_b",
            "rule": (
                "Arm D promoted only if mean real-validation composite_score "
                "exceeds Arm B by >= 1 SE across replicate seeds AND "
                "lift sign positive in >=2/3 seeds."
            ),
        },
        "promotion_eligible_arms": [s.name for s in ARM_SPECS if s.promotion_eligible],
        "non_promotion_eligible_arms": [s.name for s in ARM_SPECS if not s.promotion_eligible],
        "required_equal_fields": list(REQUIRED_EQUAL_FIELDS),
        "allowed_different_fields": list(ALLOWED_DIFFERENT_FIELDS),
        "seeds": list(seeds),
        "cost_scenarios": list(COST_SCENARIOS),
        "dry_run_only": True,
        "training_launched": False,
        "multi_phase_runner_implemented": False,
        "execution_status_arm_d": "TEMPLATE_ONLY_MULTI_PHASE_RUNNER_NOT_IMPLEMENTED",
    }


def render_manifest_md(manifest: Dict[str, Any]) -> str:
    arms_rows = "\n".join(
        f"| `{a['name']}` | `{a['role']}` | "
        f"{'✅' if a['promotion_eligible'] else '❌'} | "
        f"`{a['execution_status']}` |"
        for a in manifest["arms"]
    )
    return (
        "# Project 3 Phase 4 Compare Manifest (dry-run)\n\n"
        "**Status: PENDING_APPROVAL — no training launched.**\n\n"
        f"- schema_version: `{manifest['schema_version']}`\n"
        f"- generated_at: `{manifest['generated_at']}`\n"
        f"- project3_heldout_start: `{manifest['project3_heldout_start']}`\n"
        f"- stage_c_access: **{manifest['stage_c_access']}**\n"
        f"- reference_config: `{manifest['reference_config_path']}`\n"
        f"  - sha256: `{manifest['reference_config_hash']}`\n"
        f"- protocol_packet: `{manifest['protocol_packet_path']}`\n"
        f"  - sha256: `{manifest['protocol_packet_hash']}`\n"
        f"  - family: `{manifest['protocol_packet_family_id']}`"
        f" / `{manifest['protocol_packet_family_revision']}`\n"
        f"  - stage_b_status: `{manifest['protocol_packet_stage_b_status']}`\n\n"
        "## Arms\n\n"
        "| Name | Role | Promotion eligible | Execution status |\n"
        "|---|---|---|---|\n"
        f"{arms_rows}\n\n"
        "## Primary comparison\n\n"
        f"- treatment: **{manifest['primary_comparison']['treatment']}**\n"
        f"- control: **{manifest['primary_comparison']['control']}**\n"
        f"- rule: {manifest['primary_comparison']['rule']}\n\n"
        f"- seeds: `{manifest['seeds']}`\n"
        f"- cost_scenarios: `{manifest['cost_scenarios']}`\n"
        f"- dry_run_only: `{manifest['dry_run_only']}`\n"
        f"- training_launched: `{manifest['training_launched']}`\n"
        f"- multi_phase_runner_implemented: "
        f"`{manifest['multi_phase_runner_implemented']}`\n"
    )


# ---------------------------------------------------------------------------
def materialize(
    *,
    reference_config_path: str,
    protocol_packet_path: str,
    out_dir: str,
    seeds: List[int],
    validate_only: bool = False,
) -> Dict[str, Any]:
    with open(reference_config_path, "r", encoding="utf-8") as fh:
        reference_cfg = json.load(fh)
    with open(protocol_packet_path, "r", encoding="utf-8") as fh:
        protocol_packet = json.load(fh)

    ref_hash = _sha256(reference_config_path)
    pkt_hash = _sha256(protocol_packet_path)

    configs = build_arm_configs(
        reference_cfg=reference_cfg,
        protocol_packet=protocol_packet,
        seeds=seeds,
        protocol_packet_path=protocol_packet_path,
        protocol_packet_hash=pkt_hash,
        reference_config_path=reference_config_path,
    )

    validator = DryRunValidator(reference_cfg, protocol_packet, configs)
    validator.run_all()
    validator.raise_if_invalid()

    if validate_only:
        return {"validated": True, "errors": [], "warnings": validator.warnings}

    os.makedirs(out_dir, exist_ok=True)
    arm_paths: Dict[str, str] = {}
    for spec in ARM_SPECS:
        path = os.path.join(out_dir, spec.config_filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(configs[spec.name], fh, indent=2, sort_keys=True)
        arm_paths[spec.name] = path

    manifest = build_manifest(
        reference_config_path=reference_config_path,
        reference_config_hash=ref_hash,
        protocol_packet_path=protocol_packet_path,
        protocol_packet_hash=pkt_hash,
        seeds=seeds,
        arm_paths=arm_paths,
        protocol_packet=protocol_packet,
    )
    manifest_json = os.path.join(out_dir, "phase4_compare_manifest.json")
    manifest_md = os.path.join(out_dir, "phase4_compare_manifest.md")
    with open(manifest_json, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    with open(manifest_md, "w", encoding="utf-8") as fh:
        fh.write(render_manifest_md(manifest))

    return {
        "validated": True,
        "out_dir": out_dir,
        "arm_paths": arm_paths,
        "manifest_json": manifest_json,
        "manifest_md": manifest_md,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
def _parse_seeds(s: str) -> List[int]:
    return [int(tok) for tok in s.split(",") if tok.strip() != ""]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Project 3 Phase 4 dry-run protocol comparator. Emits four "
            "locked arm config templates (A/B/C/D) plus a manifest. "
            "Does not launch training."
        ),
    )
    ap.add_argument("--reference-config", required=True)
    ap.add_argument("--protocol-packet", required=True)
    ap.add_argument(
        "--out-dir",
        default="examples/config/project3_phase4_ethusdt_4h_sac_arms",
    )
    ap.add_argument("--seeds", default="0,1,2",
                    help="Comma-separated replicate seeds")
    ap.add_argument(
        "--dry-run-validate-protocol", action="store_true",
        help="Validate inputs and arm parity without writing any files.",
    )
    args = ap.parse_args(argv)

    try:
        result = materialize(
            reference_config_path=args.reference_config,
            protocol_packet_path=args.protocol_packet,
            out_dir=args.out_dir,
            seeds=_parse_seeds(args.seeds),
            validate_only=bool(args.dry_run_validate_protocol),
        )
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2),
              file=sys.stderr)
        return 2

    if args.dry_run_validate_protocol:
        print(json.dumps(
            {"ok": True, "validated": True, "training_launched": False},
            indent=2,
        ))
        return 0

    summary = {
        "ok": True,
        "out_dir": result["out_dir"],
        "arm_paths": result["arm_paths"],
        "manifest_json": result["manifest_json"],
        "manifest_md": result["manifest_md"],
        "training_launched": False,
        "stage_c_touched": False,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
