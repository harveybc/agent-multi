#!/usr/bin/env python3
"""Phase-1 difficulty x phase-1 LR 2x2 factorial mechanics screen (WP4).

Order 2026-08-11 §8: the prior L1 factorial varied PHASE-2 LR with the
phase-1 LR fixed at 1e-4, so it cannot say whether phase-1 learning
rate caused the collapse. This runner executes the drafted contract
``examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v1.json``:

  factors   phase-1 dynamics {normal_realistic,
                              easy_chronological_continuation}
            phase-1 learning rate {1e-4, 3e-5}
  cells     P1N_LR1E4 / P1N_LR3E5 / P1E_LR1E4 / P1E_LR3E5
  seeds     101 / 202 / 303 / 404, one seed per physical GPU; ALL FOUR
            cells of a seed run SEQUENTIALLY on that GPU (hardware
            paired within seed) in the contract's cyclic-Latin-square
            cell order (no order/thermal confounding);
  anchors   every cell starts from the exact per-seed hash-bound
            anchor, NEVER a preceding cell's terminal.

Everything else is held fixed at the ladder identity: phase-2
normal_realistic at LR 3e-5 and threshold 0.1, the l1_trained_epoch_v4
handoff, entropy fixed 0.2, replay+optimizer reset at the boundary
(structural: artifact reload), the frozen L1 v3 cost/protection
contract, the ladder's nested NON-TEST split union, sealed 2025
untouched. The held-fixed recipe is bound from the SAME ladder
contract the D0-D4 diagnostic proved (``m0_l1_mechanism_ladder_v1``);
the v3 cost block is verified verbatim against the frozen system
manifest before any model construction.

NESTED EVIDENCE ROLES (finding 224). The EXECUTING pipeline consumes
the typed nested split contract directly: the materialized config
carries ``nested_split_contract`` (exact path, sha recorded AND
verified before model construction), ``nested_split_mode=l1`` and
``selection_metric=paired_generalization_weekly_v1`` — the legacy
lexicographic/explicit-window branch is structurally unreachable (the
runner asserts it and the pipeline itself refuses a nested contract
with any other metric). Roles in execution: ``fit_train`` (11,509
scored rows) for fitting, ``train_monitor`` (2,190) as the in-sample
member, ``inner_validation`` (2,190) for selection,
``outer_validation`` (2,196) as final truth ONLY; 256 declared context
rows per evaluation role initialize causal state only (forced hold,
excluded from scores — pipeline_plugins._nested_splits is the one
implementation). ``sealed_test`` 2025 stays SEALED: no CSV path, hash,
row load or model evaluation may be emitted for it. Every cell record
binds the nested contract sha, the split-manifest sha, per-role CSV
shas, scored/context counts and exact score dates; wrong role path,
wrong count, wrong sha, outer-used-as-inner, a missing context
declaration, context counted in score, paired-metric drift or any
sealed-test materialization refuses BEFORE training.

REPLICA GATE (finding 225). ``--screen-verdict`` takes a MANDATORY
typed replica-proof file (``--replica-proof``, produced by
tools/p1lr_collect.py): exactly 16 load proofs each bound to
(experiment_identity, contract_sha256, seed, cell,
terminal_relative_path, terminal_model_sha256, loads=true). The
``replica_terminal_loads`` gate is a boolean derived from that proof —
never explanatory text — and the verdict also revalidates the
finding-221 per-checkpoint handoff facts and the nested split identity
of every record at aggregation time.

DECISION MODE (finding 226). ``--mode decision`` runs the document-38
decision path under a DISTINCT content-addressed identity (profile
``p1lr_decision_run``) and a distinct output root; every decision cell
starts from the ORIGINAL per-seed anchor, never a screen terminal.
Decision contract: same 2x2 factors/seeds, phase-2 LR 3e-5, per-cell
ceiling 2,000 pass-equivalent checkpoints (4 phase-1 + 1,996 phase-2),
paired train-monitor/inner-validation stopping with patience 60 and no
stopping conclusion before checkpoint 40, immutable best-checkpoint
restoration, then ONE final outer-validation evaluation after
selection; sealed 2025 inaccessible. Dispatch requires the corrected
screen verdict (``--screen-gate``) with a passing replica gate;
implementing this mode authorizes nothing by itself.
``--decision-verdict`` aggregates the 16 decision records and emits
the document-38 outcomes with per-seed paired main effects
(difficulty, P1 LR) plus interaction and per-cell raw weekly metrics,
all with units and horizons.

INACTIVE CELLS ARE MEASURED OUTCOMES (finding 232). A treatment that
produces a policy which never becomes activity-eligible has no best
checkpoint — that is a RESULT, not a harness exception. Such a cell
publishes ONE immutable record with ``activity_status=inactive``,
``promotion_eligible=false``, the typed termination cause and the same
custody as an active cell (terminal path + sha256 + policy-tensor
sha256 + load proof + experiment/cell identity, contract sha, nested
contract sha, anchor sha), and the seed runner CONTINUES to its next
cell (batch outcome ``SEED_COMPLETE_WITH_INACTIVE``, exit 0 — the
inactivity is named, the remaining work is never destroyed). In
decision mode an inactive cell still receives exactly ONE final outer
evaluation, of its TERMINAL artifact, as DIAGNOSTIC TRUTH ONLY;
``assert_outer_eval_artifact_role`` and
``assert_terminal_never_labeled_best`` refuse any payload or record
that relabels a terminal artifact ``best_checkpoint``. Aggregation
(both verdicts) types the 16-cell picture as ``FULL_ACTIVITY``,
``PARTIAL_ACTIVITY_SURVIVAL`` (naming exactly which cells are
inactive) or ``TOTAL_ACTIVITY_COLLAPSE``; an inactive cell NEVER
contributes an imputed zero or sentinel utility, and a paired effect
that therefore cannot be computed is typed unavailable with the exact
reason.

This is a MECHANICS SCREEN, not a performance result: one
pass-equivalent per cell (1 phase-1 epoch + 1 phase-2 epoch at 20000
timesteps). ``--screen-verdict`` aggregates the 16 records (refusing
on fewer, listing the missing), evaluates the contract's
``mechanics_screen.requires`` gates and emits the typed outcome:
``PHASE1_LR_REGION_COLLAPSED`` when every treatment combination
collapses at every seed (selected trained-checkpoint handoff_viability
in {CONSTANT_POLICY, BELOW_NORMAL_THRESHOLD}), else
``SCREEN_VIABLE_REGION`` with the exact viable cells. No performance
claims are made or implied.

Evidence contracts carried by every cell record:
  finding 223  non-empty terminal_model_path + terminal_model_sha256,
               rehashed and load-proven (policy tensor digest) — a
               cell without them is REFUSED, never certified;
  finding 221  the pipeline's typed handoff-viability evidence bound
               from the selected phase-1 checkpoint plus a
               per-checkpoint summary — refused when absent;
  WP13/198     assigned vs bound vs observed CUDA facts, the
               gpu_readiness_probe launch gate (assigned UUID must be
               visible BEFORE any framework import) and its dispatch
               GPU binding dict.

Runtime discipline reuses the proven ladder machinery: fail-closed
contract refusals before model construction, content-addressed attempt
directories, atomic (fsync+replace) records, flock-backed per-cell
exclusive claims, complete-record reuse (ALREADY_COMPLETE) and
refuse-not-overwrite on invalid existing records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _nested_splits  # noqa: E402
from pipeline_plugins import _paired_generalization as _paired  # noqa: E402
from pipeline_plugins import _system_config as sysid  # noqa: E402
from tools import gpu_readiness_probe as gpu_probe  # noqa: E402
from tools import m0_l1_mechanism_ladder as ladder  # noqa: E402
from tools.l1_factorial_screen import (  # noqa: E402
    _terminal_tensor_sha,
    atomic_write_json,
)
from tools.l1_fleet_launcher import (  # noqa: E402
    ExclusiveClaim,
    _atomic_json,
    _pid_start_identity,
    visible_gpu_uuids,
)

CONTRACT_PATH = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                 "p1_difficulty_lr_factorial_v1.json")
DECISION_EXECUTION_PROFILE_PATH = (
    REPO / "examples/config/phase_3_eth_sac_dynamics/"
    "p1lr_decision_execution_profile_v1.json")
CONTRACT_SCHEMA = "agent_multi.p1_difficulty_lr_factorial.v1"
DECISION_EXECUTION_PROFILE_SCHEMA = \
    "agent_multi.p1lr_decision_execution_profile.v1"
RECORD_SCHEMA = "agent_multi.p1_difficulty_lr_cell_record.v1"
VERDICT_SCHEMA = "agent_multi.p1_difficulty_lr_screen_verdict.v1"
DECISION_VERDICT_SCHEMA = \
    "agent_multi.p1_difficulty_lr_decision_verdict.v1"
REPLICA_PROOF_SCHEMA = "agent_multi.p1lr_replica_proof.v1"
PREFLIGHT_SCHEMA = "agent_multi.p1_difficulty_lr_preflight.v1"
HEARTBEAT_SCHEMA = "agent_multi.p1_difficulty_lr_heartbeat.v1"
HEARTBEAT_INTERVAL_S = 60

SEEDS = (101, 202, 303, 404)
CELLS = ("P1N_LR1E4", "P1N_LR3E5", "P1E_LR1E4", "P1E_LR3E5")
DYNAMICS_LEVELS = ("normal_realistic", "easy_chronological_continuation")
LR_LEVELS = (0.0001, 3e-05)

# Execution modes (finding 226): distinct content-addressed identity
# profiles AND distinct output roots per mode. The screen stays the
# default; the decision path never reuses a screen identity/record.
MODES = ("screen", "decision")
MODE_PROFILES = {"screen": "p1lr_mechanics_screen",
                 "decision": "p1lr_decision_run"}

# The exact executable selection metric (finding 224). The legacy
# lexicographic branch is UNREACHABLE for this factorial: the loader
# refuses a contract with any other metric, the materializer asserts
# the executable config carries exactly this value, and the pipeline's
# nested branch independently refuses a non-paired metric.
NESTED_SELECTION_METRIC = _paired.METRIC_NAME  # paired_generalization_weekly_v1

# Per-role facts every record must bind (finding 224): pinned csv sha,
# scored/context counts and exact score dates, verified against the
# freshly materialized split manifest before model construction.
NESTED_ROLE_FACT_KEYS = ("status", "csv_sha256", "scored_rows",
                         "context_rows", "score_start", "score_end")

# Document-38 decision outcomes (finding 226).
DECISION_OUTCOMES = (
    "PHASE1_LR_MAIN_EFFECT", "PHASE1_DIFFICULTY_MAIN_EFFECT",
    "PHASE1_LR_DIFFICULTY_INTERACTION", "NO_MATERIAL_EFFECT",
    "TOTAL_ACTIVITY_COLLAPSE", "INCONCLUSIVE")

# The ONLY config fields a factor may move (the one-intended-delta
# property test asserts each pairwise cell diff equals exactly the
# union of the differing factors' fields and NOTHING else). The LR
# factor intentionally binds BOTH pipeline knobs: the easy phase-1
# branch reads ``easy_learning_rate`` and the matched normal phase-1
# branch reads ``phase1_learning_rate`` (falling back to
# ``easy_learning_rate``) — see pipeline_plugins/
# rl_pipeline_with_solvency_curriculum.py — so setting both makes the
# level effective in whichever branch the dynamics factor selects.
FACTOR_FIELDS = {
    "phase1_dynamics": ("phase1_mode",),
    "phase1_learning_rate": ("phase1_learning_rate",
                             "easy_learning_rate"),
}

# Typed handoff-viability labels — mirrored from
# pipeline_plugins.rl_pipeline_with_solvency_curriculum (imported
# lazily at runtime; mirrored here so the aggregator never imports the
# training framework). A cross-check test asserts equality.
HANDOFF_VIABILITY_VALUES = (
    "VIABLE", "BELOW_NORMAL_THRESHOLD", "CONSTANT_POLICY", "NO_TRADE",
    "UNAVAILABLE")
COLLAPSE_LABELS = ("CONSTANT_POLICY", "BELOW_NORMAL_THRESHOLD")

# Finding 232: an inactive cell is a MEASURED OUTCOME, not a harness
# exception and not a successful checkpoint. Every cell record carries
# a typed activity status; ``inactive`` is never promotable, never
# relabeled ``best_checkpoint`` and never imputed a performance value.
ACTIVITY_STATUSES = ("active", "inactive")
INACTIVE_CAUSES = (
    # training ended before any activity-eligible checkpoint existed
    "no_activity_eligible_checkpoint",
    # a selected checkpoint existed but closed zero trades on the one
    # final outer evaluation (aggregation-time classification only)
    "zero_trades_on_outer_validation",
)
# Aggregation-time activity classification of the 16-record factorial.
ACTIVITY_CLASSIFICATIONS = ("FULL_ACTIVITY", "PARTIAL_ACTIVITY_SURVIVAL",
                            "TOTAL_ACTIVITY_COLLAPSE")
# The artifact a single final outer evaluation may load. ``terminal`` is
# DIAGNOSTIC TRUTH ONLY — relabeling it best_checkpoint is refused.
OUTER_ARTIFACT_ROLES = ("best_checkpoint", "terminal")
TERMINAL_LOAD_PROOF_SCHEMA = "agent_multi.p1lr_terminal_load_proof.v1"
# Stable trace tags per nested evaluation role (the outer default keeps
# the historical "outer" tag so P1LR traces stay byte-comparable).
_ROLE_TAG = {"outer_validation": "outer",
             "inner_validation": "inner",
             "train_monitor": "train_monitor"}

# Exit classes follow the fleet launcher/systemd contract.
EXIT_CLASS = {
    "SEED_COMPLETE": 0,
    # Finding 232: inactivity is a measured result, so the batch still
    # completes (exit 0 — nothing to restart) while the outcome NAMES
    # that at least one cell finished inactive.
    "SEED_COMPLETE_WITH_INACTIVE": 0,
    "ALREADY_COMPLETE": 0,
    "ALREADY_COMPLETE_WITH_INACTIVE": 0,
    "SCREEN_VIABLE_REGION": 0,
    "PHASE1_LR_REGION_COLLAPSED": 0,
    "PHASE1_LR_MAIN_EFFECT": 0,
    "PHASE1_DIFFICULTY_MAIN_EFFECT": 0,
    "PHASE1_LR_DIFFICULTY_INTERACTION": 0,
    "NO_MATERIAL_EFFECT": 0,
    "TOTAL_ACTIVITY_COLLAPSE": 0,
    "PREFLIGHT_PASS": 0,
    "ALREADY_RUNNING": 3,
    "REFUSED_WRONG_HOST": 4,
    "REFUSED_GPU_UNBOUND": 4,
    "REFUSED_BAD_CONTRACT": 4,
    "REFUSED_ANCHOR_UNVERIFIED": 4,
    "REFUSED_DECISION_UNGATED": 4,
    "SCREEN_REFUSED": 4,
    "DECISION_REFUSED": 4,
    "INCONCLUSIVE": 4,
    "PREFLIGHT_REFUSED": 4,
    "SEED_FAILED": 1,
}


def _sha_file(path: Path) -> str:
    return sysid.sha_file(path)


# ---------------------------------------------------------------------------
# contract loading — typed refusals BEFORE any model construction
# ---------------------------------------------------------------------------

def load_contract(path: Path = CONTRACT_PATH) -> dict:
    contract = json.loads(Path(path).read_text())
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(
            f"unknown p1-difficulty-LR contract schema "
            f"{contract.get('schema')!r}")

    factors = contract.get("factors") or {}
    if tuple(factors.get("phase1_dynamics") or ()) != DYNAMICS_LEVELS:
        raise ValueError(
            f"factors.phase1_dynamics must be exactly {DYNAMICS_LEVELS}")
    lr_levels = tuple(float(v) for v in
                      (factors.get("phase1_learning_rate") or ()))
    if lr_levels != LR_LEVELS:
        raise ValueError(
            f"factors.phase1_learning_rate must be exactly {LR_LEVELS}")

    cells = contract.get("cells") or {}
    if sorted(cells) != sorted(CELLS):
        raise ValueError(f"cells must be exactly {sorted(CELLS)}; got "
                         f"{sorted(cells)}")
    combos = set()
    for name, spec in cells.items():
        dynamics = spec.get("phase1_dynamics")
        if dynamics not in DYNAMICS_LEVELS:
            raise ValueError(f"cell {name}: unknown phase1_dynamics "
                             f"{dynamics!r}")
        lr = float(spec.get("phase1_learning_rate", float("nan")))
        if lr not in LR_LEVELS:
            raise ValueError(f"cell {name}: phase1_learning_rate must "
                             f"be one of {LR_LEVELS}")
        combos.add((dynamics, lr))
    if len(combos) != 4:
        raise ValueError(
            "the four cells must cover the full 2x2 factor cross "
            "exactly once each — a repeated combination is not a "
            "factorial")

    if [int(s) for s in contract.get("seeds") or []] != list(SEEDS):
        raise ValueError(f"seeds must be exactly {list(SEEDS)}")
    anchors = contract.get("anchors") or {}
    assignments = contract.get("assignments") or {}
    for seed in SEEDS:
        anchor = anchors.get(str(seed)) or {}
        if not anchor.get("path") or not isinstance(
                anchor.get("path"), str):
            raise ValueError(f"anchors.{seed}.path missing — an "
                             "unpinned anchor cannot anchor a screen")
        sha = anchor.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(
                f"anchors.{seed}.sha256 is not a sha256 hex digest")
        assignment = assignments.get(str(seed)) or {}
        if not assignment.get("hostname") or not assignment.get(
                "gpu_uuid"):
            raise ValueError(
                f"seed {seed}: worker assignment must pin hostname AND "
                "GPU UUID")

    order = {key: value
             for key, value in (contract.get("cell_order") or {}).items()
             if not key.startswith("$")}
    if sorted(order) != sorted(str(s) for s in SEEDS):
        raise ValueError("cell_order must declare exactly one row per "
                         "seed")
    rows = {}
    for seed in SEEDS:
        row = list(order[str(seed)])
        if sorted(row) != sorted(CELLS):
            raise ValueError(
                f"cell_order.{seed} is not a permutation of the four "
                f"cells: {row}")
        rows[seed] = row
    base_row = rows[SEEDS[0]]
    for position in range(4):
        column = {rows[seed][position] for seed in SEEDS}
        if len(column) != 4:
            raise ValueError(
                f"cell_order is not a Latin square: within-seed "
                f"position {position} repeats a cell across seeds")
    for offset, seed in enumerate(SEEDS):
        expected = [base_row[(position + offset) % 4]
                    for position in range(4)]
        if rows[seed] != expected:
            raise ValueError(
                f"cell_order.{seed} is not the declared CYCLIC Latin "
                f"square (rotation {offset} of {base_row}); got "
                f"{rows[seed]}")

    held = contract.get("held_fixed") or {}
    if float(held.get("phase2_learning_rate", float("nan"))) != 3e-05:
        raise ValueError("held_fixed.phase2_learning_rate must be 3e-05 "
                         "(the active D0 range point, order §8)")
    if held.get("phase2_dynamics") != "normal_realistic":
        raise ValueError(
            "held_fixed.phase2_dynamics must be normal_realistic")
    if float(held.get("phase2_action_threshold",
                      float("nan"))) != 0.1:
        raise ValueError("held_fixed.phase2_action_threshold must be "
                         "0.1")
    if held.get("phase1_handoff_semantics") != "l1_trained_epoch_v4":
        raise ValueError("held_fixed.phase1_handoff_semantics must be "
                         "l1_trained_epoch_v4")
    entropy = held.get("entropy") or {}
    if entropy.get("mode") != "fixed" or float(
            entropy.get("value", float("nan"))) != 0.2:
        raise ValueError("held_fixed.entropy must be fixed at 0.2")

    knobs = (contract.get("mechanics_screen") or {}).get(
        "budget_knobs") or {}
    for key in ("epoch_timesteps", "phase1_epochs", "phase2_epochs"):
        value = knobs.get(key)
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"mechanics_screen.budget_knobs.{key} must be a "
                "positive integer — the screen budget must be "
                "executable, not prose")
    if knobs["phase1_epochs"] != 1 or knobs["phase2_epochs"] != 1:
        raise ValueError(
            "the mechanics screen is ONE pass-equivalent per cell "
            "(1 phase-1 epoch + 1 phase-2 epoch); a larger budget is "
            "the decision run and needs its own order")

    if "PHASE1_LR_REGION_COLLAPSED" not in (
            contract.get("typed_outcomes") or []):
        raise ValueError("typed_outcomes must include "
                         "PHASE1_LR_REGION_COLLAPSED")
    if not contract.get("output_root"):
        raise ValueError("output_root missing")

    # Finding 224: the executable nested-role binding is a CONTRACT
    # fact, not runner prose. Paired-metric drift refuses at load time.
    metric = str(contract.get("selection_metric") or "").strip().lower()
    if metric != NESTED_SELECTION_METRIC:
        raise ValueError(
            f"selection_metric must be {NESTED_SELECTION_METRIC!r} "
            f"(got {metric!r}) — the legacy lexicographic branch is "
            "unreachable for this factorial (finding 224)")
    nested = contract.get("nested_split_contract") or {}
    if not isinstance(nested.get("path"), str) or not nested["path"]:
        raise ValueError("nested_split_contract.path missing — the "
                         "executing pipeline must consume the typed "
                         "nested contract (finding 224)")
    sha = nested.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(
            "nested_split_contract.sha256 is not a sha256 hex digest")
    if nested.get("mode") != "l1":
        raise ValueError("nested_split_contract.mode must be 'l1'")
    if int(nested.get("context_bars", -1)) != 256:
        raise ValueError("nested_split_contract.context_bars must be "
                         "the declared 256 causal context rows")
    role_facts = nested.get("role_facts") or {}
    missing_roles = [role for role in _nested_splits.ROLES
                     if role not in role_facts]
    if missing_roles:
        raise ValueError(f"nested_split_contract.role_facts missing "
                         f"roles: {missing_roles}")
    for role in _nested_splits.ROLES:
        pin = role_facts[role]
        missing_keys = [key for key in NESTED_ROLE_FACT_KEYS
                        if key not in pin]
        if missing_keys:
            raise ValueError(
                f"role_facts.{role} missing {missing_keys} — every "
                "record must bind per-role CSV sha, scored/context "
                "counts and exact score dates (finding 224)")
    if role_facts["sealed_test"].get("status") != "SEALED" or \
            role_facts["sealed_test"].get("csv_sha256") is not None:
        raise ValueError(
            "role_facts.sealed_test must be SEALED with no CSV hash — "
            "sealed 2025 may never be materialized")

    # Finding 226: the decision path must be executable, not prose.
    decision = contract.get("decision_run") or {}
    if not decision.get("output_root"):
        raise ValueError("decision_run.output_root missing — decision "
                         "mode needs its own output root")
    if str(decision["output_root"]) == str(contract["output_root"]):
        raise ValueError("decision_run.output_root must differ from "
                         "the screen output_root (distinct roots per "
                         "mode, finding 226)")
    if float(decision.get("phase2_learning_rate",
                          float("nan"))) != 3e-05:
        raise ValueError("decision_run.phase2_learning_rate must be "
                         "3e-05")
    dknobs = decision.get("budget_knobs") or {}
    for key in ("epoch_timesteps", "phase1_epochs", "phase2_max_epochs"):
        if not isinstance(dknobs.get(key), int) or dknobs[key] < 1:
            raise ValueError(f"decision_run.budget_knobs.{key} must be "
                             "a positive integer")
    ceiling = int(decision.get(
        "max_global_pass_equivalent_checkpoints", 0))
    if ceiling != 2000 or (dknobs["phase1_epochs"]
                           + dknobs["phase2_max_epochs"]) != ceiling:
        raise ValueError(
            "decision budget must total the 2000 pass-equivalent "
            "per-cell ceiling (phase1_epochs + phase2_max_epochs)")
    if int(decision.get("patience", 0)) != 60 or \
            int(decision.get("patience_floor", 0)) != 40:
        raise ValueError("decision_run patience must be 60 with floor "
                         "40 (no stopping conclusion before "
                         "checkpoint 40)")
    if decision.get("best_checkpoint_restoration") is not True:
        raise ValueError("decision_run.best_checkpoint_restoration "
                         "must be true")
    sknobs = decision.get("stopping_knobs") or {}
    if int(sknobs.get("l1_patience", 0)) != 60 or \
            int(sknobs.get("l1_patience_start_epoch", 0)) != 40:
        raise ValueError("decision_run.stopping_knobs must carry "
                         "l1_patience=60, l1_patience_start_epoch=40")
    if tuple(decision.get("decision_outcomes") or ()) != \
            DECISION_OUTCOMES:
        raise ValueError(
            f"decision_run.decision_outcomes must be exactly "
            f"{list(DECISION_OUTCOMES)} (document 38)")

    contract["_contract_sha256"] = _sha_file(Path(path))
    contract["_contract_path"] = str(path)
    return contract


def load_bindings(path: Path | None = None) -> dict:
    """The held-fixed recipe source: the SAME ladder contract the D0-D4
    diagnostic proved (data, splits, nested NON-TEST union, base
    config, plugins, frozen v3 cost manifest reference). Loaded through
    the ladder's own fail-closed loader."""
    return ladder.load_contract(path or ladder.CONTRACT_PATH)


def load_decision_execution_profile(
        path: Path = DECISION_EXECUTION_PROFILE_PATH) -> dict:
    """Load the uniform, content-addressed memory bound for decision mode."""
    profile = json.loads(Path(path).read_text())
    if profile.get("schema") != DECISION_EXECUTION_PROFILE_SCHEMA:
        raise ValueError("unknown P1LR decision execution profile schema")
    replay = profile.get("replay_buffer") or {}
    for key in ("base_config_size", "decision_size", "epoch_timesteps",
                "pass_equivalents_retained"):
        value = replay.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"decision execution profile replay_buffer.{key} must be "
                "a positive integer")
    if replay["base_config_size"] != 200_000:
        raise ValueError("decision execution profile no longer binds the "
                         "200000-transition base config")
    if replay["decision_size"] != 40_000:
        raise ValueError("decision replay buffer must remain the reviewed "
                         "40000-transition uniform bound")
    if replay["epoch_timesteps"] != 20_000 or \
            replay["pass_equivalents_retained"] != 2:
        raise ValueError("decision replay buffer must retain exactly two "
                         "20000-step pass-equivalents")
    if replay.get("uniform_across_seeds") is not True:
        raise ValueError("decision replay capacity must be uniform across "
                         "every seed and host")
    if replay.get("optimize_memory_usage") is not False:
        raise ValueError("optimize_memory_usage must stay explicitly false "
                         "for this reviewed profile")
    profile["_profile_sha256"] = _sha_file(Path(path))
    profile["_profile_path"] = str(Path(path))
    return profile


# ---------------------------------------------------------------------------
# nested-role binding (finding 224) — verified BEFORE model construction
# ---------------------------------------------------------------------------

def nested_contract_path(contract: dict) -> Path:
    return REPO / contract["nested_split_contract"]["path"]


def verify_role_semantics(nested: dict, pins: dict) -> list:
    """Pure semantic guard over the loaded nested contract + the
    factorial's pinned role facts: contiguity/ordering (an outer year
    can never occupy the inner slot), monitor-within-fit, and pinned
    score dates/counts that equal the nested contract's own role
    declarations. Returns exact refusal strings."""
    import pandas as pd

    refusals: list = []
    roles = nested.get("roles") or {}

    def ts(role: str, key: str):
        try:
            return pd.Timestamp(roles[role][key])
        except Exception:
            refusals.append(f"nested role {role}.{key} unreadable")
            return None

    fit_start, fit_end = ts("fit_train", "start"), ts("fit_train", "end")
    mon_start, mon_end = (ts("train_monitor", "start"),
                          ts("train_monitor", "end"))
    inner_start, inner_end = (ts("inner_validation", "start"),
                              ts("inner_validation", "end"))
    outer_start, outer_end = (ts("outer_validation", "start"),
                              ts("outer_validation", "end"))
    sealed_start = ts("sealed_test", "start")
    if refusals:
        return refusals
    if not (fit_start <= mon_start and mon_end <= fit_end):
        refusals.append("train_monitor is not a subset of fit_train")
    if fit_end != inner_start:
        refusals.append("fit_train and inner_validation are not "
                        "contiguous")
    if inner_end != outer_start:
        refusals.append(
            "inner_validation and outer_validation are not contiguous")
    if outer_end != sealed_start:
        refusals.append(
            "outer_validation and sealed_test are not contiguous")
    if not inner_start < outer_start:
        refusals.append(
            "outer_validation does not follow inner_validation — the "
            "outer truth year can never occupy the inner selection "
            "slot (finding 224)")

    expected_rows = nested.get("expected_rows") or {}
    for role in ("fit_train", "train_monitor", "inner_validation",
                 "outer_validation"):
        pin = pins.get(role) or {}
        if pin.get("status") != "MATERIALIZED":
            refusals.append(f"pinned role_facts.{role}.status is not "
                            "MATERIALIZED")
            continue
        if expected_rows.get(role) is not None and \
                int(pin.get("scored_rows", -1)) != \
                int(expected_rows[role]):
            refusals.append(
                f"role {role}: pinned scored_rows "
                f"{pin.get('scored_rows')!r} != nested contract "
                f"expected_rows {expected_rows[role]!r} — scored "
                "counts must exclude context (finding 224)")
        if pd.Timestamp(pin.get("score_start")) != \
                pd.Timestamp(roles[role]["start"]):
            refusals.append(
                f"role {role}: pinned score_start "
                f"{pin.get('score_start')!r} != nested contract role "
                f"start {roles[role]['start']!r} — a swapped role "
                "cannot bind (finding 224)")
    # The inner and outer pins must be distinguishable — binding the
    # outer facts into the inner slot fails here even if dates parse.
    if pins.get("inner_validation", {}).get("csv_sha256") == \
            pins.get("outer_validation", {}).get("csv_sha256"):
        refusals.append("inner_validation and outer_validation pins "
                        "share one CSV sha — roles are not distinct")
    return refusals


def verify_nested_split_binding(contract: dict, bindings: dict) -> dict:
    """Fail-closed nested binding proof, run BEFORE model construction:
    exact contract path present, sha recorded AND verified against both
    the factorial pin and the ladder held-fixed binding, context
    declaration complete (256 bars), role semantics sane. Returns the
    verified binding facts."""
    spec = contract["nested_split_contract"]
    path = nested_contract_path(contract)
    if not path.is_file():
        raise RuntimeError(
            f"nested split contract missing at {path} — wrong role "
            "path; refusing before model construction (finding 224)")
    actual_sha = _sha_file(path)
    if actual_sha != spec["sha256"]:
        raise RuntimeError(
            f"nested split contract sha {actual_sha[:16]}… != the "
            f"factorial pin {spec['sha256'][:16]}… — wrong role sha; "
            "refused (finding 224)")
    ladder_pin = bindings["common"]["nested_split_contract"]
    if spec["path"] != ladder_pin["path"] or \
            actual_sha != ladder_pin["sha256"]:
        raise RuntimeError(
            "factorial nested contract does not equal the ladder "
            "held-fixed nested binding — the held-fixed dates moved")
    # _nested_splits.load_contract refuses a missing context
    # declaration (window_size/scaling_window/max_feature_lookback).
    nested = _nested_splits.load_contract(path)
    context_bars = _nested_splits.compute_context_bars(
        window_size=int(nested["window_size"]),
        scaling_window=int(nested["scaling_window"]),
        max_feature_lookback=int(nested["max_feature_lookback"]))
    if context_bars != int(spec["context_bars"]):
        raise RuntimeError(
            f"derived context_bars {context_bars} != pinned "
            f"{spec['context_bars']} — the context declaration moved")
    semantic_refusals = verify_role_semantics(nested,
                                              spec["role_facts"])
    if semantic_refusals:
        raise RuntimeError("nested role semantics refused: "
                           + "; ".join(semantic_refusals))
    return {
        "path": str(path),
        "contract_relative_path": spec["path"],
        "sha256": actual_sha,
        "mode": spec["mode"],
        "context_bars": context_bars,
        "source_csv": nested["source_csv"],
        "source_sha256": nested["source_sha256"],
        "role_facts_pinned": {role: dict(spec["role_facts"][role])
                              for role in _nested_splits.ROLES},
    }


def verify_role_facts(manifest_roles: dict, pins: dict) -> list:
    """Pure comparison of a freshly materialized split manifest against
    the pinned role facts. Any drift — wrong CSV sha, wrong scored or
    context count (context counted in score), missing context flag,
    moved score dates, or ANY sealed-test materialization — returns
    exact refusal strings."""
    refusals: list = []
    for role in _nested_splits.ROLES:
        pin = pins.get(role) or {}
        got = manifest_roles.get(role) or {}
        if role == "sealed_test":
            if got.get("status") != "SEALED":
                refusals.append(
                    f"sealed_test status {got.get('status')!r} — "
                    "sealed 2025 may never be materialized; no CSV "
                    "path, hash, row load or evaluation may exist")
            for key in ("csv", "csv_sha256", "scored_rows",
                        "context_rows"):
                if got.get(key) is not None:
                    refusals.append(
                        f"sealed_test emits {key}={got.get(key)!r} — "
                        "sealed 2025 must stay state SEALED")
            continue
        if got.get("status") != "MATERIALIZED":
            refusals.append(f"role {role}: status "
                            f"{got.get('status')!r} != MATERIALIZED")
            continue
        for key in ("csv_sha256", "scored_rows", "context_rows",
                    "score_start", "score_end"):
            if got.get(key) != pin.get(key):
                refusals.append(
                    f"role {role}: {key} {got.get(key)!r} != pinned "
                    f"{pin.get(key)!r}")
    return refusals


def materialize_nested_roles(contract: dict, bindings: dict,
                             out_dir: Path) -> dict:
    """Materialize the nested per-role CSVs through the ONE typed
    implementation (mode l1) and verify every derived fact against the
    pinned role facts BEFORE any model construction. Returns the
    verified role facts (csv paths + shas + counts + score dates) and
    the split-manifest binding."""
    binding = verify_nested_split_binding(contract, bindings)
    nested = _nested_splits.load_contract(nested_contract_path(contract))
    split_dir = Path(out_dir) / "nested_splits"
    manifest = _nested_splits.materialize_nested_splits(
        nested, split_dir, mode=contract["nested_split_contract"]["mode"])
    refusals = verify_role_facts(manifest["roles"],
                                 binding["role_facts_pinned"])
    if refusals:
        raise RuntimeError(
            "nested role facts refused before training: "
            + "; ".join(refusals))
    manifest_path = Path(manifest["manifest_path"])
    return {
        "binding": binding,
        "split_dir": str(split_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha_file(manifest_path),
        "roles": {role: dict(entry)
                  for role, entry in manifest["roles"].items()},
    }


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------

def experiment_identity(contract: dict, bindings: dict,
                        sources: dict | None = None,
                        mode: str = "screen") -> str:
    """sha256(p1lr contract sha + held-fixed ladder-contract sha +
    nested split contract sha + paired selection metric + the four
    per-seed anchor shas + code identities + the mode profile)[:16] —
    ONE identity per mode, derived from the CORRECTED executable facts
    (finding 224); reuses no L1 decision, ladder or pre-correction
    identity, and the decision profile never collides with the screen.
    """
    if mode not in MODES:
        raise ValueError(f"unknown execution mode {mode!r}")
    sources = sources or ladder.source_identities()
    payload = {
        "contract": contract["_contract_sha256"],
        "held_fixed_bindings_contract": bindings["_contract_sha256"],
        "nested_split_contract_sha256":
            contract["nested_split_contract"]["sha256"],
        "selection_metric": NESTED_SELECTION_METRIC,
        "anchors": {str(seed): contract["anchors"][str(seed)]["sha256"]
                    for seed in SEEDS},
        "code": {name: {"commit": s["commit"],
                        "dirty_untracked_digest":
                            s["dirty_untracked_digest"]}
                 for name, s in sorted(sources.items())},
        "profile": MODE_PROFILES[mode],
    }
    if mode == "decision":
        execution_profile = load_decision_execution_profile()
        payload["decision_execution_profile_sha256"] = \
            execution_profile["_profile_sha256"]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True).encode()).hexdigest()[:16]


def output_root_for_mode(contract: dict, mode: str) -> Path:
    """Distinct output roots per mode (finding 226)."""
    if mode == "decision":
        return Path(contract["decision_run"]["output_root"]).expanduser()
    return Path(contract["output_root"]).expanduser()


def cell_identity(exp_id: str, seed: int, cell: str,
                  contract: dict) -> str:
    payload = {
        "experiment_identity": exp_id,
        "seed": int(seed),
        "cell": cell,
        "factors": dict(contract["cells"][cell]),
        "anchor_sha256": contract["anchors"][str(seed)]["sha256"],
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def intended_delta_fields(contract: dict, cell_a: str,
                          cell_b: str) -> set:
    """The EXACT set of config fields that may differ between two
    cells of the same seed: the union of FACTOR_FIELDS for every
    factor whose level differs. The property test asserts the
    materialized pairwise diff equals this set."""
    spec_a = contract["cells"][cell_a]
    spec_b = contract["cells"][cell_b]
    fields: set = set()
    for factor, config_fields in FACTOR_FIELDS.items():
        if spec_a[factor] != spec_b[factor]:
            fields.update(config_fields)
    return fields


# ---------------------------------------------------------------------------
# materialization — the ONLY path from contract to a runnable config
# ---------------------------------------------------------------------------

def materialize_cell_config(contract: dict, bindings: dict, seed: int,
                            cell: str, out_dir: Path,
                            mode: str = "screen") -> dict:
    if cell not in CELLS:
        raise ValueError(f"unknown factorial cell {cell!r}")
    if int(seed) not in SEEDS:
        raise ValueError(f"unknown factorial seed {seed!r}")
    if mode not in MODES:
        raise ValueError(f"unknown execution mode {mode!r}")
    common = bindings["common"]
    held = contract["held_fixed"]
    if mode == "decision":
        knobs = contract["decision_run"]["budget_knobs"]
    else:
        knobs = contract["mechanics_screen"]["budget_knobs"]

    base_path = REPO / common["base_config"]["path"]
    actual_base_sha = _sha_file(base_path)
    if actual_base_sha != common["base_config"]["sha256"]:
        raise RuntimeError(
            "base config drifted from the held-fixed ladder binding")

    # Held-fixed cost/protection: EXACTLY the block the ladder applies
    # as the D3_COST_PROTECTION delta, verified verbatim against the
    # LIVE frozen v3 system manifest — an embedded copy that drifted
    # from its source is a lie.
    manifest_path = REPO / bindings["l1_reference"][
        "system_manifest_path"]
    if _sha_file(manifest_path) != bindings["l1_reference"][
            "system_manifest_sha256"]:
        raise RuntimeError(
            "L1 system manifest drifted from the ladder contract")
    manifest = json.loads(manifest_path.read_text())
    v3_costs = manifest["costs"]["config_bindings"]
    if v3_costs != bindings["arms"]["D3_COST_PROTECTION"]["delta"]:
        raise RuntimeError(
            "the ladder D3 cost delta no longer equals the frozen v3 "
            "manifest cost bindings — the held-fixed cost contract is "
            "unbound")
    sysid.validate_normal_contract(v3_costs)
    if float(v3_costs["continuous_action_threshold"]) != float(
            held["phase2_action_threshold"]):
        raise RuntimeError(
            "held_fixed.phase2_action_threshold does not equal the v3 "
            "manifest continuous_action_threshold")
    if float(common["learning_rates"]["normal"]) != float(
            held["phase2_learning_rate"]):
        raise RuntimeError(
            "held_fixed.phase2_learning_rate does not equal the ladder "
            "common normal learning rate — the D0 range point moved")
    if int(knobs["epoch_timesteps"]) != int(
            common["budget"]["epoch_timesteps"]):
        raise RuntimeError(
            f"{mode} epoch_timesteps does not match the ladder pass "
            "budget — the pass-equivalent claim would be false")

    union_facts = ladder.verify_nested_nontest_union(bindings)

    # Finding 224: the nested binding is recorded AND verified before
    # model construction; a legacy-split config can never leave here.
    nested_binding = verify_nested_split_binding(contract, bindings)

    # --- the ladder-proven held-fixed recipe --------------------------
    config = json.loads(base_path.read_text())
    decision_execution_profile = None
    if mode == "decision":
        decision_execution_profile = load_decision_execution_profile()
        replay = decision_execution_profile["replay_buffer"]
        if int(config.get("buffer_size", -1)) != int(
                replay["base_config_size"]):
            raise RuntimeError(
                "base config replay capacity drifted from the decision "
                "execution profile")
        config["buffer_size"] = int(replay["decision_size"])
        config["optimize_memory_usage"] = bool(
            replay["optimize_memory_usage"])
    # No legacy split field survives: the ONLY split authority is the
    # typed nested contract below (finding 224).
    for field in ("train_years", "val_years", "test_years",
                  "train_days", "val_days", "test_days",
                  "train_start", "train_end",
                  "validation_start", "validation_end",
                  "val_start", "val_end", "test_start", "test_end",
                  "split_anchor"):
        config.pop(field, None)
    config["split_contract_note"] = (
        "typed nested split contract governs (mode l1); legacy "
        "year-count and explicit-window fields removed (finding 224)")
    if bool((common["splits"] or {}).get("evaluate_test_split", False)):
        raise RuntimeError("held-fixed contract enables the sealed "
                           "test split — refused")
    config["nested_split_contract"] = nested_binding["path"]
    config["nested_split_mode"] = nested_binding["mode"]
    config["nested_split_dir"] = str(Path(out_dir) / "nested_splits")
    config["input_data_file"] = str(Path(common["data"]["path"]))
    config["env_mode"] = "training"
    config["eval_seed"] = int(seed)
    config["train_seed"] = int(seed)
    config["ga_seed"] = int(seed)
    config["epoch_timesteps"] = int(knobs["epoch_timesteps"])
    config["l1_min_checkpoint_timesteps"] = 1
    config["evaluate_test_split"] = False
    # Evidence plumbing, UNIFORM across every cell (never a factor): a
    # fully-inactive cell is a MEASURED screen outcome and must land a
    # typed record (ladder lesson, observed 2026-08-11).
    config["inactive_terminal_is_typed_result"] = True
    # Finding 224: paired selection ONLY. The pipeline's nested branch
    # independently refuses any other metric, so the legacy
    # lexicographic branch is structurally unreachable.
    config["selection_metric"] = str(contract["selection_metric"])
    config["l1_gap_penalty_beta"] = float(
        contract.get("l1_gap_penalty_beta", 0.25))
    config["selection_min_trades"] = int(common["selection_min_trades"])
    config["quiet_mode"] = True
    config["save_model"] = str(Path(out_dir) / "model.zip")
    config["return_trace_dir"] = str(Path(out_dir) / "return_traces")
    if mode == "decision":
        # Document-38 decision stopping (finding 226): paired
        # train-monitor/inner-validation stopping, patience 60, no
        # stopping conclusion before checkpoint 40, 2000-checkpoint
        # per-cell ceiling.
        stopping = contract["decision_run"]["stopping_knobs"]
        config["max_epochs"] = int(knobs["phase2_max_epochs"])
        config["easy_max_epochs"] = int(knobs["phase1_epochs"])
        config["l1_patience"] = int(stopping["l1_patience"])
        config["l1_patience_start_epoch"] = int(
            stopping["l1_patience_start_epoch"])
        config["l1_activity_patience"] = int(
            stopping["l1_activity_patience"])
        config["l1_activity_patience_start_epoch"] = int(
            stopping["l1_activity_patience_start_epoch"])
        config["total_max_passes"] = int(stopping["total_max_passes"])
        config["phase1_max_fraction"] = float(
            stopping["phase1_max_fraction"])
        config["normal_phase_min_passes"] = int(
            stopping["normal_phase_min_passes"])
        config["easy_patience"] = 10_000
        config["execution_cost_curriculum_epochs"] = max(
            2, int(knobs["phase2_max_epochs"]))
    else:
        # Screen budget: ONE pass-equivalent, no early stopping fires.
        config["max_epochs"] = int(knobs["phase2_epochs"])
        config["easy_max_epochs"] = int(knobs["phase1_epochs"])
        config["l1_patience"] = 10_000
        config["easy_patience"] = 10_000
        config["execution_cost_curriculum_epochs"] = max(
            2, int(knobs["phase2_epochs"]))
    # Finding 191 discipline: the plugin that EXECUTES is the plugin
    # the contract names.
    config["agent_plugin"] = str(common["plugins"]["agent_plugin"])
    for key in ("env_plugin", "strategy_plugin"):
        bound = common["plugins"].get(key)
        if bound and config.get(key) != bound:
            raise RuntimeError(
                f"plugin drift: config[{key!r}]={config.get(key)!r} != "
                f"held-fixed binding {bound!r}")

    # Held fixed: the corrected v4 boundary, phase-2 LR, the v3
    # cost/protection contract (includes the 0.1 deadband), entropy.
    config["phase1_handoff_semantics"] = str(
        held["phase1_handoff_semantics"])
    config["learning_rate"] = float(held["phase2_learning_rate"])
    config.update(v3_costs)
    entropy_value = config.get("ent_coef")
    if entropy_value != float(held["entropy"]["value"]):
        raise RuntimeError(
            f"base config ent_coef={entropy_value!r} is not the "
            f"held-fixed entropy {held['entropy']['value']!r} — the "
            "entropy identity must come from the base config, not a "
            "silent override")

    # Per-seed anchor: hash-bound, NEVER a preceding cell's terminal
    # and NEVER a screen terminal (finding 226) — an anchor living
    # under either mode's output root is structurally refused.
    anchor = contract["anchors"][str(seed)]
    anchor_path = Path(anchor["path"]).expanduser()
    for other_mode in MODES:
        mode_root = output_root_for_mode(contract, other_mode)
        try:
            anchor_path.resolve().relative_to(mode_root.resolve())
        except ValueError:
            continue
        raise RuntimeError(
            f"anchor for seed {seed} lives under the {other_mode} "
            "output root — a run terminal can never anchor a cell "
            "(finding 226)")
    config["warm_start_model"] = str(anchor_path)
    config["warm_start_model_sha256"] = anchor["sha256"]

    # Structural unreachability of the legacy branch (finding 224):
    # the executable config MUST carry the nested contract and the
    # paired metric, and MUST NOT carry any legacy split window.
    assert config["nested_split_contract"] == nested_binding["path"]
    if config["selection_metric"] != NESTED_SELECTION_METRIC:
        raise RuntimeError(
            "executable selection_metric drifted from the paired "
            "comparator — the legacy lexicographic branch is forbidden")
    for legacy_key in ("train_start", "train_end", "validation_start",
                       "validation_end", "test_start", "test_end",
                       "train_years", "val_years", "test_years"):
        if config.get(legacy_key) is not None:
            raise RuntimeError(
                f"legacy split field {legacy_key!r} survived "
                "materialization — refused (finding 224)")

    # --- the cell's declared factor levels, and NOTHING else ----------
    factors = contract["cells"][cell]
    config["phase1_mode"] = str(factors["phase1_dynamics"])
    config["phase1_learning_rate"] = float(
        factors["phase1_learning_rate"])
    config["easy_learning_rate"] = float(
        factors["phase1_learning_rate"])

    config["_identity"] = {
        "experiment_contract_sha256": contract["_contract_sha256"],
        "held_fixed_bindings_contract_sha256":
            bindings["_contract_sha256"],
        "base_config_sha256": actual_base_sha,
        "data_sha256": common["data"]["sha256"],
        "mode": mode,
        "selection_metric": str(config["selection_metric"]),
        "nested_split_contract_path": nested_binding[
            "contract_relative_path"],
        "nested_split_contract_sha256": nested_binding["sha256"],
        "nested_split_mode": nested_binding["mode"],
        "nested_context_bars": nested_binding["context_bars"],
        "nested_role_facts_pinned": nested_binding["role_facts_pinned"],
        "nested_nontest_union": union_facts,
        "l1_system_manifest_sha256": bindings["l1_reference"][
            "system_manifest_sha256"],
        "anchor_sha256": anchor["sha256"],
        "seed": int(seed),
        "cell": cell,
        "factors": dict(factors),
        "held_fixed": {
            "phase2_learning_rate": float(held["phase2_learning_rate"]),
            "phase2_dynamics": str(held["phase2_dynamics"]),
            "phase2_action_threshold": float(
                held["phase2_action_threshold"]),
            "phase1_handoff_semantics": str(
                held["phase1_handoff_semantics"]),
            "entropy": dict(held["entropy"]),
            "cost_protection_contract": "l1_v3_manifest_bindings",
        },
        "plugins": dict(common["plugins"]),
    }
    if decision_execution_profile is not None:
        config["_identity"]["decision_execution_profile_sha256"] = \
            decision_execution_profile["_profile_sha256"]
        config["_identity"]["decision_replay_buffer"] = {
            "buffer_size": int(config["buffer_size"]),
            "optimize_memory_usage": bool(config["optimize_memory_usage"]),
            "uniform_across_seeds": True,
        }
    return config


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def record_is_complete(record_path: Path, expected_cell_id: str) -> bool:
    try:
        record = json.loads(record_path.read_text())
    except Exception:
        return False
    if record.get("schema") != RECORD_SCHEMA:
        return False
    if record.get("cell_identity") != expected_cell_id:
        return False
    for key in ("terminal_model_path", "terminal_model_sha256"):
        if not record.get(key):
            return False
    # Finding 232: an INACTIVE record is reusable on restart exactly
    # like an active one — same terminal custody, plus the typed
    # non-promotable invariants. A record that lost them is not
    # complete (the caller then refuses rather than overwriting).
    status = record.get("activity_status")
    if status is not None:
        if status not in ACTIVITY_STATUSES:
            return False
        if status == "inactive" and (
                record.get("promotion_eligible") is not False
                or record.get("best_model_path")
                or record.get("best_model_sha256")):
            return False
    terminal = Path(record["terminal_model_path"])
    if not terminal.is_file():
        return False
    return _sha_file(terminal) == record["terminal_model_sha256"]


def _next_attempt_dir(cell_dir: Path, cell_id: str) -> Path:
    n = 0
    while True:
        n += 1
        attempt = cell_dir / f"attempt-{cell_id}-{n:02d}"
        if not attempt.exists():
            attempt.mkdir(parents=True)
            return attempt


def bind_handoff_viability(result: dict) -> dict:
    """Surface the finding-221 evidence the pipeline attaches to every
    phase-1 checkpoint: the SELECTED-checkpoint viability plus a
    per-checkpoint summary. A cell whose pipeline result carries no
    typed selected viability is REFUSED — evidence-free handoffs are
    exactly what finding 221 forbids."""
    post_easy = (result.get("curriculum") or {}).get("post_easy") or {}
    selected = post_easy.get("selected_handoff_viability")
    if not isinstance(selected, dict):
        raise RuntimeError(
            "pipeline result carries no selected_handoff_viability "
            "block — the finding-221 evidence contract is unmet; "
            "record refused")
    label = selected.get("handoff_viability")
    if label not in HANDOFF_VIABILITY_VALUES:
        raise RuntimeError(
            f"selected handoff_viability {label!r} is not a typed "
            f"label from {list(HANDOFF_VIABILITY_VALUES)}; record "
            "refused")
    if selected.get("trained_treatment") is not True:
        raise RuntimeError(
            "the selected phase-1 handoff is not a trained treatment — "
            "under l1_trained_epoch_v4 the selected checkpoint must be "
            "a trained epoch (finding 221); record refused")
    per_checkpoint = []
    for row in post_easy.get("history") or []:
        epoch = row.get("epoch")
        evidence = row.get("handoff_viability_evidence")
        if not isinstance(evidence, dict):
            raise RuntimeError(
                f"phase-1 checkpoint epoch {epoch!r} carries no "
                "handoff_viability_evidence — the finding-221 "
                "per-checkpoint contract is unmet; record refused")
        per_checkpoint.append({
            "epoch": epoch,
            "checkpoint_source": row.get("checkpoint_source"),
            "handoff_viability": evidence.get("handoff_viability"),
            "trained_treatment": evidence.get("trained_treatment"),
            "viable_as_trained_treatment": evidence.get(
                "viable_as_trained_treatment"),
            "any_action_crosses_phase2_threshold": evidence.get(
                "any_action_crosses_phase2_threshold"),
            "probe_trades_total": evidence.get("probe_trades_total"),
        })
    return {
        "source": ("pipeline curriculum.post_easy — the finding-221 "
                   "block attached to every phase-1 checkpoint"),
        "selected": selected,
        "selected_label": label,
        "selected_is_collapse": label in COLLAPSE_LABELS,
        "per_checkpoint": per_checkpoint,
    }


# ---------------------------------------------------------------------------
# activity: the typed inactive outcome (finding 232)
# ---------------------------------------------------------------------------

def classify_cell_activity(result: dict) -> dict:
    """Finding 232: classify a finished cell as ``active`` or
    ``inactive`` from the pipeline's TYPED result — never from a
    harness exception.

    ``inactive`` means the treatment produced a policy that never became
    activity-eligible, so no best checkpoint can exist. It is a MEASURED
    outcome: the cell record still lands (with the same custody strength
    as an active cell), it is never promotable, and the seed runner
    continues to its next cell.

    A result that carries NEITHER a restorable best checkpoint NOR the
    typed inactive declaration (``activity_stopped_without_eligible_
    checkpoint`` plus a non-empty ``termination_cause``) is a harness
    failure and is REFUSED — silence is never a measurement.
    """
    best_path = result.get("best_model_path")
    has_best = bool(best_path) and Path(str(best_path)).is_file()
    declared = bool(result.get(
        "activity_stopped_without_eligible_checkpoint", False))
    cause = result.get("termination_cause")
    stop_reason = result.get("stop_reason")
    if has_best and declared:
        raise RuntimeError(
            "pipeline result is self-contradictory: it declares "
            "activity_stopped_without_eligible_checkpoint AND carries a "
            f"best checkpoint {best_path!r} — record refused "
            "(finding 232)")
    if has_best:
        return {
            "activity_status": "active",
            "promotion_eligible": True,
            "inactive_cause": None,
            "termination_cause": cause,
            "stop_reason": stop_reason,
            "evidence": ("a restorable, hash-bound best checkpoint "
                         "exists and was selected by the paired "
                         "comparator"),
        }
    if not declared:
        raise RuntimeError(
            "cell finished with NO best checkpoint and NO typed "
            "inactive declaration "
            "(activity_stopped_without_eligible_checkpoint) — an "
            "unexplained missing checkpoint is a harness failure, not a "
            "measured outcome; record refused (finding 232)")
    if not isinstance(cause, str) or not cause.strip():
        raise RuntimeError(
            "inactive cell carries no termination_cause — an inactive "
            "outcome must NAME why the policy never became "
            "activity-eligible; record refused (finding 232)")
    return {
        "activity_status": "inactive",
        "promotion_eligible": False,
        "inactive_cause": "no_activity_eligible_checkpoint",
        "termination_cause": cause.strip(),
        "stop_reason": stop_reason,
        "evidence": ("training ended before any activity-eligible "
                     "checkpoint existed; the terminal policy is bound, "
                     "hashed and load-proven as diagnostic truth only"),
    }


def terminal_load_proof(terminal_path, terminal_sha: str,
                        policy_tensor_sha: str) -> dict:
    """Finding 223/232: the terminal artifact is not merely hashed — it
    is LOADED and its policy tensors digested. The digest IS the load
    proof; an unloadable artifact never reaches here."""
    if not policy_tensor_sha or not isinstance(policy_tensor_sha, str):
        raise RuntimeError(
            "terminal artifact produced no policy-tensor digest — the "
            "load proof is unmet; record refused (finding 223/232)")
    return {
        "schema": TERMINAL_LOAD_PROOF_SCHEMA,
        "path": str(Path(terminal_path).resolve()),
        "sha256": terminal_sha,
        "policy_tensor_sha256": policy_tensor_sha,
        "loaded": True,
        "method": ("artifact loaded and its policy tensors digested "
                   "(agent_plugins.sac_agent._policy_tensor_hash) — a "
                   "digest cannot exist without a successful load"),
        "proved_utc": datetime.now(timezone.utc).isoformat(),
    }


def _walk_json(payload, path: str = ""):
    """Yield (dotted_key_path, key, value) for every mapping entry."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            yield here, str(key), value
            yield from _walk_json(value, here)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            yield from _walk_json(value, f"{path}[{index}]")


def assert_terminal_never_labeled_best(payload, *, terminal_path,
                                       terminal_sha: str,
                                       context: str) -> None:
    """Finding 232 requirement 2: the terminal artifact may be loaded
    for diagnostic truth, but it must NEVER be relabeled a best
    checkpoint. Any ``best``-named field anywhere in ``payload`` bound
    to the terminal artifact's path or sha256 is a refusal — this makes
    the relabeling structurally impossible, not merely discouraged."""
    resolved = str(Path(str(terminal_path)).resolve())
    identities = {resolved, str(terminal_path)}
    for dotted, key, value in _walk_json(payload):
        if "best" not in key.lower():
            continue
        if not isinstance(value, str):
            continue
        if value in identities or (terminal_sha and value == terminal_sha):
            raise RuntimeError(
                f"{context}: field {dotted!r} labels the TERMINAL "
                f"artifact ({value!r}) as a best checkpoint — an "
                "inactive cell's terminal policy is diagnostic truth "
                "only and can never be a best checkpoint (finding 232)")


def assert_outer_eval_artifact_role(outer: dict, *, artifact_role: str,
                                    terminal_path, terminal_sha: str,
                                    best_path, best_sha) -> None:
    """The single final outer evaluation is typed by the artifact it
    loaded. ``terminal`` payloads must carry no best-checkpoint
    identity at all; ``best_checkpoint`` payloads must evaluate exactly
    the record's best artifact and never the terminal one."""
    if artifact_role not in OUTER_ARTIFACT_ROLES:
        raise RuntimeError(
            f"unknown outer-evaluation artifact role {artifact_role!r}")
    if not isinstance(outer, dict):
        raise RuntimeError(
            "final outer evaluation did not return a typed payload")
    got = outer.get("evaluated_artifact_role")
    if got != artifact_role:
        raise RuntimeError(
            f"final outer evaluation declares artifact role {got!r} but "
            f"the runner loaded the {artifact_role!r} artifact — "
            "record refused (finding 232)")
    evaluated = outer.get("evaluated_model_path")
    terminal_resolved = str(Path(str(terminal_path)).resolve())
    if artifact_role == "terminal":
        if outer.get("best_model_path") is not None or \
                outer.get("best_model_sha256") is not None:
            raise RuntimeError(
                "final outer evaluation of an INACTIVE cell carries a "
                "best_model_path/sha — the terminal policy is never a "
                "best checkpoint (finding 232)")
        if outer.get("promotion_eligible") is not False:
            raise RuntimeError(
                "final outer evaluation of an INACTIVE cell is not "
                "marked promotion_eligible=false (finding 232)")
        if str(Path(str(evaluated)).resolve()) != terminal_resolved:
            raise RuntimeError(
                f"inactive cell's final outer evaluation loaded "
                f"{evaluated!r}, not the bound terminal artifact "
                f"{terminal_resolved!r} (finding 232)")
        assert_terminal_never_labeled_best(
            outer, terminal_path=terminal_path,
            terminal_sha=terminal_sha,
            context="final outer evaluation (inactive cell)")
        return
    if not best_path:
        raise RuntimeError(
            "best_checkpoint outer evaluation without a best artifact")
    best_resolved = str(Path(str(best_path)).resolve())
    if str(Path(str(evaluated)).resolve()) != best_resolved:
        raise RuntimeError(
            f"final outer evaluation loaded {evaluated!r} instead of "
            f"the restored best checkpoint {best_resolved!r}")
    if str(Path(str(evaluated)).resolve()) == terminal_resolved:
        raise RuntimeError(
            "final outer evaluation loaded the TERMINAL artifact while "
            "declaring it the best checkpoint (finding 232)")
    if outer.get("best_model_sha256") not in (None, best_sha):
        raise RuntimeError(
            "final outer evaluation binds a best_model_sha256 that is "
            "not the record's best checkpoint hash")


RECORD_CUSTODY_KEYS = (
    # source identities every cell record binds — an inactive record
    # carries EXACTLY the same custody as an active one (finding 223).
    "experiment_identity", "cell_identity", "contract_sha256",
    "nested_split_contract_sha256", "anchor_sha256",
    "terminal_model_path", "terminal_model_sha256",
    "terminal_policy_tensor_sha256",
)


def assert_cell_record_custody(record: dict) -> None:
    """Refuse to publish a cell record — active OR inactive — that is
    missing any custody field, mistypes its activity status, or lets an
    inactive cell carry promotion/best-checkpoint identity."""
    missing = [key for key in RECORD_CUSTODY_KEYS if not record.get(key)]
    if missing:
        raise RuntimeError(
            f"cell record is missing custody fields {missing} — an "
            "inactive cell has the SAME custody strength as an active "
            "one; record refused (finding 223/232)")
    proof = record.get("terminal_load_proof") or {}
    if proof.get("loaded") is not True or \
            proof.get("sha256") != record["terminal_model_sha256"] or \
            proof.get("policy_tensor_sha256") != record[
                "terminal_policy_tensor_sha256"]:
        raise RuntimeError(
            "cell record carries no terminal load proof bound to its "
            "own terminal hashes; record refused (finding 223/232)")
    status = record.get("activity_status")
    if status not in ACTIVITY_STATUSES:
        raise RuntimeError(
            f"cell record activity_status {status!r} is not one of "
            f"{list(ACTIVITY_STATUSES)} (finding 232)")
    if status == "inactive":
        if record.get("promotion_eligible") is not False:
            raise RuntimeError(
                "inactive cell record is not promotion_eligible=false "
                "(finding 232)")
        if record.get("best_model_path") or record.get(
                "best_model_sha256"):
            raise RuntimeError(
                "inactive cell record carries a best checkpoint — an "
                "inactive policy has none by definition (finding 232)")
        if record.get("activity_inactive_cause") not in INACTIVE_CAUSES:
            raise RuntimeError(
                "inactive cell record names no typed termination cause "
                "(finding 232)")
        if not record.get("termination_cause"):
            raise RuntimeError(
                "inactive cell record carries no termination_cause "
                "narrative (finding 232)")
        assert_terminal_never_labeled_best(
            record, terminal_path=record["terminal_model_path"],
            terminal_sha=record["terminal_model_sha256"],
            context="cell record")
    else:
        if record.get("promotion_eligible") is not True:
            raise RuntimeError(
                "active cell record is not promotion_eligible=true "
                "(finding 232)")
        if not record.get("best_model_path"):
            raise RuntimeError(
                "active cell record carries no best checkpoint path "
                "(finding 234)")
        if not record.get("best_model_sha256"):
            raise RuntimeError(
                "active cell record carries no best checkpoint hash "
                "(finding 234)")


# ---------------------------------------------------------------------------
# GPU binding: WP13 CUDA equality + the readiness-probe launch gate
# ---------------------------------------------------------------------------

def check_gpu_binding(contract: dict, seed: int, *,
                      hostname: str | None = None,
                      cuda_env: str | None = "<from-environment>",
                      observed_uuids: list | None = None,
                      enforce: bool = True) -> dict | None:
    assignment = (contract.get("assignments") or {}).get(
        str(seed)) or {}
    hostname = hostname or socket.gethostname()
    if cuda_env == "<from-environment>":
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assignment.get("hostname") != hostname:
        return {"outcome": "REFUSED_WRONG_HOST",
                "reason": (f"seed {seed} is assigned to "
                           f"{assignment.get('hostname')!r}, this is "
                           f"{hostname!r}")}
    if not enforce:
        return None
    assigned = assignment.get("gpu_uuid")
    observed = (visible_gpu_uuids() if observed_uuids is None
                else observed_uuids)
    if assigned not in observed:
        return {"outcome": "REFUSED_GPU_UNBOUND",
                "reason": f"assigned GPU {assigned} not visible on "
                          f"{hostname}"}
    if cuda_env is None:
        return {"outcome": "REFUSED_GPU_UNBOUND",
                "reason": ("CUDA_VISIBLE_DEVICES is unset; visibility "
                           "is not an execution binding (WP13)")}
    if cuda_env != assigned:
        return {"outcome": "REFUSED_GPU_UNBOUND",
                "reason": (f"CUDA_VISIBLE_DEVICES={cuda_env!r} does not "
                           f"equal the assignment {assigned!r}")}
    return None


def gpu_launch_gate(assigned_gpu_uuid: str, *,
                    heartbeat: dict | None = None) -> tuple:
    """The gpu_readiness_probe launch gate: typed refusal BEFORE any
    framework import when the assigned UUID is absent, the driver probe
    fails, or CUDA would fall back to CPU. Returns (gate_payload,
    refusal_or_None)."""
    if heartbeat is None:
        heartbeat = gpu_probe.collect_heartbeat()
    payload, exit_code = gpu_probe.launch_gate(
        heartbeat, assigned_gpu_uuid)
    if exit_code != gpu_probe.EXIT_OK:
        return payload, {
            "outcome": "REFUSED_GPU_UNBOUND",
            "reason": ("gpu_readiness_probe launch gate refused: "
                       + "; ".join(payload.get("reasons") or
                                   payload.get("blocking") or
                                   ["unknown"])),
            "gate": payload,
        }
    return payload, None


class CellHeartbeat:
    """Atomic heartbeat with assigned/bound/observed CUDA facts, in the
    proven mechanism-ladder heartbeat style."""

    def __init__(self, path: Path, *, contract: dict, seed: int,
                 cell: str, exp_id: str, cell_id: str, mode: str):
        self.path = path
        self.contract = contract
        self.seed = int(seed)
        self.cell = cell
        self.exp_id = exp_id
        self.cell_id = cell_id
        # Finding 233: the operator surfaces bind mode positionally
        # (mode root + identity). Declaring it in the heartbeat makes
        # the binding directly verifiable instead of inferred.
        self.mode = mode
        self._state: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self, **update) -> None:
        self._state.update(update)
        pid = os.getpid()
        assignment = (self.contract.get("assignments") or {}).get(
            str(self.seed)) or {}
        self._state.update({
            "schema": HEARTBEAT_SCHEMA,
            "mode": self.mode,
            "seed": self.seed,
            "cell": self.cell,
            "experiment_identity": self.exp_id,
            "cell_identity": self.cell_id,
            "pid": pid,
            "pid_start_identity": _pid_start_identity(pid),
            "hostname": socket.gethostname(),
            "assigned_gpu_uuid": assignment.get("gpu_uuid"),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"),
            "observed_gpu_uuids": visible_gpu_uuids(),
            **ladder.gpu_telemetry(assignment.get("gpu_uuid")),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.path, self._state)

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.wait(HEARTBEAT_INTERVAL_S):
                self.beat()
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# the cell run
# ---------------------------------------------------------------------------

def _default_pipeline_factory(config: dict):
    from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
        PipelinePlugin as CurriculumPipeline)
    return CurriculumPipeline(config)


def _verify_anchor(contract: dict, seed: int) -> str:
    anchor = contract["anchors"][str(seed)]
    anchor_path = Path(anchor["path"]).expanduser()
    if not anchor_path.is_file():
        raise RuntimeError(f"anchor missing: {anchor_path}")
    actual = _sha_file(anchor_path)
    if actual != anchor["sha256"]:
        raise RuntimeError(
            f"anchor hash mismatch for seed {seed}: the distributed "
            "anchor is not the contract-bound artifact")
    return actual


def run_cell(seed: int, cell: str, *, contract: dict, bindings: dict,
             exp_id: str, sources_before: dict,
             gpu_dispatch_binding: dict | None,
             gpu_gate: dict | None,
             pipeline_factory=None, agent_loader=None,
             tensor_sha_fn=None, mode: str = "screen",
             nested_roles_fn=None, outer_eval_fn=None) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown execution mode {mode!r}")
    cell_id = cell_identity(exp_id, seed, cell, contract)
    out_root = output_root_for_mode(contract, mode)
    seed_dir = out_root / exp_id / f"seed{seed}"
    cell_dir = seed_dir / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = CellHeartbeat(cell_dir / "heartbeat.json",
                              contract=contract, seed=seed, cell=cell,
                              exp_id=exp_id, cell_id=cell_id, mode=mode)

    record_path = cell_dir / "cell_record.json"
    if record_path.exists():
        if record_is_complete(record_path, cell_id):
            record = json.loads(record_path.read_text())
            record["_reuse"] = "ALREADY_COMPLETE"
            heartbeat.beat(terminal_state="ALREADY_COMPLETE")
            return record
        raise RuntimeError(
            f"seed {seed} cell {cell}: existing record fails "
            "validation; refusing to overwrite — recover it explicitly")

    claim = ExclusiveClaim(
        out_root / exp_id / "locks" /
        f"exclusive_claim.seed{seed}.{cell}.lock")
    if not claim.acquire():
        return {"outcome": "ALREADY_RUNNING",
                "experiment_identity": exp_id, "seed": seed,
                "cell": cell, "holder": claim.holder()}
    heartbeat.start()
    try:
        # Anchor custody re-proof directly before materialization: the
        # cell starts from the exact per-seed anchor, never a
        # preceding cell's terminal.
        actual_anchor = _verify_anchor(contract, seed)

        out_dir = _next_attempt_dir(cell_dir, cell_id)
        heartbeat.beat(terminal_state="RUNNING", attempt=str(out_dir),
                       progress="materializing")
        config = materialize_cell_config(contract, bindings, seed,
                                         cell, out_dir, mode=mode)
        identity = config.pop("_identity")
        resolved_sha = sysid.resolved_config_sha256(config)
        agent_name = identity["plugins"]["agent_plugin"]
        agent = (agent_loader or ladder._agent_plugin)(agent_name)

        # Finding 224: materialize + verify the nested evidence roles
        # BEFORE model construction. Wrong role path/count/sha, a
        # missing context flag, context counted in score or any
        # sealed-test materialization refuses HERE — no GPU time is
        # ever spent on the wrong roles. The pipeline re-materializes
        # into the same nested_split_dir deterministically.
        heartbeat.beat(progress="nested-role-verification")
        nested_roles = (nested_roles_fn or materialize_nested_roles)(
            contract, bindings, out_dir)

        started = datetime.now(timezone.utc)
        heartbeat.beat(progress="training")
        pipeline = (pipeline_factory or _default_pipeline_factory)(
            config)
        result = pipeline.run_pipeline(config=config, env_plugin=None,
                                       agent_plugin=agent, mode="train")
        heartbeat.beat(progress="terminal-custody")

        # Finding 223: a cell without a hash-bound, load-proven
        # terminal artifact is REFUSED, never certified.
        terminal_path = result.get("terminal_model_path")
        if not terminal_path or not Path(terminal_path).is_file():
            raise RuntimeError(
                "cell finished without a terminal artifact — invalid, "
                "record refused (finding 223)")
        terminal_sha = _sha_file(Path(terminal_path))
        terminal_tensor = (tensor_sha_fn or _terminal_tensor_sha)(
            terminal_path)
        load_proof = terminal_load_proof(terminal_path, terminal_sha,
                                         terminal_tensor)

        # Finding 232: the typed activity classification. An inactive
        # cell is a MEASURED outcome — it publishes its record and the
        # seed continues; only an UNEXPLAINED missing checkpoint raises.
        activity = classify_cell_activity(result)
        activity_status = activity["activity_status"]
        best_path = (result.get("best_model_path")
                     if activity_status == "active" else None)
        best_sha = (_sha_file(Path(best_path))
                    if best_path and Path(best_path).is_file() else None)

        # Finding 221: bind the typed handoff-viability evidence.
        viability = bind_handoff_viability(result)

        # Finding 226 (decision mode): immutable best-checkpoint
        # restoration is evidence, then ONE final outer-validation
        # evaluation after selection. Sealed 2025 stays untouched —
        # the nested manifest above already proved it SEALED.
        # Finding 232: an INACTIVE cell also receives exactly ONE final
        # outer evaluation, of its TERMINAL artifact, as diagnostic
        # truth only — assert_outer_eval_artifact_role refuses any
        # payload that relabels it a best checkpoint.
        outer_final = None
        outer_artifact_role = None
        if mode == "decision":
            if activity_status == "active":
                if not best_path or not Path(best_path).is_file():
                    raise RuntimeError(
                        "decision cell finished without a restorable "
                        "best checkpoint artifact — record refused "
                        "(finding 226: immutable best-checkpoint "
                        "restoration)")
                outer_artifact_role = "best_checkpoint"
                eval_path = best_path
            else:
                outer_artifact_role = "terminal"
                eval_path = terminal_path
            heartbeat.beat(progress="outer-validation-final",
                           activity_status=activity_status)
            outer_final = (outer_eval_fn
                           or _outer_validation_final_eval)(
                config=config, agent=agent, model_path=eval_path,
                artifact_role=outer_artifact_role,
                nested_roles=nested_roles, seed=int(seed))
            assert_outer_eval_artifact_role(
                outer_final, artifact_role=outer_artifact_role,
                terminal_path=terminal_path, terminal_sha=terminal_sha,
                best_path=best_path, best_sha=best_sha)

        finished = datetime.now(timezone.utc)
        sources_after = ladder.source_identities()
        for name in sources_before:
            sysid.assert_source_identity_unmoved(
                sources_before[name], sources_after[name])

        history = result.get("history") or []
        last = history[-1] if history else {}
        post_easy_meta = (result.get("curriculum") or {}).get(
            "post_easy") or {}
        assignment = (contract.get("assignments") or {}).get(
            str(seed)) or {}
        gpu_binding = {
            "assigned_hostname": assignment.get("hostname"),
            "observed_hostname": socket.gethostname(),
            "assigned_gpu_uuid": assignment.get("gpu_uuid"),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"),
            "observed_gpu_uuids": visible_gpu_uuids(),
        }
        try:
            import torch
            gpu_binding["torch_cuda_available"] = \
                torch.cuda.is_available()
            gpu_binding["torch_cuda_device_count"] = \
                torch.cuda.device_count()
            if torch.cuda.is_available():
                gpu_binding["torch_cuda_device_name"] = \
                    torch.cuda.get_device_name(0)
        except Exception:
            gpu_binding["torch_cuda_available"] = None

        cell_order = list(contract["cell_order"][str(seed)])
        nested_binding = nested_roles["binding"]
        record = {
            "schema": RECORD_SCHEMA,
            "mode": mode,
            "evidence_class": ("decision_run" if mode == "decision"
                               else "mechanics_screen"),
            "decision_eligible": mode == "decision",
            "performance_aggregate_eligible": mode == "decision",
            "live_promotion_eligible": False,
            "experiment_identity": exp_id,
            "cell_identity": cell_id,
            "seed": int(seed),
            "cell": cell,
            "factors": dict(contract["cells"][cell]),
            "cell_order": cell_order,
            "cell_position": cell_order.index(cell),
            "attempt_dir": str(out_dir),
            "contract_sha256": contract["_contract_sha256"],
            "resolved_config_sha256": resolved_sha,
            "identity": identity,
            # Finding 224: every record binds the nested contract sha,
            # the split-manifest sha, per-role CSV shas, scored/context
            # counts and exact score dates; sealed_test stays SEALED.
            "selection_metric": str(config["selection_metric"]),
            "nested_split_contract_path": nested_binding[
                "contract_relative_path"],
            "nested_split_contract_sha256": nested_binding["sha256"],
            "nested_split_mode": nested_binding["mode"],
            "nested_split_manifest_path": nested_roles["manifest_path"],
            "nested_split_manifest_sha256": nested_roles[
                "manifest_sha256"],
            "nested_role_facts": {
                role: {key: (nested_roles["roles"].get(role) or {})
                       .get(key) for key in NESTED_ROLE_FACT_KEYS}
                for role in _nested_splits.ROLES},
            "outer_validation_final": outer_final,
            "outer_validation_artifact_role": outer_artifact_role,
            "gpu_binding": gpu_binding,
            "gpu_dispatch_binding": gpu_dispatch_binding,
            "gpu_launch_gate": gpu_gate,
            "subject_code_identity": sources_before,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
            "phase1_mode": post_easy_meta.get("phase1_mode"),
            "phase1_handoff_semantics": config.get(
                "phase1_handoff_semantics"),
            "phase1_selection_basis": post_easy_meta.get(
                "selection_basis"),
            "phase1_best_easy_epoch": post_easy_meta.get(
                "best_easy_epoch"),
            "phase1_gradient_updates": post_easy_meta.get(
                "phase1_gradient_updates"),
            "phase1_artifact_sha256": post_easy_meta.get(
                "artifact_sha256"),
            "phase1_terminal_policy_tensor_sha256": post_easy_meta.get(
                "phase1_terminal_policy_tensor_sha256"),
            "handoff_viability": viability,
            "gradient_updates_total": last.get(
                "gradient_updates_total"),
            "epoch_history": history,
            "stop_reason": result.get("stop_reason"),
            "termination_cause": activity["termination_cause"],
            "activity_stopped_without_eligible_checkpoint": bool(
                result.get(
                    "activity_stopped_without_eligible_checkpoint",
                    False)),
            # Finding 232: the typed activity outcome, carried by EVERY
            # record. ``promotion_eligible`` says only that a selected
            # best checkpoint exists and is hash-bound; live promotion
            # stays governed by live_promotion_eligible (always false).
            "activity_status": activity_status,
            "activity_inactive_cause": activity["inactive_cause"],
            "activity_evidence": activity["evidence"],
            "promotion_eligible": activity["promotion_eligible"],
            "boundary_transfer_evidence": result.get(
                "warm_start_transfer_evidence"),
            "anchor_sha256": actual_anchor,
            "best_model_path": best_path,
            "best_model_sha256": best_sha,
            "terminal_model_path": str(Path(terminal_path).resolve()),
            "terminal_model_sha256": terminal_sha,
            "terminal_policy_tensor_sha256": terminal_tensor,
            "terminal_load_proof": load_proof,
            # The exact source identities this outcome is bound to —
            # repeated as one block so an inactive record is
            # self-describing custody, not a pointer to context.
            "source_identities": {
                "experiment_identity": exp_id,
                "cell_identity": cell_id,
                "contract_sha256": contract["_contract_sha256"],
                "nested_split_contract_sha256": nested_binding["sha256"],
                "anchor_sha256": actual_anchor,
                "resolved_config_sha256": resolved_sha,
            },
            "curriculum": result.get("curriculum"),
        }
        # Finding 223/232: identical custody for active and inactive,
        # and a structural refusal of any best-labeled terminal.
        assert_cell_record_custody(record)
        atomic_write_json(record_path, record)
        heartbeat.beat(terminal_state="CELL_COMPLETE",
                       activity_status=activity_status,
                       last_artifact=record["terminal_model_path"],
                       progress="complete")
        return record
    except Exception as exc:
        heartbeat.beat(terminal_state="CELL_FAILED",
                       error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        heartbeat.stop()
        claim.release()


# ---------------------------------------------------------------------------
# the seed batch: four cells, sequential, one GPU
# ---------------------------------------------------------------------------

def verify_screen_gate(gate: dict | None, contract: dict) -> list:
    """Finding 226: the decision path is CONDITIONAL. Dispatch demands
    the corrected screen verdict with a passing boolean replica gate;
    anything else returns exact refusal strings."""
    refusals: list = []
    if not isinstance(gate, dict):
        return ["decision mode requires the corrected screen verdict "
                "(--screen-gate PATH) — the screen gate plus the "
                "16-terminal replica proof decide (finding 226)"]
    if gate.get("schema") != VERDICT_SCHEMA:
        refusals.append(f"screen gate schema {gate.get('schema')!r} != "
                        f"{VERDICT_SCHEMA!r}")
    if gate.get("outcome") != "SCREEN_VIABLE_REGION":
        refusals.append(
            f"screen gate outcome {gate.get('outcome')!r} — only "
            "SCREEN_VIABLE_REGION authorizes the decision budget")
    if gate.get("contract_sha256") != contract["_contract_sha256"]:
        refusals.append("screen gate binds a different contract sha")
    if (gate.get("gates") or {}).get("replica_terminal_loads") \
            is not True:
        refusals.append(
            "screen gate replica_terminal_loads is not boolean true — "
            "the 16-terminal replica proof is mandatory (finding 225)")
    return refusals


def run_seed(seed: int, *, contract: dict, bindings: dict | None = None,
             enforce_gpu: bool = True, pipeline_factory=None,
             agent_loader=None, tensor_sha_fn=None,
             gate_heartbeat: dict | None = None,
             dispatch_binding_fn=None, mode: str = "screen",
             screen_gate: dict | None = None,
             nested_roles_fn=None, outer_eval_fn=None) -> dict:
    if int(seed) not in SEEDS:
        raise ValueError(f"unknown factorial seed {seed!r}")
    if mode not in MODES:
        raise ValueError(f"unknown execution mode {mode!r}")
    bindings = bindings or load_bindings()
    sources_before = ladder.source_identities()
    exp_id = experiment_identity(contract, bindings,
                                 sources=sources_before, mode=mode)
    out_root = output_root_for_mode(contract, mode)
    seed_dir = out_root / exp_id / f"seed{seed}"
    assignment = (contract.get("assignments") or {}).get(
        str(seed)) or {}

    def _seed_refusal(refusal: dict) -> dict:
        seed_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(seed_dir / "runner_heartbeat.json", {
            "schema": HEARTBEAT_SCHEMA,
            "seed": int(seed),
            "cell": None,
            "experiment_identity": exp_id,
            "terminal_state": refusal["outcome"],
            "error": refusal.get("reason"),
            "hostname": socket.gethostname(),
            "assigned_gpu_uuid": assignment.get("gpu_uuid"),
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        refusal["experiment_identity"] = exp_id
        refusal["seed"] = int(seed)
        return refusal

    # Finding 226: decision dispatch is gated on the corrected screen
    # verdict + boolean replica gate BEFORE any GPU/anchor work.
    if mode == "decision":
        gate_refusals = verify_screen_gate(screen_gate, contract)
        if gate_refusals:
            return _seed_refusal({
                "outcome": "REFUSED_DECISION_UNGATED",
                "reason": "; ".join(gate_refusals)})

    refusal = check_gpu_binding(contract, seed, enforce=enforce_gpu)
    if refusal:
        return _seed_refusal(refusal)

    gate_payload = None
    dispatch_binding = None
    if enforce_gpu:
        gate_payload, gate_refusal = gpu_launch_gate(
            assignment["gpu_uuid"], heartbeat=gate_heartbeat)
        if gate_refusal:
            return _seed_refusal(gate_refusal)
        dispatch_binding = (dispatch_binding_fn
                            or gpu_probe.dispatch_gpu_binding)(
            assignment["gpu_uuid"])

    # Per-seed anchor custody, proven ONCE before any cell spends GPU
    # time (each cell re-proves before materializing).
    try:
        _verify_anchor(contract, seed)
    except RuntimeError as exc:
        return _seed_refusal({"outcome": "REFUSED_ANCHOR_UNVERIFIED",
                              "reason": str(exc)})

    cells = list(contract["cell_order"][str(seed)])
    outcomes: dict = {}
    reused = 0
    for cell in cells:
        try:
            record = run_cell(
                seed, cell, contract=contract, bindings=bindings,
                exp_id=exp_id, sources_before=sources_before,
                gpu_dispatch_binding=dispatch_binding,
                gpu_gate=gate_payload,
                pipeline_factory=pipeline_factory,
                agent_loader=agent_loader,
                tensor_sha_fn=tensor_sha_fn, mode=mode,
                nested_roles_fn=nested_roles_fn,
                outer_eval_fn=outer_eval_fn)
        except Exception as exc:                    # noqa: BLE001
            outcomes[cell] = {"outcome": "CELL_FAILED",
                              "error": f"{type(exc).__name__}: {exc}"}
            continue
        if record.get("outcome") == "ALREADY_RUNNING":
            outcomes[cell] = {"outcome": "ALREADY_RUNNING",
                              "holder": record.get("holder")}
            continue
        if record.get("_reuse") == "ALREADY_COMPLETE":
            reused += 1
            outcomes[cell] = {"outcome": "ALREADY_COMPLETE"}
        else:
            outcomes[cell] = {"outcome": "CELL_COMPLETE"}
        # Finding 232: an inactive cell landed a record and the loop
        # CONTINUES to the seed's remaining cells — a seed batch never
        # dies on inactivity.
        outcomes[cell].update({
            "cell_identity": record.get("cell_identity"),
            "activity_status": record.get("activity_status"),
            "promotion_eligible": record.get("promotion_eligible"),
            "selected_handoff_viability": (record.get(
                "handoff_viability") or {}).get("selected_label"),
            "terminal_model_sha256": record.get(
                "terminal_model_sha256"),
        })

    inactive_cells = [cell for cell in cells
                      if (outcomes.get(cell) or {}).get(
                          "activity_status") == "inactive"]
    states = {facts["outcome"] for facts in outcomes.values()}
    if "CELL_FAILED" in states:
        outcome = "SEED_FAILED"
    elif "ALREADY_RUNNING" in states:
        outcome = "ALREADY_RUNNING"
    elif reused == len(cells):
        outcome = ("ALREADY_COMPLETE_WITH_INACTIVE" if inactive_cells
                   else "ALREADY_COMPLETE")
    else:
        outcome = ("SEED_COMPLETE_WITH_INACTIVE" if inactive_cells
                   else "SEED_COMPLETE")
    return {
        "outcome": outcome,
        "mode": mode,
        "experiment_identity": exp_id,
        "seed": int(seed),
        "cell_order": cells,
        "cells": outcomes,
        # The batch exit class NAMES the inactivity without destroying
        # remaining work: every cell after an inactive one still ran.
        "inactive_cells": inactive_cells,
        "cells_completed": sum(
            1 for facts in outcomes.values()
            if facts["outcome"] in ("CELL_COMPLETE",
                                    "ALREADY_COMPLETE")),
        "cells_expected": len(cells),
    }


# ---------------------------------------------------------------------------
# decision mode: the ONE final outer-validation evaluation (finding 226)
# ---------------------------------------------------------------------------

def _outer_validation_final_eval(*, config: dict, agent,
                                 model_path: str,
                                 artifact_role: str,
                                 nested_roles: dict,
                                 seed: int,
                                 role: str = "outer_validation",
                                 solvency_mode: str = "normal_realistic",
                                 run_id: str = "p1lr_decision") -> dict:
    """Evaluate ONE artifact ONCE on a nested evaluation role — for the
    default ``outer_validation`` role this is final truth only, after
    selection. The 256 declared context rows
    initialize causal state under the reusable ContextPrefixWrapper
    (forced hold; any account mutation raises) and are excluded from
    every metric. Sealed 2025 is never touched.

    ``artifact_role`` is the ONLY thing that decides what the payload
    may claim (finding 232):

    * ``best_checkpoint`` — an ACTIVE cell's restored, selected
      checkpoint; the payload binds ``best_model_path``/``sha`` and is
      promotion-eligible evidence within this experiment.
    * ``terminal`` — an INACTIVE cell's terminal policy, loaded for
      DIAGNOSTIC TRUTH ONLY. The payload carries no best-checkpoint
      identity whatsoever and is marked ``promotion_eligible: false``
      and ``diagnostic_only: true``. There is no code path here that
      can name a terminal artifact a best checkpoint, and
      ``assert_outer_eval_artifact_role`` refuses any payload that
      does.

    ``role``/``solvency_mode``/``run_id`` exist so the L2 program can
    reuse THIS implementation for its inner_validation member and for
    an easy-difficulty triage replay instead of forking a second
    context-prefix replay. Their defaults reproduce the P1LR decision
    behaviour byte for byte.
    """
    if role not in _nested_splits.EVAL_ROLES or role == "sealed_test":
        raise RuntimeError(
            f"refusing a nested replay of role {role!r} — sealed 2025 is "
            "structurally inaccessible and only evaluation roles replay")
    from pipeline_plugins import _return_trace as trace_mod
    from pipeline_plugins._weekly_metrics import (
        canonical_weekly_metrics_from_trace)
    from pipeline_plugins.rl_pipeline_with_validation import (
        PipelinePlugin as ValidationPipeline)

    if artifact_role not in OUTER_ARTIFACT_ROLES:
        raise RuntimeError(
            f"unknown outer-evaluation artifact role {artifact_role!r} "
            f"— must be one of {list(OUTER_ARTIFACT_ROLES)}")
    role_name = role
    role = nested_roles["roles"][role_name]
    if role.get("status") != "MATERIALIZED":
        raise RuntimeError(f"{role_name} role is not materialized")
    context_rows = int(role["context_rows"])
    eval_config = dict(config)
    eval_config["solvency_mode"] = solvency_mode
    eval_config["return_trace_dir"] = None
    pipeline = ValidationPipeline(eval_config)
    plug, env = pipeline._make_split_env(
        str(eval_config.get("env_plugin", "gym_fx_env")), eval_config,
        str(role["csv"]), agent)
    env = _nested_splits.ContextPrefixWrapper(env, context_rows)
    try:
        model = agent.load(str(model_path), env)
        obs, info = env.reset(seed=int(seed))
        prev_equity = info.get("equity")
        trace_rows: list = []
        steps = scored_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.predict(model, obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if not info.get("is_context_prefix"):
                scored_steps += 1
                trace_rows.append(trace_mod.build_trace_row(
                    env=env, step=scored_steps, action=action,
                    reward=reward, info=info, prev_equity=prev_equity,
                    asset=str(eval_config.get("asset",
                                              "unknown_asset")),
                    timeframe=str(eval_config.get("timeframe", "")),
                    split=f"{role_name}_final", seed=int(seed),
                    run_id=run_id,
                    episode_id=f"{run_id}::{_ROLE_TAG[role_name]}"
                               f"::seed{seed}"))
                prev_equity = info.get("equity")
            if steps > 1_000_000:
                raise RuntimeError("outer validation replay exceeded "
                                   "the step cap")
        expected_scored_steps = int(role["scored_rows"])
        if scored_steps != expected_scored_steps:
            raise RuntimeError(
                f"{role_name} replay scored "
                f"{scored_steps} steps but the verified manifest declares "
                f"{expected_scored_steps}; refusing an off-by-one or "
                "truncated final evaluation")
        base = env
        while hasattr(base, "env") and not hasattr(base, "summary"):
            base = base.env
        summary = base.summary() if hasattr(base, "summary") else {}
        metrics = canonical_weekly_metrics_from_trace(
            trace_rows,
            initial_cash=float(eval_config.get("initial_cash",
                                               10_000.0)),
            risk_penalty_lambda=float(
                eval_config.get("risk_penalty_lambda", 1.0)),
            metric_schema="trading.weekly.v1")
        weekly_rows = metrics.pop("weekly_rows")
        trades_total = summary.get("trades_total",
                                   summary.get("trades"))
        is_best = artifact_role == "best_checkpoint"
        evaluated_sha = _sha_file(Path(model_path))
        return {
            "role": role_name,
            "solvency_mode": solvency_mode,
            "purpose": (
                "final truth ONLY — one evaluation after selection "
                "(finding 224/226)" if is_best else
                "DIAGNOSTIC TRUTH ONLY — one evaluation of the "
                "inactive cell's terminal policy; never a selection, "
                "never a promotion (finding 232)"),
            "csv_sha256": role["csv_sha256"],
            "scored_rows": role["scored_rows"],
            "context_rows_forced_hold": context_rows,
            "context_excluded_from_metrics": True,
            "score_start": role["score_start"],
            "score_end": role["score_end"],
            "evaluated_artifact_role": artifact_role,
            "evaluated_model_path": str(Path(model_path).resolve()),
            "evaluated_model_sha256": evaluated_sha,
            "best_model_path": (str(Path(model_path).resolve())
                                if is_best else None),
            "best_model_sha256": evaluated_sha if is_best else None,
            "promotion_eligible": is_best,
            "diagnostic_only": not is_best,
            "scored_steps": scored_steps,
            "metrics": metrics,
            "weekly_return_vector": [row["return_fraction"]
                                     for row in weekly_rows],
            "trades_total": trades_total,
            "activity": {
                "traded": bool(trades_total) and int(trades_total) > 0,
                "trades_total": trades_total,
            },
            "units": dict(DECISION_METRIC_UNITS),
        }
    finally:
        try:
            plug.close()
        except Exception:
            pass


DECISION_METRIC_UNITS = {
    "weekly_return_vector": ("fraction per calendar week (W-SUN), "
                             "outer validation 2024 scored window"),
    "mean_weekly_return": "fraction per week, arithmetic mean",
    "annualized_return": ("compounded fraction per year: "
                          "(1+total_return)^(365.25/days)-1 over the "
                          "outer scored window"),
    "annual_return": "fraction per year, weekly arithmetic mean x 52",
    "mean_weekly_rap": ("fraction per week: weekly return - lambda x "
                        "weekly max drawdown (risk-adjusted "
                        "performance)"),
    "annual_rap": "fraction per year, weekly RAP arithmetic mean x 52",
    "max_drawdown_fraction": ("fraction of peak equity over the whole "
                              "outer scored window"),
    "trades_total": "count of closed trades, outer scored window",
    "activity.traded": "boolean: at least one closed trade",
}


# ---------------------------------------------------------------------------
# screen aggregation: --screen-verdict
# ---------------------------------------------------------------------------

def _discover_experiment_dir(root: Path,
                             experiment_id: str | None) -> Path:
    if experiment_id:
        return root / experiment_id
    candidates = sorted(
        d for d in root.iterdir()
        if d.is_dir() and len(d.name) == 16
        and all(c in "0123456789abcdef" for c in d.name)
    ) if root.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly ONE experiment identity under {root}; "
            f"found {[d.name for d in candidates]} — pass "
            "--experiment-id explicitly")
    return candidates[0]


def expected_terminal_relative(record: dict) -> str | None:
    """The deterministic seal-relative terminal path a replica proof
    entry must echo: seed<seed>/<cell>/<tail after the cell component
    of the record's absolute terminal path>. None when underivable."""
    try:
        seed = int(record["seed"])
        cell = str(record["cell"])
        parts = Path(str(record["terminal_model_path"])).parts
    except (KeyError, TypeError, ValueError):
        return None
    indices = [i for i, part in enumerate(parts) if part == cell]
    if not indices:
        return None
    tail = parts[indices[-1] + 1:]
    if not tail:
        return None
    return str(Path(f"seed{seed}") / cell / Path(*tail))


def load_replica_proof(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("replica proof is not a JSON object")
    return payload


def validate_replica_proof(proof: dict | None, *, contract: dict,
                           records: dict) -> tuple:
    """Finding 225: the typed 16-entry replica proof is a MANDATORY
    verdict input. Returns (loads_ok: bool, refusals: list,
    facts: dict). Every entry must bind (experiment_identity,
    contract_sha256, seed, cell, terminal_relative_path,
    terminal_model_sha256, loads=true) to the exact cell record —
    missing, duplicate, foreign, swapped, hash-altered or unloaded
    entries refuse; loads is a strict boolean, never text."""
    refusals: list = []
    facts: dict = {"entries_present": 0, "entries_expected": 16}
    if proof is None:
        return False, [
            "replica proof required: --replica-proof PATH (typed "
            "p1lr_replica_proof.v1 from tools/p1lr_collect.py) — the "
            "screen cannot be declared eligible without 16 bound "
            "replica terminal loads (finding 225)"], facts
    if proof.get("schema") != REPLICA_PROOF_SCHEMA:
        refusals.append(f"replica proof schema {proof.get('schema')!r}"
                        f" != {REPLICA_PROOF_SCHEMA!r}")
    identities = {record.get("experiment_identity")
                  for record in records.values()}
    exp_id = (sorted(identities)[0] if len(identities) == 1 else None)
    if proof.get("experiment_identity") != exp_id or exp_id is None:
        refusals.append(
            f"replica proof experiment_identity "
            f"{proof.get('experiment_identity')!r} does not equal the "
            f"records' single identity {exp_id!r}")
    if proof.get("contract_sha256") != contract["_contract_sha256"]:
        refusals.append("replica proof binds a different contract sha")
    entries = proof.get("proofs")
    if not isinstance(entries, list):
        return False, refusals + [
            "replica proof carries no 'proofs' list"], facts
    facts["entries_present"] = len(entries)
    expected_keys = {(seed, cell) for (seed, cell) in records}
    seen: dict = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            refusals.append(f"proof entry {position} is not an object")
            continue
        try:
            key = (int(entry.get("seed")), str(entry.get("cell")))
        except (TypeError, ValueError):
            refusals.append(f"proof entry {position} carries unusable "
                            "seed/cell")
            continue
        if key not in expected_keys:
            refusals.append(f"foreign proof entry for seed={key[0]} "
                            f"cell={key[1]!r}")
            continue
        if key in seen:
            refusals.append(f"duplicate proof entry for seed={key[0]} "
                            f"cell={key[1]}")
            continue
        seen[key] = entry
        record = records[key]
        if entry.get("experiment_identity") != \
                record.get("experiment_identity"):
            refusals.append(f"seed={key[0]} cell={key[1]}: proof "
                            "entry experiment_identity mismatch")
        if entry.get("contract_sha256") != \
                contract["_contract_sha256"]:
            refusals.append(f"seed={key[0]} cell={key[1]}: proof "
                            "entry contract_sha256 mismatch")
        if entry.get("terminal_model_sha256") != \
                record.get("terminal_model_sha256"):
            refusals.append(
                f"seed={key[0]} cell={key[1]}: proof terminal sha "
                "does not equal the record's terminal_model_sha256 "
                "(hash-altered or swapped)")
        expected_rel = expected_terminal_relative(record)
        if expected_rel is None or \
                entry.get("terminal_relative_path") != expected_rel:
            refusals.append(
                f"seed={key[0]} cell={key[1]}: proof relative path "
                f"{entry.get('terminal_relative_path')!r} != expected "
                f"{expected_rel!r}")
        if entry.get("loads") is not True:
            refusals.append(
                f"seed={key[0]} cell={key[1]}: replica terminal did "
                f"not load (loads={entry.get('loads')!r})")
    missing = sorted(expected_keys - set(seen))
    for seed, cell in missing:
        refusals.append(f"replica proof has NO entry for seed={seed} "
                        f"cell={cell}")
    facts["entries_bound"] = len(seen)
    facts["replica_host"] = proof.get("replica_host")
    facts["collection_tree_digest"] = proof.get(
        "collection_tree_digest")
    return not refusals, refusals, facts


def _revalidate_record_evidence(record: dict, contract: dict,
                                tag: dict) -> tuple:
    """Aggregation-time revalidation (finding 225): the finding-221
    per-checkpoint handoff facts AND the finding-224 nested split
    identity of every record — never trusted from the producing run."""
    checkpoint_failures: list = []
    nested_failures: list = []
    viability = record.get("handoff_viability") or {}
    per_checkpoint = viability.get("per_checkpoint")
    if not isinstance(per_checkpoint, list) or not per_checkpoint:
        checkpoint_failures.append(
            {**tag, "error": "per-checkpoint handoff facts missing "
             "(finding 221 aggregation-time revalidation)"})
    else:
        for row in per_checkpoint:
            if not isinstance(row, dict) or row.get(
                    "handoff_viability") not in \
                    HANDOFF_VIABILITY_VALUES:
                checkpoint_failures.append(
                    {**tag, "error": f"per-checkpoint row "
                     f"{row!r} carries no typed handoff_viability"})
                break
    nested_spec = contract["nested_split_contract"]
    if record.get("nested_split_contract_sha256") != \
            nested_spec["sha256"]:
        nested_failures.append(
            {**tag, "error": "record nested_split_contract_sha256 "
             "does not equal the contract pin (finding 224)"})
    if record.get("selection_metric") != NESTED_SELECTION_METRIC:
        nested_failures.append(
            {**tag, "error": f"record selection_metric "
             f"{record.get('selection_metric')!r} is not the paired "
             "comparator (finding 224)"})
    role_facts = record.get("nested_role_facts") or {}
    pins = nested_spec["role_facts"]
    for role in _nested_splits.ROLES:
        got = role_facts.get(role) or {}
        pin = pins.get(role) or {}
        for key in NESTED_ROLE_FACT_KEYS:
            if got.get(key) != pin.get(key):
                nested_failures.append(
                    {**tag, "error": f"nested role {role}.{key} "
                     f"{got.get(key)!r} != pinned {pin.get(key)!r} "
                     "(finding 224)"})
                break
    return checkpoint_failures, nested_failures


def record_activity_status(record: dict) -> str:
    """Aggregation-time activity status of ONE record (finding 232).
    Prefers the typed field; falls back to the pipeline's declared
    inactive flag for records written before the field existed. Never
    guesses from performance."""
    status = record.get("activity_status")
    if status in ACTIVITY_STATUSES:
        return status
    if bool(record.get(
            "activity_stopped_without_eligible_checkpoint", False)):
        return "inactive"
    if record.get("best_model_path"):
        return "active"
    return "active"


def classify_factorial_activity(inactive_count: int,
                                expected_count: int) -> str:
    """FULL_ACTIVITY / PARTIAL_ACTIVITY_SURVIVAL /
    TOTAL_ACTIVITY_COLLAPSE over the 16-cell factorial."""
    if inactive_count <= 0:
        return "FULL_ACTIVITY"
    if inactive_count >= expected_count:
        return "TOTAL_ACTIVITY_COLLAPSE"
    return "PARTIAL_ACTIVITY_SURVIVAL"


def _load_records_from_disk(contract: dict, expected: list,
                            records_root: Path | None,
                            experiment_id: str | None,
                            mode: str) -> tuple:
    root = Path(records_root
                or output_root_for_mode(contract, mode))
    exp_dir = _discover_experiment_dir(root, experiment_id)
    records: dict = {}
    for seed, cell in expected:
        path = exp_dir / f"seed{seed}" / cell / "cell_record.json"
        if path.is_file():
            try:
                records[(seed, cell)] = json.loads(path.read_text())
            except Exception:
                records[(seed, cell)] = {"_unreadable": str(path)}
    return records, exp_dir


def screen_verdict(contract: dict, *, records: dict | None = None,
                   records_root: Path | None = None,
                   experiment_id: str | None = None,
                   replica_proof: dict | None = None) -> tuple:
    """Evaluate the contract's mechanics_screen.requires gates over all
    16 (seed, cell) records and emit the typed screen outcome. Returns
    (payload, exit_code). Collapse/contract screen ONLY — no
    performance claim is computed, compared or implied. The typed
    replica proof is MANDATORY (finding 225): without it, or with any
    unbound/duplicate/foreign/altered/unloaded entry, the verdict is a
    typed refusal and ``replica_terminal_loads`` stays boolean false."""
    expected = [(seed, cell) for seed in SEEDS
                for cell in contract["cell_order"][str(seed)]]

    if records is None:
        try:
            records, _exp_dir = _load_records_from_disk(
                contract, expected, records_root, experiment_id,
                "screen")
        except RuntimeError as exc:
            return ({"schema": VERDICT_SCHEMA,
                     "outcome": "SCREEN_REFUSED",
                     "reasons": [str(exc)]},
                    EXIT_CLASS["SCREEN_REFUSED"])

    reasons: list = []
    missing = [{"seed": seed, "cell": cell}
               for seed, cell in expected
               if (seed, cell) not in records]
    if missing:
        reasons.append(
            f"records incomplete: {len(records)}/16 present — the "
            "screen refuses to conclude on a partial factorial")

    malformed: list = []
    custody_failures: list = []
    viability_failures: list = []
    checkpoint_failures: list = []
    nested_identity_failures: list = []
    identity_set: set = set()
    contract_sha_mismatch: list = []
    labels: dict = {}
    activity_failures: list = []
    inactive_cells: list = []
    active_cells: list = []
    for (seed, cell), record in sorted(records.items()):
        tag = {"seed": seed, "cell": cell}
        if record.get("schema") != RECORD_SCHEMA \
                or record.get("seed") != seed \
                or record.get("cell") != cell \
                or not record.get("cell_identity"):
            malformed.append({**tag, "error": "wrong schema or "
                              "seed/cell/identity mismatch"})
            continue
        identity_set.add(record.get("experiment_identity"))
        # Finding 232: an inactive cell is a MEASURED screen outcome —
        # it is counted and named, never a refusal — but it must carry
        # its non-promotable custody.
        status = record_activity_status(record)
        if status == "inactive":
            inactive_cells.append({
                **tag,
                "cause": record.get("activity_inactive_cause"),
                "termination_cause": record.get("termination_cause"),
                "terminal_model_sha256": record.get(
                    "terminal_model_sha256"),
            })
            if record.get("best_model_path") or record.get(
                    "promotion_eligible") is True:
                activity_failures.append(
                    {**tag, "error": "inactive record carries a best "
                     "checkpoint or promotion_eligible=true "
                     "(finding 232)"})
        else:
            active_cells.append(tag)
        # Finding 225: revalidate per-checkpoint handoff facts and
        # nested split identity AT AGGREGATION TIME.
        cp_fail, nested_fail = _revalidate_record_evidence(
            record, contract, tag)
        checkpoint_failures.extend(cp_fail)
        nested_identity_failures.extend(nested_fail)
        if record.get("contract_sha256") != \
                contract["_contract_sha256"]:
            contract_sha_mismatch.append(tag)
        # Finding 223: terminal custody fields are mandatory; when the
        # artifact is reachable here it must also rehash-match.
        terminal_path = record.get("terminal_model_path")
        terminal_sha = record.get("terminal_model_sha256")
        if not terminal_path or not terminal_sha:
            custody_failures.append(
                {**tag, "error": "missing terminal_model_path/"
                 "terminal_model_sha256 (finding 223)"})
        else:
            local = Path(terminal_path)
            if local.is_file() and _sha_file(local) != terminal_sha:
                custody_failures.append(
                    {**tag, "error": "terminal artifact rehash "
                     "mismatch (finding 223)"})
        # Finding 221: direct handoff-viability facts, typed.
        viability = record.get("handoff_viability") or {}
        selected = viability.get("selected") or {}
        label = selected.get("handoff_viability")
        if label not in HANDOFF_VIABILITY_VALUES:
            viability_failures.append(
                {**tag, "error": f"selected handoff_viability "
                 f"{label!r} is not a typed label (finding 221)"})
            continue
        if selected.get("trained_treatment") is not True:
            viability_failures.append(
                {**tag, "error": "selected checkpoint is not a "
                 "trained treatment (finding 221)"})
            continue
        if label == "UNAVAILABLE":
            viability_failures.append(
                {**tag, "error": "selected handoff viability is "
                 "UNAVAILABLE — direct facts required before any "
                 "screen conclusion"})
            continue
        labels[(seed, cell)] = label

    if malformed:
        reasons.append(f"{len(malformed)} malformed record(s)")
    if len(identity_set) > 1:
        reasons.append(
            f"identity fragmentation: records span experiment "
            f"identities {sorted(identity_set)}")
    if contract_sha_mismatch:
        reasons.append(
            f"{len(contract_sha_mismatch)} record(s) bind a different "
            "contract sha than the loaded contract")
    if custody_failures:
        reasons.append(
            f"{len(custody_failures)} record(s) fail the finding-223 "
            "terminal-custody gate")
    if viability_failures:
        reasons.append(
            f"{len(viability_failures)} record(s) fail the "
            "finding-221 handoff-viability gate")
    if checkpoint_failures:
        reasons.append(
            f"{len(checkpoint_failures)} record(s) fail the "
            "finding-221 per-checkpoint revalidation at aggregation "
            "time")
    if nested_identity_failures:
        reasons.append(
            f"{len(nested_identity_failures)} record(s) fail the "
            "finding-224 nested split identity revalidation")
    if activity_failures:
        reasons.append(
            f"{len(activity_failures)} record(s) fail the finding-232 "
            "inactive-cell custody gate")

    # Finding 225: the replica gate is a BOOLEAN derived from the
    # typed proof — never explanatory text, never assumed.
    replica_ok, replica_refusals, replica_facts = \
        validate_replica_proof(replica_proof, contract=contract,
                               records=records)
    if replica_refusals:
        reasons.extend(replica_refusals)

    # Finding 232: the screen NAMES the activity picture of the 16
    # cells alongside its collapse verdict — all inactive, some
    # inactive (exactly which), or all active.
    activity_classification = classify_factorial_activity(
        len(inactive_cells), len(expected))
    activity_block = {
        "classification": activity_classification,
        "active_cells": len(active_cells),
        "inactive_cells": len(inactive_cells),
        "cells_expected": len(expected),
        "inactive_cell_facts": inactive_cells,
        "imputation": ("none — an inactive cell is reported as a "
                       "measured outcome; no performance value is "
                       "invented for it (finding 232)"),
    }

    gates = {
        "records_16_16": not missing and not malformed,
        "identity_coherent": len(identity_set) <= 1
        and not contract_sha_mismatch,
        "terminal_custody_fields": not custody_failures,
        "handoff_viability_facts": not viability_failures,
        "per_checkpoint_facts_revalidated": not checkpoint_failures,
        "nested_split_identity_revalidated":
            not nested_identity_failures,
        "inactive_cells_non_promotable": not activity_failures,
        "replica_terminal_loads": bool(replica_ok),
    }

    if reasons:
        payload = {
            "schema": VERDICT_SCHEMA,
            "outcome": "SCREEN_REFUSED",
            "experiment_identity": (sorted(identity_set)[0]
                                    if len(identity_set) == 1 else None),
            "contract_sha256": contract["_contract_sha256"],
            "records_present": len(records),
            "records_expected": len(expected),
            "missing_records": missing,
            "malformed_records": malformed,
            "custody_failures": custody_failures,
            "viability_failures": viability_failures,
            "checkpoint_failures": checkpoint_failures,
            "nested_identity_failures": nested_identity_failures,
            "contract_sha_mismatch": contract_sha_mismatch,
            "activity_failures": activity_failures,
            "activity": activity_block,
            "replica_proof_refusals": replica_refusals,
            "replica_proof_facts": replica_facts,
            "gates": gates,
            "reasons": reasons,
        }
        return payload, EXIT_CLASS["SCREEN_REFUSED"]

    # Contract text (mechanics_screen): collapse = the selected trained
    # checkpoint's handoff_viability in {CONSTANT_POLICY,
    # BELOW_NORMAL_THRESHOLD}; PHASE1_LR_REGION_COLLAPSED iff ALL FOUR
    # treatment combinations collapse at ALL FOUR seeds.
    matrix = {cell: {str(seed): labels[(seed, cell)]
                     for seed in SEEDS} for cell in CELLS}
    collapsed_cells = [
        {"seed": seed, "cell": cell, "handoff_viability": label}
        for (seed, cell), label in sorted(labels.items())
        if label in COLLAPSE_LABELS]
    viable_cells = [
        {"seed": seed, "cell": cell, "handoff_viability": label}
        for (seed, cell), label in sorted(labels.items())
        if label not in COLLAPSE_LABELS]
    all_collapsed = len(collapsed_cells) == len(expected)
    outcome = ("PHASE1_LR_REGION_COLLAPSED" if all_collapsed
               else "SCREEN_VIABLE_REGION")
    payload = {
        "schema": VERDICT_SCHEMA,
        "outcome": outcome,
        "experiment_identity": sorted(identity_set)[0],
        "contract_sha256": contract["_contract_sha256"],
        "records_present": len(records),
        "records_expected": len(expected),
        "gates": gates,
        "activity": activity_block,
        "replica_proof_facts": replica_facts,
        "collapse_definition": (
            "selected trained-checkpoint handoff_viability in "
            f"{list(COLLAPSE_LABELS)} (contract mechanics_screen; "
            "finding 221 typed labels)"),
        "viability_matrix": matrix,
        "collapsed_cells": collapsed_cells,
        "viable_cells": viable_cells,
        "next_step": (
            "stop — do not burn the decision budget (contract "
            "collapse_outcome)" if all_collapsed else
            "decision run eligible: pass THIS verdict file as "
            "--screen-gate to --mode decision (the 16-entry replica "
            "proof is already bound into replica_terminal_loads)"),
        "performance_claims": (
            "none — collapse/contract screen only; no return, Sharpe "
            "or superiority statement is made or implied"),
    }
    return payload, EXIT_CLASS[outcome]


# ---------------------------------------------------------------------------
# decision aggregation: --decision-verdict (finding 226, document 38)
# ---------------------------------------------------------------------------

def _finite_or_none(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    import math
    return value if math.isfinite(value) else None


def decision_effects(utilities: dict,
                     unavailable_reasons: dict | None = None) -> dict:
    """Per-seed paired main effects and interaction on one declared
    utility (outer mean weekly RAP). ``utilities`` maps
    (seed, cell) -> float and contains ONLY active cells.

    Finding 232: a seed whose four cells are not all active yields a
    TYPED UNAVAILABLE entry naming the exact missing cells and their
    causes. No zero and no sentinel is ever imputed for a missing
    active pair — an effect that cannot be computed is not a number."""
    unavailable_reasons = unavailable_reasons or {}
    per_seed: dict = {}
    for seed in SEEDS:
        absent = [cell for cell in CELLS
                  if (seed, cell) not in utilities]
        if absent:
            per_seed[str(seed)] = {
                "available": False,
                "unavailable_cells": [
                    {"cell": cell,
                     "reason": unavailable_reasons.get(
                         (seed, cell),
                         "no active-cell utility for this cell")}
                    for cell in absent],
                "reason": (
                    f"seed {seed} has no computable paired effect: "
                    f"{len(absent)} of 4 cells produced no comparable "
                    f"active-cell performance ({', '.join(absent)}); "
                    "imputing zero or a sentinel would fabricate the "
                    "comparison (finding 232)"),
                "phase1_lr_effect": None,
                "phase1_difficulty_effect": None,
                "lr_x_difficulty_interaction": None,
            }
            continue
        u = {cell: float(utilities[(seed, cell)]) for cell in CELLS}
        lr_effect = 0.5 * ((u["P1N_LR3E5"] - u["P1N_LR1E4"])
                           + (u["P1E_LR3E5"] - u["P1E_LR1E4"]))
        difficulty_effect = 0.5 * ((u["P1E_LR1E4"] - u["P1N_LR1E4"])
                                   + (u["P1E_LR3E5"] - u["P1N_LR3E5"]))
        interaction = ((u["P1E_LR3E5"] - u["P1E_LR1E4"])
                       - (u["P1N_LR3E5"] - u["P1N_LR1E4"]))
        per_seed[str(seed)] = {
            "available": True,
            "phase1_lr_effect": lr_effect,
            "phase1_difficulty_effect": difficulty_effect,
            "lr_x_difficulty_interaction": interaction,
        }
    return per_seed


def _effect_material(values: list) -> bool:
    """Declared materiality rule: all four per-seed paired effects
    share ONE strict sign (sign-consistent replication)."""
    return (all(v > 0.0 for v in values)
            or all(v < 0.0 for v in values))


def decide_decision_outcome(per_seed_effects: dict,
                            activity_classification: str,
                            inactive_cells: list | None = None) -> tuple:
    """Document-38 outcome from per-seed paired effects. Declared
    priority: TOTAL_ACTIVITY_COLLAPSE, then PARTIAL_ACTIVITY_SURVIVAL
    (typed INCONCLUSIVE — the paired comparison does not exist), then a
    sign-consistent interaction, then the sign-consistent main effect
    with the larger median magnitude, then NO_MATERIAL_EFFECT."""
    from statistics import median
    inactive_cells = inactive_cells or []
    if activity_classification not in ACTIVITY_CLASSIFICATIONS:
        raise ValueError("unknown activity classification "
                         f"{activity_classification!r}")
    if activity_classification == "TOTAL_ACTIVITY_COLLAPSE":
        return "TOTAL_ACTIVITY_COLLAPSE", (
            "every one of the 16 decision cells is inactive: no cell "
            "produced a policy that closed a trade on its final outer "
            "evaluation")
    if activity_classification == "PARTIAL_ACTIVITY_SURVIVAL":
        named = ", ".join(
            f"seed{entry['seed']}/{entry['cell']}"
            f"({entry.get('cause') or 'inactive'})"
            for entry in inactive_cells)
        unavailable = sorted(
            seed for seed, facts in per_seed_effects.items()
            if not facts.get("available"))
        return "INCONCLUSIVE", (
            "partial activity survival: "
            f"{len(inactive_cells)} of 16 cells are inactive "
            f"[{named}], so the paired effect(s) for seed(s) "
            f"{unavailable} do not exist. No zero and no sentinel is "
            "imputed for a missing active pair; the surviving per-seed "
            "effects are reported as computed and the document-38 "
            "decision is withheld (finding 232)")
    lr = [e["phase1_lr_effect"] for e in per_seed_effects.values()]
    diff = [e["phase1_difficulty_effect"]
            for e in per_seed_effects.values()]
    inter = [e["lr_x_difficulty_interaction"]
             for e in per_seed_effects.values()]
    if _effect_material(inter):
        return "PHASE1_LR_DIFFICULTY_INTERACTION", (
            "the LR x difficulty interaction is sign-consistent "
            f"across all four seeds (median {median(inter):+.6g})")
    lr_material = _effect_material(lr)
    diff_material = _effect_material(diff)
    if lr_material and diff_material:
        if abs(median(lr)) >= abs(median(diff)):
            return "PHASE1_LR_MAIN_EFFECT", (
                "both main effects are sign-consistent; the P1 LR "
                f"effect has the larger median magnitude "
                f"({median(lr):+.6g} vs {median(diff):+.6g}) — both "
                "are reported in per_seed_paired_effects")
        return "PHASE1_DIFFICULTY_MAIN_EFFECT", (
            "both main effects are sign-consistent; the difficulty "
            f"effect has the larger median magnitude "
            f"({median(diff):+.6g} vs {median(lr):+.6g}) — both are "
            "reported in per_seed_paired_effects")
    if lr_material:
        return "PHASE1_LR_MAIN_EFFECT", (
            "the P1 LR paired effect is sign-consistent across all "
            f"four seeds (median {median(lr):+.6g})")
    if diff_material:
        return "PHASE1_DIFFICULTY_MAIN_EFFECT", (
            "the P1 difficulty paired effect is sign-consistent "
            f"across all four seeds (median {median(diff):+.6g})")
    return "NO_MATERIAL_EFFECT", (
        "no paired effect is sign-consistent across the four seeds "
        f"(medians: lr {median(lr):+.6g}, difficulty "
        f"{median(diff):+.6g}, interaction {median(inter):+.6g})")


def decision_verdict(contract: dict, *, records: dict | None = None,
                     records_root: Path | None = None,
                     experiment_id: str | None = None,
                     replica_proof: dict | None = None) -> tuple:
    """Aggregate the 16 DECISION records and emit the document-38
    outcome with per-seed paired main effects (difficulty, P1 LR) plus
    interaction and per-cell raw weekly metrics, all with units and
    horizons. Fail-closed: the same 16/16, identity, custody,
    per-checkpoint, nested-identity and replica-proof gates as the
    screen, PLUS a mandatory final outer evaluation on every record.
    Any refusal is typed INCONCLUSIVE with exit 4."""
    expected = [(seed, cell) for seed in SEEDS
                for cell in contract["cell_order"][str(seed)]]
    if records is None:
        try:
            records, _exp_dir = _load_records_from_disk(
                contract, expected, records_root, experiment_id,
                "decision")
        except RuntimeError as exc:
            return ({"schema": DECISION_VERDICT_SCHEMA,
                     "outcome": "INCONCLUSIVE",
                     "reasons": [str(exc)]},
                    EXIT_CLASS["INCONCLUSIVE"])

    reasons: list = []
    missing = [{"seed": seed, "cell": cell}
               for seed, cell in expected
               if (seed, cell) not in records]
    if missing:
        reasons.append(f"records incomplete: {len(records)}/16 — the "
                       "decision refuses a partial factorial")
    identity_set: set = set()
    utilities: dict = {}
    per_cell_metrics: dict = {}
    inactive_cells: list = []
    inactive_reasons: dict = {}
    for (seed, cell), record in sorted(records.items()):
        tag = {"seed": seed, "cell": cell}
        name = f"seed{seed}/{cell}"
        if record.get("schema") != RECORD_SCHEMA \
                or record.get("seed") != seed \
                or record.get("cell") != cell:
            reasons.append(f"{name}: wrong schema or seed/cell")
            continue
        identity_set.add(record.get("experiment_identity"))
        if record.get("mode") != "decision" or \
                record.get("evidence_class") != "decision_run":
            reasons.append(f"{name}: not a decision_run record — "
                           "screen records never aggregate here")
        if record.get("contract_sha256") != \
                contract["_contract_sha256"]:
            reasons.append(f"{name}: contract sha mismatch")
        cp_fail, nested_fail = _revalidate_record_evidence(
            record, contract, tag)
        for failure in cp_fail + nested_fail:
            reasons.append(f"{name}: {failure['error']}")
        terminal_path = record.get("terminal_model_path")
        if not terminal_path or not record.get("terminal_model_sha256"):
            reasons.append(f"{name}: terminal custody incomplete "
                           "(finding 223)")
        outer = record.get("outer_validation_final")
        if not isinstance(outer, dict):
            reasons.append(f"{name}: no final outer-validation "
                           "evaluation — outer truth is mandatory "
                           "exactly once after selection (an INACTIVE "
                           "cell still receives one, of its terminal "
                           "artifact, as diagnostic truth)")
            continue
        # Finding 232: typed activity FIRST — an inactive cell is a
        # measured outcome whose diagnostic outer evaluation is
        # reported but NEVER enters the paired utility set.
        status = record_activity_status(record)
        artifact_role = outer.get("evaluated_artifact_role")
        metrics = outer.get("metrics") or {}
        trades = outer.get("trades_total")
        traded = bool(trades) and int(trades) > 0
        inactive_cause = None
        if status == "inactive":
            inactive_cause = (record.get("activity_inactive_cause")
                              or "no_activity_eligible_checkpoint")
            if record.get("promotion_eligible") is True or \
                    record.get("best_model_path"):
                reasons.append(
                    f"{name}: inactive record carries a best "
                    "checkpoint or promotion_eligible=true "
                    "(finding 232)")
            if artifact_role not in (None, "terminal"):
                reasons.append(
                    f"{name}: inactive cell's final outer evaluation "
                    f"declares artifact role {artifact_role!r} — an "
                    "inactive cell may only evaluate its terminal "
                    "artifact, diagnostically (finding 232)")
            if outer.get("best_model_path") or outer.get(
                    "best_model_sha256"):
                reasons.append(
                    f"{name}: inactive cell's outer evaluation binds a "
                    "best-checkpoint identity (finding 232)")
        elif not traded:
            # A selected checkpoint that closed zero trades on the one
            # final outer evaluation is equally non-comparable.
            status = "inactive"
            inactive_cause = "zero_trades_on_outer_validation"
        if status == "inactive":
            inactive_cells.append({
                **tag,
                "cause": inactive_cause,
                "termination_cause": record.get("termination_cause"),
                "trades_total": trades,
                "evaluated_artifact_role": artifact_role,
                "terminal_model_sha256": record.get(
                    "terminal_model_sha256"),
            })
            inactive_reasons[(seed, cell)] = (
                f"cell is inactive ({inactive_cause}); its outer "
                "evaluation is diagnostic truth about a "
                "non-promotable policy and is not a comparable "
                "performance value")
        else:
            utility = _finite_or_none(metrics.get("mean_weekly_rap"))
            if utility is None:
                reasons.append(f"{name}: outer mean_weekly_rap missing "
                               "or non-finite")
                continue
            if artifact_role not in (None, "best_checkpoint"):
                reasons.append(
                    f"{name}: active cell's final outer evaluation "
                    f"declares artifact role {artifact_role!r} — an "
                    "active cell's outer truth is its restored best "
                    "checkpoint (finding 232)")
            utilities[(seed, cell)] = utility
        per_cell_metrics[name] = {
            "activity_status": status,
            "inactive_cause": inactive_cause,
            "performance_eligible": status == "active",
            "evaluated_artifact_role": artifact_role,
            "weekly_return_vector": outer.get("weekly_return_vector"),
            "mean_weekly_return": metrics.get("mean_weekly_return"),
            "annualized_compounded_return": metrics.get(
                "annualized_return"),
            "annual_return_weekly_mean_x52": metrics.get(
                "annual_return"),
            "mean_weekly_rap": metrics.get("mean_weekly_rap"),
            "annual_rap_weekly_mean_x52": metrics.get("annual_rap"),
            "max_drawdown_fraction": metrics.get(
                "max_drawdown_fraction"),
            "evaluation_weeks": metrics.get("evaluation_weeks"),
            "trades_total": trades,
            "activity": outer.get("activity"),
            "units_and_horizons": dict(DECISION_METRIC_UNITS),
        }
    if len(identity_set) > 1:
        reasons.append(f"identity fragmentation: {sorted(identity_set)}")

    replica_ok, replica_refusals, replica_facts = \
        validate_replica_proof(replica_proof, contract=contract,
                               records=records)
    if replica_refusals:
        reasons.extend(replica_refusals)

    activity_classification = classify_factorial_activity(
        len(inactive_cells), len(expected))
    payload = {
        "schema": DECISION_VERDICT_SCHEMA,
        "contract_sha256": contract["_contract_sha256"],
        "experiment_identity": (sorted(identity_set)[0]
                                if len(identity_set) == 1 else None),
        "records_present": len(records),
        "records_expected": len(expected),
        "missing_records": missing,
        "gates": {
            "records_16_16": not missing,
            "identity_coherent": len(identity_set) <= 1,
            "replica_terminal_loads": bool(replica_ok),
        },
        "replica_proof_facts": replica_facts,
        "effect_basis": ("outer_validation mean_weekly_rap (fraction "
                         "per week, 2024 scored window); paired "
                         "within seed"),
        "materiality_rule": ("an effect is material iff all four "
                             "per-seed paired effects share one "
                             "strict sign"),
        # Finding 232: the three distinguishable activity outcomes.
        "activity_classification": activity_classification,
        "activity_classification_values": list(
            ACTIVITY_CLASSIFICATIONS),
        "active_cells": len(expected) - len(inactive_cells),
        "inactive_cells": inactive_cells,
        "imputation_policy": (
            "NONE — a cell without comparable active performance is "
            "excluded from the paired utility set and its per-seed "
            "effect is typed unavailable with the exact reason; zero "
            "and sentinel values are never imputed (finding 232)"),
        "per_cell_metrics": per_cell_metrics,
        "reasons": reasons,
    }
    if reasons:
        payload["outcome"] = "INCONCLUSIVE"
        payload["outcome_rationale"] = ("refusals precede evaluation: "
                                        + "; ".join(reasons[:10]))
        return payload, EXIT_CLASS["INCONCLUSIVE"]

    per_seed_effects = decision_effects(utilities, inactive_reasons)
    outcome, rationale = decide_decision_outcome(
        per_seed_effects, activity_classification, inactive_cells)
    payload["per_seed_paired_effects"] = per_seed_effects
    payload["paired_effects_available"] = sorted(
        seed for seed, facts in per_seed_effects.items()
        if facts.get("available"))
    payload["paired_effects_unavailable"] = sorted(
        seed for seed, facts in per_seed_effects.items()
        if not facts.get("available"))
    payload["outcome"] = outcome
    payload["outcome_rationale"] = rationale
    return payload, EXIT_CLASS[outcome]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# no-training materialization preflight
# ---------------------------------------------------------------------------

def preflight(contract: dict, bindings: dict | None = None) -> tuple:
    """No-training preflight (acceptance boundary): prove the exact
    nested role counts/hashes/score dates, paired selection, sealed
    2025 absence, distinct mode identities and 16 distinct cell
    identities per mode — by REAL materialization into a temp dir,
    without constructing any model. Returns (payload, exit_code)."""
    import tempfile

    bindings = bindings or load_bindings()
    payload: dict = {
        "schema": PREFLIGHT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract["_contract_sha256"],
        "held_fixed_bindings_contract_sha256":
            bindings["_contract_sha256"],
        "training_used": False,
        "refusals": [],
    }
    try:
        with tempfile.TemporaryDirectory(
                prefix="p1lr_preflight_") as scratch:
            nested_roles = materialize_nested_roles(
                contract, bindings, Path(scratch))
            payload["nested_split_contract_path"] = nested_roles[
                "binding"]["contract_relative_path"]
            payload["nested_split_contract_sha256"] = nested_roles[
                "binding"]["sha256"]
            payload["nested_split_mode"] = nested_roles["binding"][
                "mode"]
            payload["nested_context_bars"] = nested_roles["binding"][
                "context_bars"]
            payload["nested_role_facts"] = {
                role: {key: (nested_roles["roles"].get(role) or {})
                       .get(key) for key in NESTED_ROLE_FACT_KEYS}
                for role in _nested_splits.ROLES}
            sealed = nested_roles["roles"].get("sealed_test") or {}
            payload["sealed_test_state"] = sealed.get("status")
            sealed_csv = Path(scratch) / "nested_splits" / \
                "sealed_test.csv"
            payload["sealed_test_csv_absent"] = not sealed_csv.exists()
            if not payload["sealed_test_csv_absent"]:
                payload["refusals"].append(
                    "sealed_test.csv exists in the preflight "
                    "materialization — sealed 2025 was touched")

            sources = ladder.source_identities()
            identities: dict = {}
            for mode in MODES:
                exp_id = experiment_identity(contract, bindings,
                                             sources=sources, mode=mode)
                cell_ids = {}
                metrics_seen = set()
                sealed_flags = set()
                for seed in SEEDS:
                    for cell in contract["cell_order"][str(seed)]:
                        config = materialize_cell_config(
                            contract, bindings, seed, cell,
                            Path(scratch) / mode / str(seed) / cell,
                            mode=mode)
                        config.pop("_identity")
                        metrics_seen.add(config["selection_metric"])
                        sealed_flags.add(
                            bool(config["evaluate_test_split"]))
                        cell_ids[f"{seed}:{cell}"] = cell_identity(
                            exp_id, seed, cell, contract)
                if len(set(cell_ids.values())) != 16:
                    payload["refusals"].append(
                        f"mode {mode}: cell identities are not 16 "
                        "distinct values")
                identities[mode] = {
                    "experiment_identity": exp_id,
                    "output_root": str(output_root_for_mode(contract,
                                                            mode)),
                    "cell_identities": cell_ids,
                    "selection_metrics_materialized": sorted(
                        metrics_seen),
                    "evaluate_test_split_values": sorted(sealed_flags),
                }
                if metrics_seen != {NESTED_SELECTION_METRIC}:
                    payload["refusals"].append(
                        f"mode {mode}: materialized selection metric "
                        f"set {sorted(metrics_seen)} is not exactly "
                        f"the paired comparator")
                if sealed_flags != {False}:
                    payload["refusals"].append(
                        f"mode {mode}: evaluate_test_split leaked true")
            payload["modes"] = identities
            if identities["screen"]["experiment_identity"] == \
                    identities["decision"]["experiment_identity"]:
                payload["refusals"].append(
                    "screen and decision experiment identities "
                    "collide — modes are not content-addressed apart")
    except Exception as exc:  # noqa: BLE001 — typed refusal
        payload["refusals"].append(f"{type(exc).__name__}: {exc}")

    payload["outcome"] = ("PREFLIGHT_PASS" if not payload["refusals"]
                          else "PREFLIGHT_REFUSED")
    return payload, EXIT_CLASS[payload["outcome"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, choices=list(SEEDS),
                       help="run THIS seed's four cells sequentially "
                            "in the contract cell order on the seed's "
                            "assigned GPU")
    group.add_argument("--screen-verdict", action="store_true",
                       help="aggregate all 16 screen records and emit "
                            "the typed screen outcome (REQUIRES "
                            "--replica-proof, finding 225)")
    group.add_argument("--decision-verdict", action="store_true",
                       help="aggregate all 16 decision records and "
                            "emit the document-38 outcome (REQUIRES "
                            "--replica-proof)")
    group.add_argument("--preflight", action="store_true",
                       help="no-training materialization preflight: "
                            "prove nested role counts/hashes, paired "
                            "selection, sealed-test absence and 16 "
                            "distinct cell identities per mode")
    parser.add_argument("--mode", choices=list(MODES), default="screen",
                        help="execution mode for --seed: the default "
                             "'screen' preserves current behavior; "
                             "'decision' runs the document-38 path "
                             "under a distinct identity/output root "
                             "and REQUIRES --screen-gate")
    parser.add_argument("--screen-gate", type=Path, default=None,
                        help="decision mode: the corrected screen "
                             "verdict JSON (outcome "
                             "SCREEN_VIABLE_REGION with a passing "
                             "boolean replica gate)")
    parser.add_argument("--replica-proof", type=Path, default=None,
                        help="verdicts: the typed 16-entry replica "
                             "proof file from tools/p1lr_collect.py — "
                             "mandatory; absence is a typed refusal")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--no-gpu-check", action="store_true",
                        help="skip GPU-UUID binding enforcement and "
                             "the readiness launch gate (socket-free "
                             "tests only; a fleet launch MUST enforce)")
    parser.add_argument("--records-root", type=Path, default=None,
                        help="verdicts: records root override "
                             "(default: the mode's output root)")
    parser.add_argument("--experiment-id", default=None,
                        help="verdicts: experiment identity to "
                             "aggregate (required when several exist)")
    parser.add_argument("--output", type=Path, default=None,
                        help="verdicts/preflight: also write the "
                             "payload JSON here (atomic)")
    args = parser.parse_args()
    contract = load_contract(args.contract)

    def _emit(payload: dict, exit_code: int) -> int:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(args.output, payload)
        print(json.dumps(payload, default=str), flush=True)
        return exit_code

    if args.preflight:
        payload, exit_code = preflight(contract)
        return _emit(payload, exit_code)

    if args.screen_verdict or args.decision_verdict:
        proof = None
        if args.replica_proof is not None:
            proof = load_replica_proof(args.replica_proof)
        verdict_fn = (decision_verdict if args.decision_verdict
                      else screen_verdict)
        payload, exit_code = verdict_fn(
            contract, records_root=args.records_root,
            experiment_id=args.experiment_id, replica_proof=proof)
        return _emit(payload, exit_code)

    screen_gate = None
    if args.mode == "decision" and args.screen_gate is not None:
        screen_gate = json.loads(args.screen_gate.read_text())
    try:
        summary = run_seed(args.seed, contract=contract,
                           enforce_gpu=not args.no_gpu_check,
                           mode=args.mode, screen_gate=screen_gate)
    except Exception as exc:                        # noqa: BLE001
        print(json.dumps({"outcome": "SEED_FAILED", "seed": args.seed,
                          "error": f"{type(exc).__name__}: {exc}"},
                         default=str), flush=True)
        return EXIT_CLASS["SEED_FAILED"]
    print(json.dumps(summary, default=str), flush=True)
    return EXIT_CLASS[summary["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
