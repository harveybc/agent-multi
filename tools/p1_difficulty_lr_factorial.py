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
CONTRACT_SCHEMA = "agent_multi.p1_difficulty_lr_factorial.v1"
RECORD_SCHEMA = "agent_multi.p1_difficulty_lr_cell_record.v1"
VERDICT_SCHEMA = "agent_multi.p1_difficulty_lr_screen_verdict.v1"
HEARTBEAT_SCHEMA = "agent_multi.p1_difficulty_lr_heartbeat.v1"
HEARTBEAT_INTERVAL_S = 60

SEEDS = (101, 202, 303, 404)
CELLS = ("P1N_LR1E4", "P1N_LR3E5", "P1E_LR1E4", "P1E_LR3E5")
DYNAMICS_LEVELS = ("normal_realistic", "easy_chronological_continuation")
LR_LEVELS = (0.0001, 3e-05)

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

# Exit classes follow the fleet launcher/systemd contract.
EXIT_CLASS = {
    "SEED_COMPLETE": 0,
    "ALREADY_COMPLETE": 0,
    "SCREEN_VIABLE_REGION": 0,
    "PHASE1_LR_REGION_COLLAPSED": 0,
    "ALREADY_RUNNING": 3,
    "REFUSED_WRONG_HOST": 4,
    "REFUSED_GPU_UNBOUND": 4,
    "REFUSED_BAD_CONTRACT": 4,
    "REFUSED_ANCHOR_UNVERIFIED": 4,
    "SCREEN_REFUSED": 4,
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

    contract["_contract_sha256"] = _sha_file(Path(path))
    contract["_contract_path"] = str(path)
    return contract


def load_bindings(path: Path | None = None) -> dict:
    """The held-fixed recipe source: the SAME ladder contract the D0-D4
    diagnostic proved (data, splits, nested NON-TEST union, base
    config, plugins, frozen v3 cost manifest reference). Loaded through
    the ladder's own fail-closed loader."""
    return ladder.load_contract(path or ladder.CONTRACT_PATH)


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------

def experiment_identity(contract: dict, bindings: dict,
                        sources: dict | None = None) -> str:
    """sha256(p1lr contract sha + held-fixed ladder-contract sha + the
    four per-seed anchor shas + code identities + 'p1lr_mechanics_
    screen')[:16] — ONE identity for the whole screen; reuses no L1
    decision or ladder identity."""
    sources = sources or ladder.source_identities()
    payload = {
        "contract": contract["_contract_sha256"],
        "held_fixed_bindings_contract": bindings["_contract_sha256"],
        "anchors": {str(seed): contract["anchors"][str(seed)]["sha256"]
                    for seed in SEEDS},
        "code": {name: {"commit": s["commit"],
                        "dirty_untracked_digest":
                            s["dirty_untracked_digest"]}
                 for name, s in sorted(sources.items())},
        "profile": "p1lr_mechanics_screen",
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True).encode()).hexdigest()[:16]


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
                            cell: str, out_dir: Path) -> dict:
    if cell not in CELLS:
        raise ValueError(f"unknown factorial cell {cell!r}")
    if int(seed) not in SEEDS:
        raise ValueError(f"unknown factorial seed {seed!r}")
    common = bindings["common"]
    held = contract["held_fixed"]
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
            "screen epoch_timesteps does not match the ladder pass "
            "budget — the pass-equivalent claim would be false")

    union_facts = ladder.verify_nested_nontest_union(bindings)

    # --- the ladder-proven held-fixed recipe --------------------------
    config = json.loads(base_path.read_text())
    for field in ("train_years", "val_years", "test_years"):
        config.pop(field, None)
    config["split_contract_note"] = (
        "explicit split dates govern; year-count shorthand removed")
    splits = dict(common["splits"])
    if bool(splits.pop("evaluate_test_split", False)):
        raise RuntimeError("held-fixed contract enables the sealed "
                           "test split — refused")
    config.update(splits)
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
    config["selection_metric"] = str(common["selection_metric"])
    config["selection_min_trades"] = int(common["selection_min_trades"])
    config["quiet_mode"] = True
    config["save_model"] = str(Path(out_dir) / "model.zip")
    config["return_trace_dir"] = str(Path(out_dir) / "return_traces")
    # Screen budget: ONE pass-equivalent, no early stopping can fire.
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

    # Per-seed anchor: hash-bound, NEVER a preceding cell's terminal.
    anchor = contract["anchors"][str(seed)]
    config["warm_start_model"] = str(
        Path(anchor["path"]).expanduser())
    config["warm_start_model_sha256"] = anchor["sha256"]

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
        "nested_split_contract_sha256": common["nested_split_contract"][
            "sha256"],
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
                 cell: str, exp_id: str, cell_id: str):
        self.path = path
        self.contract = contract
        self.seed = int(seed)
        self.cell = cell
        self.exp_id = exp_id
        self.cell_id = cell_id
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
             tensor_sha_fn=None) -> dict:
    cell_id = cell_identity(exp_id, seed, cell, contract)
    out_root = Path(contract["output_root"]).expanduser()
    seed_dir = out_root / exp_id / f"seed{seed}"
    cell_dir = seed_dir / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = CellHeartbeat(cell_dir / "heartbeat.json",
                              contract=contract, seed=seed, cell=cell,
                              exp_id=exp_id, cell_id=cell_id)

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
                                         cell, out_dir)
        identity = config.pop("_identity")
        resolved_sha = sysid.resolved_config_sha256(config)
        agent_name = identity["plugins"]["agent_plugin"]
        agent = (agent_loader or ladder._agent_plugin)(agent_name)

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
        best_path = result.get("best_model_path")
        best_sha = (_sha_file(Path(best_path))
                    if best_path and Path(best_path).is_file() else None)

        # Finding 221: bind the typed handoff-viability evidence.
        viability = bind_handoff_viability(result)

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
        record = {
            "schema": RECORD_SCHEMA,
            "evidence_class": "mechanics_screen",
            "decision_eligible": False,
            "performance_aggregate_eligible": False,
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
            "termination_cause": result.get("termination_cause"),
            "activity_stopped_without_eligible_checkpoint": bool(
                result.get(
                    "activity_stopped_without_eligible_checkpoint",
                    False)),
            "boundary_transfer_evidence": result.get(
                "warm_start_transfer_evidence"),
            "anchor_sha256": actual_anchor,
            "best_model_path": best_path,
            "best_model_sha256": best_sha,
            "terminal_model_path": str(Path(terminal_path).resolve()),
            "terminal_model_sha256": terminal_sha,
            "terminal_policy_tensor_sha256": terminal_tensor,
            "curriculum": result.get("curriculum"),
        }
        atomic_write_json(record_path, record)
        heartbeat.beat(terminal_state="CELL_COMPLETE",
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

def run_seed(seed: int, *, contract: dict, bindings: dict | None = None,
             enforce_gpu: bool = True, pipeline_factory=None,
             agent_loader=None, tensor_sha_fn=None,
             gate_heartbeat: dict | None = None,
             dispatch_binding_fn=None) -> dict:
    if int(seed) not in SEEDS:
        raise ValueError(f"unknown factorial seed {seed!r}")
    bindings = bindings or load_bindings()
    sources_before = ladder.source_identities()
    exp_id = experiment_identity(contract, bindings,
                                 sources=sources_before)
    out_root = Path(contract["output_root"]).expanduser()
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
                tensor_sha_fn=tensor_sha_fn)
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
        outcomes[cell].update({
            "cell_identity": record.get("cell_identity"),
            "selected_handoff_viability": (record.get(
                "handoff_viability") or {}).get("selected_label"),
            "terminal_model_sha256": record.get(
                "terminal_model_sha256"),
        })

    states = {facts["outcome"] for facts in outcomes.values()}
    if "CELL_FAILED" in states:
        outcome = "SEED_FAILED"
    elif "ALREADY_RUNNING" in states:
        outcome = "ALREADY_RUNNING"
    elif reused == len(cells):
        outcome = "ALREADY_COMPLETE"
    else:
        outcome = "SEED_COMPLETE"
    return {
        "outcome": outcome,
        "experiment_identity": exp_id,
        "seed": int(seed),
        "cell_order": cells,
        "cells": outcomes,
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


def screen_verdict(contract: dict, *, records: dict | None = None,
                   records_root: Path | None = None,
                   experiment_id: str | None = None) -> tuple:
    """Evaluate the contract's mechanics_screen.requires gates over all
    16 (seed, cell) records and emit the typed screen outcome. Returns
    (payload, exit_code). Collapse/contract screen ONLY — no
    performance claim is computed, compared or implied."""
    expected = [(seed, cell) for seed in SEEDS
                for cell in contract["cell_order"][str(seed)]]

    exp_dir = None
    if records is None:
        root = Path(records_root
                    or Path(contract["output_root"]).expanduser())
        try:
            exp_dir = _discover_experiment_dir(root, experiment_id)
        except RuntimeError as exc:
            return ({"schema": VERDICT_SCHEMA,
                     "outcome": "SCREEN_REFUSED",
                     "reasons": [str(exc)]},
                    EXIT_CLASS["SCREEN_REFUSED"])
        records = {}
        for seed, cell in expected:
            path = exp_dir / f"seed{seed}" / cell / "cell_record.json"
            if path.is_file():
                try:
                    records[(seed, cell)] = json.loads(
                        path.read_text())
                except Exception:
                    records[(seed, cell)] = {"_unreadable": str(path)}

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
    identity_set: set = set()
    contract_sha_mismatch: list = []
    labels: dict = {}
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

    gates = {
        "records_16_16": not missing and not malformed,
        "identity_coherent": len(identity_set) <= 1
        and not contract_sha_mismatch,
        "terminal_custody_fields": not custody_failures,
        "handoff_viability_facts": not viability_failures,
        "replica_terminal_loads": (
            "EXTERNAL_COLLECTOR_REQUIRED — replica load proof per "
            "seed batch is the sealed collector's job (finding 223); "
            "this verdict does not certify it"),
    }

    if reasons:
        payload = {
            "schema": VERDICT_SCHEMA,
            "outcome": "SCREEN_REFUSED",
            "experiment_identity": (sorted(identity_set)[0]
                                    if len(identity_set) == 1 else None),
            "records_present": len(records),
            "records_expected": len(expected),
            "missing_records": missing,
            "malformed_records": malformed,
            "custody_failures": custody_failures,
            "viability_failures": viability_failures,
            "contract_sha_mismatch": contract_sha_mismatch,
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
            "decision run eligible under document-38 stopping for the "
            "viable region ONLY, after the sealed collector proves "
            "replica terminal loads (finding 223)"),
        "performance_claims": (
            "none — collapse/contract screen only; no return, Sharpe "
            "or superiority statement is made or implied"),
    }
    return payload, EXIT_CLASS[outcome]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, choices=list(SEEDS),
                       help="run THIS seed's four cells sequentially "
                            "in the contract cell order on the seed's "
                            "assigned GPU")
    group.add_argument("--screen-verdict", action="store_true",
                       help="aggregate all 16 records and emit the "
                            "typed screen outcome")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--no-gpu-check", action="store_true",
                        help="skip GPU-UUID binding enforcement and "
                             "the readiness launch gate (socket-free "
                             "tests only; a fleet launch MUST enforce)")
    parser.add_argument("--records-root", type=Path, default=None,
                        help="screen-verdict: records root override "
                             "(default: the contract output_root)")
    parser.add_argument("--experiment-id", default=None,
                        help="screen-verdict: experiment identity to "
                             "aggregate (required when several exist)")
    parser.add_argument("--output", type=Path, default=None,
                        help="screen-verdict: also write the verdict "
                             "JSON here (atomic)")
    args = parser.parse_args()
    contract = load_contract(args.contract)

    if args.screen_verdict:
        payload, exit_code = screen_verdict(
            contract, records_root=args.records_root,
            experiment_id=args.experiment_id)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(args.output, payload)
        print(json.dumps(payload, default=str), flush=True)
        return exit_code

    try:
        summary = run_seed(args.seed, contract=contract,
                           enforce_gpu=not args.no_gpu_check)
    except Exception as exc:                        # noqa: BLE001
        print(json.dumps({"outcome": "SEED_FAILED", "seed": args.seed,
                          "error": f"{type(exc).__name__}: {exc}"},
                         default=str), flush=True)
        return EXIT_CLASS["SEED_FAILED"]
    print(json.dumps(summary, default=str), flush=True)
    return EXIT_CLASS[summary["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
