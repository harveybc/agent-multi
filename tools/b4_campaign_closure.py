#!/usr/bin/env python3
"""R1 (order 2026-09-11): turn the authorized v7 stop into a VERIFIABLE
campaign adjudication.

The owner already authorized stopping and quarantining the third cell,
and amendment 16 recorded that decision in the design chain. What did
not exist was a campaign-level record that a reader can CHECK: the
ledger still called all twelve cells PENDING — including the two that
were sealed — so nothing on disk distinguished "finished", "stopped
mid-flight" and "never begun".

This tool produces that record. It re-derives everything from durable
bytes:

  * the two COMPLETED cells are verified BY DESCRIPTOR — exact terminal
    schema, cell identity, the ledger's cell-config digest, the per-bar
    CSV and its digest, the checkpoint and its digest, the claim that
    binds the terminal, and the seal;
  * the third cell is INVENTORIED in full — every file, its size and
    its digest — and only then classified. A cell is called partial
    because its artifacts say so, never because a directory exists;
  * the nine untouched cells are named, with the evidence of their
    absence;
  * costs are re-derived from durable intervals, not copied from the
    amendment that charged them.

Boundaries it holds:

  * it NEVER completes a cell by label. Nine cells without a terminal
    stay NOT_STARTED and the partial stays QUARANTINED_PARTIAL;
  * the partial contributes to no comparison, and this act makes
    neither completed cell promotable — both terminals already carry
    g1_eligible=false and checkpoint_promotable=false, and the closure
    asserts that rather than changing it;
  * it does not relaunch, and it proves the launch gate refuses BEFORE
    any accelerator library is imported;
  * it writes exactly one append-only record and proves, by digesting
    the whole tree before and after, that no training artifact moved.

    python tools/b4_campaign_closure.py --results-root <v7 root> \
        --materialization-root <v5 root> [--emit]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

CLOSURE_SCHEMA = "agent_multi.b4_campaign_closure.v1"
CLOSURE_LOG = "B4_CAMPAIGN_CLOSURE.jsonl"

COMPLETED = "COMPLETED_VERIFIED"
PARTIAL = "QUARANTINED_PARTIAL"
NOT_STARTED = "NOT_STARTED"

TERMINAL_SCHEMA = "agent_multi.b4_cell_terminal.v1"

#: the exact field set a v7 terminal carries. An extra field is a
#: different artifact, not a richer one.
TERMINAL_KEYS = frozenset({
    "schema", "cell", "terminal", "g1_eligible", "checkpoint_promotable",
    "attempt_id", "cell_config_sha256", "artifact_class",
    "checkpoint_sha256", "checkpoint_path", "per_bar_csv",
    "per_bar_sha256", "scored_index_sha256", "scored_bars",
    "counter_semantics", "sealed_2025_used", "wall_seconds",
    "effective_limits", "authorization_record_sha256",
    "amendment_11_sha256", "campaign_generation", "recovery_acta_sha256",
    "pinned_execution_commit", "latest_amendment_sha256"})

#: artifacts whose presence would contradict a partial classification.
COMPLETION_ARTIFACTS = ("B4_CELL_TERMINAL.json", "results.json")

#: fields that describe WHEN and HOW LONG, never WHAT. They stay in the
#: record and stay out of the adjudication identity, so re-running the
#: closure on unchanged evidence appends nothing.
VOLATILE_FOR_IDENTITY = frozenset({"closed_at", "measurement"})


#: how long the measurement took. Cost is worth reporting and is NOT
#: part of what was measured: two inventories of identical bytes are the
#: same observation whether one took 2 s and the other 4 s.
MEASUREMENT: dict = {"inventory_seconds": {}}


class ClosureRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()).hexdigest()


def utc_now() -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"))


def inventory(root: Path, *, exclude: tuple[str, ...] = ()) -> dict:
    """Every regular file under `root`, by digest and size."""
    out: dict[str, dict] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if rel in exclude:
            continue
        out[rel] = {"bytes": p.stat().st_size, "sha256": sha_file(p)}
    return out


# ------------------------------------------------------------ COMPLETED
def verify_completed_cell(results_root: Path, cell: str,
                          ledger_cell: dict) -> dict:
    """Descriptor verification of a sealed cell.

    Deliberately does NOT consult the launch gate. A closure is not a
    launch: requiring an open launch to READ finished evidence would
    make a campaign unclosable exactly when it most needs closing.
    """
    cell_dir = results_root / cell
    term_p = cell_dir / "B4_CELL_TERMINAL.json"
    if not term_p.is_file():
        raise ClosureRefusal(f"{cell}: no terminal to verify")
    term = json.loads(term_p.read_text())

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"descriptor": name, "verified": bool(ok),
                       "detail": detail})
        if not ok:
            raise ClosureRefusal(f"{cell}: {name} — {detail}")

    check("terminal_schema_exact", set(term) == TERMINAL_KEYS,
          f"unexpected={sorted(set(term) - TERMINAL_KEYS)} "
          f"missing={sorted(TERMINAL_KEYS - set(term))}")
    check("terminal_schema_name", term["schema"] == TERMINAL_SCHEMA,
          term["schema"])
    check("cell_identity", term["cell"] == cell, term["cell"])
    check("terminal_state", term["terminal"] == "COMPLETED",
          term["terminal"])
    check("cell_config_digest_binds_ledger",
          term["cell_config_sha256"] == ledger_cell["cell_config_sha256"],
          term["cell_config_sha256"])
    check("sealed_2025_absence_proved", term["sealed_2025_used"] is False,
          repr(term["sealed_2025_used"]))

    per_bar = Path(term["per_bar_csv"])
    if not per_bar.is_absolute():
        per_bar = cell_dir / per_bar.name
    check("per_bar_present", per_bar.is_file(), str(per_bar.name))
    check("per_bar_digest", sha_file(per_bar) == term["per_bar_sha256"],
          term["per_bar_sha256"])
    rows = sum(1 for _ in per_bar.open()) - 1
    check("per_bar_rows_match_scored_bars", rows == term["scored_bars"],
          f"rows={rows} scored_bars={term['scored_bars']}")

    ckpt = Path(term["checkpoint_path"])
    if not ckpt.is_absolute():
        ckpt = cell_dir / ckpt.name
    check("checkpoint_present", ckpt.is_file(), ckpt.name)
    check("checkpoint_digest", sha_file(ckpt) == term["checkpoint_sha256"],
          term["checkpoint_sha256"])

    claim_p = next(cell_dir.glob("CLAIM_*.json"), None)
    check("claim_present", claim_p is not None, str(claim_p))
    claim = json.loads(claim_p.read_text())
    check("claim_binds_this_attempt",
          claim.get("attempt_id") == term["attempt_id"],
          f"{claim.get('attempt_id')} vs {term['attempt_id']}")
    check("claim_binds_this_cell", claim.get("cell") == cell,
          str(claim.get("cell")))

    # The terminal is bound by the TWO-PHASE SEAL, not by the claim.
    # A claim is written when the attempt starts, so its
    # terminal_sha256 is null by construction; the intent names the
    # terminal it is about to seal and the completion binds that
    # intent. Checking the claim for the terminal digest would refuse
    # every correctly sealed cell — as it did here before I read the
    # records instead of assuming their shape.
    terminal_digest = sha_file(term_p)
    intent = cell_dir / f"SEAL_INTENT_{term['attempt_id']}.json"
    seal = cell_dir / f"SEAL_COMPLETE_{term['attempt_id']}.json"
    check("seal_intent_present", intent.is_file(), intent.name)
    check("seal_complete_present", seal.is_file(), seal.name)
    intent_doc = json.loads(intent.read_text())
    seal_doc = json.loads(seal.read_text())
    check("seal_intent_binds_terminal_digest",
          intent_doc.get("terminal_sha256") == terminal_digest,
          str(intent_doc.get("terminal_sha256")))
    check("seal_intent_binds_this_cell", intent_doc.get("cell") == cell,
          str(intent_doc.get("cell")))
    check("seal_complete_binds_the_intent",
          seal_doc.get("intent_sha256") == sha_file(intent),
          str(seal_doc.get("intent_sha256")))
    check("seal_complete_binds_the_same_terminal",
          seal_doc.get("terminal_sha256") == terminal_digest,
          str(seal_doc.get("terminal_sha256")))
    check("seal_complete_binds_this_attempt",
          seal_doc.get("attempt_id") == term["attempt_id"],
          str(seal_doc.get("attempt_id")))

    wall = term["wall_seconds"]
    check("wall_seconds_is_a_positive_measurement",
          isinstance(wall, (int, float)) and not isinstance(wall, bool)
          and wall > 0, repr(wall))

    # The closure ASSERTS the non-promotion the terminal already
    # carries. It has no authority to change it, and a closure that
    # quietly promoted a cell would be the worst possible artifact.
    check("not_g1_eligible", term["g1_eligible"] is False,
          repr(term["g1_eligible"]))
    check("checkpoint_not_promotable",
          term["checkpoint_promotable"] is False,
          repr(term["checkpoint_promotable"]))

    return {
        "cell": cell,
        "classification": COMPLETED,
        "attempt_id": term["attempt_id"],
        "terminal_sha256": terminal_digest,
        "seal_intent_sha256": sha_file(intent),
        "seal_sha256": sha_file(seal),
        "per_bar_sha256": term["per_bar_sha256"],
        "scored_bars": term["scored_bars"],
        "wall_seconds_declared": wall,
        "g1_eligible": term["g1_eligible"],
        "checkpoint_promotable": term["checkpoint_promotable"],
        "descriptors_verified": checks,
        "enters_comparisons": False,
        "enters_comparisons_reason": (
            "the campaign population is incomplete; a partial "
            "population is adjudicated, never compared"),
    }


# -------------------------------------------------------------- PARTIAL
def inventory_partial_cell(results_root: Path, cell: str,
                           ledger_cell: dict) -> dict:
    """Inventory EVERY artifact of the stopped cell, then classify."""
    cell_dir = results_root / cell
    started = time.perf_counter()
    files = inventory(cell_dir)
    MEASUREMENT["inventory_seconds"][cell] = round(
        time.perf_counter() - started, 3)

    present = set(files)
    completion_present = [a for a in COMPLETION_ARTIFACTS if a in present]
    if completion_present:
        raise ClosureRefusal(
            f"{cell}: carries {completion_present} — a cell with "
            "completion artifacts is not a partial and must be verified "
            "as completed or refused, never quarantined by assumption")
    seals = sorted(f for f in present if f.startswith("SEAL_"))
    if seals:
        raise ClosureRefusal(f"{cell}: carries seals {seals}")

    claim_name = next((f for f in present if f.startswith("CLAIM_")), None)
    lease_name = next((f for f in present if f.startswith("LEASE_")), None)
    if claim_name is None or lease_name is None:
        raise ClosureRefusal(
            f"{cell}: a stopped attempt must carry both a claim and a "
            f"lease; claim={claim_name} lease={lease_name}")
    claim = json.loads((cell_dir / claim_name).read_text())
    lease = json.loads((cell_dir / lease_name).read_text())
    if claim["attempt_id"] != lease["attempt_id"]:
        raise ClosureRefusal(
            f"{cell}: claim and lease name different attempts")
    # A claim's terminal_sha256 is null by construction at start, so
    # its emptiness proves nothing. The evidence that nothing was
    # sealed is the ABSENCE of the seal pair, checked above.
    if claim.get("cell") != cell or lease.get("cell") != cell:
        raise ClosureRefusal(f"{cell}: claim/lease name another cell")

    stop_local = cell_dir / "STOP"
    stop_campaign = results_root / "CAMPAIGN_STOP"
    if not stop_local.exists() or not stop_campaign.exists():
        raise ClosureRefusal(
            f"{cell}: quarantine requires BOTH durable stop signals; "
            f"cell STOP={stop_local.exists()} "
            f"campaign STOP={stop_campaign.exists()}")

    status_p = cell_dir / "cell_runtime/status.json"
    status = json.loads(status_p.read_text()) if status_p.is_file() else {}

    return {
        "cell": cell,
        "classification": PARTIAL,
        "attempt_id": claim["attempt_id"],
        "cell_config_sha256_expected": ledger_cell["cell_config_sha256"],
        "artifact_count": len(files),
        "artifact_bytes": sum(f["bytes"] for f in files.values()),
        "artifact_inventory_sha256": sha_obj(files),
        "completion_artifacts_present": completion_present,
        "stop_signals": {"cell": True, "campaign": True},
        "last_durable_progress": {
            "epoch_completed": status.get("epoch_completed", "UNAVAILABLE"),
            "num_timesteps": status.get("num_timesteps", "UNAVAILABLE"),
            "stop_reason": status.get("stop_reason", "UNAVAILABLE"),
            "last_durable_artifact": status.get("last_durable_artifact",
                                                "UNAVAILABLE"),
        },
        "enters_comparisons": False,
        "enters_comparisons_reason": (
            "no terminal, no per-bar evidence and no seal: there is "
            "nothing to compare. Its artifacts are preserved as "
            "evidence and contribute no model, replay, RNG or "
            "checkpoint to any later generation"),
        "scientific_contribution": "NONE",
    }


# ---------------------------------------------------------- NOT_STARTED
def record_not_started(results_root: Path, cell: str) -> dict:
    cell_dir = results_root / cell
    if cell_dir.exists():
        raise ClosureRefusal(
            f"{cell}: a directory exists, so absence is not the evidence")
    return {
        "cell": cell,
        "classification": NOT_STARTED,
        "evidence": "no cell directory exists under the campaign root",
        "enters_comparisons": False,
        "enters_comparisons_reason": "the cell was never executed",
    }


# --------------------------------------------------------------- costs
def rederive_costs(results_root: Path, completed: list[dict],
                   partial: list[dict]) -> dict:
    """Costs from durable bytes only.

    For a sealed cell the terminal's own measurement governs. For the
    stopped cell the only durable interval that survives v7 is
    claim -> operator STOP file. The supervised reap record that
    amendment 16 charged from was introduced for v8 and has no v7
    counterpart, so this figure is a LOWER BOUND and is reported as
    one. A closure may not reduce a charge it cannot fully re-derive.
    """
    per_cell: dict[str, float] = {}
    for entry in completed:
        per_cell[entry["cell"]] = float(entry["wall_seconds_declared"])

    partial_costs: dict[str, dict] = {}
    for entry in partial:
        cell_dir = results_root / entry["cell"]
        claim_p = next(cell_dir.glob("CLAIM_*.json"))
        claim = json.loads(claim_p.read_text())
        start = float(claim["claimed_wall"])
        stop_at = (cell_dir / "STOP").stat().st_mtime
        derived = round(stop_at - start, 1)
        if derived <= 0:
            raise ClosureRefusal(
                f"{entry['cell']}: the stop signal predates the claim")
        partial_costs[entry["cell"]] = {
            "claim_to_stop_signal_seconds": derived,
            "bound": "LOWER_BOUND",
            "why": ("the STOP file records when the stop was SIGNALLED. "
                    "The charge closes at the externally observed reap, "
                    "which is later by the escalation window and has no "
                    "durable v7 record; this closure therefore does not "
                    "lower any charge already made"),
        }
    return {
        "completed_wall_seconds": per_cell,
        "completed_wall_seconds_total": round(sum(per_cell.values()), 1),
        "quarantined_partial": partial_costs,
        "cost_units": "wall_seconds",
        "device": "gpu_attached_cell_wall",
    }


# ---------------------------------------------------------- adjudication
def adjudicate(classified: list[dict], ledger: dict) -> dict:
    counts: dict[str, int] = {}
    for entry in classified:
        counts[entry["classification"]] = \
            counts.get(entry["classification"], 0) + 1
    declared = len(ledger["cells"])
    completed = counts.get(COMPLETED, 0)
    if completed == declared:
        raise ClosureRefusal(
            "every declared cell is complete — this is a completion, not "
            "a closure, and belongs to the campaign verifier")
    return {
        "declared_cells": declared,
        "counts": counts,
        "verdict": "SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT",
        "contract": (
            "the pre-existing contract is the complete-population gate: "
            "the campaign verifier re-derives ALL declared cells before "
            "any comparison. With "
            f"{completed}/{declared} cells sealed the population is "
            "incomplete, so no effect, ranking or promotion may be read "
            "from it"),
        "explicitly_not_done": [
            "no cell was completed by label",
            "no partial artifact was promoted to a result",
            "no comparison, effect or ranking was computed",
            "no completed cell became promotable by this act",
            "no campaign was relaunched, resumed or rescheduled",
        ],
    }


# -------------------------------------------------------- relaunch proof
def prove_relaunch_refuses() -> dict:
    """The gate must refuse, and must refuse before CUDA exists."""
    accel_before = sorted(m for m in sys.modules
                          if m.split(".")[0] in ("torch", "tensorflow",
                                                 "jax", "cupy"))
    import b4_authority as b4a
    try:
        b4a.require_v6_launch_open()
    except SystemExit as exc:
        refusal = str(exc)
    except Exception as exc:                                # noqa: BLE001
        refusal = f"{type(exc).__name__}: {exc}"
    else:
        raise ClosureRefusal(
            "the launch gate did NOT refuse — a quarantined campaign "
            "whose launch is open is not quarantined")
    accel_after = sorted(m for m in sys.modules
                         if m.split(".")[0] in ("torch", "tensorflow",
                                                "jax", "cupy"))
    return {
        "launch_gate": "b4_authority.require_v6_launch_open",
        "refused": True,
        "refusal": refusal[:400],
        "accelerator_modules_before": accel_before,
        "accelerator_modules_after": accel_after,
        "refused_before_any_accelerator_import":
            accel_before == accel_after == [],
    }


# ---------------------------------------------------------------- main
def build_closure(results_root: Path, mat_root: Path) -> dict:
    ledger_p = results_root / "CAMPAIGN_LEDGER.json"
    ledger = json.loads(ledger_p.read_text())
    if ledger.get("materialization_sha256") != \
            sha_file(mat_root / "B4_MATERIALIZATION.json"):
        raise ClosureRefusal(
            "the ledger does not bind this materialization root")

    classified: list[dict] = []
    for cell in sorted(ledger["cells"]):
        entry = ledger["cells"][cell]
        cell_dir = results_root / cell
        if (cell_dir / "B4_CELL_TERMINAL.json").is_file():
            classified.append(verify_completed_cell(results_root, cell,
                                                    entry))
        elif cell_dir.is_dir():
            classified.append(inventory_partial_cell(results_root, cell,
                                                     entry))
        else:
            classified.append(record_not_started(results_root, cell))

    completed = [c for c in classified if c["classification"] == COMPLETED]
    partial = [c for c in classified if c["classification"] == PARTIAL]

    closure = {
        "schema": CLOSURE_SCHEMA,
        "closed_at": utc_now(),
        "campaign_generation": ledger["generation_provenance"][
            "campaign_generation"],
        "results_root_logical": results_root.name,
        "materialization_sha256": ledger["materialization_sha256"],
        "population_sha256": ledger["population_sha256"],
        "campaign_ledger_sha256": sha_file(ledger_p),
        "owner_authorization": {
            "consumed": True,
            "source": ("the owner's prior authorization to stop and "
                       "quarantine the third cell, recorded in design "
                       "amendment 16 (v7_root_disposition)"),
            "requested_again": False,
        },
        "cells": classified,
        "costs": rederive_costs(results_root, completed, partial),
        "adjudication": adjudicate(classified, ledger),
        "relaunch": prove_relaunch_refuses(),
        "measurement": dict(MEASUREMENT),
    }
    body = {k: closure[k] for k in sorted(closure)}
    closure["closure_sha256"] = sha_obj(body)
    # The SCIENTIFIC identity excludes when the closure was taken. Two
    # closures of the same evidence are the same adjudication even
    # though they were written at different instants, so re-running the
    # tool appends nothing new — while the history stays append-only.
    closure["adjudication_sha256"] = sha_obj(
        {k: v for k, v in body.items()
         if k not in VOLATILE_FOR_IDENTITY})
    return closure


def last_closure(results_root: Path) -> dict | None:
    log = results_root / CLOSURE_LOG
    if not log.is_file():
        return None
    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--materialization-root", type=Path, required=True)
    ap.add_argument("--emit", action="store_true",
                    help="append the record to the campaign root")
    args = ap.parse_args(argv)

    results_root = args.results_root.expanduser()
    before = inventory(results_root, exclude=(CLOSURE_LOG,))
    closure = build_closure(results_root, args.materialization_root
                            .expanduser())

    if args.emit:
        previous = last_closure(results_root)
        if previous and previous.get("adjudication_sha256") == \
                closure["adjudication_sha256"]:
            closure["emitted"] = False
            closure["already_closed_at"] = previous["closed_at"]
            print(json.dumps(closure, indent=1, sort_keys=True))
            return 0
        log = results_root / CLOSURE_LOG
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(closure, sort_keys=True) + "\n")
        closure["emitted"] = True
        after = inventory(results_root, exclude=(CLOSURE_LOG,))
        if after != before:
            changed = sorted(set(before) ^ set(after)) or [
                k for k in before if before[k] != after.get(k)]
            raise ClosureRefusal(
                f"emitting the closure changed training artifacts: "
                f"{changed[:5]}")
        closure["artifact_immutability_proof"] = {
            "files_digested": len(before),
            "bytes_digested": sum(f["bytes"] for f in before.values()),
            "tree_sha256_before": sha_obj(before),
            "tree_sha256_after": sha_obj(after),
            "unchanged": True,
        }

    print(json.dumps(closure, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
