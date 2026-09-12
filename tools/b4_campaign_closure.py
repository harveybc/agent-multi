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

R11-R12 (order 2026-09-12) — EVERY ARTIFACT IS READ ONCE.

The first version verified and consumed through SEPARATE path opens:
it parsed the terminal and hashed it from a second open, re-opened the
per-bar ledger to count rows, and re-read `status.json` after
inventorying it. The PRE shows what that costs — a terminal reading
999999 on disk adjudicated as 123.4 with every descriptor reporting
`verified`, and a per-bar ledger whose rows were counted from bytes
that were never hashed.

Now a retained root descriptor resolves each path component by
component with O_NOFOLLOW, `fstat` establishes regular-file, owner and
mode on that same descriptor, the bytes are read once, and the digest,
the parse, the row count and every rule are computed from THOSE bytes.
Classification reads a directory SNAPSHOT taken once; no `exists`,
`is_file` or `glob` runs at decision time. A file that appears after
the snapshot cannot change the run that used it.

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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from descriptor_custody import (Artifact, Custody,  # noqa: E402
                                CustodyRefusal, DirSnapshot)

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
def verify_completed_cell(custody: Custody, snap: DirSnapshot,
                          cell: str, ledger_cell: dict) -> dict:
    """Descriptor verification of a sealed cell.

    Every object below is read ONCE through `custody`; the digest, the
    parse and the counts all come from those same bytes. Nothing here
    takes a path.

    Deliberately does NOT consult the launch gate. A closure is not a
    launch: requiring an open launch to READ finished evidence would
    make a campaign unclosable exactly when it most needs closing.
    """
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"descriptor": name, "verified": bool(ok),
                       "detail": detail})
        if not ok:
            raise ClosureRefusal(f"{cell}: {name} — {detail}")

    check("terminal_present_in_snapshot",
          snap.has_file("B4_CELL_TERMINAL.json"),
          "listed once from the cell's own directory descriptor")
    terminal = custody.read(f"{cell}/B4_CELL_TERMINAL.json")
    term = terminal.json()
    terminal_digest = terminal.sha256          # SAME bytes as `term`

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

    per_bar_name = Path(term["per_bar_csv"]).name
    check("per_bar_present_in_snapshot", snap.has_file(per_bar_name),
          per_bar_name)
    per_bar = custody.read(f"{cell}/{per_bar_name}")
    check("per_bar_digest", per_bar.sha256 == term["per_bar_sha256"],
          term["per_bar_sha256"])
    # R11: the rows are counted in the bytes that were just hashed.
    check("per_bar_rows_match_scored_bars",
          per_bar.data_rows() == term["scored_bars"],
          f"rows={per_bar.data_rows()} scored_bars={term['scored_bars']}")

    ckpt_name = Path(term["checkpoint_path"]).name
    check("checkpoint_present_in_snapshot", snap.has_file(ckpt_name),
          ckpt_name)
    checkpoint = custody.read(f"{cell}/{ckpt_name}")
    check("checkpoint_digest",
          checkpoint.sha256 == term["checkpoint_sha256"],
          term["checkpoint_sha256"])

    claim_names = snap.matching(prefix="CLAIM_", suffix=".json")
    check("exactly_one_claim", len(claim_names) == 1, str(claim_names))
    claim_art = custody.read(f"{cell}/{claim_names[0]}")
    claim = claim_art.json()
    check("claim_binds_this_attempt",
          claim.get("attempt_id") == term["attempt_id"],
          f"{claim.get('attempt_id')} vs {term['attempt_id']}")
    check("claim_binds_this_cell", claim.get("cell") == cell,
          str(claim.get("cell")))

    # The terminal is bound by the TWO-PHASE SEAL, not by the claim.
    # A claim is written when the attempt starts, so its
    # terminal_sha256 is null by construction; checking the claim for
    # the terminal digest would refuse every correctly sealed cell.
    attempt = term["attempt_id"]
    intent_name = f"SEAL_INTENT_{attempt}.json"
    seal_name = f"SEAL_COMPLETE_{attempt}.json"
    check("seal_intent_present_in_snapshot", snap.has_file(intent_name),
          intent_name)
    check("seal_complete_present_in_snapshot", snap.has_file(seal_name),
          seal_name)
    intent = custody.read(f"{cell}/{intent_name}")
    seal = custody.read(f"{cell}/{seal_name}")
    intent_doc, seal_doc = intent.json(), seal.json()
    check("seal_intent_binds_terminal_digest",
          intent_doc.get("terminal_sha256") == terminal_digest,
          str(intent_doc.get("terminal_sha256")))
    check("seal_intent_binds_this_cell", intent_doc.get("cell") == cell,
          str(intent_doc.get("cell")))
    # R11: the intent's digest comes from the SAME read that was
    # parsed above, not from a second open.
    check("seal_complete_binds_the_intent",
          seal_doc.get("intent_sha256") == intent.sha256,
          str(seal_doc.get("intent_sha256")))
    check("seal_complete_binds_the_same_terminal",
          seal_doc.get("terminal_sha256") == terminal_digest,
          str(seal_doc.get("terminal_sha256")))
    check("seal_complete_binds_this_attempt",
          seal_doc.get("attempt_id") == attempt,
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
        "attempt_id": attempt,
        "terminal_sha256": terminal_digest,
        "seal_intent_sha256": intent.sha256,
        "seal_sha256": seal.sha256,
        "per_bar_sha256": per_bar.sha256,
        "checkpoint_sha256": checkpoint.sha256,
        "scored_bars": term["scored_bars"],
        "wall_seconds_declared": wall,
        "g1_eligible": term["g1_eligible"],
        "checkpoint_promotable": term["checkpoint_promotable"],
        "descriptors_verified": checks,
        "custody": {
            "reads": 6,
            "artifacts": [a.facts() for a in
                          (terminal, per_bar, checkpoint, claim_art,
                           intent, seal)],
        },
        "enters_comparisons": False,
        "enters_comparisons_reason": (
            "the campaign population is incomplete; a partial "
            "population is adjudicated, never compared"),
    }


# -------------------------------------------------------------- PARTIAL
def inventory_partial_cell(custody: Custody, root_snap: DirSnapshot,
                           cell: str, ledger_cell: dict) -> dict:
    """Inventory EVERY artifact of the stopped cell, then classify.

    R12: the tree is walked through directory SNAPSHOTS and each file
    is read once; the classification and the per-file digests come from
    those same reads. The previous version combined `rglob`, `is_file`,
    `stat` and a path hash, and then re-opened `status.json` — so the
    progress it adjudicated was not the progress it inventoried.
    """
    started = time.perf_counter()
    files: dict[str, dict] = {}
    artifacts: dict[str, Artifact] = {}
    pending = [cell]
    while pending:
        rel = pending.pop()
        snap = custody.snapshot(rel)
        if snap.others:
            raise ClosureRefusal(
                f"{cell}: {rel} contains entries that are neither "
                f"regular files nor directories: {sorted(snap.others)} "
                "— a link or a device in an evidence tree is not "
                "evidence")
        for name in snap.dirs:
            pending.append(f"{rel}/{name}")
        for name in snap.files:
            art = custody.read(f"{rel}/{name}")
            key = str(Path(f"{rel}/{name}").relative_to(cell))
            files[key] = {"bytes": art.size, "sha256": art.sha256,
                          "custody_weakness": art.custody_weakness}
            artifacts[key] = art
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

    claim_key = next((f for f in present if f.startswith("CLAIM_")), None)
    lease_key = next((f for f in present if f.startswith("LEASE_")), None)
    if claim_key is None or lease_key is None:
        raise ClosureRefusal(
            f"{cell}: a stopped attempt must carry both a claim and a "
            f"lease; claim={claim_key} lease={lease_key}")
    claim = artifacts[claim_key].json()
    lease = artifacts[lease_key].json()
    if claim["attempt_id"] != lease["attempt_id"]:
        raise ClosureRefusal(
            f"{cell}: claim and lease name different attempts")
    if claim.get("cell") != cell or lease.get("cell") != cell:
        raise ClosureRefusal(f"{cell}: claim/lease name another cell")
    # A claim's terminal_sha256 is null by construction at start, so
    # its emptiness proves nothing. The evidence that nothing was
    # sealed is the ABSENCE of the seal pair, checked above from the
    # inventory itself.

    if "STOP" not in present:
        raise ClosureRefusal(
            f"{cell}: quarantine requires the cell's own durable STOP "
            "signal and it is not in the inventory")
    if not root_snap.has_file("CAMPAIGN_STOP"):
        raise ClosureRefusal(
            f"{cell}: quarantine requires the CAMPAIGN stop signal and "
            "the campaign root snapshot does not list it")

    status_key = "cell_runtime/status.json"
    status = (artifacts[status_key].json()
              if status_key in artifacts else {})

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
            "read_from": ("the same inventory read, never a second open"
                          if status_key in artifacts else "ABSENT"),
        },
        "claim_wall": float(claim["claimed_wall"]),
        "stop_mtime_ns": artifacts["STOP"].mtime_ns,
        "enters_comparisons": False,
        "enters_comparisons_reason": (
            "no terminal, no per-bar evidence and no seal: there is "
            "nothing to compare. Its artifacts are preserved as "
            "evidence and contribute no model, replay, RNG or "
            "checkpoint to any later generation"),
        "scientific_contribution": "NONE",
    }


# ---------------------------------------------------------- NOT_STARTED
def published_root_facts(root_snap: DirSnapshot) -> dict:
    """The campaign root as EVIDENCE, with this closure's own record
    excluded.

    The append-only log lives in the root it describes, so including it
    would make the first closure change the snapshot every later
    closure publishes — the record would alter the thing it is about.
    The exclusion is named in the facts, never silent.
    """
    facts = root_snap.facts()
    facts["files"] = [f for f in facts["files"] if f != CLOSURE_LOG]
    facts["excluded"] = [CLOSURE_LOG]
    facts["excluded_reason"] = (
        "this closure's own append-only record; including it would "
        "make the closure change the snapshot it publishes")
    return facts


def record_not_started(root_snap: DirSnapshot, cell: str) -> dict:
    """R12: absence is read from the ROOT SNAPSHOT taken once.

    The previous version asked the filesystem at classification time,
    so a cell that appeared afterwards kept the label NOT_STARTED while
    its directory — and even a terminal — existed on disk.
    """
    if cell in root_snap.dirs or cell in root_snap.files:
        raise ClosureRefusal(
            f"{cell}: the root snapshot lists it, so absence is not the "
            "evidence")
    return {
        "cell": cell,
        "classification": NOT_STARTED,
        "evidence": ("the campaign root directory, listed once from its "
                     "own descriptor, contains no entry for this cell"),
        "root_snapshot_sha256": sha_obj(
            published_root_facts(root_snap)),
        "enters_comparisons": False,
        "enters_comparisons_reason": "the cell was never executed",
    }


# --------------------------------------------------------------- costs
def rederive_costs(completed: list[dict],
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
        # R11: both numbers were already established by the single
        # inventory read — the claim's own bytes and the fstat taken on
        # the STOP file's descriptor. Re-opening either here would
        # reintroduce the window this rewrite exists to close.
        start = float(entry["claim_wall"])
        stop_at = entry["stop_mtime_ns"] / 1e9
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
    """One retained root descriptor; one snapshot per directory; one
    read per artifact."""
    custody = Custody(results_root)
    mat_custody = Custody(mat_root)
    try:
        ledger_art = custody.read("CAMPAIGN_LEDGER.json")
        ledger = ledger_art.json()
        mat_art = mat_custody.read("B4_MATERIALIZATION.json")
        if ledger.get("materialization_sha256") != mat_art.sha256:
            raise ClosureRefusal(
                "the ledger does not bind this materialization root")

        # R12: ONE listing of the campaign root decides which cells
        # have a directory at all. Every classification below reads
        # this snapshot, never the filesystem.
        root_snap = custody.snapshot()

        classified: list[dict] = []
        for cell in sorted(ledger["cells"]):
            entry = ledger["cells"][cell]
            if cell not in root_snap.dirs:
                classified.append(record_not_started(root_snap, cell))
                continue
            cell_snap = custody.snapshot(cell)
            if cell_snap.has_file("B4_CELL_TERMINAL.json"):
                classified.append(verify_completed_cell(
                    custody, cell_snap, cell, entry))
            else:
                classified.append(inventory_partial_cell(
                    custody, root_snap, cell, entry))

        completed = [c for c in classified
                     if c["classification"] == COMPLETED]
        partial = [c for c in classified
                   if c["classification"] == PARTIAL]
        ledger_digest = ledger_art.sha256
        custody_reads = len(custody.reads())
        # The closure's OWN append-only log lives in this root, so it
        # appears in the snapshot the moment the first closure is
        # emitted. Publishing it would make every re-closure a new
        # adjudication of the same evidence — the record would change
        # the thing it describes. It is excluded and the exclusion is
        # named; it is not campaign evidence.
        root_facts = published_root_facts(root_snap)
        weaknesses = sorted({a["custody_weakness"]
                             for a in custody.reads()})
    finally:
        custody.close()
        mat_custody.close()

    closure = {
        "schema": CLOSURE_SCHEMA,
        "closed_at": utc_now(),
        "campaign_generation": ledger["generation_provenance"][
            "campaign_generation"],
        "results_root_logical": results_root.name,
        "materialization_sha256": ledger["materialization_sha256"],
        "population_sha256": ledger["population_sha256"],
        "campaign_ledger_sha256": ledger_digest,
        "custody": {
            "discipline": "one path resolution -> one descriptor -> "
                          "one read -> all facts; classification reads "
                          "directory snapshots taken once",
            "artifact_reads": custody_reads,
            "root_snapshot": root_facts,
            "observed_weaknesses": weaknesses,
        },
        "owner_authorization": {
            "consumed": True,
            "source": ("the owner's prior authorization to stop and "
                       "quarantine the third cell, recorded in design "
                       "amendment 16 (v7_root_disposition)"),
            "requested_again": False,
        },
        "cells": classified,
        "costs": rederive_costs(completed, partial),
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


#: R13: the closure's own executable surface. Every module the
#: adjudication can reach, so a mutation anywhere in it changes the
#: identity a reviewer is asked to accept.
CLOSURE_CODE = (
    "tools/b4_campaign_closure.py",
    "tools/descriptor_custody.py",
    "tools/b4_authority.py",
)


def closure_code_identity(repo: Path = REPO,
                          ignore: tuple[str, ...] = ()) -> dict:
    """Bind this closure's code, its commit and a clean worktree.

    A re-adjudication is only as trustworthy as the program that
    produced it, so the program is named file by file. A dirty tree is
    recorded rather than hidden: a reviewer must be able to see that
    the bytes reviewed are the bytes that ran.
    """
    repo = Path(repo)
    files = {}
    for rel in sorted(CLOSURE_CODE):
        f = repo / rel
        if not f.is_file():
            raise ClosureRefusal(
                f"the closure surface names {rel}, which is absent — "
                "an identity that skips a missing file is not one")
        files[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    commit = subprocess.run(("git", "-C", str(repo), "rev-parse", "HEAD"),
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(("git", "-C", str(repo), "status",
                            "--porcelain"),
                           capture_output=True, text=True).stdout.strip()
    # An artifact cannot be required to be absent from the tree it
    # describes: writing the submission dirties the worktree the
    # submission reports on. The exclusion is named, never silent.
    # `git status --porcelain` is XY<space>PATH, but a rename carries
    # an arrow and a staged line pads differently; splitting on
    # whitespace is the only parse that does not silently eat the
    # first character of a path — which it did.
    dirty_files = sorted(ln.split(maxsplit=1)[-1]
                         for ln in dirty.splitlines() if ln.strip())
    excluded = sorted(f for f in dirty_files
                      if any(f.endswith(i) for i in ignore))
    dirty_files = [f for f in dirty_files if f not in excluded]
    return {
        "files": files,
        "surface_sha256": sha_obj(files),
        "commit": commit or "UNAVAILABLE",
        "worktree_clean": not dirty_files,
        "dirty_paths": dirty_files[:20],
        "excluded_from_cleanliness": excluded,
        "excluded_reason": ("this submission's own file; an artifact "
                            "cannot be required to be absent from the "
                            "tree it describes"),
        "interpreter": sys.version.split()[0],
    }


def build_submission(closure: dict, *, results_root: Path,
                     read_root: Path, repo: Path = REPO) -> dict:
    """R14: a re-adjudication SUBMISSION. It authorizes nothing.

    It states which root was read, whether that root was the preserved
    original or a copy, what the adjudication found, and exactly which
    code produced it — and asks for review. The preserved B4 root is
    not opened for writing by this tool at any point.
    """
    doc = {
        "schema": "agent_multi.b4_readjudication_submission.v1",
        "submitted_at": utc_now(),
        "campaign_generation": closure["campaign_generation"],
        "results_root_logical": closure["results_root_logical"],
        "root_actually_read": str(read_root),
        "read_the_preserved_root": Path(read_root).resolve()
        == Path(results_root).resolve(),
        "adjudication": closure["adjudication"],
        "adjudication_sha256": closure["adjudication_sha256"],
        "campaign_ledger_sha256": closure["campaign_ledger_sha256"],
        "custody": closure["custody"],
        "cell_digests": {
            c["cell"]: {k: c[k] for k in
                        ("classification", "terminal_sha256",
                         "per_bar_sha256", "artifact_inventory_sha256")
                        if k in c}
            for c in closure["cells"]},
        "costs": closure["costs"],
        "code_identity": closure_code_identity(
            repo, ignore=("B4_READJUDICATION_SUBMISSION_2026_09_12.json",)),
        "grants_nothing":
            "a submission states what was re-adjudicated and asks for a "
            "decision. It opens no campaign, promotes no cell, and does "
            "not authorize reading or writing the preserved root",
        "requires": "EXTERNAL_REVIEW",
    }
    doc["submission_sha256"] = sha_obj(
        {k: doc[k] for k in sorted(doc)})
    return doc


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
    ap.add_argument("--submission", type=Path, default=None,
                    help="write a NON-AUTHORIZING re-adjudication "
                         "submission for external review")
    ap.add_argument("--preserved-root", type=Path, default=None,
                    help="the root this submission is ABOUT, when the "
                         "reading happened on a copy of it")
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

    if args.submission:
        sub = build_submission(
            closure, results_root=(args.preserved_root or results_root)
            .expanduser(), read_root=results_root)
        args.submission.parent.mkdir(parents=True, exist_ok=True)
        args.submission.write_text(
            json.dumps(sub, indent=1, sort_keys=True) + "\n")
        closure["submission_file"] = str(args.submission.name)
        closure["submission_sha256"] = sub["submission_sha256"]

    print(json.dumps(closure, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
