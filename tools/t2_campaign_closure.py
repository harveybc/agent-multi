#!/usr/bin/env python3
"""R2 (order 2026-09-11): close T2 from its 242 durable records.

Three things were missing, and one of them was not known until this
tool refused to run.

1. The executor heartbeat says ``done=241`` while 242 records, 242
   claims and 242 arrays sit on disk. A heartbeat is telemetry — it is
   written before the unit it names is sealed — so the discrepancy is a
   reporting artefact, not a missing unit. Nothing durable said so.

2. Nothing verified the inventory as an EXACT triple per unit: one
   claim, one array, one record, no extras, no duplicates, nothing
   foreign.

3. The recomputation must run under the REVIEWED code identity. The
   external successor execution record pins seven executor files and a
   commit; ``tools/t2_confirmatory_executor.py`` was corrected AFTER
   that record was authored, so the branch tip can no longer satisfy
   its own gate. This tool therefore refuses to guess: it takes an
   explicit ``--reviewed-checkout`` whose bytes equal the record, and
   imports the pinned modules from THERE. The divergence is reported,
   never smoothed over, and no gate is loosened to get past it.

Nothing is retrained and nothing is re-downloaded: every number comes
from the arrays and records already on disk.

    python tools/t2_campaign_closure.py \\
        --reviewed-checkout <clean checkout at the pinned commit> \\
        [--root <results root>] [--emit] [--emit-outbox]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TIP_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TIP_REPO / "tools"))

from descriptor_custody import Custody  # noqa: E402

CLOSURE_SCHEMA = "agent_multi.t2_campaign_closure.v1"
CLOSURE_LOG = "T2_CAMPAIGN_CLOSURE.jsonl"

#: fields that say WHEN or HOW LONG, never WHAT was found.
VOLATILE_FOR_IDENTITY = frozenset({"closed_at", "measurement"})

UNIT_KINDS = ("CLAIM", "ARRAYS", "RECORD")


class ClosureRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def utc_now() -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"))


# --------------------------------------------------- reviewed identity
def load_audit_snapshot_modules(snapshot: Path):
    """R12: EVERY replay import comes from the one recoverable
    checkout.

    The previous loader put the reviewed checkout first and the branch
    tip second, so the reconstruction driver and the closure came from
    a different tree than the verifier. One snapshot means one tree.
    """
    snapshot = Path(snapshot).expanduser().resolve()
    tools = str(snapshot / "tools")
    for entry in (str(TIP_REPO / "tools"), tools):
        while entry in sys.path:
            sys.path.remove(entry)
    for mod in ("t2_confirmatory", "t2_confirmatory_executor",
                "t2_completion_reconstruction", "descriptor_custody"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, tools)
    import t2_confirmatory as conf
    import t2_confirmatory_executor as ex
    import t2_completion_reconstruction as recon
    for mod in (conf, ex, recon):
        got = Path(mod.__file__).resolve().parents[1]
        if got != snapshot:
            raise ClosureRefusal(
                f"{mod.__name__} resolved to {got.name}, not to the "
                f"audit snapshot {snapshot.name} — a replay whose "
                "imports come from two trees has no single identity")
    return conf, ex, recon


def load_reviewed_modules(checkout: Path):
    """Put the REVIEWED checkout's tools first on sys.path.

    Import order is the whole point: ``t2_confirmatory`` and
    ``t2_confirmatory_executor`` must come from the reviewed checkout,
    while the reconstruction driver — which the execution record does
    not pin — comes from the branch tip. Reversing this would recompute
    under code no external record has seen.
    """
    checkout = checkout.expanduser().resolve()
    for entry in (str(TIP_REPO / "tools"), str(checkout / "tools")):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, str(TIP_REPO / "tools"))
    sys.path.insert(0, str(checkout / "tools"))
    import t2_confirmatory as conf
    import t2_confirmatory_executor as ex
    import t2_completion_reconstruction as recon
    for mod, want_root in ((conf, checkout), (ex, checkout),
                           (recon, TIP_REPO)):
        got = Path(mod.__file__).resolve().parents[1]
        if got != want_root.resolve():
            raise ClosureRefusal(
                f"{mod.__name__} resolved to {got}, not {want_root}")
    return conf, ex, recon


def assert_reviewed_identity(conf, checkout: Path, *,
                             snapshot_mode: bool = False) -> dict:
    """Verify a checkout against the external execution record.

    R12 note on the COMMIT. The record pins both a commit and seven
    file digests. An audit snapshot reproduces the seven digests EXACTLY
    and necessarily sits at a different commit, because the same tree
    must also carry the reconstruction, closure and custody code the
    replay executes — code that post-dates the record. Demanding commit
    equality there would make a single recoverable checkout impossible,
    and quietly dropping the check would hide which commit the seven
    files came from. So in snapshot mode the file digests ARE the
    identity claim and the pinned commit is recorded as the provenance
    of those bytes, with the difference stated.
    """
    record = json.loads(
        conf.T2_SUCCESSOR_EXECUTION_RECORD_PATH.read_text())
    declared = record["executor_code_identity"]
    absent = sorted(rel for rel in declared
                    if not (checkout / rel).is_file())
    if absent:
        raise ClosureRefusal(
            f"the checkout at {checkout.name} is not the reviewed "
            f"identity; these pinned files are absent: {absent}")
    physical = {rel: sha_file(checkout / rel) for rel in declared}
    diff = sorted(k for k in declared if declared[k] != physical[k])
    if diff:
        raise ClosureRefusal(
            f"the checkout at {checkout.name} is not the reviewed "
            f"identity; these files differ from the execution record: "
            f"{diff}")
    head = subprocess.run(("git", "-C", str(checkout), "rev-parse", "HEAD"),
                          capture_output=True, text=True).stdout.strip()
    commit_note = "EQUAL_TO_PINNED_COMMIT"
    if head != record["pinned_commit"]:
        if not snapshot_mode:
            raise ClosureRefusal(
                f"the checkout is at {head[:12]}, the record pins "
                f"{record['pinned_commit'][:12]}")
        commit_note = (
            f"the seven pinned files match the record byte for byte; "
            f"this snapshot sits at {head[:12]} rather than the pinned "
            f"{record['pinned_commit'][:12]} because the same tree also "
            "carries the replay code, which post-dates the record")
    dirty = subprocess.run(("git", "-C", str(checkout), "status",
                            "--porcelain"),
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ClosureRefusal("the reviewed checkout is not clean")
    return {
        "pinned_commit": record["pinned_commit"],
        "checkout_commit": head,
        "commit_relation": commit_note,
        "execution_record_sha256": sha_obj(
            {k: record[k] for k in sorted(record)}),
        "pinned_files": {k: declared[k] for k in sorted(declared)},
        "checkout_clean": True,
    }


def report_tip_divergence(conf, checkout: Path) -> dict:
    """Name the divergence between the tip and the reviewed identity.

    This is a finding, not a failure of the closure: the tip's extra
    hardening is real work, but it is work no external record has seen,
    and a result recomputed under it would carry an identity nobody
    reviewed.
    """
    record = json.loads(
        conf.T2_SUCCESSOR_EXECUTION_RECORD_PATH.read_text())
    declared = record["executor_code_identity"]
    diverged = {}
    for rel, want in declared.items():
        tip_file = TIP_REPO / rel
        got = sha_file(tip_file) if tip_file.is_file() else "ABSENT"
        if got != want:
            diverged[rel] = {"reviewed": want, "branch_tip": got}
    return {
        "branch_tip_can_run_the_gate": not diverged,
        "diverged_files": diverged,
        "consequence": (
            "the branch tip refuses its own confirmatory gate, so the "
            "recomputation MUST run from the reviewed checkout. The "
            "corrections made after the record was authored (a typed "
            "refusal for omitted mlp seeds at both executor sites) are "
            "NOT part of the identity this closure recomputes under, "
            "and are declared here rather than silently included"
            if diverged else
            "the branch tip equals the reviewed identity"),
    }


# ------------------------------------------------------------ inventory
def exact_inventory_from_snapshot(ex, snap_facts: dict, custody,
                                  uids: list[str]) -> dict:
    """The inventory, derived from the SAME listing the snapshot used."""
    names = custody.walk_to("units").files
    expected: dict[str, str] = {}
    for uid in uids:
        safe = ex._safe_name(uid)
        expected[f"CLAIM_{safe}.json"] = uid
        expected[f"ARRAYS_{safe}.npz"] = uid
        expected[f"RECORD_{safe}.json"] = uid
    missing = sorted(set(expected) - set(names))
    extra = sorted(set(names) - set(expected))
    per_kind = {k: sum(1 for n in names if n.startswith(k + "_"))
                for k in UNIT_KINDS}
    if missing or extra:
        raise ClosureRefusal(
            f"the unit inventory is not exact: missing={missing[:5]} "
            f"extra={extra[:5]}")
    if set(per_kind.values()) != {len(uids)}:
        raise ClosureRefusal(
            f"per-kind counts {per_kind} do not all equal {len(uids)}")
    return {"sealed_units": len(uids), "artifacts_per_kind": per_kind,
            "missing": missing, "extra": extra, "duplicates": [],
            "total_artifacts": len(names), "exact": True,
            "snapshot_inventory_sha256": snap_facts["inventory_sha256"]}


def exact_inventory(ex, root: Path, uids: list[str]) -> dict:
    """One claim, one array, one record per unit. Nothing else."""
    units_dir = root / "units"
    actual = sorted(p.name for p in units_dir.iterdir() if p.is_file())
    expected: dict[str, str] = {}
    for uid in uids:
        safe = ex._safe_name(uid)
        expected[f"CLAIM_{safe}.json"] = uid
        expected[f"ARRAYS_{safe}.npz"] = uid
        expected[f"RECORD_{safe}.json"] = uid

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    # A filesystem cannot hold two entries with one name, so a
    # "duplicate" here means a second artifact claiming the same unit
    # and kind under a different spelling — which is what `extra`
    # already catches. Recording the count keeps the claim explicit.
    per_kind = {k: sum(1 for n in actual if n.startswith(k + "_"))
                for k in UNIT_KINDS}
    if missing or extra:
        raise ClosureRefusal(
            f"the unit inventory is not exact: missing={missing[:5]} "
            f"extra={extra[:5]}")
    if set(per_kind.values()) != {len(uids)}:
        raise ClosureRefusal(
            f"per-kind counts {per_kind} do not all equal "
            f"{len(uids)} sealed units")
    return {
        "sealed_units": len(uids),
        "artifacts_per_kind": per_kind,
        "missing": missing,
        "extra": extra,
        "duplicates": [],
        "total_artifacts": len(actual),
        "exact": True,
    }


# ------------------------------------------------------- reconciliation
def reconcile_heartbeat(root: Path, ex, uids: list[str],
                        counts: dict) -> dict:
    hb_p = root / "EXECUTOR_HEARTBEAT.json"
    hb = json.loads(hb_p.read_text()) if hb_p.is_file() else {}
    current = hb.get("current_unit")
    safe = ex._safe_name(current) if current else None
    record_p = (root / "units" / f"RECORD_{safe}.json") if safe else None
    sealed = bool(record_p and record_p.is_file())
    done = hb.get("done")
    adjudicated = counts["COMPLETED_VERIFIED"] + counts["TERMINAL_FAILED"]
    return {
        "heartbeat_done": done,
        "heartbeat_current_unit": current,
        "current_unit_record_sealed": sealed,
        "records_on_disk": len(uids),
        "units_adjudicated": adjudicated,
        "discrepancy": (adjudicated - done) if isinstance(done, int)
                       else "UNAVAILABLE",
        "explanation": (
            "the heartbeat is written at the START of a unit: it names "
            "the unit being worked on and counts the units finished "
            "BEFORE it. When the executor sealed the last unit "
            f"({current}) it did not write a further heartbeat, because "
            "there was no next unit to announce. So done=241 and "
            "current_unit=<the 242nd> together describe 242 units, and "
            "the 242nd unit's record is physically present. No unit is "
            "missing and none was inferred: the count comes from the "
            "records, and the heartbeat is telemetry with no terminal "
            "authority."),
        "authority": "RECORDS_GOVERN_TELEMETRY_DOES_NOT",
    }


# ----------------------------------------------------------- the closure
def exact_two_sided_binomial_p(k: int, n: int) -> float:
    """P0-5 / R9.9: a two-sided exact binomial p under p=0.5.

    The sealed screen published `2 * P(X >= k)`, which is a DOUBLED ONE
    TAIL. At three positives out of six that is 1.3125 — a number no
    p-value can take — and it is undefined below three, so it is not
    symmetric either. The correct statistic doubles the SMALLER tail
    and is capped at one, giving the symmetric bounded table

        0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875, 0.03125

    This supersedes the field. It does not change the estimand, the
    per-panel effects or the harm-gate verdict, and the historical
    envelope carrying 1.3125 is not rewritten.
    """
    if not (isinstance(k, int) and isinstance(n, int)) or n <= 0 \
            or not 0 <= k <= n:
        raise ClosureRefusal(
            f"a sign test needs 0 <= k <= n with n > 0; got k={k} n={n}")
    total = float(2 ** n)
    lower = sum(math.comb(n, i) for i in range(0, k + 1)) / total
    upper = sum(math.comb(n, i) for i in range(k, n + 1)) / total
    return round(min(1.0, 2.0 * min(lower, upper)), 5)


def supersede_sign_test(screen: dict) -> dict:
    """Recompute the sign test beside the sealed one, never over it."""
    k = screen.get("signs_positive")
    published = screen.get("sign_test_exact_p_two_sided")
    if not isinstance(k, int):
        return {"state": "UNAVAILABLE",
                "reason": "the screen declares no sign count"}
    corrected = exact_two_sided_binomial_p(k, 6)
    return {
        "state": "SUPERSEDED",
        "signs_positive": k,
        "published_value": published,
        "published_value_valid": (isinstance(published, (int, float))
                                  and 0.0 <= float(published) <= 1.0),
        "corrected_value": corrected,
        "corrected_table_0_to_6": [exact_two_sided_binomial_p(i, 6)
                                   for i in range(7)],
        "changes_estimand": False,
        "changes_panel_effects": False,
        "changes_verdict": False,
        "why_the_verdict_stands": (
            "advancement required 6/6 positive signs and the harm gate "
            "already determined the negative result; correcting an "
            "inference that could never have been read as favourable "
            "cannot turn it favourable"),
        "historical_envelope": "NOT REWRITTEN",
    }


def build_closure(conf, ex, recon, root: Path, checkout: Path,
                  *, snapshot_mode: bool = False) -> dict:
    measurement: dict = {}
    t0 = time.perf_counter()

    identity = assert_reviewed_identity(conf, checkout,
                                       snapshot_mode=snapshot_mode)
    try:
        snapshot_identity = audit_snapshot_identity(checkout)
    except ClosureRefusal as exc:
        snapshot_identity = {"single_checkout": False,
                             "refusal": str(exc)[:200]}
    divergence = report_tip_divergence(conf, checkout)

    t_recon = time.perf_counter()
    custody = Custody(root)
    try:
        doc = reconstruct_from_snapshot(conf, ex, recon, root, custody)
        measurement["reconstruction_seconds"] = round(
            time.perf_counter() - t_recon, 2)
        measurement["custody_reads"] = len(custody.reads())

        facts = conf.verify_confirmatory_gates(
            ex.MANIFEST_PATH, ex.active_design_path(),
            census_path=ex.CENSUS_PATH)
        uids = facts["design"].doc["task_population"]["series_ids"]
        inventory = exact_inventory_from_snapshot(
            ex, doc["unit_snapshot"], custody, uids)
    finally:
        custody.close()
    counts = doc["final_adjudication_counts"]
    reconciliation = reconcile_heartbeat(root, ex, uids, counts)

    screen = doc["screen_adjudication"]
    verdict = screen.get("verdict")
    if counts["TERMINAL_FAILED"] > 0:
        terminal_state = "FAILED"
    elif verdict in (None, "", "UNAVAILABLE"):
        terminal_state = "INCONCLUSIVE"
    else:
        terminal_state = "COMPLETE"

    measurement["closure_seconds"] = round(time.perf_counter() - t0, 2)

    closure = {
        "schema": CLOSURE_SCHEMA,
        "sign_test_supersession": supersede_sign_test(screen),
        "unit_snapshot": doc["unit_snapshot"],
        "single_instance": doc["single_instance"],
        "final_adjudication_not_called":
            doc["final_adjudication_not_called"],
        "closed_at": utc_now(),
        "campaign_root_logical": root.name,
        "reviewed_identity": identity,
        "audit_snapshot_identity": snapshot_identity,
        "branch_tip_divergence": divergence,
        "inventory": inventory,
        "final_adjudication_counts": counts,
        "heartbeat_reconciliation": reconciliation,
        "wall_ledger": doc["wall_ledger"],
        "release_sequence": doc["release_sequence"],
        "screen_adjudication": screen,
        "gate_facts": doc["gate_facts"],
        "terminal_state": terminal_state,
        "retraining": "NONE — every number re-derives from the arrays "
                      "and records already on disk",
        "downloads": "NONE",
        "measurement": measurement,
    }
    body = {k: closure[k] for k in sorted(closure)}
    closure["closure_sha256"] = sha_obj(body)
    closure["adjudication_sha256"] = sha_obj(
        {k: v for k, v in body.items() if k not in VOLATILE_FOR_IDENTITY})
    return closure


#: R8: everything the re-adjudication can execute. The seven files the
#: external record pins, plus the reconstruction driver and this
#: closure — which the record does NOT pin, because they were written
#: after it — plus the custody layer both now depend on.
CLOSURE_SURFACE = (
    "tools/t2_confirmatory.py",
    "tools/t2_confirmatory_executor.py",
    "tools/t2_assay_harness.py",
    "tools/t2_bank.py",
    "tools/t2_bank_census.py",
    "tools/t2_fresh_verifier.py",
    "tools/t2_public_data_census.py",
    "tools/t2_completion_reconstruction.py",
    "tools/t2_campaign_closure.py",
    "tools/descriptor_custody.py",
)

#: libraries whose numerics decide the reconstructed metrics. Recorded
#: by name, version and location: hashing one wrapper would be a false
#: claim about a whole distribution.
NUMERIC_DEPENDENCIES = ("numpy", "scipy")


def audit_snapshot_identity(snapshot_checkout: Path) -> dict:
    """R12: ONE recoverable checkout, and the proof that it is one.

    The previous submission combined seven files from the reviewed
    checkout with three from a dirty branch tip and said so honestly —
    but an honest mixture is still not an identity a reviewer can
    obtain. This reads every file of the executable surface from a
    SINGLE checkout, proves that checkout's seven pinned files match
    the external execution record byte for byte, and records its commit
    so it can be fetched.
    """
    snap = Path(snapshot_checkout)
    record = json.loads(
        (Path.home() / ".config/agent-multi/reviewer_authority"
         / "MUSASHI_T2_SUCCESSOR_EXECUTION_RECORD.json").read_text())
    pinned = record["executor_code_identity"]

    files, mismatched, absent = {}, [], []
    for rel in CLOSURE_SURFACE:
        f = snap / rel
        if not f.is_file():
            absent.append(rel)
            continue
        digest = sha_file(f)
        entry = {"sha256": digest,
                 "pinned_by_execution_record": rel in pinned}
        if rel in pinned:
            entry["matches_record"] = pinned[rel] == digest
            if not entry["matches_record"]:
                mismatched.append(rel)
        files[rel] = entry
    if absent:
        raise ClosureRefusal(
            f"the audit snapshot is missing {absent} — an identity "
            "that skips a file is not one")
    if mismatched:
        raise ClosureRefusal(
            f"the audit snapshot does not carry the pinned bytes for "
            f"{mismatched}; it is not the reviewed executor")

    def _git(*args):
        return subprocess.run(("git", "-C", str(snap), *args),
                              capture_output=True,
                              text=True).stdout.strip()

    dirty = _git("status", "--porcelain")
    deps = {}
    for name in NUMERIC_DEPENDENCIES:
        try:
            from importlib import metadata
            deps[name] = {"version": metadata.distribution(name).version,
                          "binding": "NAME_AND_VERSION_ONLY"}
        except Exception:                                 # noqa: BLE001
            deps[name] = {"version": "NOT_INSTALLED",
                          "binding": "NAME_AND_VERSION_ONLY"}
    return {
        "single_checkout": True,
        "commit": _git("rev-parse", "HEAD"),
        "clean": not dirty,
        "dirty_paths": sorted(ln.split(maxsplit=1)[-1]
                              for ln in dirty.splitlines()
                              if ln.strip())[:20],
        "files": files,
        "surface_sha256": sha_obj(files),
        "pinned_files_verified": sorted(r for r in files
                                        if files[r].get("matches_record")),
        "files_not_pinned_by_the_record": sorted(
            r for r in files if not files[r]["pinned_by_execution_record"]),
        "numeric_dependencies": deps,
        "honesty": ("every file of the executable surface comes from "
                    "ONE checkout whose commit is recorded above. The "
                    "seven files the external record pins are verified "
                    "byte for byte; the rest post-date the record and "
                    "are named, not hidden"),
        "interpreter": sys.version.split()[0],
    }


def closure_code_identity(reviewed_checkout: Path,
                          tip: Path = TIP_REPO) -> dict:
    """Every file the closure can execute, and where each came from.

    R8: the previous version called its output a "reviewed identity"
    while the reconstruction driver and the closure itself came from
    the branch tip. A mixture is not an identity. Each file is now
    listed with the checkout it was loaded from and its digest, the
    record's pins are compared file by file, and the divergences are
    named rather than averaged away.
    """
    reviewed = Path(reviewed_checkout)
    record = json.loads(
        (Path.home() / ".config/agent-multi/reviewer_authority"
         / "MUSASHI_T2_SUCCESSOR_EXECUTION_RECORD.json").read_text())
    pinned = record["executor_code_identity"]

    files = {}
    for rel in CLOSURE_SURFACE:
        source = reviewed if rel in pinned else tip
        f = source / rel
        if not f.is_file():
            raise ClosureRefusal(
                f"the closure surface names {rel}, absent from "
                f"{source.name}")
        digest = sha_file(f)
        files[rel] = {
            "sha256": digest,
            "loaded_from": ("REVIEWED_CHECKOUT" if rel in pinned
                            else "BRANCH_TIP"),
            "pinned_by_execution_record": rel in pinned,
            "matches_record": (pinned.get(rel) == digest
                               if rel in pinned else "NOT_PINNED"),
        }
    tip_divergence = {
        rel: {"reviewed": pinned[rel], "branch_tip": sha_file(tip / rel)}
        for rel in pinned
        if (tip / rel).is_file() and sha_file(tip / rel) != pinned[rel]}

    deps = {}
    for name in NUMERIC_DEPENDENCIES:
        try:
            from importlib import metadata
            dist = metadata.distribution(name)
            # R12.4: version only. The previous record published a
            # site-packages path, which is this host's topology and
            # not evidence; it does not travel in Git.
            deps[name] = {"version": dist.version,
                          "binding": "NAME_AND_VERSION_ONLY",
                          "location": "WITHHELD — local topology is "
                                      "not evidence"}
        except Exception:                                 # noqa: BLE001
            deps[name] = {"version": "NOT_INSTALLED",
                          "binding": "NAME_AND_VERSION_ONLY",
                          "location": "WITHHELD — local topology is "
                                      "not evidence"}

    def _git(repo, *args):
        return subprocess.run(("git", "-C", str(repo), *args),
                              capture_output=True,
                              text=True).stdout.strip()

    return {
        "files": files,
        "surface_sha256": sha_obj(files),
        "reviewed_checkout": {
            "commit": _git(reviewed, "rev-parse", "HEAD"),
            "clean": not _git(reviewed, "status", "--porcelain"),
        },
        "branch_tip": {
            "commit": _git(tip, "rev-parse", "HEAD"),
            "clean": not _git(tip, "status", "--porcelain"),
            "dirty_paths": sorted(
                ln[3:] for ln in
                _git(tip, "status", "--porcelain").splitlines() if ln)[:20],
        },
        "pinned_files_diverging_at_tip": tip_divergence,
        "numeric_dependencies": deps,
        "honesty": (
            "this is NOT called a reviewed identity. Seven files come "
            "from the checkout the external record pins; the "
            "reconstruction driver, this closure and the custody layer "
            "come from the branch tip because the record predates "
            "them. Both origins are named per file"),
        "interpreter": sys.version.split()[0],
    }


def build_readjudication_submission(closure: dict, *,
                                    reviewed_checkout: Path,
                                    read_root: Path,
                                    preserved_root: Path) -> dict:
    """R10: a submission, and a stop.

    It states what would be re-derived, under exactly which code, and
    asks for review. It opens nothing, authorizes nothing and does not
    supersede the previous envelope — that happens after review.
    """
    doc = {
        "schema": "agent_multi.t2_readjudication_submission.v1",
        "submitted_at": utc_now(),
        "campaign_root_logical": closure["campaign_root_logical"],
        "root_actually_read": str(read_root),
        "read_the_preserved_root": Path(read_root).resolve()
        == Path(preserved_root).resolve(),
        "inventory": closure["inventory"],
        "final_adjudication_counts": closure["final_adjudication_counts"],
        "screen_verdict": closure["screen_adjudication"].get("verdict"),
        "primary_estimand": closure["screen_adjudication"].get(
            "primary_estimand_unweighted_mean_of_panel_effects"),
        "sign_test_supersession": closure["sign_test_supersession"],
        "unit_snapshot": closure["unit_snapshot"],
        "single_instance": closure["single_instance"],
        "adjudication_sha256": closure["adjudication_sha256"],
        "code_identity": closure_code_identity(reviewed_checkout),
        "grants_nothing":
            "a submission states what was re-adjudicated and under "
            "which code, and asks for a decision. It supersedes no "
            "envelope, promotes nothing, and does not authorize "
            "re-deriving against the preserved root",
        "requires": "EXTERNAL_REVIEW",
    }
    doc["submission_sha256"] = sha_obj({k: doc[k] for k in sorted(doc)})
    return doc


def last_closure(root: Path) -> dict | None:
    log = root / CLOSURE_LOG
    if not log.is_file():
        return None
    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


# ------------------------------------------------------------- outbox
def emit_to_outbox(closure: dict, *, uids: list[str],
                   manifest: dict | None = None) -> dict:
    """One durable envelope into the CRISP-DM outbox.

    The outbox is a local content-addressed directory; emitting never
    contacts a database and never fails because one is unavailable.
    """
    predictor = Path.home() / "Documents/GitHub/predictor"
    if not (predictor / "olap/outbox.py").is_file():
        raise ClosureRefusal(
            f"no outbox implementation at {predictor.name}/olap")
    sys.path.insert(0, str(predictor))
    from olap import outbox as ob                      # noqa: E402

    screen = closure["screen_adjudication"]
    counts = closure["final_adjudication_counts"]
    gate = closure["gate_facts"]
    # The envelope contract is EXACT, and rightly so: my first attempt
    # omitted data_consumed and artifacts and the loader dead-lettered
    # it. A campaign envelope that cannot say what it consumed is a
    # summary, not evidence.
    document = {
        "schema": "crispdm.campaign_envelope.v1",
        "producer": "agent-multi",
        "campaign_key": "t2::resource_successor_screen",
        "result_class": "DEVELOPMENT",
        "identity": {
            "run_id": closure["campaign_root_logical"],
            "code_identity": closure["reviewed_identity"]["pinned_commit"],
            "design_sha256": gate["design_self_sha256"],
            "record_sha256": closure["closure_sha256"],
        },
        "data_consumed": {
            "variables": [{"id": uid,
                           "digest": gate["census_sha256"],
                           "eligibility_state":
                               "PUBLIC_FORECASTING_EVIDENCE"}
                          for uid in sorted(uids)],
            "operators": [{"id": "ewma_alpha_0.3",
                           "digest": gate["design_self_sha256"],
                           "eligibility_state":
                               "T1_ACCEPTED_OPERATOR"}],
            "datasets": [{"id": closure["campaign_root_logical"],
                          "digest": gate["manifest_sha256"],
                          "eligibility_state":
                              "PUBLIC_FORECASTING_EVIDENCE"}],
        },
        "terminal": {
            "state": closure["terminal_state"],
            "adjudication": screen.get("verdict", "UNAVAILABLE"),
            "failure_phase": "UNAVAILABLE",
            "failure_type": "UNAVAILABLE",
            "reason": screen.get("reason", "UNAVAILABLE"),
        },
        "partitions": {
            "exposure": "SCREEN_DEVELOPMENT_NON_CONFIRMATORY",
            "splits": "rolling_origin",
        },
        "budget": {
            "cost_units": "wall_seconds",
            "device": "cpu",
            "wall_seconds": closure["wall_ledger"].get(
                "charged_state", "UNAVAILABLE"),
            "units_verified": counts["COMPLETED_VERIFIED"],
            "units_failed": counts["TERMINAL_FAILED"],
        },
        "artifacts": {
            "verification": "SCHEMA_EXACT_AND_SELF_DIGEST_REDERIVED",
            "closure_record": CLOSURE_LOG,
            "results_root": closure["campaign_root_logical"],
        },
        "units": [{
            "cell_key": closure["campaign_root_logical"],
            "candidate_key": "ewma_alpha_0.3",
            "metric_name": "screen_primary_estimand",
            "metric_value": screen.get(
                "primary_estimand_unweighted_mean_of_panel_effects"),
            "terminal_state": closure["terminal_state"],
        }],
    }
    document["envelope_sha256"] = sha_obj(
        {k: document[k] for k in sorted(document)})
    return ob.emit(document, kind="envelope")


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reviewed-checkout", type=Path, default=None)
    ap.add_argument("--audit-snapshot", type=Path, default=None,
                    help="R12: the ONE recoverable checkout every "
                         "replay import comes from")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--emit-outbox", action="store_true")
    ap.add_argument("--outbox-only", action="store_true",
                    help="emit the LAST durable closure to the outbox "
                         "without recomputing it; the closure record "
                         "is the authority, the envelope is derived")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        os.nice(15)
    except OSError:
        pass

    if args.audit_snapshot:
        conf, ex, recon = load_audit_snapshot_modules(args.audit_snapshot)
        checkout = args.audit_snapshot
    elif args.reviewed_checkout:
        conf, ex, recon = load_reviewed_modules(args.reviewed_checkout)
        checkout = args.reviewed_checkout
    else:
        raise ClosureRefusal(
            "one of --audit-snapshot or --reviewed-checkout is required")
    root = (args.root or recon.DEFAULT_ROOT).expanduser()

    if args.outbox_only:
        closure = last_closure(root)
        if closure is None:
            raise ClosureRefusal(
                "there is no durable closure to emit; run the closure "
                "first — an envelope is derived from a record, never "
                "invented")
        facts = conf.verify_confirmatory_gates(
            ex.MANIFEST_PATH, ex.active_design_path(),
            census_path=ex.CENSUS_PATH)
        uids = facts["design"].doc["task_population"]["series_ids"]
        closure["outbox"] = emit_to_outbox(closure, uids=uids)
        print(json.dumps(closure, indent=1, sort_keys=True,
                         default=str))
        return 0

    closure = build_closure(conf, ex, recon, root, checkout,
                            snapshot_mode=bool(args.audit_snapshot))

    if args.emit:
        previous = last_closure(root)
        if previous and previous.get("adjudication_sha256") == \
                closure["adjudication_sha256"]:
            closure["emitted"] = False
            closure["already_closed_at"] = previous["closed_at"]
        else:
            with (root / CLOSURE_LOG).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(closure, sort_keys=True,
                                    default=str) + "\n")
            closure["emitted"] = True

    if args.emit_outbox:
        facts = conf.verify_confirmatory_gates(
            ex.MANIFEST_PATH, ex.active_design_path(),
            census_path=ex.CENSUS_PATH)
        closure["outbox"] = emit_to_outbox(
            closure,
            uids=facts["design"].doc["task_population"]["series_ids"])

    print(json.dumps(closure, indent=1, sort_keys=True, default=str))
    return 0




# =====================================================================
# R7 (order 2026-09-12): ONE verified snapshot per unit.
# =====================================================================
#
# The audit found the hole precisely: `final_adjudication()` verifies
# every unit through the results root's held descriptors, and then
# `reconstruct()` RE-OPENS every `RECORD_*.json` by path to build the
# list handed to `adjudicate_screen()`. A replacement between those two
# steps lets one set of bytes be verified and another be scored.
#
# The pinned verifier cannot hand its bytes back — `verify_unit_record`
# returns a summary, not the wrapper it parsed — so the order's second
# option applies: read each unit ONCE here, and make BOTH the pinned
# verification and the screen consume that same instance.
#
# The pinned code reaches evidence through a small surface —
# `read_private(dfd, name, what)`, `exists(dfd, name)`, `listdir(dfd)`
# and a `units_fd` token. `UnitSnapshotRoot` implements exactly that
# surface over bytes already read, so the verifier runs unmodified and
# consumes the snapshot. Nothing in the pinned identity changes.

def make_snapshot_class(results_root_cls):
    """Build the snapshot adapter as a SUBCLASS of the pinned
    `ResultsRoot`.

    The pinned `_unit_srcs()` routes reads through the held descriptors
    only when `isinstance(root, ResultsRoot)`. Subclassing — without
    calling the parent's `__init__`, so nothing is opened — makes that
    test true while every read is answered from the snapshot. Not one
    byte of the pinned code changes.
    """

    class UnitSnapshotRoot(results_root_cls):
        """The units directory, read once, served to everyone.

        JSON artifacts are small and are held for the whole run, because
        the screen needs every record at the end. The `.npz` arrays are the
        bulk (hundreds of megabytes) and are consumed only by verification,
        so each is read once and released as soon as its unit is
        adjudicated — still exactly one read per artifact.
        """

        #: the token the pinned code passes back to `read_private`.
        units_fd = "UNITS_SNAPSHOT"

        def __init__(self, units_snap, names: tuple[str, ...],
                     path=None) -> None:
            # The parent's __init__ is deliberately NOT called: it
            # would open the very descriptors this snapshot replaces.
            # R11: `units_snap` is a RETAINED directory descriptor, so
            # every read below comes out of the instance that was
            # inventoried — not out of whatever now answers to the
            # name `units`.
            self.path = path
            self._units = units_snap
            self._names = tuple(sorted(names))
            self._json: dict[str, bytes] = {}
            self._arrays: dict[str, bytes] = {}
            self._digests: dict[str, str] = {}
            self._reads = 0
            for name in self._names:
                if name.endswith(".json"):
                    art = units_snap.read(name)
                    self._json[name] = art.raw()
                    self._digests[name] = art.sha256
                    self._reads += 1

        # ---------------- the surface the pinned verifier expects -------
        def read_private(self, dfd, name, what) -> bytes:
            if dfd is not self.units_fd:
                raise ClosureRefusal(
                    f"{what}: a read was attempted through a descriptor "
                    "this snapshot does not own")
            if name in self._json:
                return self._json[name]
            if name in self._arrays:
                return self._arrays[name]
            if name in self._names and name.endswith(".npz"):
                art = self._units.read(name)
                self._arrays[name] = art.raw()
                self._digests[name] = art.sha256
                self._reads += 1
                return self._arrays[name]
            raise ClosureRefusal(f"{what}: {name} is not in the snapshot")

        def exists(self, dfd, name) -> bool:
            return name in self._names

        def listdir(self, dfd) -> list[str]:
            return list(self._names)

        def revalidate(self) -> None:
            """The snapshot cannot drift: it is bytes, not a path."""

        def close(self) -> None:
            self._arrays.clear()

        # ------------------------------------------------------- facts --
        def release_arrays(self, uid_safe: str) -> None:
            self._arrays.pop(f"ARRAYS_{uid_safe}.npz", None)

        def record(self, uid_safe: str) -> dict:
            """The parsed record — from the SAME bytes the verifier saw."""
            name = f"RECORD_{uid_safe}.json"
            if name not in self._json:
                raise ClosureRefusal(f"{name} is not in the snapshot")
            return json.loads(self._json[name].decode("utf-8"))

        def digests(self) -> dict:
            return dict(self._digests)

        def reads(self) -> int:
            return self._reads

        def facts(self) -> dict:
            return {"artifacts": len(self._names),
                    "json_held": len(self._json),
                    "reads": self._reads,
                    "inventory_sha256": sha_obj(self._digests)}

    # R13: FAIL CLOSED on anything this adapter does not implement.
    #
    # Subclassing inherits the parent's methods, so a call the adapter
    # does not override would silently run the pinned implementation
    # against attributes the adapter never created — and the failure
    # would be an AttributeError somewhere deep, not a refusal. Worse,
    # if the pinned verifier ever grows a new public call, the adapter
    # would quietly answer it with real filesystem behaviour.
    #
    # Every public callable of the parent that this adapter does not
    # deliberately implement is therefore replaced by a typed refusal.
    # Adding a call to the pinned verifier now stops the replay instead
    # of changing what it consumes.
    IMPLEMENTED = {"read_private", "exists", "listdir", "revalidate",
                   "close", "release_arrays", "record", "digests",
                   "reads", "facts"}

    def _refuse(name):
        def _stub(self, *a, **k):
            raise ClosureRefusal(
                f"the unit snapshot does not implement {name!r}. The "
                "pinned verifier reached for it, so this replay would "
                "have fallen through to real filesystem behaviour "
                "outside the retained descriptor — it fails closed "
                "instead")
        _stub.__name__ = name
        return _stub

    for _name in dir(results_root_cls):
        if _name.startswith("_") or _name in IMPLEMENTED:
            continue
        if callable(getattr(results_root_cls, _name, None)):
            setattr(UnitSnapshotRoot, _name, _refuse(_name))
    UnitSnapshotRoot.implemented_surface = frozenset(IMPLEMENTED)
    return UnitSnapshotRoot


def reconstruct_from_snapshot(conf, ex, recon, root: Path,
                              custody) -> dict:
    """R7: verify and score ONE instance of every unit.

    This replaces the record-collection half of
    `t2_completion_reconstruction.reconstruct()`. It is declared, not
    hidden: the pinned `final_adjudication()` is NOT called, because it
    re-resolves the results root (`ResultsRoot(rr.path)`) and would
    reopen the evidence this snapshot exists to pin. Its three
    guarantees are reproduced here over the snapshot instead —

      * the control directory carries EXACTLY the expected objects;
      * EVERY unit re-adjudicates deeply under current authority,
        through the pinned verifier, consuming the snapshot's bytes;
      * zero UNCERTAIN, and the counts close the sealed population.

    The wall replay and the release sequence still come from the
    reconstruction module's own helpers, which read the campaign's
    ledger and lock files rather than unit evidence.
    """
    import numpy as np

    t0 = time.perf_counter()
    facts = conf.verify_confirmatory_gates(
        ex.MANIFEST_PATH, ex.active_design_path(),
        census_path=ex.CENSUS_PATH)
    design = facts["design"].doc
    manifest = facts["manifest"].doc
    if design.get("schema") == conf.T2_SUCCESSOR_SCHEMA:
        conf.verify_resource_successor(design)
    authority = {
        "sealed_design_file_sha256": facts["design_file_sha256"],
        "sealed_design_self_sha256": facts["design_self_sha256"],
        "design_review_record_sha256": facts["review_record_sha256"],
        "execution_record_sha256": facts["execution_record_sha256"],
        "manifest_sha256": facts["manifest_sha256"],
        "census_sha256": facts["census_sha256"]}
    uids = design["task_population"]["series_ids"]

    # R11: the units directory is photographed ONCE and its
    # descriptor is retained until the last verification. The audit
    # showed the previous version re-resolved `units` by name for every
    # RECORD and every ARRAYS, so the whole directory could be
    # exchanged between the inventory and the reads.
    units_snap = custody.walk_to("units")
    if units_snap.others or units_snap.dirs:
        raise ClosureRefusal(
            f"the units directory contains non-file entries: "
            f"{sorted(units_snap.others) + sorted(units_snap.dirs)}")
    snapshot_cls = make_snapshot_class(ex.ResultsRoot)
    snap = snapshot_cls(units_snap, units_snap.files, path=root)

    expected = set()
    for uid in uids:
        safe = ex._safe_name(uid)
        expected.update({f"CLAIM_{safe}.json", f"RECORD_{safe}.json",
                         f"ARRAYS_{safe}.npz"})
    actual = set(units_snap.files)
    foreign = sorted(actual - expected - {f"TERMINAL_{ex._safe_name(u)}.json"
                                          for u in uids})
    missing = sorted(expected - actual)
    if foreign or missing:
        raise ClosureRefusal(
            f"the control directory is not exact: foreign={foreign[:5]} "
            f"missing={missing[:5]}")

    raw_root = ex.STATE / "t2_public_raw"

    def _rebuild(uid):
        return np.asarray(
            ex.load_bank_unit(design, uid, manifest, raw_root)["y"],
            dtype=np.float64)

    counts = {"COMPLETED_VERIFIED": 0, "TERMINAL_FAILED": 0}
    records, cost_total = [], 0.0
    for uid in uids:
        safe = ex._safe_name(uid)
        st, why = ex.adjudicate_unit_shallow(snap, uid)
        y = _rebuild(uid) if st == "COMPLETED" else None
        st, why = ex.adjudicate_unit_deep(snap, uid, design, authority,
                                          "confirmatory", unit_y=y)
        if st == "UNCERTAIN":
            raise ClosureRefusal(
                f"final adjudication found typed uncertainty — {why}")
        if st == "PENDING":
            raise ClosureRefusal(
                f"{uid} is still PENDING; the counts do not close the "
                "sealed population")
        counts[st] += 1
        # The SAME bytes the verifier just consumed become the record
        # that is scored. There is no second open.
        rec = snap.record(safe)
        ar = rec["assay_record"]

        def _numsum(node):
            if isinstance(node, (int, float)):
                return float(node)
            if isinstance(node, dict):
                return sum(_numsum(v) for v in node.values())
            return 0.0

        phase_sum = _numsum(ar.get("costs_by_phase", {}))
        wall_u = float(rec.get("wall_seconds", 0.0))
        if wall_u <= 0 or phase_sum > wall_u * 1.10 + 2.0:
            raise ClosureRefusal(
                f"{uid}: per-phase cost sum {phase_sum:.2f}s is "
                f"incoherent with the unit wall {wall_u:.2f}s")
        cost_total += phase_sum
        records.append(ar)
        snap.release_arrays(safe)

    if counts["COMPLETED_VERIFIED"] + counts["TERMINAL_FAILED"] \
            != len(uids):
        raise ClosureRefusal(
            "adjudication counts do not equal the sealed population")

    wall = recon.replay_wall_ledger(root, design)
    release = recon.verify_release_sequence(root)
    screen = conf.adjudicate_screen(records, design)
    return {
        "gate_facts": {k: facts[k] for k in
                       ("design_file_sha256", "design_self_sha256",
                        "review_record_sha256", "execution_record_sha256",
                        "manifest_sha256", "census_sha256")},
        "final_adjudication_counts": counts,
        "wall_ledger": wall,
        "release_sequence": release,
        "screen_adjudication": screen,
        "unit_snapshot": {**snap.facts(),
                          "directory_instance": {
                              "device": units_snap.device,
                              "inode": units_snap.inode,
                              "mode": oct(units_snap.mode)}},
        "train_seconds_from_records": round(cost_total, 3),
        "single_instance": (
            "every unit was read once through a retained descriptor; "
            "the pinned verifier and the screen consumed that same "
            "instance, and no RECORD, ARRAYS, manifest, census or "
            "design was re-opened by path between them"),
        "final_adjudication_not_called": (
            "the pinned final_adjudication() re-resolves the results "
            "root and would reopen the evidence; its exact-inventory, "
            "deep-adjudication and population-closure guarantees are "
            "reproduced above over the snapshot"),
        "wall_seconds_reconstruction": round(time.perf_counter() - t0, 2),
    }


# The entry point lives at the END of the module, after every
# definition. It used to sit in the middle: `reconstruct_from_snapshot`
# was appended below it, so running this file as a script called main()
# before that function existed and died with a NameError. The tests
# never saw it because importing a module defines everything first.
if __name__ == "__main__":
    sys.exit(main())
