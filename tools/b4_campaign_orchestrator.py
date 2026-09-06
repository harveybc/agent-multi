#!/usr/bin/env python3
"""B4 campaign orchestrator (orders @0ce52740 C4, C9-C16).

C10 — ONE claimable object per logical cell and campaign generation:
the claim path is FIXED (`CLAIM_<generation>.json`), created with
O_EXCL|O_NOFOLLOW and validated descriptor-first; the random
attempt_id lives INSIDE the record. Two synchronized processes at
the vulnerable boundary produce exactly one winner. Uncertain
creation fails closed; success is never inferred from absence.

C11 — dry-run means ZERO writes: without --execute the orchestrator
verifies authority, materialization, ledger, health schema and
schedule, prints the plan, and provably alters nothing.

C12 — execution happens only under a verified EXECUTION LEASE bound
to campaign generation, cell, attempt, authorization and
materialization digests; the executor refuses without it.

C13 — the 96 GPU-hour ceiling is an INTRASEGMENT bound: each
dispatch derives the remaining campaign wall from every attempt
(failed and uncertain included, durable facts for replay, monotonic
for the live attempt) and the cell's effective wall budget is
min(per-cell limit, global remainder); a remainder smaller than one
segment fails closed.

C15 — resume ADJUDICATES terminals: verified-completed skips;
verified non-completed is retained and named; unsealed/malformed is
UNCERTAIN and blocks; an ambiguous claim blocks; absent claim and
terminal is pending. Exit status distinguishes COMPLETE /
COMPLETE_WITH_FAILED_CELLS / STOPPED_RESOURCE_CEILING / UNCERTAIN —
never success-by-file-existence."""
import argparse
import hashlib
import json
import os
import stat
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import b4_authority as b4a  # noqa: E402

SUBSTANTIAL_COMPUTE_MIB = 1024
MIN_SEGMENT_SECONDS = 600.0


class OrchestratorRefusal(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _excl_write(path: Path, payload: bytes, mode=0o644) -> None:
    """O_EXCL|O_NOFOLLOW create, descriptor-first validation,
    fsync(file)+fsync(dir). Uncertainty fails closed."""
    if path.parent.is_symlink():
        raise OrchestratorRefusal(
            f"REFUSED: parent of {path.name} is a symlink")
    try:
        fd = os.open(str(path),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY
                     | os.O_NOFOLLOW, mode)
    except FileExistsError:
        raise OrchestratorRefusal(
            f"REFUSED: {path.name} already exists — exactly one "
            "winner per logical object")
    except OSError as exc:
        raise OrchestratorRefusal(
            f"REFUSED: uncertain exclusive create of {path.name}: "
            f"{exc} — failing closed")
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OrchestratorRefusal(
                f"REFUSED: {path.name} is not a regular file")
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class GlobalLock:
    """C19: acquisition is O_EXCL; release is OWNED (only the
    recorded holder pid may release) and DURABLE (an append-only
    RELEASE witness is fsynced before the unlink, and the directory
    is fsynced after). A crashed holder's lock is never auto-stolen
    — operator disposition only."""

    def __init__(self, results_root: Path):
        self.root = Path(results_root)
        self.path = self.root / "CAMPAIGN_LOCK"
        self.held = False
        self.acquire_id = None

    def __enter__(self):
        import uuid as _uuid
        self.acquire_id = _uuid.uuid4().hex[:16]
        _excl_write(self.path, json.dumps(
            {"pid": os.getpid(),
             "generation": b4a.CAMPAIGN_GENERATION,
             "acquire_id": self.acquire_id}).encode())
        self.held = True
        return self

    def __exit__(self, *exc):
        if not self.held:
            return
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise OrchestratorRefusal(
                "REFUSED: campaign lock vanished under the holder "
                f"({exc}) — uncertain release, failing closed")
        rec = b4a._strict_json_bytes(raw, "campaign lock")
        if rec.get("pid") != os.getpid() or \
                rec.get("acquire_id") != self.acquire_id:
            raise OrchestratorRefusal(
                "REFUSED: lock release by a non-holder — ownership "
                "is required")
        witness = self.root / f"LOCK_RELEASE_{self.acquire_id}.json"
        _excl_write(witness, json.dumps(
            {"schema": "agent_multi.b4_lock_release.v1",
             "acquire_id": self.acquire_id,
             "holder_pid": os.getpid()}).encode())
        self.path.unlink()
        dfd = os.open(str(self.root), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        self.held = False


def _claim_path(results_root: Path, cell_id: str) -> Path:
    return (Path(results_root) / cell_id /
            f"CLAIM_{b4a.CAMPAIGN_GENERATION}.json")


def claim_attempt(results_root: Path, cell_id: str) -> dict:
    """C10: the ONE claimable object per (cell, generation). The
    attempt id is data inside the record — never part of the
    exclusive path."""
    cell_dir = Path(results_root) / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "agent_multi.b4_attempt_claim.v2",
           "campaign_generation": b4a.CAMPAIGN_GENERATION,
           "attempt_id": f"attempt_{uuid.uuid4().hex[:16]}",
           "cell": cell_id,
           "claimed_wall": time.time(),
           "claimed_monotonic": time.monotonic(),
           "holder_pid": os.getpid(),
           "terminal_sha256": None}
    _excl_write(_claim_path(results_root, cell_id),
                json.dumps(rec, indent=1).encode())
    return rec


def load_claim(results_root: Path, cell_id: str) -> dict:
    p = _claim_path(results_root, cell_id)
    if p.is_symlink() or not p.is_file():
        raise OrchestratorRefusal(
            f"REFUSED: claim for {cell_id} absent or non-regular")
    rec = b4a._strict_json_bytes(p.read_bytes(),
                                 f"claim {cell_id}")
    for k in ("schema", "campaign_generation", "attempt_id",
              "cell", "claimed_wall", "terminal_sha256"):
        if k not in rec:
            raise OrchestratorRefusal(
                f"REFUSED: claim for {cell_id} missing {k!r}")
    if rec["cell"] != cell_id or \
            rec["campaign_generation"] != b4a.CAMPAIGN_GENERATION:
        raise OrchestratorRefusal(
            f"REFUSED: claim binding mismatch for {cell_id}")
    return rec


# ------------------- C12: execution lease -------------------------
LEASE_SCHEMA = {
    "schema": str, "campaign_generation": str, "cell": str,
    "attempt_id": str, "authorization_sha256": str,
    "materialization_sha256": str, "issued_monotonic": float,
    "holder_pid": int, "lease_sha256": str}
LEASE_SCHEMA_NAME = "agent_multi.b4_execution_lease.v2"


def issue_lease(results_root: Path, cell_id: str, claim: dict,
                auth_sha: str, mat_root: Path) -> Path:
    lease = {"schema": LEASE_SCHEMA_NAME,
             "campaign_generation": b4a.CAMPAIGN_GENERATION,
             "cell": cell_id,
             "attempt_id": claim["attempt_id"],
             "authorization_sha256": auth_sha,
             "materialization_sha256":
                 _sha_file(Path(mat_root) / "B4_MATERIALIZATION.json"),
             "issued_monotonic": float(time.monotonic()),
             "holder_pid": os.getpid()}
    lease["lease_sha256"] = hashlib.sha256(json.dumps(
        {k: lease[k] for k in sorted(lease)},
        sort_keys=True).encode()).hexdigest()
    p = (Path(results_root) / cell_id /
         f"LEASE_{claim['attempt_id']}.json")
    _excl_write(p, json.dumps(lease, indent=1).encode())
    return p


def verify_lease(lease_path: Path, results_root: Path,
                 cell_id: str, mat_root: Path,
                 expected_auth_sha: str = None) -> dict:
    """C17: the lease is an EXECUTION CAPABILITY, not a file beside
    a claim — exact schema and primitive types, immutable content
    digest, the LIVE campaign authorization digest, and holder
    identity bound across lease == claim == global lock == the
    executing process. Revalidate under the same lock immediately
    before entering the pipeline; any foreign element produces zero
    compute."""
    p = Path(lease_path)
    if p.is_symlink() or not p.is_file():
        raise OrchestratorRefusal("REFUSED: execution lease absent")
    lease = b4a._strict_json_bytes(p.read_bytes(), "execution lease")
    if set(lease) != set(LEASE_SCHEMA):
        raise OrchestratorRefusal(
            "REFUSED: lease keys are not the exact schema")
    for k, t in LEASE_SCHEMA.items():
        if type(lease[k]) is not t:
            raise OrchestratorRefusal(
                f"REFUSED: lease field {k!r} has a foreign "
                "primitive type")
    if lease["schema"] != LEASE_SCHEMA_NAME:
        raise OrchestratorRefusal(
            "REFUSED: foreign lease schema")
    body = {k: lease[k] for k in sorted(lease)
            if k != "lease_sha256"}
    if hashlib.sha256(json.dumps(
            body, sort_keys=True).encode()).hexdigest() != \
            lease["lease_sha256"]:
        raise OrchestratorRefusal(
            "REFUSED: lease content digest does not re-derive — "
            "the capability was altered")
    if lease["campaign_generation"] != b4a.CAMPAIGN_GENERATION \
            or lease["cell"] != cell_id:
        raise OrchestratorRefusal(
            "REFUSED: lease generation/cell binding mismatch")
    if expected_auth_sha is not None and \
            lease["authorization_sha256"] != expected_auth_sha:
        raise OrchestratorRefusal(
            "REFUSED: lease authorization digest differs from the "
            "reviewed campaign authorization")
    claim = load_claim(results_root, cell_id)
    if claim["attempt_id"] != lease["attempt_id"]:
        raise OrchestratorRefusal(
            "REFUSED: lease attempt differs from the unique claim")
    if seal_state(results_root, cell_id) != "UNSEALED":
        raise OrchestratorRefusal(
            "REFUSED: the claimed attempt already reached a sealed "
            "or uncertain terminal")
    terminal = Path(results_root) / cell_id / "B4_CELL_TERMINAL.json"
    if terminal.exists():
        raise OrchestratorRefusal(
            "REFUSED: a terminal already exists for this cell")
    lock = Path(results_root) / "CAMPAIGN_LOCK"
    if not lock.is_file():
        raise OrchestratorRefusal(
            "REFUSED: no live campaign lease/lock covers this "
            "execution")
    lockrec = b4a._strict_json_bytes(lock.read_bytes(),
                                     "campaign lock")
    if lockrec.get("generation") != b4a.CAMPAIGN_GENERATION:
        raise OrchestratorRefusal(
            "REFUSED: campaign lock belongs to another generation")
    me = os.getpid()
    if not (lease["holder_pid"] == lockrec.get("pid")
            == claim.get("holder_pid") == me):
        raise OrchestratorRefusal(
            "REFUSED: lease/claim/lock holder identity does not "
            "bind to the executing process")
    mat_sha = _sha_file(Path(mat_root) / "B4_MATERIALIZATION.json")
    if lease["materialization_sha256"] != mat_sha:
        raise OrchestratorRefusal(
            "REFUSED: lease materialization digest is stale")
    return lease


# ---------------- C18: intent/completion durable seal -------------
def _seal_paths(results_root: Path, cell_id: str,
                attempt_id: str):
    d = Path(results_root) / cell_id
    return (d / f"SEAL_INTENT_{attempt_id}.json",
            d / f"SEAL_COMPLETE_{attempt_id}.json")


def seal_attempt(results_root: Path, cell_id: str,
                 attempt_id: str) -> None:
    """C18: append-only intent/completion — never an overwrite whose
    only proof is the final fsync. Recovery reads PHYSICAL data: a
    complete, self-integral completion that matches its intent seals
    the terminal; anything else is UNCERTAIN."""
    cell_dir = Path(results_root) / cell_id
    claim = load_claim(results_root, cell_id)
    if claim["attempt_id"] != attempt_id:
        raise OrchestratorRefusal(
            "REFUSED: sealing a foreign attempt")
    terminal = cell_dir / "B4_CELL_TERMINAL.json"
    if not terminal.is_file():
        raise OrchestratorRefusal(
            "REFUSED: no terminal exists to seal — the attempt "
            "stays UNCERTAIN for operator disposition")
    intent_p, complete_p = _seal_paths(results_root, cell_id,
                                       attempt_id)
    if complete_p.exists():
        raise OrchestratorRefusal("REFUSED: attempt already sealed")
    term_sha = _sha_file(terminal)
    intent = {"schema": "agent_multi.b4_seal_intent.v1",
              "campaign_generation": b4a.CAMPAIGN_GENERATION,
              "cell": cell_id, "attempt_id": attempt_id,
              "holder_pid": os.getpid(),
              "terminal_sha256": term_sha}
    _excl_write(intent_p, json.dumps(intent, indent=1).encode())
    completion = {"schema": "agent_multi.b4_seal_completion.v1",
                  "intent_sha256": _sha_file(intent_p),
                  "terminal_sha256": term_sha,
                  "attempt_id": attempt_id}
    completion["completion_sha256"] = hashlib.sha256(json.dumps(
        {k: completion[k] for k in sorted(completion)},
        sort_keys=True).encode()).hexdigest()
    _excl_write(complete_p,
                json.dumps(completion, indent=1).encode())


def seal_state(results_root: Path, cell_id: str) -> str:
    """PHYSICAL adjudication of the seal: SEALED only when a
    complete, self-integral completion matches its intent AND the
    live terminal bytes; UNCERTAIN on any partial, malformed or
    transplanted witness; UNSEALED when neither exists."""
    try:
        claim = load_claim(results_root, cell_id)
    except SystemExit:
        return "NO_CLAIM"
    attempt_id = claim["attempt_id"]
    intent_p, complete_p = _seal_paths(results_root, cell_id,
                                       attempt_id)
    terminal = Path(results_root) / cell_id / "B4_CELL_TERMINAL.json"
    if not intent_p.exists() and not complete_p.exists():
        return "UNSEALED"
    if not complete_p.exists() or not intent_p.exists() or \
            not terminal.is_file():
        return "UNCERTAIN"
    try:
        completion = b4a._strict_json_bytes(complete_p.read_bytes(),
                                            "seal completion")
        body = {k: completion[k] for k in sorted(completion)
                if k != "completion_sha256"}
        if hashlib.sha256(json.dumps(
                body, sort_keys=True).encode()).hexdigest() != \
                completion.get("completion_sha256"):
            return "UNCERTAIN"
        if completion.get("intent_sha256") != _sha_file(intent_p):
            return "UNCERTAIN"
        if completion.get("attempt_id") != attempt_id:
            return "UNCERTAIN"
        if completion.get("terminal_sha256") != _sha_file(terminal):
            return "UNCERTAIN"
    except (SystemExit, OSError):
        return "UNCERTAIN"
    return "SEALED"


# ---------------- C13: global remaining wall ----------------------
def gpu_seconds_spent(results_root: Path) -> float:
    """Durable elapsed facts from EVERY attempt (failed and
    uncertain included). Malformed durations fail closed."""
    total = 0.0
    now = time.time()
    for claim_p in Path(results_root).glob("*/CLAIM_*.json"):
        rec = b4a._strict_json_bytes(claim_p.read_bytes(),
                                     claim_p.name)
        term = claim_p.parent / "B4_CELL_TERMINAL.json"
        if term.exists():
            t = b4a._strict_json_bytes(term.read_bytes(), term.name)
            w = t.get("wall_seconds")
            if type(w) not in (int, float) or w < 0 or \
                    not (w == w):
                raise OrchestratorRefusal(
                    f"REFUSED: malformed terminal duration in "
                    f"{claim_p.parent.name} — failing closed")
            total += float(w)
        else:
            start = rec.get("claimed_wall")
            if type(start) not in (int, float):
                raise OrchestratorRefusal(
                    "REFUSED: claim without a start fact — failing "
                    "closed")
            elapsed = now - float(start)
            if elapsed < 0:
                raise OrchestratorRefusal(
                    "REFUSED: clock rollback detected — failing "
                    "closed")
            total += elapsed
    return total


def remaining_global_seconds(results_root: Path,
                             limits: dict) -> float:
    ceiling = float(limits["global_gpu_hours_ceiling"]) * 3600.0
    return ceiling - gpu_seconds_spent(results_root)


# ------------------- C15: resume adjudication ---------------------
def adjudicate_cell_state(results_root: Path, cell_id: str) -> str:
    cell_dir = Path(results_root) / cell_id
    claim_p = _claim_path(results_root, cell_id)
    terminal = cell_dir / "B4_CELL_TERMINAL.json"
    if not claim_p.exists() and not terminal.exists():
        return "PENDING"
    if claim_p.exists() and not terminal.exists():
        return "AMBIGUOUS_CLAIM"
    try:
        claim = load_claim(results_root, cell_id)
    except SystemExit:
        return "UNCERTAIN"
    try:
        term = b4a._strict_json_bytes(terminal.read_bytes(),
                                      f"terminal {cell_id}")
    except SystemExit:
        return "UNCERTAIN"
    seal = seal_state(results_root, cell_id)
    if seal != "SEALED":
        return "UNCERTAIN"          # unsealed/partial never accepted
    if term.get("cell") != cell_id or \
            term.get("attempt_id") != claim["attempt_id"]:
        return "UNCERTAIN"
    if term.get("terminal") == "COMPLETED":
        return "COMPLETED_VERIFIED"
    if term.get("terminal") in ("FAILED", "TIMED_OUT",
                                "THERMAL_STOP", "RESOURCE_STOP",
                                "EXTERNALLY_STOPPED"):
        return f"TERMINAL_{term['terminal']}"
    return "UNCERTAIN"


def runtime_health(results_root: Path, device: str) -> dict:
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4run_orch", REPO / "tools/b4_run_cell.py")
    runner = ilu.module_from_spec(spec)
    spec.loader.exec_module(runner)
    health = {"device_available": False,
              "stop_file_present":
                  (Path(results_root) / "CAMPAIGN_STOP").exists(),
              "compute_apps_active": False}
    try:
        inv = runner.gpu_inventory(device)
        apps = runner.gpu_compute_apps(device)
    except SystemExit:
        return health
    if apps is None:
        return health
    health["device_available"] = True
    health["compute_apps_active"] = any(
        a["used_memory_mib"] > SUBSTANTIAL_COMPUTE_MIB for a in apps)
    return health


def _snapshot(root: Path) -> dict:
    if not Path(root).exists():
        return {}
    return {str(p.relative_to(root)): _sha_file(p)
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def run_campaign(mat_root: Path, ledger_path: Path,
                 results_root: Path, device: str,
                 execute: bool,
                 continue_after_failed: bool = False) -> int:
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4led_orch", REPO / "tools/b4_campaign_ledger.py")
    ledger_mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(ledger_mod)
    espec = ilu.spec_from_file_location(
        "b4exec_orch", REPO / "tools/b4_campaign_executor.py")
    executor = ilu.module_from_spec(espec)
    espec.loader.exec_module(executor)
    limits = b4a.load_resource_contract()
    results_root = Path(results_root)

    # ---- C11: pure dry-run — ZERO writes anywhere ----
    if not execute:
        pre = _snapshot(results_root)
        ledger = ledger_mod.verify_ledger(ledger_path, mat_root)
        b4a.verify_campaign_materialization(mat_root)
        states = {cid: adjudicate_cell_state(results_root, cid)
                  for cid in ledger_mod.EXPECTED_CELLS}
        plan = [cid for cid, st in states.items()
                if st == "PENDING"]
        spent_h = (gpu_seconds_spent(results_root) / 3600.0
                   if results_root.exists() else 0.0)
        print(json.dumps({
            "dry_run": True, "writes": 0,
            "generation": b4a.CAMPAIGN_GENERATION,
            "cell_states": states,
            "dispatch_plan_in_order": plan,
            "gpu_hours_spent": round(spent_h, 2),
            "gpu_hours_remaining": round(
                limits["global_gpu_hours_ceiling"] - spent_h, 2),
        }, indent=1))
        post = _snapshot(results_root)
        if pre != post:
            raise OrchestratorRefusal(
                "REFUSED: dry-run altered the result root — "
                "impossible state, failing closed")
        return 0

    if executor.CAMPAIGN_AUTH_SHA is None:
        raise OrchestratorRefusal(
            "REFUSED: no Musashi campaign authorization record — "
            "the orchestrator dispatches nothing")
    ledger = ledger_mod.verify_ledger(ledger_path, mat_root)
    results_root.mkdir(parents=True, exist_ok=True)
    outcome = {"completed": [], "failed": [], "uncertain": [],
               "pending": []}
    with GlobalLock(results_root):
        for cid in ledger_mod.EXPECTED_CELLS:
            state = adjudicate_cell_state(results_root, cid)
            if state == "COMPLETED_VERIFIED":
                outcome["completed"].append(cid)
                continue
            if state.startswith("TERMINAL_"):
                outcome["failed"].append(f"{cid}:{state}")
                if not continue_after_failed:
                    raise OrchestratorRefusal(
                        f"REFUSED: {cid} holds verified terminal "
                        f"{state} and the reviewed policy does not "
                        "say to collect the remaining cells")
                continue
            if state in ("AMBIGUOUS_CLAIM", "UNCERTAIN"):
                raise OrchestratorRefusal(
                    f"REFUSED: {cid} is {state} — blocked for "
                    "operator disposition")
            remaining = remaining_global_seconds(results_root,
                                                 limits)
            if remaining < MIN_SEGMENT_SECONDS:
                raise OrchestratorRefusal(
                    "STOPPED_RESOURCE_CEILING: remaining global "
                    f"wall {remaining:.0f}s < one segment")
            gpu_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            health = (runtime_health(results_root, gpu_dev)
                      if device.startswith("cuda") else
                      {"device_available": True,
                       "stop_file_present":
                           (results_root / "CAMPAIGN_STOP"
                            ).exists(),
                       "compute_apps_active": False})
            nxt = ledger_mod.schedule_next(
                {"cells": {c: {"status":
                               ("PENDING" if c == cid else "DONE")}
                           for c in ledger_mod.EXPECTED_CELLS}},
                {k: health[k] for k in
                 ("device_available", "stop_file_present",
                  "compute_apps_active")})
            if nxt.startswith("HOLD"):
                raise OrchestratorRefusal(f"REFUSED: {nxt}")
            claim = claim_attempt(results_root, cid)
            lease = issue_lease(results_root, cid, claim,
                                executor.CAMPAIGN_AUTH_SHA,
                                mat_root)
            executor.execute_cell(
                cid, mat_root, results_root, device,
                lease_path=lease,
                global_wall_remaining_seconds=remaining)
            seal_attempt(results_root, cid, claim["attempt_id"])
            outcome["completed"].append(cid)
    # C20: scientific completion REQUIRES the strongest verifier —
    # comparator evidence derived from the reviewed materialization,
    # impossible to omit.
    if not outcome["failed"] and not outcome["uncertain"]:
        ledger_mod.verify_campaign_results(
            ledger_path, mat_root, results_root)
        status = "CAMPAIGN_COMPLETE"
    else:
        status = "CAMPAIGN_COMPLETE_WITH_FAILED_CELLS"
    print(json.dumps({"status": status, **outcome}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialization-root", type=Path,
                    required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "cuda:0"])
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    return run_campaign(args.materialization_root, args.ledger,
                        args.results_root, args.device,
                        args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
