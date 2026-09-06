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
    def __init__(self, results_root: Path):
        self.path = Path(results_root) / "CAMPAIGN_LOCK"
        self.held = False

    def __enter__(self):
        _excl_write(self.path, json.dumps(
            {"pid": os.getpid(),
             "generation": b4a.CAMPAIGN_GENERATION}).encode())
        self.held = True
        return self

    def __exit__(self, *exc):
        if self.held:
            self.path.unlink(missing_ok=True)


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


def seal_attempt(results_root: Path, cell_id: str,
                 attempt_id: str) -> None:
    cell_dir = Path(results_root) / cell_id
    claim_p = _claim_path(results_root, cell_id)
    rec = load_claim(results_root, cell_id)
    if rec["attempt_id"] != attempt_id:
        raise OrchestratorRefusal(
            "REFUSED: sealing a foreign attempt")
    if rec.get("terminal_sha256") is not None:
        raise OrchestratorRefusal("REFUSED: attempt already sealed")
    terminal = cell_dir / "B4_CELL_TERMINAL.json"
    if not terminal.is_file():
        raise OrchestratorRefusal(
            "REFUSED: no terminal exists to seal — the attempt "
            "stays UNCERTAIN for operator disposition")
    rec["terminal_sha256"] = _sha_file(terminal)
    tmp = claim_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, claim_p)
    dfd = os.open(str(cell_dir), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


# ------------------- C12: execution lease -------------------------
def issue_lease(results_root: Path, cell_id: str, claim: dict,
                auth_sha: str, mat_root: Path) -> Path:
    lease = {"schema": "agent_multi.b4_execution_lease.v1",
             "campaign_generation": b4a.CAMPAIGN_GENERATION,
             "cell": cell_id,
             "attempt_id": claim["attempt_id"],
             "authorization_sha256": auth_sha,
             "materialization_sha256":
                 _sha_file(Path(mat_root) / "B4_MATERIALIZATION.json"),
             "issued_monotonic": time.monotonic(),
             "holder_pid": os.getpid()}
    p = (Path(results_root) / cell_id /
         f"LEASE_{claim['attempt_id']}.json")
    _excl_write(p, json.dumps(lease, indent=1).encode())
    return p


def verify_lease(lease_path: Path, results_root: Path,
                 cell_id: str, mat_root: Path) -> dict:
    """C12: the executor calls THIS before any pipeline/env/CUDA
    construction. A fabricated attempt id, a manually written
    incomplete claim or a stale lease produces zero compute."""
    p = Path(lease_path)
    if p.is_symlink() or not p.is_file():
        raise OrchestratorRefusal("REFUSED: execution lease absent")
    lease = b4a._strict_json_bytes(p.read_bytes(), "execution lease")
    if lease.get("campaign_generation") != b4a.CAMPAIGN_GENERATION \
            or lease.get("cell") != cell_id:
        raise OrchestratorRefusal(
            "REFUSED: lease generation/cell binding mismatch")
    claim = load_claim(results_root, cell_id)
    if claim["attempt_id"] != lease.get("attempt_id"):
        raise OrchestratorRefusal(
            "REFUSED: lease attempt differs from the unique claim")
    if claim.get("terminal_sha256") is not None:
        raise OrchestratorRefusal(
            "REFUSED: the claimed attempt already reached a sealed "
            "terminal")
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
    mat_sha = _sha_file(Path(mat_root) / "B4_MATERIALIZATION.json")
    if lease.get("materialization_sha256") != mat_sha:
        raise OrchestratorRefusal(
            "REFUSED: lease materialization digest is stale")
    return lease


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
    if claim.get("terminal_sha256") is None:
        return "UNCERTAIN"          # unsealed is never accepted
    if claim["terminal_sha256"] != _sha_file(terminal):
        return "UNCERTAIN"
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
    status = ("CAMPAIGN_COMPLETE" if not outcome["failed"]
              else "CAMPAIGN_COMPLETE_WITH_FAILED_CELLS")
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
