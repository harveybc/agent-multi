#!/usr/bin/env python3
"""B4 campaign orchestrator (order @0ce52740, C4).

The ONE dispatcher: consumes ledger + authorization record + runtime
health and executes the twelve-cell population in FIXED order. It

- claims a durable, unique attempt_id with O_CREAT|O_EXCL BEFORE any
  CUDA construction;
- holds a global exclusive lock across the PENDING -> IN_FLIGHT ->
  terminal transaction (real concurrency = 1);
- accounts global GPU wall hours from ALL attempts, failed included,
  and refuses to dispatch at the authorized ceiling;
- refuses on global stop-file, uncertain telemetry, foreign compute
  workload, or ANY ambiguous attempt (claimed, no terminal);
- never reads scores, returns or gate direction — scheduling is
  health-only by construction (it never opens a per-bar file);
- resumes only from a recognized bound bundle when an explicit
  policy in the authorization record allows it; absent that policy,
  an ambiguous attempt is BLOCKED for disposition, never re-issued.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import b4_authority as b4a  # noqa: E402

SUBSTANTIAL_COMPUTE_MIB = 1024


class OrchestratorRefusal(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class GlobalLock:
    """Real mutual exclusion via O_CREAT|O_EXCL lock file carrying
    the holder pid; released on context exit."""

    def __init__(self, results_root: Path):
        self.path = Path(results_root) / "CAMPAIGN_LOCK"
        self.fd = None

    def __enter__(self):
        try:
            self.fd = os.open(str(self.path),
                              os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise OrchestratorRefusal(
                "REFUSED: another orchestrator holds the campaign "
                f"lock ({self.path}) — concurrency is one")
        os.write(self.fd, str(os.getpid()).encode())
        os.fsync(self.fd)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            self.path.unlink(missing_ok=True)


def claim_attempt(results_root: Path, cell_id: str) -> dict:
    """Durable exclusive attempt claim BEFORE CUDA. The claim file
    is fsynced; a crash after the claim and before a terminal leaves
    the cell AMBIGUOUS and blocked, never PENDING again."""
    attempt_id = f"attempt_{uuid.uuid4().hex[:16]}"
    cell_dir = Path(results_root) / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cell_dir.glob("ATTEMPT_*.json"))
    terminal = cell_dir / "B4_CELL_TERMINAL.json"
    if existing and not terminal.exists():
        raise OrchestratorRefusal(
            f"REFUSED: {cell_id} holds a claimed attempt without a "
            "terminal — AMBIGUOUS; blocked for disposition, never "
            "re-issued")
    if terminal.exists():
        raise OrchestratorRefusal(
            f"REFUSED: {cell_id} already terminal — attempts are "
            "never reused")
    claim = cell_dir / f"ATTEMPT_{attempt_id}.json"
    rec = {"schema": "agent_multi.b4_attempt_claim.v1",
           "attempt_id": attempt_id, "cell": cell_id,
           "claimed_wall": time.time(),
           "pid": os.getpid(),
           "terminal_sha256": None}
    fd = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o644)
    try:
        os.write(fd, json.dumps(rec, indent=1).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(str(cell_dir), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return rec


def seal_attempt(results_root: Path, cell_id: str,
                 attempt_id: str) -> None:
    """After a terminal exists, bind its digest into the claim
    (write-once via replace of a file we own)."""
    cell_dir = Path(results_root) / cell_id
    claim = cell_dir / f"ATTEMPT_{attempt_id}.json"
    terminal = cell_dir / "B4_CELL_TERMINAL.json"
    rec = json.loads(claim.read_bytes())
    if rec.get("terminal_sha256") is not None:
        raise OrchestratorRefusal(
            "REFUSED: attempt already sealed")
    rec["terminal_sha256"] = _sha_file(terminal)
    tmp = claim.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, claim)


def gpu_hours_spent(results_root: Path) -> float:
    """Global GPU wall accounting from ALL attempts (failed and
    ambiguous included): terminal wall_seconds where present, else
    elapsed since the claim."""
    total = 0.0
    now = time.time()
    for claim in Path(results_root).glob("*/ATTEMPT_*.json"):
        rec = json.loads(claim.read_bytes())
        cell_dir = claim.parent
        term = cell_dir / "B4_CELL_TERMINAL.json"
        if term.exists():
            t = json.loads(term.read_bytes())
            total += float(t.get("wall_seconds", 0.0) or 0.0)
        else:
            total += max(0.0, now - float(rec["claimed_wall"]))
    return total / 3600.0


def runtime_health(results_root: Path, device: str) -> dict:
    """Health facts ONLY — this function never opens a per-bar file,
    a terminal's scores or any return series."""
    runner_spec = __import__("importlib.util", fromlist=["util"])
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4run_orch", REPO / "tools/b4_run_cell.py")
    runner = ilu.module_from_spec(spec)
    spec.loader.exec_module(runner)
    health = {"device_available": False,
              "stop_file_present":
                  (Path(results_root) / "CAMPAIGN_STOP").exists(),
              "compute_apps_active": False,
              "gpu_temperature_celsius": None,
              "gpu_memory_free_mib": None}
    try:
        inv = runner.gpu_inventory(device)
        apps = runner.gpu_compute_apps(device)
    except SystemExit:
        return health   # telemetry uncertain -> not available
    if apps is None:
        return health
    health["device_available"] = True
    health["gpu_temperature_celsius"] = inv["temperature_celsius"]
    health["gpu_memory_free_mib"] = (inv["memory_total_mib"]
                                     - inv["memory_used_mib"])
    health["compute_apps_active"] = any(
        a["used_memory_mib"] > SUBSTANTIAL_COMPUTE_MIB for a in apps)
    return health


def run_campaign(mat_root: Path, ledger_path: Path,
                 results_root: Path, device: str,
                 execute: bool) -> int:
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "b4led_orch", REPO / "tools/b4_campaign_ledger.py")
    ledger_mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(ledger_mod)
    espec = ilu.spec_from_file_location(
        "b4exec_orch", REPO / "tools/b4_campaign_executor.py")
    executor = ilu.module_from_spec(espec)
    espec.loader.exec_module(executor)
    if execute and executor.CAMPAIGN_AUTH_SHA is None:
        raise OrchestratorRefusal(
            "REFUSED: no Musashi campaign authorization record — "
            "the orchestrator dispatches nothing")
    limits = b4a.load_resource_contract()
    ledger = ledger_mod.verify_ledger(ledger_path, mat_root)
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    with GlobalLock(results_root):
        for cid in ledger_mod.EXPECTED_CELLS:
            terminal = (results_root / cid /
                        "B4_CELL_TERMINAL.json")
            if terminal.exists():
                continue
            spent = gpu_hours_spent(results_root)
            if spent >= limits["global_gpu_hours_ceiling"]:
                raise OrchestratorRefusal(
                    f"REFUSED: global GPU ceiling reached "
                    f"({spent:.1f} h >= "
                    f"{limits['global_gpu_hours_ceiling']} h)")
            gpu_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            health = runtime_health(results_root, gpu_dev) if \
                device.startswith("cuda") else {
                    "device_available": True,
                    "stop_file_present":
                        (results_root / "CAMPAIGN_STOP").exists(),
                    "compute_apps_active": False}
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
            if not execute:
                print(json.dumps(
                    {"cell": cid, "claimed": claim["attempt_id"],
                     "would_execute": True,
                     "gpu_hours_spent": round(spent, 2)}))
                seal_attempt(results_root, cid,
                             claim["attempt_id"])
                return 0
            executor.execute_cell(cid, mat_root, results_root,
                                  device,
                                  attempt_id=claim["attempt_id"])
            seal_attempt(results_root, cid, claim["attempt_id"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialization-root", type=Path,
                    required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "cuda:0"])
    ap.add_argument("--execute", action="store_true",
                    help="without this flag: claim-and-report only")
    args = ap.parse_args(argv)
    return run_campaign(args.materialization_root, args.ledger,
                        args.results_root, args.device,
                        args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
