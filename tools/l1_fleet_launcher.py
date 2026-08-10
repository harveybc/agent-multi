#!/usr/bin/env python3
"""Durable, idempotent L1 factorial launcher (order §2/WP1).

One seed per invocation, contract-enforced hostname + GPU-UUID
assignment, one exclusive ``flock`` per experiment/seed and one per
cell (the exclusive_claim files under ``<experiment>/locks/``). A
second invocation while the first is alive returns ``ALREADY_RUNNING``
with the holder's PID/start identity; a completed seed returns
``ALREADY_COMPLETE``; any contract violation returns a typed refusal.
There is never a second writer: the kernel drops the flock with the
process, so a crashed holder frees the claim, and recovery lands in a
NEW content-addressed attempt directory (the runner never overwrites a
partial one).

A heartbeat file (atomic replace) carries seed, cell, attempt, PID and
PID start identity, progress, last artifact and terminal state — the
durable workload signal the fleet doctrine requires. The systemd
template under ``examples/systemd/l1-factorial@.service`` restarts on
failure; the flock makes a restart racing a live holder exit
``ALREADY_RUNNING`` instead of double-writing.

Typed outcomes (stdout JSON, one line per seed run) and their exit
classes (order §2/WP7, finding 188):

  exit 0  SEED_COMPLETE | ALREADY_COMPLETE   clean terminal
  exit 3  ALREADY_RUNNING                    clean no-op (SuccessExit)
  exit 4  REFUSED_WRONG_HOST | REFUSED_GPU_UNBOUND |
          REFUSED_BAD_CONTRACT               typed configuration
                                             refusal; heartbeat
                                             written; never blindly
                                             restarted
  exit 1  SEED_FAILED                        real failure; the ONLY
                                             class Restart=on-failure
                                             retries
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import l1_factorial_screen as runner  # noqa: E402

LAUNCHER_VERSION = "l1_fleet_launcher.v2"
HEARTBEAT_INTERVAL_S = 60

# Exit classes are a CONTRACT shared with the systemd unit; changing a
# code without changing the unit is a finding, not a refactor.
EXIT_CLASS = {
    "SEED_COMPLETE": 0,
    "ALREADY_COMPLETE": 0,
    "ALREADY_RUNNING": 3,
    "REFUSED_WRONG_HOST": 4,
    "REFUSED_GPU_UNBOUND": 4,
    "REFUSED_BAD_CONTRACT": 4,
    "SEED_FAILED": 1,
}


def _pid_start_identity(pid: int) -> str | None:
    """Kernel start time of the pid — a PID alone is reusable, the
    (pid, starttime) pair is not."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[19]
    except Exception:
        return None


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True,
                              default=str) + "\n")
    os.replace(tmp, path)


def visible_gpu_uuids() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:
        return []
    return [seg.split(")")[0] for seg in out.split("UUID: ")[1:]]


class ExclusiveClaim:
    """flock-backed exclusive claim with a PID/start-identity sidecar."""

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self._fd = fd
        pid = os.getpid()
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({
            "pid": pid,
            "pid_start_identity": _pid_start_identity(pid),
            "acquired_utc": datetime.now(timezone.utc).isoformat(),
            "launcher_version": LAUNCHER_VERSION,
        }).encode())
        os.fsync(fd)
        return True

    def holder(self) -> dict:
        try:
            return json.loads(self.path.read_text() or "{}")
        except Exception:
            return {}

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


class SeedLauncher:
    def __init__(self, *, contract: dict, manifest: dict, seed: int,
                 smoke: bool, hostname: str | None = None,
                 gpu_uuids: list[str] | None = None,
                 run_cell_fn=None, enforce_gpu: bool = True):
        self.contract = contract
        self.manifest = manifest
        self.seed = seed
        self.smoke = smoke
        self.hostname = hostname or socket.gethostname()
        self.gpu_uuids = (visible_gpu_uuids() if gpu_uuids is None
                          else gpu_uuids)
        self.run_cell_fn = run_cell_fn or runner.run_cell
        self.enforce_gpu = enforce_gpu
        self._heartbeat_path: Path | None = None
        self._hb_state: dict = {}
        self._hb_stop = threading.Event()

    # -- assignment ----------------------------------------------------
    def check_assignment(self) -> dict | None:
        assignment = (self.contract.get("assignments") or {}).get(
            str(self.seed))
        if not assignment:
            return {"outcome": "REFUSED_BAD_CONTRACT",
                    "reason": f"seed {self.seed} has no contract "
                              "assignment"}
        if assignment.get("hostname") != self.hostname:
            return {"outcome": "REFUSED_WRONG_HOST",
                    "reason": (f"seed {self.seed} is assigned to "
                               f"{assignment.get('hostname')!r}, this is "
                               f"{self.hostname!r}")}
        if self.enforce_gpu and \
                assignment.get("gpu_uuid") not in self.gpu_uuids:
            return {"outcome": "REFUSED_GPU_UNBOUND",
                    "reason": (f"assigned GPU {assignment.get('gpu_uuid')}"
                               f" not visible on {self.hostname}")}
        return None

    # -- heartbeat -----------------------------------------------------
    def _heartbeat(self, **update) -> None:
        if self._heartbeat_path is None:
            return
        self._hb_state.update(update)
        pid = os.getpid()
        self._hb_state.update({
            "schema": "agent_multi.l1_launcher_heartbeat.v1",
            "seed": self.seed,
            "pid": pid,
            "pid_start_identity": _pid_start_identity(pid),
            "hostname": self.hostname,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        _atomic_json(self._heartbeat_path, self._hb_state)

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(HEARTBEAT_INTERVAL_S):
            self._heartbeat()

    # -- run -----------------------------------------------------------
    def run(self) -> dict:
        exp_id = runner.experiment_identity(self.contract, self.manifest,
                                            self.smoke)
        out_root = Path(self.contract["output_root"]).expanduser()
        exp_dir = out_root / exp_id
        locks = exp_dir / "locks"
        self._heartbeat_path = (exp_dir / f"seed{self.seed}" /
                                "launcher_heartbeat.json")
        self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        refusal = self.check_assignment()
        if refusal:
            # A configuration refusal is VISIBLE: the heartbeat carries
            # the typed outcome so an idle worker is never silent.
            self._heartbeat(terminal_state=refusal["outcome"],
                            error=refusal.get("reason"))
            refusal["experiment_id"] = exp_id
            return refusal

        seed_claim = ExclusiveClaim(
            locks / f"exclusive_claim.seed{self.seed}.lock")
        if not seed_claim.acquire():
            return {"outcome": "ALREADY_RUNNING",
                    "experiment_id": exp_id,
                    "holder": seed_claim.holder()}
        hb_thread = threading.Thread(target=self._heartbeat_loop,
                                     daemon=True)
        hb_thread.start()
        try:
            cells = list(self.contract["cells"])
            completed, reused = [], []
            for cell in cells:
                cell_claim = ExclusiveClaim(
                    locks / f"exclusive_claim.seed{self.seed}"
                            f".{cell}.lock")
                if not cell_claim.acquire():
                    return {"outcome": "ALREADY_RUNNING",
                            "experiment_id": exp_id, "cell": cell,
                            "holder": cell_claim.holder()}
                try:
                    self._heartbeat(cell=cell, terminal_state="RUNNING",
                                    progress=f"{len(completed)}/"
                                             f"{len(cells)} cells")
                    record = self.run_cell_fn(
                        cell, self.seed, contract=self.contract,
                        manifest=self.manifest, smoke=self.smoke)
                    (reused if record.get("_reuse") == "ALREADY_COMPLETE"
                     else completed).append(cell)
                    self._heartbeat(
                        attempt=record.get("attempt_dir"),
                        last_artifact=record.get("terminal_model_path"))
                finally:
                    cell_claim.release()
            outcome = ("ALREADY_COMPLETE"
                       if reused and not completed else "SEED_COMPLETE")
            self._heartbeat(terminal_state=outcome, cell=None,
                            progress=f"{len(cells)}/{len(cells)} cells")
            return {"outcome": outcome, "experiment_id": exp_id,
                    "completed_cells": completed, "reused_cells": reused}
        except Exception as exc:
            self._heartbeat(terminal_state="SEED_FAILED",
                            error=f"{type(exc).__name__}: {exc}")
            return {"outcome": "SEED_FAILED", "experiment_id": exp_id,
                    "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._hb_stop.set()
            seed_claim.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-gpu-check", action="store_true",
                        help="skip GPU-UUID visibility enforcement "
                             "(CPU smoke only)")
    parser.add_argument("--contract", default=None,
                        help="contract path override (tests)")
    parser.add_argument("--manifest", default=None,
                        help="system manifest path override (tests)")
    args = parser.parse_args()
    contract = runner.load_contract(
        Path(args.contract)) if args.contract else runner.load_contract()
    manifest = runner.load_system_manifest(
        Path(args.manifest)) if args.manifest \
        else runner.load_system_manifest()
    launcher = SeedLauncher(contract=contract, manifest=manifest,
                            seed=args.seed, smoke=args.smoke,
                            enforce_gpu=not args.no_gpu_check)
    result = launcher.run()
    print(json.dumps(result, default=str), flush=True)
    return EXIT_CLASS.get(result["outcome"], 1)


if __name__ == "__main__":
    sys.exit(main())
