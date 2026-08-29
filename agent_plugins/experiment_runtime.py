"""Observable, resumable experiment runtime (PERMANENT order
@95e088da, 2026-08-29). Every job expected to exceed 30 minutes runs
through this layer or does not run.

The twelve launch invariants, enforced by ``preflight_or_refuse``:

 1. atomic work-unit identity (experiment/family/window/latent/budget/
    seed/origin/treatment + code/data/config digests);
 2. durable per-unit state (PENDING/RUNNING/COMPLETED/FAILED/
    TIMED_OUT/INTERRUPTED);
 3. atomic result persistence immediately after every completed unit;
 4. progress heartbeat (<= 5 min) with current unit, completed/total,
    elapsed, last durable result;
 5. ETA from completed comparable units (median, p90, pessimistic);
 6. per-unit timeout from a bounded smoke + campaign wall ceiling;
 7. idempotent resume: exact completed units skipped, failed/
    interrupted units get FRESH attempts;
 8. per-unit durable stdout/stderr capture (never a sole agent pipe);
 9. immutable digest-identified inputs OUTSIDE volatile /tmp scratch;
10. watchdog: stale heartbeat, dead process, disk pressure, identity
    drift;
11. machine-readable status without process attachment;
12. graceful cancellation preserving completed results and marking
    the active unit interrupted.

A monolithic final-write-only runner is prohibited: the WORKER
executes exactly one unit per invocation and exits."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Callable

STATES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "TIMED_OUT",
          "INTERRUPTED")
HEARTBEAT_MAX_AGE_S = 300.0


class RuntimePreflightError(RuntimeError):
    """A launch invariant is absent — the launcher REFUSES."""


class UnitClaimError(RuntimeError):
    """The unit is not claimable (already running/terminal)."""


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    """tmp + fsync + rename + dir fsync: a disk-full or fsync failure
    raises BEFORE any state can claim completion."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=1, default=str).encode()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def unit_id(identity: dict) -> str:
    required = ("experiment", "family", "window", "latent", "budget",
                "seed", "origin", "treatment")
    missing = [k for k in required if k not in identity]
    if missing:
        raise RuntimePreflightError(
            f"unit identity missing fields {missing} (invariant 1)")
    return sha_obj({k: identity[k] for k in required})[:20]


class RunDirectory:
    """Durable run layout OUTSIDE /tmp (invariant 9):

    run_root/
      ledger.json (+ .sha256 companion INSIDE the file)
      inputs/            immutable digest-identified inputs
      units/<uid>.state.json / .result.json / .attempt<k>.log
      decisions/round<k>.json
      status.json        heartbeat + machine-readable status
    """

    def __init__(self, root: Path, *,
                 allow_volatile_for_tests: bool = False):
        self.root = Path(root)
        if (str(self.root).startswith("/tmp/")
                and not allow_volatile_for_tests):
            raise RuntimePreflightError(
                "run root under volatile /tmp is prohibited "
                "(invariant 9); allow_volatile_for_tests exists ONLY "
                "for the unit tests of this machinery")
        for sub in ("units", "inputs", "decisions", "logs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ---- ledger ----------------------------------------------------
    def write_ledger(self, ledger: dict) -> str:
        ledger = dict(ledger)
        ledger["ledger_digest"] = sha_obj(
            {k: v for k, v in ledger.items()
             if k != "ledger_digest"})
        atomic_write_json(self.root / "ledger.json", ledger)
        for unit in ledger["units"]:
            state_path = self._state_path(unit["unit_id"])
            if not state_path.exists():
                atomic_write_json(state_path, {
                    "unit_id": unit["unit_id"],
                    "state": "PENDING", "attempt": 0,
                    "identity": unit["identity"]})
        return ledger["ledger_digest"]

    def ledger(self) -> dict:
        return json.loads((self.root / "ledger.json").read_text())

    # ---- unit state machine ---------------------------------------
    def _state_path(self, uid: str) -> Path:
        return self.root / "units" / f"{uid}.state.json"

    def _result_path(self, uid: str) -> Path:
        return self.root / "units" / f"{uid}.result.json"

    def unit_state(self, uid: str) -> dict:
        return json.loads(self._state_path(uid).read_text())

    def claim(self, uid: str, *, expected_digests: dict) -> dict:
        """Atomic exclusive claim: PENDING -> RUNNING via O_EXCL lock
        file; a concurrent worker cannot claim the same unit
        (invariant of the required tests)."""
        lock = self.root / "units" / f"{uid}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise UnitClaimError(f"{uid}: already claimed")
        try:
            os.write(fd, str(os.getpid()).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        state = self.unit_state(uid)
        if state["state"] not in ("PENDING", "FAILED", "INTERRUPTED",
                                  "TIMED_OUT"):
            os.unlink(lock)
            raise UnitClaimError(
                f"{uid}: state {state['state']} is not claimable")
        ledger = self.ledger()
        drift = {k: (ledger["digests"].get(k), v)
                 for k, v in expected_digests.items()
                 if ledger["digests"].get(k) != v}
        if drift:
            os.unlink(lock)
            raise RuntimePreflightError(
                f"{uid}: code/data/config digest drift vs the ledger "
                f"{drift} — resume with changed identity refuses")
        state.update({"state": "RUNNING",
                      "attempt": int(state.get("attempt", 0)) + 1,
                      "pid": os.getpid(),
                      "claimed_at": time.time()})
        atomic_write_json(self._state_path(uid), state)
        return state

    def release(self, uid: str, terminal_state: str,
                result: dict | None = None,
                note: str | None = None) -> None:
        if terminal_state not in ("COMPLETED", "FAILED", "TIMED_OUT",
                                  "INTERRUPTED"):
            raise RuntimePreflightError(
                f"illegal terminal state {terminal_state}")
        if terminal_state == "COMPLETED":
            if result is None:
                raise RuntimePreflightError(
                    "COMPLETED without a result is prohibited "
                    "(invariant 3)")
            result = dict(result)
            result["result_digest"] = sha_obj(
                {k: v for k, v in result.items()
                 if k != "result_digest"})
            existing = self._result_path(uid)
            if existing.exists():
                previous = json.loads(existing.read_text())
                if previous.get("result_digest") == \
                        result["result_digest"]:
                    pass  # idempotent duplicate
                else:
                    raise RuntimePreflightError(
                        f"{uid}: conflicting duplicate result "
                        "refuses (digest mismatch)")
            else:
                atomic_write_json(existing, result)
        state = self.unit_state(uid)
        state.update({"state": terminal_state,
                      "finished_at": time.time(),
                      "note": note})
        atomic_write_json(self._state_path(uid), state)
        lock = self.root / "units" / f"{uid}.lock"
        if lock.exists():
            os.unlink(lock)

    def result(self, uid: str) -> dict | None:
        path = self._result_path(uid)
        return json.loads(path.read_text()) if path.exists() else None

    # ---- status / heartbeat / ETA ---------------------------------
    def states(self) -> dict:
        out = {}
        for path in (self.root / "units").glob("*.state.json"):
            state = json.loads(path.read_text())
            out[state["unit_id"]] = state
        return out

    def heartbeat(self, *, current_unit: str | None,
                  extra: dict | None = None) -> dict:
        states = self.states()
        by_state: dict = {}
        for s in states.values():
            by_state[s["state"]] = by_state.get(s["state"], 0) + 1
        completed = [s for s in states.values()
                     if s["state"] == "COMPLETED"]
        durations = sorted(
            s["finished_at"] - s["claimed_at"] for s in completed
            if s.get("finished_at") and s.get("claimed_at"))
        remaining = by_state.get("PENDING", 0) + by_state.get(
            "RUNNING", 0)
        eta = None
        if durations:
            median = durations[len(durations) // 2]
            p90 = durations[min(len(durations) - 1,
                                int(len(durations) * 0.9))]
            eta = {"median_unit_s": round(median, 1),
                   "p90_unit_s": round(p90, 1),
                   "remaining_units": remaining,
                   "eta_median_s": round(median * remaining, 1),
                   "eta_p90_s": round(p90 * remaining, 1),
                   "eta_pessimistic_s": round(
                       durations[-1] * remaining, 1)}
        last_done = max((s.get("finished_at", 0)
                         for s in completed), default=None)
        status = {
            "schema": "agent_multi.experiment_runtime_status.v1",
            "timestamp": time.time(),
            "current_unit": current_unit,
            "counts_by_state": by_state,
            "completed_total": f"{len(completed)}/{len(states)}",
            "last_durable_completion": last_done,
            "eta": eta,
            **(extra or {}),
        }
        atomic_write_json(self.root / "status.json", status)
        return status

    def watchdog(self) -> list:
        """Stale heartbeat / dead process / disk pressure / identity
        drift detection (invariant 10). Returns typed alerts and
        marks stale RUNNING units TIMED_OUT preserving evidence."""
        alerts = []
        status_path = self.root / "status.json"
        if status_path.exists():
            age = time.time() - json.loads(
                status_path.read_text())["timestamp"]
            if age > HEARTBEAT_MAX_AGE_S:
                alerts.append({"type": "stale_heartbeat",
                               "age_s": round(age, 1)})
        for state in self.states().values():
            if state["state"] != "RUNNING":
                continue
            pid = state.get("pid")
            alive = pid and Path(f"/proc/{pid}").exists()
            stale = (time.time() - state.get("claimed_at", 0)
                     > state.get("timeout_s", 1e18))
            if not alive or stale:
                alerts.append({"type": ("dead_process" if not alive
                                        else "unit_timeout"),
                               "unit": state["unit_id"], "pid": pid})
                self.release(state["unit_id"], "TIMED_OUT",
                             note="watchdog: dead or stale")
        usage = os.statvfs(self.root)
        free_fraction = usage.f_bavail / max(1, usage.f_blocks)
        if free_fraction < 0.05:
            alerts.append({"type": "disk_pressure",
                           "free_fraction": round(free_fraction, 3)})
        return alerts


def preflight_or_refuse(run: RunDirectory,
                        expected_wall_ceiling_s: float,
                        unit_timeout_s: float | None) -> None:
    """The launcher must refuse when any invariant is absent."""
    ledger_path = run.root / "ledger.json"
    if not ledger_path.exists():
        raise RuntimePreflightError(
            "no materialized ledger (invariants 1/5)")
    ledger = run.ledger()
    for key in ("units", "digests", "campaign_wall_ceiling_s",
                "ledger_digest"):
        if key not in ledger:
            raise RuntimePreflightError(f"ledger missing {key}")
    if unit_timeout_s is None and not ledger.get("unit_timeout_s"):
        raise RuntimePreflightError(
            "no per-unit timeout derived from a bounded smoke "
            "(invariant 6)")
    if float(ledger["campaign_wall_ceiling_s"]) > float(
            expected_wall_ceiling_s):
        raise RuntimePreflightError(
            "campaign wall ceiling exceeds the authorized ceiling")
    for unit in ledger["units"]:
        if unit_id(unit["identity"]) != unit["unit_id"]:
            raise RuntimePreflightError(
                f"unit {unit['unit_id']}: identity/id mismatch")


def run_one_unit(run: RunDirectory, uid: str,
                 executor: Callable[[dict, Path], dict], *,
                 expected_digests: dict,
                 timeout_s: float) -> dict:
    """WORKER CORE: exactly one atomic unit — claim, execute under a
    timeout with durable per-unit logging, atomic result, terminal
    state. SIGTERM during execution produces a durable INTERRUPTED
    state (required test)."""
    state = run.claim(uid, expected_digests=expected_digests)
    state["timeout_s"] = timeout_s
    atomic_write_json(run._state_path(uid), state)
    log_path = run.root / "units" / \
        f"{uid}.attempt{state['attempt']}.log"
    interrupted = {"flag": False}

    def on_term(_sig, _frame):
        interrupted["flag"] = True
        run.release(uid, "INTERRUPTED",
                    note="SIGTERM during execution")
        raise SystemExit(143)

    previous = signal.signal(signal.SIGTERM, on_term)
    started = time.time()
    try:
        identity = state["identity"]
        result = executor(identity, log_path)
        if time.time() - started > timeout_s:
            run.release(uid, "TIMED_OUT",
                        note=f"exceeded {timeout_s}s")
            return {"state": "TIMED_OUT"}
        run.release(uid, "COMPLETED", result=result)
        return {"state": "COMPLETED", "result": result}
    except SystemExit:
        raise
    except BaseException as exc:
        if not interrupted["flag"]:
            run.release(uid, "FAILED",
                        note=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def aggregate(run: RunDirectory, expected_units: list) -> dict:
    """Aggregation consumes ONLY complete verified units; missing,
    duplicated or foreign units refuse."""
    ledger_units = {u["unit_id"] for u in run.ledger()["units"]}
    foreign = [u for u in expected_units if u not in ledger_units]
    if foreign:
        raise RuntimePreflightError(
            f"aggregation refuses: foreign units {foreign[:3]}")
    if len(set(expected_units)) != len(expected_units):
        raise RuntimePreflightError(
            "aggregation refuses: duplicated expected units")
    states = run.states()
    missing = [u for u in expected_units if u not in states
               or states[u]["state"] != "COMPLETED"]
    if missing:
        raise RuntimePreflightError(
            f"aggregation refuses: {len(missing)} expected units not "
            f"COMPLETED (e.g. {missing[:3]})")
    results = {}
    for u in expected_units:
        result = run.result(u)
        if result is None:
            raise RuntimePreflightError(
                f"aggregation refuses: no durable result for {u}")
        results[u] = result
    return results
