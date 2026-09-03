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
        """R1 (2026-09-03): every read verifies the ledger's own
        digest — a tampered or torn ledger REFUSES instead of
        steering the campaign."""
        ledger = json.loads((self.root / "ledger.json").read_text())
        expected = ledger.get("ledger_digest")
        actual = sha_obj({k: v for k, v in ledger.items()
                          if k != "ledger_digest"})
        if expected != actual:
            raise RuntimePreflightError(
                "ledger self-digest mismatch — tampered or torn "
                "ledger refuses")
        return ledger

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
                note: str | None = None,
                attempt: int | None = None) -> None:
        """R1 (2026-09-03): compare-and-set semantics.

        * a TERMINAL state never overwrites another terminal state
          (the only exception is an IDEMPOTENT duplicate COMPLETED
          with a bit-identical result);
        * when ``attempt`` is given, a stale actor from an earlier
          attempt cannot finalize a unit that was re-claimed;
        * COMPLETED requires the unit to be RUNNING.
        """
        if terminal_state not in ("COMPLETED", "FAILED", "TIMED_OUT",
                                  "INTERRUPTED"):
            raise RuntimePreflightError(
                f"illegal terminal state {terminal_state}")
        state = self.unit_state(uid)
        current = state.get("state")
        if attempt is not None and \
                int(state.get("attempt", 0)) != int(attempt):
            raise RuntimePreflightError(
                f"{uid}: stale actor (attempt {attempt} vs current "
                f"{state.get('attempt')}) — release refused")
        if terminal_state == "COMPLETED":
            if result is None:
                raise RuntimePreflightError(
                    "COMPLETED without a result is prohibited "
                    "(invariant 3)")
            result = dict(result)
            result["unit_id"] = uid
            result["result_digest"] = sha_obj(
                {k: v for k, v in result.items()
                 if k != "result_digest"})
            existing = self._result_path(uid)
            if existing.exists():
                previous = json.loads(existing.read_text())
                if previous.get("result_digest") == \
                        result["result_digest"]:
                    return  # idempotent duplicate — nothing moves
                raise RuntimePreflightError(
                    f"{uid}: conflicting duplicate result "
                    "refuses (digest mismatch)")
            if current in ("COMPLETED", "FAILED", "TIMED_OUT",
                           "INTERRUPTED"):
                raise RuntimePreflightError(
                    f"{uid}: terminal state {current!r} never "
                    "overwritten by COMPLETED — a late completer "
                    "refuses")
            if current != "RUNNING":
                raise RuntimePreflightError(
                    f"{uid}: COMPLETED requires RUNNING, found "
                    f"{current!r}")
            atomic_write_json(existing, result)
        else:
            if current in ("COMPLETED", "FAILED", "TIMED_OUT",
                           "INTERRUPTED"):
                raise RuntimePreflightError(
                    f"{uid}: terminal state {current!r} never "
                    f"overwritten by {terminal_state} — refused")
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

    @staticmethod
    def stratum_key(identity: dict) -> str:
        """R1 (2026-09-03): comparable units share family, window,
        latent, treatment and budget — ETA is computed WITHIN a
        stratum, never across (permanent order 95e088da)."""
        return "|".join(str(identity.get(k)) for k in (
            "treatment", "family", "window", "latent", "budget"))

    def heartbeat(self, *, current_unit: str | None,
                  workers: int | None = None,
                  device_class: str | None = None,
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
        pooled = None
        if durations:
            median = durations[len(durations) // 2]
            p90 = durations[min(len(durations) - 1,
                                int(len(durations) * 0.9))]
            pooled = {"median_unit_s": round(median, 1),
                      "p90_unit_s": round(p90, 1),
                      "remaining_units": remaining,
                      "note": "POOLED across strata — diagnostic "
                              "only, never the published ETA"}
        # ---- stratified ETA (CE1): per-stratum medians/p90 over
        # comparable units, divided by the workers actually
        # available; unmeasured strata are declared, not guessed
        strata: dict = {}
        for s in states.values():
            key = self.stratum_key(s.get("identity") or {})
            entry = strata.setdefault(key, {
                "done_s": [], "remaining": 0})
            if s["state"] == "COMPLETED" and s.get("finished_at")                     and s.get("claimed_at"):
                entry["done_s"].append(
                    s["finished_at"] - s["claimed_at"])
            elif s["state"] in ("PENDING", "RUNNING"):
                entry["remaining"] += 1
        eta_med = eta_p90 = 0.0
        unmeasured = []
        per_stratum = {}
        for key, entry in sorted(strata.items()):
            done = sorted(entry["done_s"])
            rem = entry["remaining"]
            if not rem:
                continue
            if not done:
                unmeasured.append({"stratum": key,
                                   "remaining": rem})
                continue
            med = done[len(done) // 2]
            p90 = done[min(len(done) - 1, int(len(done) * 0.9))]
            per_stratum[key] = {
                "median_s": round(med, 1), "p90_s": round(p90, 1),
                "measured": len(done), "remaining": rem}
            eta_med += med * rem
            eta_p90 += p90 * rem
        effective_workers = max(1, int(workers or 1))
        eta = {
            "stratified": per_stratum,
            "unmeasured_strata": unmeasured,
            "workers_assumed": effective_workers,
            "eta_interval_s": [
                round(eta_med / effective_workers, 1),
                round(eta_p90 / effective_workers, 1)],
            "assumptions": [
                "per-stratum median/p90 over completed comparable "
                "units only",
                f"divided by {effective_workers} workers with "
                "ideal packing (real packing can only be worse "
                "than the lower bound, better than serial)",
                "unmeasured strata excluded and listed — the "
                "interval is a lower bound until they measure"],
            "pooled_unstratified_diagnostic": pooled,
        } if strata else None
        active = [{"unit": s["unit_id"], "pid": s.get("pid"),
                   "attempt": s.get("attempt"),
                   "elapsed_s": round(
                       time.time() - s.get("claimed_at", time.time()),
                       1),
                   "stratum": self.stratum_key(
                       s.get("identity") or {})}
                  for s in states.values()
                  if s["state"] == "RUNNING"]
        last_done = max((s.get("finished_at", 0)
                         for s in completed), default=None)
        status = {
            "schema": "agent_multi.experiment_runtime_status.v2",
            "timestamp": time.time(),
            "current_unit": current_unit,
            "active_units": active,
            "device_class": device_class,
            "counts_by_state": by_state,
            "completed_total": f"{len(completed)}/{len(states)}",
            "last_durable_completion": last_done,
            "eta": eta,
            **(extra or {}),
        }
        atomic_write_json(self.root / "status.json", status)
        return status

    def watchdog(self, *, kill_child=None,
                 expected_digests: dict | None = None,
                 temperature_reader=None,
                 gpu_max_c: float = 87.0,
                 cpu_max_c: float = 95.0) -> list:
        """Stale heartbeat / dead process / unit timeout / disk
        pressure / THERMAL / IDENTITY-DRIFT detection (invariant 10,
        completed per R1 2026-09-03).

        Race-free timeout: a RUNNING unit whose process is ALIVE is
        never marked TIMED_OUT unless ``kill_child(pid)`` terminates
        AND REAPS it first — only a confirmed-dead child can be
        released, with attempt CAS, so a zombie can never write
        COMPLETED over a TIMED_OUT terminal. Without a killer the
        watchdog only ALERTS on an alive-but-stale unit."""
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
            if not alive:
                alerts.append({"type": "dead_process",
                               "unit": state["unit_id"], "pid": pid})
                self.release(state["unit_id"], "TIMED_OUT",
                             note="watchdog: process dead",
                             attempt=state.get("attempt"))
            elif stale:
                if kill_child is None:
                    alerts.append({"type": "unit_timeout_alive",
                                   "unit": state["unit_id"],
                                   "pid": pid,
                                   "action": "ALERT ONLY — no "
                                             "killer supplied; a "
                                             "live process is never "
                                             "marked terminal"})
                    continue
                reaped = bool(kill_child(pid))
                if not reaped:
                    alerts.append({"type": "unit_timeout_kill_failed",
                                   "unit": state["unit_id"],
                                   "pid": pid})
                    continue
                alerts.append({"type": "unit_timeout",
                               "unit": state["unit_id"], "pid": pid,
                               "child_reaped": True})
                self.release(state["unit_id"], "TIMED_OUT",
                             note="watchdog: timed out; child "
                                  "terminated and reaped first",
                             attempt=state.get("attempt"))
        usage = os.statvfs(self.root)
        free_fraction = usage.f_bavail / max(1, usage.f_blocks)
        if free_fraction < 0.05:
            alerts.append({"type": "disk_pressure",
                           "free_fraction": round(free_fraction, 3)})
        if expected_digests:
            try:
                ledger_digests = self.ledger().get("digests", {})
            except RuntimePreflightError as exc:
                alerts.append({"type": "identity_drift",
                               "detail": f"ledger unreadable: {exc}"})
                ledger_digests = None
            if ledger_digests is not None:
                drift = {k: (ledger_digests.get(k, "")[:12], v[:12])
                         for k, v in expected_digests.items()
                         if ledger_digests.get(k) != v}
                if drift:
                    alerts.append({"type": "identity_drift",
                                   "drift": drift})
        reader = temperature_reader or read_temperatures
        temps = reader()
        gpu_t = temps.get("gpu_max_c")
        cpu_t = temps.get("cpu_max_c")
        if gpu_t is not None and gpu_t >= gpu_max_c:
            alerts.append({"type": "thermal", "device": "gpu",
                           "temperature_c": gpu_t,
                           "limit_c": gpu_max_c})
        if cpu_t is not None and cpu_t >= cpu_max_c:
            alerts.append({"type": "thermal", "device": "cpu",
                           "temperature_c": cpu_t,
                           "limit_c": cpu_max_c})
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
                    note="SIGTERM during execution",
                    attempt=state["attempt"])
        raise SystemExit(143)

    previous = signal.signal(signal.SIGTERM, on_term)
    started = time.time()
    try:
        identity = state["identity"]
        result = executor(identity, log_path)
        if time.time() - started > timeout_s:
            run.release(uid, "TIMED_OUT",
                        note=f"exceeded {timeout_s}s",
                        attempt=state["attempt"])
            return {"state": "TIMED_OUT"}
        run.release(uid, "COMPLETED", result=result,
                    attempt=state["attempt"])
        return {"state": "COMPLETED", "result": result}
    except SystemExit:
        raise
    except BaseException as exc:
        if not interrupted["flag"]:
            run.release(uid, "FAILED",
                        note=f"{type(exc).__name__}: {exc}",
                        attempt=state["attempt"])
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def read_temperatures() -> dict:
    """Best-effort thermal read: max GPU temp via nvidia-smi, max CPU
    thermal zone via sysfs. Absent sensors yield None — the watchdog
    then reports nothing rather than guessing."""
    import subprocess
    gpu = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader"], capture_output=True,
            text=True, timeout=5)
        values = [float(v) for v in out.stdout.split() if v.strip()]
        gpu = max(values) if values else None
    except Exception:
        gpu = None
    cpu = None
    try:
        zones = list(Path("/sys/class/thermal").glob(
            "thermal_zone*/temp"))
        values = []
        for zone in zones:
            try:
                values.append(int(zone.read_text().strip()) / 1000.0)
            except (OSError, ValueError):
                continue
        cpu = max(values) if values else None
    except Exception:
        cpu = None
    return {"gpu_max_c": gpu, "cpu_max_c": cpu}


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
        # R1 (2026-09-03): recompute the digest over the CONTENT and
        # demand unit-result correspondence — a tampered or foreign
        # result refuses instead of steering the decision
        recomputed = sha_obj({k: v for k, v in result.items()
                              if k != "result_digest"})
        if result.get("result_digest") != recomputed:
            raise RuntimePreflightError(
                f"aggregation refuses: result digest mismatch for "
                f"{u} — content was altered after completion")
        bound = result.get("unit_id")
        if bound is not None and bound != u:
            raise RuntimePreflightError(
                f"aggregation refuses: result of {u} is bound to "
                f"{bound} — unit/result correspondence broken")
        results[u] = result
    return results
