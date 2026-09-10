"""B4 C51-C53: the external per-cell watchdog.

One cell attempt runs in ONE supervised child process under a
capability bound to generation/cell/attempt, materialization,
authorization + recovery record, parent identity, the per-cell
wall deadline and the remaining global budget. The parent's
deadline is MONOTONIC and independent of callbacks, the GIL,
model code, CUDA progress and child telemetry.

Escalation (C52), executed and persisted in order: durable stop
request -> bounded graceful interval -> SIGTERM (process group)
-> second bounded interval -> SIGKILL -> mandatory reap with an
empty-process-group proof -> CUDA process inventory -> no next
cell. The supervisor writes incident records; it NEVER fabricates
a scientific terminal. A child terminal is accepted only if it
was durably complete BEFORE the stop request and passes the
existing productive adjudication; otherwise the attempt is
QUARANTINED_RUNTIME_STALL or QUARANTINED_EXTERNAL_STOP.

Liveness (C53): the parent-observed progress contract is the
durable write stream of the cell (status.json is written once
per epoch — measured median 171 s / p90 184 s per epoch on the
two completed v7 cells — plus any new file under cell_runtime/
and checkpoints/). LIVENESS_MAX_SILENCE_S = 1800 s (~10x the
per-epoch p90; the audited stall was silent ~29 h). A busy CPU
or GPU is NOT progress: unchanged durable progress past the
silence bound stops the child even while a heartbeat thread
beats. GPU accounting closes at the externally observed reap
time in a separate append-only supervisor record that never
rewrites a claim or lease; failed and quarantined time counts
against the global ceiling, re-derived from durable intervals
only.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

POLL_S = 5.0
GRACE_STOP_S = 60.0
GRACE_TERM_S = 30.0
LIVENESS_MAX_SILENCE_S = 1800.0
CELL_WALL_S = 43200.0
GLOBAL_CEILING_S = 345600.0          # 96 h resource contract
PROGRESS_SUBDIRS = ("cell_runtime", "checkpoints",
                    "return_traces")


class SupervisorRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _self_sha(doc, key):
    return hashlib.sha256(json.dumps(
        {k: doc[k] for k in sorted(doc) if k != key},
        sort_keys=True).encode()).hexdigest()


def _excl_append_json(path: Path, doc: dict):
    doc = dict(doc)
    doc["record_sha256"] = _self_sha(doc, "record_sha256")
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, json.dumps(doc, indent=1,
                                sort_keys=True).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return doc["record_sha256"]


def build_capability(generation, cell_id, attempt_id, mat_root,
                     auth_sha, recovery_acta_sha, cell_dir,
                     wall_s, global_remaining_s):
    cap = {"schema": "agent_multi.b4_cell_capability.v1",
           "campaign_generation": generation,
           "cell": cell_id, "attempt_id": attempt_id,
           "materialization_root": str(mat_root),
           "authorization_record_sha256": auth_sha,
           "recovery_acta_sha256": recovery_acta_sha,
           "parent_pid": os.getpid(),
           "parent_session_id": os.getsid(0),
           "cell_wall_seconds": float(wall_s),
           "global_remaining_seconds": float(
               global_remaining_s)}
    cap["capability_sha256"] = _self_sha(cap,
                                         "capability_sha256")
    path = Path(cell_dir) / f"CAPABILITY_{attempt_id}.json"
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, json.dumps(cap, indent=1,
                                sort_keys=True).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return path, cap


def _progress_stamp(cell_dir: Path):
    """The durable-progress signal: newest mtime + file count
    across the cell's runtime write stream."""
    newest = 0.0
    count = 0
    for sub in PROGRESS_SUBDIRS:
        d = Path(cell_dir) / sub
        if d.is_dir():
            for p in d.rglob("*"):
                if p.is_file():
                    count += 1
                    m = p.stat().st_mtime
                    if m > newest:
                        newest = m
    return (newest, count)


def _read_heartbeat(cell_dir: Path):
    p = Path(cell_dir) / "SUPERVISOR_HEARTBEAT.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {"malformed": True}


def _group_empty(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def _cuda_inventory(device: str):
    if not str(device).startswith("cuda"):
        return {"queried": False,
                "note": "cpu device — no CUDA inventory"}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        pids = [ln.split(",")[0].strip()
                for ln in out.stdout.splitlines() if ln.strip()]
        return {"queried": True, "compute_pids": pids}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"queried": True,
                "error": type(exc).__name__}


def run_cell_supervised(cell_id, out_root, cell_dir, attempt_id,
                        child_cmd, device="cpu",
                        generation="", mat_root="",
                        auth_sha="0" * 64,
                        recovery_acta_sha="0" * 64,
                        wall_s=CELL_WALL_S,
                        global_remaining_s=GLOBAL_CEILING_S,
                        liveness_silence_s=(
                            LIVENESS_MAX_SILENCE_S),
                        grace_stop_s=GRACE_STOP_S,
                        grace_term_s=GRACE_TERM_S,
                        poll_s=POLL_S) -> dict:
    """Supervise ONE cell attempt in ONE child process. Returns
    the supervisor outcome; never fabricates a terminal."""
    cell_dir = Path(cell_dir)
    out_root = Path(out_root)
    cap_path, cap = build_capability(
        generation, cell_id, attempt_id, mat_root, auth_sha,
        recovery_acta_sha, cell_dir, wall_s, global_remaining_s)
    claim_wallclock = time.time()
    t0 = time.monotonic()
    deadline = t0 + min(float(wall_s),
                        float(global_remaining_s))
    child = subprocess.Popen(
        list(child_cmd) + ["--capability", str(cap_path)],
        start_new_session=True)
    pgid = os.getpgid(child.pid)
    last_progress = _progress_stamp(cell_dir)
    last_advance_mono = time.monotonic()
    last_hb_seq = -1
    incident_n = 0
    stop_reason = None
    stop_request_wallclock = None

    def incident(reason, extra=None):
        nonlocal incident_n
        incident_n += 1
        _excl_append_json(
            cell_dir / (f"SUPERVISOR_INCIDENT_{attempt_id}_"
                        f"{incident_n:03d}.json"),
            {"schema": "agent_multi.b4_supervisor_incident.v1",
             "attempt_id": attempt_id, "cell": cell_id,
             "seq": incident_n, "reason": reason,
             "monotonic_since_start": round(
                 time.monotonic() - t0, 3),
             "wallclock": time.time(),
             **(extra or {})})

    while True:
        rc = child.poll()
        if rc is not None:
            break
        now = time.monotonic()
        if now > deadline:
            stop_reason = "HARD_WALL"
            break
        if (out_root / "CAMPAIGN_STOP").exists() or \
                (cell_dir / "STOP").exists():
            stop_reason = "EXTERNAL_STOP"
            break
        hb = _read_heartbeat(cell_dir)
        if hb is not None and not hb.get("malformed"):
            if hb.get("child_pid") != child.pid or \
                    type(hb.get("seq")) is not int or \
                    hb["seq"] < last_hb_seq:
                stop_reason = "HEARTBEAT_CONTRACT_VIOLATION"
                break
            last_hb_seq = hb["seq"]
        stamp = _progress_stamp(cell_dir)
        if stamp != last_progress:
            last_progress = stamp
            last_advance_mono = now
        elif now - last_advance_mono > float(liveness_silence_s):
            # a beating heartbeat or busy CPU/GPU is NOT progress
            stop_reason = "TELEMETRY_STALL"
            break
        time.sleep(poll_s)

    escalation = []
    if stop_reason is not None:
        # C52 ordered protocol, durably persisted step by step
        stop_request_wallclock = time.time()
        sp = cell_dir / "STOP"
        if not sp.exists():
            fd = os.open(str(sp), os.O_CREAT | os.O_EXCL
                         | os.O_WRONLY, 0o600)
            try:
                os.write(fd, stop_reason.encode())
                os.fsync(fd)
            finally:
                os.close(fd)
        escalation.append("stop_request_durable")
        incident(stop_reason,
                 {"escalation_step": "stop_request",
                  "stop_request_wallclock":
                      stop_request_wallclock})
        t_stop = time.monotonic()
        while time.monotonic() - t_stop < grace_stop_s:
            if child.poll() is not None:
                break
            time.sleep(0.2)
        if child.poll() is None:
            os.killpg(pgid, signal.SIGTERM)
            escalation.append("sigterm_group")
            incident(stop_reason,
                     {"escalation_step": "sigterm"})
            t_term = time.monotonic()
            while time.monotonic() - t_term < grace_term_s:
                if child.poll() is not None:
                    break
                time.sleep(0.2)
        if child.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
            escalation.append("sigkill_group")
            incident(stop_reason,
                     {"escalation_step": "sigkill"})
    # C52.6: mandatory reap + empty-group proof
    child.wait(timeout=60)
    reap_wallclock = time.time()
    t_reap = time.monotonic()
    while not _group_empty(pgid):
        if time.monotonic() - t_reap > 30:
            raise SupervisorRefusal(
                f"process group {pgid} did not empty after "
                "SIGKILL — manual intervention required")
        time.sleep(0.2)
    escalation.append("reaped_group_empty")
    cuda_inv = _cuda_inventory(device)
    # ---- classification: NEVER fabricate a terminal ----
    # The supervisor only states FACTS about the child terminal;
    # sealing and final adjudication belong to the orchestrator.
    terminal_p = cell_dir / "B4_CELL_TERMINAL.json"
    outcome = None
    if terminal_p.exists():
        term_mtime = terminal_p.stat().st_mtime
        try:
            term_doc = json.loads(terminal_p.read_text())
        except (json.JSONDecodeError, OSError):
            term_doc = {}
        after_stop = (stop_request_wallclock is not None
                      and term_mtime > stop_request_wallclock)
        if not after_stop:
            outcome = "TERMINAL_PRESENT_ON_TIME"
        elif term_doc.get("terminal") not in (None, "COMPLETED"):
            # a typed stop acknowledgement written during the
            # graceful interval is the legitimate cooperative
            # path
            outcome = "TERMINAL_PRESENT_GRACEFUL_ACK"
        else:
            # C55.5: a COMPLETED terminal written after the stop
            # request can NEVER become a completion
            outcome = None
    if outcome is None:
        outcome = ("QUARANTINED_EXTERNAL_STOP"
                   if stop_reason == "EXTERNAL_STOP"
                   else "QUARANTINED_RUNTIME_STALL")
        _excl_append_json(
            cell_dir / f"SUPERVISOR_ATTEMPT_CLASS_"
                       f"{attempt_id}.json",
            {"schema":
                 "agent_multi.b4_supervisor_attempt_class.v1",
             "attempt_id": attempt_id, "cell": cell_id,
             "classification": outcome,
             "stop_reason": stop_reason,
             "child_returncode": child.returncode,
             "terminal_present": terminal_p.exists(),
             "terminal_late_completed": bool(
                 terminal_p.exists()
                 and stop_request_wallclock is not None
                 and terminal_p.stat().st_mtime
                 > stop_request_wallclock)})
    # C53.3: close accounting at the externally observed reap
    close_sha = _excl_append_json(
        cell_dir / f"SUPERVISOR_GPU_CLOSE_{attempt_id}.json",
        {"schema": "agent_multi.b4_supervisor_gpu_close.v1",
         "attempt_id": attempt_id, "cell": cell_id,
         "open_wallclock": claim_wallclock,
         "close_wallclock": reap_wallclock,
         "charged_seconds": round(
             reap_wallclock - claim_wallclock, 3),
         "close_reason": stop_reason or "CHILD_EXIT",
         "counts_against_global_ceiling": True})
    return {"outcome": outcome,
            "stop_reason": stop_reason,
            "escalation": escalation,
            "child_returncode": child.returncode,
            "group_empty": True,
            "cuda_inventory": cuda_inv,
            "gpu_close_sha256": close_sha,
            "no_next_cell_scheduled": stop_reason is not None,
            "capability_sha256": cap["capability_sha256"]}


# ---------- C53.4/5: durable-interval accounting ----------

def recompute_gpu_charges(roots, overrides=None,
                          ceiling_s=GLOBAL_CEILING_S) -> dict:
    """Re-derive every charge from durable intervals only:
    [claim mtime -> terminal/seal mtime | supervisor close |
    override]. The rounded dry-run summaries are NEVER inputs.
    `overrides` maps 'cell_id' -> close epoch-seconds for
    historical attempts whose close is fixed by external record
    bytes (the v7 operator stop). Quarantined/failed time counts
    against the ceiling."""
    per_cell = {}
    total = 0.0
    for root in roots:
        root = Path(root)
        for cdir in sorted(p for p in root.iterdir()
                           if p.is_dir()
                           and p.name.startswith("o20")):
            claims = sorted(cdir.glob("CLAIM_*.json"))
            if not claims:
                continue
            open_t = min(c.stat().st_mtime for c in claims)
            close_candidates = []
            term = cdir / "B4_CELL_TERMINAL.json"
            if term.exists():
                close_candidates.append(term.stat().st_mtime)
            for s in cdir.glob("SEAL_COMPLETE_*.json"):
                close_candidates.append(s.stat().st_mtime)
            for s in cdir.glob("SUPERVISOR_GPU_CLOSE_*.json"):
                doc = json.loads(s.read_text())
                close_candidates.append(
                    float(doc["close_wallclock"]))
            if overrides and cdir.name in overrides:
                close_candidates.append(
                    float(overrides[cdir.name]))
            if not close_candidates:
                raise SupervisorRefusal(
                    f"{cdir.name}: an open interval has no "
                    "durable close and no external override — "
                    "charges cannot be derived")
            close_t = max(close_candidates)
            sec = max(0.0, close_t - open_t)
            per_cell[f"{root.name}/{cdir.name}"] = round(sec, 1)
            total += sec
    return {"per_cell": per_cell,
            "total_charged_seconds": round(total, 1),
            "ceiling_seconds": float(ceiling_s),
            "remaining_seconds": round(ceiling_s - total, 1),
            "source": "durable intervals only — no rounded "
                      "summary is an input"}
