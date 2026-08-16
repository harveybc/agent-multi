#!/usr/bin/env python3
"""P1LR idle guard: bounded recovery for a stalled factorial seed.

Order 2026-08-11 §7.8 (finding 229): an ASSIGNED GPU idle for more than
15 minutes while a P1LR cell is still pending on its seed (cell records
incomplete, no runner process alive, no heartbeat progress) emits ONE
deduplicated incident through the fleet incident ledger and triggers
BOUNDED service recovery — at most N ``systemctl --user restart`` calls
of the seed's ``p1lr-screen@<seed>.service`` unit with exponential
backoff, and only when the unit actually exists. When the unit does not
exist, the incident itself carries the exact relaunch command as
remediation. Never an unbounded restart loop; a typed ``REFUSED_*``
runner heartbeat withholds restarts entirely, because a configuration
refusal is not transient.

Deduplication follows the gpu_readiness_probe pattern: one observation
per state CHANGE (idle condition raised), one recovery notice on direct
progress-resumption evidence, and a failed emission keeps the old state
so the next poll retries (the ledger's fingerprint dedup makes a retry
safe). Guard state lives in the guard's OWN state file — never inside
the run's output root, which stays strictly read-only.

MODE (finding 233): the guard's mode is EXPLICIT and VALIDATED. Output
root, unit name, expected heartbeat/record mode and total cells all
derive from it through ``multifront_status.p1lr_mode_binding``, so a
decision run under ``decision_run.output_root`` is never guarded by
reading the screen root — the defect that reported an idle 0/16 screen
while four decision processes held four busy GPUs.

TERMINAL COMPLETION (order 2026-08-15 §3): a completed seed used to
render ``idle: false`` — the guard answered "not stalled" to a question
nobody asked, while ``process_alive`` was false, no cell was pending and
no next job was running. Completion is not health. Idleness is now
CLASSIFIED by :func:`classify_seed_idleness`: a terminal seed is never
``stalled`` (it needs a dispatch, not a restart), but it is ``idle``
unless the durable transition queue proves the successor is dispatched
or in flight. The proof comes only from
``tools/experiment_transition_queue.py`` records — never from a
heartbeat, a shell process, a chat message or operator memory — and the
same module owns the ONE deduplicated undispatched-successor incident,
through the same fleet ledger this guard already uses.

IDENTITY-AWARE MATCHING (finding AUD-GEN-20260815-250): a process
counts as a seed's runner ONLY when its own cmdline/cwd facts prove
contract identity + mode + seed + output root — NEVER seed alone. The
retired seed-only pgrep pattern let a v2 decision PID with ``--seed
101`` make terminal v1 seed 101 render ``busy_process_alive``; that
pattern survives only as :func:`legacy_seed_only_pattern` for the
regression proof. The guard also discovers the ACTIVE durable
transition from the queue records and REFUSES (typed, exit 2) to
supervise an identity that conflicts with the active dispatched chain,
and it withholds restarts while a foreign or unprovable same-seed
process lives — one writer per seed, always.

Socket-free by construction for tests: process facts, GPU telemetry,
unit existence, restart calls and ledger emissions are injected; the
default wiring uses /proc cmdline+cwd via pgrep candidates /
nvidia-smi (tools.m0_l1_mechanism_ladder.gpu_telemetry) / systemctl
and tools/incident_emit.py. Intended to run from a per-host systemd
user timer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import experiment_transition_queue as etq  # noqa: E402
from tools.multifront_status import (  # noqa: E402  (path set above)
    P1LR_MODES,
    P1LR_UNIT_TEMPLATES,
    P1lrModeRefusal,
    classify_p1lr_failure,
    p1lr_mode_binding,
    p1lr_verify_identity_binding,
)

STATE_SCHEMA = "agent_multi.p1lr_idle_guard_state.v1"
REPORT_SCHEMA = "agent_multi.p1lr_idle_guard_report.v1"
SOURCE = "p1lr_idle_guard"
FRONT = "front1"
SEVERITY = "P2"
# Kept for compatibility with older callers; the guard itself always
# takes the unit template from the validated mode binding.
UNIT_TEMPLATE = P1LR_UNIT_TEMPLATES["screen"]
DEFAULT_MODE = "screen"

DEFAULT_CONTRACT = (REPO_ROOT / "examples/config/phase_3_eth_sac_dynamics/"
                               "p1_difficulty_lr_factorial_v1.json")
DEFAULT_STATE_DIR = Path.home() / ".local/state/agent-multi/p1lr-idle-guard"
DEFAULT_IDLE_AFTER_SECONDS = 900.0          # §7.8: 15 minutes
DEFAULT_GPU_IDLE_UTILIZATION_PCT = 10       # below this the GPU reads idle
DEFAULT_MAX_RESTARTS = 3                    # bounded recovery cap
DEFAULT_RESTART_BACKOFF_SECONDS = 900.0     # doubles per attempt

# Order 2026-08-15 §3: the durable transition queue is the ONLY authority
# for "is a successor dispatched?". It lives outside every run root.
DEFAULT_TRANSITION_QUEUE_DIR = etq.DEFAULT_QUEUE_DIR

#: Typed idleness classes. ``idle`` is a CONSEQUENCE of the class, never
#: a synonym for "stalled": a terminal experiment with no dispatched
#: successor is idle without being stalled.
IDLE_CLASSES = (
    "busy_process_alive",
    "pending_recent_progress",
    "pending_gpu_busy",
    "pending_gpu_utilization_unknown",
    "stalled_pending_cells",
    "completed_untransitioned",
    "completed_transition_in_progress",
    "completed_transitioned",
)

# The decision unit NEVER defaults to screen: its relaunch command
# carries --mode decision through the shipped unit, whose ExecStartPre
# verifies the pinned screen gate before any training starts.
MODE_UNIT_FILES = {"screen": "examples/systemd/p1lr-screen@.service",
                   "decision": "examples/systemd/p1lr-decision@.service"}


def idle_event_code(seed: int) -> str:
    return f"p1lr_gpu_idle_pending.seed{seed}"


def cap_event_code(seed: int) -> str:
    return f"p1lr_idle_restart_cap.seed{seed}"


def relaunch_command(seed: int, mode: str = DEFAULT_MODE) -> str:
    """The exact owner/operator relaunch command when the unit is
    missing, for THIS mode's unit (examples/systemd/p1lr-<mode>@.service).
    The decision unit itself pins --mode decision and the verified
    screen gate, so the relaunch can never silently fall back to a
    screen run."""
    if mode not in P1LR_MODES:
        raise P1lrModeRefusal(
            "P1LR_MODE_INVALID",
            f"unknown P1LR execution mode {mode!r}: one of "
            f"{list(P1LR_MODES)}", supplied_mode=mode)
    unit = P1LR_UNIT_TEMPLATES[mode].format(seed=seed)
    return (f"cp {MODE_UNIT_FILES[mode]} "
            "~/.config/systemd/user/ && systemctl --user daemon-reload && "
            f"systemctl --user enable --now {unit}")


# ---------------------------------------------------------------------------
# guard state (own state file; the run root is never written)
# ---------------------------------------------------------------------------

def default_state() -> dict:
    return {"schema": STATE_SCHEMA, "seeds": {}, "updated_utc": None}


def default_seed_state() -> dict:
    return {"idle_active": False, "cap_emitted": False,
            "restarts": [], "first_pending_seen_utc": None}


def load_state(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_state()
    if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
        return default_state()
    state = default_state()
    state.update(value)
    if not isinstance(state.get("seeds"), dict):
        state["seeds"] = {}
    return state


def save_state(path: Path, state: dict) -> None:
    state = dict(state)
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# default fact providers (tests inject fakes; nothing here runs in tests)
# ---------------------------------------------------------------------------

def legacy_seed_only_pattern(seed: int) -> str:
    """The RETIRED seed-only pgrep pattern, kept importable ONLY so the
    regression proof of finding AUD-GEN-20260815-250 can reproduce the
    defect it caused: this pattern matches ANY runner with the same
    ``--seed``, whatever its contract, mode, chain or output root — so
    a v2 decision PID with ``--seed 101`` made terminal v1 seed 101
    render ``busy_process_alive``. It is never used for matching."""
    return f"p1_difficulty_lr_factorial\\.py --seed {seed}(\\s|$)"


def default_process_facts() -> list[dict]:
    """Observed runner-candidate process facts on THIS host — pid,
    argv (``/proc/<pid>/cmdline``) and working directory
    (``/proc/<pid>/cwd``). Candidates come from a broad pgrep; the
    IDENTITY decision is made by :func:`match_seed_processes`, never
    here — finding 250: process matching requires contract identity,
    mode, seed and output root, NEVER seed alone."""
    try:
        out = subprocess.run(["pgrep", "-f", "p1_difficulty_lr_factorial"],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    facts: list[dict] = []
    for token in (out.stdout or "").split():
        try:
            pid = int(token)
        except ValueError:
            continue
        entry: dict[str, Any] = {"pid": pid, "cmdline": None, "cwd": None}
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            entry["cmdline"] = [a for a in raw.decode("utf-8", "replace")
                                .split("\0") if a]
        except OSError:
            pass
        try:
            entry["cwd"] = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            pass
        facts.append(entry)
    return facts


#: The runner's own --contract default (tools/p1_difficulty_lr_factorial
#: .py), relative to the runner's repo root: a runner launched WITHOUT
#: --contract is a v1-contract runner, and the matcher says so instead
#: of guessing.
RUNNER_DEFAULT_CONTRACT_RELPATH = (
    "examples/config/phase_3_eth_sac_dynamics/"
    "p1_difficulty_lr_factorial_v1.json")


def parse_runner_cmdline(cmdline: Optional[list]) -> Optional[dict]:
    """Typed runner facts from one argv, or None when the argv is not a
    ``p1_difficulty_lr_factorial.py`` invocation (a bash wrapper, an ssh
    line or an unrelated process never parses as a runner: the script
    name must be its OWN argv element)."""
    if not isinstance(cmdline, list):
        return None
    script_idx = next(
        (i for i, a in enumerate(cmdline)
         if isinstance(a, str)
         and a.endswith("p1_difficulty_lr_factorial.py")), None)
    if script_idx is None:
        return None
    args = [str(a) for a in cmdline[script_idx + 1:]]

    def flag(name: str) -> Optional[str]:
        for i, arg in enumerate(args):
            if arg == name and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
        return None

    seed_raw = flag("--seed")
    try:
        seed = int(seed_raw) if seed_raw is not None else None
    except ValueError:
        seed = None
    return {"script": str(cmdline[script_idx]),
            "seed": seed,
            "mode": flag("--mode") or "screen",   # the runner's default
            "contract": flag("--contract"),
            "screen_gate": flag("--screen-gate")}


def _resolve_process_contract(parsed: dict, cwd: Optional[str],
                              ) -> Optional[Path]:
    """The ABSOLUTE contract path a runner process is executing, resolved
    against its /proc cwd (relative launches) or its script location
    (the runner's own --contract default). None when unresolvable."""
    contract = parsed.get("contract")
    if contract is None:
        script = Path(str(parsed.get("script") or ""))
        if not script.is_absolute():
            if not cwd:
                return None
            script = Path(cwd) / script
        contract = str(script.parent.parent
                       / RUNNER_DEFAULT_CONTRACT_RELPATH)
    path = Path(str(contract)).expanduser()
    if not path.is_absolute():
        if not cwd:
            return None
        path = Path(cwd) / path
    return path


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def match_seed_processes(process_facts: list[dict], *, seed: int,
                         binding: dict, expected_contract_sha256: str,
                         sha256_file_fn: Optional[
                             Callable[[Path], Optional[str]]] = None,
                         ) -> dict[str, Any]:
    """Identity-aware seed-process matching (finding 250).

    A process COUNTS AS THIS SEED'S RUNNER only when ALL of these are
    proven from its own facts: it is a runner invocation, its ``--seed``
    equals ``seed``, its ``--mode`` equals the guarded mode, and its
    contract file (resolved against the process's own cwd) hashes to
    ``expected_contract_sha256`` — which content-addresses the chain
    identity AND the output root, since both derive from the contract.
    NEVER seed alone.

    Everything else is typed, not discarded:

    ``foreign_same_seed``
        a runner holding this seed for a DIFFERENT mode or contract —
        Musashi's exact observation (a v2 decision PID on terminal v1
        seed 101). It never renders this identity alive, and the guard
        withholds restarts while it lives (a restart beside it would
        create a second writer on the seed's GPU).
    ``unprovable_same_seed``
        a runner holding this seed whose contract identity cannot be
        PROVEN (cwd or contract unreadable). Unproven is not mine — it
        never renders alive — and unproven is not free either: restarts
        are withheld exactly as for a foreign process.
    """
    hash_fn = sha256_file_fn or _sha256_file
    expected_root = str(Path(str(binding["output_root"]))
                        .expanduser()).rstrip("/")
    matched: list[dict] = []
    foreign: list[dict] = []
    unprovable: list[dict] = []
    for proc in process_facts or []:
        parsed = parse_runner_cmdline(proc.get("cmdline"))
        if parsed is None or parsed.get("seed") != seed:
            continue
        entry: dict[str, Any] = {
            "pid": proc.get("pid"),
            "cwd": proc.get("cwd"),
            "mode": parsed["mode"],
            "contract_argument": parsed.get("contract"),
        }
        if parsed["mode"] != binding["mode"]:
            entry["mismatch"] = ("mode_mismatch: process runs "
                                 f"--mode {parsed['mode']}, this guard "
                                 f"binds mode {binding['mode']}")
            foreign.append(entry)
            continue
        contract_path = _resolve_process_contract(parsed, proc.get("cwd"))
        if contract_path is None:
            entry["mismatch"] = ("contract_unresolvable: no cwd fact to "
                                 "resolve the process contract path")
            unprovable.append(entry)
            continue
        entry["contract_path"] = str(contract_path)
        observed_sha = hash_fn(contract_path)
        entry["contract_sha256"] = observed_sha
        if observed_sha is None:
            entry["mismatch"] = ("contract_unprovable: the process "
                                 "contract file is unreadable from this "
                                 "guard, so its identity cannot be proven")
            unprovable.append(entry)
            continue
        if observed_sha != expected_contract_sha256:
            entry["mismatch"] = (
                "contract_identity_mismatch: process contract sha256 "
                f"{observed_sha[:16]}… differs from the guarded contract "
                f"{expected_contract_sha256[:16]}… — a different chain "
                "identity and a different output root")
            foreign.append(entry)
            continue
        entry["output_root"] = expected_root
        matched.append(entry)
    return {
        "alive": bool(matched),
        "basis": ("process cmdline/cwd facts: runner script + --seed + "
                  "--mode + contract sha256 (content-addresses chain "
                  "identity and output root); never seed alone "
                  "(finding 250)"),
        "matched": matched,
        "foreign_same_seed": foreign,
        "unprovable_same_seed": unprovable,
    }


def default_gpu_telemetry(gpu_uuid: Optional[str]) -> dict:
    """Reuses the proven nvidia-smi helper (absent facts stay None)."""
    from tools.m0_l1_mechanism_ladder import gpu_telemetry
    return gpu_telemetry(gpu_uuid)


def default_unit_exists(unit: str) -> bool:
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", unit, "-p", "LoadState",
             "--value"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    return out.strip() == "loaded"


def default_restart(unit: str) -> dict:
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "restart", unit],
            capture_output=True, text=True, timeout=60)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "stderr": proc.stderr.strip()[:200]}
    except Exception as exc:
        return {"ok": False, "returncode": None,
                "stderr": f"{type(exc).__name__}: {exc}"[:200]}


class LedgerEmitter:
    """Adapter onto tools/incident_emit.py (the fleet shim). Imported
    lazily so tests never touch the real ledger."""

    def __init__(self, machine: str, front: str = FRONT):
        self.machine = machine
        self.front = front

    def observe(self, event_code: str, severity: str, summary: str,
                payload: dict, affected_object: str = "-") -> bool:
        import incident_emit
        return incident_emit.observe_incident(
            source=SOURCE, event_code=event_code, severity=severity,
            summary=summary, front=self.front, machine=self.machine,
            affected_object=affected_object, payload=payload)

    def recover(self, event_code: str, evidence: dict,
                affected_object: str = "-") -> bool:
        import incident_emit
        return incident_emit.recover_incident(
            source=SOURCE, event_code=event_code, evidence=evidence,
            front=self.front, machine=self.machine,
            affected_object=affected_object)


# ---------------------------------------------------------------------------
# fact collection (read-only towards the run root)
# ---------------------------------------------------------------------------

def _iso(value: Any) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _seed_cells(contract: dict, seed: int,
                binding: Optional[dict] = None) -> list[str]:
    if binding is not None:
        cells = binding["cells_per_seed"].get(str(seed))
        if cells:
            return list(cells)
    cells_map = contract.get("cells") or {}
    order = (contract.get("cell_order") or {}).get(str(seed)) or []
    ordered = [c for c in order if c in cells_map]
    return ordered or list(cells_map)


def seed_facts(contract: dict, identity: str, seed: int,
               now: datetime, binding: Optional[dict] = None) -> dict:
    """Observed facts for one seed: record count, newest progress mtime
    (cell heartbeats, seed refusal heartbeat, cell records) and the
    newest heartbeat's terminal_state (to detect typed refusals).

    The output root comes from the VALIDATED mode binding (finding 233),
    never from ``contract['output_root']`` unconditionally."""
    if binding is None:
        binding = p1lr_mode_binding(contract, DEFAULT_MODE)
    root = Path(str(binding["output_root"])).expanduser()
    seed_dir = root / identity / f"seed{seed}"
    cells = _seed_cells(contract, seed, binding)
    records = [c for c in cells
               if (seed_dir / c / "cell_record.json").is_file()]
    progress_paths = ([seed_dir / c / "heartbeat.json" for c in cells]
                      + [seed_dir / "runner_heartbeat.json"]
                      + [seed_dir / c / "cell_record.json" for c in cells])
    newest_mtime: Optional[float] = None
    newest_heartbeat: tuple[Optional[float], Optional[Path]] = (None, None)
    for path in progress_paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest_mtime is None or mtime > newest_mtime:
            newest_mtime = mtime
        if path.name.endswith("heartbeat.json") and (
                newest_heartbeat[0] is None or mtime > newest_heartbeat[0]):
            newest_heartbeat = (mtime, path)
    terminal_state = None
    heartbeat_error = None
    heartbeat_failure_class = None
    heartbeat_cell = None
    if newest_heartbeat[1] is not None:
        try:
            loaded = json.loads(newest_heartbeat[1].read_text())
            if isinstance(loaded, dict):
                terminal_state = loaded.get("terminal_state")
                # WP0 rule 4: the error text and the runner-declared
                # failure class ride along so the poll classifies a
                # SEED_FAILED-due-to-source-drift distinctly.
                heartbeat_error = loaded.get("error")
                heartbeat_failure_class = loaded.get("failure_class")
                heartbeat_cell = loaded.get("cell")
        except (OSError, ValueError):
            pass
    progress_age = (round(now.timestamp() - newest_mtime, 1)
                    if newest_mtime is not None else None)
    return {
        "cells_total": len(cells),
        "records_landed": len(records),
        "pending_cells": [c for c in cells if c not in records],
        "progress_age_seconds": progress_age,
        "last_heartbeat_terminal_state": terminal_state,
        "last_heartbeat_error": heartbeat_error,
        "last_heartbeat_failure_class": heartbeat_failure_class,
        "last_heartbeat_cell": heartbeat_cell,
        "seed_dir": str(seed_dir),
    }


# ---------------------------------------------------------------------------
# idleness classification (order 2026-08-15 §3 bullets 1 and 2)
# ---------------------------------------------------------------------------

def classify_seed_idleness(*, pending: bool, process_alive: bool,
                           gpu_idle_now: bool,
                           gpu_utilization_unknown: bool,
                           observed_idle: float,
                           idle_after_seconds: float,
                           cells_total: int, records_landed: int,
                           transition_state: Optional[str],
                           ) -> dict[str, Any]:
    """Type WHY a seed is or is not idle. Completion is never health.

    Two facts that used to be conflated are now separate:

    ``stalled``
        the §7.8 condition — cells still PENDING, no runner process, an
        idle assigned GPU and no progress for longer than the threshold.
        This is the only condition that justifies a bounded restart.

    ``idle``
        the assigned GPU has no useful work. A terminal seed whose
        successor is not proven dispatched is idle: ``process_alive:
        false`` plus a complete 16/16 plus no next executable job is
        FLEET IDLE TIME, and the owner's cardinal rule says it must
        surface. It renders ``completed_untransitioned``, never
        ``idle: false``.

    ``transition_state`` is the durable queue's verdict
    (``tools/experiment_transition_queue.transition_status``). ``None``
    means no durable record exists, which is itself
    ``completed_untransitioned`` — unproven is not dispatched.
    """
    terminal_complete = (not pending and cells_total > 0
                         and records_landed >= cells_total)
    if process_alive:
        return {
            "idle_class": "busy_process_alive", "idle": False,
            "stalled": False, "terminal_complete": terminal_complete,
            "idle_class_reason": ("a runner process for this seed is "
                                  "alive on this host"),
        }
    if terminal_complete:
        if transition_state == "transitioned":
            return {
                "idle_class": "completed_transitioned", "idle": False,
                "stalled": False, "terminal_complete": True,
                "idle_class_reason": (
                    f"all {records_landed}/{cells_total} cell records "
                    "landed and the durable transition queue proves the "
                    "approved successor is DISPATCHED"),
            }
        if transition_state == "transition_dispatch_in_progress":
            return {
                "idle_class": "completed_transition_in_progress",
                "idle": False, "stalled": False, "terminal_complete": True,
                "idle_class_reason": (
                    f"all {records_landed}/{cells_total} cell records "
                    "landed and a LIVE dispatch lease is transitioning "
                    "the fleet to the approved successor"),
            }
        return {
            "idle_class": "completed_untransitioned", "idle": True,
            "stalled": False, "terminal_complete": True,
            "idle_class_reason": (
                f"all {records_landed}/{cells_total} cell records landed, "
                "no runner process is alive and NO successor is proven "
                "dispatched by the durable transition queue: this is "
                "fleet idle time after a terminal completion, not health "
                "(a terminal seed is not a stalled seed — the remedy is "
                "dispatch, never a restart)"),
        }
    if gpu_utilization_unknown:
        return {
            "idle_class": "pending_gpu_utilization_unknown", "idle": False,
            "stalled": False, "terminal_complete": False,
            "idle_class_reason": ("assigned-GPU utilization is unreadable; "
                                  "the guard alerts on observed facts only"),
        }
    if not gpu_idle_now:
        return {
            "idle_class": "pending_gpu_busy", "idle": False,
            "stalled": False, "terminal_complete": False,
            "idle_class_reason": ("cells are pending and the assigned GPU "
                                  "is busy: work is being done"),
        }
    if observed_idle > idle_after_seconds:
        return {
            "idle_class": "stalled_pending_cells", "idle": True,
            "stalled": True, "terminal_complete": False,
            "idle_class_reason": (
                f"cells are pending, no runner process is alive, the "
                f"assigned GPU is idle and progress stopped "
                f"{observed_idle:.0f}s ago (> {idle_after_seconds:.0f}s)"),
        }
    return {
        "idle_class": "pending_recent_progress", "idle": False,
        "stalled": False, "terminal_complete": False,
        "idle_class_reason": (
            f"cells are pending and progress was observed {observed_idle:.0f}s "
            f"ago, within the {idle_after_seconds:.0f}s threshold"),
    }


def guard_transition_authority(records: list, *, identity: str,
                               experiment: str, now: datetime,
                               lease_seconds: float =
                               etq.DEFAULT_CLAIM_LEASE_SECONDS,
                               ) -> dict[str, Any]:
    """Does the ACTIVE durable transition authorize guarding this
    identity? (finding AUD-GEN-20260815-250, requirement 3.)

    The queue records are the ONLY authority. Three typed verdicts pass:

    - ``identity_is_active_chain`` — the guarded identity IS the
      dispatched/in-flight successor chain;
    - ``identity_is_terminal_predecessor_of_active_chain`` — the guarded
      identity is the current (ending) job of an active record: watching
      a terminal job's artifacts is legitimate, and classification
      renders it ``completed_transitioned``;
    - ``no_active_authority_for_family`` — nothing is dispatched in this
      experiment family, so nothing can conflict.

    Anything else — an active chain exists in the family and the guarded
    identity is neither it nor its predecessor — is a CONFLICT. The
    caller refuses; it never adopts the active chain silently.
    """
    family = etq.experiment_family(experiment)
    authorities = etq.active_chain_authorities(
        records, now=now, lease_seconds=lease_seconds, family=family)
    if not authorities:
        return {"conflict": False,
                "value": "no_active_authority_for_family",
                "family": family, "authorities": [],
                "reason": (f"no durable record proves an active dispatched "
                           f"chain in experiment family {family!r}"),
                "corrective_command": None}
    # Only the RULING (newest-dispatch) authority grants a pass: an
    # older chain whose record was never explicitly superseded — the
    # historical v1 decision — must not certify itself alive against
    # the chain that owns the fleet now.
    ruling = authorities[0]
    if ruling.get("chain_id") == identity:
        return {"conflict": False,
                "value": "identity_is_active_chain",
                "family": family, "authority": ruling,
                "authorities": authorities,
                "reason": (f"identity {identity} IS the ruling "
                           f"{ruling['transition_state']} chain of "
                           f"transition {ruling['transition_id']}"),
                "corrective_command": None}
    if ruling.get("current_identity") == identity:
        return {"conflict": False,
                "value":
                    "identity_is_terminal_predecessor_of_active_chain",
                "family": family, "authority": ruling,
                "authorities": authorities,
                "reason": (f"identity {identity} is the terminal "
                           "current job of ruling transition "
                           f"{ruling['transition_id']}; its successor "
                           f"chain is {ruling['chain_id']}"),
                "corrective_command": None}
    corrective = (
        "tools/p1lr_idle_guard.py --mode decision "
        f"--contract {ruling.get('next_contract_path')} "
        f"--identity {ruling.get('chain_id')}")
    superseded = next((a for a in authorities[1:]
                       if identity in (a.get("chain_id"),
                                       a.get("current_identity"))), None)
    if superseded is not None:
        return {"conflict": True,
                "value": "identity_is_superseded_older_chain",
                "family": family, "authorities": authorities,
                "reason": (
                    f"identity {identity} belongs to OLDER transition "
                    f"{superseded['transition_id']} (dispatched "
                    f"{superseded.get('dispatched_utc')}), but the ruling "
                    f"authority of family {family!r} is chain "
                    f"{ruling.get('chain_id')} of transition "
                    f"{ruling['transition_id']} (dispatched "
                    f"{ruling.get('dispatched_utc')}): supervising the "
                    "superseded identity could restart the WRONG "
                    "unit/identity against the live chain (finding 250)"),
                "corrective_command": corrective}
    return {"conflict": True,
            "value": "identity_conflicts_active_transition",
            "family": family, "authorities": authorities,
            "reason": (
                f"identity {identity} is neither the ruling active chain "
                f"{ruling.get('chain_id')} nor its terminal predecessor "
                f"{ruling.get('current_identity')}, yet that chain owns "
                f"experiment family {family!r} by the durable queue: "
                f"supervising {identity} could restart the WRONG "
                "unit/identity against the live chain (finding 250)"),
            "corrective_command": corrective}


def _transition_view(queue_dir: Optional[Path], *, experiment: str,
                     mode: str, identity: str, now: datetime,
                     ) -> tuple[Optional[dict], dict[str, Any]]:
    """(durable record or None, typed transition status).

    Read-only towards the queue: the guard never invents a fleet-terminal
    claim it cannot observe (it sees only its own host's seeds). The
    fleet-level observer enrols the record; the guard consumes it.
    """
    if queue_dir is None:
        return None, dict(etq.transition_status(None, now=now),
                          queue_dir=None,
                          basis="transition_queue_not_configured")
    records, _unreadable = etq.load_records(queue_dir)
    record = etq.find_record(records, experiment=experiment, mode=mode,
                             identity=identity)
    status = etq.transition_status(record, now=now)
    status["queue_dir"] = str(Path(queue_dir).expanduser())
    return record, status


# ---------------------------------------------------------------------------
# the poll: pure over injected facts
# ---------------------------------------------------------------------------

def poll(*, contract: dict, identity: str, state: dict, now: datetime,
         local_hostname: str,
         emitter: Any,
         process_alive_fn: Callable[[int], bool],
         gpu_telemetry_fn: Callable[[Optional[str]], dict],
         unit_exists_fn: Callable[[str], bool],
         restart_fn: Callable[[str], dict],
         seed_facts_fn: Optional[Callable[..., dict]] = None,
         idle_after_seconds: float = DEFAULT_IDLE_AFTER_SECONDS,
         gpu_idle_utilization_pct: int = DEFAULT_GPU_IDLE_UTILIZATION_PCT,
         max_restarts: int = DEFAULT_MAX_RESTARTS,
         restart_backoff_seconds: float = DEFAULT_RESTART_BACKOFF_SECONDS,
         mode: str = DEFAULT_MODE,
         transition_queue_dir: Optional[Path] = None,
         transition_emitter: Any = None,
         process_facts_fn: Optional[Callable[[], list]] = None,
         expected_contract_sha256: Optional[str] = None,
         enforce_transition_authority: bool = True,
         ) -> dict:
    """One guard cycle over the seeds ASSIGNED TO THIS HOST.

    ``mode`` is validated and binds the output root, the unit name and
    the per-seed cell total (finding 233): in decision mode every fact
    and the whole report come from ``decision_run.output_root``.

    ``transition_queue_dir`` binds the durable transition queue (order
    2026-08-15 §3). It answers exactly one question — is a successor
    proven dispatched? — and it answers it from records on disk, so the
    verdict survives a reboot that erases every heartbeat and process.
    When a locally-terminal seed's transition is stuck past its declared
    budget, the queue module emits ONE deduplicated incident through the
    same fleet ledger and closes that SAME event code on recovery.

    FINDING 250: when ``process_facts_fn`` is supplied (the default
    wiring always supplies it), a seed reads ``process_alive`` ONLY for
    a process proven — via :func:`match_seed_processes` over the
    injected cmdline/cwd facts — to run THIS contract identity in THIS
    mode on THIS seed; ``expected_contract_sha256`` is then mandatory.
    A same-seed process of another chain (or of unprovable identity) is
    typed ``foreign``/``unprovable`` and WITHHOLDS restarts: a bounded
    recovery beside a live foreign writer would put two writers on one
    seed's GPU. And when ``enforce_transition_authority`` holds and the
    durable queue proves an ACTIVE dispatched chain in this experiment
    family that is neither the guarded identity nor its direct
    predecessor, the WHOLE poll returns a typed refusal report — no
    incident, no restart, no silent adoption of the successor identity.

    Returns {"schema", "mode", "output_root", "transition", "seeds":
    {seed: facts+actions}, "emitted", "recovered", "restarts", "state"}
    where ``state`` is the new dedup state to persist; on an authority
    conflict the report instead carries a ``refusal`` block and empty
    action lists. The run root is only read, never written.
    """
    binding = p1lr_mode_binding(contract, mode)
    experiment = str(contract.get("experiment") or "")
    if process_facts_fn is not None and not expected_contract_sha256:
        raise ValueError(
            "poll(): process_facts_fn requires expected_contract_sha256 — "
            "identity-aware matching cannot run without the guarded "
            "contract identity (finding 250: never seed alone)")
    transition_record, transition = _transition_view(
        transition_queue_dir, experiment=experiment, mode=binding["mode"],
        identity=identity, now=now)

    # Finding 250: discover the ACTIVE durable transition and refuse a
    # conflicting identity BEFORE any incident or restart machinery.
    if transition_queue_dir is not None and enforce_transition_authority:
        records_all, _unreadable = etq.load_records(transition_queue_dir)
        authority = guard_transition_authority(
            records_all, identity=identity, experiment=experiment,
            now=now)
        if authority["conflict"]:
            return {
                "schema": REPORT_SCHEMA,
                "generated_at": now.isoformat(),
                "hostname": local_hostname,
                "identity": identity,
                "mode": binding["mode"],
                "mode_basis": "explicit_validated_parameter",
                "output_root": binding["output_root"],
                "unit_template": binding["unit_template"],
                "cells_total": binding["total_cells"],
                "transition": transition,
                "transition_authority": authority,
                "refusal": {
                    "error_code":
                        "P1LR_GUARD_IDENTITY_CONFLICTS_ACTIVE_TRANSITION",
                    "reason": authority["reason"],
                    "refusal_contract": (
                        "the guard refuses to supervise an identity that "
                        "conflicts with the ACTIVE durable transition: it "
                        "emits no incident, restarts nothing and never "
                        "silently adopts the successor identity (finding "
                        "AUD-GEN-20260815-250)"),
                    "guarded_identity": identity,
                    "guarded_experiment": experiment,
                    "guarded_mode": binding["mode"],
                    "active_authorities": authority["authorities"],
                    "corrective_command": authority["corrective_command"],
                },
                "seeds": {},
                "terminal_seeds_local": [],
                "emitted": [],
                "recovered": [],
                "restarts": [],
                "state": state,
            }
    facts_fn = seed_facts_fn or seed_facts
    assignments = contract.get("assignments") or {}
    seeds = [int(s) for s in (contract.get("seeds") or [])
             if str(s) in assignments]
    state = dict(state)
    state["seeds"] = {k: dict(v) for k, v in
                      (state.get("seeds") or {}).items()}
    emitted: list[dict] = []
    recovered: list[dict] = []
    restarts: list[dict] = []
    report_seeds: dict[str, Any] = {}
    terminal_seeds: list[int] = []
    # One process-table observation per poll (finding 250): the same
    # facts answer every seed, and the matcher — never a seed-only
    # pattern — decides which processes belong to THIS identity.
    observed_processes = (list(process_facts_fn() or [])
                          if process_facts_fn is not None else None)

    for seed in seeds:
        assignment = assignments.get(str(seed)) or {}
        if assignment.get("hostname") != local_hostname:
            continue
        unit = binding["unit_template"].format(seed=seed)
        gpu_uuid = assignment.get("gpu_uuid")
        seed_state = state["seeds"].setdefault(str(seed),
                                               default_seed_state())
        facts = facts_fn(contract, identity, seed, now, binding)
        pending = facts["records_landed"] < facts["cells_total"]
        process_match: Optional[dict] = None
        if observed_processes is not None:
            process_match = match_seed_processes(
                observed_processes, seed=seed, binding=binding,
                expected_contract_sha256=str(expected_contract_sha256))
            process_alive = process_match["alive"]
        else:
            process_alive = bool(process_alive_fn(seed))
        foreign_or_unproven = bool(
            process_match
            and (process_match["foreign_same_seed"]
                 or process_match["unprovable_same_seed"]))
        telemetry = gpu_telemetry_fn(gpu_uuid) or {}
        utilization = telemetry.get("gpu_utilization_pct")
        gpu_idle_now = (utilization is not None
                        and utilization < gpu_idle_utilization_pct)
        refusal = str(facts.get("last_heartbeat_terminal_state")
                      or "").startswith("REFUSED_")

        # Observed idleness duration: heartbeat/record progress age; a
        # seed with NO artifact at all is bounded by this guard's own
        # first-pending observation (never an invented age).
        progress_age = facts["progress_age_seconds"]
        if progress_age is not None:
            observed_idle = progress_age
            idle_basis = ("newest heartbeat/record mtime under the seed "
                          "directory")
            seed_state["first_pending_seen_utc"] = None
        else:
            first_seen = _iso(seed_state.get("first_pending_seen_utc"))
            if pending and first_seen is None:
                seed_state["first_pending_seen_utc"] = now.isoformat()
                first_seen = now
            observed_idle = (round((now - first_seen).total_seconds(), 1)
                             if first_seen else 0.0)
            idle_basis = ("no heartbeat or record has EVER been observed; "
                          "idleness bounded by this guard's first pending "
                          "observation")

        # Unknown utilization (nvidia-smi unreadable) NEVER reads as
        # idle: the guard alerts on observed facts only. Completion is
        # never health: a terminal seed with no dispatched successor is
        # idle, and it is classified, not silently dismissed (§3).
        classification = classify_seed_idleness(
            pending=pending, process_alive=process_alive,
            gpu_idle_now=gpu_idle_now,
            gpu_utilization_unknown=utilization is None,
            observed_idle=observed_idle,
            idle_after_seconds=idle_after_seconds,
            cells_total=facts["cells_total"],
            records_landed=facts["records_landed"],
            transition_state=transition.get("value"))
        idle = bool(classification["idle"])
        stalled = bool(classification["stalled"])
        if classification["terminal_complete"]:
            terminal_seeds.append(seed)

        actions: list[str] = []
        entry: dict[str, Any] = {
            "seed": seed, "unit": unit, "assigned_gpu_uuid": gpu_uuid,
            "mode": binding["mode"],
            "output_root": binding["output_root"],
            "seed_dir": facts.get("seed_dir"),
            "pending": pending,
            "records_landed": {"value": facts["records_landed"],
                               "of": facts["cells_total"],
                               "unit": "cell_records"},
            "process_alive": process_alive,
            "gpu_utilization_pct": utilization,
            "gpu_temperature_c": telemetry.get("gpu_temperature_c"),
            "gpu_utilization_unknown": utilization is None,
            "observed_idle_seconds": observed_idle,
            "idle_basis": idle_basis,
            "last_heartbeat_terminal_state":
                facts.get("last_heartbeat_terminal_state"),
            "idle": idle,
            "idle_class": classification["idle_class"],
            "idle_class_reason": classification["idle_class_reason"],
            "stalled": stalled,
            "terminal_complete": classification["terminal_complete"],
            "actions": actions,
        }
        if process_match is not None:
            entry["process_match"] = {
                "basis": process_match["basis"],
                "matched": process_match["matched"],
                "foreign_same_seed": process_match["foreign_same_seed"],
                "unprovable_same_seed":
                    process_match["unprovable_same_seed"],
            }
        if classification["terminal_complete"]:
            entry["transition_state"] = transition.get("value")
            entry["transition_id"] = transition.get("transition_id")
            entry["next_job_id"] = transition.get("next_job_id")
            entry["transition_blockers"] = transition.get("blockers")

        # WP0 rule 4: a failed/refused newest heartbeat is CLASSIFIED —
        # a SEED_FAILED-due-to-source-drift renders failure_class
        # source_drift with retry_eligible true (the bounded restart /
        # systemd Restart=on-failure IS the scheduled retry), never
        # silent progress.
        failure = classify_p1lr_failure(
            facts.get("last_heartbeat_terminal_state"),
            facts.get("last_heartbeat_error"),
            declared_class=facts.get("last_heartbeat_failure_class"),
            unit=unit, cell=facts.get("last_heartbeat_cell"))
        if failure is not None:
            entry["failure"] = failure

        if stalled:
            unit_exists = bool(unit_exists_fn(unit))
            entry["unit_exists"] = unit_exists
            affected = f"{identity}/seed{seed}"
            if not seed_state.get("idle_active"):
                payload = {
                    "identity": identity, "seed": seed, "unit": unit,
                    "mode": binding["mode"],
                    "output_root": binding["output_root"],
                    "assigned_gpu_uuid": gpu_uuid,
                    "gpu_utilization_pct": utilization,
                    "gpu_temperature_c": telemetry.get("gpu_temperature_c"),
                    "records_landed": facts["records_landed"],
                    "cells_total": facts["cells_total"],
                    "pending_cells": facts["pending_cells"],
                    "observed_idle_seconds": observed_idle,
                    "idle_after_seconds": idle_after_seconds,
                    "process_alive": False,
                    "unit_exists": unit_exists,
                    "last_heartbeat_terminal_state":
                        facts.get("last_heartbeat_terminal_state"),
                    "remediation": (
                        "bounded automatic restart in progress"
                        if unit_exists and not refusal else
                        ("configuration refusal "
                         f"({facts.get('last_heartbeat_terminal_state')}): "
                         "fix the refusal, then systemctl --user "
                         f"reset-failed {unit} && systemctl --user start "
                         f"{unit}") if refusal else
                        relaunch_command(seed, binding["mode"])),
                }
                summary = (
                    f"P1LR {binding['mode']} seed {seed} on "
                    f"{local_hostname}: assigned GPU "
                    f"{gpu_uuid} idle "
                    f"{observed_idle:.0f}s > {idle_after_seconds:.0f}s with "
                    f"{len(facts['pending_cells'])} cell(s) pending and no "
                    "runner process/heartbeat progress")
                if not unit_exists:
                    summary += (f" — unit missing; relaunch: "
                                f"{relaunch_command(seed, binding['mode'])}")
                if emitter.observe(idle_event_code(seed), SEVERITY,
                                   summary, payload,
                                   affected_object=affected):
                    seed_state["idle_active"] = True
                    emitted.append({"event_code": idle_event_code(seed),
                                    "seed": seed})
                    actions.append("incident_emitted")
                else:
                    actions.append("incident_emission_failed_will_retry")

            # Bounded service recovery (never an unbounded loop).
            restart_history = list(seed_state.get("restarts") or [])
            if foreign_or_unproven:
                # Finding 250: a live same-seed process of ANOTHER (or
                # unprovable) identity owns this seed's GPU right now.
                # Restarting this identity's unit beside it would create
                # a second writer — the restart is withheld, typed.
                actions.append(
                    "restart_withheld_foreign_seed_process"
                    if process_match["foreign_same_seed"]
                    else "restart_withheld_unproven_seed_process")
            elif refusal:
                actions.append("restart_withheld_typed_refusal")
            elif not unit_exists:
                actions.append("restart_withheld_unit_missing")
                entry["relaunch_command"] = relaunch_command(
                    seed, binding["mode"])
            elif len(restart_history) >= max_restarts:
                actions.append("restart_cap_reached")
                if not seed_state.get("cap_emitted"):
                    cap_payload = {
                        "identity": identity, "seed": seed, "unit": unit,
                        "mode": binding["mode"],
                        "output_root": binding["output_root"],
                        "restart_attempts": len(restart_history),
                        "max_restarts": max_restarts,
                        "remediation": (
                            "bounded recovery exhausted; inspect "
                            f"journalctl --user -u {unit}, fix the fault, "
                            f"then systemctl --user restart {unit}"),
                    }
                    if emitter.observe(
                            cap_event_code(seed), SEVERITY,
                            (f"P1LR seed {seed} on {local_hostname}: "
                             f"{max_restarts} bounded restart(s) exhausted "
                             "without progress; manual intervention "
                             "required"),
                            cap_payload, affected_object=affected):
                        seed_state["cap_emitted"] = True
                        emitted.append({
                            "event_code": cap_event_code(seed),
                            "seed": seed})
            else:
                last_restart = _iso(restart_history[-1]) \
                    if restart_history else None
                backoff = (restart_backoff_seconds
                           * (2 ** max(0, len(restart_history) - 1)))
                due = (last_restart is None
                       or (now - last_restart).total_seconds() >= backoff)
                if due:
                    result = restart_fn(unit)
                    restart_history.append(now.isoformat())
                    seed_state["restarts"] = restart_history
                    restarts.append({"seed": seed, "unit": unit,
                                     "attempt": len(restart_history),
                                     "of": max_restarts,
                                     "result": result})
                    actions.append("restart_attempted")
                else:
                    actions.append("restart_waiting_backoff")
                    entry["next_restart_due_in_seconds"] = round(
                        backoff - (now - last_restart).total_seconds(), 1)
        elif seed_state.get("idle_active"):
            # Direct progress-resumption evidence -> ONE recovery notice.
            evidence = {
                "summary": (f"P1LR seed {seed} on {local_hostname} "
                            "resumed progress"),
                "records_landed": facts["records_landed"],
                "cells_total": facts["cells_total"],
                "process_alive": process_alive,
                "progress_age_seconds": facts["progress_age_seconds"],
                "gpu_utilization_pct": utilization,
                "pending": pending,
            }
            affected = f"{identity}/seed{seed}"
            if emitter.recover(idle_event_code(seed), evidence,
                               affected_object=affected):
                if seed_state.get("cap_emitted"):
                    emitter.recover(cap_event_code(seed), evidence,
                                    affected_object=affected)
                state["seeds"][str(seed)] = default_seed_state()
                recovered.append({"event_code": idle_event_code(seed),
                                  "seed": seed})
                actions.append("recovery_emitted")
            else:
                actions.append("recovery_emission_failed_will_retry")

        # §3 bullet 1: a terminal seed is NOT a stalled seed — it is
        # never restarted here. Its idleness is a TRANSITION problem and
        # the durable queue owns both the verdict and the incident.
        if classification["idle_class"] == "completed_untransitioned":
            actions.append("restart_withheld_terminal_complete")
            actions.append("completed_untransitioned")

        report_seeds[str(seed)] = entry

    # §3 bullet 6: ONE deduplicated incident for a stuck transition, with
    # its dedup marker inside the DURABLE record (so a reboot or a second
    # host cannot produce a second incident), closed on the same code.
    transition_action: dict[str, Any] = {
        "action": "none",
        "reason": ("no locally terminal seed, or no durable transition "
                   "record to evaluate"),
    }
    if terminal_seeds and transition_record is not None:
        updated, transition_action = etq.evaluate_transition_incident(
            transition_record, emitter=transition_emitter or emitter,
            now=now)
        if updated.get("incident") != transition_record.get("incident"):
            etq.save_record(transition_queue_dir, updated, now=now)
            transition = etq.transition_status(updated, now=now)
            transition["queue_dir"] = str(
                Path(transition_queue_dir).expanduser())
        if transition_action["action"] == "incident_emitted":
            emitted.append({"event_code": transition_action["event_code"],
                            "transition_id": updated.get("transition_id")})
        elif transition_action["action"] == "incident_recovered":
            recovered.append({"event_code": transition_action["event_code"],
                              "transition_id": updated.get("transition_id")})
    elif terminal_seeds:
        transition_action = {
            "action": "no_durable_record",
            "reason": ("seeds are terminal on this host but no durable "
                       "transition record exists for this identity: the "
                       "fleet-level observer must enrol it before a "
                       "successor can be proven dispatched"),
        }

    return {"schema": REPORT_SCHEMA,
            "generated_at": now.isoformat(),
            "hostname": local_hostname,
            "identity": identity,
            # Finding 233: the report is BOUND to the mode it guarded —
            # in decision mode every count above came from the decision
            # root, and the report says so.
            "mode": binding["mode"],
            "mode_basis": "explicit_validated_parameter",
            "output_root": binding["output_root"],
            "unit_template": binding["unit_template"],
            "cells_total": binding["total_cells"],
            # §3 bullets 1/4: the terminal-to-next-job transition, taken
            # from durable records ONLY — never a heartbeat, a shell
            # process, a chat message or operator memory.
            "transition": transition,
            "transition_action": transition_action,
            "terminal_seeds_local": terminal_seeds,
            "seeds": report_seeds,
            "emitted": emitted,
            "recovered": recovered,
            "restarts": restarts,
            "state": state}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_identity(output_root: str) -> Optional[str]:
    from tools.multifront_status import _p1lr_discover_identity
    return _p1lr_discover_identity(output_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--mode", choices=list(P1LR_MODES),
                        default=DEFAULT_MODE,
                        help="P1LR execution mode to guard (finding 233): "
                             "output root, unit name, expected heartbeat/"
                             "record mode and total cells derive from it. "
                             "'decision' binds every fact AND the report "
                             "to decision_run.output_root; an identity "
                             "belonging to the other mode's root refuses")
    parser.add_argument("--identity", default=None,
                        help="experiment identity; discovered from the "
                             "newest local runner heartbeat under the "
                             "SELECTED mode's output root when omitted")
    parser.add_argument("--report-out", type=Path, default=None,
                        help="also write the report JSON here (atomic); "
                             "the report is bound to the guarded mode's "
                             "output root")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR,
                        help="guard dedup/backoff state; NEVER inside the "
                             "run's output root")
    parser.add_argument("--transition-queue-dir", type=Path,
                        default=DEFAULT_TRANSITION_QUEUE_DIR,
                        help="durable terminal-to-next-job queue (order "
                             "2026-08-15 §3): the ONLY authority for "
                             "whether a successor is dispatched, and the "
                             "home of the deduplicated undispatched "
                             "incident; NEVER inside a run's output root")
    parser.add_argument("--no-transition-queue", action="store_true",
                        help="do not consult the durable transition queue "
                             "(a terminal seed then always renders "
                             "completed_untransitioned, because nothing "
                             "can prove a successor was dispatched)")
    parser.add_argument("--idle-after-seconds", type=float,
                        default=DEFAULT_IDLE_AFTER_SECONDS)
    parser.add_argument("--gpu-idle-utilization-pct", type=int,
                        default=DEFAULT_GPU_IDLE_UTILIZATION_PCT)
    parser.add_argument("--max-restarts", type=int,
                        default=DEFAULT_MAX_RESTARTS)
    parser.add_argument("--restart-backoff-seconds", type=float,
                        default=DEFAULT_RESTART_BACKOFF_SECONDS)
    parser.add_argument("--dry-run", action="store_true",
                        help="report facts only: no incident emission, no "
                             "restart, no state mutation")
    args = parser.parse_args()

    try:
        contract = json.loads(args.contract.read_text())
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": f"contract unreadable: "
                                   f"{type(exc).__name__}"}))
        return 2
    try:
        binding = p1lr_mode_binding(contract, args.mode)
    except P1lrModeRefusal as refusal:
        print(json.dumps({"error_code": refusal.code,
                          "error": refusal.reason, **refusal.facts},
                         indent=1, sort_keys=True))
        return 2
    identity = args.identity or _discover_identity(binding["output_root"])
    if not identity:
        print(json.dumps({
            "error_code": "P1LR_NO_IDENTITY",
            "mode": binding["mode"],
            "output_root": binding["output_root"],
            "error": "no experiment identity: none supplied and no runner "
                     f"heartbeat discoverable under the local "
                     f"{binding['mode']} output root "
                     f"{binding['output_root']}"}, indent=1, sort_keys=True))
        return 2
    try:
        identity_presence = p1lr_verify_identity_binding(binding, identity)
    except P1lrModeRefusal as refusal:
        print(json.dumps({"error_code": refusal.code,
                          "error": refusal.reason, **refusal.facts},
                         indent=1, sort_keys=True))
        return 2

    hostname = socket.gethostname()
    state_path = args.state_dir / "state.json"
    state = load_state(state_path)

    if args.dry_run:
        class _NullEmitter:
            def observe(self, *a, **kw):
                return False

            def recover(self, *a, **kw):
                return False

        emitter: Any = _NullEmitter()
        transition_emitter: Any = emitter
        restart_fn: Callable[[str], dict] = lambda unit: {
            "ok": False, "returncode": None, "stderr": "dry-run"}
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        emitter = LedgerEmitter(hostname)
        # The SAME fleet ledger, under the transition queue's own source
        # name — one alerting stack, two truthful producers.
        transition_emitter = etq.LedgerEmitter(hostname)
        restart_fn = default_restart

    # Finding 250: the guarded contract's sha256 IS the identity the
    # process matcher requires — matching is contract identity + mode +
    # seed + output root, NEVER seed alone.
    contract_sha256 = hashlib.sha256(
        args.contract.read_bytes()).hexdigest()

    report = poll(
        contract=contract, identity=identity, state=state,
        now=datetime.now(timezone.utc), local_hostname=hostname,
        emitter=emitter,
        process_alive_fn=lambda seed: False,  # superseded by the matcher
        process_facts_fn=default_process_facts,
        expected_contract_sha256=contract_sha256,
        gpu_telemetry_fn=default_gpu_telemetry,
        unit_exists_fn=default_unit_exists, restart_fn=restart_fn,
        idle_after_seconds=args.idle_after_seconds,
        gpu_idle_utilization_pct=args.gpu_idle_utilization_pct,
        max_restarts=args.max_restarts,
        restart_backoff_seconds=args.restart_backoff_seconds,
        mode=args.mode,
        transition_queue_dir=(None if args.no_transition_queue
                              else args.transition_queue_dir),
        transition_emitter=transition_emitter)
    refused = bool(report.get("refusal"))
    if not args.dry_run and not refused:
        save_state(state_path, report["state"])
    printable = {k: v for k, v in report.items() if k != "state"}
    printable["identity_presence"] = identity_presence
    printable["contract_sha256"] = contract_sha256
    printable["dry_run"] = bool(args.dry_run)
    text = json.dumps(printable, indent=1, sort_keys=True)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.report_out.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(args.report_out)
    print(text)
    # A typed identity/authority refusal is exit 2, like every other
    # refusal-to-bind: the guard never supervises a conflicting identity.
    return 2 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
