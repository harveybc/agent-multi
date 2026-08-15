#!/usr/bin/env python3
"""Durable terminal-to-next-job transition queue (order 2026-08-15 §3).

The defect this module closes was directly observed by the auditor on
2026-08-15: the P1LR decision identity ``c0e53cf18b7d60dd`` was TERMINAL
with 16/16 cell records, ``process_alive`` was false, no cell was pending
and no next job was running — and the fleet still reported ``idle:
false`` merely because the previous experiment had completed. Fleet-level
idle time after a terminal completion was invisible, which violates the
owner's cardinal rule that useful machines must never sit idle.

The fix is a DURABLE record of the transition itself. A transition is a
first-class object with its own identity, persisted as one JSON file per
terminal job under a state directory that is NEVER inside any run's
output root. It carries the current job, the terminal result identity,
the next approved job, the materialization state, the dispatch state and
the blockers. Everything an operator would otherwise have to remember.

Reconstruction contract (§3): after a reboot the transition is rebuilt
from these records ALONE — :func:`reconstruct_transitions` reads nothing
but the queue directory. No heartbeat, no shell process, no chat
message and no operator memory can create, complete or close a
transition; a heartbeat can only ever be corroborating evidence recorded
INTO a durable record.

Dispatch is leased and FAILS CLOSED. :func:`claim_dispatch` refuses a
second, different claim while a claim is live (``TRANSITION_DUPLICATE_
DISPATCH``), refuses a job that is not materialized, refuses a blocked
job, and refuses a chain identity that conflicts with the record's
proven chain (``TRANSITION_CHAIN_CONFLICT``). Healthy nodes may still
join the ONE proven common chain through :func:`join_chain`, and a node
that is unavailable is recorded by :func:`mark_node` as a typed fact —
an unavailable node degrades the fleet breadth of a dispatch, it never
forks a second chain and never silently blocks the healthy ones.

Alerting reuses the existing fleet ledger through
``tools/incident_emit.py`` (there is exactly one alerting stack). The
deduplication state lives IN the durable record, so a reboot cannot
produce a second incident for the same undispatched successor and a
recovery closes the SAME event code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

RECORD_SCHEMA = "agent_multi.experiment_transition_record.v1"
RECONSTRUCTION_SCHEMA = "agent_multi.experiment_transition_reconstruction.v1"
SOURCE = "experiment_transition_queue"
FRONT = "front1"
SEVERITY = "P1"

DEFAULT_QUEUE_DIR = (Path.home()
                     / ".local/state/agent-multi/experiment-transition-queue")

#: How long an approved successor may stay undispatched before the fleet
#: is declared idle-by-omission. Declared per record; this is the default.
DEFAULT_TRANSITION_BUDGET_SECONDS = 3600.0
#: A dispatch claim is a LEASE. A claimer that reboots mid-dispatch stops
#: renewing it; only after the lease expires may another actor reclaim,
#: and the reclaim is journalled as such (never as a fresh dispatch).
DEFAULT_CLAIM_LEASE_SECONDS = 900.0

MATERIALIZATION_STATES = ("not_started", "in_progress", "materialized",
                          "blocked", "invalid")
DISPATCH_STATES = ("undispatched", "dispatch_claimed", "dispatched",
                   "failed", "superseded")
NODE_STATES = ("ready", "unavailable", "dispatched", "failed")

#: Typed transition states. ``completed_untransitioned`` is the state the
#: order requires the fleet to surface: a terminal experiment with no
#: dispatched successor.
TRANSITION_STATES = ("current_job_running", "completed_untransitioned",
                     "transition_dispatch_in_progress", "transitioned",
                     "superseded")

UNDISPATCHED_EVENT_PREFIX = "experiment_transition_undispatched"

_JOURNAL_MAX = 500


class TransitionRefusal(ValueError):
    """A typed refusal of a transition mutation.

    Every refusal is FAIL-CLOSED: the record is left untouched and the
    caller must not dispatch. ``as_block()`` renders the refusal for a
    status packet with its corrective evidence attached.
    """

    def __init__(self, code: str, reason: str, **facts: Any):
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason
        self.facts = facts

    def as_block(self, **extra: Any) -> dict[str, Any]:
        block: dict[str, Any] = {
            "source": SOURCE,
            "basis": "refusal",
            "error_code": self.code,
            "reason": self.reason,
            "refusal_contract": (
                "a refused transition mutation leaves the durable record "
                "unchanged and dispatches nothing: a duplicate or "
                "chain-conflicting dispatch fails closed (order "
                "2026-08-15 §3)"),
        }
        block.update(self.facts)
        block.update(extra)
        return block


# ---------------------------------------------------------------------------
# identity, time and durable storage
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> Optional[datetime]:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def transition_id(experiment: str, mode: str, identity: str) -> str:
    """The durable identity of ONE transition.

    Derived only from the job that is ending — experiment, execution mode
    and content-addressed run identity — so the record is stable while
    the successor is still unknown, still being approved, or being
    re-approved. A transition is therefore enrolled at most once no
    matter how many hosts observe the same terminal completion.
    """
    material = f"{experiment}|{mode}|{identity}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def record_path(queue_dir: Path, tid: str) -> Path:
    return Path(queue_dir).expanduser() / f"{tid}.json"


def save_record(queue_dir: Path, record: dict, *,
                now: Optional[datetime] = None) -> Path:
    """Persist a record atomically (tmp + fsync + replace).

    A torn record after a power loss would defeat the whole point of the
    durable queue, so the bytes are flushed to disk before the rename and
    the containing directory is synced afterwards.
    """
    record = dict(record)
    record["updated_utc"] = (now or _now()).isoformat()
    directory = Path(queue_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = record_path(directory, record["transition_id"])
    tmp = path.with_suffix(".json.tmp")
    payload = json.dumps(record, indent=1, sort_keys=True)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
    return path


def load_record(path: Path) -> Optional[dict]:
    """One durable record, or None when it is absent/unreadable/foreign.

    A file that does not carry :data:`RECORD_SCHEMA` is NOT coerced: an
    unrecognised document is not a transition and must never be treated
    as one.
    """
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("schema") != RECORD_SCHEMA:
        return None
    if not value.get("transition_id"):
        return None
    return value


def load_records(queue_dir: Path) -> tuple[list[dict], list[dict]]:
    """(valid records, typed unreadable entries) under ``queue_dir``."""
    directory = Path(queue_dir).expanduser()
    records: list[dict] = []
    unreadable: list[dict] = []
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError as exc:
        return records, [{"path": str(directory),
                          "reason": f"queue directory unreadable: "
                                    f"{type(exc).__name__}"}]
    for path in paths:
        record = load_record(path)
        if record is None:
            unreadable.append({
                "path": str(path),
                "reason": ("not a readable "
                           f"{RECORD_SCHEMA} document; never coerced into "
                           "a transition")})
            continue
        records.append(record)
    return records, unreadable


def _journal(record: dict, event: str, now: datetime,
             **detail: Any) -> None:
    entries = list(record.get("journal") or [])
    entries.append({"utc": now.isoformat(), "event": event,
                    "detail": detail or None})
    record["journal"] = entries[-_JOURNAL_MAX:]


# ---------------------------------------------------------------------------
# record construction and mutation
# ---------------------------------------------------------------------------

def new_transition_record(
    *,
    experiment: str,
    mode: str,
    identity: str,
    output_root: Optional[str] = None,
    contract_path: Optional[str] = None,
    contract_sha256: Optional[str] = None,
    cells_total: Optional[int] = None,
    now: Optional[datetime] = None,
    transition_budget_seconds: float = DEFAULT_TRANSITION_BUDGET_SECONDS,
    observed_by: Optional[str] = None,
) -> dict:
    """A durable transition record for a job that is ending.

    The record starts with NO terminal result and NO successor: those are
    added by :func:`record_terminal_result` and :func:`approve_successor`
    as they become facts. Every field the order names is present from
    the beginning (current job, terminal result, next approved job,
    materialization state, dispatch state, blockers) so no consumer ever
    has to distinguish "absent field" from "unknown state".
    """
    moment = now or _now()
    tid = transition_id(experiment, mode, identity)
    record = {
        "schema": RECORD_SCHEMA,
        "transition_id": tid,
        "created_utc": moment.isoformat(),
        "updated_utc": moment.isoformat(),
        "observed_by": observed_by or socket.gethostname(),
        "current_job": {
            "experiment": experiment,
            "mode": mode,
            "identity": identity,
            "output_root": output_root,
            "contract_path": contract_path,
            "contract_sha256": contract_sha256,
            "cells_total": cells_total,
        },
        "terminal_result": None,
        "next_job": None,
        "materialization_state": "not_started",
        "dispatch_state": "undispatched",
        "dispatch": None,
        "blockers": [],
        "nodes": {},
        "transition_budget_seconds": float(transition_budget_seconds),
        "incident": {"event_code": undispatched_event_code(tid),
                     "open": False, "opened_utc": None,
                     "closed_utc": None},
        "journal": [],
        "durability_contract": (
            "this record is the ONLY authority for the transition; on "
            "reboot it is reconstructed from this file alone — never "
            "from a heartbeat, a shell process, a chat message or "
            "operator memory (order 2026-08-15 §3)"),
    }
    _journal(record, "transition_enrolled", moment, experiment=experiment,
             mode=mode, identity=identity)
    return record


def undispatched_event_code(tid: str) -> str:
    """The ONE event code for 'approved successor still undispatched'.

    Keyed on the transition identity, so every emission and every
    recovery for one transition collapses onto a single ledger
    fingerprint — one incident, never a message flood.
    """
    return f"{UNDISPATCHED_EVENT_PREFIX}.{tid}"


def record_terminal_result(record: dict, *, identity: str,
                           records_landed: int, cells_total: int,
                           terminal_utc: Optional[str] = None,
                           evidence_root: Optional[str] = None,
                           now: Optional[datetime] = None,
                           observed_by: Optional[str] = None) -> dict:
    """Seal the terminal result of the current job into the record.

    ``terminal_utc`` starts the transition budget. When the caller can
    observe the real completion time (newest cell record), that instant
    is used; otherwise the observation time is recorded AS the
    observation time and labelled, never presented as the completion
    time.
    """
    moment = now or _now()
    record = dict(record)
    declared = _iso(terminal_utc)
    record["terminal_result"] = {
        "identity": identity,
        "records_landed": int(records_landed),
        "cells_total": int(cells_total),
        "complete": int(records_landed) >= int(cells_total) > 0,
        "terminal_utc": (declared or moment).isoformat(),
        "terminal_utc_basis": ("observed_completion_time" if declared
                               else "first_terminal_observation_by_queue"),
        "evidence_root": evidence_root,
        "observed_by": observed_by or record.get("observed_by"),
    }
    _journal(record, "terminal_result_recorded", moment,
             identity=identity, records_landed=int(records_landed),
             cells_total=int(cells_total))
    return record


def approve_successor(record: dict, *, job_id: str,
                      experiment: Optional[str] = None,
                      contract_path: Optional[str] = None,
                      contract_sha256: Optional[str] = None,
                      plan_sha256: Optional[str] = None,
                      approved_by: str,
                      approval_reference: Optional[str] = None,
                      chain_id: Optional[str] = None,
                      now: Optional[datetime] = None) -> dict:
    """Bind the NEXT approved job to this transition.

    ``chain_id`` is the proven common chain every healthy node must join.
    Re-approving a different chain for an already-dispatched transition
    is refused: the dispatched chain is history and a second chain would
    fork the arm (order 2026-08-15 §2.5, "never create parallel
    independent chains for one arm").
    """
    moment = now or _now()
    record = dict(record)
    existing = dict(record.get("next_job") or {})
    if record.get("dispatch_state") in ("dispatch_claimed", "dispatched"):
        proven = existing.get("chain_id")
        if chain_id is not None and proven is not None and chain_id != proven:
            raise TransitionRefusal(
                "TRANSITION_CHAIN_CONFLICT",
                f"transition {record['transition_id']} is already "
                f"{record['dispatch_state']} on proven chain {proven!r}; "
                f"approving chain {chain_id!r} would fork the arm into two "
                "parallel independent chains",
                transition_id=record["transition_id"],
                proven_chain_id=proven, conflicting_chain_id=chain_id,
                dispatch_state=record.get("dispatch_state"))
    record["next_job"] = {
        "id": job_id,
        "experiment": experiment,
        "contract_path": contract_path,
        "contract_sha256": contract_sha256,
        "plan_sha256": plan_sha256,
        "approved_by": approved_by,
        "approval_reference": approval_reference,
        "chain_id": chain_id if chain_id is not None
        else existing.get("chain_id"),
        "approved_utc": moment.isoformat(),
    }
    _journal(record, "successor_approved", moment, job_id=job_id,
             approved_by=approved_by, chain_id=record["next_job"]["chain_id"])
    return record


def set_materialization(record: dict, state: str, *,
                        detail: Optional[str] = None,
                        now: Optional[datetime] = None) -> dict:
    if state not in MATERIALIZATION_STATES:
        raise TransitionRefusal(
            "TRANSITION_MATERIALIZATION_STATE_INVALID",
            f"unknown materialization state {state!r}: one of "
            f"{list(MATERIALIZATION_STATES)}",
            supplied_state=state,
            known_states=list(MATERIALIZATION_STATES))
    moment = now or _now()
    record = dict(record)
    record["materialization_state"] = state
    _journal(record, "materialization_state_set", moment, state=state,
             detail=detail)
    return record


def add_blocker(record: dict, *, code: str, detail: str,
                owner_action_required: bool = False,
                dependency: Optional[str] = None,
                now: Optional[datetime] = None) -> dict:
    """Record a blocker. Re-adding the same code refreshes its detail and
    PRESERVES ``since_utc`` — a blocker's age is evidence."""
    moment = now or _now()
    record = dict(record)
    blockers = [dict(b) for b in (record.get("blockers") or [])]
    for blocker in blockers:
        if blocker.get("code") == code:
            blocker.update({"detail": detail,
                            "owner_action_required": owner_action_required,
                            "dependency": dependency,
                            "last_seen_utc": moment.isoformat()})
            break
    else:
        blockers.append({"code": code, "detail": detail,
                         "owner_action_required": owner_action_required,
                         "dependency": dependency,
                         "since_utc": moment.isoformat(),
                         "last_seen_utc": moment.isoformat()})
        _journal(record, "blocker_added", moment, code=code)
    record["blockers"] = blockers
    return record


def clear_blocker(record: dict, code: str, *,
                  now: Optional[datetime] = None) -> dict:
    moment = now or _now()
    record = dict(record)
    before = list(record.get("blockers") or [])
    after = [b for b in before if b.get("code") != code]
    record["blockers"] = after
    if len(after) != len(before):
        _journal(record, "blocker_cleared", moment, code=code)
    return record


def mark_node(record: dict, *, host: str, state: str,
              reason: Optional[str] = None,
              chain_id: Optional[str] = None,
              now: Optional[datetime] = None) -> dict:
    """Type ONE node's participation in this transition.

    An ``unavailable`` node is a typed fact, not a veto: the dispatch may
    still proceed on the healthy nodes and the record shows exactly which
    breadth was achieved.
    """
    if state not in NODE_STATES:
        raise TransitionRefusal(
            "TRANSITION_NODE_STATE_INVALID",
            f"unknown node state {state!r}: one of {list(NODE_STATES)}",
            supplied_state=state, known_states=list(NODE_STATES))
    moment = now or _now()
    record = dict(record)
    nodes = {k: dict(v) for k, v in (record.get("nodes") or {}).items()}
    entry = nodes.get(host, {})
    entry.update({"state": state, "observed_utc": moment.isoformat()})
    if reason is not None:
        entry["reason"] = reason
    if chain_id is not None:
        entry["chain_id"] = chain_id
    nodes[host] = entry
    record["nodes"] = nodes
    _journal(record, "node_state_set", moment, host=host, state=state,
             reason=reason)
    return record


def _live_claim(record: dict, now: datetime, lease_seconds: float,
                ) -> tuple[Optional[dict], bool, bool]:
    """(claim, live, liveness_provable) for the current dispatch claim.

    A claim whose timestamp will not parse can never be PROVEN live. That
    asymmetry is deliberate and it cuts both ways: the status stops
    claiming a dispatch is in flight (unproven is not progress), and
    :func:`claim_dispatch` still refuses a different claimer (unproven is
    not free either). Neither side gets to guess.
    """
    claim = record.get("dispatch")
    if not isinstance(claim, dict) or not claim.get("claim_id"):
        return None, False, True
    claimed = _iso(claim.get("renewed_utc") or claim.get("claimed_utc"))
    if claimed is None:
        return claim, False, False
    return claim, (now - claimed).total_seconds() <= lease_seconds, True


def claim_dispatch(record: dict, *, claim_id: str, host: str,
                   chain_id: Optional[str] = None,
                   now: Optional[datetime] = None,
                   lease_seconds: float = DEFAULT_CLAIM_LEASE_SECONDS,
                   ) -> dict:
    """Take the EXCLUSIVE lease to dispatch this transition's successor.

    Fail-closed refusals (order 2026-08-15 §3, "duplicate dispatch must
    fail closed"):

    ``TRANSITION_NO_APPROVED_SUCCESSOR``
        nothing is approved to dispatch.
    ``TRANSITION_NOT_MATERIALIZED``
        the successor is not materialized; an unmaterialized job is never
        dispatched on the strength of an intention.
    ``TRANSITION_BLOCKED``
        an unresolved blocker stands; clearing it is an explicit act.
    ``TRANSITION_CHAIN_CONFLICT``
        the supplied chain differs from the record's proven chain.
    ``TRANSITION_DUPLICATE_DISPATCH``
        another live claim or a completed dispatch already exists.

    Re-claiming with the SAME ``claim_id`` is idempotent (a retry after a
    lost response must not be a refusal), and renews the lease. A claim
    whose lease has expired — the reboot-during-dispatch case — may be
    taken over by a new claim_id, and that takeover is journalled as
    ``dispatch_claim_reclaimed_after_lease_expiry``, never as a fresh
    dispatch.
    """
    moment = now or _now()
    record = dict(record)
    tid = record.get("transition_id")
    next_job = dict(record.get("next_job") or {})
    if not next_job.get("id"):
        raise TransitionRefusal(
            "TRANSITION_NO_APPROVED_SUCCESSOR",
            f"transition {tid} carries no approved next job: there is "
            "nothing to dispatch",
            transition_id=tid)
    if record.get("materialization_state") != "materialized":
        raise TransitionRefusal(
            "TRANSITION_NOT_MATERIALIZED",
            f"transition {tid} successor {next_job['id']!r} is "
            f"{record.get('materialization_state')!r}, not 'materialized': "
            "an unmaterialized job is never dispatched",
            transition_id=tid, next_job_id=next_job.get("id"),
            materialization_state=record.get("materialization_state"))
    blockers = list(record.get("blockers") or [])
    if blockers:
        raise TransitionRefusal(
            "TRANSITION_BLOCKED",
            f"transition {tid} carries {len(blockers)} unresolved "
            f"blocker(s): {[b.get('code') for b in blockers]}",
            transition_id=tid, blockers=blockers)

    proven = next_job.get("chain_id")
    if chain_id is not None and proven is not None and chain_id != proven:
        raise TransitionRefusal(
            "TRANSITION_CHAIN_CONFLICT",
            f"transition {tid} is bound to proven chain {proven!r}; a "
            f"dispatch on chain {chain_id!r} would create a second "
            "parallel independent chain for one arm",
            transition_id=tid, proven_chain_id=proven,
            conflicting_chain_id=chain_id)

    claim, live, provable = _live_claim(record, moment, lease_seconds)
    state = record.get("dispatch_state")
    if state == "dispatched":
        if claim and claim.get("claim_id") == claim_id:
            return record  # idempotent re-assertion of the winning claim
        raise TransitionRefusal(
            "TRANSITION_DUPLICATE_DISPATCH",
            f"transition {tid} was already DISPATCHED by claim "
            f"{(claim or {}).get('claim_id')!r} on host "
            f"{(claim or {}).get('claimed_by')!r}; a second dispatch is "
            "refused (fail closed). A healthy node joins the proven "
            "chain with join_chain() instead",
            transition_id=tid, dispatch_state=state,
            existing_claim=claim, attempted_claim_id=claim_id,
            attempted_by=host,
            proven_chain_id=proven)
    if state == "dispatch_claimed" and claim is not None:
        if claim.get("claim_id") == claim_id:
            claim = dict(claim)
            claim["renewed_utc"] = moment.isoformat()
            record["dispatch"] = claim
            _journal(record, "dispatch_claim_renewed", moment,
                     claim_id=claim_id, host=host)
            return record
        if live or not provable:
            raise TransitionRefusal(
                "TRANSITION_DUPLICATE_DISPATCH",
                f"transition {tid} already carries a dispatch claim "
                f"{claim.get('claim_id')!r} from host "
                f"{claim.get('claimed_by')!r} that is "
                + ("LIVE" if live else
                   "of UNPROVABLE liveness (unparsable claim timestamp)")
                + "; a concurrent second dispatch is refused (fail closed)",
                transition_id=tid, dispatch_state=state,
                existing_claim=claim, attempted_claim_id=claim_id,
                attempted_by=host, claim_liveness_provable=provable,
                lease_seconds=lease_seconds)
        _journal(record, "dispatch_claim_reclaimed_after_lease_expiry",
                 moment, previous_claim_id=claim.get("claim_id"),
                 previous_host=claim.get("claimed_by"),
                 new_claim_id=claim_id, new_host=host,
                 lease_seconds=lease_seconds)

    record["dispatch_state"] = "dispatch_claimed"
    record["dispatch"] = {
        "claim_id": claim_id,
        "claimed_by": host,
        "claimed_utc": moment.isoformat(),
        "renewed_utc": moment.isoformat(),
        "lease_seconds": float(lease_seconds),
        "chain_id": chain_id if chain_id is not None else proven,
        "dispatched_utc": None,
    }
    if chain_id is not None and proven is None:
        next_job["chain_id"] = chain_id
        record["next_job"] = next_job
    _journal(record, "dispatch_claimed", moment, claim_id=claim_id,
             host=host, chain_id=record["dispatch"]["chain_id"])
    return record


def join_chain(record: dict, *, host: str, chain_id: str,
               now: Optional[datetime] = None) -> dict:
    """A healthy node joins the ONE proven common chain.

    This is the collaborative path the order allows ("healthy nodes may
    continue a proven common chain"): it never takes the dispatch lease
    and never forks the arm. A different chain id is a
    ``TRANSITION_CHAIN_CONFLICT`` refusal.
    """
    moment = now or _now()
    record = dict(record)
    tid = record.get("transition_id")
    proven = (dict(record.get("next_job") or {}).get("chain_id")
              or dict(record.get("dispatch") or {}).get("chain_id"))
    if proven is None:
        raise TransitionRefusal(
            "TRANSITION_NO_PROVEN_CHAIN",
            f"transition {tid} has no proven chain yet: a node cannot "
            "join a chain that no dispatch has established",
            transition_id=tid, host=host, attempted_chain_id=chain_id)
    if chain_id != proven:
        raise TransitionRefusal(
            "TRANSITION_CHAIN_CONFLICT",
            f"host {host} presented chain {chain_id!r} but transition "
            f"{tid} is running proven chain {proven!r}: joining would "
            "create a second parallel independent chain for one arm",
            transition_id=tid, host=host, proven_chain_id=proven,
            conflicting_chain_id=chain_id)
    if record.get("dispatch_state") not in ("dispatch_claimed",
                                            "dispatched"):
        raise TransitionRefusal(
            "TRANSITION_NOT_DISPATCHING",
            f"transition {tid} is {record.get('dispatch_state')!r}: there "
            "is no in-flight dispatch for a node to join",
            transition_id=tid, host=host,
            dispatch_state=record.get("dispatch_state"))
    return mark_node(record, host=host, state="dispatched",
                     chain_id=chain_id, now=moment,
                     reason="joined the proven common chain")


def confirm_dispatch(record: dict, *, claim_id: str,
                     now: Optional[datetime] = None,
                     evidence: Optional[dict] = None) -> dict:
    """Promote a held claim to a completed dispatch.

    Only the holder of the lease may confirm; anything else is a
    duplicate-dispatch refusal, so a stale actor that wakes up after a
    reboot cannot mark someone else's dispatch as its own.
    """
    moment = now or _now()
    record = dict(record)
    claim = dict(record.get("dispatch") or {})
    if claim.get("claim_id") != claim_id:
        raise TransitionRefusal(
            "TRANSITION_DUPLICATE_DISPATCH",
            f"claim {claim_id!r} does not hold the dispatch lease for "
            f"transition {record.get('transition_id')} (held by "
            f"{claim.get('claim_id')!r}); confirmation is refused",
            transition_id=record.get("transition_id"),
            existing_claim=claim or None, attempted_claim_id=claim_id)
    claim["dispatched_utc"] = moment.isoformat()
    if evidence:
        claim["evidence"] = evidence
    record["dispatch"] = claim
    record["dispatch_state"] = "dispatched"
    _journal(record, "dispatch_confirmed", moment, claim_id=claim_id,
             chain_id=claim.get("chain_id"))
    return record


def fail_dispatch(record: dict, *, claim_id: str, reason: str,
                  now: Optional[datetime] = None) -> dict:
    """Release a claim that could not be completed.

    The transition returns to ``undispatched`` WITH a blocker, so it
    stays visible as ``completed_untransitioned`` instead of quietly
    looking finished.
    """
    moment = now or _now()
    record = dict(record)
    claim = dict(record.get("dispatch") or {})
    if claim.get("claim_id") != claim_id:
        raise TransitionRefusal(
            "TRANSITION_DUPLICATE_DISPATCH",
            f"claim {claim_id!r} does not hold the dispatch lease for "
            f"transition {record.get('transition_id')}; release refused",
            transition_id=record.get("transition_id"),
            existing_claim=claim or None, attempted_claim_id=claim_id)
    record["dispatch_state"] = "undispatched"
    record["dispatch"] = None
    _journal(record, "dispatch_failed", moment, claim_id=claim_id,
             reason=reason)
    return add_blocker(record, code="DISPATCH_FAILED", detail=reason,
                       now=moment)


# ---------------------------------------------------------------------------
# typed status: the fleet-level answer
# ---------------------------------------------------------------------------

def transition_status(record: Optional[dict], *,
                      now: Optional[datetime] = None,
                      lease_seconds: float = DEFAULT_CLAIM_LEASE_SECONDS,
                      ) -> dict[str, Any]:
    """The typed transition state — the fleet-level answer (§3 bullet 1).

    A terminal experiment whose successor is not dispatched renders
    ``completed_untransitioned``, whatever the previous experiment's own
    completion looked like. A terminal seed with no pending cells is NOT
    reported as stalled: ``stalled`` is deliberately absent from this
    vocabulary, because completion is not a stall — it is an
    un-transitioned completion, and the remedy is dispatch, not restart.

    ``record is None`` is itself an answer: a terminal job with NO
    durable record is ``completed_untransitioned`` with the
    ``NO_DURABLE_TRANSITION_RECORD`` blocker. Missing bookkeeping never
    renders as healthy.
    """
    moment = now or _now()
    if record is None:
        return {
            "value": "completed_untransitioned",
            "basis": "no_durable_record",
            "reason": ("no durable transition record exists for this "
                       "terminal job: the successor cannot be proven "
                       "dispatched, so the fleet is un-transitioned"),
            "transition_id": None,
            "next_job_id": None,
            "materialization_state": None,
            "dispatch_state": None,
            "blockers": [{"code": "NO_DURABLE_TRANSITION_RECORD",
                          "detail": ("enroll the transition with "
                                     "experiment_transition_queue.ensure_"
                                     "terminal_record() so the next job is "
                                     "reconstructible after a reboot"),
                          "owner_action_required": False}],
            "over_budget": None,
            "successor_dispatched": False,
        }

    dispatch_state = record.get("dispatch_state") or "undispatched"
    terminal = record.get("terminal_result")
    claim, live, provable = _live_claim(record, moment, lease_seconds)
    expired = bool(claim) and not live
    next_job = dict(record.get("next_job") or {})
    blockers = list(record.get("blockers") or [])

    if not terminal:
        value = "current_job_running"
        reason = ("the current job has not recorded a terminal result; "
                  "there is nothing to transition from yet")
    elif dispatch_state == "superseded":
        value = "superseded"
        reason = "this transition was superseded by another record"
    elif dispatch_state == "dispatched":
        value = "transitioned"
        reason = (f"successor {next_job.get('id')!r} is DISPATCHED on "
                  f"proven chain {next_job.get('chain_id')!r}")
    elif dispatch_state == "dispatch_claimed" and live:
        value = "transition_dispatch_in_progress"
        reason = (f"dispatch lease {(claim or {}).get('claim_id')!r} is "
                  f"live on {(claim or {}).get('claimed_by')!r}")
    elif dispatch_state == "dispatch_claimed" and not provable:
        value = "completed_untransitioned"
        reason = ("a dispatch claim exists but its timestamp will not "
                  "parse, so its liveness cannot be PROVEN; an unprovable "
                  "claim is never rendered as a dispatch in flight")
    elif dispatch_state == "dispatch_claimed":
        value = "completed_untransitioned"
        reason = ("a dispatch claim exists but its lease EXPIRED (the "
                  "claimer stopped renewing — the reboot-during-dispatch "
                  "case); the successor is not proven dispatched")
    else:
        value = "completed_untransitioned"
        reason = (f"terminal result recorded and successor "
                  f"{next_job.get('id') or 'unapproved'} is "
                  f"{dispatch_state}: no successor job is running")

    budget = float(record.get("transition_budget_seconds")
                   or DEFAULT_TRANSITION_BUDGET_SECONDS)
    started = _iso((terminal or {}).get("terminal_utc"))
    elapsed = (round((moment - started).total_seconds(), 1)
               if started else None)
    over = (value == "completed_untransitioned" and elapsed is not None
            and elapsed > budget)
    if value == "completed_untransitioned" and not next_job.get("id"):
        blockers = blockers + [{
            "code": "NO_APPROVED_SUCCESSOR",
            "detail": ("no next job is approved for this terminal result; "
                       "the fleet has no executable work queued"),
            "owner_action_required": True}]
    return {
        "value": value,
        "basis": "durable_record",
        "reason": reason,
        "transition_id": record.get("transition_id"),
        "current_job": record.get("current_job"),
        "terminal_result": terminal,
        "next_job_id": next_job.get("id"),
        "next_job": next_job or None,
        "chain_id": next_job.get("chain_id"),
        "materialization_state": record.get("materialization_state"),
        "dispatch_state": dispatch_state,
        "dispatch_claim_expired": expired if claim else None,
        "dispatch_claim_liveness_provable": provable if claim else None,
        "blockers": blockers,
        "nodes": record.get("nodes") or {},
        "elapsed_since_terminal_seconds": elapsed,
        "transition_budget_seconds": budget,
        "over_budget": over,
        "over_budget_seconds": (round(elapsed - budget, 1)
                                if over and elapsed is not None else None),
        "successor_dispatched": value == "transitioned",
        "incident": record.get("incident"),
    }


def find_record(records: Iterable[dict], *, experiment: str, mode: str,
                identity: str) -> Optional[dict]:
    """The durable record for one (experiment, mode, identity) job."""
    tid = transition_id(experiment, mode, identity)
    for record in records:
        if record.get("transition_id") == tid:
            return record
    return None


def ensure_terminal_record(queue_dir: Path, *, experiment: str, mode: str,
                           identity: str, records_landed: int,
                           cells_total: int,
                           output_root: Optional[str] = None,
                           contract_path: Optional[str] = None,
                           contract_sha256: Optional[str] = None,
                           terminal_utc: Optional[str] = None,
                           evidence_root: Optional[str] = None,
                           now: Optional[datetime] = None,
                           transition_budget_seconds: float =
                           DEFAULT_TRANSITION_BUDGET_SECONDS,
                           observed_by: Optional[str] = None) -> dict:
    """Enrol (idempotently) the transition of a job observed TERMINAL.

    Called by the observer that first sees a complete run. Because
    :func:`transition_id` is content-addressed on the ending job, four
    hosts observing the same completion converge on ONE record and the
    terminal timestamp is written once — a later observation never
    restarts the transition budget.
    """
    moment = now or _now()
    tid = transition_id(experiment, mode, identity)
    existing = load_record(record_path(queue_dir, tid))
    record = existing or new_transition_record(
        experiment=experiment, mode=mode, identity=identity,
        output_root=output_root, contract_path=contract_path,
        contract_sha256=contract_sha256, cells_total=cells_total,
        now=moment, transition_budget_seconds=transition_budget_seconds,
        observed_by=observed_by)
    if not record.get("terminal_result"):
        record = record_terminal_result(
            record, identity=identity, records_landed=records_landed,
            cells_total=cells_total, terminal_utc=terminal_utc,
            evidence_root=evidence_root, now=moment,
            observed_by=observed_by)
        save_record(queue_dir, record, now=moment)
    elif existing is None:
        save_record(queue_dir, record, now=moment)
    return record


# ---------------------------------------------------------------------------
# reboot reconstruction (§3 bullet 4)
# ---------------------------------------------------------------------------

def reconstruct_transitions(queue_dir: Path, *,
                            now: Optional[datetime] = None,
                            lease_seconds: float =
                            DEFAULT_CLAIM_LEASE_SECONDS) -> dict[str, Any]:
    """Rebuild every transition from the durable records ALONE.

    This function deliberately reads NOTHING except ``queue_dir``: no
    heartbeat, no process table, no systemd state, no chat message, no
    operator memory. That is the whole reboot contract — after Omega
    comes back up, what the fleet owes the owner is exactly what these
    files say, and a claim whose lease expired while the host was down
    surfaces as ``completed_untransitioned`` with
    ``dispatch_claim_expired`` rather than as a dispatch that is somehow
    still in flight.
    """
    moment = now or _now()
    records, unreadable = load_records(queue_dir)
    statuses = [transition_status(record, now=moment,
                                  lease_seconds=lease_seconds)
                for record in records]
    open_transitions = [s for s in statuses
                        if s["value"] in ("completed_untransitioned",
                                          "transition_dispatch_in_progress")]
    return {
        "schema": RECONSTRUCTION_SCHEMA,
        "generated_at": moment.isoformat(),
        "queue_dir": str(Path(queue_dir).expanduser()),
        "basis": ("durable transition records on disk ONLY; no heartbeat, "
                  "shell process, chat message or operator memory is "
                  "consulted (order 2026-08-15 §3)"),
        "records_total": len(records),
        "unreadable": unreadable,
        "transitions": statuses,
        "open_transitions": open_transitions,
        "completed_untransitioned": [
            s for s in statuses if s["value"] == "completed_untransitioned"],
        "expired_dispatch_claims": [
            s for s in statuses if s.get("dispatch_claim_expired")],
        "fleet_idle_after_terminal_completion": bool(
            [s for s in statuses if s["value"] == "completed_untransitioned"]),
    }


# ---------------------------------------------------------------------------
# incident lifecycle (§3 bullet 6) — ONE ledger, ONE incident
# ---------------------------------------------------------------------------

class LedgerEmitter:
    """Adapter onto ``tools/incident_emit.py`` — the ONE fleet alerting
    path. Imported lazily so tests never touch the real ledger."""

    def __init__(self, machine: Optional[str] = None, front: str = FRONT):
        self.machine = machine or socket.gethostname()
        self.front = front

    def observe(self, event_code: str, severity: str, summary: str,
                payload: dict, affected_object: str = "-") -> bool:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import incident_emit
        return incident_emit.observe_incident(
            source=SOURCE, event_code=event_code, severity=severity,
            summary=summary, front=self.front, machine=self.machine,
            affected_object=affected_object, payload=payload)

    def recover(self, event_code: str, evidence: dict,
                affected_object: str = "-") -> bool:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import incident_emit
        return incident_emit.recover_incident(
            source=SOURCE, event_code=event_code, evidence=evidence,
            front=self.front, machine=self.machine,
            affected_object=affected_object)


def evaluate_transition_incident(record: dict, *, emitter: Any,
                                 now: Optional[datetime] = None,
                                 lease_seconds: float =
                                 DEFAULT_CLAIM_LEASE_SECONDS,
                                 ) -> tuple[dict, dict[str, Any]]:
    """ONE deduplicated incident per stuck transition; recovery CLOSES it.

    Emission happens exactly once, when a terminal job's successor is
    still undispatched past the record's declared transition budget. The
    dedup marker lives INSIDE the durable record, so a reboot, a second
    host or a second poll cannot produce a second incident, and the
    recovery uses the SAME event code — no message floods.

    A failed emission deliberately leaves the marker closed so the next
    poll retries; the ledger's fingerprint dedup makes that retry safe.

    Returns (possibly updated record, action report).
    """
    moment = now or _now()
    record = dict(record)
    status = transition_status(record, now=moment,
                               lease_seconds=lease_seconds)
    incident = dict(record.get("incident")
                    or {"event_code":
                        undispatched_event_code(record["transition_id"]),
                        "open": False})
    event_code = incident.get("event_code") or undispatched_event_code(
        record["transition_id"])
    incident["event_code"] = event_code
    current = dict(record.get("current_job") or {})
    affected = f"{current.get('experiment')}/{current.get('identity')}"
    action = "none"

    if status["value"] == "completed_untransitioned" and status["over_budget"]:
        if not incident.get("open"):
            payload = {
                "transition_id": record["transition_id"],
                "current_job": current,
                "terminal_result": record.get("terminal_result"),
                "next_job": record.get("next_job"),
                "materialization_state": record.get("materialization_state"),
                "dispatch_state": record.get("dispatch_state"),
                "blockers": status["blockers"],
                "nodes": record.get("nodes"),
                "elapsed_since_terminal_seconds":
                    status["elapsed_since_terminal_seconds"],
                "transition_budget_seconds":
                    status["transition_budget_seconds"],
                "over_budget_seconds": status["over_budget_seconds"],
                "remediation": (
                    "materialize and dispatch the approved successor, then "
                    "confirm_dispatch() on the durable record; the same "
                    "event code is closed automatically on recovery"),
            }
            summary = (
                f"experiment {current.get('experiment')} identity "
                f"{current.get('identity')} is TERMINAL "
                f"({(record.get('terminal_result') or {}).get('records_landed')}"
                f"/{(record.get('terminal_result') or {}).get('cells_total')} "
                f"records) and its successor "
                f"{status['next_job_id'] or '(none approved)'} is still "
                f"{status['dispatch_state'] or 'undispatched'} "
                f"{status['over_budget_seconds']:.0f}s beyond the "
                f"{status['transition_budget_seconds']:.0f}s transition "
                "budget: the fleet is idle after a terminal completion")
            if emitter.observe(event_code, SEVERITY, summary, payload,
                               affected_object=affected):
                incident.update({"open": True,
                                 "opened_utc": moment.isoformat(),
                                 "closed_utc": None})
                _journal(record, "transition_incident_observed", moment,
                         event_code=event_code)
                action = "incident_emitted"
            else:
                action = "incident_emission_failed_will_retry"
        else:
            action = "incident_already_open_deduplicated"
    elif incident.get("open") and status["value"] in (
            "transitioned", "transition_dispatch_in_progress", "superseded"):
        evidence = {
            "summary": (f"successor {status['next_job_id']} is "
                        f"{status['value']} on chain {status['chain_id']}"),
            "transition_id": record["transition_id"],
            "dispatch_state": status["dispatch_state"],
            "chain_id": status["chain_id"],
            "nodes": record.get("nodes"),
        }
        if emitter.recover(event_code, evidence, affected_object=affected):
            incident.update({"open": False,
                             "closed_utc": moment.isoformat()})
            _journal(record, "transition_incident_recovered", moment,
                     event_code=event_code)
            action = "incident_recovered"
        else:
            action = "recovery_emission_failed_will_retry"

    record["incident"] = incident
    return record, {"action": action, "event_code": event_code,
                    "transition_state": status["value"],
                    "over_budget": status["over_budget"],
                    "affected_object": affected}


def sweep_transition_incidents(queue_dir: Path, *, emitter: Any,
                               now: Optional[datetime] = None,
                               lease_seconds: float =
                               DEFAULT_CLAIM_LEASE_SECONDS,
                               ) -> dict[str, Any]:
    """Run :func:`evaluate_transition_incident` across the whole queue and
    persist any dedup-state change. Safe to run from a timer on every
    host: the marker is durable, so only ONE emission survives."""
    moment = now or _now()
    records, unreadable = load_records(queue_dir)
    actions = []
    for record in records:
        updated, action = evaluate_transition_incident(
            record, emitter=emitter, now=moment, lease_seconds=lease_seconds)
        if updated.get("incident") != record.get("incident"):
            save_record(queue_dir, updated, now=moment)
        actions.append(action)
    return {"generated_at": moment.isoformat(),
            "queue_dir": str(Path(queue_dir).expanduser()),
            "records_total": len(records), "unreadable": unreadable,
            "actions": actions}


# ---------------------------------------------------------------------------
# CLI: inspect / reconstruct / enrol / approve / dispatch
# ---------------------------------------------------------------------------

def _print(value: Any) -> None:
    print(json.dumps(value, indent=1, sort_keys=True))


def _load_or_die(queue_dir: Path, tid: str) -> dict:
    record = load_record(record_path(queue_dir, tid))
    if record is None:
        raise TransitionRefusal("TRANSITION_RECORD_NOT_FOUND",
                                f"no durable record {tid} under {queue_dir}",
                                transition_id=tid, queue_dir=str(queue_dir))
    return record


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR,
                        help="durable transition records; NEVER inside a "
                             "run's output root")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("reconstruct", help="rebuild every transition from the "
                                       "durable records alone (reboot path)")

    show = sub.add_parser("show", help="typed status of one transition")
    show.add_argument("transition_id")

    enrol = sub.add_parser("enrol", help="enrol a TERMINAL job (idempotent)")
    enrol.add_argument("--experiment", required=True)
    enrol.add_argument("--mode", required=True)
    enrol.add_argument("--identity", required=True)
    enrol.add_argument("--records-landed", type=int, required=True)
    enrol.add_argument("--cells-total", type=int, required=True)
    enrol.add_argument("--output-root", default=None)
    enrol.add_argument("--terminal-utc", default=None)
    enrol.add_argument("--budget-seconds", type=float,
                       default=DEFAULT_TRANSITION_BUDGET_SECONDS)

    approve = sub.add_parser("approve", help="bind the next approved job")
    approve.add_argument("transition_id")
    approve.add_argument("--job-id", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--approval-reference", default=None)
    approve.add_argument("--contract-path", default=None)
    approve.add_argument("--contract-sha256", default=None)
    approve.add_argument("--plan-sha256", default=None)
    approve.add_argument("--chain-id", default=None)

    materialize = sub.add_parser("materialize",
                                 help="set the materialization state")
    materialize.add_argument("transition_id")
    materialize.add_argument("state", choices=list(MATERIALIZATION_STATES))
    materialize.add_argument("--detail", default=None)

    claim = sub.add_parser("claim", help="take the dispatch lease "
                                         "(fails closed on duplicates)")
    claim.add_argument("transition_id")
    claim.add_argument("--claim-id", required=True)
    claim.add_argument("--chain-id", default=None)

    confirm = sub.add_parser("confirm", help="confirm a held dispatch")
    confirm.add_argument("transition_id")
    confirm.add_argument("--claim-id", required=True)

    join = sub.add_parser("join", help="join the proven common chain")
    join.add_argument("transition_id")
    join.add_argument("--host", required=True)
    join.add_argument("--chain-id", required=True)

    node = sub.add_parser("node", help="type one node's participation")
    node.add_argument("transition_id")
    node.add_argument("--host", required=True)
    node.add_argument("--state", required=True, choices=list(NODE_STATES))
    node.add_argument("--reason", default=None)

    sweep = sub.add_parser("sweep", help="incident lifecycle over the queue")
    sweep.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    queue_dir = args.queue_dir
    host = socket.gethostname()
    now = _now()

    try:
        if args.command == "reconstruct":
            _print(reconstruct_transitions(queue_dir, now=now))
            return 0
        if args.command == "show":
            _print(transition_status(_load_or_die(queue_dir,
                                                  args.transition_id),
                                     now=now))
            return 0
        if args.command == "enrol":
            record = ensure_terminal_record(
                queue_dir, experiment=args.experiment, mode=args.mode,
                identity=args.identity, records_landed=args.records_landed,
                cells_total=args.cells_total, output_root=args.output_root,
                terminal_utc=args.terminal_utc, now=now,
                transition_budget_seconds=args.budget_seconds,
                observed_by=host)
            _print(transition_status(record, now=now))
            return 0

        record = _load_or_die(queue_dir, args.transition_id)
        if args.command == "approve":
            record = approve_successor(
                record, job_id=args.job_id, approved_by=args.approved_by,
                approval_reference=args.approval_reference,
                contract_path=args.contract_path,
                contract_sha256=args.contract_sha256,
                plan_sha256=args.plan_sha256, chain_id=args.chain_id,
                now=now)
        elif args.command == "materialize":
            record = set_materialization(record, args.state,
                                         detail=args.detail, now=now)
        elif args.command == "claim":
            record = claim_dispatch(record, claim_id=args.claim_id,
                                    host=host, chain_id=args.chain_id,
                                    now=now)
        elif args.command == "confirm":
            record = confirm_dispatch(record, claim_id=args.claim_id,
                                      now=now)
        elif args.command == "join":
            record = join_chain(record, host=args.host,
                                chain_id=args.chain_id, now=now)
        elif args.command == "node":
            record = mark_node(record, host=args.host, state=args.state,
                               reason=args.reason, now=now)
        elif args.command == "sweep":
            class _NullEmitter:
                def observe(self, *a, **kw):
                    return False

                def recover(self, *a, **kw):
                    return False

            emitter: Any = (_NullEmitter() if args.dry_run
                            else LedgerEmitter(host))
            _print(sweep_transition_incidents(queue_dir, emitter=emitter,
                                              now=now))
            return 0
        save_record(queue_dir, record, now=now)
        _print(transition_status(record, now=now))
        return 0
    except TransitionRefusal as refusal:
        _print(refusal.as_block())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
