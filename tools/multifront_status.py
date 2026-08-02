#!/usr/bin/env python3
"""Consolidated machine-readable multi-front status contract.

Implements owner-approved improvements 1 (consolidated status) and 4
(queue-state taxonomy) from the 2026-08-01 acceptance contract. Aggregates
existing tier-0 evidence sources read-only; never invents a value — a source
that cannot be read yields an explicit `unavailable` entry instead.

Every numeric field carries `unit` and `horizon`; every section records its
source, fetch time and freshness. `basis` distinguishes `observed` (read
directly from a source) from `derived` (computed here, formula named).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

SCHEMA = "agent_multi.multifront_status.v1"

QUEUE_STATES = {
    "running",
    "materialized",
    "dependency_blocked",
    "proposed",
    "owner_blocked",
}

# States that can never be simultaneously true for one item, and field
# requirements per state (acceptance contract section 4).
_REQUIRES_HASHES = {"running", "materialized"}


class QueueStateError(ValueError):
    """A queue item violates the canonical taxonomy."""


_SHA256_HEX = 64

# Per-state field contract (finding 036): a field allowed in one state is
# forbidden in every state that does not list it.
_STATE_FIELDS: dict[str, dict[str, set[str]]] = {
    "running": {"required": set(), "forbidden": {"dependency", "owner_blocked_reason"}},
    "materialized": {"required": set(), "forbidden": {"dependency", "owner_blocked_reason"}},
    "dependency_blocked": {"required": {"dependency"}, "forbidden": {"owner_blocked_reason"}},
    "proposed": {"required": set(), "forbidden": {"dependency", "owner_blocked_reason"}},
    "owner_blocked": {"required": {"owner_blocked_reason"}, "forbidden": {"dependency"}},
}


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == _SHA256_HEX and all(
        character in "0123456789abcdef" for character in text.lower()
    )


def validate_queue_item(item: Mapping[str, Any]) -> None:
    state = item.get("state")
    if state not in QUEUE_STATES:
        raise QueueStateError(f"unknown queue state: {state!r}")
    states_claimed = item.get("also_states") or []
    if states_claimed:
        raise QueueStateError(
            "a queue item has exactly one canonical state; "
            f"got extra states {states_claimed!r}"
        )
    contract = _STATE_FIELDS[state]
    for field in contract["forbidden"]:
        if item.get(field):
            raise QueueStateError(f"{state} item cannot carry {field}")
    for field in contract["required"]:
        if not item.get(field):
            raise QueueStateError(f"{state} item must carry {field}")
    hashes = item.get("hashes") or {}
    for key, value in hashes.items():
        if key.endswith("_sha256") and value and not _valid_sha256(value):
            raise QueueStateError(f"{key} is not a valid SHA-256 hex digest")
    if state in _REQUIRES_HASHES:
        if not _valid_sha256(hashes.get("config_sha256")) and not _valid_sha256(
            hashes.get("plan_sha256")
        ):
            raise QueueStateError(
                f"{state} item requires a syntactically valid "
                "config_sha256 or plan_sha256"
            )


def validate_queue(items: list[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for item in items:
        item_id = str(item.get("id") or "")
        if not item_id:
            raise QueueStateError("queue item requires an id")
        if item_id in seen:
            raise QueueStateError(f"duplicate queue item id: {item_id}")
        seen.add(item_id)
        validate_queue_item(item)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_seconds(iso_ts: Optional[str]) -> Optional[float]:
    if not iso_ts:
        return None
    try:
        ts = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - ts).total_seconds(), 1)
    except ValueError:
        return None


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _load_json_file(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _get_url(url: str, timeout: float) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


def _as_dict(value: Any) -> dict:
    """Finding 037: valid JSON of unexpected shape must degrade, not raise.

    Any nested section accessed with `.get()` goes through this, so a truthy
    list/str/number in place of an object reads as an empty section and the
    downstream field becomes explicitly unavailable.
    """
    return value if isinstance(value, dict) else {}


def _direct_count(value: Any) -> Optional[int]:
    """Parse a direct venue count; only non-negative true integers qualify.

    Booleans, strings, floats and negatives carry no documented coercion
    contract and therefore read as unavailable (finding 037), never as a
    number invented by coercion.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def collect(
    *,
    snapshot_path: Path,
    watchdog_path: Path,
    social_db_path: Path,
    supervisor_url: str,
    l0_heartbeat_path: Path = Path.home()
    / ".local/state/lts/demo-execution-l0/heartbeat.json",
    l0_db_path: Path = Path.home() / ".local/state/lts/demo-execution-l0.sqlite",
    timeout: float = 6.0,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    fronts: dict[str, Any] = {}

    def register(name: str, locator: str, payload_ts: Optional[str]) -> None:
        sources.append(
            {
                "name": name,
                "locator": locator,
                "fetched_at": _now(),
                "payload_generated_at": payload_ts,
                "freshness_seconds": _age_seconds(payload_ts),
            }
        )

    # ── Front 1: optimization (supervisor API, observed) ──
    status = _get_url(f"{supervisor_url}/api/status", timeout)
    network = _get_url(f"{supervisor_url}/api/network", timeout)
    if not isinstance(status, dict):
        status = None  # truthy wrong-type payload degrades to unavailable
    workers = _as_dict(status.get("workers")) if status else {}
    worker = next(iter(workers.values()), None) if workers else None
    if status and isinstance(worker, dict):
        population = _as_dict(worker.get("shared_population"))
        candidate = _as_dict(worker.get("candidate"))
        eta = _as_dict(worker.get("candidate_eta"))
        register("supervisor_status", f"{supervisor_url}/api/status", status.get("updated_at"))
        fronts["f1_optimization"] = {
            "basis": "observed",
            "plan_id": status.get("plan_id"),
            "plan_sha256": status.get("plan_hash"),
            "job_id": status.get("job_id"),
            "phase": status.get("phase"),
            "stage": {"value": candidate.get("stage"), "of": candidate.get("total_stages"), "name": candidate.get("stage_name"), "unit": "ordinal", "horizon": "campaign"},
            "generation": {"value": population.get("generation"), "unit": "count", "horizon": "job"},
            "generation_evaluated": {"value": population.get("evaluated"), "of": population.get("pop_size"), "unit": "candidates", "horizon": "generation"},
            "best_fitness": {"value": worker.get("best_performance"), "unit": "dimensionless_full_period_proxy", "horizon": "job_0", "note": "owner-ratified Alternative A: initialization evidence only; job 1 selects with robust_weekly_rap_fitness (fraction/week)"},
            "candidates_per_hour_recent": {"value": eta.get("candidates_per_hour"), "unit": "candidates/hour", "horizon": "recent_window", "basis": "derived", "formula": "median of matched start/result log pairs (supervisor)"},
        }
    else:
        unavailable.append({"field": "f1_optimization", "reason": "supervisor status unreachable, empty, or wrong type"})

    if not isinstance(network, dict):
        network = None  # truthy wrong-type payload degrades to unavailable
    if network:
        register("supervisor_network", f"{supervisor_url}/api/network", None)
        anchors = set()
        tips = set()
        for participant in _as_dict(network.get("participants")).values():
            nested = _as_dict(_as_dict(_as_dict(participant).get("status")).get("workers"))
            for w in nested.values():
                w = _as_dict(w)
                anchors.add((w.get("finalized_height"), str(w.get("finalized_hash"))[:12]))
                tips.add(str(w.get("tip_hash"))[:12])
        fronts.setdefault("f1_optimization", {})["chain_coherence"] = {
            "basis": "observed",
            "source": "supervisor_network",
            "distinct_unfinalized_tips": {"value": len(tips), "unit": "count", "horizon": "instant"},
            "distinct_finalized_anchors": {
                "value": sorted(
                    [list(a) for a in anchors], key=lambda x: (x[0] is None, x)
                ),
                "unit": "(block_height, hash_prefix12) pairs",
                "horizon": "instant",
            },
            "note": "anchor divergence must converge before archive; no mutation",
        }

    # ── Front 2: venues (watchdog packet, observed) ──
    watchdog = _load_json_file(watchdog_path)
    if isinstance(watchdog, dict) and watchdog:
        register("paper_execution_watchdog", str(watchdog_path), watchdog.get("generated_at"))
        alpaca = _as_dict(watchdog.get("alpaca"))
        ibkr = _as_dict(watchdog.get("ibkr"))
        mt5 = _as_dict(watchdog.get("mt5"))
        heartbeat = _as_dict(mt5.get("heartbeat"))
        # Finding 035: order/position counts come ONLY from direct venue
        # payloads. If any venue count is missing, the aggregate is
        # unavailable — never zero-by-absence. Alerts stay a separate field.
        # Finding 037: wrong-type sections and non-numeric counts also
        # degrade to unavailability instead of crashing or coercing.
        venue_orders: dict[str, Any] = {
            "alpaca": _as_dict(alpaca.get("detail")).get("open_orders"),
            "ibkr": _as_dict(ibkr.get("latest_complete")).get("open_orders"),
            "mt5": _as_dict(mt5.get("latest_snapshot")).get("orders_total"),
        }
        venue_positions: dict[str, Any] = {
            "alpaca": _as_dict(alpaca.get("detail")).get("open_positions"),
            "ibkr": _as_dict(ibkr.get("latest_complete")).get("open_positions"),
            "mt5": _as_dict(mt5.get("latest_snapshot")).get("positions_total"),
        }

        def _aggregate(counts: Mapping[str, Any], label: str) -> dict[str, Any]:
            parsed = {k: _direct_count(v) for k, v in counts.items()}
            missing = sorted(k for k, v in counts.items() if v is None)
            invalid = sorted(
                k for k, v in counts.items() if v is not None and parsed[k] is None
            )
            if missing or invalid:
                reasons = []
                if missing:
                    reasons.append(f"missing direct counts from: {', '.join(missing)}")
                if invalid:
                    reasons.append(
                        f"non-numeric direct counts from: {', '.join(invalid)}"
                    )
                unavailable.append(
                    {
                        "field": f"f2_business_reality.{label}.aggregate",
                        "reason": "; ".join(reasons),
                    }
                )
                total: Optional[int] = None
            else:
                total = sum(v for v in parsed.values() if v is not None)
            return {
                "per_venue": parsed,
                "aggregate": total,
                "unit": label,
                "horizon": "instant",
                "basis": "observed",
                "source": "paper_execution_watchdog",
            }

        fronts["f2_business_reality"] = {
            "basis": "observed",
            "active_events": watchdog.get("active_event_keys"),
            "alpaca_sessions": {"value": alpaca.get("complete_sessions"), "unit": "sessions", "horizon": "cumulative", "note": "cumulative, not continuous-window"},
            "ibkr_sessions": {"value": ibkr.get("complete_sessions"), "unit": "sessions", "horizon": "cumulative"},
            "mt5_heartbeat_age": {"value": heartbeat.get("age_seconds"), "unit": "seconds", "horizon": "instant"},
            "mt5_read_only": mt5.get("read_only"),
            "open_orders": _aggregate(venue_orders, "orders"),
            "open_positions": _aggregate(venue_positions, "positions"),
        }
    else:
        unavailable.append({"field": "f2_business_reality", "reason": "watchdog packet unreadable or wrong type"})

    # ── Front 2b: L0 demo-execution runner (heartbeat + ledger, observed) ──
    l0_heartbeat = _load_json_file(l0_heartbeat_path)
    if isinstance(l0_heartbeat, dict) and l0_heartbeat:
        register("l0_demo_execution", str(l0_heartbeat_path), l0_heartbeat.get("at"))
        l0_counts: dict[str, Any] = {}
        try:
            l0_con = sqlite3.connect(f"file:{l0_db_path}?mode=ro", uri=True)
            l0_counts = {
                "decisions": l0_con.execute(
                    "SELECT COUNT(*) FROM decisions"
                ).fetchone()[0],
                "would_be_orders": l0_con.execute(
                    "SELECT COUNT(*) FROM decisions WHERE outcome LIKE 'would_be%'"
                ).fetchone()[0],
                "lifecycle_events": l0_con.execute(
                    "SELECT COUNT(*) FROM lifecycle_events"
                ).fetchone()[0],
            }
            l0_con.close()
        except sqlite3.Error:
            unavailable.append(
                {"field": "f2_business_reality.l0_demo_execution.ledger",
                 "reason": "L0 ledger unreadable"}
            )
        fronts.setdefault("f2_business_reality", {})["l0_demo_execution"] = {
            "basis": "observed",
            "source": "l0_demo_execution",
            "heartbeat_age": {"value": _age_seconds(l0_heartbeat.get("at")),
                              "unit": "seconds", "horizon": "instant"},
            "last_outcome": l0_heartbeat.get("outcome"),
            "halt_state": l0_heartbeat.get("halt_state"),
            "capability_evidence": l0_heartbeat.get("capability_evidence"),
            "network_submissions": {
                "value": l0_heartbeat.get("network_submissions_session"),
                "unit": "submissions", "horizon": "runner_session",
                "note": "structurally zero: the sink has no network path",
            },
            "ledger": {"value": l0_counts or None, "unit": "rows",
                       "horizon": "cumulative"},
        }
    else:
        unavailable.append(
            {"field": "f2_business_reality.l0_demo_execution",
             "reason": "L0 runner heartbeat missing or wrong type"}
        )

    # ── Front 3: social (OLAP counts, observed) ──
    try:
        con = sqlite3.connect(f"file:{social_db_path}?mode=ro", uri=True)
        posts = con.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        runs = con.execute("SELECT COUNT(*) FROM collection_runs").fetchone()[0]
        drafts = con.execute("SELECT COUNT(*) FROM drafts").fetchone()[0]
        con.close()
        register("social_intelligence_olap", str(social_db_path), None)
        fronts["f3_social"] = {
            "basis": "observed",
            "collection_runs": {"value": runs, "unit": "runs", "horizon": "cumulative"},
            "posts_collected": {"value": posts, "unit": "posts", "horizon": "cumulative"},
            "drafts": {"value": drafts, "unit": "drafts", "horizon": "cumulative", "note": "publishing gated on human approval"},
        }
    except sqlite3.Error:
        unavailable.append({"field": "f3_social", "reason": "social OLAP unreadable"})

    # ── Front 4: audit/evidence (snapshot packet, observed) ──
    snapshot = _load_json_file(snapshot_path)
    if isinstance(snapshot, dict) and isinstance(snapshot.get("meta"), dict):
        meta = snapshot["meta"]
        register("audit_snapshot", str(snapshot_path), meta.get("generated_at"))
        fronts["f4_audit_evidence"] = {
            "basis": "observed",
            "source": "audit_snapshot",
            "snapshot_sha256": meta.get("snapshot_sha256"),
            "tests_packet_available": bool(_as_dict(snapshot.get("tests")).get("available")),
        }
    else:
        unavailable.append(
            {
                "field": "f4_audit_evidence",
                "reason": "audit snapshot unreadable, wrong type, or missing meta",
            }
        )

    # ── Queue (taxonomy of section 4) ──
    # Finding 036: only explicitly known supervisor states enter the
    # executable queue; anything else (failed, completed, unknown) is exposed
    # in queue_excluded, never disguised as materialized work.
    _SUPERVISOR_STATE_MAP = {"running": "running", "queued": "dependency_blocked"}
    queue: list[dict[str, Any]] = []
    queue_excluded: list[dict[str, Any]] = []
    plan_jobs = network.get("plan_jobs") if network else None
    if plan_jobs is not None and not isinstance(plan_jobs, list):
        # Finding 037: a wrong-type plan_jobs section degrades explicitly.
        unavailable.append(
            {"field": "queue.f1", "reason": "plan_jobs is not a list"}
        )
        plan_jobs = []
    if plan_jobs:
        for job in plan_jobs:
            if not isinstance(job, dict):
                queue_excluded.append(
                    {
                        "id": repr(job)[:80],
                        "front": "f1",
                        "supervisor_status": "wrong_type",
                        "reason": "plan job entry is not an object; recorded as history/error",
                    }
                )
                continue
            job_status = str(job.get("status") or "")
            state = _SUPERVISOR_STATE_MAP.get(job_status)
            if state is None:
                queue_excluded.append(
                    {
                        "id": str(job.get("job_id")),
                        "front": "f1",
                        "supervisor_status": job_status or "unknown",
                        "reason": "not an executable-queue state; recorded as history/error",
                    }
                )
                continue
            plan_hash = network.get("plan_hash")
            if state in _REQUIRES_HASHES and not _valid_sha256(plan_hash):
                # Finding 037: a malformed live hash must not make our own
                # taxonomy validator raise; the job is excluded explicitly.
                queue_excluded.append(
                    {
                        "id": str(job.get("job_id")),
                        "front": "f1",
                        "supervisor_status": job_status,
                        "reason": "plan_sha256 missing or malformed; cannot enter executable queue",
                    }
                )
                continue
            entry: dict[str, Any] = {
                "id": str(job.get("job_id")),
                "front": "f1",
                "state": state,
                "hashes": {"plan_sha256": plan_hash},
            }
            if state == "dependency_blocked":
                entry["dependency"] = "job-0 champion/elite archive (fail-closed materializer)"
            queue.append(entry)
    queue.append(
        {
            "id": "protected-canaries-m3",
            "front": "f2",
            "state": "dependency_blocked",
            "dependency": "24-hour continuous windows + owner review (doc 22 M2->M3)",
            "hashes": {},
        }
    )
    queue.append(
        {
            "id": "darwinex-zero-subscription",
            "front": "f2",
            "state": "owner_blocked",
            "owner_blocked_reason": "recurring spending not approved (owner, 2026-08-01)",
            "hashes": {},
        }
    )
    validate_queue(queue)

    return {
        "schema": SCHEMA,
        "generated_at": _now(),
        "sources": sources,
        "fronts": fronts,
        "queue": queue,
        "queue_excluded": queue_excluded,
        "unavailable": unavailable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path.home() / ".local/state/agent-multi/audit-snapshots/latest.json")
    parser.add_argument("--watchdog", type=Path, default=Path.home() / ".local/state/lts/paper-execution-watchdog/latest.json")
    parser.add_argument("--social-db", type=Path, default=Path.home() / ".local/state/agent-multi/social-intelligence.sqlite")
    parser.add_argument("--supervisor-url", default="http://127.0.0.1:8795")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packet = collect(
        snapshot_path=args.snapshot,
        watchdog_path=args.watchdog,
        social_db_path=args.social_db,
        supervisor_url=args.supervisor_url,
    )
    text = json.dumps(packet, indent=1, sort_keys=True)
    if args.output:
        args.output.write_text(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        print(json.dumps({"written": str(args.output), "sha256": digest}))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
