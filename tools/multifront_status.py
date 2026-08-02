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
    if state == "running" and item.get("owner_blocked_reason"):
        raise QueueStateError("running item cannot carry owner_blocked_reason")
    if state in _REQUIRES_HASHES:
        hashes = item.get("hashes") or {}
        if not hashes.get("config_sha256") and not hashes.get("plan_sha256"):
            raise QueueStateError(
                f"{state} item requires config_sha256 or plan_sha256"
            )
    if state == "dependency_blocked" and not item.get("dependency"):
        raise QueueStateError("dependency_blocked item must name its dependency")
    if state == "owner_blocked" and not item.get("owner_blocked_reason"):
        raise QueueStateError("owner_blocked item must name the owner decision")


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


def collect(
    *,
    snapshot_path: Path,
    watchdog_path: Path,
    social_db_path: Path,
    supervisor_url: str,
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
    if status and status.get("workers"):
        worker = next(iter(status["workers"].values()))
        population = worker.get("shared_population") or {}
        candidate = worker.get("candidate") or {}
        eta = worker.get("candidate_eta") or {}
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
        unavailable.append({"field": "f1_optimization", "reason": "supervisor status unreachable or empty"})

    if network:
        anchors = set()
        tips = set()
        for participant in (network.get("participants") or {}).values():
            for w in ((participant.get("status") or {}).get("workers") or {}).values():
                anchors.add((w.get("finalized_height"), str(w.get("finalized_hash"))[:12]))
                tips.add(str(w.get("tip_hash"))[:12])
        fronts.setdefault("f1_optimization", {})["chain_coherence"] = {
            "basis": "observed",
            "distinct_unfinalized_tips": len(tips),
            "distinct_finalized_anchors": sorted(
                [list(a) for a in anchors], key=lambda x: (x[0] is None, x)
            ),
            "note": "anchor divergence must converge before archive; no mutation",
        }

    # ── Front 2: venues (watchdog packet, observed) ──
    watchdog = _load_json_file(watchdog_path)
    if watchdog:
        register("paper_execution_watchdog", str(watchdog_path), watchdog.get("generated_at"))
        mt5 = watchdog.get("mt5") or {}
        heartbeat = mt5.get("heartbeat") or {}
        fronts["f2_business_reality"] = {
            "basis": "observed",
            "active_events": watchdog.get("active_event_keys"),
            "alpaca_sessions": {"value": (watchdog.get("alpaca") or {}).get("complete_sessions"), "unit": "sessions", "horizon": "cumulative", "note": "cumulative, not continuous-window"},
            "ibkr_sessions": {"value": (watchdog.get("ibkr") or {}).get("complete_sessions"), "unit": "sessions", "horizon": "cumulative"},
            "mt5_heartbeat_age": {"value": heartbeat.get("age_seconds"), "unit": "seconds", "horizon": "instant"},
            "mt5_read_only": mt5.get("read_only"),
            "orders_anywhere": {"value": 0 if not watchdog.get("active_event_keys") else None, "unit": "orders", "horizon": "instant", "basis": "derived", "formula": "zero only when no exposure event is active; venue payloads carry the direct counts"},
        }
    else:
        unavailable.append({"field": "f2_business_reality", "reason": "watchdog packet unreadable"})

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
    if snapshot:
        meta = snapshot.get("meta") or {}
        register("audit_snapshot", str(snapshot_path), meta.get("generated_at"))
        fronts["f4_audit_evidence"] = {
            "basis": "observed",
            "snapshot_sha256": meta.get("snapshot_sha256"),
            "tests_packet_available": bool((snapshot.get("tests") or {}).get("available")),
        }
    else:
        unavailable.append({"field": "f4_audit_evidence", "reason": "audit snapshot unreadable"})

    # ── Queue (taxonomy of section 4) ──
    queue: list[dict[str, Any]] = []
    if network and network.get("plan_jobs"):
        for job in network["plan_jobs"]:
            job_status = str(job.get("status") or "")
            if job_status == "running":
                state = "running"
            elif job_status == "queued":
                state = "dependency_blocked"
            else:
                state = "materialized"
            entry: dict[str, Any] = {
                "id": str(job.get("job_id")),
                "front": "f1",
                "state": state,
                "hashes": {"plan_sha256": network.get("plan_hash")},
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
