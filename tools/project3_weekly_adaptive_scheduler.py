#!/usr/bin/env python3
"""Prune Project 3 weekly-pool backlog with pragmatic successive halving.

The phase orchestrator intentionally creates broad experiment batches. This
script owns the opposite job: once partial evidence exists, keep enough probes
to learn from each candidate and defer the rest of the low-value factorial
grid. Deferred subjobs are not deleted; they can be restored manually if later
evidence makes them interesting again.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
FINANCIAL_ROOT = Path("/home/harveybc/Documents/GitHub/financial-data")
DEFAULT_DB = FINANCIAL_ROOT / "experiments" / "weekly_walkforward_pool" / "project3_weekly_pool.sqlite"

sys.path.insert(0, str(TOOLS_DIR))
from project3_weekly_pool import connect, init_db  # noqa: E402


DEFAULT_PHASES = (
    "asset_preset_broadening_phase5_v1",
    "risk_adjusted_reward_phase7_v3",
    "sltp_risk_geometry_phase8_v3",
)
DEFERRED_STATUS = "deferred"
NEGATIVE_GATE_SCORE = -1_000_000.0


@dataclass
class SubjobRow:
    external_id: str
    job_id: int
    status: str
    priority: int
    validation_start: str
    depends_on_subjob_id: str | None
    warm_start_parent_subjob_id: str | None
    result_json: str | None


@dataclass
class Candidate:
    job_id: int
    external_id: str
    asset: str
    timeframe: str
    phase: str
    rows: list[SubjobRow]
    metrics: list[float]

    @property
    def pending(self) -> list[SubjobRow]:
        return [row for row in self.rows if row.status == "pending"]

    @property
    def deferred(self) -> list[SubjobRow]:
        return [row for row in self.rows if row.status == DEFERRED_STATUS]

    @property
    def running_count(self) -> int:
        return sum(1 for row in self.rows if row.status == "running")

    @property
    def done_count(self) -> int:
        return len(self.metrics)

    @property
    def mean(self) -> float | None:
        if not self.metrics:
            return None
        return sum(self.metrics) / len(self.metrics)

    @property
    def lcb(self) -> float | None:
        if not self.metrics:
            return None
        if len(self.metrics) == 1:
            return self.metrics[0]
        mean = self.mean
        assert mean is not None
        variance = sum((value - mean) ** 2 for value in self.metrics) / (len(self.metrics) - 1)
        return mean - math.sqrt(variance) / math.sqrt(len(self.metrics))

    @property
    def std(self) -> float | None:
        if not self.metrics:
            return None
        if len(self.metrics) == 1:
            return 0.0
        mean = self.mean
        assert mean is not None
        variance = sum((value - mean) ** 2 for value in self.metrics) / (len(self.metrics) - 1)
        return math.sqrt(variance)

    @property
    def worst(self) -> float | None:
        if not self.metrics:
            return None
        return min(self.metrics)

    def l2_score(self, std_penalty: float, worst_loss_penalty: float) -> float | None:
        mean = self.mean
        std = self.std
        worst = self.worst
        if mean is None or std is None or worst is None:
            return None
        return mean - float(std_penalty) * std - float(worst_loss_penalty) * max(0.0, -worst)

    @property
    def has_pending_dependencies(self) -> bool:
        return any(row.depends_on_subjob_id or row.warm_start_parent_subjob_id for row in self.pending)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _metric_from_result(result_json: str | None) -> float | None:
    if not result_json:
        return None
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError:
        return None
    if result.get("trade_gate_passed") is False:
        return NEGATIVE_GATE_SCORE
    for key in (
        "train_validation_l1_score",
        "train_validation_selection_score",
        "l2_week_score",
        "score",
        "raw_score",
        "train_validation_risk_adjusted_composite_score",
        "train_validation_composite_score",
    ):
        value = result.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    train_tail = result.get("train_tail_total_return")
    validation = result.get("validation_total_return")
    if train_tail is not None and validation is not None:
        try:
            return 0.5 * float(train_tail) + 0.5 * float(validation)
        except (TypeError, ValueError):
            return None
    return None


def _load_candidates(conn: sqlite3.Connection, phases: tuple[str, ...]) -> list[Candidate]:
    rows = conn.execute(
        """
        SELECT j.id AS job_id,
               j.external_id AS job_external_id,
               j.asset,
               j.timeframe,
               COALESCE(json_extract(j.config_json, '$.experiment_phase'), 'base') AS phase,
               s.external_id AS subjob_external_id,
               s.status,
               s.priority,
               s.validation_start,
               s.depends_on_subjob_id,
               s.warm_start_parent_subjob_id,
               s.result_json
        FROM jobs j
        JOIN subjobs s ON s.job_id=j.id
        ORDER BY j.id, s.priority, s.validation_start, s.id
        """
    ).fetchall()
    by_job: dict[int, dict[str, Any]] = {}
    phase_set = set(phases)
    for row in rows:
        phase = str(row["phase"])
        if phase_set and phase not in phase_set:
            continue
        job_id = int(row["job_id"])
        bucket = by_job.setdefault(
            job_id,
            {
                "external_id": str(row["job_external_id"]),
                "asset": str(row["asset"]),
                "timeframe": str(row["timeframe"]),
                "phase": phase,
                "rows": [],
                "metrics": [],
            },
        )
        subjob = SubjobRow(
            external_id=str(row["subjob_external_id"]),
            job_id=job_id,
            status=str(row["status"]),
            priority=100 if row["priority"] is None else int(row["priority"]),
            validation_start=str(row["validation_start"]),
            depends_on_subjob_id=row["depends_on_subjob_id"],
            warm_start_parent_subjob_id=row["warm_start_parent_subjob_id"],
            result_json=row["result_json"],
        )
        bucket["rows"].append(subjob)
        if subjob.status == "done":
            metric = _metric_from_result(subjob.result_json)
            if metric is not None:
                bucket["metrics"].append(metric)
    return [
        Candidate(
            job_id=job_id,
            external_id=str(data["external_id"]),
            asset=str(data["asset"]),
            timeframe=str(data["timeframe"]),
            phase=str(data["phase"]),
            rows=list(data["rows"]),
            metrics=list(data["metrics"]),
        )
        for job_id, data in by_job.items()
    ]


def _ordered_pending(candidate: Candidate) -> list[SubjobRow]:
    return sorted(candidate.pending, key=lambda row: (row.priority, row.validation_start, row.external_id))


def _ordered_deferred(candidate: Candidate) -> list[SubjobRow]:
    return sorted(candidate.deferred, key=lambda row: (row.priority, row.validation_start, row.external_id))


def _select_probe_keep_ids(candidate: Candidate, slots: int) -> set[str]:
    pending = _ordered_pending(candidate)
    if slots <= 0 or not pending:
        return set()
    if slots >= len(pending):
        return {row.external_id for row in pending}
    if candidate.has_pending_dependencies:
        return {row.external_id for row in pending[:slots]}
    if slots == 1:
        return {pending[len(pending) // 2].external_id}
    keep_indexes = {
        round(idx * (len(pending) - 1) / (slots - 1))
        for idx in range(slots)
    }
    return {pending[idx].external_id for idx in sorted(keep_indexes)}


def _select_deferred_promote_ids(candidate: Candidate, slots: int) -> set[str]:
    deferred = _ordered_deferred(candidate)
    if slots <= 0 or not deferred:
        return set()
    if slots >= len(deferred):
        return {row.external_id for row in deferred}
    if candidate.has_pending_dependencies:
        return {row.external_id for row in deferred[:slots]}
    if slots == 1:
        return {deferred[len(deferred) // 2].external_id}
    keep_indexes = {
        round(idx * (len(deferred) - 1) / (slots - 1))
        for idx in range(slots)
    }
    return {deferred[idx].external_id for idx in sorted(keep_indexes)}


def _apply_dependency_closure(
    candidates: list[Candidate],
    defer_reasons: dict[str, str],
) -> None:
    pending_by_id = {
        row.external_id: row
        for candidate in candidates
        for row in candidate.pending
    }
    changed = True
    while changed:
        changed = False
        for subjob_id, row in list(pending_by_id.items()):
            if subjob_id in defer_reasons:
                continue
            parents = [row.depends_on_subjob_id, row.warm_start_parent_subjob_id]
            parent = next((parent for parent in parents if parent in defer_reasons), None)
            if parent:
                defer_reasons[subjob_id] = f"dependency_parent_deferred:{parent}"
                changed = True


def build_defer_plan(
    conn: sqlite3.Connection,
    *,
    phases: tuple[str, ...],
    min_probe_weeks: int,
    probe_quota: int,
    keep_top_per_asset_timeframe: int,
    mean_floor: float,
    lcb_floor: float,
    l2_floor: float | None = None,
    l2_std_penalty: float = 0.25,
    l2_worst_loss_penalty: float = 0.50,
) -> dict[str, Any]:
    candidates = _load_candidates(conn, phases)
    defer_reasons: dict[str, str] = {}
    candidate_summaries: list[dict[str, Any]] = []

    for candidate in candidates:
        pending = _ordered_pending(candidate)
        if not pending:
            continue
        mean = candidate.mean
        lcb = candidate.lcb
        std = candidate.std
        worst = candidate.worst
        l2_score = candidate.l2_score(l2_std_penalty, l2_worst_loss_penalty)
        summary = {
            "job_id": candidate.external_id,
            "asset": candidate.asset,
            "timeframe": candidate.timeframe,
            "phase": candidate.phase,
            "done": candidate.done_count,
            "running": candidate.running_count,
            "pending": len(pending),
            "mean": mean,
            "lcb": lcb,
            "std": std,
            "worst": worst,
            "l2_score": l2_score,
            "deferred": 0,
            "reason": None,
        }
        if candidate.done_count < min_probe_weeks:
            remaining_slots = max(0, probe_quota - candidate.done_count - candidate.running_count)
            keep_ids = _select_probe_keep_ids(candidate, remaining_slots)
            for row in pending:
                if row.external_id not in keep_ids:
                    defer_reasons[row.external_id] = (
                        f"probe_cap_until_min_evidence:"
                        f"done={candidate.done_count},running={candidate.running_count},quota={probe_quota}"
                    )
            summary["deferred"] = sum(1 for row in pending if row.external_id in defer_reasons)
            if summary["deferred"]:
                summary["reason"] = "probe_cap_until_min_evidence"
            candidate_summaries.append(summary)
            continue
        floor = lcb_floor if l2_floor is None else l2_floor
        if mean is None or lcb is None or l2_score is None:
            continue
        if mean < mean_floor or l2_score < floor:
            reason = (
                "weak_evidence:"
                f"done={candidate.done_count},mean={mean:.8f},std={std:.8f},"
                f"worst={worst:.8f},l2={l2_score:.8f}"
            )
            for row in pending:
                defer_reasons[row.external_id] = reason
            summary["deferred"] = len(pending)
            summary["reason"] = "weak_evidence"
        candidate_summaries.append(summary)

    ranked_groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if (
            candidate.done_count >= min_probe_weeks
            and candidate.pending
            and candidate.mean is not None
            and candidate.l2_score(l2_std_penalty, l2_worst_loss_penalty) is not None
        ):
            ranked_groups[(candidate.asset, candidate.timeframe)].append(candidate)

    keep_candidate_ids: set[int] = set()
    for group_candidates in ranked_groups.values():
        ranked = sorted(
            group_candidates,
            key=lambda item: (
                item.l2_score(l2_std_penalty, l2_worst_loss_penalty)
                if item.l2_score(l2_std_penalty, l2_worst_loss_penalty) is not None
                else NEGATIVE_GATE_SCORE,
                item.mean if item.mean is not None else NEGATIVE_GATE_SCORE,
                item.done_count,
            ),
            reverse=True,
        )
        keep_candidate_ids.update(item.job_id for item in ranked[:keep_top_per_asset_timeframe])
        for loser in ranked[keep_top_per_asset_timeframe:]:
            reason = (
                "not_top_candidate_for_asset_timeframe:"
                f"ranked_by_l2_keep_top={keep_top_per_asset_timeframe},"
                f"done={loser.done_count},mean={loser.mean:.8f},"
                f"l2={loser.l2_score(l2_std_penalty, l2_worst_loss_penalty):.8f}"
            )
            for row in loser.pending:
                defer_reasons.setdefault(row.external_id, reason)

    _apply_dependency_closure(candidates, defer_reasons)

    reason_counts = Counter(defer_reasons.values())
    affected_jobs = {
        candidate.external_id
        for candidate in candidates
        if any(row.external_id in defer_reasons for row in candidate.pending)
    }
    return {
        "generated_at": utc_now(),
        "phases": list(phases),
        "policy": {
            "min_probe_weeks": min_probe_weeks,
            "probe_quota": probe_quota,
            "keep_top_per_asset_timeframe": keep_top_per_asset_timeframe,
            "mean_floor": mean_floor,
            "lcb_floor": lcb_floor,
            "l2_floor": lcb_floor if l2_floor is None else l2_floor,
            "l2_formula": "mean - std_penalty * std - worst_loss_penalty * max(0,-worst)",
            "l2_std_penalty": l2_std_penalty,
            "l2_worst_loss_penalty": l2_worst_loss_penalty,
        },
        "candidate_count": len(candidates),
        "affected_job_count": len(affected_jobs),
        "defer_subjob_count": len(defer_reasons),
        "reason_counts": dict(reason_counts),
        "candidate_summaries": [
            item for item in candidate_summaries if item.get("deferred")
        ][:80],
        "defer_reasons": defer_reasons,
        "kept_evidence_candidate_count": len(keep_candidate_ids),
    }


def apply_defer_plan(conn: sqlite3.Connection, plan: dict[str, Any], *, status: str = DEFERRED_STATUS) -> dict[str, Any]:
    defer_reasons: dict[str, str] = dict(plan.get("defer_reasons") or {})
    if not defer_reasons:
        return {"updated": 0, "status": status}
    now = utc_now()
    updated = 0
    by_reason = Counter(defer_reasons.values())
    with conn:
        for subjob_id, reason in defer_reasons.items():
            cur = conn.execute(
                """
                UPDATE subjobs
                SET status=?,
                    error=?,
                    claimed_by=NULL,
                    claimed_at=NULL,
                    heartbeat_at=NULL,
                    updated_at=?
                WHERE external_id=? AND status='pending'
                """,
                (status, f"adaptive_scheduler:{reason}", now, subjob_id),
            )
            updated += int(cur.rowcount)
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (
                "adaptive_scheduler_defer",
                "project3_weekly_adaptive_scheduler",
                _json(
                    {
                        "updated": updated,
                        "status": status,
                        "phases": plan.get("phases"),
                        "policy": plan.get("policy"),
                        "affected_job_count": plan.get("affected_job_count"),
                        "reason_counts": dict(by_reason),
                    }
                ),
                now,
            ),
        )
    return {"updated": updated, "status": status}


def build_promote_plan(
    conn: sqlite3.Connection,
    *,
    phases: tuple[str, ...],
    min_probe_weeks: int,
    promote_quota: int,
    keep_top_per_asset_timeframe: int,
    mean_floor: float,
    lcb_floor: float,
    l2_floor: float | None = None,
    l2_std_penalty: float = 0.25,
    l2_worst_loss_penalty: float = 0.50,
) -> dict[str, Any]:
    candidates = _load_candidates(conn, phases)
    promote_reasons: dict[str, str] = {}
    candidate_summaries: list[dict[str, Any]] = []

    ranked_groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.done_count >= min_probe_weeks and candidate.deferred:
            l2_score = candidate.l2_score(l2_std_penalty, l2_worst_loss_penalty)
            if candidate.mean is not None and l2_score is not None:
                ranked_groups[(candidate.asset, candidate.timeframe)].append(candidate)

    top_candidate_ids: set[int] = set()
    floor = lcb_floor if l2_floor is None else l2_floor
    for group_candidates in ranked_groups.values():
        viable = [
            candidate
            for candidate in group_candidates
            if candidate.mean is not None
            and candidate.mean >= mean_floor
            and (candidate.l2_score(l2_std_penalty, l2_worst_loss_penalty) or NEGATIVE_GATE_SCORE) >= floor
        ]
        ranked = sorted(
            viable,
            key=lambda item: (
                item.l2_score(l2_std_penalty, l2_worst_loss_penalty)
                if item.l2_score(l2_std_penalty, l2_worst_loss_penalty) is not None
                else NEGATIVE_GATE_SCORE,
                item.mean if item.mean is not None else NEGATIVE_GATE_SCORE,
                item.done_count,
            ),
            reverse=True,
        )
        top_candidate_ids.update(item.job_id for item in ranked[:keep_top_per_asset_timeframe])

    for candidate in candidates:
        deferred = _ordered_deferred(candidate)
        if not deferred:
            continue
        target = 0
        reason = None
        if candidate.done_count < min_probe_weeks:
            target = max(
                0,
                min(
                    promote_quota,
                    min_probe_weeks - candidate.done_count - candidate.running_count - len(candidate.pending),
                ),
            )
            reason = f"complete_min_probe_evidence:done={candidate.done_count},target={min_probe_weeks}"
        elif candidate.job_id in top_candidate_ids:
            target = max(0, min(promote_quota, len(deferred)))
            reason = (
                "extend_top_l2_candidate:"
                f"done={candidate.done_count},mean={candidate.mean:.8f},"
                f"l2={candidate.l2_score(l2_std_penalty, l2_worst_loss_penalty):.8f}"
            )
        if target <= 0:
            continue
        promote_ids = _select_deferred_promote_ids(candidate, target)
        for subjob_id in promote_ids:
            promote_reasons[subjob_id] = reason or "promote_deferred"
        candidate_summaries.append(
            {
                "job_id": candidate.external_id,
                "asset": candidate.asset,
                "timeframe": candidate.timeframe,
                "phase": candidate.phase,
                "done": candidate.done_count,
                "running": candidate.running_count,
                "pending": len(candidate.pending),
                "deferred": len(deferred),
                "promoted": len(promote_ids),
                "mean": candidate.mean,
                "l2_score": candidate.l2_score(l2_std_penalty, l2_worst_loss_penalty),
                "reason": reason,
            }
        )

    return {
        "generated_at": utc_now(),
        "phases": list(phases),
        "policy": {
            "min_probe_weeks": min_probe_weeks,
            "promote_quota": promote_quota,
            "keep_top_per_asset_timeframe": keep_top_per_asset_timeframe,
            "mean_floor": mean_floor,
            "lcb_floor": lcb_floor,
            "l2_floor": floor,
            "l2_formula": "mean - std_penalty * std - worst_loss_penalty * max(0,-worst)",
            "l2_std_penalty": l2_std_penalty,
            "l2_worst_loss_penalty": l2_worst_loss_penalty,
        },
        "candidate_count": len(candidates),
        "promote_subjob_count": len(promote_reasons),
        "candidate_summaries": candidate_summaries[:80],
        "promote_reasons": promote_reasons,
    }


def apply_promote_plan(conn: sqlite3.Connection, plan: dict[str, Any]) -> dict[str, Any]:
    promote_reasons: dict[str, str] = dict(plan.get("promote_reasons") or {})
    if not promote_reasons:
        return {"updated": 0, "status": "pending"}
    now = utc_now()
    updated = 0
    by_reason = Counter(promote_reasons.values())
    with conn:
        for subjob_id, reason in promote_reasons.items():
            cur = conn.execute(
                """
                UPDATE subjobs
                SET status='pending',
                    error=?,
                    claimed_by=NULL,
                    claimed_at=NULL,
                    heartbeat_at=NULL,
                    updated_at=?
                WHERE external_id=? AND status=?
                """,
                (f"adaptive_scheduler_promote:{reason}", now, subjob_id, DEFERRED_STATUS),
            )
            updated += int(cur.rowcount)
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (
                "adaptive_scheduler_promote",
                "project3_weekly_adaptive_scheduler",
                _json(
                    {
                        "updated": updated,
                        "phases": plan.get("phases"),
                        "policy": plan.get("policy"),
                        "reason_counts": dict(by_reason),
                    }
                ),
                now,
            ),
        )
    return {"updated": updated, "status": "pending"}


def _status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        str(row["status"]): int(row["n"])
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM subjobs GROUP BY status")
    }


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    conn = connect(args.db)
    init_db(conn)
    before = _status_counts(conn)
    promote_plan = {"promote_reasons": {}}
    promoted = {"updated": 0, "status": "pending"}
    active_count = before.get("pending", 0) + before.get("running", 0)
    if args.promote_deferred and active_count <= args.promote_when_active_lte:
        promote_plan = build_promote_plan(
            conn,
            phases=tuple(args.phase),
            min_probe_weeks=args.min_probe_weeks,
            promote_quota=args.promote_quota,
            keep_top_per_asset_timeframe=args.keep_top_per_asset_timeframe,
            mean_floor=args.mean_floor,
            lcb_floor=args.lcb_floor,
            l2_floor=args.l2_floor,
            l2_std_penalty=args.l2_std_penalty,
            l2_worst_loss_penalty=args.l2_worst_loss_penalty,
        )
        if args.apply:
            promoted = apply_promote_plan(conn, promote_plan)
    plan = build_defer_plan(
        conn,
        phases=tuple(args.phase),
        min_probe_weeks=args.min_probe_weeks,
        probe_quota=args.probe_quota,
        keep_top_per_asset_timeframe=args.keep_top_per_asset_timeframe,
        mean_floor=args.mean_floor,
        lcb_floor=args.lcb_floor,
        l2_floor=args.l2_floor,
        l2_std_penalty=args.l2_std_penalty,
        l2_worst_loss_penalty=args.l2_worst_loss_penalty,
    )
    applied = {"updated": 0, "status": args.defer_status}
    if args.apply:
        applied = apply_defer_plan(conn, plan, status=args.defer_status)
    after = _status_counts(conn)
    plan_out = {
        key: value
        for key, value in plan.items()
        if key != "defer_reasons"
    }
    plan_out["before_counts"] = before
    plan_out["after_counts"] = after
    plan_out["apply"] = bool(args.apply)
    plan_out["applied"] = applied
    plan_out["promote_deferred"] = bool(args.promote_deferred)
    plan_out["promoted"] = promoted
    plan_out["promote_plan"] = {
        key: value
        for key, value in promote_plan.items()
        if key != "promote_reasons"
    }
    return plan_out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--phase", action="append", default=None)
    ap.add_argument("--min-probe-weeks", type=int, default=5)
    ap.add_argument("--probe-quota", type=int, default=8)
    ap.add_argument("--keep-top-per-asset-timeframe", type=int, default=2)
    ap.add_argument("--mean-floor", type=float, default=0.0)
    ap.add_argument("--lcb-floor", type=float, default=0.0005)
    ap.add_argument("--l2-floor", type=float, default=None)
    ap.add_argument("--l2-std-penalty", type=float, default=0.25)
    ap.add_argument("--l2-worst-loss-penalty", type=float, default=0.50)
    ap.add_argument("--defer-status", default=DEFERRED_STATUS)
    ap.add_argument("--promote-deferred", action="store_true")
    ap.add_argument("--promote-quota", type=int, default=3)
    ap.add_argument("--promote-when-active-lte", type=int, default=0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--sleep-sec", type=int, default=1800)
    args = ap.parse_args()
    if args.phase is None:
        args.phase = list(DEFAULT_PHASES)
    return args


def main() -> None:
    args = parse_args()
    while True:
        payload = run_once(args)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if not args.daemon:
            return
        time.sleep(max(300, args.sleep_sec))


if __name__ == "__main__":
    main()
