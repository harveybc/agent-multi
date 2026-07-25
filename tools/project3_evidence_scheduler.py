#!/usr/bin/env python3
"""Promote completed evidence stages into bounded downstream sweeps."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from project3_evidence_pool import connect, enqueue_plan, init_db


UPSTREAM_STAGES = ("E1_BASE_SOURCE_SCREEN", "E1_EXTERNAL_SOURCE_SCREEN")
E2_STAGE = "E2_PREPROCESSING_CONTEXT"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _job_id(config: dict[str, Any]) -> str:
    digest = hashlib.sha256(_json(config).encode("utf-8")).hexdigest()[:16]
    return (
        f"e2__{config['asset']}__{config['timeframe']}__"
        f"{config['base_feature_bundle']}__{digest}"
    )


def _finished(conn, stages: tuple[str, ...]) -> bool:
    placeholders = ",".join("?" for _ in stages)
    active = conn.execute(
        f"SELECT COUNT(*) FROM jobs WHERE stage IN ({placeholders}) AND status IN ('pending','running')",
        stages,
    ).fetchone()[0]
    total = conn.execute(
        f"SELECT COUNT(*) FROM jobs WHERE stage IN ({placeholders})",
        stages,
    ).fetchone()[0]
    return total > 0 and active == 0


def _top_e1_contracts(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT
                config_json,
                validation_annual_rap,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        json_extract(config_json, '$.asset'),
                        json_extract(config_json, '$.timeframe')
                    ORDER BY validation_annual_rap DESC, job_id ASC
                ) AS rank_in_cell
            FROM evidence_result_olap
            WHERE stage IN ('E1_BASE_SOURCE_SCREEN','E1_EXTERNAL_SOURCE_SCREEN')
              AND status='completed'
              AND validation_annual_rap IS NOT NULL
        )
        SELECT config_json, validation_annual_rap
        FROM ranked WHERE rank_in_cell=1
        ORDER BY json_extract(config_json, '$.asset'), json_extract(config_json, '$.timeframe')
        """
    ).fetchall()
    return [
        {
            "config": json.loads(row["config_json"]),
            "validation_annual_rap": row["validation_annual_rap"],
        }
        for row in rows
    ]


def _add_variant(variants: dict[str, dict[str, Any]], config: dict[str, Any]) -> None:
    variants[hashlib.sha256(_json(config).encode("utf-8")).hexdigest()] = config


def _e2_variants(base: dict[str, Any]) -> list[dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    _add_variant(variants, dict(base))

    for mode in ("none", "rolling_zscore", "expanding_zscore", "rolling_robust"):
        cfg = {**base, "preprocessing_mode": mode}
        _add_variant(variants, cfg)
    for history in (24, 72, 168, 336, 720, 2160, 4320):
        cfg = {
            **base,
            "preprocessing_mode": "rolling_zscore",
            "scaling_history_hours": history,
        }
        _add_variant(variants, cfg)
    for clip in (None, 3, 5, 10, 20):
        cfg = {**base, "clip_value": clip}
        _add_variant(variants, cfg)
    for context_hours in (24, 72, 168, 336, 720, 2160):
        for representation in ("last", "summary", "sparse_lags"):
            cfg = {
                **base,
                "context_hours": context_hours,
                "context_representation": representation,
            }
            _add_variant(variants, cfg)
    for method in ("rank_ic_topk", "mutual_info_topk"):
        for budget in (8, 12, 16, 24, 32, 48, 64, 96, 128):
            cfg = {
                **base,
                "feature_selection_method": method,
                "feature_budget": budget,
            }
            _add_variant(variants, cfg)
    if str(base.get("external_context_bundle") or "none") != "none":
        for lag in (0, 24, 168, 744):
            cfg = {**base, "external_context_lag_hours": lag}
            _add_variant(variants, cfg)
    return list(variants.values())


def promote_e2(conn, *, materialized_plan: Path | None = None) -> dict[str, int | str]:
    existing = conn.execute("SELECT COUNT(*) FROM jobs WHERE stage=?", (E2_STAGE,)).fetchone()[0]
    if existing:
        return {"status": "already_enqueued", "jobs": int(existing)}
    if not _finished(conn, UPSTREAM_STAGES):
        return {"status": "waiting_for_e1", "jobs": 0}
    contracts = _top_e1_contracts(conn)
    if not contracts:
        return {"status": "no_completed_e1_contracts", "jobs": 0}

    campaign_row = conn.execute(
        "SELECT campaign_id, plan_json FROM campaigns ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if campaign_row is None:
        raise RuntimeError("no campaign found")
    plan = json.loads(campaign_row["plan_json"])
    jobs = list(plan.get("jobs") or [])
    added = []
    for contract in contracts:
        for config in _e2_variants(contract["config"]):
            added.append(
                {
                    "job_id": _job_id(config),
                    "stage": E2_STAGE,
                    "task_type": "feature_proxy_screen",
                    "priority": 200,
                    "max_attempts": 3,
                    "config": config,
                }
            )
    plan["jobs"] = jobs + added
    plan.setdefault("materialized_stage_counts", {})[E2_STAGE] = len(added)
    result = enqueue_plan(conn, plan)
    if materialized_plan:
        materialized_plan.parent.mkdir(parents=True, exist_ok=True)
        materialized_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "enqueued",
        "jobs": len(added),
        "inserted": result["inserted"],
        "existing": result["existing"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--materialized-plan")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    conn = connect(args.db)
    init_db(conn)
    output = Path(args.materialized_plan) if args.materialized_plan else None
    while True:
        result = promote_e2(conn, materialized_plan=output)
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.once:
            return
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
