#!/usr/bin/env python3
"""Promote completed evidence stages into bounded downstream sweeps."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Any

from project3_evidence_pool import connect, enqueue_plan, init_db


UPSTREAM_STAGES = ("E1_BASE_SOURCE_SCREEN", "E1_EXTERNAL_SOURCE_SCREEN")
E2_STAGE = "E2_PREPROCESSING_CONTEXT"
E2_INTERACTION_STAGE = "E2_INTERACTION_CONFIRMATION"


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


def _top_contracts(
    conn,
    stages: tuple[str, ...],
    *,
    per_cell: int = 1,
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in stages)
    rows = conn.execute(
        f"""
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
            WHERE stage IN ({placeholders})
              AND status='completed'
              AND validation_annual_rap IS NOT NULL
        )
        SELECT config_json, validation_annual_rap
        FROM ranked WHERE rank_in_cell <= ?
        ORDER BY json_extract(config_json, '$.asset'), json_extract(config_json, '$.timeframe')
        """,
        (*stages, int(per_cell)),
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

    for mode in (
        "none",
        "rolling_zscore",
        "expanding_zscore",
        "rolling_robust",
        "rolling_rank_gaussian",
    ):
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
        for representation in (
            "last",
            "summary",
            "sparse_lags",
            "raw_sequence",
            "multiscale_sequence",
        ):
            cfg = {
                **base,
                "context_hours": context_hours,
                "context_representation": representation,
            }
            _add_variant(variants, cfg)
    for method in (
        "rank_ic_topk",
        "mutual_info_topk",
        "redundancy_stability_topk",
        "regime_conditioned_topk",
        "sparse_mask",
    ):
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
        for policy in (
            "causal_ffill",
            "causal_ffill_plus_missing_indicator",
            "train_median",
        ):
            cfg = {**base, "missing_value_policy": policy}
            _add_variant(variants, cfg)
        for staleness in (24, 168, 744, 2160):
            cfg = {**base, "max_staleness_hours": staleness}
            _add_variant(variants, cfg)
    for reference_set in (
        "none",
        "btc_eth",
        "crypto_leaders",
        "fx_leaders",
        "portfolio_candidates",
    ):
        cfg = {**base, "cross_asset_reference_set": reference_set}
        _add_variant(variants, cfg)
    for volatility_window_hours in (24, 72, 168, 336):
        cfg = {
            **base,
            "cross_asset_reference_set": "portfolio_candidates",
            "cross_asset_volatility_window_hours": volatility_window_hours,
        }
        _add_variant(variants, cfg)
    timeframe_hours = {"15m": 0.25, "1h": 1.0, "4h": 4.0}[str(base["timeframe"])]
    for target_hours in (1, 4, 12, 24, 72, 168):
        if target_hours < timeframe_hours:
            continue
        cfg = {**base, "target_horizon_hours": target_hours}
        _add_variant(variants, cfg)
    for definition in (
        "forward_return",
        "cost_adjusted_forward_return",
        "triple_barrier",
        "future_rap",
    ):
        cfg = {**base, "target_definition": definition}
        _add_variant(variants, cfg)
    for threshold in (0.85, 0.90, 0.95, 0.98):
        cfg = {
            **base,
            "feature_selection_method": "redundancy_stability_topk",
            "redundancy_threshold": threshold,
        }
        _add_variant(variants, cfg)
    for folds in (3, 5, 8):
        cfg = {
            **base,
            "feature_selection_method": "redundancy_stability_topk",
            "stability_folds": folds,
        }
        _add_variant(variants, cfg)
    for enabled in (False, True):
        cfg = {**base, "log_transform_positive_features": enabled}
        _add_variant(variants, cfg)
    for signal in (
        "close",
        "log_close",
        "return",
        "log_return",
        "volume",
        "volatility",
        "spread_proxy",
    ):
        cfg = {
            **base,
            "wavelet_family": "causal_multiscale_rolling",
            "transform_input_signal": signal,
        }
        _add_variant(variants, cfg)
    for levels in ([1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]):
        cfg = {
            **base,
            "wavelet_family": "causal_multiscale_rolling",
            "wavelet_levels": levels,
        }
        _add_variant(variants, cfg)
    for base_scale_hours in (4, 8, 24):
        cfg = {
            **base,
            "wavelet_family": "causal_multiscale_rolling",
            "wavelet_base_scale_hours": base_scale_hours,
        }
        _add_variant(variants, cfg)
    for volatility_window_hours in (24, 72, 168, 336):
        cfg = {
            **base,
            "wavelet_family": "causal_multiscale_rolling",
            "transform_input_signal": "volatility",
            "transform_volatility_window_hours": volatility_window_hours,
        }
        _add_variant(variants, cfg)
    for detrend_window_hours in (24, 72, 168, 336):
        cfg = {
            **base,
            "hilbert_input_signal": "detrended_close",
            "transform_detrend_window_hours": detrend_window_hours,
        }
        _add_variant(variants, cfg)
    for signal in ("detrended_close", "log_return", "volatility"):
        for window_hours in (72, 168, 336, 720):
            cfg = {
                **base,
                "hilbert_input_signal": signal,
                "hilbert_window_hours": window_hours,
            }
            _add_variant(variants, cfg)
    for signal in ("log_return", "volatility", "volume_change"):
        for window_hours in (72, 168, 336, 720):
            cfg = {
                **base,
                "multitaper_input_signal": signal,
                "multitaper_window_hours": window_hours,
            }
            _add_variant(variants, cfg)
    for time_bandwidth, taper_count in ((2.5, 3), (3.5, 5), (4.5, 7)):
        cfg = {
            **base,
            "multitaper_input_signal": "log_return",
            "multitaper_time_bandwidth": time_bandwidth,
            "multitaper_taper_count": taper_count,
        }
        _add_variant(variants, cfg)
    for sample_interval_hours in (6, 24, 72):
        cfg = {
            **base,
            "hilbert_input_signal": "log_return",
            "transform_sample_interval_hours": sample_interval_hours,
        }
        _add_variant(variants, cfg)
    for signal in ("detrended_close", "log_return"):
        cfg = {**base, "emd_input_signal": signal}
        _add_variant(variants, cfg)
    for windows in ([8, 32, 128], [24, 168, 720], [72, 336, 2160]):
        cfg = {
            **base,
            "emd_input_signal": "log_return",
            "emd_window_hours": windows,
        }
        _add_variant(variants, cfg)
    for signal in ("close", "log_close"):
        for d in (0.2, 0.4, 0.6, 0.8):
            cfg = {
                **base,
                "fracdiff_input_signal": signal,
                "fracdiff_d": d,
            }
            _add_variant(variants, cfg)
    for history_hours in (168, 720, 2160):
        cfg = {
            **base,
            "fracdiff_input_signal": "log_close",
            "fracdiff_max_history_hours": history_hours,
        }
        _add_variant(variants, cfg)
    return list(variants.values())


def _diff(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if base.get(key) != value
        and key
        not in {
            "input_data_file",
            "data_root",
            "asset",
            "timeframe",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        }
    }


def _interaction_variants(
    base: dict[str, Any],
    ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    _add_variant(variants, dict(base))
    patches = [_diff(base, item["config"]) for item in ranked]
    patches = [patch for patch in patches if patch]
    for patch in patches:
        _add_variant(variants, {**base, **patch})
    for size in (2, 3):
        for combination in itertools.combinations(patches[:5], size):
            merged = dict(base)
            for patch in combination:
                merged.update(patch)
            _add_variant(variants, merged)
    return list(variants.values())


def promote_e2(conn, *, materialized_plan: Path | None = None) -> dict[str, int | str]:
    existing = conn.execute("SELECT COUNT(*) FROM jobs WHERE stage=?", (E2_STAGE,)).fetchone()[0]
    if existing:
        return {"status": "already_enqueued", "jobs": int(existing)}
    if not _finished(conn, UPSTREAM_STAGES):
        return {"status": "waiting_for_e1", "jobs": 0}
    contracts = _top_contracts(conn, UPSTREAM_STAGES)
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


def promote_e2_interactions(
    conn,
    *,
    materialized_plan: Path | None = None,
) -> dict[str, int | str]:
    existing = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE stage=?",
        (E2_INTERACTION_STAGE,),
    ).fetchone()[0]
    if existing:
        return {"status": "already_enqueued", "jobs": int(existing)}
    if not _finished(conn, (E2_STAGE,)):
        return {"status": "waiting_for_e2", "jobs": 0}
    ranked = _top_contracts(conn, (E2_STAGE,), per_cell=5)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in ranked:
        config = item["config"]
        grouped.setdefault((str(config["asset"]), str(config["timeframe"])), []).append(item)
    if not grouped:
        return {"status": "no_completed_e2_contracts", "jobs": 0}

    campaign_row = conn.execute(
        "SELECT campaign_id, plan_json FROM campaigns ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if campaign_row is None:
        raise RuntimeError("no campaign found")
    plan = json.loads(campaign_row["plan_json"])
    added = []
    for items in grouped.values():
        base = dict(items[0]["config"])
        for config in _interaction_variants(base, items):
            added.append(
                {
                    "job_id": _job_id(config).replace("e2__", "e2i__", 1),
                    "stage": E2_INTERACTION_STAGE,
                    "task_type": "feature_proxy_screen",
                    "priority": 300,
                    "max_attempts": 3,
                    "config": config,
                }
            )
    plan["jobs"] = list(plan.get("jobs") or []) + added
    plan.setdefault("materialized_stage_counts", {})[E2_INTERACTION_STAGE] = len(added)
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


def promote_all(conn, *, materialized_plan: Path | None = None) -> dict[str, Any]:
    e2 = promote_e2(conn, materialized_plan=materialized_plan)
    e2_interactions = promote_e2_interactions(conn, materialized_plan=materialized_plan)
    return {"e2": e2, "e2_interactions": e2_interactions}


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
        result = promote_all(conn, materialized_plan=output)
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.once:
            return
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
