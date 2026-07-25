#!/usr/bin/env python3
"""Build the first hierarchical Project 3 evidence-recovery campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("/home/harveybc/Documents/GitHub/financial-data")
PRESETS = {
    "baseline_12",
    "tech_full",
    "tech_stat",
    "tech_stat_decomp",
    "sota_low_cost",
    "learned_lstm",
    "learned_cnn",
    "kitchen_sink_guarded",
}
CRYPTO_SUFFIXES = ("usdt", "usdt_perp")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _job_id(stage: str, config: dict[str, Any]) -> str:
    digest = hashlib.sha256(_json(config).encode("utf-8")).hexdigest()[:16]
    asset = str(config["asset"])
    timeframe = str(config["timeframe"])
    preset = str(config.get("base_feature_bundle") or "audit")
    return f"{stage.lower()}__{asset}__{timeframe}__{preset}__{digest}"


def _base_config(
    *,
    data_root: Path,
    asset: str,
    timeframe: str,
    preset: str,
    input_file: Path,
) -> dict[str, Any]:
    target_hours = {"15m": 1, "1h": 4, "4h": 24}.get(timeframe, 24)
    return {
        "asset": asset,
        "timeframe": timeframe,
        "base_feature_bundle": preset,
        "input_data_file": str(input_file),
        "data_root": str(data_root),
        "train_start": "2021-01-01T00:00:00",
        "train_end": "2021-12-31T23:59:59",
        "validation_start": "2022-01-01T00:00:00",
        "validation_end": "2022-12-31T23:59:59",
        "test_start": "2023-01-01T00:00:00",
        "test_end": "2023-12-31T23:59:59",
        "target_horizon_hours": target_hours,
        "transaction_cost_fraction": 0.0005,
        "risk_penalty_lambda": 1.0,
        "metric_schema": "project3.evidence.metrics.v2",
    }


def _job(stage: str, task_type: str, priority: int, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": _job_id(stage, config),
        "stage": stage,
        "task_type": task_type,
        "priority": priority,
        "max_attempts": 3,
        "config": config,
    }


def _discover_inputs(data_root: Path) -> list[tuple[str, str, str, Path]]:
    root = data_root / "experiments" / "stage_a_screening" / "inputs"
    rows = []
    for path in sorted(root.glob("*/*/*/train.csv")):
        preset = path.parent.name
        timeframe = path.parent.parent.name
        asset = path.parent.parent.parent.name
        if preset in PRESETS and timeframe in {"15m", "1h", "4h"}:
            rows.append((asset, timeframe, preset, path))
    return rows


def build_plan(data_root: Path, campaign_id: str) -> dict[str, Any]:
    inputs = _discover_inputs(data_root)
    jobs: list[dict[str, Any]] = []

    # E0 proves file integrity and makes missing external coverage visible.
    for asset, timeframe, preset, path in inputs:
        config = {
            **_base_config(
                data_root=data_root,
                asset=asset,
                timeframe=timeframe,
                preset=preset,
                input_file=path,
            ),
            "external_context_bundle": "none",
            "external_context_lag_hours": 0,
        }
        jobs.append(_job("E0_DATA_CONTRACT", "data_contract_audit", 10, config))

    baseline_by_asset_tf = {
        (asset, timeframe): path
        for asset, timeframe, preset, path in inputs
        if preset == "baseline_12"
    }
    for (asset, timeframe), path in sorted(baseline_by_asset_tf.items()):
        bundles = ["macro_core", "market_core", "macro_market"]
        if asset.endswith(CRYPTO_SUFFIXES):
            bundles.extend(["onchain_asset", "funding_core", "crypto_context", "all_non_cryptoquant"])
        else:
            bundles.append("economic_calendar")
        for bundle in bundles:
            config = {
                **_base_config(
                    data_root=data_root,
                    asset=asset,
                    timeframe=timeframe,
                    preset="baseline_12",
                    input_file=path,
                ),
                "external_context_bundle": bundle,
                "external_context_lag_hours": 168 if "macro" in bundle or bundle == "all_non_cryptoquant" else 24,
            }
            jobs.append(_job("E0_EXTERNAL_COVERAGE", "data_contract_audit", 20, config))

    # E1 isolates base bundles and external source bundles with one conservative
    # preprocessing/context contract. No validation or test rows select features.
    for asset, timeframe, preset, path in inputs:
        for method in ("rank_ic_topk", "mutual_info_topk"):
            config = {
                **_base_config(
                    data_root=data_root,
                    asset=asset,
                    timeframe=timeframe,
                    preset=preset,
                    input_file=path,
                ),
                "external_context_bundle": "none",
                "external_context_lag_hours": 0,
                "feature_selection_method": method,
                "feature_budget": 32,
                "preprocessing_mode": "rolling_zscore",
                "scaling_history_hours": 168,
                "clip_value": 10,
                "context_hours": 168,
                "context_representation": "summary",
                "ridge_alpha": 1.0,
                "action_threshold_quantile": 0.65,
                "minimum_split_rows": 200,
            }
            jobs.append(_job("E1_BASE_SOURCE_SCREEN", "feature_proxy_screen", 100, config))

    for (asset, timeframe), path in sorted(baseline_by_asset_tf.items()):
        bundles = ["macro_core", "market_core", "macro_market"]
        if asset.endswith(CRYPTO_SUFFIXES):
            bundles.extend(["onchain_asset", "funding_core", "crypto_context", "all_non_cryptoquant"])
        for bundle in bundles:
            config = {
                **_base_config(
                    data_root=data_root,
                    asset=asset,
                    timeframe=timeframe,
                    preset="baseline_12",
                    input_file=path,
                ),
                "external_context_bundle": bundle,
                "external_context_lag_hours": 168 if "macro" in bundle or bundle == "all_non_cryptoquant" else 24,
                "feature_selection_method": "mutual_info_topk",
                "feature_budget": 32,
                "preprocessing_mode": "rolling_zscore",
                "scaling_history_hours": 168,
                "clip_value": 10,
                "context_hours": 168,
                "context_representation": "summary",
                "ridge_alpha": 1.0,
                "action_threshold_quantile": 0.65,
                "minimum_split_rows": 200,
            }
            jobs.append(_job("E1_EXTERNAL_SOURCE_SCREEN", "feature_proxy_screen", 110, config))

    return {
        "schema_version": "project3.evidence.plan.v1",
        "campaign_id": campaign_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameter_registry": "examples/config/evidence_sweep/project3_parameter_registry_v1.json",
        "metric_schema": "project3.evidence.metrics.v2",
        "selection_contract": {
            "train_period": "2021",
            "validation_period": "2022",
            "test_period": "2023",
            "feature_selection_uses": ["train"],
            "model_selection_uses": ["validation"],
            "reporting_uses": ["validation", "test"],
            "positive_profit_required": False
        },
        "stage_order": [
            "E0_DATA_CONTRACT",
            "E0_EXTERNAL_COVERAGE",
            "E1_BASE_SOURCE_SCREEN",
            "E1_EXTERNAL_SOURCE_SCREEN",
            "E2_PREPROCESSING_CONTEXT",
            "E2_INTERACTION_CONFIRMATION",
            "E3_REPRESENTATION_MODEL",
            "E4_WEEKLY_RETRAINING_CONFIRMATION",
            "DOIN_L2_ASSET_OPTIMIZATION",
            "PORTFOLIO_OPTIMIZATION"
        ],
        "future_stage_templates": {
            "E2_PREPROCESSING_CONTEXT": {
                "source": "top robust E1 contracts per asset/timeframe and cross-asset global effects",
                "parameters": [
                    "preprocessing.mode",
                    "preprocessing.scaling_history_hours",
                    "preprocessing.clip_value",
                    "observation.context_hours",
                    "observation.context_representation",
                    "selection.feature_budget",
                    "selection.method",
                    "data.external_context_lag_hours",
                    "features.cross_asset_volatility_window_hours",
                    "features.transform_input_signal",
                    "features.transform_volatility_window_hours",
                    "features.transform_detrend_window_hours",
                    "features.transform_sample_interval_hours",
                    "features.wavelet_family",
                    "features.wavelet_base_scale_hours",
                    "features.wavelet_levels",
                    "features.hilbert_input_signal",
                    "features.hilbert_window_hours",
                    "features.multitaper_input_signal",
                    "features.multitaper_window_hours",
                    "features.multitaper_time_bandwidth",
                    "features.multitaper_taper_count",
                    "features.emd_input_signal",
                    "features.emd_backend",
                    "features.emd_window_hours",
                    "features.fracdiff_input_signal",
                    "features.fracdiff_d",
                    "features.fracdiff_max_history_hours"
                ],
                "promotion_rule": "rank by validation annual_rap, inspect test stability; no positive-return gate"
            },
            "E3_REPRESENTATION_MODEL": {
                "source": "top E2 contracts",
                "parameters": [
                    "representation.encoder",
                    "representation.latent_dimension",
                    "representation.pretraining_objective",
                    "agent.model_family",
                    "agent.net_arch"
                ],
                "promotion_rule": "cost-aware validation/test evidence and runtime, not cosmetic significance gates"
            },
            "E2_INTERACTION_CONFIRMATION": {
                "source": "top five E2 contracts per asset/timeframe",
                "parameters": [
                    "pairwise and three-way combinations of validation-ranked E2 changes"
                ],
                "promotion_rule": "validation annual_rap ranking under one execution protocol hash; no positive-return gate"
            },
            "E4_WEEKLY_RETRAINING_CONFIRMATION": {
                "source": "one selected contract per asset/timeframe",
                "parameters": [
                    "evaluation.weekly_retraining",
                    "evaluation.retraining_history_years"
                ],
                "required_outputs": [
                    "mean_weekly_return",
                    "annualized_return",
                    "mean_weekly_rap",
                    "annual_rap",
                    "max_drawdown",
                    "champion_model_artifact",
                    "resolved_config"
                ]
            }
        },
        "jobs": jobs,
        "job_counts": {
            stage: sum(1 for job in jobs if job["stage"] == stage)
            for stage in sorted({job["stage"] for job in jobs})
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--campaign-id", default="project3-evidence-recovery-20260725-v1")
    parser.add_argument(
        "--output",
        default="examples/config/evidence_sweep/project3_evidence_recovery_campaign_v1.json",
    )
    args = parser.parse_args()
    plan = build_plan(Path(args.data_root), args.campaign_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "job_counts": plan["job_counts"], "jobs": len(plan["jobs"])}, indent=2))


if __name__ == "__main__":
    main()
