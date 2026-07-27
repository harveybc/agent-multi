#!/usr/bin/env python3
"""Materialize full-genome per-asset configs from completed E4 evidence."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


CORE_COLUMNS = {"DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"}
SCHEMA_VERSION = "agent_multi.project3_full_genome_config.v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _feature_group(column: str) -> str:
    name = column.lower()
    if name.startswith(("return_", "log_return_", "roc_", "mom_")):
        return "returns_momentum"
    if name.startswith(
        (
            "sma_",
            "ema_",
            "close_sma_",
            "macd",
            "trend_",
            "ema_cross_",
            "bb_",
        )
    ):
        return "trend"
    if name.startswith(
        (
            "rsi_",
            "stoch_",
            "williams_",
            "cci_",
            "mfi_",
        )
    ):
        return "oscillators"
    if name.startswith(
        (
            "atr_",
            "natr_",
            "hist_vol_",
            "realized_var_",
            "vol_regime_",
        )
    ):
        return "volatility"
    if name.startswith(("obv", "volume_", "vwap_")):
        return "volume"
    if name.startswith(
        (
            "statistical__",
            "roll_",
            "autocorr_",
            "sqret_",
            "hurst_",
            "zscore_",
        )
    ):
        return "statistics"
    if name.startswith("wavelet_"):
        return "wavelet"
    if name.startswith("ht_"):
        return "hilbert"
    if name.startswith("mt_"):
        return "multitaper"
    if name.startswith("emd_"):
        return "emd"
    if name.startswith("fracdiff_"):
        return "fracdiff"
    if name.startswith(("cnn_", "lstm_", "latent_", "learned_cnn_", "learned_lstm_")):
        return "learned"
    if name.startswith("sota_pair_"):
        return "cross_asset"
    if name.startswith("sota_hmm_"):
        return "regime"
    if name.startswith(
        (
            "sota_intrabar_",
            "intrabar_count",
            "sota_realized_",
            "sota_bipower_",
            "sota_jump_",
            "sota_tripower_",
        )
    ):
        return "microstructure"
    if name.startswith("sota_funding_"):
        return "derivatives"
    if name.startswith("event_"):
        return "economic_events"
    if name.startswith("external__"):
        if "economic_calendar" in name:
            return "economic_events"
        if any(token in name for token in ("fred", "oecd", "world_bank", "ecb", "boj")):
            return "macro"
        if any(token in name for token in ("onchain", "coinmetrics", "glassnode")):
            return "onchain"
        if "funding" in name:
            return "derivatives"
        return "external_market"
    return "other"


def _headers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.reader(handle))
    return [str(value) for value in row]


def _training_rows(
    path: Path,
    *,
    start: str,
    end: str,
) -> int:
    import pandas as pd

    dates = pd.read_csv(path, usecols=["DATE_TIME"])
    values = pd.to_datetime(dates["DATE_TIME"], utc=True, errors="raise")
    return int(
        (
            (values >= pd.Timestamp(start, tz="UTC"))
            & (values <= pd.Timestamp(end, tz="UTC"))
        ).sum()
    )


def _e4_candidates(
    db_path: Path,
    *,
    asset: str,
    timeframe: str,
) -> list[dict[str, Any]]:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    candidates: list[dict[str, Any]] = []
    with sqlite3.connect(uri, uri=True) as connection:
        for external_id, config_json, result_json in connection.execute(
            """
            SELECT external_id, config_json, result_json
            FROM jobs
            WHERE stage = 'E4_ASSET_POLICY_TRAINING'
              AND status = 'completed'
            """
        ):
            config = json.loads(config_json)
            if (
                str(config.get("asset", "")).upper() != asset.upper()
                or str(config.get("timeframe", "")).lower() != timeframe.lower()
            ):
                continue
            result = json.loads(result_json)
            validation = dict((result.get("summary") or {}).get("validation") or {})
            candidates.append(
                {
                    "external_id": external_id,
                    "config": config,
                    "result": result,
                    "validation_annual_rap": float(
                        validation.get("annual_rap", float("-inf"))
                    ),
                }
            )
    if not candidates:
        raise ValueError(f"no completed E4 evidence for {asset}@{timeframe}")
    return sorted(
        candidates,
        key=lambda item: (
            item["validation_annual_rap"],
            -int(item["config"].get("training_seed") or 0),
        ),
        reverse=True,
    )


def _choice_gene(
    name: str,
    choices: list[Any],
    *,
    target: str | None = None,
    choice_patches: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "kind": "categorical",
        "choices": choices,
    }
    if target:
        result["target"] = target
    if choice_patches:
        result["choice_patches"] = choice_patches
    return result


def _mixed_genome(
    *,
    timeframe_hours: int,
    feature_groups: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context_hours = [24, 72, 168, 336, 720]
    context_patches = {
        str(hours): {
            "window_size": max(2, int(round(hours / timeframe_hours))),
        }
        for hours in context_hours
    }
    genes: list[dict[str, Any]] = [
        _choice_gene(
            "preprocessing_mode",
            ["none", "rolling_zscore", "expanding_zscore"],
            target="feature_scaling",
        ),
        _choice_gene(
            "scaling_history_bars",
            [24, 42, 84, 168, 336],
            target="feature_scaling_window",
        ),
        _choice_gene(
            "feature_clip",
            [3.0, 5.0, 10.0, 20.0],
            target="feature_clip",
        ),
        _choice_gene(
            "context_hours",
            context_hours,
            choice_patches=context_patches,
        ),
        _choice_gene(
            "net_architecture",
            ["128x128", "256x256", "512x256", "256x256x128"],
            choice_patches={
                "128x128": {"net_arch": [128, 128]},
                "256x256": {"net_arch": [256, 256]},
                "512x256": {"net_arch": [512, 256]},
                "256x256x128": {"net_arch": [256, 256, 128]},
            },
        ),
        {
            "name": "learning_rate_gene",
            "kind": "log_float",
            "low": 1e-5,
            "high": 1e-3,
            "target": "learning_rate",
        },
        _choice_gene(
            "batch_size_gene",
            [128, 256, 512],
            target="batch_size",
        ),
        _choice_gene(
            "buffer_size_gene",
            [100_000, 200_000, 500_000],
            target="buffer_size",
        ),
        _choice_gene(
            "learning_starts_gene",
            [2_000, 5_000, 10_000],
            target="learning_starts",
        ),
        {
            "name": "gamma_gene",
            "kind": "float",
            "low": 0.94,
            "high": 0.9995,
            "target": "gamma",
        },
        {
            "name": "tau_gene",
            "kind": "log_float",
            "low": 0.0005,
            "high": 0.02,
            "target": "tau",
        },
        _choice_gene(
            "train_freq_gene",
            [1, 2, 4, 8],
            target="train_freq",
        ),
        _choice_gene(
            "gradient_steps_gene",
            [1, 2, 4, 8],
            target="gradient_steps",
        ),
        _choice_gene(
            "entropy_gene",
            ["auto", 0.01, 0.05, 0.1],
            target="ent_coef",
        ),
        {
            "name": "action_threshold_gene",
            "kind": "float",
            "low": 0.03,
            "high": 0.40,
            "target": "continuous_action_threshold",
        },
        {
            "name": "relative_volume_gene",
            "kind": "float",
            "low": 0.01,
            "high": 0.25,
            "target": "rel_volume",
        },
        {
            "name": "stop_loss_atr_gene",
            "kind": "float",
            "low": 0.75,
            "high": 5.0,
            "target": "k_sl",
        },
        {
            "name": "take_profit_atr_gene",
            "kind": "float",
            "low": 1.0,
            "high": 8.0,
            "target": "k_tp",
        },
    ]
    for group in sorted(feature_groups):
        genes.append(
            {
                "name": f"feature_group__{group}",
                "kind": "boolean",
                "target": f"_feature_group_{group}",
            }
        )
    return genes, context_patches


def materialize_config(
    *,
    db_path: Path,
    asset: str,
    timeframe: str,
    output: Path,
    data_root: Path,
    artifact_root: str,
    smoke: bool = False,
) -> dict[str, Any]:
    selected = _e4_candidates(db_path, asset=asset, timeframe=timeframe)[0]
    evidence_config = selected["config"]
    evidence_result = selected["result"]
    canonical = copy.deepcopy(evidence_result["resolved_config"])

    wide_path = (
        data_root
        / "experiments"
        / "stage_a_screening"
        / "inputs"
        / asset.lower()
        / timeframe.lower()
        / "kitchen_sink_guarded"
        / "train.csv"
    )
    if not wide_path.is_file():
        raise FileNotFoundError(
            "full-genome optimization requires the kitchen_sink_guarded input; "
            f"expected {wide_path}. Refusing to fall back to the narrower E4 input "
            f"{evidence_config.get('input_data_file')!r}."
        )
    columns = [
        value for value in _headers(wide_path) if value not in CORE_COLUMNS
    ]
    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(_feature_group(column), []).append(column)

    timeframe_hours = int(str(timeframe).lower().removesuffix("h"))
    epoch_timesteps = _training_rows(
        wide_path,
        start=str(evidence_config["train_start"]),
        end=str(evidence_config["train_end"]),
    )
    if epoch_timesteps < 100:
        raise ValueError("full-genome training split is unexpectedly small")
    genes, _ = _mixed_genome(
        timeframe_hours=timeframe_hours,
        feature_groups=groups,
    )

    selected_groups = {
        _feature_group(str(value))
        for value in evidence_result.get("selected_features") or []
    }
    if not selected_groups:
        selected_groups = {"trend"}
    initial = {
        "preprocessing_mode": str(
            evidence_config.get("preprocessing_mode") or "rolling_zscore"
        ),
        "scaling_history_bars": max(
            2,
            int(
                round(
                    float(evidence_config.get("scaling_history_hours") or 168)
                    / timeframe_hours
                )
            ),
        ),
        "feature_clip": float(evidence_config.get("clip_value") or 10.0),
        "context_hours": int(evidence_config.get("context_hours") or 168),
        "net_architecture": "256x256",
        "learning_rate_gene": float(evidence_config["learning_rate"]),
        "batch_size_gene": int(evidence_config["batch_size"]),
        "buffer_size_gene": 200_000,
        "learning_starts_gene": int(
            evidence_config.get("learning_starts") or 5_000
        ),
        "gamma_gene": float(evidence_config["gamma"]),
        "tau_gene": float(evidence_config["tau"]),
        "train_freq_gene": int(evidence_config["train_freq"]),
        "gradient_steps_gene": int(evidence_config["gradient_steps"]),
        "entropy_gene": evidence_config.get("ent_coef") or "auto",
        "action_threshold_gene": float(
            canonical["asset_policy"]["continuous_action_threshold"]
        ),
        "relative_volume_gene": float(evidence_config.get("rel_volume") or 0.05),
        "stop_loss_atr_gene": float(evidence_config.get("k_sl") or 2.0),
        "take_profit_atr_gene": float(evidence_config.get("k_tp") or 3.0),
    }
    for group in sorted(groups):
        initial[f"feature_group__{group}"] = group in selected_groups

    name = f"phase_1_asset_policy_{asset.lower()}_{timeframe.lower()}_full_genome_v1"
    canonical["experiment"].update(
        {
            "name": name + ("_smoke" if smoke else ""),
            "mode": "train",
            "quiet_mode": False,
            "role": (
                "full_genome_contract_smoke"
                if smoke
                else "full_genome_asset_policy_optimization"
            ),
            "phase": "D1_FULL_GENOME_PER_ASSET",
            "scientific_scope": (
                "non_promotable_smoke"
                if smoke
                else "full_fidelity_per_asset_optimization"
            ),
        }
    )
    canonical["data"].update(
        {
            "input_data_file": str(wide_path),
            "dataset_manifest_file": str(
                wide_path.with_name("train_metadata.json")
            ),
            "features_preset": "kitchen_sink_guarded",
            "data_profile": "kitchen_sink_guarded",
            "feature_list": columns,
        }
    )
    canonical["environment"].update(
        {
            "preprocessor_plugin": "feature_window_preprocessor",
            "feature_columns": columns,
            "feature_binary_columns": [
                value for value in columns if value.startswith("vol_regime_")
            ],
            "feature_scaling": "rolling_zscore",
            "feature_scaling_window": max(24, 168 // timeframe_hours),
            "feature_clip": 10.0,
            "window_size": max(2, 168 // timeframe_hours),
            "include_price_window": False,
            "include_agent_state": True,
            "require_feature_aware_preprocessor": True,
            "precomputed_causal_features": False,
        }
    )
    canonical["asset_policy"].update(
        {
            "window_size": max(2, 168 // timeframe_hours),
            "project3_strict": True,
        }
    )

    max_epochs = 4 if smoke else 2_000
    patience = 1 if smoke else 60
    patience_start = 1 if smoke else 40
    canonical["training"].update(
        {
            "epoch_timesteps": epoch_timesteps,
            "max_epochs": max_epochs,
            "l1_patience": patience,
            "l1_patience_start_epoch": patience_start,
            "l1_min_delta": 0.00001,
            "l1_min_checkpoint_timesteps": 5_001,
            "evaluate_test_split": False,
            "selection_metric": "risk_adjusted_return",
            "risk_penalty_lambda": 1.0,
            "device": "cuda",
        }
    )
    optimization = canonical["optimization"]
    optimization.pop("hyperparameter_bounds", None)
    optimization.update(
        {
            "enabled": True,
            "plugin": "project3_full_genome_optimizer",
            "metric": "train_validation_l1_score",
            "higher_is_better": True,
            "ga_fitness_split": "train",
            "ga_population": 1 if smoke else 20,
            "ga_generations": 0 if smoke else 24,
            "ga_cxpb": 0.5,
            "ga_mutpb": 0.25,
            "ga_eval_timesteps": epoch_timesteps * max_epochs,
            "ga_seed": 2703,
            "optimization_patience": 5,
            "optimization_resume": False,
            "optimization_capture_model_artifact": True,
            "optimization_require_model_artifact": True,
            "optimization_run_final_pipeline": False,
            "optimization_reject_action_collapse": True,
            "optimization_stages": (
                [
                    {
                        "name": "smoke",
                        "params": "all",
                        "generations": 0,
                        "patience": 1,
                    }
                ]
                if smoke
                else [
                    {
                        "name": "data_observation",
                        "params": [
                            "preprocessing_mode",
                            "scaling_history_bars",
                            "feature_clip",
                            "context_hours",
                            *[
                                f"feature_group__{group}"
                                for group in sorted(groups)
                            ],
                        ],
                        "generations": 6,
                        "patience": 4,
                    },
                    {
                        "name": "model_training",
                        "params": [
                            "net_architecture",
                            "learning_rate_gene",
                            "batch_size_gene",
                            "buffer_size_gene",
                            "learning_starts_gene",
                            "gamma_gene",
                            "tau_gene",
                            "train_freq_gene",
                            "gradient_steps_gene",
                            "entropy_gene",
                        ],
                        "generations": 8,
                        "patience": 5,
                    },
                    {
                        "name": "execution_risk",
                        "params": [
                            "action_threshold_gene",
                            "relative_volume_gene",
                            "stop_loss_atr_gene",
                            "take_profit_atr_gene",
                        ],
                        "generations": 4,
                        "patience": 3,
                    },
                    {
                        "name": "joint_refinement",
                        "params": "all",
                        "generations": 6,
                        "patience": 5,
                    },
                ]
            ),
            "mixed_genome_schema": genes,
            "mixed_genome_feature_groups": groups,
            "mixed_genome_required_feature_group": "trend",
            "mixed_genome_max_observation_elements": 4_096,
            "mixed_genome_max_replay_observation_values": 300_000_000,
            "mixed_genome_repair_rules": [],
            "initial_candidate_decoded": initial,
            "e4_baseline_job_id": selected["external_id"],
            "e4_baseline_validation_annual_rap": selected[
                "validation_annual_rap"
            ],
            "full_fidelity_protocol": not smoke,
        }
    )
    root = (
        f"{artifact_root}/full_genome/{asset.lower()}_{timeframe.lower()}"
        + ("/smoke" if smoke else "")
    )
    canonical["artifacts"].update(
        {
            "artifact_root": root,
            "save_model": f"{root}/final_policy.zip",
            "results_file": f"{root}/results.json",
            "resolved_config_file": f"{root}/resolved_config.json",
            "config_manifest_file": f"{root}/config_manifest.json",
            "optimizer_output_file": f"{root}/optimizer_output.json",
            "return_trace_dir": f"{root}/return_traces",
        }
    )
    optimization.update(
        {
            "optimization_statistics": f"{root}/optimization_stats.json",
            "optimization_parameters_file": (
                f"{root}/optimization_parameters.json"
            ),
            "optimization_resume_file": f"{root}/optimization_resume.json",
            "optimization_candidate_history": f"{root}/candidate_history.csv",
            "optimization_champion_model_file": f"{root}/champion_policy.zip",
        }
    )
    canonical["walk_forward"].update(
        {
            "enabled": False,
            "weekly_retraining": False,
            "selection_uses_test": False,
        }
    )
    canonical["deployment"].update(
        {
            "channel": "research_component",
            "lifecycle": "not_for_live_orders",
            "promotion_gate": (
                "non_promotable_smoke"
                if smoke
                else "full_genome_three_seed_confirmation_required"
            ),
        }
    )
    canonical.setdefault("code", {}).setdefault("contract_versions", {})[
        "full_genome_config"
    ] = SCHEMA_VERSION
    _atomic_json(output, canonical)
    return {
        "output": str(output),
        "asset": asset.upper(),
        "timeframe": timeframe,
        "feature_count": len(columns),
        "feature_groups": {key: len(value) for key, value in groups.items()},
        "epoch_timesteps": epoch_timesteps,
        "max_epochs": max_epochs,
        "l1_patience": patience,
        "e4_baseline_job_id": selected["external_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--artifact-root",
        default="${ARTIFACT_ROOT}",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = materialize_config(
        db_path=args.db,
        asset=args.asset,
        timeframe=args.timeframe,
        output=args.output,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
