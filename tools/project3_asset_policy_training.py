#!/usr/bin/env python3
"""Train one E4 SAC policy from an evidence-selected E3 data contract."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

from project3_evidence_metrics import METRIC_SCHEMA, canonical_trading_metrics
from project3_evidence_screen import (
    CORE_COLUMNS,
    _causal_scale,
    _context_features,
    _feature_columns,
    _load_base,
    _sha256,
    _timeframe_hours,
    _transform_warmup_hours,
    add_configured_transform_features,
    merge_cross_asset_context,
    merge_external_context,
    source_files,
)


E4_PROTOCOL_ID = "project3.asset_policy.sac_mlp.single_fit.v1"
E4_PROTOCOL_HASH = hashlib.sha256(
    (
        "E3-selected features and causal representation are frozen into a "
        "content-hashed policy dataset; SAC MLP uses validation-only L1 "
        "checkpoint selection; protected test is evaluated once after the "
        "checkpoint is frozen; Backtrader applies configured costs and risk"
    ).encode("utf-8")
).hexdigest()

_POLICY_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "config"
    / "phase_1_asset_policy"
    / "optimization"
    / "phase_1_asset_policy_btcusdt_1h_sac_optimization_config.json"
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ordered_hash(values: list[str]) -> str:
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _artifact_root(config: dict[str, Any]) -> Path:
    configured = config.get("e4_artifact_root")
    if configured:
        root = Path(str(configured)).expanduser()
    else:
        artifact_id = str(config.get("e4_artifact_id") or "unidentified")
        root = (
            Path.home()
            / ".local"
            / "share"
            / "agent-multi"
            / "evidence-e4"
            / artifact_id
        )
    machine_id = str(config.get("_machine_id") or "local")
    return (root / machine_id).resolve()


def _set_cuda_visibility(machine_id: str) -> None:
    """Pin the two gamma workers before torch is imported."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    if machine_id == "gamma-5070ti":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif machine_id == "gamma-5090":
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def _source_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(str(config["data_root"])).resolve()
    manifests: list[dict[str, Any]] = []
    for path in source_files(config):
        resolved = path.resolve()
        try:
            relative = str(resolved.relative_to(root))
        except ValueError:
            relative = str(resolved)
        manifests.append(
            {
                "relative_path": relative,
                "sha256": _sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return manifests


def materialize_policy_dataset(
    config: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Freeze the exact causal E3 observation columns into a policy CSV."""
    selected = [str(value) for value in config.get("upstream_selected_features") or []]
    if not selected:
        raise ValueError("E4 requires upstream_selected_features from E3")

    train_start = pd.Timestamp(str(config["train_start"]), tz="UTC")
    test_end = pd.Timestamp(str(config["test_end"]), tz="UTC")
    frame = _load_base(config)
    warmup_start = train_start - pd.Timedelta(
        hours=_transform_warmup_hours(config)
    )
    frame, source_meta = merge_external_context(
        frame,
        config,
        output_start=warmup_start,
        output_end=test_end,
    )
    frame, cross_asset_meta = merge_cross_asset_context(frame, config)
    frame, transform_meta = add_configured_transform_features(frame, config)

    available = _feature_columns(frame)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(
            "E4 selected features missing from materialized input: "
            + ", ".join(missing)
        )

    timeframe_hours = _timeframe_hours(str(config["timeframe"]))
    scaling_rows = max(
        2,
        int(round(float(config["scaling_history_hours"]) / timeframe_hours)),
    )
    feature_frame = frame[selected].copy()
    if bool(config.get("log_transform_positive_features", False)):
        for column in selected:
            values = pd.to_numeric(feature_frame[column], errors="coerce")
            if values.dropna().empty or float(values.min(skipna=True)) < 0.0:
                continue
            feature_frame[column] = np.log1p(values)
    scaled = _causal_scale(
        feature_frame,
        selected,
        mode=str(config["preprocessing_mode"]),
        window_rows=scaling_rows,
        clip=config.get("clip_value"),
    )
    context_rows = max(
        2,
        int(round(float(config["context_hours"]) / timeframe_hours)),
    )
    observations = _context_features(
        scaled,
        selected,
        representation=str(config["context_representation"]),
        context_rows=context_rows,
    )
    observation_columns = [str(value) for value in observations.columns]
    core = [
        "DATE_TIME",
        *[
            name
            for name in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")
            if name in frame.columns
        ],
    ]
    policy = pd.concat([frame[core], observations], axis=1)
    policy = policy[
        (policy["DATE_TIME"] >= train_start)
        & (policy["DATE_TIME"] <= test_end)
    ].copy()
    policy = policy.replace([float("inf"), float("-inf")], pd.NA)
    required_core = [
        value for value in ("OPEN", "HIGH", "LOW", "CLOSE") if value in policy
    ]
    policy = policy.dropna(subset=[*required_core, *observation_columns])
    if "VOLUME" in policy:
        policy["VOLUME"] = pd.to_numeric(
            policy["VOLUME"], errors="coerce"
        ).fillna(0.0)
    if policy.empty:
        raise ValueError("E4 policy dataset is empty after causal warm-up")

    minimum_rows = int(config.get("minimum_split_rows") or 200)
    split_bounds = {
        "train": (config["train_start"], config["train_end"]),
        "validation": (config["validation_start"], config["validation_end"]),
        "test": (config["test_start"], config["test_end"]),
    }
    split_rows: dict[str, int] = {}
    for split, (start, end) in split_bounds.items():
        mask = (
            (policy["DATE_TIME"] >= pd.Timestamp(str(start), tz="UTC"))
            & (policy["DATE_TIME"] <= pd.Timestamp(str(end), tz="UTC"))
        )
        split_rows[split] = int(mask.sum())
        if split_rows[split] < minimum_rows:
            raise ValueError(
                f"E4 {split} has {split_rows[split]} rows; need {minimum_rows}"
            )

    # The validation pipeline uses naive timestamps; preserve UTC wall time
    # explicitly so split comparisons cannot mix aware and naive datetimes.
    policy["DATE_TIME"] = pd.to_datetime(
        policy["DATE_TIME"], utc=True
    ).dt.tz_localize(None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    policy.to_csv(temporary, index=False)
    temporary.replace(destination)
    return {
        "path": str(destination),
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "row_count": len(policy),
        "split_rows": split_rows,
        "selected_features": selected,
        "selected_features_sha256": _ordered_hash(selected),
        "observation_columns": observation_columns,
        "observation_columns_sha256": _ordered_hash(observation_columns),
        "context_rows": context_rows,
        "source_meta": source_meta,
        "cross_asset_meta": cross_asset_meta,
        "transform_meta": transform_meta,
        "source_manifest": _source_manifest(config),
    }


def _canonical_rows(
    summary: dict[str, Any],
    *,
    risk_penalty_lambda: float,
    initial_cash: float,
    timeframe: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for split in ("validation", "test"):
        split_summary = dict((summary.get("splits") or {}).get(split) or {})
        trace_path = split_summary.get("return_trace_file")
        if not trace_path:
            raise ValueError(f"E4 {split} return trace is missing")
        trace = pd.read_csv(trace_path, usecols=["timestamp", "equity"])
        timestamps = pd.to_datetime(trace["timestamp"], utc=True, errors="raise")
        initial_timestamp = timestamps.iloc[0] - pd.Timedelta(
            hours=_timeframe_hours(timeframe)
        )
        timestamps = pd.concat(
            [pd.Series([initial_timestamp]), timestamps],
            ignore_index=True,
        )
        equity = pd.concat(
            [
                pd.Series([float(initial_cash)], dtype=float),
                pd.to_numeric(trace["equity"], errors="raise"),
            ],
            ignore_index=True,
        )
        canonical = canonical_trading_metrics(
            timestamps=timestamps,
            equity=equity,
            risk_penalty_lambda=risk_penalty_lambda,
        )
        summaries[split] = canonical["canonical"]
        for metric in canonical["metrics"]:
            rows.append({**metric, "split": split})
    validation_rap = summaries["validation"].get("annual_rap")
    rows.append(
        {
            "metric_schema": METRIC_SCHEMA,
            "metric_name": "optimization_score",
            "value": validation_rap,
            "unit": "dimensionless",
            "horizon": "experiment",
            "aggregation": "selection_objective",
            "split": "validation",
        }
    )
    return rows, summaries


def _resolved_training_config(
    config: dict[str, Any],
    *,
    dataset: dict[str, Any],
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from app.canonical_config import resolve_config
    from app.config import DEFAULT_VALUES

    template_path = Path(
        str(config.get("policy_template_file") or _POLICY_TEMPLATE)
    ).resolve()
    template = json.loads(template_path.read_text(encoding="utf-8"))
    canonical = copy.deepcopy(template)
    asset = str(config["asset"]).upper()
    timeframe = str(config["timeframe"])
    artifact_id = str(config.get("e4_artifact_id") or "e4")

    canonical["experiment"].update(
        {
            "name": f"e4_{asset.lower()}_{timeframe}_{artifact_id}",
            "mode": "train",
            "quiet_mode": True,
            "role": "asset_policy_training",
            "phase": "E4_ASSET_POLICY_TRAINING",
        }
    )
    canonical["data"].update(
        {
            "asset": asset,
            "timeframe": timeframe,
            "input_data_file": dataset["path"],
            "dataset_manifest_file": str(
                artifact_root / "policy_dataset_manifest.json"
            ),
            "features_preset": str(config["base_feature_bundle"]),
            "data_profile": str(config["base_feature_bundle"]),
            "feature_list": dataset["observation_columns"],
            "train_start": config["train_start"],
            "train_end": config["train_end"],
            "validation_start": config["validation_start"],
            "validation_end": config["validation_end"],
            "test_start": config["test_start"],
            "test_end": config["test_end"],
            "min_split_rows": int(config.get("minimum_split_rows") or 200),
        }
    )
    canonical["environment"].update(
        {
            "simulation_engine": "backtrader",
            "preprocessor_plugin": "feature_window_preprocessor",
            "feature_columns": dataset["observation_columns"],
            "feature_binary_columns": [],
            "feature_scaling": "none",
            "feature_scaling_window": 2,
            "feature_clip": 0.0,
            "window_size": 1,
            "include_price_window": False,
            "include_agent_state": True,
            "require_feature_aware_preprocessor": True,
            "precomputed_causal_features": True,
            "precomputed_feature_contract_sha256": dataset["sha256"],
            "initial_cash": float(config.get("initial_cash") or 10000.0),
        }
    )
    canonical["asset_policy"].update(
        {
            "plugin": "project3_sac_actor_critic_agent",
            "window_size": 1,
            "action_space_mode": "continuous",
            "continuous_action_threshold": float(
                config.get("continuous_action_threshold") or 0.18798110300070983
            ),
        }
    )
    canonical["training"].update(
        {
            "pipeline_plugin": "rl_pipeline_with_validation",
            "epoch_timesteps": int(config.get("epoch_timesteps") or 4000),
            "max_epochs": int(config.get("max_epochs") or 2000),
            "l1_patience": int(config.get("l1_patience") or 60),
            "l1_patience_start_epoch": int(
                config.get("l1_patience_start_epoch") or 40
            ),
            "l1_min_delta": float(config.get("l1_min_delta") or 0.00001),
            "l1_min_checkpoint_timesteps": int(
                config.get("l1_min_checkpoint_timesteps") or 5001
            ),
            "evaluate_test_split": True,
            "selection_metric": "risk_adjusted_return",
            "risk_penalty_lambda": float(config.get("risk_penalty_lambda") or 1.0),
            "learning_rate": float(
                config.get("learning_rate") or 0.00035593553866490607
            ),
            "batch_size": int(config.get("batch_size") or 256),
            "gamma": float(config.get("gamma") or 0.957651059044391),
            "tau": float(config.get("tau") or 0.001547040562484868),
            "train_freq": int(config.get("train_freq") or 1),
            "gradient_steps": int(config.get("gradient_steps") or 8),
            "ent_coef": config.get("ent_coef") or "auto",
            "net_arch": list(config.get("net_arch") or [256, 256]),
            "learning_starts": int(config.get("learning_starts") or 5000),
            "train_seed": int(config["training_seed"]),
            "eval_seed": int(config["training_seed"]),
            "device": "cuda",
        }
    )
    canonical["optimization"]["enabled"] = False
    canonical["risk"].update(
        {
            "commission": float(
                config.get("transaction_cost_fraction") or 0.0005
            ),
            "slippage": float(config.get("slippage_fraction") or 0.0),
            "rel_volume": float(config.get("rel_volume") or 0.05),
            "k_sl": float(config.get("k_sl") or 2.0),
            "k_tp": float(config.get("k_tp") or 3.0),
        }
    )
    canonical["artifacts"].update(
        {
            "artifact_root": str(artifact_root),
            "save_model": str(artifact_root / "champion_policy.zip"),
            "results_file": str(artifact_root / "results.json"),
            "resolved_config_file": str(artifact_root / "resolved_config.json"),
            "config_manifest_file": str(artifact_root / "config_manifest.json"),
            "return_trace_dir": str(artifact_root / "return_traces"),
        }
    )
    resolution = resolve_config(DEFAULT_VALUES, file_config=canonical)
    runtime = dict(resolution.runtime)
    runtime.update(
        {
            "return_trace_dir": str(artifact_root / "return_traces"),
            "write_results_sidecar": True,
            "feature_list": dataset["observation_columns"],
        }
    )
    return resolution.canonical.model_dump(mode="json"), runtime


def _run_training_subprocess(
    *,
    runtime: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    """Run Torch/SB3 in a child so each pool job releases GPU memory."""
    runtime_path = artifact_root / "runtime_config.json"
    summary_path = artifact_root / "training_summary.json"
    _atomic_json(runtime_path, runtime)
    script = (
        "import json,sys;"
        "from pathlib import Path;"
        "from app.main import _run;"
        "cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));"
        "result=_run(cfg);"
        "Path(sys.argv[2]).write_text("
        "json.dumps(result,indent=2,sort_keys=True,default=str)+'\\n',"
        "encoding='utf-8')"
    )
    subprocess.run(
        [sys.executable, "-c", script, str(runtime_path), str(summary_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_test_champion(champion: Path) -> None:
    script = (
        "import sys;"
        "from stable_baselines3 import SAC;"
        "model=SAC.load(sys.argv[1],device='cpu');"
        "del model"
    )
    subprocess.run(
        [sys.executable, "-c", script, str(champion)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def asset_policy_training(config: dict[str, Any]) -> dict[str, Any]:
    """Execute one auditable E4 policy training job."""
    if config.get("upstream_evaluation_protocol_id") is None:
        raise ValueError("E4 requires an upstream E3 evaluation protocol")
    machine_id = str(config.get("_machine_id") or "local")
    _set_cuda_visibility(machine_id)
    root = _artifact_root(config)
    root.mkdir(parents=True, exist_ok=True)

    dataset_path = root / "policy_input.csv"
    dataset = materialize_policy_dataset(config, dataset_path)
    _atomic_json(root / "policy_dataset_manifest.json", dataset)
    canonical_config, runtime = _resolved_training_config(
        config,
        dataset=dataset,
        artifact_root=root,
    )
    _atomic_json(root / "resolved_config.json", canonical_config)

    training_summary = _run_training_subprocess(
        runtime=runtime,
        artifact_root=root,
    )
    champion = Path(str(runtime["save_model"])).resolve()
    if not champion.exists():
        alternative = champion.with_suffix(champion.suffix + ".zip")
        if alternative.exists():
            champion = alternative
    if not champion.exists() or champion.stat().st_size <= 0:
        raise RuntimeError("E4 training did not create a loadable champion policy")
    _load_test_champion(champion)

    metric_rows, canonical_summary = _canonical_rows(
        training_summary,
        risk_penalty_lambda=float(config.get("risk_penalty_lambda") or 1.0),
        initial_cash=float(config.get("initial_cash") or 10000.0),
        timeframe=str(config["timeframe"]),
    )
    result = {
        "task_type": "asset_policy_training",
        "evaluation_protocol_id": E4_PROTOCOL_ID,
        "evaluation_protocol_hash": E4_PROTOCOL_HASH,
        "metric_rows": metric_rows,
        "summary": {
            "validation": canonical_summary["validation"],
            "test": canonical_summary["test"],
            "selection_uses_validation": True,
            "selection_uses_test": False,
            "training_seed": int(config["training_seed"]),
            "machine_id": machine_id,
            "upstream_job_id": config.get("upstream_job_id"),
        },
        "artifacts": [
            {
                "artifact_type": "champion_model",
                "path": str(champion),
                "sha256": _sha256(champion),
                "size_bytes": champion.stat().st_size,
                "metadata": {
                    "format": "stable_baselines3_zip",
                    "machine_id": machine_id,
                    "load_tested": True,
                },
            },
            {
                "artifact_type": "policy_dataset",
                "path": dataset["path"],
                "sha256": dataset["sha256"],
                "size_bytes": dataset["size_bytes"],
            },
            {
                "artifact_type": "resolved_config",
                "path": str(root / "resolved_config.json"),
                "sha256": _sha256(root / "resolved_config.json"),
                "size_bytes": (root / "resolved_config.json").stat().st_size,
            },
        ],
        "resolved_config": canonical_config,
        "selected_features": dataset["selected_features"],
        "selected_features_sha256": dataset["selected_features_sha256"],
        "observation_columns": dataset["observation_columns"],
        "observation_columns_sha256": dataset["observation_columns_sha256"],
        "data_contract_sha256": dataset["sha256"],
        "source_manifest": dataset["source_manifest"],
        "resolved_parameters": {
            key: value
            for key, value in config.items()
            if not str(key).startswith("_")
        },
    }
    _atomic_json(root / "e4_result.json", result)
    result["artifacts"][0]["content_base64"] = base64.b64encode(
        champion.read_bytes()
    ).decode("ascii")
    return result
