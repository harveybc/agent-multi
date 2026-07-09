#!/usr/bin/env python3
"""Materialize an agent-multi config for one Project 3 weekly pool subjob."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from project3_weekly_pool import connect, init_db  # noqa: E402
from project3_event_token_transformer import (  # noqa: E402
    encode_event_token_transformer,
)


SUPPORTED_CONTEXT_FAMILIES = (
    "event_token_attention_v1",
    "event_token_transformer_v1",
)
DEFAULT_OUTPUT_PREFIX_BY_FAMILY = {
    "event_token_attention_v1": "ctx_evt",
    "event_token_transformer_v1": "ctx_evt_tr",
}
DEFAULT_WINDOW_SIZE = 32
DEFAULT_MIN_SPLIT_ROWS = 40


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "weekly_walkforward_pool"
HELDOUT_START = "2025-01-01 00:00:00"
ORACLE_BC_SUBJOB_SUFFIX = "_oracle_bc_v1"
ORACLE_BC_FOLLOWUP_SUFFIX_RE = re.compile(r"_oracle_bc_e\d+_oracle_bc_followup_phase\d+_v\d+$")
STABLE_SAC_OVERRIDES: dict[str, Any] = {
    "window_size": 32,
    "learning_rate": 0.0001,
    "batch_size": 256,
    "buffer_size": 200000,
    "learning_starts": 1000,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": "auto",
    "train_freq": 1,
    "gradient_steps": 1,
    "use_sde": False,
    "target_update_interval": 1,
    "target_entropy": "auto",
    "device": "cuda",
    "project3_strict": True,
    "strategy_plugin": "direct_atr_sltp",
    "atr_period": 14,
    "k_sl": 2.0,
    "k_tp": 3.0,
    "rel_volume": 0.05,
    "size_mode": "notional",
    "leverage": 1.0,
    "min_order_volume": 0.0,
    "max_order_volume": 100.0,
    "epoch_timesteps": 2000,
    "max_epochs": 500,
    "l1_patience": 20,
    "l1_min_delta": 0.0001,
    "early_stop_train_tail_days": 7,
    "early_stop_min_trades": 1,
    "early_stop_no_trade_penalty": 1000000.0,
    "total_timesteps": 1000000,
}


def _weekly_observed_split_rows(subjob: dict[str, Any]) -> list[int]:
    observed_rows: list[int] = []
    for key in ("validation_rows", "test_rows"):
        try:
            value = int(subjob.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            observed_rows.append(value)
    return observed_rows


def _weekly_min_split_rows(subjob: dict[str, Any]) -> int:
    """Use a split minimum that cannot exceed normal weekly row availability."""
    observed_rows = _weekly_observed_split_rows(subjob)
    if not observed_rows:
        return DEFAULT_MIN_SPLIT_ROWS
    return max(8, min(DEFAULT_MIN_SPLIT_ROWS, min(observed_rows)))


def _weekly_window_size(subjob: dict[str, Any], requested: int = DEFAULT_WINDOW_SIZE) -> int:
    observed_rows = _weekly_observed_split_rows(subjob)
    if not observed_rows:
        return requested
    min_rows = min(observed_rows)
    if min_rows > requested:
        return requested
    return max(8, min(requested, min_rows - 7))


def _parse_dt(value: Any) -> datetime:
    text = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return out


def _pearson_abs(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return abs(cov / math.sqrt(vx * vy))


def _context_source_columns(header: list[str], profile: dict[str, Any]) -> list[str]:
    explicit = [str(col) for col in profile.get("source_columns") or [] if str(col) in header]
    prefixes = tuple(str(prefix) for prefix in profile.get("source_prefixes") or ["event_"])
    prefixed = [
        col
        for col in header
        if prefixes
        and any(col.startswith(prefix) for prefix in prefixes)
        and col not in explicit
    ]
    return [*explicit, *prefixed]


def _encode_attention_v1(
    *,
    rows: list[dict[str, Any]],
    source_columns: list[str],
    train_indices: list[int],
    price_column: str,
    profile: dict[str, Any],
    output_prefix: str,
) -> dict[str, Any]:
    """Train-only correlation-weighted attention embedding (the first bridge).

    Mutates ``rows`` in place and returns the manifest fragment.
    """
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in source_columns:
        values = [_safe_float(rows[idx].get(col)) for idx in train_indices]
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1)
        std = math.sqrt(var) if var > 1.0e-12 else 1.0
        means[col] = mean
        stds[col] = std

    target_by_idx: dict[int, float] = {}
    train_set = set(train_indices)
    for idx in train_indices:
        if idx + 1 in train_set:
            now_price = max(1.0e-12, _safe_float(rows[idx].get(price_column)))
            next_price = max(1.0e-12, _safe_float(rows[idx + 1].get(price_column)))
            target_by_idx[idx] = math.log(next_price / now_price)
        else:
            target_by_idx[idx] = 0.0
    targets = [target_by_idx[idx] for idx in train_indices]
    token_scores: dict[str, float] = {}
    for col in source_columns:
        xs = [
            (_safe_float(rows[idx].get(col)) - means[col]) / stds[col]
            for idx in train_indices
        ]
        token_scores[col] = _pearson_abs(xs, targets)

    dim = int(profile.get("embedding_dim", 8))
    if dim <= 0 or dim > 64:
        raise ValueError("context embedding_dim must be between 1 and 64")
    seed = int(profile.get("seed", 0))
    rng = random.Random(seed)
    projection = {
        col: [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        for col in source_columns
    }
    embedding_columns = [f"{output_prefix}_{i:02d}" for i in range(dim)]
    diagnostic_columns = [f"{output_prefix}_attn_mass", f"{output_prefix}_token_count"]

    for row in rows:
        z_values = {
            col: (_safe_float(row.get(col)) - means[col]) / stds[col]
            for col in source_columns
        }
        raw_weights = {
            col: abs(z) * max(token_scores[col], 1.0e-6)
            for col, z in z_values.items()
        }
        weight_sum = sum(raw_weights.values())
        if weight_sum <= 0.0:
            weights = {col: 1.0 / len(source_columns) for col in source_columns}
            attn_mass = 0.0
        else:
            weights = {col: raw_weights[col] / weight_sum for col in source_columns}
            attn_mass = weight_sum
        emb = []
        for i in range(dim):
            value = 0.0
            for col in source_columns:
                value += weights[col] * math.tanh(z_values[col]) * projection[col][i]
            emb.append(value)
        for col, value in zip(embedding_columns, emb):
            row[col] = f"{value:.10g}"
        row[diagnostic_columns[0]] = f"{attn_mass:.10g}"
        row[diagnostic_columns[1]] = str(len(source_columns))

    return {
        "embedding_columns": embedding_columns,
        "diagnostic_columns": diagnostic_columns,
        "token_scores": token_scores,
        "means": means,
        "stds": stds,
    }


def _build_context_embedding_file(
    *,
    job: dict[str, Any],
    subjob: dict[str, Any],
    run_dir: Path,
    features: list[str],
) -> tuple[str, list[str], dict[str, Any] | None]:
    profile = dict(job.get("context_embedding_profile") or {})
    if not profile or not bool(profile.get("enabled", False)):
        return str(job["input_data_file"]), features, None
    family = str(profile.get("family") or "event_token_attention_v1")
    if family not in SUPPORTED_CONTEXT_FAMILIES:
        raise ValueError(
            f"unsupported context_embedding_profile.family={family!r}; "
            f"supported={list(SUPPORTED_CONTEXT_FAMILIES)}"
        )

    input_path = Path(str(job["input_data_file"]))
    if not input_path.exists():
        raise FileNotFoundError(f"context embedding input_data_file not found: {input_path}")
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"context embedding input CSV has no rows: {input_path}")
    date_column = str(profile.get("date_column") or "DATE_TIME")
    price_column = str(profile.get("target_price_column") or "CLOSE")
    if date_column not in header:
        raise ValueError(f"context embedding date column {date_column!r} missing from {input_path}")
    if price_column not in header:
        raise ValueError(f"context embedding price column {price_column!r} missing from {input_path}")

    source_columns = _context_source_columns(header, profile)
    if not source_columns:
        if bool(profile.get("required", True)):
            raise ValueError(
                "context embedding profile found no source columns; "
                f"prefixes={profile.get('source_prefixes') or ['event_']}"
            )
        return str(job["input_data_file"]), features, None

    train_start = _parse_dt(subjob["train_start"])
    train_end = _parse_dt(subjob["train_end"])
    train_indices = [
        idx
        for idx, row in enumerate(rows)
        if train_start <= _parse_dt(row[date_column]) < train_end
    ]
    if len(train_indices) < int(profile.get("min_fit_rows", 50)):
        raise ValueError(
            f"context embedding fit rows too small: {len(train_indices)} "
            f"< {int(profile.get('min_fit_rows', 50))}"
        )

    output_prefix = str(
        profile.get("output_prefix")
        or DEFAULT_OUTPUT_PREFIX_BY_FAMILY.get(family, "ctx_evt")
    )

    if family == "event_token_transformer_v1":
        result = encode_event_token_transformer(
            rows=rows,
            source_columns=source_columns,
            train_indices=train_indices,
            price_column=price_column,
            profile=profile,
            output_prefix=output_prefix,
            safe_float=_safe_float,
        )
    else:
        result = _encode_attention_v1(
            rows=rows,
            source_columns=source_columns,
            train_indices=train_indices,
            price_column=price_column,
            profile=profile,
            output_prefix=output_prefix,
        )

    embedding_columns = result.pop("embedding_columns")
    diagnostic_columns = result.pop("diagnostic_columns")
    output_header = [*header, *embedding_columns, *diagnostic_columns]

    context_dir = run_dir / "context_embedding"
    context_dir.mkdir(parents=True, exist_ok=True)
    output_path = context_dir / "input_with_context_embedding.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_header)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "schema_version": "project3_context_embedding_manifest_v1",
        "family": family,
        "input_data_file": str(input_path),
        "output_data_file": str(output_path),
        "fit_scope": "train_only",
        "fit_window_start": subjob["train_start"],
        "fit_window_end": subjob["train_end"],
        "source_columns": source_columns,
        "embedding_columns": embedding_columns,
        "diagnostic_columns": diagnostic_columns,
        "embedding_dim": len(embedding_columns),
        "seed": int(profile.get("seed", 0)),
        "training_summary": {},
        "model_config": {},
    }
    metadata.update(result)
    manifest_path = context_dir / "context_embedding_manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    merged_features = list(features)
    for col in [*embedding_columns, *diagnostic_columns]:
        if col not in merged_features:
            merged_features.append(col)
    metadata["manifest_file"] = str(manifest_path)
    return str(output_path), merged_features, metadata


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(value)


def _oracle_behavior_source_subjob_id_candidates(subjob: dict[str, Any]) -> list[str]:
    explicit = subjob.get("oracle_behavior_source_subjob_id")
    if explicit:
        return [str(explicit)]
    source_subjob_id = str(subjob["external_id"])
    source_subjob_id = ORACLE_BC_FOLLOWUP_SUFFIX_RE.sub("", source_subjob_id)
    candidates = [source_subjob_id]
    if source_subjob_id.endswith(ORACLE_BC_SUBJOB_SUFFIX):
        candidates.append(source_subjob_id[: -len(ORACLE_BC_SUBJOB_SUFFIX)])
    return list(dict.fromkeys(candidates))


def _fetch_subjob(conn: sqlite3.Connection, subjob_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    row = conn.execute(
        """
        SELECT s.*, j.config_json
        FROM subjobs s
        JOIN jobs j ON j.id = s.job_id
        WHERE s.external_id=?
        """,
        (subjob_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown subjob_id: {subjob_id}")
    subjob = dict(row)
    job = _json_loads(subjob.pop("config_json"))
    return job, subjob


def _fetch_parent_policy(conn: sqlite3.Connection, parent_subjob_id: str) -> Path:
    row = conn.execute(
        "SELECT status, run_dir, result_json FROM subjobs WHERE external_id=?",
        (parent_subjob_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown warm-start parent subjob: {parent_subjob_id}")
    if row["status"] != "done":
        raise RuntimeError(
            f"warm-start parent {parent_subjob_id} is not done yet; status={row['status']}"
        )
    run_dir = row["run_dir"]
    if run_dir:
        policy = Path(run_dir) / "policy.zip"
        if policy.exists():
            return policy
    result = _json_loads(row["result_json"])
    policy = Path(str(result.get("run_dir") or "")) / "policy.zip"
    if policy.exists():
        return policy
    raise FileNotFoundError(f"warm-start parent policy not found for {parent_subjob_id}")


def build_config(
    job: dict[str, Any],
    subjob: dict[str, Any],
    output_root: Path,
    warm_start_model: Path | None = None,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / subjob["external_id"]
    trace_dir = run_dir / "return_traces"
    hparams = dict(job.get("hyperparameters") or {})
    features = list(job.get("feature_columns") or job.get("selected_features") or [])
    if not features:
        raise ValueError(f"job {job.get('job_id')} has no feature columns")
    input_data_file, features, context_embedding_metadata = _build_context_embedding_file(
        job=job,
        subjob=subjob,
        run_dir=run_dir,
        features=features,
    )

    cfg: dict[str, Any] = {
        "mode": "train",
        "quiet_mode": False,
        "agent_plugin": job.get("agent_plugin", "project3_sac_actor_critic_agent"),
        "env_plugin": job.get("env_plugin", "gym_fx_env"),
        "pipeline_plugin": job.get("pipeline_plugin", "rl_pipeline_with_validation"),
        "asset": f"{job['asset']}_{job['timeframe']}",
        "timeframe": job["timeframe"],
        "input_data_file": input_data_file,
        "date_column": "DATE_TIME",
        "price_column": "CLOSE",
        "window_size": DEFAULT_WINDOW_SIZE,
        "feature_list": features,
        "feature_columns": features,
        "features_preset": job.get("feature_preset"),
        "execution_profile": job.get("execution_profile"),
        "preprocessing_profile": job.get("preprocessing_profile"),
        "train_years": int(job["train_years"]),
        "validation_days": int(job.get("validation_days", 7)),
        "test_days": int(job.get("test_days", 7)),
        "train_start": subjob["train_start"],
        "train_end": subjob["train_end"],
        "validation_start": subjob["validation_start"],
        "validation_end": subjob["validation_end"],
        "test_start": subjob["test_start"],
        "test_end": subjob["test_end"],
        "min_split_rows": _weekly_min_split_rows(subjob),
        "heldout_start": HELDOUT_START,
        "stage_c_access": "DENIED",
        "final_stage_c_evaluation": False,
        "stage_c_acknowledged": False,
        "_project3_weekly_pool": True,
        "_stage_c_firewall": "DENIED",
        "_weekly_pool_job_id": job["job_id"],
        "_weekly_pool_subjob_id": subjob["external_id"],
        "_weekly_anchor_id": subjob["weekly_anchor_id"],
        "_training_policy": job.get("training_policy", "scratch_n_years"),
        "_fine_tune_months": job.get("fine_tune_months"),
        "_depends_on_subjob_id": subjob.get("depends_on_subjob_id"),
        "_warm_start_parent_subjob_id": subjob.get("warm_start_parent_subjob_id"),
        "_input_data_sha256": job.get("input_data_sha256"),
        "_context_embedding_profile": job.get("context_embedding_profile"),
        "_context_embedding_manifest_file": (
            context_embedding_metadata.get("manifest_file")
            if context_embedding_metadata
            else None
        ),
        "_context_embedding_feature_columns": (
            context_embedding_metadata.get("embedding_columns")
            if context_embedding_metadata
            else None
        ),
        "_train_rows": subjob.get("train_rows"),
        "_validation_rows": subjob.get("validation_rows"),
        "_test_rows": subjob.get("test_rows"),
        "train_seed": 0,
        "eval_seed": 0,
        "save_model": str(run_dir / "policy.zip"),
        "results_file": str(run_dir / "results.json"),
        "save_config": str(run_dir / "config_out.json"),
        "progress_file": str(run_dir / "training_progress.json"),
        "training_progress_file": str(run_dir / "training_progress.json"),
        "return_trace_dir": str(trace_dir),
        "return_trace_file": str(trace_dir / "evaluation_return_trace.csv"),
        "commission": 0.0002,
        "slippage": 0.0,
        "initial_cash": 10000.0,
        "stage_b_force_close_obs": True,
        "force_close_dow": 4,
        "force_close_hour": 20,
        "force_close_window_hours": 4,
        "monday_entry_window_hours": 4,
    }
    execution_profile = str(job.get("execution_profile") or "").strip().lower()
    if execution_profile == "event_no_trade_overlay_v1":
        if "event_no_trade_window_active" not in features:
            raise ValueError(
                "event_no_trade_overlay_v1 requires event_no_trade_window_active "
                f"in job {job.get('job_id')} feature columns"
            )
        cfg.update(
            {
                "event_context_execution_overlay": True,
                "event_context_no_trade_column": "event_no_trade_window_active",
                "event_context_no_trade_threshold": 0.5,
                "event_context_block_new_entries": True,
                "event_context_force_flat": True,
                "event_context_spread_stress_column": "event_spread_stress_multiplier",
                "event_context_slippage_stress_column": "event_slippage_stress_multiplier",
                "_event_context_overlay_policy": "force_flat_and_block_entries",
            }
        )
    cfg.update(STABLE_SAC_OVERRIDES)
    cfg.update(hparams)
    requested_window_size = int(cfg.get("window_size", DEFAULT_WINDOW_SIZE))
    adapted_window_size = _weekly_window_size(subjob, requested_window_size)
    if adapted_window_size < requested_window_size:
        cfg["_window_size_adapted_for_weekly_rows"] = True
        cfg["_window_size_previous"] = requested_window_size
        cfg["window_size"] = adapted_window_size
    if bool(job.get("oracle_behavior_pretrain_enabled", False)):
        source_job_id = job.get("oracle_behavior_source_job_id") or job.get("source_job_id")
        labels_root = job.get("oracle_behavior_labels_dir")
        if not source_job_id or not labels_root:
            raise ValueError(
                "oracle_behavior_pretrain_enabled requires "
                "oracle_behavior_source_job_id and oracle_behavior_labels_dir"
            )
        label_dir = Path(str(labels_root)) / str(source_job_id)
        label_candidates = [
            (source_subjob_id, label_dir / f"{source_subjob_id}_oracle_behavior_labels.csv")
            for source_subjob_id in _oracle_behavior_source_subjob_id_candidates(subjob)
        ]
        source_subjob_id, labels_file = next(
            ((candidate_id, path) for candidate_id, path in label_candidates if path.exists()),
            label_candidates[0],
        )
        if not labels_file.exists():
            tried = ", ".join(str(path) for _, path in label_candidates)
            raise FileNotFoundError(f"oracle behavior labels not found; tried: {tried}")
        cfg.update(
            {
                "oracle_behavior_pretrain_enabled": True,
                "oracle_behavior_labels_file": str(labels_file),
                "oracle_behavior_source_job_id": str(source_job_id),
                "oracle_behavior_source_subjob_id": str(source_subjob_id),
                "oracle_behavior_pretrain_variant": job.get(
                    "oracle_behavior_pretrain_variant",
                    "oracle_bc_pretrain_then_sac_v1",
                ),
                "oracle_behavior_pretrain_epochs": int(job.get("oracle_behavior_pretrain_epochs", 3)),
                "oracle_behavior_pretrain_batch_size": int(job.get("oracle_behavior_pretrain_batch_size", 512)),
                "oracle_behavior_pretrain_hold_fraction": float(job.get("oracle_behavior_pretrain_hold_fraction", 0.10)),
                "oracle_behavior_pretrain_max_samples": int(job.get("oracle_behavior_pretrain_max_samples", 0)),
            }
        )
    if warm_start_model is not None:
        cfg["warm_start_model"] = str(warm_start_model)
        cfg["_warm_start_model"] = str(warm_start_model)
    return cfg


def materialize(db_path: str | Path, subjob_id: str, output_root: str | Path = DEFAULT_EXPERIMENT_ROOT) -> Path:
    conn = connect(db_path)
    init_db(conn)
    job, subjob = _fetch_subjob(conn, subjob_id)
    parent_id = subjob.get("warm_start_parent_subjob_id") or subjob.get("depends_on_subjob_id")
    warm_start_model = _fetch_parent_policy(conn, parent_id) if parent_id else None
    output_root = Path(output_root)
    cfg = build_config(job, subjob, output_root, warm_start_model)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{subjob_id}.json"
    config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_dir = Path(cfg["save_model"]).parent
    with conn:
        conn.execute(
            "UPDATE subjobs SET config_path=?, run_dir=?, updated_at=datetime('now') WHERE external_id=?",
            (str(config_path), str(run_dir), subjob_id),
        )
    return config_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--subjob-id", required=True)
    ap.add_argument("--output-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    args = ap.parse_args()
    path = materialize(args.db, args.subjob_id, args.output_root)
    print(json.dumps({"config_path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
