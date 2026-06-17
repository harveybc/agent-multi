#!/usr/bin/env python3
"""Automatically enqueue the next useful Project 3 weekly-pool phase.

The weekly pool workers deliberately know only how to claim subjobs. This
orchestrator owns the higher-level question: when the queue is running low,
which non-duplicative experiment batch should be created next?
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
AGENT_ROOT = TOOLS_DIR.parents[0]
FINANCIAL_ROOT = Path("/home/harveybc/Documents/GitHub/financial-data")
PLAN_WORKER = FINANCIAL_ROOT / "_scripts" / "workers" / "project3_weekly_pool_plan_worker.py"
DEFAULT_DB = FINANCIAL_ROOT / "experiments" / "weekly_walkforward_pool" / "project3_weekly_pool.sqlite"
DEFAULT_OUTPUT_DIR = FINANCIAL_ROOT / "experiments" / "weekly_walkforward_pool" / "auto_phase_plans"
DEFAULT_LABEL_DIR = FINANCIAL_ROOT / "experiments" / "oracle_behavior_pretraining" / "labels_auto"
DEFAULT_INPUT_ROOT = FINANCIAL_ROOT / "experiments" / "stage_a_screening" / "inputs"
PYTHON_BIN = "/home/harveybc/anaconda3/envs/tensorflow/bin/python"

sys.path.insert(0, str(TOOLS_DIR))
from project3_weekly_pool import connect, enqueue_plan, init_db  # noqa: E402


EVENT_CONTEXT_SOURCE_JOBS = [
    "ethusdt_4h_sota_low_cost_plus_event_engineered_v1_sac_warm_start_chain_1y",
    "ethusdt_4h_sota_low_cost_sac_warm_start_chain_1y",
    "ethusdt_4h_sota_low_cost_plus_event_engineered_v1_sac_scratch_1y",
    "ethusdt_4h_sota_low_cost_sac_scratch_1y",
    "ethusdt_4h_sota_low_cost_plus_event_engineered_v1_sac_warm_start_chain_3y",
    "ethusdt_4h_sota_low_cost_sac_warm_start_chain_3y",
    "ethusdt_4h_sota_low_cost_plus_event_engineered_v1_sac_scratch_3y",
    "ethusdt_4h_sota_low_cost_sac_scratch_3y",
]


@dataclass(frozen=True)
class PhaseResult:
    phase_id: str
    inserted_jobs: int
    inserted_subjobs: int
    plan_paths: tuple[str, ...]
    message: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _phase_suffix(phase_id: str) -> str:
    return phase_id.replace("-", "_").replace(" ", "_")


def _backlog(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        row["status"]: int(row["n"])
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM subjobs GROUP BY status")
    }


def _pending_running(conn: sqlite3.Connection) -> int:
    counts = _backlog(conn)
    return counts.get("pending", 0) + counts.get("running", 0)


def _phase_has_subjobs(conn: sqlite3.Connection, phase_id: str) -> bool:
    suffix = _phase_suffix(phase_id)
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM subjobs WHERE external_id LIKE ?",
        (f"%{suffix}%",),
    ).fetchone()
    return bool(row and row["n"])


def _record_event(conn: sqlite3.Connection, event_type: str, subject_id: str, payload: dict[str, Any]) -> None:
    with conn:
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (event_type, subject_id, _json(payload), utc_now()),
        )


def _load_job(conn: sqlite3.Connection, job_id: str) -> tuple[sqlite3.Row, dict[str, Any]] | None:
    row = conn.execute("SELECT * FROM jobs WHERE external_id=?", (job_id,)).fetchone()
    if row is None:
        return None
    return row, json.loads(row["config_json"])


def _done_subjobs(conn: sqlite3.Connection, job_row: sqlite3.Row, limit: int | None = None) -> list[sqlite3.Row]:
    sql = """
        SELECT *
        FROM subjobs
        WHERE job_id=? AND status='done'
        ORDER BY priority ASC, train_end ASC, external_id ASC
    """
    if limit is not None:
        sql += " LIMIT ?"
        return list(conn.execute(sql, (job_row["id"], limit)))
    return list(conn.execute(sql, (job_row["id"],)))


def _clone_jobs_with_seeds(
    conn: sqlite3.Connection,
    *,
    source_job_ids: list[str],
    seeds: list[int],
    phase_id: str,
    output_dir: Path,
    max_subjobs_per_job: int | None = None,
    extra_job_fields: dict[str, Any] | None = None,
) -> Path:
    jobs: list[dict[str, Any]] = []
    suffix = _phase_suffix(phase_id)
    for seed in seeds:
        for source_job_id in source_job_ids:
            loaded = _load_job(conn, source_job_id)
            if loaded is None:
                continue
            job_row, source_job = loaded
            source_rows = _done_subjobs(conn, job_row, max_subjobs_per_job)
            if not source_rows:
                continue
            source_ids = {row["external_id"] for row in source_rows}
            cloned_job = json.loads(json.dumps(source_job))
            new_job_id = f"{source_job_id}_seed{seed}_{suffix}"
            cloned_job.update(
                {
                    "job_id": new_job_id,
                    "candidate_id": new_job_id,
                    "source_job_id": source_job_id,
                    "experiment_phase": phase_id,
                    "experiment_rationale": (
                        "Automatically enqueued non-duplicative seed robustness "
                        "batch from the phase orchestrator."
                    ),
                }
            )
            if extra_job_fields:
                cloned_job.update(extra_job_fields)
            hparams = dict(cloned_job.get("hyperparameters") or {})
            hparams.update({"train_seed": seed, "eval_seed": seed, "seed_robustness_seed": seed})
            cloned_job["hyperparameters"] = hparams
            subjobs = []
            for row in source_rows:
                subjob = {
                    "subjob_id": f"{row['external_id']}_seed{seed}_{suffix}",
                    "weekly_anchor_id": row["weekly_anchor_id"],
                    "train_start": row["train_start"],
                    "train_end": row["train_end"],
                    "validation_start": row["validation_start"],
                    "validation_end": row["validation_end"],
                    "test_start": row["test_start"],
                    "test_end": row["test_end"],
                    "train_rows": row["train_rows"],
                    "validation_rows": row["validation_rows"],
                    "test_rows": row["test_rows"],
                    "priority": int(row["priority"] or 100) + seed,
                    "source_subjob_id": row["external_id"],
                }
                for dep_key in ("depends_on_subjob_id", "warm_start_parent_subjob_id"):
                    parent = row[dep_key]
                    if parent:
                        if parent not in source_ids:
                            raise RuntimeError(
                                f"{row['external_id']} depends on {parent}, which is not in source clone set"
                            )
                        subjob[dep_key] = f"{parent}_seed{seed}_{suffix}"
                subjobs.append(subjob)
            cloned_job["subjobs"] = subjobs
            jobs.append(cloned_job)

    plan = {
        "schema_version": "project3_weekly_walkforward_pool_plan_v1",
        "plan_id": phase_id,
        "generated_at": utc_now(),
        "stage_c_access": "DENIED",
        "training_launched": False,
        "source_jobs": source_job_ids,
        "seeds": seeds,
        "purpose": "Automatic phase batch generated by project3_weekly_phase_orchestrator.py",
        "jobs": jobs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{phase_id}.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _suffix_generated_plan(
    plan: dict[str, Any],
    *,
    phase_id: str,
    seed: int,
    job_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suffix = _phase_suffix(phase_id)
    out = json.loads(json.dumps(plan))
    out["plan_id"] = phase_id
    out["experiment_phase"] = phase_id
    out["generated_at"] = utc_now()
    out["stage_c_access"] = "DENIED"
    out["training_launched"] = False
    global_subjob_map = {
        subjob["subjob_id"]: f"{subjob['subjob_id']}_seed{seed}_{suffix}"
        for job in out.get("jobs", [])
        for subjob in job.get("subjobs", [])
    }
    for job in out.get("jobs", []):
        old_job_id = job["job_id"]
        new_job_id = f"{old_job_id}_seed{seed}_{suffix}"
        job["job_id"] = new_job_id
        job["candidate_id"] = new_job_id
        job["source_job_id"] = old_job_id
        job["experiment_phase"] = phase_id
        if job_overrides:
            job.update(json.loads(json.dumps(job_overrides)))
        hparams = dict(job.get("hyperparameters") or {})
        hparams.update({"train_seed": seed, "eval_seed": seed, "phase_seed": seed})
        job["hyperparameters"] = hparams
        for subjob in job.get("subjobs", []):
            old_subjob_id = subjob["subjob_id"]
            subjob["subjob_id"] = global_subjob_map[old_subjob_id]
            subjob["source_subjob_id"] = old_subjob_id
            subjob["priority"] = int(subjob.get("priority", 100)) + seed
            for dep_key in ("depends_on_subjob_id", "warm_start_parent_subjob_id"):
                parent = subjob.get(dep_key)
                if parent:
                    subjob[dep_key] = global_subjob_map[parent]
    return out


def _run_plan_worker(
    *,
    python_bin: str,
    input_data_file: Path,
    output_dir: Path,
    plan_stem: str,
    train_years: str,
    policies: str,
    fine_tune_months: str,
    max_anchors: int,
    early_stop_train_tail_days: int,
    validation_days: int,
    test_days: int,
    feature_column_mode: str = "all_available",
    execution_profile: str | None = None,
) -> Path:
    cmd = [
        python_bin,
        str(PLAN_WORKER),
        "--input-data-file",
        str(input_data_file),
        "--feature-column-mode",
        feature_column_mode,
        "--train-years",
        train_years,
        "--policies",
        policies,
        "--fine-tune-months",
        fine_tune_months,
        "--max-anchors",
        str(max_anchors),
        "--early-stop-train-tail-days",
        str(early_stop_train_tail_days),
        "--validation-days",
        str(validation_days),
        "--test-days",
        str(test_days),
        "--output-dir",
        str(output_dir),
        "--plan-stem",
        plan_stem,
    ]
    if execution_profile:
        cmd.extend(["--execution-profile", execution_profile])
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"plan worker failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return output_dir / f"{plan_stem}.json"


def _build_worker_generated_phase(
    *,
    python_bin: str,
    phase_id: str,
    specs: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for idx, spec in enumerate(specs, start=1):
        source_path = _run_plan_worker(
            python_bin=python_bin,
            input_data_file=Path(spec["input_data_file"]),
            output_dir=output_dir,
            plan_stem=f"{phase_id}_source_{idx}",
            train_years=str(spec.get("train_years", "1,3")),
            policies=str(spec.get("policies", "scratch,warm_start_chain,fine_tune_recent_window")),
            fine_tune_months=str(spec.get("fine_tune_months", "12,6,3")),
            max_anchors=int(spec.get("max_anchors", 24)),
            early_stop_train_tail_days=int(spec.get("early_stop_train_tail_days", 7)),
            validation_days=int(spec.get("validation_days", 7)),
            test_days=int(spec.get("test_days", 7)),
            feature_column_mode=str(spec.get("feature_column_mode", "all_available")),
            execution_profile=spec.get("execution_profile"),
        )
        plan = json.loads(source_path.read_text(encoding="utf-8"))
        suffixed = _suffix_generated_plan(
            plan,
            phase_id=phase_id,
            seed=int(spec.get("seed", 4)),
            job_overrides=spec.get("job_overrides"),
        )
        out_path = output_dir / f"{phase_id}_{idx}.json"
        out_path.write_text(json.dumps(suffixed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(out_path)
    return paths


def _existing_input(asset: str, timeframe: str, preset: str) -> Path | None:
    path = DEFAULT_INPUT_ROOT / asset / timeframe / preset / "train.csv"
    return path if path.exists() else None


def _asset_broadening_specs() -> list[dict[str, Any]]:
    assets = ["solusdt", "ethusdt", "btcusdt", "bnbusdt", "xrpusdt", "adausdt"]
    timeframes = ["4h", "1h"]
    presets = ["sota_low_cost", "kitchen_sink_guarded", "crypto_full", "tech_stat_decomp"]
    specs: list[dict[str, Any]] = []
    for asset in assets:
        for timeframe in timeframes:
            for preset in presets:
                path = _existing_input(asset, timeframe, preset)
                if path is None:
                    continue
                specs.append(
                    {
                        "input_data_file": str(path),
                        "train_years": "1,3",
                        "policies": "warm_start_chain,fine_tune_recent_window",
                        "fine_tune_months": "6,3",
                        "max_anchors": 20 if timeframe == "4h" else 12,
                        "early_stop_train_tail_days": 7,
                        "validation_days": 7,
                        "test_days": 7,
                        "feature_column_mode": "all_available",
                        "seed": 5,
                    }
                )
    return specs


def _top_completed_jobs(conn: sqlite3.Connection, *, limit: int, min_weeks: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT j.external_id AS job_id,
               COUNT(*) AS n,
               AVG(json_extract(s.result_json, '$.train_validation_composite_score')) AS composite_avg
        FROM subjobs s
        JOIN jobs j ON j.id=s.job_id
        WHERE s.status='done'
          AND s.result_json IS NOT NULL
          AND json_extract(s.result_json, '$.trade_gate_passed') = 1
        GROUP BY j.external_id
        HAVING n >= ?
        ORDER BY composite_avg DESC
        LIMIT ?
        """,
        (min_weeks, limit),
    ).fetchall()
    return [row["job_id"] for row in rows]


def _build_oracle_bc_phase(
    conn: sqlite3.Connection,
    *,
    db_path: str,
    python_bin: str,
    phase_id: str,
    output_dir: Path,
    label_dir: Path,
    top_n: int = 3,
    min_completed_weeks: int = 6,
    epochs: tuple[int, ...] = (1, 3, 5),
) -> list[Path]:
    label_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            python_bin,
            str(TOOLS_DIR / "project3_oracle_behavior_labels.py"),
            "--db",
            db_path,
            "--output-dir",
            str(label_dir),
            "--top-n",
            str(top_n),
            "--min-completed-weeks",
            str(min_completed_weeks),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"oracle label generation failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    manifest_path = label_dir / "oracle_behavior_label_batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_job_ids = [job["job_id"] for job in manifest.get("jobs", [])]
    paths: list[Path] = []
    suffix = _phase_suffix(phase_id)
    for epoch_count in epochs:
        jobs = []
        for source_job_id in source_job_ids:
            loaded = _load_job(conn, source_job_id)
            if loaded is None:
                continue
            job_row, source_job = loaded
            source_rows = _done_subjobs(conn, job_row)
            label_job_dir = label_dir / source_job_id
            label_source_ids = {
                path.name.removesuffix("_oracle_behavior_labels.csv")
                for path in label_job_dir.glob("*_oracle_behavior_labels.csv")
            }
            source_rows = [row for row in source_rows if row["external_id"] in label_source_ids]
            if not source_rows:
                continue
            source_ids = {row["external_id"] for row in source_rows}
            cloned = json.loads(json.dumps(source_job))
            new_job_id = f"{source_job_id}_oracle_bc_e{epoch_count}_{suffix}"
            cloned.update(
                {
                    "job_id": new_job_id,
                    "candidate_id": new_job_id,
                    "source_job_id": source_job_id,
                    "oracle_behavior_pretrain_enabled": True,
                    "oracle_behavior_source_job_id": source_job_id,
                    "oracle_behavior_labels_dir": str(label_dir),
                    "oracle_behavior_pretrain_variant": "oracle_bc_pretrain_then_sac_v1",
                    "oracle_behavior_pretrain_epochs": epoch_count,
                    "oracle_behavior_pretrain_batch_size": 512,
                    "oracle_behavior_pretrain_hold_fraction": 0.10,
                    "oracle_behavior_pretrain_max_samples": 0,
                    "experiment_phase": phase_id,
                }
            )
            subjobs = []
            for row in source_rows:
                subjob = {
                    "subjob_id": f"{row['external_id']}_oracle_bc_e{epoch_count}_{suffix}",
                    "weekly_anchor_id": row["weekly_anchor_id"],
                    "train_start": row["train_start"],
                    "train_end": row["train_end"],
                    "validation_start": row["validation_start"],
                    "validation_end": row["validation_end"],
                    "test_start": row["test_start"],
                    "test_end": row["test_end"],
                    "train_rows": row["train_rows"],
                    "validation_rows": row["validation_rows"],
                    "test_rows": row["test_rows"],
                    "priority": int(row["priority"] or 100) + epoch_count,
                    "oracle_behavior_source_subjob_id": row["external_id"],
                }
                for dep_key in ("depends_on_subjob_id", "warm_start_parent_subjob_id"):
                    parent = row[dep_key]
                    if parent and parent in source_ids:
                        subjob[dep_key] = f"{parent}_oracle_bc_e{epoch_count}_{suffix}"
                subjobs.append(subjob)
            cloned["subjobs"] = subjobs
            jobs.append(cloned)
        plan = {
            "schema_version": "project3_weekly_walkforward_pool_plan_v1",
            "plan_id": f"{phase_id}_e{epoch_count}",
            "generated_at": utc_now(),
            "stage_c_access": "DENIED",
            "training_launched": False,
            "oracle_behavior_label_manifest": str(manifest_path),
            "purpose": "Automatic oracle-behavior pretraining follow-up on current winners.",
            "jobs": jobs,
        }
        path = output_dir / f"{phase_id}_e{epoch_count}.json"
        path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _enqueue_paths(conn: sqlite3.Connection, paths: list[Path], *, dry_run: bool) -> PhaseResult:
    inserted_jobs = 0
    inserted_subjobs = 0
    for path in paths:
        if dry_run:
            plan = json.loads(path.read_text(encoding="utf-8"))
            inserted_jobs += len(plan.get("jobs", []))
            inserted_subjobs += sum(len(job.get("subjobs", [])) for job in plan.get("jobs", []))
            continue
        result = enqueue_plan(conn, path)
        inserted_jobs += int(result.get("inserted_jobs", 0))
        inserted_subjobs += int(result.get("inserted_subjobs", 0))
    return PhaseResult("", inserted_jobs, inserted_subjobs, tuple(str(path) for path in paths), "")


def _phase_event_seed_robustness(conn: sqlite3.Connection, args: argparse.Namespace) -> list[Path]:
    return [
        _clone_jobs_with_seeds(
            conn,
            source_job_ids=EVENT_CONTEXT_SOURCE_JOBS,
            seeds=[1, 2, 3],
            phase_id="eventctx_seed_robustness_phase2_v1",
            output_dir=Path(args.output_dir),
        )
    ]


def _phase_event_metric_windows(conn: sqlite3.Connection, args: argparse.Namespace) -> list[Path]:
    del conn
    specs = []
    for early_stop_days in (14, 28):
        for input_path in (
            DEFAULT_INPUT_ROOT / "ethusdt" / "4h" / "sota_low_cost" / "train.csv",
            DEFAULT_INPUT_ROOT / "ethusdt" / "4h" / "sota_low_cost_plus_event_engineered_v1" / "train.csv",
        ):
            if input_path.exists():
                specs.append(
                    {
                        "input_data_file": str(input_path),
                        "train_years": "1,3",
                        "policies": "scratch,warm_start_chain,fine_tune_recent_window",
                        "fine_tune_months": "12,6,3",
                        "max_anchors": 32,
                        "early_stop_train_tail_days": early_stop_days,
                        "validation_days": 7,
                        "test_days": 7,
                        "feature_column_mode": "all_available",
                        "seed": 4,
                    }
                )
    return _build_worker_generated_phase(
        python_bin=args.python_bin,
        phase_id="eventctx_metric_window_phase3_v1",
        specs=specs,
        output_dir=Path(args.output_dir),
    )


def _phase_asset_preset_broadening(conn: sqlite3.Connection, args: argparse.Namespace) -> list[Path]:
    del conn
    return _build_worker_generated_phase(
        python_bin=args.python_bin,
        phase_id="asset_preset_broadening_phase5_v1",
        specs=_asset_broadening_specs(),
        output_dir=Path(args.output_dir),
    )


def _phase_event_token_embedding(conn: sqlite3.Connection, args: argparse.Namespace) -> list[Path]:
    del conn
    input_path = DEFAULT_INPUT_ROOT / "ethusdt" / "4h" / "sota_low_cost_plus_event_engineered_v1" / "train.csv"
    if not input_path.exists():
        return []
    return _build_worker_generated_phase(
        python_bin=args.python_bin,
        phase_id="event_token_embedding_phase4_v1",
        specs=[
            {
                "input_data_file": str(input_path),
                "train_years": "1,3",
                "policies": "scratch,warm_start_chain,fine_tune_recent_window",
                "fine_tune_months": "6,3",
                "max_anchors": 24,
                "early_stop_train_tail_days": 7,
                "validation_days": 7,
                "test_days": 7,
                "feature_column_mode": "all_available",
                "seed": 6,
                "job_overrides": {
                    "context_embedding_profile": {
                        "enabled": True,
                        "family": "event_token_attention_v1",
                        "source_prefixes": ["event_"],
                        "output_prefix": "ctx_evt",
                        "embedding_dim": 8,
                        "seed": 20260617,
                        "fit_scope": "train_only_per_subjob",
                        "required": True,
                        "min_fit_rows": 50,
                    },
                    "feature_preset": "sota_low_cost_plus_event_token_embedding_v1",
                    "experiment_rationale": (
                        "Train-only event token attention embedding added to "
                        "event-engineered ETHUSDT 4h inputs."
                    ),
                },
            }
        ],
        output_dir=Path(args.output_dir),
    )


def _phase_oracle_bc_followup(conn: sqlite3.Connection, args: argparse.Namespace) -> list[Path]:
    return _build_oracle_bc_phase(
        conn,
        db_path=args.db,
        python_bin=args.python_bin,
        phase_id="oracle_bc_followup_phase6_v1",
        output_dir=Path(args.output_dir),
        label_dir=Path(args.label_dir),
    )


PHASES = [
    ("eventctx_seed_robustness_phase2_v1", _phase_event_seed_robustness),
    ("eventctx_metric_window_phase3_v1", _phase_event_metric_windows),
    ("event_token_embedding_phase4_v1", _phase_event_token_embedding),
    ("asset_preset_broadening_phase5_v1", _phase_asset_preset_broadening),
    ("oracle_bc_followup_phase6_v1", _phase_oracle_bc_followup),
]


def _adaptive_top_seed_extension(conn: sqlite3.Connection, args: argparse.Namespace) -> list[Path] | None:
    for seed in range(args.adaptive_seed_start, args.adaptive_seed_stop + 1):
        phase_id = f"adaptive_top_seed_extension_seed{seed}_v1"
        if _phase_has_subjobs(conn, phase_id):
            continue
        source_job_ids = _top_completed_jobs(conn, limit=args.adaptive_top_n, min_weeks=args.adaptive_min_weeks)
        if not source_job_ids:
            return None
        return [
            _clone_jobs_with_seeds(
                conn,
                source_job_ids=source_job_ids,
                seeds=[seed],
                phase_id=phase_id,
                output_dir=Path(args.output_dir),
                max_subjobs_per_job=args.adaptive_max_subjobs_per_job or None,
            )
        ]
    return None


def maybe_enqueue_next_phase(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    backlog = _pending_running(conn)
    counts = _backlog(conn)
    if backlog >= args.min_backlog:
        return {
            "event": "phase_orchestrator_noop",
            "reason": "backlog_above_threshold",
            "min_backlog": args.min_backlog,
            "backlog": backlog,
            "subjob_counts": counts,
        }

    for phase_id, builder in PHASES:
        if _phase_has_subjobs(conn, phase_id):
            continue
        paths = builder(conn, args)
        result = _enqueue_paths(conn, paths, dry_run=args.dry_run)
        payload = {
            "event": "phase_orchestrator_enqueue",
            "phase_id": phase_id,
            "inserted_jobs": result.inserted_jobs,
            "inserted_subjobs": result.inserted_subjobs,
            "plan_paths": result.plan_paths,
            "dry_run": args.dry_run,
        }
        if not args.dry_run:
            _record_event(conn, "phase_orchestrator_enqueue", phase_id, payload)
        if result.inserted_subjobs > 0 or args.dry_run:
            return payload

    adaptive_paths = _adaptive_top_seed_extension(conn, args)
    if adaptive_paths:
        phase_id = Path(adaptive_paths[0]).stem
        result = _enqueue_paths(conn, adaptive_paths, dry_run=args.dry_run)
        payload = {
            "event": "phase_orchestrator_enqueue",
            "phase_id": phase_id,
            "inserted_jobs": result.inserted_jobs,
            "inserted_subjobs": result.inserted_subjobs,
            "plan_paths": result.plan_paths,
            "dry_run": args.dry_run,
        }
        if not args.dry_run:
            _record_event(conn, "phase_orchestrator_enqueue", phase_id, payload)
        return payload

    return {
        "event": "phase_orchestrator_exhausted",
        "reason": "all configured phases already present and adaptive seed range exhausted",
        "subjob_counts": counts,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--label-dir", default=str(DEFAULT_LABEL_DIR))
    ap.add_argument("--python-bin", default=PYTHON_BIN)
    ap.add_argument("--min-backlog", type=int, default=120)
    ap.add_argument("--sleep-sec", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adaptive-top-n", type=int, default=4)
    ap.add_argument("--adaptive-min-weeks", type=int, default=6)
    ap.add_argument("--adaptive-seed-start", type=int, default=6)
    ap.add_argument("--adaptive-seed-stop", type=int, default=9)
    ap.add_argument("--adaptive-max-subjobs-per-job", type=int, default=32)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.python_bin).exists() and shutil.which(args.python_bin) is None:
        raise FileNotFoundError(f"python-bin not found: {args.python_bin}")
    conn = connect(args.db)
    init_db(conn)
    while True:
        try:
            payload = maybe_enqueue_next_phase(conn, args)
        except Exception as exc:  # Keep daemon alive; operator can inspect pool_events/logs.
            payload = {
                "event": "phase_orchestrator_error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
            _record_event(conn, "phase_orchestrator_error", "project3_weekly_phase_orchestrator", payload)
        payload["generated_at"] = utc_now()
        print(json.dumps(payload, sort_keys=True), flush=True)
        if args.once:
            return
        time.sleep(max(30, args.sleep_sec))


if __name__ == "__main__":
    main()
