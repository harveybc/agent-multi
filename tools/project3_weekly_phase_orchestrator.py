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

# Event-engineered source jobs carry the ``event_*`` columns the transformer
# encoder needs as tokens, so the transformer phase clones only those.
EVENT_TOKEN_TRANSFORMER_SOURCE_JOBS = [
    job_id for job_id in EVENT_CONTEXT_SOURCE_JOBS if "event_engineered_v1" in job_id
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
                            subjob.setdefault("dropped_external_dependencies", {})[dep_key] = parent
                            continue
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
    included_subjob_ids = {
        subjob["subjob_id"]
        for job in out.get("jobs", [])
        for subjob in job.get("subjobs", [])
    }
    skipped_subjobs: list[dict[str, str]] = []
    changed = True
    while changed:
        changed = False
        for job in out.get("jobs", []):
            for subjob in job.get("subjobs", []):
                subjob_id = subjob["subjob_id"]
                if subjob_id not in included_subjob_ids:
                    continue
                for dep_key in ("depends_on_subjob_id", "warm_start_parent_subjob_id"):
                    parent = subjob.get(dep_key)
                    if parent and parent not in included_subjob_ids:
                        included_subjob_ids.remove(subjob_id)
                        skipped_subjobs.append(
                            {
                                "subjob_id": subjob_id,
                                "missing_dependency_key": dep_key,
                                "missing_dependency": parent,
                            }
                        )
                        changed = True
                        break
    kept_jobs = []
    for job in out.get("jobs", []):
        kept_subjobs = [
            subjob for subjob in job.get("subjobs", []) if subjob["subjob_id"] in included_subjob_ids
        ]
        if kept_subjobs:
            job["subjobs"] = kept_subjobs
            kept_jobs.append(job)
    out["jobs"] = kept_jobs
    if skipped_subjobs:
        out["skipped_subjobs_due_missing_dependencies"] = skipped_subjobs
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
                        "policies": "scratch,warm_start_chain,fine_tune_recent_window",
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
    annual_rows = conn.execute(
        """
        SELECT candidate_id,
               unique_weeks,
               annual_rap,
               mean_weekly_l1_score
        FROM weekly_result_full_year_protocol_olap
        WHERE metric_block='validation_year'
          AND has_near_full_year_coverage = 1
          AND unique_weeks >= ?
        ORDER BY COALESCE(mean_weekly_l1_score, annual_rap) DESC,
                 annual_rap DESC
        LIMIT ?
        """,
        (max(48, min_weeks), limit),
    ).fetchall()
    annual_candidate_ids = [row["candidate_id"] for row in annual_rows if row["candidate_id"]]
    if annual_candidate_ids:
        placeholders = ",".join("?" for _ in annual_candidate_ids)
        job_rows = conn.execute(
            f"""
            SELECT external_id
            FROM jobs
            WHERE candidate_id IN ({placeholders})
              AND json_extract(config_json, '$.evaluation_protocol') = 'full_year_validation_test_v1'
              AND json_extract(config_json, '$.evaluation_block') IN ('validation_year', 'test_year')
            ORDER BY
              CASE json_extract(config_json, '$.evaluation_block')
                WHEN 'validation_year' THEN 0
                WHEN 'test_year' THEN 1
                ELSE 2
              END,
              external_id
            """,
            tuple(annual_candidate_ids),
        ).fetchall()
        annual_job_ids = [row["external_id"] for row in job_rows if row["external_id"]]
        if annual_job_ids:
            return annual_job_ids
    rows = conn.execute(
        """
        SELECT j.external_id AS job_id,
               COUNT(*) AS n,
               AVG(COALESCE(
                   json_extract(s.result_json, '$.train_validation_l1_score'),
                   json_extract(s.result_json, '$.train_validation_selection_score'),
                   json_extract(s.result_json, '$.train_validation_risk_adjusted_composite_score'),
                   json_extract(s.result_json, '$.train_validation_composite_score')
               )) AS selection_avg
        FROM subjobs s
        JOIN jobs j ON j.id=s.job_id
        WHERE s.status='done'
          AND s.result_json IS NOT NULL
          AND json_extract(s.result_json, '$.trade_gate_passed') = 1
        GROUP BY j.external_id
        HAVING n >= ?
        ORDER BY selection_avg DESC
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


def _event_token_transformer_profile() -> dict[str, Any]:
    return {
        "enabled": True,
        "family": "event_token_transformer_v1",
        "source_prefixes": ["event_"],
        "output_prefix": "ctx_evt_tr",
        "embedding_dim": 16,
        "hidden_size": 16,
        "num_heads": 2,
        "num_blocks": 2,
        "ff_dim": 32,
        "seed": 20260617,
        "fit_scope": "train_only_per_subjob",
        "required": True,
        "min_fit_rows": 50,
    }


def _phase_event_token_transformer(conn: sqlite3.Connection, args: argparse.Namespace) -> list[Path]:
    """Clone completed event-engineered anchors with the transformer profile.

    Cloning already-completed subjobs (rather than regenerating anchors via the
    plan worker) keeps this phase self-contained and identical in window
    coverage to the matched ``event_token_attention_v1`` baseline, which is what
    the acceptance comparison requires.
    """
    if not EVENT_TOKEN_TRANSFORMER_SOURCE_JOBS:
        return []
    return [
        _clone_jobs_with_seeds(
            conn,
            source_job_ids=EVENT_TOKEN_TRANSFORMER_SOURCE_JOBS,
            seeds=[7],
            phase_id="event_token_transformer_phase_next_v1",
            output_dir=Path(args.output_dir),
            extra_job_fields={
                "context_embedding_profile": _event_token_transformer_profile(),
                "feature_preset": "sota_low_cost_plus_event_token_transformer_v1",
                "experiment_rationale": (
                    "Train-only event-token transformer embedding "
                    "(event_token_transformer_v1) cloned from completed "
                    "event-engineered ETHUSDT 4h anchors."
                ),
            },
        )
    ]


def _phase_oracle_bc_followup(conn: sqlite3.Connection, args: argparse.Namespace) -> list[Path]:
    return _build_oracle_bc_phase(
        conn,
        db_path=args.db,
        python_bin=args.python_bin,
        phase_id="oracle_bc_followup_phase6_v1",
        output_dir=Path(args.output_dir),
        label_dir=Path(args.label_dir),
    )


def _phase_risk_adjusted_followup(conn: sqlite3.Connection, args: argparse.Namespace) -> list[Path]:
    """Clone current winners into a small profit-risk sweep.

    This is intentionally not a broad Cartesian sweep. It starts from already
    evidenced candidates, swaps the training reward to drawdown-penalized PnL,
    and asks the level-1 early stopper to rank checkpoints by gap-penalized
    train-tail/validation RAP. Test stays report-only.
    """
    source_job_ids = _top_completed_jobs(conn, limit=2, min_weeks=48)
    if not source_job_ids:
        return []
    phase_id = "risk_adjusted_reward_phase7_v3"
    suffix = _phase_suffix(phase_id)
    lambdas = (0.25, 0.5, 1.0)
    rel_volumes = (0.05, 0.075, 0.10)
    l1_gap_beta = 0.25
    max_subjobs_per_job = 52
    jobs: list[dict[str, Any]] = []
    for source_job_id in source_job_ids:
        loaded = _load_job(conn, source_job_id)
        if loaded is None:
            continue
        job_row, source_job = loaded
        source_rows = _done_subjobs(conn, job_row, max_subjobs_per_job)
        if not source_rows:
            continue
        for risk_lambda in lambdas:
            for rel_volume in rel_volumes:
                risk_tag = f"rap_l{str(risk_lambda).replace('.', 'p')}_rv{str(rel_volume).replace('.', 'p')}"
                new_job_id = f"{source_job_id}_{risk_tag}_{suffix}"
                cloned_job = json.loads(json.dumps(source_job))
                cloned_job.update(
                    {
                        "job_id": new_job_id,
                        "candidate_id": new_job_id,
                        "source_job_id": source_job_id,
                        "experiment_phase": phase_id,
                        "experiment_rationale": (
                            "Profit-risk follow-up on current winners: "
                            "dd_penalized_reward plus level-1 selection by "
                            "gap-penalized train-tail/validation "
                            "risk_adjusted_return. Test remains report-only."
                        ),
                        "risk_adjusted_followup": True,
                        "risk_penalty_lambda": risk_lambda,
                        "risk_sizing_rel_volume": rel_volume,
                        "l1_generalization_gap_penalty_beta": l1_gap_beta,
                    }
                )
                hparams = dict(cloned_job.get("hyperparameters") or {})
                hparams.update(
                    {
                        "reward_plugin": "dd_penalized_reward",
                        "penalty_lambda": risk_lambda,
                        "selection_metric": "risk_adjusted_return",
                        "risk_penalty_lambda": risk_lambda,
                        "l1_generalization_gap_penalty_beta": l1_gap_beta,
                        "rel_volume": rel_volume,
                        "train_seed": 8,
                        "eval_seed": 8,
                        "phase_seed": 8,
                    }
                )
                cloned_job["hyperparameters"] = hparams
                subjobs = []
                for row in source_rows:
                    subjob = {
                        "subjob_id": f"{row['external_id']}_{risk_tag}_{suffix}",
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
                        "depends_on_subjob_id": row["external_id"],
                        "warm_start_parent_subjob_id": row["external_id"],
                        # Keep this follow-up behind the currently running broadening queue
                        # unless an operator explicitly reprioritizes it after worker rollout.
                        "priority": int(row["priority"] or 100) + 10_000,
                        "source_subjob_id": row["external_id"],
                    }
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
        "purpose": (
            "Small risk-adjusted reward/sizing sweep over already evidenced "
            "weekly walk-forward winners."
        ),
        "risk_metric": "RAP = total_return - lambda * max_drawdown_fraction",
        "l1_metric": "mean(train_tail_RAP, validation_RAP) - beta * abs(train_tail_RAP - validation_RAP)",
        "l1_generalization_gap_penalty_beta": l1_gap_beta,
        "lambda_values": list(lambdas),
        "rel_volume_values": list(rel_volumes),
        "jobs": jobs,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{phase_id}.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [path]


def _phase_sltp_risk_geometry_followup(conn: sqlite3.Connection, args: argparse.Namespace) -> list[Path]:
    """Clone winners into a compact SL/TP/risk-control sweep.

    This phase is the pragmatic follow-up after profit-only and RAP probes. It
    keeps the historical 2ATR/3ATR, rel_volume=0.05 behavior as an explicit
    control, then compares fixed ATR geometry against exposure-aware SL/TP
    geometry. The selection metric remains gap-penalized train-tail/validation
    RAP; same-week test remains report-only.
    """
    source_job_ids = _top_completed_jobs(conn, limit=2, min_weeks=48)
    if not source_job_ids:
        return []
    phase_id = "sltp_risk_geometry_phase8_v3"
    suffix = _phase_suffix(phase_id)
    risk_lambda = 0.5
    l1_gap_beta = 0.25
    max_subjobs_per_job = 52
    profiles = [
        {
            "tag": "control_rv0p05_sl2_tp3",
            "family": "fixed_atr_sltp_control",
            "rel_volume": 0.05,
            "k_sl": 2.0,
            "k_tp": 3.0,
            "sltp_risk_mode": "fixed_atr",
        },
        {
            "tag": "fixed_rv0p05_sl1p25_tp1p25",
            "family": "fixed_atr_sltp_geometry",
            "rel_volume": 0.05,
            "k_sl": 1.25,
            "k_tp": 1.25,
            "sltp_risk_mode": "fixed_atr",
        },
        {
            "tag": "fixed_rv0p05_sl1p5_tp2",
            "family": "fixed_atr_sltp_geometry",
            "rel_volume": 0.05,
            "k_sl": 1.5,
            "k_tp": 2.0,
            "sltp_risk_mode": "fixed_atr",
        },
        {
            "tag": "fixed_rv0p05_sl2_tp4",
            "family": "fixed_atr_sltp_geometry",
            "rel_volume": 0.05,
            "k_sl": 2.0,
            "k_tp": 4.0,
            "sltp_risk_mode": "fixed_atr",
        },
        {
            "tag": "fixed_rv0p10_sl2_tp3",
            "family": "fixed_atr_sltp_volume",
            "rel_volume": 0.10,
            "k_sl": 2.0,
            "k_tp": 3.0,
            "sltp_risk_mode": "fixed_atr",
        },
        {
            "tag": "fixed_rv0p10_sl1p5_tp2",
            "family": "fixed_atr_sltp_volume",
            "rel_volume": 0.10,
            "k_sl": 1.5,
            "k_tp": 2.0,
            "sltp_risk_mode": "fixed_atr",
        },
        {
            "tag": "fixed_rv0p25_sl1p25_tp1p5",
            "family": "fixed_atr_sltp_high_exposure",
            "rel_volume": 0.25,
            "k_sl": 1.25,
            "k_tp": 1.5,
            "sltp_risk_mode": "fixed_atr",
            "max_planned_loss_fraction": 0.025,
        },
        {
            "tag": "fixed_rv0p50_sl1p25_tp1p25",
            "family": "fixed_atr_sltp_high_exposure",
            "rel_volume": 0.50,
            "k_sl": 1.25,
            "k_tp": 1.25,
            "sltp_risk_mode": "fixed_atr",
            "max_planned_loss_fraction": 0.025,
        },
        {
            "tag": "aware_rv0p10_base",
            "family": "rel_volume_aware_sltp",
            "rel_volume": 0.10,
            "k_sl": 2.0,
            "k_tp": 3.0,
            "sltp_risk_mode": "rel_volume_aware_atr",
        },
        {
            "tag": "aware_rv0p25_base",
            "family": "rel_volume_aware_sltp",
            "rel_volume": 0.25,
            "k_sl": 2.0,
            "k_tp": 3.0,
            "sltp_risk_mode": "rel_volume_aware_atr",
        },
        {
            "tag": "aware_rv0p50_base",
            "family": "rel_volume_aware_sltp",
            "rel_volume": 0.50,
            "k_sl": 2.0,
            "k_tp": 3.0,
            "sltp_risk_mode": "rel_volume_aware_atr",
            "max_planned_loss_fraction": 0.025,
        },
        {
            "tag": "margin_aware_rv0p50_cap2p5",
            "family": "margin_aware_sltp",
            "rel_volume": 0.50,
            "k_sl": 2.0,
            "k_tp": 3.0,
            "sltp_risk_mode": "margin_aware_atr",
            "max_planned_loss_fraction": 0.025,
        },
    ]
    jobs: list[dict[str, Any]] = []
    for source_job_id in source_job_ids:
        loaded = _load_job(conn, source_job_id)
        if loaded is None:
            continue
        job_row, source_job = loaded
        source_rows = _done_subjobs(conn, job_row, max_subjobs_per_job)
        if not source_rows:
            continue
        for profile in profiles:
            if float(profile["k_tp"]) < float(profile["k_sl"]):
                raise ValueError(f"invalid SL/TP profile has TP < SL: {profile['tag']}")
            new_job_id = f"{source_job_id}_{profile['tag']}_{suffix}"
            cloned_job = json.loads(json.dumps(source_job))
            cloned_job.update(
                {
                    "job_id": new_job_id,
                    "candidate_id": new_job_id,
                    "source_job_id": source_job_id,
                    "experiment_phase": phase_id,
                    "experiment_rationale": (
                        "Trade-level risk geometry follow-up: compare fixed ATR "
                        "SL/TP, high-exposure conservative geometry, and "
                        "rel_volume-aware SL/TP while selecting checkpoints by "
                        "gap-penalized train-tail/validation RAP."
                    ),
                    "sltp_risk_geometry_followup": True,
                    "sltp_profile_tag": profile["tag"],
                    "sltp_profile_family": profile["family"],
                    "risk_adjusted_followup": True,
                    "risk_penalty_lambda": risk_lambda,
                    "risk_sizing_rel_volume": profile["rel_volume"],
                    "l1_generalization_gap_penalty_beta": l1_gap_beta,
                }
            )
            hparams = dict(cloned_job.get("hyperparameters") or {})
            hparams.update(
                {
                    "reward_plugin": "dd_penalized_reward",
                    "penalty_lambda": risk_lambda,
                    "selection_metric": "risk_adjusted_return",
                    "risk_penalty_lambda": risk_lambda,
                    "l1_generalization_gap_penalty_beta": l1_gap_beta,
                    "rel_volume": profile["rel_volume"],
                    "atr_period": 14,
                    "k_sl": profile["k_sl"],
                    "k_tp": profile["k_tp"],
                    "sltp_risk_mode": profile["sltp_risk_mode"],
                    "baseline_rel_volume": 0.05,
                    "max_risk_rel_volume": 0.50,
                    "min_reward_risk_ratio": 1.0,
                    "rel_volume_sl_shrink_alpha": 0.35,
                    "rel_volume_tp_shrink_alpha": 0.20,
                    "min_k_sl": 1.0,
                    "max_planned_loss_fraction": profile.get("max_planned_loss_fraction"),
                    "sltp_profile_tag": profile["tag"],
                    "sltp_profile_family": profile["family"],
                    "train_seed": 9,
                    "eval_seed": 9,
                    "phase_seed": 9,
                }
            )
            cloned_job["hyperparameters"] = hparams
            subjobs = []
            for row in source_rows:
                subjobs.append(
                    {
                        "subjob_id": f"{row['external_id']}_{profile['tag']}_{suffix}",
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
                        "depends_on_subjob_id": row["external_id"],
                        "warm_start_parent_subjob_id": row["external_id"],
                        "priority": int(row["priority"] or 100) + 500,
                        "source_subjob_id": row["external_id"],
                    }
                )
            cloned_job["subjobs"] = subjobs
            jobs.append(cloned_job)

    plan = {
        "schema_version": "project3_weekly_walkforward_pool_plan_v1",
        "plan_id": phase_id,
        "generated_at": utc_now(),
        "stage_c_access": "DENIED",
        "training_launched": False,
        "source_jobs": source_job_ids,
        "purpose": (
            "Compact trade-level risk control sweep over already evidenced "
            "weekly walk-forward winners."
        ),
        "risk_metric": "RAP = total_return - 0.5 * max_drawdown_fraction",
        "l1_metric": "mean(train_tail_RAP, validation_RAP) - beta * abs(train_tail_RAP - validation_RAP)",
        "l1_generalization_gap_penalty_beta": l1_gap_beta,
        "max_risk_rel_volume": 0.50,
        "baseline": {"rel_volume": 0.05, "k_sl": 2.0, "k_tp": 3.0},
        "profiles": profiles,
        "jobs": jobs,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{phase_id}.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [path]


PHASES = [
    ("eventctx_seed_robustness_phase2_v1", _phase_event_seed_robustness),
    ("eventctx_metric_window_phase3_v1", _phase_event_metric_windows),
    ("event_token_embedding_phase4_v1", _phase_event_token_embedding),
    ("event_token_transformer_phase_next_v1", _phase_event_token_transformer),
    ("asset_preset_broadening_phase5_v1", _phase_asset_preset_broadening),
    ("oracle_bc_followup_phase6_v1", _phase_oracle_bc_followup),
    ("risk_adjusted_reward_phase7_v3", _phase_risk_adjusted_followup),
    ("sltp_risk_geometry_phase8_v3", _phase_sltp_risk_geometry_followup),
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
