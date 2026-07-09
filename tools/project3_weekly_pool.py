#!/usr/bin/env python3
"""SQLite job pool for Project 3 weekly walk-forward experiments."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "project3_weekly_walkforward_pool_v1"
HELDOUT_START = datetime.fromisoformat("2025-01-01T00:00:00+00:00")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_dt(value: Any) -> datetime:
    if value is None:
        raise ValueError("missing datetime value")
    text = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT NOT NULL UNIQUE,
            candidate_id TEXT NOT NULL,
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            model_family TEXT NOT NULL,
            train_years INTEGER NOT NULL,
            training_policy TEXT NOT NULL,
            input_data_file TEXT NOT NULL,
            feature_count INTEGER NOT NULL,
            config_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS subjobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT NOT NULL UNIQUE,
            job_id INTEGER NOT NULL REFERENCES jobs(id),
            weekly_anchor_id TEXT NOT NULL,
            train_start TEXT NOT NULL,
            train_end TEXT NOT NULL,
            validation_start TEXT NOT NULL,
            validation_end TEXT NOT NULL,
            test_start TEXT NOT NULL,
            test_end TEXT NOT NULL,
            train_rows INTEGER,
            validation_rows INTEGER,
            test_rows INTEGER,
            depends_on_subjob_id TEXT,
            warm_start_parent_subjob_id TEXT,
            priority INTEGER NOT NULL DEFAULT 100,
            status TEXT NOT NULL DEFAULT 'pending',
            claimed_by TEXT,
            claimed_at TEXT,
            heartbeat_at TEXT,
            completed_at TEXT,
            config_path TEXT,
            run_dir TEXT,
            result_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS machine_heartbeats (
            machine_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            current_subjob_id TEXT,
            gpu_summary TEXT,
            message TEXT,
            heartbeat_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pool_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            subject_id TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS result_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subjob_id TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            path TEXT NOT NULL,
            size_bytes INTEGER,
            mtime TEXT,
            sha256 TEXT,
            content_tail TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(subjob_id, artifact_type, path)
        );

        CREATE INDEX IF NOT EXISTS idx_subjobs_status_priority
            ON subjobs(status, priority, id);
        CREATE INDEX IF NOT EXISTS idx_subjobs_job_id ON subjobs(job_id);
        CREATE INDEX IF NOT EXISTS idx_result_artifacts_subjob_id ON result_artifacts(subjob_id);
        CREATE INDEX IF NOT EXISTS idx_result_artifacts_type ON result_artifacts(artifact_type);
        DROP VIEW IF EXISTS weekly_result_artifact_olap;
        DROP VIEW IF EXISTS weekly_result_full_year_protocol_olap;
        DROP VIEW IF EXISTS weekly_result_validation_year_olap;
        DROP VIEW IF EXISTS weekly_result_validation_week_olap;
        DROP VIEW IF EXISTS weekly_result_test_year_olap;
        DROP VIEW IF EXISTS weekly_result_test_week_olap;
        DROP VIEW IF EXISTS weekly_result_olap;
        CREATE VIEW weekly_result_olap AS
        SELECT
            s.external_id AS subjob_id,
            j.external_id AS job_id,
            j.candidate_id,
            j.asset,
            j.timeframe,
            j.model_family,
            j.train_years,
            j.training_policy,
            j.input_data_file,
            j.feature_count,
            json_extract(j.config_json, '$.experiment_phase') AS experiment_phase,
            json_extract(j.config_json, '$.evaluation_protocol') AS evaluation_protocol,
            json_extract(j.config_json, '$.evaluation_block') AS evaluation_block,
            CAST(json_extract(j.config_json, '$.configured_validation_year') AS INTEGER) AS configured_validation_year,
            CAST(json_extract(j.config_json, '$.configured_test_year') AS INTEGER) AS configured_test_year,
            CAST(json_extract(j.config_json, '$.annual_eval_min_weeks') AS INTEGER) AS annual_eval_min_weeks,
            COALESCE(
                json_extract(j.config_json, '$.hyperparameters.sltp_profile_tag'),
                json_extract(j.config_json, '$.sltp_profile_tag'),
                json_extract(s.result_json, '$.sltp_risk_mode'),
                j.candidate_id
            ) AS olap_profile_key,
            CAST(strftime('%Y', s.validation_start) AS INTEGER) AS validation_year,
            date(s.validation_start) AS validation_week_start,
            CAST(strftime('%Y', s.test_start) AS INTEGER) AS test_year,
            date(s.test_start) AS test_week_start,
            s.weekly_anchor_id,
            s.train_start,
            s.train_end,
            s.validation_start,
            s.validation_end,
            s.test_start,
            s.test_end,
            s.status,
            s.completed_at,
            j.config_json AS job_config_json,
            s.result_json AS subjob_result_json,
            CAST(json_extract(s.result_json, '$.score') AS REAL) AS score,
            CAST(json_extract(s.result_json, '$.raw_score') AS REAL) AS raw_score,
            json_extract(s.result_json, '$.selection_metric') AS selection_metric,
            CAST(json_extract(s.result_json, '$.risk_penalty_lambda') AS REAL) AS risk_penalty_lambda,
            CAST(json_extract(s.result_json, '$.train_validation_composite_score') AS REAL) AS composite,
            CAST(json_extract(s.result_json, '$.train_validation_risk_adjusted_composite_score') AS REAL) AS risk_composite,
            CAST(COALESCE(
                json_extract(s.result_json, '$.train_validation_l1_score'),
                json_extract(s.result_json, '$.train_validation_selection_score')
            ) AS REAL) AS l1_score,
            CAST(json_extract(s.result_json, '$.train_validation_selection_mean_score') AS REAL) AS l1_mean_score,
            CAST(json_extract(s.result_json, '$.train_validation_selection_gap') AS REAL) AS l1_gap,
            CAST(json_extract(s.result_json, '$.train_validation_selection_gap_penalty') AS REAL) AS l1_gap_penalty,
            CAST(COALESCE(
                json_extract(s.result_json, '$.l1_generalization_gap_penalty_beta'),
                json_extract(j.config_json, '$.l1_generalization_gap_penalty_beta'),
                json_extract(j.config_json, '$.hyperparameters.l1_generalization_gap_penalty_beta')
            ) AS REAL) AS l1_gap_beta,
            CAST(json_extract(s.result_json, '$.train_tail_total_return') AS REAL) AS train_tail_return,
            CAST(json_extract(s.result_json, '$.validation_total_return') AS REAL) AS validation_return,
            CAST(json_extract(s.result_json, '$.test_total_return') AS REAL) AS test_return,
            CAST(json_extract(s.result_json, '$.train_tail_max_drawdown_fraction') AS REAL) AS train_tail_drawdown,
            CAST(json_extract(s.result_json, '$.validation_max_drawdown_fraction') AS REAL) AS validation_drawdown,
            CAST(json_extract(s.result_json, '$.test_max_drawdown_fraction') AS REAL) AS test_drawdown,
            CAST(json_extract(s.result_json, '$.train_tail_risk_adjusted_total_return') AS REAL) AS train_tail_rap,
            CAST(json_extract(s.result_json, '$.validation_risk_adjusted_total_return') AS REAL) AS validation_rap,
            CAST(json_extract(s.result_json, '$.test_risk_adjusted_total_return') AS REAL) AS test_rap,
            CAST(json_extract(s.result_json, '$.rel_volume') AS REAL) AS rel_volume,
            CAST(json_extract(s.result_json, '$.business_risk_fraction') AS REAL) AS business_risk_fraction,
            json_extract(s.result_json, '$.strategy_plugin') AS strategy_plugin,
            json_extract(s.result_json, '$.sltp_risk_mode') AS sltp_risk_mode,
            CAST(json_extract(s.result_json, '$.atr_period') AS INTEGER) AS atr_period,
            CAST(json_extract(s.result_json, '$.k_sl') AS REAL) AS k_sl,
            CAST(json_extract(s.result_json, '$.k_tp') AS REAL) AS k_tp,
            CAST(json_extract(s.result_json, '$.reward_risk_ratio') AS REAL) AS reward_risk_ratio,
            CAST(json_extract(s.result_json, '$.stop_loss_atr_exposure_multiplier') AS REAL) AS stop_loss_atr_exposure_multiplier,
            CAST(json_extract(s.result_json, '$.take_profit_atr_exposure_multiplier') AS REAL) AS take_profit_atr_exposure_multiplier,
            CAST(json_extract(s.result_json, '$.train_tail_trades_total') AS INTEGER) AS train_tail_trades,
            CAST(json_extract(s.result_json, '$.validation_trades_total') AS INTEGER) AS validation_trades,
            CAST(json_extract(s.result_json, '$.test_trades_total') AS INTEGER) AS test_trades,
            s.config_path,
            s.run_dir
        FROM subjobs s
        JOIN jobs j ON j.id = s.job_id
        WHERE s.result_json IS NOT NULL;

        DROP VIEW IF EXISTS weekly_result_test_week_olap;
        CREATE VIEW weekly_result_test_week_olap AS
        SELECT
            asset,
            timeframe,
            model_family,
            candidate_id,
            train_years,
            training_policy,
            input_data_file,
            feature_count,
            experiment_phase,
            evaluation_protocol,
            evaluation_block,
            configured_validation_year,
            configured_test_year,
            annual_eval_min_weeks,
            olap_profile_key,
            sltp_risk_mode,
            rel_volume,
            business_risk_fraction,
            risk_penalty_lambda,
            atr_period,
            k_sl,
            k_tp,
            reward_risk_ratio,
            stop_loss_atr_exposure_multiplier,
            take_profit_atr_exposure_multiplier,
            validation_year,
            validation_week_start,
            test_year,
            test_week_start,
            MIN(test_end) AS test_end,
            COUNT(*) AS subjob_rows,
            COUNT(DISTINCT job_id) AS candidate_rows,
            AVG(score) AS mean_score,
            AVG(raw_score) AS mean_raw_score,
            AVG(composite) AS mean_composite,
            AVG(risk_composite) AS mean_risk_composite,
            AVG(l1_score) AS mean_l1_score,
            AVG(l1_mean_score) AS mean_l1_mean_score,
            AVG(l1_gap) AS mean_l1_gap,
            AVG(l1_gap_penalty) AS mean_l1_gap_penalty,
            AVG(train_tail_return) AS mean_train_tail_return,
            AVG(validation_return) AS mean_validation_return,
            AVG(test_return) AS mean_test_return,
            AVG(train_tail_drawdown) AS mean_train_tail_drawdown,
            AVG(validation_drawdown) AS mean_validation_drawdown,
            AVG(test_drawdown) AS mean_test_drawdown,
            AVG(train_tail_rap) AS mean_train_tail_rap,
            AVG(validation_rap) AS mean_validation_rap,
            AVG(test_rap) AS mean_test_rap,
            MIN(test_rap) AS min_test_rap,
            MAX(test_rap) AS max_test_rap,
            AVG(train_tail_trades) AS mean_train_tail_trades,
            AVG(validation_trades) AS mean_validation_trades,
            AVG(test_trades) AS mean_test_trades,
            MIN(completed_at) AS first_completed_at,
            MAX(completed_at) AS last_completed_at
        FROM weekly_result_olap
        WHERE test_rap IS NOT NULL
        GROUP BY
            asset,
            timeframe,
            model_family,
            candidate_id,
            train_years,
            training_policy,
            input_data_file,
            feature_count,
            experiment_phase,
            evaluation_protocol,
            evaluation_block,
            configured_validation_year,
            configured_test_year,
            annual_eval_min_weeks,
            olap_profile_key,
            sltp_risk_mode,
            rel_volume,
            business_risk_fraction,
            risk_penalty_lambda,
            atr_period,
            k_sl,
            k_tp,
            reward_risk_ratio,
            stop_loss_atr_exposure_multiplier,
            take_profit_atr_exposure_multiplier,
            validation_year,
            validation_week_start,
            test_year,
            test_week_start;

        DROP VIEW IF EXISTS weekly_result_validation_week_olap;
        CREATE VIEW weekly_result_validation_week_olap AS
        SELECT
            asset,
            timeframe,
            model_family,
            candidate_id,
            train_years,
            training_policy,
            input_data_file,
            feature_count,
            experiment_phase,
            evaluation_protocol,
            evaluation_block,
            configured_validation_year,
            configured_test_year,
            annual_eval_min_weeks,
            olap_profile_key,
            sltp_risk_mode,
            rel_volume,
            business_risk_fraction,
            risk_penalty_lambda,
            atr_period,
            k_sl,
            k_tp,
            reward_risk_ratio,
            stop_loss_atr_exposure_multiplier,
            take_profit_atr_exposure_multiplier,
            validation_year,
            validation_week_start,
            MIN(validation_end) AS validation_end,
            COUNT(*) AS subjob_rows,
            COUNT(DISTINCT job_id) AS candidate_rows,
            AVG(score) AS mean_score,
            AVG(raw_score) AS mean_raw_score,
            AVG(composite) AS mean_composite,
            AVG(risk_composite) AS mean_risk_composite,
            AVG(l1_score) AS mean_l1_score,
            AVG(l1_mean_score) AS mean_l1_mean_score,
            AVG(l1_gap) AS mean_l1_gap,
            AVG(l1_gap_penalty) AS mean_l1_gap_penalty,
            AVG(train_tail_return) AS mean_train_tail_return,
            AVG(validation_return) AS mean_validation_return,
            AVG(train_tail_drawdown) AS mean_train_tail_drawdown,
            AVG(validation_drawdown) AS mean_validation_drawdown,
            AVG(train_tail_rap) AS mean_train_tail_rap,
            AVG(validation_rap) AS mean_validation_rap,
            MIN(validation_rap) AS min_validation_rap,
            MAX(validation_rap) AS max_validation_rap,
            AVG(train_tail_trades) AS mean_train_tail_trades,
            AVG(validation_trades) AS mean_validation_trades,
            MIN(completed_at) AS first_completed_at,
            MAX(completed_at) AS last_completed_at
        FROM weekly_result_olap
        WHERE validation_rap IS NOT NULL
        GROUP BY
            asset,
            timeframe,
            model_family,
            candidate_id,
            train_years,
            training_policy,
            input_data_file,
            feature_count,
            experiment_phase,
            evaluation_protocol,
            evaluation_block,
            configured_validation_year,
            configured_test_year,
            annual_eval_min_weeks,
            olap_profile_key,
            sltp_risk_mode,
            rel_volume,
            business_risk_fraction,
            risk_penalty_lambda,
            atr_period,
            k_sl,
            k_tp,
            reward_risk_ratio,
            stop_loss_atr_exposure_multiplier,
            take_profit_atr_exposure_multiplier,
            validation_year,
            validation_week_start;

        DROP VIEW IF EXISTS weekly_result_test_year_olap;
        CREATE VIEW weekly_result_test_year_olap AS
        SELECT
            asset,
            timeframe,
            model_family,
            candidate_id,
            train_years,
            training_policy,
            input_data_file,
            feature_count,
            experiment_phase,
            evaluation_protocol,
            evaluation_block,
            configured_validation_year,
            configured_test_year,
            annual_eval_min_weeks,
            olap_profile_key,
            sltp_risk_mode,
            rel_volume,
            business_risk_fraction,
            risk_penalty_lambda,
            atr_period,
            k_sl,
            k_tp,
            reward_risk_ratio,
            stop_loss_atr_exposure_multiplier,
            take_profit_atr_exposure_multiplier,
            test_year,
            COUNT(*) AS unique_test_weeks,
            SUM(subjob_rows) AS subjob_rows,
            SUM(candidate_rows) AS candidate_rows,
            MIN(test_week_start) AS first_test_week,
            MAX(test_week_start) AS last_test_week,
            AVG(mean_score) AS mean_weekly_score,
            AVG(mean_raw_score) AS mean_weekly_raw_score,
            AVG(mean_composite) AS mean_weekly_composite,
            AVG(mean_risk_composite) AS mean_weekly_risk_composite,
            AVG(mean_l1_score) AS mean_weekly_l1_score,
            AVG(mean_l1_mean_score) AS mean_weekly_l1_mean_score,
            AVG(mean_l1_gap) AS mean_weekly_l1_gap,
            AVG(mean_l1_gap_penalty) AS mean_weekly_l1_gap_penalty,
            AVG(mean_train_tail_return) AS mean_weekly_train_tail_return,
            AVG(mean_validation_return) AS mean_weekly_validation_return,
            AVG(mean_test_return) AS mean_weekly_test_return,
            AVG(mean_train_tail_drawdown) AS mean_weekly_train_tail_drawdown,
            AVG(mean_validation_drawdown) AS mean_weekly_validation_drawdown,
            AVG(mean_test_drawdown) AS mean_weekly_test_drawdown,
            AVG(mean_train_tail_rap) AS mean_weekly_train_tail_rap,
            AVG(mean_validation_rap) AS mean_weekly_validation_rap,
            AVG(mean_test_rap) AS mean_weekly_test_rap,
            SUM(mean_test_return) AS sum_weekly_test_return,
            SUM(mean_test_return) AS observed_test_return,
            52.0 * AVG(mean_test_return) AS projected_annual_test_return_52w,
            SUM(mean_test_return) AS annual_test_return,
            SUM(mean_test_drawdown) AS sum_weekly_test_drawdown,
            SUM(mean_test_rap) AS sum_weekly_test_rap,
            SUM(mean_test_rap) AS observed_test_rap,
            52.0 * AVG(mean_test_rap) AS projected_annual_test_rap_52w,
            SUM(mean_test_rap) AS annual_test_rap,
            MIN(mean_test_rap) AS worst_weekly_test_rap,
            MAX(mean_test_rap) AS best_weekly_test_rap,
            AVG(mean_test_trades) AS mean_weekly_test_trades,
            ROUND(COUNT(*) / 52.0, 4) AS coverage_ratio_52w,
            CASE WHEN COUNT(*) >= COALESCE(annual_eval_min_weeks, 48) THEN 1 ELSE 0 END AS has_near_full_year_coverage,
            MIN(first_completed_at) AS first_completed_at,
            MAX(last_completed_at) AS last_completed_at
        FROM weekly_result_test_week_olap
        GROUP BY
            asset,
            timeframe,
            model_family,
            candidate_id,
            train_years,
            training_policy,
            input_data_file,
            feature_count,
            experiment_phase,
            evaluation_protocol,
            evaluation_block,
            configured_validation_year,
            configured_test_year,
            annual_eval_min_weeks,
            olap_profile_key,
            sltp_risk_mode,
            rel_volume,
            business_risk_fraction,
            risk_penalty_lambda,
            atr_period,
            k_sl,
            k_tp,
            reward_risk_ratio,
            stop_loss_atr_exposure_multiplier,
            take_profit_atr_exposure_multiplier,
            test_year;

        DROP VIEW IF EXISTS weekly_result_validation_year_olap;
        CREATE VIEW weekly_result_validation_year_olap AS
        SELECT
            asset,
            timeframe,
            model_family,
            candidate_id,
            train_years,
            training_policy,
            input_data_file,
            feature_count,
            experiment_phase,
            evaluation_protocol,
            evaluation_block,
            configured_validation_year,
            configured_test_year,
            annual_eval_min_weeks,
            olap_profile_key,
            sltp_risk_mode,
            rel_volume,
            business_risk_fraction,
            risk_penalty_lambda,
            atr_period,
            k_sl,
            k_tp,
            reward_risk_ratio,
            stop_loss_atr_exposure_multiplier,
            take_profit_atr_exposure_multiplier,
            validation_year,
            COUNT(*) AS unique_validation_weeks,
            SUM(subjob_rows) AS subjob_rows,
            SUM(candidate_rows) AS candidate_rows,
            MIN(validation_week_start) AS first_validation_week,
            MAX(validation_week_start) AS last_validation_week,
            AVG(mean_score) AS mean_weekly_score,
            AVG(mean_raw_score) AS mean_weekly_raw_score,
            AVG(mean_composite) AS mean_weekly_composite,
            AVG(mean_risk_composite) AS mean_weekly_risk_composite,
            AVG(mean_l1_score) AS mean_weekly_l1_score,
            AVG(mean_l1_mean_score) AS mean_weekly_l1_mean_score,
            AVG(mean_l1_gap) AS mean_weekly_l1_gap,
            AVG(mean_l1_gap_penalty) AS mean_weekly_l1_gap_penalty,
            AVG(mean_train_tail_return) AS mean_weekly_train_tail_return,
            AVG(mean_validation_return) AS mean_weekly_validation_return,
            AVG(mean_train_tail_drawdown) AS mean_weekly_train_tail_drawdown,
            AVG(mean_validation_drawdown) AS mean_weekly_validation_drawdown,
            AVG(mean_train_tail_rap) AS mean_weekly_train_tail_rap,
            AVG(mean_validation_rap) AS mean_weekly_validation_rap,
            SUM(mean_validation_return) AS sum_weekly_validation_return,
            SUM(mean_validation_return) AS observed_validation_return,
            52.0 * AVG(mean_validation_return) AS projected_annual_validation_return_52w,
            SUM(mean_validation_return) AS annual_validation_return,
            SUM(mean_validation_drawdown) AS sum_weekly_validation_drawdown,
            SUM(mean_validation_rap) AS sum_weekly_validation_rap,
            SUM(mean_validation_rap) AS observed_validation_rap,
            52.0 * AVG(mean_validation_rap) AS projected_annual_validation_rap_52w,
            SUM(mean_validation_rap) AS annual_validation_rap,
            MIN(mean_validation_rap) AS worst_weekly_validation_rap,
            MAX(mean_validation_rap) AS best_weekly_validation_rap,
            AVG(mean_validation_trades) AS mean_weekly_validation_trades,
            ROUND(COUNT(*) / 52.0, 4) AS coverage_ratio_52w,
            CASE WHEN COUNT(*) >= COALESCE(annual_eval_min_weeks, 48) THEN 1 ELSE 0 END AS has_near_full_year_coverage,
            MIN(first_completed_at) AS first_completed_at,
            MAX(last_completed_at) AS last_completed_at
        FROM weekly_result_validation_week_olap
        GROUP BY
            asset,
            timeframe,
            model_family,
            candidate_id,
            train_years,
            training_policy,
            input_data_file,
            feature_count,
            experiment_phase,
            evaluation_protocol,
            evaluation_block,
            configured_validation_year,
            configured_test_year,
            annual_eval_min_weeks,
            olap_profile_key,
            sltp_risk_mode,
            rel_volume,
            business_risk_fraction,
            risk_penalty_lambda,
            atr_period,
            k_sl,
            k_tp,
            reward_risk_ratio,
            stop_loss_atr_exposure_multiplier,
            take_profit_atr_exposure_multiplier,
            validation_year;

        DROP VIEW IF EXISTS weekly_result_full_year_protocol_olap;
        CREATE VIEW weekly_result_full_year_protocol_olap AS
        SELECT
            'validation_year' AS metric_block,
            asset,
            timeframe,
            model_family,
            candidate_id,
            train_years,
            training_policy,
            input_data_file,
            feature_count,
            experiment_phase,
            evaluation_protocol,
            evaluation_block,
            configured_validation_year,
            configured_test_year,
            annual_eval_min_weeks,
            olap_profile_key,
            sltp_risk_mode,
            rel_volume,
            business_risk_fraction,
            risk_penalty_lambda,
            atr_period,
            k_sl,
            k_tp,
            reward_risk_ratio,
            stop_loss_atr_exposure_multiplier,
            take_profit_atr_exposure_multiplier,
            validation_year AS metric_year,
            unique_validation_weeks AS unique_weeks,
            subjob_rows,
            first_validation_week AS first_week,
            last_validation_week AS last_week,
            mean_weekly_l1_score,
            mean_weekly_l1_mean_score,
            mean_weekly_l1_gap,
            mean_weekly_l1_gap_penalty,
            mean_weekly_validation_return AS mean_weekly_return,
            sum_weekly_validation_return AS sum_weekly_return,
            observed_validation_return AS observed_return,
            projected_annual_validation_return_52w AS projected_annual_return_52w,
            annual_validation_return AS annual_return,
            mean_weekly_validation_drawdown AS mean_weekly_drawdown,
            sum_weekly_validation_drawdown AS sum_weekly_drawdown,
            mean_weekly_validation_rap AS mean_weekly_rap,
            sum_weekly_validation_rap AS sum_weekly_rap,
            observed_validation_rap AS observed_rap,
            projected_annual_validation_rap_52w AS projected_annual_rap_52w,
            annual_validation_rap AS annual_rap,
            worst_weekly_validation_rap AS worst_weekly_rap,
            best_weekly_validation_rap AS best_weekly_rap,
            mean_weekly_validation_trades AS mean_weekly_trades,
            coverage_ratio_52w,
            has_near_full_year_coverage,
            first_completed_at,
            last_completed_at
        FROM weekly_result_validation_year_olap
        WHERE evaluation_protocol = 'full_year_validation_test_v1'
          AND evaluation_block = 'validation_year'
        UNION ALL
        SELECT
            'test_year' AS metric_block,
            asset,
            timeframe,
            model_family,
            candidate_id,
            train_years,
            training_policy,
            input_data_file,
            feature_count,
            experiment_phase,
            evaluation_protocol,
            evaluation_block,
            configured_validation_year,
            configured_test_year,
            annual_eval_min_weeks,
            olap_profile_key,
            sltp_risk_mode,
            rel_volume,
            business_risk_fraction,
            risk_penalty_lambda,
            atr_period,
            k_sl,
            k_tp,
            reward_risk_ratio,
            stop_loss_atr_exposure_multiplier,
            take_profit_atr_exposure_multiplier,
            test_year AS metric_year,
            unique_test_weeks AS unique_weeks,
            subjob_rows,
            first_test_week AS first_week,
            last_test_week AS last_week,
            mean_weekly_l1_score,
            mean_weekly_l1_mean_score,
            mean_weekly_l1_gap,
            mean_weekly_l1_gap_penalty,
            mean_weekly_test_return AS mean_weekly_return,
            sum_weekly_test_return AS sum_weekly_return,
            observed_test_return AS observed_return,
            projected_annual_test_return_52w AS projected_annual_return_52w,
            annual_test_return AS annual_return,
            mean_weekly_test_drawdown AS mean_weekly_drawdown,
            sum_weekly_test_drawdown AS sum_weekly_drawdown,
            mean_weekly_test_rap AS mean_weekly_rap,
            sum_weekly_test_rap AS sum_weekly_rap,
            observed_test_rap AS observed_rap,
            projected_annual_test_rap_52w AS projected_annual_rap_52w,
            annual_test_rap AS annual_rap,
            worst_weekly_test_rap AS worst_weekly_rap,
            best_weekly_test_rap AS best_weekly_rap,
            mean_weekly_test_trades AS mean_weekly_trades,
            coverage_ratio_52w,
            has_near_full_year_coverage,
            first_completed_at,
            last_completed_at
        FROM weekly_result_test_year_olap
        WHERE evaluation_protocol = 'full_year_validation_test_v1'
          AND evaluation_block = 'test_year';

        DROP VIEW IF EXISTS weekly_result_artifact_olap;
        CREATE VIEW weekly_result_artifact_olap AS
        SELECT
            ra.subjob_id,
            j.external_id AS job_id,
            j.candidate_id,
            j.asset,
            j.timeframe,
            j.model_family,
            j.train_years,
            j.training_policy,
            j.input_data_file,
            j.feature_count,
            json_extract(j.config_json, '$.experiment_phase') AS experiment_phase,
            s.weekly_anchor_id,
            s.train_start,
            s.train_end,
            s.validation_start,
            s.validation_end,
            s.test_start,
            s.test_end,
            CAST(strftime('%Y', s.test_start) AS INTEGER) AS test_year,
            date(s.test_start) AS test_week_start,
            s.status AS subjob_status,
            s.completed_at,
            s.config_path,
            s.run_dir,
            ra.artifact_type,
            ra.path,
            ra.size_bytes,
            ra.mtime,
            ra.sha256,
            ra.content_tail,
            ra.metadata_json,
            ra.updated_at AS artifact_updated_at
        FROM result_artifacts ra
        LEFT JOIN subjobs s ON s.external_id = ra.subjob_id
        LEFT JOIN jobs j ON j.id = s.job_id;
        """
    )
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(subjobs)")
    }
    for column in ("depends_on_subjob_id", "warm_start_parent_subjob_id"):
        if column not in existing:
            conn.execute(f"ALTER TABLE subjobs ADD COLUMN {column} TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_subjobs_depends_on ON subjobs(depends_on_subjob_id)"
    )
    conn.commit()


def _validate_plan(plan: dict[str, Any], known_parent_ids: set[str] | None = None) -> None:
    known_parent_ids = known_parent_ids or set()
    if plan.get("stage_c_access") != "DENIED":
        raise ValueError("plan must carry stage_c_access='DENIED'")
    if plan.get("training_launched") not in (False, None):
        raise ValueError("plan must carry training_launched=false before enqueue")
    subjob_ids = {
        subjob.get("subjob_id")
        for job in plan.get("jobs", [])
        for subjob in job.get("subjobs", [])
    }
    for job in plan.get("jobs", []):
        for subjob in job.get("subjobs", []):
            parent = subjob.get("depends_on_subjob_id") or subjob.get("warm_start_parent_subjob_id")
            if parent and parent not in subjob_ids and parent not in known_parent_ids:
                raise ValueError(
                    f"subjob {subjob.get('subjob_id')} depends on unknown parent {parent}"
                )
            dates = [
                _parse_dt(subjob["train_start"]),
                _parse_dt(subjob["train_end"]),
                _parse_dt(subjob["validation_start"]),
                _parse_dt(subjob["validation_end"]),
                _parse_dt(subjob["test_start"]),
                _parse_dt(subjob["test_end"]),
            ]
            if not dates[0] < dates[1] <= dates[2] < dates[3] <= dates[4] < dates[5]:
                raise ValueError(f"invalid split ordering in {subjob.get('subjob_id')}")
            if dates[5] > HELDOUT_START:
                raise ValueError(f"subjob {subjob.get('subjob_id')} reaches Stage C heldout")


def enqueue_plan(conn: sqlite3.Connection, plan_path: str | Path) -> dict[str, int]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    known_parent_ids = {
        str(row["external_id"])
        for row in conn.execute("SELECT external_id FROM subjobs WHERE status='done'")
    }
    _validate_plan(plan, known_parent_ids)
    now = utc_now()
    inserted_jobs = 0
    inserted_subjobs = 0
    with conn:
        for job in plan.get("jobs", []):
            feature_columns = job.get("feature_columns") or job.get("selected_features") or []
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    external_id, candidate_id, asset, timeframe, model_family,
                    train_years, training_policy, input_data_file, feature_count,
                    config_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job["job_id"],
                    job.get("candidate_id", job["job_id"]),
                    job["asset"],
                    job["timeframe"],
                    job.get("model_family", "sac"),
                    int(job["train_years"]),
                    job.get("training_policy", "scratch_n_years"),
                    job["input_data_file"],
                    len(feature_columns),
                    _json(job),
                    now,
                    now,
                ),
            )
            inserted_jobs += cur.rowcount
            row = conn.execute("SELECT id FROM jobs WHERE external_id = ?", (job["job_id"],)).fetchone()
            if row is None:
                raise RuntimeError(f"job disappeared after enqueue: {job['job_id']}")
            for subjob in job.get("subjobs", []):
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO subjobs (
                        external_id, job_id, weekly_anchor_id,
                        train_start, train_end, validation_start, validation_end,
                        test_start, test_end, train_rows, validation_rows, test_rows,
                        depends_on_subjob_id, warm_start_parent_subjob_id,
                        priority, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        subjob["subjob_id"],
                        row["id"],
                        subjob["weekly_anchor_id"],
                        subjob["train_start"],
                        subjob["train_end"],
                        subjob["validation_start"],
                        subjob["validation_end"],
                        subjob["test_start"],
                        subjob["test_end"],
                        subjob.get("train_rows"),
                        subjob.get("validation_rows"),
                        subjob.get("test_rows"),
                        subjob.get("depends_on_subjob_id"),
                        subjob.get("warm_start_parent_subjob_id"),
                        int(subjob.get("priority", 100)),
                        now,
                        now,
                    ),
                )
                inserted_subjobs += cur.rowcount
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("enqueue_plan", str(plan_path), _json({"jobs": inserted_jobs, "subjobs": inserted_subjobs}), now),
        )
    return {"inserted_jobs": inserted_jobs, "inserted_subjobs": inserted_subjobs}


def claim_subjob(conn: sqlite3.Connection, machine_id: str) -> dict[str, Any] | None:
    now = utc_now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            """
            SELECT s.*, j.external_id AS job_external_id, j.config_json
            FROM subjobs s
            JOIN jobs j ON j.id = s.job_id
            WHERE s.status = 'pending'
              AND (
                s.depends_on_subjob_id IS NULL
                OR EXISTS (
                    SELECT 1
                    FROM subjobs parent
                    WHERE parent.external_id = s.depends_on_subjob_id
                      AND parent.status = 'done'
                )
              )
            ORDER BY s.priority ASC, s.id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO machine_heartbeats(machine_id, status, current_subjob_id, heartbeat_at)
                VALUES (?, 'idle', NULL, ?)
                ON CONFLICT(machine_id) DO UPDATE SET
                    status='idle', current_subjob_id=NULL, heartbeat_at=excluded.heartbeat_at
                """,
                (machine_id, now),
            )
            conn.commit()
            return None
        conn.execute(
            """
            UPDATE subjobs
            SET status='running', claimed_by=?, claimed_at=?, heartbeat_at=?, updated_at=?
            WHERE id=?
            """,
            (machine_id, now, now, now, row["id"]),
        )
        conn.execute(
            """
            INSERT INTO machine_heartbeats(machine_id, status, current_subjob_id, heartbeat_at)
            VALUES (?, 'running', ?, ?)
            ON CONFLICT(machine_id) DO UPDATE SET
                status='running',
                current_subjob_id=excluded.current_subjob_id,
                heartbeat_at=excluded.heartbeat_at
            """,
            (machine_id, row["external_id"], now),
        )
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("claim", row["external_id"], _json({"machine_id": machine_id}), now),
        )
        conn.commit()
        out = dict(row)
        out["job_config"] = json.loads(row["config_json"])
        out.pop("config_json", None)
        return out
    except Exception:
        conn.rollback()
        raise


def heartbeat(
    conn: sqlite3.Connection,
    machine_id: str,
    subjob_id: str | None,
    status: str,
    message: str | None = None,
    gpu_summary: str | None = None,
) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            """
            INSERT INTO machine_heartbeats(machine_id, status, current_subjob_id, message, gpu_summary, heartbeat_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(machine_id) DO UPDATE SET
                status=excluded.status,
                current_subjob_id=excluded.current_subjob_id,
                message=excluded.message,
                gpu_summary=excluded.gpu_summary,
                heartbeat_at=excluded.heartbeat_at
            """,
            (machine_id, status, subjob_id, message, gpu_summary, now),
        )
        if subjob_id:
            conn.execute(
                "UPDATE subjobs SET heartbeat_at=?, updated_at=? WHERE external_id=?",
                (now, now, subjob_id),
            )


def complete_subjob(conn: sqlite3.Connection, subjob_id: str, result: dict[str, Any]) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            """
            UPDATE subjobs
            SET status='done', completed_at=?, heartbeat_at=?, updated_at=?, result_json=?
            WHERE external_id=?
            """,
            (now, now, now, _json(result), subjob_id),
        )
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("complete", subjob_id, _json(result), now),
        )


def fail_subjob(conn: sqlite3.Connection, subjob_id: str, error: str) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            """
            UPDATE subjobs
            SET status='failed', completed_at=?, heartbeat_at=?, updated_at=?, error=?
            WHERE external_id=?
            """,
            (now, now, now, error, subjob_id),
        )
        conn.execute(
            "INSERT INTO pool_events(event_type, subject_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("fail", subjob_id, _json({"error": error}), now),
        )


def status(conn: sqlite3.Connection) -> dict[str, Any]:
    counts = {
        row["status"]: row["n"]
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM subjobs GROUP BY status")
    }
    jobs = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
    machines = [dict(row) for row in conn.execute("SELECT * FROM machine_heartbeats ORDER BY machine_id")]
    running = [
        dict(row)
        for row in conn.execute(
            """
            SELECT s.external_id, s.weekly_anchor_id, s.claimed_by, s.heartbeat_at,
                   s.train_start, s.train_end, s.validation_start, s.validation_end, s.test_start, s.test_end,
                   s.depends_on_subjob_id, s.warm_start_parent_subjob_id,
                   j.asset, j.timeframe, j.model_family, j.train_years, j.training_policy
            FROM subjobs s
            JOIN jobs j ON j.id = s.job_id
            WHERE s.status='running'
            ORDER BY s.claimed_at
            """
        )
    ]
    machine_conflicts = [
        dict(row)
        for row in conn.execute(
            """
            SELECT
                claimed_by AS machine_id,
                COUNT(*) AS running_subjobs,
                GROUP_CONCAT(external_id, '\n') AS subjob_ids,
                MIN(claimed_at) AS oldest_claimed_at,
                MAX(heartbeat_at) AS newest_heartbeat_at
            FROM subjobs
            WHERE status = 'running'
              AND claimed_by IS NOT NULL
            GROUP BY claimed_by
            HAVING COUNT(*) > 1
            ORDER BY claimed_by
            """
        )
    ]
    stale_running = [
        dict(row)
        for row in conn.execute(
            """
            SELECT
                external_id,
                claimed_by,
                claimed_at,
                heartbeat_at,
                ROUND((julianday('now') - julianday(heartbeat_at)) * 24 * 60, 2) AS minutes_since_heartbeat
            FROM subjobs
            WHERE status = 'running'
              AND heartbeat_at IS NOT NULL
              AND (julianday('now') - julianday(heartbeat_at)) * 24 * 60 >= 3.0
            ORDER BY minutes_since_heartbeat DESC
            """
        )
    ]
    try:
        annual_protocol_best = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    metric_block,
                    asset,
                    timeframe,
                    model_family,
                    candidate_id,
                    training_policy,
                    experiment_phase,
                    metric_year,
                    unique_weeks,
                    coverage_ratio_52w,
                    has_near_full_year_coverage,
                    olap_profile_key,
                    sltp_risk_mode,
                    rel_volume,
                    k_sl,
                    k_tp,
                    mean_weekly_return,
                    observed_return,
                    projected_annual_return_52w,
                    annual_return,
                    mean_weekly_rap,
                    observed_rap,
                    projected_annual_rap_52w,
                    annual_rap,
                    mean_weekly_drawdown,
                    mean_weekly_l1_score,
                    mean_weekly_l1_gap
                FROM weekly_result_full_year_protocol_olap
                WHERE has_near_full_year_coverage = 1
                ORDER BY metric_block, annual_rap DESC
                LIMIT 12
                """
            )
        ]
    except sqlite3.OperationalError:
        annual_protocol_best = []
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "job_count": jobs,
        "subjob_counts": counts,
        "machines": machines,
        "running": running,
        "machine_conflicts": machine_conflicts,
        "stale_running": stale_running,
        "annual_protocol_best": annual_protocol_best,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("--plan", required=True)
    claim = sub.add_parser("claim")
    claim.add_argument("--machine-id", required=True)
    hb = sub.add_parser("heartbeat")
    hb.add_argument("--machine-id", required=True)
    hb.add_argument("--subjob-id")
    hb.add_argument("--status", default="running")
    hb.add_argument("--message")
    hb.add_argument("--gpu-summary")
    done = sub.add_parser("complete")
    done.add_argument("--subjob-id", required=True)
    done.add_argument("--result-json", default="{}")
    fail = sub.add_parser("fail")
    fail.add_argument("--subjob-id", required=True)
    fail.add_argument("--error", required=True)
    sub.add_parser("status")
    args = ap.parse_args()

    conn = connect(args.db)
    init_db(conn)
    if args.cmd == "init":
        print(json.dumps({"ok": True, "db": args.db}, indent=2))
    elif args.cmd == "enqueue":
        print(json.dumps(enqueue_plan(conn, args.plan), indent=2))
    elif args.cmd == "claim":
        print(json.dumps(claim_subjob(conn, args.machine_id), indent=2))
    elif args.cmd == "heartbeat":
        heartbeat(conn, args.machine_id, args.subjob_id, args.status, args.message, args.gpu_summary)
        print(json.dumps({"ok": True}, indent=2))
    elif args.cmd == "complete":
        complete_subjob(conn, args.subjob_id, json.loads(args.result_json))
        print(json.dumps({"ok": True}, indent=2))
    elif args.cmd == "fail":
        fail_subjob(conn, args.subjob_id, args.error)
        print(json.dumps({"ok": True}, indent=2))
    elif args.cmd == "status":
        print(json.dumps(status(conn), indent=2))


if __name__ == "__main__":
    main()
