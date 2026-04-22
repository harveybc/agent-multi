#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
etl_migrate_v2.py -- reliable ETL for experiment configs and per-horizon metrics.

Replaces prior version with:
 - robust upsert_experiment (fills extracted columns used by Metabase GUI)
 - load_results_summary (legacy summary loader, kept)
 - load_performance_metrics (new; fills fact_performance from per-horizon CSV)
 - clear, verbose logging and defensive parsing
"""

# Standard library imports
import argparse                     # CLI parsing
import json                         # config JSON
import logging                      # structured logs
import os                           # env vars
import re                           # parsing Metric text
import sys                          # exit codes
from typing import Dict, Optional   # typing hints

# Third-party imports
import pandas as pd                 # CSV parsing
from sqlalchemy import create_engine, text  # DB access

# Constants
SCHEMA = "public"

# Logging config (honor LOG_LEVEL)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s %(message)s")

# Regex to match metric lines like "Train MAE H1" (case-insensitive)
_METRIC_RE = re.compile(r'^\s*(Train|Validation|Test)\s+(.+?)\s+H(\d+)\s*$', flags=re.IGNORECASE)

# -----------------------------
# Database connection builder
# -----------------------------
def build_engine_from_pg_env():
    """Build SQLAlchemy engine using PG* env vars with sane defaults (local dev)."""
    host = os.getenv("PGHOST", "127.0.0.1")
    port = int(os.getenv("PGPORT", "5432"))
    dbname = os.getenv("PGDATABASE", "predictor_olap")
    user = os.getenv("PGUSER", "metabase")
    password = os.getenv("PGPASSWORD", "metabase_pass")
    dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    safe_dsn = f"postgresql://{user}:*****@{host}:{port}/{dbname}"
    engine = create_engine(dsn, pool_pre_ping=True, future=True)
    logging.info("Connected to PostgreSQL via DSN: %s", safe_dsn)
    return engine

# -----------------------------
# Ensure schema + tables + seeds
# -----------------------------
def ensure_schema_and_tables(engine):
        """
        Idempotent DDL for required tables and basic seed data (splits/horizons/metrics).
        This function is safe to run many times.
        """
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {SCHEMA};

        CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_project (
            project_key TEXT PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_phase (
            phase_key   TEXT PRIMARY KEY,
            project_key TEXT NOT NULL REFERENCES {SCHEMA}.dim_project(project_key) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_experiment (
            experiment_key TEXT PRIMARY KEY,
            project_key    TEXT NOT NULL REFERENCES {SCHEMA}.dim_project(project_key) ON DELETE CASCADE,
            phase_key      TEXT NOT NULL REFERENCES {SCHEMA}.dim_phase(phase_key)   ON DELETE CASCADE,
            config_json    JSONB,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            -- extracted numeric/filter fields
            max_steps_train    INTEGER,
            max_steps_test     INTEGER,
            intermediate_layers INTEGER,
            initial_layer_size INTEGER,
            layer_size_divisor INTEGER,
            learning_rate      DOUBLE PRECISION,
            activation         TEXT,
            l2_reg             DOUBLE PRECISION,
            kl_weight          DOUBLE PRECISION,
            kl_anneal_epochs   INTEGER,
            early_patience     INTEGER,
            start_from_epoch   INTEGER,
            use_returns        BOOLEAN,
            mmd_lambda         DOUBLE PRECISION,
            window_size        INTEGER,
            batch_size         INTEGER,
            min_delta          DOUBLE PRECISION,
            epochs             INTEGER,
            stl_period         INTEGER,
            predicted_horizons JSONB,
            use_stl            BOOLEAN,
            use_wavelets       BOOLEAN,
            use_multi_tapper   BOOLEAN,
            -- categorical/boolean fields (Metabase GUI friendly)
            predictor_plugin   TEXT,
            optimizer_plugin   TEXT,
            pipeline_plugin    TEXT,
            preprocessor_plugin TEXT,
            use_strategy       BOOLEAN,
            use_daily          BOOLEAN,
            mc_samples         INTEGER
        );

        CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_dataset_split (
            split_key TEXT PRIMARY KEY,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_horizon (
            horizon_key INTEGER PRIMARY KEY,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_metric (
            metric_key TEXT PRIMARY KEY,
            metric_type TEXT,
            direction   TEXT
        );

        CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_performance (
            id             BIGSERIAL PRIMARY KEY,
            experiment_key TEXT NOT NULL REFERENCES {SCHEMA}.dim_experiment(experiment_key) ON DELETE CASCADE,
            phase_key      TEXT NOT NULL REFERENCES {SCHEMA}.dim_phase(phase_key) ON DELETE CASCADE,
            split_key      TEXT NOT NULL REFERENCES {SCHEMA}.dim_dataset_split(split_key),
            horizon_key    INTEGER NOT NULL REFERENCES {SCHEMA}.dim_horizon(horizon_key),
            metric_key     TEXT NOT NULL REFERENCES {SCHEMA}.dim_metric(metric_key),
            avg_value      DOUBLE PRECISION,
            std_dev        DOUBLE PRECISION,
            min_value      DOUBLE PRECISION,
            max_value      DOUBLE PRECISION,
            loaded_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (experiment_key, phase_key, split_key, horizon_key, metric_key)
        );

        CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_results_summary (
            id             BIGSERIAL PRIMARY KEY,
            experiment_key TEXT NOT NULL REFERENCES {SCHEMA}.dim_experiment(experiment_key) ON DELETE CASCADE,
            phase_key      TEXT,
            metric         TEXT NOT NULL,
            avg_value      DOUBLE PRECISION,
            std_dev        DOUBLE PRECISION,
            min_value      DOUBLE PRECISION,
            max_value      DOUBLE PRECISION,
            loaded_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (experiment_key, metric)
        );

        CREATE INDEX IF NOT EXISTS idx_fact_perf_experiment
            ON {SCHEMA}.fact_performance (experiment_key);

        CREATE INDEX IF NOT EXISTS idx_fact_perf_metric
            ON {SCHEMA}.fact_performance (metric_key);

        CREATE INDEX IF NOT EXISTS idx_dim_experiment_cfg_gin
            ON {SCHEMA}.dim_experiment USING gin (config_json);
        """
        with engine.begin() as conn:
                conn.exec_driver_sql(ddl)
                # Backfill/migrate existing tables that may predate new columns
                # These are safe, idempotent ALTERs for older databases.
                try:
                        conn.exec_driver_sql(f"""
                                ALTER TABLE {SCHEMA}.fact_performance
                                    ADD COLUMN IF NOT EXISTS avg_value   DOUBLE PRECISION,
                                    ADD COLUMN IF NOT EXISTS std_dev     DOUBLE PRECISION,
                                    ADD COLUMN IF NOT EXISTS min_value   DOUBLE PRECISION,
                                    ADD COLUMN IF NOT EXISTS max_value   DOUBLE PRECISION,
                                    ADD COLUMN IF NOT EXISTS loaded_at   TIMESTAMPTZ NOT NULL DEFAULT NOW();
                        """)
                except Exception as exc:
                        logging.warning("Schema migrate (fact_performance columns) skipped or failed: %s", exc)

                # Legacy column: metric_value (older schema used a single value column)
                # - Allow NULL to prevent insert failures when not provided
                # - Backfill from avg_value when available
                try:
                        # Drop NOT NULL if present (idempotent using information_schema)
                        conn.exec_driver_sql(f"""
                                DO $$
                                BEGIN
                                    IF EXISTS (
                                        SELECT 1
                                        FROM information_schema.columns
                                        WHERE table_schema = '{SCHEMA}'
                                            AND table_name   = 'fact_performance'
                                            AND column_name  = 'metric_value'
                                            AND is_nullable  = 'NO'
                                    ) THEN
                                        EXECUTE 'ALTER TABLE {SCHEMA}.fact_performance ALTER COLUMN metric_value DROP NOT NULL';
                                    END IF;
                                END$$;
                        """)
                        # Backfill metric_value from avg_value when missing
                        conn.exec_driver_sql(f"""
                                UPDATE {SCHEMA}.fact_performance
                                     SET metric_value = avg_value
                                 WHERE metric_value IS NULL AND avg_value IS NOT NULL;
                        """)
                except Exception as exc:
                        logging.warning("Schema migrate (metric_value relax/backfill) skipped or failed: %s", exc)

                # Ensure a unique index exists for the ON CONFLICT clause used by the loader.
                # Older databases may miss the UNIQUE constraint even if the table exists.
                # First, remove duplicates that would block unique index creation.
                try:
                        conn.exec_driver_sql(f"""
                                DELETE FROM {SCHEMA}.fact_performance t
                                USING {SCHEMA}.fact_performance t2
                                WHERE t.id > t2.id
                                    AND t.experiment_key = t2.experiment_key
                                    AND t.phase_key = t2.phase_key
                                    AND t.split_key = t2.split_key
                                    AND t.horizon_key = t2.horizon_key
                                    AND t.metric_key = t2.metric_key;
                        """)
                except Exception as exc:
                        logging.warning("Duplicate cleanup for fact_performance skipped or failed: %s", exc)

                # Create a unique index if it does not exist (supports ON CONFLICT by column list)
                try:
                        conn.exec_driver_sql(f"""
                                DO $$
                                BEGIN
                                    IF NOT EXISTS (
                                        SELECT 1 FROM pg_class c
                                        JOIN pg_namespace n ON n.oid = c.relnamespace
                                        WHERE c.relkind = 'i'
                                            AND c.relname = 'ux_fact_performance_keys'
                                            AND n.nspname = '{SCHEMA}'
                                    ) THEN
                                        EXECUTE 'CREATE UNIQUE INDEX ux_fact_performance_keys ON {SCHEMA}.fact_performance (
                                                            experiment_key, phase_key, split_key, horizon_key, metric_key)';
                                    END IF;
                                END$$;
                        """)
                except Exception as exc:
                        logging.warning("Creating unique index for fact_performance failed: %s", exc)

                # seed splits / horizons / canonical metrics
                conn.exec_driver_sql(f"""
                        INSERT INTO {SCHEMA}.dim_dataset_split (split_key, description) VALUES
                            ('train','Training set'), ('validation','Validation set'), ('test','Test set')
                        ON CONFLICT DO NOTHING;
                """)
                conn.exec_driver_sql(f"""
                        INSERT INTO {SCHEMA}.dim_horizon (horizon_key, description) VALUES
                            (1,'Horizon 1'), (2,'Horizon 2'), (3,'Horizon 3'),
                            (4,'Horizon 4'), (5,'Horizon 5'), (6,'Horizon 6')
                        ON CONFLICT DO NOTHING;
                """)
                conn.exec_driver_sql(f"""
                        INSERT INTO {SCHEMA}.dim_metric (metric_key, metric_type, direction) VALUES
                            ('MAE','error','lower_is_better'),
                            ('R2','fit','higher_is_better'),
                            ('SNR','signal_to_noise','higher_is_better'),
                            ('Uncertainty','uncertainty','lower_is_better'),
                            ('Naive_MAE','baseline','lower_is_better'),
                            ('Naive MAE','baseline','lower_is_better')
                        ON CONFLICT DO NOTHING;
                """)

# -----------------------------
# Upsert helpers
# -----------------------------
def upsert_project(engine, project_key: str):
    """Insert project if missing."""
    sql = f"INSERT INTO {SCHEMA}.dim_project (project_key) VALUES (:k) ON CONFLICT DO NOTHING;"
    with engine.begin() as conn:
        conn.execute(text(sql), {"k": project_key})
    logging.info("Upserted project: %s", project_key)

def upsert_phase(engine, project_key: str, phase_key: str):
    """Insert or update phase row (project linkage)."""
    sql = f"""
    INSERT INTO {SCHEMA}.dim_phase (phase_key, project_key)
    VALUES (:p, :proj)
    ON CONFLICT (phase_key) DO UPDATE SET project_key = EXCLUDED.project_key;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {"p": phase_key, "proj": project_key})
    logging.info("Upserted phase: %s (project=%s)", phase_key, project_key)

def upsert_experiment(engine, project_key: str, phase_key: str, experiment_key: str, config_json: Optional[Dict]):
    """
    Upsert experiment and populate extracted columns that Metabase GUI needs
    (categoricals & numeric filters). Uses JSONB to store full config as well.
    """
    cfg = config_json or {}

    # Extract expected fields (None when absent)
    extracted = {
        "max_steps_train": cfg.get("max_steps_train"),
        "max_steps_test": cfg.get("max_steps_test"),
        "intermediate_layers": cfg.get("intermediate_layers"),
        "initial_layer_size": cfg.get("initial_layer_size"),
        "layer_size_divisor": cfg.get("layer_size_divisor"),
        "learning_rate": cfg.get("learning_rate"),
        "activation": cfg.get("activation"),
        "l2_reg": cfg.get("l2_reg"),
        "kl_weight": cfg.get("kl_weight"),
        "kl_anneal_epochs": cfg.get("kl_anneal_epochs"),
        "early_patience": cfg.get("early_patience"),
        "start_from_epoch": cfg.get("start_from_epoch"),
        "use_returns": cfg.get("use_returns"),
        "mmd_lambda": cfg.get("mmd_lambda"),
        "window_size": cfg.get("window_size"),
        "batch_size": cfg.get("batch_size"),
        "min_delta": cfg.get("min_delta"),
        "epochs": cfg.get("epochs"),
        "stl_period": cfg.get("stl_period"),
        "predicted_horizons": json.dumps(cfg.get("predicted_horizons")) if cfg.get("predicted_horizons") is not None else None,
        "use_stl": cfg.get("use_stl"),
        "use_wavelets": cfg.get("use_wavelets"),
        "use_multi_tapper": cfg.get("use_multi_tapper"),
        "predictor_plugin": cfg.get("predictor_plugin"),
        "optimizer_plugin": cfg.get("optimizer_plugin"),
        "pipeline_plugin": cfg.get("pipeline_plugin"),
        "preprocessor_plugin": cfg.get("preprocessor_plugin"),
        "use_strategy": cfg.get("use_strategy"),
        "use_daily": cfg.get("use_daily"),
        "mc_samples": cfg.get("mc_samples"),
    }

    sql = f"""
    INSERT INTO {SCHEMA}.dim_experiment (
      experiment_key, project_key, phase_key, config_json,
      max_steps_train, max_steps_test, intermediate_layers, initial_layer_size,
      layer_size_divisor, learning_rate, activation, l2_reg, kl_weight,
      kl_anneal_epochs, early_patience, start_from_epoch, use_returns,
      mmd_lambda, window_size, batch_size, min_delta, epochs, stl_period,
      predicted_horizons, use_stl, use_wavelets, use_multi_tapper,
      predictor_plugin, optimizer_plugin, pipeline_plugin, preprocessor_plugin,
      use_strategy, use_daily, mc_samples
    )
    VALUES (
      :e, :proj, :ph, CAST(:cfg AS JSONB),
      :max_steps_train, :max_steps_test, :intermediate_layers, :initial_layer_size,
      :layer_size_divisor, :learning_rate, :activation, :l2_reg, :kl_weight,
      :kl_anneal_epochs, :early_patience, :start_from_epoch, :use_returns,
      :mmd_lambda, :window_size, :batch_size, :min_delta, :epochs, :stl_period,
      CAST(:predicted_horizons AS JSONB), :use_stl, :use_wavelets, :use_multi_tapper,
      :predictor_plugin, :optimizer_plugin, :pipeline_plugin, :preprocessor_plugin,
      :use_strategy, :use_daily, :mc_samples
    )
    ON CONFLICT (experiment_key) DO UPDATE SET
      project_key = EXCLUDED.project_key,
      phase_key = EXCLUDED.phase_key,
      config_json = EXCLUDED.config_json,
      max_steps_train = EXCLUDED.max_steps_train,
      max_steps_test = EXCLUDED.max_steps_test,
      intermediate_layers = EXCLUDED.intermediate_layers,
      initial_layer_size = EXCLUDED.initial_layer_size,
      layer_size_divisor = EXCLUDED.layer_size_divisor,
      learning_rate = EXCLUDED.learning_rate,
      activation = EXCLUDED.activation,
      l2_reg = EXCLUDED.l2_reg,
      kl_weight = EXCLUDED.kl_weight,
      kl_anneal_epochs = EXCLUDED.kl_anneal_epochs,
      early_patience = EXCLUDED.early_patience,
      start_from_epoch = EXCLUDED.start_from_epoch,
      use_returns = EXCLUDED.use_returns,
      mmd_lambda = EXCLUDED.mmd_lambda,
      window_size = EXCLUDED.window_size,
      batch_size = EXCLUDED.batch_size,
      min_delta = EXCLUDED.min_delta,
      epochs = EXCLUDED.epochs,
      stl_period = EXCLUDED.stl_period,
      predicted_horizons = EXCLUDED.predicted_horizons,
      use_stl = EXCLUDED.use_stl,
      use_wavelets = EXCLUDED.use_wavelets,
      use_multi_tapper = EXCLUDED.use_multi_tapper,
      predictor_plugin = EXCLUDED.predictor_plugin,
      optimizer_plugin = EXCLUDED.optimizer_plugin,
      pipeline_plugin = EXCLUDED.pipeline_plugin,
      preprocessor_plugin = EXCLUDED.preprocessor_plugin,
      use_strategy = EXCLUDED.use_strategy,
      use_daily = EXCLUDED.use_daily,
      mc_samples = EXCLUDED.mc_samples;
    """

    params = {"e": experiment_key, "proj": project_key, "ph": phase_key, "cfg": json.dumps(cfg) if cfg is not None else None}
    params.update(extracted)

    with engine.begin() as conn:
        conn.execute(text(sql), params)

    logging.info("Upserted experiment: %s (ioin=%s, max_steps_train=%s)",
                 experiment_key, extracted.get("predictor_plugin"), extracted.get("max_steps_train"))

# -----------------------------
# Legacy summary loader (unchanged)
# -----------------------------
def load_results_summary(engine, project_key: str, phase_key: str, experiment_key: str, results_csv: str) -> int:
    """Existing summary loader — retained for compatibility (one-row-per-metric)."""
    try:
        df = pd.read_csv(results_csv)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    except Exception as exc:
        logging.error("Failed to read results CSV '%s': %s", results_csv, exc, exc_info=True)
        raise

    required = ["Metric", "Average", "Std_Dev", "Min", "Max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.error("Results CSV missing required columns: %s (file=%s)", missing, results_csv)
        raise ValueError(f"Missing columns: {missing}")

    for col in ["Average", "Std_Dev", "Min", "Max"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sql = f"""
    INSERT INTO {SCHEMA}.fact_results_summary
      (experiment_key, phase_key, metric, avg_value, std_dev, min_value, max_value)
    VALUES
      (:experiment_key, :phase_key, :metric, :avg, :std, :minv, :maxv)
    ON CONFLICT (experiment_key, metric) DO UPDATE
      SET avg_value = EXCLUDED.avg_value,
          std_dev   = EXCLUDED.std_dev,
          min_value = EXCLUDED.min_value,
          max_value = EXCLUDED.max_value,
          loaded_at = NOW();
    """

    rows = 0
    with engine.begin() as conn:
        for rec in df.itertuples(index=False):
            # Use sanitized attribute names from itertuples
            metric_text = getattr(rec, "Metric")
            conn.execute(text(sql), {
                "experiment_key": experiment_key,
                "phase_key": phase_key,
                "metric": metric_text,
                "avg":   getattr(rec, "Average"),
                "std":   getattr(rec, "Std_Dev"),
                "minv":  getattr(rec, "Min"),
                "maxv":  getattr(rec, "Max"),
            })
            rows += 1

    logging.info("Loaded results summary rows: %d (experiment=%s)", rows, experiment_key)
    return rows

# -----------------------------
# NEW loader: per-horizon facts
# -----------------------------
def load_performance_metrics(engine, project_key: str, phase_key: str, experiment_key: str, results_csv: str) -> int:
    """
    Parse the per-horizon CSV rows and upsert into fact_performance.
    Returns number of rows written.
    """
    try:
        df = pd.read_csv(results_csv)
    except Exception as exc:
        logging.error("Failed to read results CSV '%s': %s", results_csv, exc, exc_info=True)
        raise

    # Normalize headers for reliable access (Metric, Average, Std Dev, Min, Max -> Metric, Average, Std_Dev, Min, Max)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    if "Metric" not in df.columns:
        raise ValueError("Results CSV missing 'Metric' column")

    # Determine if legacy column metric_value exists; if so, include it to satisfy NOT NULL legacy schemas
    include_metric_value = False
    try:
        with engine.connect() as conn_check:
            res = conn_check.execute(text(
                """
                SELECT 1 FROM information_schema.columns
                 WHERE table_schema = :schema AND table_name = 'fact_performance' AND column_name = 'metric_value'
                """
            ), {"schema": SCHEMA}).fetchone()
            include_metric_value = res is not None
    except Exception as exc:
        logging.debug("Could not determine presence of metric_value column: %s", exc)

    # Upsert SQL for fact_performance (we will execute many times inside single txn)
    if include_metric_value:
        upsert_sql = f"""
        INSERT INTO {SCHEMA}.fact_performance
            (experiment_key, phase_key, split_key, horizon_key, metric_key, metric_value, avg_value, std_dev, min_value, max_value)
        VALUES
            (:experiment_key, :phase_key, :split_key, :horizon_key, :metric_key, :metric_value, :avg, :std, :minv, :maxv)
        ON CONFLICT (experiment_key, phase_key, split_key, horizon_key, metric_key) DO UPDATE
            SET metric_value = EXCLUDED.metric_value,
                avg_value   = EXCLUDED.avg_value,
                std_dev     = EXCLUDED.std_dev,
                min_value   = EXCLUDED.min_value,
                max_value   = EXCLUDED.max_value,
                loaded_at   = NOW();
        """
    else:
        upsert_sql = f"""
        INSERT INTO {SCHEMA}.fact_performance
            (experiment_key, phase_key, split_key, horizon_key, metric_key, avg_value, std_dev, min_value, max_value)
        VALUES
            (:experiment_key, :phase_key, :split_key, :horizon_key, :metric_key, :avg, :std, :minv, :maxv)
        ON CONFLICT (experiment_key, phase_key, split_key, horizon_key, metric_key) DO UPDATE
            SET avg_value = EXCLUDED.avg_value,
                std_dev   = EXCLUDED.std_dev,
                min_value = EXCLUDED.min_value,
                max_value = EXCLUDED.max_value,
                loaded_at = NOW();
        """
    # helper to canonicalize metric name to keys used in dim_metric
    def canonical_metric_key(raw_metric_name: str) -> str:
        rn = raw_metric_name.strip().lower().replace("_", " ")
        if rn in ("naive mae", "naive_mae"):
            return "Naive_MAE"
        if rn == "mae" or rn.endswith(" mae"):
            return "MAE"
        if rn in ("r2", "r^2") or "r2" in rn:
            return "R2"
        if "snr" in rn:
            return "SNR"
        if "uncertainty" in rn:
            return "Uncertainty"
        return raw_metric_name.strip().replace(" ", "_")


    rows_written = 0
    skipped = 0

    # We'll do a single transaction per-experiment; use per-row SAVEPOINTs to avoid aborting the whole txn
    with engine.begin() as conn:
        # include_metric_value and upsert_sql were computed above; ensure they are available here
        for idx, row in df.iterrows():
            raw_metric = str(row.get("Metric", "")).strip()
            m = _METRIC_RE.match(raw_metric)
            if not m:
                logging.debug("Skipping unexpected metric format: '%s'", raw_metric)
                skipped += 1
                continue

            split_raw, metric_name_raw, horizon_raw = m.group(1), m.group(2), m.group(3)
            split_key = split_raw.strip().lower()  # train/validation/test -> dim_dataset_split uses lowercase
            try:
                horizon_key = int(horizon_raw)
            except Exception:
                logging.warning("Invalid horizon value in '%s' — skipping", raw_metric)
                skipped += 1
                continue

            metric_key = canonical_metric_key(metric_name_raw)

            # ensure horizon exists in dim_horizon
            try:
                with conn.begin_nested():  # savepoint
                    conn.execute(text(f"""
                        INSERT INTO {SCHEMA}.dim_horizon (horizon_key, description)
                        VALUES (:hk, :desc)
                        ON CONFLICT (horizon_key) DO NOTHING;
                    """), {"hk": horizon_key, "desc": f"Horizon {horizon_key}"})
            except Exception as exc:
                logging.warning("Failed to ensure dim_horizon for H%s: %s", horizon_key, exc)

            # ensure metric exists in dim_metric (auto-create conservative entry)
            try:
                with conn.begin_nested():  # savepoint
                    conn.execute(text(f"""
                        INSERT INTO {SCHEMA}.dim_metric (metric_key, metric_type, direction)
                        VALUES (:mk, 'unknown', 'unknown') ON CONFLICT (metric_key) DO NOTHING;
                    """), {"mk": metric_key})
            except Exception as exc:
                logging.warning("Failed to ensure dim_metric for %s: %s", metric_key, exc)

            # parse numeric fields with coercion
            def _to_float(v):
                try:
                    return float(v) if v is not None and (str(v).strip() != "") else None
                except Exception:
                    return None

            avg = _to_float(row.get("Average"))
            std = _to_float(row.get("Std_Dev")) if "Std_Dev" in df.columns else None
            minv = _to_float(row.get("Min")) if "Min" in df.columns else None
            maxv = _to_float(row.get("Max")) if "Max" in df.columns else None

            # If avg is None it's probably something wrong — log and skip
            if avg is None:
                logging.warning("Skipping row with non-numeric 'Average' for metric '%s' in %s", raw_metric, results_csv)
                skipped += 1
                continue

            # upsert to fact_performance within a savepoint to isolate row failures
            try:
                with conn.begin_nested():  # savepoint
                    params = {
                        "experiment_key": experiment_key,
                        "phase_key": phase_key,
                        "split_key": split_key,
                        "horizon_key": horizon_key,
                        "metric_key": metric_key,
                        "avg": avg,
                        "std": std,
                        "minv": minv,
                        "maxv": maxv
                    }
                    if include_metric_value:
                        # Mirror avg into legacy metric_value
                        params["metric_value"] = avg
                    conn.execute(text(upsert_sql), params)
                rows_written += 1
            except Exception as exc:
                logging.error("Failed to upsert fact_performance for %s / %s / H%s / %s: %s",
                              experiment_key, split_key, horizon_key, metric_key, exc, exc_info=True)

    logging.info("Performance loader: written=%d skipped=%d (from file=%s, experiment=%s)",
                 rows_written, skipped, results_csv, experiment_key)
    return rows_written

# -----------------------------
# Main CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-key", required=True)
    ap.add_argument("--phase-key", required=True)
    ap.add_argument("--experiment-key", required=True)
    ap.add_argument("--experiment-config", required=True)
    ap.add_argument("--results-csv", required=True)
    args = ap.parse_args()

    engine = build_engine_from_pg_env()
    ensure_schema_and_tables(engine)

    # load config JSON
    try:
        with open(args.experiment_config, "r") as fh:
            cfg = json.load(fh)
    except Exception as exc:
        logging.error("Failed to read experiment config '%s': %s", args.experiment_config, exc, exc_info=True)
        sys.exit(3)

    # upsert dims
    upsert_project(engine, args.project_key)
    upsert_phase(engine, args.project_key, args.phase_key)
    upsert_experiment(engine, args.project_key, args.phase_key, args.experiment_key, cfg)

    # load summary (kept) and detailed performance facts
    try:
        _ = load_results_summary(engine, args.project_key, args.phase_key, args.experiment_key, args.results_csv)
    except Exception as exc:
        logging.warning("Summary loader failed (will still attempt performance loader): %s", exc, exc_info=True)

    try:
        n = load_performance_metrics(engine, args.project_key, args.phase_key, args.experiment_key, args.results_csv)
        if n == 0:
            logging.warning("No per-horizon metrics were loaded from '%s' (experiment=%s)", args.results_csv, args.experiment_key)
    except Exception as exc:
        logging.error("Failed to load per-horizon performance metrics: %s", exc, exc_info=True)
        sys.exit(5)

    logging.info("ETL completed for experiment '%s'.", args.experiment_key)
    sys.exit(0)

if __name__ == "__main__":
    main()
