#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init_db.py

Idempotent initialization for OLAP schema used by the ETL.

Creates:
  - dim_project, dim_phase, dim_experiment
  - dim_dataset_split, dim_horizon, dim_metric
  - fact_performance, fact_results_summary

This file is safe to run repeatedly (idempotent).
"""
import os
import logging
from sqlalchemy import create_engine

SCHEMA = "public"

DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

-- Projects
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_project (
  project_key TEXT PRIMARY KEY
);

-- Phases
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_phase (
  phase_key   TEXT PRIMARY KEY,
  project_key TEXT NOT NULL REFERENCES {SCHEMA}.dim_project(project_key) ON DELETE CASCADE
);

-- Experiments (config JSON + extracted columns for Metabase filtering)
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_experiment (
  experiment_key TEXT PRIMARY KEY,
  project_key    TEXT NOT NULL REFERENCES {SCHEMA}.dim_project(project_key) ON DELETE CASCADE,
  phase_key      TEXT NOT NULL REFERENCES {SCHEMA}.dim_phase(phase_key)   ON DELETE CASCADE,
  config_json    JSONB,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Extracted numeric / filterable fields
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

  -- Categorical/boolean fields for Metabase GUI filters
  predictor_plugin    TEXT,
  optimizer_plugin    TEXT,
  pipeline_plugin     TEXT,
  preprocessor_plugin TEXT,
  use_strategy        BOOLEAN,
  use_daily           BOOLEAN,
  mc_samples          INTEGER
);

-- Dataset splits
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_dataset_split (
  split_key TEXT PRIMARY KEY,
  description TEXT
);

-- Horizons (1..N)
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_horizon (
  horizon_key INTEGER PRIMARY KEY,
  description TEXT
);

-- Metrics (canonical set; loader may add more entries as needed)
CREATE TABLE IF NOT EXISTS {SCHEMA}.dim_metric (
  metric_key TEXT PRIMARY KEY,
  metric_type TEXT,
  direction   TEXT
);

-- Fact: per-experiment × split × horizon × metric
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

-- Fact: per-experiment summary (kept for high-level views)
CREATE TABLE IF NOT EXISTS {SCHEMA}.fact_results_summary (
  id             BIGSERIAL PRIMARY KEY,
  experiment_key TEXT NOT NULL REFERENCES {SCHEMA}.dim_experiment(experiment_key) ON DELETE CASCADE,
  phase_key      TEXT REFERENCES {SCHEMA}.dim_phase(phase_key) ON DELETE CASCADE,
  metric         TEXT NOT NULL,
  avg_value      DOUBLE PRECISION,
  std_dev        DOUBLE PRECISION,
  min_value      DOUBLE PRECISION,
  max_value      DOUBLE PRECISION,
  loaded_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (experiment_key, metric)
);

-- Indexes to speed common access patterns
CREATE INDEX IF NOT EXISTS idx_fact_perf_experiment ON {SCHEMA}.fact_performance (experiment_key);
CREATE INDEX IF NOT EXISTS idx_fact_perf_metric     ON {SCHEMA}.fact_performance (metric_key);
CREATE INDEX IF NOT EXISTS idx_dim_experiment_cfg_gin ON {SCHEMA}.dim_experiment USING gin (config_json);
"""

# Seed data: splits, horizons, canonical metrics
# Note: include both "Naive MAE" and "Naive_MAE" variants to tolerate CSV differences.
SEED = f"""
INSERT INTO {SCHEMA}.dim_dataset_split (split_key, description) VALUES
  ('train','Training set'), ('validation','Validation set'), ('test','Test set')
ON CONFLICT DO NOTHING;

INSERT INTO {SCHEMA}.dim_horizon (horizon_key, description) VALUES
  (1,'Horizon 1'), (2,'Horizon 2'), (3,'Horizon 3'),
  (4,'Horizon 4'), (5,'Horizon 5'), (6,'Horizon 6')
ON CONFLICT DO NOTHING;

-- Insert canonical metric keys. We add both spaced and underscore variants for Naive MAE
INSERT INTO {SCHEMA}.dim_metric (metric_key, metric_type, direction) VALUES
  ('MAE','error','lower_is_better'),
  ('R2','fit','higher_is_better'),
  ('SNR','signal_to_noise','higher_is_better'),
  ('Uncertainty','uncertainty','lower_is_better'),
  ('Naive MAE','baseline','lower_is_better'),
  ('Naive_MAE','baseline','lower_is_better')
ON CONFLICT DO NOTHING;
"""

def build_engine_from_pg_env():
    host = os.getenv("PGHOST", "127.0.0.1")
    port = int(os.getenv("PGPORT", "5432"))
    dbname = os.getenv("PGDATABASE", "predictor_olap")
    user = os.getenv("PGUSER", "metabase")
    password = os.getenv("PGPASSWORD", "metabase_pass")
    dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(dsn, pool_pre_ping=True, future=True)
    logging.info("Connected to PostgreSQL at %s:%s database=%s user=%s", host, port, dbname, user)
    return engine

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = build_engine_from_pg_env()
    logging.info("Creating schema and tables (SCHEMA=%s)...", SCHEMA)
    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)
        conn.exec_driver_sql(SEED)
    logging.info("Database initialization complete.")

if __name__ == "__main__":
    main()
