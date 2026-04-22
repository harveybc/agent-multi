"""
OLAP upload helpers for binary classification experiments.

Re-uses the core ETL functions from olap.etl_migrate_v2 but seeds
binary-specific metrics into dim_metric before loading.
"""
from __future__ import annotations
from typing import Dict, Optional
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _try_import_etl():
    """Import ETL module; add olap/ to path if necessary."""
    try:
        from olap.etl_migrate_v2 import (
            build_engine_from_pg_env,
            ensure_schema_and_tables,
            upsert_project,
            upsert_phase,
            upsert_experiment,
            load_results_summary,
            load_performance_metrics,
            SCHEMA,
        )
        return (
            build_engine_from_pg_env,
            ensure_schema_and_tables,
            upsert_project,
            upsert_phase,
            upsert_experiment,
            load_results_summary,
            load_performance_metrics,
            SCHEMA,
        )
    except ImportError:
        # Try adding parent dir in case olap is a sibling folder
        predictor_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, predictor_root)
        try:
            from olap.etl_migrate_v2 import (
                build_engine_from_pg_env,
                ensure_schema_and_tables,
                upsert_project,
                upsert_phase,
                upsert_experiment,
                load_results_summary,
                load_performance_metrics,
                SCHEMA,
            )
            return (
                build_engine_from_pg_env,
                ensure_schema_and_tables,
                upsert_project,
                upsert_phase,
                upsert_experiment,
                load_results_summary,
                load_performance_metrics,
                SCHEMA,
            )
        except ImportError:
            return None


# Binary-specific metric seeds (direction for Metabase dashboards)
_BINARY_METRICS_SEED = [
    ("Accuracy",       "classification", "higher_is_better"),
    ("Precision",      "classification", "higher_is_better"),
    ("Recall",         "classification", "higher_is_better"),
    ("F1",             "classification", "higher_is_better"),
    ("AUC_ROC",        "classification", "higher_is_better"),
    ("AUC_PR",         "classification", "higher_is_better"),
    ("MCC",            "classification", "higher_is_better"),
    ("Brier",          "calibration",    "lower_is_better"),
    ("LogLoss",        "calibration",    "lower_is_better"),
    ("Pos_Rate_True",  "distribution",   "neutral"),
    ("Pos_Rate_Pred",  "distribution",   "neutral"),
    ("Uncertainty",    "uncertainty",     "lower_is_better"),
]


def _seed_binary_metrics(engine, schema: str):
    """Ensure all binary-specific metrics exist in dim_metric."""
    from sqlalchemy import text
    sql = f"""
        INSERT INTO {schema}.dim_metric (metric_key, metric_type, direction)
        VALUES (:mk, :mt, :d)
        ON CONFLICT (metric_key) DO NOTHING;
    """
    with engine.begin() as conn:
        for mk, mt, d in _BINARY_METRICS_SEED:
            conn.execute(text(sql), {"mk": mk, "mt": mt, "d": d})
    logger.info("Seeded %d binary metric dimension entries.", len(_BINARY_METRICS_SEED))


def upload_binary_experiment(
    config_json: Dict,
    results_csv: str,
    project_key: str = "ioin",
    phase_key: str = "phase_1b_binary",
    experiment_key: Optional[str] = None,
) -> bool:
    """
    Upload a binary experiment config and results to the OLAP cube.

    Returns True on success, False on failure (non-fatal — pipeline continues).
    """
    etl = _try_import_etl()
    if etl is None:
        logger.warning("OLAP ETL module not available — skipping upload.")
        return False

    (
        build_engine_from_pg_env,
        ensure_schema_and_tables,
        upsert_project,
        upsert_phase,
        upsert_experiment,
        load_results_summary,
        load_performance_metrics,
        SCHEMA,
    ) = etl

    # Derive experiment key from config if not provided
    if experiment_key is None:
        ioin = config_json.get("predictor_plugin", "unknown")
        signal = config_json.get("signal_type", "unknown")
        experiment_key = f"{ioin}_{signal}"

    try:
        engine = build_engine_from_pg_env()
        ensure_schema_and_tables(engine)
        _seed_binary_metrics(engine, SCHEMA)

        upsert_project(engine, project_key)
        upsert_phase(engine, project_key, phase_key)
        upsert_experiment(engine, project_key, phase_key, experiment_key, config_json)

        # Load results summary (all rows, legacy table)
        try:
            load_results_summary(engine, project_key, phase_key, experiment_key, results_csv)
        except Exception as exc:
            logger.warning("Summary loader failed (continuing): %s", exc)

        # Load per-horizon performance metrics (fact_performance)
        n = load_performance_metrics(engine, project_key, phase_key, experiment_key, results_csv)
        if n == 0:
            logger.warning("No per-horizon metrics loaded from %s", results_csv)

        logger.info("OLAP upload complete for experiment '%s' (%d metrics).", experiment_key, n)
        return True

    except Exception as exc:
        logger.error("OLAP upload failed for experiment '%s': %s", experiment_key, exc, exc_info=True)
        return False
