#!/usr/bin/env python3
"""
Reset OLAP tables by truncating them (RESTART IDENTITY CASCADE).

Uses PG* environment variables (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD).
Defaults: 127.0.0.1:5432, predictor_olap, metabase / metabase_pass

Safety: requires --yes to execute; otherwise prints what it would do.
"""
import argparse
import logging
import os
import sys
from typing import List

from sqlalchemy import create_engine, text


SCHEMA_DEFAULT = "public"


def build_engine_from_pg_env():
    host = os.getenv("PGHOST", "127.0.0.1")
    port = int(os.getenv("PGPORT", "5432"))
    dbname = os.getenv("PGDATABASE", "predictor_olap")
    user = os.getenv("PGUSER", "metabase")
    password = os.getenv("PGPASSWORD", "metabase_pass")
    dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(dsn, pool_pre_ping=True, future=True)


def get_default_tables() -> List[str]:
    # Limit to the OLAP tables created/used by this project
    return [
        "fact_performance",
        "fact_results_summary",
        "dim_experiment",
        "dim_phase",
        "dim_project",
        "dim_dataset_split",
        "dim_horizon",
        "dim_metric",
    ]


def trunc_tables(engine, schema: str, tables: List[str]):
    if not tables:
        logging.info("No tables to truncate.")
        return
    ident_list = ", ".join(f'{schema}."{t}"' for t in tables)
    sql = f"TRUNCATE TABLE {ident_list} RESTART IDENTITY CASCADE;"
    with engine.begin() as conn:
        conn.execute(text(sql))


def main():
    parser = argparse.ArgumentParser(description="Truncate OLAP tables (RESTART IDENTITY CASCADE)")
    parser.add_argument("--schema", default=SCHEMA_DEFAULT, help="Schema containing the OLAP tables (default: public)")
    parser.add_argument(
        "--tables",
        nargs="*",
        help="Explicit tables to truncate (defaults to known OLAP tables)",
    )
    parser.add_argument("--yes", action="store_true", help="Confirm execution")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tables = args.tables or get_default_tables()
    logging.info("Target schema: %s", args.schema)
    logging.info("Tables to truncate: %s", ", ".join(tables))

    if not args.yes:
        logging.warning("Dry-run: add --yes to execute the TRUNCATE.")
        sys.exit(0)

    engine = build_engine_from_pg_env()
    try:
        trunc_tables(engine, args.schema, tables)
        logging.info("Truncate completed.")
    except Exception as exc:
        logging.error("Failed to truncate tables: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
