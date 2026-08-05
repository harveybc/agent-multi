#!/usr/bin/env python3
"""Socket-free reproducer for the 2026-08-05 ETH curriculum audit."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TRADING_CONTRACTS_SRC = REPO.parent / "trading-contracts/src"
sys.path[:0] = [str(REPO), str(TRADING_CONTRACTS_SRC)]

from app.canonical_config import resolve_config
from app.metrics import compute_optimization_fitness
from pipeline_plugins._lexicographic_selection import (
    evaluate_selection_contract,
)


PREDICTOR = REPO.parent / "predictor"
CONFIG = (
    REPO
    / "examples/config/phase_2_eth_curriculum/optimization/"
    "phase_2_eth_en_v1.json"
)
FIXTURE = REPO / "tools/eth_curriculum_fixture.py"
DATA = (
    PREDICTOR
    / "examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv"
)
ARCHIVED_CHAIN = (
    Path.home()
    / ".local/state/agent-multi/doin-campaigns/"
    "phase-2-eth-curriculum-invalid-audit-20260805/omega/"
    "doin-data-eth-en-v1-omega/chain.db"
)


def _dataset_evidence() -> dict:
    boundaries = {
        "train": (
            datetime.fromisoformat("2017-09-28T04:00:00"),
            datetime.fromisoformat("2024-01-01T00:00:00"),
        ),
        "validation": (
            datetime.fromisoformat("2024-01-01T00:00:00"),
            datetime.fromisoformat("2025-01-01T00:00:00"),
        ),
        "test": (
            datetime.fromisoformat("2025-01-01T00:00:00"),
            datetime.fromisoformat("2026-01-01T00:00:00"),
        ),
    }
    counts = {name: 0 for name in boundaries}
    timestamps: list[datetime] = []
    with DATA.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime.fromisoformat(row["DATE_TIME"])
            timestamps.append(stamp)
            for name, (start, end) in boundaries.items():
                counts[name] += int(start <= stamp < end)
    return {
        "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "rows": len(timestamps),
        "split_rows": counts,
        "duplicates": len(timestamps) - len(set(timestamps)),
        "monotonic": timestamps == sorted(timestamps),
        "first": timestamps[0].isoformat(),
        "last": timestamps[-1].isoformat(),
    }


def _chain_evidence() -> dict:
    if not ARCHIVED_CHAIN.exists():
        return {"available": False, "path": str(ARCHIVED_CHAIN)}
    connection = sqlite3.connect(ARCHIVED_CHAIN)
    try:
        blocks = connection.execute(
            "SELECT block_index, hash, previous_hash, tx_count, "
            "weighted_performance_sum, threshold FROM blocks "
            "ORDER BY block_index"
        ).fetchall()
        accepted = connection.execute(
            "SELECT block_index, tx_type, domain_id, "
            "json_extract(payload, '$.verified_performance') "
            "FROM transactions WHERE tx_type = 'optimae_accepted' "
            "ORDER BY block_index, tx_index"
        ).fetchall()
    finally:
        connection.close()
    return {
        "available": True,
        "path": str(ARCHIVED_CHAIN),
        "sha256": hashlib.sha256(ARCHIVED_CHAIN.read_bytes()).hexdigest(),
        "blocks": [list(row) for row in blocks],
        "accepted_transactions": [list(row) for row in accepted],
    }


def main() -> int:
    canonical = json.loads(CONFIG.read_text(encoding="utf-8"))
    runtime = resolve_config({}, file_config=canonical).runtime

    fitness_error = None
    try:
        compute_optimization_fitness(
            {
                "mean_weekly_return": 0.01,
                "max_drawdown_fraction": 0.1,
                "total_return": 0.2,
            },
            runtime,
            object(),
        )
    except Exception as exc:  # evidence records the exact fail-closed error
        fitness_error = f"{type(exc).__name__}: {exc}"

    lexicographic_winner = evaluate_selection_contract(
        {
            "mean_weekly_return": 0.01,
            "max_drawdown_fraction": 0.9,
            "total_return": 0.0,
            "trades_total": 12,
        },
        min_trades=12,
    )
    transport_winner = evaluate_selection_contract(
        {
            "mean_weekly_return": 0.00995,
            "max_drawdown_fraction": 0.0,
            "total_return": 0.0,
            "trades_total": 12,
        },
        min_trades=12,
    )

    genome = runtime["mixed_genome_schema"]
    preprocessing = next(
        gene for gene in genome if gene.get("name") == "preprocessing_mode"
    )
    fixture_source = FIXTURE.read_text(encoding="utf-8")
    result = {
        "schema": "agent_multi.audit.eth_curriculum_repro.v1",
        "network_used": False,
        "runtime_contract": {
            "selection_metric": runtime.get("selection_metric"),
            "optimization_metric": runtime.get("optimization_metric"),
            "experiment_name": runtime.get("experiment_name"),
            "optimization_champion_model_file": runtime.get(
                "optimization_champion_model_file"
            ),
            "optimization_parameters_file": runtime.get(
                "optimization_parameters_file"
            ),
            "optimization_resume_file": runtime.get(
                "optimization_resume_file"
            ),
            "outer_fitness_error": fitness_error,
        },
        "lexicographic_counterexample": {
            "authoritative_A_gt_B": (
                lexicographic_winner["ordered_tuple"]
                > transport_winner["ordered_tuple"]
            ),
            "transport_A_gt_B": (
                lexicographic_winner["transport_scalar"]
                > transport_winner["transport_scalar"]
            ),
            "A_transport": lexicographic_winner["transport_scalar"],
            "B_transport": transport_winner["transport_scalar"],
        },
        "invalid_genome_surface": {
            "preprocessing_choices": preprocessing.get("choices"),
            "repair_rules": runtime.get("mixed_genome_repair_rules"),
            "requires_feature_aware_preprocessor": runtime.get(
                "require_feature_aware_preprocessor"
            ),
            "precomputed_causal_features": runtime.get(
                "precomputed_causal_features"
            ),
        },
        "fixture_contract": {
            "evaluate_test_split_true": (
                'config["evaluate_test_split"] = True' in fixture_source
            ),
            "repo_report_exists": (
                REPO / "eth_fixture_full/fixture_report.json"
            ).exists(),
        },
        "dataset": _dataset_evidence(),
        "archived_chain": _chain_evidence(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
