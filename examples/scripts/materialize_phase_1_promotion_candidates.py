#!/usr/bin/env python3
"""Freeze a small, validation-only Pareto set from a DOIN Phase 1 chain.

This is deliberately a promotion *input* generator.  It never opens, ranks on,
or emits protected-test evidence.  The resulting manifest is consumed by the
weekly-retrained promotion evaluator after a candidate is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen


SCHEMA_VERSION = "agent_multi.phase1_promotion_candidates.v1"
REQUIRED_METRICS = (
    "train_validation_l1_score",
    "risk_adjusted_total_return",
    "total_return",
    "max_drawdown_fraction",
    "trades_total",
)
PROTECTED_TEST_PREFIXES = ("test_", "protected_test")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:  # nosec B310: caller controls the local DOIN URL
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return payload


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _assert_no_protected_test_evidence(metrics: dict[str, Any]) -> None:
    leaked = [
        key
        for key, value in metrics.items()
        if key.startswith(PROTECTED_TEST_PREFIXES) and value is not None
    ]
    if leaked:
        raise ValueError(
            "Promotion candidate materialization refuses protected-test evidence: "
            + ", ".join(sorted(leaked))
        )


def _candidate_from_transaction(
    *,
    block: dict[str, Any],
    transaction: dict[str, Any],
    domain_id: str,
    min_trades: int,
) -> dict[str, Any] | None:
    if transaction.get("tx_type") != "optimae_accepted":
        return None
    if transaction.get("domain_id") != domain_id:
        return None

    payload = transaction.get("payload")
    if not isinstance(payload, dict):
        return None
    performance = payload.get("verified_performance")
    metrics = payload.get("champion_metrics")
    parameters = payload.get("parameters")
    if performance is None or not isinstance(metrics, dict) or not isinstance(parameters, dict):
        return None
    if any(metrics.get(key) is None for key in REQUIRED_METRICS):
        return None
    _assert_no_protected_test_evidence(metrics)

    l2_fitness = _finite_number(performance, field="verified_performance")
    validation_rap = _finite_number(
        metrics["risk_adjusted_total_return"], field="risk_adjusted_total_return"
    )
    validation_return = _finite_number(metrics["total_return"], field="total_return")
    max_drawdown = _finite_number(
        metrics["max_drawdown_fraction"], field="max_drawdown_fraction"
    )
    trades_total = int(_finite_number(metrics["trades_total"], field="trades_total"))
    if trades_total < min_trades:
        return None

    parameter_hash = _sha256(parameters)
    metric_key = {
        "l2_fitness": l2_fitness,
        "validation_rap": validation_rap,
        "validation_return": validation_return,
        "max_drawdown_fraction": max_drawdown,
        "trades_total": trades_total,
    }
    artifact_hash = metrics.get("model_artifact_sha256")
    return {
        "candidate_id": "sha256:" + _sha256({"parameters": parameter_hash, "metrics": metric_key}),
        "parameter_hash": "sha256:" + parameter_hash,
        "parameters": parameters,
        "l2_fitness": l2_fitness,
        "validation_evidence": {
            "total_return": validation_return,
            "risk_adjusted_total_return": validation_rap,
            "max_drawdown_fraction": max_drawdown,
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "trades_total": trades_total,
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "train_tail_selection_score": metrics.get("train_tail_selection_score"),
            "validation_selection_score": metrics.get("validation_selection_score"),
            "train_validation_selection_gap": metrics.get("train_validation_selection_gap"),
            "train_validation_selection_gap_penalty": metrics.get(
                "train_validation_selection_gap_penalty"
            ),
        },
        "artifact": {
            "sha256": artifact_hash,
            "bytes": metrics.get("model_artifact_bytes"),
            "format": metrics.get("model_artifact_format"),
            "retrieval": "doin_chain_embedded_or_replicated",
        },
        "source": {
            "block_index": block.get("index"),
            "block_hash_prefix": block.get("hash"),
            "transaction_id": transaction.get("id"),
            "peer_id": transaction.get("peer_id"),
            "timestamp": transaction.get("timestamp"),
            "optimization_config_hash": payload.get("optimization_config_hash"),
            "data_hash": payload.get("data_hash"),
        },
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether ``left`` is no worse in every promotion dimension."""
    left_metrics = left["validation_evidence"]
    right_metrics = right["validation_evidence"]
    comparisons = (
        left["l2_fitness"] >= right["l2_fitness"],
        left_metrics["risk_adjusted_total_return"]
        >= right_metrics["risk_adjusted_total_return"],
        left_metrics["total_return"] >= right_metrics["total_return"],
        left_metrics["max_drawdown_fraction"]
        <= right_metrics["max_drawdown_fraction"],
    )
    strictly_better = (
        left["l2_fitness"] > right["l2_fitness"],
        left_metrics["risk_adjusted_total_return"]
        > right_metrics["risk_adjusted_total_return"],
        left_metrics["total_return"] > right_metrics["total_return"],
        left_metrics["max_drawdown_fraction"]
        < right_metrics["max_drawdown_fraction"],
    )
    return all(comparisons) and any(strictly_better)


def reconcile_candidates(
    chain: dict[str, Any], *, domain_id: str, min_trades: int, max_candidates: int
) -> list[dict[str, Any]]:
    """Deduplicate chain acceptances then retain a bounded validation Pareto set."""
    best_by_equivalent_evidence: dict[tuple[str, float, float, float, float, int], dict[str, Any]] = {}
    for block in chain.get("blocks", []):
        if not isinstance(block, dict):
            continue
        for transaction in block.get("transactions", []):
            if not isinstance(transaction, dict):
                continue
            candidate = _candidate_from_transaction(
                block=block,
                transaction=transaction,
                domain_id=domain_id,
                min_trades=min_trades,
            )
            if candidate is None:
                continue
            evidence = candidate["validation_evidence"]
            key = (
                candidate["parameter_hash"],
                candidate["l2_fitness"],
                evidence["risk_adjusted_total_return"],
                evidence["total_return"],
                evidence["max_drawdown_fraction"],
                evidence["trades_total"],
            )
            incumbent = best_by_equivalent_evidence.get(key)
            if incumbent is None or candidate["source"]["block_index"] < incumbent["source"]["block_index"]:
                best_by_equivalent_evidence[key] = candidate

    candidates = list(best_by_equivalent_evidence.values())
    pareto = [
        candidate
        for candidate in candidates
        if not any(
            other["candidate_id"] != candidate["candidate_id"] and _dominates(other, candidate)
            for other in candidates
        )
    ]
    pareto.sort(
        key=lambda item: (
            -item["l2_fitness"],
            -item["validation_evidence"]["risk_adjusted_total_return"],
            item["validation_evidence"]["max_drawdown_fraction"],
            item["source"]["block_index"],
        )
    )
    for rank, candidate in enumerate(pareto[:max_candidates], start=1):
        candidate["promotion_rank"] = rank
        candidate["promotion_status"] = "frozen_pending_weekly_retrained_protected_test"
    return pareto[:max_candidates]


def materialize_manifest(args: argparse.Namespace) -> dict[str, Any]:
    chain = _read_json(args.chain_json) if args.chain_json else _fetch_json(args.chain_url)
    config = _read_json(args.config)
    candidates = reconcile_candidates(
        chain,
        domain_id=args.domain_id,
        min_trades=args.min_trades,
        max_candidates=args.max_candidates,
    )
    if not candidates:
        raise ValueError("No validation-only candidates met the configured promotion minimums")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_scope": {
            "domain_id": args.domain_id,
            "asset": args.asset,
            "timeframe": args.timeframe,
            "source_split": "validation_only",
            "protected_test": {
                "period": args.protected_test_period,
                "status": "not_opened",
                "rule": "Excluded from reconciliation, ranking, and this manifest.",
            },
        },
        "source": {
            "chain_height": chain.get("height"),
            "chain_url": args.chain_url if not args.chain_json else None,
            "chain_json": str(args.chain_json) if args.chain_json else None,
            "experiment_config": str(args.config),
            "experiment_config_sha256": "sha256:" + _sha256(config),
        },
        "selection_rules": {
            "minimum_validation_trades": args.min_trades,
            "pareto_objectives": [
                "maximize_train_validation_l1_score",
                "maximize_validation_risk_adjusted_total_return",
                "maximize_validation_total_return",
                "minimize_validation_max_drawdown_fraction",
            ],
            "deduplication": "parameter_hash_and_exact_validation_metric_vector",
        },
        "promotion_blockers": [
            "WEEKLY_RETRAINED_PROTECTED_TEST_NOT_RUN",
            "RELEASE_VALIDATION_NOT_RUN",
            "COMPONENT_COMPATIBILITY_NOT_RUN",
        ],
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--chain-url", help="DOIN dashboard /api/chain endpoint")
    source.add_argument("--chain-json", type=Path, help="Saved /api/chain payload for offline replay")
    parser.add_argument("--config", type=Path, required=True, help="Canonical Phase 1 config JSON")
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--protected-test-period", required=True)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.min_trades < 1:
        parser.error("--min-trades must be at least 1")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be at least 1")

    manifest = materialize_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"materialized {len(manifest['candidates'])} validation-only promotion candidate(s) "
        f"at {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
