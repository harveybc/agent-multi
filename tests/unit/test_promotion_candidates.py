from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/scripts/materialize_phase_1_promotion_candidates.py"
)
_SPEC = importlib.util.spec_from_file_location("promotion_candidates", _SCRIPT)
assert _SPEC and _SPEC.loader
promotion_candidates = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(promotion_candidates)


def _accepted(
    block: int,
    *,
    parameters: dict,
    l2: float,
    rap: float,
    total_return: float,
    drawdown: float,
    trades: int,
    protected_test: float | None = None,
) -> dict:
    metrics = {
        "train_validation_l1_score": l2,
        "risk_adjusted_total_return": rap,
        "total_return": total_return,
        "max_drawdown_fraction": drawdown,
        "trades_total": trades,
        "model_artifact_sha256": f"artifact-{block}",
    }
    if protected_test is not None:
        metrics["test_total_return"] = protected_test
    return {
        "index": block,
        "hash": f"block-{block}",
        "transactions": [
            {
                "id": f"tx-{block}",
                "tx_type": "optimae_accepted",
                "domain_id": "trading-asset-policy-solusdt-4h-sac-v1",
                "peer_id": "node-a",
                "timestamp": "2026-07-14T00:00:00+00:00",
                "payload": {
                    "verified_performance": l2,
                    "parameters": parameters,
                    "champion_metrics": metrics,
                },
            }
        ],
    }


def test_reconcile_candidates_deduplicates_and_keeps_validation_pareto_set() -> None:
    chain = {
        "height": 3,
        "blocks": [
            _accepted(
                1,
                parameters={"threshold": 0.1},
                l2=0.05,
                rap=0.20,
                total_return=0.24,
                drawdown=0.04,
                trades=120,
            ),
            _accepted(
                2,
                parameters={"threshold": 0.1},
                l2=0.05,
                rap=0.20,
                total_return=0.24,
                drawdown=0.04,
                trades=120,
            ),
            _accepted(
                3,
                parameters={"threshold": 0.2},
                l2=0.04,
                rap=0.18,
                total_return=0.20,
                drawdown=0.05,
                trades=130,
            ),
        ],
    }

    candidates = promotion_candidates.reconcile_candidates(
        chain,
        domain_id="trading-asset-policy-solusdt-4h-sac-v1",
        min_trades=1,
        max_candidates=3,
    )

    assert len(candidates) == 1
    assert candidates[0]["source"]["block_index"] == 1
    assert candidates[0]["promotion_rank"] == 1
    assert candidates[0]["promotion_status"] == "frozen_pending_weekly_retrained_protected_test"


def test_reconciliation_rejects_protected_test_evidence() -> None:
    chain = {
        "height": 1,
        "blocks": [
            _accepted(
                1,
                parameters={"threshold": 0.1},
                l2=0.05,
                rap=0.20,
                total_return=0.24,
                drawdown=0.04,
                trades=120,
                protected_test=0.5,
            )
        ],
    }

    with pytest.raises(ValueError, match="protected-test evidence"):
        promotion_candidates.reconcile_candidates(
            chain,
            domain_id="trading-asset-policy-solusdt-4h-sac-v1",
            min_trades=1,
            max_candidates=3,
        )


def test_materialized_manifest_stays_explicitly_blocked_without_test(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"schema_version":"trading_experiment.v1"}\n', encoding="utf-8")
    chain = tmp_path / "chain.json"
    chain.write_text(
        __import__("json").dumps(
            {
                "height": 1,
                "blocks": [
                    _accepted(
                        1,
                        parameters={"threshold": 0.1},
                        l2=0.05,
                        rap=0.20,
                        total_return=0.24,
                        drawdown=0.04,
                        trades=120,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        chain_json=chain,
        chain_url=None,
        config=config,
        domain_id="trading-asset-policy-solusdt-4h-sac-v1",
        asset="SOLUSDT",
        timeframe="4h",
        protected_test_period="2023",
        min_trades=1,
        max_candidates=3,
    )

    manifest = promotion_candidates.materialize_manifest(args)

    assert manifest["selection_scope"]["protected_test"]["status"] == "not_opened"
    assert "WEEKLY_RETRAINED_PROTECTED_TEST_NOT_RUN" in manifest["promotion_blockers"]
    assert manifest["candidates"][0]["validation_evidence"]["total_return"] == pytest.approx(0.24)
