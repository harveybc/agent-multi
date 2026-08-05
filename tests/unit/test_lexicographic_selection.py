"""ETH order §9: the transparent constrained/lexicographic selection
contract — activity without a profit gate, weekly-return primacy,
drawdown/total tie-breaks, transport-only scalar, fail-closed gates."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline_plugins._lexicographic_selection import (
    INELIGIBLE_TRANSPORT_SCALAR,
    evaluate_selection_contract,
)
from pipeline_plugins.rl_pipeline_with_validation import _selection_value


def _summary(**overrides):
    summary = {
        "mean_weekly_return": 0.004, "max_drawdown_fraction": 0.08,
        "total_return": 0.11, "trades_total": 40,
    }
    summary.update(overrides)
    return summary


def test_eligible_contract_and_ordered_tuple():
    contract = evaluate_selection_contract(_summary(), min_trades=10)
    assert contract["eligible"] is True
    assert contract["ordered_tuple"] == [0.004, -0.08, 0.11]
    assert contract["components"]["trades_total"] == 40
    assert "never display" in contract["transport_note"]


def test_losses_are_valid_evidence_no_profit_gate():
    contract = evaluate_selection_contract(
        _summary(mean_weekly_return=-0.02, total_return=-0.3),
        min_trades=10)
    assert contract["eligible"] is True                # negative is fine
    assert contract["ordered_tuple"][0] == -0.02


def test_activity_gate_without_profit_gate():
    contract = evaluate_selection_contract(
        _summary(trades_total=3), min_trades=10)
    assert contract["eligible"] is False
    assert any("no profit gate applies" in reason
               for reason in contract["ineligible_reasons"])
    assert contract["transport_scalar"] == INELIGIBLE_TRANSPORT_SCALAR


def test_failures_and_missing_metrics_fail_closed():
    assert not evaluate_selection_contract(
        _summary(error="boom"), min_trades=0)["eligible"]
    assert not evaluate_selection_contract(
        _summary(mean_weekly_return=float("nan")), min_trades=0)["eligible"]
    assert not evaluate_selection_contract({}, min_trades=0)["eligible"]


def test_lexicographic_order_preserved_by_transport_scalar():
    better_weekly = evaluate_selection_contract(
        _summary(mean_weekly_return=0.005, max_drawdown_fraction=0.5),
        min_trades=0)
    worse_weekly = evaluate_selection_contract(
        _summary(mean_weekly_return=0.004, max_drawdown_fraction=0.01),
        min_trades=0)
    assert (better_weekly["transport_scalar"]
            > worse_weekly["transport_scalar"])        # weekly dominates
    tie_low_dd = evaluate_selection_contract(
        _summary(max_drawdown_fraction=0.02), min_trades=0)
    tie_high_dd = evaluate_selection_contract(
        _summary(max_drawdown_fraction=0.30), min_trades=0)
    assert tie_low_dd["transport_scalar"] > tie_high_dd["transport_scalar"]
    tie_total_hi = evaluate_selection_contract(
        _summary(total_return=0.5), min_trades=0)
    tie_total_lo = evaluate_selection_contract(
        _summary(total_return=0.1), min_trades=0)
    assert (tie_total_hi["transport_scalar"]
            > tie_total_lo["transport_scalar"])


def test_selection_value_integration_persists_contract():
    summary = _summary(_selection_min_trades=10)
    value = _selection_value(
        summary, selection_metric="lexicographic_weekly_v1",
        risk_lambda=1.0)
    assert summary["selection_contract"]["eligible"] is True
    assert value == summary["selection_contract"]["transport_scalar"]
    ineligible = _summary(trades_total=1, _selection_min_trades=10)
    assert _selection_value(
        ineligible, selection_metric="lexicographic_weekly_v1",
        risk_lambda=1.0) == INELIGIBLE_TRANSPORT_SCALAR
