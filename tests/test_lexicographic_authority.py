"""Adversarial proofs for the authoritative lexicographic contract.

Corrections for AUD-F1-20260805-108/109/112: the order key must be a
proven order-preserving encoding of the quantized tuple, the objective
must be implemented end to end, and every failure must map to the
rejected-result schema — never to a bare comparable sentinel.
"""
from __future__ import annotations

import math
import random

import pytest

from app.metrics import SelectionIneligibleError, compute_optimization_fitness
from optimizer_plugins.default_optimizer import (
    REJECTED_FITNESS,
    _rejected_result,
)
from pipeline_plugins import _lexicographic_selection as lex
from pipeline_plugins.rl_pipeline_with_validation import (
    _selection_pair_details,
)


def test_audit_counterexample_verbatim():
    """AUD-F1-20260805-112 counterexample: A=(0.01000,-0.9,0) must beat
    B=(0.00995,0,0) in BOTH the tuple and the scalar encoding."""
    a_tuple = [0.01, -0.9, 0.0]
    b_tuple = [0.00995, -0.0, 0.0]
    assert lex.compare_ordered_tuples(a_tuple, b_tuple) > 0
    a_key = lex.encode_order_key(0.01, 0.9, 0.0)
    b_key = lex.encode_order_key(0.00995, 0.0, 0.0)
    assert a_key > b_key, "scalar encoding reversed the audited order"


def test_order_key_matches_quantized_tuple_order_property():
    """Property test over the full bounded component ranges: scalar
    comparison equals quantized tuple comparison for every pair."""
    rng = random.Random(112)

    def sample():
        return (
            rng.uniform(lex.WEEKLY_MIN * 1.2, lex.WEEKLY_MAX * 1.2),
            rng.uniform(lex.DD_MIN - 0.1, lex.DD_MAX + 0.1),
            rng.uniform(lex.TOTAL_MIN - 1.0, lex.TOTAL_MAX + 2.0),
        )

    for _ in range(5000):
        wa, da, ta = sample()
        wb, db, tb = sample()
        qa = lex.quantized_tuple(wa, da, ta)
        qb = lex.quantized_tuple(wb, db, tb)
        ka = lex.encode_order_key(wa, da, ta)
        kb = lex.encode_order_key(wb, db, tb)
        assert (ka > kb) == (qa > qb)
        assert (ka == kb) == (qa == qb)


def test_order_key_is_float64_exact_and_positive():
    top = lex.encode_order_key(lex.WEEKLY_MAX, lex.DD_MIN, lex.TOTAL_MAX)
    assert top < float(2 ** 53)
    assert top == int(top)
    bottom = lex.encode_order_key(lex.WEEKLY_MIN, lex.DD_MAX, lex.TOTAL_MIN)
    assert bottom >= 1.0
    # C2 (order 2026-08-19): the ineligible key is None — typed
    # non-orderable, no numeric relation to any eligible key exists.
    assert lex.INELIGIBLE_ORDER_KEY is None


def test_ineligible_loses_to_every_eligible():
    contract = lex.evaluate_selection_contract(
        {"mean_weekly_return": -0.4, "max_drawdown_fraction": 1.0,
         "total_return": -0.99, "trades_total": 100},
        min_trades=1)
    assert contract["eligible"]
    assert contract["transport_scalar"] is not None
    assert contract["transport_scalar"] >= 1.0
    rejected = lex.evaluate_selection_contract(
        {"mean_weekly_return": 0.4, "max_drawdown_fraction": 0.0,
         "total_return": 5.0, "trades_total": 0},
        min_trades=1)
    assert not rejected["eligible"]
    assert rejected["transport_scalar"] is None
    with pytest.raises(lex.IneligibleOrderKeyError):
        lex.require_orderable(rejected)


def test_metric_branch_implemented_and_fail_closed():
    """AUD-F1-20260805-108: the configured objective must complete for an
    eligible summary and raise the typed error when ineligible."""
    config = {"optimization_metric": "lexicographic_weekly_v1",
              "selection_min_trades": 12}
    summary = {"mean_weekly_return": 0.002, "max_drawdown_fraction": 0.05,
               "total_return": 0.1, "trades_total": 40}
    key = compute_optimization_fitness(summary, config, object())
    assert key == lex.encode_order_key(0.002, 0.05, 0.1)
    with pytest.raises(SelectionIneligibleError):
        compute_optimization_fitness(
            {**summary, "trades_total": 3}, config, object())
    with pytest.raises(SelectionIneligibleError):
        compute_optimization_fitness(
            {**summary, "mean_weekly_return": float("nan")}, config,
            object())


def test_metric_branch_prefers_validation_split():
    config = {"optimization_metric": "lexicographic_weekly_v1",
              "selection_min_trades": 1}
    summary = {
        "mean_weekly_return": 0.5, "max_drawdown_fraction": 0.0,
        "total_return": 9.9, "trades_total": 999,
        "splits": {"validation": {
            "mean_weekly_return": 0.001, "max_drawdown_fraction": 0.2,
            "total_return": 0.05, "trades_total": 20}},
    }
    key = compute_optimization_fitness(summary, config, object())
    assert key == lex.encode_order_key(0.001, 0.2, 0.05)


def test_rejected_result_schema():
    """AUD-F1-20260805-109: one rejected-result schema for every
    optimizer-boundary failure; DOIN-visible flags always present."""
    fitness, payload = _rejected_result("evaluation_error", "boom")
    assert fitness == REJECTED_FITNESS
    assert payload["candidate_rejected"] is True
    assert payload["candidate_rejected_reason"].startswith(
        "evaluation_error")
    assert payload["fitness"] == REJECTED_FITNESS
    assert payload["rejection_type"] == "evaluation_error"


def test_checkpoint_selection_is_validation_only_for_lexicographic():
    """§9: no train-tail averaging or gap penalty may distort the
    lexicographic checkpoint comparison."""
    train_tail = {"mean_weekly_return": 0.4, "max_drawdown_fraction": 0.0,
                  "total_return": 5.0, "trades_total": 50,
                  "_selection_min_trades": 1}
    validation = {"mean_weekly_return": 0.001,
                  "max_drawdown_fraction": 0.3, "total_return": 0.02,
                  "trades_total": 30, "_selection_min_trades": 1}
    details = _selection_pair_details(
        train_tail, validation,
        selection_metric="lexicographic_weekly_v1", risk_lambda=1.0,
        gap_penalty_beta=0.25)
    expected = lex.encode_order_key(0.001, 0.3, 0.02)
    assert details["train_validation_selection_score"] == expected
    assert details["train_validation_selection_gap_penalty"] == 0.0
    assert validation["selection_contract"]["eligible"] is True


def test_clamping_never_reorders_within_bounds():
    """Out-of-bounds values clamp to the boundary and therefore tie with
    the boundary — they can never leapfrog an in-bounds better value."""
    inside = lex.encode_order_key(0.49, 0.0, 0.0)
    beyond = lex.encode_order_key(7.0, 0.0, 0.0)
    boundary = lex.encode_order_key(lex.WEEKLY_MAX, 0.0, 0.0)
    assert beyond == boundary
    assert inside < boundary
    assert math.isfinite(beyond)
