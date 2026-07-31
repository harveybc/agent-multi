from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution_policy_plugins.adaptive_order_router import (
    Plugin,
    directive_to_action_patch,
)


CONTEXT = {
    "reference_price": 100.0,
    "atr": 2.0,
    "spread_rate": 0.0004,
}


def test_router_holds_inside_deadband() -> None:
    decision = Plugin().route(target_exposure=0.04, context=CONTEXT)
    assert decision is None
    assert directive_to_action_patch(decision) == {"hold": True}


def test_router_uses_market_for_urgent_liquid_entry() -> None:
    decision = Plugin().route(
        target_exposure=0.9,
        urgency=0.9,
        confidence=0.9,
        context={**CONTEXT, "breakout_score": 1.0},
    )
    assert decision is not None
    assert decision.side == "buy"
    assert decision.target_exposure == pytest.approx(0.9)
    assert decision.order_type == "market"
    assert decision.entry_price is None


def test_router_uses_stop_for_high_confidence_breakout() -> None:
    router = Plugin({"market_max_spread_bps": 1.0})
    decision = router.route(
        target_exposure=-0.9,
        urgency=0.9,
        confidence=0.9,
        context={**CONTEXT, "breakout_score": 0.9},
    )
    assert decision is not None
    assert decision.side == "sell"
    assert decision.order_type == "stop"
    assert decision.entry_price < CONTEXT["reference_price"]
    patch = directive_to_action_patch(decision)
    assert patch["target_exposure"] == pytest.approx(-0.9)
    assert patch["entry_execution"]["trigger_price"] == decision.entry_price
    assert patch["entry_execution"]["unfilled_fallback"] == "market"


def test_router_uses_limit_for_nonurgent_entry() -> None:
    decision = Plugin().route(
        target_exposure=0.6,
        urgency=0.4,
        confidence=0.7,
        context={**CONTEXT, "breakout_score": 0.2},
    )
    assert decision is not None
    assert decision.order_type == "limit"
    assert decision.entry_price < CONTEXT["reference_price"]
    assert decision.valid_for_bars == 2
    assert (
        directive_to_action_patch(decision)["entry_execution"]["limit_price"]
        == decision.entry_price
    )


def test_router_is_deterministic() -> None:
    router = Plugin()
    kwargs = {
        "target_exposure": -0.55,
        "urgency": 0.3,
        "confidence": 0.8,
        "context": {**CONTEXT, "breakout_score": 0.4},
    }
    assert router.route(**kwargs) == router.route(**kwargs)


def test_unavailable_market_cannot_emit_an_entry_directive() -> None:
    decision = Plugin().route(
        target_exposure=0.9,
        context=CONTEXT,
        market_available=False,
    )

    assert decision is None
    assert directive_to_action_patch(decision) == {"hold": True}


def test_unavailable_market_does_not_hide_an_invalid_target() -> None:
    with pytest.raises(ValueError, match="target_exposure"):
        Plugin().route(
            target_exposure=2.0,
            context=CONTEXT,
            market_available=False,
        )


@pytest.mark.parametrize(
    "signal_kwargs",
    [
        {"signal_valid": False},
        {"signal_age_seconds": 61.0, "max_signal_age_seconds": 60.0},
    ],
)
def test_invalid_or_stale_signal_cannot_emit_an_entry_directive(
    signal_kwargs: dict[str, object],
) -> None:
    decision = Plugin().route(
        target_exposure=-1.0,
        context=CONTEXT,
        **signal_kwargs,
    )

    assert decision is None
    assert directive_to_action_patch(decision) == {"hold": True}


def test_signal_age_requires_an_explicit_freshness_limit() -> None:
    with pytest.raises(ValueError, match="max_signal_age_seconds"):
        Plugin().route(
            target_exposure=0.5,
            context=CONTEXT,
            signal_age_seconds=1.0,
        )


def test_versioned_router_profile_loads_into_plugin() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/config/execution_router"
        / "project3_adaptive_order_router_v1.json"
    )
    profile = json.loads(path.read_text(encoding="utf-8"))
    router = Plugin(profile["parameters"])

    assert profile["schema_version"] == "adaptive_order_router.v1"
    assert router.get_debug_info()["unfilled_fallback"] == "cancel"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("market_urgency_threshold", 1.1),
        ("passive_valid_for_bars", 0),
        ("limit_offset_atr_multiple", -0.1),
        ("unfilled_fallback", "magic"),
    ],
)
def test_router_rejects_invalid_configuration(key: str, value: object) -> None:
    with pytest.raises(ValueError):
        Plugin({key: value})
