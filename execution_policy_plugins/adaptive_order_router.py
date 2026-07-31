"""Deterministic market/limit/stop router for research and live adapters.

The router is deliberately account independent. It chooses entry mechanics
from an asset policy's signed target exposure and market context; account
sizing and broker-specific order construction remain downstream concerns.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


_ORDER_TYPES = {"market", "limit", "stop"}
_FALLBACKS = {"market", "cancel"}


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _unit_interval(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class ExecutionContext:
    reference_price: float
    atr: float = 0.0
    spread_rate: float = 0.0
    breakout_score: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionContext":
        reference_price = _finite(
            "reference_price", value.get("reference_price"), minimum=0.0
        )
        if reference_price <= 0.0:
            raise ValueError("reference_price must be > 0")
        return cls(
            reference_price=reference_price,
            atr=_finite("atr", value.get("atr", 0.0), minimum=0.0),
            spread_rate=_finite(
                "spread_rate", value.get("spread_rate", 0.0), minimum=0.0
            ),
            breakout_score=_unit_interval(
                "breakout_score", value.get("breakout_score", 0.0)
            ),
        )


@dataclass(frozen=True)
class ExecutionDirective:
    side: str
    target_exposure: float
    order_type: str
    urgency: float
    entry_price: float | None
    valid_for_bars: int | None
    fallback: str | None
    reason_code: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Plugin:
    """Route a signed asset target into deterministic execution mechanics."""

    plugin_params = {
        "execution_deadband": 0.05,
        "market_urgency_threshold": 0.75,
        "market_max_spread_bps": 8.0,
        "stop_breakout_threshold": 0.65,
        "limit_offset_spread_multiple": 0.5,
        "limit_offset_atr_multiple": 0.05,
        "stop_offset_spread_multiple": 0.5,
        "stop_offset_atr_multiple": 0.05,
        "passive_valid_for_bars": 2,
        "stop_valid_for_bars": 2,
        "unfilled_fallback": "market",
    }

    plugin_debug_vars = list(plugin_params)

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.params = dict(self.plugin_params)
        if config:
            self.set_params(**dict(config))

    def set_params(self, **kwargs: Any) -> None:
        for key in self.params:
            if key in kwargs:
                self.params[key] = kwargs[key]
        self._validate()

    def get_debug_info(self) -> dict[str, Any]:
        return {name: self.params[name] for name in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: dict[str, Any]) -> None:
        debug_info.update(self.get_debug_info())

    def _validate(self) -> None:
        for name in (
            "execution_deadband",
            "market_urgency_threshold",
            "stop_breakout_threshold",
        ):
            _unit_interval(name, self.params[name])
        for name in (
            "market_max_spread_bps",
            "limit_offset_spread_multiple",
            "limit_offset_atr_multiple",
            "stop_offset_spread_multiple",
            "stop_offset_atr_multiple",
        ):
            _finite(name, self.params[name], minimum=0.0)
        for name in ("passive_valid_for_bars", "stop_valid_for_bars"):
            if int(self.params[name]) < 1:
                raise ValueError(f"{name} must be >= 1")
        fallback = str(self.params["unfilled_fallback"]).strip().lower()
        if fallback not in _FALLBACKS:
            raise ValueError(
                f"unfilled_fallback must be one of {sorted(_FALLBACKS)}"
            )

    def route(
        self,
        *,
        target_exposure: float,
        context: Mapping[str, Any] | ExecutionContext,
        urgency: float | None = None,
        confidence: float | None = None,
        market_available: bool = True,
        signal_valid: bool = True,
        signal_age_seconds: float | None = None,
        max_signal_age_seconds: float | None = None,
    ) -> ExecutionDirective | None:
        """Return an account-independent entry directive or ``None`` for hold."""
        if not isinstance(market_available, bool):
            raise ValueError("market_available must be a boolean")
        if not isinstance(signal_valid, bool):
            raise ValueError("signal_valid must be a boolean")
        exposure = _finite("target_exposure", target_exposure)
        if exposure < -1.0 or exposure > 1.0:
            raise ValueError("target_exposure must be in [-1, 1]")
        stale = False
        if signal_age_seconds is not None:
            age = _finite("signal_age_seconds", signal_age_seconds, minimum=0.0)
            if max_signal_age_seconds is None:
                raise ValueError(
                    "max_signal_age_seconds is required with signal_age_seconds"
                )
            maximum_age = _finite(
                "max_signal_age_seconds",
                max_signal_age_seconds,
                minimum=0.0,
            )
            stale = age > maximum_age
        elif max_signal_age_seconds is not None:
            _finite(
                "max_signal_age_seconds",
                max_signal_age_seconds,
                minimum=0.0,
            )
        if not market_available or not signal_valid or stale:
            return None
        deadband = float(self.params["execution_deadband"])
        if abs(exposure) <= deadband:
            return None

        market = (
            context
            if isinstance(context, ExecutionContext)
            else ExecutionContext.from_mapping(context)
        )
        resolved_urgency = _unit_interval(
            "urgency",
            abs(exposure) if urgency is None else urgency,
        )
        resolved_confidence = _unit_interval(
            "confidence",
            abs(exposure) if confidence is None else confidence,
        )
        decision_strength = resolved_urgency * resolved_confidence
        side = "buy" if exposure > 0.0 else "sell"
        spread_bps = market.spread_rate * 10_000.0

        if (
            resolved_urgency >= float(self.params["market_urgency_threshold"])
            and spread_bps <= float(self.params["market_max_spread_bps"])
        ):
            return ExecutionDirective(
                side=side,
                target_exposure=exposure,
                order_type="market",
                urgency=resolved_urgency,
                entry_price=None,
                valid_for_bars=None,
                fallback=None,
                reason_code="urgent_liquid_entry",
            )

        if (
            market.breakout_score * decision_strength
            >= float(self.params["stop_breakout_threshold"])
        ):
            offset = self._offset(
                market,
                spread_multiple=float(
                    self.params["stop_offset_spread_multiple"]
                ),
                atr_multiple=float(self.params["stop_offset_atr_multiple"]),
            )
            price = (
                market.reference_price + offset
                if side == "buy"
                else market.reference_price - offset
            )
            return ExecutionDirective(
                side=side,
                target_exposure=exposure,
                order_type="stop",
                urgency=resolved_urgency,
                entry_price=price,
                valid_for_bars=int(self.params["stop_valid_for_bars"]),
                fallback=str(self.params["unfilled_fallback"]).lower(),
                reason_code="breakout_confirmation",
            )

        offset = self._offset(
            market,
            spread_multiple=float(
                self.params["limit_offset_spread_multiple"]
            ),
            atr_multiple=float(self.params["limit_offset_atr_multiple"]),
        )
        price = (
            market.reference_price - offset
            if side == "buy"
            else market.reference_price + offset
        )
        return ExecutionDirective(
            side=side,
            target_exposure=exposure,
            order_type="limit",
            urgency=resolved_urgency,
            entry_price=price,
            valid_for_bars=int(self.params["passive_valid_for_bars"]),
            fallback=str(self.params["unfilled_fallback"]).lower(),
            reason_code="passive_price_improvement",
        )

    @staticmethod
    def _offset(
        context: ExecutionContext,
        *,
        spread_multiple: float,
        atr_multiple: float,
    ) -> float:
        spread = context.reference_price * context.spread_rate
        return max(
            context.reference_price * 1e-8,
            spread * spread_multiple,
            context.atr * atr_multiple,
        )


def directive_to_action_patch(
    directive: ExecutionDirective | None,
) -> dict[str, Any]:
    """Convert a route decision to the engine-neutral gym-fx action fields."""
    if directive is None:
        return {"hold": True}
    if directive.order_type not in _ORDER_TYPES:
        raise ValueError(f"unsupported order_type={directive.order_type!r}")
    patch: dict[str, Any] = {
        "target_exposure": directive.target_exposure,
        "entry_execution": {
            "order_type": directive.order_type,
        },
    }
    entry = patch["entry_execution"]
    if directive.valid_for_bars is not None:
        entry["valid_for_bars"] = directive.valid_for_bars
    if directive.fallback is not None:
        entry["unfilled_fallback"] = directive.fallback
    if directive.order_type == "limit":
        entry["limit_price"] = directive.entry_price
    elif directive.order_type == "stop":
        entry["trigger_price"] = directive.entry_price
    return patch
