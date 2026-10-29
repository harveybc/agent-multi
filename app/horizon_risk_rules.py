"""Configurable EURUSD/USD order construction, separate from action selection."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math

from app.zigzag_risk_profiles import size_from_risk


def coefficients(**overrides):
    g = dict(a_sl=1., b_sl=0., a_tp=1., b_tp=0., buffer_pips=1., rho=.005)
    if set(overrides) - set(g):
        raise ValueError('UNKNOWN_RISK_GENE')
    g.update(overrides)
    if (not all(math.isfinite(v) for v in g.values()) or min(g.values()) < 0
            or g['a_sl'] <= 0 or g['a_tp'] <= 0 or g['rho'] > 1):
        raise ValueError('INVALID_RISK_GENE')
    return g


def levels(direction, entry_quote, tp_reference, mae_reference, sigma, genes, tick):
    if (direction not in (-1, 1) or not all(math.isfinite(v) for v in
            (entry_quote, tp_reference, mae_reference, sigma, tick))
            or min(entry_quote, tp_reference, tick) <= 0 or min(mae_reference, sigma) < 0):
        raise ValueError('INVALID_LEVEL_INPUT')
    g = coefficients(**genes)
    D = lambda x: Decimal(str(x))
    e, s, step = D(entry_quote), D(direction), D(tick)
    ds = D(g['a_sl'])*D(mae_reference) + D(g['b_sl'])*D(sigma) + D(g['buffer_pips'])*D(.0001)
    dt = D(g['a_tp'])*D(tp_reference) - D(g['b_tp'])*D(sigma)
    # Both prices round downward for a buy, upward for a sell: SL out, TP in.
    rounding = ROUND_FLOOR if direction == 1 else ROUND_CEILING
    rounded = lambda x: float((x/step).to_integral_value(rounding=rounding)*step)
    return dict(sl=rounded(e-s*ds), tp=rounded(e+s*dt),
                raw_sl_distance=float(ds), raw_tp_distance=float(dt))


def propose(*, direction, entry_quote, liquidation_quote, balance,
            tp_reference, mae_reference, sigma, genes, sizing, tick,
            min_distance, quality, loss_cost_per_unit, funding_reserve_per_unit):
    """Fill-time template: quotes must already be known, never next-open foresight.

    EURUSD, USD account, linear per-EUR costs; spread is embodied in quotes.
    Returned risk is a modeled stop loss, not a bound on gap losses.
    Calendar/position gates and action arbitration remain external.
    """
    g = coefficients(**genes)
    values = (liquidation_quote, balance, min_distance, quality,
              loss_cost_per_unit, funding_reserve_per_unit)
    if (not all(math.isfinite(v) for v in values) or min(liquidation_quote, balance) <= 0
            or min(min_distance, loss_cost_per_unit, funding_reserve_per_unit) < 0
            or not 0 <= quality <= 1
            or direction not in (-1, 1)
            or direction*(entry_quote-liquidation_quote) < -1e-12):
        raise ValueError('INVALID_ORDER_INPUT')
    if sizing.rel_volume != .1 or sizing.leverage != 50 or sizing.basis != 'margin':
        raise ValueError('BUSINESS_SIZING_CONTRACT')
    lv = levels(direction, entry_quote, tp_reference, mae_reference, sigma, g, tick)
    if (lv['sl'] <= 0 or lv['tp'] <= 0 or lv['raw_tp_distance'] <= 0
            or lv['raw_sl_distance'] <= 0
            or any(direction*(lv['tp']-p) <= 0 or direction*(p-lv['sl']) <= 0
                   or min(abs(lv['tp']-p), abs(p-lv['sl'])) < min_distance-1e-12
                   for p in (entry_quote, liquidation_quote))):
        return dict(status='NOP', reason='LEVELS_NOT_EXECUTABLE', q=0., units=0, **lv)
    maximum = sizing.units_max(balance, entry_quote)
    if maximum < sizing.min_units:
        return dict(status='NOP', reason='BELOW_MINIMUM_UNITS', q=0., units=0, **lv)
    loss_per_unit = direction*(entry_quote-lv['sl']) + loss_cost_per_unit + funding_reserve_per_unit
    risk_budget = g['rho']*balance
    sized = size_from_risk(balance, sizing.rel_volume, risk_budget,
                           maximum*loss_per_unit, quality)
    if sized['q'] == 0:
        return dict(status='NOP', reason=sized['reason'], q=0., units=0, **lv)
    units = sizing.order_units(balance, entry_quote, percentual_order_size=sized['q'],
                               risk_accepts_minimum=False)
    if units is None:
        return dict(status='NOP', reason='BELOW_MINIMUM_UNITS', q=0., units=0, **lv)
    margin, loss = sizing.required_margin(units, entry_quote), units*loss_per_unit
    if margin > .1*balance+1e-9 or loss > quality*risk_budget+1e-9:
        raise ValueError('NATIVE_SIZING_CAP_BREACH')
    return dict(status='ORDER', direction=direction, **lv, q=sized['q'], units=int(units),
                margin=margin, risk_budget=risk_budget, estimated_loss=loss,
                loss_at_q1=maximum*loss_per_unit, loss_per_unit=loss_per_unit,
                quality=quality, genes=g)
