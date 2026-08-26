"""Pre-execution tests ordered by the post-P1 order §3: formulas, strict
lag, B0 zero exposure, B1 causal entry, scored-index identity, sealed
absence."""
import math

import numpy as np
import pandas as pd
import pytest

from tools.screen_b_baselines import (BARS_PER_YEAR, CONTEXT_BARS,
                                      SIGMA_TARGET, VOL_WINDOW,
                                      ScreenBError, materialize_origin,
                                      rule_positions)


def _close(n=800, seed=7):
    rng = np.random.default_rng(seed)
    return np.exp(np.cumsum(rng.normal(0, 0.01, n))) * 1000.0


def test_b0_zero_exposure_everywhere():
    pos = rule_positions(_close(), "B0", CONTEXT_BARS)
    assert np.all(pos == 0.0)


def test_b1_causal_entry_at_first_scored_bar_only():
    pos = rule_positions(_close(), "B1", CONTEXT_BARS)
    assert np.all(pos[:CONTEXT_BARS] == 0.0)      # flat during context
    assert np.all(pos[CONTEXT_BARS:] == 1.0)      # long from first scored


def test_b2a_exact_lookback_180_on_close_t_minus_1():
    close = _close()
    t = CONTEXT_BARS + 5
    pos = rule_positions(close, "B2a", CONTEXT_BARS)
    expected = np.sign(close[t - 1] - close[t - 1 - 180])
    assert pos[t] == expected


def test_b2b_exact_lookback_540():
    close = _close()
    t = CONTEXT_BARS + 3
    pos = rule_positions(close, "B2b", CONTEXT_BARS)
    expected = np.sign(close[t - 1] - close[t - 1 - 540])
    assert pos[t] == expected


def test_b3_formula_target_window_annualization_cap():
    close = _close()
    t = CONTEXT_BARS + 10
    pos = rule_positions(close, "B3", CONTEXT_BARS)
    logret = np.diff(np.log(close), prepend=np.nan)
    sigma = float(np.std(logret[t - VOL_WINDOW:t], ddof=1)) * math.sqrt(
        BARS_PER_YEAR)
    frac = min(1.0, SIGMA_TARGET / sigma)
    sign = np.sign(close[t - 1] - close[t - 181])
    assert pos[t] == pytest.approx(sign * frac)
    assert np.all(np.abs(pos) <= 1.0 + 1e-12)  # leverage cap 1


def test_strict_lag_no_t_information():
    close = _close()
    t = CONTEXT_BARS + 20
    for arm in ("B2a", "B2b", "B3"):
        base = rule_positions(close, arm, CONTEXT_BARS)[t]
        mutated = close.copy()
        mutated[t] *= 1.5          # information AT t must not leak
        assert rule_positions(mutated, arm, CONTEXT_BARS)[t] == base
        mutated2 = close.copy()
        mutated2[t - 1] *= 1.5     # t-1 information MAY change pos[t]
        # (not asserted to change — only that t cannot)


def test_lag_sensitivity_to_t_minus_1():
    close = _close()
    t = CONTEXT_BARS + 20
    base = rule_positions(close, "B2a", CONTEXT_BARS)[t]
    mutated = close.copy()
    # force the OPPOSITE sign of the base decision (stay positive for log)
    ref = close[t - 1 - 180]
    mutated[t - 1] = ref * (0.5 if base > 0 else 2.0)
    assert rule_positions(mutated, "B2a", CONTEXT_BARS)[t] == -base


def _frame(dates):
    return pd.DataFrame({"DATE_TIME": pd.to_datetime(dates),
                         "CLOSE": np.linspace(100, 200, len(dates)),
                         "typical_price": np.linspace(100, 200,
                                                      len(dates))})


def test_origin_materialization_sealed_absence(tmp_path):
    dates = pd.date_range("2024-06-01", periods=CONTEXT_BARS + 100,
                          freq="4h")
    # extends into 2025 -> the 2024 slice would carry sealed rows
    df = _frame(list(dates) + list(
        pd.date_range("2025-01-01", periods=10, freq="4h")))
    with pytest.raises(ScreenBError, match="sealed-period"):
        materialize_origin(df, 2025, tmp_path)


def test_origin_scored_index_and_context(tmp_path):
    dates = pd.date_range("2021-10-01", periods=CONTEXT_BARS + 400,
                          freq="4h")
    df = _frame(dates)
    o = materialize_origin(df, 2022, tmp_path)
    assert o["scored_start_index"] == CONTEXT_BARS
    assert o["scored_rows"] == o["rows"] - CONTEXT_BARS
    assert len(o["scored_index_sha256"]) == 64
    assert o["score_start"].startswith("2022-01-01")


def test_insufficient_context_refused(tmp_path):
    dates = pd.date_range("2022-01-01", periods=300, freq="4h")
    with pytest.raises(ScreenBError, match="context"):
        materialize_origin(_frame(dates), 2022, tmp_path)


# --- C5: trial identity, idempotency, stats-input validation -------------

def test_trial_ids_deterministic_and_distinct(tmp_path):
    from tools.screen_b_baselines import trial_id
    o = {"year": 2022, "csv_sha256": "a" * 64}
    t1 = trial_id("B1", o, "primary", "e" * 64, "c" * 64)
    t2 = trial_id("B1", o, "primary", "e" * 64, "c" * 64)
    t3 = trial_id("B2a", o, "primary", "e" * 64, "c" * 64)
    assert t1 == t2 and t1 != t3 and len(t1) == 32


def test_ledger_idempotent_and_refuses_conflicts(tmp_path):
    from tools.screen_b_baselines import ScreenBError, register_trials
    ledger = tmp_path / "ledger.jsonl"
    row = {"trial_id": "x" * 32, "screen": "B", "arm": "B1",
           "origin": 2022}
    register_trials(ledger, [row])
    register_trials(ledger, [row])          # same content -> skip
    assert len(ledger.read_text().splitlines()) == 1
    conflict = dict(row, origin=2023)
    with pytest.raises(ScreenBError, match="DIFFERENT content"):
        register_trials(ledger, [conflict])


def test_stats_inputs_refuse_diagnostic_arms():
    from tools.screen_b_baselines import (ScreenBError,
                                          validate_stats_inputs)
    ok = {"g1_eligible": True, "cost_set": "alpaca_ethusd",
          "cost_authority": "alpaca venue primary"}
    diag = {"g1_eligible": False, "cost_set": "zero_cost",
            "cost_authority": "DIAGNOSTIC_ONLY"}
    assert validate_stats_inputs([ok, diag]) == [ok]
    with pytest.raises(ScreenBError, match="no g1-eligible"):
        validate_stats_inputs([diag])


def test_calibration_grid_is_predeclared_with_fixed_control():
    from tools.screen_b_baselines import (CALIBRATION_GRID,
                                          FIXED_CONTROL_ENVELOPE)
    assert CALIBRATION_GRID[0] == FIXED_CONTROL_ENVELOPE
    atr_cells = CALIBRATION_GRID[1:]
    assert len(atr_cells) == 6
    assert {(c["atr_sl_mult"], round(c["atr_tp_mult"] / c["atr_sl_mult"],
                                     2)) for c in atr_cells} == {
        (1.5, 1.5), (1.5, 2.0), (2.0, 1.5), (2.0, 2.0),
        (3.0, 1.5), (3.0, 2.0)}
    assert all(c["collision_rule"] == "stop_first_pessimistic"
               for c in CALIBRATION_GRID)


def test_envelope_criterion_gates_before_ranking():
    from tools.screen_b_baselines import envelope_criterion
    churny = [{"close_reason_counts": {"envelope_close_sl": 2000},
               "activity_position_changes": 100,
               "net_total_return": 1.0, "max_drawdown_fraction": 0.1}] * 4
    assert envelope_criterion(churny)["refusal"] == "pathological_churn"
    idle = [{"close_reason_counts": {}, "activity_position_changes": 0,
             "net_total_return": 0.0, "max_drawdown_fraction": 0.0}] * 4
    assert envelope_criterion(idle)["refusal"] == "no_activity"
    ok = [{"close_reason_counts": {"envelope_close_sl": 30},
           "activity_position_changes": 40,
           "net_total_return": 0.10, "max_drawdown_fraction": 0.05}] * 4
    c = envelope_criterion(ok)
    assert c["eligible"] and abs(c["composite_median"] - 0.05) < 1e-12


def test_calibration_slice_is_pre_score_year(tmp_path):
    import pandas as pd, numpy as np
    from tools.screen_b_baselines import (CONTEXT_BARS,
                                          materialize_calibration_slice)
    dates = pd.date_range("2020-06-01", periods=CONTEXT_BARS + 3000,
                          freq="4h")
    df = pd.DataFrame({"DATE_TIME": dates,
                       "CLOSE": np.linspace(100, 200, len(dates)),
                       "typical_price": np.linspace(100, 200,
                                                    len(dates))})
    cal = materialize_calibration_slice(df, 2022, tmp_path)
    assert cal["year"] == 2021
    assert cal["score_start"].startswith("2021-01-01")
    assert "2022" not in cal["score_end"][:4]
