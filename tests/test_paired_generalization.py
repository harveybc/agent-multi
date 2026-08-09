"""WP2 comparator + stopping tests (order §6.3 classes)."""
import pytest

from pipeline_plugins._paired_generalization import (
    PairedComparatorError,
    PairedStoppingState,
    paired_generalization_weekly_v1 as paired,
)


def summary(utility=0.5, trades=10, weekly=0.001, dd=0.02):
    return {"robust_weekly_rap_fitness": utility, "trades_total": trades,
            "mean_weekly_return": weekly, "max_drawdown_fraction": dd,
            "commission_paid": 1.0}


def test_scores_mean_gap_penalty():
    r = paired(summary(0.6), summary(0.4), beta=0.25)
    assert r["eligible"] and r["mean"] == pytest.approx(0.5)
    assert r["gap"] == pytest.approx(0.2)
    assert r["paired_score"] == pytest.approx(0.5 - 0.05)


def test_activity_failure_on_either_split_is_ineligible():
    assert not paired(summary(trades=0), summary(), beta=0.2)["eligible"]
    assert not paired(summary(), summary(trades=0), beta=0.2)["eligible"]


def test_lexicographic_order_key_alone_cannot_pair():
    lex_only = {"selection_contract": {"order_key": 1e15},
                "trades_total": 10}
    r = paired(lex_only, summary(), beta=0.2)
    assert not r["eligible"]
    assert any("ordinal" in reason or "utility" in reason
               for reason in r["ineligibility_reasons"])


def test_mixed_utility_sources_refuse():
    fallback_only = {"mean_weekly_rap": 0.3, "trades_total": 10,
                     "mean_weekly_return": 0.001,
                     "max_drawdown_fraction": 0.02}
    r = paired(summary(), fallback_only, beta=0.2)
    assert not r["eligible"]


def test_deterministic_tie_break():
    a = paired(summary(0.5), summary(0.5), beta=0.2, candidate_id="a")
    b = paired(summary(0.5), summary(0.5), beta=0.2, candidate_id="b")
    assert a["tie_break"] != b["tie_break"]
    assert a["tie_break"][:-1] == b["tie_break"][:-1]


def _state():
    return PairedStoppingState(patience=3, floor=2, min_delta=0.01,
                               split_identity="sid")


def test_validation_up_train_down_no_false_improvement():
    state = _state()
    state.update(paired(summary(0.5), summary(0.5), beta=0.5), 1)
    # validation rises but train collapses: gap erases the mean gain
    out = state.update(paired(summary(0.1), summary(0.9), beta=0.5), 2)
    assert not out["improved"]


def test_both_improve_resets_patience():
    state = _state()
    state.update(paired(summary(0.4), summary(0.4), beta=0.2), 1)
    state.update(paired(summary(0.35), summary(0.35), beta=0.2), 3)
    assert state.waited == 1
    out = state.update(paired(summary(0.6), summary(0.6), beta=0.2), 4)
    assert out["improved"] and state.waited == 0


def test_gap_growth_erasing_mean_gain_never_resets():
    state = _state()
    state.update(paired(summary(0.5), summary(0.5), beta=1.0), 1)
    out = state.update(paired(summary(0.9), summary(0.3), beta=1.0), 3)
    assert not out["improved"] and state.waited == 1


def test_ineligible_never_touches_improvement_patience():
    state = _state()
    state.update(paired(summary(0.5), summary(0.5), beta=0.2), 1)
    before = state.waited
    state.update(paired(summary(trades=0), summary(), beta=0.2), 3)
    assert state.waited == before


def test_stop_fires_after_patience_past_floor_only():
    state = _state()
    state.update(paired(summary(0.9), summary(0.9), beta=0.2), 1)
    assert not state.update(paired(summary(0.1), summary(0.1), beta=0.2), 1)["stop"]
    for index in (2, 3):
        out = state.update(paired(summary(0.1), summary(0.1), beta=0.2), index)
    out = state.update(paired(summary(0.1), summary(0.1), beta=0.2), 4)
    assert out["stop"]


def test_resume_preserves_state_and_split_identity():
    state = _state()
    state.update(paired(summary(0.7), summary(0.7), beta=0.2), 1)
    state.update(paired(summary(0.1), summary(0.1), beta=0.2), 3)
    revived = PairedStoppingState.from_state(state.to_state(),
                                             split_identity="sid")
    assert revived.best_score == state.best_score
    assert revived.waited == state.waited
    with pytest.raises(PairedComparatorError, match="identity mismatch"):
        PairedStoppingState.from_state(state.to_state(),
                                       split_identity="other")


def test_bad_beta_refuses():
    with pytest.raises(PairedComparatorError):
        paired(summary(), summary(), beta=float("nan"))
