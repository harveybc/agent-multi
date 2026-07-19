from __future__ import annotations

import pytest

from optimizer_plugins.default_optimizer import _action_collapse_evidence
from pipeline_plugins._observation_contract import validate_observation_contract
from pipeline_plugins.rl_pipeline import (
    _action_summary_fields,
    _new_action_stats,
    _update_action_stats,
)


def _summary_for_actions(actions: list[float], coerced: int) -> dict:
    stats = _new_action_stats(continuous_threshold=0.1)
    for action in actions:
        _update_action_stats(stats, [action], {"coerced_action": coerced})
    return _action_summary_fields(stats, {"trades_total": 1})


def test_action_summary_detects_exact_policy_collapse() -> None:
    summary = _summary_for_actions([-0.96402758] * 128, coerced=2)

    assert summary["action_raw_std"] == pytest.approx(0.0, abs=1e-7)
    assert summary["action_raw_range"] == pytest.approx(0.0)
    assert summary["action_dominant_side"] == "short"
    assert summary["action_dominant_rate"] == pytest.approx(1.0)


def test_optimizer_rejects_collapse_on_train_tail_and_validation() -> None:
    constant = _summary_for_actions([-0.96402758] * 128, coerced=2)
    evidence = _action_collapse_evidence(
        {"splits": {"train_tail": constant, "validation": constant}},
        {
            "optimization_reject_action_collapse": True,
            "optimization_action_collapse_min_steps": 64,
            "optimization_min_action_std": 1e-5,
            "optimization_max_dominant_action_rate": 0.999,
        },
    )

    assert evidence["policy_action_collapse_rejected"] is True
    assert evidence["candidate_rejected_reason"] == "deterministic_policy_action_collapse"
    assert evidence["policy_action_collapse_splits"] == ["train_tail", "validation"]


def test_optimizer_accepts_observation_sensitive_actions() -> None:
    varied = _summary_for_actions([-0.8, -0.2, 0.0, 0.3, 0.9] * 26, coerced=1)
    evidence = _action_collapse_evidence(
        {"splits": {"train_tail": varied, "validation": varied}},
        {"optimization_reject_action_collapse": True},
    )

    assert evidence["policy_action_collapse_rejected"] is False
    assert evidence["candidate_rejected_reason"] is None


def test_feature_aware_contract_rejects_default_preprocessor() -> None:
    with pytest.raises(ValueError, match="feature_window_preprocessor"):
        validate_observation_contract(
            {
                "require_feature_aware_preprocessor": True,
                "preprocessor_plugin": "default_preprocessor",
            }
        )


def test_feature_aware_contract_rejects_raw_price_window() -> None:
    with pytest.raises(ValueError, match="raw price window"):
        validate_observation_contract(
            {
                "require_feature_aware_preprocessor": True,
                "preprocessor_plugin": "feature_window_preprocessor",
                "feature_columns": ["return_1"],
                "feature_scaling": "rolling_zscore",
                "include_price_window": True,
            }
        )


def test_feature_aware_contract_accepts_causal_scaled_features() -> None:
    validate_observation_contract(
        {
            "require_feature_aware_preprocessor": True,
            "preprocessor_plugin": "feature_window_preprocessor",
            "feature_columns": ["return_1", "rsi_14"],
            "feature_scaling": "rolling_zscore",
            "include_price_window": False,
        }
    )
