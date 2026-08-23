import pytest

from agent_plugins.feature_families import (
    baseline_grouped_architecture,
    semantic_feature_families,
)


def test_semantic_families_are_exhaustive_and_disjoint():
    columns = ["return_1", "macd", "rsi_14", "atr_14", "obv_delta_20"]
    groups = semantic_feature_families(columns)
    flattened = [item for values in groups.values() for item in values]
    assert sorted(flattened) == sorted(columns)
    assert len(flattened) == len(set(flattened))


def test_unknown_feature_refuses_instead_of_joining_miscellaneous():
    with pytest.raises(ValueError, match="maps to 0"):
        semantic_feature_families(["mystery_signal"])


def test_baseline_is_explicitly_configurable():
    architecture = baseline_grouped_architecture(
        ["return_1", "macd", "rsi_14", "atr_14", "obv_delta_20"]
    )
    assert architecture["fusion"]["plugin"] == "gated_fusion"
    assert {branch["plugin"] for branch in architecture["branches"]} == {
        "tcn_branch", "transformer_branch", "gru_branch"
    }
