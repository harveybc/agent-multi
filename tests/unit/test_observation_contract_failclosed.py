"""AUD-P1LR-20260815-235: the observation contract is fail-closed.

The old guard opened with ``if not config.get(
"require_feature_aware_preprocessor", False): return``.  Every experiment
that mattered declared neither field, so it validated nothing while 64
unnormalized raw-price dimensions reached the actor.  These tests hold
the reversed default in place: absence of a declaration REFUSES, the
opt-out must be written down, and the L2 program's declaration actually
reaches a materialized runtime config.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline_plugins._observation_contract import (
    UNDECLARED_OBSERVATION_CONTRACT,
    UNWAIVED_RAW_OBSERVATION,
    apply_observation_contract,
    declared_observation_contract,
    feature_columns_sha256,
    validate_observation_contract,
)

REPO = Path(__file__).resolve().parents[2]
L2_CONTRACT = (
    REPO / "examples/config/phase_3_eth_sac_dynamics/l2_curriculum_arms_v1.json")
BASE_CONFIG = (
    REPO / "examples/results/project3_ethusdt_4h_sac_train_val_test_v2"
           "/config_out.json")


def _feature_aware_config(**overrides):
    config = {
        "require_feature_aware_preprocessor": True,
        "preprocessor_plugin": "feature_window_preprocessor",
        "feature_columns": ["return_1", "rsi_14"],
        "feature_scaling": "rolling_zscore",
        "include_price_window": False,
    }
    config.update(overrides)
    return config


# --------------------------------------------------------------------------
# the reversed default
# --------------------------------------------------------------------------

def test_undeclared_observation_contract_refuses():
    """The exact hole that let three campaigns measure nothing."""
    with pytest.raises(ValueError, match=UNDECLARED_OBSERVATION_CONTRACT):
        validate_observation_contract({})


def test_undeclared_refuses_even_for_a_full_looking_runtime_config():
    config = _feature_aware_config()
    config.pop("require_feature_aware_preprocessor")
    with pytest.raises(ValueError, match=UNDECLARED_OBSERVATION_CONTRACT):
        validate_observation_contract(config)


def test_the_real_base_config_alone_is_now_refused():
    """The base config every L1/L2 candidate is materialized from.

    It declares ``include_price_window: true`` and no contract. Under the
    old default it sailed through; it must not any more.
    """
    config = json.loads(BASE_CONFIG.read_text())
    assert config["include_price_window"] is True
    assert "require_feature_aware_preprocessor" not in config
    with pytest.raises(ValueError, match=UNDECLARED_OBSERVATION_CONTRACT):
        validate_observation_contract(config)


def test_opt_out_without_a_written_reason_refuses():
    with pytest.raises(ValueError, match=UNWAIVED_RAW_OBSERVATION):
        validate_observation_contract(
            {"require_feature_aware_preprocessor": False,
             "include_price_window": True})


def test_opt_out_with_a_short_reason_refuses():
    with pytest.raises(ValueError, match=UNWAIVED_RAW_OBSERVATION):
        validate_observation_contract(
            {"require_feature_aware_preprocessor": False,
             "observation_contract_waiver_reason": "n/a"})


def test_written_opt_out_is_allowed_and_typed():
    facts = validate_observation_contract(
        {"require_feature_aware_preprocessor": False,
         "include_price_window": True,
         "observation_contract_waiver_reason":
             "legacy anchor embeds the raw window; diagnostic only"})
    assert facts["outcome"] == "OBSERVATION_CONTRACT_WAIVED"
    assert facts["feature_aware"] is False
    assert facts["include_price_window"] is True


def test_feature_aware_declaration_returns_typed_facts():
    facts = validate_observation_contract(_feature_aware_config())
    assert facts["outcome"] == "FEATURE_AWARE_OBSERVATION_CONTRACT"
    assert facts["feature_aware"] is True
    assert facts["include_price_window"] is False
    assert facts["feature_column_count"] == 2


# --------------------------------------------------------------------------
# none of the original four refusals were weakened
# --------------------------------------------------------------------------

def test_raw_price_window_still_refused_under_a_feature_aware_declaration():
    with pytest.raises(ValueError, match="raw price window"):
        validate_observation_contract(
            _feature_aware_config(include_price_window=True))


def test_default_preprocessor_still_refused():
    with pytest.raises(ValueError, match="feature_window_preprocessor"):
        validate_observation_contract(
            _feature_aware_config(preprocessor_plugin="default_preprocessor"))


def test_duplicate_feature_columns_still_refused():
    with pytest.raises(ValueError, match="duplicates"):
        validate_observation_contract(
            _feature_aware_config(feature_columns=["a", "a"]))


def test_non_causal_scaling_still_refused():
    with pytest.raises(ValueError, match="causal z-score"):
        validate_observation_contract(
            _feature_aware_config(feature_scaling="minmax"))


# --------------------------------------------------------------------------
# the L2 program's declaration actually reaches a runtime config
# --------------------------------------------------------------------------

def test_l2_contract_declares_the_corrected_observation_fields():
    contract = json.loads(L2_CONTRACT.read_text())
    block = contract["observation_contract"]
    assert block["require_feature_aware_preprocessor"] is True
    assert block["include_price_window"] is False
    assert block["preprocessor_plugin"] == "feature_window_preprocessor"
    assert block["feature_scaling"] == "rolling_zscore"


def test_l2_declaration_binds_onto_the_base_config_and_then_validates():
    """End to end: base config -> content-addressed contract -> validator."""
    config = json.loads(BASE_CONFIG.read_text())
    config["_identity"] = {
        "experiment_contract_sha256":
            hashlib.sha256(L2_CONTRACT.read_bytes()).hexdigest()}

    declared, source = declared_observation_contract(config)
    assert declared is not None
    assert source.endswith("l2_curriculum_arms_v1.json")

    bound, provenance = apply_observation_contract(config)
    assert provenance["declared"] is True
    assert provenance["applied"]["include_price_window"] == {
        "from": True, "to": False}
    assert bound["include_price_window"] is False
    assert bound["require_feature_aware_preprocessor"] is True

    facts = validate_observation_contract(bound)
    assert facts["outcome"] == "FEATURE_AWARE_OBSERVATION_CONTRACT"
    assert facts["feature_column_count"] == 83


def test_apply_is_a_no_op_without_a_declaration():
    config = {"include_price_window": True}
    bound, provenance = apply_observation_contract(config)
    assert bound is config
    assert provenance["declared"] is False
    assert provenance["applied"] == {}


def test_apply_refuses_a_contract_that_binds_a_non_observation_field():
    config = {"observation_contract": {"include_price_window": False,
                                       "learning_rate": 1e-3}}
    with pytest.raises(ValueError, match="unknown keys"):
        apply_observation_contract(config)


def test_apply_refuses_a_contract_pinned_to_a_different_feature_set():
    config = {
        "feature_columns": ["return_1"],
        "observation_contract": {
            "include_price_window": False,
            "feature_columns_sha256": feature_columns_sha256(["other"]),
        },
    }
    with pytest.raises(ValueError, match="different feature set"):
        apply_observation_contract(config)


def test_feature_columns_digest_is_order_sensitive():
    assert feature_columns_sha256(["a", "b"]) != feature_columns_sha256(
        ["b", "a"])


def test_a_runner_that_forgets_the_binding_refuses_instead_of_running_dead():
    """The safe state when the contract never reaches the runtime config.

    ``l2_curriculum_arms.evaluate_candidate`` pops ``_identity`` before it
    hands the config to the pipeline, so a candidate config that carries
    no inline ``observation_contract`` block reaches the validator with no
    declaration at all.  Under the OLD default that ran the raw price
    window silently.  It must now refuse.
    """
    base = json.loads(BASE_CONFIG.read_text())
    base.pop("_identity", None)
    bound, provenance = apply_observation_contract(base)
    assert provenance["declared"] is False
    with pytest.raises(ValueError, match=UNDECLARED_OBSERVATION_CONTRACT):
        validate_observation_contract(bound)


def test_the_inline_binding_is_all_a_runner_needs():
    """The one line a runner adds: config['observation_contract'] = ...

    Given it, the base config's raw price window is replaced by the
    program's declaration and the validator passes.
    """
    contract = json.loads(L2_CONTRACT.read_text())
    config = json.loads(BASE_CONFIG.read_text())
    config["observation_contract"] = contract["observation_contract"]

    bound, provenance = apply_observation_contract(config)
    assert provenance["source"] == "config.observation_contract"
    assert bound["include_price_window"] is False
    assert validate_observation_contract(bound)["feature_aware"] is True
