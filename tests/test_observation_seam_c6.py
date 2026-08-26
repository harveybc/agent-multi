"""C6/finding 322: observation authority at the pipeline seam."""
import pytest

from pipeline_plugins._observation_contract import (
    apply_observation_contract, feature_columns_sha256,
    verify_flattened_dimension)


COLS = [f"f{i}" for i in range(83)]


def _cfg(**over):
    base = {
        "feature_columns": list(COLS),
        "require_feature_aware_preprocessor": True,
        "include_price_window": False,
        "include_agent_state": True,
        "window_size": 32,
        "observation_contract": {
            "require_feature_aware_preprocessor": True,
            "include_price_window": False,
            "include_agent_state": True,
            "window_size": 32,
            "feature_columns_sha256": feature_columns_sha256(COLS),
            "expected_flattened_dimension": 2660,
        },
    }
    base.update(over)
    return base


class _Space:
    """Duck-typed Dict space stand-in with a known flatdim."""

    def __init__(self, dim):
        import numpy as np
        from gymnasium import spaces
        self._s = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,))

    @property
    def space(self):
        return self._s


def test_inline_declaration_applies_at_seam():
    cfg, prov = apply_observation_contract(_cfg())
    assert prov["declared"] is True
    assert prov["source"] == "config.observation_contract"


def test_reordered_features_refused_by_sha_pin():
    cfg = _cfg(feature_columns=list(reversed(COLS)))
    with pytest.raises(ValueError, match="different feature set"):
        apply_observation_contract(cfg)


def test_extra_feature_refused_by_sha_pin():
    cfg = _cfg(feature_columns=["typical_price"] + COLS)
    with pytest.raises(ValueError, match="different feature set"):
        apply_observation_contract(cfg)


def test_missing_feature_refused_by_sha_pin():
    cfg = _cfg(feature_columns=COLS[:-1])
    with pytest.raises(ValueError, match="different feature set"):
        apply_observation_contract(cfg)


def test_wrong_flattened_dimension_refused_at_env_seam():
    cfg = _cfg()
    with pytest.raises(ValueError, match="REFUSED before model"):
        verify_flattened_dimension(cfg, _Space(2692).space)


def test_correct_flattened_dimension_passes():
    facts = verify_flattened_dimension(_cfg(), _Space(2660).space)
    assert facts["checked"] and facts["actual_flattened_dimension"] == 2660


def test_undeclared_contract_is_noop_for_dimension_check():
    cfg = _cfg()
    del cfg["observation_contract"]
    facts = verify_flattened_dimension(cfg, _Space(1234).space)
    assert facts == {"checked": False, "source": "undeclared"}


def test_required_declaration_omitted_refuses():
    # finding 327: the B4 flag makes ABSENCE a refusal, not a no-op
    cfg = _cfg()
    del cfg["observation_contract"]
    cfg["require_observation_declaration"] = True
    with pytest.raises(ValueError, match="finding 327"):
        apply_observation_contract(cfg)


def test_required_declaration_present_passes():
    cfg = _cfg()
    cfg["require_observation_declaration"] = True
    _cfg2, prov = apply_observation_contract(cfg)
    assert prov["declared"] is True
