"""The committed FROZEN_DEVELOPMENT ETH 4h manifest binds: 83 features, split dates, pilot allowed, DEVELOPMENT class."""
from __future__ import annotations

from pathlib import Path

import pytest

from ._mechanisms import ROOT, require

MANIFEST = ROOT / "examples/config/rl_temporal/SELECTED_FEATURE_MANIFEST.eth_4h.v1.FROZEN_DEVELOPMENT.json"


def test_frozen_development_manifest_binds_and_allows_pilot():
    Binding = require("ARMS", "rl_temporal.lake_binding", "SelectedFeatureBinding", "manifest binding")
    b = Binding.from_manifest(MANIFEST, selected_variant="A_all_admissible_control",
                              expected_sha256="fdff0c85fc376cd6930cede4981a64b076022bd701c6a045339689e3ab892d4c")
    assert b.status == "FROZEN_DEVELOPMENT" and b.frozen and len(b.feature_order) == 83
    assert b.feature_order[:3] == ("return_1", "log_return_1", "return_5")
    assert b.task["availability_class"] == "DEVELOPMENT"
    b.refuse_real_data_pilot()
    res = b.verify_resources()
    assert res["model_ready_view"]["reason"] == "NOT_RESOLVABLE" or res["model_ready_view"]["verified"]
    with pytest.raises(ValueError, match="selected_variant"):
        Binding.from_manifest(MANIFEST, selected_variant="B_screen_union")


def test_cells_carry_split_dates_and_development_class(tmp_path):
    Binding = require("ARMS", "rl_temporal.lake_binding", "SelectedFeatureBinding", "manifest binding")
    build = require("ARMS", "rl_temporal.arms", "build_arm_config", "arm config builder")
    b = Binding.from_manifest(MANIFEST, selected_variant="A_all_admissible_control")
    cfg = build("RL-S1", b, seed=101, out_dir=str(tmp_path), window=24, sample_hours=4)
    assert cfg["train_start"].startswith("2017-09-28") and cfg["train_end"].startswith("2024-01-01")
    assert cfg["validation_start"].startswith("2024-01-01") and cfg["validation_end"].startswith("2025-01-01")
    assert cfg["test_start"].startswith("2025-01-01") and cfg["evaluate_test_split"] is False
    assert cfg["pilot_gate"] == {"manifest_status": "FROZEN_DEVELOPMENT", "real_data_fit_allowed": True,
                                 "availability_class": "DEVELOPMENT", "evidence_class": "DEVELOPMENT", "reason": None}
    assert cfg["heartbeat_file"].endswith("heartbeat.json") and cfg["hard_limits"]["max_wall_s"] == 14400
    assert cfg["representation"]["modular_config"]["sample_hours"] == 4
