from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.live_parity import (
    audit_experiment_live_parity,
    load_live_contract,
    validate_live_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "examples"
    / "config"
    / "live_parity"
    / "project3_realtime_feature_asset_contract_v1.json"
)
EXPERIMENT_PATH = (
    ROOT
    / "examples"
    / "config"
    / "phase_1_asset_policy"
    / "optimization"
    / "phase_1_asset_policy_usdcad_4h_protected_easy_v2.json"
)


def _experiment() -> dict:
    return json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8"))


def test_current_usdcad_is_research_eligible_but_fails_closed_for_live() -> None:
    report = audit_experiment_live_parity(_experiment(), load_live_contract(CONTRACT_PATH))

    assert report["research_eligible"] is True
    assert report["live_inference_eligible"] is False
    assert report["live_execution_eligible"] is False
    assert "runtime_capability_not_integrated:closed_ohlcv_bars" in report["blockers"][
        "live_inference"
    ]
    assert "runtime_parity_not_passed:closed_ohlcv_bars" in report["blockers"][
        "live_inference"
    ]


def test_integrated_feed_and_protected_route_can_pass_live_gate() -> None:
    contract = load_live_contract(CONTRACT_PATH)
    contract = copy.deepcopy(contract)
    contract["runtime_capabilities"]["closed_ohlcv_bars"].update(
        {"status": "integrated", "parity_status": "passed"}
    )
    cell = next(
        value
        for value in contract["asset_cells"]
        if value["asset"] == "usdcad" and value["timeframe"] == "4h"
    )
    cell["data_route_status"] = "observed"
    cell["execution_route_status"] = "protected_canary_passed"

    report = audit_experiment_live_parity(_experiment(), contract)

    assert report["research_eligible"] is True
    assert report["live_inference_eligible"] is True
    assert report["live_execution_eligible"] is True


def test_unregistered_feature_profile_is_not_silently_promoted() -> None:
    experiment = _experiment()
    experiment["data"]["data_profile"] = "future_paid_oracle"

    report = audit_experiment_live_parity(experiment, load_live_contract(CONTRACT_PATH))

    assert report["research_eligible"] is False
    assert report["live_inference_eligible"] is False
    assert "feature_profile_unregistered:future_paid_oracle" in report["blockers"]["research"]


def test_unknown_runtime_route_is_rejected_by_contract_validation() -> None:
    contract = load_live_contract(CONTRACT_PATH)
    contract = copy.deepcopy(contract)
    contract["asset_cells"][0]["data_routes"].append("imaginary_realtime_oracle")

    with pytest.raises(ValueError, match="unknown runtime source"):
        validate_live_contract(contract)
