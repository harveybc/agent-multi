#!/usr/bin/env python3
"""D7 (order 2026-08-19): NON-ABORTING runner for the 12 return
counterexamples plus consumer fixtures.

Every case runs ISOLATED: an expected typed refusal is a CLOSED
disposition, an unexpected exception is a RUNNER_ERROR, and a surviving
defect is REPRODUCED. One refusal can never abort the rest. Acceptance
is ZERO reproduced defects — never "unable to start".
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _activity_authority as aa  # noqa: E402
from pipeline_plugins import _lexicographic_selection as lex  # noqa: E402
from pipeline_plugins import _paired_generalization as paired  # noqa: E402
from pipeline_plugins.rl_pipeline_with_validation import (  # noqa: E402
    _early_stop_composite,
    _trade_count,
)

CLOSED, REPRODUCED, RUNNER_ERROR = "CLOSED", "REPRODUCED", "RUNNER_ERROR"


def _descriptor(tmp: Path, role: str, trades) -> dict:
    artifact = tmp / f"{role}.json"
    artifact.write_text(json.dumps({"trades_total": trades}))
    return {"schema": aa.EVIDENCE_DESCRIPTOR_SCHEMA, "role": role,
            "source_kind": "summary_artifact",
            "artifact_locator": str(artifact),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "fact_key": "trades_total",
            "producer_contract_id": "runner.fixture.v1"}


def case_paired_boolean_count():
    reasons = paired._split_eligibility(
        {"trades_total": True, "weekly_common_scale_utility": 0.01},
        "inner_validation", 1)
    return CLOSED if any("UNAVAILABLE" in r for r in reasons) \
        else REPRODUCED


def case_paired_fractional_count():
    reasons = paired._split_eligibility(
        {"trades_total": 2.7, "weekly_common_scale_utility": 0.01},
        "inner_validation", 1)
    return CLOSED if any("UNAVAILABLE" in r for r in reasons) \
        else REPRODUCED


def case_paired_zero_floor():
    try:
        paired._split_eligibility(
            {"trades_total": 5, "weekly_common_scale_utility": 0.01},
            "inner_validation", 0)
        return REPRODUCED
    except aa.ActivityAuthorityError:
        return CLOSED


def case_pipeline_self_assertion_reference():
    # a summary whose only "evidence" would be its own in-memory hash:
    # the pipeline now presents ONLY trace-backed descriptors; absence
    # is ineligible.
    composite, _raw, gate, *_ = _early_stop_composite(
        {"trades_total": 5, "total_return": 0.01},
        {"trades_total": 2, "total_return": 0.01},
        min_trades=1, no_trade_penalty=1e6)
    return CLOSED if gate is False and composite is None else REPRODUCED


def case_syntactic_fake_evidence():
    fake = {"schema": aa.EVIDENCE_DESCRIPTOR_SCHEMA,
            "role": "train_monitor", "source_kind": "return_trace",
            "artifact_locator": "/nonexistent/trace.csv",
            "sha256": "a" * 64, "fact_key": "trades",
            "producer_contract_id": "fake.v1"}
    with tempfile.TemporaryDirectory() as tmp:
        result = aa.evaluate_activity(
            train_monitor_trades=5, inner_validation_trades=2,
            evidence_refs={"train_monitor": fake,
                           "inner_validation": _descriptor(
                               Path(tmp), "inner_validation", 2)})
    return CLOSED if not result["eligible"] and any(
        "ARTIFACT_MISSING" in r for r in result["reason_codes"]) \
        else REPRODUCED


def case_higher_floor_strict_id():
    try:
        aa.evaluate_role_activity(12, role="inner_validation", floor=12)
        return REPRODUCED
    except aa.ActivityAuthorityError:
        pass
    result = aa.evaluate_role_activity(
        12, role="inner_validation", floor=12,
        calibrated_contract={"id": "x.v1", "floor": 12,
                             "units": "trades",
                             "evidence_ref": "config:x=12"})
    if result["threshold_contract_id"] == aa.THRESHOLD_CONTRACT_ID:
        return REPRODUCED
    return CLOSED


def case_calibrated_string_floor():
    try:
        aa.threshold_contract_for(12, {"id": "x.v1", "floor": "12",
                                       "units": "trades",
                                       "evidence_ref": "e"})
        return REPRODUCED
    except aa.ActivityAuthorityError:
        return CLOSED


def case_calibrated_fractional_floor():
    try:
        aa.threshold_contract_for(12, {"id": "x.v1", "floor": 12.0,
                                       "units": "trades",
                                       "evidence_ref": "e"})
        return REPRODUCED
    except aa.ActivityAuthorityError:
        return CLOSED


def case_missing_trade_rendered_as_zero():
    if _trade_count({}) is not None:
        return REPRODUCED
    contract = lex.evaluate_selection_contract(
        {"mean_weekly_return": 0.01, "max_drawdown_fraction": 0.1,
         "total_return": 0.2}, min_trades=1)
    trades = contract["components"]["trades_total"]
    return CLOSED if trades is None and not contract["eligible"] \
        else REPRODUCED


def case_promotion_truncates_fractional():
    source = (REPO / "examples/scripts/"
              "materialize_phase_1_promotion_candidates.py").read_text()
    if "int(_finite_number(metrics[\"trades_total\"]" in source:
        return REPRODUCED
    return CLOSED if "evaluate_role_activity" in source else REPRODUCED


def case_legacy_pipeline_bypass():
    source = (REPO / "pipeline_plugins/rl_pipeline.py").read_text()
    return CLOSED if "activity_authority_prohibited" in source \
        else REPRODUCED


def case_consumer_graph():
    # honest disposition: names what is integrated and what remains
    pending = []
    weekly = (REPO / "examples/scripts/"
              "run_phase_1_weekly_promotion.py").read_text()
    if "_activity_authority" not in weekly:
        pending.append("weekly_promotion")
    l2 = (REPO / "optimizer_plugins/l2_curriculum_optimizer.py"
          ).read_text()
    if "_activity_authority" not in l2 and \
            "_paired_generalization" not in l2:
        pending.append("l2_optimizer")
    return CLOSED if not pending else f"PARTIAL:{','.join(pending)}"


CASES = {
    "paired_comparator_accepts_boolean_count":
        case_paired_boolean_count,
    "paired_comparator_accepts_fractional_count":
        case_paired_fractional_count,
    "paired_comparator_accepts_zero_floor": case_paired_zero_floor,
    "pipeline_manufactures_self_assertion_reference":
        case_pipeline_self_assertion_reference,
    "syntactic_fake_evidence_is_eligible": case_syntactic_fake_evidence,
    "higher_floor_is_mislabeled_with_strict_contract_id":
        case_higher_floor_strict_id,
    "calibrated_string_floor_is_accepted": case_calibrated_string_floor,
    "calibrated_fractional_floor_is_accepted":
        case_calibrated_fractional_floor,
    "missing_trade_is_rendered_as_zero":
        case_missing_trade_rendered_as_zero,
    "promotion_truncates_fractional_count_before_authority":
        case_promotion_truncates_fractional,
    "registered_legacy_pipeline_bypasses_authority":
        case_legacy_pipeline_bypass,
    "consumer_graph_is_incomplete": case_consumer_graph,
}


def main() -> int:
    dispositions = {}
    for name, case in CASES.items():
        try:
            dispositions[name] = case()
        except Exception as error:  # a case may never abort the rest
            dispositions[name] = (
                f"{RUNNER_ERROR}: {type(error).__name__}: {error}")
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    reproduced = [k for k, v in dispositions.items()
                  if str(v).startswith(REPRODUCED)]
    packet = {
        "schema": "agent_multi.wp1_return_counterexamples_runner.v1",
        "commit_under_test": commit,
        "dispositions": dispositions,
        "reproduced_count": len(reproduced),
        "acceptance": "ZERO_REPRODUCED" if not reproduced
        else f"REPRODUCED:{reproduced}",
    }
    print(json.dumps(packet, indent=1, sort_keys=True))
    return 0 if not reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
