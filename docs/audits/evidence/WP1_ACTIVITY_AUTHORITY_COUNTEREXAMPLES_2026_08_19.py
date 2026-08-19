#!/usr/bin/env python3
"""Independent counterexamples for activity-authority WP1 at 3069d564."""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _activity_authority as activity  # noqa: E402
from pipeline_plugins import _lexicographic_selection as selection  # noqa: E402
from pipeline_plugins import rl_pipeline_with_validation as pipeline  # noqa: E402


def _capture(callable_):
    try:
        return {"returned": callable_(), "raised": None}
    except Exception as error:  # The exception class is the evidence.
        return {
            "returned": None,
            "raised": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }


def main() -> int:
    malformed_counts = {
        repr(value): _capture(
            lambda value=value: activity.evaluate_role_activity(
                value, role="inner_validation"
            )
        )
        for value in (1.5, True, "1", float("inf"))
    }
    malformed_floors = {
        repr(value): _capture(
            lambda value=value: activity.resolve_floor(
                {"activity_floor": value}
            )
        )
        for value in (1.5, True, "1", float("inf"))
    }

    evidence_free = activity.evaluate_activity(
        train_monitor_trades=1,
        inner_validation_trades=1,
    )
    floor_one = activity.evaluate_role_activity(
        12, role="inner_validation", floor=1
    )
    floor_twelve = activity.evaluate_role_activity(
        12, role="inner_validation", floor=12
    )
    missing_trade_selection = selection.evaluate_selection_contract(
        {
            "mean_weekly_return": 0.01,
            "max_drawdown_fraction": 0.10,
            "total_return": 0.20,
        },
        min_trades=0,
    )
    silent_zero_floor = selection.evaluate_selection_contract(
        {
            "mean_weekly_return": -0.01,
            "max_drawdown_fraction": 0.10,
            "total_return": -0.20,
            "trades_total": 1,
        },
        min_trades=0,
    )
    missing_pipeline_trade = _capture(lambda: pipeline._trade_count({}))

    reproduced = {
        "fractional_count_is_truncated_and_eligible": (
            malformed_counts["1.5"]["returned"]["eligible"] is True
            and malformed_counts["1.5"]["returned"]["trades"] == 1
        ),
        "boolean_count_is_eligible": (
            malformed_counts["True"]["returned"]["eligible"] is True
        ),
        "string_count_is_eligible": (
            malformed_counts["'1'"]["returned"]["eligible"] is True
        ),
        "infinite_count_crashes_instead_of_typed_unavailable": (
            malformed_counts["inf"]["raised"]["type"] == "OverflowError"
        ),
        "malformed_floors_are_coerced_or_crash": (
            malformed_floors["1.5"]["returned"] == 1
            and malformed_floors["True"]["returned"] == 1
            and malformed_floors["'1'"]["returned"] == 1
            and malformed_floors["inf"]["raised"]["type"]
            == "OverflowError"
        ),
        "missing_activity_evidence_still_allows_selection": (
            evidence_free["eligible"] is True
            and evidence_free["selection_score_permitted"] is True
            and evidence_free["evidence_refs"] == []
            and evidence_free["active_weeks_available"] is False
            and evidence_free["exposure_fraction_available"] is False
        ),
        "different_floors_share_one_contract_id": (
            floor_one["threshold_contract_id"]
            == floor_twelve["threshold_contract_id"]
        ),
        "missing_trade_fact_is_rendered_as_zero": (
            missing_trade_selection["components"]["trades_total"] == 0
        ),
        "ineligible_candidate_still_has_numeric_transport_scalar": (
            missing_trade_selection["eligible"] is False
            and missing_trade_selection["transport_scalar"] == 0.0
        ),
        "zero_floor_is_silently_promoted_to_one": (
            silent_zero_floor["eligible"] is True
            and silent_zero_floor["components"]["min_trades_required"] == 1
        ),
        "pipeline_missing_trade_crashes_before_typed_authority": (
            missing_pipeline_trade["raised"]["type"] == "ValueError"
        ),
    }
    packet = {
        "schema": "agent_multi.wp1_activity_authority_counterexamples.v1",
        "commit_under_test": "3069d56450eff9fd505c9e0a3089a7ef69d2c0bd",
        "malformed_counts": malformed_counts,
        "malformed_floors": malformed_floors,
        "evidence_free_result": evidence_free,
        "floor_one": floor_one,
        "floor_twelve": floor_twelve,
        "missing_trade_selection": missing_trade_selection,
        "silent_zero_floor_selection": silent_zero_floor,
        "missing_pipeline_trade": missing_pipeline_trade,
        "reproduced": reproduced,
        "all_reproduced": all(reproduced.values()),
    }
    print(json.dumps(packet, indent=2, sort_keys=True, allow_nan=False))
    return 0 if packet["all_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
