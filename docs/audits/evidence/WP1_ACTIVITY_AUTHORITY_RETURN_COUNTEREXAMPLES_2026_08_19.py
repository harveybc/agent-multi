#!/usr/bin/env python3
"""Independent return-audit counterexamples for activity authority WP1.

Unlike the delivery's post-fix evidence, this runner captures every case and
never lets one expected refusal prevent the remaining cases from executing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _capture(call: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"returned": call(), "raised": None}
    except Exception as error:  # The exact exception is audit evidence.
        return {
            "returned": None,
            "raised": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }


def _promotion_module():
    path = REPO / "examples/scripts/materialize_phase_1_promotion_candidates.py"
    spec = importlib.util.spec_from_file_location("promotion_under_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from pipeline_plugins import _activity_authority as activity
    from pipeline_plugins import _lexicographic_selection as selection
    from pipeline_plugins import _paired_generalization as paired
    from pipeline_plugins.rl_pipeline_with_validation import (
        _activity_evidence_ref,
    )

    promotion = _promotion_module()
    utility = {
        "robust_weekly_rap_fitness": -0.2,
        "mean_weekly_return": -0.01,
        "max_drawdown_fraction": 0.5,
    }

    missing_selection = selection.evaluate_selection_contract(
        {
            "mean_weekly_return": 0.01,
            "max_drawdown_fraction": 0.10,
            "total_return": 0.20,
        },
        min_trades=1,
    )
    higher_floor_selection = selection.evaluate_selection_contract(
        {
            "mean_weekly_return": 0.01,
            "max_drawdown_fraction": 0.10,
            "total_return": 0.20,
            "trades_total": 12,
        },
        min_trades=12,
    )

    fake_hash = "a" * 40
    fake_evidence = activity.evaluate_activity(
        train_monitor_trades=1,
        inner_validation_trades=1,
        evidence_refs={
            "train_monitor": fake_hash,
            "inner_validation": fake_hash,
        },
    )

    promotion_tx = {
        "tx_type": "optimae_accepted",
        "domain_id": "audit-domain",
        "payload": {
            "verified_performance": 1.0,
            "parameters": {"audit": True},
            "champion_metrics": {
                "train_validation_l1_score": 0.1,
                "risk_adjusted_total_return": 0.1,
                "total_return": 0.1,
                "max_drawdown_fraction": 0.1,
                "trades_total": 1.9,
            },
        },
    }
    promotion_candidate = promotion._candidate_from_transaction(
        block={"index": 1},
        transaction=promotion_tx,
        domain_id="audit-domain",
        min_trades=1,
    )

    paired_bool = paired.paired_generalization_weekly_v1(
        {**utility, "trades_total": True},
        {**utility, "trades_total": True},
        beta=0.25,
    )
    paired_fractional = paired.paired_generalization_weekly_v1(
        {**utility, "trades_total": 1.5},
        {**utility, "trades_total": 1.5},
        beta=0.25,
    )
    paired_zero_floor = paired.paired_generalization_weekly_v1(
        {**utility, "trades_total": 0},
        {**utility, "trades_total": 0},
        beta=0.25,
        min_trades_a=0,
        min_trades_b=0,
    )

    calibrated_string_floor = _capture(
        lambda: activity.threshold_contract_for(
            12,
            {
                "id": "audit.activity_floor.v1",
                "floor": "12",
                "units": "trades",
                "evidence_ref": "audit",
            },
        )
    )
    calibrated_fractional_floor = _capture(
        lambda: activity.threshold_contract_for(
            12,
            {
                "id": "audit.activity_floor.v1",
                "floor": 12.9,
                "units": "trades",
                "evidence_ref": "audit",
            },
        )
    )

    after_map_path = (
        REPO
        / "docs/audits/evidence/WP1_ACTIVITY_CONSUMER_MAP_AFTER_2026_08_19.json"
    )
    after_map = json.loads(after_map_path.read_text(encoding="utf-8"))
    pending_consumers = sorted(
        name
        for name, facts in after_map["consumers_after"].items()
        if "PENDING" in json.dumps(facts, sort_keys=True)
    )

    setup_text = (REPO / "setup.py").read_text(encoding="utf-8")
    legacy_configs = sorted(
        str(path.relative_to(REPO))
        for path in (REPO / "examples/config").rglob("*.json")
        if '"pipeline_plugin": "rl_pipeline"' in path.read_text(
            encoding="utf-8", errors="replace"
        )
    )

    facts = {
        "promotion_fractional_count": {
            "accepted": promotion_candidate is not None,
            "input": 1.9,
            "persisted": (
                promotion_candidate["validation_evidence"]["trades_total"]
                if promotion_candidate
                else None
            ),
        },
        "paired_bool": paired_bool,
        "paired_fractional": paired_fractional,
        "paired_zero_floor": paired_zero_floor,
        "fake_40_hex_evidence": fake_evidence,
        "self_assertion_reference": _activity_evidence_ref(
            "train_monitor", {"trades_total": 1, "total_return": 0.1}
        ),
        "higher_floor_selection": higher_floor_selection,
        "calibrated_string_floor": calibrated_string_floor,
        "calibrated_fractional_floor": calibrated_fractional_floor,
        "missing_trade_selection": missing_selection,
        "pending_consumers": pending_consumers,
        "legacy_rl_pipeline": {
            "registered": (
                "rl_pipeline=pipeline_plugins.rl_pipeline:PipelinePlugin"
                in setup_text
            ),
            "config_count": len(legacy_configs),
            "sample_configs": legacy_configs[:5],
        },
    }

    reproduced = {
        "promotion_truncates_fractional_count_before_authority": bool(
            promotion_candidate
            and promotion_candidate["validation_evidence"]["trades_total"] == 1
        ),
        "paired_comparator_accepts_boolean_count": paired_bool["eligible"],
        "paired_comparator_accepts_fractional_count": paired_fractional[
            "eligible"
        ],
        "paired_comparator_accepts_zero_floor": paired_zero_floor["eligible"],
        "syntactic_fake_evidence_is_eligible": fake_evidence["eligible"],
        "pipeline_manufactures_self_assertion_reference": (
            ":summary:sha256:"
            in facts["self_assertion_reference"]
        ),
        "higher_floor_is_mislabeled_with_strict_contract_id": (
            higher_floor_selection["components"]["threshold_contract_id"]
            == activity.THRESHOLD_CONTRACT_ID
        ),
        "calibrated_string_floor_is_accepted": (
            calibrated_string_floor["raised"] is None
        ),
        "calibrated_fractional_floor_is_accepted": (
            calibrated_fractional_floor["raised"] is None
        ),
        "missing_trade_is_rendered_as_zero": (
            missing_selection["components"]["trades_total"] == 0
        ),
        "consumer_graph_is_incomplete": bool(pending_consumers),
        "registered_legacy_pipeline_bypasses_authority": bool(
            facts["legacy_rl_pipeline"]["registered"]
            and facts["legacy_rl_pipeline"]["config_count"]
        ),
    }
    packet = {
        "schema": "agent_multi.wp1_activity_authority_return_audit.v1",
        "commit_under_test": "4e8134049bb2eb71f3f773b4be7c8a16fd90c257",
        "facts": facts,
        "reproduced": reproduced,
        "reproduced_count": sum(bool(value) for value in reproduced.values()),
        "case_count": len(reproduced),
        "all_reproduced": all(reproduced.values()),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if packet["all_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
