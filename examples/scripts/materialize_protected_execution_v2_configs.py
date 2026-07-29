#!/usr/bin/env python3
"""Build immutable protected-entry easy and curriculum v2 configs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


ORDER_GENES = [
    {
        "choices": ["adaptive", "market", "limit", "stop"],
        "kind": "categorical",
        "name": "entry_order_mode_gene",
        "target": "entry_order_mode",
    },
    {
        "high": 0.90,
        "kind": "float",
        "low": 0.55,
        "name": "market_urgency_threshold_gene",
        "target": "market_urgency_threshold",
    },
    {
        "choices": [1.0, 2.0, 4.0, 8.0],
        "kind": "categorical",
        "name": "market_max_spread_bps_gene",
        "target": "market_max_spread_bps",
    },
    {
        "high": 0.85,
        "kind": "float",
        "low": 0.35,
        "name": "stop_breakout_threshold_gene",
        "target": "stop_breakout_threshold",
    },
    {
        "high": 0.25,
        "kind": "float",
        "low": 0.01,
        "name": "limit_offset_atr_gene",
        "target": "limit_offset_atr_multiple",
    },
    {
        "high": 0.25,
        "kind": "float",
        "low": 0.01,
        "name": "stop_offset_atr_gene",
        "target": "stop_offset_atr_multiple",
    },
]
ORDER_GENE_NAMES = [item["name"] for item in ORDER_GENES]
ORDER_INITIAL = {
    "entry_order_mode_gene": "adaptive",
    "market_urgency_threshold_gene": 0.75,
    "market_max_spread_bps_gene": 4.0,
    "stop_breakout_threshold_gene": 0.65,
    "limit_offset_atr_gene": 0.05,
    "stop_offset_atr_gene": 0.05,
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _replace_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_strings(item, old, new)
            for key, item in value.items()
        }
    return value


def _add_order_genes(config: dict[str, Any]) -> None:
    optimization = config["optimization"]
    schema = optimization["mixed_genome_schema"]
    existing = {str(item.get("name")) for item in schema}
    schema.extend(
        copy.deepcopy(item)
        for item in ORDER_GENES
        if item["name"] not in existing
    )
    optimization["initial_candidate_decoded"].update(ORDER_INITIAL)
    for stage in optimization["optimization_stages"]:
        if stage.get("name") in {"execution_risk", "bounded_joint_refinement"}:
            params = stage.get("params")
            if isinstance(params, list):
                for name in ORDER_GENE_NAMES:
                    if name not in params:
                        params.append(name)


def _apply_safety_contract(config: dict[str, Any]) -> None:
    environment = config["environment"]
    environment.update(
        {
            "require_protected_entries": True,
            "entry_order_mode": "adaptive",
            "full_spread_rate": 0.0001,
            "market_urgency_threshold": 0.75,
            "market_max_spread_bps": 4.0,
            "stop_breakout_threshold": 0.65,
            "limit_offset_spread_multiple": 0.5,
            "limit_offset_atr_multiple": 0.05,
            "stop_offset_spread_multiple": 0.5,
            "stop_offset_atr_multiple": 0.05,
            "breakout_lookback": 12,
        }
    )
    training = config["training"]
    training.update(
        {
            "early_stop_min_train_tail_trades": 1,
            "early_stop_min_validation_trades": 12,
        }
    )
    optimization = config["optimization"]
    optimization.update(
        {
            "optimization_reject_insufficient_activity": True,
            "optimization_min_trades_by_split": {
                "train_tail": 1,
                "validation": 12,
            },
        }
    )


def build_easy(source: dict[str, Any]) -> dict[str, Any]:
    config = _replace_strings(
        copy.deepcopy(source),
        "full_genome/usdcad_4h",
        "protected_easy/usdcad_4h",
    )
    config["experiment"]["name"] = (
        "phase_1_asset_policy_usdcad_4h_protected_easy_v2"
    )
    config["experiment"]["description"] = (
        "Easy non-zero-cost full-genome optimization with mandatory SL/TP "
        "and evolvable market/limit/stop/adaptive protected entries."
    )
    config["environment"]["execution_difficulty"] = "easy_floor"
    config["risk"]["commission"] = 0.00005
    config["risk"]["slippage"] = 0.000075
    _apply_safety_contract(config)
    _add_order_genes(config)
    return config


def build_curriculum(source: dict[str, Any]) -> dict[str, Any]:
    config = _replace_strings(
        copy.deepcopy(source),
        "execution_curriculum/usdcad_4h",
        "protected_curriculum/usdcad_4h",
    )
    config["experiment"]["name"] = (
        "phase_1_asset_policy_usdcad_4h_protected_curriculum_v2"
    )
    config["experiment"]["description"] = (
        "Warm-started protected-entry execution curriculum from easy through "
        "nominal and stress costs with robust weekly RAP selection."
    )
    _apply_safety_contract(config)
    _add_order_genes(config)
    for stage in config["optimization"]["optimization_stages"]:
        if stage.get("name") == "bounded_joint_refinement":
            continue
        if stage.get("name") == "execution_risk":
            continue
        if stage.get("name") == "cost_adaptation":
            params = stage.get("params")
            if isinstance(params, list):
                for name in ORDER_GENE_NAMES:
                    if name not in params:
                        params.append(name)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easy-source", type=Path, required=True)
    parser.add_argument("--curriculum-source", type=Path, required=True)
    parser.add_argument("--easy-output", type=Path, required=True)
    parser.add_argument("--curriculum-output", type=Path, required=True)
    args = parser.parse_args()
    _write(args.easy_output, build_easy(_load(args.easy_source)))
    _write(
        args.curriculum_output,
        build_curriculum(_load(args.curriculum_source)),
    )
    print(args.easy_output)
    print(args.curriculum_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
