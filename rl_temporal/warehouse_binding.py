"""The result record a warehouse row must be built from.

Every field that decides what a number means is bound here by name; a record
that lacks one is refused by that name. A DRAFT manifest cannot label a RESULT,
and a heuristic baseline enters only with its naive-gate evidence.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

RESULT_SCHEMA = "rl_temporal_result.v1"

_REQUIRED = {
    "arm": str, "algorithm": str, "representation": str, "task": dict, "data": dict,
    "representation_identity": dict, "seed": int, "paired_seed_group": str, "actions": dict, "reward": dict,
    "evaluation_population": dict, "costs": dict, "metrics": dict, "baselines": dict, "resources": dict,
    "versions": dict, "status": str,
}
_NESTED = {
    "task": ("dataset_id", "selected_feature_manifest_sha256", "manifest_status"),
    "data": ("model_ready_view_sha256", "train_rows", "validation_rows"),
    "representation_identity": ("modular_config_sha256", "layout_digest", "regimes", "donor_identity", "parameter_count"),
    "actions": ("action_space_mode",),
    "reward": ("plugin", "frozen_before_fit"),
    "evaluation_population": ("split", "episodes", "rows", "selection_metric"),
    "costs": ("commission", "slippage", "financing_enabled"),
    "metrics": ("net_return", "max_drawdown_fraction", "sharpe", "turnover_units", "trades_closed", "exposure_fraction"),
    "baselines": ("no_trade", "heuristic"),
    "resources": ("host_alias", "device", "wall_s", "peak_rss_bytes", "gradient_updates", "pretraining_cost"),
    "versions": ("stable_baselines3", "torch", "engine_pin"),
}
_STATUSES = ("RESULT", "DEVELOPMENT_NOT_CONFIRMATORY", "PILOT_NOT_A_RESULT", "FIXTURE_NOT_A_RESULT", "SKIPPED", "FAILED")


def validate_result_record(record: Dict[str, Any]) -> Dict[str, Any]:
    for key, typ in _REQUIRED.items():
        if key not in record:
            raise ValueError(f"result record lacks required binding {key!r}")
        if typ is int and isinstance(record[key], bool) or not isinstance(record[key], typ):
            raise ValueError(f"result record field {key!r} must be {typ.__name__}")
    for key, fields in _NESTED.items():
        for f in fields:
            if f not in record[key]:
                raise ValueError(f"result record {key}.{f} is missing")
    if record["status"] not in _STATUSES:
        raise ValueError(f"status must be one of {_STATUSES}")
    if record["status"] == "RESULT" and record["task"]["manifest_status"] not in ("FROZEN", "FROZEN_DEVELOPMENT"):
        raise ValueError(f"a RESULT cannot be labelled from a {record['task']['manifest_status']} manifest "
                         "(DRAFT_NOT_FROZEN selection may be recomputed; only a frozen manifest binds a result)")
    if record["task"]["manifest_status"] == "FROZEN_DEVELOPMENT" and not record["task"].get("availability_class"):
        raise ValueError("a FROZEN_DEVELOPMENT manifest must carry task.availability_class (DEVELOPMENT evidence)")
    sharpe = record["metrics"]["sharpe"]
    if not isinstance(sharpe, dict) or "convention" not in sharpe or "value" not in sharpe:
        raise ValueError("metrics.sharpe must carry value and convention (and undefined_reason when None)")
    if sharpe["value"] is None and not sharpe.get("undefined_reason"):
        raise ValueError("an undefined sharpe must say why (undefined_reason)")
    heuristic = record["baselines"]["heuristic"]
    if heuristic != "UNAVAILABLE":
        gate = heuristic.get("naive_gate") if isinstance(heuristic, dict) else None
        if not (isinstance(gate, dict) and gate.get("passed") is True and gate.get("evidence")):
            raise ValueError("baselines.heuristic needs naive_gate {passed: true, evidence: ...} or must be 'UNAVAILABLE'")
    if record["actions"]["action_space_mode"] not in ("discrete", "continuous"):
        raise ValueError("actions.action_space_mode must be discrete or continuous")
    return {"schema": RESULT_SCHEMA, "digest": record_digest(record), "status": record["status"]}


def record_digest(record: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
