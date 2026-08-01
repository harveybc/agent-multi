from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


CONTRACT_SCHEMA = "agent_multi.live_feature_asset_contract.v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _normalize_asset(value: object) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def validate_live_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError(f"unsupported live parity schema: {contract.get('schema_version')!r}")

    required_objects = (
        "decision_policy",
        "feature_profiles",
        "sources",
        "runtime_capabilities",
        "runtime_sources",
        "execution_venues",
        "instrument_mappings",
    )
    for key in required_objects:
        if not isinstance(contract.get(key), Mapping):
            raise ValueError(f"live parity contract requires object {key!r}")

    cells = contract.get("asset_cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("live parity contract requires a non-empty asset_cells list")

    seen: set[tuple[str, str]] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("every asset cell must be an object")
        key = (_normalize_asset(cell.get("asset")), str(cell.get("timeframe") or "").lower())
        if not all(key):
            raise ValueError("every asset cell requires asset and timeframe")
        if key in seen:
            raise ValueError(f"duplicate asset cell: {key[0]}@{key[1]}")
        seen.add(key)

        mapping_id = str(cell.get("instrument_mapping_id") or "")
        if mapping_id not in contract["instrument_mappings"]:
            raise ValueError(f"asset cell references unknown instrument mapping: {mapping_id!r}")
        for source_id in cell.get("data_routes") or []:
            if source_id not in contract["runtime_sources"]:
                raise ValueError(f"asset cell references unknown runtime source: {source_id!r}")
        for venue_id in cell.get("execution_routes") or []:
            if venue_id not in contract["execution_venues"]:
                raise ValueError(f"asset cell references unknown execution venue: {venue_id!r}")

    for profile_name, profile in contract["feature_profiles"].items():
        if not isinstance(profile, Mapping):
            raise ValueError(f"feature profile must be an object: {profile_name!r}")
        for source_id in profile.get("source_ids") or []:
            if source_id not in contract["sources"]:
                raise ValueError(
                    f"feature profile {profile_name!r} references unknown source {source_id!r}"
                )
        for capability_id in profile.get("required_capabilities") or []:
            if capability_id not in contract["runtime_capabilities"]:
                raise ValueError(
                    f"feature profile {profile_name!r} references unknown capability "
                    f"{capability_id!r}"
                )


def load_live_contract(path: str | Path) -> dict[str, Any]:
    contract = _load_json(path)
    validate_live_contract(contract)
    return contract


def audit_experiment_live_parity(
    experiment: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify research, live-inference and live-execution eligibility.

    The audit is intentionally fail-closed. Historical availability, documented
    provider capability and an observed quote are not substitutes for an
    integrated runtime feed plus a passed causal/numerical parity test.
    """

    validate_live_contract(contract)
    data = experiment.get("data") if isinstance(experiment.get("data"), Mapping) else {}
    asset = _normalize_asset(data.get("asset"))
    timeframe = str(data.get("timeframe") or "").lower()
    profile_name = str(data.get("data_profile") or data.get("features_preset") or "")

    blockers: dict[str, list[str]] = {
        "research": [],
        "live_inference": [],
        "live_execution": [],
    }

    profiles = contract["feature_profiles"]
    profile = profiles.get(profile_name)
    if not isinstance(profile, Mapping):
        blockers["research"].append(f"feature_profile_unregistered:{profile_name or 'missing'}")
        blockers["live_inference"].append("feature_profile_not_live_audited")
        profile = {}

    source_ids = profile.get("source_ids") if isinstance(profile.get("source_ids"), list) else []
    if not source_ids:
        blockers["research"].append("feature_profile_has_no_source_contract")
        blockers["live_inference"].append("feature_profile_has_no_runtime_source")

    source_evidence: list[dict[str, Any]] = []
    for source_id in source_ids:
        source = contract["sources"].get(source_id)
        if not isinstance(source, Mapping):
            blockers["research"].append(f"source_unregistered:{source_id}")
            blockers["live_inference"].append(f"source_unregistered:{source_id}")
            continue
        source_evidence.append(
            {
                "source_id": source_id,
                "availability_class": source.get("availability_class"),
                "research_status": source.get("research_status"),
                "live_status": source.get("live_status"),
            }
        )
        if source.get("research_status") not in {"available", "available_point_in_time"}:
            blockers["research"].append(f"source_not_research_ready:{source_id}")
        if source.get("availability_class") == "research_only":
            blockers["live_inference"].append(f"research_only_source:{source_id}")

    required_capabilities = profile.get("required_capabilities")
    if not isinstance(required_capabilities, list):
        required_capabilities = []
    capability_evidence: list[dict[str, Any]] = []
    ready_capability_statuses = set(
        contract["decision_policy"].get("live_capability_ready_statuses") or []
    )
    for capability_id in required_capabilities:
        capability = contract["runtime_capabilities"].get(capability_id)
        if not isinstance(capability, Mapping):
            blockers["live_inference"].append(f"runtime_capability_unregistered:{capability_id}")
            continue
        capability_evidence.append(
            {
                "capability_id": capability_id,
                "status": capability.get("status"),
                "parity_status": capability.get("parity_status"),
            }
        )
        if capability.get("status") not in ready_capability_statuses:
            blockers["live_inference"].append(f"runtime_capability_not_integrated:{capability_id}")
        if capability.get("parity_status") != "passed":
            blockers["live_inference"].append(f"runtime_parity_not_passed:{capability_id}")

    matching_cells = [
        cell
        for cell in contract["asset_cells"]
        if _normalize_asset(cell.get("asset")) == asset
        and str(cell.get("timeframe") or "").lower() == timeframe
    ]
    cell = matching_cells[0] if matching_cells else None
    if cell is None:
        blockers["research"].append(f"asset_cell_not_selected:{asset or 'missing'}@{timeframe or 'missing'}")
        blockers["live_inference"].append("asset_data_route_unregistered")
        blockers["live_execution"].append("asset_execution_route_unregistered")
    else:
        data_ready_statuses = set(
            contract["decision_policy"].get("live_data_ready_statuses") or []
        )
        execution_ready_statuses = set(
            contract["decision_policy"].get("live_execution_ready_statuses") or []
        )
        if cell.get("data_route_status") not in data_ready_statuses:
            blockers["live_inference"].append(
                f"asset_data_route_not_ready:{cell.get('data_route_status')}"
            )
        if cell.get("execution_route_status") not in execution_ready_statuses:
            blockers["live_execution"].append(
                f"protected_execution_not_ready:{cell.get('execution_route_status')}"
            )

    blockers["live_execution"].extend(blockers["live_inference"])
    blockers = {key: sorted(set(values)) for key, values in blockers.items()}

    return {
        "schema_version": "agent_multi.live_parity_audit.v1",
        "asset": asset,
        "timeframe": timeframe,
        "data_profile": profile_name,
        "selected_set": cell.get("selection_set") if cell else None,
        "research_eligible": not blockers["research"],
        "live_inference_eligible": not blockers["research"] and not blockers["live_inference"],
        "live_execution_eligible": (
            not blockers["research"]
            and not blockers["live_inference"]
            and not blockers["live_execution"]
        ),
        "blockers": blockers,
        "source_evidence": source_evidence,
        "capability_evidence": capability_evidence,
        "asset_cell": dict(cell) if cell else None,
    }


def audit_experiment_files(
    experiment_path: str | Path, contract_path: str | Path
) -> dict[str, Any]:
    return audit_experiment_live_parity(_load_json(experiment_path), load_live_contract(contract_path))
