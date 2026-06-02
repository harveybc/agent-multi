"""Project 3 Stage 3X SAC smoke-run plan generator.

Consumes the financial-data Stage 3X selected feature contracts and emits
locked SAC smoke/follow-up configs (selected contracts x seeds x costs).
**Never launches training. Never touches Stage C.**

Input contract
--------------
``--selected-contracts`` must point at a JSON file with schema
``project3_stage3x_selected_feature_contracts_v1`` (produced by
``financial-data/_scripts/workers/stage3x_target_relation_screen_worker.py``).
Top-level must have ``contracts: [...]`` with each entry exposing at minimum:

    contract_id, genome_id, asset, timeframe, feature_preset,
    feature_selection_method, preprocessing_profile, input_data_file,
    selected_features (non-empty list)

Optional preprocessing fields that are preserved when present:
``scaling_mode``, ``feature_scaling_window``, ``feature_clip``,
``window_size``, ``stage_b_force_close_obs`` (or ``force_close_*`` flags).

Schema
------
Manifest schema_version: ``project3_stage3x_sac_smoke_plan_v1``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "project3_stage3x_sac_smoke_plan_v1"
INPUT_SCHEMA_VERSION = "project3_stage3x_selected_feature_contracts_v1"
MARKET_STATE_PROFILES_SCHEMA = "project3_stage3x_selected_market_state_profiles_v1"
HELDOUT_START = "2025-01-01"

DEFAULT_TOP_N = 3
DEFAULT_SEEDS: Sequence[int] = (0, 1, 2)
DEFAULT_COST_SCENARIO = "base"
DEFAULT_COST_SCENARIOS: Sequence[str] = (DEFAULT_COST_SCENARIO,)
ALLOWED_COST_SCENARIOS = ("base", "plus_50pct", "plus_100pct")
DEFAULT_BASE_COMMISSION = 0.0002
DEFAULT_BASE_SLIPPAGE = 0.0
COST_SCENARIO_MULTIPLIERS: Dict[str, float] = {
    "base": 1.0,
    "plus_50pct": 1.5,
    "plus_100pct": 2.0,
}

DEFAULT_SAC_AGENT_PLUGIN = "project3_sac_actor_critic_agent"
DEFAULT_PIPELINE_PLUGIN = "rl_pipeline_with_validation"
DEFAULT_ENV_PLUGIN = "gym_fx_env"

PRESERVED_PREPROCESSING_KEYS = (
    "broker_profile",
    "market_type",
    "regulatory_profile",
    "trade_rate_band_id",
    "calendar_policy_id",
    "market_state_profile_id",
    "market_state_profile_hash",
    "market_state_profile_family",
    "market_state_profile_name",
    "market_state_encoder_config_hash",
    "market_state_weekly_anchor_id",
    "market_state_source_columns",
    "market_state_selected_columns",
    "scaling_mode",
    "feature_scaling_window",
    "feature_clip",
    "window_size",
    "split_anchor",
    "train_days",
    "val_days",
    "test_days",
    "min_split_rows",
    "stage_b_force_close_obs",
    "force_close_dow",
    "force_close_hour",
    "force_close_window_hours",
    "monday_entry_window_hours",
    "force_close_window_bars",
    "force_close_penalty_coef",
    "parent_contract_id",
    "micro_nsga_generation",
    "micro_nsga_individual_id",
    "micro_nsga_variant",
    "total_timesteps",
    "learning_rate",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "gamma",
    "tau",
    "train_freq",
    "gradient_steps",
    "ent_coef",
    "target_entropy",
    "use_sde",
    "net_arch",
    "continuous_action_threshold",
    "epoch_timesteps",
    "max_epochs",
    "l1_patience",
    "l1_min_delta",
)


class SmokePlanError(RuntimeError):
    """Raised when the smoke plan cannot be safely generated."""


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded_cell_name(contract_id: str, seed: int, cost_scenario: str) -> str:
    """Return a short, deterministic filesystem name for one smoke cell.

    The full contract id is kept inside the config and manifest. The path
    component must be bounded because iterative micro-NSGA ids append ancestry
    across generations and can exceed Linux filename limits.
    """
    digest = hashlib.sha1(
        f"{contract_id}__s{seed}__{cost_scenario}".encode("utf-8")
    ).hexdigest()[:16]
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", contract_id)[:64].strip("._-")
    return f"stage3x_smoke__{readable}__s{seed}__{cost_scenario}__{digest}"


def _load_selected_contracts(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise SmokePlanError(f"selected-contracts file not found: {path}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SmokePlanError(f"selected-contracts not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise SmokePlanError("selected-contracts root must be a JSON object")
    schema = str(doc.get("schema_version") or "")
    if schema and schema != INPUT_SCHEMA_VERSION:
        raise SmokePlanError(
            f"unexpected selected-contracts schema_version: {schema!r} "
            f"(expected {INPUT_SCHEMA_VERSION!r})"
        )
    if str(doc.get("stage_c_access", "DENIED")).upper() != "DENIED":
        raise SmokePlanError(
            "selected-contracts has non-DENIED stage_c_access; refusing"
        )
    if bool(doc.get("training_launched")):
        raise SmokePlanError(
            "selected-contracts marked training_launched=true; refusing"
        )
    contracts = doc.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        raise SmokePlanError("selected-contracts has no contracts")
    return doc


def _load_market_state_profiles(path: str) -> Dict[str, Any]:
    """Load and validate selected_market_state_profiles.json.

    Refuses to load if Stage C is not DENIED, training_launched is true, or
    the schema_version does not match the expected v1 contract.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise SmokePlanError(f"selected-market-state-profiles file not found: {path}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SmokePlanError(
            f"selected-market-state-profiles not valid JSON: {exc}"
        ) from exc
    if not isinstance(doc, dict):
        raise SmokePlanError("selected-market-state-profiles root must be a JSON object")
    schema = str(doc.get("schema_version") or "")
    if schema and schema != MARKET_STATE_PROFILES_SCHEMA:
        raise SmokePlanError(
            f"unexpected selected-market-state-profiles schema_version: {schema!r} "
            f"(expected {MARKET_STATE_PROFILES_SCHEMA!r})"
        )
    if str(doc.get("stage_c_access", "DENIED")).upper() != "DENIED":
        raise SmokePlanError(
            "selected-market-state-profiles has non-DENIED stage_c_access; refusing"
        )
    if bool(doc.get("training_launched")):
        raise SmokePlanError(
            "selected-market-state-profiles marked training_launched=true; refusing"
        )
    profiles = doc.get("contracts")
    if not isinstance(profiles, list) or not profiles:
        raise SmokePlanError("selected-market-state-profiles has no profile contracts")
    return doc


def _normalize_asset_key(asset: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(asset or "").lower()).strip("_")


def _build_profile_index(profiles_doc: Mapping[str, Any]) -> Dict[tuple, Dict[str, Any]]:
    """Index profiles by (normalized asset, timeframe), keeping highest score."""
    index: Dict[tuple, Dict[str, Any]] = {}
    for profile in profiles_doc.get("contracts", []):
        if not isinstance(profile, Mapping):
            continue
        asset_key = _normalize_asset_key(profile.get("target_asset"))
        timeframe = str(profile.get("timeframe") or "").lower()
        if not asset_key or not timeframe:
            continue
        key = (asset_key, timeframe)
        score = float(profile.get("relation_score") or 0.0)
        prior = index.get(key)
        if prior is None or score > float(prior.get("relation_score") or 0.0):
            index[key] = dict(profile)
    return index


def _attach_market_state_profile(
    contract: Dict[str, Any],
    profile_index: Mapping[tuple, Mapping[str, Any]],
    *,
    require: bool,
    idx: int,
) -> Optional[Dict[str, Any]]:
    """Merge a market-state profile into ``contract`` in place.

    When ``require`` is True and no profile matches the contract's
    (asset, timeframe), raises SmokePlanError. Returns the attached profile
    dict (or None when not required and no match was found).

    Side effects when a profile is attached:
      - extends ``contract['selected_features']`` with the profile's
        ``selected_columns`` (dedup, order-preserving);
      - sets ``market_state_profile_*`` keys preserved into the locked config.
    """
    asset_key = _normalize_asset_key(contract.get("asset"))
    timeframe = str(contract.get("timeframe") or "").lower()
    profile = profile_index.get((asset_key, timeframe))
    if profile is None:
        if require:
            raise SmokePlanError(
                f"contract[{idx}] {contract.get('contract_id')!r} has no matching "
                f"market-state profile for asset={asset_key!r} timeframe={timeframe!r}; "
                "refusing (fail-closed)"
            )
        return None
    profile_hash = str(profile.get("market_state_profile_hash") or "").strip()
    profile_id = str(profile.get("market_state_profile_id") or "").strip()
    selected_columns = profile.get("selected_columns") or []
    if not profile_hash or not profile_id or not selected_columns:
        raise SmokePlanError(
            f"contract[{idx}] {contract.get('contract_id')!r}: matched market-state "
            "profile is missing id/hash/selected_columns; refusing (fail-closed)"
        )

    existing = list(contract.get("selected_features") or [])
    merged: List[str] = list(existing)
    seen = set(existing)
    for col in selected_columns:
        col_str = str(col)
        if col_str not in seen:
            merged.append(col_str)
            seen.add(col_str)
    contract["selected_features"] = merged

    contract["market_state_profile_id"] = profile_id
    contract["market_state_profile_hash"] = profile_hash
    contract["market_state_profile_family"] = profile.get("profile_family")
    contract["market_state_profile_name"] = profile.get("profile_name")
    contract["market_state_encoder_config_hash"] = profile.get("encoder_config_hash")
    contract["market_state_weekly_anchor_id"] = profile.get("weekly_anchor_id")
    contract["market_state_source_columns"] = list(profile.get("source_columns") or [])
    contract["market_state_selected_columns"] = [str(c) for c in selected_columns]
    return profile


def _validate_contract(contract: Mapping[str, Any], *, idx: int) -> None:
    required = (
        "contract_id", "genome_id", "asset", "timeframe", "feature_preset",
        "preprocessing_profile", "input_data_file", "selected_features",
    )
    missing = [k for k in required if not contract.get(k)]
    features = contract.get("selected_features")
    if not isinstance(features, list) or not features:
        raise SmokePlanError(
            f"contract[{idx}] {contract.get('contract_id')!r} has empty or "
            "missing selected_features; refusing (fail-closed)"
        )
    if missing:
        raise SmokePlanError(
            f"contract[{idx}] {contract.get('contract_id')!r} missing "
            f"required field(s): {missing}"
        )
    if str(contract.get("stage_c_access", "DENIED")).upper() != "DENIED":
        raise SmokePlanError(
            f"contract[{idx}] has non-DENIED stage_c_access; refusing"
        )


def _select_top_n(contracts: Sequence[Mapping[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if top_n <= 0:
        raise SmokePlanError(f"--top-n must be positive, got {top_n}")

    def _score(row: Mapping[str, Any]) -> tuple:
        return (
            float(row.get("screen_score") or 0.0),
            float(row.get("proxy_net_return") or 0.0),
            float(row.get("best_abs_validation_ic") or 0.0),
        )

    ranked = sorted(contracts, key=_score, reverse=True)
    return [dict(row) for row in ranked[:top_n]]


def _build_locked_config(
    *,
    contract: Mapping[str, Any],
    seed: int,
    cost_scenario: str,
    output_root: Path,
) -> Dict[str, Any]:
    features = [str(f) for f in contract["selected_features"]]
    if not features:
        raise SmokePlanError(
            f"contract {contract.get('contract_id')!r} resolved to empty "
            "feature list at build time; refusing (fail-closed)"
        )

    asset = str(contract["asset"])
    timeframe = str(contract["timeframe"])
    contract_id = str(contract["contract_id"])
    base_commission = float(contract.get("commission") or DEFAULT_BASE_COMMISSION)
    base_slippage = float(contract.get("slippage") or DEFAULT_BASE_SLIPPAGE)
    cost_multiplier = COST_SCENARIO_MULTIPLIERS[cost_scenario]
    run_dirname = _bounded_cell_name(contract_id, seed, cost_scenario)
    run_dir = output_root / "runs" / run_dirname
    trace_dir = run_dir / "return_traces"

    cfg: Dict[str, Any] = {
        # Stage B/3X lock fields (defense in depth)
        "_project3_stage3x_sac_smoke": True,
        "_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED": True,
        "_project3_stage_b_lock": True,
        "_stage_b_role": "candidate",
        "heldout_start": HELDOUT_START,
        "stage_c_access": "DENIED",
        "final_stage_c_evaluation": False,
        "stage_c_acknowledged": False,

        # Algorithm wiring — SAC only.
        "agent_plugin": DEFAULT_SAC_AGENT_PLUGIN,
        "pipeline_plugin": DEFAULT_PIPELINE_PLUGIN,
        "env_plugin": DEFAULT_ENV_PLUGIN,

        # Stage 3X contract identity.
        "_stage3x_contract_id": contract_id,
        "_stage3x_genome_id": contract.get("genome_id"),
        "_stage3x_feature_preset": contract.get("feature_preset"),
        "_stage3x_feature_selection_method": contract.get("feature_selection_method"),
        "_stage3x_preprocessing_profile": contract.get("preprocessing_profile"),
        "_stage3x_source_family": contract.get("source_family"),
        "_stage3x_screen_score": contract.get("screen_score"),
        "_stage3x_best_abs_validation_ic": contract.get("best_abs_validation_ic"),
        "_stage3x_proxy_net_return": contract.get("proxy_net_return"),
        "_stage3x_proxy_trades": contract.get("proxy_trades"),

        # Data + features.
        "asset": f"{asset}_{timeframe}",
        "timeframe": timeframe,
        "input_data_file": str(contract["input_data_file"]),
        "date_column": "DATE_TIME",
        "price_column": "CLOSE",
        "feature_list": list(features),
        "feature_columns": list(features),
        "features_preset": contract.get("feature_preset"),
        "commission": base_commission * cost_multiplier,
        "slippage": base_slippage * cost_multiplier,
        "_base_commission": base_commission,
        "_base_slippage": base_slippage,
        "_cost_multiplier": cost_multiplier,
        "stage_b_force_close_obs": True,
        "force_close_dow": 4,
        "force_close_hour": 20,
        "force_close_window_hours": 4,
        "monday_entry_window_hours": 4,

        # Chronological train/val/test split. Use the end of the train-only
        # file so the test slice stays before the 2025 heldout firewall. The
        # split years adapt to shorter histories such as BTC perpetuals.
        **_split_params_for_contract(contract),

        # Seeds + smoke cost.
        "train_seed": int(seed),
        "eval_seed": int(seed),
        "_cost_scenario": cost_scenario,
    }

    # Preserve selected preprocessing fields where present on the contract.
    for key in PRESERVED_PREPROCESSING_KEYS:
        if key in contract and contract[key] is not None:
            cfg[key] = contract[key]

    # Deterministic per-run output paths.
    cfg["save_model"] = str(run_dir / "policy.zip")
    cfg["results_file"] = str(run_dir / "results.json")
    cfg["save_config"] = str(run_dir / "config_out.json")
    cfg["progress_file"] = str(run_dir / "training_progress.json")
    cfg["training_progress_file"] = str(run_dir / "training_progress.json")

    # Return-trace / evidence wiring (consumed by financial-data evaluator).
    cfg["return_trace_dir"] = str(trace_dir)
    cfg["return_trace_file"] = str(trace_dir / "evaluation_return_trace.csv")

    return cfg


def _parse_date(value: str) -> _dt.datetime | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return _dt.datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        try:
            return _dt.datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def _data_date_span_years(path: str) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    first: _dt.datetime | None = None
    last: _dt.datetime | None = None
    with p.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            value = row.get("DATE_TIME") or row.get("date") or row.get("timestamp")
            parsed = _parse_date(str(value or ""))
            if parsed is None:
                continue
            if first is None:
                first = parsed
            last = parsed
    if first is None or last is None or last <= first:
        return None
    return (last - first).days / 365.25


def _split_params_for_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Choose integer-year splits that fit inside the train-only data file."""
    span_years = _data_date_span_years(str(contract.get("input_data_file") or ""))
    train_years = 4
    if span_years is not None and span_years < 6.0:
        train_years = 2 if span_years >= 4.0 else 1
    return {
        "split_anchor": "end",
        "train_years": train_years,
        "val_years": 1,
        "test_years": 1,
    }


def _expected_evidence_file(cfg: Mapping[str, Any]) -> str:
    trace_dir = cfg.get("return_trace_dir")
    if trace_dir:
        return str(Path(trace_dir) / "evidence.json")
    trace_file = cfg.get("return_trace_file")
    if trace_file:
        return str(Path(trace_file).parent / "evidence.json")
    return ""


def _config_filename(contract_id: str, seed: int, cost_scenario: str) -> str:
    return f"{_bounded_cell_name(contract_id, seed, cost_scenario)}.json"


def build_smoke_plan(
    *,
    selected_contracts_path: str,
    output_dir: str,
    top_n: int = DEFAULT_TOP_N,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cost_scenario: str = DEFAULT_COST_SCENARIO,
    cost_scenarios: Sequence[str] | None = None,
    selected_market_state_profiles_path: Optional[str] = None,
    require_market_state_profile: bool = False,
    write_files: bool = True,
) -> Dict[str, Any]:
    """Build (and optionally write) the Stage 3X SAC smoke plan.

    Returns the manifest dict. When ``write_files`` is True, also writes
    the manifest JSON/MD and one locked config per (contract, seed) cell.
    """
    if not seeds:
        raise SmokePlanError("--seeds must list at least one seed")
    selected_costs = list(cost_scenarios) if cost_scenarios is not None else [cost_scenario]
    if not selected_costs:
        raise SmokePlanError("--cost-scenarios must list at least one scenario")
    unknown_costs = [c for c in selected_costs if c not in ALLOWED_COST_SCENARIOS]
    if unknown_costs:
        raise SmokePlanError(
            f"unknown cost scenarios {unknown_costs!r}; allowed "
            f"{ALLOWED_COST_SCENARIOS!r}"
        )

    doc = _load_selected_contracts(selected_contracts_path)
    contracts = doc["contracts"]
    for idx, contract in enumerate(contracts):
        _validate_contract(contract, idx=idx)

    profile_index: Dict[tuple, Dict[str, Any]] = {}
    profiles_doc: Optional[Dict[str, Any]] = None
    profiles_sha: Optional[str] = None
    require_profile = bool(require_market_state_profile)
    if selected_market_state_profiles_path:
        profiles_doc = _load_market_state_profiles(selected_market_state_profiles_path)
        profile_index = _build_profile_index(profiles_doc)
        with Path(selected_market_state_profiles_path).open("rb") as fh:
            profiles_sha = _sha256_bytes(fh.read())
        # When an explicit profile manifest is provided, fail closed unless
        # every contract resolves to a market-state profile.
        require_profile = True
    elif require_profile:
        raise SmokePlanError(
            "--require-market-state-profile set but no "
            "--selected-market-state-profiles file provided"
        )

    if profile_index or require_profile:
        for idx, contract in enumerate(contracts):
            _attach_market_state_profile(
                contract, profile_index, require=require_profile, idx=idx,
            )

    top_contracts = _select_top_n(contracts, top_n)

    output_root = Path(output_dir)
    if write_files:
        config_dir = output_root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        for stale in config_dir.glob("*.json"):
            stale.unlink()
        (output_root / "runs").mkdir(parents=True, exist_ok=True)

    with Path(selected_contracts_path).open("rb") as fh:
        contracts_sha = _sha256_bytes(fh.read())

    config_entries: List[Dict[str, Any]] = []
    for contract in top_contracts:
        for seed in seeds:
            for selected_cost in selected_costs:
                cfg = _build_locked_config(
                    contract=contract,
                    seed=int(seed),
                    cost_scenario=selected_cost,
                    output_root=output_root,
                )
                # Defense-in-depth post-build invariants.
                assert cfg["stage_c_access"] == "DENIED"
                assert cfg["final_stage_c_evaluation"] is False
                assert cfg["stage_c_acknowledged"] is False
                assert cfg["_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"] is True
                assert cfg["_project3_stage3x_sac_smoke"] is True
                assert "sac" in str(cfg["agent_plugin"]).lower()
                assert cfg["feature_list"], "feature_list must be non-empty"
                assert cfg["feature_columns"], "feature_columns must be non-empty"
                assert cfg.get("return_trace_dir") or cfg.get("return_trace_file")
                assert cfg.get("progress_file")
                # Stage 3X market-state profile fail-closed: if a profile id
                # is attached, the hash must be present and the profile's
                # selected columns must reach the locked feature_list.
                profile_id = cfg.get("market_state_profile_id")
                if profile_id:
                    profile_hash = cfg.get("market_state_profile_hash")
                    if not profile_hash:
                        raise SmokePlanError(
                            f"contract {contract.get('contract_id')!r} has "
                            "market_state_profile_id without a profile hash; "
                            "refusing (fail-closed)"
                        )
                    selected_cols = cfg.get("market_state_selected_columns") or []
                    feature_set = set(cfg["feature_list"])
                    missing_cols = [c for c in selected_cols if c not in feature_set]
                    if missing_cols:
                        raise SmokePlanError(
                            f"contract {contract.get('contract_id')!r}: market-state "
                            f"profile columns missing from feature_list: {missing_cols}"
                        )
                elif require_profile:
                    raise SmokePlanError(
                        f"contract {contract.get('contract_id')!r} has no attached "
                        "market_state_profile but require flag is set; "
                        "refusing (fail-closed)"
                    )

                filename = _config_filename(
                    str(contract["contract_id"]), int(seed), selected_cost,
                )
                config_path = output_root / "configs" / filename
                if write_files:
                    with config_path.open("w", encoding="utf-8") as fh:
                        json.dump(cfg, fh, indent=2, sort_keys=True, default=str)

                run_dir = str(Path(str(cfg["save_model"])).parent)
                config_entries.append({
                    "contract_id": contract["contract_id"],
                    "genome_id": contract.get("genome_id"),
                    "asset": contract.get("asset"),
                    "timeframe": contract.get("timeframe"),
                    "feature_preset": contract.get("feature_preset"),
                    "preprocessing_profile": contract.get("preprocessing_profile"),
                    "seed": int(seed),
                    "cost_scenario": selected_cost,
                    "agent_plugin": cfg["agent_plugin"],
                    "config_file": str(config_path),
                    "run_dir": run_dir,
                    "progress_file": cfg["progress_file"],
                    "return_trace_dir": cfg["return_trace_dir"],
                    "return_trace_file": cfg["return_trace_file"],
                    "expected_evidence_file": _expected_evidence_file(cfg),
                    "feature_count": len(cfg["feature_list"]),
                    "market_state_profile_id": cfg.get("market_state_profile_id"),
                    "market_state_profile_hash": cfg.get("market_state_profile_hash"),
                    "market_state_profile_family": cfg.get("market_state_profile_family"),
                    "market_state_profile_name": cfg.get("market_state_profile_name"),
                    "market_state_encoder_config_hash": cfg.get(
                        "market_state_encoder_config_hash"
                    ),
                    "market_state_weekly_anchor_id": cfg.get(
                        "market_state_weekly_anchor_id"
                    ),
                    "market_state_selected_columns": cfg.get(
                        "market_state_selected_columns"
                    ),
                })

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utcnow_iso(),
        "selected_contracts_file": str(Path(selected_contracts_path).resolve()),
        "selected_contracts_sha256": contracts_sha,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "selected_market_state_profiles_file": (
            str(Path(selected_market_state_profiles_path).resolve())
            if selected_market_state_profiles_path else None
        ),
        "selected_market_state_profiles_sha256": profiles_sha,
        "market_state_profile_schema_version": (
            MARKET_STATE_PROFILES_SCHEMA if profiles_doc else None
        ),
        "market_state_profile_count": sum(
            1 for c in config_entries if c.get("market_state_profile_id")
        ),
        "heldout_start": HELDOUT_START,
        "stage_c_access": "DENIED",
        "final_stage_c_evaluation": False,
        "stage_c_acknowledged": False,
        "training_launched": False,
        "_project3_stage3x_sac_smoke": True,
        "top_n": int(top_n),
        "seeds": [int(s) for s in seeds],
        "cost_scenario": selected_costs[0] if len(selected_costs) == 1 else "multiple",
        "cost_scenarios": list(selected_costs),
        "selected_contract_ids": [c["contract_id"] for c in top_contracts],
        "configs": config_entries,
        "config_count": len(config_entries),
        "rules": {
            "smoke_only_base_cost": list(selected_costs) == ["base"],
            "targeted_cost_followup": list(selected_costs) != ["base"],
            "broad_gpu_launch_allowed": False,
            "stage_c_forbidden": True,
            "sac_only": True,
        },
    }

    if write_files:
        json_path = output_root / "stage3x_sac_smoke_plan_manifest.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        md_path = output_root / "stage3x_sac_smoke_plan_manifest.md"
        md_path.write_text(_render_markdown(manifest), encoding="utf-8")
        manifest["manifest_file"] = str(json_path.resolve())
        manifest["manifest_markdown_file"] = str(md_path.resolve())

    return manifest


def _render_markdown(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Project 3 Stage 3X SAC Smoke Plan",
        "",
        f"- schema_version: `{manifest['schema_version']}`",
        f"- generated_at: {manifest['generated_at']}",
        f"- selected_contracts_file: `{manifest['selected_contracts_file']}`",
        f"- selected_contracts_sha256: `{manifest['selected_contracts_sha256']}`",
        f"- heldout_start: `{manifest['heldout_start']}`",
        f"- stage_c_access: **{manifest['stage_c_access']}**",
        f"- training_launched: **{manifest['training_launched']}**",
        f"- top_n: {manifest['top_n']}",
        f"- seeds: {manifest['seeds']}",
        f"- cost_scenarios: `{manifest['cost_scenarios']}`",
        f"- config_count: {manifest['config_count']}",
        "",
        f"## Configs ({manifest['config_count']})",
        "",
        "| contract_id | seed | cost | features | config |",
        "|---|---:|---|---:|---|",
    ]
    for c in manifest["configs"]:
        lines.append(
            f"| `{c['contract_id']}` | {c['seed']} | {c['cost_scenario']} | "
            f"{c['feature_count']} | `{Path(c['config_file']).name}` |"
        )
    lines.append("")
    lines.append(
        "> Every emitted config is locked: `stage_c_access=DENIED`, "
        "`_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED=true`, "
        "`_project3_stage3x_sac_smoke=true`. This tool never launches training."
    )
    return "\n".join(lines) + "\n"


def _parse_csv_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate Project 3 Stage 3X locked SAC smoke-run configs from "
            "the financial-data selected feature contracts. Does not launch "
            "training. Stage C remains DENIED."
        ),
    )
    p.add_argument(
        "--selected-contracts",
        required=False,
        default=(
            "/home/harveybc/Documents/GitHub/financial-data/experiments/"
            "stage3x_target_relation_screen/selected_feature_contracts.json"
        ),
        help="Path to financial-data selected_feature_contracts.json "
             "(schema project3_stage3x_selected_feature_contracts_v1).",
    )
    p.add_argument(
        "--output-dir",
        default="./experiments/stage3x_sac_smoke_plan",
        help="Directory to receive manifest + locked configs.",
    )
    p.add_argument(
        "--top-n", type=int, default=DEFAULT_TOP_N,
        help=f"Number of top contracts to schedule (default {DEFAULT_TOP_N}).",
    )
    p.add_argument(
        "--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=f"Comma-separated seed list (default {','.join(str(s) for s in DEFAULT_SEEDS)}).",
    )
    p.add_argument(
        "--cost-scenario", default=DEFAULT_COST_SCENARIO,
        choices=list(ALLOWED_COST_SCENARIOS),
        help="Single cost scenario to schedule. Kept for backward compatibility.",
    )
    p.add_argument(
        "--cost-scenarios", default="",
        help=(
            "Comma-separated cost scenarios for targeted follow-up. "
            "Overrides --cost-scenario when provided."
        ),
    )
    p.add_argument(
        "--selected-market-state-profiles",
        default="",
        help=(
            "Optional path to financial-data selected_market_state_profiles.json "
            f"(schema {MARKET_STATE_PROFILES_SCHEMA!r}). When provided, each "
            "smoke contract must resolve to a market-state profile, the "
            "profile's selected_columns are merged into feature_list, and "
            "the profile id+hash are preserved on the locked config."
        ),
    )
    p.add_argument(
        "--require-market-state-profile", action="store_true",
        help=(
            "Fail closed when no --selected-market-state-profiles file is "
            "provided or when any contract has no matching profile."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Do not write any files; print the manifest summary only.",
    )
    p.add_argument(
        "--validate-only", action="store_true",
        help="Validate inputs and build the manifest in-memory; write nothing.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    write_files = not (args.dry_run or args.validate_only)
    try:
        manifest = build_smoke_plan(
            selected_contracts_path=args.selected_contracts,
            output_dir=args.output_dir,
            top_n=int(args.top_n),
            seeds=_parse_csv_int_list(args.seeds),
            cost_scenario=args.cost_scenario,
            cost_scenarios=(
                _parse_csv_str_list(args.cost_scenarios)
                if args.cost_scenarios
                else None
            ),
            selected_market_state_profiles_path=(
                args.selected_market_state_profiles or None
            ),
            require_market_state_profile=bool(args.require_market_state_profile),
            write_files=write_files,
        )
    except SmokePlanError as exc:
        print(f"[stage3x-sac-smoke-plan] ERROR: {exc}", file=sys.stderr)
        return 2
    summary = {
        "schema_version": manifest["schema_version"],
        "config_count": manifest["config_count"],
        "selected_contract_ids": manifest["selected_contract_ids"],
        "seeds": manifest["seeds"],
        "cost_scenario": manifest["cost_scenario"],
        "cost_scenarios": manifest["cost_scenarios"],
        "stage_c_access": manifest["stage_c_access"],
        "training_launched": manifest["training_launched"],
        "manifest_file": manifest.get("manifest_file"),
        "wrote_files": write_files,
        "selected_market_state_profiles_file": manifest.get(
            "selected_market_state_profiles_file"
        ),
        "market_state_profile_count": manifest.get("market_state_profile_count"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
