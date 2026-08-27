"""WP-PRETRAIN library v3 (Data-First order @7886de39;
DATA-SOTA-341..346 and 347..352 corrected).

Discipline added by 347..352 on top of the 341..346 corrections:

* 347 — origin authority is VERIFIED, not syntactic: every temporal
  field parses as strict timezone-aware ISO (naive == UTC, impossible
  calendar dates refuse) and later origins require a typed
  ``earlier_origin_decision`` whose referenced manifest is loaded,
  digest-verified, origin-ordered and chronologically anterior to this
  contract's materialization.
* 348 — the branch assignment is a COMPLETE ORDERED PARTITION of
  ``feature_columns``: every feature exactly once, families non-empty,
  within-family order canonical (global column order); the global and
  per-family ordered digests are persisted and bound to identity.
* 349 — the causal fit period splits chronologically into
  train / calibration / monitor: objective weights calibrate ONCE on
  calibration, training touches only train, monitor only checkpoints
  and reports; boundaries, counts and digests are persisted.
* 350 — ONE input domain: the transferred encoder always consumes the
  exact runtime-preprocessed tensor (masking is corruption, not a
  domain change); normalization policies apply to reconstruction
  TARGETS only (objective-side adapters, excluded from transfer).
* 341..346 — causal per-origin fit boundary, executing-preprocessor
  windows, mask-safe visible-only target statistics, typed per-family
  policies with declared eps, monotone quantile head + diagnostics,
  complete resume identity over atomic digest-sealed generations.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_plugins._observation_contract import feature_columns_sha256

FIRST_ORIGIN = "o2022"
DEVELOPMENT_OUTER_START = "2024-01-01T00:00:00+00:00"
NORMALIZATION_POLICIES = ("identity_preprocessed", "window_zscore_visible")
OBJECTIVE_DOMAINS = ("runtime_domain_with_target_adapters",
                     "single_domain_raw_targets")
ORIGIN_DECISION_SCHEMA = "agent_multi.origin_decision.v1"


class PretrainContractError(ValueError):
    """Typed refusal: the pretraining contract, its resume identity or
    its artifact generation is invalid. Never construct, never train."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def parse_iso_utc(value: Any, label: str) -> datetime:
    """DATA-SOTA-347: strict timezone-aware ISO parsing. Impossible
    calendar dates and non-strings refuse; a naive timestamp is
    declared UTC."""
    if not isinstance(value, str) or not value.strip():
        raise PretrainContractError(
            f"{label} must be a non-empty ISO-8601 string, got "
            f"{value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PretrainContractError(
            f"{label}={value!r} is not a valid ISO-8601 timestamp: "
            f"{exc}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# --------------------------------------------------- 348: partition

def validate_branch_partition(columns: list[str],
                              branches: list[dict[str, Any]]
                              ) -> dict[str, Any]:
    """DATA-SOTA-348: the branch assignment must be a COMPLETE ordered
    partition of ``columns`` — every feature exactly once, no empty
    family, and within-family order canonical (the global column
    order), so a reordered identity cannot masquerade as the same
    assignment."""
    if not columns:
        raise PretrainContractError("feature_columns must not be empty")
    position = {name: i for i, name in enumerate(columns)}
    claimed: dict[str, str] = {}
    family_digests: dict[str, str] = {}
    for number, branch in enumerate(branches):
        family = str(branch.get("name") or "")
        features = list(branch.get("features") or [])
        if not family:
            raise PretrainContractError(
                f"branches[{number}] has no name")
        if not features:
            raise PretrainContractError(
                f"branch {family} claims no features (empty family)")
        unknown = [f for f in features if f not in position]
        if unknown:
            raise PretrainContractError(
                f"branch {family} features not in feature_columns: "
                f"{unknown}")
        for name in features:
            if name in claimed:
                raise PretrainContractError(
                    f"feature {name} assigned to both "
                    f"{claimed[name]} and {family}")
            claimed[name] = family
        order = [position[f] for f in features]
        if order != sorted(order):
            raise PretrainContractError(
                f"branch {family} features are not in canonical global "
                f"column order (DATA-SOTA-348: reordered identities "
                f"refuse): {features}")
        family_digests[family] = feature_columns_sha256(features)
    missing = [name for name in columns if name not in claimed]
    if missing:
        raise PretrainContractError(
            f"features without a branch (incomplete partition, "
            f"DATA-SOTA-348): {missing}")
    return {"global_ordered_digest": feature_columns_sha256(columns),
            "family_ordered_digests": family_digests,
            "coverage": {"feature_count": len(columns),
                         "family_count": len(branches),
                         "assignment": {b["name"]: list(b["features"])
                                        for b in branches}}}


# --------------------------------------------------- 347: origin chain

def verify_earlier_origin_decision(contract: dict[str, Any],
                                   repo_root: Path) -> dict[str, Any]:
    """DATA-SOTA-347/356: a later origin materializes only against a
    LOADED, digest-verified, chronologically anterior frozen decision
    of its EXACT immediate predecessor — never a bare string, never a
    skipped origin."""
    parsed = validate_contract(contract)
    origin = contract["score_origin"]
    predecessor = parsed.get("predecessor_origin_id")
    if predecessor is None:
        raise PretrainContractError(
            f"origin {origin.get('origin_id')} is the first plan "
            f"origin; no earlier decision applies")
    decision = contract.get("earlier_origin_decision")
    if not isinstance(decision, dict):
        raise PretrainContractError(
            f"origin {origin['origin_id']} requires a typed "
            f"earlier_origin_decision object (DATA-SOTA-347)")
    for key in ("origin_id", "decided_at", "artifact",
                "artifact_sha256"):
        if not str(decision.get(key) or "").strip():
            raise PretrainContractError(
                f"earlier_origin_decision.{key} is required")
    artifact_path = repo_root / str(decision["artifact"])
    if not artifact_path.is_file():
        raise PretrainContractError(
            f"earlier_origin_decision.artifact absent: "
            f"{decision['artifact']}")
    actual_sha = sha256_file(artifact_path)
    if actual_sha != str(decision["artifact_sha256"]):
        raise PretrainContractError(
            f"earlier_origin_decision digest mismatch: declared "
            f"{decision['artifact_sha256']}, actual {actual_sha}")
    manifest = json.loads(artifact_path.read_text())
    if manifest.get("schema") != ORIGIN_DECISION_SCHEMA:
        raise PretrainContractError(
            f"decision artifact schema must be "
            f"{ORIGIN_DECISION_SCHEMA}, got {manifest.get('schema')!r}")
    if str(decision["origin_id"]) != predecessor:
        raise PretrainContractError(
            f"earlier_origin_decision references "
            f"{decision['origin_id']!r} but the IMMEDIATE predecessor "
            f"is {predecessor!r} (DATA-SOTA-356)")
    if str(manifest.get("origin_id")) != str(decision["origin_id"]):
        raise PretrainContractError(
            f"decision artifact origin {manifest.get('origin_id')!r} "
            f"differs from referenced {decision['origin_id']!r}")
    earlier_start = parse_iso_utc(manifest.get("score_start"),
                                  "decision artifact score_start")
    this_start = parse_iso_utc(origin["score_start"],
                               "score_origin.score_start")
    if earlier_start >= this_start:
        raise PretrainContractError(
            f"referenced origin {decision['origin_id']} does not "
            f"precede {origin['origin_id']} "
            f"({earlier_start.isoformat()} >= {this_start.isoformat()})")
    decided_at = parse_iso_utc(decision["decided_at"],
                               "earlier_origin_decision.decided_at")
    frozen_at = parse_iso_utc(manifest.get("decided_at"),
                              "decision artifact decided_at")
    if decided_at != frozen_at:
        raise PretrainContractError(
            f"decided_at mismatch: contract "
            f"{decided_at.isoformat()} vs artifact "
            f"{frozen_at.isoformat()}")
    materialized_at = parse_iso_utc(
        contract.get("materialized_at"), "materialized_at")
    if decided_at >= materialized_at:
        raise PretrainContractError(
            f"the earlier decision ({decided_at.isoformat()}) must "
            f"predate this origin's materialization "
            f"({materialized_at.isoformat()})")
    return {"artifact": str(decision["artifact"]),
            "artifact_sha256": actual_sha,
            "origin_id": str(decision["origin_id"]),
            "decided_at": decided_at.isoformat()}


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    from feature_branch_plugins._topology import (TopologyError,
                                                  require_int_list,
                                                  strict_int, strict_real)
    if contract.get("schema") != "agent_multi.pretrain_contract.v4":
        raise PretrainContractError(
            f"unsupported pretrain contract schema "
            f"{contract.get('schema')!r}")
    try:
        window = strict_int(contract.get("window_size"), "window_size", 2)
        stride = strict_int(contract.get("window_stride", 1),
                            "window_stride", 1)
        seed = strict_int(contract.get("seed"), "seed", 0)
        epochs = strict_int(contract.get("epochs"), "epochs", 1)
        warmup = strict_int(contract.get("warmup_bars"), "warmup_bars", 2)
        batch = strict_int(
            (contract.get("optimizer") or {}).get("batch_size"),
            "optimizer.batch_size", 1)
        lr = strict_real((contract.get("optimizer") or {}).get("lr"),
                         "optimizer.lr")
        fractions = contract.get("partition_fractions") or {}
        calibration_fraction = strict_real(
            fractions.get("calibration"),
            "partition_fractions.calibration")
        monitor_fraction = strict_real(
            fractions.get("monitor"), "partition_fractions.monitor")
    except TopologyError as exc:
        raise PretrainContractError(str(exc)) from exc
    if lr <= 0:
        raise PretrainContractError(f"optimizer.lr must be > 0, got {lr}")
    # ---- DATA-SOTA-349: chronological three-way partition
    for label, value in (("calibration", calibration_fraction),
                         ("monitor", monitor_fraction)):
        if not (0.0 < value <= 0.4):
            raise PretrainContractError(
                f"partition_fractions.{label} must lie in (0, 0.4], "
                f"got {value}")
    if calibration_fraction + monitor_fraction >= 0.8:
        raise PretrainContractError(
            "train partition would fall below 20% of eligible windows")

    # ---- DATA-SOTA-341/347: causal per-origin boundary, strict ISO
    origin = contract.get("score_origin") or {}
    origin_id = str(origin.get("origin_id") or "")
    if not origin_id:
        raise PretrainContractError(
            "score_origin.origin_id is required (DATA-SOTA-341)")
    score_start = parse_iso_utc(origin.get("score_start"),
                                "score_origin.score_start")
    fit_end = parse_iso_utc(contract.get("fit_end"), "fit_end")
    if fit_end >= score_start:
        raise PretrainContractError(
            f"fit_end={fit_end.isoformat()} does not precede "
            f"score_origin {origin_id} score_start="
            f"{score_start.isoformat()}: monitor and inner validation "
            f"years are reserved (DATA-SOTA-341)")
    if fit_end >= parse_iso_utc(DEVELOPMENT_OUTER_START,
                                "development_outer boundary"):
        raise PretrainContractError(
            f"fit_end={fit_end.isoformat()} reaches development_outer "
            f"(2024+); sealed 2025 is structurally excluded")
    # ---- DATA-SOTA-356: ordered origin plan with exact predecessors
    plan = contract.get("origin_plan")
    if not isinstance(plan, list) or not plan:
        raise PretrainContractError(
            "origin_plan is required: an ordered list of origins with "
            "exact predecessor_origin_id (DATA-SOTA-356)")
    seen_ids: list[str] = []
    previous_start = None
    for number, entry in enumerate(plan):
        entry_id = str((entry or {}).get("origin_id") or "")
        if not entry_id:
            raise PretrainContractError(
                f"origin_plan[{number}].origin_id is required")
        if entry_id in seen_ids:
            raise PretrainContractError(
                f"origin_plan contains duplicate origin {entry_id}")
        entry_start = parse_iso_utc(entry.get("score_start"),
                                    f"origin_plan[{number}].score_start")
        if previous_start is not None and entry_start <= previous_start:
            raise PretrainContractError(
                f"origin_plan score_starts must strictly increase; "
                f"{entry_id} does not follow chronologically")
        declared_pred = entry.get("predecessor_origin_id")
        expected_pred = seen_ids[-1] if seen_ids else None
        if declared_pred != expected_pred:
            raise PretrainContractError(
                f"origin_plan[{number}] ({entry_id}) declares "
                f"predecessor {declared_pred!r}; the immediate "
                f"predecessor is {expected_pred!r} (DATA-SOTA-356)")
        seen_ids.append(entry_id)
        previous_start = entry_start
    if origin_id not in seen_ids:
        raise PretrainContractError(
            f"score_origin {origin_id} is not declared in origin_plan "
            f"(unknown origin refuses; DATA-SOTA-356)")
    predecessor_origin_id = (
        None if origin_id == seen_ids[0]
        else seen_ids[seen_ids.index(origin_id) - 1])
    plan_start = parse_iso_utc(
        plan[seen_ids.index(origin_id)]["score_start"],
        "origin_plan score_start")
    if plan_start != score_start:
        raise PretrainContractError(
            f"score_origin.score_start {score_start.isoformat()} "
            f"differs from origin_plan ({plan_start.isoformat()})")

    if predecessor_origin_id is not None:
        if not isinstance(contract.get("earlier_origin_decision"),
                          dict):
            raise PretrainContractError(
                f"origin {origin_id} requires a typed "
                f"earlier_origin_decision object — a bare string "
                f"cannot mint an origin (DATA-SOTA-347)")
        parse_iso_utc(contract.get("materialized_at"),
                      "materialized_at")
        declared = str(contract["earlier_origin_decision"].get(
            "origin_id") or "")
        if declared != predecessor_origin_id:
            raise PretrainContractError(
                f"earlier_origin_decision references {declared!r} but "
                f"the IMMEDIATE predecessor of {origin_id} is "
                f"{predecessor_origin_id!r} — skipping an unresolved "
                f"origin refuses (DATA-SOTA-356)")

    # ---- DATA-SOTA-342: executing observation pipeline binding
    pipe = contract.get("observation_pipeline") or {}
    if str(pipe.get("preprocessor_plugin") or "") != \
            "feature_window_preprocessor":
        raise PretrainContractError(
            "observation_pipeline.preprocessor_plugin must name the "
            "executing preprocessor (feature_window_preprocessor); "
            "pretraining windows must come from the same transform the "
            "env applies (DATA-SOTA-342)")
    if not str(pipe.get("source_config") or ""):
        raise PretrainContractError(
            "observation_pipeline.source_config is required "
            "(DATA-SOTA-342)")

    # ---- DATA-SOTA-350: one declared objective input domain
    domain = str(contract.get("objective_domain") or "")
    if domain not in OBJECTIVE_DOMAINS:
        raise PretrainContractError(
            f"objective_domain must be one of {OBJECTIVE_DOMAINS} "
            f"(DATA-SOTA-350: domains are never mixed silently), got "
            f"{domain!r}")

    branches = contract.get("branches") or []
    if not branches:
        raise PretrainContractError("branches must not be empty")
    partition = validate_branch_partition(
        list(contract.get("feature_columns") or []), branches)

    # ---- DATA-SOTA-344: typed per-family TARGET policies
    policies = contract.get("normalization_policies")
    if not isinstance(policies, dict) or not policies:
        raise PretrainContractError(
            "normalization_policies is required: every branch family "
            "declares a typed policy (DATA-SOTA-344)")
    names = [str(b.get("name")) for b in branches]
    if sorted(policies) != sorted(names):
        raise PretrainContractError(
            f"normalization_policies must cover the branch families "
            f"exactly: policies for {sorted(policies)}, branches "
            f"{sorted(names)} (DATA-SOTA-344)")
    parsed_policies: dict[str, dict[str, Any]] = {}
    for family, spec in policies.items():
        policy = str((spec or {}).get("policy") or "")
        if policy not in NORMALIZATION_POLICIES:
            raise PretrainContractError(
                f"normalization_policies[{family}].policy must be one "
                f"of {NORMALIZATION_POLICIES}, got {policy!r}")
        if (domain == "single_domain_raw_targets"
                and policy != "identity_preprocessed"):
            raise PretrainContractError(
                f"objective_domain single_domain_raw_targets requires "
                f"identity_preprocessed policies; {family} declares "
                f"{policy}")
        eps = None
        if policy == "window_zscore_visible":
            try:
                eps = strict_real(spec.get("eps"),
                                  f"normalization_policies[{family}].eps")
            except TopologyError as exc:
                raise PretrainContractError(str(exc)) from exc
            if eps <= 0:
                raise PretrainContractError(
                    f"normalization_policies[{family}].eps must be > 0")
        parsed_policies[str(family)] = {"policy": policy, "eps": eps}

    # ---- DATA-SOTA-345/349: predeclared calibration-set balancing
    balancing = contract.get("objective_balancing") or {}
    if str(balancing.get("method") or "") not in (
            "inverse_initial_loss", "frozen_gradient_norm"):
        raise PretrainContractError(
            "objective_balancing.method must be 'inverse_initial_loss' "
            "(control) or 'frozen_gradient_norm' (M1 treatment) — "
            "resolved through pretrain_balancing.plugins")
    combiner = contract.get("gradient_combiner") or {}
    if str(combiner.get("plugin") or "") not in ("ordinary_sum",
                                                 "pcgrad"):
        raise PretrainContractError(
            "gradient_combiner.plugin must be 'ordinary_sum' or "
            "'pcgrad' (M1; resolved through "
            "pretrain_combiner.plugins)")
    try:
        floor = strict_real(balancing.get("floor"),
                            "objective_balancing.floor")
    except TopologyError as exc:
        raise PretrainContractError(str(exc)) from exc
    if floor <= 0:
        raise PretrainContractError("objective_balancing.floor must be "
                                    "> 0")

    objectives = contract.get("objectives") or {}
    known = {"masked_patch_reconstruction", "multi_horizon_quantile",
             "hierarchical_contrastive", "volatility", "barrier_hit"}
    unknown = set(objectives) - known
    if unknown:
        raise PretrainContractError(
            f"unknown objectives {sorted(unknown)}; wired: "
            f"{sorted(known)}")
    if not objectives:
        raise PretrainContractError("at least one objective is required")
    for name, spec in objectives.items():
        try:
            weight = strict_real((spec or {}).get("weight"),
                                 f"objectives.{name}.weight")
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        if weight <= 0:
            raise PretrainContractError(
                f"objectives.{name}.weight must be > 0, got {weight}")
    if "masked_patch_reconstruction" in objectives:
        spec = objectives["masked_patch_reconstruction"]
        try:
            span = strict_int(spec.get("mask_span"), "mask_span", 1)
            ratio = strict_real(spec.get("mask_ratio"), "mask_ratio")
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        if not (0.0 < ratio < 1.0):
            raise PretrainContractError(
                f"mask_ratio must lie in (0, 1), got {ratio}")
        if span >= window:
            raise PretrainContractError(
                f"mask_span={span} must be < window_size={window}")
    if "multi_horizon_quantile" in objectives:
        spec = objectives["multi_horizon_quantile"]
        try:
            horizons = require_int_list(spec, "horizons", 1)
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        quantiles = spec.get("quantiles")
        if not isinstance(quantiles, (list, tuple)) or not quantiles:
            raise PretrainContractError("quantiles must be non-empty")
        previous = 0.0
        for q in quantiles:
            try:
                q = strict_real(q, "quantile")
            except TopologyError as exc:
                raise PretrainContractError(str(exc)) from exc
            if not (0.0 < q < 1.0):
                raise PretrainContractError(
                    f"quantiles must lie in (0, 1), got {q}")
            if q <= previous:
                raise PretrainContractError(
                    "quantiles must be strictly increasing (the "
                    "monotone head orders its outputs by quantile)")
            previous = q
        if len(set(horizons)) != len(horizons):
            raise PretrainContractError("horizons must be unique")

    def _unique_horizons(spec, label):
        try:
            hs = require_int_list(spec, "horizons", 1)
        except TopologyError as exc:
            raise PretrainContractError(f"{label}: {exc}") from exc
        if len(set(hs)) != len(hs):
            raise PretrainContractError(f"{label}.horizons must be "
                                        f"unique")
        return hs

    if "hierarchical_contrastive" in objectives:
        spec = objectives["hierarchical_contrastive"]
        try:
            scales = require_int_list(spec, "scales", 1)
            strict_real(spec.get("temperature"), "temperature")
            strict_int(spec.get("projection_dim"), "projection_dim", 1)
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        if float(spec["temperature"]) <= 0:
            raise PretrainContractError("temperature must be > 0")
        if len(set(scales)) != len(scales) or any(
                sc < 2 or sc > window // 2 for sc in scales):
            raise PretrainContractError(
                f"scales must be unique ints in [2, window//2] "
                f"(causal in-window smoothing views), got {scales}")
        negatives = spec.get("negatives") or {}
        try:
            strict_int(negatives.get("exclusion_steps"),
                       "negatives.exclusion_steps", 0)
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        if str(negatives.get("source") or "") != "train_only":
            raise PretrainContractError(
                "negatives.source must be 'train_only' (negatives "
                "never come from calibration/monitor)")
        if not str(negatives.get("false_negative_policy") or "").strip():
            raise PretrainContractError(
                "negatives.false_negative_policy must be declared")

    if "volatility" in objectives:
        spec = objectives["volatility"]
        _unique_horizons(spec, "volatility")
        if str(spec.get("estimator") or "") !=                 "realized_vol_close_to_close":
            raise PretrainContractError(
                "volatility.estimator must be EXPLICITLY "
                "'realized_vol_close_to_close' — no default target "
                "formula")
        if not str(spec.get("units") or "").strip():
            raise PretrainContractError(
                "volatility.units must be declared")
        annualization = spec.get("annualization")
        if annualization != "none" and not (
                isinstance(annualization, dict)
                and isinstance(annualization.get("periods_per_year"),
                               int)
                and not isinstance(annualization.get(
                    "periods_per_year"), bool)
                and annualization["periods_per_year"] >= 1):
            raise PretrainContractError(
                "volatility.annualization must be 'none' or "
                "{periods_per_year: int>=1} — declared, never implied")
        try:
            eps = strict_real(spec.get("epsilon"), "volatility.epsilon")
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        if eps <= 0:
            raise PretrainContractError("volatility.epsilon must be > 0")

    if "barrier_hit" in objectives:
        spec = objectives["barrier_hit"]
        _unique_horizons(spec, "barrier_hit")
        scale = spec.get("barrier_scale") or {}
        if str(scale.get("estimator") or "") !=                 "trailing_realized_vol_close_to_close":
            raise PretrainContractError(
                "barrier_scale.estimator must be EXPLICITLY "
                "'trailing_realized_vol_close_to_close' (past-only)")
        try:
            lookback = strict_int(scale.get("lookback"),
                                  "barrier_scale.lookback", 2)
            strict_real(scale.get("epsilon"), "barrier_scale.epsilon")
            upper = strict_real(spec.get("upper_mult"), "upper_mult")
            lower = strict_real(spec.get("lower_mult"), "lower_mult")
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        if upper <= 0 or lower <= 0:
            raise PretrainContractError(
                "upper_mult and lower_mult must be > 0")
        if lookback > warmup:
            raise PretrainContractError(
                f"barrier_scale.lookback {lookback} exceeds "
                f"warmup_bars {warmup}: the past-only scale must fit "
                f"inside the causal warmup")
        if str(spec.get("same_bar_collision") or "") !=                 "conservative_adverse_first":
            raise PretrainContractError(
                "same_bar_collision must be EXPLICITLY "
                "'conservative_adverse_first'")
        if str(spec.get("class_weights_from") or "") !=                 "calibration_only":
            raise PretrainContractError(
                "class_weights_from must be 'calibration_only'")
        ohlc = spec.get("ohlc_columns") or {}
        if sorted(ohlc.keys()) != ["close", "high", "low", "open"] or \
                not all(str(v).strip() for v in ohlc.values()):
            raise PretrainContractError(
                "barrier_hit.ohlc_columns must declare open/high/low/"
                "close column names — labels are EXECUTABLE intrabar "
                "first-touch, close-only data refuses (DATA-SOTA-364)")

    # DATA-SOTA-353 + WP1: the purge horizon is the maximum FORWARD
    # horizon across ALL objectives
    all_horizons = [1]
    if "multi_horizon_quantile" in objectives:
        all_horizons += list(objectives["multi_horizon_quantile"][
            "horizons"])
    if "volatility" in objectives:
        all_horizons += list(objectives["volatility"]["horizons"])
    if "barrier_hit" in objectives:
        all_horizons += list(objectives["barrier_hit"]["horizons"])

    return {"window_size": window, "window_stride": stride, "seed": seed,
            "epochs": epochs, "batch_size": batch, "lr": lr,
            "fit_end": fit_end, "warmup_bars": warmup,
            "calibration_fraction": calibration_fraction,
            "monitor_fraction": monitor_fraction,
            "origin_id": origin_id, "score_start": score_start,
            "origin_plan": plan,
            "predecessor_origin_id": predecessor_origin_id,
            "objective_domain": domain,
            "balancing_floor": floor,
            "balancing_method": str(balancing["method"]),
            "gradient_combiner": dict(combiner),
            "max_horizon_all_objectives": max(all_horizons),
            "normalization_policies": parsed_policies,
            "partition": partition}


def load_fit_slice(csv_path, contract: dict[str, Any]):
    """Load ONLY the causal per-origin fit slice: rows after ``fit_end``
    (reserved monitor/validation years, development_outer, sealed) are
    never loaded into memory."""
    import pandas as pd

    parsed = validate_contract(contract)
    date_col = str(contract.get("date_column") or "DATE_TIME")
    close_col = str(contract.get("close_column") or "CLOSE")
    columns = list(contract["feature_columns"])
    usecols = [date_col] + columns + (
        [close_col] if close_col not in columns else [])
    barrier = (contract.get("objectives") or {}).get("barrier_hit")
    if barrier:
        for name in barrier["ohlc_columns"].values():
            if name not in usecols:
                usecols.append(name)
    df = pd.read_csv(csv_path, usecols=usecols)
    stamps = pd.to_datetime(df[date_col], utc=True)
    df = df.loc[stamps <= parsed["fit_end"]]
    if df.empty:
        raise PretrainContractError(
            f"no rows at or before fit_end="
            f"{parsed['fit_end'].isoformat()}")
    check_cols = columns + [close_col] + (
        [c for c in barrier["ohlc_columns"].values()
         if c not in columns + [close_col]] if barrier else [])
    if df[check_cols].isna().any().any():
        bad = df[check_cols].isna().any()
        raise PretrainContractError(
            f"NaNs in fit slice columns: {sorted(bad[bad].index)}")
    return df.reset_index(drop=True), columns, close_col


def build_step_index(n_rows: int, warmup_bars: int, stride: int,
                     max_horizon: int,
                     max_windows: int | None) -> list[int]:
    """Eligible preprocessor steps t: the executing window covers rows
    [t-window, t) so the last OBSERVED bar is t-1; require full scaler
    warmup (t >= warmup_bars) and every forward target close[t-1+h]
    inside the fit slice (t-1+max_horizon <= n_rows-1)."""
    first = warmup_bars
    last = n_rows - max_horizon
    steps = list(range(first, last + 1, stride))
    if not steps:
        raise PretrainContractError(
            f"no eligible step: {n_rows} rows, warmup {warmup_bars}, "
            f"max horizon {max_horizon}")
    if max_windows is not None and len(steps) > max_windows:
        steps = steps[-max_windows:]  # keep the newest fit-slice steps
    return steps


def collect_preprocessed_windows(df, contract: dict[str, Any],
                                 env_config: dict[str, Any],
                                 steps: list[int]):
    """DATA-SOTA-342: emit pretraining windows through the SAME
    executing preprocessor plugin the GymFxEnv calls, with the same
    config — one shared transform, not a reimplementation."""
    import numpy as np

    from app.plugin_loader import load_plugin

    plugin_name = contract["observation_pipeline"]["preprocessor_plugin"]
    plugin_class, _ = load_plugin("preprocessor.plugins", plugin_name)
    preprocessor = plugin_class()
    neutral = {"initial_cash": 1000.0, "equity": 1000.0, "price": 0.0,
               "position": 0, "position_units": 0.0, "entry_price": 0.0,
               "holding_bars": 0, "bar_index": 0, "total_bars": len(df)}
    windows = []
    for t in steps:
        obs = preprocessor.make_observation(
            data=df, step=t, bridge_state={**neutral, "bar_index": t},
            config=env_config)
        windows.append(np.asarray(obs["features"], dtype=np.float32))
    return np.stack(windows, axis=0)  # (N, T, F)


def three_way_split(steps: list[int], calibration_fraction: float,
                    monitor_fraction: float, purge_steps: int):
    """DATA-SOTA-349/353: chronological PURGED train / calibration /
    monitor. ``purge_steps`` is derived MECHANICALLY by the caller as
    ``max(horizons)`` — never a free constant — so the final target row
    of each partition precedes the first scored anchor of the next.
    Returns (train, calibration, monitor, purged)."""
    from feature_branch_plugins._topology import TopologyError, strict_int
    try:
        purge_steps = strict_int(purge_steps, "purge_steps", 1)
    except TopologyError as exc:
        raise PretrainContractError(str(exc)) from exc
    n = len(steps)
    n_monitor = max(1, int(round(n * monitor_fraction)))
    n_calibration = max(1, int(round(n * calibration_fraction)))
    calibration_end = n - n_monitor - purge_steps
    train_end = calibration_end - n_calibration - purge_steps
    if train_end < 1:
        raise PretrainContractError(
            f"purged three-way split leaves no training window "
            f"({n} steps, {n_calibration} calibration, {n_monitor} "
            f"monitor, purge {purge_steps} at each boundary)")
    train = steps[:train_end]
    calibration = steps[calibration_end - n_calibration:calibration_end]
    monitor = steps[n - n_monitor:]
    purged = (steps[train_end:calibration_end - n_calibration]
              + steps[calibration_end:n - n_monitor])
    return train, calibration, monitor, purged


def assert_purged_boundaries(train: list[int], calibration: list[int],
                             monitor: list[int], max_horizon: int,
                             stride: int) -> None:
    """DATA-SOTA-353: the last target row of a partition must PRECEDE
    the first scored anchor row of the next (target of step t anchors
    at t-1 and extends to t-1+max_horizon; anchor of step t' is
    t'-1)."""
    for name, left, right in (("train->calibration", train, calibration),
                              ("calibration->monitor", calibration,
                               monitor)):
        if left[-1] - 1 + max_horizon >= right[0] - 1:
            raise PretrainContractError(
                f"target overlap at {name}: last target row "
                f"{left[-1] - 1 + max_horizon} does not precede first "
                f"scored anchor {right[0] - 1} (DATA-SOTA-353)")


def partition_evidence(name: str, steps: list[int], stamps,
                       window_size: int, max_horizon: int,
                       context_rows: int) -> dict:
    """DATA-SOTA-349/353: scored windows, INPUT-CONTEXT range and
    TARGET range bound separately per partition. Context rows are
    shared causal PAST (observation window + scaler history) — they are
    context-only and never counted as scored windows or metrics."""
    return {"partition": name,
            "scored_windows": len(steps),
            "first_step": steps[0], "last_step": steps[-1],
            "first_observed_bar": str(stamps[steps[0] - 1]),
            "last_observed_bar": str(stamps[steps[-1] - 1]),
            "input_context": {
                "first_context_row": max(0, steps[0] - context_rows),
                "last_observation_row": steps[-1] - 1,
                "context_rows_before_first_step": context_rows,
                "role": "context_only_shared_causal_past"},
            "target_range": {
                "first_target_row": steps[0],
                "last_target_row": steps[-1] - 1 + max_horizon},
            "steps_sha256": sha256_obj(steps)}


def sample_span_mask(batch: int, window: int, ratio: float, span: int,
                     generator):
    """(B, T) boolean temporal mask covering ~ratio of the window in
    contiguous spans; always leaves >=1 masked and >=1 visible step."""
    import torch

    n_spans = max(1, int(round(window * ratio / span)))
    mask = torch.zeros(batch, window, dtype=torch.bool)
    starts = torch.randint(0, window - span + 1, (batch, n_spans),
                           generator=generator)
    for i in range(batch):
        for s in starts[i].tolist():
            mask[i, s:s + span] = True
        if mask[i].all():  # keep at least one visible step
            mask[i, 0] = False
    return mask


def masked_visible_normalize(windows, mask, eps: float):
    """DATA-SOTA-343: normalization statistics come from VISIBLE steps
    only — changing masked raw values cannot change any visible
    normalized value. eps is the DECLARED contract value
    (DATA-SOTA-344), applied here, not a hardcoded default."""
    import torch

    visible = (~mask).unsqueeze(-1).to(windows.dtype)   # (B, T, 1)
    count = visible.sum(dim=1, keepdim=True)
    mean = (windows * visible).sum(dim=1, keepdim=True) / count
    var = (((windows - mean) ** 2) * visible).sum(dim=1,
                                                  keepdim=True) / count
    return (windows - mean) / (torch.sqrt(var) + eps)


def reconstruction_target(values, mask, policy: dict[str, Any]):
    """DATA-SOTA-350: the policy transforms the reconstruction TARGET
    only. The encoder always consumes the runtime-preprocessed tensor;
    no per-objective input domain exists."""
    if policy["policy"] == "identity_preprocessed":
        return values  # executing preprocessor output, untouched
    return masked_visible_normalize(values, mask, float(policy["eps"]))


def masked_reconstruction_loss(encoder, head, values, target, mask):
    """Encode the MASKED runtime tensor, reconstruct the (possibly
    policy-transformed) target, score ONLY masked steps."""
    import torch

    masked_in = values.masked_fill(mask.unsqueeze(-1), 0.0)
    pred = head(encoder(masked_in)).view(target.shape)
    diff = (pred - target)[mask]
    if diff.numel() == 0:
        return torch.zeros((), dtype=values.dtype)
    return (diff ** 2).mean()


def build_monotone_quantile_head(dim: int, n_horizons: int,
                                 n_quantiles: int):
    """DATA-SOTA-345: quantile outputs are monotone BY CONSTRUCTION —
    the first output is the lowest quantile and the rest are cumulative
    softplus increments, so crossing is structurally impossible."""
    import torch
    import torch.nn as nn

    class MonotoneQuantileHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(dim, n_horizons * n_quantiles)

        def forward(self, embedding):
            raw = self.proj(embedding).view(-1, n_horizons, n_quantiles)
            base = raw[..., :1]
            if n_quantiles == 1:
                return base
            steps = torch.nn.functional.softplus(raw[..., 1:])
            return torch.cat([base, base + torch.cumsum(steps, dim=-1)],
                             dim=-1)

    return MonotoneQuantileHead()


def pinball_loss(pred, target, quantiles):
    """pred (B, H, Q), target (B, H): mean quantile (pinball) loss."""
    import torch

    q = torch.tensor(list(quantiles), dtype=pred.dtype,
                     device=pred.device).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def quantile_crossing_rate(pred) -> float:
    """Fraction of adjacent quantile pairs where a higher quantile
    predicts BELOW a lower one (measured even though the monotone head
    makes it structurally zero)."""
    if pred.shape[-1] < 2:
        return 0.0
    crossings = (pred[..., 1:] < pred[..., :-1])
    return float(crossings.float().mean())


def forward_log_return_targets(close_values, steps, horizons):
    """(N, H) strictly-forward log returns from the LAST OBSERVED bar of
    each executing window: log(close[t-1+h] / close[t-1]) for step t
    (the window covers rows [t-window, t))."""
    import numpy as np

    close = np.asarray(close_values, dtype=np.float64)
    if (close <= 0).any():
        raise PretrainContractError("non-positive close in fit slice")
    anchor = np.asarray(steps) - 1
    cols = [np.log(close[anchor + h] / close[anchor]) for h in horizons]
    return np.stack(cols, axis=1).astype(np.float32)


def objective_gradient_diagnostics(encoder, losses: dict[str, Any]):
    """DATA-SOTA-345: per-objective gradient norm over the SHARED
    encoder parameters plus pairwise cosine (gradient conflict)."""
    import torch

    grads = {}
    params = [p for p in encoder.parameters() if p.requires_grad]
    for name, loss in losses.items():
        g = torch.autograd.grad(loss, params, retain_graph=True,
                                allow_unused=True)
        flat = torch.cat([x.reshape(-1) for x in g if x is not None])
        grads[name] = flat
    report: dict[str, Any] = {
        "norms": {k: round(float(v.norm()), 6) for k, v in grads.items()}}
    names = sorted(grads)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            denom = float(grads[a].norm() * grads[b].norm())
            cos = float(grads[a] @ grads[b]) / denom if denom > 0 else 0.0
            report[f"cosine:{a}|{b}"] = round(cos, 6)
    return report


def balance_objective_weights(initial_losses: dict[str, float],
                              declared: dict[str, float],
                              floor: float) -> dict[str, float]:
    """Predeclared bounded balancing (DATA-SOTA-345/349): effective
    weight = declared / max(initial CALIBRATION loss, floor), frozen
    before epoch 0. The monitor never calibrates."""
    return {name: declared[name] / max(float(initial_losses[name]), floor)
            for name in declared}


# ------------------------------------------------------------ durability

def resume_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """DATA-SOTA-346: EVERY identity field binds resume — nothing about
    the executing code, data, contract or environment may drift."""
    return dict(manifest["identity"])


def refuse_on_identity_drift(saved: dict[str, Any],
                             current: dict[str, Any]) -> None:
    drift = {k: (saved.get(k), current.get(k))
             for k in set(saved) | set(current)
             if saved.get(k) != current.get(k)}
    if drift:
        raise PretrainContractError(
            "resume identity drift REFUSED: "
            + "; ".join(f"{k}: saved={a!r} current={b!r}"
                        for k, (a, b) in sorted(drift.items())))


def _fsync_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_generation(out_dir: Path, checkpoint_obj: dict[str, Any],
                     manifest_obj: dict[str, Any],
                     generation: int) -> None:
    """DATA-SOTA-346: checkpoint + manifest land as ONE atomic fsynced
    generation sealed by digest; a reader can always detect a torn
    pair."""
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(checkpoint_obj, buffer)
    ckpt_bytes = buffer.getvalue()
    manifest_bytes = json.dumps(manifest_obj, indent=1).encode()
    _fsync_write_bytes(out_dir / "checkpoint.pt", ckpt_bytes)
    _fsync_write_bytes(out_dir / "pretrain_manifest.json",
                       manifest_bytes)
    seal = {"schema": "agent_multi.pretrain_generation.v1",
            "generation": generation,
            "checkpoint_sha256": hashlib.sha256(ckpt_bytes).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                manifest_bytes).hexdigest()}
    _fsync_write_bytes(out_dir / "generation.json",
                       json.dumps(seal, indent=1).encode())


def load_generation(out_dir: Path):
    """Verify the generation seal; a torn checkpoint/manifest pair is a
    typed refusal, never a silent resume."""
    import torch

    seal_path = out_dir / "generation.json"
    ckpt_path = out_dir / "checkpoint.pt"
    manifest_path = out_dir / "pretrain_manifest.json"
    if not (seal_path.is_file() and ckpt_path.is_file()
            and manifest_path.is_file()):
        raise PretrainContractError(
            "resume REFUSED: no sealed generation in output dir")
    seal = json.loads(seal_path.read_text())
    actual_ckpt = sha256_file(ckpt_path)
    actual_manifest = sha256_file(manifest_path)
    if (seal.get("checkpoint_sha256") != actual_ckpt
            or seal.get("manifest_sha256") != actual_manifest):
        raise PretrainContractError(
            "resume REFUSED: TORN GENERATION — checkpoint/manifest do "
            "not match the generation seal (DATA-SOTA-346)")
    return (torch.load(ckpt_path, weights_only=False),
            json.loads(manifest_path.read_text()),
            int(seal["generation"]))


def canonical_feature_digest(columns) -> str:
    # F01 digest unity: the pipeline's canonical serialization, reused.
    return feature_columns_sha256(columns)


# ------------------------- WP1 objectives (order 2026-08-27, post-transfer)

def realized_volatility_targets(close_values, steps, horizons,
                                epsilon: float, periods_per_year):
    """(N, H) log realized volatility, STRICTLY FORWARD from the last
    observed bar a = t-1 of step t: r_i = log(close[a+i]/close[a+i-1]),
    vol_h = sqrt(mean(r_1..r_h ^2)) (per-bar close-to-close realized
    std), annualized by sqrt(periods_per_year) when declared, target =
    log(vol + epsilon). Formula is the contract's declared estimator —
    never a default."""
    import numpy as np

    close = np.asarray(close_values, dtype=np.float64)
    if (close <= 0).any():
        raise PretrainContractError("non-positive close in fit slice")
    log_close = np.log(close)
    returns = np.diff(log_close)  # returns[i] = log(close[i+1]/close[i])
    anchor = np.asarray(steps) - 1
    columns = []
    for h in horizons:
        window_returns = np.stack(
            [returns[anchor + i] for i in range(h)], axis=1)
        vol = np.sqrt((window_returns ** 2).mean(axis=1))
        if periods_per_year is not None:
            vol = vol * np.sqrt(float(periods_per_year))
        columns.append(np.log(vol + epsilon))
    return np.stack(columns, axis=1).astype(np.float32)


def validate_ohlc(open_, high, low, close):
    """DATA-SOTA-364: OHLC must be finite, non-inverted and aligned
    (high >= max(open, close), low <= min(open, close)) — anything else
    REFUSES; there is NO close-only fallback."""
    import numpy as np

    arrays = {"open": np.asarray(open_, dtype=np.float64),
              "high": np.asarray(high, dtype=np.float64),
              "low": np.asarray(low, dtype=np.float64),
              "close": np.asarray(close, dtype=np.float64)}
    n = len(arrays["close"])
    for name, arr in arrays.items():
        if len(arr) != n:
            raise PretrainContractError(
                f"misaligned OHLC: {name} has {len(arr)} rows, close "
                f"has {n} (DATA-SOTA-364)")
        if not np.isfinite(arr).all():
            raise PretrainContractError(
                f"non-finite OHLC values in {name} (DATA-SOTA-364)")
        if (arr <= 0).any():
            raise PretrainContractError(
                f"non-positive OHLC values in {name}")
    if (arrays["high"] < arrays["low"]).any():
        raise PretrainContractError(
            "inverted OHLC: high < low (DATA-SOTA-364)")
    body_max = np.maximum(arrays["open"], arrays["close"])
    body_min = np.minimum(arrays["open"], arrays["close"])
    if (arrays["high"] < body_max).any() or             (arrays["low"] > body_min).any():
        raise PretrainContractError(
            "misaligned OHLC: high/low do not contain the open/close "
            "body (DATA-SOTA-364)")
    return arrays


def barrier_hit_labels(open_, high, low, close, steps, horizons,
                       lookback: int, upper_mult: float,
                       lower_mult: float, epsilon: float):
    """(N, H) int64 EXECUTABLE first-touch labels (DATA-SOTA-364):
    0 = first UPPER hit, 1 = first LOWER hit, 2 = neither/censored.

    Barrier scale: PAST-ONLY trailing realized vol (close-to-close)
    over ``lookback`` bars ending at the anchor a = t-1; barriers fixed
    at close[a]. The walk over (a, a+h] uses timestamp-aligned
    HIGH/LOW: future HIGH >= upper hits the upper barrier, future LOW
    <= lower hits the lower barrier — exactly how the shared execution
    envelope triggers intrabar (gap-throughs included, since
    high >= open). Both barriers inside ONE bar resolve by the declared
    conservative rule: ADVERSE (lower) first. Close-only data REFUSES
    — no fallback."""
    import numpy as np

    arrays = validate_ohlc(open_, high, low, close)
    close_arr = arrays["close"]
    log_close = np.log(close_arr)
    returns = np.diff(log_close)
    anchor = np.asarray(steps) - 1
    trailing = np.stack(
        [returns[anchor - lookback + i] for i in range(lookback)],
        axis=1)
    scale = np.sqrt((trailing ** 2).mean(axis=1)) + epsilon
    upper_level = close_arr[anchor] * (1.0 + upper_mult * scale)
    lower_level = close_arr[anchor] * (1.0 - lower_mult * scale)
    n = len(anchor)
    max_h = max(horizons)
    future_high = np.stack(
        [arrays["high"][anchor + i] for i in range(1, max_h + 1)],
        axis=1)
    future_low = np.stack(
        [arrays["low"][anchor + i] for i in range(1, max_h + 1)],
        axis=1)
    upper_hit = future_high >= upper_level[:, None]
    lower_hit = future_low <= lower_level[:, None]
    labels = np.full((n, len(horizons)), 2, dtype=np.int64)
    for row in range(n):
        first_upper = np.argmax(upper_hit[row]) if upper_hit[row].any() \
            else max_h + 1
        first_lower = np.argmax(lower_hit[row]) if lower_hit[row].any() \
            else max_h + 1
        for col, h in enumerate(horizons):
            up = first_upper if first_upper < h else max_h + 1
            lo = first_lower if first_lower < h else max_h + 1
            if up == max_h + 1 and lo == max_h + 1:
                labels[row, col] = 2
            elif lo <= up:  # same-bar collision: adverse-first
                labels[row, col] = 1
            else:
                labels[row, col] = 0
    return labels


def frozen_class_weights(labels, n_classes: int = 3):
    """DATA-SOTA-349 discipline for barrier classes: balance weights
    derive from the CALIBRATION labels only and are frozen. Absent
    classes get weight 1.0 (recorded)."""
    import numpy as np

    labels = np.asarray(labels)
    weights = []
    for col in range(labels.shape[1]):
        counts = np.bincount(labels[:, col], minlength=n_classes)
        total = counts.sum()
        weights.append([float(total / (n_classes * c)) if c > 0 else 1.0
                        for c in counts])
    return weights  # (H, n_classes)


def causal_scale_view(windows, scale: int):
    """Causal in-window smoothing view for the hierarchical contrastive
    objective: average-pool along time at ``scale`` then upsample back
    (repeat) to the original length. Uses ONLY values inside the
    window — no future, calibration or monitor access."""
    import torch

    b, t, c = windows.shape
    usable = (t // scale) * scale
    pooled = windows[:, t - usable:, :].reshape(
        b, usable // scale, scale, c).mean(dim=2)
    smoothed = pooled.repeat_interleave(scale, dim=1)
    if smoothed.shape[1] < t:  # left-pad with the oldest smoothed value
        pad = smoothed[:, :1, :].expand(b, t - smoothed.shape[1], c)
        smoothed = torch.cat([pad, smoothed], dim=1)
    return smoothed


def build_projection_head(dim: int, projection_dim: int):
    """Contrastive projection head — an EXCLUDED transfer adapter."""
    import torch.nn as nn

    return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                         nn.Linear(dim, projection_dim))


def hierarchical_contrastive_loss(encoder, projection, windows,
                                  batch_positions, scales,
                                  temperature: float,
                                  exclusion_steps: int):
    """InfoNCE over causal multi-scale views. Positives: (window,
    smoothed view) of the SAME training window per declared scale.
    Negatives: the OTHER train windows in the batch, deterministically,
    minus temporal neighbors within ``exclusion_steps`` of the anchor
    (the declared false-negative policy). Returns (loss, diagnostics:
    embedding_std, effective_negatives_mean, per_scale)."""
    import torch
    import torch.nn.functional as F

    anchor_embedding = encoder(windows)
    anchor_z = F.normalize(projection(anchor_embedding), dim=-1)
    positions = torch.as_tensor(batch_positions)
    distance = (positions[:, None] - positions[None, :]).abs()
    negative_mask = distance > exclusion_steps  # excludes self too
    effective_negatives = negative_mask.sum(dim=1).float()
    total = torch.zeros((), dtype=anchor_z.dtype)
    per_scale = {}
    for scale in scales:
        view_z = F.normalize(
            projection(encoder(causal_scale_view(windows, scale))),
            dim=-1)
        logits = anchor_z @ view_z.T / temperature       # (B, B)
        positive = torch.diagonal(logits)
        masked = logits.masked_fill(~negative_mask, float("-inf"))
        denominator = torch.logsumexp(
            torch.cat([positive.unsqueeze(1), masked], dim=1), dim=1)
        scale_loss = (denominator - positive).mean()
        per_scale[f"scale_{scale}"] = round(
            float(scale_loss.detach()), 6)
        total = total + scale_loss
    loss = total / len(scales)
    diagnostics = {
        "embedding_std": round(float(anchor_embedding.std()), 6),
        "projection_std": round(float(anchor_z.std()), 6),
        "effective_negatives_mean": round(
            float(effective_negatives.mean()), 2),
        "per_scale": per_scale}
    return loss, diagnostics


def build_volatility_head(dim: int, n_horizons: int):
    import torch.nn as nn

    return nn.Linear(dim, n_horizons)


def build_barrier_head(dim: int, n_horizons: int, n_classes: int = 3):
    import torch.nn as nn

    return nn.Linear(dim, n_horizons * n_classes)


def barrier_loss(pred, labels, class_weights):
    """pred (B, H*3) -> per-horizon 3-class CE with FROZEN
    calibration-derived weights."""
    import torch
    import torch.nn.functional as F

    b = pred.shape[0]
    n_h = labels.shape[1]
    logits = pred.view(b, n_h, 3)
    total = torch.zeros((), dtype=pred.dtype)
    for col in range(n_h):
        weight = torch.tensor(class_weights[col], dtype=pred.dtype)
        total = total + F.cross_entropy(logits[:, col, :],
                                        labels[:, col], weight=weight)
    return total / n_h
