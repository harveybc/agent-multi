"""Stage-integrated ``easy -> normal`` solvency curriculum (order WP-D).

For EVERY candidate, in every outer DOIN stage, this pipeline:

1. decodes/freezes the candidate contract exactly as the validation
   pipeline does (same splits, same observation contract);
2. trains first under ``easy_chronological_continuation`` (train-only
   relaxed solvency dynamics, WP-C) with early stopping and a declared
   maximum budget — the easy score controls only the budget, never
   selection;
3. saves an immutable ``post_easy`` artifact (weights, sha256, easy
   event counters, history);
4. continues under ``normal_realistic`` FROM the learned weights via the
   validation pipeline's warm start; the SB3 artifact reload at the
   dynamics boundary yields a fresh replay buffer, so unrealistic easy
   transitions can never contaminate normal updates;
5. saves the ``post_normal`` artifact and returns the parent pipeline's
   transparent normal-validation selection result unchanged.

Evaluation is structurally normal: the parent's evaluation path forces
``normal_realistic`` for every split env, and gym-fx itself refuses easy
dynamics outside ``env_mode=training``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from agent_plugins.sac_agent import _policy_tensor_hash
from pipeline_plugins import _actor_liveness as _liveness
from pipeline_plugins import _paired_generalization as _paired
from pipeline_plugins._observation_contract import (
    apply_observation_contract,
)
from pipeline_plugins.rl_pipeline_with_validation import (
    PipelinePlugin as ValidationPipelinePlugin,
    _verify_artifact_sha256,
)

EASY_MODE = "easy_chronological_continuation"
NORMAL_MODE = "normal_realistic"

# Phase-1 handoff selection semantics (finding 220 / mechanism-ladder
# order WP3). The M0 screen of 2026-08-07 ran the *v3* boundary: the
# epoch-0 warm-start baseline (the anchor itself) was ELIGIBLE to be
# selected as the post_easy handoff, selection scored the easy-probe
# economic equity, and the normal-handoff probe GATED the save. The
# corrected L1 boundary (v4, findings 159/160/195/200) makes epoch 0
# structurally ineligible, selects on the paired comparator over
# trained epochs only, and demotes the normal probe to telemetry. The
# bounded D0-D4 diagnostic must reproduce the M0 boundary *as the
# screen ran it*, so the semantics are a typed, config-gated choice —
# the default is and stays the corrected v4 behavior.
HANDOFF_SEMANTICS_L1_V4 = "l1_trained_epoch_v4"
HANDOFF_SEMANTICS_M0_V3 = "m0_epoch0_eligible_v3"
_HANDOFF_SEMANTICS = (HANDOFF_SEMANTICS_L1_V4, HANDOFF_SEMANTICS_M0_V3)

# ---------------------------------------------------------------------------
# Handoff-viability evidence (AUD-F1-20260811-221 / order WP3 §7).
#
# `easy_activity_eligible` counts trades and non-hold actions under the
# phase-1 threshold only; near-constant policies whose raw actions never
# reach the phase-2 deadband pass it while their normal probes trade
# zero. EVERY phase-1 checkpoint record therefore carries a typed
# evidence block measuring the DETERMINISTIC raw action distribution on
# train-monitor and inner validation, evaluated against BOTH thresholds
# on the SAME captured action vector (the WP2 same-vector design).
#
# The block is EVIDENCE, never a selector: the paired comparator (v4)
# and the as-run v3 economic score remain the only selection
# authorities, and no invented numeric activity floor exists here — the
# constant-policy call derives its tolerance purely from dtype
# precision at the observed action scale.
#
# AUD-F1-20260812-231 (gap closed here): the evidence rollouts run on
# train_monitor and inner_validation, whose manifest role entries
# declare a causal-context prefix. Those prefix rows are INPUT to the
# observation window, never measurement, so they are separated by the
# same mechanism the corrected selector uses — the role and its
# `context_rows` resolved from the VERIFIED nested manifest, the
# ContextPrefixWrapper installed inside `_make_split_env` before the
# rollout — and contribute nothing to the distribution, the quantiles,
# the mapped direction counts, the constant-policy classification or
# the action-vector sha256. The counts of both populations are recorded
# so an auditor sees the separation directly instead of inferring it.
# ---------------------------------------------------------------------------
HANDOFF_VIABILITY_SCHEMA = (
    "agent_multi.solvency_curriculum.handoff_viability.v1")
SELECTED_HANDOFF_SCHEMA = (
    "agent_multi.solvency_curriculum.selected_handoff_viability.v1")
VIABILITY_VIABLE = "VIABLE"
VIABILITY_BELOW_NORMAL_THRESHOLD = "BELOW_NORMAL_THRESHOLD"
VIABILITY_CONSTANT_POLICY = "CONSTANT_POLICY"
VIABILITY_NO_TRADE = "NO_TRADE"
VIABILITY_UNAVAILABLE = "UNAVAILABLE"
HANDOFF_VIABILITY_VALUES = (
    VIABILITY_VIABLE,
    VIABILITY_BELOW_NORMAL_THRESHOLD,
    VIABILITY_CONSTANT_POLICY,
    VIABILITY_NO_TRADE,
    VIABILITY_UNAVAILABLE,
)
PROVENANCE_ANCHOR_PASSTHROUGH = "anchor_passthrough"
PROVENANCE_TRAINED_EPOCH = "trained_epoch"

# Declared, machine-readable statements of WHAT the evidence measured
# and HOW the causal prefix was separated (finding 231 requirement 2).
EVIDENCE_POPULATION = (
    "scored rows only: actions requested on causal-context rows are"
    " forced to hold by the ContextPrefixWrapper and are excluded from"
    " every statistic in this block and from action_vector_sha256")
CONTEXT_SEPARATION_MECHANISM = (
    "role and context_rows resolved from the VERIFIED nested split"
    " manifest (never a file name, never a row position); the"
    " ContextPrefixWrapper is installed inside _make_split_env before"
    " the rollout — the SAME path the executing selector uses; a role"
    " that declares zero context rows and a legacy run with no nested"
    " manifest are built unwrapped and unchanged")

# gym-fx (app/env.py) maps continuous actions through this default when
# the config omits the deadband. Recording the source alongside the
# value keeps the evidence from silently guessing a threshold.
_GYM_FX_DEFAULT_CONTINUOUS_THRESHOLD = 0.33
_HANDOFF_SPLIT_NAMES = ("train_monitor", "inner_validation")


def _resolve_action_threshold(raw: Any) -> tuple:
    """Resolve a deadband exactly as gym-fx would; declare the source."""
    if raw is None:
        return _GYM_FX_DEFAULT_CONTINUOUS_THRESHOLD, "gym_fx_default"
    return float(raw), "config"


def _non_hold_fraction(values: "np.ndarray", threshold: float) -> float:
    """Fraction of raw actions the gym-fx continuous mapping would
    treat as non-hold: strict ``|a| > 0`` at threshold zero, otherwise
    ``|a| >= thr`` (mirrors gym-fx app/env.py exactly)."""
    if values.size == 0:
        return 0.0
    thr = float(threshold)
    if thr == 0.0:
        return float(np.mean(np.abs(values) > 0.0))
    return float(np.mean(np.abs(values) >= thr))


def _mapped_direction_counts(
        values: "np.ndarray", threshold: float) -> Dict[str, int]:
    thr = float(threshold)
    if thr == 0.0:
        long_mask = values > 0.0
        short_mask = values < 0.0
    else:
        long_mask = values >= thr
        short_mask = values <= -thr
    long_count = int(np.count_nonzero(long_mask))
    short_count = int(np.count_nonzero(short_mask))
    return {
        "long": long_count,
        "short": short_count,
        "hold": int(values.size - long_count - short_count),
    }


def _dtype_constant_classification(
        values: "np.ndarray") -> Dict[str, Any]:
    """Exact/near-constant facts derived ONLY from dtype precision at
    the observed action scale — never from an invented behavior floor.

    Declared tolerance: ``finfo(dtype).eps * max(1.0, max|a|)`` (one
    machine epsilon at the observed scale, floored at the gym-fx action
    unit scale ``|a| <= 1``). ``near_constant`` holds iff the full
    peak-to-peak spread of the deterministic action vector fits inside
    that tolerance; ``exact_constant`` iff exactly one bitwise-unique
    value was emitted."""
    dtype = values.dtype
    eps = float(np.finfo(dtype).eps)
    observed_max_abs = float(np.max(np.abs(values)))
    tolerance = eps * max(1.0, observed_max_abs)
    peak_to_peak = float(np.max(values) - np.min(values))
    unique_exact = int(np.unique(values).size)
    unique_under_tolerance = int(np.unique(
        np.round(values.astype(np.float64) / tolerance)).size)
    return {
        "dtype": str(dtype),
        "dtype_eps": eps,
        "observed_max_abs": observed_max_abs,
        "near_constant_tolerance": tolerance,
        "tolerance_derivation":
            "finfo(dtype).eps * max(1.0, observed_max_abs)",
        "unique_action_count_exact": unique_exact,
        "unique_action_count_under_tolerance": unique_under_tolerance,
        "peak_to_peak": peak_to_peak,
        "exact_constant": bool(unique_exact == 1),
        "near_constant": bool(peak_to_peak <= tolerance),
        "classification_rule": (
            "exact_constant iff one bitwise-unique action;"
            " near_constant iff peak_to_peak <="
            " near_constant_tolerance"),
    }


def _probe_protection_facts(summary: Any) -> Dict[str, Any]:
    """Bind protected-entry facts out of an existing normal handoff
    probe summary; typed ``None`` when the summary carries none."""
    diagnostics = (summary or {}).get("execution_diagnostics") \
        if isinstance(summary, dict) else None
    if not isinstance(diagnostics, dict):
        return {"available": False, "protected_entries": None,
                "protected_entry_rejections": None}
    entry_keys = ("protected_market_entries", "protected_limit_entries",
                  "protected_stop_entries")
    seen = [key for key in entry_keys if key in diagnostics]
    rejections = diagnostics.get("protected_entry_rejections")
    return {
        "available": True,
        "protected_entries": (
            sum(int(diagnostics[key] or 0) for key in seen)
            if seen else None),
        "protected_entry_rejections": (
            None if rejections is None else int(rejections)),
    }


def _context_separation_facts(rollout: Dict[str, Any]) -> Dict[str, Any]:
    """The causal-context separation this rollout actually performed —
    recorded on EVERY per-split block, available or not, so an auditor
    can read the two populations directly (finding 231 requirement 2)."""
    return {
        "nested_role": rollout.get("nested_role"),
        "context_rows_declared": int(rollout["context_rows_declared"]),
        "context_prefix_steps": int(rollout["context_prefix_steps"]),
        "scored_steps": int(rollout["scored_steps"]),
        "total_env_steps": int(rollout["total_env_steps"]),
        "evidence_population": EVIDENCE_POPULATION,
    }


def _handoff_split_evidence(
        rollout: Dict[str, Any],
        phase1_threshold: float,
        phase2_threshold: float) -> tuple:
    """Per-split evidence block; returns ``(evidence_values, block)``
    where ``evidence_values`` is ``None`` when the split is unusable.

    ``rollout['values']`` carries the SCORED raw actions only: the
    context-prefix actions were separated by `_handoff_action_rollout`
    before this function saw anything, so every statistic below — and
    ``action_vector_sha256`` in particular — is computed over the scored
    interval alone."""
    separation = _context_separation_facts(rollout)
    values = rollout["values"]
    if values.size == 0:
        return None, {
            "available": False,
            **separation,
            "error": "deterministic rollout produced zero scored"
                     " observations",
        }
    if np.issubdtype(values.dtype, np.floating):
        evidence_values = values
    else:
        evidence_values = values.astype(np.float32)
    non_finite = int(np.count_nonzero(~np.isfinite(evidence_values)))
    if non_finite:
        return None, {
            "available": False,
            "observation_count": int(values.size),
            **separation,
            "non_finite_actions": non_finite,
            "error": f"{non_finite} non-finite raw actions",
        }
    abs_values = np.abs(evidence_values)
    block = {
        "available": True,
        "observation_count": int(values.size),
        **separation,
        "raw_action_dim": int(rollout["raw_action_dim"]),
        "raw_action_dtype": str(rollout["raw_dtype"]),
        "evidence_dtype": str(evidence_values.dtype),
        "non_finite_actions": 0,
        "action_vector_sha256": hashlib.sha256(
            np.ascontiguousarray(evidence_values).tobytes()).hexdigest(),
        "raw_action": {
            "min": float(np.min(evidence_values)),
            "max": float(np.max(evidence_values)),
            "mean": float(np.mean(evidence_values)),
            "std": float(np.std(evidence_values)),
        },
        "abs_action_quantiles": {
            f"p{int(q * 100):02d}": float(np.quantile(abs_values, q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "abs_action_iqr": float(
            np.quantile(abs_values, 0.75) - np.quantile(abs_values, 0.25)),
        "fraction_non_hold_phase1_threshold": _non_hold_fraction(
            evidence_values, phase1_threshold),
        "fraction_non_hold_phase2_threshold": _non_hold_fraction(
            evidence_values, phase2_threshold),
        "fraction_abs_ge_phase2_threshold": float(
            np.mean(abs_values >= float(phase2_threshold))),
        "mapped_counts_phase1_threshold": _mapped_direction_counts(
            evidence_values, phase1_threshold),
        "mapped_counts_phase2_threshold": _mapped_direction_counts(
            evidence_values, phase2_threshold),
        "constant_policy_classification":
            _dtype_constant_classification(evidence_values),
        # AUD-P1LR-20260815-235: WHY the action distribution looks the
        # way it does. A constant handoff action with a dead first layer
        # is a different fact from a constant action with a live one.
        "actor_liveness": rollout.get("actor_liveness"),
    }
    return evidence_values, block


def _assert_handoff_evidence_invariants(
        evidence: Dict[str, Any]) -> None:
    """Source assertions (finding 221): epoch-zero anchor telemetry can
    never be classified as a trained treatment, and the viability label
    stays inside its typed enum."""
    if evidence["handoff_viability"] not in HANDOFF_VIABILITY_VALUES:
        raise RuntimeError(
            f"unknown handoff_viability"
            f" {evidence['handoff_viability']!r}; expected one of"
            f" {list(HANDOFF_VIABILITY_VALUES)}")
    epoch = int(evidence["epoch"])
    provenance = evidence["policy_provenance"]
    trained = bool(evidence["trained_treatment"])
    if epoch == 0:
        if provenance != PROVENANCE_ANCHOR_PASSTHROUGH or trained:
            raise RuntimeError(
                "epoch-0 anchor telemetry must carry the"
                " anchor_passthrough marker and can never be a trained"
                " treatment (finding 221)")
        if bool(evidence["viable_as_trained_treatment"]):
            raise RuntimeError(
                "epoch-0 anchor telemetry can never be handoff-viable"
                " AS A TRAINED TREATMENT (finding 221)")
    elif provenance != PROVENANCE_TRAINED_EPOCH or not trained:
        raise RuntimeError(
            f"trained epoch {epoch} must carry trained_epoch"
            " provenance and trained_treatment=True")
    if bool(evidence["viable_as_trained_treatment"]) != bool(
            trained
            and evidence["handoff_viability"] == VIABILITY_VIABLE):
        raise RuntimeError(
            "viable_as_trained_treatment must equal"
            " (trained_treatment AND handoff_viability == VIABLE)")
    # Finding 231: a usable split block measures the SCORED interval and
    # nothing else. The observation count is the scored count, and the
    # separated prefix is exactly the count the manifest declared.
    for name, block in (evidence.get("splits") or {}).items():
        if not isinstance(block, dict) or not block.get("available"):
            continue
        declared = int(block["context_rows_declared"])
        separated = int(block["context_prefix_steps"])
        scored = int(block["scored_steps"])
        if separated != declared:
            raise RuntimeError(
                f"{name}: the manifest declares {declared} causal"
                f" context rows but the evidence rollout separated"
                f" {separated}")
        if int(block["observation_count"]) != scored:
            raise RuntimeError(
                f"{name}: evidence observation_count"
                f" {block['observation_count']} != scored_steps"
                f" {scored} — the action evidence must measure the"
                " scored interval only (finding 231)")


def _assert_selected_handoff_invariants(
        selected: Dict[str, Any]) -> None:
    """Source assertions on the SELECTED handoff representation (order
    WP3 §7): a diagnostic terminal fallback and the v3 epoch-0 anchor
    path may hand off, but neither is ever represented as a selected
    viable trained handoff."""
    if selected["handoff_viability"] not in HANDOFF_VIABILITY_VALUES:
        raise RuntimeError(
            "selected handoff carries an unknown viability label"
            f" {selected['handoff_viability']!r}")
    if selected["anchor_passthrough"] and selected["trained_treatment"]:
        raise RuntimeError(
            "an anchor passthrough can never be represented as a"
            " trained treatment (finding 221)")
    if selected["selected_as_viable_handoff"]:
        if selected["selection_is_diagnostic_fallback"]:
            raise RuntimeError(
                "a diagnostic terminal fallback must never be"
                " represented as a selected viable handoff (order WP3)")
        if selected["anchor_passthrough"] or \
                not selected["trained_treatment"]:
            raise RuntimeError(
                "only a TRAINED epoch may be represented as a selected"
                " viable handoff; the v3 epoch-0 anchor path is"
                " anchor_passthrough (finding 221)")
        if selected["handoff_viability"] != VIABILITY_VIABLE:
            raise RuntimeError(
                "selected_as_viable_handoff requires"
                " handoff_viability == VIABLE")


class PipelinePlugin(ValidationPipelinePlugin):
    plugin_params = {
        **ValidationPipelinePlugin.plugin_params,
        "easy_epoch_timesteps": None,     # default: epoch_timesteps
        "easy_max_epochs": 4,             # declared maximum budget
        "easy_patience": 2,               # early stop (budget control only)
        "easy_patience_start_epoch": 1,
        "easy_min_delta": 0.0,
        # Easy removes the action deadband and uses the existing strictly
        # positive easy-floor execution costs. Normal evaluation restores
        # the candidate's original threshold and cost contract.
        "easy_continuous_action_threshold": 0.0,
        "easy_commission_fraction_per_side": 0.00005,
        "easy_full_spread_rate": 0.0001,
        "easy_slippage_bps_per_side": 0.25,
        "easy_min_trades": 1,
        "phase1_handoff_semantics": HANDOFF_SEMANTICS_L1_V4,
    }

    plugin_debug_vars = [
        *ValidationPipelinePlugin.plugin_debug_vars,
        "easy_epoch_timesteps", "easy_max_epochs", "easy_patience",
        "easy_patience_start_epoch", "easy_min_delta",
        "easy_continuous_action_threshold",
        "easy_commission_fraction_per_side", "easy_full_spread_rate",
        "easy_slippage_bps_per_side", "easy_min_trades",
        "phase1_handoff_semantics",
    ]

    def _easy_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        easy = dict(config)
        threshold = float(easy.get(
            "easy_continuous_action_threshold",
            self.params["easy_continuous_action_threshold"],
        ))
        commission = float(easy.get(
            "easy_commission_fraction_per_side",
            self.params["easy_commission_fraction_per_side"],
        ))
        full_spread = float(easy.get(
            "easy_full_spread_rate", self.params["easy_full_spread_rate"]
        ))
        slippage_bps = float(easy.get(
            "easy_slippage_bps_per_side",
            self.params["easy_slippage_bps_per_side"],
        ))
        minimum_trades = int(easy.get(
            "easy_min_trades", self.params["easy_min_trades"]
        ))
        if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ValueError(
                "easy_continuous_action_threshold must be finite in [0, 1)"
            )
        if (
            not all(math.isfinite(value) for value in (
                commission, full_spread, slippage_bps
            ))
            or min(commission, full_spread, slippage_bps) <= 0.0
        ):
            raise ValueError(
                "easy execution costs must all be finite and strictly positive"
            )
        if minimum_trades < 1:
            raise ValueError("easy_min_trades must be >= 1")

        easy.update({
            "solvency_mode": EASY_MODE,
            "env_mode": "training",
            "continuous_action_threshold": threshold,
            "commission": commission,
            "full_spread_rate": full_spread,
            "slippage": full_spread / 2.0 + slippage_bps / 10_000.0,
            "easy_min_trades": minimum_trades,
        })
        # M0 order §8.1: the easy phase keeps its own learning rate while
        # the NORMAL fine-tune rate varies per arm. Without this
        # override, a reduced normal LR would silently also slow the
        # easy phase and confound the mechanism screen.
        easy_lr = easy.get("easy_learning_rate")
        if easy_lr is not None:
            if isinstance(easy_lr, bool) or not isinstance(
                easy_lr, (int, float)
            ):
                raise ValueError("easy_learning_rate must be a number")
            easy_lr = float(easy_lr)
            if not math.isfinite(easy_lr) or easy_lr <= 0.0:
                raise ValueError(
                    "easy_learning_rate must be finite and strictly"
                    " positive"
                )
            easy["learning_rate"] = easy_lr
        return easy

    # ------------------------------------------------------------------
    def _context_aware_split_env(
        self,
        env_plugin_name: str,
        config: Dict[str, Any],
        csv_path: str,
        agent_plugin,
    ) -> tuple:
        """Build one evaluation env through the SAME manifest-verified,
        context-aware path as the corrected executing selector
        (AUD-F1-20260812-231).

        Returns ``(plug, env, role, context_rows)``. The role and its
        ``context_rows`` come from the parent's VERIFIED nested split
        manifest via `_resolve_nested_role` — never from the file name,
        never from a row position. A role that DECLARES causal context
        is built with the ContextPrefixWrapper installed by
        `_make_split_env`, exactly as `_eval_on_split` does. A role that
        declares none — and any legacy run with no nested manifest at
        all — keeps the historical four-argument factory call and is
        unchanged; nothing is ever inherited silently.
        """
        nested = self._resolve_nested_role(config, csv_path)
        role = nested[0] if nested else None
        context_rows = int(nested[1]["context_rows"]) if nested else 0
        if context_rows:
            plug, env = self._make_split_env(
                env_plugin_name, config, csv_path, agent_plugin,
                context_rows=context_rows,
            )
        else:
            plug, env = self._make_split_env(
                env_plugin_name, config, csv_path, agent_plugin)
        return plug, env, role, context_rows

    @staticmethod
    def _assert_context_separation(
        *,
        split: str,
        declared: int,
        separated: int,
        scored: int,
    ) -> None:
        """Fail-closed backstop shared by the evidence rollouts: a role
        that declares causal context must have had its prefix separated
        by the wrapper, and must reach at least one scored row."""
        if declared and separated != declared:
            raise ValueError(
                f"{split}: the manifest declares {declared} causal"
                f" context rows but the rollout separated {separated}"
                " — the env was not wrapped, or the episode ended"
                " inside the prefix; refusing to measure it")
        if declared and scored == 0:
            raise ValueError(
                f"{split}: the episode produced no scored rows after"
                " its causal context prefix")

    # ------------------------------------------------------------------
    def _easy_probe(
        self,
        env_plugin_name: str,
        easy_config: Dict[str, Any],
        csv_path: str,
        agent_plugin,
        model,
        seed: int,
    ) -> Dict[str, Any]:
        """One deterministic rollout on a FRESH easy env; returns the
        ECONOMIC outcome (operational equity minus recapitalization debt)
        plus solvency event counters. Used exclusively for the easy
        phase's early-stopping budget — never for selection.

        The probe runs on the ``fit_train`` role, which the nested
        manifest declares with ZERO context rows (fit training consumes
        its own leading bars), so it is built unwrapped exactly as
        before. It is nonetheless routed through the same
        manifest-verified constructor so the zero is a VERIFIED fact
        rather than an assumption, and the separation counts it observed
        are recorded on the checkpoint row."""
        plug, env, role, context_rows = self._context_aware_split_env(
            env_plugin_name, easy_config, csv_path, agent_plugin)
        try:
            obs, _info = env.reset(seed=seed)
            terminated = truncated = False
            last_info: Dict[str, Any] = {}
            context_prefix_steps = 0
            scored_steps = 0
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, last_info = env.step(
                    action)
                if bool(last_info.get("is_context_prefix")):
                    context_prefix_steps += 1
                else:
                    scored_steps += 1
            self._assert_context_separation(
                split="easy_probe", declared=context_rows,
                separated=context_prefix_steps, scored=scored_steps)
            action_diagnostics = dict(
                last_info.get("action_diagnostics") or {}
            )
            execution_diagnostics = dict(
                last_info.get("execution_diagnostics") or {}
            )
            return {
                "economic_equity": float(
                    last_info.get("economic_equity", float("nan"))),
                "recapitalization_debt": float(
                    last_info.get("recapitalization_debt", 0.0)),
                "recapitalization_count": int(
                    last_info.get("recapitalization_count", 0)),
                "would_margin_call_count": int(
                    last_info.get("would_margin_call_count", 0)),
                "termination_cause": last_info.get("termination_cause"),
                "trades_total": int(last_info.get("trades", 0) or 0),
                "non_hold_actions": int(
                    action_diagnostics.get("non_hold_actions", 0) or 0
                ),
                "entry_actions_seen": int(
                    execution_diagnostics.get("entry_actions_seen", 0) or 0
                ),
                "entry_orders_submitted": int(
                    execution_diagnostics.get(
                        "entry_orders_submitted", 0
                    ) or 0
                ),
                "protected_entry_rejections": int(
                    execution_diagnostics.get(
                        "protected_entry_rejections", 0
                    ) or 0
                ),
                "action_diagnostics": action_diagnostics,
                "execution_diagnostics": execution_diagnostics,
                "nested_role": role,
                "context_rows_declared": int(context_rows),
                "context_prefix_steps": int(context_prefix_steps),
                "scored_steps": int(scored_steps),
            }
        finally:
            try:
                plug.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _handoff_action_rollout(
        self,
        env_plugin_name: str,
        config: Dict[str, Any],
        csv_path: str,
        agent_plugin,
        model,
        seed: int,
    ) -> Dict[str, Any]:
        """One deterministic rollout on a fresh env capturing the RAW
        policy action at every SCORED step — the first action component,
        exactly the value gym-fx maps through the deadband — BEFORE the
        env applies any threshold.

        AUD-F1-20260812-231: train_monitor and inner_validation declare a
        causal-context prefix, and the actions the policy takes on those
        rows are input to the observation window, not measurement of the
        scored interval. The env is therefore built through
        `_context_aware_split_env` — the same manifest-verified,
        wrapper-installing path the executing selector uses — and every
        step tagged ``is_context_prefix`` is SEPARATED here: it is forced
        to hold by the wrapper and its raw action never enters the
        returned vector, so it cannot reach the distribution, the
        quantiles, the mapped counts, the constant-policy call or the
        action-vector sha256.

        Fail-closed: a declared prefix that the rollout did not separate
        (an unwrapped env), a prefix row reported after the score
        boundary, or an episode with no scored row REFUSES. Because the
        only caller wraps this in the typed UNAVAILABLE handler, a
        refusal degrades the evidence — it never gates a phase.
        """
        plug, env, role, context_rows = self._context_aware_split_env(
            env_plugin_name, config, csv_path, agent_plugin)
        try:
            obs, _info = env.reset(seed=seed)
            scored_values = []
            raw_dim = None
            context_prefix_steps = 0
            total_steps = 0
            terminated = truncated = False
            # AUD-P1LR-20260815-235: the phase-1 HANDOFF artifact is the
            # one selection actually promotes, and the one measured at
            # 21/256 live first-layer units. Its observations are already
            # in hand here, so type it before phase 2 inherits it.
            liveness_sampler = _liveness.StridedObservationSampler(
                int(config.get(
                    "actor_liveness_probe_observations",
                    self.params["actor_liveness_probe_observations"],
                ) or 0))
            while not (terminated or truncated):
                liveness_sampler.offer(obs)
                action, _state = model.predict(obs, deterministic=True)
                flat = np.asarray(action).reshape(-1)
                value = flat[0] if flat.size else 0.0
                obs, _reward, terminated, truncated, info = env.step(
                    action)
                total_steps += 1
                if bool(info.get("is_context_prefix")):
                    if scored_values:
                        raise ValueError(
                            "a causal-context row was reported after the"
                            " score boundary — the context prefix must"
                            " precede every scored row")
                    context_prefix_steps += 1
                else:
                    if raw_dim is None:
                        raw_dim = int(flat.size)
                    scored_values.append(value)
                if total_steps > 1_000_000:
                    break
            self._assert_context_separation(
                split="handoff_action_rollout", declared=context_rows,
                separated=context_prefix_steps,
                scored=len(scored_values))
            values = np.asarray(scored_values)
            return {
                "values": values,
                "actor_liveness": _liveness.actor_liveness_facts(
                    model=model,
                    observations=liveness_sampler.batch(),
                    actions=values if values.size else None,
                    split=str(role or "handoff_action_rollout"),
                    phase=EASY_MODE,
                    min_live_unit_fraction=float(config.get(
                        "actor_liveness_min_live_unit_fraction",
                        self.params[
                            "actor_liveness_min_live_unit_fraction"],
                    )),
                ),
                "raw_action_dim": int(raw_dim or 0),
                "raw_dtype": str(values.dtype),
                "nested_role": role,
                "context_rows_declared": int(context_rows),
                "context_prefix_steps": int(context_prefix_steps),
                "scored_steps": int(len(scored_values)),
                "total_env_steps": int(total_steps),
            }
        finally:
            try:
                plug.close()
            except Exception:
                pass

    def _build_handoff_viability_evidence(
        self,
        *,
        env_plugin_name: str,
        normal_config: Dict[str, Any],
        paths: Dict[str, str],
        agent_plugin,
        model,
        seed: int,
        epoch: int,
        phase1_threshold_raw: Any,
        normal_probe_facts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Typed handoff-viability evidence for ONE phase-1 checkpoint
        (finding 221). Evidence only — never a gate, never a selector;
        any rollout failure lands as a typed UNAVAILABLE record."""
        phase1_threshold, phase1_source = _resolve_action_threshold(
            phase1_threshold_raw)
        phase2_threshold, phase2_source = _resolve_action_threshold(
            normal_config.get("continuous_action_threshold"))
        evidence_config = dict(normal_config)
        evidence_config["solvency_mode"] = NORMAL_MODE
        split_paths = {
            "train_monitor": paths.get("train_tail", paths["train"]),
            "inner_validation": paths["val"],
        }
        splits: Dict[str, Any] = {}
        vectors = []
        for split_name in _HANDOFF_SPLIT_NAMES:
            try:
                rollout = self._handoff_action_rollout(
                    env_plugin_name, evidence_config,
                    split_paths[split_name], agent_plugin, model, seed)
                evidence_values, block = _handoff_split_evidence(
                    rollout, phase1_threshold, phase2_threshold)
                splits[split_name] = block
                if evidence_values is not None:
                    vectors.append(evidence_values)
            except Exception as exc:  # evidence must never gate phase 1
                splits[split_name] = {
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        train_tail_trades = int(normal_probe_facts["train_tail_trades"])
        validation_trades = int(normal_probe_facts["validation_trades"])
        probe_trades_total = train_tail_trades + validation_trades
        usable = [name for name in _HANDOFF_SPLIT_NAMES
                  if splits[name].get("available")]
        reasons = []
        combined_constant = None
        any_cross = False
        if not usable:
            viability = VIABILITY_UNAVAILABLE
            reasons.append(
                "no split produced usable deterministic action evidence")
            reasons.extend(
                f"{name}: {splits[name].get('error', 'unavailable')}"
                for name in _HANDOFF_SPLIT_NAMES)
        else:
            combined_constant = _dtype_constant_classification(
                np.concatenate(vectors))
            any_cross = any(
                splits[name]["fraction_non_hold_phase2_threshold"] > 0.0
                for name in usable)
            if combined_constant["exact_constant"]:
                viability = VIABILITY_CONSTANT_POLICY
                reasons.append(
                    "exactly one bitwise-unique deterministic action"
                    f" across {len(usable)} split(s)")
            elif combined_constant["near_constant"]:
                viability = VIABILITY_CONSTANT_POLICY
                reasons.append(
                    "peak_to_peak="
                    f"{combined_constant['peak_to_peak']!r} <= dtype-"
                    "derived tolerance "
                    f"{combined_constant['near_constant_tolerance']!r}")
            elif not any_cross:
                viability = VIABILITY_BELOW_NORMAL_THRESHOLD
                reasons.append(
                    "max |a|="
                    f"{combined_constant['observed_max_abs']!r} never"
                    " reaches the phase-2 threshold"
                    f" {phase2_threshold!r} on any split")
            elif probe_trades_total == 0:
                viability = VIABILITY_NO_TRADE
                reasons.append(
                    "actions cross the phase-2 threshold but the normal"
                    " handoff probes recorded zero trades (train_tail="
                    f"{train_tail_trades},"
                    f" validation={validation_trades})")
            else:
                viability = VIABILITY_VIABLE
                reasons.append(
                    "non-constant actions cross the phase-2 threshold"
                    " and the normal probes traded (train_tail="
                    f"{train_tail_trades},"
                    f" validation={validation_trades})")
        trained_treatment = int(epoch) > 0
        # Finding 231: the separation is reported at the top of the
        # block too, so an auditor reads "how many rows were context and
        # how many were scored" without opening the per-split blocks.
        causal_context = {
            "context_rows_source":
                "verified nested split manifest role entry"
                " (roles[<role>].context_rows)",
            "separation_mechanism": CONTEXT_SEPARATION_MECHANISM,
            "evidence_population": EVIDENCE_POPULATION,
            "context_prefix_steps_total": sum(
                int(splits[name]["context_prefix_steps"])
                for name in usable),
            "scored_steps_total": sum(
                int(splits[name]["scored_steps"]) for name in usable),
            "roles": {
                name: splits[name].get("nested_role")
                for name in _HANDOFF_SPLIT_NAMES
            },
        }
        evidence = {
            "schema": HANDOFF_VIABILITY_SCHEMA,
            "epoch": int(epoch),
            "policy_provenance": (
                PROVENANCE_TRAINED_EPOCH if trained_treatment
                else PROVENANCE_ANCHOR_PASSTHROUGH),
            "trained_treatment": trained_treatment,
            "deterministic": True,
            "rollout_seed": int(seed),
            "rollout_solvency_mode": NORMAL_MODE,
            "threshold_semantics": (
                "gym_fx_continuous_v1: thr==0 -> non-hold iff |a| > 0;"
                " thr>0 -> non-hold iff |a| >= thr"),
            "threshold_application": (
                "both thresholds evaluated on the SAME raw action"
                " vector captured from one deterministic"
                " normal-realistic rollout per split (WP2 same-vector"
                " design); that vector is the SCORED interval only —"
                " causal-context rows are separated before any"
                " threshold is applied (finding 231)"),
            "phase1_threshold": float(phase1_threshold),
            "phase1_threshold_source": phase1_source,
            "phase2_threshold": float(phase2_threshold),
            "phase2_threshold_source": phase2_source,
            "causal_context": causal_context,
            "splits": splits,
            "normal_probe": {
                "source": "existing normal handoff probe rollouts",
                "train_tail_trades": train_tail_trades,
                "validation_trades": validation_trades,
                "train_tail_protection": _probe_protection_facts(
                    normal_probe_facts.get("train_tail_summary")),
                "validation_protection": _probe_protection_facts(
                    normal_probe_facts.get("validation_summary")),
            },
            "probe_trades_total": probe_trades_total,
            "constant_policy_classification": combined_constant,
            "any_action_crosses_phase2_threshold": bool(any_cross),
            "handoff_viability": viability,
            "classification_reasons": reasons,
            "authority": (
                "evidence only — selection stays with the paired"
                " comparator (v4) / as-run v3 score; this block never"
                " selects"),
            "viable_as_trained_treatment": bool(
                trained_treatment and viability == VIABILITY_VIABLE),
        }
        _assert_handoff_evidence_invariants(evidence)
        return evidence

    def _train_easy_phase(
        self,
        *,
        config: Dict[str, Any],
        env_plugin,
        agent_plugin,
    ) -> Dict[str, Any]:
        """Phase 1: train under easy dynamics, save post_easy immutably."""
        handoff_semantics = str(config.get(
            "phase1_handoff_semantics",
            self.params["phase1_handoff_semantics"]))
        if handoff_semantics not in _HANDOFF_SEMANTICS:
            raise ValueError(
                f"unknown phase1_handoff_semantics "
                f"{handoff_semantics!r}; expected one of "
                f"{sorted(_HANDOFF_SEMANTICS)}")
        phase1_mode = str(config.get("phase1_mode", EASY_MODE))
        if phase1_mode == NORMAL_MODE:
            # matched-boundary NORMAL phase 1 (doc 38 §5.3): identical
            # phase structure, selection and handoff machinery; only the
            # solvency dynamics differ from the easy arm.
            easy_config = dict(config)
            easy_config["solvency_mode"] = NORMAL_MODE
            easy_config["env_mode"] = "training"
            phase1_lr = easy_config.get("phase1_learning_rate",
                                        easy_config.get("easy_learning_rate"))
            if phase1_lr is not None:
                easy_config["learning_rate"] = float(phase1_lr)
            easy_config["easy_min_trades"] = int(
                easy_config.get("easy_min_trades",
                                self.params["easy_min_trades"]))
        elif phase1_mode == EASY_MODE:
            easy_config = self._easy_training_config(config)
        else:
            raise ValueError(f"unknown phase1_mode {phase1_mode!r}")
        env_plugin_name = easy_config.get("env_plugin", "gym_fx_env")
        paths = self._split_csv(easy_config)
        # FIT TRAINING env: built unwrapped, deliberately. fit_train
        # declares zero context rows in the nested manifest (it consumes
        # its own leading bars while learning), and the ContextPrefixWrapper
        # is an EVALUATION boundary — it must never force holds during
        # training (finding 231; parent `_make_split_env` contract).
        _plug, easy_env = self._make_split_env(
            env_plugin_name, easy_config, paths["train"], agent_plugin)
        try:
            warm_start_model = easy_config.get("warm_start_model")
            if warm_start_model:
                warm_start_path = Path(str(warm_start_model))
                if not warm_start_path.exists():
                    raise FileNotFoundError(
                        f"warm_start_model not found: {warm_start_path}"
                    )
                _verify_artifact_sha256(
                    warm_start_path,
                    easy_config.get("warm_start_model_sha256"),
                )
                training_loader = getattr(
                    agent_plugin, "load_for_training", None
                )
                if callable(training_loader):
                    model = training_loader(
                        str(warm_start_path), easy_env, easy_config
                    )
                else:
                    model = agent_plugin.load(str(warm_start_path), easy_env)
                try:
                    model.set_env(easy_env)
                except Exception:
                    pass
            else:
                model = agent_plugin.build(easy_env, easy_config)
            # AUD-F1-20260808-160: weight change is proved from the
            # canonical policy tensor digest, never from archive bytes.
            anchor_tensor_sha = _policy_tensor_hash(model.policy)
            epoch_ts = int(
                easy_config.get("easy_epoch_timesteps")
                or easy_config.get("epoch_timesteps",
                                   self.params["epoch_timesteps"]))
            max_epochs = int(easy_config.get(
                "easy_max_epochs", self.params["easy_max_epochs"]))
            patience = int(easy_config.get(
                "easy_patience", self.params["easy_patience"]))
            patience_start_epoch = max(1, int(easy_config.get(
                "easy_patience_start_epoch",
                self.params["easy_patience_start_epoch"])))
            min_delta = float(easy_config.get(
                "easy_min_delta", self.params["easy_min_delta"]))
            seed = int(easy_config.get("eval_seed",
                                       self.params["eval_seed"]))
            minimum_trades = int(easy_config["easy_min_trades"])
            best = -math.inf
            best_epoch = None
            waited = 0
            last_improvement_epoch = None
            stop_reason = "max_epochs_budget"
            history = []
            save_model = str(config.get("save_model")
                             or "./agent_model.zip")
            post_easy_path = Path(save_model).with_suffix("")
            post_easy_path = post_easy_path.parent / (
                post_easy_path.name + ".post_easy.zip")
            post_easy_path.parent.mkdir(parents=True, exist_ok=True)

            def evaluate_checkpoint(epoch: int, source: str) -> None:
                nonlocal best, best_epoch, waited, last_improvement_epoch
                probe = self._easy_probe(
                    env_plugin_name, easy_config, paths["train"],
                    agent_plugin, model, seed)
                probe["epoch"] = epoch
                probe["checkpoint_source"] = source
                probe["easy_activity_eligible"] = bool(
                    probe["trades_total"] >= minimum_trades
                    and probe["non_hold_actions"] > 0
                    and probe["entry_actions_seen"] > 0
                    and probe["entry_orders_submitted"] > 0
                    and probe["protected_entry_rejections"] == 0
                )
                normal_config = dict(config)
                normal_config["solvency_mode"] = NORMAL_MODE
                normal_train_tail = self._eval_on_split(
                    env_plugin_name,
                    normal_config,
                    paths.get("train_tail", paths["train"]),
                    agent_plugin,
                    model,
                    seed,
                    "train_tail_epoch",
                )
                normal_validation = self._eval_on_split(
                    env_plugin_name,
                    normal_config,
                    paths["val"],
                    agent_plugin,
                    model,
                    seed,
                    "validation_epoch",
                )
                normal_default_min = int(normal_config.get(
                    "early_stop_min_trades", 1
                ))
                configured_train_tail_min = normal_config.get(
                    "early_stop_min_train_tail_trades"
                )
                configured_validation_min = normal_config.get(
                    "early_stop_min_validation_trades"
                )
                normal_min_train_tail = int(
                    normal_default_min
                    if configured_train_tail_min is None
                    else configured_train_tail_min
                )
                normal_min_validation = int(
                    normal_default_min
                    if configured_validation_min is None
                    else configured_validation_min
                )
                normal_train_tail_trades = int(
                    normal_train_tail.get("trades_total", 0) or 0
                )
                normal_validation_trades = int(
                    normal_validation.get("trades_total", 0) or 0
                )
                probe["normal_handoff_probe"] = {
                    "train_tail_trades": normal_train_tail_trades,
                    "validation_trades": normal_validation_trades,
                    "minimum_train_tail_trades": normal_min_train_tail,
                    "minimum_validation_trades": normal_min_validation,
                }
                normal_handoff_eligible = bool(
                    normal_train_tail_trades >= normal_min_train_tail
                    and normal_validation_trades >= normal_min_validation
                )
                # Finding 221 (order WP3 §7): typed handoff-viability
                # evidence on EVERY phase-1 checkpoint record — both
                # thresholds on the same raw action vector, the
                # existing normal-probe facts bound in. Evidence only;
                # it never gates and never selects.
                probe["handoff_viability_evidence"] = (
                    self._build_handoff_viability_evidence(
                        env_plugin_name=env_plugin_name,
                        normal_config=normal_config,
                        paths=paths,
                        agent_plugin=agent_plugin,
                        model=model,
                        seed=seed,
                        epoch=epoch,
                        phase1_threshold_raw=easy_config.get(
                            "continuous_action_threshold"),
                        normal_probe_facts={
                            "train_tail_trades": normal_train_tail_trades,
                            "validation_trades": normal_validation_trades,
                            "train_tail_summary": normal_train_tail,
                            "validation_summary": normal_validation,
                        },
                    ))
                if handoff_semantics == HANDOFF_SEMANTICS_M0_V3:
                    # Mechanism-ladder D0 (finding 220): the boundary AS
                    # THE M0 SCREEN RAN IT — the normal probe GATES, the
                    # score is the easy-probe economic equity, and the
                    # epoch-0 warm-start baseline is save-eligible. This
                    # reproduces post_easy.v3 selection (observed:
                    # best_easy_epoch=0 on the active M0 arm).
                    probe["normal_handoff_probe_telemetry_only"] = False
                    probe["normal_handoff_eligible"] = (
                        normal_handoff_eligible)
                    probe["activity_eligible"] = bool(
                        probe["easy_activity_eligible"]
                        and normal_handoff_eligible)
                    probe["handoff_eligible"] = True
                    score = probe["economic_equity"]
                    improved = False
                    if (
                        probe["activity_eligible"]
                        and math.isfinite(score)
                        and score > best + min_delta
                    ):
                        best = score
                        best_epoch = epoch
                        waited = 0
                        last_improvement_epoch = epoch
                        improved = True
                        agent_plugin.save(model, str(post_easy_path))
                    elif (best_epoch is not None
                          and epoch > patience_start_epoch):
                        waited += 1
                    probe["checkpoint_improved"] = improved
                    probe["early_stop_patience_used"] = waited
                    history.append(probe)
                    return
                probe["normal_handoff_probe_telemetry_only"] = True
                # AUD-F1-20260808-159: the normal probe is TELEMETRY.
                # It never gates whether the easy treatment exists.
                probe["normal_handoff_eligible_telemetry"] = (
                    normal_handoff_eligible)
                # checkpoint A: easy utility on the monitor year;
                # checkpoint B: normal-realistic inner validation
                easy_monitor = self._eval_on_split(
                    env_plugin_name, easy_config,
                    paths.get("train_tail", paths["train"]),
                    agent_plugin, model, seed, "easy_monitor_epoch")
                easy_monitor_return = easy_monitor.get("total_return")
                probe["phase1_monitor_total_return"] = easy_monitor_return
                probe["phase1_monitor_positive_return"] = bool(
                    isinstance(easy_monitor_return, (int, float))
                    and math.isfinite(float(easy_monitor_return))
                    and float(easy_monitor_return) > 0.0)
                paired = _paired.paired_generalization_weekly_v1(
                    easy_monitor, normal_validation,
                    beta=float(easy_config.get(
                        "l1_gap_penalty_beta", 0.25)),
                    label_a="easy_train_monitor",
                    label_b="normal_inner_validation",
                    candidate_id=f"easy_epoch_{epoch}")
                probe["paired_selection"] = {
                    "eligible": paired["eligible"],
                    "paired_score": paired["paired_score"],
                    "reasons": paired["ineligibility_reasons"],
                }
                probe["handoff_eligible"] = epoch > 0
                # epoch 0 is baseline telemetry and is STRUCTURALLY
                # ineligible as a treatment handoff (order §7.5)
                improved = False
                if (
                    epoch > 0
                    and paired["eligible"]
                    and paired["paired_score"] is not None
                    and paired["paired_score"] > best + min_delta
                ):
                    best = paired["paired_score"]
                    best_epoch = epoch
                    waited = 0
                    last_improvement_epoch = epoch
                    improved = True
                    agent_plugin.save(model, str(post_easy_path))
                elif (best_epoch is not None
                      and epoch > patience_start_epoch):
                    waited += 1
                probe["checkpoint_improved"] = improved
                probe["early_stop_patience_used"] = waited
                history.append(probe)

            if warm_start_model:
                evaluate_checkpoint(0, "warm_start_baseline")

            for epoch in range(1, max_epochs + 1):
                model.learn(total_timesteps=epoch_ts,
                            reset_num_timesteps=False)
                evaluate_checkpoint(epoch, "easy_training_epoch")
                if (best_epoch is not None
                        and epoch > patience_start_epoch
                        and waited >= patience):
                    stop_reason = "easy_early_stop"
                    break

            trained_epochs = [row for row in history
                              if row.get("epoch", 0) > 0]
            if not trained_epochs:
                raise RuntimeError(
                    "phase 1 trained zero epochs; a declared phase-1 arm"
                    " cannot hand off (invariant 1)")
            selection_basis = "paired_comparator_best_trained_epoch"
            if best_epoch is None or not post_easy_path.exists():
                if handoff_semantics == HANDOFF_SEMANTICS_M0_V3:
                    # v3 as-run had NO terminal fallback: a phase 1 with
                    # no easy-and-normal activity-eligible checkpoint
                    # refused the handoff outright.
                    raise RuntimeError(
                        "easy curriculum produced no easy-and-normal "
                        "activity-eligible checkpoint under the as-run "
                        "M0 v3 handoff semantics: required trades>="
                        f"{minimum_trades}, non-hold actions, entry "
                        "actions, submitted entries, zero protected-"
                        "entry rejections in easy; train-tail and "
                        "validation trade gates must also pass in "
                        "normal")
                # invariant 3: a failed/inactive phase-1 result is still
                # handed off and measured — the TERMINAL trained epoch,
                # never the anchor.
                best_epoch = trained_epochs[-1]["epoch"]
                selection_basis = "terminal_trained_epoch_fallback"
                agent_plugin.save(model, str(post_easy_path))
            elif handoff_semantics == HANDOFF_SEMANTICS_M0_V3:
                selection_basis = "m0_v3_economic_equity_epoch0_eligible"
            updates_after = int(getattr(model, "_n_updates", 0) or 0)
            if updates_after <= 0:
                raise RuntimeError(
                    "phase 1 applied zero gradient updates; refusing a"
                    " sham handoff")
            terminal_tensor_sha = _policy_tensor_hash(model.policy)
            if terminal_tensor_sha == anchor_tensor_sha:
                raise RuntimeError(
                    "phase-1 final policy is tensor-identical to the"
                    " anchor; archive-byte difference is not weight"
                    " change (AUD-F1-20260808-160)")
            post_easy_sha = _verify_artifact_sha256(post_easy_path, None)
            selected_rows = [row for row in history
                             if int(row.get("epoch", -1))
                             == int(best_epoch)]
            if len(selected_rows) != 1:
                raise RuntimeError(
                    "expected exactly one checkpoint record for the"
                    f" selected epoch {best_epoch}; found"
                    f" {len(selected_rows)}")
            selected_evidence = selected_rows[0][
                "handoff_viability_evidence"]
            selection_is_fallback = (
                selection_basis == "terminal_trained_epoch_fallback")
            selected_handoff = {
                "schema": SELECTED_HANDOFF_SCHEMA,
                "best_easy_epoch": int(best_epoch),
                "selection_basis": selection_basis,
                "phase1_handoff_semantics": handoff_semantics,
                "handoff_viability":
                    selected_evidence["handoff_viability"],
                "policy_provenance":
                    selected_evidence["policy_provenance"],
                "trained_treatment": bool(
                    selected_evidence["trained_treatment"]),
                "anchor_passthrough": (
                    selected_evidence["policy_provenance"]
                    == PROVENANCE_ANCHOR_PASSTHROUGH),
                "selection_is_diagnostic_fallback":
                    selection_is_fallback,
                # Order WP3 §7: a diagnostic terminal fallback may hand
                # off an INACTIVE record but is NEVER represented as a
                # selected viable handoff; the v3 epoch-0 anchor path
                # is anchor_passthrough, never a trained treatment.
                "selected_as_viable_handoff": bool(
                    not selection_is_fallback
                    and selected_evidence["trained_treatment"]
                    and selected_evidence["handoff_viability"]
                    == VIABILITY_VIABLE),
                "viability_is_selection_authority": False,
            }
            _assert_selected_handoff_invariants(selected_handoff)
            meta = {
                "schema": "agent_multi.solvency_curriculum.post_easy.v4",
                "anchor_policy_tensor_sha256": anchor_tensor_sha,
                "phase1_terminal_policy_tensor_sha256": terminal_tensor_sha,
                "phase1_gradient_updates": updates_after,
                "selection_basis": selection_basis,
                "phase1_handoff_semantics": handoff_semantics,
                "epoch0_structurally_ineligible": (
                    handoff_semantics != HANDOFF_SEMANTICS_M0_V3),
                "normal_probe_is_telemetry_only": (
                    handoff_semantics != HANDOFF_SEMANTICS_M0_V3),
                "artifact": str(post_easy_path),
                "artifact_sha256": post_easy_sha,
                # Finding 195: the persisted mode is the mode that RAN.
                # Normal phase-1 records say normal; only easy says easy.
                "solvency_mode": (EASY_MODE
                                  if phase1_mode == EASY_MODE
                                  else NORMAL_MODE),
                "best_easy_epoch": best_epoch,
                "phase1_stop_reason": stop_reason,
                "phase1_stopped_epoch": int(trained_epochs[-1]["epoch"]),
                "phase1_last_improvement_epoch": last_improvement_epoch,
                "phase1_patience": patience,
                "phase1_patience_start_epoch": patience_start_epoch,
                "phase1_patience_used": waited,
                "phase1_min_delta": min_delta,
                "phase1_eligible_checkpoint_count": sum(
                    1 for h in trained_epochs
                    if bool((h.get("paired_selection") or {}).get(
                        "eligible")
                    if handoff_semantics != HANDOFF_SEMANTICS_M0_V3
                    else h.get("activity_eligible"))),
                "phase1_positive_monitor_epoch_count": sum(
                    1 for h in trained_epochs
                    if h.get("phase1_monitor_positive_return") is True),
                "phase1_first_positive_monitor_epoch": next(
                    (int(h["epoch"]) for h in trained_epochs
                     if h.get("phase1_monitor_positive_return") is True),
                    None),
                # Finding 200: trained epochs are epoch > 0 ONLY; the
                # epoch-0 baseline evaluation is telemetry, counted
                # separately. The legacy alias carries the SAME truthful
                # trained count (aliases never lie).
                "phase1_epochs_run": sum(
                    1 for h in history if int(h.get("epoch", 0)) > 0),
                "phase1_baseline_evaluations": sum(
                    1 for h in history if int(h.get("epoch", 0)) == 0),
                "easy_epochs_run": sum(
                    1 for h in history if int(h.get("epoch", 0)) > 0),
                "easy_budget_epochs": max_epochs,
                "easy_epoch_timesteps": epoch_ts,
                "activity_contract": {
                    "minimum_trades": minimum_trades,
                    "requires_non_hold_action": True,
                    "requires_entry_action": True,
                    "requires_submitted_entry": True,
                    "maximum_protected_entry_rejections": 0,
                    # Under the corrected v4 boundary the normal-handoff
                    # probe is telemetry ONLY (finding 160/195); under
                    # the as-run M0 v3 reproduction it gates, and these
                    # facts say which one actually ran.
                    "normal_handoff_probe_is_telemetry_only": (
                        handoff_semantics != HANDOFF_SEMANTICS_M0_V3),
                    "normal_handoff_activity_gates_selection": (
                        handoff_semantics == HANDOFF_SEMANTICS_M0_V3),
                },
                "phase1_mode": phase1_mode,
                "selected_handoff_viability": selected_handoff,
                "phase1_difficulty": {
                    # easy-relaxation fields; None under normal phase-1
                    # (the candidate's own contract governs there)
                    "continuous_action_threshold": easy_config.get(
                        "continuous_action_threshold"),
                    "commission_fraction_per_side": easy_config.get(
                        "commission"),
                    "full_spread_rate": easy_config.get("full_spread_rate"),
                    "slippage_rate_per_side": easy_config.get("slippage"),
                },
                "history": history,
            }
            meta_path = Path(str(post_easy_path) + ".meta.json")
            meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True),
                                 encoding="utf-8")
            return {"path": str(post_easy_path), "sha256": post_easy_sha,
                    "meta": meta}
        finally:
            try:
                easy_env.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        *,
        config: Dict[str, Any],
        env_plugin,
        agent_plugin,
        mode: str = "train",
    ) -> Dict[str, Any]:
        # AUD-P1LR-20260815-235: phase 1 trains from THIS config, not
        # from the one the parent pipeline later binds, so the program's
        # observation contract has to reach the candidate here or the
        # easy phase would still build the raw price window.
        config, _observation_contract_application = (
            apply_observation_contract(config))
        if str(mode).lower() != "train" or not bool(
                config.get("solvency_curriculum_enabled", True)):
            passthrough = dict(config)
            passthrough["solvency_mode"] = NORMAL_MODE
            return super().run_pipeline(
                config=passthrough, env_plugin=env_plugin,
                agent_plugin=agent_plugin, mode=mode)

        if bool(config.get("require_constant_lr_across_phases", False)):
            lr_values = {
                float(config.get("learning_rate")),
                float(config.get("phase1_learning_rate")),
                float(config.get("easy_learning_rate")),
            }
            if len(lr_values) != 1:
                raise ValueError(
                    "causal difficulty contrast requires one constant "
                    "learning rate across phase 1 and phase 2; got "
                    f"{sorted(lr_values)}")

        ledger = config.get("total_max_passes")
        if ledger is not None:
            total_max = int(ledger)
            phase1 = int(config.get("easy_max_epochs", 0))
            phase2 = int(config.get("max_epochs", 0))
            fraction = float(config.get("phase1_max_fraction", 0.5))
            normal_min = int(config.get("normal_phase_min_passes", 1))
            if phase1 + phase2 > total_max:
                raise ValueError(
                    f"two-phase budget exceeds total_max_passes:"
                    f" {phase1}+{phase2} > {total_max}")
            if (bool(config.get("require_exact_total_phase_budget", False))
                    and phase1 + phase2 != total_max):
                raise ValueError(
                    "causal difficulty contrast requires the declared "
                    "phase budgets to exactly consume total_max_passes: "
                    f"{phase1}+{phase2} != {total_max}")
            if phase1 > int(fraction * total_max):
                raise ValueError(
                    f"phase-1 budget {phase1} exceeds phase1_max_fraction"
                    f" {fraction} of {total_max}")
            if phase2 < normal_min:
                raise ValueError(
                    f"normal phase {phase2} below normal_phase_min_passes"
                    f" {normal_min}")
        post_easy = self._train_easy_phase(
            config=config, env_plugin=env_plugin,
            agent_plugin=agent_plugin)

        normal_config = dict(config)
        normal_config["solvency_mode"] = NORMAL_MODE
        # Warm continuation from the learned weights; the artifact reload
        # gives the normal learner a FRESH replay buffer at the dynamics
        # boundary (easy transitions never enter normal updates).
        normal_config["warm_start_model"] = post_easy["path"]
        normal_config["warm_start_model_sha256"] = post_easy["sha256"]
        normal_config["warm_start_expand_observation_space"] = False

        result = super().run_pipeline(
            config=normal_config, env_plugin=env_plugin,
            agent_plugin=agent_plugin, mode="train")
        # The parent re-applies an already-bound contract and therefore
        # records an empty diff; the phase-1 application is the one that
        # actually moved the observation fields, so keep it.
        result["observation_contract_phase1"] = (
            _observation_contract_application)
        if _observation_contract_application.get("applied"):
            result["observation_contract"] = (
                _observation_contract_application)
        normal_history = result.get("history") or []
        normal_trained = [row for row in normal_history
                          if int(row.get("epoch", 0)) > 0]
        normal_best_epochs = [int(row["epoch"])
                              for row in normal_history
                              if row.get("checkpoint_improved") is True]
        phase1_mode = str(config.get("phase1_mode", EASY_MODE))
        result["curriculum"] = {
            "schema": "agent_multi.solvency_curriculum.result.v1",
            "phases": [phase1_mode, NORMAL_MODE],
            "post_easy": post_easy["meta"],
            "post_normal_artifact": result.get("best_model_path"),
            "replay_buffer_boundary": (
                "fresh buffer via artifact reload at the dynamics"
                " boundary; easy transitions excluded from normal"
                " updates"),
            "selection_basis": str(
                config.get("selection_metric")
                or "normal_validation_only"),
            "learning_rate_contract": {
                "constant_across_phases": bool(
                    config.get("require_constant_lr_across_phases", False)),
                "phase1_learning_rate": (
                    float(config["phase1_learning_rate"])
                    if config.get("phase1_learning_rate") is not None
                    else None),
                "phase2_learning_rate": (
                    float(config["learning_rate"])
                    if config.get("learning_rate") is not None else None),
            },
            "phase_summaries": {
                "phase1": {
                    "mode": phase1_mode,
                    "max_epochs": int(config.get("easy_max_epochs", 0)),
                    "epochs_run": post_easy["meta"].get(
                        "phase1_epochs_run"),
                    "best_epoch": post_easy["meta"].get(
                        "best_easy_epoch"),
                    "stopped_epoch": post_easy["meta"].get(
                        "phase1_stopped_epoch"),
                    "stop_reason": post_easy["meta"].get(
                        "phase1_stop_reason"),
                    "patience": int(config.get("easy_patience", 0)),
                    "patience_start_epoch": int(config.get(
                        "easy_patience_start_epoch", 1)),
                    "min_delta": float(config.get("easy_min_delta", 0.0)),
                },
                "phase2": {
                    "mode": NORMAL_MODE,
                    "max_epochs": int(config.get("max_epochs", 0)),
                    "epochs_run": len(normal_trained),
                    "best_epoch": (normal_best_epochs[-1]
                                   if normal_best_epochs else None),
                    "stopped_epoch": (int(normal_trained[-1]["epoch"])
                                      if normal_trained else None),
                    "stop_reason": result.get("stop_reason"),
                    "patience": int(config.get("l1_patience", 0)),
                    "patience_start_epoch": int(config.get(
                        "l1_patience_start_epoch", 1)),
                    "min_delta": float(config.get("l1_min_delta", 0.0)),
                    "activity_patience": int(config.get(
                        "l1_activity_patience", 0)),
                },
            },
        }
        return result
