"""
rl_pipeline_with_validation.py — train/val/test pipeline with L1 early stopping.

This pipeline mirrors predictor's three-mode pattern (train / inference /
optimization upstream) and adds per-epoch validation evaluation with
level-1 early stopping based on a composite watch metric:

    selection_mean = 0.5 * (train_tail_score + val_score)
    composite = selection_mean - beta * abs(train_tail_score - val_score)

When `selection_metric=risk_adjusted_return`, train_tail_score and val_score
are RAP = total_return - lambda * max_drawdown_fraction. Patience resets when
the L1 composite improves over the best so far. Training stops when patience
>= configured `l1_patience` or `max_epochs` is hit.
The train-side watch window is the last week of the training period, not
the full multi-year training slice, so a large historical train return cannot
hide no-trade or bad validation behavior.

Per-epoch logs include:
    epoch | L1 patience X/N | L2 patience Y/M | trades | win% | sharpe |
    profit | balance      (validation rollout)

Splits are time-ordered:
    train: first `train_years` (chronological)
    val:   next  `val_years`
    test:  next  `test_years` (typically the last block)

Final output: ASCII table over train/val/test plus results.json with the
same content next to the saved model.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from copy import deepcopy
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import _actor_liveness as _liveness
from . import _lexicographic_selection as _lex
from . import _paired_generalization as _paired
from . import _return_trace as _trace_mod
from ._weekly_metrics import canonical_weekly_metrics_from_trace
from ._observation_contract import (
    apply_observation_contract,
    validate_observation_contract,
)
from .rl_pipeline import (
    _action_summary_fields,
    _new_action_stats,
    _update_action_stats,
)
from agent_plugins._progress_callback import make_progress_callback


_METRIC_KEYS = ("trades_total", "win_pct", "sharpe_ratio", "total_return", "final_equity")


def _verify_artifact_sha256(path: Path, expected: str | None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected:
        normalized = str(expected).removeprefix("sha256:")
        if normalized != actual:
            raise ValueError(
                f"warm-start model sha256 mismatch: expected {normalized}, "
                f"got {actual}"
            )
    return actual


_VERIFIED_NESTED_MANIFESTS: Dict[
    Tuple[str, int, int], Tuple[Dict[str, Any], Tuple[Tuple[Any, ...], ...]]
] = {}


def _nested_split_out_dir(config: Dict[str, Any]) -> Path:
    """The ONE derivation of the nested split directory, shared by the
    materializer and the role resolver so they can never disagree."""
    return Path(
        config.get("nested_split_dir")
        or Path(config.get("save_model", "./agent_model.zip"))
        .resolve().parent / "nested_splits")


def _verified_nested_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load and VERIFY the nested split manifest (every materialized
    role csv is re-hashed) before any role fact is trusted.

    Cached per manifest and materialized-role file stat signatures: a
    decision cell materializes its splits once per phase, so unchanged
    files are not re-hashed every epoch.  A role CSV rewrite invalidates
    the cache even when the manifest itself did not change.
    """
    from . import _nested_splits

    path = Path(manifest_path)
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _VERIFIED_NESTED_MANIFESTS.get(key)
    if cached is not None:
        manifest, verified_role_stats = cached
        try:
            current_role_stats = _nested_role_file_stats(manifest)
        except OSError:
            current_role_stats = ()
        if current_role_stats == verified_role_stats:
            return manifest

    manifest = _nested_splits.verify_split_manifest(path)
    role_stats = _nested_role_file_stats(manifest)
    _VERIFIED_NESTED_MANIFESTS[key] = (manifest, role_stats)
    return manifest


def _nested_role_file_stats(manifest: Dict[str, Any]) -> Tuple[Tuple[Any, ...], ...]:
    """Stable signatures for every materialized role file in a manifest."""
    facts = []
    for role, entry in sorted((manifest.get("roles") or {}).items()):
        if entry.get("status") != "MATERIALIZED":
            continue
        csv_path = Path(str(entry["csv"])).resolve()
        stat = csv_path.stat()
        facts.append((role, str(csv_path), int(stat.st_mtime_ns),
                      int(stat.st_size)))
    return tuple(facts)


def _replay_buffer_size(model: Any) -> int | None:
    """Current replay-buffer occupancy, or None when the model has no
    readable buffer. Used to prove an evaluation rollout — and its
    causal prefix in particular — writes no transitions."""
    buffer = getattr(model, "replay_buffer", None)
    if buffer is None:
        return None
    size = getattr(buffer, "size", None)
    try:
        value = size() if callable(size) else size
        return int(value) if value is not None else None
    except Exception:
        return None


def _load_env_plugin(name: str, config: Dict[str, Any]):
    eps = entry_points().select(group="env.plugins")
    ep = next((e for e in eps if e.name == name), None)
    if ep is None:
        raise ImportError(f"env plugin '{name}' not found")
    klass = ep.load()
    inst = klass(config)
    inst.set_params(**config)
    return inst


def _win_pct(summary: Dict[str, Any]) -> float:
    won = summary.get("trades_won")
    total = summary.get("trades_total")
    try:
        won = float(won) if won is not None else 0.0
        total = float(total) if total is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return (won / total * 100.0) if total > 0 else 0.0


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _set_env_training_progress(env: Any, progress: float) -> bool:
    """Propagate normalized training progress through common Gym wrappers."""
    bounded = max(0.0, min(1.0, float(progress)))
    visited: set[int] = set()

    def visit(current: Any) -> bool:
        if current is None or id(current) in visited:
            return False
        visited.add(id(current))
        setter = getattr(current, "set_training_progress", None)
        if callable(setter):
            setter(bounded)
            return True
        changed = False
        for child in getattr(current, "envs", ()) or ():
            changed = visit(child) or changed
        inner = getattr(current, "env", None)
        if inner is not current:
            changed = visit(inner) or changed
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not current:
            changed = visit(unwrapped) or changed
        return changed

    return visit(env)


def _training_progress_for_epoch(
    epoch: int,
    *,
    max_epochs: int,
    curriculum_epochs: int | None,
) -> float:
    """Map epochs to a bounded curriculum horizon independent of hard cap."""
    horizon = int(curriculum_epochs or max_epochs)
    if horizon < 2:
        raise ValueError("execution_cost_curriculum_epochs must be >= 2")
    return min(1.0, max(0.0, (int(epoch) - 1) / (horizon - 1)))


def _trade_count(summary: Dict[str, Any]) -> int:
    return int(_safe_float(summary.get("trades_total")) or 0)


def _drawdown_fraction(summary: Dict[str, Any]) -> float:
    """Return max drawdown as a positive fraction of equity.

    Backtrader's DrawDown analyzer reports ``max.drawdown`` as a percentage
    value (for example 2.5 means 2.5%). Some older/imported summaries may
    already carry fractional values under ``max_drawdown``; keep that fallback
    deliberately conservative.
    """
    raw_pct = _safe_float(summary.get("max_drawdown_pct"))
    if not math.isnan(raw_pct):
        return max(0.0, raw_pct / 100.0)
    raw_fraction = _safe_float(summary.get("max_drawdown"))
    if not math.isnan(raw_fraction):
        return abs(raw_fraction)
    return 0.0


def _risk_adjusted_return(summary: Dict[str, Any], risk_lambda: float) -> float:
    ret = _safe_float(summary.get("total_return"))
    if math.isnan(ret):
        ret = 0.0
    return ret - float(risk_lambda) * _drawdown_fraction(summary)


def _annotate_risk_adjusted(summary: Dict[str, Any], risk_lambda: float) -> None:
    drawdown = _drawdown_fraction(summary)
    ret = _safe_float(summary.get("total_return"))
    if math.isnan(ret):
        ret = 0.0
    summary["max_drawdown_fraction"] = drawdown
    summary["risk_penalty_lambda"] = float(risk_lambda)
    summary["risk_adjusted_total_return"] = ret - float(risk_lambda) * drawdown


def _resolve_l1_min_checkpoint_timesteps(
    config: Dict[str, Any],
    default: int | None = None,
) -> int:
    """Return the earliest timestep at which an L1 checkpoint is trainable.

    Off-policy agents do not update their networks before ``learning_starts``.
    Letting an earlier rollout compete for the best checkpoint makes every
    hyperparameter candidate collapse to the same seeded, untrained policy.
    """
    configured = config.get("l1_min_checkpoint_timesteps", default)
    if configured is None:
        learning_starts = max(0, int(config.get("learning_starts", 0)))
        return learning_starts + 1 if learning_starts else 0
    return max(0, int(configured))


def _update_l1_checkpoint_state(
    *,
    composite: float,
    best_composite: float,
    no_improve: int,
    min_delta: float,
    eligible: bool,
    patience_eligible: bool | None = None,
) -> tuple[float, int, bool]:
    """Update checkpoint state without charging pre-patience warm-up epochs."""
    if not eligible:
        return best_composite, no_improve, False
    improved = composite > (best_composite + min_delta)
    if improved:
        return composite, 0, True
    if patience_eligible is False:
        return best_composite, no_improve, False
    return best_composite, no_improve + 1, False


def _checkpoint_is_eligible(
    *,
    num_timesteps: int,
    minimum_timesteps: int,
    trade_gate_passed: bool,
) -> bool:
    """Require both trained weights and observable trading activity.

    A no-trade rollout can carry a finite (penalized) scalar and would
    otherwise become the first checkpoint merely because ``-inf`` has not
    been replaced yet.  Such a checkpoint is not usable by the portfolio or
    by the next curriculum phase.
    """
    return bool(
        int(num_timesteps) >= int(minimum_timesteps) and trade_gate_passed
    )


def _activity_stop_disposition(
    *,
    best_checkpoint_saved: bool,
    streak: int,
    start_epoch: int,
    budget: int,
) -> tuple[str, str]:
    """Describe activity exhaustion without erasing an earlier checkpoint."""
    prefix = (
        f"activity-ineligible for {int(streak)} consecutive epochs after "
        f"epoch {int(start_epoch)} (budget={int(budget)})"
    )
    if best_checkpoint_saved:
        return (
            "activity_stop_after_best_checkpoint",
            prefix + "; the current policy lost the trade gate, while the "
            "previously saved activity-eligible checkpoint is retained",
        )
    return (
        "activity_stop_no_eligible_checkpoint",
        prefix + "; the trade gate never passed, so no eligible checkpoint "
        "exists",
    )


# The complete set of selection metrics _selection_value implements.
# This module OWNS the invariant; app/config_validation.py observes it
# through runtime_implemented_metrics() and must never restate it. Any
# branch added below must be added here, and vice versa — the surface
# test in tests/test_config_validation.py holds the two together.
IMPLEMENTED_SELECTION_METRICS = frozenset(
    {
        _lex.METRIC_NAME,
        _paired.METRIC_NAME,
        "robust_weekly_rap_fitness",
        "robust_weekly_rap",
        "execution_curriculum_robust_fitness",
        "risk_adjusted_return",
        "risk_adjusted_total_return",
        "rap",
        "total_return",
    }
)


def _selection_value(summary: Dict[str, Any], *, selection_metric: str, risk_lambda: float) -> float:
    metric = str(selection_metric or "total_return").strip().lower()
    if metric == _lex.METRIC_NAME:
        # ETH order §9: transparent constrained/lexicographic contract on
        # validation. The scalar returned here is DEAP transport ONLY;
        # the persisted ordered tuple in the summary is authoritative and
        # nothing here may be displayed as return or profit.
        contract = _lex.evaluate_selection_contract(
            summary,
            min_trades=int(summary.get("_selection_min_trades", 0)),
        )
        summary["selection_contract"] = contract
        return float(contract["transport_scalar"])
    if metric == _paired.METRIC_NAME:
        value, source = _paired._split_utility(summary)
        if value is None:
            raise ValueError(
                "paired_generalization_weekly_v1 requires a finite"
                " common-scale weekly utility per split; none present")
        return float(value)
    if metric in {
        "robust_weekly_rap_fitness",
        "robust_weekly_rap",
        "execution_curriculum_robust_fitness",
    }:
        value = _safe_float(summary.get("robust_weekly_rap_fitness"))
        if math.isnan(value):
            raise ValueError("robust validation summary is missing finite fitness")
        return value
    if metric in {"risk_adjusted_return", "risk_adjusted_total_return", "rap"}:
        return _risk_adjusted_return(summary, risk_lambda)
    ret = _safe_float(summary.get("total_return"))
    return 0.0 if math.isnan(ret) else ret


def _selection_pair_details(
    train_tail_summary: Dict[str, Any],
    val_summary: Dict[str, Any],
    *,
    selection_metric: str,
    risk_lambda: float,
    gap_penalty_beta: float,
) -> Dict[str, float]:
    train_tail_score = _selection_value(
        train_tail_summary,
        selection_metric=selection_metric,
        risk_lambda=risk_lambda,
    )
    val_score = _selection_value(
        val_summary,
        selection_metric=selection_metric,
        risk_lambda=risk_lambda,
    )
    if str(selection_metric or "").strip().lower() == _lex.METRIC_NAME:
        # §9 / AUD-F1-20260805-108: the lexicographic contract selects on
        # VALIDATION only. Averaging order keys or subtracting a gap
        # penalty would break the proven ordering, so the checkpoint
        # score IS the validation order key.
        return {
            "train_tail_selection_score": train_tail_score,
            "validation_selection_score": val_score,
            "train_validation_selection_mean_score": val_score,
            "train_validation_selection_gap": 0.0,
            "train_validation_selection_gap_penalty": 0.0,
            "train_validation_selection_score": val_score,
        }
    mean_score = 0.5 * (train_tail_score + val_score)
    gap = abs(train_tail_score - val_score)
    gap_penalty = float(gap_penalty_beta) * gap
    return {
        "train_tail_selection_score": train_tail_score,
        "validation_selection_score": val_score,
        "train_validation_selection_mean_score": mean_score,
        "train_validation_selection_gap": gap,
        "train_validation_selection_gap_penalty": gap_penalty,
        "train_validation_selection_score": mean_score - gap_penalty,
    }


def _early_stop_composite(
    train_tail_summary: Dict[str, Any],
    val_summary: Dict[str, Any],
    *,
    min_trades: int | None = None,
    min_train_tail_trades: int | None = None,
    min_validation_trades: int | None = None,
    no_trade_penalty: float,
    selection_metric: str = "total_return",
    risk_lambda: float = 1.0,
    gap_penalty_beta: float = 0.25,
) -> Tuple[float, float, bool, float, float, int, int]:
    train_tail_ret = _safe_float(train_tail_summary.get("total_return"))
    val_ret = _safe_float(val_summary.get("total_return"))
    if math.isnan(train_tail_ret):
        train_tail_ret = 0.0
    if math.isnan(val_ret):
        val_ret = 0.0
    details = _selection_pair_details(
        train_tail_summary,
        val_summary,
        selection_metric=selection_metric,
        risk_lambda=risk_lambda,
        gap_penalty_beta=gap_penalty_beta,
    )
    raw = details["train_validation_selection_score"]
    train_tail_trades = _trade_count(train_tail_summary)
    val_trades = _trade_count(val_summary)
    legacy_min = max(0, int(min_trades or 0))
    train_tail_min = max(
        0,
        int(
            legacy_min
            if min_train_tail_trades is None
            else min_train_tail_trades
        ),
    )
    validation_min = max(
        0,
        int(
            legacy_min
            if min_validation_trades is None
            else min_validation_trades
        ),
    )
    trade_gate_passed = (
        train_tail_trades >= train_tail_min
        and val_trades >= validation_min
    )
    composite = raw if trade_gate_passed else raw - no_trade_penalty
    return composite, raw, trade_gate_passed, train_tail_ret, val_ret, train_tail_trades, val_trades


def _normalize_split_label(name: str) -> str:
    n = str(name).strip().lower()
    if n in _trace_mod.ALLOWED_SPLITS:
        return n
    if n in ("val", "valid"):
        return "validation"
    if n.endswith("_epoch") and n[: -len("_epoch")] in ("train", "validation", "val"):
        return n if n in _trace_mod.ALLOWED_SPLITS else "evaluation"
    return "evaluation"


def _format_table(rows: List[Tuple[str, Dict[str, Any]]]) -> str:
    headers = ["Split", "Trades", "Win %", "Sharpe", "Profit", "Balance"]
    fmt_rows = []
    for name, s in rows:
        fmt_rows.append([
            name,
            str(int(_safe_float(s.get("trades_total")) or 0)),
            f"{_win_pct(s):.2f}",
            f"{_safe_float(s.get('sharpe_ratio')):.4f}",
            f"{_safe_float(s.get('total_return')) * 100:.2f}%",
            f"{_safe_float(s.get('final_equity')):.2f}",
        ])
    widths = [max(len(h), max(len(r[i]) for r in fmt_rows)) for i, h in enumerate(headers)]
    sep = "+".join("-" * (w + 2) for w in widths)
    sep = "+" + sep + "+"
    def fmt_row(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    out = [sep, fmt_row(headers), sep]
    for r in fmt_rows:
        out.append(fmt_row(r))
    out.append(sep)
    return "\n".join(out)


class PipelinePlugin:
    plugin_params: Dict[str, Any] = {
        # split widths (years)
        "train_years": 4,
        "val_years": 1,
        "test_years": 1,
        "train_days": None,
        "val_days": None,
        "test_days": None,
        "train_start": None,
        "train_end": None,
        "validation_start": None,
        "validation_end": None,
        "val_start": None,
        "val_end": None,
        "test_start": None,
        "test_end": None,
        "min_split_rows": 100,
        "split_anchor": "start",  # "start" or "end" of dataset

        # epoch loop
        "epoch_timesteps": 2_000,
        "max_epochs": 2_000,
        "l1_patience": 60,
        "l1_patience_start_epoch": 40,
        # AUD-P1LR-20260815-234: the activity-ineligible budget is a
        # SECOND terminator and it fires far earlier than the improvement
        # patience against an inactive policy. It lived as a bare literal
        # inside the epoch loop, so it never appeared in the plugin's
        # declared parameter surface and no contract reader could see it.
        # A rule that can end training is declared here or it is hidden.
        "l1_activity_patience": 40,
        # None inherits l1_patience_start_epoch.
        "l1_activity_patience_start_epoch": None,
        "l1_min_delta": 1e-5,
        "l1_min_checkpoint_timesteps": None,
        "early_stop_train_tail_days": 7,
        "early_stop_min_trades": 1,
        "early_stop_min_train_tail_trades": None,
        "early_stop_min_validation_trades": None,
        "early_stop_no_trade_penalty": 1_000_000.0,
        "selection_metric": "total_return",
        "risk_penalty_lambda": 1.0,
        "l1_generalization_gap_penalty_beta": 0.25,

        # eval
        "eval_seed": 0,
        "train_seed": 0,
        "save_model": "./agent_model.zip",
        "load_model": None,
        "warm_start_model": None,
        "return_trace_dir": None,
        "evaluate_test_split": True,
        "write_results_sidecar": True,
        "execution_cost_curriculum_epochs": None,

        # AUD-P1LR-20260815-235 — actor liveness probe. Every checkpoint
        # is measured on a strided sample of the REAL validation
        # observations it was just scored on, so a first layer that
        # cannot learn is a typed fact at epoch 1 instead of a
        # post-mortem after an 80-epoch grind. Measurement is ALWAYS on:
        # it costs one matmul over a 256-row batch and there is no flag
        # to forget. A fully dead first layer (zero live units, hence
        # zero gradient) REFUSES by default because every remaining
        # epoch is already decided.
        "actor_liveness_probe_observations":
            _liveness.DEFAULT_PROBE_OBSERVATIONS,
        "actor_liveness_min_live_unit_fraction":
            _liveness.DEFAULT_MIN_LIVE_UNIT_FRACTION,
        "refuse_dead_actor": True,
        "refuse_constant_policy_actor": False,
    }

    plugin_debug_vars = [
        "train_years", "val_years", "test_years",
        "train_days", "val_days", "test_days",
        "train_start", "train_end", "validation_start", "validation_end",
        "val_start", "val_end", "test_start", "test_end",
        "min_split_rows",
        "epoch_timesteps", "max_epochs", "l1_patience",
        "l1_patience_start_epoch",
        "l1_activity_patience", "l1_activity_patience_start_epoch",
        "l1_min_delta",
        "l1_min_checkpoint_timesteps",
        "early_stop_train_tail_days", "early_stop_min_trades",
        "early_stop_min_train_tail_trades",
        "early_stop_min_validation_trades",
        "early_stop_no_trade_penalty",
        "selection_metric", "risk_penalty_lambda", "l1_generalization_gap_penalty_beta",
        "warm_start_model", "return_trace_dir", "evaluate_test_split",
        "write_results_sidecar", "execution_cost_curriculum_epochs",
        "actor_liveness_probe_observations",
        "actor_liveness_min_live_unit_fraction", "refuse_dead_actor",
        "refuse_constant_policy_actor",
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        self._tempdir: Optional[tempfile.TemporaryDirectory] = None
        # Set by _split_csv when a nested contract materializes: the
        # role resolver falls back to it so a split evaluation launched
        # with a derived config copy (curriculum phase 1) still resolves
        # its role from the same verified manifest.
        self._nested_split_manifest_path: Optional[str] = None
        # Per-split observation samples captured by the most recent
        # rollout, keyed by split label. Measurement scratch only: it is
        # never serialized and never influences selection.
        self._liveness_observations: Dict[str, Any] = {}
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

    def get_debug_info(self) -> Dict[str, Any]:
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info: Dict[str, Any]) -> None:
        debug_info.update(self.get_debug_info())

    # ------------------------------------------------------------------
    def _split_csv(self, config: Dict[str, Any]) -> Dict[str, str]:
        # WP1 (doc 38 §3): nested-contract configs delegate to the ONE
        # typed split implementation; this method never grows nested
        # date parsing of its own. Legacy configs keep the path below.
        contract_path = config.get("nested_split_contract")
        if contract_path:
            from . import _nested_splits

            contract = _nested_splits.load_contract(Path(contract_path))
            out_dir = _nested_split_out_dir(config)
            manifest = _nested_splits.materialize_nested_splits(
                contract, out_dir,
                mode=str(config.get("nested_split_mode", "l1")))
            config["nested_split_manifest"] = manifest["manifest_path"]
            self._nested_split_manifest_path = manifest["manifest_path"]
            # Doc 38 §5: a nested decision config must use the paired
            # comparator; the validation-only lexicographic branch is
            # structurally out of reach for it.
            metric = str(config.get("selection_metric") or "").strip().lower()
            if metric != _paired.METRIC_NAME:
                raise ValueError(
                    "nested_split_contract requires selection_metric="
                    f"{_paired.METRIC_NAME!r}; got {metric!r} — the"
                    " legacy validation-only branch is forbidden for"
                    " nested decision configs")
            roles = manifest["roles"]
            paths = {
                "train": roles["fit_train"]["csv"],
                "train_monitor": roles["train_monitor"]["csv"],
                "validation": roles["inner_validation"]["csv"],
                "outer_validation": roles["outer_validation"]["csv"],
                # loop-compatibility aliases: the in-sample member of
                # the L1 pair is the 2022 monitor year (never a tail
                # sliver), and 'val' is inner validation 2023.
                "train_tail": roles["train_monitor"]["csv"],
                "val": roles["inner_validation"]["csv"],
            }
            if roles["sealed_test"].get("status") == "MATERIALIZED":
                paths["test"] = roles["sealed_test"]["csv"]
            return paths
        src = config["input_data_file"]
        date_col = config.get("date_column", "DATE_TIME")
        df = pd.read_csv(src)
        if date_col not in df.columns:
            raise ValueError(f"date_column '{date_col}' missing from {src}")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        train_y = float(config.get("train_years", self.params["train_years"]))
        val_y = float(config.get("val_years", self.params["val_years"]))
        test_y = float(config.get("test_years", self.params["test_years"]))
        train_d = _safe_float_or_none(config.get("train_days", self.params["train_days"]))
        val_d = _safe_float_or_none(config.get("val_days", self.params["val_days"]))
        test_d = _safe_float_or_none(config.get("test_days", self.params["test_days"]))
        min_split_rows = int(config.get("min_split_rows", self.params["min_split_rows"]))
        anchor = str(config.get("split_anchor", self.params["split_anchor"])).lower()
        use_day_splits = train_d is not None and val_d is not None and test_d is not None
        explicit_train_start = config.get("train_start", self.params["train_start"])
        explicit_train_end = config.get("train_end", self.params["train_end"])
        explicit_val_start = (
            config.get("validation_start", self.params["validation_start"])
            or config.get("val_start", self.params["val_start"])
        )
        explicit_val_end = (
            config.get("validation_end", self.params["validation_end"])
            or config.get("val_end", self.params["val_end"])
        )
        explicit_test_start = config.get("test_start", self.params["test_start"])
        explicit_test_end = config.get("test_end", self.params["test_end"])
        explicit_ranges = [
            explicit_train_start,
            explicit_train_end,
            explicit_val_start,
            explicit_val_end,
            explicit_test_start,
            explicit_test_end,
        ]
        use_explicit_splits = all(v not in (None, "") for v in explicit_ranges)
        if any(v not in (None, "") for v in explicit_ranges) and not use_explicit_splits:
            raise ValueError(
                "Explicit weekly split windows require train_start, train_end, "
                "validation_start/val_start, validation_end/val_end, test_start, and test_end."
            )

        first = df[date_col].iloc[0]
        last = df[date_col].iloc[-1]
        if use_explicit_splits:
            train_start = pd.Timestamp(explicit_train_start)
            train_end = pd.Timestamp(explicit_train_end)
            val_start = pd.Timestamp(explicit_val_start)
            val_end = pd.Timestamp(explicit_val_end)
            test_start = pd.Timestamp(explicit_test_start)
            test_end = pd.Timestamp(explicit_test_end)
            if not train_start < train_end <= val_start < val_end <= test_start < test_end:
                raise ValueError(
                    "Explicit weekly split windows must be ordered as "
                    "train_start < train_end <= validation_start < validation_end <= test_start < test_end."
                )
        elif anchor == "end":
            test_end = last
            if use_day_splits:
                test_start = test_end - pd.DateOffset(days=int(test_d))
                val_end = test_start
                val_start = val_end - pd.DateOffset(days=int(val_d))
                train_end = val_start
                train_start = train_end - pd.DateOffset(days=int(train_d))
            else:
                test_start = test_end - pd.DateOffset(years=int(test_y))
                val_end = test_start
                val_start = val_end - pd.DateOffset(years=int(val_y))
                train_end = val_start
                train_start = train_end - pd.DateOffset(years=int(train_y))
        else:
            train_start = first
            if use_day_splits:
                train_end = train_start + pd.DateOffset(days=int(train_d))
                val_start = train_end
                val_end = val_start + pd.DateOffset(days=int(val_d))
                test_start = val_end
                test_end = test_start + pd.DateOffset(days=int(test_d))
            else:
                train_end = train_start + pd.DateOffset(years=int(train_y))
                val_start = train_end
                val_end = val_start + pd.DateOffset(years=int(val_y))
                test_start = val_end
                test_end = test_start + pd.DateOffset(years=int(test_y))

        train_df = df[(df[date_col] >= train_start) & (df[date_col] < train_end)]
        val_df = df[(df[date_col] >= val_start) & (df[date_col] < val_end)]
        test_df = df[(df[date_col] >= test_start) & (df[date_col] < test_end)]
        train_tail_days = _safe_float_or_none(
            config.get("early_stop_train_tail_days", self.params["early_stop_train_tail_days"])
        )
        train_tail_df = train_df
        if train_tail_days is not None and train_tail_days > 0:
            train_tail_start = train_end - pd.DateOffset(days=int(train_tail_days))
            train_tail_df = df[(df[date_col] >= train_tail_start) & (df[date_col] < train_end)]
            configured_window = config.get("window_size")
            if configured_window not in (None, ""):
                min_env_rows = max(3, int(configured_window) + 2)
                if len(train_tail_df) < min_env_rows:
                    train_tail_df = train_df.tail(min_env_rows)

        for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
            if len(part) < min_split_rows:
                raise ValueError(
                    f"{name} split has only {len(part)} rows; minimum is {min_split_rows} (range "
                    f"{train_start if name=='train' else val_start if name=='val' else test_start} "
                    f"-> {train_end if name=='train' else val_end if name=='val' else test_end}). "
                    f"Adjust split_anchor, *_years, *_days, or min_split_rows."
                )

        self._tempdir = tempfile.TemporaryDirectory(prefix="agent_multi_split_")
        out_dir = Path(self._tempdir.name)
        paths = {}
        for name, part in (
            ("train", train_df),
            ("train_tail", train_tail_df),
            ("val", val_df),
            ("test", test_df),
        ):
            p = out_dir / f"{name}.csv"
            part.to_csv(p, index=False)
            paths[name] = str(p)
        if not config.get("quiet_mode"):
            print(
                f"[split] train={len(train_df):>6} rows ({train_df[date_col].iloc[0].date()} -> {train_df[date_col].iloc[-1].date()})  "
                f"val={len(val_df):>5} rows ({val_df[date_col].iloc[0].date()} -> {val_df[date_col].iloc[-1].date()})  "
                f"test={len(test_df):>5} rows ({test_df[date_col].iloc[0].date()} -> {test_df[date_col].iloc[-1].date()})"
            )
        return paths

    def _nested_manifest_path(
        self, config: Dict[str, Any]
    ) -> Optional[Path]:
        """Where the verified nested manifest lives for this run, or
        None when the config declares no nested contract at all.

        The instance-recorded path is only ever consulted for a config
        that itself declares the nested contract, so a legacy run can
        never inherit a manifest from an earlier nested run.
        """
        explicit = config.get("nested_split_manifest")
        if explicit:
            return Path(str(explicit))
        if not (config.get("nested_split_contract")
                or config.get("nested_split_dir")):
            return None
        derived = _nested_split_out_dir(config) / "nested_split_manifest.json"
        if derived.is_file():
            return derived
        recorded = self._nested_split_manifest_path
        if recorded:
            return Path(str(recorded))
        return derived              # missing → the caller fails closed

    def _resolve_nested_role(
        self, config: Dict[str, Any], csv_path: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Resolve (role, verified role entry) for ``csv_path`` from the
        VERIFIED nested split manifest — finding 231 requirement 1.

        The role and its ``context_rows`` come from the manifest's own
        re-hashed role entries; nothing is inferred from the file name
        or from row positions. A run without a nested contract returns
        None (legacy split path, no causal prefix). A csv that the
        manifest does not declare is REFUSED rather than silently
        scored as if it carried no context.
        """
        manifest_path = self._nested_manifest_path(config)
        if manifest_path is None:
            return None
        if not manifest_path.is_file():
            raise ValueError(
                "nested split contract declared but its manifest is "
                f"missing at {manifest_path} — refusing to evaluate a "
                "split whose role and context rows cannot be verified")
        manifest = _verified_nested_manifest(manifest_path)
        target = Path(csv_path).resolve()
        for role, entry in (manifest.get("roles") or {}).items():
            if entry.get("status") != "MATERIALIZED":
                continue
            if Path(str(entry["csv"])).resolve() != target:
                continue
            context_rows = entry.get("context_rows")
            if (isinstance(context_rows, bool)
                    or not isinstance(context_rows, int)
                    or context_rows < 0):
                raise ValueError(
                    f"nested role {role}: context_rows is "
                    f"{context_rows!r} — the manifest must declare a "
                    "non-negative integer count of causal context rows")
            return role, entry
        raise ValueError(
            f"{csv_path} is not a materialized role of the verified "
            f"nested split manifest {manifest_path} — refusing to score "
            "a slice with no declared role or context semantics")

    def _make_split_env(self, env_plugin_name: str, base_config: Dict[str, Any], csv_path: str, agent_plugin,
                        context_rows: int | None = None):
        """Build one evaluation/training env for a split csv.

        ``context_rows`` installs the reusable ContextPrefixWrapper
        (finding 231 requirement 2): the leading causal-context rows of
        an evaluation role are forced to hold, tagged
        ``is_context_prefix`` and refused any account mutation, BEFORE
        the env reaches ``_rollout``. It is passed by the internal
        selection path (`_eval_on_split`) from the verified manifest.
        Callers that own their own prefix boundary — the final
        outer-validation helper — leave it None and keep their wrapper,
        and fit training is built without it.
        """
        cfg = deepcopy(base_config)
        cfg["input_data_file"] = csv_path
        env_plugin = _load_env_plugin(env_plugin_name, cfg)
        env = env_plugin.make_env(cfg)
        wrap = getattr(agent_plugin, "wrap_env", None)
        if callable(wrap):
            env = wrap(env, cfg)
        if context_rows:
            from . import _nested_splits

            env = _nested_splits.ContextPrefixWrapper(env, int(context_rows))
        return env_plugin, env

    def _eval_on_split(
        self,
        env_plugin_name: str,
        config: Dict[str, Any],
        csv_path: str,
        agent_plugin,
        model,
        seed: int,
        split_name: str,
    ) -> Dict[str, Any]:
        """Build a fresh env just for evaluation, roll out once, close it.

        Critical: never reuse the training env for evaluation — that pollutes
        SAC's replay buffer with terminal/post-reset transitions and freezes
        the actor weights from epoch 2 onward.

        WP-C guard: evaluation is structurally ``normal_realistic`` — no
        split evaluation can ever run under relaxed solvency dynamics,
        regardless of the training configuration.

        AUD-F1-20260812-231: this is the EXECUTING internal selection
        path. Under a nested contract the role and its declared
        ``context_rows`` are resolved from the verified manifest and the
        ContextPrefixWrapper is installed before the rollout, so
        train_monitor and inner_validation are scored on their declared
        scored rows only — never on the causal prefix that precedes
        them.
        """
        config = {**config, "solvency_mode": "normal_realistic"}
        nested = self._resolve_nested_role(config, csv_path)
        context_rows = int(nested[1]["context_rows"]) if nested else 0
        if context_rows:
            # A role that DECLARES causal context is built with its
            # prefix boundary installed; a role that declares none keeps
            # the historical four-argument factory call.
            plug, env = self._make_split_env(
                env_plugin_name, config, csv_path, agent_plugin,
                context_rows=context_rows,
            )
        else:
            plug, env = self._make_split_env(
                env_plugin_name, config, csv_path, agent_plugin)
        try:
            split_label = _normalize_split_label(split_name)
            run_id = _trace_mod.make_run_id(config)
            episode_id = f"{run_id}::{split_label}"
            asset = str(config.get("asset", "unknown_asset"))
            timeframe = str(config.get("timeframe", config.get("timeframe_label", "")))

            summary = self._rollout(
                env, agent_plugin, model, seed,
                asset=asset, timeframe=timeframe, split=split_label,
                run_id=run_id, episode_id=episode_id,
                continuous_threshold=_safe_float_or_none(
                    config.get("continuous_action_threshold")
                ),
                context_rows=context_rows,
                capture_observations=int(
                    config.get(
                        "actor_liveness_probe_observations",
                        self.params["actor_liveness_probe_observations"],
                    )
                    or 0
                ),
            )
            # Measurement scratch: it leaves the summary here so no
            # record, sidecar or hash ever sees a raw observation batch.
            self._liveness_observations[split_label] = summary.pop(
                "_actor_liveness_observations", None)
            if nested is not None:
                role, entry = nested
                expected_scored_rows = int(entry["scored_rows"])
                if int(summary.get("scored_steps", -1)) != expected_scored_rows:
                    raise ValueError(
                        f"{role}: rollout scored {summary.get('scored_steps')} "
                        f"steps but the verified manifest declares "
                        f"{expected_scored_rows}; refusing an off-by-one or "
                        "truncated evaluation")
                summary["nested_role"] = role
                summary["nested_role_csv_sha256"] = entry.get("csv_sha256")
                summary["nested_role_scored_rows"] = entry.get("scored_rows")
                summary["nested_role_context_rows"] = context_rows
            summary["_selection_min_trades"] = int(
                config.get("selection_min_trades", 0))
            trace_rows = summary.get("_return_trace_rows")
            if trace_rows:
                weekly_metrics = canonical_weekly_metrics_from_trace(
                    trace_rows,
                    initial_cash=float(config.get("initial_cash", 10_000.0)),
                    risk_penalty_lambda=float(
                        config.get("risk_penalty_lambda", 1.0)
                    ),
                    metric_schema="trading.weekly.v1",
                )
                if not bool(config.get("_retain_weekly_rows", False)):
                    weekly_metrics.pop("weekly_rows", None)
                summary.update(weekly_metrics)
            if not bool(config.get("_retain_return_trace_rows", False)):
                summary.pop("_return_trace_rows", None)
            trace_dir = config.get("return_trace_dir")
            if trace_dir and trace_rows is not None:
                trace_path = _trace_mod.derive_split_trace_path(str(trace_dir), split_label)
                # Per-split config view so the metadata sidecar's data_file
                # hash matches the slice that was actually evaluated.
                split_config = dict(config)
                split_config["_run_config_hash"] = config.get("_run_config_hash") or _trace_mod._hash_config(config)
                split_config["input_data_file"] = csv_path
                split_config["_split"] = split_label
                metadata = _trace_mod.write_return_trace(
                    trace_path,
                    trace_rows,
                    config=split_config,
                    split=split_label,
                    seed=seed,
                    asset=asset,
                    timeframe=timeframe,
                    run_id=run_id,
                    episode_id=episode_id,
                    feature_list=config.get("feature_list"),
                    env=env,
                )
                summary["return_trace_file"] = metadata["trace_file"]
                summary["return_trace_metadata_file"] = metadata["metadata_file"]
                # Stash the full sidecar so _final_eval can roll the per-split
                # metadata items into the run-level evidence index.
                summary["_return_trace_metadata"] = metadata
            return summary
        finally:
            try:
                plug.close()
            except Exception:
                pass

    @staticmethod
    def _rollout(
        env, agent_plugin, model, seed: int,
        *,
        asset: str = "unknown_asset",
        timeframe: str = "",
        split: str = "evaluation",
        run_id: str = "run",
        episode_id: str = "run::ep0",
        continuous_threshold: float | None = None,
        context_rows: int = 0,
        capture_observations: int = 0,
    ) -> Dict[str, Any]:
        """Roll the policy once and return the SCORED outcome.

        AUD-F1-20260812-231 requirement 5: causal-context rows are input,
        never measurement. A step tagged ``is_context_prefix`` by the
        ContextPrefixWrapper contributes nothing to the reward, the
        action statistics, the canonical return trace, the weekly
        metrics or the metric horizon; the counts of both populations
        are reported. ``context_rows`` is the count DECLARED by the
        verified manifest: if a role declares context and the rollout
        never sees a tagged prefix — an unwrapped env — the evaluation
        is refused instead of silently scoring the prefix.
        """
        declared_context_rows = int(context_rows or 0)
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        prefix_reward_excluded = 0.0
        steps = 0
        scored_steps = 0
        context_prefix_steps = 0
        score_boundary_equity: float | None = None
        done = False
        trace_rows: List[Dict[str, Any]] = []
        action_stats = _new_action_stats(
            continuous_threshold=continuous_threshold,
        )
        prev_equity = _safe_float_or_none(info.get("equity"))
        replay_size_before = _replay_buffer_size(model)
        # AUD-P1LR-20260815-235: the observations the policy is ABOUT to
        # act on are the only honest probe batch for a liveness check,
        # and they are already in hand here — capturing a strided sample
        # costs no extra env step and no second rollout.
        liveness_sampler = _liveness.StridedObservationSampler(
            capture_observations)
        while not done:
            liveness_sampler.offer(obs)
            action = agent_plugin.predict(model, obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = bool(terminated or truncated)
            if bool(info.get("is_context_prefix")):
                if scored_steps:
                    raise ValueError(
                        f"{split}: a causal-context row was reported after "
                        "the score boundary — the context prefix must "
                        "precede every scored row")
                context_prefix_steps += 1
                prefix_reward_excluded += float(reward)
                if steps > 1_000_000:
                    break
                continue
            if scored_steps == 0:
                # The score boundary: equity here is the opening equity
                # of the scored interval and must be the reset equity —
                # the wrapper refused any prefix mutation to get here.
                score_boundary_equity = prev_equity
                replay_size_at_boundary = _replay_buffer_size(model)
                if (replay_size_before is not None
                        and replay_size_at_boundary is not None
                        and replay_size_at_boundary != replay_size_before):
                    raise ValueError(
                        f"{split}: the replay buffer grew during the "
                        f"causal context prefix ({replay_size_before} -> "
                        f"{replay_size_at_boundary})")
            scored_steps += 1
            equity = _safe_float_or_none(info.get("equity"))
            _update_action_stats(action_stats, action, info)
            trace_rows.append(
                _trace_mod.build_trace_row(
                    env=env,
                    step=scored_steps,
                    action=action,
                    reward=reward,
                    info=info,
                    prev_equity=prev_equity,
                    asset=asset,
                    timeframe=timeframe,
                    split=split,
                    seed=int(seed),
                    run_id=run_id,
                    episode_id=episode_id,
                )
            )
            prev_equity = equity
            total_reward += float(reward)
            if steps > 1_000_000:
                break
        if declared_context_rows and context_prefix_steps != declared_context_rows:
            raise ValueError(
                f"{split}: the manifest declares {declared_context_rows} "
                f"causal context rows but the rollout separated "
                f"{context_prefix_steps} — the evaluation env was not "
                "wrapped, or the episode ended inside the prefix; "
                "refusing to score it")
        if declared_context_rows and scored_steps == 0:
            raise ValueError(
                f"{split}: the episode produced no scored rows after its "
                "causal context prefix")
        replay_size_after = _replay_buffer_size(model)
        if (replay_size_before is not None and replay_size_after is not None
                and replay_size_after != replay_size_before):
            raise ValueError(
                f"{split}: the replay buffer grew during evaluation "
                f"({replay_size_before} -> {replay_size_after})")
        base = env
        while hasattr(base, "env") and not hasattr(base, "summary"):
            base = base.env
        summary = base.summary() if hasattr(base, "summary") else {}
        summary["episode_reward"] = total_reward
        summary["episode_length"] = scored_steps
        summary["scored_steps"] = scored_steps
        summary["context_prefix_steps"] = context_prefix_steps
        summary["context_rows_declared"] = declared_context_rows
        summary["context_prefix_reward_excluded"] = prefix_reward_excluded
        summary["score_boundary_opening_equity"] = score_boundary_equity
        summary["total_env_steps"] = steps
        summary.update(_action_summary_fields(action_stats, summary))
        summary["_return_trace_rows"] = trace_rows
        # Private, popped by _eval_on_split before the summary can reach
        # any record or sidecar.
        summary["_actor_liveness_observations"] = liveness_sampler.batch()
        return summary

    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        *,
        config: Dict[str, Any],
        env_plugin,
        agent_plugin,
        mode: str = "train",
    ) -> Dict[str, Any]:
        mode = str(mode).lower()
        # AUD-P1LR-20260815-235: an experiment program may declare the
        # observation fields its candidates run under. Binding them HERE
        # — before the split envs, the preprocessor and the model exist
        # — is what makes the fail-closed validator below effective; it
        # used to return on its first line because nothing between the
        # base config and this call ever declared the guard.
        config, observation_contract_application = apply_observation_contract(
            config)
        env_plugin_name = config.get("env_plugin", "gym_fx_env")
        validate_observation_contract(config)
        self._liveness_observations = {}
        liveness_probe_size = int(
            config.get(
                "actor_liveness_probe_observations",
                self.params["actor_liveness_probe_observations"],
            )
            or 0
        )
        liveness_min_live_fraction = float(
            config.get(
                "actor_liveness_min_live_unit_fraction",
                self.params["actor_liveness_min_live_unit_fraction"],
            )
        )
        liveness_refuse_dead = bool(
            config.get(
                "refuse_dead_actor",
                self.params["refuse_dead_actor"],
            )
        )
        liveness_refuse_constant = bool(
            config.get(
                "refuse_constant_policy_actor",
                self.params["refuse_constant_policy_actor"],
            )
        )
        actor_liveness_history: List[Dict[str, Any]] = []
        try:
            paths = self._split_csv(config)

            train_env_plugin, train_env = self._make_split_env(
                env_plugin_name, config, paths["train"], agent_plugin
            )

            try:
                if mode == "inference":
                    load_path = config.get("load_model")
                    if not load_path:
                        raise ValueError("inference mode requires config['load_model']")
                    model = agent_plugin.load(load_path, train_env)
                    final = self._final_eval(
                        agent_plugin, model, train_env,
                        env_plugin_name, paths, config, agent_plugin
                    )
                    return final

                # training mode. Optional warm-start continues from a previous
                # weekly checkpoint but evaluates/saves under the current
                # split windows and run id.
                warm_start_model = config.get("warm_start_model", self.params["warm_start_model"])
                if warm_start_model:
                    warm_start_path = Path(str(warm_start_model))
                    if not warm_start_path.exists():
                        raise FileNotFoundError(f"warm_start_model not found: {warm_start_path}")
                    _verify_artifact_sha256(
                        warm_start_path,
                        config.get("warm_start_model_sha256"),
                    )
                    if not config.get("quiet_mode"):
                        print(f"[train] warm-start loading {warm_start_path}", flush=True)
                    expansion_loader = getattr(
                        agent_plugin,
                        "load_with_observation_expansion",
                        None,
                    )
                    if bool(
                        config.get(
                            "warm_start_expand_observation_space",
                            False,
                        )
                    ):
                        if not callable(expansion_loader):
                            raise ValueError(
                                "agent does not support warm-start observation expansion"
                            )
                        model = expansion_loader(
                            str(warm_start_path),
                            train_env,
                            config,
                        )
                    else:
                        training_loader = getattr(
                            agent_plugin, "load_for_training", None
                        )
                        if callable(training_loader):
                            model = training_loader(
                                str(warm_start_path), train_env, config
                            )
                        else:
                            model = agent_plugin.load(
                                str(warm_start_path), train_env
                            )
                    try:
                        model.set_env(train_env)
                    except Exception:
                        pass
                else:
                    model = agent_plugin.build(train_env, config)
                pretrain_summary = None
                pretrain_behavior = getattr(agent_plugin, "pretrain_behavior", None)
                if callable(pretrain_behavior) and bool(config.get("oracle_behavior_pretrain_enabled", False)):
                    pretrain_summary = pretrain_behavior(model, train_env, config)
                if not hasattr(model, "learn"):
                    best_model_path = config.get("save_model") or "./agent_model.zip"
                    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
                    agent_plugin.save(model, best_model_path)
                    final = self._final_eval(
                        agent_plugin, model, train_env,
                        env_plugin_name, paths, config, agent_plugin,
                    )
                    final["mode"] = "deterministic_baseline"
                    final["history"] = []
                    final["best_composite"] = None
                    final["best_model_path"] = str(Path(best_model_path).resolve())
                    final["oracle_behavior_pretrain"] = pretrain_summary
                    return final

                epoch_ts = int(config.get("epoch_timesteps", self.params["epoch_timesteps"]))
                max_epochs = int(config.get("max_epochs", self.params["max_epochs"]))
                total_progress_timesteps = int(config.get("total_timesteps") or epoch_ts * max_epochs)
                l1_patience = int(config.get("l1_patience", self.params["l1_patience"]))
                l1_patience_start_epoch = max(
                    1,
                    int(
                        config.get(
                            "l1_patience_start_epoch",
                            self.params["l1_patience_start_epoch"],
                        )
                    ),
                )
                l1_min_delta = float(config.get("l1_min_delta", self.params["l1_min_delta"]))
                l1_min_checkpoint_timesteps = _resolve_l1_min_checkpoint_timesteps(
                    config,
                    self.params["l1_min_checkpoint_timesteps"],
                )
                seed = int(config.get("eval_seed", self.params["eval_seed"]))

                # L2 patience info shown in logs (driven externally by optimizer if any)
                l2_patience = config.get("optimization_patience", "-")
                l2_counter = config.get("_l2_counter", "-")

                # AUD-F1-20260806-127: bounded budget for epochs that
                # produce no eligible trading activity, tracked apart
                # from improvement patience.
                activity_patience = int(
                    config.get("l1_activity_patience")
                    if config.get("l1_activity_patience") is not None
                    else self.params["l1_activity_patience"])
                _activity_start = config.get(
                    "l1_activity_patience_start_epoch")
                if _activity_start is None:
                    _activity_start = self.params[
                        "l1_activity_patience_start_epoch"]
                if _activity_start is None:
                    _activity_start = l1_patience_start_epoch
                activity_patience_start_epoch = max(
                    1, int(_activity_start))
                activity_ineligible_streak = 0
                activity_stop_reason: str | None = None

                best_composite = -math.inf
                no_improve = 0
                best_checkpoint_saved = False
                best_model_path = config.get("save_model") or "./agent_model.zip"
                Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)

                history: List[Dict[str, Any]] = []

                # A supplied warm start is already a trained artifact.  Earn
                # its place by evaluating it under the current NORMAL split
                # contract before the first update, then retain it as the
                # floor that subsequent epochs must beat.  This prevents an
                # active curriculum handoff from being replaced by an epoch-1
                # policy that collapsed to HOLD everywhere.
                if warm_start_model and bool(
                    config.get("warm_start_baseline_checkpoint_enabled", True)
                ):
                    baseline_train_tail = self._eval_on_split(
                        env_plugin_name,
                        config,
                        paths.get("train_tail", paths["train"]),
                        agent_plugin,
                        model,
                        seed,
                        "train_tail_epoch",
                    )
                    baseline_val = self._eval_on_split(
                        env_plugin_name,
                        config,
                        paths["val"],
                        agent_plugin,
                        model,
                        seed,
                        "validation_epoch",
                    )
                    baseline_selection_metric = str(
                        config.get(
                            "selection_metric", self.params["selection_metric"]
                        )
                    )
                    baseline_risk_lambda = float(
                        config.get(
                            "risk_penalty_lambda",
                            self.params["risk_penalty_lambda"],
                        )
                    )
                    baseline_gap_beta = float(
                        config.get(
                            "l1_generalization_gap_penalty_beta",
                            self.params[
                                "l1_generalization_gap_penalty_beta"
                            ],
                        )
                    )
                    for summary in (baseline_train_tail, baseline_val):
                        _annotate_risk_adjusted(summary, baseline_risk_lambda)
                    baseline_min_trades = int(
                        config.get(
                            "early_stop_min_trades",
                            self.params["early_stop_min_trades"],
                        )
                    )
                    baseline_train_tail_min = config.get(
                        "early_stop_min_train_tail_trades",
                        self.params["early_stop_min_train_tail_trades"],
                    )
                    baseline_validation_min = config.get(
                        "early_stop_min_validation_trades",
                        self.params["early_stop_min_validation_trades"],
                    )
                    (
                        baseline_composite,
                        baseline_raw,
                        baseline_trade_gate,
                        _baseline_train_tail_return,
                        _baseline_val_return,
                        baseline_train_tail_trades,
                        baseline_val_trades,
                    ) = _early_stop_composite(
                        baseline_train_tail,
                        baseline_val,
                        min_trades=baseline_min_trades,
                        min_train_tail_trades=(
                            baseline_min_trades
                            if baseline_train_tail_min is None
                            else int(baseline_train_tail_min)
                        ),
                        min_validation_trades=(
                            baseline_min_trades
                            if baseline_validation_min is None
                            else int(baseline_validation_min)
                        ),
                        no_trade_penalty=float(
                            config.get(
                                "early_stop_no_trade_penalty",
                                self.params["early_stop_no_trade_penalty"],
                            )
                        ),
                        selection_metric=baseline_selection_metric,
                        risk_lambda=baseline_risk_lambda,
                        gap_penalty_beta=baseline_gap_beta,
                    )
                    history.append({
                        "epoch": 0,
                        "checkpoint_source": "warm_start_normal_baseline",
                        "selection_metric": baseline_selection_metric,
                        "composite_raw": baseline_raw,
                        "composite": baseline_composite,
                        "checkpoint_improved": bool(
                            baseline_trade_gate),
                        "early_stop_trade_gate_passed": baseline_trade_gate,
                        "train_tail_trades": baseline_train_tail_trades,
                        "val_trades": baseline_val_trades,
                    })
                    if baseline_trade_gate:
                        agent_plugin.save(model, best_model_path)
                        best_composite = baseline_composite
                        best_checkpoint_saved = True

                if not config.get("quiet_mode"):
                    print(
                        f"[train] starting: epoch_timesteps={epoch_ts} max_epochs={max_epochs} "
                        f"l1_patience={l1_patience} "
                        f"l1_patience_start_epoch={l1_patience_start_epoch} "
                        f"(L1=mean(train_tail_score,val_score)-beta*gap, no-trade penalized)"
                    )

                def _policy_checksum(m) -> Tuple[float, float, float]:
                    try:
                        actor = sum(float(p.detach().abs().sum().item())
                                    for p in m.policy.actor.parameters())
                    except Exception:
                        actor = float("nan")
                    try:
                        critic = sum(float(p.detach().abs().sum().item())
                                     for p in m.policy.critic.parameters())
                    except Exception:
                        critic = float("nan")
                    try:
                        # AUD (M0 order §6 Q4): D1 recorded NaN here for
                        # every epoch because log_ent_coef exists only in
                        # automatic entropy mode. A FIXED coefficient is
                        # a direct fact and must be recorded as such.
                        if getattr(m, "log_ent_coef", None) is not None:
                            ent = float(m.log_ent_coef.detach().exp().item())
                        else:
                            raw_coef = getattr(m, "ent_coef", None)
                            ent = (
                                float(raw_coef)
                                if isinstance(raw_coef, (int, float))
                                else float("nan")
                            )
                    except Exception:
                        ent = float("nan")
                    return actor, critic, ent

                stop_reason = "max_epochs_budget"
                for epoch in range(1, max_epochs + 1):
                    _set_env_training_progress(
                        train_env,
                        _training_progress_for_epoch(
                            epoch,
                            max_epochs=max_epochs,
                            curriculum_epochs=config.get(
                                "execution_cost_curriculum_epochs"
                            ),
                        ),
                    )
                    a_b, c_b, e_b = _policy_checksum(model)
                    nts_before = int(getattr(model, "num_timesteps", 0))
                    rb_before = int(getattr(getattr(model, "replay_buffer", None), "size", lambda: 0)()) if hasattr(model, "replay_buffer") else 0
                    # On epoch 1 we set up cleanly; on subsequent epochs use
                    # reset_num_timesteps=False to *continue* training on the
                    # same SAC instance without re-initializing the schedule.
                    model.learn(
                        total_timesteps=epoch_ts,
                        reset_num_timesteps=(epoch == 1),
                        log_interval=max(1, epoch_ts // 1000),
                        callback=make_progress_callback(config, total_progress_timesteps),
                    )
                    a_a, c_a, e_a = _policy_checksum(model)
                    nts_after = int(getattr(model, "num_timesteps", 0))
                    rb_after = int(getattr(getattr(model, "replay_buffer", None), "size", lambda: 0)()) if hasattr(model, "replay_buffer") else 0

                    train_summary = self._eval_on_split(
                        env_plugin_name, config, paths["train"], agent_plugin, model, seed, "train_epoch"
                    )
                    train_tail_summary = self._eval_on_split(
                        env_plugin_name, config, paths.get("train_tail", paths["train"]),
                        agent_plugin, model, seed, "train_tail_epoch"
                    )
                    val_summary = self._eval_on_split(
                        env_plugin_name, config, paths["val"], agent_plugin, model, seed, "validation_epoch"
                    )
                    selection_metric = str(
                        config.get("selection_metric", self.params["selection_metric"])
                    )
                    risk_lambda = float(
                        config.get("risk_penalty_lambda", self.params["risk_penalty_lambda"])
                    )
                    l1_gap_beta = float(
                        config.get(
                            "l1_generalization_gap_penalty_beta",
                            self.params["l1_generalization_gap_penalty_beta"],
                        )
                    )
                    for split_summary in (train_summary, train_tail_summary, val_summary):
                        _annotate_risk_adjusted(split_summary, risk_lambda)

                    train_ret = _safe_float(train_summary.get("total_return"))
                    if math.isnan(train_ret):
                        train_ret = 0.0
                    early_stop_min_trades = int(
                        config.get("early_stop_min_trades", self.params["early_stop_min_trades"])
                    )
                    configured_train_tail_min = config.get(
                        "early_stop_min_train_tail_trades",
                        self.params["early_stop_min_train_tail_trades"],
                    )
                    configured_validation_min = config.get(
                        "early_stop_min_validation_trades",
                        self.params["early_stop_min_validation_trades"],
                    )
                    early_stop_min_train_tail_trades = int(
                        early_stop_min_trades
                        if configured_train_tail_min is None
                        else configured_train_tail_min
                    )
                    early_stop_min_validation_trades = int(
                        early_stop_min_trades
                        if configured_validation_min is None
                        else configured_validation_min
                    )
                    no_trade_penalty = float(
                        config.get(
                            "early_stop_no_trade_penalty",
                            self.params["early_stop_no_trade_penalty"],
                        )
                    )
                    (
                        composite,
                        composite_raw,
                        trade_gate_passed,
                        train_tail_ret,
                        val_ret,
                        train_tail_trades,
                        val_trades,
                    ) = _early_stop_composite(
                        train_tail_summary,
                        val_summary,
                        min_trades=early_stop_min_trades,
                        min_train_tail_trades=early_stop_min_train_tail_trades,
                        min_validation_trades=early_stop_min_validation_trades,
                        no_trade_penalty=no_trade_penalty,
                        selection_metric=selection_metric,
                        risk_lambda=risk_lambda,
                        gap_penalty_beta=l1_gap_beta,
                    )
                    selection_details = _selection_pair_details(
                        train_tail_summary,
                        val_summary,
                        selection_metric=selection_metric,
                        risk_lambda=risk_lambda,
                        gap_penalty_beta=l1_gap_beta,
                    )

                    checkpoint_eligible = _checkpoint_is_eligible(
                        num_timesteps=nts_after,
                        minimum_timesteps=l1_min_checkpoint_timesteps,
                        trade_gate_passed=trade_gate_passed,
                    )
                    patience_eligible = (
                        checkpoint_eligible
                        and epoch > l1_patience_start_epoch
                    )
                    # AUD-F1-20260806-127: activity-ineligible epochs
                    # consume their OWN bounded budget. They are never
                    # charged to improvement patience — doing so would
                    # reward a candidate for emitting trivial trades to
                    # survive, which is selection pressure toward noise.
                    # A candidate that cannot become active after the
                    # warm-up is terminated and rejected, not run to the
                    # hard epoch cap.
                    if checkpoint_eligible:
                        activity_ineligible_streak = 0
                    elif epoch > activity_patience_start_epoch:
                        activity_ineligible_streak += 1
                    best_composite, no_improve, improved = _update_l1_checkpoint_state(
                        composite=composite,
                        best_composite=best_composite,
                        no_improve=no_improve,
                        min_delta=l1_min_delta,
                        eligible=checkpoint_eligible,
                        patience_eligible=patience_eligible,
                    )
                    if improved:
                        agent_plugin.save(model, best_model_path)
                        best_checkpoint_saved = True

                    # AUD-P1LR-20260815-235: type the actor at THIS
                    # epoch. The probe combines uniformly distributed
                    # samples from the fit and validation rollouts that
                    # were just scored, so the typed fact covers both
                    # sides of the L1 contract.
                    liveness_observations = (
                        _liveness.combine_observation_batches(
                            self._liveness_observations.get("train_epoch"),
                            self._liveness_observations.get(
                                "validation_epoch"),
                        )
                    )
                    liveness = _liveness.actor_liveness_facts(
                        model=model,
                        observations=liveness_observations,
                        action_raw_std=val_summary.get("action_raw_std"),
                        epoch=epoch,
                        split="train_epoch+validation_epoch",
                        phase=str(config.get("solvency_mode")
                                  or "normal_realistic"),
                        min_live_unit_fraction=liveness_min_live_fraction,
                    )
                    actor_liveness_history.append(liveness)

                    history.append({
                        "epoch": epoch,
                        "actor_liveness": liveness,
                        "actor_liveness_classification":
                            liveness["classification"],
                        "actor_live_unit_fraction":
                            liveness.get("live_unit_fraction"),
                        "actor_constant_policy":
                            liveness.get("constant_policy"),
                        "train_total_return": train_ret,
                        "train_tail_total_return": train_tail_ret,
                        "val_total_return": val_ret,
                        "selection_metric": selection_metric,
                        "risk_penalty_lambda": risk_lambda,
                        "l1_generalization_gap_penalty_beta": l1_gap_beta,
                        "train_tail_risk_adjusted_total_return": train_tail_summary.get(
                            "risk_adjusted_total_return"
                        ),
                        "val_risk_adjusted_total_return": val_summary.get(
                            "risk_adjusted_total_return"
                        ),
                        "train_tail_max_drawdown_fraction": train_tail_summary.get(
                            "max_drawdown_fraction"
                        ),
                        "val_max_drawdown_fraction": val_summary.get(
                            "max_drawdown_fraction"
                        ),
                        **selection_details,
                        "composite_raw": composite_raw,
                        "composite": composite,
                        "checkpoint_improved": improved,
                        "best_composite": best_composite if best_checkpoint_saved else None,
                        "l1_checkpoint_eligible": checkpoint_eligible,
                        "l1_patience_eligible": patience_eligible,
                        "l1_patience_start_epoch": l1_patience_start_epoch,
                        "l1_min_checkpoint_timesteps": l1_min_checkpoint_timesteps,
                        "early_stop_trade_gate_passed": trade_gate_passed,
                        "early_stop_min_trades": early_stop_min_trades,
                        "early_stop_min_train_tail_trades": (
                            early_stop_min_train_tail_trades
                        ),
                        "early_stop_min_validation_trades": (
                            early_stop_min_validation_trades
                        ),
                        "l1_patience_used": no_improve,
                        "l1_patience_max": l1_patience,
                        "policy_actor_l1_before": a_b,
                        "policy_actor_l1_after": a_a,
                        "policy_actor_delta": a_a - a_b,
                        "policy_critic_l1_before": c_b,
                        "policy_critic_l1_after": c_a,
                        "policy_critic_delta": c_a - c_b,
                        "ent_coef": e_a,
                        # M0 order §7: per-epoch training telemetry. The
                        # eval rollouts already measure raw/thresholded
                        # action behavior; copy the validation-side facts
                        # into the durable history instead of losing
                        # them, and record the SB3 counters/losses that
                        # D1 never captured. Absent facts stay None —
                        # never zero.
                        "replay_buffer_size": (
                            int(model.replay_buffer.size())
                            if getattr(model, "replay_buffer", None)
                            is not None else None
                        ),
                        "gradient_updates_total": (
                            int(getattr(model, "_n_updates"))
                            if hasattr(model, "_n_updates") else None
                        ),
                        "actor_loss": _safe_float_or_none(
                            getattr(model, "logger", None)
                            and model.logger.name_to_value.get(
                                "train/actor_loss")),
                        "critic_loss": _safe_float_or_none(
                            getattr(model, "logger", None)
                            and model.logger.name_to_value.get(
                                "train/critic_loss")),
                        "ent_coef_loss": _safe_float_or_none(
                            getattr(model, "logger", None)
                            and model.logger.name_to_value.get(
                                "train/ent_coef_loss")),
                        "val_action_raw_mean": _safe_float_or_none(
                            val_summary.get("action_raw_mean")),
                        "val_action_raw_std": _safe_float_or_none(
                            val_summary.get("action_raw_std")),
                        "val_action_raw_min": _safe_float_or_none(
                            val_summary.get("action_raw_min")),
                        "val_action_raw_max": _safe_float_or_none(
                            val_summary.get("action_raw_max")),
                        "val_action_non_hold_rate": _safe_float_or_none(
                            val_summary.get("action_non_hold_rate")),
                        "val_action_dominant_rate": _safe_float_or_none(
                            val_summary.get("action_dominant_rate")),
                        "val_action_deadband_rate": _safe_float_or_none(
                            val_summary.get("action_deadband_rate")),
                        "val_entry_orders_submitted": val_summary.get(
                            "execution_entry_orders_submitted"),
                        "val_no_trade_diagnosis": val_summary.get(
                            "no_trade_diagnosis"),
                        "train_trades": int(_safe_float(train_summary.get("trades_total")) or 0),
                        "train_win_pct": _win_pct(train_summary),
                        "train_sharpe": _safe_float(train_summary.get("sharpe_ratio")),
                        "train_profit_pct": train_ret * 100.0,
                        "train_balance": _safe_float(train_summary.get("final_equity")),
                        "train_tail_trades": train_tail_trades,
                        "train_tail_win_pct": _win_pct(train_tail_summary),
                        "train_tail_sharpe": _safe_float(train_tail_summary.get("sharpe_ratio")),
                        "train_tail_profit_pct": train_tail_ret * 100.0,
                        "train_tail_balance": _safe_float(train_tail_summary.get("final_equity")),
                        "val_trades": val_trades,
                        "val_win_pct": _win_pct(val_summary),
                        "val_sharpe": _safe_float(val_summary.get("sharpe_ratio")),
                        "val_profit_pct": val_ret * 100.0,
                        "val_balance": _safe_float(val_summary.get("final_equity")),
                    })

                    # AUD-F1-20260806-127: the label must distinguish
                    # "still warming up" from "produced no eligible
                    # trading activity", which are different states.
                    if patience_eligible:
                        l1_status = f"{no_improve}/{l1_patience}"
                    elif checkpoint_eligible:
                        l1_status = (
                            f"epoch-warmup<={l1_patience_start_epoch}")
                    elif int(nts_after) < int(
                            l1_min_checkpoint_timesteps):
                        l1_status = (
                            f"step-warmup<{l1_min_checkpoint_timesteps}")
                    else:
                        l1_status = (
                            f"no-activity {activity_ineligible_streak}"
                            f"/{activity_patience}")
                    checkpoint_status = (
                        "(IMPROVED, model saved)"
                        if improved
                        else "(checkpoint ineligible)" if not checkpoint_eligible else ""
                    )
                    print(
                        f"[epoch {epoch:>3}/{max_epochs}] "
                        f"L1 {l1_status}  "
                        f"L2 {l2_counter}/{l2_patience}  "
                        f"{selection_metric} composite={composite:+.4f} raw={composite_raw:+.4f} "
                        f"trade_gate={'PASS' if trade_gate_passed else 'FAIL'} "
                        f"best={best_composite:+.4f} "
                        f"{checkpoint_status} "
                        f"actor|w|={a_a:.2f} Δa={a_a-a_b:+.4f} "
                        f"critic|w|={c_a:.2f} Δc={c_a-c_b:+.4f} ent={e_a:.4f} "
                        f"steps={nts_before}->{nts_after} buf={rb_before}->{rb_after}",
                        flush=True,
                    )
                    print(
                        f"            TRAIN trades={int(_safe_float(train_summary.get('trades_total')) or 0):>4} "
                        f"win%={_win_pct(train_summary):>5.2f} "
                        f"sharpe={_safe_float(train_summary.get('sharpe_ratio')):+.4f} "
                        f"profit={train_ret*100:+.2f}% "
                        f"bal={_safe_float(train_summary.get('final_equity')):.2f}",
                        flush=True,
                    )
                    print(
                        f"            TRAIN_TAIL trades={train_tail_trades:>4} "
                        f"win%={_win_pct(train_tail_summary):>5.2f} "
                        f"sharpe={_safe_float(train_tail_summary.get('sharpe_ratio')):+.4f} "
                        f"profit={train_tail_ret*100:+.2f}% "
                        f"bal={_safe_float(train_tail_summary.get('final_equity')):.2f}",
                        flush=True,
                    )
                    print(
                        f"            VAL   trades={int(_safe_float(val_summary.get('trades_total')) or 0):>4} "
                        f"win%={_win_pct(val_summary):>5.2f} "
                        f"sharpe={_safe_float(val_summary.get('sharpe_ratio')):+.4f} "
                        f"profit={val_ret*100:+.2f}% "
                        f"bal={_safe_float(val_summary.get('final_equity')):.2f}",
                        flush=True,
                    )
                    if liveness["classification"] != _liveness.ALIVE:
                        print(
                            "            "
                            + _liveness.liveness_summary_line(liveness),
                            flush=True,
                        )
                    # Conservative policy: no first-layer unit fired on the
                    # combined fit/validation probe. The record preserves the
                    # sampled support; it does not claim unobserved rows were
                    # examined.
                    _liveness.assert_actor_alive(
                        liveness,
                        refuse_dead=liveness_refuse_dead,
                        refuse_constant=liveness_refuse_constant)

                    if patience_eligible and no_improve >= l1_patience:
                        stop_reason = "l1_early_stop"
                        print(
                            f"[train] L1 EARLY STOP at epoch {epoch} "
                            f"(no improvement for {no_improve} epochs, patience={l1_patience})",
                            flush=True,
                        )
                        break

                    # AUD-F1-20260806-127: stop burning GPU on a
                    # candidate that never becomes activity-eligible.
                    if (
                        activity_patience > 0
                        and activity_ineligible_streak >= activity_patience
                    ):
                        stop_reason, activity_stop_reason = (
                            _activity_stop_disposition(
                                best_checkpoint_saved=best_checkpoint_saved,
                                streak=activity_ineligible_streak,
                                start_epoch=activity_patience_start_epoch,
                                budget=activity_patience,
                            )
                        )
                        print(
                            f"[train] ACTIVITY STOP at epoch {epoch}:"
                            f" {activity_stop_reason}", flush=True)
                        break

                if not best_checkpoint_saved:
                    stop_detail = (
                        "training ended before an activity-eligible L1 "
                        "checkpoint became available: "
                        f"num_timesteps={int(getattr(model, 'num_timesteps', 0))}, "
                        f"l1_min_checkpoint_timesteps={l1_min_checkpoint_timesteps}; "
                        "train-tail and validation trade gates must both pass"
                        + (
                            f"; {activity_stop_reason}"
                            if activity_stop_reason else ""
                        )
                    )
                    if not bool(config.get(
                            "inactive_terminal_is_typed_result", False)):
                        raise RuntimeError(stop_detail)
                    # Doc 38 / repair spec §7.1: in a matched factorial a
                    # cell whose policy never becomes activity-eligible
                    # is a MEASURED OUTCOME ("inactive"), not a harness
                    # failure. Save the terminal weights, evaluate them,
                    # and return a typed result so the cell record lands
                    # and aggregation judges activity directly. Raising
                    # here killed the whole seed's remaining cells
                    # (observed: seeds 202/303, 2026-08-09).
                    print(
                        f"[train] INACTIVE TERMINAL RESULT: {stop_detail}",
                        flush=True)
                    terminal_model_path = str(
                        Path(best_model_path).with_suffix("")
                    ) + ".terminal.zip"
                    agent_plugin.save(model, terminal_model_path)
                    final = self._final_eval(
                        agent_plugin, model, train_env,
                        env_plugin_name, paths, config, agent_plugin,
                    )
                    final["mode"] = mode
                    final["history"] = history
                    final["best_composite"] = best_composite
                    final["best_model_path"] = None
                    final["activity_stopped_without_eligible_checkpoint"] = (
                        True)
                    final["stop_reason"] = stop_reason
                    final["termination_cause"] = stop_detail
                    final["artifacts"] = {
                        "best_checkpoint": None,
                        "terminal": {
                            "path": str(
                                Path(terminal_model_path).resolve()),
                            "sha256": _verify_artifact_sha256(
                                Path(terminal_model_path), None),
                            "num_timesteps": int(
                                getattr(model, "num_timesteps", 0)),
                        },
                    }
                    final["terminal_model_path"] = str(
                        Path(terminal_model_path).resolve())
                    final["oracle_behavior_pretrain"] = pretrain_summary
                    final["warm_start_transfer_evidence"] = getattr(
                        model, "warm_start_transfer_evidence", None)
                    final["observation_contract"] = (
                        observation_contract_application)
                    final["actor_liveness_history"] = actor_liveness_history
                    final["actor_liveness"] = (
                        actor_liveness_history[-1]
                        if actor_liveness_history else None)
                    return final

                # AUD-F1-20260806-129: preserve the TERMINAL policy —
                # the weights as they exist at the last training step —
                # BEFORE reloading the best checkpoint destroys them.
                # Both artifacts are typed, hashed and load-proven so a
                # downstream packet can evaluate both or fail loudly.
                terminal_model_path = str(
                    Path(best_model_path).with_suffix("")) + ".terminal.zip"
                agent_plugin.save(model, terminal_model_path)
                terminal_num_timesteps = int(
                    getattr(model, "num_timesteps", 0))

                # Reload best model for final evaluation.
                model = agent_plugin.load(best_model_path, train_env)

                final = self._final_eval(
                    agent_plugin, model, train_env,
                    env_plugin_name, paths, config, agent_plugin,
                )
                final["mode"] = mode
                final["history"] = history
                final["best_composite"] = best_composite
                final["stop_reason"] = stop_reason
                final["best_model_path"] = str(Path(best_model_path).resolve())
                final["artifacts"] = {
                    "best_checkpoint": {
                        "path": str(Path(best_model_path).resolve()),
                        "sha256": _verify_artifact_sha256(
                            Path(best_model_path), None),
                    },
                    "terminal": {
                        "path": str(Path(terminal_model_path).resolve()),
                        "sha256": _verify_artifact_sha256(
                            Path(terminal_model_path), None),
                        "num_timesteps": terminal_num_timesteps,
                    },
                }
                final["terminal_model_path"] = str(
                    Path(terminal_model_path).resolve())
                final["oracle_behavior_pretrain"] = pretrain_summary
                # M0 order §7: the warm-start boundary evidence lives on
                # the model object; export it or it dies with the model.
                final["warm_start_transfer_evidence"] = getattr(
                    model, "warm_start_transfer_evidence", None)
                final["observation_contract"] = (
                    observation_contract_application)
                final["actor_liveness_history"] = actor_liveness_history
                final["actor_liveness"] = (
                    actor_liveness_history[-1]
                    if actor_liveness_history else None)
                return final
            finally:
                try:
                    train_env_plugin.close()
                except Exception:
                    pass
        finally:
            if self._tempdir is not None:
                try:
                    self._tempdir.cleanup()
                except Exception:
                    pass
                self._tempdir = None

    # ------------------------------------------------------------------
    def _final_eval(
        self,
        agent_plugin,
        model,
        train_env,
        env_plugin_name: str,
        paths: Dict[str, str],
        config: Dict[str, Any],
        agent_plugin_for_wrap,
    ) -> Dict[str, Any]:
        seed = int(config.get("eval_seed", self.params["eval_seed"]))
        train_summary = self._eval_on_split(
            env_plugin_name, config, paths["train"], agent_plugin_for_wrap, model, seed, "train"
        )
        train_tail_summary = self._eval_on_split(
            env_plugin_name, config, paths.get("train_tail", paths["train"]),
            agent_plugin_for_wrap, model, seed, "train_tail"
        )
        val_summary = self._eval_on_split(
            env_plugin_name, config, paths["val"], agent_plugin_for_wrap, model, seed, "validation"
        )
        evaluate_test = bool(
            config.get("evaluate_test_split", self.params["evaluate_test_split"])
        )
        if evaluate_test:
            test_summary = self._eval_on_split(
                env_plugin_name,
                config,
                paths["test"],
                agent_plugin_for_wrap,
                model,
                seed,
                "test",
            )
        else:
            test_summary = {
                "evaluation_skipped": True,
                "skip_reason": "protected_test_disabled_for_optimization",
            }
        selection_metric = str(config.get("selection_metric", self.params["selection_metric"]))
        risk_lambda = float(config.get("risk_penalty_lambda", self.params["risk_penalty_lambda"]))
        l1_gap_beta = float(
            config.get(
                "l1_generalization_gap_penalty_beta",
                self.params["l1_generalization_gap_penalty_beta"],
            )
        )
        metric_summaries = [train_summary, train_tail_summary, val_summary]
        if evaluate_test:
            metric_summaries.append(test_summary)
        for split_summary in metric_summaries:
            _annotate_risk_adjusted(split_summary, risk_lambda)
        selection_details = _selection_pair_details(
            train_tail_summary,
            val_summary,
            selection_metric=selection_metric,
            risk_lambda=risk_lambda,
            gap_penalty_beta=l1_gap_beta,
        )
        risk_adjusted_mean = 0.5 * (
            float(train_tail_summary.get("risk_adjusted_total_return") or 0.0)
            + float(val_summary.get("risk_adjusted_total_return") or 0.0)
        )

        rows = [
            ("Train", train_summary),
            ("TrainTail", train_tail_summary),
            ("Validation", val_summary),
        ]
        if evaluate_test:
            rows.append(("Test", test_summary))
        table = _format_table(rows)
        print("\n=== Final results (best-composite checkpoint) ===")
        print(table, flush=True)

        # Pop transient evidence-bearing fields out of each split summary
        # before exporting, then build the run-level evidence index.
        metadata_items: List[Dict[str, Any]] = []
        for s in (train_summary, train_tail_summary, val_summary, test_summary):
            meta = s.pop("_return_trace_metadata", None)
            if meta is not None:
                metadata_items.append(meta)

        # Build the export payload
        out = {
            "splits": {
                "train": train_summary,
                "train_tail": train_tail_summary,
                "validation": val_summary,
                "test": test_summary,
            },
            "summary_table": table,
            "selection_metric": selection_metric,
            "risk_penalty_lambda": risk_lambda,
            "l1_generalization_gap_penalty_beta": l1_gap_beta,
            "train_validation_risk_adjusted_composite_score": risk_adjusted_mean,
            "train_validation_risk_adjusted_mean_score": risk_adjusted_mean,
            "train_validation_l1_score": selection_details["train_validation_selection_score"],
            **selection_details,
        }
        # also surface top-level metrics from validation for compatibility
        out.update({
            "total_return": val_summary.get("total_return"),
            "mean_weekly_return": val_summary.get("mean_weekly_return"),
            "annualized_return": val_summary.get("annualized_return"),
            "annual_return": val_summary.get("annual_return"),
            "annual_return_method": val_summary.get("annual_return_method"),
            "mean_weekly_rap": val_summary.get("mean_weekly_rap"),
            "annual_rap": val_summary.get("annual_rap"),
            "annual_rap_method": val_summary.get("annual_rap_method"),
            "evaluation_weeks": val_summary.get("evaluation_weeks"),
            "evaluation_days": val_summary.get("evaluation_days"),
            "robust_weekly_rap_fitness": val_summary.get(
                "robust_weekly_rap_fitness"
            ),
            "worst_scenario_weekly_rap": val_summary.get(
                "worst_scenario_weekly_rap"
            ),
            "lower_tail_cvar_weekly_rap": val_summary.get(
                "lower_tail_cvar_weekly_rap"
            ),
            "scenario_weekly_rap_dispersion": val_summary.get(
                "scenario_weekly_rap_dispersion"
            ),
            "risk_adjusted_total_return": val_summary.get("risk_adjusted_total_return"),
            "max_drawdown_fraction": val_summary.get("max_drawdown_fraction"),
            "final_equity": val_summary.get("final_equity"),
            "max_drawdown_pct": val_summary.get("max_drawdown_pct"),
            "sharpe_ratio": val_summary.get("sharpe_ratio"),
            "trades_total": val_summary.get("trades_total"),
            "episode_reward": val_summary.get("episode_reward"),
            "episode_length": val_summary.get("episode_length"),
            "eval_seed": seed,
        })

        if metadata_items:
            evidence = _trace_mod.build_return_trace_evidence(
                metadata_items,
                config=config,
                run_id=_trace_mod.make_run_id(config),
                pipeline_plugin="rl_pipeline_with_validation",
            )
            trace_dir = config.get("return_trace_dir")
            evidence_path = _trace_mod.derive_evidence_path(
                trace_dir=str(trace_dir) if trace_dir else None,
                trace_file=metadata_items[0].get("trace_file"),
            )
            evidence["evidence_file"] = _trace_mod.write_return_trace_evidence(
                evidence, evidence_path,
            )
            out["return_trace_evidence"] = evidence
            out["return_trace_evidence_file"] = evidence["evidence_file"]

        # Save results.json next to the model file
        model_path = config.get("save_model")
        if model_path and bool(
            config.get("write_results_sidecar", self.params["write_results_sidecar"])
        ):
            results_path = Path(model_path).with_name("results.json")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with results_path.open("w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, default=str)
            print(f"[results] wrote {results_path}", flush=True)
        return out
