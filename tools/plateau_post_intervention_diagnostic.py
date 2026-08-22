"""POST_HOC_EXPLORATORY intervention diagnostic over the closed screen.

Order: MUSASHI_TO_GENERAL_SATOSHI_PLR_CLOSURE_VERDICT_AND_NEXT_ORDER_2026_08_22 WP1
Finding: AUD-F1-20260822-PLR-07.

Consumes the SAME eight committed reports as the official aggregation
and extracts what the predeclared global-best endpoint could not see:
the already-paid post-intervention trajectories. Every emitted number
is labeled ``POST_HOC_EXPLORATORY``. This tool may not — and cannot —
mutate the official ``INCONCLUSIVE`` aggregate, select a checkpoint,
or create promotion authority: it writes one diagnostic JSON and
nothing else.

Deterministic by construction: pure functions over committed files;
no clock, no randomness, no network.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

_AGG_SPEC = importlib.util.spec_from_file_location(
    "plateau_screen_aggregate",
    Path(__file__).resolve().parent / "plateau_screen_aggregate.py")
_agg = importlib.util.module_from_spec(_AGG_SPEC)
_AGG_SPEC.loader.exec_module(_agg)

# C1 (AUD-F1-20260822-PLR-08): the canonical pre-intervention
# projection. Every non-treatment per-epoch fact — economic,
# observation/action, optimization-state and model-movement — must be
# EXACTLY equal across the pair for every pre-treatment epoch, or the
# diagnostic refuses. The ONLY excluded field is the treatment record
# itself.
TREATMENT_FIELDS = frozenset({"plateau_lr"})
PROJECTION_FIELDS = (
    "composite", "composite_raw",
    "val_total_return", "val_trades", "val_max_drawdown_fraction",
    "val_sharpe", "val_balance", "val_profit_pct", "val_win_pct",
    "val_action_raw_std", "val_action_raw_mean",
    "val_action_non_hold_rate", "val_action_deadband_rate",
    "train_total_return", "train_trades", "train_sharpe",
    "train_tail_total_return", "train_tail_trades",
    "train_tail_max_drawdown_fraction", "train_tail_sharpe",
    "policy_actor_delta", "policy_critic_delta",
    "policy_actor_l1_after", "policy_critic_l1_after",
    "ent_coef", "gradient_updates_total", "replay_buffer_size",
    "observed_learning_rates",
    "best_composite", "checkpoint_improved", "l1_checkpoint_eligible",
    "actor_loss", "critic_loss",
)

SEEDS = (101, 202, 303, 404)
LABEL = "POST_HOC_EXPLORATORY"
MONITOR_UNITS = ("risk-adjusted validation/train-tail return fraction "
                 "(easy_checkpoint_monitor composite)")
AUC_UNITS = "monitor-value-fraction x epochs (trapezoidal)"


class DiagnosticError(ValueError):
    """Typed refusal: identity, alignment or fact-integrity violation."""


def _finite(name, value, *, epoch=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticError(
            f"{name} at epoch {epoch} must be a real number, got "
            f"{value!r}")
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise DiagnosticError(
            f"{name} at epoch {epoch} must be finite, got {value!r}")
    return v


def _optional(row, key):
    """Optional per-epoch fact: absent stays typed-unavailable, never 0."""
    v = row.get(key)
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return None if (math.isnan(f) or math.isinf(f)) else f


def _load_report(path: Path, seed: int):
    doc = json.loads(path.read_text())
    if (doc.get("budgets") or {}).get("seed") != seed:
        raise DiagnosticError(
            f"{path.name}: report seed {(doc.get('budgets') or {}).get('seed')!r} "
            f"does not match filename seed {seed}; mismatched pair "
            "identity")
    hist = doc.get("history") or []
    if not hist:
        raise DiagnosticError(f"{path.name}: no history")
    for i, row in enumerate(hist, start=1):
        if row.get("epoch") != i:
            raise DiagnosticError(
                f"{path.name}: history epoch {row.get('epoch')!r} at "
                f"position {i}; non-contiguous epochs refuse (off-by-"
                "one hazard)")
    return doc, hist


def _projection(rows) -> list:
    return [{k: row.get(k) for k in PROJECTION_FIELDS}
            for row in rows]


def diagnose_seed(fixed_path: Path, plateau_path: Path,
                  seed: int) -> dict:
    fdoc, fh = _load_report(fixed_path, seed)
    pdoc, ph = _load_report(plateau_path, seed)
    # C1: ONE identity authority — the aggregator's exact pair
    # verification, with no compatibility exception of any kind. A
    # changed commit, config, data, split, device, budget, metric or
    # arm semantics refuses before any number is emitted.
    try:
        _agg.exact_pair_identity(fdoc, pdoc, seed)
    except _agg.ScreenAggregationError as exc:
        raise DiagnosticError(f"seed {seed}: {exc}") from exc
    reductions = [r["epoch"] for r in ph
                  if (r.get("plateau_lr") or {}).get("reduced")]
    if any((r.get("plateau_lr") or {}).get("reduced") for r in fh):
        raise DiagnosticError(
            f"seed {seed}: fixed arm shows LR reductions; not a valid "
            "fixed/plateau pair")
    if not reductions:
        return {"seed": seed, "label": LABEL,
                "intervention": "none — no LR reduction occurred; "
                                "no treatment window exists"}
    t0 = min(reductions)
    # C1: the pre-intervention prefix must be identical on the FULL
    # canonical projection — every non-treatment fact — not only the
    # monitor scalar.
    n_prefix = min(t0, len(fh), len(ph))
    proj_f = _projection(fh[:n_prefix])
    proj_p = _projection(ph[:n_prefix])
    for i, (ra, rb) in enumerate(zip(proj_f, proj_p), start=1):
        for k in PROJECTION_FIELDS:
            if ra[k] != rb[k]:
                raise DiagnosticError(
                    f"seed {seed}: pre-intervention projection "
                    f"differs at epoch {i} field {k!r}: "
                    f"{ra[k]!r} vs {rb[k]!r} — changed prefix refuses")
    projection_sha = hashlib.sha256(json.dumps(
        proj_f, sort_keys=True).encode()).hexdigest()
    n_aligned = min(len(fh), len(ph))
    if n_aligned <= t0:
        return {"seed": seed, "label": LABEL,
                "first_reduction_epoch": t0,
                "intervention": "reduction fired on/after the shorter "
                                "arm's terminal epoch; empty aligned "
                                "post-intervention window"}
    window = list(range(t0 + 1, n_aligned + 1))
    f_post = [_finite("fixed composite", fh[e - 1]["composite"],
                      epoch=e) for e in window]
    p_post = [_finite("plateau composite", ph[e - 1]["composite"],
                      epoch=e) for e in window]
    deltas = [p - f for p, f in zip(p_post, f_post)]
    # trapezoidal AUC of the delta curve over the aligned window
    auc = sum((deltas[i] + deltas[i + 1]) / 2.0
              for i in range(len(deltas) - 1))

    def _fact_delta(key):
        fv = _optional(fh[n_aligned - 1], key)
        pv = _optional(ph[n_aligned - 1], key)
        if fv is None or pv is None:
            return "unavailable"
        return pv - fv

    def _sum_delta(key):
        fs = [_optional(r, key) for r in fh[t0:n_aligned]]
        ps = [_optional(r, key) for r in ph[t0:n_aligned]]
        if any(v is None for v in fs + ps):
            return "unavailable"
        return sum(ps) - sum(fs)

    return {
        "seed": seed, "label": LABEL,
        "first_reduction_epoch": t0,
        "all_reduction_epochs": reductions,
        "aligned_window_epochs": [window[0], window[-1]],
        "unaligned_tail_epochs": {"fixed": len(fh) - n_aligned,
                                  "plateau": len(ph) - n_aligned},
        "prefix_identical": True,
        "prefix_projection_fields": list(PROJECTION_FIELDS),
        "prefix_projection_sha256": projection_sha,
        "treatment_fields_excluded": sorted(TREATMENT_FIELDS),
        "per_epoch_monitor_delta": [
            {"epoch": e, "delta": d} for e, d in zip(window, deltas)],
        "best_post_fixed": max(f_post),
        "best_post_plateau": max(p_post),
        "best_post_delta": max(p_post) - max(f_post),
        "terminal_delta": deltas[-1],
        "auc_delta": auc,
        "auc_units": AUC_UNITS,
        "monitor_units": MONITOR_UNITS,
        "terminal_fact_deltas": {
            "val_total_return_fraction": _fact_delta("val_total_return"),
            "val_max_drawdown_fraction": _fact_delta(
                "val_max_drawdown_fraction"),
            "val_trades": _fact_delta("val_trades"),
            "train_tail_trades": _fact_delta("train_tail_trades"),
            "val_action_raw_std": _fact_delta("val_action_raw_std"),
            "val_action_non_hold_rate": _fact_delta(
                "val_action_non_hold_rate"),
        },
        "post_window_sum_deltas": {
            "policy_actor_movement": _sum_delta("policy_actor_delta"),
            "policy_critic_movement": _sum_delta("policy_critic_delta"),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    per_seed = {}
    for seed in SEEDS:
        fixed = args.screen_dir / f"seed{seed}_fixed_report.json"
        plateau = args.screen_dir / f"seed{seed}_plateau_report.json"
        missing = [str(p) for p in (fixed, plateau) if not p.is_file()]
        if missing:
            print(json.dumps({"outcome": "REFUSED_INCOMPLETE_INPUT",
                              "missing": missing}))
            return 2
        per_seed[seed] = diagnose_seed(fixed, plateau, seed)
    measured = {s: d for s, d in per_seed.items()
                if "best_post_delta" in d}
    signs = {s: {"best_post": ("+" if d["best_post_delta"] > 0 else
                               "-" if d["best_post_delta"] < 0 else "0"),
                 "terminal": ("+" if d["terminal_delta"] > 0 else
                              "-" if d["terminal_delta"] < 0 else "0"),
                 "auc": ("+" if d["auc_delta"] > 0 else
                         "-" if d["auc_delta"] < 0 else "0")}
             for s, d in measured.items()}
    bp = [d["best_post_delta"] for d in measured.values()]
    result = {
        "schema": "agent_multi.plateau_post_intervention_diag.v1",
        "label": LABEL,
        "authority": ("NONE — exploratory diagnostic only; the official "
                      "screen outcome remains INCONCLUSIVE; no "
                      "checkpoint selection; no promotion authority"),
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "sign_table": signs,
        "dispersion_best_post_delta": ({
            "median": statistics.median(bp), "min": min(bp),
            "max": max(bp)} if bp else "unavailable"),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=1))
    print(json.dumps({"label": LABEL, "sign_table": signs,
                      "median_best_post_delta":
                          statistics.median(bp) if bp else None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
