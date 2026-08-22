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
import json
import math
import statistics
import sys
from pathlib import Path

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


def _load_history(path: Path, seed: int) -> list:
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
    return hist, doc.get("data_sha256")


def diagnose_seed(fixed_path: Path, plateau_path: Path,
                  seed: int) -> dict:
    fh, f_sha = _load_history(fixed_path, seed)
    ph, p_sha = _load_history(plateau_path, seed)
    if f_sha != p_sha:
        raise DiagnosticError(
            f"seed {seed}: arms carry different data_sha256; "
            "mismatched pair identity")
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
    # Pre-intervention prefix must be bit-identical: same seed, same
    # everything, LR untouched before the first reduction.
    prefix_max_abs_diff = 0.0
    n_prefix = min(t0, len(fh), len(ph))
    for i in range(n_prefix):
        a = _finite("fixed composite", fh[i]["composite"], epoch=i + 1)
        b = _finite("plateau composite", ph[i]["composite"], epoch=i + 1)
        prefix_max_abs_diff = max(prefix_max_abs_diff, abs(a - b))
    if prefix_max_abs_diff != 0.0:
        raise DiagnosticError(
            f"seed {seed}: pre-intervention prefix differs (max abs "
            f"diff {prefix_max_abs_diff}); the arms are not the same "
            "experiment before epoch "
            f"{t0} — changed prefix refuses")
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
        "prefix_max_abs_diff": 0.0,
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
