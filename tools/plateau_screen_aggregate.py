"""Aggregate the bounded 120/40/40-day scheduler mechanism screen.

Orders: AUDIT_SATOSHI_WP3_PLATEAU_LR_2026_08_21 §Orders 2-3.

PREDECLARED DECISION RULE (fixed before any arm completed; changing it
after seeing results is adaptation and is refused by review):

- Primary causal endpoint per arm: best eligible checkpoint-monitor
  value (``composite`` at the selected epoch).
- Secondary endpoints: validation total return at the selected epoch,
  epochs to stop, selected-epoch validation trades.
- Paired delta per seed = plateau arm − fixed arm (same seed, same GPU,
  identical everything but LR policy).
- Outcome:
  - ``SHORT_SCREEN_SIGNAL_FOR_PLATEAU``  — ≥3 of 4 seeds with primary
    delta > 0 AND median primary delta > 0;
  - ``SHORT_SCREEN_SIGNAL_AGAINST``       — ≥3 of 4 seeds with primary
    delta < 0 AND median primary delta < 0;
  - ``INCONCLUSIVE``                      — anything else.
- Four seeds are never called statistically conclusive; the report
  carries direction and dispersion only.

CAUSAL EXCLUSIONS (AUD-F1-20260821-PLR-03): arm order is fixed-first on
every GPU, so wall-clock, temperature and host-load facts are
order-confounded. They are reported under ``descriptive_only`` and are
mechanically excluded from the outcome fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

SEEDS = (101, 202, 303, 404)
OUTCOMES = ("SHORT_SCREEN_SIGNAL_FOR_PLATEAU",
            "SHORT_SCREEN_SIGNAL_AGAINST", "INCONCLUSIVE")


class ScreenAggregationError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScreenAggregationError(
            f"{name} must be a finite number, got {value!r}")
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        raise ScreenAggregationError(f"{name} must be finite: {value!r}")
    return v


def arm_facts(report_path: Path) -> dict:
    doc = json.loads(report_path.read_text())
    history = doc.get("history") or []
    if not history:
        raise ScreenAggregationError(
            f"{report_path} carries no history; refusing")
    eligible = [h for h in history if h.get("l1_checkpoint_eligible")
                and h.get("checkpoint_improved")]
    if not eligible:
        raise ScreenAggregationError(
            f"{report_path}: no eligible improved checkpoint; typed "
            "refusal, not a zero")
    best = max(eligible, key=lambda h: _finite("composite",
                                               h["composite"]))
    reductions = [
        {"epoch": h["epoch"],
         "old_lr": h["plateau_lr"]["old_lr"],
         "new_lr": h["plateau_lr"]["new_lr"]}
        for h in history
        if (h.get("plateau_lr") or {}).get("reduced")]
    lrs = [((h.get("observed_learning_rates") or {}).get("actor"))
           for h in history]
    return {
        "report": str(report_path),
        "report_sha256": _sha(report_path),
        "stop_reason": doc.get("stop_reason"),
        "epochs_run": doc.get("epochs_run"),
        "best_epoch": best["epoch"],
        "best_monitor_value": _finite("composite", best["composite"]),
        "best_val_total_return": _finite(
            "val_total_return", best["val_total_return"]),
        "best_val_trades": best.get("val_trades"),
        "best_train_tail_trades": best.get("train_tail_trades"),
        "best_val_max_drawdown_fraction": best.get(
            "val_max_drawdown_fraction"),
        "lr_reductions": reductions,
        "observed_actor_lr_first": lrs[0] if lrs else None,
        "observed_actor_lr_last": lrs[-1] if lrs else None,
        "monitor_curve_len": len(history),
        "data_sha256": doc.get("data_sha256"),
        "descriptive_only": {
            "elapsed_seconds": doc.get("elapsed_seconds"),
            "excluded_from_causal_conclusion": True,
            "reason": ("arm order is not counterbalanced "
                       "(AUD-F1-20260821-PLR-03)")},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)

    pairs = {}
    for seed in SEEDS:
        fixed = args.screen_dir / f"seed{seed}_fixed_report.json"
        plateau = args.screen_dir / f"seed{seed}_plateau_report.json"
        missing = [str(p) for p in (fixed, plateau) if not p.is_file()]
        if missing:
            print(json.dumps({"outcome": "REFUSED_INCOMPLETE_SCREEN",
                              "missing": missing}))
            return 2
        f, p = arm_facts(fixed), arm_facts(plateau)
        if f.get("data_sha256") != p.get("data_sha256"):
            raise ScreenAggregationError(
                f"seed {seed}: arms trained on different data hashes")
        pairs[seed] = {
            "fixed": f, "plateau": p,
            "delta_primary_best_monitor_value":
                p["best_monitor_value"] - f["best_monitor_value"],
            "delta_best_val_total_return":
                p["best_val_total_return"] - f["best_val_total_return"],
            "delta_epochs_run":
                (p["epochs_run"] or 0) - (f["epochs_run"] or 0),
            "delta_best_val_trades":
                (p["best_val_trades"] or 0) - (f["best_val_trades"] or 0),
        }

    primary = [pairs[s]["delta_primary_best_monitor_value"]
               for s in SEEDS]
    pos = sum(1 for d in primary if d > 0)
    neg = sum(1 for d in primary if d < 0)
    med = statistics.median(primary)
    if pos >= 3 and med > 0:
        outcome = OUTCOMES[0]
    elif neg >= 3 and med < 0:
        outcome = OUTCOMES[1]
    else:
        outcome = OUTCOMES[2]

    result = {
        "schema": "agent_multi.plateau_screen_aggregate.v1",
        "screen_label": ("bounded 120/40/40-day scheduler mechanism "
                         "screen (AUD-F1-20260821-PLR-02); no claim "
                         "about the multi-year easy curriculum"),
        "predeclared_rule": ("primary=best eligible monitor value; "
                             "FOR if >=3/4 seeds delta>0 and median>0; "
                             "AGAINST mirrored; else INCONCLUSIVE; "
                             "4 seeds are never conclusive"),
        "outcome": outcome,
        "primary_deltas_by_seed": {str(s): pairs[s][
            "delta_primary_best_monitor_value"] for s in SEEDS},
        "dispersion": {
            "median": med,
            "min": min(primary), "max": max(primary),
            "positive_seeds": pos, "negative_seeds": neg},
        "pairs": pairs,
        "promotion": "REFUSED — no checkpoint from this screen may be "
                     "promoted (audit 2026-08-21 ML verdict)",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=1))
    print(json.dumps({"outcome": outcome,
                      "median_primary_delta": med,
                      "positive_seeds": pos, "negative_seeds": neg}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
