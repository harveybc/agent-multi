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


BOUNDED_LABEL = "BOUNDED_120_40_40_DAY_SCHEDULER_SCREEN"

# The plateau spec an arm may differ by (AUD-F1-20260821-PLR-06).
PREDECLARED_PLATEAU_SPEC = {
    "factor": 0.5, "lr_patience": 20, "min_lr": 1e-6,
    "threshold": 1e-6, "cooldown": 0, "start_epoch": 40}


def _split_identity(doc: dict) -> dict:
    """Split identity = rows + boundary timestamps per INPUT role.

    Trace sha256 is a model-rollout OUTPUT and legitimately differs
    between arms; it does not participate in pair identity."""
    traces = ((doc.get("split_facts") or {}).get("traces")) or {}
    out = {}
    for role in ("train_epoch", "train_tail_epoch", "validation_epoch"):
        t = traces.get(role) or {}
        out[role] = {"rows": t.get("rows"),
                     "first_timestamp": t.get("first_timestamp"),
                     "last_timestamp": t.get("last_timestamp")}
    return out


def _verify_arm_semantics(facts: dict, doc: dict, policy: str,
                          report_path: Path) -> None:
    history = doc.get("history") or []
    reductions = facts["lr_reductions"]
    if policy == "fixed":
        if reductions:
            raise ScreenAggregationError(
                f"{report_path}: fixed arm shows LR reductions "
                f"{reductions}; not a fixed arm (PLR-06)")
        lrs = {row.get("observed_learning_rates", {}).get("actor")
               for row in history
               if row.get("observed_learning_rates")}
        if len(lrs) > 1:
            raise ScreenAggregationError(
                f"{report_path}: fixed arm observed multiple actor "
                f"LRs {sorted(lrs)}; scheduler leaked in")
    else:
        if not any(row.get("plateau_lr") is not None
                   for row in history):
            raise ScreenAggregationError(
                f"{report_path}: plateau arm carries no scheduler "
                "records; machinery absent (PLR-06)")
        for r in reductions:
            if abs(r["new_lr"] - r["old_lr"] * 0.5) > 1e-18:
                raise ScreenAggregationError(
                    f"{report_path}: reduction at epoch {r['epoch']} "
                    f"is not the predeclared halving: {r}")
            if r["new_lr"] < PREDECLARED_PLATEAU_SPEC["min_lr"]:
                raise ScreenAggregationError(
                    f"{report_path}: reduction below min_lr: {r}")


def verify_pair(seed: int, fixed_doc: dict, plateau_doc: dict,
                fixed_facts: dict, plateau_facts: dict,
                fixed_path: Path, plateau_path: Path) -> dict:
    """PLR-06: exact pair identity; arms differ only as predeclared."""
    if fixed_facts["report_sha256"] == plateau_facts["report_sha256"]:
        raise ScreenAggregationError(
            f"seed {seed}: identical report files supplied for both "
            "arms; duplicate evidence refused")
    contracts = {}
    for label, doc, path in (("fixed", fixed_doc, fixed_path),
                             ("plateau", plateau_doc, plateau_path)):
        # §C.6 (post-outage order): the 93880beb compatibility path
        # was removed after committing the one migrated screen result.
        # Every report must carry its canonical contracts explicitly.
        pair = doc.get("pair_contract")
        arm = doc.get("arm_contract")
        if not pair or not arm:
            raise ScreenAggregationError(
                f"{path}: missing canonical pair_contract/arm_contract; "
                "the frozen-tip derivation path was retired (§C.6)")
        contracts[label] = {"pair": pair, "arm": arm}
        if pair.get("seed") != seed:
            raise ScreenAggregationError(
                f"{path}: pair_contract seed {pair.get('seed')!r} does "
                f"not match filename seed {seed}; swapped or "
                "mislabelled report (PLR-06)")
        if not doc.get("accepted", False):
            raise ScreenAggregationError(
                f"{path}: arm not accepted; incomplete or typed-"
                "negative arms cannot enter a directional outcome")
        label_str = ((doc.get("stopping_contract") or {}).get(
            "classification"))
        if label_str != BOUNDED_LABEL:
            raise ScreenAggregationError(
                f"{path}: classification {label_str!r} is not the "
                "bounded screen label (PLR-02/PLR-06; the legacy "
                "exception was retired in §C.6)")
    pf, pp = contracts["fixed"]["pair"], contracts["plateau"]["pair"]
    if pf != pp:
        diff = {k: (pf.get(k), pp.get(k))
                for k in set(pf) | set(pp) if pf.get(k) != pp.get(k)}
        raise ScreenAggregationError(
            f"seed {seed}: pair_contract mismatch between arms: "
            f"{diff}; extra factor differences refuse (PLR-06)")
    if _split_identity(fixed_doc) != _split_identity(plateau_doc):
        raise ScreenAggregationError(
            f"seed {seed}: split rows/timestamps differ between arms")
    af, ap = contracts["fixed"]["arm"], contracts["plateau"]["arm"]
    if af.get("scheduler_policy") != "fixed":
        raise ScreenAggregationError(
            f"seed {seed}: fixed-labelled report declares policy "
            f"{af.get('scheduler_policy')!r}; swapped arms (PLR-06)")
    if ap.get("scheduler_policy") != "plateau":
        raise ScreenAggregationError(
            f"seed {seed}: plateau-labelled report declares policy "
            f"{ap.get('scheduler_policy')!r}; swapped arms (PLR-06)")
    spec = ap.get("plateau_spec")
    if spec != PREDECLARED_PLATEAU_SPEC:
        raise ScreenAggregationError(
            f"seed {seed}: plateau arm spec {spec!r} is not the "
            f"predeclared {PREDECLARED_PLATEAU_SPEC!r}")
    _verify_arm_semantics(fixed_facts, fixed_doc, "fixed", fixed_path)
    _verify_arm_semantics(plateau_facts, plateau_doc, "plateau",
                          plateau_path)
    return contracts


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
        fixed_doc = json.loads(fixed.read_text())
        plateau_doc = json.loads(plateau.read_text())
        contracts = verify_pair(seed, fixed_doc, plateau_doc, f, p,
                                fixed, plateau)
        pairs[seed] = {
            "identity": contracts,
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

    hashes = [pairs[s][arm]["report_sha256"]
              for s in SEEDS for arm in ("fixed", "plateau")]
    if len(set(hashes)) != len(hashes):
        raise ScreenAggregationError(
            "duplicate report content across arms/seeds; refused "
            "(PLR-06)")

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
