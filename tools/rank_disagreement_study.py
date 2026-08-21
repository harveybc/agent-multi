#!/usr/bin/env python3
"""WP1 (order 2026-08-21): zero-GPU rank disagreement study.

Consumes a smoke report WITH per-epoch history (durable evidence),
scores every epoch under BOTH contracts, ranks, and reports the rank
delta plus adversarial fixtures. Mechanically asserts no test fact was
loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _easy_contracts as ec  # noqa: E402

CAL = {"activity_plateau_low_rate": 50.0,
       "activity_plateau_high_rate": 300.0}

ADVERSARIAL = {
    "zero_trades": dict(trades=0, val_ret=0.0, val_dd=0.0, tt=0.0),
    "one_losing_trade": dict(trades=1, val_ret=-0.02, val_dd=0.02,
                             tt=-0.01),
    "active_moderate_loss": dict(trades=120, val_ret=-0.05,
                                 val_dd=0.10, tt=-0.04),
    "catastrophic_loss": dict(trades=120, val_ret=-0.95, val_dd=0.95,
                              tt=-0.90),
    "overtrading": dict(trades=5000, val_ret=0.01, val_dd=0.05,
                        tt=0.02),
    "equal_activity_low_risk": dict(trades=120, val_ret=0.05,
                                    val_dd=0.02, tt=0.05),
    "equal_activity_high_risk": dict(trades=120, val_ret=0.05,
                                     val_dd=0.30, tt=0.05),
    "equal_econ_small_gap": dict(trades=120, val_ret=0.05,
                                 val_dd=0.05, tt=0.06),
    "equal_econ_big_gap": dict(trades=120, val_ret=0.05, val_dd=0.05,
                               tt=0.90),
}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    history = report.get("history") or []
    if not history:
        print(json.dumps({"outcome": "REFUSED_NO_HISTORY",
                          "detail": "the report carries no per-epoch "
                          "history; durable evidence is required"}))
        return 2

    forbidden = [k for row in history for k in row
                 if "test" in str(k).lower()]
    rows = []
    for row in history:
        epoch = row.get("epoch")
        # EC-02: RAW values into the strict validators — no defaulting,
        # no int()/float() pre-coercion. Any missing, boolean, string,
        # fractional, negative or non-finite fact REFUSES the study.
        # Replication defect 2026-08-21: the EC-02 rewrite read keys
        # ("train_tail_return", ...) that the EXECUTING pipeline history
        # never produces, so the strict path refused every real report
        # and the corrected happy path had never run end to end. The
        # binding below names the canonical rl_pipeline_with_validation
        # history keys verbatim; there is no aliasing or fallback — a
        # report without them still refuses.
        raw = {"train_tail_trades": row.get("train_tail_trades"),
               "val_trades": row.get("val_trades"),
               "train_tail_total_return":
                   row.get("train_tail_total_return"),
               "val_total_return": row.get("val_total_return"),
               "train_tail_max_drawdown_fraction":
                   row.get("train_tail_max_drawdown_fraction"),
               "val_max_drawdown_fraction":
                   row.get("val_max_drawdown_fraction")}
        try:
            monitor = ec.easy_checkpoint_monitor(
                train_tail_return=raw["train_tail_total_return"],
                validation_return=raw["val_total_return"],
                train_tail_drawdown=raw[
                    "train_tail_max_drawdown_fraction"],
                validation_drawdown=raw["val_max_drawdown_fraction"])
            fit = ec.easy_doin_candidate_fitness(
                closed_trades=raw["val_trades"],
                scored_rows=2190,
                validation_return=raw["val_total_return"],
                validation_drawdown=raw["val_max_drawdown_fraction"],
                train_tail_return=raw["train_tail_total_return"],
                activity_config=CAL)
        except ec.EasyContractError as error:
            print(json.dumps({
                "outcome": "REFUSED_EPOCH_FACTS",
                "epoch": epoch, "detail": str(error)[:160],
                "raw": {k: repr(v) for k, v in raw.items()}}))
            return 2
        mc, fc = monitor["components"], fit["components"]
        rows.append({
            "epoch": epoch,
            # EC-04: every raw input first-class
            **{f"raw_{k}": v for k, v in raw.items()},
            # decomposed monitor outputs
            "monitor_value": monitor["value"],
            "monitor_train_tail_rap": mc["train_tail_rap"],
            "monitor_validation_rap": mc["validation_rap"],
            "monitor_gap": mc["gap"],
            "monitor_gap_penalty": mc["gap_penalty"],
            # decomposed fitness outputs
            "fitness_activity_band": fc["activity_band"],
            "fitness_annualized_rate": fc["annualized_rate"],
            "fitness_activity_utility": fc["activity_utility"],
            "fitness_validation_economics":
                fc["validation_economics"],
            "fitness_gap_bounded": fc["gap_bounded"],
            "fitness_lex_key": json.dumps(fit["lex_key"]),
            "eligible": fit["eligible"], "reason": fit["reason"],
            "monitor_contract_id": monitor["contract_id"],
            "fitness_contract_id": fit["contract_id"],
            "source_report": str(args.report),
        })
    import hashlib as _hashlib
    report_sha = _hashlib.sha256(
        args.report.read_bytes()).hexdigest()
    for row in rows:
        row["source_report_sha256"] = report_sha
    by_monitor = sorted(
        [r for r in rows if r["monitor_value"] is not None],
        key=lambda r: r["monitor_value"], reverse=True)
    by_fitness = sorted(
        [r for r in rows if r["eligible"]],
        key=lambda r: json.loads(r["fitness_lex_key"]), reverse=True)
    for rank, row in enumerate(by_monitor, 1):
        row["monitor_rank"] = rank
    fit_ranks = {id(r): rank for rank, r in enumerate(by_fitness, 1)}
    for row in rows:
        row["fitness_rank"] = fit_ranks.get(id(row))
        row["rank_delta"] = (
            (row.get("monitor_rank") or 0) - row["fitness_rank"]
            if row.get("fitness_rank") and row.get("monitor_rank")
            else None)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    adversarial = {}
    for name, c in ADVERSARIAL.items():
        f = ec.easy_doin_candidate_fitness(
            closed_trades=c["trades"], scored_rows=2190,
            validation_return=c["val_ret"],
            validation_drawdown=c["val_dd"],
            train_tail_return=c["tt"], activity_config=CAL)
        adversarial[name] = {"lex_key": f["lex_key"],
                             "eligible": f["eligible"]}
    order = sorted(adversarial, key=lambda n: adversarial[n]["lex_key"],
                   reverse=True)
    summary = {
        "schema": "agent_multi.rank_disagreement_study.v1",
        "epochs": len(rows),
        "no_test_fact_loaded": not forbidden,
        "forbidden_keys_found": forbidden,
        "top_by_monitor": [r["epoch"] for r in by_monitor[:3]],
        "top_by_fitness": [r["epoch"] for r in by_fitness[:3]],
        "max_abs_rank_delta": max(
            (abs(r["rank_delta"]) for r in rows
             if r["rank_delta"] is not None), default=0),
        "adversarial_ordering_best_to_worst": order,
        "csv": str(args.out_csv),
    }
    args.out_report.write_text(json.dumps(summary, indent=1,
                                          sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if summary["no_test_fact_loaded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
