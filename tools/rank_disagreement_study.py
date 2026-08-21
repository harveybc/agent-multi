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
        tt_tr = row.get("train_tail_trades")
        vv_tr = row.get("val_trades")
        tt_ret = row.get("train_tail_return", 0.0) or 0.0
        vv_ret = row.get("val_return", row.get("validation_return",
                                               0.0)) or 0.0
        monitor = ec.easy_checkpoint_monitor(
            train_tail_return=float(tt_ret),
            validation_return=float(vv_ret),
            train_tail_drawdown=float(row.get(
                "train_tail_drawdown", 0.0) or 0.0),
            validation_drawdown=float(row.get(
                "val_drawdown", 0.0) or 0.0))
        try:
            fit = ec.easy_doin_candidate_fitness(
                closed_trades=int(vv_tr or 0),
                scored_rows=2190,
                validation_return=float(vv_ret),
                validation_drawdown=float(row.get(
                    "val_drawdown", 0.0) or 0.0),
                train_tail_return=float(tt_ret),
                activity_config=CAL)
            key = fit["lex_key"]
            eligible = fit["eligible"]
            reason = fit["reason"]
        except ec.EasyContractError as error:
            key, eligible, reason = None, False, str(error)[:60]
        rows.append({"epoch": epoch, "train_tail_trades": tt_tr,
                     "val_trades": vv_tr,
                     "monitor_value": monitor["value"],
                     "monitor_components": json.dumps(
                         monitor["components"]),
                     "fitness_lex_key": json.dumps(key),
                     "eligible": eligible, "reason": reason})
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
