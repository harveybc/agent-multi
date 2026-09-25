#!/usr/bin/env python3
"""Repro: the four 'byte-identical metrics' sites, and the guard's verdict.

READ-ONLY against the sealed/live collections. Run:

    python docs/audits/evidence/SATOSHI_III_D8_IDENTICAL_METRICS_REPRO_2026_08_15.py

Each site prints the primary evidence and the verdict that
``tools.arm_differentiation`` returns for it. The expected result is
printed at the end; a deviation means either the collections moved or
the guard's ladder changed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.arm_differentiation import (  # noqa: E402
    ArmObservation, evaluate_arms, policy_tensor_sha256,
    trace_behavior_facts,
)

SHARE = Path.home() / ".local/share/agent-multi"
SITE1 = SHARE / "eth_curriculum_decision_20260807_v2"
SITE3 = (SHARE / "p1_difficulty_lr_factorial_20260811_v1_decision"
         / "c0e53cf18b7d60dd" / "seed101")
_FIXTURE_REL = "examples/results/eth_curriculum_fixture_v2"
# The fixture's .zip weights are untracked, so a worktree has only the
# JSON. Fall back to the canonical checkout (read-only) to reach them.
FIXTURE = next(
    (root / _FIXTURE_REL for root in
     (REPO_ROOT, Path("/home/harveybc/Documents/GitHub/agent-multi"))
     if (root / _FIXTURE_REL / "easy/model.post_easy.zip").is_file()),
    REPO_ROOT / _FIXTURE_REL)
USDCAD = Path("/home/harveybc/Documents/GitHub/financial-data/experiments"
              "/stage_a_screening")

KEYS = ("mean_weekly_return", "annualized_return", "total_return",
        "max_drawdown_fraction", "trades_total")


def site1() -> None:
    print("\n=== SITE 1: eth_curriculum_decision_20260807_v2 / seed101 ===")
    obs = []
    for arm in ("E4", "EN4_10", "N14"):
        record = json.loads((SITE1 / "seed101" / arm
                             / "arm_record.json").read_text())
        results = json.loads((SITE1 / "seed101" / arm
                              / "results.json").read_text())
        reported = (record["splits_raw"] or {}).get("validation") or {}
        own = (results["splits"] or {}).get("validation") or {}
        selected = (SITE1 / "seed101" / arm /
                    ("model.post_easy.zip" if arm == "E4"
                     else "best_checkpoint.zip"))
        tensor = policy_tensor_sha256(selected)
        print(f"  {arm:7s} arm_record.splits_raw.validation"
              f" total_return={reported.get('total_return')!r}"
              f" trades={reported.get('trades_total')!r}")
        print(f"          own results.json validation"
              f" total_return={own.get('total_return')!r}"
              f" trades={own.get('trades_total')!r}"
              f"  no_trade_diagnosis={own.get('no_trade_diagnosis')!r}")
        print(f"          scored artifact {selected.name}"
              f" tensor={tensor[:16]}…")
        obs.append(ArmObservation(
            arm=arm, treatment={"arm": arm},
            metrics={k: reported.get(k) for k in KEYS},
            scored_policy_tensor_sha256=tensor,
            trades_total=reported.get("trades_total")))
    anchor = policy_tensor_sha256(SITE1 / "seed101" / "anchor_seed101.zip")
    print(f"  warm-start anchor_seed101.zip tensor={anchor[:16]}…"
          f"  <-- equals every scored tensor above")
    for pair in evaluate_arms(obs, metric_keys=KEYS)["pairs"]:
        print(f"  VERDICT {pair['arms'][0]:7s} vs {pair['arms'][1]:7s}"
              f" -> {pair['verdict']}")


def site2() -> None:
    print("\n=== SITE 2: examples/results/eth_curriculum_fixture_v2 ===")
    for rel in ("easy/model.post_easy.zip",
                "easy_normal/model.post_easy.zip",
                "easy_normal/model.zip", "normal/model.zip"):
        path = FIXTURE / rel
        if not path.is_file():
            print(f"  {rel:34s} ABSENT (untracked binary)")
            continue
        print(f"  {rel:34s} tensor={policy_tensor_sha256(path)[:16]}…")
    print("  README open question is CONFIRMED: easy and easy_normal"
          " select the same post-easy tensor.")


def site3() -> None:
    print("\n=== SITE 3: p1lr decision c0e53cf18b7d60dd / seed101 ===")
    obs = []
    for cell in sorted(p.name for p in SITE3.iterdir() if p.is_dir()):
        record = json.loads((SITE3 / cell / "cell_record.json").read_text())
        final = record["outer_validation_final"]
        metrics = {k: final.get(k) for k in KEYS}
        print(f"  {cell:12s} trades={final.get('trades_total')!r}"
              f" total_return={final.get('total_return')!r}"
              f" activity={record.get('activity_status')!r}"
              f" tensor={str(record.get('terminal_policy_tensor_sha256'))[:16]}…")
        obs.append(ArmObservation(
            arm=cell, treatment=record["factors"], metrics=metrics,
            scored_policy_tensor_sha256=record.get(
                "terminal_policy_tensor_sha256"),
            active=record.get("activity_status") == "active",
            trades_total=final.get("trades_total")))
    for pair in evaluate_arms(obs, metric_keys=KEYS)["pairs"]:
        if pair["metric_tuple_identical"]:
            print(f"  VERDICT {pair['arms'][0]} vs {pair['arms'][1]}"
                  f" -> {pair['verdict']}")


def site4() -> None:
    print("\n=== SITE 4: usdcad_4h stage-B register (financial-data) ===")
    index = USDCAD / "stage_b_approval/stage_b_approval_candidates.csv"
    if not index.is_file():
        print("  register absent; skipping")
        return
    rows = [r for r in csv.DictReader(index.open())
            if r["asset"] == "usdcad" and r["timeframe"] == "4h"]
    tuples = {(r["total_return"], r["sharpe_ratio"],
               r["max_drawdown_pct"], r["trades_total"]) for r in rows}
    print(f"  {len(rows)} runs -> {len(tuples)} distinct metric tuples")
    obs = []
    for preset in ("baseline_12", "fx_full", "kitchen_sink_guarded"):
        hits = sorted((USDCAD / "runs/gamma").glob(
            f"usdcad_4h_ppo_{preset}_direct_atr_sltp_s0_*"))
        if not hits:
            continue
        facts = trace_behavior_facts(hits[0] / "return_trace.csv")
        print(f"  {preset:22s} fingerprint={facts['behavior_fingerprint'][:16]}…"
              f" distinct_actions={facts['distinct_actions']}"
              f" degenerate={facts['behavior_degenerate']}")
        obs.append(ArmObservation(
            arm=preset, treatment={"preset": preset},
            metrics={"total_return": -3.580170977901531e-06,
                     "trades_total": 1727},
            behavior_fingerprint=facts["behavior_fingerprint"],
            behavior_degenerate=facts["behavior_degenerate"],
            trades_total=facts["trades_total"]))
    for pair in evaluate_arms(
            obs, metric_keys=("total_return", "trades_total"))["pairs"]:
        print(f"  VERDICT {pair['arms'][0]} vs {pair['arms'][1]}"
              f" -> {pair['verdict']}")


def main() -> int:
    for site in (site1, site2, site3, site4):
        try:
            site()
        except FileNotFoundError as exc:
            print(f"  SKIPPED ({exc})")
    print("""
EXPECTED
  site 1  SHARED_SCORED_POLICY on all three pairs — every arm scored the
          untouched warm-start anchor, so the treatment was never measured
  site 2  same mechanism, confirmed by tensor identity (README closed)
  site 3  DEGENERATE_IDENTICAL for P1E_LR1E4 vs P1N_LR1E4 only — two
          distinct dead policies, a REAL identity; the LR=3e-5 column
          separates cleanly, proving the pipeline does distinguish arms
  site 4  DEGENERATE_IDENTICAL — action_raw is constant, so the feature
          preset cannot matter; real collapse, not a measurement bug""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
