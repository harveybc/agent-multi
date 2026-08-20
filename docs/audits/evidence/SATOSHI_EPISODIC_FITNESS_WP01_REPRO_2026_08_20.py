#!/usr/bin/env python3
"""EAF-009 corrected reproducer: non-aborting, per-case dispositions.

Runs every audited case against the EXECUTING module with an EXPLICIT
diagnostic candidate contract (the central WP4 candidate, labelled
diagnostic — chosen for reproduction only, not calibration authority).
One expected refusal can never abort the rest. Exit 0 iff zero defects
reproduce.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _episodic_activity_fitness as ef  # noqa: E402

DIAGNOSTIC_CONTRACT = {
    "activity_plateau_low_rate": 50.0,
    "activity_plateau_high_rate": 300.0,
    "_label": "diagnostic candidate (WP4 central); not calibration",
}


def score(trades, ret, dd=0.0, **kw):
    cfg = {k: v for k, v in DIAGNOSTIC_CONTRACT.items()
           if not k.startswith("_")}
    cfg.update(kw.pop("config", {}))
    return ef.evaluate_episode(
        total_return=ret, max_drawdown_fraction=dd, sharpe=None,
        closed_trades=trades, scored_rows=2190, config=cfg,
        **kw)["selection_value"]


def run(name, fn, dispositions):
    try:
        reproduced = bool(fn())
        dispositions[name] = "REPRODUCED" if reproduced else "CLOSED"
    except ef.EpisodicFitnessError as error:
        dispositions[name] = f"CLOSED_TYPED_REFUSAL: {str(error)[:80]}"
    except Exception as error:  # noqa: BLE001 — never abort the rest
        dispositions[name] = (
            f"RUNNER_ERROR: {type(error).__name__}: {error}")


def main() -> int:
    d: dict = {}
    run("one_trade_still_wins",
        lambda: score(1, -0.00001) > score(40, -0.05, 0.1), d)
    run("deep_losses_alias",
        lambda: len({score(10, r, 1.0)
                     for r in (-1.0, -10.0, -100.0)}) < 3, d)
    run("extreme_finite_loss_below_sentinel",
        lambda: score(120, -1e9, 1.0) <= -100.0, d)
    run("invalid_config_turns_loss_positive",
        lambda: score(40, -0.05,
                      config={"loss_economic_weight": -5}) > 0, d)
    run("small_sentinel_beats_active_loss",
        lambda: score(40, -0.2,
                      config={"zero_trade_sentinel": -1.0}) < -1.0, d)
    run("bars_negative_accepted",
        lambda: score(10, 0.01, bars_per_year=-2190), d)
    run("bars_boolean_accepted",
        lambda: score(10, 0.01, bars_per_year=True), d)
    run("bars_fractional_accepted",
        lambda: score(10, 0.01, bars_per_year=1.5), d)
    run("plateau_default_invented",
        lambda: ef.evaluate_episode(
            total_return=0.01, max_drawdown_fraction=0.0, sharpe=None,
            closed_trades=10, scored_rows=2190), d)
    run("single_crossing_survives_handoff",
        lambda: ef.assert_handoff_survivable(
            [0.0] * 499 + [0.5], normal_threshold=0.1,
            min_normal_crossings=4)["survivable"], d)
    reproduced = [k for k, v in d.items() if v == "REPRODUCED"]
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    print(json.dumps({
        "schema": "agent_multi.episodic_fitness_repro.v2",
        "commit_under_test": commit,
        "diagnostic_contract": DIAGNOSTIC_CONTRACT,
        "dispositions": d,
        "reproduced": reproduced,
        "acceptance": ("ZERO_REPRODUCED" if not reproduced
                       else f"REPRODUCED:{reproduced}"),
    }, indent=1, sort_keys=True))
    return 0 if not reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
