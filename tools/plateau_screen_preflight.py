"""C4 preflight: replay the PROPOSED early-intervention scheduler over
the four existing FIXED monitor histories and persist the predicted
first interventions.

Order: MUSASHI_TO_GENERAL_SATOSHI_WP1_WP3_CORRECTION_AND_GPU_DISPATCH_ORDER_2026_08_22 §C4.

Dispatch rule (mechanical, fail-closed): if fewer than THREE of the
four seeds would receive their first LR reduction STRICTLY BEFORE
their historical global-best epoch, the proposed screen has no
observable treatment window and dispatch is REFUSED (exit 2).

The replay is counterfactual — an actual reduction would change the
trajectory — which is exactly what a treatment-window preflight needs:
it asks only WHEN the intervention would first fire on the untreated
curve. Uses the REAL controller implementation, not a re-derivation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline_plugins import _sac_plateau_lr as pl  # noqa: E402

SEEDS = (101, 202, 303, 404)
PROPOSED = {"factor": 0.5, "lr_patience": 8, "min_lr": 1e-6,
            "threshold": 1e-6, "cooldown": 0, "start_epoch": 0}
MIN_SEEDS_WITH_WINDOW = 3


def replay(history) -> dict:
    ctrl = pl.SacPlateauLrController(initial_lr=3e-4, **PROPOSED)
    first = None
    for row in history:
        rec = ctrl.observe(epoch=row["epoch"],
                           monitor_value=row["composite"],
                           apply_fn=lambda lr: {"replay": lr})
        if rec["reduced"] and first is None:
            first = row["epoch"]
    eligible = [r for r in history if r.get("l1_checkpoint_eligible")]
    best = max(eligible, key=lambda r: r["composite"])
    return {"predicted_first_reduction_epoch": first,
            "historical_global_best_epoch": best["epoch"],
            "intervenes_before_best":
                first is not None and first < best["epoch"],
            "epochs_replayed": len(history)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    per_seed = {}
    for seed in SEEDS:
        path = args.screen_dir / f"seed{seed}_fixed_report.json"
        if not path.is_file():
            print(json.dumps({"outcome": "REFUSED_INCOMPLETE_INPUT",
                              "missing": str(path)}))
            return 2
        doc = json.loads(path.read_text())
        per_seed[seed] = replay(doc["history"])
    with_window = sum(1 for d in per_seed.values()
                      if d["intervenes_before_best"])
    permitted = with_window >= MIN_SEEDS_WITH_WINDOW
    result = {
        "schema": "agent_multi.plateau_screen_preflight.v1",
        "proposed_contract": PROPOSED,
        "controller_contract_id": pl.CONTRACT_ID,
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "seeds_with_treatment_window": with_window,
        "min_required": MIN_SEEDS_WITH_WINDOW,
        "dispatch_permitted": permitted,
        "scope_statement": (
            "A negative screen result rejects THIS bounded-ETH "
            "scheduler specification (120/40/40-day window, this "
            "contract) — not plateau scheduling as a universal "
            "mechanism."),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=1))
    print(json.dumps({"dispatch_permitted": permitted,
                      "seeds_with_treatment_window": with_window,
                      "per_seed_first_reduction": {
                          str(s): per_seed[s][
                              "predicted_first_reduction_epoch"]
                          for s in SEEDS}}))
    return 0 if permitted else 2


if __name__ == "__main__":
    sys.exit(main())
