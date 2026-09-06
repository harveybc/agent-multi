#!/usr/bin/env python3
"""T1 adjudicator (work plan 43 §5.4, order T0-T1 §7).

Pure and fail-closed: derives everything from the per-observation
measurement records, never from producer aggregates. Emits ONLY
LAB_CALIBRATED, LAB_REJECTED or INCONCLUSIVE, always bounded by
regime (family x perturbation x SNR band). A better RMSE or SNR gain
with worse task utility or tails does not pass. An informative
residual forbids the name "noise removal" (the operator may survive
as a TRANSFORMATION, labeled). The statistical unit is the
independent (family, perturbation, snr, seed) combination — rows and
windows are never replicas."""
import json
import sys
from pathlib import Path

import numpy as np

SNR_GAIN_MIN_DB = 0.5
UTILITY_TOLERANCE = 0.02          # relative R2 drop tolerated
EXTREME_RETENTION_MIN = 0.5
TAIL_RATIO_MAX = 1.5
RESIDUAL_INFO_MAX = 0.01          # incremental R2 of the residual
MIN_SEEDS = 3


class AdjudicationRefusal(SystemExit):
    pass


def _regime(rec: dict) -> str:
    snr = rec["declared_snr_db"]
    snr_s = ("het" + "_".join(str(s) for s in snr)
             if isinstance(snr, list) else str(snr))
    return f"{rec['family']}|{rec['perturbation']}|snr{snr_s}"


def adjudicate(measurements: dict) -> dict:
    records = measurements["records"]
    expected = (measurements["expected_units"]
                * measurements["expected_operators"])
    if len(records) != expected:
        raise AdjudicationRefusal(
            f"REFUSED: population incomplete — {len(records)} "
            f"records != expected {expected}")
    by_key = {}
    for r in records:
        by_key.setdefault((r["operator"], _regime(r)),
                          []).append(r)
    verdicts = {}
    for (op, regime), recs in sorted(by_key.items()):
        if op == "centered_mean_oracle":
            verdicts[(op, regime)] = {
                "verdict": "NON_CAUSAL_ORACLE_ONLY",
                "reason": "diagnostic ceiling; never eligible"}
            continue
        seeds = {r["seed"] for r in recs}
        refusals = [r for r in recs
                    if r["status"] == "TYPED_REFUSAL"]
        if refusals:
            verdicts[(op, regime)] = {
                "verdict": "INCONCLUSIVE",
                "reason": ("typed refusal in regime (e.g. "
                           "unlicensed missingness): "
                           + refusals[0]["refusal"][:80]),
                "seeds": len(seeds)}
            continue
        if len(seeds) < MIN_SEEDS:
            verdicts[(op, regime)] = {
                "verdict": "INCONCLUSIVE",
                "reason": f"only {len(seeds)} seeds "
                          f"< {MIN_SEEDS}"}
            continue
        gains, util_deltas, retentions, tails, resid_info = \
            [], [], [], [], []
        delays = []
        for r in recs:
            for pv in r["per_variable"]:
                gains.append(pv["snr_gain_db"])
                retentions.append(pv["extreme_retention"])
                tails.append(pv["tail_ratio"])
                delays.append(abs(pv["delay_bars"]))
                for h in ("h1", "h5"):
                    a = pv["assays"][h]
                    base = max(abs(a["X"]), 1e-6)
                    util_deltas.append((a["D"] - a["X"]) / base)
                    resid_info.append(
                        a["residual_incremental_r2"])
        med_gain = float(np.median(gains))
        med_util = float(np.median(util_deltas))
        med_ret = float(np.median(retentions))
        med_tail = float(np.median(tails))
        med_resid = float(np.median(resid_info))
        med_delay = float(np.median(delays))
        facts = {"median_snr_gain_db": round(med_gain, 3),
                 "median_relative_utility_delta_D_vs_X":
                     round(med_util, 4),
                 "median_extreme_retention": round(med_ret, 3),
                 "median_tail_ratio": round(med_tail, 3),
                 "median_residual_incremental_r2":
                     round(med_resid, 4),
                 "median_abs_delay_bars": med_delay,
                 "seeds": len(seeds)}
        if op == "identity":
            verdicts[(op, regime)] = {
                "verdict": "LAB_CALIBRATED",
                "reason": "mandatory no-op control", **facts}
            continue
        failures = []
        if med_gain < SNR_GAIN_MIN_DB:
            failures.append(
                f"no material SNR gain ({med_gain:.2f} dB)")
        if med_util < -UTILITY_TOLERANCE:
            failures.append(
                f"downstream utility WORSENS ({med_util:.3f} "
                "relative) — reconstruction alone never passes")
        if med_ret < EXTREME_RETENTION_MIN and \
                med_gain >= SNR_GAIN_MIN_DB:
            failures.append(
                f"material extreme destruction (retention "
                f"{med_ret:.2f})")
        if med_tail > TAIL_RATIO_MAX:
            failures.append(f"tail inflation ({med_tail:.2f}x)")
        informative = med_resid > RESIDUAL_INFO_MAX
        if not failures:
            verdict = "LAB_CALIBRATED"
            note = ("RESIDUAL_INFORMATIVE — reported as a "
                    "TRANSFORMATION, not verified noise removal"
                    if informative else "verified in-regime")
        elif med_util < -UTILITY_TOLERANCE or \
                med_ret < EXTREME_RETENTION_MIN:
            verdict = "LAB_REJECTED"
            note = "; ".join(failures)
        else:
            verdict = "INCONCLUSIVE"
            note = "; ".join(failures)
        verdicts[(op, regime)] = {"verdict": verdict,
                                  "reason": note,
                                  "residual_informative":
                                      informative, **facts}
    out = {"schema": "agent_multi.t1_adjudication.v1",
           "population": {"records": len(records),
                          "expected": expected},
           "thresholds": {
               "snr_gain_min_db": SNR_GAIN_MIN_DB,
               "utility_tolerance_rel": UTILITY_TOLERANCE,
               "extreme_retention_min": EXTREME_RETENTION_MIN,
               "tail_ratio_max": TAIL_RATIO_MAX,
               "residual_info_max_r2": RESIDUAL_INFO_MAX,
               "min_seeds": MIN_SEEDS},
           "statistical_unit": ("independent (family, perturbation, "
                                "snr, seed) combination; never rows "
                                "or windows"),
           "verdicts": {f"{op}::{rg}": v
                        for (op, rg), v in verdicts.items()}}
    counts = {}
    for v in verdicts.values():
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    out["verdict_counts"] = counts
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    m = json.loads(args.measurements.read_text())
    out = adjudicate(m)
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["verdict_counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
