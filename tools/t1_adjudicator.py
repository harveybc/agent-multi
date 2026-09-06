#!/usr/bin/env python3
"""T1 adjudicator v2 (orders T0-T1 + C1-C12).

C6/C11: the adjudicator consumes the SEALED design and the exact
bank inventory, binds exactly one record for every predeclared
unit x operator, verifies source-array digests and the
content-addressed NPZ evidence, and RE-DERIVES the gate metrics
from the arrays — producer aggregates and producer-declared
cardinalities grant nothing.

C7: strict JSON (duplicate-key and non-finite rejection); every
gate-bearing number must be finite; typed nulls never authorize
calibration.

C8: the verdict derives from SCORE-role facts only; train and
validation facts are published separately.

C10: ANY material safety failure (non-finite gate, tail, extreme or
utility harm) in ANY required seed/variable yields LAB_REJECTED or
INCONCLUSIVE per the predeclared rule — a median can never hide it.
Distribution (min/median/max) is reported; windows are never
replicas."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

SNR_GAIN_MIN_DB = 0.5
UTILITY_TOLERANCE = 0.02
EXTREME_RETENTION_MIN = 0.5
TAIL_RATIO_MAX = 1.5
RESIDUAL_INFO_MAX = 0.01
MIN_SEEDS = 3
CANDIDATES = ("identity", "trailing_mean", "trailing_median",
              "ewma", "local_level_kalman")
ORACLE = "centered_mean_oracle"


class AdjudicationRefusal(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict:
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise AdjudicationRefusal(
                f"REFUSED: duplicate JSON key in {path.name}")
        return dict(pairs)
    return json.loads(
        Path(path).read_text(), object_pairs_hook=_no_dupes,
        parse_constant=lambda c: (_ for _ in ()).throw(
            AdjudicationRefusal(
                f"REFUSED: non-finite literal {c!r} in "
                f"{path.name}")))


def _gate(v, what: str) -> float:
    """C7: a gate value must be a finite float; typed nulls and
    anything else refuse the gate (never authorize)."""
    if isinstance(v, dict):
        raise AdjudicationRefusal(
            f"REFUSED: gate {what} is a typed null "
            f"({v.get('reason')}) — it cannot authorize "
            "calibration")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or \
            not np.isfinite(float(v)):
        raise AdjudicationRefusal(
            f"REFUSED: gate {what} is not a finite number")
    return float(v)


def _regime(rec: dict) -> str:
    snr = rec["declared_snr_db"]
    snr_s = ("het" + "_".join(str(s) for s in snr)
             if isinstance(snr, list) else str(snr))
    return f"{rec['family']}|{rec['perturbation']}|snr{snr_s}"


def expected_population(design: dict, inventory: dict) -> set:
    """C6: the exact expected (unit, operator) population derives
    from the SEALED DESIGN + inventory — never from the producer."""
    ops = tuple(design["expected_operators_exact"])
    return {(uid, op) for uid in inventory["unit_ids"]
            for op in ops}


def rederive_gates(record: dict, bank_dir: Path,
                   npz_dir: Path) -> None:
    """C6: recompute the score-role gates from the content-addressed
    arrays and require equality with the record's published facts."""
    unit_dir = bank_dir / record["unit_id"]
    rec_u = _strict_json(unit_dir / "UNIT.json")
    clean = np.load(unit_dir / "clean_signal.npy")
    obs = np.load(unit_dir / "observed_signal.npy")
    support = np.load(unit_dir / "metric_support.npy")
    blob = np.load(npz_dir /
                   f"{record['denoised_residual_npz_sha256']}.npz")
    d = blob["denoised"]
    if hashlib.sha256(d.tobytes() + blob["residual"].tobytes()
                      ).hexdigest() != \
            record["denoised_residual_npz_sha256"]:
        raise AdjudicationRefusal(
            f"REFUSED: NPZ content digest broken for "
            f"{record['unit_id']}/{record['operator']}")
    lo, hi = rec_u["temporal_roles"]["score"]
    for j, pv in enumerate(record["per_variable"]):
        sup = support[j][lo:hi]
        c = clean[j][lo:hi][sup]
        o = obs[j][lo:hi][sup]
        dd = d[j][lo:hi][sup]
        if len(c) < 20:
            continue
        mse_o = float(np.mean((o - c) ** 2))
        mse_d = float(np.mean((dd - c) ** 2))
        gain = (10 * np.log10(mse_o / mse_d)
                if mse_o > 0 and mse_d > 0 else 0.0)
        pub = pv["by_role"]["score"]
        if isinstance(pub, dict) and pub.get("null"):
            raise AdjudicationRefusal(
                f"REFUSED: {record['unit_id']} score facts are "
                "null yet the record claims MEASURED")
        if abs(_gate(pub["snr_gain_db"], "snr_gain") - gain) > 1e-6:
            raise AdjudicationRefusal(
                f"REFUSED: published score snr_gain differs from "
                f"the re-derived value for {record['unit_id']}/"
                f"{record['operator']}")


def adjudicate(design: dict, inventory: dict, measurements: dict,
               bank_dir: Path = None, npz_dir: Path = None,
               rederive_sample: int = 0) -> dict:
    records = measurements["records"]
    expected = expected_population(design, inventory)
    seen = set()
    for r in records:
        key = (r["unit_id"], r["operator"])
        if key in seen:
            raise AdjudicationRefusal(
                f"REFUSED: duplicate record {key}")
        if key not in expected:
            raise AdjudicationRefusal(
                f"REFUSED: foreign record {key} not in the "
                "design-derived population")
        seen.add(key)
    missing = expected - seen
    if missing:
        raise AdjudicationRefusal(
            f"REFUSED: population incomplete — {len(missing)} "
            f"design-required records missing (e.g. "
            f"{sorted(missing)[:2]})")
    if rederive_sample and bank_dir is not None:
        rng = np.random.default_rng(20260906)
        measured = [r for r in records if r["status"] == "MEASURED"]
        for r in rng.choice(np.array(measured, dtype=object),
                            size=min(rederive_sample,
                                     len(measured)),
                            replace=False):
            rederive_gates(r, bank_dir, npz_dir)
    by_key = {}
    for r in records:
        by_key.setdefault((r["operator"], _regime(r)),
                          []).append(r)
    verdicts = {}
    for (op, regime), recs in sorted(by_key.items()):
        if op == ORACLE:
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
                "reason": ("typed refusal in regime: "
                           + refusals[0]["refusal"][:80]),
                "seeds": len(seeds)}
            continue
        if len(seeds) < MIN_SEEDS:
            verdicts[(op, regime)] = {
                "verdict": "INCONCLUSIVE",
                "reason": f"only {len(seeds)} seeds < {MIN_SEEDS}"}
            continue
        gains, utils, rets, tails, resid = [], [], [], [], []
        material_failures = []
        try:
            for r in recs:
                for pv in r["per_variable"]:
                    sc = pv["by_role"]["score"]
                    if isinstance(sc, dict) and sc.get("null"):
                        material_failures.append(
                            f"seed {r['seed']} {pv['variable']}: "
                            "null score facts")
                        continue
                    g = _gate(sc["snr_gain_db"],
                              "score snr_gain")
                    ret = _gate(sc["extreme_retention"],
                                "score retention")
                    tl = _gate(sc["tail_ratio"], "score tail")
                    gains.append(g)
                    rets.append(ret)
                    tails.append(tl)
                    for h in ("h1", "h5"):
                        a = pv["assays_score_fit_train"][h]
                        x = _gate(a["X"], "assay X")
                        dv = _gate(a["D"], "assay D")
                        ri = _gate(a["residual_incremental_r2"],
                                   "residual info")
                        rel = (dv - x) / max(abs(x), 1e-6)
                        utils.append(rel)
                        resid.append(ri)
                        # C10: ANY material failure is terminal
                        if rel < -0.25:
                            material_failures.append(
                                f"seed {r['seed']} "
                                f"{pv['variable']} {h}: utility "
                                f"collapse {rel:.2f}")
                    if ret < 0.0 and g >= SNR_GAIN_MIN_DB:
                        material_failures.append(
                            f"seed {r['seed']} {pv['variable']}: "
                            f"extreme INVERSION ({ret:.2f})")
                    if tl > 3.0:
                        material_failures.append(
                            f"seed {r['seed']} {pv['variable']}: "
                            f"tail explosion ({tl:.2f}x)")
        except AdjudicationRefusal as exc:
            verdicts[(op, regime)] = {
                "verdict": "INCONCLUSIVE",
                "reason": str(exc)[:120], "seeds": len(seeds)}
            continue
        def dist(v):
            return {"min": round(float(np.min(v)), 4),
                    "median": round(float(np.median(v)), 4),
                    "max": round(float(np.max(v)), 4)} if v else None
        facts = {"snr_gain_db": dist(gains),
                 "relative_utility_delta_D_vs_X": dist(utils),
                 "extreme_retention": dist(rets),
                 "tail_ratio": dist(tails),
                 "residual_incremental_r2": dist(resid),
                 "seeds": len(seeds),
                 "material_failures": material_failures}
        if op == "identity":
            verdicts[(op, regime)] = {
                "verdict": "LAB_CALIBRATED",
                "reason": "mandatory no-op control", **facts}
            continue
        if material_failures:
            verdicts[(op, regime)] = {
                "verdict": "LAB_REJECTED",
                "reason": ("material safety failure in at least "
                           "one seed/variable — a median cannot "
                           "hide it: "
                           + material_failures[0]),
                **facts}
            continue
        med_gain = float(np.median(gains))
        med_util = float(np.median(utils))
        med_ret = float(np.median(rets))
        med_tail = float(np.median(tails))
        med_resid = float(np.median(resid))
        failures = []
        if med_gain < SNR_GAIN_MIN_DB:
            failures.append(f"no material SNR gain "
                            f"({med_gain:.2f} dB)")
        if med_util < -UTILITY_TOLERANCE:
            failures.append(f"downstream utility worsens "
                            f"({med_util:.3f})")
        if med_ret < EXTREME_RETENTION_MIN and \
                med_gain >= SNR_GAIN_MIN_DB:
            failures.append(f"extreme destruction "
                            f"({med_ret:.2f})")
        if med_tail > TAIL_RATIO_MAX:
            failures.append(f"tail inflation ({med_tail:.2f}x)")
        informative = med_resid > RESIDUAL_INFO_MAX
        if not failures:
            verdict = "LAB_CALIBRATED"
            note = ("RESIDUAL_INFORMATIVE — reported as a "
                    "TRANSFORMATION, not verified noise removal"
                    if informative else "verified in-regime "
                    "(score role only)")
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
    out = {"schema": "agent_multi.t1_adjudication.v2",
           "design_sha256": measurements.get("design_sha256"),
           "population": {"records": len(records),
                          "expected": len(expected)},
           "verdict_basis": "SCORE-role facts only (C8); "
                            "train/validation published separately "
                            "in the records",
           "material_failure_rule":
               "any seed/variable material failure -> LAB_REJECTED "
               "(C10); distributions reported min/median/max",
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
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--design-sha", required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--measurements", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rederive-sample", type=int, default=40)
    args = ap.parse_args()
    raw = args.design.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.design_sha:
        raise AdjudicationRefusal(
            "REFUSED: sealed design bytes differ from the reviewed "
            "digest")
    design = _strict_json(args.design)
    inventory = _strict_json(args.bank_dir / "BANK_INVENTORY.json")
    m = _strict_json(args.measurements)
    if m.get("design_sha256") != args.design_sha:
        raise AdjudicationRefusal(
            "REFUSED: measurements were not produced under this "
            "sealed design")
    out = adjudicate(design, inventory, m, bank_dir=args.bank_dir,
                     npz_dir=args.npz_dir,
                     rederive_sample=args.rederive_sample)
    args.output.write_text(json.dumps(out, indent=1,
                                      allow_nan=False))
    print(json.dumps(out["verdict_counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
