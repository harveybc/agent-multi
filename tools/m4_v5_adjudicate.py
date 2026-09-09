"""M4 C34/C35: development gate adjudication and the CANDIDATE
calibration adjudication for Musashi review.

C34 gates (fail -> M4_V5_DEVELOPMENT_GATE_FAILED): fresh
verification of the corrected DEVELOPMENT run, the easy positive
control passing its frozen criterion, random-label never
licensed, tapes/geneses/lineages exact (enforced by the fresh
verifier).

C35 (only if C34 passes, over the sealed 16-generator
CALIBRATION run): learnability margins, PROPOSED family/width
eligibility (explicitly labeled for Musashi review — not a
sealed pre-outcome rule), per-confirmatory-cell generator-level
dispersion of the paired restricted endpoint (seed-averaged)
with its chi-square UCB95 and the sealed UCB<=6 support gate,
the executable M0/M1/M2 ladder disposition, and the exact
CONFIRMATION slot list with typed ineligible slots. No
CONFIRMATION array or outcome is generated, loaded or scored.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import m4_residual_capacity as m4  # noqa: E402
import m4_v5_protocol as pv  # noqa: E402
import m4_v5_runner as rn  # noqa: E402

PROPOSED_ELIGIBILITY = {
    "rule": ("a family/noise/width cell is ELIGIBLE when >= 12 "
             "of its 16 CALIBRATION generators return "
             "LEARNABLE_UNDER_FROZEN_BUDGET and none returns "
             "NUMERICALLY_INVALID"),
    "status": "PROPOSED_ELIGIBILITY_RULE_FOR_MUSASHI_REVIEW",
    "min_learnable": 12,
}


class AdjudicationRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def c34_development_gate(design, dev_root) -> dict:
    v = rn.verify_run_v5(design, Path(dev_root),
                         ("DEVELOPMENT",))
    tab = m4._strict_json_file(
        Path(dev_root) / "LEARNABILITY_TABLE_DEVELOPMENT.json",
        "dev table")
    gates = {
        "fresh_verifier": v["verified"] is True,
        "easy_positive_control":
            tab["easy_positive_control_passes"] is True,
        "random_label_not_licensed": not any(
            "random_label" in k for k in tab["cells"]),
        "tapes_geneses_lineages_exact": v["verified"] is True,
    }
    return {"gates": gates,
            "passed": all(gates.values()),
            "margin": tab["margin"]}


def c35_calibration_adjudication(design, cal_root) -> dict:
    cal_root = Path(cal_root)
    v = rn.verify_run_v5(design, cal_root, ("CALIBRATION",))
    if v["verified"] is not True:
        raise AdjudicationRefusal("calibration run does not "
                                  "verify")
    tab = m4._strict_json_file(
        cal_root / "LEARNABILITY_TABLE_CALIBRATION.json",
        "cal table")
    # ---- eligibility (PROPOSED rule, labeled) ----
    per_cell = {}
    for uid, cell in tab["cells"].items():
        _, _, fam, nz, w, _g = uid.split("::")
        key = f"{fam}::{nz}::{w}"
        per_cell.setdefault(key, {"LEARNABLE_UNDER_FROZEN_"
                                  "BUDGET": 0,
                                  "OPTIMIZATION_LIMITED": 0,
                                  "NUMERICALLY_INVALID": 0,
                                  "total": 0})
        per_cell[key][cell["outcome"]] += 1
        per_cell[key]["total"] += 1
    eligibility = {}
    for key, c in sorted(per_cell.items()):
        ok = (c["LEARNABLE_UNDER_FROZEN_BUDGET"]
              >= PROPOSED_ELIGIBILITY["min_learnable"]
              and c["NUMERICALLY_INVALID"] == 0)
        eligibility[key] = {
            "counts": c,
            "eligible_under_proposed_rule": bool(ok)}
    # ---- dispersion per confirmatory cell x width ----
    sums = m4._strict_json_file  # alias for brevity below
    summaries = {}
    incomplete_units = []
    for p in sorted((cal_root / "intervention").glob(
            "*_summary.json")):
        r = sums(p, p.name)
        if r.get("unit_status") == \
                "NUMERICALLY_INVALID_TASK_TRAINING":
            incomplete_units.append(r["unit_id"])
            continue
        summaries[r["unit_id"]] = r
    dispersion = {}
    supported_all = True
    for fam, nz in rn._conf_cells(design):
        for w in rn.CONF_WIDTHS:
            per_gen = {}
            for gi in range(design["populations_v5"][
                    "CALIBRATION_per_cell"]):
                diffs = []
                for ms in range(rn.CAL_SEEDS):
                    uid = (f"intervention::CALIBRATION::{fam}"
                           f"::{nz}::w{w}::g{gi}::s{ms}")
                    r = summaries.get(uid)
                    if r is None:
                        continue
                    diffs.append(
                        r["paired_primary_difference"])
                if diffs:
                    per_gen[f"g{gi}"] = float(np.mean(diffs))
            key = f"{fam}::{nz}::w{w}"
            if len(per_gen) >= 3:
                d = pv.dispersion_from_paired(per_gen)
                dispersion[key] = d
                elig = eligibility.get(
                    f"{fam}::{nz}::{w}",
                    {"eligible_under_proposed_rule": False})
                if elig["eligible_under_proposed_rule"] and \
                        not d["supported"]:
                    supported_all = False
            else:
                dispersion[key] = {
                    "status": "INSUFFICIENT_GENERATORS"}
    # ---- ladder over CALIBRATION arms ----
    rows = []
    groups = []
    for uid, r in summaries.items():
        nuis = [1.0 if r["task_kind"] == "temporal" else 0.0,
                1.0 if r["noise_coord"] == "white" else 0.0]
        for arm, a in r["arms"].items():
            if a["stopping_cause"] == "NUMERICAL_ANOMALY":
                continue        # incomplete arm, never survival
            rows.append({
                "param_count": 10 * r["width"] + 1,
                "nuisance": nuis,
                "checkpoint_loss": a["checkpoint_loss_stop"],
                "task_updates":
                    r["checkpoint_lineage"][arm]["updates"],
                "compressed_len":
                    a["descriptors"]["compressed_len_zlib9"],
                "spectral_rank":
                    a["descriptors"]["spectral_rank_W1_1e3"],
                "prune_fraction":
                    a["descriptors"]["prune_fraction_1e3"],
                "stop_traj_slope": r["stop_trajectory_slope"],
                "fail_batch": a["fail_batch"]})
            groups.append(r["generator_id"])
    ladder = pv.ladder_compare(rows, groups)
    # ---- CONFIRMATION slot list with typed ineligibles ----
    slots = []
    for fam, nz in rn._conf_cells(design):
        for w in rn.CONF_WIDTHS:
            key = f"{fam}::{nz}::w{w}"
            ek = f"{fam}::{nz}::{w}"
            elig = eligibility.get(ek, {
                "eligible_under_proposed_rule": False})
            slots.append({
                "cell": key,
                "reserved_generators":
                    design["populations_v5"][
                        "CONFIRMATION_reserved_per_cell"],
                "typed_status":
                    "ELIGIBLE_UNDER_PROPOSED_RULE"
                    if elig["eligible_under_proposed_rule"]
                    else "INELIGIBLE_UNDER_PROPOSED_RULE",
                "dispersion": dispersion.get(key)})
    doc = {"schema":
           "agent_multi.m4_v5_calibration_adjudication."
           "candidate.v1",
           "authority": "CANDIDATE_FOR_MUSASHI_REVIEW_NO_"
                        "CONFIRMATION_AUTHORITY",
           "design_sha256": design["design_sha256"],
           "calibration_margin": tab["margin"],
           "proposed_eligibility_rule": PROPOSED_ELIGIBILITY,
           "eligibility": eligibility,
           "dispersion": dispersion,
           "precision_supported_on_eligible_cells":
               bool(supported_all),
           "ladder": ladder,
           "incomplete_units_in_denominator": incomplete_units,
           "confirmation_slots": slots}
    doc["record_sha256"] = m4._self_sha(doc, "record_sha256")
    return doc


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-root", type=Path, required=True)
    ap.add_argument("--cal-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    design = pv.load_design_v5()
    g34 = c34_development_gate(design, a.dev_root)
    print(json.dumps({"c34": g34}, indent=1))
    if not g34["passed"]:
        print(json.dumps(
            {"disposition": "M4_V5_DEVELOPMENT_GATE_FAILED"}))
        return 1
    if a.cal_root is None:
        return 0
    doc = c35_calibration_adjudication(design, a.cal_root)
    if a.out:
        fd = os.open(str(a.out), os.O_CREAT | os.O_EXCL
                     | os.O_WRONLY, 0o600)
        try:
            os.write(fd, json.dumps(doc, indent=1).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
    print(json.dumps({
        "calibration_margin": doc["calibration_margin"],
        "precision_supported":
            doc["precision_supported_on_eligible_cells"],
        "ladder_status": doc["ladder"].get("status"),
        "eligible_cells": sum(
            1 for e in doc["eligibility"].values()
            if e["eligible_under_proposed_rule"]),
        "total_cells": len(doc["eligibility"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
