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

ADMITTED_ENDPOINT_STATES = ("ACQUISITION_ENDPOINT",
                            "RETENTION_ENDPOINT",
                            "CAP_REACHED")
ATTRITION_ALLOWANCE = 0.20      # sealed in v4/v5 precision basis
MIN_COMPLETE_FOR_DISPERSION = 3


def _arm_complete(a) -> bool:
    """C31C: an arm is complete only when it ends in an endpoint
    state admitted by the sealed restricted-endpoint contract AND
    carries a valid descriptor. NUMERICAL_ANOMALY, invalid task
    training, invalid descriptor, resource stops, missing records
    and uncertain states are NOT complete."""
    if a is None:
        return False
    if a.get("stopping_cause") not in ADMITTED_ENDPOINT_STATES:
        return False
    if a.get("descriptors", {}).get(
            "numerically_invalid_descriptor") is True:
        return False
    return True


def _complete_primary_pair(summary) -> bool:
    if summary is None or summary.get("unit_status") is not None:
        return False
    arms = summary.get("arms", {})
    return (_arm_complete(arms.get("initialization"))
            and _arm_complete(arms.get("calibration_stop")))


def _complete_quartet(summary) -> bool:
    if summary is None or summary.get("unit_status") is not None:
        return False
    arms = summary.get("arms", {})
    import m4_v5_protocol as _pv
    return all(_arm_complete(arms.get(a))
               for a in _pv.CHECKPOINTS)


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
    planned_per_cell = design["populations_v5"][
        "CALIBRATION_per_cell"]
    min_complete = max(
        MIN_COMPLETE_FOR_DISPERSION,
        int(np.ceil(planned_per_cell
                    * (1 - ATTRITION_ALLOWANCE))))
    for fam, nz in rn._conf_cells(design):
        for w in rn.CONF_WIDTHS:
            per_gen = {}
            incomplete = []
            for gi in range(planned_per_cell):
                diffs = []
                seeds_complete = 0
                for ms in range(rn.CAL_SEEDS):
                    uid = (f"intervention::CALIBRATION::{fam}"
                           f"::{nz}::w{w}::g{gi}::s{ms}")
                    r = summaries.get(uid)
                    if _complete_primary_pair(r):
                        seeds_complete += 1
                        diffs.append(
                            r["paired_primary_difference"])
                # C31C: EXACTLY all three sealed seeds, or the
                # generator is INCOMPLETE_PAIRED_GENERATOR — one
                # or two seeds are never averaged as complete.
                if seeds_complete == rn.CAL_SEEDS:
                    per_gen[f"g{gi}"] = float(np.mean(diffs))
                else:
                    incomplete.append(
                        {"generator": f"g{gi}",
                         "seeds_complete": seeds_complete,
                         "status":
                             "INCOMPLETE_PAIRED_GENERATOR"})
            key = f"{fam}::{nz}::w{w}"
            base = {"planned_generators": planned_per_cell,
                    "complete_generators": len(per_gen),
                    "incomplete_generators": incomplete,
                    "attrition_allowance": ATTRITION_ALLOWANCE,
                    "min_complete_required": min_complete}
            if len(per_gen) >= min_complete:
                d = pv.dispersion_from_paired(per_gen)
                dispersion[key] = {**base, **d}
                elig = eligibility.get(
                    key, {"eligible_under_proposed_rule": False})
                if elig["eligible_under_proposed_rule"] and \
                        not d["supported"]:
                    supported_all = False
            else:
                dispersion[key] = {
                    **base,
                    "status": "CALIBRATION_INCOMPLETE"}
                elig = eligibility.get(
                    key, {"eligible_under_proposed_rule": False})
                if elig["eligible_under_proposed_rule"]:
                    supported_all = False
    # ---- ladder over CALIBRATION arms ----
    rows = []
    groups = []
    ladder_excluded = []
    for uid, r in summaries.items():
        if not _complete_quartet(r):
            reasons = []
            for arm_n, a_ in r.get("arms", {}).items():
                if not _arm_complete(a_):
                    reasons.append(
                        f"{arm_n}:"
                        f"{a_.get('stopping_cause')}"
                        if a_.get("descriptors", {}).get(
                            "numerically_invalid_descriptor")
                        is not True else
                        f"{arm_n}:INVALID_DESCRIPTOR")
            ladder_excluded.append({"unit_id": uid,
                                    "reasons": reasons})
            continue
        nuis = [1.0 if r["task_kind"] == "temporal" else 0.0,
                1.0 if r["noise_coord"] == "white" else 0.0]
        for arm, a in r["arms"].items():
            if a["stopping_cause"] == "NUMERICAL_ANOMALY":
                continue        # unreachable post-quartet-gate
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
            elig = eligibility.get(key, {
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
           "ladder_excluded_units": ladder_excluded,
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
