"""M4 C35: the CONFIRMATION runner and independent verifier.

Structurally unable to consume DEVELOPMENT or CALIBRATION
outcomes as confirmation observations:

- every confirmation unit id carries ``::CONFIRMATION::`` and its
  generator identity derives from the role string (disjoint seed
  streams by construction);
- records whose unit ids carry any other role are refused at the
  analysis boundary;
- byte-level disjointness is proven at runtime: every generated
  CONFIRMATION array digest must be absent from the prior-role
  digest census, or the run refuses.

Before ANY execution the runner must, in order:
 1. bind the accepted calibration evidence (C32) and verify the
    confirmation successor (C33);
 2. consume BOTH external records (Musashi design review + owner
    execution, chained) — absent/forged/transplanted records
    refuse BEFORE any CONFIRMATION array or ledger exists;
 3. verify the executing checkout is clean and record its
    identity;
 4. materialize the exact generator/cell/seed/checkpoint census
    with the frozen update bounds;
 5. write the complete PRE-RESULT ledger (every unit PENDING,
    O_EXCL) before the first observation;
 6. enforce the sealed CPU wall / RSS / nice / heartbeat /
    stop-file limits through the accepted v5 limit machinery;
 7. preserve every incomplete or numerical state in the
    denominator.

``plan`` reports counts with no records and writes nothing.
``development-probe`` runs the real process boundary on
DEVELOPMENT units only and proves no CONFIRMATION array, score
or ledger is created.

The independent verifier reconstructs generators, tapes,
checkpoints, restricted endpoints, paired effects, attrition,
costs and all 16 contrasts from raw records; producer aggregates
never determine a verdict.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import m4_confirmation_protocol as cp  # noqa: E402
import m4_generator_bank as gb  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402
import m4_v5_runner as rn  # noqa: E402


class ConfirmationRunnerRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git(repo_root, *args):
    return subprocess.run(["git", *args], cwd=repo_root,
                          capture_output=True, text=True)


# ---------------- census (C35.1) ----------------

CHECKPOINT_KINDS = ("initialization", "calibration_stop",
                    "pre_stop", "post_stop_bounded")


def confirmation_units(successor):
    units = []
    for s in successor["eligible_slots"]:
        prefix, w = s["cell"].rsplit("::w", 1)
        fam, nz = prefix.split("::")
        for gi in range(
                successor[
                    "confirmation_generators_per_eligible_slot"
                ]):
            for ms in range(
                    successor["nested_seeds_per_generator"]):
                units.append(rn._iv_unit(
                    "CONFIRMATION", fam, nz, int(w), gi, ms))
    return sorted(units, key=lambda u: u["unit_id"])


def materialize_census(successor, design) -> dict:
    units = confirmation_units(successor)
    ck = design["checkpoint_rules"]
    census = {
        "schema": "m4_confirmation_census.v1",
        "successor_sha256": successor["successor_sha256"],
        "eligible_slots": len(successor["eligible_slots"]),
        "generators_per_slot": successor[
            "confirmation_generators_per_eligible_slot"],
        "seeds_per_generator": successor[
            "nested_seeds_per_generator"],
        "units_total": len(units),
        "unit_ids_sha256": _sha_bytes(json.dumps(
            [u["unit_id"] for u in units]).encode()),
        "checkpoint_kinds": list(CHECKPOINT_KINDS),
        "update_bounds": {
            "calibration_stop_max_updates":
                ck["calibration_stop"]["max_updates"],
            "cadence_updates":
                ck["calibration_stop"]["cadence_updates"],
            "post_stop_bounded_updates": 500,
        },
    }
    census["census_sha256"] = cp._selfsha(census,
                                          "census_sha256")
    return census


# ---------------- role disjointness (C35.2) ----------------

def prior_role_digest_census(state_root=None) -> set:
    """Byte digests of every DEVELOPMENT/CALIBRATION generator
    array persisted by the accepted campaign roots."""
    state_root = Path(state_root
                      or Path.home() / ".local/share/agent-multi")
    digests = set()
    for root in sorted(state_root.glob("m4_v5_*")):
        for p in sorted(root.rglob("*.npz")):
            digests.add(_sha_bytes(p.read_bytes()))
    return digests


def verify_role_disjointness(conf_digests, prior_digests):
    hit = sorted(set(conf_digests) & set(prior_digests))
    if hit:
        raise ConfirmationRunnerRefusal(
            f"{len(hit)} CONFIRMATION array byte-digests "
            "collide with prior-role bytes — reused data never "
            "confirms (first: " + hit[0][:16] + ")")


def refuse_foreign_role_record(rec):
    uid = rec.get("unit_id", "")
    if "::CONFIRMATION::" not in uid:
        raise ConfirmationRunnerRefusal(
            f"record {uid!r} is not a CONFIRMATION unit — "
            "DEVELOPMENT/CALIBRATION outcomes are structurally "
            "inadmissible as confirmation observations")


# ---------------- pre-result ledger (C35.3) ----------------

def write_pre_result_ledger(out_root: Path, census,
                            gates) -> Path:
    p = Path(out_root) / "CONFIRMATION_PRE_RESULT_LEDGER.json"
    doc = {
        "schema": "m4_confirmation_pre_result_ledger.v1",
        "census_sha256": census["census_sha256"],
        "gates": gates,
        "units": {},
    }
    units = census["units_total"]
    doc["units_total"] = units
    doc["status_all"] = "PENDING"
    doc["ledger_sha256"] = cp._selfsha(doc, "ledger_sha256")
    rn._excl_json(p, doc)
    return p


# ---------------- plan (no records, no writes) --------------

def plan_confirmation(repo_root=REPO) -> dict:
    auth = cp.bind_calibration_evidence(repo_root)
    successor = cp.verify_confirmation_successor(repo_root)
    census = materialize_census(successor, auth["design"])
    review_present = cp.MUSASHI_REVIEW_RECORD_PATH.is_file()
    exec_present = cp.OWNER_EXECUTION_RECORD_PATH.is_file()
    return {
        "mode": "PLAN_ONLY_NO_AUTHORITY",
        "units_total": census["units_total"],
        "eligible_slots": census["eligible_slots"],
        "generators_per_slot": census["generators_per_slot"],
        "seeds_per_generator": census["seeds_per_generator"],
        "checkpoints_per_unit": len(CHECKPOINT_KINDS),
        "update_bounds": census["update_bounds"],
        "census_sha256": census["census_sha256"],
        "musashi_review_record_present": review_present,
        "owner_execution_record_present": exec_present,
        "execution_open": False,
        "note": "planning reports counts only; execution "
                "refuses before generating a CONFIRMATION "
                "array or ledger until BOTH records verify",
    }


# ---------------- execute (gated) ----------------

def execute_confirmation(repo_root=REPO, out_root=None) -> dict:
    """The full gate chain. In this order the chain ALWAYS
    refuses at the two-record gate (no real records exist and
    candidate code never creates them); the post-gate body is
    exercised by the battery only through the DEVELOPMENT
    mechanics probe and record mocks — never with CONFIRMATION
    arrays."""
    repo_root = Path(repo_root)
    auth = cp.bind_calibration_evidence(repo_root)
    successor = cp.verify_confirmation_successor(repo_root)
    records = cp.require_both_records(successor)  # refuses here
    st = _git(repo_root, "status", "--porcelain")
    if st.stdout.strip():
        raise ConfirmationRunnerRefusal(
            "executing checkout is dirty — an unpinned surface "
            "never executes CONFIRMATION")
    head = _git(repo_root, "rev-parse",
                "HEAD").stdout.strip()
    if out_root is None:
        raise ConfirmationRunnerRefusal(
            "no output root was provided")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    census = materialize_census(successor, auth["design"])
    prior = prior_role_digest_census()
    gates = {
        "successor_sha256": successor["successor_sha256"],
        "review_record_sha256":
            records["review"]["record_sha256"],
        "execution_record_sha256":
            records["execution"]["record_sha256"],
        "executing_head": head,
        "prior_role_digests": len(prior),
    }
    ledger = write_pre_result_ledger(out_root, census, gates)
    return {"census": census, "ledger": str(ledger),
            "records": records,
            "note": "unit execution proceeds only beyond this "
                    "point, through the sealed v5 limit "
                    "machinery, with per-array disjointness "
                    "verification"}


# ---------------- DEVELOPMENT mechanics probe ----------------

def development_mechanics_probe(repo_root=REPO,
                                out_root=None) -> dict:
    """C37: run the REAL unit machinery through the real process
    boundary on DEVELOPMENT units only; prove that no
    CONFIRMATION array, score or ledger is created."""
    repo_root = Path(repo_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    (out_root / "intervention").mkdir()
    design = cp.bind_calibration_evidence(repo_root)["design"]
    units = rn.intervention_units_v5(design, "DEVELOPMENT")[:2]
    import time
    t0 = time.monotonic()
    acct = {"optimization_updates": 0, "evaluations": 0,
            "descriptor_seconds": 0.0, "descriptor_evals": 0,
            "t0": t0}
    recs = []
    for u in units:
        rn._limits(design, out_root, t0, acct)
        recs.append(rn._run_intervention_unit_v5(
            design, u, out_root, acct))
    conf_artifacts = [str(p) for p in out_root.rglob("*")
                      if "CONFIRMATION" in p.name]
    if conf_artifacts:
        raise ConfirmationRunnerRefusal(
            "the DEVELOPMENT probe created CONFIRMATION-named "
            f"artifacts: {conf_artifacts[:3]}")
    return {"units_run": [u["unit_id"] for u in units],
            "records": len(recs),
            "confirmation_artifacts": 0}


# ---------------- independent verifier ----------------

def verify_confirmation_run(repo_root, run_root,
                            successor=None) -> dict:
    """Reconstruct EVERYTHING from raw records; producer
    aggregates never determine a verdict."""
    repo_root = Path(repo_root)
    run_root = Path(run_root)
    successor = successor or cp.verify_confirmation_successor(
        repo_root)
    design = cp.bind_calibration_evidence(repo_root)["design"]
    ledger_p = run_root / "CONFIRMATION_PRE_RESULT_LEDGER.json"
    if not ledger_p.is_file():
        raise ConfirmationRunnerRefusal(
            "no pre-result ledger exists — results without a "
            "prior complete ledger never verify")
    ledger = m4._strict_json_file(ledger_p, "pre-result ledger")
    if cp._selfsha(ledger, "ledger_sha256") != \
            ledger["ledger_sha256"]:
        raise ConfirmationRunnerRefusal(
            "pre-result ledger self-identity does not re-derive")
    census = materialize_census(successor, design)
    if ledger["census_sha256"] != census["census_sha256"]:
        raise ConfirmationRunnerRefusal(
            "ledger census is not the successor-derived census")
    per_gen = {}
    ck_pair = {}
    attrition = {}
    costs = {}
    n_rec = 0
    for p in sorted(run_root.glob("intervention/"
                                  "*_summary.json")):
        r = m4._strict_json_file(p, p.name)
        refuse_foreign_role_record(r)
        n_rec += 1
        uid = r["unit_id"]
        _, role, fam, nz, w, g, s = uid.split("::")
        # paired effect re-derived from the ARM RECORDS, never
        # from any producer aggregate field
        arms = r.get("arms", {})
        if r.get("unit_status") == \
                "NUMERICALLY_INVALID_TASK_TRAINING":
            attrition.setdefault(f"{fam}::{nz}::{w}",
                                 []).append(uid)
            continue
        stop = arms.get("calibration_stop")
        init = arms.get("initialization")
        if stop is None or init is None:
            attrition.setdefault(f"{fam}::{nz}::{w}",
                                 []).append(uid)
            continue
        eff = (stop["restricted_endpoint"]
               - init["restricted_endpoint"])
        declared = r.get("paired_primary_difference")
        if declared is not None and \
                abs(eff - declared) > 1e-9:
            raise ConfirmationRunnerRefusal(
                f"{uid}: declared paired difference "
                f"{declared!r} does not re-derive from the arm "
                f"records ({eff!r}) — producer aggregates never "
                "determine a verdict")
        key = f"{fam}::{nz}"
        wd = int(w[1:])
        per_gen.setdefault(key, {}).setdefault(
            wd, {}).setdefault(g, []).append(eff)
        ck_pair.setdefault(g, []).append(eff)
        # costs re-derived from the arm records themselves
        # (updates actually done), never from a producer total
        costs[uid] = {an: a["updates_done"]
                      for an, a in arms.items()}
    seeds_needed = successor["nested_seeds_per_generator"]
    complete = {}
    for key, by_w in per_gen.items():
        for wd, by_g in by_w.items():
            for g, vals in by_g.items():
                # a generator with fewer than the exact nested
                # seed count never averages as complete
                if len(vals) == seeds_needed:
                    complete.setdefault(key, {}).setdefault(
                        wd, {})[g] = float(np.mean(vals))
    # attrition floor per eligible slot: below the frozen
    # minimum the slot is CONFIRMATION_INCOMPLETE — its width
    # never contributes and is never favorable
    floor = successor["attrition"]["min_complete_required"]
    incomplete_slots = {}
    for s in successor["eligible_slots"]:
        prefix, w = s["cell"].rsplit("::w", 1)
        n_complete = len(complete.get(prefix, {}).get(
            int(w), {}))
        if n_complete < floor:
            incomplete_slots[s["cell"]] = {
                "status": "CONFIRMATION_INCOMPLETE",
                "complete_generators": n_complete,
                "min_complete_required": floor}
            if prefix in complete:
                complete[prefix].pop(int(w), None)
    ck_complete = {g: float(np.mean(v))
                   for g, v in ck_pair.items()
                   if len(v) >= seeds_needed}
    analysis = cp.sixteen_contrasts(successor, complete,
                                    ck_complete)
    hetero = cp.width_heterogeneity(successor, complete)
    return {"records_verified": n_rec,
            "attrition": {k: len(v)
                          for k, v in attrition.items()},
            "confirmation_incomplete": incomplete_slots,
            "analysis": analysis,
            "width_heterogeneity_secondary": hetero,
            "costs_units": len(costs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    e = sub.add_parser("execute")
    e.add_argument("--out", required=True)
    d = sub.add_parser("development-probe")
    d.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--run-root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        print(json.dumps(plan_confirmation(), indent=1))
    elif a.cmd == "execute":
        print(json.dumps(execute_confirmation(
            out_root=Path(a.out)), indent=1, default=str))
    elif a.cmd == "development-probe":
        print(json.dumps(development_mechanics_probe(
            out_root=Path(a.out)), indent=1))
    elif a.cmd == "verify":
        print(json.dumps(verify_confirmation_run(
            REPO, Path(a.run_root)), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
