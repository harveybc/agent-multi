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
``execute`` runs the COMPLETE path: ``execute_confirmation``
(the gate chain, which refuses with the records absent and
creates nothing) followed by ``execute_confirmation_units``, the
unit census body, and one session report. ``development-probe``
and ``development-execution-probe`` run the real process
boundary on DEVELOPMENT units only and prove no CONFIRMATION
array, score or ledger is created.

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
import time
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


# ------- per-array role disjointness in the ARRAY domain -------
#
# prior_role_digest_census() above digests whole persisted *.npz
# FILES (v5 durable arm states). A generator ARRAY digest can
# never equal a zip-archive digest, so that census alone is a
# weak superset: it can only ever catch a byte-identical file.
# The execution body therefore ALSO re-derives the prior-role
# generator array digests for exactly the cells it is about to
# run, and unions them into the census it checks against. This
# only ENLARGES the forbidden set; the sealed
# verify_role_disjointness() comparison and its refusal are
# untouched, and the bank's own role-namespace proof
# (gb.assert_role_disjointness) is unchanged.

PRIOR_ROLES = ("DEVELOPMENT", "CALIBRATION")
_ARRAY_DIGEST_KEYS = ("X_train_sha256", "y_train_sha256",
                      "X_stop_sha256", "y_stop_sha256",
                      "X_held_sha256", "y_held_sha256")


def generator_array_digests(g) -> set:
    """The digests of the SIX byte-disjoint arrays this generator
    actually hands the consumer, recomputed by the bank's own
    consumer contract before use."""
    man = g["manifest"]
    return {man[k] for k in _ARRAY_DIGEST_KEYS}


def cells_of_units(units) -> list:
    return sorted({(u["family"], u["noise_coord"])
                   for u in units})


def role_array_digests(design, cells,
                       roles=PRIOR_ROLES) -> set:
    """Array-domain digest census of the named prior roles over
    the given (family, noise) cells. Every generator is built
    WITHOUT allow_confirmation, so this census can never contain
    a CONFIRMATION byte."""
    counts = {
        "DEVELOPMENT":
            design["populations_v5"]["DEVELOPMENT_per_cell"],
        "CALIBRATION":
            design["populations_v5"]["CALIBRATION_per_cell"]}
    out = set()
    for fam, nz in cells:
        for role in roles:
            if role not in counts:
                raise ConfirmationRunnerRefusal(
                    f"{role!r} is not a prior role")
            for gi in range(counts[role]):
                g = gb.generate(role, fam, nz, gi)
                gb.consumer_verify(g)
                out |= generator_array_digests(g)
    return out


# ---------------- per-unit records (C35.4) ----------------

ABORTED_DIR = "ABORTED_PARTIALS"


def unit_record_path(out_root, unit_id) -> Path:
    return (Path(out_root) / "intervention"
            / f"{rn._safe(unit_id)}_summary.json")


def _fsync_dir(p: Path):
    fd = os.open(str(p), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_unit_record(path, rec: dict) -> Path:
    """COMPLETE-OR-ABSENT. The bytes are written and fsynced
    under a temporary name and only then renamed into place, so a
    record carrying its FINAL name is never a partial. (The
    sealed _excl_json creates the final name FIRST, which would
    leave a truncated record readable as complete if the process
    died mid-write; the census must never read one as done.)"""
    path = Path(path)
    tmp = path.with_name(
        path.name + f".part{os.urandom(6).hex()}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, json.dumps(rec, indent=1,
                                sort_keys=True).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return path


def read_complete_unit_record(path, unit, expect_role):
    """None when the unit is not recorded; the record when it IS
    complete and binds this unit; a typed refusal when a record
    exists that cannot be read as complete — a damaged or foreign
    record is UNCERTAIN, never silently re-run and never counted
    in the numerator."""
    path = Path(path)
    if not path.is_file():
        return None
    rec = m4._strict_json_file(path, path.name)
    if "record_sha256" not in rec or \
            m4._self_sha(rec, "record_sha256") != \
            rec["record_sha256"]:
        raise ConfirmationRunnerRefusal(
            f"{path.name}: unit record self-identity does not "
            "re-derive — a partial or altered record is "
            "UNCERTAIN, never a completed unit")
    if rec.get("unit_id") != unit["unit_id"]:
        raise ConfirmationRunnerRefusal(
            f"{path.name}: record binds "
            f"{rec.get('unit_id')!r}, not {unit['unit_id']!r}")
    if expect_role == "CONFIRMATION":
        refuse_foreign_role_record(rec)
    return rec


def abort_partial_unit(out_root, unit) -> int:
    """A unit is ATOMIC and its record is the ONLY completion
    marker. Bytes left behind by an interrupted attempt (the
    per-batch arm logs and their durable states) are never
    consumed: they are moved aside and PRESERVED under
    ABORTED_PARTIALS/, and the unit runs again from scratch. The
    sealed 'partial log without its durable predecessor state is
    UNCERTAIN' refusal is therefore never met with bytes we
    kept — at ~0.5 s per unit, re-running the single in-flight
    unit is cheaper and safer than any silent resume, and no
    byte is destroyed."""
    out = Path(out_root)
    iv = out / "intervention"
    safe = rn._safe(unit["unit_id"])
    stale = sorted(p for p in iv.glob(safe + "__*") if p.is_file())
    stale += sorted(p for p in iv.glob(safe + "_summary.json.part*")
                    if p.is_file())
    if not stale:
        return 0
    base = out / ABORTED_DIR / safe
    base.mkdir(parents=True, exist_ok=True)
    os.chmod(out / ABORTED_DIR, 0o700)
    k = 1 + len([q for q in base.glob("attempt_*") if q.is_dir()])
    dest = base / f"attempt_{k:03d}"
    dest.mkdir(mode=0o700)
    for p in stale:
        os.replace(p, dest / p.name)
    return len(stale)


# ---------------- the execution body (C35.5) ----------------

def new_accounting() -> dict:
    """The accounting shape the sealed _limits and unit machinery
    expect; every increment is made by the sealed functions."""
    return {"optimization_updates": 0, "evaluations": 0,
            "descriptor_seconds": 0.0, "descriptor_evals": 0,
            "t0": time.monotonic()}


def next_session_index(out_root) -> int:
    return 1 + len([p for p in
                    Path(out_root).glob("SESSION_*_REPORT.json")
                    if p.is_file()])


def write_session_report(out_root, doc) -> Path:
    """One append-only report per session. Named without the
    token CONFIRMATION so the DEVELOPMENT probe's zero-artifact
    assertion stays exactly as sealed; the role is a field."""
    out = Path(out_root)
    n = next_session_index(out)
    doc = dict(doc)
    doc["schema"] = "m4_confirmation_session_report.v1"
    doc["session"] = n
    doc["report_sha256"] = cp._selfsha(doc, "report_sha256")
    p = out / f"SESSION_{n:03d}_REPORT.json"
    rn._excl_json(p, doc)
    return p


def execute_confirmation_units(design, units, out_root, acct,
                               *, expect_role,
                               allow_confirmation,
                               prior_digests,
                               batch_units=None) -> dict:
    """THE UNIT LOOP — the body that was missing.

    It drives the SEALED v5 machinery, it does not replace it:
    rn._limits() for the wall/RSS/nice/heartbeat/stop-file
    contract, rn._run_intervention_unit_v5() for the four
    checkpoint arms, gb.consumer_verify() for the bytes in use,
    verify_role_disjointness() for byte-level role isolation,
    refuse_foreign_role_record() at the record boundary, and the
    accounting dict the sealed limit machinery reads.

    Resumption: the pre-pass classifies every unit as COMPLETE
    (a self-re-deriving record that binds it) or PENDING; a
    complete unit is never re-run, a damaged record refuses, and
    an interrupted unit's partial bytes are set aside before it
    runs again. Records are written atomically, so the
    completion marker can never be a partial.

    It NEVER decides whether it may run: the gate chain above it
    does that, and it must already have passed."""
    if not prior_digests:
        raise ConfirmationRunnerRefusal(
            "the prior-role digest census is EMPTY — byte "
            "disjointness is never proven against nothing")
    out = Path(out_root)
    iv = out / "intervention"
    iv.mkdir(mode=0o700, exist_ok=True)
    complete = 0
    pending = []
    for u in units:
        if f"::{expect_role}::" not in u["unit_id"]:
            raise ConfirmationRunnerRefusal(
                f"unit {u['unit_id']!r} is not a "
                f"{expect_role} unit — one census, one role")
        rp = unit_record_path(out, u["unit_id"])
        if read_complete_unit_record(rp, u, expect_role) \
                is not None:
            complete += 1
        else:
            pending.append(u)
    new = 0
    aborted = 0
    gen_checked = set()
    status = None
    for u in pending:
        if batch_units is not None and new >= batch_units:
            status = "BATCH_FILLED_RESUMABLE"
            break
        stop = rn._limits(design, out, acct["t0"], acct)
        if stop:
            status = stop
            break
        aborted += abort_partial_unit(out, u)
        gid = u["generator_id"]
        if gid not in gen_checked:
            g = gb.generate(u["role"], u["family"],
                            u["noise_coord"],
                            u["generator_index"],
                            allow_confirmation=allow_confirmation)
            gb.consumer_verify(g)
            verify_role_disjointness(
                generator_array_digests(g), prior_digests)
            gen_checked.add(gid)
        rec = rn._run_intervention_unit_v5(
            design, u, out, acct,
            allow_confirmation=allow_confirmation)
        if rec.get("unit_id") != u["unit_id"]:
            raise ConfirmationRunnerRefusal(
                "the produced record does not bind its own unit")
        if expect_role == "CONFIRMATION":
            refuse_foreign_role_record(rec)
        write_unit_record(unit_record_path(out, u["unit_id"]),
                          rec)
        complete += 1
        new += 1
    left = len(units) - complete
    return {"role": expect_role,
            "units_total": len(units),
            "units_complete": complete,
            "units_pending": left,
            "units_new_this_session": new,
            "partial_attempts_set_aside": aborted,
            "generators_disjointness_verified": len(gen_checked),
            "prior_role_digests": len(prior_digests),
            "census_complete": left == 0,
            "session_status": status or (
                "CENSUS_COMPLETE" if left == 0
                else "INCOMPLETE_RESUMABLE"),
            "accounting": {
                "optimization_updates":
                    acct["optimization_updates"],
                "evaluations": acct["evaluations"],
                "descriptor_evals": acct["descriptor_evals"],
                "descriptor_seconds":
                    round(acct["descriptor_seconds"], 6)},
            "wall_seconds": round(
                time.monotonic() - acct["t0"], 2)}


def run_confirmation(repo_root=REPO, out_root=None,
                     batch_units=None) -> dict:
    """The COMPLETE execution path: the gate chain FIRST (with
    the two records absent it refuses and creates nothing), then
    the unit census body, then one session report. This is what
    the `execute` subcommand runs.

    The body lives one call BELOW execute_confirmation on
    purpose. The sealed POST battery
    (docs/audits/evidence/repro_runs/m4_c32_c38_post_2026_09_10)
    proves the gate is load-bearing by removing it and calling
    execute_confirmation directly; if the 3024-unit loop lived
    inside that function, re-running that sealed proof would
    itself manufacture an unauthorized CONFIRMATION census.
    Gate stage and body stay separate functions, and the
    execution path runs both."""
    repo_root = Path(repo_root)
    gate = execute_confirmation(repo_root, out_root)
    auth = cp.bind_calibration_evidence(repo_root)
    successor = cp.verify_confirmation_successor(repo_root)
    design = auth["design"]
    units = confirmation_units(successor)
    if len(units) != gate["census"]["units_total"]:
        raise ConfirmationRunnerRefusal(
            "the census the body would run is not the census "
            "the pre-result ledger enumerated")
    prior = (prior_role_digest_census()
             | role_array_digests(design, cells_of_units(units)))
    out = Path(out_root)
    acct = new_accounting()
    body = execute_confirmation_units(
        design, units, out, acct, expect_role="CONFIRMATION",
        allow_confirmation=True, prior_digests=prior,
        batch_units=batch_units)
    body["census_sha256"] = gate["census"]["census_sha256"]
    body["pre_result_ledger"] = str(gate["ledger"])
    body["review_record_sha256"] = \
        gate["records"]["review"]["record_sha256"]
    body["execution_record_sha256"] = \
        gate["records"]["execution"]["record_sha256"]
    report = write_session_report(out, body)
    return {**body, "session_report": str(report)}


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
    if out_root.exists():
        # RESUMPTION of an authorized run. The PRE-RESULT ledger
        # is written ONCE, before the first observation, and is
        # never rewritten: a later session verifies it instead.
        ledger = out_root / "CONFIRMATION_PRE_RESULT_LEDGER.json"
        if not ledger.is_file():
            raise ConfirmationRunnerRefusal(
                "the output root exists without its pre-result "
                "ledger — observations never resume into a root "
                "whose prior ledger cannot be read")
        led = m4._strict_json_file(ledger, "pre-result ledger")
        if cp._selfsha(led, "ledger_sha256") != \
                led["ledger_sha256"]:
            raise ConfirmationRunnerRefusal(
                "pre-result ledger self-identity does not "
                "re-derive — UNCERTAIN, never resumed")
        if led["census_sha256"] != census["census_sha256"]:
            raise ConfirmationRunnerRefusal(
                "the existing ledger binds a different census")
        # `prior_role_digests` is a live count of the state root
        # and legitimately moves between sessions; every
        # SCIENTIFIC gate identity must be identical.
        for k in ("successor_sha256", "review_record_sha256",
                  "execution_record_sha256", "executing_head"):
            if led["gates"].get(k) != gates[k]:
                raise ConfirmationRunnerRefusal(
                    f"resumption gate {k} differs from the "
                    "ledger — an unpinned surface never "
                    "continues a CONFIRMATION run")
    else:
        out_root.mkdir(parents=True, exist_ok=False)
        os.chmod(out_root, 0o700)
        ledger = write_pre_result_ledger(out_root, census, gates)
    return {"census": census, "ledger": str(ledger),
            "records": records,
            "note": "the gate chain ends here; unit execution "
                    "is execute_confirmation_units(), driven by "
                    "run_confirmation() through the sealed v5 "
                    "limit machinery with per-array "
                    "disjointness verification"}


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


def development_execution_probe(repo_root=REPO, out_root=None,
                                n_units=None,
                                batch_units=None) -> dict:
    """C37+: the SAME execution body, driven over DEVELOPMENT
    units only, through the real process boundary.

    No CONFIRMATION byte is touched: allow_confirmation stays
    False, so the bank's kill-17 guard would refuse one, and the
    prior-role census the body checks against is the real
    CALIBRATION array census for the same cells. The sealed
    assertion that no CONFIRMATION-named artifact appears is
    re-asserted here verbatim."""
    repo_root = Path(repo_root)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    design = cp.bind_calibration_evidence(repo_root)["design"]
    units = rn.intervention_units_v5(design, "DEVELOPMENT")
    if n_units is not None:
        units = units[:n_units]
    prior = role_array_digests(
        design, cells_of_units(units), roles=("CALIBRATION",))
    acct = new_accounting()
    body = execute_confirmation_units(
        design, units, out, acct, expect_role="DEVELOPMENT",
        allow_confirmation=False, prior_digests=prior,
        batch_units=batch_units)
    conf_artifacts = [str(p) for p in out.rglob("*")
                      if "CONFIRMATION" in p.name]
    if conf_artifacts:
        raise ConfirmationRunnerRefusal(
            "the DEVELOPMENT probe created CONFIRMATION-named "
            f"artifacts: {conf_artifacts[:3]}")
    body["confirmation_artifacts"] = 0
    body["units_run"] = [u["unit_id"] for u in units]
    write_session_report(out, dict(body))
    return body


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
    e.add_argument("--batch-units", type=int, default=None,
                   help="units this session may newly run; "
                        "omit to run the whole census (the "
                        "measured basis is ~0.5 s and ~103 MiB "
                        "per unit, so 3024 units are ~25 min "
                        "of one CPU inside the 172800 s wall)")
    g = sub.add_parser("gate-only")
    g.add_argument("--out", required=True)
    d = sub.add_parser("development-probe")
    d.add_argument("--out", required=True)
    p = sub.add_parser("development-execution-probe")
    p.add_argument("--out", required=True)
    p.add_argument("--units", type=int, default=None)
    p.add_argument("--batch-units", type=int, default=None)
    v = sub.add_parser("verify")
    v.add_argument("--run-root", required=True)
    a = ap.parse_args(argv)
    if a.cmd in ("execute", "development-probe",
                 "development-execution-probe"):
        # the sealed resources block: cpu_only, cuda_hidden,
        # nice 15, one logical worker
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            delta = 15 - os.nice(0)
            if delta > 0:
                os.nice(delta)
        except OSError:
            pass
    if a.cmd == "plan":
        print(json.dumps(plan_confirmation(), indent=1))
    elif a.cmd == "execute":
        print(json.dumps(run_confirmation(
            out_root=Path(a.out),
            batch_units=a.batch_units), indent=1, default=str))
    elif a.cmd == "gate-only":
        print(json.dumps(execute_confirmation(
            out_root=Path(a.out)), indent=1, default=str))
    elif a.cmd == "development-probe":
        print(json.dumps(development_mechanics_probe(
            out_root=Path(a.out)), indent=1))
    elif a.cmd == "development-execution-probe":
        print(json.dumps(development_execution_probe(
            out_root=Path(a.out), n_units=a.units,
            batch_units=a.batch_units), indent=1))
    elif a.cmd == "verify":
        print(json.dumps(verify_confirmation_run(
            REPO, Path(a.run_root)), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
