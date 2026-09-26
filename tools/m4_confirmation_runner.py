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
import m4_v5_protocol as pv  # noqa: E402
import m4_v5_runner as rn  # noqa: E402

# DR04 R1: the digest of the executable analysis surface, re-exported so
# every gate, ledger and verdict names the same one identity.
implementation_digest = cp.implementation_digest
implementation_file_digests = cp.implementation_file_digests


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


def census_units(successor, role="CONFIRMATION"):
    """The EXACT population a successor document declares, enumerated for
    one role. ``role`` never widens anything: it only names whose units
    these are, and every consumer checks that the role it expects is the
    role it got. A DEVELOPMENT fixture successor therefore exercises this
    same enumeration, and the CONFIRMATION enumeration is unchanged."""
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
                    role, fam, nz, int(w), gi, ms))
    return sorted(units, key=lambda u: u["unit_id"])


def confirmation_units(successor):
    return census_units(successor, "CONFIRMATION")


def materialize_census(successor, design,
                       role="CONFIRMATION") -> dict:
    units = census_units(successor, role)
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
    if role != "CONFIRMATION":
        # The sealed CONFIRMATION census bytes are left byte-identical
        # (its sha256 is quoted in the accepted evidence); any other role
        # carries its role IN the census, so a fixture census can never be
        # mistaken for the confirmatory one.
        census["census_role"] = role
    census["census_sha256"] = cp._selfsha(census,
                                          "census_sha256")
    return census


def census_role(census) -> str:
    return census.get("census_role", "CONFIRMATION")


# ------- DEVELOPMENT fixture population (DR04) -------
#
# Every repair below has to be demonstrable positively AND negatively
# without constructing one byte of confirmatory data. A DEVELOPMENT
# fixture successor is a document of the SAME schema whose declared
# population is small and whose role is DEVELOPMENT, so the identical
# census / seed-set / attrition / contrast / verdict code path runs over
# real (non-confirmatory) records.
#
# It can never produce a confirmatory verdict: the verifier requires the
# order-pinned successor identity whenever the expected role is
# CONFIRMATION, the census carries its role, the ledger gate map carries
# its role, and the verdict carries NON_CONFIRMATORY in its own field.

DEV_FIXTURE_CELLS = (("sine", "clean"), ("chirp", "clean"))
DEV_FIXTURE_WIDTHS = (16, 64)


def development_fixture_successor(generators=2, seeds=3,
                                  cells=DEV_FIXTURE_CELLS,
                                  widths=DEV_FIXTURE_WIDTHS,
                                  successor=None) -> dict:
    """A DEVELOPMENT-role successor document: same schema, same frozen
    16-slot contrast family, tiny declared population, floor 2.

    It is NEVER installed, never written into the repository and never
    accepted where the CONFIRMATION successor is required."""
    doc = {
        "schema": "m4_development_fixture_successor.v1",
        "role": "DEVELOPMENT",
        "grants_nothing": "DEVELOPMENT_FIXTURE_NO_AUTHORITY_NO_"
                          "SCIENTIFIC_VERDICT",
        "eligible_slots": [
            {"cell": f"{fam}::{nz}::w{w}",
             "confirmation_generators": generators}
            for fam, nz in cells for w in widths],
        "confirmation_generators_per_eligible_slot": generators,
        "nested_seeds_per_generator": seeds,
        "attrition": {"allowance": 0.2,
                      "planned_generators_per_slot": generators,
                      "min_complete_required": generators,
                      "rule": "the fixture requires its whole declared "
                              "population; below it the slot is "
                              "INCOMPLETE, never favorable"},
        "contrast_family_16": list(
            (successor or cp.verify_confirmation_successor(REPO))[
                "contrast_family_16"]),
        "m2_status": {"calibration_gain": cp.ORDER_FACTS["m2_gain"]},
    }
    doc["successor_sha256"] = cp._selfsha(doc, "successor_sha256")
    return doc


# ---------------- run lock and per-unit claims (DR04 R6) ----------------

RUN_LOCK_NAME = "M4_RUN_SESSION.lock"
LOCKS_DIR = "ABORTED_LOCKS"


def _boot_id() -> str:
    try:
        return Path(
            "/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "BOOT_ID_UNAVAILABLE"


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError, ValueError, TypeError):
        return True
    return True


def _claim_doc(kind) -> dict:
    doc = {"schema": f"m4_{kind}.v1", "pid": os.getpid(),
           "boot_id": _boot_id(), "epoch": round(time.time(), 3),
           "token": os.urandom(8).hex()}
    doc["claim_sha256"] = cp._selfsha(doc, "claim_sha256")
    return doc


def _holder_is_live(path) -> bool:
    """A claim whose writer is still running on this boot is LIVE; any
    other claim is the residue of an interrupted attempt."""
    try:
        doc = m4._strict_json_file(path, path.name)
    except SystemExit:
        return False
    if doc.get("boot_id") != _boot_id():
        return False
    return _pid_alive(doc.get("pid", -1))


def acquire_run_lock(out_root) -> dict:
    """ONE writer per run root. A live holder REFUSES — two sessions can
    never execute the same census concurrently. A lock left behind by an
    interrupted session is preserved under ABORTED_LOCKS/ (never deleted)
    and the new session proceeds."""
    out = Path(out_root)
    p = out / RUN_LOCK_NAME
    stale = 0
    for attempt in (1, 2):
        doc = _claim_doc("run_session_lock")
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL
                         | os.O_WRONLY, 0o600)
        except FileExistsError:
            if _holder_is_live(p):
                raise ConfirmationRunnerRefusal(
                    f"another session holds the run lock {p.name} "
                    "(live pid on this boot) — concurrent sessions never "
                    "execute the same census")
            if attempt == 2:
                raise ConfirmationRunnerRefusal(
                    f"{p.name} reappeared while it was being set aside "
                    "— UNCERTAIN, never raced")
            d = out / LOCKS_DIR
            d.mkdir(mode=0o700, exist_ok=True)
            k = 1 + len([q for q in d.glob("lock_*") if q.is_file()])
            os.replace(p, d / f"lock_{k:03d}.json")
            stale += 1
            continue
        try:
            os.write(fd, json.dumps(doc, indent=1,
                                    sort_keys=True).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        return {"path": p, "doc": doc, "stale_locks_set_aside": stale}
    raise ConfirmationRunnerRefusal("run lock could not be acquired")


def release_run_lock(lock):
    p = Path(lock["path"])
    if not p.is_file():
        return
    doc = m4._strict_json_file(p, p.name)
    if doc.get("token") != lock["doc"]["token"]:
        raise ConfirmationRunnerRefusal(
            "the run lock was replaced while this session held it — "
            "UNCERTAIN")
    os.unlink(p)


def unit_claim_path(out_root, unit_id) -> Path:
    return (Path(out_root) / "intervention"
            / f"{rn._safe(unit_id)}__CLAIM.json")


def claim_unit(out_root, unit_id) -> dict:
    """An O_EXCL claim per unit, so even without the run lock two
    sessions can never fit the same unit twice."""
    p = unit_claim_path(out_root, unit_id)
    doc = _claim_doc("unit_claim")
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                     0o600)
    except FileExistsError:
        raise ConfirmationRunnerRefusal(
            f"unit {unit_id!r} is already claimed by another session — "
            "a unit is executed exactly once")
    try:
        os.write(fd, json.dumps(doc, indent=1,
                                sort_keys=True).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return doc


def release_unit_claim(out_root, unit_id):
    p = unit_claim_path(out_root, unit_id)
    if p.is_file():
        os.unlink(p)


# ---------------- cumulative accounting (DR04 R6) ----------------

def prior_session_reports(out_root) -> list:
    """Every previous session report, self-identity re-derived. A report
    that does not re-derive makes the cumulative account UNCERTAIN."""
    out = []
    for p in sorted(Path(out_root).glob("SESSION_*_REPORT.json")):
        doc = m4._strict_json_file(p, p.name)
        if cp._selfsha(doc, "report_sha256") != \
                doc.get("report_sha256"):
            raise ConfirmationRunnerRefusal(
                f"{p.name}: session report self-identity does not "
                "re-derive — the cumulative account is UNCERTAIN")
        out.append(doc)
    return out


def cumulative_prior(out_root) -> dict:
    """The cumulative cost of every prior session in this run root.

    Without this, each session restarted the sealed wall from zero and N
    interruptions bought N times the frozen budget."""
    tot = {"sessions": 0, "wall_seconds": 0.0,
           "optimization_updates": 0, "evaluations": 0,
           "descriptor_evals": 0, "descriptor_seconds": 0.0,
           "units_new": 0}
    if not Path(out_root).exists():
        return tot
    for doc in prior_session_reports(out_root):
        tot["sessions"] += 1
        tot["wall_seconds"] += float(
            doc.get("session_wall_seconds",
                    doc.get("wall_seconds", 0.0)) or 0.0)
        tot["units_new"] += int(doc.get("units_new_this_session", 0))
        a = doc.get("accounting", {})
        for k in ("optimization_updates", "evaluations",
                  "descriptor_evals"):
            tot[k] += int(a.get(k, 0) or 0)
        tot["descriptor_seconds"] += float(
            a.get("descriptor_seconds", 0.0) or 0.0)
    tot["wall_seconds"] = round(tot["wall_seconds"], 3)
    tot["descriptor_seconds"] = round(tot["descriptor_seconds"], 6)
    return tot


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


IDENTITY_FIELDS = ("kind", "role", "family", "noise_coord", "width",
                   "generator_index", "generator_id", "model_seed",
                   "unit_role")
TERMINAL_UNIT_KEYS = ("manifest_sha256", "task_kind", "tape_id",
                      "tape_digest", "tape_tol", "genesis_digest",
                      "checkpoint_lineage", "stop_trajectory_digest",
                      "stop_trajectory_slope", "selected_stop_update",
                      "arms", "paired_primary_difference")
ARM_KEYS = ("restricted_endpoint", "cap_reached", "stopping_cause",
            "updates_done", "retention_margin", "fail_batch",
            "final_params_digest", "descriptors",
            "checkpoint_loss_stop")
LINEAGE_KEYS = ("params_digest", "parent", "updates")
INVALID_STATUS = "NUMERICALLY_INVALID_TASK_TRAINING"


def refuse_role_mismatch(rec, expect_role):
    """The sealed check refuses any unit id that does not carry
    ``::CONFIRMATION::``. This one refuses any record whose role is not
    EXACTLY the role of the census being read — in either direction, so a
    CONFIRMATION record can never be counted in a DEVELOPMENT fixture
    either."""
    uid = rec.get("unit_id", "")
    if f"::{expect_role}::" not in uid or \
            rec.get("role") not in (None, expect_role):
        raise ConfirmationRunnerRefusal(
            f"record {uid!r} (role {rec.get('role')!r}) is not a "
            f"{expect_role} unit — one census, one role, and outcomes of "
            "another role are structurally inadmissible")


def arm_log_path(out_root, unit_id, arm) -> Path:
    return (Path(out_root) / "intervention"
            / f"{rn._safe(unit_id)}__{arm}.jsonl")


def authenticate_arm_log(out_root, rec, arm) -> dict:
    """Read the RAW per-batch log of one arm and re-derive the arm facts
    from it. A producer summary is never sufficient: the endpoints, the
    update count, the stopping cause and the failing batch must all come
    back out of the authenticated lines, so a consistently altered pair of
    endpoints fails even when the declared difference agrees."""
    uid = rec["unit_id"]
    lp = arm_log_path(out_root, uid, arm)
    if not lp.is_file():
        raise ConfirmationRunnerRefusal(
            f"{lp.name} is ABSENT — a summary without its raw per-batch "
            f"log is never verified evidence for {uid}/{arm}")
    sp = rn._state_path(Path(out_root), uid, arm)
    if sp.exists() or Path(str(sp) + ".meta.json").exists():
        raise ConfirmationRunnerRefusal(
            f"{uid}/{arm}: a durable resume state survives beside a "
            "COMPLETE record — UNCERTAIN, never read as finished")
    lines = lp.read_text().splitlines()
    if not lines:
        raise ConfirmationRunnerRefusal(
            f"{lp.name} is EMPTY — no raw evidence, no verification")
    recs = []
    for i, line in enumerate(lines):
        r = m4._strict_json_text(line, f"{lp.name} {i}")
        if "record_sha256" not in r or \
                m4._self_sha(r, "record_sha256") != r["record_sha256"]:
            raise ConfirmationRunnerRefusal(
                f"{lp.name} line {i}: raw record self-identity does not "
                "re-derive — altered raw evidence is never authentic")
        if r.get("tape_digest") != rec.get("tape_digest"):
            raise ConfirmationRunnerRefusal(
                f"{lp.name} line {i} binds a foreign tape")
        if r.get("batch") != i:
            raise ConfirmationRunnerRefusal(
                f"{lp.name} line {i}: batch index {r.get('batch')!r} is "
                "not contiguous from zero — a log with holes is never "
                "a complete history")
        recs.append(r)
    a = rec["arms"][arm]
    endpoint = 0
    for r in recs:
        if r["outcome"] == "ACCEPTED":
            endpoint = r["cumulative_associations"]
    endpoint = min(endpoint, pv.MAX_BATCHES * pv.ASSOC_BATCH)
    if a["restricted_endpoint"] != endpoint:
        raise ConfirmationRunnerRefusal(
            f"{uid}/{arm}: declared restricted endpoint "
            f"{a['restricted_endpoint']!r} does not re-derive from the "
            f"raw log ({endpoint!r}) — producer aggregates never "
            "determine a verdict")
    updates = len(recs) * m4.UPDATES_PER_BATCH
    if a["updates_done"] != updates:
        raise ConfirmationRunnerRefusal(
            f"{uid}/{arm}: declared {a['updates_done']!r} updates, the "
            f"raw log accounts for {updates!r}")
    last = recs[-1]["outcome"]
    cap = len(recs) == pv.MAX_BATCHES and last == "ACCEPTED"
    if bool(a["cap_reached"]) is not cap:
        raise ConfirmationRunnerRefusal(
            f"{uid}/{arm}: declared cap_reached {a['cap_reached']!r} "
            "does not re-derive from the raw log")
    cause = "CAP_REACHED" if cap else last
    if a["stopping_cause"] != cause:
        raise ConfirmationRunnerRefusal(
            f"{uid}/{arm}: declared stopping cause "
            f"{a['stopping_cause']!r} does not re-derive ({cause!r})")
    fail_b = None if cap else (len(recs) - 1 if last != "ACCEPTED"
                               else None)
    if a["fail_batch"] != fail_b:
        raise ConfirmationRunnerRefusal(
            f"{uid}/{arm}: declared failing batch {a['fail_batch']!r} "
            f"does not re-derive ({fail_b!r})")
    margin = a["retention_margin"]
    streak = 0
    for r in recs:
        if r["ret_loss"] is None:
            continue
        knife = abs(float(r["ret_loss"]) - float(margin)) <= 1e-6
        streak = streak + 1 if float(r["ret_loss"]) > float(margin) \
            else 0
        if not knife and r["retention_streak"] != streak:
            raise ConfirmationRunnerRefusal(
                f"{uid}/{arm} batch {r['batch']}: the retention streak "
                "does not re-derive from the raw retention loss and the "
                "declared margin")
        streak = r["retention_streak"]
    return {"lines": len(recs), "restricted_endpoint": endpoint,
            "updates_done": updates}


def validate_unit_record(rec, unit, expect_role, out_root=None):
    """The COMPLETE terminal schema of a finished unit.

    The atomic file name is not a validation of completion, and neither is
    a self-consistent hash over two fields. A record is complete only when
    it carries its identity, its typed terminal state, its lineage, its
    tape and manifest identity, its four arms — and, when out_root is
    given, when the RAW per-batch logs those arms claim exist, authenticate
    and re-derive the arm facts."""
    refuse_role_mismatch(rec, expect_role)
    for k in IDENTITY_FIELDS:
        if k not in rec:
            raise ConfirmationRunnerRefusal(
                f"{unit['unit_id']}: record omits the identity field "
                f"{k!r} — an incomplete identity is never a completed "
                "unit")
        if rec[k] != unit[k]:
            raise ConfirmationRunnerRefusal(
                f"{unit['unit_id']}: record identity field {k!r} is "
                f"{rec[k]!r}, the census says {unit[k]!r}")
    if rec["generator_id"] != gb.generator_id(
            unit["role"], unit["family"], unit["noise_coord"],
            unit["generator_index"]):
        raise ConfirmationRunnerRefusal(
            f"{unit['unit_id']}: generator identity does not re-derive "
            "from role/family/noise/index")
    status = rec.get("unit_status")
    if status is not None:
        if status != INVALID_STATUS:
            raise ConfirmationRunnerRefusal(
                f"{unit['unit_id']}: unit_status {status!r} is not a "
                "typed terminal state")
        if not isinstance(rec.get("invalid_at_update"), int):
            raise ConfirmationRunnerRefusal(
                f"{unit['unit_id']}: a typed-invalid unit must name the "
                "update at which it became invalid")
        for k in ("manifest_sha256", "tape_id", "tape_digest",
                  "task_kind"):
            if k not in rec:
                raise ConfirmationRunnerRefusal(
                    f"{unit['unit_id']}: typed-invalid record omits "
                    f"{k!r}")
        if out_root is not None:
            stray = [arm for arm in pv.CHECKPOINTS
                     if arm_log_path(out_root, rec["unit_id"],
                                     arm).exists()]
            if stray:
                raise ConfirmationRunnerRefusal(
                    f"{unit['unit_id']}: typed-invalid unit carries arm "
                    f"logs {stray[:2]} — UNCERTAIN")
        return {"status": INVALID_STATUS, "arms": {}}
    for k in TERMINAL_UNIT_KEYS:
        if k not in rec:
            raise ConfirmationRunnerRefusal(
                f"{unit['unit_id']}: record omits {k!r} — the terminal "
                "schema is complete or the unit is not done")
    if set(rec["checkpoint_lineage"]) != set(pv.CHECKPOINTS):
        raise ConfirmationRunnerRefusal(
            f"{unit['unit_id']}: the four checkpoints are not all "
            "present in the lineage")
    for name, li in rec["checkpoint_lineage"].items():
        for k in LINEAGE_KEYS:
            if k not in li:
                raise ConfirmationRunnerRefusal(
                    f"{unit['unit_id']}: lineage {name} omits {k!r}")
    if rec["genesis_digest"] != \
            rec["checkpoint_lineage"]["initialization"]["params_digest"]:
        raise ConfirmationRunnerRefusal(
            f"{unit['unit_id']}: genesis digest does not equal the "
            "initialization lineage")
    if set(rec["arms"]) != set(pv.CHECKPOINTS):
        raise ConfirmationRunnerRefusal(
            f"{unit['unit_id']}: the four checkpoint arms are not all "
            "present")
    for arm, a in rec["arms"].items():
        for k in ARM_KEYS:
            if k not in a:
                raise ConfirmationRunnerRefusal(
                    f"{unit['unit_id']}/{arm}: arm omits {k!r}")
    want = (rec["arms"]["calibration_stop"]["restricted_endpoint"]
            - rec["arms"]["initialization"]["restricted_endpoint"])
    if rec["paired_primary_difference"] != want:
        raise ConfirmationRunnerRefusal(
            f"{unit['unit_id']}: the declared paired difference does not "
            "equal the difference of its own arm endpoints")
    logs = {}
    if out_root is not None:
        for arm in pv.CHECKPOINTS:
            logs[arm] = authenticate_arm_log(out_root, rec, arm)
    return {"status": "COMPLETE", "arms": logs}


def read_complete_unit_record(path, unit, expect_role,
                              out_root=None):
    """None when the unit is not recorded; the record when it IS
    complete and binds this unit; a typed refusal when a record
    exists that cannot be read as complete — a damaged or foreign
    record is UNCERTAIN, never silently re-run and never counted
    in the numerator.

    DR04 R5: "complete" now means the COMPLETE terminal schema — identity,
    typed state, lineage, tape/manifest identity, four arms — and, when
    out_root is given, the authenticated raw per-batch logs. A document
    carrying only a unit id and its own hash is no longer a completed
    unit."""
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
    validate_unit_record(rec, unit, expect_role,
                         out_root if out_root is not None
                         else path.parent.parent)
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
    cp_ = unit_claim_path(out, unit["unit_id"])
    if cp_.is_file() and _holder_is_live(cp_):
        raise ConfirmationRunnerRefusal(
            f"unit {unit['unit_id']!r} is claimed by a LIVE session — "
            "its bytes are never set aside under another writer")
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

def new_accounting(prior_wall_seconds=0.0) -> dict:
    """The accounting shape the sealed _limits and unit machinery
    expect; every increment is made by the sealed functions.

    DR04 R6: ``prior_wall_seconds`` moves t0 BACK by the wall already
    spent in this run root, so the sealed wall limit is enforced over the
    CUMULATIVE cost of the run and not once per interruption."""
    return {"optimization_updates": 0, "evaluations": 0,
            "descriptor_seconds": 0.0, "descriptor_evals": 0,
            "t0": time.monotonic() - float(prior_wall_seconds),
            "session_t0": time.monotonic(),
            "prior_wall_seconds": round(float(prior_wall_seconds), 3)}


def next_session_index(out_root) -> int:
    return 1 + len([p for p in
                    Path(out_root).glob("SESSION_*_REPORT.json")
                    if p.is_file()])


def write_session_report(out_root, doc) -> Path:
    """One append-only report per session. Named without the
    token CONFIRMATION so the DEVELOPMENT probe's zero-artifact
    assertion stays exactly as sealed; the role is a field."""
    out = Path(out_root)
    prior = cumulative_prior(out)
    n = prior["sessions"] + 1
    if n != next_session_index(out):
        raise ConfirmationRunnerRefusal(
            "the session index and the verified prior-report census "
            "disagree — the cumulative account is UNCERTAIN")
    doc = dict(doc)
    doc["schema"] = "m4_confirmation_session_report.v1"
    doc["session"] = n
    doc["prior_sessions"] = prior["sessions"]
    doc["cumulative"] = {
        "sessions": n,
        "wall_seconds": round(prior["wall_seconds"]
                              + float(doc.get("session_wall_seconds",
                                              0.0)), 3),
        "optimization_updates": prior["optimization_updates"]
        + int(doc.get("accounting", {}).get(
            "optimization_updates", 0)),
        "evaluations": prior["evaluations"]
        + int(doc.get("accounting", {}).get("evaluations", 0)),
        "descriptor_evals": prior["descriptor_evals"]
        + int(doc.get("accounting", {}).get("descriptor_evals", 0)),
        "units_new": prior["units_new"]
        + int(doc.get("units_new_this_session", 0))}
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
    # DR04 R6: one writer per run root, and the wall counts the whole run.
    lock = acquire_run_lock(out)
    prior_cost = cumulative_prior(out)
    if acct.get("prior_wall_seconds", 0.0) == 0.0 and \
            prior_cost["wall_seconds"] > 0.0:
        acct["t0"] -= prior_cost["wall_seconds"]
        acct["prior_wall_seconds"] = prior_cost["wall_seconds"]
    acct.setdefault("session_t0", acct["t0"])
    try:
        complete_recs = {}
        pending = []
        for u in units:
            if f"::{expect_role}::" not in u["unit_id"]:
                raise ConfirmationRunnerRefusal(
                    f"unit {u['unit_id']!r} is not a "
                    f"{expect_role} unit — one census, one role")
            rp = unit_record_path(out, u["unit_id"])
            rec = read_complete_unit_record(rp, u, expect_role, out)
            if rec is not None:
                complete_recs[u["unit_id"]] = (u, rec)
            else:
                pending.append(u)
        # DR04 R5: the disjointness proof is REVALIDATED for resumed
        # units, not inherited from a previous session's word, and each
        # resumed record must bind the generator it claims.
        gen_checked = set()
        for u, rec in complete_recs.values():
            gid = u["generator_id"]
            if gid in gen_checked:
                continue
            g = gb.generate(u["role"], u["family"], u["noise_coord"],
                            u["generator_index"],
                            allow_confirmation=allow_confirmation)
            gb.consumer_verify(g)
            verify_role_disjointness(
                generator_array_digests(g), prior_digests)
            gen_checked.add(gid)
            if rec.get("manifest_sha256") != \
                    g["manifest"]["manifest_sha256"]:
                raise ConfirmationRunnerRefusal(
                    f"{u['unit_id']}: the resumed record binds manifest "
                    f"{rec.get('manifest_sha256')!r}, the generator it "
                    "names re-derives another — UNCERTAIN")
        complete = len(complete_recs)
        new = 0
        aborted = 0
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
            claim_unit(out, u["unit_id"])
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
            validate_unit_record(rec, u, expect_role, out)
            write_unit_record(unit_record_path(out, u["unit_id"]),
                              rec)
            release_unit_claim(out, u["unit_id"])
            complete += 1
            new += 1
        left = len(units) - complete
        now = time.monotonic()
        return {"role": expect_role,
                "units_total": len(units),
                "units_complete": complete,
                "units_pending": left,
                "units_new_this_session": new,
                "partial_attempts_set_aside": aborted,
                "stale_locks_set_aside":
                    lock["stale_locks_set_aside"],
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
                "prior_sessions_wall_seconds":
                    prior_cost["wall_seconds"],
                "session_wall_seconds": round(
                    now - acct["session_t0"], 2),
                "wall_seconds": round(now - acct["t0"], 2)}
    finally:
        release_run_lock(lock)


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

LEDGER_GATE_KEYS = ("census_role", "successor_sha256",
                    "review_record_sha256", "execution_record_sha256",
                    "executing_head",
                    "executing_implementation_sha256",
                    "prior_role_digests")
DEV_GATE_AUTHORITY = ("DEVELOPMENT_FIXTURE_NO_AUTHORITY_NO_"
                      "SCIENTIFIC_VERDICT")


def validate_ledger_gates(census, gates):
    """DR04 R1/R3: the ledger's gate map is EVIDENCE, and it is validated.

    An empty gate map used to be accepted and never read again; a run whose
    ledger names no successor, no records and no executing implementation
    can never reach a verdict."""
    role = census_role(census)
    if not isinstance(gates, dict) or set(gates) != set(
            LEDGER_GATE_KEYS):
        raise ConfirmationRunnerRefusal(
            "the pre-result ledger gate map is not the exact schema "
            f"(missing/extra: "
            f"{sorted(set(gates or {}) ^ set(LEDGER_GATE_KEYS))}) — a "
            "ledger that names no authority is never a ledger")
    if gates["census_role"] != role:
        raise ConfirmationRunnerRefusal(
            f"the ledger gate role {gates['census_role']!r} is not the "
            f"census role {role!r}")
    if gates["successor_sha256"] != census["successor_sha256"]:
        raise ConfirmationRunnerRefusal(
            "the ledger gate successor is not the census successor")
    hexish = ("review_record_sha256", "execution_record_sha256",
              "executing_head", "executing_implementation_sha256")
    if role == "CONFIRMATION":
        for k in hexish:
            v = gates[k]
            if not (isinstance(v, str) and len(v) in (40, 64)
                    and all(c in "0123456789abcdef" for c in v)):
                raise ConfirmationRunnerRefusal(
                    f"ledger gate {k!r} is not a recorded digest — a "
                    "CONFIRMATION ledger names its authority and its "
                    "executable implementation or it does not exist")
    else:
        for k in ("review_record_sha256", "execution_record_sha256"):
            if gates[k] != DEV_GATE_AUTHORITY:
                raise ConfirmationRunnerRefusal(
                    f"a {role} fixture ledger must carry "
                    f"{DEV_GATE_AUTHORITY!r} in {k!r} — a fixture never "
                    "carries anything that resembles an approval digest")


def write_pre_result_ledger(out_root: Path, census,
                            gates) -> Path:
    validate_ledger_gates(census, gates)
    p = Path(out_root) / "CONFIRMATION_PRE_RESULT_LEDGER.json"
    doc = {
        "schema": "m4_confirmation_pre_result_ledger.v1",
        "census_sha256": census["census_sha256"],
        "census_role": census_role(census),
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
    sti = _git(repo_root, "status", "--porcelain", "--",
               *cp.IMPLEMENTATION_FILES)
    if sti.stdout.strip():
        raise ConfirmationRunnerRefusal(
            "the executing implementation differs from the commit it "
            f"claims ({sti.stdout.strip().splitlines()[0]!r}) — "
            "authority binds bytes, never a file name")
    head = _git(repo_root, "rev-parse",
                "HEAD").stdout.strip()
    impl = cp.implementation_digest(repo_root)
    if out_root is None:
        raise ConfirmationRunnerRefusal(
            "no output root was provided")
    out_root = Path(out_root)
    census = materialize_census(successor, auth["design"])
    prior = prior_role_digest_census()
    gates = {
        "census_role": "CONFIRMATION",
        "successor_sha256": successor["successor_sha256"],
        "review_record_sha256":
            records["review"]["record_sha256"],
        "execution_record_sha256":
            records["execution"]["record_sha256"],
        "executing_head": head,
        "executing_implementation_sha256": impl,
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
        for k in ("census_role", "successor_sha256",
                  "review_record_sha256",
                  "execution_record_sha256", "executing_head",
                  "executing_implementation_sha256"):
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


def development_verification_probe(repo_root=REPO, out_root=None,
                                  generators=2, seeds=3,
                                  batch_units=None) -> dict:
    """DR04: the COMPLETE path — census, pre-result ledger, execution body
    and the independent verifier — over a DEVELOPMENT fixture population.

    Every repair is exercised on the SAME code the CONFIRMATION path runs;
    only the role, the population and the authority declaration differ, and
    all three are named in the ledger, the census and the verdict. No
    CONFIRMATION array is constructed, no authority record is read,
    created or simulated, and the verdict explicitly carries
    NON_CONFIRMATORY."""
    repo_root = Path(repo_root)
    out = Path(out_root)
    design = cp.bind_calibration_evidence(repo_root)["design"]
    pinned = cp.verify_confirmation_successor(repo_root)
    fixture = development_fixture_successor(
        generators=generators, seeds=seeds, successor=pinned)
    census = materialize_census(fixture, design, "DEVELOPMENT")
    units = census_units(fixture, "DEVELOPMENT")
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    ledger = out / "CONFIRMATION_PRE_RESULT_LEDGER.json"
    if not ledger.is_file():
        write_pre_result_ledger(out, census, {
            "census_role": "DEVELOPMENT",
            "successor_sha256": fixture["successor_sha256"],
            "review_record_sha256": DEV_GATE_AUTHORITY,
            "execution_record_sha256": DEV_GATE_AUTHORITY,
            "executing_head": _git(repo_root, "rev-parse",
                                   "HEAD").stdout.strip(),
            "executing_implementation_sha256":
                cp.implementation_digest(repo_root),
            "prior_role_digests": 0})
    prior = role_array_digests(design, cells_of_units(units),
                              roles=("CALIBRATION",))
    acct = new_accounting()
    body = execute_confirmation_units(
        design, units, out, acct, expect_role="DEVELOPMENT",
        allow_confirmation=False, prior_digests=prior,
        batch_units=batch_units)
    leaked = [str(q) for q in out.rglob("*")
              if "CONFIRMATION" in q.name
              and q.name != "CONFIRMATION_PRE_RESULT_LEDGER.json"]
    if leaked:
        raise ConfirmationRunnerRefusal(
            f"the DEVELOPMENT fixture created CONFIRMATION-named "
            f"artifacts: {leaked[:3]}")
    write_session_report(out, dict(body))
    out_doc = {"body": body,
               "fixture_successor_sha256":
                   fixture["successor_sha256"],
               "census_sha256": census["census_sha256"],
               "census_units_total": census["units_total"]}
    if body["census_complete"]:
        out_doc["verification"] = verify_confirmation_run(
            repo_root, out, fixture, expect_role="DEVELOPMENT")
    else:
        out_doc["verification"] = {
            "verdict": "NOT_ATTEMPTED_CENSUS_INCOMPLETE",
            "units_pending": body["units_pending"]}
    return out_doc


# ---------------- independent verifier ----------------

CHECKPOINT_POPULATION_RULE = (
    "one observation per VERIFIED COMPLETE generator identity (family, "
    "noise, width, generator index), the nested model seeds averaged "
    "first, below-floor slots excluded; generator indices are NEVER "
    "pooled across families, noise regimes or widths")


def analyse_complete_population(successor,
                                per_generator_seed_effects) -> dict:
    """DR04 R2/R4: the analysis stage, pure and exact.

    ``per_generator_seed_effects`` is
    ``{"fam::nz": {width: {"gN": {model_seed: paired effect}}}}`` — a
    SEED-KEYED map, so a repeated seed cannot be mistaken for a nested
    repetition and a row count can never stand in for the sealed seed set.

    In order: the exact seed set makes a generator complete; the attrition
    floor is applied per eligible slot BEFORE any contrast; the fifteenth
    contrast is built from the surviving complete generator IDENTITIES with
    the cross-cell/width aggregation named, never from a bare generator
    index; and nothing is called complete while any identity is partial or
    any slot is below its floor."""
    seeds_needed = successor["nested_seeds_per_generator"]
    want_seeds = set(range(seeds_needed))
    complete = {}
    partial = {}
    for cell, by_w in per_generator_seed_effects.items():
        for wd, by_g in by_w.items():
            for gkey, by_seed in by_g.items():
                if not isinstance(by_seed, dict):
                    raise ConfirmationRunnerRefusal(
                        f"{cell}::w{wd}::{gkey}: the effects are not "
                        "keyed by model seed — a row count is never the "
                        "sealed seed set")
                if set(by_seed) == want_seeds:
                    complete.setdefault(cell, {}).setdefault(
                        wd, {})[gkey] = float(
                            np.mean([by_seed[k]
                                     for k in sorted(by_seed)]))
                else:
                    partial[f"{cell}::w{wd}::{gkey}"] = sorted(by_seed)
    floor = successor["attrition"]["min_complete_required"]
    incomplete_slots = {}
    for s in successor["eligible_slots"]:
        prefix, w = s["cell"].rsplit("::w", 1)
        n_complete = len(complete.get(prefix, {}).get(int(w), {}))
        if n_complete < floor:
            incomplete_slots[s["cell"]] = {
                "status": "CONFIRMATION_INCOMPLETE",
                "complete_generators": n_complete,
                "min_complete_required": floor}
            if prefix in complete:
                complete[prefix].pop(int(w), None)
    ck_complete = {}
    ck_population = {}
    for s in successor["eligible_slots"]:
        if s["cell"] in incomplete_slots:
            continue
        prefix, w = s["cell"].rsplit("::w", 1)
        gens = complete.get(prefix, {}).get(int(w), {})
        ck_population[s["cell"]] = len(gens)
        for gkey, val in gens.items():
            ck_complete[f"{prefix}::w{w}::{gkey}"] = val
    analysis = cp.sixteen_contrasts(successor, complete, ck_complete)
    population_complete = not incomplete_slots and not partial
    for k in analysis["contrasts"]:
        analysis["contrasts"][k]["population_complete"] = \
            population_complete
    return {"complete": complete,
            "generators_incomplete_seed_sets": partial,
            "confirmation_incomplete": incomplete_slots,
            "checkpoint_population_rule": CHECKPOINT_POPULATION_RULE,
            "checkpoint_population": ck_population,
            "population_complete": population_complete,
            "analysis": analysis,
            "width_heterogeneity_secondary":
                cp.width_heterogeneity(successor, complete)}


def reconstruct_unit(design, unit, rec, run_root, gcache,
                     acct) -> dict:
    """DR04 R3: rebuild the unit from its own inputs and compare EVERY
    fact with the record and with the raw per-batch logs.

    Generator, manifest, tape, the four checkpoints and their lineage, all
    four arms, every batch line, the descriptors and the paired difference.
    A producer's summary is never sufficient; a consistently altered pair
    of endpoints fails here even when the declared difference agrees,
    because the endpoints themselves are re-derived."""
    uid = unit["unit_id"]
    key = (unit["role"], unit["family"], unit["noise_coord"],
           unit["generator_index"])
    if key not in gcache:
        g = gb.generate(*key,
                        allow_confirmation=(unit["role"]
                                            == "CONFIRMATION"))
        gb.consumer_verify(g)
        gcache[key] = g
    g = gcache[key]
    if rec["manifest_sha256"] != g["manifest"]["manifest_sha256"]:
        raise ConfirmationRunnerRefusal(
            f"{uid}: the record binds a manifest the generator it names "
            "does not re-derive")
    kind = pv.task_kind(unit["family"])
    if rec["task_kind"] != kind:
        raise ConfirmationRunnerRefusal(
            f"{uid}: declared task kind does not re-derive")
    tape = pv.association_tape(design["design_sha256"], g,
                               unit["width"], unit["model_seed"])
    if rec["tape_digest"] != tape["digest"] or \
            rec["tape_id"] != tape["tape_id"]:
        raise ConfirmationRunnerRefusal(
            f"{uid}: tape identity does not re-derive — arms never share "
            "an unverified tape")
    ck = pv.build_checkpoints(g, unit["width"], unit["model_seed"])
    if ck["numerically_invalid"]:
        if rec.get("unit_status") != INVALID_STATUS or \
                rec.get("invalid_at_update") != ck["invalid_at_update"]:
            raise ConfirmationRunnerRefusal(
                f"{uid}: the replay is numerically invalid at "
                f"{ck['invalid_at_update']!r}; the record does not "
                "declare that typed state")
        return {"status": INVALID_STATUS}
    if rec.get("unit_status") is not None:
        raise ConfirmationRunnerRefusal(
            f"{uid}: claims an invalid status the replay does not derive")
    if rec["genesis_digest"] != \
            ck["checkpoints"]["initialization"]["params_digest"]:
        raise ConfirmationRunnerRefusal(
            f"{uid}: genesis does not re-derive")
    if rec["stop_trajectory_digest"] != ck["stop_trajectory_digest"] \
            or rec["selected_stop_update"] != ck["selected_stop_update"]:
        raise ConfirmationRunnerRefusal(
            f"{uid}: the early-stopping trajectory does not re-derive")
    for name, c in ck["checkpoints"].items():
        li = rec["checkpoint_lineage"][name]
        if li["params_digest"] != c["params_digest"] or \
                li["parent"] != c["parent"] or \
                li["updates"] != c["updates"]:
            raise ConfirmationRunnerRefusal(
                f"{uid}: checkpoint {name} lineage does not replay")
    acct["optimization_updates"] += \
        ck["checkpoints"]["post_stop_bounded"]["updates"]
    endpoints = {}
    for arm in pv.CHECKPOINTS:
        res = pv.run_intervention(g, tape, ck["checkpoints"][arm], kind)
        acct["optimization_updates"] += res["updates_done"]
        acct["evaluations"] += len(res["records"])
        a = rec["arms"][arm]
        if a["restricted_endpoint"] != res["restricted_endpoint"] or \
                bool(a["cap_reached"]) is not res["cap_reached"] or \
                a["stopping_cause"] != res["stopping_cause"] or \
                a["updates_done"] != res["updates_done"] or \
                a["retention_margin"] != res["retention_margin"] or \
                a["final_params_digest"] != res["final_params_digest"]:
            raise ConfirmationRunnerRefusal(
                f"{uid}/{arm}: the endpoint facts do not equal the "
                "replayed derivation")
        lp = arm_log_path(run_root, uid, arm)
        lines = lp.read_text().splitlines()
        if len(lines) != len(res["records"]):
            raise ConfirmationRunnerRefusal(
                f"{lp.name}: the raw log length does not equal the "
                "replay")
        for i, line in enumerate(lines):
            raw = m4._strict_json_text(line, f"{lp.name} {i}")
            claim = {k: raw[k] for k in raw
                     if k not in ("record_sha256", "tape_digest")}
            if claim != res["records"][i]:
                raise ConfirmationRunnerRefusal(
                    f"{lp.name} record {i} does not replay")
        fresh = rn._descriptors(ck["checkpoints"][arm]["params"],
                               {"descriptor_seconds": 0.0,
                                "descriptor_evals": 0})
        have = dict(a["descriptors"])
        fresh.pop("descriptor_seconds", None)
        have.pop("descriptor_seconds", None)
        if fresh != have:
            raise ConfirmationRunnerRefusal(
                f"{uid}/{arm}: descriptors do not re-derive from the "
                "original float64 parameters")
        acct["descriptor_evals"] += 1
        loss = round(pv.loss_task(
            kind, ck["checkpoints"][arm]["params"],
            g["X_stop"], g["y_stop"]), 8)
        if a["checkpoint_loss_stop"] != loss:
            raise ConfirmationRunnerRefusal(
                f"{uid}/{arm}: the stop-split checkpoint loss does not "
                "re-derive")
        endpoints[arm] = res["restricted_endpoint"]
    eff = (endpoints["calibration_stop"]
           - endpoints["initialization"])
    if rec["paired_primary_difference"] != eff:
        raise ConfirmationRunnerRefusal(
            f"{uid}: the paired primary difference does not re-derive "
            "from the replayed endpoints")
    return {"status": "COMPLETE", "paired_primary_difference": eff,
            "endpoints": endpoints}


def verify_confirmation_run(repo_root, run_root,
                            successor=None, *,
                            expect_role="CONFIRMATION") -> dict:
    """Reconstruct EVERYTHING from raw records; producer aggregates never
    determine a verdict.

    DR04, in this order and with no way round any step:
      1. the successor is the ORDER-PINNED one whenever the verdict could
         be confirmatory, and never the pinned one for a fixture role;
      2. the pre-result ledger exists, self-re-derives, binds the
         successor-derived census AND its role;
      3. its gate map is VALIDATED — authority digests and the digest of
         the executable implementation, which must equal the
         implementation doing the verifying; for CONFIRMATION the two
         external records are re-read and must match the ledger;
      4. every record file carries the CANONICAL name of the unit it
         declares, that unit is IN the census, and no identity appears
         twice — a repeated seed is refused by name;
      5. the population is COMPLETE: a missing record refuses, it never
         narrows the denominator;
      6. every record is authenticated and NUMERICALLY RE-DERIVED —
         generator, manifest, tape, checkpoints, lineage, four arms, every
         raw per-batch line, descriptors, paired difference;
      7. a generator is complete only with the EXACT sealed seed set;
      8. the attrition floor is applied per eligible slot BEFORE any
         contrast;
      9. all 16 contrasts, the fifteenth included, are built from verified
         complete generator IDENTITIES (family, noise, width, index) with
         the cross-cell/width aggregation named explicitly;
     10. VERIFIED is returned only when the population is complete and
         every re-derivation held."""
    repo_root = Path(repo_root)
    run_root = Path(run_root)
    design = cp.bind_calibration_evidence(repo_root)["design"]
    pinned = cp.verify_confirmation_successor(repo_root)
    if expect_role == "CONFIRMATION":
        successor = successor or pinned
        if successor["successor_sha256"] != \
                pinned["successor_sha256"]:
            raise ConfirmationRunnerRefusal(
                "a confirmatory verdict is only ever derived from the "
                "order-pinned successor identity — a substituted "
                "population never confirms")
    else:
        if successor is None:
            raise ConfirmationRunnerRefusal(
                f"a {expect_role} verification must name its own fixture "
                "successor; it never borrows the confirmatory one")
        if successor["successor_sha256"] == pinned["successor_sha256"]:
            raise ConfirmationRunnerRefusal(
                "the order-pinned successor declares the CONFIRMATION "
                f"population; it never yields a {expect_role} verdict")
    impl = cp.implementation_digest(repo_root)
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
    if ledger.get("census_role", "CONFIRMATION") != expect_role:
        raise ConfirmationRunnerRefusal(
            f"the ledger was written for role "
            f"{ledger.get('census_role')!r}, this verification expects "
            f"{expect_role!r} — a fixture root never yields a "
            "CONFIRMATION verdict and a CONFIRMATION root is never read "
            "as a fixture")
    census = materialize_census(successor, design, expect_role)
    if ledger["census_sha256"] != census["census_sha256"]:
        raise ConfirmationRunnerRefusal(
            "ledger census is not the successor-derived census")
    validate_ledger_gates(census, ledger["gates"])
    gates = ledger["gates"]
    if gates["executing_implementation_sha256"] != impl:
        raise ConfirmationRunnerRefusal(
            "the ledger was written by implementation "
            f"{gates['executing_implementation_sha256'][:16]}… and this "
            f"verifier is {impl[:16]}… — a verdict is never read out of "
            "a run produced by other code")
    authority = {"role": expect_role}
    if expect_role == "CONFIRMATION":
        records = cp.require_both_records(successor, impl)
        if records["review"]["record_sha256"] != \
                gates["review_record_sha256"] or \
                records["execution"]["record_sha256"] != \
                gates["execution_record_sha256"]:
            raise ConfirmationRunnerRefusal(
                "the installed authority records are not the records the "
                "pre-result ledger was written under")
        authority = {
            "role": expect_role,
            "review_record_sha256": records["review"]["record_sha256"],
            "execution_record_sha256":
                records["execution"]["record_sha256"],
            "reviewed_implementation_sha256":
                records["review"]["reviewed_implementation_sha256"]}
        verdict_authority = "CONFIRMATORY"
    else:
        authority["declaration"] = DEV_GATE_AUTHORITY
        verdict_authority = (f"NON_CONFIRMATORY_{expect_role}_FIXTURE_"
                             "NO_SCIENTIFIC_VERDICT")
    # ---- 4: exact census membership, canonical names, no duplicates ----
    by_uid = {u["unit_id"]: u for u in census_units(successor,
                                                   expect_role)}
    seen = {}
    for p in sorted(run_root.glob("intervention/*_summary.json")):
        r = m4._strict_json_file(p, p.name)
        uid = r.get("unit_id")
        if not isinstance(uid, str):
            raise ConfirmationRunnerRefusal(
                f"{p.name}: the record declares no unit identity")
        if expect_role == "CONFIRMATION":
            refuse_foreign_role_record(r)
        refuse_role_mismatch(r, expect_role)
        canon = unit_record_path(run_root, uid)
        if p != canon:
            u = by_uid.get(uid)
            repeat = ""
            if canon.is_file() and u is not None:
                repeat = (f" and it REPEATS model seed s{u['model_seed']} "
                          f"of generator {u['generator_id']}, whose "
                          "canonical record already exists — a repetition "
                          "is never an independent observation")
            raise ConfirmationRunnerRefusal(
                f"{p.name} is not the canonical record name of the unit "
                f"it declares ({canon.name}) — an arbitrarily named "
                f"document is never an observation{repeat}")
        if uid not in by_uid:
            raise ConfirmationRunnerRefusal(
                f"{uid} is NOT in the exact census — an out-of-census "
                "identity is never counted (census "
                f"{census['units_total']} units, "
                f"{census['census_sha256'][:16]}…)")
        if uid in seen:
            u = by_uid[uid]
            raise ConfirmationRunnerRefusal(
                f"{uid} appears more than once: model seed "
                f"s{u['model_seed']} of generator {u['generator_id']} is "
                "REPEATED — a repetition is never an independent "
                "observation")
        seen[uid] = r
    missing = sorted(set(by_uid) - set(seen))
    if missing:
        raise ConfirmationRunnerRefusal(
            f"POPULATION_INCOMPLETE: {len(missing)} of "
            f"{len(by_uid)} census units have no record (first "
            f"{missing[0]}) — an incomplete population refuses, it never "
            "narrows the denominator")
    # ---- 5/6: authenticate and numerically re-derive every record ----
    acct = {"optimization_updates": 0, "evaluations": 0,
            "descriptor_seconds": 0.0, "descriptor_evals": 0}
    gcache = {}
    attrition = {}
    per_gen = {}
    costs = {}
    for uid in sorted(seen):
        unit = by_uid[uid]
        rec = seen[uid]
        if "record_sha256" not in rec or \
                m4._self_sha(rec, "record_sha256") != \
                rec["record_sha256"]:
            raise ConfirmationRunnerRefusal(
                f"{uid}: the record self-identity does not re-derive — "
                "an unauthenticated summary is never verified evidence")
        validate_unit_record(rec, unit, expect_role, run_root)
        rep = reconstruct_unit(design, unit, rec, run_root, gcache,
                               acct)
        slot = f"{unit['family']}::{unit['noise_coord']}::w{unit['width']}"
        if rep["status"] == INVALID_STATUS:
            attrition.setdefault(slot, []).append(uid)
            continue
        cell = f"{unit['family']}::{unit['noise_coord']}"
        per_gen.setdefault(cell, {}).setdefault(
            unit["width"], {}).setdefault(
            f"g{unit['generator_index']}", {})[
            unit["model_seed"]] = rep["paired_primary_difference"]
        costs[uid] = {an: a["updates_done"]
                      for an, a in rec["arms"].items()}
    stage = analyse_complete_population(successor, per_gen)
    complete = stage["complete"]
    partial = stage["generators_incomplete_seed_sets"]
    incomplete_slots = stage["confirmation_incomplete"]
    ck_population = stage["checkpoint_population"]
    analysis = stage["analysis"]
    hetero = stage["width_heterogeneity_secondary"]
    population_complete = stage["population_complete"]
    verdict = "VERIFIED" if population_complete else \
        "POPULATION_INCOMPLETE_NO_PRIMARY_VERDICT"
    return {"verdict": verdict,
            "verdict_authority": verdict_authority,
            "expect_role": expect_role,
            "successor_sha256": successor["successor_sha256"],
            "census_sha256": census["census_sha256"],
            "census_units_total": census["units_total"],
            "implementation_sha256": impl,
            "authority": authority,
            "records_verified": len(seen),
            "records_numerically_rederived":
                len(seen) - sum(len(v) for v in attrition.values()),
            "reconstruction": "FULL_REPLAY_FROM_RAW_RECORDS",
            "attrition": {k: len(v) for k, v in attrition.items()},
            "generators_incomplete_seed_sets": partial,
            "confirmation_incomplete": incomplete_slots,
            "population_complete": population_complete,
            "checkpoint_population_rule": CHECKPOINT_POPULATION_RULE,
            "checkpoint_population": ck_population,
            "analysis": analysis,
            "width_heterogeneity_secondary": hetero,
            "replay_accounting": {
                "optimization_updates": acct["optimization_updates"],
                "evaluations": acct["evaluations"],
                "descriptor_evals": acct["descriptor_evals"]},
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
    w = sub.add_parser("development-verification-probe")
    w.add_argument("--out", required=True)
    w.add_argument("--generators", type=int, default=2)
    w.add_argument("--seeds", type=int, default=3)
    w.add_argument("--batch-units", type=int, default=None)
    sub.add_parser("authority-digest")
    a = ap.parse_args(argv)
    if a.cmd in ("execute", "development-probe",
                 "development-execution-probe",
                 "development-verification-probe"):
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
    elif a.cmd == "development-verification-probe":
        print(json.dumps(development_verification_probe(
            out_root=Path(a.out), generators=a.generators,
            seeds=a.seeds, batch_units=a.batch_units),
            indent=1, default=str))
    elif a.cmd == "authority-digest":
        # What an approval must pin: the bytes that would run. Printing it
        # authorizes nothing and installs nothing.
        print(json.dumps({
            "implementation_sha256": cp.implementation_digest(REPO),
            "files": cp.implementation_file_digests(REPO),
            "reviewed_tip_pinned_by_the_order": cp.REVIEWED_TIP,
            "note": "the Musashi design-review record must carry this "
                    "implementation_sha256 in reviewed_implementation_"
                    "sha256; the gate compares it with the live bytes "
                    "before any ledger exists"}, indent=1))
    elif a.cmd == "verify":
        print(json.dumps(verify_confirmation_run(
            REPO, Path(a.run_root)), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
