"""DR04 POST: each frozen counterexample, answered on the same public path.

Reads as a pair with
``docs/audits/evidence/repro_runs/m4_dr04_verifiability_pre_2026_09_26.{py,out}``,
which was committed BEFORE the repair. Every probe below is the PRE probe,
re-run against the repaired candidate, plus the POSITIVE case the repair has
to keep working.

This is evidence, not an approval and not a scientific measurement. It is
the successor's own POST; the auditor's review of the repaired candidate is
a separate act.

What is NOT done here, as in the PRE:
  - no CONFIRMATION array is constructed, fitted or scored;
  - no external authority record is created, installed or simulated (the
    implementation binding is proven at the ledger and verification
    boundary, with no record of any kind);
  - CONFIRMATION is not run: the gate chain still refuses first and
    creates nothing;
  - every positive case runs over a DEVELOPMENT fixture population — real
    generators, real fits, real raw per-batch logs — through the SAME code
    the CONFIRMATION path runs.

Run (CPU only, light):
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \\
  $HOME/.local/bin/crispdm-run -m 3G -t 1800 -n m4verify -- \\
    python3 -B docs/audits/evidence/repro_runs/\\
m4_dr04_verifiability_post_2026_09_26.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))

import m4_confirmation_protocol as cp  # noqa: E402
import m4_confirmation_runner as cr  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402
import m4_v5_protocol as pv  # noqa: E402
import m4_v5_runner as rn  # noqa: E402


def say(tag, payload):
    print(f"{tag} {json.dumps(payload, sort_keys=True, default=str)}"
          if not isinstance(payload, str) else f"{tag} {payload}",
          flush=True)


def refused(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not refuse")


DESIGN = cp.bind_calibration_evidence(REPO)["design"]
PINNED = cp.verify_confirmation_successor(REPO)
CENSUS = cr.materialize_census(PINNED, DESIGN)
IMPL = cp.implementation_digest(REPO)
say("EXACT_CENSUS_UNCHANGED", [CENSUS["units_total"],
                               CENSUS["census_sha256"]])
say("IMPLEMENTATION_DIGEST", {"implementation_sha256": IMPL,
                              "files": len(cp.IMPLEMENTATION_FILES)})

TMP = Path(tempfile.mkdtemp(prefix="m4_dr04_post_"))
os.chmod(TMP, 0o700)
try:
    # ============ the POSITIVE case: the complete entrypoint ============
    RUN = TMP / "dev_fixture_run"
    probe = cr.development_verification_probe(REPO, RUN, generators=2,
                                             seeds=3)
    body = probe["body"]
    ver = probe["verification"]
    FX = cr.development_fixture_successor(generators=2, seeds=3,
                                          successor=PINNED)
    say("POSITIVE_COMPLETE_ENTRYPOINT_DEVELOPMENT_FIXTURE", {
        "units_total": body["units_total"],
        "units_complete": body["units_complete"],
        "generators_disjointness_verified":
            body["generators_disjointness_verified"],
        "session_status": body["session_status"],
        "raw_per_batch_logs": len(list(RUN.glob("intervention/*.jsonl"))),
        "verdict": ver["verdict"],
        "verdict_authority": ver["verdict_authority"],
        "reconstruction": ver["reconstruction"],
        "records_numerically_rederived":
            ver["records_numerically_rederived"],
        "population_complete": ver["population_complete"],
        "checkpoint_population": ver["checkpoint_population"]})

    def world(name):
        dst = TMP / name
        shutil.copytree(RUN, dst)
        return dst

    def verify(root):
        return cr.verify_confirmation_run(REPO, root, FX,
                                          expect_role="DEVELOPMENT")

    def rehash(path, mutate):
        rec = json.loads(Path(path).read_text())
        mutate(rec)
        rec.pop("record_sha256", None)
        rec["record_sha256"] = m4._self_sha(rec, "record_sha256")
        Path(path).write_text(json.dumps(rec, indent=1, sort_keys=True))

    UNITS = cr.census_units(FX, "DEVELOPMENT")
    U0 = UNITS[0]

    # ---------------- R3 / PRE P1 ----------------
    w = world("p1_fabricated")
    for arm in pv.CHECKPOINTS:
        cr.arm_log_path(w, U0["unit_id"], arm).unlink()
    fab = {"unit_id": U0["unit_id"],
           "arms": {"initialization": {"restricted_endpoint": 0,
                                       "updates_done": 0},
                    "calibration_stop": {"restricted_endpoint": 8,
                                         "updates_done": 400}},
           "paired_primary_difference": 8}
    fab["record_sha256"] = m4._self_sha(fab, "record_sha256")
    cr.unit_record_path(w, U0["unit_id"]).write_text(json.dumps(fab))
    say("R3_FABRICATED_SUMMARY_REFUSED", refused(verify, w)[:120])

    w = world("p1_endpoint_pair")
    rehash(cr.unit_record_path(w, U0["unit_id"]), lambda r: [
        r["arms"]["initialization"].__setitem__(
            "restricted_endpoint",
            r["arms"]["initialization"]["restricted_endpoint"] + 8),
        r["arms"]["calibration_stop"].__setitem__(
            "restricted_endpoint",
            r["arms"]["calibration_stop"]["restricted_endpoint"] + 8)])
    say("R3_CONSISTENTLY_ALTERED_ENDPOINT_PAIR_REFUSED",
        refused(verify, w)[:150])

    w = world("p1_raw_log")
    cr.arm_log_path(w, U0["unit_id"], "pre_stop").unlink()
    say("R3_MISSING_RAW_LOG_REFUSED", refused(verify, w)[:120])

    w = world("p1_raw_line")
    lp = cr.arm_log_path(w, U0["unit_id"], "initialization")
    lines = lp.read_text().splitlines()
    doc = json.loads(lines[0])
    doc["cumulative_associations"] = 999
    lines[0] = json.dumps(doc, sort_keys=True)
    lp.write_text("\n".join(lines) + "\n")
    say("R3_ALTERED_RAW_LINE_REFUSED", refused(verify, w)[:120])

    empty = TMP / "p1_gates"
    empty.mkdir()
    say("R3_EMPTY_GATE_MAP_REFUSED_WHEN_WRITTEN", {
        "refusal": refused(cr.write_pre_result_ledger, empty, CENSUS,
                           {})[:120],
        "ledger_created": (
            empty / "CONFIRMATION_PRE_RESULT_LEDGER.json").exists()})

    # The whole fabricated CONFIRMATION world of the PRE cannot be built:
    # a CONFIRMATION ledger names its authority digests, which candidate
    # code cannot produce and this POST does not simulate.
    say("R1_R3_FABRICATED_CONFIRMATION_WORLD_CANNOT_BE_BUILT", {
        "no_gates": refused(cr.write_pre_result_ledger, empty, CENSUS,
                            {})[:70],
        "fixture_token_in_a_confirmation_ledger": refused(
            cr.write_pre_result_ledger, empty, CENSUS, {
                "census_role": "CONFIRMATION",
                "successor_sha256": PINNED["successor_sha256"],
                "review_record_sha256": cr.DEV_GATE_AUTHORITY,
                "execution_record_sha256": cr.DEV_GATE_AUTHORITY,
                "executing_head": "1" * 40,
                "executing_implementation_sha256": IMPL,
                "prior_role_digests": 0})[:70],
        "anything_created": sorted(p.name for p in empty.iterdir())})

    # ---------------- R2 / PRE P2 ----------------
    w = world("p2_out_of_census")
    alien = rn._iv_unit("DEVELOPMENT", U0["family"], U0["noise_coord"],
                        U0["width"], 9000, 0)
    rec = json.loads(cr.unit_record_path(w, U0["unit_id"]).read_text())
    rec.update({k: alien[k] for k in cr.IDENTITY_FIELDS})
    rec["unit_id"] = alien["unit_id"]
    rec.pop("record_sha256")
    rec["record_sha256"] = m4._self_sha(rec, "record_sha256")
    cr.unit_record_path(w, alien["unit_id"]).write_text(json.dumps(rec))
    say("R2_OUT_OF_CENSUS_IDENTITY_REFUSED", refused(verify, w)[:130])

    w = world("p2_repeated_seed")
    canon = cr.unit_record_path(w, U0["unit_id"])
    for i in range(3):
        (w / "intervention" / f"copy{i}_summary.json").write_text(
            canon.read_text())
    msg = refused(verify, w)
    say("R2_REPEATED_SEED_REFUSED_BY_NAME", {
        "names_the_seed": f"s{U0['model_seed']}" in msg,
        "names_the_generator": U0["generator_id"] in msg,
        "refusal": msg[-150:]})

    w = world("p2_identity")
    rehash(cr.unit_record_path(w, U0["unit_id"]),
           lambda r: r.__setitem__("model_seed", 2))
    say("R2_IDENTITY_FIELD_MUST_REDERIVE", refused(verify, w)[:120])

    say("R2_POSITIVE_EXACT_SEED_SET_NOT_A_ROW_COUNT", {
        "three_copies_of_one_seed": sorted(
            cr.analyse_complete_population(
                FX, {"sine::clean": {16: {"g0": {0: 1.0}},
                                     64: {"g0": {0: 1.0}}}})[
                "generators_incomplete_seed_sets"]),
        "a_list_of_rows_refuses": refused(
            cr.analyse_complete_population, FX,
            {"sine::clean": {16: {"g0": [1.0, 1.0, 1.0]}}})[:90]})

    # ---------------- R4 / PRE P3 ----------------
    w = world("p3_incomplete")
    cr.unit_record_path(w, UNITS[-1]["unit_id"]).unlink()
    say("R4_INCOMPLETE_POPULATION_REFUSES_NEVER_NARROWS",
        refused(verify, w)[:170])

    below = {}
    for s in FX["eligible_slots"]:
        pre, wd = s["cell"].rsplit("::w", 1)
        below.setdefault(pre, {})[int(wd)] = {
            "g0": {0: 5.0, 1: 5.0, 2: 5.0}}
    st = cr.analyse_complete_population(FX, below)
    ck = st["analysis"]["contrasts"]["checkpoint_effect::primary_pair"]
    say("R4_FIFTEENTH_CONTRAST_AFTER_ATTRITION", {
        "slots_below_floor": len(st["confirmation_incomplete"]),
        "floor": FX["attrition"]["min_complete_required"],
        "checkpoint_status": ck["status"],
        "checkpoint_reject": ck["reject_at_alpha"],
        "checkpoint_population": st["checkpoint_population"],
        "population_complete": st["population_complete"]})

    pooled = {}
    for s in FX["eligible_slots"]:
        pre, wd = s["cell"].rsplit("::w", 1)
        pooled.setdefault(pre, {})[int(wd)] = {
            f"g{i}": {0: 1.0 + i, 1: 1.0 + i, 2: 1.0 + i}
            for i in range(2)}
    st = cr.analyse_complete_population(FX, pooled)
    say("R4_GENERATOR_INDICES_NEVER_POOL_ACROSS_CELLS", {
        "distinct_identities": sum(st["checkpoint_population"].values()),
        "n_generators_in_the_contrast": st["analysis"]["contrasts"][
            "checkpoint_effect::primary_pair"]["n_generators"],
        "rule": st["checkpoint_population_rule"][:80]})

    # ---------------- R5 / PRE P4 ----------------
    w = world("p4_thin")
    thin = {"unit_id": U0["unit_id"]}
    thin["record_sha256"] = m4._self_sha(thin, "record_sha256")
    cr.unit_record_path(w, U0["unit_id"]).write_text(json.dumps(thin))
    say("R5_UNIT_ID_AND_HASH_ONLY_IS_NOT_COMPLETE", refused(
        cr.execute_confirmation_units, DESIGN, UNITS, w,
        cr.new_accounting(), expect_role="DEVELOPMENT",
        allow_confirmation=False,
        prior_digests={"dr04_post_synthetic_prior_digest"})[:130])

    w = world("p4_disjoint")
    g = cr.gb.generate(U0["role"], U0["family"], U0["noise_coord"],
                       U0["generator_index"])
    say("R5_RESUME_REVALIDATES_DISJOINTNESS", refused(
        cr.execute_confirmation_units, DESIGN, UNITS, w,
        cr.new_accounting(), expect_role="DEVELOPMENT",
        allow_confirmation=False,
        prior_digests=cr.generator_array_digests(g))[:130])

    w = world("p4_manifest")
    rehash(cr.unit_record_path(w, U0["unit_id"]),
           lambda r: r.__setitem__("manifest_sha256", "0" * 64))
    say("R5_RESUMED_RECORD_MUST_BIND_ITS_GENERATOR", refused(
        cr.execute_confirmation_units, DESIGN, UNITS, w,
        cr.new_accounting(), expect_role="DEVELOPMENT",
        allow_confirmation=False,
        prior_digests={"dr04_post_synthetic_prior_digest"})[:130])

    w = world("p4_positive")
    prior = cr.role_array_digests(DESIGN, cr.cells_of_units(UNITS),
                                  roles=("CALIBRATION",))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(w.glob("intervention/*_summary.json"))}
    again = cr.execute_confirmation_units(
        DESIGN, UNITS, w, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests=prior)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(w.glob("intervention/*_summary.json"))}
    say("R5_POSITIVE_RESUME_RERUNS_NOTHING", {
        "units_complete": again["units_complete"],
        "units_new_this_session": again["units_new_this_session"],
        "generators_disjointness_verified":
            again["generators_disjointness_verified"],
        "records_byte_identical": after == before})

    # ---------------- R6 ----------------
    w = world("r6_live_lock")
    live = {"schema": "m4_run_session_lock.v1", "pid": os.getpid(),
            "boot_id": cr._boot_id(), "epoch": 1.0, "token": "live"}
    live["claim_sha256"] = cp._selfsha(live, "claim_sha256")
    (w / cr.RUN_LOCK_NAME).write_text(json.dumps(live))
    say("R6_LIVE_RUN_LOCK_REFUSES", refused(
        cr.execute_confirmation_units, DESIGN, UNITS, w,
        cr.new_accounting(), expect_role="DEVELOPMENT",
        allow_confirmation=False,
        prior_digests={"dr04_post_synthetic_prior_digest"})[:120])

    w = world("r6_stale_lock")
    stale = dict(live, boot_id="A_PREVIOUS_BOOT", token="stale")
    stale.pop("claim_sha256")
    stale["claim_sha256"] = cp._selfsha(stale, "claim_sha256")
    (w / cr.RUN_LOCK_NAME).write_text(json.dumps(stale))
    b = cr.execute_confirmation_units(
        DESIGN, UNITS, w, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests=prior)
    kept = sorted((w / cr.LOCKS_DIR).glob("lock_*.json"))
    say("R6_STALE_LOCK_PRESERVED_AND_RUN_CONTINUES", {
        "stale_locks_set_aside": b["stale_locks_set_aside"],
        "preserved": [q.name for q in kept],
        "token_preserved": json.loads(kept[0].read_text())["token"],
        "lock_released": not (w / cr.RUN_LOCK_NAME).exists()})

    w = world("r6_live_claim")
    cr.unit_record_path(w, U0["unit_id"]).unlink()
    claim = dict(live, token="claim")
    claim.pop("claim_sha256")
    claim["schema"] = "m4_unit_claim.v1"
    claim["claim_sha256"] = cp._selfsha(claim, "claim_sha256")
    cr.unit_claim_path(w, U0["unit_id"]).write_text(json.dumps(claim))
    say("R6_LIVE_UNIT_CLAIM_REFUSES_DOUBLE_EXECUTION", refused(
        cr.execute_confirmation_units, DESIGN, UNITS, w,
        cr.new_accounting(), expect_role="DEVELOPMENT",
        allow_confirmation=False, prior_digests=prior)[:130])

    w = world("r6_wall")
    cr.unit_record_path(w, U0["unit_id"]).unlink()
    spent = cr.cumulative_prior(w)["wall_seconds"]
    tight = dict(DESIGN, resources=dict(DESIGN["resources"],
                                        max_wall_seconds=spent / 2.0))
    b = cr.execute_confirmation_units(
        tight, UNITS, w, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests=prior)
    say("R6_SEALED_WALL_COUNTS_THE_WHOLE_RUN", {
        "prior_sessions_wall_seconds": b["prior_sessions_wall_seconds"],
        "tightened_wall": tight["resources"]["max_wall_seconds"],
        "session_status": b["session_status"],
        "units_new_this_session": b["units_new_this_session"]})

    cum = cr.cumulative_prior(RUN)
    say("R6_CUMULATIVE_ACCOUNTING", {
        "sessions": cum["sessions"], "units_new": cum["units_new"],
        "optimization_updates": cum["optimization_updates"],
        "wall_seconds": cum["wall_seconds"]})
    w = world("r6_bad_report")
    rp = sorted(w.glob("SESSION_*_REPORT.json"))[0]
    bad = json.loads(rp.read_text())
    bad["units_new_this_session"] = 999
    rp.write_text(json.dumps(bad))
    say("R6_UNVERIFIABLE_PRIOR_REPORT_IS_UNCERTAIN",
        refused(cr.cumulative_prior, w)[:120])

    # ---------------- R1 ----------------
    w = world("r1_other_code")
    lp = w / "CONFIRMATION_PRE_RESULT_LEDGER.json"
    led = json.loads(lp.read_text())
    recorded = led["gates"]["executing_implementation_sha256"]
    led["gates"]["executing_implementation_sha256"] = "a" * 64
    led.pop("ledger_sha256")
    led["ledger_sha256"] = cp._selfsha(led, "ledger_sha256")
    lp.write_text(json.dumps(led))
    say("R1_LEDGER_FROM_OTHER_CODE_NEVER_YIELDS_A_VERDICT", {
        "ledger_recorded": recorded == IMPL,
        "refusal": refused(verify, w)[:130]})

    alt = TMP / "alt_impl" / "tools"
    alt.mkdir(parents=True)
    for rel in cp.IMPLEMENTATION_FILES:
        shutil.copy(REPO / rel, alt / Path(rel).name)
    same = cp.implementation_digest(TMP / "alt_impl")
    target = alt / "m4_confirmation_runner.py"
    target.write_text(target.read_text() + "\n# one changed byte\n")
    say("R1_ONE_CHANGED_BYTE_CHANGES_THE_AUTHORITY_DIGEST", {
        "identical_copy_same_digest": same == IMPL,
        "one_byte_changed_digest_differs":
            cp.implementation_digest(TMP / "alt_impl") != IMPL,
        "review_schema_pins_it":
            "reviewed_implementation_sha256" in cp._REVIEW_KEYS,
        "no_record_created_or_simulated_in_this_post":
            not cp.MUSASHI_REVIEW_RECORD_PATH.exists()
            and not cp.OWNER_EXECUTION_RECORD_PATH.exists()})

    # ---------------- roles never cross ----------------
    say("ROLES_NEVER_CROSS", {
        "fixture_root_as_confirmatory": refused(
            cr.verify_confirmation_run, REPO, RUN, FX,
            expect_role="CONFIRMATION")[:90],
        "pinned_successor_as_a_fixture": refused(
            cr.verify_confirmation_run, REPO, RUN, PINNED,
            expect_role="DEVELOPMENT")[:90]})

    # ---------------- controls, unchanged ----------------
    ctl = TMP / "control_gate"
    say("CONTROL_ABSENT_RECORDS_REFUSE_BEFORE_ANY_WRITE", {
        "refusal": refused(cr.execute_confirmation, REPO, ctl)[:70],
        "out_root_created": ctl.exists(),
        "review_record_present":
            cp.MUSASHI_REVIEW_RECORD_PATH.is_file(),
        "execution_record_present":
            cp.OWNER_EXECUTION_RECORD_PATH.is_file()})
    say("CONTROL_ARRAY_DIGEST_COMPARATOR_REFUSES_A_COLLISION",
        bool(refused(cr.verify_role_disjointness, {"c"}, {"c"})))
    say("CONTROL_KILL_17_STILL_CLOSED", refused(
        cr.gb.generate, "CONFIRMATION", "sine", "white", 0)[:70])

    # F6: the four facts, separately
    conf = cr.gb.generate("CONFIRMATION", "sine", "white", 0,
                          allow_confirmation=True)
    state = Path.home() / ".local/share/agent-multi"
    say("F6_FOUR_FACTS", {
        "1_in_memory_construction_for_the_disjointness_proof":
            "HAPPENS (the bank's named exception): "
            f"{len(cr.generator_array_digests(conf))} array digests",
        "2_materialization": "DOES NOT HAPPEN: "
            f"{sorted(p.name for p in state.glob('*m4*confirmation*'))} "
            "in the state root",
        "3_fitting": "DOES NOT HAPPEN: " + refused(
            rn._run_intervention_unit_v5, DESIGN,
            rn._iv_unit("CONFIRMATION", "sine", "white", 16, 0, 0),
            RUN, cr.new_accounting())[:60],
        "4_scoring": "DOES NOT HAPPEN: no CONFIRMATION record, session "
                     "report or verification document has ever existed"})

    leaked = sorted(str(q) for q in TMP.rglob("*")
                    if "CONFIRMATION" in q.name
                    and q.name != "CONFIRMATION_PRE_RESULT_LEDGER.json")
    assert leaked == [], leaked
    say("POST_CONFIRMED",
        "every frozen counterexample is refused BY NAME on the same "
        "public path, each repair has a positive case through the "
        "complete entrypoint, and CONFIRMATION remains closed: the "
        "two-record gate refuses first and creates nothing")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
