"""DR04 PRE: Musashi's M4 verifiability counterexamples, frozen on the
REAL PUBLIC PATH of the candidate under review, BEFORE any repair.

Scope and standing
------------------
This is a PRE battery, not an approval and not a measurement. It is
committed BEFORE the repair so the repair is demonstrably answering probes
that predate it. Its reference is the auditor's dictamen
``docs/audits/work_plan/MUSASHI_DAY_REVIEW_2026_09_26.md`` F2 and its
evidence ``docs/audits/evidence/DAY_REVIEW_2026_09_26/{M4_PROBE.txt,
M4_FINDINGS.md}`` (findings F1-F6), reproduced here against
agent-multi@0de54534.

Difference from the auditor's own reproducer: he extracted function ASTs
from Git blobs and drove them with in-memory fixtures. This battery calls
the SAME defects through the module's PUBLIC surface only —
``cp.bind_calibration_evidence``, ``cp.verify_confirmation_successor``,
``cr.materialize_census``, ``cr.write_pre_result_ledger``,
``cr.verify_confirmation_run``, ``cr.execute_confirmation_units``,
``cr.execute_confirmation`` — with no monkeypatching of the verifier, no
private-helper surgery and no mocked authority reader.

What is NOT done here
---------------------
- No CONFIRMATION generator is constructed, no model is fitted, no score
  is produced. The "confirmation" documents below are FABRICATED SUMMARY
  JSON with no arrays, no logs and no lineage: that they are counted
  verified is the finding.
- No external authority record is created, simulated, mocked or installed.
  P5 replaces the auditor's mocked record reader with a structural probe of
  the public schema plus two ledgers written through the public ledger
  writer; no approval document of any kind is touched.
- Nothing is written outside a private temporary root, which is removed.

Run (CPU only, light):
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  $HOME/.local/bin/crispdm-run -m 2G -t 900 -n m4verify -- \
    python3 -B docs/audits/evidence/repro_runs/\
m4_dr04_verifiability_pre_2026_09_26.py
"""
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
import m4_v5_runner as rn  # noqa: E402

OUT = []


def say(tag, payload):
    line = f"{tag} {json.dumps(payload, sort_keys=True)}" \
        if not isinstance(payload, str) else f"{tag} {payload}"
    OUT.append(line)
    print(line, flush=True)


DESIGN = cp.bind_calibration_evidence(REPO)["design"]
SUCC = cp.verify_confirmation_successor(REPO)
CENSUS = cr.materialize_census(SUCC, DESIGN)
say("EXACT_CENSUS", [CENSUS["units_total"], CENSUS["census_sha256"]])

TMP = Path(tempfile.mkdtemp(prefix="m4_dr04_pre_"))
os.chmod(TMP, 0o700)


def fresh_root(name, gates=None):
    """A run root with the PUBLIC pre-result ledger and nothing else.

    ``gates`` defaults to ``{}`` — the empty gate map the current
    ``write_pre_result_ledger`` accepts and the current verifier never
    validates. No authority record is read, created or simulated.
    """
    root = TMP / name
    root.mkdir(mode=0o700)
    (root / "intervention").mkdir(mode=0o700)
    cr.write_pre_result_ledger(root, CENSUS,
                              {} if gates is None else gates)
    return root


def fabricate_summary(root, fam, nz, width, gi, seed, effect,
                      alias=None):
    """A FABRICATED per-unit summary: two arm endpoints and a declared
    difference, with no raw per-batch log, no unit self-hash, no lineage,
    no manifest and no generator. Nothing scientific is computed."""
    uid = (f"intervention::CONFIRMATION::{fam}::{nz}"
           f"::w{width}::g{gi}::s{seed}")
    rec = {
        "FABRICATED_ADVERSARIAL_FIXTURE": "DR04 PRE probe; not a "
                                          "measurement and not evidence",
        "unit_id": uid,
        "arms": {
            "initialization": {"restricted_endpoint": 0,
                               "updates_done": 0},
            "calibration_stop": {"restricted_endpoint": effect,
                                 "updates_done": 400}},
        "paired_primary_difference": effect}
    name = alias or f"{rn._safe(uid)}_summary.json"
    p = root / "intervention" / name
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    os.chmod(p, 0o600)
    return p


try:
    # ---------------- P1 (auditor F1) ----------------
    # 234 fabricated summaries, an empty gate map, zero raw logs and zero
    # record hashes: counted verified, and a Holm rejection is declared.
    root = fresh_root("p1")
    for width in (16, 64):
        for gi in range(39):
            for seed in range(3):
                fabricate_summary(root, "sine", "clean", width, gi,
                                  seed, 8 + gi % 5)
    r = cr.verify_confirmation_run(REPO, root, SUCC)
    v = r["analysis"]["contrasts"]["intervention_effect::sine::clean"]
    ledger = m4._strict_json_file(
        root / "CONFIRMATION_PRE_RESULT_LEDGER.json", "ledger")
    assert r["records_verified"] == 234 and v["reject_at_alpha"]
    say("P1_FABRICATED_SUMMARIES_REACH_REJECTION", {
        "records_verified": r["records_verified"],
        "n_generators": v["n_generators"],
        "p_holm": v["p_holm"],
        "reject_at_alpha": v["reject_at_alpha"],
        "raw_per_batch_logs": len(list(root.glob(
            "intervention/*.jsonl"))),
        "records_carrying_record_sha256": 0,
        "ledger_gates": ledger["gates"]})

    # ---------------- P2 (auditor F2) ----------------
    # Three differently NAMED copies of s0, for 39 generator indices that
    # are not in the census at all (g9000..g9038): 39 complete generators.
    root = fresh_root("p2")
    for width in (16, 64):
        for gi in range(9000, 9039):
            for copy_id in range(3):
                fabricate_summary(
                    root, "sine", "clean", width, gi, 0, 8 + gi % 5,
                    alias=f"w{width}_g{gi}_copy{copy_id}_summary.json")
    r = cr.verify_confirmation_run(REPO, root, SUCC)
    v = r["analysis"]["contrasts"]["intervention_effect::sine::clean"]
    in_census = json.dumps(
        [u["unit_id"] for u in cr.confirmation_units(SUCC)])
    assert v["n_generators"] == 39 and v["reject_at_alpha"]
    say("P2_DUPLICATE_SEED_AND_OUT_OF_CENSUS_GENERATORS", {
        "n_generators": v["n_generators"],
        "reject_at_alpha": v["reject_at_alpha"],
        "distinct_model_seeds_present": 1,
        "seeds_required_by_successor":
            SUCC["nested_seeds_per_generator"],
        "generator_index_min": 9000,
        "any_g9000_id_in_census": "::g9000::" in in_census,
        "filenames_are_arbitrary": True})

    # ---------------- P3a (auditor F3a) ----------------
    # 30 generators, one width, three seeds: every one of the 21 eligible
    # slots is below the frozen floor of 39, and the FIFTEENTH contrast is
    # nevertheless evaluated and rejects.
    root = fresh_root("p3a")
    for gi in range(30):
        for seed in range(3):
            fabricate_summary(root, "sine", "clean", 16, gi, seed,
                              8 + gi % 5)
    r = cr.verify_confirmation_run(REPO, root, SUCC)
    v = r["analysis"]["contrasts"]["checkpoint_effect::primary_pair"]
    assert len(r["confirmation_incomplete"]) == 21 and v["reject_at_alpha"]
    say("P3A_FIFTEENTH_CONTRAST_IGNORES_ATTRITION_AND_POPULATION", {
        "incomplete_slots": len(r["confirmation_incomplete"]),
        "min_complete_required":
            SUCC["attrition"]["min_complete_required"],
        "census_units_total": CENSUS["units_total"],
        "records_present": r["records_verified"],
        "n_generators": v["n_generators"],
        "p_holm": v["p_holm"],
        "reject_at_alpha": v["reject_at_alpha"]})

    # ---------------- P3b (auditor F3b) ----------------
    # 30 sine and 30 chirp generator identities, each with only s0 and s1:
    # sixty incomplete identities collapse by bare g index into thirty
    # "complete" checkpoint observations, and reject.
    root = fresh_root("p3b")
    for gi in range(30):
        for fam in ("sine", "chirp"):
            for seed in (0, 1):
                fabricate_summary(root, fam, "clean", 16, gi, seed,
                                  8 + gi % 5)
    r = cr.verify_confirmation_run(REPO, root, SUCC)
    v = r["analysis"]["contrasts"]["checkpoint_effect::primary_pair"]
    assert v["n_generators"] == 30 and v["reject_at_alpha"]
    say("P3B_DISTINCT_FAMILY_GENERATORS_COLLAPSE_BY_G_INDEX", {
        "distinct_family_generator_identities": 60,
        "every_identity_missing_seed_2": True,
        "reported_n_generators": v["n_generators"],
        "reject_at_alpha": v["reject_at_alpha"]})

    # ---------------- P4 (auditor F4) ----------------
    # Resumption: a DEVELOPMENT document consisting only of unit_id and a
    # correct self-hash is counted COMPLETE and the census is declared
    # complete, with zero generators and zero disjointness proofs. The
    # public body is called; no engine, no generator, no fit.
    root = TMP / "p4"
    root.mkdir(mode=0o700)
    unit = rn.intervention_units_v5(DESIGN, "DEVELOPMENT")[0]
    thin = {"unit_id": unit["unit_id"]}
    thin["record_sha256"] = m4._self_sha(thin, "record_sha256")
    (root / "intervention").mkdir(mode=0o700)
    tp = cr.unit_record_path(root, unit["unit_id"])
    tp.write_text(json.dumps(thin, indent=1, sort_keys=True))
    os.chmod(tp, 0o600)
    body = cr.execute_confirmation_units(
        DESIGN, [unit], root, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests={"dr04_pre_synthetic_prior_digest"})
    assert body["census_complete"] and body["units_complete"] == 1
    say("P4_RESUME_ACCEPTS_UNIT_ID_AND_HASH_ONLY", {
        "record_keys": sorted(thin),
        "units_complete": body["units_complete"],
        "units_new_this_session": body["units_new_this_session"],
        "census_complete": body["census_complete"],
        "generators_disjointness_verified":
            body["generators_disjointness_verified"],
        "arms_in_record": 0, "lineage_in_record": 0,
        "typed_terminal_status_in_record": 0})

    # ---------------- P5 (auditor F5) ----------------
    # Authority is not bound to the implementation that will run. No
    # record reader is mocked here: the public review schema is inspected,
    # and two ledgers are written through the PUBLIC ledger writer with
    # different fabricated code heads. Both verify; the gate map is never
    # validated, so it can even be empty (P1).
    impl_keys = sorted(k for k in cp._REVIEW_KEYS
                       if "implementation" in k or "code" in k)
    gate_names = []
    for head in ("1" * 40, "2" * 40):
        root = fresh_root(f"p5_{head[0]}", gates={
            "successor_sha256": SUCC["successor_sha256"],
            "executing_head": head})
        fabricate_summary(root, "sine", "clean", 16, 0, 0, 8)
        r = cr.verify_confirmation_run(REPO, root, SUCC)
        led = m4._strict_json_file(
            root / "CONFIRMATION_PRE_RESULT_LEDGER.json", "ledger")
        gate_names.append(led["gates"]["executing_head"])
        assert r["records_verified"] == 1
    assert len(set(gate_names)) == 2
    say("P5_AUTHORITY_DOES_NOT_PIN_THE_EXECUTABLE_IMPLEMENTATION", {
        "review_schema_keys_naming_the_implementation": impl_keys,
        "ledger_gate_keys_naming_the_implementation": [],
        "different_heads_both_verified": gate_names,
        "verifier_validates_gates": False})

    # ---------------- P6 (auditor F6) ----------------
    # The published "zero CONFIRMATION arrays" sentence. Static reading of
    # the committed bytes; nothing is executed and no array is built here.
    hits = {}
    for rel in ("tests/test_m4_confirmation_execution_body.py",
                "docs/audits/evidence/repro_runs/"
                "m4_confirmation_exec_body_post_2026_09_26.py"):
        txt = (REPO / rel).read_text()
        hits[rel] = txt.count('allow_confirmation=True')
    say("P6_ZERO_ARRAY_SENTENCE_IS_NOT_WHAT_THE_CODE_DOES", {
        "call_sites_building_CONFIRMATION_arrays": hits,
        "bank_docstring_allows_the_disjointness_proof_exception":
            "only the explicit" in (
                REPO / "tools/m4_generator_bank.py").read_text(),
        "four_facts_to_distinguish": ["disjointness_proof",
                                      "materialization", "fitting",
                                      "scoring"]})

    # ---------------- controls ----------------
    # The absent-record gate still refuses BEFORE writing anything: the
    # real record paths are used and are absent on this host.
    ctl = TMP / "control_gate"
    try:
        cr.execute_confirmation(REPO, ctl)
        raise AssertionError("absent records must refuse")
    except SystemExit as e:
        msg = str(e)
    say("CONTROL_ABSENT_RECORDS_REFUSE_BEFORE_ANY_WRITE", {
        "refusal": msg[:70],
        "out_root_created": ctl.exists(),
        "review_record_present":
            cp.MUSASHI_REVIEW_RECORD_PATH.is_file(),
        "execution_record_present":
            cp.OWNER_EXECUTION_RECORD_PATH.is_file()})

    try:
        cr.verify_role_disjointness({"c"}, {"c"})
        raise AssertionError("collision must refuse")
    except SystemExit:
        pass
    say("CONTROL_ARRAY_DIGEST_COMPARATOR_REFUSES_A_COLLISION", True)

    say("PRE_FROZEN", "six counterexamples reproduced through the public "
                      "path; nothing repaired in this commit")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
