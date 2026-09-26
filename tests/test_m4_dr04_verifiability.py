"""DR04: the auditor's M4 counterexamples, as permanent tests.

Every test here descends from one probe in Musashi's dictamen of 2026-09-26
(``docs/audits/work_plan/MUSASHI_DAY_REVIEW_2026_09_26.md`` F2 and its
evidence ``docs/audits/evidence/DAY_REVIEW_2026_09_26/M4_FINDINGS.md``
F1-F6), frozen first as a PRE battery
(``docs/audits/evidence/repro_runs/m4_dr04_verifiability_pre_2026_09_26.py``)
and converted here into tests of the REAL PUBLIC PATH.

Rules this battery keeps:

- **No new confirmatory data.** Every positive case runs over a
  DEVELOPMENT fixture population — real generators, real fits, real raw
  per-batch logs — through the same code the CONFIRMATION path runs. The
  fabricated CONFIRMATION documents of the PRE battery appear only to prove
  that they can no longer reach the record layer at all.
- **No authority record is created, installed or simulated here.** The one
  place that needs a record double is the sealed protocol battery, which
  already owns that machinery (``_mk_records``, author ``test-double``, a
  monkeypatched path); the implementation binding is proven here without any
  record, at the ledger and verification boundary.
- **No gate is weakened, reordered or bypassed**, and CONFIRMATION is never
  executed: the gate chain still refuses with the two records absent.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import m4_confirmation_protocol as cp  # noqa: E402
import m4_confirmation_runner as cr  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402
import m4_v5_protocol as pv  # noqa: E402
import m4_v5_runner as rn  # noqa: E402

CLI = [sys.executable, str(REPO / "tools"
                           / "m4_confirmation_runner.py")]
ENV = {"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
       "HOME": str(Path.home()), "PYTHONDONTWRITEBYTECODE": "1",
       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def _cli(*args):
    return subprocess.run([*CLI, *args], capture_output=True, text=True,
                          timeout=1800, env=ENV)


@pytest.fixture(scope="module")
def design():
    return cp.bind_calibration_evidence(REPO)["design"]


@pytest.fixture(scope="module")
def pinned():
    return cp.verify_confirmation_successor(REPO)


@pytest.fixture(scope="module")
def fixture_successor(pinned):
    """A DEVELOPMENT fixture population: 2 cells x 2 widths x 2
    generators x 3 seeds = 24 real units, floor 2."""
    return cr.development_fixture_successor(generators=2, seeds=3,
                                           successor=pinned)


@pytest.fixture(scope="module")
def dev_run(tmp_path_factory):
    """ONE real DEVELOPMENT fixture run, built through the complete
    entrypoint and verified. Every mutation test works on a copy."""
    root = tmp_path_factory.mktemp("dr04") / "run"
    out = cr.development_verification_probe(REPO, root, generators=2,
                                           seeds=3)
    assert out["body"]["census_complete"] is True
    assert out["verification"]["verdict"] == "VERIFIED"
    return root


@pytest.fixture
def copy_run(dev_run, tmp_path):
    dst = tmp_path / "run"
    shutil.copytree(dev_run, dst)
    return dst


def _units(fixture_successor):
    return cr.census_units(fixture_successor, "DEVELOPMENT")


def _rewrite(path, mutate):
    rec = json.loads(Path(path).read_text())
    mutate(rec)
    rec.pop("record_sha256", None)
    rec["record_sha256"] = m4._self_sha(rec, "record_sha256")
    Path(path).write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec


def _verify(root, fixture_successor):
    return cr.verify_confirmation_run(REPO, root, fixture_successor,
                                      expect_role="DEVELOPMENT")


# ===================================================================
# PRE P1 / auditor F1 — raw, authenticated evidence and numerical
# re-derivation before anything reads VERIFIED
# ===================================================================

def test_p1_positive_real_run_verifies_by_full_replay(dev_run,
                                                      fixture_successor):
    out = _verify(dev_run, fixture_successor)
    assert out["verdict"] == "VERIFIED"
    assert out["verdict_authority"] == (
        "NON_CONFIRMATORY_DEVELOPMENT_FIXTURE_NO_SCIENTIFIC_VERDICT")
    assert out["reconstruction"] == "FULL_REPLAY_FROM_RAW_RECORDS"
    assert out["records_verified"] == 24
    assert out["records_numerically_rederived"] == 24
    assert out["population_complete"] is True
    # the raw per-batch logs the verdict rests on actually exist
    logs = sorted(dev_run.glob("intervention/*.jsonl"))
    assert len(logs) == 24 * len(pv.CHECKPOINTS)
    assert out["replay_accounting"]["optimization_updates"] > 0


def test_p1_negative_fabricated_summary_never_reads_verified(
        copy_run, fixture_successor):
    """The PRE counterexample: a summary with two arm endpoints and a
    declared difference, no raw log, no lineage, no manifest."""
    unit = _units(fixture_successor)[0]
    p = cr.unit_record_path(copy_run, unit["unit_id"])
    for arm in pv.CHECKPOINTS:
        cr.arm_log_path(copy_run, unit["unit_id"], arm).unlink()
    rec = {"unit_id": unit["unit_id"],
           "arms": {"initialization": {"restricted_endpoint": 0,
                                       "updates_done": 0},
                    "calibration_stop": {"restricted_endpoint": 8,
                                         "updates_done": 400}},
           "paired_primary_difference": 8}
    rec["record_sha256"] = m4._self_sha(rec, "record_sha256")
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    with pytest.raises(SystemExit) as e:
        _verify(copy_run, fixture_successor)
    assert "identity field" in str(e.value) or "omits" in str(e.value)


def test_p1_negative_missing_raw_log_refuses(copy_run,
                                             fixture_successor):
    unit = _units(fixture_successor)[0]
    cr.arm_log_path(copy_run, unit["unit_id"], "pre_stop").unlink()
    with pytest.raises(SystemExit, match="ABSENT"):
        _verify(copy_run, fixture_successor)


def test_p1_negative_consistently_altered_endpoint_pair_refuses(
        copy_run, fixture_successor):
    """F1's sharpest case: shift BOTH endpoints by the same amount so the
    declared difference still agrees. Only re-derivation from the raw log
    can catch it."""
    unit = _units(fixture_successor)[0]
    p = cr.unit_record_path(copy_run, unit["unit_id"])
    before = json.loads(p.read_text())
    _rewrite(p, lambda r: [
        r["arms"]["initialization"].__setitem__(
            "restricted_endpoint",
            r["arms"]["initialization"]["restricted_endpoint"] + 8),
        r["arms"]["calibration_stop"].__setitem__(
            "restricted_endpoint",
            r["arms"]["calibration_stop"]["restricted_endpoint"] + 8)])
    after = json.loads(p.read_text())
    assert (after["paired_primary_difference"]
            == before["paired_primary_difference"])
    with pytest.raises(SystemExit, match="does not re-derive from the raw"):
        _verify(copy_run, fixture_successor)


def test_p1_negative_altered_raw_log_line_refuses(copy_run,
                                                  fixture_successor):
    unit = _units(fixture_successor)[0]
    lp = cr.arm_log_path(copy_run, unit["unit_id"], "initialization")
    lines = lp.read_text().splitlines()
    doc = json.loads(lines[0])
    doc["cumulative_associations"] = 999
    lines[0] = json.dumps(doc, sort_keys=True)
    lp.write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="does not re-derive"):
        _verify(copy_run, fixture_successor)


def test_p1_negative_empty_gate_map_is_refused(fixture_successor,
                                               design, tmp_path):
    """The PRE ledger carried ``gates={}`` and the verifier never looked.
    The gate map is now evidence and it is validated when WRITTEN."""
    census = cr.materialize_census(fixture_successor, design,
                                   "DEVELOPMENT")
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(SystemExit, match="exact schema"):
        cr.write_pre_result_ledger(root, census, {})
    assert not (root / "CONFIRMATION_PRE_RESULT_LEDGER.json").exists()


def test_p1_negative_fabricated_confirmation_world_cannot_be_built(
        pinned, design, tmp_path):
    """PRE P1/P2/P3 verbatim: the fabricated CONFIRMATION world no longer
    even gets a ledger, because a CONFIRMATION ledger names its authority
    and its executable implementation — which no candidate code can
    produce."""
    census = cr.materialize_census(pinned, design)
    assert census["units_total"] == 3024
    root = tmp_path / "fab"
    root.mkdir()
    with pytest.raises(SystemExit, match="exact schema"):
        cr.write_pre_result_ledger(root, census, {})
    with pytest.raises(SystemExit, match="recorded digest"):
        cr.write_pre_result_ledger(root, census, {
            "census_role": "CONFIRMATION",
            "successor_sha256": pinned["successor_sha256"],
            "review_record_sha256": "not-a-digest",
            "execution_record_sha256": "not-a-digest",
            "executing_head": "1" * 40,
            "executing_implementation_sha256":
                cp.implementation_digest(REPO),
            "prior_role_digests": 0})
    # and a CONFIRMATION ledger may never carry the fixture declaration
    with pytest.raises(SystemExit, match="recorded digest"):
        cr.write_pre_result_ledger(root, census, {
            "census_role": "CONFIRMATION",
            "successor_sha256": pinned["successor_sha256"],
            "review_record_sha256": cr.DEV_GATE_AUTHORITY,
            "execution_record_sha256": cr.DEV_GATE_AUTHORITY,
            "executing_head": "1" * 40,
            "executing_implementation_sha256":
                cp.implementation_digest(REPO),
            "prior_role_digests": 0})
    assert list(root.iterdir()) == []


# ===================================================================
# PRE P2 / auditor F2 — exact census, distinct seeds, complete identity
# ===================================================================

def test_p2_negative_out_of_census_identity_refuses(copy_run,
                                                    fixture_successor):
    unit = _units(fixture_successor)[0]
    alien = rn._iv_unit("DEVELOPMENT", unit["family"],
                        unit["noise_coord"], unit["width"], 9000, 0)
    src = cr.unit_record_path(copy_run, unit["unit_id"])
    rec = json.loads(src.read_text())
    rec.update({k: alien[k] for k in cr.IDENTITY_FIELDS})
    rec["unit_id"] = alien["unit_id"]
    rec.pop("record_sha256")
    rec["record_sha256"] = m4._self_sha(rec, "record_sha256")
    cr.unit_record_path(copy_run, alien["unit_id"]).write_text(
        json.dumps(rec, indent=1, sort_keys=True))
    with pytest.raises(SystemExit, match="NOT in the exact census"):
        _verify(copy_run, fixture_successor)


def test_p2_negative_repeated_seed_is_refused_by_name(
        copy_run, fixture_successor):
    """Three differently NAMED copies of one seed were counted as three
    nested repetitions. A record under any other name is refused, and the
    refusal names the repeated seed and its generator."""
    unit = _units(fixture_successor)[0]
    canon = cr.unit_record_path(copy_run, unit["unit_id"])
    for i in range(2):
        (copy_run / "intervention"
         / f"copy{i}_summary.json").write_text(canon.read_text())
    with pytest.raises(SystemExit) as e:
        _verify(copy_run, fixture_successor)
    msg = str(e.value)
    assert "canonical record name" in msg
    assert f"s{unit['model_seed']}" in msg
    assert unit["generator_id"] in msg


def test_p2_negative_identity_fields_must_re_derive(copy_run,
                                                     fixture_successor):
    unit = _units(fixture_successor)[0]
    p = cr.unit_record_path(copy_run, unit["unit_id"])
    _rewrite(p, lambda r: r.__setitem__("model_seed", 2))
    with pytest.raises(SystemExit, match="identity field 'model_seed'"):
        _verify(copy_run, fixture_successor)


def test_p2_negative_generator_identity_must_re_derive(
        copy_run, fixture_successor):
    unit = _units(fixture_successor)[0]
    p = cr.unit_record_path(copy_run, unit["unit_id"])
    _rewrite(p, lambda r: r.__setitem__("generator_id", "FORGED-g0"))
    with pytest.raises(SystemExit, match="identity field 'generator_id'"):
        _verify(copy_run, fixture_successor)


def test_p2_positive_exact_seed_set_makes_a_generator_complete(
        fixture_successor):
    """The sealed seed set, not a row count: three copies of one seed are
    one seed."""
    eff, thin = {}, {}
    for s in fixture_successor["eligible_slots"]:
        pre, w = s["cell"].rsplit("::w", 1)
        eff.setdefault(pre, {})[int(w)] = {
            f"g{i}": {0: 1.0, 1: 1.0, 2: 1.0} for i in range(2)}
        thin.setdefault(pre, {})[int(w)] = {
            f"g{i}": {0: 1.0} for i in range(2)}
    ok = cr.analyse_complete_population(fixture_successor, eff)
    assert "g0" in ok["complete"]["sine::clean"][16]
    assert ok["population_complete"] is True
    bad = cr.analyse_complete_population(fixture_successor, thin)
    assert bad["complete"] == {}
    assert "sine::clean::w16::g0" in \
        bad["generators_incomplete_seed_sets"]
    assert bad["population_complete"] is False
    with pytest.raises(SystemExit, match="keyed by model seed"):
        cr.analyse_complete_population(
            fixture_successor,
            {"sine::clean": {16: {"g0": [1.0, 1.0, 1.0]}}})


# ===================================================================
# PRE P3 / auditor F3 — attrition and a complete population for every
# contrast, the fifteenth included
# ===================================================================

def test_p3_negative_incomplete_population_refuses_never_narrows(
        copy_run, fixture_successor):
    unit = _units(fixture_successor)[-1]
    cr.unit_record_path(copy_run, unit["unit_id"]).unlink()
    with pytest.raises(SystemExit,
                       match="POPULATION_INCOMPLETE") as e:
        _verify(copy_run, fixture_successor)
    assert "never narrows the denominator" in str(e.value)


def test_p3_negative_fifteenth_contrast_excludes_below_floor_slots(
        fixture_successor):
    floor = fixture_successor["attrition"]["min_complete_required"]
    eff = {}
    for s in fixture_successor["eligible_slots"]:
        pre, w = s["cell"].rsplit("::w", 1)
        eff.setdefault(pre, {})[int(w)] = {
            "g0": {0: 5.0, 1: 5.0, 2: 5.0}}          # 1 < floor of 2
    out = cr.analyse_complete_population(fixture_successor, eff)
    assert len(out["confirmation_incomplete"]) == len(
        fixture_successor["eligible_slots"])
    assert all(v["complete_generators"] < floor
               for v in out["confirmation_incomplete"].values())
    ck = out["analysis"]["contrasts"]["checkpoint_effect::primary_pair"]
    assert ck["status"] == "NOT_EVALUABLE"
    assert ck["reject_at_alpha"] is False
    assert out["checkpoint_population"] == {}
    assert out["population_complete"] is False


def test_p3_negative_generator_indices_never_pool_across_cells(
        fixture_successor):
    """F3b: sixty distinct identities collapsed into thirty observations
    because the contrast keyed on the bare ``gN`` token."""
    eff = {}
    for s in fixture_successor["eligible_slots"]:
        pre, w = s["cell"].rsplit("::w", 1)
        eff.setdefault(pre, {})[int(w)] = {
            f"g{i}": {0: 1.0 + i, 1: 1.0 + i, 2: 1.0 + i}
            for i in range(2)}
    out = cr.analyse_complete_population(fixture_successor, eff)
    ck = out["analysis"]["contrasts"]["checkpoint_effect::primary_pair"]
    # 2 cells x 2 widths x 2 generators = 8 distinct identities, no pooling
    assert ck["n_generators"] == 8
    assert sum(out["checkpoint_population"].values()) == 8
    assert "family" in out["checkpoint_population_rule"] or \
        "families" in out["checkpoint_population_rule"]


def test_p3_positive_the_fifteenth_contrast_names_its_population(
        dev_run, fixture_successor):
    out = _verify(dev_run, fixture_successor)
    ck = out["analysis"]["contrasts"]["checkpoint_effect::primary_pair"]
    assert ck["n_generators"] == 8
    assert out["checkpoint_population"] == {
        "sine::clean::w16": 2, "sine::clean::w64": 2,
        "chirp::clean::w16": 2, "chirp::clean::w64": 2}
    assert "NEVER pooled" in out["checkpoint_population_rule"]
    assert ck["population_complete"] is True


# ===================================================================
# PRE P4 / auditor F4 — resumption with the complete schema
# ===================================================================

def test_p4_negative_unit_id_and_hash_only_is_not_complete(
        copy_run, fixture_successor, design):
    unit = _units(fixture_successor)[0]
    thin = {"unit_id": unit["unit_id"]}
    thin["record_sha256"] = m4._self_sha(thin, "record_sha256")
    cr.unit_record_path(copy_run, unit["unit_id"]).write_text(
        json.dumps(thin, indent=1, sort_keys=True))
    with pytest.raises(SystemExit, match="omits the identity field"):
        cr.execute_confirmation_units(
            design, _units(fixture_successor), copy_run,
            cr.new_accounting(), expect_role="DEVELOPMENT",
            allow_confirmation=False,
            prior_digests={"dr04_synthetic_prior_digest"})


def test_p4_negative_record_without_lineage_is_not_complete(
        copy_run, fixture_successor, design):
    unit = _units(fixture_successor)[0]
    p = cr.unit_record_path(copy_run, unit["unit_id"])
    _rewrite(p, lambda r: r.pop("checkpoint_lineage"))
    with pytest.raises(SystemExit, match="omits 'checkpoint_lineage'"):
        cr.read_complete_unit_record(p, unit, "DEVELOPMENT", copy_run)


def test_p4_negative_leftover_durable_state_is_uncertain(
        copy_run, fixture_successor):
    unit = _units(fixture_successor)[0]
    sp = rn._state_path(copy_run, unit["unit_id"], "initialization")
    sp.write_bytes(b"leftover")
    with pytest.raises(SystemExit, match="durable resume state"):
        cr.read_complete_unit_record(
            cr.unit_record_path(copy_run, unit["unit_id"]), unit,
            "DEVELOPMENT", copy_run)


def test_p4_negative_resume_revalidates_the_disjointness_proof(
        copy_run, fixture_successor, design):
    """A resumed unit's role disjointness is proven again, not inherited:
    poison the prior-role census with the fixture's OWN array digests and
    the resume refuses even though every unit is already complete."""
    units = _units(fixture_successor)
    g = cr.gb.generate(units[0]["role"], units[0]["family"],
                       units[0]["noise_coord"],
                       units[0]["generator_index"])
    with pytest.raises(SystemExit, match="collide with prior"):
        cr.execute_confirmation_units(
            design, units, copy_run, cr.new_accounting(),
            expect_role="DEVELOPMENT", allow_confirmation=False,
            prior_digests=cr.generator_array_digests(g))


def test_p4_negative_resumed_record_must_bind_its_generator(
        copy_run, fixture_successor, design):
    unit = _units(fixture_successor)[0]
    p = cr.unit_record_path(copy_run, unit["unit_id"])
    _rewrite(p, lambda r: r.__setitem__("manifest_sha256", "0" * 64))
    with pytest.raises(SystemExit, match="manifest"):
        cr.execute_confirmation_units(
            design, _units(fixture_successor), copy_run,
            cr.new_accounting(), expect_role="DEVELOPMENT",
            allow_confirmation=False,
            prior_digests={"dr04_synthetic_prior_digest"})


def test_p4_positive_resume_reruns_nothing_and_proves_disjointness(
        copy_run, fixture_successor, design):
    units = _units(fixture_successor)
    prior = cr.role_array_digests(design, cr.cells_of_units(units),
                                  roles=("CALIBRATION",))
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(copy_run.glob(
                   "intervention/*_summary.json"))}
    body = cr.execute_confirmation_units(
        design, units, copy_run, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests=prior)
    assert body["units_complete"] == 24
    assert body["units_new_this_session"] == 0
    assert body["census_complete"] is True
    assert body["generators_disjointness_verified"] == 4
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(copy_run.glob(
                 "intervention/*_summary.json"))}
    assert after == digests, "a completed record was rewritten on resume"


def test_p4_negative_foreign_role_record_refuses_both_ways():
    with pytest.raises(SystemExit, match="one census, one role"):
        cr.refuse_role_mismatch(
            {"unit_id": "intervention::CONFIRMATION::sine::clean::w16"
                        "::g0::s0", "role": "CONFIRMATION"},
            "DEVELOPMENT")
    with pytest.raises(SystemExit, match="one census, one role"):
        cr.refuse_role_mismatch(
            {"unit_id": "intervention::DEVELOPMENT::sine::clean::w16"
                        "::g0::s0", "role": "DEVELOPMENT"},
            "CONFIRMATION")


# ===================================================================
# PRE P5 / auditor F5 — authority bound to the reviewed implementation
# ===================================================================

def test_p5_positive_the_review_schema_pins_the_implementation():
    assert "reviewed_implementation_sha256" in cp._REVIEW_KEYS
    d = cp.implementation_digest(REPO)
    assert len(d) == 64 and d == cp.implementation_digest(REPO)
    files = cp.implementation_file_digests(REPO)
    assert set(files) == set(cp.IMPLEMENTATION_FILES)
    for rel in ("tools/m4_confirmation_runner.py",
                "tools/m4_confirmation_protocol.py",
                "tools/m4_v5_runner.py",
                "tools/m4_generator_bank.py"):
        assert rel in files


def test_p5_negative_one_changed_byte_changes_the_authority_digest(
        tmp_path):
    """A recorded revision names a commit; this digest binds the bytes."""
    fake = tmp_path / "repo" / "tools"
    fake.mkdir(parents=True)
    for rel in cp.IMPLEMENTATION_FILES:
        shutil.copy(REPO / rel, fake / Path(rel).name)
    assert cp.implementation_digest(tmp_path / "repo") == \
        cp.implementation_digest(REPO)
    target = fake / "m4_confirmation_runner.py"
    target.write_text(target.read_text() + "\n# one changed byte\n")
    assert cp.implementation_digest(tmp_path / "repo") != \
        cp.implementation_digest(REPO)


def test_p5_negative_absent_implementation_file_refuses(tmp_path):
    (tmp_path / "tools").mkdir()
    with pytest.raises(SystemExit, match="is ABSENT"):
        cp.implementation_digest(tmp_path)


def test_p5_negative_a_ledger_from_other_code_never_yields_a_verdict(
        copy_run, fixture_successor):
    """Two different heads were both recorded and both verified. A ledger
    is now bound to the implementation that wrote it."""
    lp = copy_run / "CONFIRMATION_PRE_RESULT_LEDGER.json"
    led = json.loads(lp.read_text())
    assert led["gates"]["executing_implementation_sha256"] == \
        cp.implementation_digest(REPO)
    led["gates"]["executing_implementation_sha256"] = "a" * 64
    led.pop("ledger_sha256")
    led["ledger_sha256"] = cp._selfsha(led, "ledger_sha256")
    lp.write_text(json.dumps(led, indent=1, sort_keys=True))
    with pytest.raises(SystemExit,
                       match="never read out of a run produced by other"):
        _verify(copy_run, fixture_successor)


def test_p5_negative_a_substituted_successor_never_confirms(
        copy_run, fixture_successor, pinned):
    with pytest.raises(SystemExit, match="order-pinned successor"):
        cr.verify_confirmation_run(REPO, copy_run, fixture_successor,
                                   expect_role="CONFIRMATION")
    with pytest.raises(SystemExit,
                       match="never yields a DEVELOPMENT verdict"):
        cr.verify_confirmation_run(REPO, copy_run, pinned,
                                   expect_role="DEVELOPMENT")


def test_p5_negative_a_fixture_ledger_never_verifies_as_confirmatory(
        dev_run):
    r = _cli("verify", "--run-root", str(dev_run))
    assert r.returncode != 0
    assert "CONFIRMATION" in (r.stdout + r.stderr)


def test_p5_the_gate_chain_still_refuses_and_creates_nothing(tmp_path):
    out = tmp_path / "never"
    r = _cli("execute", "--out", str(out))
    assert r.returncode != 0
    assert "ABSENT" in (r.stdout + r.stderr)
    assert not out.exists()
    assert not cp.MUSASHI_REVIEW_RECORD_PATH.exists()
    assert not cp.OWNER_EXECUTION_RECORD_PATH.exists()


# ===================================================================
# DR04 R6 — cumulative accounting, interruption and concurrency
# ===================================================================

def test_r6_negative_a_live_run_lock_refuses(copy_run,
                                             fixture_successor, design):
    doc = {"schema": "m4_run_session_lock.v1", "pid": os.getpid(),
           "boot_id": cr._boot_id(), "epoch": 1.0, "token": "live"}
    doc["claim_sha256"] = cp._selfsha(doc, "claim_sha256")
    (copy_run / cr.RUN_LOCK_NAME).write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="holds the run lock"):
        cr.execute_confirmation_units(
            design, _units(fixture_successor), copy_run,
            cr.new_accounting(), expect_role="DEVELOPMENT",
            allow_confirmation=False,
            prior_digests={"dr04_synthetic_prior_digest"})


def test_r6_positive_a_stale_lock_is_preserved_and_the_run_continues(
        copy_run, fixture_successor, design):
    doc = {"schema": "m4_run_session_lock.v1", "pid": os.getpid(),
           "boot_id": "A_PREVIOUS_BOOT", "epoch": 1.0, "token": "stale"}
    doc["claim_sha256"] = cp._selfsha(doc, "claim_sha256")
    (copy_run / cr.RUN_LOCK_NAME).write_text(json.dumps(doc))
    units = _units(fixture_successor)
    prior = cr.role_array_digests(design, cr.cells_of_units(units),
                                  roles=("CALIBRATION",))
    body = cr.execute_confirmation_units(
        design, units, copy_run, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests=prior)
    assert body["stale_locks_set_aside"] == 1
    kept = sorted((copy_run / cr.LOCKS_DIR).glob("lock_*.json"))
    assert len(kept) == 1
    assert json.loads(kept[0].read_text())["token"] == "stale"
    assert not (copy_run / cr.RUN_LOCK_NAME).exists()


def test_r6_negative_a_live_unit_claim_refuses(copy_run,
                                               fixture_successor, design):
    unit = _units(fixture_successor)[0]
    cr.unit_record_path(copy_run, unit["unit_id"]).unlink()
    doc = {"schema": "m4_unit_claim.v1", "pid": os.getpid(),
           "boot_id": cr._boot_id(), "epoch": 1.0, "token": "live"}
    doc["claim_sha256"] = cp._selfsha(doc, "claim_sha256")
    cr.unit_claim_path(copy_run, unit["unit_id"]).write_text(
        json.dumps(doc))
    units = _units(fixture_successor)
    prior = cr.role_array_digests(design, cr.cells_of_units(units),
                                  roles=("CALIBRATION",))
    with pytest.raises(SystemExit, match="claimed by a LIVE session"):
        cr.execute_confirmation_units(
            design, units, copy_run, cr.new_accounting(),
            expect_role="DEVELOPMENT", allow_confirmation=False,
            prior_digests=prior)


def test_r6_positive_a_stale_unit_claim_is_set_aside_and_rerun(
        copy_run, fixture_successor, design):
    unit = _units(fixture_successor)[0]
    cr.unit_record_path(copy_run, unit["unit_id"]).unlink()
    doc = {"schema": "m4_unit_claim.v1", "pid": os.getpid(),
           "boot_id": "A_PREVIOUS_BOOT", "epoch": 1.0, "token": "stale"}
    doc["claim_sha256"] = cp._selfsha(doc, "claim_sha256")
    cr.unit_claim_path(copy_run, unit["unit_id"]).write_text(
        json.dumps(doc))
    units = _units(fixture_successor)
    prior = cr.role_array_digests(design, cr.cells_of_units(units),
                                  roles=("CALIBRATION",))
    body = cr.execute_confirmation_units(
        design, units, copy_run, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests=prior)
    assert body["units_new_this_session"] == 1
    assert body["partial_attempts_set_aside"] >= 1
    assert body["census_complete"] is True
    assert not cr.unit_claim_path(copy_run, unit["unit_id"]).exists()
    kept = list((copy_run / cr.ABORTED_DIR).rglob("*__CLAIM.json"))
    assert len(kept) == 1


def test_r6_positive_accounting_is_cumulative_across_sessions(dev_run):
    reports = cr.prior_session_reports(dev_run)
    assert len(reports) >= 1
    cum = cr.cumulative_prior(dev_run)
    assert cum["sessions"] == len(reports)
    assert cum["units_new"] == 24
    assert cum["optimization_updates"] > 0
    last = reports[-1]
    assert last["cumulative"]["units_new"] == 24
    assert last["cumulative"]["wall_seconds"] >= \
        last["session_wall_seconds"]


def test_r6_negative_the_sealed_wall_counts_the_WHOLE_run(
        copy_run, fixture_successor, design):
    """Each session used to restart the frozen wall from zero, so N
    interruptions bought N times the budget."""
    unit = _units(fixture_successor)[0]
    cr.unit_record_path(copy_run, unit["unit_id"]).unlink()
    spent = cr.cumulative_prior(copy_run)["wall_seconds"]
    assert spent > 0
    tight = dict(design, resources=dict(
        design["resources"], max_wall_seconds=spent / 2.0))
    units = _units(fixture_successor)
    prior = cr.role_array_digests(design, cr.cells_of_units(units),
                                  roles=("CALIBRATION",))
    body = cr.execute_confirmation_units(
        tight, units, copy_run, cr.new_accounting(),
        expect_role="DEVELOPMENT", allow_confirmation=False,
        prior_digests=prior)
    assert body["session_status"] == "WALL_STOP"
    assert body["units_new_this_session"] == 0
    assert body["prior_sessions_wall_seconds"] == spent


def test_r6_negative_an_unverifiable_prior_report_is_uncertain(
        copy_run):
    p = sorted(copy_run.glob("SESSION_*_REPORT.json"))[0]
    doc = json.loads(p.read_text())
    doc["units_new_this_session"] = 999
    p.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="UNCERTAIN"):
        cr.cumulative_prior(copy_run)


# ===================================================================
# the COMPLETE entrypoint, positive and negative
# ===================================================================

def test_entrypoint_positive_development_verification_probe(tmp_path):
    out = tmp_path / "probe"
    r = _cli("development-verification-probe", "--out", str(out),
             "--generators", "2", "--seeds", "3")
    assert r.returncode == 0, r.stderr[-1200:]
    doc = json.loads(r.stdout)
    assert doc["body"]["role"] == "DEVELOPMENT"
    assert doc["body"]["units_total"] == 24
    assert doc["body"]["census_complete"] is True
    v = doc["verification"]
    assert v["verdict"] == "VERIFIED"
    assert v["verdict_authority"].startswith("NON_CONFIRMATORY")
    assert v["records_numerically_rederived"] == 24
    assert v["population_complete"] is True
    assert [p for p in out.rglob("*") if "CONFIRMATION" in p.name] == [
        out / "CONFIRMATION_PRE_RESULT_LEDGER.json"]
    assert not (out / cr.RUN_LOCK_NAME).exists()


def test_entrypoint_negative_batched_then_incomplete(tmp_path):
    out = tmp_path / "probe"
    r = _cli("development-verification-probe", "--out", str(out),
             "--generators", "2", "--seeds", "3", "--batch-units", "3")
    assert r.returncode == 0, r.stderr[-1200:]
    doc = json.loads(r.stdout)
    assert doc["body"]["units_new_this_session"] == 3
    assert doc["body"]["session_status"] == "BATCH_FILLED_RESUMABLE"
    assert doc["verification"]["verdict"] == \
        "NOT_ATTEMPTED_CENSUS_INCOMPLETE"
    # and the verifier itself refuses that root, it does not narrow
    fx = cr.development_fixture_successor(generators=2, seeds=3)
    with pytest.raises(SystemExit, match="POPULATION_INCOMPLETE"):
        cr.verify_confirmation_run(REPO, out, fx,
                                   expect_role="DEVELOPMENT")


def test_entrypoint_authority_digest_is_printable_and_binds_nothing(
        tmp_path):
    r = _cli("authority-digest")
    assert r.returncode == 0
    doc = json.loads(r.stdout)
    assert doc["implementation_sha256"] == cp.implementation_digest(REPO)
    assert doc["reviewed_tip_pinned_by_the_order"] == cp.REVIEWED_TIP
    assert not cp.MUSASHI_REVIEW_RECORD_PATH.exists()


def test_plan_still_reports_counts_without_authority():
    r = _cli("plan")
    assert r.returncode == 0
    doc = json.loads(r.stdout)
    assert doc["units_total"] == 3024
    assert doc["execution_open"] is False
    assert doc["musashi_review_record_present"] is False
    assert doc["owner_execution_record_present"] is False


# ===================================================================
# PRE P6 / auditor F6 — the four facts the "zero arrays" sentence blurred
# ===================================================================

def test_f6_four_facts_about_confirmation_arrays(tmp_path):
    """The published sentence "no CONFIRMATION array, score or ledger was
    created … at any point" was false as written. Four facts, separately:

    1. in-memory CONSTRUCTION for the byte-level disjointness proof
       HAPPENS — the bank's docstring permits exactly that exception;
    2. MATERIALIZATION to storage does NOT happen;
    3. FITTING on CONFIRMATION bytes does NOT happen;
    4. SCORING of CONFIRMATION bytes does NOT happen.
    """
    import m4_generator_bank as gb
    design = cp.bind_calibration_evidence(REPO)["design"]

    # fact 1: construction, in memory, only through the named exception
    with pytest.raises(SystemExit, match="RESERVED"):
        gb.generate("CONFIRMATION", "sine", "white", 0)
    g = gb.generate("CONFIRMATION", "sine", "white", 0,
                    allow_confirmation=True)
    digests = cr.generator_array_digests(g)
    assert len(digests) == 6
    assert digests.isdisjoint(
        cr.role_array_digests(design, [("sine", "white")]))
    assert "only the explicit" in gb.generate.__doc__

    # fact 2: nothing of it is materialised, here or in the state root
    assert list(tmp_path.iterdir()) == []
    state = Path.home() / ".local/share/agent-multi"
    assert list(state.glob("*m4*confirmation*")) == []
    assert not cp.MUSASHI_REVIEW_RECORD_PATH.exists()
    assert not cp.OWNER_EXECUTION_RECORD_PATH.exists()

    # fact 3: no fit on those bytes — the default caller refuses, and the
    # gate chain refuses before the execution body is ever reached
    out = tmp_path / "never"
    unit = rn._iv_unit("CONFIRMATION", "sine", "white", 16, 0, 0)
    (tmp_path / "o" / "intervention").mkdir(parents=True)
    with pytest.raises(SystemExit, match="RESERVED"):
        rn._run_intervention_unit_v5(design, unit, tmp_path / "o",
                                     cr.new_accounting())
    r = _cli("execute", "--out", str(out))
    assert r.returncode != 0 and "ABSENT" in (r.stdout + r.stderr)
    assert not out.exists()

    # fact 4: nothing scored — no CONFIRMATION record, report or verdict
    assert list(state.rglob("SESSION_*_REPORT.json")) == []
    assert cr.prior_role_digest_census(state) is not None
