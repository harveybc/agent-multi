"""C37: acceptance battery for the M4 CONFIRMATION protocol.

The fifteen ordered kills plus guard-removal mutants. Every
mutation dies on ITS OWN needle; the real repository is only
ever mutated through save/restore of the (yet-uncommitted)
successor bytes, and every external-record world lives in tmp.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import m4_confirmation_protocol as cp  # noqa: E402
import m4_confirmation_runner as cr  # noqa: E402

SUCC = REPO / cp.SUCCESSOR_PATH


def _selfsha(doc, key):
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()


@pytest.fixture()
def succ_doc():
    return cp.verify_confirmation_successor(REPO)


@pytest.fixture()
def mutate_successor():
    """Yield a mutator that rewrites the successor with a
    repaired self-digest; ALWAYS restores the original bytes."""
    orig = SUCC.read_bytes()

    def mut(fn):
        doc = json.loads(orig.decode())
        fn(doc)
        doc["successor_sha256"] = _selfsha(doc,
                                           "successor_sha256")
        SUCC.write_text(json.dumps(doc, indent=1))
        return doc

    yield mut
    SUCC.write_bytes(orig)


def _mk_records(tmp, monkeypatch, successor_sha,
                forge_review=None, forge_exec=None,
                skip_exec=False):
    d = tmp / "auth"
    d.mkdir(exist_ok=True)
    rev = {
        "schema": "musashi_m4_confirmation_design_review.v1",
        "author": "test-double", "role": "EXTERNAL_AUDITOR",
        "date": "2026-09-10",
        "reviewed_successor_sha256": successor_sha,
        "reviewed_tip": cp.REVIEWED_TIP,
        # DR04 R1: a record that names a revision is not a record that
        # binds the code that will run. Every double pins the live digest.
        "reviewed_implementation_sha256":
            cp.implementation_digest(REPO),
        "reviewed_analysis_statement":
            "16-slot family, Holm supersession, 21 slots",
        "decision":
            "M4_CONFIRMATION_DESIGN_APPROVED_FOR_EXECUTION"}
    if forge_review:
        forge_review(rev)
    rev["record_sha256"] = _selfsha(rev, "record_sha256")
    rp = d / "review.json"
    rp.write_text(json.dumps(rev))
    monkeypatch.setattr(cp, "MUSASHI_REVIEW_RECORD_PATH", rp)
    ep = d / "exec.json"
    if not skip_exec:
        ex = {
            "schema": "owner_m4_confirmation_execution.v1",
            "author": "test-double", "role": "OWNER",
            "date": "2026-09-10",
            "authorized_successor_sha256": successor_sha,
            "authorized_review_record_sha256":
                rev["record_sha256"],
            "authorized_population_statement":
                "21 slots x 48 generators x 3 seeds",
            "cpu_limits_statement":
                "sealed v5 wall/RSS/nice/heartbeat/stop",
            "decision":
                "M4_CONFIRMATION_EXECUTION_AUTHORIZED_CPU_ONLY"}
        if forge_exec:
            forge_exec(ex)
        ex["record_sha256"] = _selfsha(ex, "record_sha256")
        ep.write_text(json.dumps(ex))
    monkeypatch.setattr(cp, "OWNER_EXECUTION_RECORD_PATH", ep)
    return rev


# ---- kill 1: substituted attempt-2 adjudication ----
def test_substituted_attempt2_adjudication_refuses(monkeypatch):
    monkeypatch.setattr(
        cp, "ADJUDICATION_PATH",
        "docs/audits/evidence/M4_V5_CALIBRATION_ADJUDICATION_"
        "ATTEMPT2_NON_GOVERNING_2026_09_09.json")
    with pytest.raises(SystemExit,
                       match="order-pinned identity"):
        cp.bind_calibration_evidence(REPO)


# ---- kill 2: changed 12/16 threshold ----
def test_changed_threshold_refuses(mutate_successor):
    mutate_successor(lambda d: d["eligibility_rule"].update(
        min_learnable_under_frozen_budget=11))
    with pytest.raises(SystemExit, match="diverges|mutated"):
        cp.verify_confirmation_successor(REPO)


# ---- kill 3: one changed eligible slot ----
def test_changed_eligible_slot_refuses(mutate_successor):
    def mut(d):
        d["eligible_slots"][0]["cell"] = \
            "parity4::clean::w16"
    mutate_successor(mut)
    with pytest.raises(SystemExit, match="diverges|mutated"):
        cp.verify_confirmation_successor(REPO)


# ---- kill 4: calling the policy predeclared ----
def test_predeclared_label_refuses(mutate_successor):
    mutate_successor(lambda d: d.update(
        selection_rule_label="PREDECLARED"))
    with pytest.raises(SystemExit, match="diverges|mutated|"
                       "CALIBRATION_DERIVED"):
        cp.verify_confirmation_successor(REPO)


# ---- kill 5: two widths as independent hypotheses ----
def test_width_as_extra_primary_refuses(succ_doc):
    fake_family = list(succ_doc["contrast_family_16"]) + [
        "intervention_effect::sine::clean::w16"]
    doc = dict(succ_doc, contrast_family_16=fake_family)
    with pytest.raises(SystemExit,
                       match="unknown contrast|16"):
        cp.sixteen_contrasts(doc, {}, {})


# ---- kill 6: silently choosing the better width ----
def test_equal_width_average_never_best_pick(succ_doc):
    pge = {}
    for s in succ_doc["eligible_slots"]:
        pre, w = s["cell"].rsplit("::w", 1)
        pge.setdefault(pre, {})[int(w)] = {}
    # sine::clean: w16 strongly positive, w64 strongly negative
    for g in range(45):
        pge["sine::clean"][16][f"g{g}"] = 1.0
        pge["sine::clean"][64][f"g{g}"] = -1.0
        for pre, by_w in pge.items():
            if pre == "sine::clean":
                continue
            for w in by_w:
                by_w[w].setdefault(f"g{g}", 0.1)
    ck = {f"g{g}": 0.1 for g in range(45)}
    out = cp.sixteen_contrasts(succ_doc, pge, ck)
    r = out["contrasts"]["intervention_effect::sine::clean"]
    assert r["effect_mean"] == pytest.approx(0.0), (
        "the primary effect must be the EQUAL average across "
        "frozen widths, never the better width")
    het = cp.width_heterogeneity(succ_doc, pge)
    assert all(v["classification"] == "SECONDARY_HETEROGENEITY"
               for v in het.values())


# ---- kill 7: dropping an ineligible contrast from Holm ----
def test_holm_always_over_16(succ_doc):
    pge = {}
    for s in succ_doc["eligible_slots"]:
        pre, w = s["cell"].rsplit("::w", 1)
        pge.setdefault(pre, {})[int(w)] = {
            f"g{i}": 0.2 + 0.01 * i for i in range(45)}
    ck = {f"g{i}": 0.1 for i in range(45)}
    out = cp.sixteen_contrasts(succ_doc, pge, ck)
    assert out["n_slots"] == 16
    assert len(out["contrasts"]) == 16
    ne = [k for k, v in out["contrasts"].items()
          if v.get("status") == "NOT_EVALUABLE"]
    for k in ne:
        assert out["contrasts"][k]["p_holm"] == 1.0
    assert ("incremental_prediction::M2_vs_M1"
            in out["contrasts"])


# ---- kill 8: fitting or scoring M2 on CONFIRMATION ----
def test_m2_rehabilitation_refuses(mutate_successor):
    mutate_successor(lambda d: d["m2_status"].update(
        status="ADVANCES"))
    with pytest.raises(SystemExit, match="diverges|mutated"):
        cp.verify_confirmation_successor(REPO)


def test_m2_always_placeholder(succ_doc):
    out = cp.sixteen_contrasts(succ_doc, {}, {})
    m2 = out["contrasts"]["incremental_prediction::M2_vs_M1"]
    assert m2["status"].startswith("NON_REJECTING_PLACEHOLDER")
    assert m2["p_holm"] == 1.0
    assert m2["reject_at_alpha"] is False


# ---- kills 9/10/12/15: the ANALYSIS STAGE, exactly ----
#
# DR04: these four properties are properties of the analysis stage, and
# they are now tested on the stage itself, which is pure and exact
# (cr.analyse_complete_population). Before DR04 they were driven by
# fabricated CONFIRMATION summary documents fed to the verifier; that is
# precisely the route the auditor showed could manufacture a rejection, and
# the repaired verifier refuses it at the authority gate. The record layer
# that FEEDS this stage is covered end to end, over real records, in
# tests/test_m4_dr04_verifiability.py.

def _effects(succ_doc, n_gen=40, n_slots=2, seeds=3,
             drop_seed_of=None, invalid_of=None):
    """{"fam::nz": {width: {"gN": {model_seed: paired effect}}}} — the
    SEED-KEYED map the analysis stage consumes. ``drop_seed_of`` removes a
    nested seed; ``invalid_of`` models a unit that stayed in the
    denominator as typed-invalid (no effect at all)."""
    out = {}
    attrited = []
    for s in succ_doc["eligible_slots"][:n_slots]:
        pre, w = s["cell"].rsplit("::w", 1)
        for gi in range(n_gen):
            eff = 0.3 + 0.01 * gi
            per_seed = {}
            for ms in range(seeds):
                if drop_seed_of == (s["cell"], gi) and ms == seeds - 1:
                    continue
                if invalid_of == (s["cell"], gi) and ms == 0:
                    attrited.append(f"{s['cell']}::g{gi}::s{ms}")
                    continue
                per_seed[ms] = eff
            out.setdefault(pre, {}).setdefault(
                int(w), {})[f"g{gi}"] = per_seed
    return out, attrited


def test_missing_nested_seed_never_completes(succ_doc):
    cell = succ_doc["eligible_slots"][0]["cell"]
    eff, _ = _effects(succ_doc, n_gen=40, drop_seed_of=(cell, 0))
    out = cr.analyse_complete_population(succ_doc, eff)
    pre, w = cell.rsplit("::w", 1)
    # g0 has 2 of 3 seeds -> never averaged as complete, and it is NAMED
    assert f"{pre}::w{w}::g0" in out["generators_incomplete_seed_sets"]
    assert "g0" not in out["complete"][pre][int(w)]
    assert out["population_complete"] is False
    r = out["analysis"]["contrasts"][f"intervention_effect::{pre}"]
    if r["status"] == "EVALUATED":
        assert r["n_generators"] <= 39
    assert r["population_complete"] is False


def test_attrition_beyond_allowance_never_favorable(succ_doc):
    eff, _ = _effects(succ_doc, n_gen=30)
    out = cr.analyse_complete_population(succ_doc, eff)
    for cell, st in out["confirmation_incomplete"].items():
        assert st["status"] == "CONFIRMATION_INCOMPLETE"
        assert st["complete_generators"] < st[
            "min_complete_required"]
    assert len(out["confirmation_incomplete"]) >= 2
    for k, r in out["analysis"]["contrasts"].items():
        if k.startswith("intervention_effect::"):
            assert r.get("reject_at_alpha") is not True or \
                r.get("n_generators", 0) >= 39
    # DR04 R4: the FIFTEENTH contrast no longer escapes attrition —
    # below-floor slots contribute no observation at all
    ck = out["analysis"]["contrasts"]["checkpoint_effect::primary_pair"]
    assert ck["reject_at_alpha"] is not True
    assert all(c not in out["checkpoint_population"]
               for c in out["confirmation_incomplete"])
    assert out["population_complete"] is False


def test_forged_producer_aggregate_refuses(succ_doc):
    """A declared paired difference that does not equal the difference of
    the record's own arm endpoints refuses at the record boundary."""
    design = cp.bind_calibration_evidence(REPO)["design"]
    unit = cr.census_units(succ_doc, "CONFIRMATION")[0]
    rec = dict(unit)
    rec.update({
        "manifest_sha256": "0" * 64, "task_kind": "regression",
        "tape_id": "t", "tape_digest": "d", "tape_tol": 0.1,
        "genesis_digest": "g", "stop_trajectory_digest": "s",
        "stop_trajectory_slope": 0.0, "selected_stop_update": 1,
        "checkpoint_lineage": {
            k: {"params_digest": "g" if k == "initialization" else k,
                "parent": None, "updates": 0}
            for k in ("initialization", "pre_stop",
                      "calibration_stop", "post_stop_bounded")},
        "arms": {k: {"restricted_endpoint": 10, "cap_reached": False,
                     "stopping_cause": "ACQUISITION_ENDPOINT",
                     "updates_done": 400, "retention_margin": 1.0,
                     "fail_batch": 0, "final_params_digest": "f",
                     "descriptors": {}, "checkpoint_loss_stop": 1.0}
                 for k in ("initialization", "pre_stop",
                           "calibration_stop", "post_stop_bounded")},
        "paired_primary_difference": 99.9})
    with pytest.raises(SystemExit,
                       match="does not equal the difference"):
        cr.validate_unit_record(rec, unit, "CONFIRMATION")
    assert design is not None


def test_numerical_failure_stays_in_denominator(succ_doc):
    cell = succ_doc["eligible_slots"][0]["cell"]
    eff, attrited = _effects(succ_doc, n_gen=40,
                             invalid_of=(cell, 0))
    assert attrited, "the typed-invalid unit must be named"
    out = cr.analyse_complete_population(succ_doc, eff)
    pre, w = cell.rsplit("::w", 1)
    # the generator that lost a seed to numerical failure is NOT complete
    # and the loss is visible, never silently removed from the denominator
    assert f"{pre}::w{w}::g0" in out["generators_incomplete_seed_sets"]
    assert out["population_complete"] is False


# ---- kill 11: CAL/CONF byte overlap ----
def test_role_byte_overlap_refuses():
    a = {"d1", "d2", "d3"}
    b = {"d3", "d4"}
    with pytest.raises(SystemExit, match="never confirms"):
        cr.verify_role_disjointness(a, b)
    cr.verify_role_disjointness({"d1"}, {"d2"})


def test_foreign_role_record_refuses():
    with pytest.raises(SystemExit,
                       match="structurally inadmissible"):
        cr.refuse_foreign_role_record(
            {"unit_id":
             "intervention::CALIBRATION::am::clean::w16"
             "::g0::s0"})


# ---- kill 13: absent, forged, transplanted records ----
def test_absent_records_refuse(succ_doc, tmp_path,
                               monkeypatch):
    monkeypatch.setattr(cp, "MUSASHI_REVIEW_RECORD_PATH",
                        tmp_path / "nope.json")
    with pytest.raises(SystemExit, match="ABSENT"):
        cp.require_both_records(succ_doc)


def test_forged_record_self_refuses(succ_doc, tmp_path,
                                    monkeypatch):
    _mk_records(tmp_path, monkeypatch,
                succ_doc["successor_sha256"])
    p = cp.MUSASHI_REVIEW_RECORD_PATH
    doc = json.loads(p.read_text())
    doc["reviewed_analysis_statement"] = "tampered"
    p.write_text(json.dumps(doc))
    with pytest.raises(SystemExit,
                       match="does not re-derive"):
        cp.require_both_records(succ_doc)


def test_transplanted_record_refuses(succ_doc, tmp_path,
                                     monkeypatch):
    _mk_records(tmp_path, monkeypatch, "ab" * 32)
    with pytest.raises(SystemExit, match="DIFFERENT successor"):
        cp.require_both_records(succ_doc)


def test_broken_owner_chain_refuses(succ_doc, tmp_path,
                                    monkeypatch):
    _mk_records(tmp_path, monkeypatch,
                succ_doc["successor_sha256"],
                forge_exec=lambda e: e.update(
                    authorized_review_record_sha256="cd" * 32))
    with pytest.raises(SystemExit, match="chain"):
        cp.require_both_records(succ_doc)


# ---- DR04 R1: authority binds the EXECUTABLE implementation ----
def test_review_pinning_another_implementation_refuses(
        succ_doc, tmp_path, monkeypatch):
    """A review record that pins a different implementation digest
    authorizes nothing: a recorded revision names a commit, this binds the
    bytes that would run. The role token and the decision string are
    untouched, and nothing is installed at the real record path."""
    _mk_records(tmp_path, monkeypatch,
                succ_doc["successor_sha256"],
                forge_review=lambda r: r.update(
                    reviewed_implementation_sha256="ab" * 32))
    with pytest.raises(SystemExit,
                       match="a review of other code authorizes nothing"):
        cp.require_both_records(succ_doc)
    assert not (Path.home() / ".local/share/agent-multi"
                / "m4_confirmation_authority").exists()


def test_review_without_an_implementation_digest_refuses(
        succ_doc, tmp_path, monkeypatch):
    _mk_records(tmp_path, monkeypatch,
                succ_doc["successor_sha256"],
                forge_review=lambda r: r.update(
                    reviewed_implementation_sha256="<to be filled>"))
    with pytest.raises(SystemExit, match="template|64-hex"):
        cp.require_both_records(succ_doc)


def test_review_pinning_the_live_implementation_passes_that_check(
        succ_doc, tmp_path, monkeypatch):
    """The positive: with the live digest pinned, the implementation check
    is satisfied and the chain proceeds to the owner record."""
    _mk_records(tmp_path, monkeypatch,
                succ_doc["successor_sha256"])
    both = cp.require_both_records(succ_doc)
    assert both["review"]["reviewed_implementation_sha256"] == \
        cp.implementation_digest(REPO)
    assert both["execution"]["authorized_review_record_sha256"] == \
        both["review"]["record_sha256"]


def test_template_placeholder_refuses(succ_doc, tmp_path,
                                      monkeypatch):
    _mk_records(tmp_path, monkeypatch,
                succ_doc["successor_sha256"],
                forge_review=lambda r: r.update(
                    author="<musashi-signature-name>"))
    with pytest.raises(SystemExit, match="template"):
        cp.require_both_records(succ_doc)


# ---- kill 14: scientific change after review ----
def test_scientific_change_after_review_refuses(
        tmp_path, monkeypatch, mutate_successor):
    original = cp.verify_confirmation_successor(REPO)
    # both records pin the REVIEWED successor identity
    _mk_records(tmp_path, monkeypatch,
                original["successor_sha256"])
    # the successor is then scientifically changed (self
    # repaired) — the changed document never reaches execution:
    mutated = mutate_successor(
        lambda d: d["attrition"].update(allowance=0.5))
    # path 1: live verification refuses the changed bytes
    with pytest.raises(SystemExit, match="diverges"):
        cp.verify_confirmation_successor(REPO)
    # path 2: even presented directly, the record chain refuses
    with pytest.raises(SystemExit, match="DIFFERENT successor"):
        cp.require_both_records(mutated)


# ---- two-record gate closes execution entirely ----
def test_execute_refuses_before_any_artifact(tmp_path,
                                             monkeypatch):
    monkeypatch.setattr(cp, "MUSASHI_REVIEW_RECORD_PATH",
                        tmp_path / "absent.json")
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="ABSENT"):
        cr.execute_confirmation(REPO, out)
    assert not out.exists(), (
        "execution refused AFTER creating artifacts — the "
        "refusal must come first")


def test_plan_reports_counts_without_authority():
    plan = cr.plan_confirmation(REPO)
    assert plan["units_total"] == 3024
    assert plan["eligible_slots"] == 21
    assert plan["execution_open"] is False


# ---- guard-removal mutants (subprocess) ----
MUTANTS = {
    "sixteen_check_off": (
        "m4_confirmation_protocol.py",
        '''    if len(pvals) != 16:
        raise ConfirmationProtocolRefusal(
            f"{len(pvals)} slots reached Holm — the frozen "
            "procedure runs over ALL 16 including placeholders")''',
        "    if False:\n        pass",
        "fifteen_slot_analysis_admitted"),
    "rederive_off": (
        "m4_confirmation_runner.py",
        '''    if rec["paired_primary_difference"] != want:
        raise ConfirmationRunnerRefusal(''',
        '''    if False:
        raise ConfirmationRunnerRefusal(''',
        "forged_aggregate_admitted"),
    "floor_off": (
        "m4_confirmation_runner.py",
        "        if n_complete < floor:",
        "        if False:",
        "over_attrition_admitted"),
    "disjoint_off": (
        "m4_confirmation_runner.py",
        '''    hit = sorted(set(conf_digests) & set(prior_digests))
    if hit:''',
        '''    hit = sorted(set(conf_digests) & set(prior_digests))
    if False:''',
        "byte_overlap_admitted"),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_guard_removal_mutant_bites(name, tmp_path):
    fname, old, new, marker = MUTANTS[name]
    src = (REPO / "tools" / fname).read_text()
    assert old in src, (name, "anchor missing")
    mdir = tmp_path / "mut"
    mdir.mkdir()
    for f in ("m4_confirmation_protocol.py",
              "m4_confirmation_runner.py"):
        s = (REPO / "tools" / f).read_text()
        if f == fname:
            s = s.replace(old, new)
        (mdir / f).write_text(s)
    driver = mdir / "driver.py"
    driver.write_text(f"""
import importlib.util
import json, sys
sys.path.insert(0, {str(REPO / 'tools')!r})


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load("m4_confirmation_protocol",
           {str(mdir)!r} + "/m4_confirmation_protocol.py")
cr = _load("m4_confirmation_runner",
           {str(mdir)!r} + "/m4_confirmation_runner.py")
assert cp.__file__.startswith({str(mdir)!r}), cp.__file__
assert cr.__file__.startswith({str(mdir)!r}), cr.__file__
name = {name!r}
try:
    if name == "sixteen_check_off":
        doc = cp.verify_confirmation_successor({str(REPO)!r})
        fam15 = doc["contrast_family_16"][:15]
        doc2 = dict(doc, contrast_family_16=fam15)
        out = cp.sixteen_contrasts(doc2, {{}}, {{}})
        print("MUTANT_ADMITTED", len(out["contrasts"]))
    elif name == "disjoint_off":
        cr.verify_role_disjointness({{"x"}}, {{"x"}})
        print("MUTANT_ADMITTED overlap")
    else:
        print("MUTANT_NEEDS_WORLD")
except SystemExit as e:
    print("REFUSED", str(e)[:60])
""")
    rc = subprocess.run([sys.executable, str(driver)],
                        capture_output=True, text=True,
                        timeout=300)
    if name in ("sixteen_check_off", "disjoint_off"):
        assert "MUTANT_ADMITTED" in rc.stdout, (
            name, rc.stdout, rc.stderr[-300:])
        clean = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, "
             f"{str(REPO / 'tools')!r})\n"
             "import m4_confirmation_runner as cr\n"
             "import m4_confirmation_protocol as cp\n"
             + ("cr.verify_role_disjointness({'x'},{'x'})"
                if name == "disjoint_off" else
                f"doc = cp.verify_confirmation_successor("
                f"{str(REPO)!r})\n"
                "d2 = dict(doc, contrast_family_16="
                "doc['contrast_family_16'][:15])\n"
                "cp.sixteen_contrasts(d2, {}, {})")],
            capture_output=True, text=True, timeout=300)
        assert clean.returncode != 0, (
            name, "the LIVE guard does not refuse")


def test_guard_removal_rederive_and_floor(tmp_path):
    """rederive_off and floor_off are exercised in-process against
    mutated module copies — the forged aggregate on a record, and the
    over-attrition slot on the analysis stage."""
    import importlib.util
    succ_doc = cp.verify_confirmation_successor(REPO)
    for name in ("rederive_off", "floor_off"):
        fname, old, new, marker = MUTANTS[name]
        src = (REPO / "tools" / fname).read_text()
        assert old in src, (name, "anchor missing")
        mdir = tmp_path / name
        mdir.mkdir()
        (mdir / fname).write_text(src.replace(old, new))
        spec = importlib.util.spec_from_file_location(
            f"mut_{name}", mdir / fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if name == "rederive_off":
            unit = cr.census_units(succ_doc, "CONFIRMATION")[0]
            rec = dict(unit)
            rec.update({
                "manifest_sha256": "0" * 64,
                "task_kind": "regression", "tape_id": "t",
                "tape_digest": "d", "tape_tol": 0.1,
                "genesis_digest": "g", "stop_trajectory_digest": "s",
                "stop_trajectory_slope": 0.0,
                "selected_stop_update": 1,
                "checkpoint_lineage": {
                    k: {"params_digest":
                        "g" if k == "initialization" else k,
                        "parent": None, "updates": 0}
                    for k in ("initialization", "pre_stop",
                              "calibration_stop",
                              "post_stop_bounded")},
                "arms": {k: {"restricted_endpoint": 10,
                             "cap_reached": False,
                             "stopping_cause": "ACQUISITION_ENDPOINT",
                             "updates_done": 400,
                             "retention_margin": 1.0,
                             "fail_batch": 0,
                             "final_params_digest": "f",
                             "descriptors": {},
                             "checkpoint_loss_stop": 1.0}
                         for k in ("initialization", "pre_stop",
                                   "calibration_stop",
                                   "post_stop_bounded")},
                "paired_primary_difference": 99.9})
            # the mutant ADMITS the forged aggregate
            mod.validate_unit_record(rec, unit, "CONFIRMATION")
            with pytest.raises(SystemExit):
                cr.validate_unit_record(rec, unit, "CONFIRMATION")
        else:
            eff, _ = _effects(succ_doc, n_gen=30)
            out = mod.analyse_complete_population(succ_doc, eff)
            assert out["confirmation_incomplete"] == {}, (
                "mutant did not admit the over-attrition slot")
            live = cr.analyse_complete_population(succ_doc, eff)
            assert live["confirmation_incomplete"] != {}, (
                "the LIVE floor guard does not bite")
