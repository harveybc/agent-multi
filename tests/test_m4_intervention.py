"""M4 C17-C24 battery: the twelve ordered kills plus foundation
facts — every forgery, role overlap, denominator drop and
double-invocation dies against the productive verifier."""
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import m4_generator_bank as gb  # noqa: E402
import m4_intervention_design as dz  # noqa: E402
import m4_intervention_runner as rn  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402


def _small_design():
    d = copy.deepcopy(dz.load_design_v4())
    cp = d["candidate_population"]
    cp["structured_boolean_families"] = ["identity", "parity4"]
    cp["temporal_families"] = ["sine"]
    cp["noise_regimes_temporal"] = ["clean", "white"]
    cp["hidden_widths"] = [16]
    d["population_census"][
        "development_generators_per_cell"] = 1
    d["four_unit_rule"]["units"] = [
        {"family": "identity", "noise": "clean", "width": 16,
         "generator_index": 0, "model_seed": 0}]
    del d["design_sha256"]
    d["design_sha256"] = m4._self_sha(d, "design_sha256")
    return d


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    d = _small_design()
    out = tmp_path_factory.mktemp("m4iv") / "run"
    rn.execute(d, out)
    assert rn.verify_run(d, out)["verified"] is True
    return d, out


@pytest.fixture()
def world(small_run, tmp_path):
    d, src = small_run
    out = tmp_path / "run"
    shutil.copytree(src, out)
    return d, out


def _repair_report(out):
    p = out / "RUN_REPORT.json"
    rep = json.loads(p.read_text())
    rep["artifacts_sha256"] = rn._inventory(out)
    rep.pop("record_sha256")
    rep["record_sha256"] = m4._self_sha(rep, "record_sha256")
    p.write_text(json.dumps(rep, indent=1, sort_keys=True))


def _rewrite(path, mutate):
    doc = json.loads(Path(path).read_text())
    mutate(doc)
    doc.pop("record_sha256", None)
    doc["record_sha256"] = m4._self_sha(doc, "record_sha256")
    Path(path).write_text(json.dumps(doc, indent=1,
                                     sort_keys=True))


def test_kill_1_role_overlap_refuses(world, monkeypatch):
    """Kill 1: DEVELOPMENT/CALIBRATION/CONFIRMATION identities
    are disjoint by bytes; collapsing the role out of the seed
    refuses, and a foreign-role unit smuggled into the ledger
    refuses against the sealed population."""
    shas = gb.assert_role_disjointness()
    assert len(set(shas.values())) == 3
    real_seed = gb._seed
    with monkeypatch.context() as mp:
        mp.setattr(gb, "_seed",
                   lambda *parts: real_seed(
                       *[x for x in parts
                         if x not in gb.ROLES]))
        with pytest.raises(SystemExit, match="NOT disjoint"):
            gb.assert_role_disjointness()
    d, out = world
    _rewrite(out / "RUN_LEDGER.json",
             lambda led: led["units"][0].__setitem__(
                 "generator_id",
                 led["units"][0]["generator_id"].replace(
                     "DEVELOPMENT", "CONFIRMATION")))
    _repair_report(out)
    with pytest.raises(SystemExit,
                       match="does not enumerate the sealed "
                             "population"):
        rn.verify_run(d, out)


def test_kill_2_train_only_scaling():
    """Kill 2: every noise scale derives from the TRAIN slice
    only — a latent whose held-out tail explodes must not change
    the disturbance scale."""
    rng = np.random.default_rng(7)
    latent = np.concatenate([np.ones(200) * 0.5,
                             np.ones(64) * 50.0])
    tr = slice(0, 200)
    d = gb._disturbance("white", rng, latent, tr)
    expected = 0.1 * float(np.std(latent[tr]) + 1e-12)
    observed = float(np.std(d[tr]))
    assert abs(observed - expected) / expected < 0.35
    total_scale = 0.1 * float(np.std(latent))
    assert observed < total_scale / 10   # future leak would 40x


def test_kill_3_random_label_never_licensed(world):
    """Kill 3: the negative control can never be licensed as
    structured learning — it is excluded from cells, and inflated
    random-label improvements RAISE the margin, demoting marginal
    cells instead of passing them."""
    d, out = world
    recs = [json.loads(p.read_text())
            for p in sorted((out / "screen").glob("*.json"))]
    tab = rn.derive_learnability_table(d, recs)
    assert not any("random_label" in k for k in tab["cells"])
    forged = []
    for r in recs:
        r = dict(r)
        if r["family"] == "random_label":
            r["improvement"] = 0.5
        forged.append(r)
    tab2 = rn.derive_learnability_table(d, forged)
    assert tab2["margin"] >= 0.5
    assert all(v["outcome"] != "LEARNABLE_UNDER_FROZEN_BUDGET"
               for v in tab2["cells"].values()
               if v["improvement"] is not None
               and v["improvement"] < 0.5)


def test_kill_4_5_forged_batch_records_refuse(world):
    """Kills 4/5: a forged cumulative count (forgotten early
    association hidden) and a forged retention streak (one
    failure counted as two) both die on the transition replay."""
    d, out = world
    lp = next((out / "intervention").glob("*_control.jsonl"))
    lines = lp.read_text().splitlines()
    r0 = json.loads(lines[0])
    assert not (r0["cumulative_ok"] and
                r0["cumulative_acquired"]
                == r0["cumulative_associations"])
    r0["cumulative_acquired"] = r0["cumulative_associations"]
    r0["cumulative_ok"] = True
    r0["outcome"] = "ACCEPTED"
    r0.pop("record_sha256")
    r0["record_sha256"] = m4._self_sha(r0, "record_sha256")
    lines[0] = json.dumps(r0, sort_keys=True)
    lp.write_text("\n".join(lines) + "\n")
    _repair_report(out)
    with pytest.raises(SystemExit, match="does not replay"):
        rn.verify_run(d, out)


def test_kill_5_streak_forgery_refuses(world):
    d, out = world
    lp = next((out / "intervention").glob("*_control.jsonl"))
    lines = lp.read_text().splitlines()
    r0 = json.loads(lines[0])
    r0["retention_streak"] = 2
    r0["outcome"] = "RETENTION_ENDPOINT"
    r0.pop("record_sha256")
    r0["record_sha256"] = m4._self_sha(r0, "record_sha256")
    lines[0] = json.dumps(r0, sort_keys=True)
    lp.write_text("\n".join(lines) + "\n")
    _repair_report(out)
    with pytest.raises(SystemExit, match="does not replay"):
        rn.verify_run(d, out)


def test_kill_6_max_batches_is_censored(world):
    """Kill 6: MAX_BATCHES claimed as an OBSERVED endpoint
    (censored=false) refuses against the replayed derivation;
    and the sealed design types it right-censored."""
    d, out = world
    assert "RIGHT-CENSORED" in \
        d["estimands"]["primary_endpoint"]
    sp = next((out / "intervention").glob("*_summary.json"))
    doc = json.loads(sp.read_text())
    arm = ("treatment"
           if doc["arms"]["treatment"]["censored"]
           else "control"
           if doc["arms"]["control"]["censored"] else None)
    if arm is None:
        arm = "treatment"
        doc["arms"][arm]["censored"] = True   # make forgeable
        doc["arms"][arm]["stopping_cause"] = "MAX_BATCHES"
    doc["arms"][arm]["censored"] = False
    doc.pop("record_sha256")
    doc["record_sha256"] = m4._self_sha(doc, "record_sha256")
    sp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    _repair_report(out)
    with pytest.raises(SystemExit,
                       match="censoring facts do not equal|"
                             "does not reconstruct"):
        rn.verify_run(d, out)


def test_kill_7_nesting_is_sealed():
    """Kill 7: checkpoints and seeds are nested repetitions —
    never independent generators — sealed in the design."""
    d = dz.load_design_v4()
    assert "NEVER treated as independent" in \
        d["estimands"]["analysis_2_checkpoint_effect"]
    assert d["population_census"]["model_seeds_nested"] == 3
    assert "GENERATOR is the independent unit" in \
        d["candidate_population"]["unit_definition"]
    assert "nested" in d["statistics"]["nested_seeds"]


def test_kill_8_denominator_is_complete(world):
    """Kill 8: a missing/failed unit can never leave the
    denominator — deleting a screen record refuses on the exact
    inventory, and the table counts every enumerated cell."""
    d, out = world
    recs = [json.loads(p.read_text())
            for p in sorted((out / "screen").glob("*.json"))]
    tab = rn.derive_learnability_table(d, recs)
    total = sum(v["total"]
                for v in tab["family_noise_summary"].values())
    structured = [r for r in recs if r["family"] not in
                  ("random_label", "easy_constant")]
    assert total == len(structured)
    victim = sorted((out / "screen").glob("*.json"))[0]
    victim.unlink()
    _repair_report(out)
    with pytest.raises(SystemExit,
                       match="does not enumerate the sealed "
                             "population|does not replay|"
                             "No such file|screen record"):
        rn.verify_run(d, out)


def test_kill_9_forged_accounting_refuses(world):
    d, out = world
    p = out / "RUN_REPORT.json"
    _rewrite(p, lambda rep: rep["accounting"].__setitem__(
        "optimization_updates",
        rep["accounting"]["optimization_updates"] + 1))
    with pytest.raises(SystemExit,
                       match="accounting does not equal"):
        rn.verify_run(d, out)
    _rewrite(p, lambda rep: rep["accounting"].__setitem__(
        "optimization_updates",
        rep["accounting"]["optimization_updates"] - 1) or
        rep["accounting"].__setitem__("descriptor_evals", 0))
    with pytest.raises(SystemExit,
                       match="descriptor cost accounting"):
        rn.verify_run(d, out)


def test_kill_10_relabeled_family_refuses(world):
    """Kill 10: an unlearnable family relabeled as learnable in
    the committed table dies on the frozen-rule re-derivation."""
    d, out = world
    p = out / "LEARNABILITY_TABLE.json"
    tab = json.loads(p.read_text())
    victim = next(iter(tab["cells"]))
    cur = tab["cells"][victim]["outcome"]
    tab["cells"][victim]["outcome"] = (
        "OPTIMIZATION_LIMITED"
        if cur == "LEARNABLE_UNDER_FROZEN_BUDGET"
        else "LEARNABLE_UNDER_FROZEN_BUDGET")
    tab.pop("record_sha256")
    tab["record_sha256"] = m4._self_sha(tab, "record_sha256")
    p.write_text(json.dumps(tab, indent=1, sort_keys=True))
    _repair_report(out)
    with pytest.raises(SystemExit, match="does not re-derive"):
        rn.verify_run(d, out)


def test_kill_11_forged_summary_refuses(world):
    d, out = world
    sp = next((out / "intervention").glob("*_summary.json"))
    _rewrite(sp, lambda doc: doc.__setitem__(
        "paired_difference_endpoint",
        doc["paired_difference_endpoint"] + 8))
    _repair_report(out)
    with pytest.raises(SystemExit,
                       match="paired difference does not "
                             "re-derive|does not equal the "
                             "replayed"):
        rn.verify_run(d, out)


def test_kill_12_second_invocation_overwrites_nothing(world):
    d, out = world
    before = {str(p.relative_to(out)): m4._sha_file(p)
              for p in sorted(out.rglob("*")) if p.is_file()
              and p.name != "M4_RUN_HEARTBEAT.json"}
    r = rn.execute(d, out)
    assert r["units_new_this_session"] == 0
    assert r["units_resumed_verified"] > 0
    after = {str(p.relative_to(out)): m4._sha_file(p)
             for p in sorted(out.rglob("*")) if p.is_file()
             and p.name != "M4_RUN_HEARTBEAT.json"}
    new_files = set(after) - set(before)
    assert all(n.startswith("RUN_REPORT_resume_")
               for n in new_files)
    assert all(after[k] == before[k] for k in before)


def test_foundation_design_facts():
    """C17/C18/C21 sealed facts: population census, meaningful
    products, precision table, contrast family, reduction and
    four-unit rules all present and coherent."""
    d = dz.load_design_v4()
    c = d["population_census"]
    assert c["structured_family_noise_cells"] == 29
    assert c["learnability_screen_units"] == 248
    assert d["confirmatory_contrast_family"]["count"] == 16
    prec = d["precision"]
    assert prec["assumptions"]["alpha"] == 0.05
    assert prec["minimum_confirmation_generators_by_sd"][
        "sd2"] == 10
    assert d["scientific_outcome"] == "NONE"
    assert d["supersedes_design_sha256"] == \
        m4._strict_json_file(m4.DESIGN_PATH_V3, "v3")[
            "design_sha256"]
    with pytest.raises(SystemExit, match="nonsensical product"):
        gb.generate("DEVELOPMENT", "parity4", "white", 0)
    v4 = copy.deepcopy(d)
    v4["retention"] = {"metric": "changed"}
    v4.pop("design_sha256")
    v4["design_sha256"] = m4._self_sha(v4, "design_sha256")
    with pytest.raises(SystemExit,
                       match="changes an accepted v3 mechanic"):
        dz.verify_design_supersession_v4(v4)
