"""M4 C33 battery: the eighteen ordered kills against the v5
protocol — paired tapes, one genesis, truthful semantics, causal
splits, real checkpoints, role custody, durable resume, private
artifacts, canonical reporting, the pre-outcome boundary and the
structural CONFIRMATION closure."""
import copy
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import m4_generator_bank as gb  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402
import m4_v5_protocol as pv  # noqa: E402
import m4_v5_runner as rn  # noqa: E402


def _small_v5():
    d = copy.deepcopy(pv.load_design_v5())
    cp = d["candidate_population"]
    cp["structured_boolean_families"] = ["identity"]
    cp["temporal_families"] = ["sine"]
    cp["noise_regimes_temporal"] = ["clean"]
    cp["hidden_widths"] = [16]
    d["populations_v5"]["DEVELOPMENT_per_cell"] = 1
    d["populations_v5"]["CALIBRATION_per_cell"] = 2
    d["four_unit_rule"]["units"] = [
        {"family": "identity", "noise": "clean", "width": 16,
         "generator_index": 0, "model_seed": 0}]
    del d["design_sha256"]
    d["design_sha256"] = m4._self_sha(d, "design_sha256")
    return d


@pytest.fixture(scope="module")
def dev_run(tmp_path_factory):
    d = _small_v5()
    out = tmp_path_factory.mktemp("m4v5") / "run"
    r = rn.execute_v5(d, out, ("DEVELOPMENT",))
    assert r["verified"] is True
    return d, out


@pytest.fixture()
def world(dev_run, tmp_path):
    d, src = dev_run
    out = tmp_path / "run"
    shutil.copytree(src, out)
    for q in out.rglob("*"):
        if q.is_file():
            os.chmod(q, 0o600)
    return d, out


def _repair_report(out):
    p = out / "RUN_REPORT.json"
    rep = json.loads(p.read_text())
    rep["artifacts_sha256"] = rn._inventory_v5(out)
    rep.pop("record_sha256")
    rep["record_sha256"] = m4._self_sha(rep, "record_sha256")
    p.write_text(json.dumps(rep, indent=1, sort_keys=True))
    os.chmod(p, 0o600)


def test_kill_1_one_tape_across_arms(world):
    """Kill 1: the association tape identity excludes arm and
    checkpoint names — both arms consume byte-identical tapes,
    and a record bound to a foreign tape refuses."""
    d, out = world
    g = gb.generate("DEVELOPMENT", "identity", "clean", 0)
    t1 = pv.association_tape(d["design_sha256"], g, 16, 0)
    t2 = pv.association_tape(d["design_sha256"], g, 16, 0)
    assert t1["digest"] == t2["digest"]
    assert "treatment" not in t1["tape_id"]
    assert "control" not in t1["tape_id"]
    sp = next((out / "intervention").glob("*_summary.json"))
    lp = next((out / "intervention").glob(
        "*__calibration_stop.jsonl"))
    lines = lp.read_text().splitlines()
    r0 = json.loads(lines[0])
    r0["tape_digest"] = "a" * 64
    r0.pop("record_sha256")
    r0["record_sha256"] = m4._self_sha(r0, "record_sha256")
    lines[0] = json.dumps(r0, sort_keys=True)
    lp.write_text("\n".join(lines) + "\n")
    os.chmod(lp, 0o600)
    _repair_report(out)
    with pytest.raises(SystemExit,
                       match="foreign tape|does not replay"):
        rn.verify_run_v5(d, out, ("DEVELOPMENT",))


def test_kill_2_one_genesis(world):
    """Kill 2: initialization IS the genesis — identical for
    every arm of the unit; a forged genesis digest refuses."""
    d, out = world
    g = gb.generate("DEVELOPMENT", "identity", "clean", 0)
    ck = pv.build_checkpoints(g, 16, 0)
    ge = pv.genesis_params(g, 16, 0)
    assert ck["checkpoints"]["initialization"][
        "params_digest"] == m4._params_digest(ge)
    sp = next((out / "intervention").glob("*_summary.json"))
    doc = json.loads(sp.read_text())
    doc["genesis_digest"] = "b" * 64
    doc.pop("record_sha256")
    doc["record_sha256"] = m4._self_sha(doc, "record_sha256")
    sp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.chmod(sp, 0o600)
    _repair_report(out)
    with pytest.raises(SystemExit,
                       match="genesis does not re-derive"):
        rn.verify_run_v5(d, out, ("DEVELOPMENT",))


def test_kill_3_baseline_evaluated_on_evaluation():
    """Kill 3: the Boolean baseline is the train-selected class
    EVALUATED on evaluation rows — the PRE's eight mismatches
    are dead."""
    for fam in gb.BOOL_FAMILIES:
        for gi in range(2):
            g = gb.generate("DEVELOPMENT", fam, "clean", gi)
            p = pv.genesis_params(g, 16, 0)
            _, base, _ = pv.heldout_metric("boolean", g, p)
            maj_class = 1.0 if float(
                (g["y_train"] > 0.5).mean()) >= 0.5 else 0.0
            want = float(((g["y_held"] > 0.5)
                          == (maj_class > 0.5)).mean())
            assert base == want


def test_kill_4_sigmoid_executes():
    """Kill 4: the Boolean head EXECUTES sigmoid — outputs live
    in (0,1) and the sealed design says EXECUTED."""
    g = gb.generate("DEVELOPMENT", "parity4", "clean", 0)
    p = pv.genesis_params(g, 16, 0)
    for _ in range(200):
        rng = np.random.default_rng(_)
        i = rng.integers(0, len(g["y_train"]), size=16)
        pv.sgd_step_task("boolean", p, g["X_train"][i],
                         g["y_train"][i], 0.05)
    _, out = pv.forward_task("boolean", p, g["X_held"])
    assert (out > 0).all() and (out < 1).all()
    d5 = pv.load_design_v5()
    assert "EXECUTED" in d5["architectures"]["boolean_head"]
    assert "sigmoid" in d5["architectures"]["boolean_head"]


def test_kill_5_family_compatible_tapes():
    """Kill 5: association inputs/targets are family-compatible
    — Boolean tapes live on {-1,+1}x{0,1}; temporal tapes use
    the observation process with continuous train-marginal
    targets."""
    d = _small_v5()
    gB = gb.generate("DEVELOPMENT", "identity", "clean", 0)
    tB = pv.association_tape(d["design_sha256"], gB, 16, 0)
    assert np.isin(tB["X"], (-1.0, 1.0)).all()
    assert set(np.unique(tB["y"])) <= {0.0, 1.0}
    gT = gb.generate("DEVELOPMENT", "sine", "white", 0)
    tT = pv.association_tape(d["design_sha256"], gT, 16, 0)
    assert not np.isin(tT["X"], (-1.0, 1.0)).all()
    assert np.isin(tT["y"], gT["y_train"]).all()
    assert tT["tol"] != pv.ACQ_TOL_BOOL


def test_kill_6_stop_selects_evaluation_scores():
    """Kill 6: STOP rows select the checkpoint; EVALUATION rows
    score retention/learnability — never the same slice."""
    src = (REPO / "tools/m4_v5_protocol.py").read_text()
    sel = src[src.index("def build_checkpoints"):
              src.index("def run_intervention")]
    assert 'g["X_stop"]' in sel and 'g["X_held"]' not in sel
    ri = src[src.index("def run_intervention"):
             src.index("def dispersion_from_paired")]
    assert 'g["X_held"]' in ri and 'g["X_stop"]' not in ri
    hm = src[src.index("def heldout_metric"):
             src.index("def improvement")]
    assert 'g["X_held"]' in hm and "X_stop" not in hm


def test_kill_7_missing_checkpoint_refuses(world):
    d, out = world
    sp = next((out / "intervention").glob("*_summary.json"))
    doc = json.loads(sp.read_text())
    doc["checkpoint_lineage"].pop("pre_stop")
    doc.pop("record_sha256")
    doc["record_sha256"] = m4._self_sha(doc, "record_sha256")
    sp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.chmod(sp, 0o600)
    _repair_report(out)
    with pytest.raises(SystemExit,
                       match="four declared checkpoints"):
        rn.verify_run_v5(d, out, ("DEVELOPMENT",))


def test_kill_8_role_bytes_disjoint(monkeypatch):
    shas = gb.assert_role_disjointness("sine", "white", 0)
    assert len(set(shas.values())) == 3
    real = gb._seed
    with monkeypatch.context() as mp:
        mp.setattr(gb, "_seed",
                   lambda *parts: real(
                       *[x for x in parts if x not in gb.ROLES]))
        with pytest.raises(SystemExit, match="NOT disjoint"):
            gb.assert_role_disjointness("sine", "white", 0)


def test_kill_9_confirmation_fixed_and_dispersion_floor():
    """Kill 9: CONFIRMATION is fixed at 48 BEFORE calibration;
    a two-generator dispersion estimate refuses."""
    d5 = pv.load_design_v5()
    assert d5["populations_v5"][
        "CONFIRMATION_reserved_per_cell"] == 48
    assert "fixed" in d5["populations_v5"][
        "confirmation_boundary"] or "48 fixed" in \
        d5["populations_v5"]["confirmation_boundary"]
    with pytest.raises(SystemExit, match="at least 3"):
        pv.dispersion_from_paired({"g0": 1.0, "g1": 2.0})
    disp = pv.dispersion_from_paired(
        {f"g{i}": float(i % 5) for i in range(16)})
    assert disp["sd_ucb95"] > disp["sd"]


def test_kill_10_ladder_never_advances_without_execution():
    rows = [{"param_count": 100, "nuisance": [0.0],
             "checkpoint_loss": 0.5, "task_updates": 100,
             "compressed_len": 500, "spectral_rank": 8,
             "prune_fraction": 0.1, "stop_traj_slope": -0.01,
             "fail_batch": 3}] * 6
    r = pv.ladder_compare(rows, ["g0", "g0", "g1", "g1",
                                 "g2", "g2"])
    assert r["status"] == "M4_LADDER_POPULATION_INSUFFICIENT"
    groups = [f"g{i}" for i in range(6) for _ in (0, 1)]
    rows12 = []
    for i in range(12):
        rows12.append({"param_count": 100 + i,
                       "nuisance": [float(i % 2)],
                       "checkpoint_loss": 0.5, "task_updates":
                       100, "compressed_len": 500,
                       "spectral_rank": 8,
                       "prune_fraction": 0.1,
                       "stop_traj_slope": -0.01,
                       "fail_batch": (i % 5) + 1})
    r2 = pv.ladder_compare(rows12, groups)
    assert r2["status"] == "EXECUTED"
    assert "m2_vs_m1_t" in r2 and "integrated_brier" in r2


def test_kill_11_only_cap_censors():
    """Kill 11: run_intervention outcomes are ACQUISITION/
    RETENTION/CAP only; resource stops never appear as arm
    outcomes (they refuse at unit boundaries), so nothing but
    the cap can censor."""
    src = (REPO / "tools/m4_v5_protocol.py").read_text()
    ri = src[src.index("def run_intervention"):
             src.index("def dispersion_from_paired")]
    for tok in ("WALL_STOP", "RSS_STOP", "STOP_REQUESTED"):
        assert tok not in ri
    assert '"CAP_REACHED"' in ri
    d5 = pv.load_design_v5()
    ep = d5["estimands"]["primary_endpoint"]
    assert "INCOMPLETE scientific units" in ep
    assert "never survival censoring" in ep
    assert "Gehan" in ep and "removed" in ep


def test_kill_12_durable_resume_not_restart(world, tmp_path,
                                            monkeypatch):
    """Kill 12: an interrupted arm resumes from its durable
    per-batch predecessor — the completed history equals an
    uninterrupted run; a partial log WITHOUT its state is
    UNCERTAIN and refuses."""
    d, _ = world
    out = tmp_path / "resume"
    (out / "intervention").mkdir(parents=True)
    os.chmod(out, 0o700)
    g = gb.generate("DEVELOPMENT", "identity", "clean", 0)
    kind = pv.task_kind("identity")
    tape = pv.association_tape(d["design_sha256"], g, 16, 0)
    ck = pv.build_checkpoints(g, 16, 0)
    u = rn.intervention_units_v5(d, "DEVELOPMENT")[0]
    ckpt = dict(ck["checkpoints"]["calibration_stop"])
    ckpt["_name"] = "calibration_stop"
    acct = {"optimization_updates": 0, "evaluations": 0,
            "descriptor_seconds": 0.0, "descriptor_evals": 0}

    class Boom(Exception):
        pass
    calls = {"n": 0}
    real_ri = pv.run_intervention

    def boom_ri(g_, tape_, ck_, kind_, on_batch=None,
                start=None):
        def cb(rec, pp, st, margin, endpoint):
            on_batch(rec, pp, st, margin, endpoint)
            calls["n"] += 1
            if calls["n"] == 1:
                raise Boom()
        return real_ri(g_, tape_, ck_, kind_, on_batch=cb,
                       start=start)
    with monkeypatch.context() as mp:
        mp.setattr(pv, "run_intervention", boom_ri)
        with pytest.raises(Boom):
            rn._run_arm_durable(d, u, g, tape, ckpt, kind, out,
                                acct)
    lp = (out / "intervention"
          / f"{rn._safe(u['unit_id'])}__calibration_stop.jsonl")
    sp = rn._state_path(out, u["unit_id"], "calibration_stop")
    assert lp.exists() and sp.exists()
    resumed = rn._run_arm_durable(d, u, g, tape, ckpt, kind,
                                  out, acct)
    fresh = pv.run_intervention(g, tape,
                                ck["checkpoints"][
                                    "calibration_stop"], kind)
    assert resumed["restricted_endpoint"] == \
        fresh["restricted_endpoint"]
    assert resumed["stopping_cause"] == fresh["stopping_cause"]
    assert not sp.exists()          # state removed on completion
    lines = lp.read_text().splitlines()
    assert len(lines) == len(fresh["records"])
    for i, line in enumerate(lines):
        rec = json.loads(line)
        claim = {k: rec[k] for k in rec
                 if k not in ("record_sha256", "tape_digest")}
        assert claim == fresh["records"][i]
    # partial log WITHOUT state -> UNCERTAIN
    out2 = tmp_path / "orphan"
    (out2 / "intervention").mkdir(parents=True)
    lp2 = (out2 / "intervention"
           / f"{rn._safe(u['unit_id'])}__calibration_stop.jsonl")
    lp2.write_text(lines[0] + "\n")
    with pytest.raises(SystemExit, match="UNCERTAIN"):
        rn._run_arm_durable(d, u, g, tape, ckpt, kind, out2,
                            acct)


def test_kill_13_group_writable_refuses(world):
    d, out = world
    victim = next((out / "screen").glob("*.json"))
    os.chmod(victim, 0o664)
    with pytest.raises(SystemExit, match="private 0600"):
        rn.verify_run_v5(d, out, ("DEVELOPMENT",))


def test_kill_14_lookalike_and_unknown_field_refuse(world):
    d, out = world
    (out / "RUN_REPORT_fake.json").write_text("{}")
    os.chmod(out / "RUN_REPORT_fake.json", 0o600)
    with pytest.raises(SystemExit, match="lookalike"):
        rn.verify_run_v5(d, out, ("DEVELOPMENT",))
    os.unlink(out / "RUN_REPORT_fake.json")
    p = out / "RUN_REPORT.json"
    rep = json.loads(p.read_text())
    rep["extra_field"] = "x"
    rep.pop("record_sha256")
    rep["record_sha256"] = m4._self_sha(rep, "record_sha256")
    p.write_text(json.dumps(rep, indent=1, sort_keys=True))
    os.chmod(p, 0o600)
    with pytest.raises(SystemExit, match="exact schema"):
        rn.verify_run_v5(d, out, ("DEVELOPMENT",))


def test_kill_15_execute_cannot_succeed_unverified(tmp_path,
                                                   monkeypatch):
    d = _small_v5()
    with monkeypatch.context() as mp:
        mp.setattr(rn, "verify_run_v5",
                   lambda *a, **k: {"verified": False})
        with pytest.raises(SystemExit,
                           match="did not verify"):
            rn.execute_v5(d, tmp_path / "unv", ("DEVELOPMENT",))


def test_kill_16_pre_outcome_boundary(tmp_path, monkeypatch):
    """Kill 16: the SEALED v5 refuses to score before its bytes
    are committed and pushed; fixtures pass through."""
    sealed = pv.load_design_v5()
    clone = tmp_path / "sealed_clone.json"
    clone.write_text(json.dumps(sealed, indent=1))
    with monkeypatch.context() as mp:
        mp.setattr(pv, "DESIGN_PATH_V5", clone)
        mp.setattr(rn.pv, "DESIGN_PATH_V5", clone)
        with pytest.raises(SystemExit,
                           match="PRE_OUTCOME_BOUNDARY"):
            rn.execute_v5(sealed, tmp_path / "b", ("DEVELOPMENT",))


def test_kill_17_confirmation_structurally_closed():
    with pytest.raises(SystemExit, match="RESERVED"):
        gb.generate("CONFIRMATION", "sine", "white", 0)
    d = _small_v5()
    with pytest.raises(SystemExit,
                       match="CONFIRMATION is reserved"):
        rn.execute_v5(d, Path("/nonexistent"),
                      ("CONFIRMATION",))
    slots = rn.reserved_confirmation_slots(d)
    assert all(s.startswith("RESERVED::CONFIRMATION::")
               for s in slots)


def test_kill_18_edited_aggregate_refuses(world):
    d, out = world
    p = out / "LEARNABILITY_TABLE_DEVELOPMENT.json"
    tab = json.loads(p.read_text())
    k = next(iter(tab["cells"]))
    cur = tab["cells"][k]["outcome"]
    tab["cells"][k]["outcome"] = (
        "OPTIMIZATION_LIMITED"
        if cur == "LEARNABLE_UNDER_FROZEN_BUDGET"
        else "LEARNABLE_UNDER_FROZEN_BUDGET")
    tab.pop("record_sha256")
    tab["record_sha256"] = m4._self_sha(tab, "record_sha256")
    p.write_text(json.dumps(tab, indent=1, sort_keys=True))
    os.chmod(p, 0o600)
    _repair_report(out)
    with pytest.raises(SystemExit, match="does not re-derive"):
        rn.verify_run_v5(d, out, ("DEVELOPMENT",))
