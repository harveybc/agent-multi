"""T0-T1 §9 adversarial battery: twelve mandatory cases + directed
mutations against the REAL bank, lab and adjudicator."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
PREP = Path(os.environ.get(
    "B4_T1_PREPROCESSOR_ROOT",
    Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))
# spec-load by file: agent-multi has its OWN `app` package, so a
# name import would collide inside the full suite.
import importlib.util as _ilu  # noqa: E402
_co_spec = _ilu.spec_from_file_location(
    "t1_causal_operators", PREP / "app/causal_operators.py")
co = _ilu.module_from_spec(_co_spec)
_co_spec.loader.exec_module(co)
import t1_adjudicator as adj  # noqa: E402
import t1_known_truth_bank as bank  # noqa: E402

MEAS = Path.home() / (".local/share/agent-multi/"
                     "t1_measurements_v4_20260906.json")
BANK = Path.home() / ".local/share/agent-multi/t1_bank_v3_20260906"
NPZ = Path.home() / ".local/share/agent-multi/t1_npz_v4_20260906"
DESIGN = (REPO / "docs/audits/evidence/"
          "T1_LAB_DESIGN_V4_2026_09_06.json")
PUB = Path.home() / (".local/share/agent-multi/"
                     "t1_adjudication_v4_20260906.json")
MANIFEST = Path.home() / (".local/share/agent-multi/"
                          "t1_measurements_v4_20260906_MANIFEST"
                          ".json")
_meas_present = pytest.mark.skipif(not MEAS.is_file(),
                                   reason="measurements absent")


# 1. centered filter presented as causal -> refuses
def test_a1_centered_filter_cannot_pose_as_causal():
    spec = {"schema": co.SCHEMA_VERSION,
            "operator_id": "sneaky_smooth", "kind":
            "centered_mean_oracle", "version": "1",
            "params": {"window": 5}, "columns": ["a"],
            "fit_role": "train", "lookback": 5,
            "availability_rule": "bar_close"}
    # C15: an understated causal lookback dies FIRST
    with pytest.raises(co.CausalOperatorError,
                       match="cannot be understated"):
        co.validate_spec(spec)
    # honest future-reach declaration still cannot look causal
    spec2 = dict(spec, lookback=co.LOOKBACK_NON_CAUSAL)
    with pytest.raises(co.CausalOperatorError,
                       match="never look causal"):
        co.validate_spec(spec2)


# 2. fit contaminated with validation/test -> refuses
def test_a2_contaminated_fit_role_refuses():
    spec = {"schema": co.SCHEMA_VERSION, "operator_id": "e",
            "kind": "ewma", "version": "1",
            "params": {"alpha": 0.3}, "columns": ["a"],
            "fit_role": "train", "lookback": -1,
            "availability_rule": "bar_close"}
    x = np.random.default_rng(0).normal(0, 1, (100, 1))
    with pytest.raises(co.CausalOperatorError, match="licenses"):
        co.fit(spec, x, ["a"], "validation",
               co.make_train_contract(x, 100))
    bad = dict(spec, fit_role="validation")
    with pytest.raises(co.CausalOperatorError,
                       match="training role"):
        co.validate_spec(bad)


# 3. reconstruction better + extreme destroyed -> LAB_REJECTED
def _pv(gain=5.0, util=0.01, retention=0.9, resid=0.0,
        tail=1.0):
    def role(g):
        return {"mse_observed": 1.0, "mse_denoised": 0.5,
                "snr_gain_db": g, "extreme_retention": retention,
                "tail_ratio": tail}
    return {"variable": "v0", "true_additive_snr_db": 10,
            "true_total_error_snr_db": 10,
            "snr_estimator_std_error": 0.1, "delay_bars": 1,
            "by_role": {"train": role(gain), "validation":
                        role(gain), "score": role(gain)},
            "assays_score_fit_train": {
                h: {"X": 0.5, "D": 0.5 * (1 + util),
                    "XDR": 0.5 + resid,
                    "width_control_X_nuisance": 0.5,
                    "residual_incremental_r2": resid}
                for h in ("h1", "h5")}}


def _fake_measurements(gain=5.0, util=0.01, retention=0.9,
                       resid=0.0, tail=1.0):
    recs = []
    for seed in (11, 12, 13):
        recs.append({
            "schema": "agent_multi.t1_measurement.v2",
            "unit_id": f"bumps__white__snr10__hom__seed{seed}",
            "operator": "ewma",
            "family": "bumps", "perturbation": "white",
            "declared_snr_db": 10, "heterogeneous": False,
            "seed": seed, "causal": True, "oracle_only": False,
            "source_digests": {}, "status": "MEASURED",
            "cpu_wall_seconds": 0.1,
            "per_variable": [_pv(gain, util, retention, resid,
                                 tail)],
            "denoised_residual_npz_sha256": "e" * 64,
            "output_column_expansion": {"X": 1, "D": 1, "XDR": 3},
            "peak_rss_bytes": 1,
            "artifact_sha256": "a" * 64,
            "spec_sha256": "b" * 64})
    return {"records": recs, "design_sha256": "d" * 64}


FAKE_DESIGN = {"expected_operators_exact": ["ewma"]}
FAKE_UIDS = [f"bumps__white__snr10__hom__seed{s}"
             for s in (11, 12, 13)]
FAKE_INV = {"schema": "agent_multi.t1_bank_inventory.v2",
            "predeclared_cells": 1, "seeds": [11, 12, 13],
            "units_total": 3, "unit_ids": list(FAKE_UIDS),
            "units": {u: {"unit_json_sha256": "c" * 64,
                          "arrays": {n: "d" * 64 for n in (
                              "clean_signal", "additive_noise",
                              "observed_signal",
                              "metric_support")}}
                      for u in FAKE_UIDS},
            "design_rule": "fixture"}


def _adjudicate(m):
    return adj.adjudicate(FAKE_DESIGN, FAKE_INV, m)


def test_a3_reconstruction_up_extreme_destroyed_rejected():
    out = _adjudicate(_fake_measurements(
        gain=5.0, util=0.0, retention=0.1))
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] == "LAB_REJECTED"
    assert "extreme" in v["reason"]


# 4. residual with target utility -> demoted to TRANSFORMATION
def test_a4_informative_residual_demoted():
    out = _adjudicate(_fake_measurements(
        gain=5.0, util=0.01, retention=0.9, resid=0.2))
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] == "LAB_CALIBRATED"
    assert "TRANSFORMATION" in v["reason"]
    assert v["residual_informative"] is True


# 5. true SNR declared on natural data -> the bank has no such unit
def test_a5_no_true_snr_outside_known_truth():
    src = (REPO / "tools/t1_known_truth_bank.py").read_text()
    assert "true_additive_snr_db" in src
    lab = (REPO / "tools/t1_lab_run.py").read_text()
    # every true-* fact the lab reports flows FROM the bank unit
    # record (known clean+noise); the lab never mints one.
    assert 'rec["true_additive_snr_db"]' in lab
    assert 'rec["noise_std_per_var"]' in lab
    assert "def true_snr" not in lab


# 6. positive aggregate with a failed/absent unit -> refuses
def test_a6_incomplete_population_refuses():
    """C6: the population derives from DESIGN+INVENTORY — a missing
    design-required record refuses; producer counts grant nothing."""
    m = _fake_measurements()
    m["records"] = m["records"][:2]
    with pytest.raises(SystemExit, match="population incomplete"):
        _adjudicate(m)
    m2 = _fake_measurements()
    m2["records"].append(dict(
        m2["records"][0],
        unit_id="bumps__white__snr10__hom__seed99", seed=99))
    with pytest.raises(SystemExit, match="foreign record"):
        _adjudicate(m2)
    # C20: a non-canonical identity dies even earlier
    m3 = _fake_measurements()
    m3["records"].append(dict(m3["records"][0],
                              unit_id="foreign_unit"))
    with pytest.raises(SystemExit, match="canonical shape"):
        _adjudicate(m3)


# 7. inflated support from windows of one process -> unit rule
def test_a7_windows_are_never_replicas():
    m = _fake_measurements()
    m["records"] = m["records"][:1]
    inv1 = copy.deepcopy(FAKE_INV)
    inv1["unit_ids"] = [FAKE_UIDS[0]]
    inv1["units"] = {FAKE_UIDS[0]: FAKE_INV["units"][FAKE_UIDS[0]]}
    inv1["units_total"] = 1
    out = adj.adjudicate(FAKE_DESIGN, inv1, m)
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] == "INCONCLUSIVE"
    assert "seeds" in v["reason"]


# 8. re-digested artifact after mutation -> refuses
def test_a8_redigested_artifact_refuses(tmp_path):
    spec = {"schema": co.SCHEMA_VERSION, "operator_id": "e",
            "kind": "ewma", "version": "1",
            "params": {"alpha": 0.3}, "columns": ["a"],
            "fit_role": "train", "lookback": -1,
            "availability_rule": "bar_close"}
    x = np.random.default_rng(0).normal(0, 1, (100, 1))
    art = co.fit(spec, x, ["a"], "train",
                 co.make_train_contract(x, 100))
    doc = copy.deepcopy(art)
    doc["spec"]["params"]["alpha"] = 0.9
    import hashlib as h
    doc["artifact_sha256"] = h.sha256(
        json.dumps(doc, sort_keys=True).encode()).hexdigest()
    with pytest.raises(co.CausalOperatorError,
                       match="does not re-derive"):
        co.verify_artifact(doc)


# 9. producer self-declaring LAB_CALIBRATED -> ignored
def test_a9_producer_verdict_ignored():
    # C20: a smuggled producer verdict field now refuses at the
    # exact-schema boundary
    m = _fake_measurements(gain=-3.0, util=-0.5, retention=0.9)
    for r in m["records"]:
        r["verdict"] = "LAB_CALIBRATED"      # forged producer field
    with pytest.raises(SystemExit, match="exact schema"):
        _adjudicate(m)
    # and without the smuggled field, bad facts still adjudicate
    # to rejection on their own
    m2 = _fake_measurements(gain=-3.0, util=-0.5, retention=0.9)
    out = _adjudicate(m2)
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] == "LAB_REJECTED"


# 10. two transforms changing column order -> refuses
def test_a10_column_order_change_refuses():
    spec = {"schema": co.SCHEMA_VERSION, "operator_id": "e",
            "kind": "ewma", "version": "1",
            "params": {"alpha": 0.3}, "columns": ["a", "b"],
            "fit_role": "train", "lookback": -1,
            "availability_rule": "bar_close"}
    x = np.random.default_rng(0).normal(0, 1, (50, 2))
    art = co.fit(spec, x, ["a", "b"], "train",
                 co.make_train_contract(x, 50))
    with pytest.raises(co.CausalOperatorError, match="rejected"):
        co.transform_batch(art, x, ["b", "a"],
                           co.make_bar_close_contract(50))


# 11. batch matching only the last value, not the full prefix
def test_a11_full_prefix_parity_not_just_last_value():
    spec = {"schema": co.SCHEMA_VERSION, "operator_id": "e",
            "kind": "trailing_mean", "version": "1",
            "params": {"window": 4}, "columns": ["a"],
            "fit_role": "train", "lookback": 3,
            "availability_rule": "bar_close"}
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (60, 1))
    art = co.fit(spec, x, ["a"], "train",
                 co.make_train_contract(x, 60))
    tc = co.make_bar_close_contract(60)
    batch = co.transform_batch(art, x, ["a"], tc)
    state = co.init_state(art, "synthetic_bar_close", 1.0)
    frag = []
    for i in range(60):
        out, state = co.transform_incremental(
            art, state, x[i], ["a"],
            co.make_bar_close_contract(1, float(i)))
        frag.append(out)
    frag = np.concatenate(frag)
    assert frag.tobytes() == batch.tobytes()   # EVERY byte, not [-1]


# 12. accidental CUDA/network access -> guard present and typed
def test_a12_cpu_only_guard():
    lab = (REPO / "tools/t1_lab_run.py").read_text()
    assert "CUDA visible — T1 is CPU-only" in lab
    bank_src = (REPO / "tools/t1_known_truth_bank.py").read_text()
    for tok in ("import requests", "import urllib", "http://",
                "https://", "urlopen"):
        assert tok not in bank_src


# directed mutations on the REAL adjudication
@_meas_present
def test_m1_real_population_rederives():
    design = json.loads(DESIGN.read_text())
    inv = json.loads((BANK / "BANK_INVENTORY.json").read_text())
    m = json.loads(MEAS.read_text())
    out = adj.adjudicate(design, inv, m)
    assert out["population"]["records"] == 1152
    assert out["verdict_counts"]["NON_CAUSAL_ORACLE_ONLY"] == 64


@_meas_present
def test_m2_mutated_record_changes_verdict_derivation():
    design = json.loads(DESIGN.read_text())
    inv = json.loads((BANK / "BANK_INVENTORY.json").read_text())
    m = json.loads(MEAS.read_text())
    out1 = adj.adjudicate(design, inv,
                          json.loads(json.dumps(m)))
    for r in m["records"]:
        if r["operator"] == "trailing_median" and \
                r["status"] == "MEASURED":
            for pv in r["per_variable"]:
                for role in pv["by_role"].values():
                    if isinstance(role, dict) and \
                            "extreme_retention" in role:
                        role["extreme_retention"] = 1.0
                for h in ("h1", "h5"):
                    a = pv["assays_score_fit_train"][h]
                    if isinstance(a["X"], float):
                        a["D"] = a["X"] + 0.5
    out2 = adj.adjudicate(design, inv, m)
    assert out1["verdict_counts"] != out2["verdict_counts"]


@_meas_present
def test_m4_rederivation_catches_forged_npz_gates():
    """C6: a published gate that disagrees with the re-derived
    array value refuses."""
    design = json.loads(DESIGN.read_text())
    inv = json.loads((BANK / "BANK_INVENTORY.json").read_text())
    m = json.loads(MEAS.read_text())
    victim = next(r for r in m["records"]
                  if r["status"] == "MEASURED"
                  and r["operator"] == "ewma")
    victim["per_variable"][0]["by_role"]["score"][
        "snr_gain_db"] = 99.9
    with pytest.raises(SystemExit, match="re-derived"):
        adj.rederive_gates(victim, BANK, NPZ)


def test_c7_nan_gates_never_authorize():
    """C7/C20 POST: a raw NaN gate refuses TYPED at the consuming
    schema boundary; the lab's own typed-null shape adjudicates
    INCONCLUSIVE — neither path can calibrate."""
    m = _fake_measurements()
    for r in m["records"]:
        for pv in r["per_variable"]:
            pv["by_role"]["score"]["snr_gain_db"] = float("nan")
    with pytest.raises(SystemExit, match="non-finite"):
        _adjudicate(m)
    m2 = _fake_measurements()
    for r in m2["records"]:
        for pv in r["per_variable"]:
            pv["by_role"]["score"] = {"null": True,
                                      "reason": "support<20"}
    out = _adjudicate(m2)
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] in ("INCONCLUSIVE", "LAB_REJECTED")


def test_c10_one_material_failure_cannot_hide():
    """C10 POST: two good seeds cannot mask one destroyed seed."""
    m = _fake_measurements(gain=5.0, util=0.01, retention=0.9)
    bad = m["records"][2]["per_variable"][0]
    for role in bad["by_role"].values():
        role["extreme_retention"] = -0.5     # inversion in ONE seed
    out = _adjudicate(m)
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] == "LAB_REJECTED"
    assert "cannot hide" in v["reason"] or "material" in v["reason"]


def test_c5_delayed_units_bind_observation_identity(tmp_path):
    """C5 POST: delayed observed == rolled(clean+additive) EXACTLY
    on support; the on-support identity is asserted at
    materialization."""
    rec = bank.materialize_unit(
        {"family": "heavisine", "perturbation": "delayed",
         "snr_db": 10, "heterogeneous": False}, 11, tmp_path)
    u = tmp_path / rec["unit_id"]
    clean = np.load(u / "clean_signal.npy")
    add = np.load(u / "additive_noise.npy")
    obs = np.load(u / "observed_signal.npy")
    sup = np.load(u / "metric_support.npy")
    k = rec["distortion"]["delay_bars"]
    assert np.array_equal(obs[:, k:], (clean + add)[:, :-k])
    assert not sup[:, :k].any() and sup[:, k:].all()
    assert rec["true_total_observation_error_snr_db"][0] < \
        rec["true_additive_snr_db"][0]


def test_m3_bank_unit_digests_rederive(tmp_path):
    cell = {"family": "sine", "perturbation": "white",
            "snr_db": 10, "heterogeneous": False}
    rec = bank.materialize_unit(cell, 99, tmp_path)
    clean = np.load(tmp_path / rec["unit_id"] / "clean_signal.npy")
    assert hashlib.sha256(
        np.ascontiguousarray(clean).tobytes()).hexdigest() == \
        rec["digests"]["clean_signal"]
    rec2 = bank.materialize_unit(cell, 99, tmp_path / "again")
    assert rec2["digests"] == rec["digests"]   # seeded determinism


# ============ C13-C20 acceptance battery (order §4) ============

_v3_present = pytest.mark.skipif(
    not MEAS.is_file() or not BANK.is_dir(),
    reason="v3 artifacts absent")


def _v3_meas():
    return json.loads(MEAS.read_text())


@_v3_present
def test_c16_three_record_forgery_dies():
    """§4: the exact PRE counterexample — three ewma am|white|snr5
    records with favorable utility/tail/extreme producer fields —
    now dies record by record in the full re-derivation."""
    m = _v3_meas()
    forged = [r for r in m["records"]
              if r["operator"] == "ewma"
              and r["unit_id"].startswith("am__white__snr5__")]
    assert len(forged) == 3
    for r in forged:
        for pv in r["per_variable"]:
            for h in pv["assays_score_fit_train"].values():
                h["D"] = h["X"] + 0.05
                h["XDR"] = h["width_control_X_nuisance"] + 0.05
                h["residual_incremental_r2"] = 0.05
            sc = pv["by_role"]["score"]
            sc["extreme_retention"] = 0.99
            sc["tail_ratio"] = 0.5
        with pytest.raises(SystemExit, match="differs from the"):
            adj.rederive_all_facts(r, BANK, NPZ)


@_v3_present
@pytest.mark.parametrize("path,field", [
    ("role", "mse_observed"), ("role", "mse_denoised"),
    ("role", "snr_gain_db"), ("role", "extreme_retention"),
    ("role", "tail_ratio"),
    ("assay", "X"), ("assay", "D"), ("assay", "XDR"),
    ("assay", "width_control_X_nuisance"),
    ("assay", "residual_incremental_r2"),
    ("pv", "snr_estimator_std_error"), ("pv", "delay_bars")])
def test_c16_every_metric_mutation_caught(path, field):
    """§4.6: EVERY decision-bearing metric mutation is caught by
    the independent re-derivation, one field at a time."""
    m = _v3_meas()
    r = next(x for x in m["records"]
             if x["status"] == "MEASURED"
             and x["operator"] == "trailing_mean")
    pv = r["per_variable"][0]
    if path == "role":
        pv["by_role"]["score"][field] = \
            float(pv["by_role"]["score"][field]) + 0.31
    elif path == "assay":
        pv["assays_score_fit_train"]["h1"][field] = 0.777
    elif field == "delay_bars":
        pv["delay_bars"] = int(pv["delay_bars"]) + 3
    else:
        pv[field] = float(pv[field]) + 0.5 \
            if not isinstance(pv[field], dict) else 0.5
    with pytest.raises(SystemExit,
                       match="differs from|delay"):
        adj.rederive_all_facts(r, BANK, NPZ)


@_v3_present
def test_c17_physical_replacement_dies(tmp_path):
    """§4.5: replacing unit bytes under an unchanged sealed
    inventory now refuses — every physical byte is bound."""
    import shutil
    inv = json.loads((BANK / "BANK_INVENTORY.json").read_text())
    uid = "am__white__snr5__hom__seed11"
    (tmp_path / uid).mkdir()
    for f in (BANK / uid).iterdir():
        shutil.copyfile(f, tmp_path / uid / f.name)
    adj.check_inventory(inv)
    adj.verify_unit_bytes(inv, tmp_path, uid)     # honest copy ok
    clean = np.load(tmp_path / uid / "clean_signal.npy")
    np.save(tmp_path / uid / "clean_signal.npy", clean * 0.5)
    with pytest.raises(SystemExit, match="differ from the"):
        adj.verify_unit_bytes(inv, tmp_path, uid)
    # metadata replacement dies too
    (tmp_path / uid / "clean_signal.npy").write_bytes(
        (BANK / uid / "clean_signal.npy").read_bytes())
    u = json.loads((tmp_path / uid / "UNIT.json").read_text())
    u["declared_snr_db"] = 99
    (tmp_path / uid / "UNIT.json").write_text(json.dumps(u))
    with pytest.raises(SystemExit, match="metadata bytes differ"):
        adj.verify_unit_bytes(inv, tmp_path, uid)


@_v3_present
def test_c18_c23_no_path_to_review_authority(tmp_path):
    """C18/C23: rewritten measurements refuse against the
    manifest; the honest pair is NON-AUTHORIZING (exit 3); there
    is NO consuming API for reviewer records; a forged
    quantitative publication refuses on complete equality."""
    import subprocess
    design_sha = hashlib.sha256(DESIGN.read_bytes()).hexdigest()
    base = [sys.executable,
            str(REPO / "tools/t1_independent_verifier.py"),
            "--design", str(DESIGN), "--design-sha", design_sha,
            "--bank-dir", str(BANK), "--npz-dir", str(NPZ),
            "--measurement-manifest", str(MANIFEST),
            "--published", str(PUB)]
    # (a) coherently rewritten measurements != manifest
    m = _v3_meas()
    m["records"][0]["cpu_wall_seconds"] = 9.9
    fake_m = tmp_path / "rewritten.json"
    fake_m.write_text(json.dumps(m))
    rc = subprocess.run(base + ["--measurements", str(fake_m)],
                        capture_output=True, text=True)
    assert rc.returncode not in (0, 3)
    assert "population manifest" in rc.stderr + rc.stdout
    # (b) the honest pair: strongest outcome is exit 3
    rc2 = subprocess.run(
        base + ["--measurements", str(MEAS)],
        capture_output=True, text=True)
    assert rc2.returncode == 3
    assert "SELF_CONSISTENT_ONLY_NOT_AUTHORIZING" in rc2.stdout
    assert "complete_publication_equality" in rc2.stdout
    # (c) NO consuming API: the flag itself does not exist
    rr = tmp_path / "rr.json"
    rr.write_text(json.dumps({
        "schema": "agent_multi.t1_reviewed_record.v1",
        "reviewer": "candidate-self-review",
        "measurements_sha256": adj._sha_file(MEAS),
        "publication_sha256": adj._sha_file(PUB)}))
    rc3 = subprocess.run(
        base + ["--measurements", str(MEAS),
                "--reviewed-record", str(rr)],
        capture_output=True, text=True)
    assert rc3.returncode == 2          # argparse: unknown flag
    # (d) forged quantitative field -> refused on COMPLETE
    # equality (the exact PRE probe)
    pub = json.loads(PUB.read_text())
    pub["verdicts"]["ewma::am|white|snr-5"][
        "snr_gain_db"]["median"] = 999
    forged = tmp_path / "pub999.json"
    forged.write_text(json.dumps(pub))
    rc4 = subprocess.run(
        [sys.executable,
         str(REPO / "tools/t1_independent_verifier.py"),
         "--design", str(DESIGN), "--design-sha", design_sha,
         "--bank-dir", str(BANK), "--npz-dir", str(NPZ),
         "--measurement-manifest", str(MANIFEST),
         "--published", str(forged),
         "--measurements", str(MEAS),
         "--submission-out", str(tmp_path / "sub.json")],
        capture_output=True, text=True)
    assert rc4.returncode not in (0, 3)
    blob = rc4.stderr + rc4.stdout
    assert "differs from the complete re-derivation" in blob
    assert "snr_gain_db" in blob and "median" in blob
    assert not (tmp_path / "sub.json").exists()


def test_c23_candidate_code_has_no_authority_tokens():
    """C23: structural assert — no candidate tool contains a
    branch capable of emitting a review-authority label."""
    for tool in ("t1_known_truth_bank.py", "t1_lab_run.py",
                 "t1_adjudicator.py",
                 "t1_independent_verifier.py"):
        text = (REPO / "tools" / tool).read_text()
        for token in ("REPRODUCED_UNDER_REVIEWED_IDENTITY",
                      "reviewed_record", "reviewed-record",
                      '"ACCEPTED"', '"AUTHORIZED"',
                      "REVIEW_APPROVED"):
            assert token not in text, (tool, token)


@_v3_present
def test_c21_identity_mutations_refuse_before_evidence(tmp_path):
    """C21: every executable-identity mutation refuses before any
    evidence is read — for each of the sealed digests."""
    design = json.loads(DESIGN.read_text())
    for field in ("t1_known_truth_bank_sha256",
                  "t1_lab_run_sha256",
                  "t1_adjudicator_sha256",
                  "t1_independent_verifier_sha256"):
        forged = json.loads(DESIGN.read_text())
        forged["code_identity"][field] = "0" * 64
        with pytest.raises(SystemExit,
                           match="differ from the sealed"):
            adj.verify_complete_code_identity(forged)
    with pytest.raises(SystemExit, match="causal operator"):
        adj.verify_complete_code_identity(
            design, operator_sha="0" * 64)
    # live subprocess: a design naming other bytes refuses rc!=0
    import subprocess
    forged = json.loads(DESIGN.read_text())
    forged["code_identity"]["t1_lab_run_sha256"] = "0" * 64
    fp = tmp_path / "design_forged.json"
    fp.write_text(json.dumps(forged, indent=1))
    rc = subprocess.run(
        [sys.executable,
         str(REPO / "tools/t1_independent_verifier.py"),
         "--design", str(fp),
         "--design-sha",
         hashlib.sha256(fp.read_bytes()).hexdigest(),
         "--bank-dir", str(BANK), "--npz-dir", str(NPZ),
         "--measurement-manifest", str(MANIFEST),
         "--published", str(PUB),
         "--measurements", str(MEAS)],
        capture_output=True, text=True)
    assert rc.returncode not in (0, 3)
    assert "differ from the sealed" in rc.stderr + rc.stdout


@_v3_present
@pytest.mark.parametrize("field,setter", [
    ("verdict", lambda v: v.update(
        verdict="LAB_REJECTED"
        if v["verdict"] == "LAB_CALIBRATED"
        else "LAB_CALIBRATED")),
    ("reason", lambda v: v.update(reason="all good")),
    ("snr median", lambda v: v["snr_gain_db"].update(median=999.0)),
    ("snr min", lambda v: v["snr_gain_db"].update(min=-1.0)),
    ("snr max", lambda v: v["snr_gain_db"].update(max=100.0)),
    ("tail median", lambda v: v["tail_ratio"].update(median=0.1)),
    ("retention", lambda v: v["extreme_retention"].update(
        median=0.99)),
    ("utility", lambda v: v["relative_utility_delta_D_vs_X"]
     .update(median=0.5)),
    ("residual dist", lambda v: v["residual_incremental_r2"]
     .update(median=0.5)),
    ("seeds", lambda v: v.update(seeds=99)),
    ("residual flag", lambda v: v.update(
        residual_informative=not v.get("residual_informative",
                                       False))),
    ("material failures", lambda v: v.update(
        material_failures=["smuggled clean bill"]))])
def test_c22_every_field_class_mutation_refuses(field, setter):
    """C22: mutation of every publication field class refuses on
    the complete comparison."""
    rederived = json.loads(PUB.read_text())
    published = json.loads(PUB.read_text())
    setter(published["verdicts"]["ewma::am|white|snr-5"])
    with pytest.raises(SystemExit,
                       match="complete re-derivation"):
        adj.require_publication_equality(rederived, published)


@_v3_present
def test_c22_population_metadata_and_counts_refuse():
    rederived = json.loads(PUB.read_text())
    for mut in (
            lambda p: p["verdict_counts"].update(
                LAB_CALIBRATED=181),
            lambda p: p.update(verdict_basis="vibes"),
            lambda p: p.update(material_failure_rule="x"),
            lambda p: p["population"].update(records=9),
            lambda p: p.update(design_sha256="f" * 64)):
        published = json.loads(PUB.read_text())
        mut(published)
        with pytest.raises(SystemExit,
                           match="complete re-derivation"):
            adj.require_publication_equality(rederived, published)
    # smuggled extra top-level field dies on the exact schema
    published = json.loads(PUB.read_text())
    published["endorsement"] = "shiny"
    with pytest.raises(SystemExit, match="exact schema"):
        adj.require_publication_equality(rederived, published)


@_v3_present
def test_mut_c22_complete_equality_is_the_guard(tmp_path):
    """Reverting to label/count comparison re-admits the 999
    forgery — proving C22 bites."""
    import importlib.util as ilu
    text = (REPO / "tools/t1_adjudicator.py").read_text()
    old = ("    if _canonical_bytes(rederived) == "
           "_canonical_bytes(published):")
    assert old in text
    mut_text = text.replace(
        old,
        "    if rederived['verdict_counts'] == "
        "published['verdict_counts']:")
    mp = tmp_path / "adj_mut_c22.py"
    mp.write_text(mut_text)
    spec = ilu.spec_from_file_location("adj_mut_c22", mp)
    mut = ilu.module_from_spec(spec)
    spec.loader.exec_module(mut)
    rederived = json.loads(PUB.read_text())
    published = json.loads(PUB.read_text())
    published["verdicts"]["ewma::am|white|snr-5"][
        "snr_gain_db"]["median"] = 999.0
    with pytest.raises(SystemExit):
        adj.require_publication_equality(rederived, published)
    mut.require_publication_equality(rederived, published)  # blind


@_v3_present
def test_mut_c21_identity_guard_bites(tmp_path):
    """Removing the startup identity comparison re-admits a
    foreign-bytes run — proving C21 bites."""
    import importlib.util as ilu
    text = (REPO / "tools/t1_adjudicator.py").read_text()
    old = "        if sealed[k] != live[k]:"
    assert old in text
    mp = tmp_path / "adj_mut_c21.py"
    mp.write_text(text.replace(
        old, "        if False and sealed[k] != live[k]:"))
    spec = ilu.spec_from_file_location("adj_mut_c21", mp)
    mut = ilu.module_from_spec(spec)
    spec.loader.exec_module(mut)
    # the mutant lives in tmp; its identity source stays the real
    # tools directory (only the comparison guard is mutated)
    mut.executed_code_identity = adj.executed_code_identity
    forged = json.loads(DESIGN.read_text())
    forged["code_identity"]["t1_lab_run_sha256"] = "0" * 64
    with pytest.raises(SystemExit):
        adj.verify_complete_code_identity(forged)
    mut.verify_complete_code_identity(forged)       # mutant blind


def test_c19_future_rows_cannot_reach_controls():
    """§4.9: the width control is INDEPENDENT of the observed
    series — mutating validation/score rows changes neither the
    nuisance channels nor any train-role feature of any arm."""
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "t1lab_test", REPO / "tools/t1_lab_run.py")
    lab = ilu.module_from_spec(spec)
    spec.loader.exec_module(lab)
    roles = {"train": (0, 1228), "validation": (1228, 1638),
             "score": (1638, 2048)}
    rng = np.random.default_rng(7)
    obs = rng.normal(0, 1, 2048)
    obs_mut = obs.copy()
    obs_mut[1228:] = 999.0            # every non-train row mutated
    n1 = lab.nuisance_channels(obs, roles, "u|op|v0")
    n2 = lab.nuisance_channels(obs_mut, roles, "u|op|v0")
    for a, b in zip(n1, n2):
        assert a.tobytes() == b.tobytes()
    # train-role features of every arm are untouched
    lo_t, hi_t = roles["train"]
    for series in ([obs], [obs, n1[0], n1[1]]):
        series_mut = [s.copy() for s in series]
        series_mut[0] = obs_mut
        Xt = lab._lag_matrix(series, lo_t, hi_t, 1)
        Xt_mut = lab._lag_matrix(series_mut, lo_t, hi_t, 1)
        assert Xt.tobytes() == Xt_mut.tobytes()
    # determinism: same identity -> same channels
    again = lab.nuisance_channels(obs, roles, "u|op|v0")
    assert again[0].tobytes() == n1[0].tobytes()
    # distinct identities -> distinct channels
    other = lab.nuisance_channels(obs, roles, "u|op|v1")
    assert other[0].tobytes() != n1[0].tobytes()


def test_c19_no_circular_shift_remains():
    lab = (REPO / "tools/t1_lab_run.py").read_text()
    assert "np.roll" not in lab
    assert "NUISANCE_SHIFTS" not in lab


def test_c20_smuggled_nested_fields_refuse():
    """§4: the PRE C20 probe — unknown/mistyped nested fields now
    refuse typed before grouping."""
    m = _fake_measurements()
    m["records"][0]["attacker_note"] = "smuggled"
    with pytest.raises(SystemExit, match="exact schema"):
        _adjudicate(m)
    m2 = _fake_measurements()
    m2["records"][0]["per_variable"][0]["extra_nested"] = {"x": 1}
    with pytest.raises(SystemExit, match="exact schema"):
        _adjudicate(m2)
    m3 = _fake_measurements()
    m3["records"][0]["per_variable"][0]["by_role"]["score"][
        "extreme_retention"] = True       # bool as number
    with pytest.raises(SystemExit, match="number|bool"):
        _adjudicate(m3)
    m4 = _fake_measurements()
    m4["records"][0]["status"] = "TOTALLY_FINE"
    with pytest.raises(SystemExit, match="invalid status"):
        _adjudicate(m4)
    m5 = _fake_measurements()
    m5["records"][0]["seed"] = 12         # inconsistent with uid
    with pytest.raises(SystemExit, match="inconsistent"):
        _adjudicate(m5)


def test_history_designs_immutable_and_superseded():
    """v1-v3 designs remain byte-intact; each generation names its
    predecessor's exact bytes as superseded history."""
    e = REPO / "docs/audits/evidence"
    v1 = e / "T1_LAB_DESIGN_2026_09_06.json"
    v2 = e / "T1_LAB_DESIGN_V2_2026_09_06.json"
    v3 = e / "T1_LAB_DESIGN_V3_2026_09_06.json"
    assert v1.is_file() and v2.is_file() and v3.is_file()
    d3 = json.loads(v3.read_text())
    assert d3["supersedes"]["design_v2_sha256"] == \
        hashlib.sha256(v2.read_bytes()).hexdigest()
    d4 = json.loads(DESIGN.read_text())
    assert d4["supersedes"]["design_v3_sha256"] == \
        hashlib.sha256(v3.read_bytes()).hexdigest()


@_v3_present
def test_mut_c16_rederivation_is_the_guard(tmp_path):
    """§4.13: removing the assay comparison re-admits the forgery
    — proving the C16 re-derivation is what kills it."""
    import importlib.util as ilu
    srcp = REPO / "tools/t1_adjudicator.py"
    text = srcp.read_text()
    old = '''                _rd_close(pub_h[name], got[name],
                          f"{record['unit_id']}/"
                          f"{record['operator']} h{h} {name}")'''
    assert old in text
    mp = tmp_path / "adj_mut_c16.py"
    mp.write_text(text.replace(old, "                pass"))
    spec = ilu.spec_from_file_location("adj_mut_c16", mp)
    mut = ilu.module_from_spec(spec)
    spec.loader.exec_module(mut)
    m = _v3_meas()
    r = next(x for x in m["records"]
             if x["status"] == "MEASURED"
             and x["operator"] == "ewma"
             and x["unit_id"].startswith("am__white__snr5__"))
    for pv in r["per_variable"]:
        for h in pv["assays_score_fit_train"].values():
            h["D"] = h["X"] + 0.05        # only the utility arm
    with pytest.raises(SystemExit):
        adj.rederive_all_facts(copy.deepcopy(r), BANK, NPZ)
    mut.rederive_all_facts(copy.deepcopy(r), BANK, NPZ)  # passes


@_v3_present
def test_mut_c17_byte_binding_is_the_guard(tmp_path):
    import importlib.util as ilu
    import shutil
    srcp = REPO / "tools/t1_adjudicator.py"
    text = srcp.read_text()
    old = '''        if _sha_file(ud / f"{name}.npy") != want:'''
    assert old in text
    mp = tmp_path / "adj_mut_c17.py"
    mp.write_text(text.replace(old, "        if False:"))
    spec = ilu.spec_from_file_location("adj_mut_c17", mp)
    mut = ilu.module_from_spec(spec)
    spec.loader.exec_module(mut)
    inv = json.loads((BANK / "BANK_INVENTORY.json").read_text())
    uid = "sine__white__snr10__hom__seed11"
    (tmp_path / uid).mkdir()
    for f in (BANK / uid).iterdir():
        shutil.copyfile(f, tmp_path / uid / f.name)
    clean = np.load(tmp_path / uid / "clean_signal.npy")
    np.save(tmp_path / uid / "clean_signal.npy", clean * 0.5)
    with pytest.raises(SystemExit):
        adj.verify_unit_bytes(inv, tmp_path, uid)
    mut.verify_unit_bytes(inv, tmp_path, uid)      # mutant blind
