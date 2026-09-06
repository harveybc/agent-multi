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
PREP = Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"
sys.path.insert(0, str(PREP))
from app import causal_operators as co  # noqa: E402
import t1_adjudicator as adj  # noqa: E402
import t1_known_truth_bank as bank  # noqa: E402

MEAS = Path.home() / (".local/share/agent-multi/"
                     "t1_measurements_v2_20260906.json")
BANK = Path.home() / ".local/share/agent-multi/t1_bank_v2_20260906"
NPZ = Path.home() / ".local/share/agent-multi/t1_npz_v2_20260906"
DESIGN = (REPO / "docs/audits/evidence/"
          "T1_LAB_DESIGN_V2_2026_09_06.json")
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
    with pytest.raises(co.CausalOperatorError,
                       match="never look causal"):
        co.validate_spec(spec)


# 2. fit contaminated with validation/test -> refuses
def test_a2_contaminated_fit_role_refuses():
    spec = {"schema": co.SCHEMA_VERSION, "operator_id": "e",
            "kind": "ewma", "version": "1",
            "params": {"alpha": 0.3}, "columns": ["a"],
            "fit_role": "train", "lookback": 1,
            "availability_rule": "bar_close"}
    x = np.random.default_rng(0).normal(0, 1, (100, 1))
    with pytest.raises(co.CausalOperatorError, match="licenses"):
        co.fit(spec, x, ["a"], "validation")
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
            "unit_id": f"u{seed}", "operator": "ewma",
            "family": "bumps", "perturbation": "white",
            "declared_snr_db": 10, "heterogeneous": False,
            "seed": seed, "causal": True, "oracle_only": False,
            "status": "MEASURED",
            "per_variable": [_pv(gain, util, retention, resid,
                                 tail)]})
    return {"records": recs, "design_sha256": "d" * 64}


FAKE_DESIGN = {"expected_operators_exact": ["ewma"]}
FAKE_INV = {"unit_ids": ["u11", "u12", "u13"]}


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
    m2["records"].append(dict(m2["records"][0],
                              unit_id="foreign_unit"))
    with pytest.raises(SystemExit, match="foreign record"):
        _adjudicate(m2)


# 7. inflated support from windows of one process -> unit rule
def test_a7_windows_are_never_replicas():
    m = _fake_measurements()
    m["records"] = m["records"][:1]
    out = adj.adjudicate(FAKE_DESIGN, {"unit_ids": ["u11"]}, m)
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] == "INCONCLUSIVE"
    assert "seeds" in v["reason"]


# 8. re-digested artifact after mutation -> refuses
def test_a8_redigested_artifact_refuses(tmp_path):
    spec = {"schema": co.SCHEMA_VERSION, "operator_id": "e",
            "kind": "ewma", "version": "1",
            "params": {"alpha": 0.3}, "columns": ["a"],
            "fit_role": "train", "lookback": 1,
            "availability_rule": "bar_close"}
    x = np.random.default_rng(0).normal(0, 1, (100, 1))
    art = co.fit(spec, x, ["a"], "train")
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
    m = _fake_measurements(gain=-3.0, util=-0.5, retention=0.9)
    for r in m["records"]:
        r["verdict"] = "LAB_CALIBRATED"      # forged producer field
    out = _adjudicate(m)
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] == "LAB_REJECTED"


# 10. two transforms changing column order -> refuses
def test_a10_column_order_change_refuses():
    spec = {"schema": co.SCHEMA_VERSION, "operator_id": "e",
            "kind": "ewma", "version": "1",
            "params": {"alpha": 0.3}, "columns": ["a", "b"],
            "fit_role": "train", "lookback": 1,
            "availability_rule": "bar_close"}
    x = np.random.default_rng(0).normal(0, 1, (50, 2))
    art = co.fit(spec, x, ["a", "b"], "train")
    with pytest.raises(co.CausalOperatorError, match="rejected"):
        co.transform_batch(art, x, ["b", "a"])


# 11. batch matching only the last value, not the full prefix
def test_a11_full_prefix_parity_not_just_last_value():
    spec = {"schema": co.SCHEMA_VERSION, "operator_id": "e",
            "kind": "trailing_mean", "version": "1",
            "params": {"window": 4}, "columns": ["a"],
            "fit_role": "train", "lookback": 4,
            "availability_rule": "bar_close"}
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (60, 1))
    art = co.fit(spec, x, ["a"], "train")
    tc = co.make_bar_close_contract(60)
    batch = co.transform_batch(art, x, ["a"], tc)
    state = co.init_state(art)
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
    """C7 POST: NaN in every gate now REFUSES the gate and the
    regime becomes INCONCLUSIVE, never calibrated."""
    m = _fake_measurements()
    for r in m["records"]:
        for pv in r["per_variable"]:
            pv["by_role"]["score"]["snr_gain_db"] = float("nan")
    out = _adjudicate(m)
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] == "INCONCLUSIVE"
    assert "finite" in v["reason"]


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
