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

MEAS = Path.home() / ".local/share/agent-multi/t1_measurements_20260906.json"
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
def _fake_measurements(gain, util, retention, resid=0.0,
                       tail=1.0):
    recs = []
    for seed in (11, 12, 13):
        recs.append({
            "unit_id": f"u{seed}", "operator": "ewma",
            "family": "bumps", "perturbation": "white",
            "declared_snr_db": 10, "heterogeneous": False,
            "seed": seed, "causal": True, "oracle_only": False,
            "status": "MEASURED",
            "per_variable": [{
                "variable": "v0", "true_snr_db": 10,
                "snr_estimator_std_error": 0.1,
                "mse_observed": 1.0, "mse_denoised": 0.5,
                "snr_gain_db": gain, "delay_bars": 1,
                "extreme_retention": retention,
                "tail_ratio": tail,
                "ljung_box_p_residual_train": 0.5,
                "assays": {h: {"X": 0.5, "D": 0.5 * (1 + util),
                               "XDR": 0.5 + resid,
                               "capacity_control_XXX": 0.5,
                               "residual_incremental_r2": resid,
                               "persistence_obs_mse": 1.0}
                           for h in ("h1", "h5")}}]})
    return {"records": recs, "expected_units": 3,
            "expected_operators": 1}


def test_a3_reconstruction_up_extreme_destroyed_rejected():
    out = adj.adjudicate(_fake_measurements(
        gain=5.0, util=0.0, retention=0.1))
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] == "LAB_REJECTED"
    assert "extreme" in v["reason"]


# 4. residual with target utility -> demoted to TRANSFORMATION
def test_a4_informative_residual_demoted():
    out = adj.adjudicate(_fake_measurements(
        gain=5.0, util=0.01, retention=0.9, resid=0.2))
    v = list(out["verdicts"].values())[0]
    assert v["verdict"] == "LAB_CALIBRATED"
    assert "TRANSFORMATION" in v["reason"]
    assert v["residual_informative"] is True


# 5. true SNR declared on natural data -> the bank has no such unit
def test_a5_no_true_snr_outside_known_truth():
    src = (REPO / "tools/t1_known_truth_bank.py").read_text()
    assert "true_realized_snr_db" in src
    lab = (REPO / "tools/t1_lab_run.py").read_text()
    # every true-* fact the lab reports flows FROM the bank unit
    # record (known clean+noise); the lab never mints one.
    assert 'rec["true_realized_snr_db"]' in lab
    assert 'rec["noise_std_per_var"]' in lab
    assert "def true_snr" not in lab


# 6. positive aggregate with a failed/absent unit -> refuses
def test_a6_incomplete_population_refuses():
    m = _fake_measurements(5.0, 0.01, 0.9)
    m["expected_units"] = 4
    with pytest.raises(SystemExit, match="population incomplete"):
        adj.adjudicate(m)


# 7. inflated support from windows of one process -> unit rule
def test_a7_windows_are_never_replicas():
    m = _fake_measurements(5.0, 0.01, 0.9)
    m["records"] = m["records"][:1]     # one seed only
    m["expected_units"] = 1
    out = adj.adjudicate(m)
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
    out = adj.adjudicate(m)
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
    batch = co.transform_batch(art, x, ["a"])
    state = co.init_state(art)
    frag = []
    for i in range(60):
        out, state = co.transform_incremental(art, state,
                                              x[i], ["a"])
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
    m = json.loads(MEAS.read_text())
    out = adj.adjudicate(m)
    assert out["population"]["records"] == 1152
    assert out["verdict_counts"]["NON_CAUSAL_ORACLE_ONLY"] == 64


@_meas_present
def test_m2_mutated_record_changes_verdict_derivation():
    m = json.loads(MEAS.read_text())
    out1 = adj.adjudicate(json.loads(json.dumps(m)))
    for r in m["records"]:
        if r["operator"] == "trailing_median" and \
                r["status"] == "MEASURED":
            for pv in r["per_variable"]:
                pv["extreme_retention"] = 1.0
                for h in ("h1", "h5"):
                    pv["assays"][h]["D"] = \
                        pv["assays"][h]["X"] + 0.5
    out2 = adj.adjudicate(m)
    assert out1["verdict_counts"] != out2["verdict_counts"]


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
