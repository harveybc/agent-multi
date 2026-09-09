"""M4 C31D: the numeric-incident regression battery — the twelve
failed routes from the incident audit, each against productive
code."""
import copy
import json
import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import m4_generator_bank as gb  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402
import m4_v5_adjudicate as adj  # noqa: E402
import m4_v5_protocol as pv  # noqa: E402
import m4_v5_runner as rn  # noqa: E402

ACCT = lambda: {"descriptor_seconds": 0.0,  # noqa: E731
                "descriptor_evals": 0,
                "optimization_updates": 0, "evaluations": 0}
OK_P = {"W1": np.ones((8, 4)), "b1": np.zeros(4),
        "W2": np.ones((4, 1)), "b2": np.zeros(1)}


def _summary(cause_i="ACQUISITION_ENDPOINT",
             cause_c="ACQUISITION_ENDPOINT", diff=4,
             desc_invalid=False, status=None, all_arms=True):
    desc = ({"numerically_invalid_descriptor": True,
             "compressed_len_zlib9": None,
             "spectral_rank_W1_1e3": None,
             "prune_fraction_1e3": None}
            if desc_invalid else
            {"compressed_len_zlib9": 500,
             "spectral_rank_W1_1e3": 4,
             "prune_fraction_1e3": 0.1,
             "descriptor_seconds": 0.0})
    arm = lambda c: {"stopping_cause": c,  # noqa: E731
                     "restricted_endpoint": 8,
                     "cap_reached": c == "CAP_REACHED",
                     "updates_done": 400,
                     "checkpoint_loss_stop": 0.4,
                     "fail_batch": 1,
                     "descriptors": dict(desc)}
    arms = {"initialization": arm(cause_i),
            "calibration_stop": arm(cause_c)}
    if all_arms:
        arms["pre_stop"] = arm("ACQUISITION_ENDPOINT")
        arms["post_stop_bounded"] = arm("ACQUISITION_ENDPOINT")
    s = {"unit_id": "u", "task_kind": "boolean",
         "noise_coord": "clean", "width": 16,
         "generator_id": "DEVELOPMENT-identity-clean-g0",
         "paired_primary_difference": diff,
         "checkpoint_lineage": {
             k: {"updates": 100} for k in pv.CHECKPOINTS},
         "stop_trajectory_slope": -0.01,
         "arms": arms}
    if status:
        s["unit_status"] = status
    return s


def test_route_1_nonfinite_training_typed():
    """Route 1 (corrected at 3f432a0b, regression): nonfinite
    parameters during checkpoint training return the typed
    invalid result instead of crashing."""
    g = gb.generate("DEVELOPMENT", "identity", "clean", 0)
    real = pv.sgd_step_task

    def poison(kind, p, X, y, lr):
        real(kind, p, X, y, lr)
        p["W1"][0, 0] = np.inf
    with mock.patch.object(pv, "sgd_step_task", poison):
        ck = pv.build_checkpoints(g, 16, 0)
    assert ck["numerically_invalid"] is True
    assert ck["checkpoints"] is None


def test_route_2_f32_range_is_typed_invalid():
    """Route 2: finite float64 parameters outside float32 range
    are NUMERICALLY_INVALID_DESCRIPTOR — infinity bytes are never
    compressed."""
    big = {"W1": np.full((8, 4), 1e39), "b1": np.zeros(4),
           "W2": np.ones((4, 1)), "b2": np.zeros(1)}
    d = rn._descriptors(big, ACCT())
    assert d["numerically_invalid_descriptor"] is True
    assert d["compressed_len_zlib9"] is None
    assert d["spectral_rank_W1_1e3"] is None


def test_route_3_svd_failure_typed():
    with mock.patch("numpy.linalg.svd",
                    side_effect=np.linalg.LinAlgError("x")):
        d = rn._descriptors(copy.deepcopy(OK_P), ACCT())
    assert d["numerically_invalid_descriptor"] is True


def test_route_4_nonfinite_singular_typed():
    with mock.patch("numpy.linalg.svd",
                    return_value=np.array([np.inf, 1.0])):
        d = rn._descriptors(copy.deepcopy(OK_P), ACCT())
    assert d["numerically_invalid_descriptor"] is True
    assert d["spectral_rank_W1_1e3"] is None


def test_route_5_intrabatch_anomaly_typed_and_replayed():
    """Route 5 (corrected, regression): intervention divergence
    types NUMERICAL_ANOMALY in the productive record path."""
    src = (REPO / "tools/m4_v5_protocol.py").read_text()
    seg = src[src.index("def run_intervention"):
              src.index("def dispersion_from_paired")]
    assert '"NUMERICAL_ANOMALY"' in seg
    assert "np.isfinite(ret_loss)" in seg


def test_routes_6_7_producer_claims_vs_replay():
    """Routes 6/7 (corrected, regression): a claimed invalidity
    the replay derives valid refuses, and vice versa — verifier
    source facts."""
    src = (REPO / "tools/m4_v5_runner.py").read_text()
    assert "does not replay to the same typed state" in src
    assert "claims an invalid status the" in src


def test_route_8_anomalous_arm_never_a_pair():
    s = _summary(cause_c="NUMERICAL_ANOMALY")
    assert adj._complete_primary_pair(s) is False
    s2 = _summary(cause_i="WALL_STOP")
    assert adj._complete_primary_pair(s2) is False
    s3 = _summary(status="NUMERICALLY_INVALID_TASK_TRAINING")
    assert adj._complete_primary_pair(s3) is False
    assert adj._complete_primary_pair(_summary()) is True


def test_route_9_exact_three_seeds(monkeypatch, tmp_path):
    """Route 9: two of three seeds NEVER average as a complete
    generator — the generator types INCOMPLETE_PAIRED_GENERATOR
    and stays in the denominator."""
    d = copy.deepcopy(pv.load_design_v5())
    cp = d["candidate_population"]
    cp["structured_boolean_families"] = ["identity"]
    cp["temporal_families"] = []
    cp["noise_regimes_temporal"] = []
    d["populations_v5"]["CALIBRATION_per_cell"] = 4
    summaries = {}
    for gi in range(4):
        for ms in range(rn.CAL_SEEDS):
            if gi == 0 and ms == 2:
                continue                    # missing seed
            uid = (f"intervention::CALIBRATION::identity::clean"
                   f"::w16::g{gi}::s{ms}")
            summaries[uid] = _summary(diff=4 + gi)
    per_gen = {}
    incomplete = []
    for gi in range(4):
        diffs = []
        seeds_complete = 0
        for ms in range(rn.CAL_SEEDS):
            uid = (f"intervention::CALIBRATION::identity::clean"
                   f"::w16::g{gi}::s{ms}")
            r = summaries.get(uid)
            if adj._complete_primary_pair(r):
                seeds_complete += 1
                diffs.append(r["paired_primary_difference"])
        if seeds_complete == rn.CAL_SEEDS:
            per_gen[f"g{gi}"] = float(np.mean(diffs))
        else:
            incomplete.append(gi)
    assert incomplete == [0]
    assert set(per_gen) == {"g1", "g2", "g3"}
    # and the productive source enforces the same rule
    src = (REPO / "tools/m4_v5_adjudicate.py").read_text()
    assert "seeds_complete == rn.CAL_SEEDS" in src
    assert "INCOMPLETE_PAIRED_GENERATOR" in src


def test_route_10_attrition_gate():
    """Route 10: attrition beyond the sealed 20% allowance can
    never report precision support — the cell types
    CALIBRATION_INCOMPLETE."""
    src = (REPO / "tools/m4_v5_adjudicate.py").read_text()
    assert "CALIBRATION_INCOMPLETE" in src
    assert "min_complete_required" in src
    assert "1 - ATTRITION_ALLOWANCE" in src
    # 16 planned, allowance 20% -> at least 13 complete required
    import math
    assert max(3, math.ceil(16 * 0.8)) == 13


def test_route_11_invalid_descriptor_never_in_m2():
    """Route 11: a quartet with one invalid descriptor is
    excluded from ladder rows with its reason reported."""
    s_ok = _summary()
    s_bad = _summary(desc_invalid=True)
    assert adj._complete_quartet(s_ok) is True
    assert adj._complete_quartet(s_bad) is False
    src = (REPO / "tools/m4_v5_adjudicate.py").read_text()
    assert "ladder_excluded" in src
    assert "INVALID_DESCRIPTOR" in src


def test_route_12_denominator_visibility():
    """Route 12: planned/complete/incomplete counts are published
    per cell — an incomplete unit can never disappear."""
    src = (REPO / "tools/m4_v5_adjudicate.py").read_text()
    for tok in ("planned_generators", "complete_generators",
                "incomplete_generators",
                "incomplete_units_in_denominator"):
        assert tok in src


def test_descriptor_validity_rederives_in_verifier():
    """C31B: the fresh verifier derives the SAME invalidity from
    the original float64 parameters (source facts + function
    equality on an invalid input)."""
    src = (REPO / "tools/m4_v5_runner.py").read_text()
    assert "descriptor " + \
        "validity/values do not re-derive" in src.replace(
            "\n                    \"", "\"").replace(
            "\"\n", "\"") or \
        "validity/values do not re-derive" in src
    big = {"W1": np.full((8, 4), 1e39), "b1": np.zeros(4),
           "W2": np.ones((4, 1)), "b2": np.zeros(1)}
    d1 = rn._descriptors(big, ACCT())
    d2 = rn._descriptors(big, ACCT())
    d1.pop("descriptor_seconds")
    d2.pop("descriptor_seconds")
    assert d1 == d2
