"""N2 refusal tests (order @8fce8da0 §6): the census aggregate
refuses or downgrades on missing/failed units, license failures and
negative-control failures; geometry invariants hold; the losses and
the dependence-aware bootstrap behave as predeclared."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "tcn2", REPO / "tools" / "target_horizon_census_n2.py")
tcn2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcn2)

from agent_plugins.experiment_runtime import (  # noqa: E402
    RunDirectory, unit_id)


@pytest.fixture()
def home_tmp():
    root = (Path.home() / ".cache" / "tcn2_tests" / uuid.uuid4().hex)
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


N = 216
WKS = ("w1", "w2", "w3", "w4")


def _geometry():
    return {"sufficient": True, "n": 2200, "frontier": 1533,
            "embargo_rows": 3, "cal_len": 176, "score_len": N,
            "windows": {
                f"w{k}": {"fit": [0, 400],
                          "cal": [400, 576],
                          "score": [663 + 219 * (k - 1),
                                    663 + 219 * (k - 1) + N]}
                for k in (1, 2, 3, 4)}}


def _result(key, wk, *, delta):
    """Synthetic terminal record. delta > 0 -> model beats every
    baseline by ~delta in per-obs loss (skill ~ delta)."""
    fam, hidx, mode = tcn2.ALL_TREATMENTS[key]
    rng = np.random.default_rng(
        abs(hash((key, wk))) % (2 ** 31))
    base = 1.0 + 0.05 * rng.standard_normal(N)
    baselines = {}
    for i, b in enumerate(tcn2.FAMILY_BASELINES[fam]):
        baselines[b] = np.abs(base + 0.01 * i) + 0.05
    ref = baselines[tcn2.FAMILY_BASELINES[fam][0]]
    model = np.abs(ref - delta * ref
                   + 0.01 * rng.standard_normal(N)) + 1e-6
    result = {
        "family": fam, "mode": mode,
        "horizon": tcn2.HORIZON_OF[fam][hidx],
        "score_rows": _geometry()["windows"][wk]["score"],
        "n_score": N, "effective_blocks": N // tcn2.BLOCK,
        "selected_model": "causal_linear",
        "model_records": {"causal_linear": {"lambda": 1.0,
                                            "cal_loss": 1.0,
                                            "cond": 10.0,
                                            "coef_norm": 1.0}},
        "baseline_losses": {k: [round(float(v), 8) for v in arr]
                            for k, arr in baselines.items()},
        "model_losses": [round(float(v), 8) for v in model],
    }
    if fam == "bar":
        result["class_support"] = {
            role: {"0": 60, "1": 60, "2": 96}
            for role in ("fit", "cal", "score")}
    else:
        result["target_variance"] = {
            role: 0.01 for role in ("fit", "cal", "score")}
    return result


def _full_run(home, *, deltas=None, drop=None, fail=None,
              overrides=None):
    """deltas: {treatment: delta}; default candidates fail
    (delta -0.05), leak controls detect (0.7), shift controls fail
    (-0.05). overrides: {(key, wk): result-mutator}."""
    run = RunDirectory(home / "census")
    units = [tcn2._identity(key, wk)
             for wk in WKS for key in tcn2.ALL_TREATMENTS]
    run.write_ledger({
        "schema": "agent_multi.target_census_ledger.v1",
        "experiment": tcn2.EXPERIMENT,
        "units": [{"unit_id": unit_id(u), "identity": u}
                  for u in units],
        "digests": {}, "campaign_wall_ceiling_s": 7200.0,
        "unit_timeout_s": 600.0,
        "predeclaration": tcn2.PREDECLARATION,
        "role_geometry": _geometry()})
    for u in units:
        uid = unit_id(u)
        key, wk = u["treatment"], u["origin"]
        if drop and (key, wk) in drop:
            continue
        run.claim(uid, expected_digests={})
        if fail and (key, wk) in fail:
            run.release(uid, "FAILED",
                        result={"error": "synthetic"}, attempt=1)
            continue
        mode = tcn2.ALL_TREATMENTS[key][2]
        default = {"candidate": -0.05, "leak": 0.7,
                   "shift": -0.05}[mode]
        delta = (deltas or {}).get(key, default)
        result = _result(key, wk, delta=delta)
        if overrides and (key, wk) in overrides:
            overrides[(key, wk)](result)
        run.release(uid, "COMPLETED", result=result, attempt=1)
    return run


class TestGeometryInvariants:

    def test_embargo_three_and_disjoint_before_frontier(self):
        geo = tcn2.role_geometry(2200)
        assert geo["sufficient"] and geo["embargo_rows"] == 3
        spans = sorted(v["score"]
                       for v in geo["windows"].values())
        for a, b in zip(spans, spans[1:]):
            assert b[0] - a[1] >= 3
        for v in geo["windows"].values():
            assert v["score"][1] <= geo["frontier"]
            assert v["cal"][1] + 3 <= v["score"][0]

    def test_sixty_units(self):
        assert len(tcn2.ALL_TREATMENTS) == 15
        assert len(tcn2.CANDIDATES) == 9
        assert len(tcn2.CONTROLS) == 6

    def test_insufficient_n_is_typed(self):
        geo = tcn2.role_geometry(150)  # L = 12 < 30 -> refuse
        assert geo["sufficient"] is False


class TestLosses:

    def test_qlike_zero_at_truth_positive_elsewhere(self):
        y = np.array([-3.0, -2.5, -4.0])
        assert float(tcn2._qlike(y, y).max()) == pytest.approx(0.0)
        assert (tcn2._qlike(y + 0.3, y) > 0).all()
        assert (tcn2._qlike(y - 0.3, y) > 0).all()

    def test_logloss_clips(self):
        probs = np.array([[1.0, 0.0, 0.0]])
        val = tcn2._logloss(probs, np.array([1]))
        assert np.isfinite(val).all()


class TestBootstrap:

    def test_uniformly_positive_diffs_are_significant(self):
        rng = np.random.default_rng(7)
        diffs = [0.1 + 0.01 * rng.standard_normal(N)
                 for _ in range(4)]
        p = tcn2._block_bootstrap_p(diffs, tcn2.BOOT_SEED)
        assert p < 0.01

    def test_zero_centered_diffs_are_not(self):
        rng = np.random.default_rng(8)
        diffs = [0.05 * rng.standard_normal(N) for _ in range(4)]
        p = tcn2._block_bootstrap_p(diffs, tcn2.BOOT_SEED)
        assert p > 0.05

    def test_deterministic(self):
        rng = np.random.default_rng(9)
        diffs = [0.01 * rng.standard_normal(N) for _ in range(4)]
        assert tcn2._block_bootstrap_p(diffs, 505) == \
            tcn2._block_bootstrap_p(diffs, 505)


class TestAggregateRefusals:

    def test_all_failing_is_clean_negative(self, home_tmp):
        _full_run(home_tmp)
        trace = tcn2.aggregate_final(home_tmp)
        assert trace["verdict"] == \
            "NO_TARGET_CANDIDATE_DEMONSTRATED"
        assert trace["problems_preserved"] == []
        assert len(trace["holm_pvalues"]) == 9

    def test_strong_candidate_is_found_and_selected(self, home_tmp):
        _full_run(home_tmp, deltas={"vol_h6": 0.15})
        trace = tcn2.aggregate_final(home_tmp)
        assert trace["verdict"] == "TARGET_CANDIDATE_FOUND"
        assert trace["selection"]["selected"] == ["vol_h6"]
        assert trace["candidates"]["vol_h6"]["outcome"] == "PASSES"

    def test_selection_caps_at_two(self, home_tmp):
        _full_run(home_tmp, deltas={"vol_h6": 0.15,
                                    "ret_h6": 0.12,
                                    "bar_h6": 0.10})
        trace = tcn2.aggregate_final(home_tmp)
        assert trace["verdict"] == "TARGET_CANDIDATE_FOUND"
        assert len(trace["selection"]["selected"]) == 2
        assert len(trace["selection"]["ranked_passers"]) == 3

    def test_missing_unit_is_inconclusive(self, home_tmp):
        _full_run(home_tmp, drop={("vol_h6", "w2")})
        trace = tcn2.aggregate_final(home_tmp)
        assert trace["verdict"] == "INCONCLUSIVE"
        assert trace["problems_preserved"]

    def test_failed_unit_preserved_not_dropped(self, home_tmp):
        _full_run(home_tmp, fail={("ret_h1", "w3")})
        trace = tcn2.aggregate_final(home_tmp)
        assert trace["verdict"] == "INCONCLUSIVE"
        assert any(p["why"] == "FAILED"
                   and p["treatment"] == "ret_h1"
                   for p in trace["problems_preserved"])

    def test_leak_control_failure_invalidates_census(self,
                                                     home_tmp):
        _full_run(home_tmp, deltas={"ctl_leak_vol_h6": 0.05})
        trace = tcn2.aggregate_final(home_tmp)
        assert trace["verdict"] == "INCONCLUSIVE"
        assert any("did NOT flag leaked" in c
                   for c in trace["cause"])

    def test_shift_control_passing_invalidates_census(self,
                                                      home_tmp):
        _full_run(home_tmp, deltas={"ctl_shift_ret_h6": 0.15})
        trace = tcn2.aggregate_final(home_tmp)
        assert trace["verdict"] == "INCONCLUSIVE"
        assert any("harness invalid" in c for c in trace["cause"])

    def test_low_class_support_is_inconclusive_candidate(
            self, home_tmp):
        """C2 adversarial (order @4c1f1532): ONE unlicensed
        candidate among eight failures — the SEALED rule makes the
        whole census INCONCLUSIVE, never a clean negative."""
        def starve(result):
            result["class_support"]["score"] = \
                {"0": 5, "1": 60, "2": 151}
        _full_run(home_tmp, overrides={
            (("bar_h6"), wk): starve for wk in WKS})
        trace = tcn2.aggregate_final(home_tmp)
        assert "bar_h6" in trace["inconclusive_candidates"]
        assert trace["candidates"]["bar_h6"]["outcome"] == \
            "INCONCLUSIVE_CANDIDATE"
        assert trace["verdict"] == "INCONCLUSIVE"
        assert any("sealed rule" in c for c in trace["cause"])

    def test_unlicensed_beside_passer_is_inconclusive(
            self, home_tmp):
        """C2 adversarial (order @4c1f1532): one unlicensed
        candidate beside an apparent passer — INCONCLUSIVE and NO
        selection, irrespective of the passer."""
        def starve(result):
            result["class_support"]["score"] = \
                {"0": 5, "1": 60, "2": 151}
        _full_run(home_tmp, deltas={"vol_h6": 0.15},
                  overrides={
                      (("bar_h6"), wk): starve for wk in WKS})
        trace = tcn2.aggregate_final(home_tmp)
        assert trace["verdict"] == "INCONCLUSIVE"
        assert "selection" not in trace
        assert "bar_h6" in trace["inconclusive_candidates"]

    def test_degenerate_model_is_inconclusive_candidate(
            self, home_tmp):
        def degenerate(result):
            result["model_losses"] = None
            result["selected_model"] = None
        _full_run(home_tmp, overrides={
            (("ret_h12"), wk): degenerate for wk in WKS})
        trace = tcn2.aggregate_final(home_tmp)
        assert "ret_h12" in trace["inconclusive_candidates"]
        assert trace["verdict"] == "INCONCLUSIVE"


class TestRealN2Rederivation:

    def test_repaired_judge_preserves_real_n2_verdict(self):
        """C2 (order @4c1f1532): the sealed-semantics repair does
        NOT alter the real N2 result, because all nine real
        candidates were licensed — proven by rederivation from the
        durable run directory."""
        run_root = (Path.home() / ".local/share/agent-multi/"
                    "target_horizon_census_n2_20260903")
        if not run_root.exists():
            pytest.skip("frozen N2 run not present on this host")
        trace = tcn2.aggregate_final(run_root)
        assert trace["inconclusive_candidates"] == []
        assert trace["verdict"] == "TARGET_CANDIDATE_FOUND"
        assert trace["selection"]["selected"] == \
            ["bar_h6", "bar_h12"]
        for key in ("bar_h6", "bar_h12"):
            assert trace["candidates"][key]["outcome"] == "PASSES"
            assert trace["candidates"][key][
                "bootstrap_p_reported"] == "<= 1/2001"


class TestFrozenAnchors:

    def test_n1_digest_constant_matches_frozen_ledger(self):
        run_root = (Path.home() / ".local/share/agent-multi/"
                    "target_identifiability_n1_20260903")
        if not run_root.exists():
            pytest.skip("frozen N1 run not present on this host")
        ledger = json.loads(
            (run_root / "diagnostic" / "ledger.json").read_text())
        assert ledger["digests"]["input_w64"] == \
            tcn2.N1_INPUT_DIGEST

    def test_predeclaration_is_committed(self):
        assert (REPO / tcn2.PREDECLARATION).exists()
        pre = json.loads((REPO / tcn2.PREDECLARATION).read_text())
        assert pre["sealed_before_any_result"] is True
        keys = ([f"ret_h{h}" for h in (1, 3, 6, 12)]
                + [f"vol_h{h}" for h in (3, 6, 12)]
                + [f"bar_h{h}" for h in (6, 12)])
        assert sorted(tcn2.CANDIDATES) == sorted(keys)
