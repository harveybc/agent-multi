"""N1.2 focused refusal tests (order @89d099aa §4): wrong pairing,
reused score rows, missing seeds, forged aggregate and
post-materialization input change all refuse or return INCONCLUSIVE."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "tid", REPO / "tools" / "target_identifiability_audit.py")
tid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tid)

from agent_plugins.experiment_runtime import (  # noqa: E402
    RunDirectory, RuntimePreflightError, unit_id)


@pytest.fixture()
def home_tmp():
    root = (Path.home() / ".cache" / "tid_tests" / uuid.uuid4().hex)
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _geometry():
    return {"sufficient": True, "n": 2200, "frontier": 1533,
            "embargo_rows": 2, "cal_len": 176, "score_len": 216,
            "windows": {
                f"w{k}": {"fit": [0, 100 * k],
                          "cal": [100 * k, 100 * k + 50],
                          "score": [700 + 220 * (k - 1),
                                    700 + 220 * (k - 1) + 216]}
                for k in (1, 2, 3, 4)}}


def _full_run(home, *, drop=None, seeds=tid.SEEDS,
              scores=None):
    """Materialize a complete synthetic ledger and complete every
    unit with plausible scores; `drop` removes units, `seeds` trims
    temporal seeds, `scores` overrides per (arm, window)."""
    run = RunDirectory(home / "diagnostic")
    units = []
    for wk in ("w1", "w2", "w3", "w4"):
        for arm in tid.CPU_ARMS:
            units.append(tid._identity(arm, wk, 0))
        for seed in seeds:
            units.append(tid._identity("direct_temporal", wk, seed))
    run.write_ledger({
        "schema": "agent_multi.target_identifiability_ledger.v2",
        "experiment": tid.EXPERIMENT,
        "units": [{"unit_id": unit_id(u), "identity": u}
                  for u in units],
        "digests": {}, "campaign_wall_ceiling_s": 21600.0,
        "unit_timeout_s": 3600.0,
        "predeclaration": tid.PREDECLARATION,
        "role_geometry": _geometry()})
    for u in units:
        uid = unit_id(u)
        if drop and (u["treatment"], u["origin"],
                     u["seed"]) in drop:
            continue
        run.claim(uid, expected_digests={})
        base = {"literal_persistence": -0.10,
                "calibrated_ar1": -0.08,
                "direct_linear": -0.30,
                "direct_temporal": -0.09}[u["treatment"]]
        value = ((scores or {}).get((u["treatment"], u["origin"]))
                 if scores else None)
        run.release(uid, "COMPLETED", result={
            "score_r2": base if value is None else value,
            "score_rows": _geometry()["windows"][u["origin"]]
            ["score"]}, attempt=1)
    return run


class TestGeometryInvariants:

    def test_overlapping_score_rows_refuse_at_materialization(self):
        """Reused score rows can never materialize: the geometry
        invariant refuses windows closer than the embargo."""
        geometry = tid.role_geometry(2200)
        assert geometry["sufficient"]
        spans = sorted(v["score"] for v in
                       geometry["windows"].values())
        for a, b in zip(spans, spans[1:]):
            assert b[0] - a[1] >= tid.EMBARGO
        # a synthetic n too small to hold four windows refuses typed
        bad = tid.role_geometry(200)
        assert bad["sufficient"] is False

    def test_cal_precedes_score_with_embargo(self):
        geometry = tid.role_geometry(2200)
        for roles in geometry["windows"].values():
            assert roles["cal"][1] + tid.EMBARGO <= \
                roles["score"][0] + tid.EMBARGO
            assert roles["cal"][1] <= roles["score"][0] - 0
            assert roles["fit"][1] == roles["cal"][0]

    def test_all_windows_end_before_consumed_frontier(self):
        geometry = tid.role_geometry(2200)
        frontier = geometry["frontier"]
        for roles in geometry["windows"].values():
            assert roles["score"][1] <= frontier
        # the frontier is the START of the consumed origin-0 monitor,
        # stricter than the order's 85% boundary
        assert frontier == int(int(2200 * 0.85) * 0.82)


class TestAggregateRefusals:

    def test_complete_run_reaches_a_licensed_verdict(self, home_tmp):
        _full_run(home_tmp)
        trace = tid.aggregate_final(home_tmp)
        assert trace["verdict"] == "PREDICTABILITY_NOT_DEMONSTRATED"
        assert trace["problems_preserved"] == []

    def test_missing_unit_forces_inconclusive(self, home_tmp):
        _full_run(home_tmp,
                  drop={("direct_linear", "w2", 0)})
        trace = tid.aggregate_final(home_tmp)
        assert trace["verdict"] == "INCONCLUSIVE_INFRASTRUCTURE"
        assert trace["problems_preserved"]

    def test_missing_seed_forces_inconclusive(self, home_tmp):
        _full_run(home_tmp,
                  drop={("direct_temporal", "w3", 303)})
        trace = tid.aggregate_final(home_tmp)
        assert trace["verdict"] == "INCONCLUSIVE_INFRASTRUCTURE"

    def test_failed_unit_preserved_not_dropped(self, home_tmp):
        run = _full_run(home_tmp,
                        drop={("calibrated_ar1", "w1", 0)})
        uid = unit_id(tid._identity("calibrated_ar1", "w1", 0))
        run.claim(uid, expected_digests={})
        run.release(uid, "FAILED", note="scientific failure",
                    attempt=1)
        trace = tid.aggregate_final(home_tmp)
        assert trace["verdict"] == "INCONCLUSIVE_INFRASTRUCTURE"
        assert any(p.get("why") == "FAILED"
                   for p in trace["problems_preserved"])

    def test_forged_aggregate_result_refuses(self, home_tmp):
        run = _full_run(home_tmp)
        uid = unit_id(tid._identity("direct_linear", "w1", 0))
        path = run.root / "units" / f"{uid}.result.json"
        res = json.loads(path.read_text())
        res["score_r2"] = 0.95  # forged without re-digesting
        path.write_text(json.dumps(res))
        with pytest.raises(RuntimePreflightError,
                           match="digest mismatch"):
            tid.aggregate_final(home_tmp)

    def test_advancing_arm_yields_representation_bottleneck(
            self, home_tmp):
        scores = {("direct_linear", wk): 0.20
                  for wk in ("w1", "w2", "w3", "w4")}
        # slight window-to-window variation so sd > 0
        scores[("direct_linear", "w2")] = 0.22
        scores[("direct_linear", "w3")] = 0.21
        scores[("direct_linear", "w4")] = 0.23
        _full_run(home_tmp, scores=scores)
        trace = tid.aggregate_final(home_tmp)
        assert trace["verdict"] == \
            "REPRESENTATION_BOTTLENECK_DEMONSTRATED"
        assert trace["paired_analysis"]["direct_linear"]["advances"]

    def test_positive_without_license_is_discordant(self, home_tmp):
        # all windows positive but margin/CI cannot license
        scores = {("direct_linear", "w1"): 0.001,
                  ("direct_linear", "w2"): 0.002,
                  ("direct_linear", "w3"): 0.001,
                  ("direct_linear", "w4"): 0.002,
                  ("literal_persistence", "w1"): 0.0,
                  ("literal_persistence", "w2"): 0.0,
                  ("literal_persistence", "w3"): 0.0,
                  ("literal_persistence", "w4"): 0.0}
        _full_run(home_tmp, scores=scores)
        trace = tid.aggregate_final(home_tmp)
        assert trace["verdict"] == "INCONCLUSIVE_DISCORDANT"


class TestInputImmutability:

    def test_post_materialization_input_change_refuses_claim(
            self, home_tmp):
        """A worker whose recomputed input digest differs from the
        ledger refuses at claim (drift)."""
        run = RunDirectory(home_tmp / "diagnostic")
        ident = tid._identity("direct_linear", "w1", 0)
        uid = unit_id(ident)
        run.write_ledger({
            "schema": "s", "experiment": tid.EXPERIMENT,
            "units": [{"unit_id": uid, "identity": ident}],
            "digests": {"input_w64": "a" * 64},
            "campaign_wall_ceiling_s": 60, "unit_timeout_s": 30,
            "role_geometry": _geometry()})
        with pytest.raises(RuntimePreflightError, match="drift"):
            run.claim(uid, expected_digests={"input_w64": "b" * 64})
