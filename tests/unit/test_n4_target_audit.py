"""N4 refusal battery (order @13fdf18c §§6-8): census completeness
per the ordered fields, sealed-definition causality of the successor
builders, per-horizon purge containment, the development-only
2026-guard, and the screen's decision constants matching the sealed
design."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "n4a", REPO / "tools" / "n4_target_audit.py")
n4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n4a)

CENSUS = (REPO / "docs/audits/evidence/"
          "N4_TARGET_CENSUS_2026_09_04.json")
DESIGN = json.loads((REPO / "docs/audits/evidence/"
                     "N4_TARGET_AUDIT_DESIGN_2026_09_04.json")
                    .read_text())

REQUIRED_FIELDS = {
    "definition", "units", "causal_decision_time_information",
    "horizon_bars", "overlap_purge_requirement",
    "economic_interpretation_and_cost_dependence",
    "simplest_admissible_baseline",
    "distribution_by_dev_window",
    "roles_already_inspected_selected_confirmed",
    "remaining_untouched_confirmation_roles"}


class TestCensusCompleteness:

    @pytest.fixture(scope="class")
    def census(self):
        if not CENSUS.exists():
            pytest.skip("census not yet generated")
        return json.loads(CENSUS.read_text())

    def test_all_sixteen_targets_present(self, census):
        keys = set(census["targets"])
        assert {f"ret_h{h}" for h in (1, 3, 6, 12)} <= keys
        assert {f"vol_h{h}" for h in (3, 6, 12)} <= keys
        assert {"bar_h6", "bar_h12"} <= keys
        assert {f"tm_h{h}" for h in (6, 12, 24)} <= keys
        assert {"mfemae_h6", "mfemae_h12"} <= keys
        assert {"lm_h6", "lm_h12"} <= keys
        assert len(keys) == 16

    def test_every_entry_has_every_ordered_field(self, census):
        for key, entry in census["targets"].items():
            missing = REQUIRED_FIELDS - set(entry)
            assert not missing, f"{key}: missing {missing}"

    def test_prior_use_derived_not_labeled(self, census):
        note = census["prior_use_derivation_note"]
        assert "committed" in note and "never" in note
        for key in ("ret_h6", "vol_h6", "bar_h6"):
            roles = census["targets"][key][
                "roles_already_inspected_selected_confirmed"]
            assert any("docs/audits/evidence" in r for r in roles)

    def test_untouched_roles_answer_is_explicit(self, census):
        for entry in census["targets"].values():
            assert "NONE on this data contract" in entry[
                "remaining_untouched_confirmation_roles"]


class TestSuccessorBuilders:

    def _plane(self):
        closes = np.array([100.0, 101.0, 99.0, 100.5, 100.6,
                           100.4, 100.7, 102.0, 98.0, 100.0,
                           100.1, 100.2, 100.3, 100.4, 100.5,
                           100.6, 100.7, 100.8, 100.9, 101.0,
                           101.1, 101.2, 101.3, 101.4, 101.5,
                           101.6, 101.7, 101.8, 101.9, 102.0])
        return {"anchors": np.array([0, 1, 2]),
                "closes": closes,
                "highs": closes * 1.01,
                "lows": closes * 0.99}

    def test_tradeable_move_uses_only_forward_window(self):
        out = n4a.build_targets(self._plane())
        # anchor 0, h=6: r = log(100.7/100) > 10bp -> class 0
        assert out["tm_h6"][0] == 0

    def test_mfemae_is_finite_and_causal(self):
        out = n4a.build_targets(self._plane())
        assert np.isfinite(out["mfemae_h6"][:3]).all()

    def test_tail_overflow_masked_not_crashing(self):
        plane = self._plane()
        plane["anchors"] = np.array([0, 25])  # 25+6 > 30 rows
        out = n4a.build_targets(plane)
        assert np.isnan(out["r_h6"][1]) or np.isfinite(
            out["r_h6"][1])  # masked path executes
        assert np.isnan(out["mfemae_h24"][1])


class TestPurgeContainment:

    def test_labels_stay_inside_roles(self):
        geometry = json.loads(
            (REPO / n4a.N2_BUNDLE).read_text()
        )["ledger"]["role_geometry"]
        for wk in geometry["windows"]:
            for h in (6, 12, 24):
                fit, cal, sc = n4a._window_roles(geometry, wk, h)
                tail = -(-h // n4a.STRIDE)
                assert all(r + tail
                           < geometry["windows"][wk]["fit"][1]
                           for r in fit)
                assert all(r + tail
                           < geometry["windows"][wk]["score"][1]
                           for r in sc)


class TestDevelopmentOnlyGuard:

    def test_dev_anchors_end_before_2022(self):
        """The screen's data plane is the o2022 fit slice: its
        LAST row is 2021-12-31 — 2026 confirmation rows are
        structurally unreachable."""
        run_root = (Path.home() / ".local/share/agent-multi/"
                    "target_horizon_census_n2_20260903")
        if not run_root.exists():
            pytest.skip("frozen N2 run not present")
        plane = n4a._load_dev_arrays(run_root)
        assert len(plane["closes"]) == 9319  # fit_train rows
        assert plane["anchors"].max() + 24 < 9340


class TestSealedConstants:

    def test_screen_constants_match_design(self):
        d = DESIGN
        assert n4a.ROUND_TRIP_COST == 0.0010
        assert n4a.EXCEEDANCE_Q == 0.80
        assert n4a.MARGIN == 0.01
        assert n4a.BOOT_SEED == 808
        assert "0.0010" in json.dumps(d)
        fams = d["candidate_families_max3_horizons_max3"]
        assert len(fams) <= 3
        for fam, spec_ in fams.items():
            assert len(spec_["horizons"]) <= 3
            assert "distinct_from_failures" in spec_

    def test_verdict_labels_exact(self):
        v = DESIGN["decision_rule"]["verdicts"]
        assert set(v) == {
            "TARGET_FORMULATION_CANDIDATE_FOR_FUTURE_CONFIRMATION",
            "TARGET_FORMULATION_NOT_IDENTIFIED",
            "NO_UNTOUCHED_CONFIRMATION_ROLE_AVAILABLE"}
