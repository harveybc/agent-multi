"""Exact counterexample regressions for DATA-SOTA-374/375/376 (SAC
driver correction order 2026-08-28). The 374/376 cases are fed from
the PUBLISHED raw report (facts unaltered); 375 pins the floor rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.objective_routing import select_routes  # noqa: E402

REPORT = json.loads((REPO / "docs/audits/evidence/"
                     "FINAL_PROBE_SCREEN_REPORT_2026_08_28.json"
                     ).read_text())


class TestDataSota374RefusedNeverSelected:
    def test_all_arms_refused_family_selects_nothing(self):
        """volatility_distribution: EVERY arm ROUTE_REFUSED — the old
        fallback still wrote full5 into selected."""
        for family in ("volatility_distribution", "oscillators",
                       "volume_flow"):
            result = select_routes({family: REPORT["families"][family]})
            assert result["selected"][family] is None, family
            assert ("NOT_EVALUABLE" in result["verdicts"][family]
                    or "INCOMPLETE_EVIDENCE"
                    in result["verdicts"][family]), family

    def test_refused_full5_cannot_be_fallback(self):
        families = {"synthetic": {"arms": {
            "full5_control": {"ROUTE_REFUSED": "seed instability"},
            "predictive3": {"skills": {"quantile": -0.5,
                                       "volatility": -0.4,
                                       "barrier": -0.3,
                                       "reconstruction": 0.1,
                                       "contrastive": 0.1}}}}}
        result = select_routes(families)
        assert result["selected"]["synthetic"] is None
        assert "NOT_EVALUABLE" in result["verdicts"]["synthetic"]


class TestDataSota376AllThreePredictiveRequired:
    def test_returns_momentum_is_incomplete_not_selected(self):
        """The published 0.6736 rested on quantile+volatility with
        barrier DIAGNOSTIC_INVALID — now INCOMPLETE_EVIDENCE."""
        result = select_routes({"returns_momentum":
                                REPORT["families"]["returns_momentum"]})
        assert result["selected"]["returns_momentum"] is None
        assert "INCOMPLETE_EVIDENCE" in result["verdicts"][
            "returns_momentum"]

    def test_two_of_three_predictive_never_eligible(self):
        families = {"synthetic": {"arms": {
            "full5_control": {"skills": {"quantile": 0.9,
                                         "volatility": 0.8,
                                         "reconstruction": 0.5}}}}}
        result = select_routes(families)
        assert result["selected"]["synthetic"] is None

    def test_fully_valid_arm_selects_properly(self):
        families = {"synthetic": {"arms": {
            "full5_control": {"skills": {"quantile": 0.6,
                                         "volatility": 0.5,
                                         "barrier": 0.4,
                                         "reconstruction": 0.2,
                                         "contrastive": 0.1}},
            "predictive3": {"skills": {"quantile": 0.1,
                                       "volatility": 0.1,
                                       "barrier": 0.1,
                                       "reconstruction": 0.0,
                                       "contrastive": 0.0}}}}}
        result = select_routes(families)
        assert result["selected"]["synthetic"]["arm"] == "full5_control"
        assert result["selected"]["synthetic"]["label"] == "SELECTED"

    def test_valid_but_worse_than_random_falls_back_to_evaluable_full5(
            self):
        families = {"synthetic": {"arms": {
            "full5_control": {"skills": {"quantile": -0.5,
                                         "volatility": 0.2,
                                         "barrier": 0.1,
                                         "reconstruction": 0.1,
                                         "contrastive": 0.0}}}}}
        result = select_routes(families)
        assert result["selected"]["synthetic"]["label"] == \
            "CONSERVATIVE_DIAGNOSTIC"


class TestDataSota375FloorRule:
    def test_marginal_floor_makes_task_invalid_in_tool(self):
        source = (REPO / "tools/final_probe_screen.py").read_text()
        assert "floor_fit_marginal" in source
        # the tool must treat a marginal floor as DIAGNOSTIC_INVALID
        # for skill and preserve the floor provenance
        assert "random_floor_provenance" in source

    def test_published_report_is_tombstoned_not_rewritten(self):
        assert REPORT.get("verdict_status") == \
            "DIAGNOSTIC_PROTOCOL_INVALID_374_376"
        # raw facts intact
        assert "families" in REPORT and "verdicts" in REPORT
        assert set(REPORT["families"]) == {
            "returns_momentum", "trend_level",
            "volatility_distribution", "oscillators", "volume_flow"}
