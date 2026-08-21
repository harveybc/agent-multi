"""WP3 (order 2026-08-20): call-path proof that the EXECUTING selector
and early-stop state consume the episodic objective, and that the
legacy scalar path refuses for this contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _episodic_activity_fitness as ef  # noqa: E402
from pipeline_plugins.rl_pipeline_with_validation import (  # noqa: E402
    PipelinePlugin,
    _early_stop_composite,
    _selection_value,
)

CAL = {"activity_plateau_low_rate": 50.0,
       "activity_plateau_high_rate": 300.0}


def _summary(trades, ret, dd=0.05, rows=2190):
    return {"trades_total": trades, "total_return": ret,
            "max_drawdown_fraction": dd, "scored_steps": rows,
            "_episodic_fitness_config": dict(CAL)}


class TestCallPath:
    def test_selector_consumes_episodic_fitness(self, monkeypatch):
        calls = []
        original = ef.evaluate_episode

        def spy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(ef, "evaluate_episode", spy)
        value = _selection_value(
            _summary(40, -0.05),
            selection_metric="episodic_activity_economic_v1",
            risk_lambda=1.0)
        assert len(calls) == 1
        assert calls[0]["closed_trades"] == 40
        assert value < 0

    def test_early_stop_state_consumes_episodic_composite(self):
        """The REAL stopping path: _early_stop_composite with the
        episodic metric — its composite IS the episodic pairing."""
        tm = _summary(85, -0.02)
        tm["activity_evidence_descriptor"] = None  # authority gate path
        iv = _summary(20, -0.03)
        composite, raw, gate, *_ = _early_stop_composite(
            tm, iv, min_trades=1, no_trade_penalty=1e6,
            selection_metric="episodic_activity_economic_v1")
        # both summaries got their episodic result attached by the
        # executing selector
        assert "episodic_fitness" in tm and "episodic_fitness" in iv
        assert tm["episodic_fitness"]["schema"] == ef.SCHEMA

    def test_summary_carries_the_full_component_record(self):
        summary = _summary(85, -0.02)
        _selection_value(summary,
                         selection_metric="episodic_activity_economic_v1",
                         risk_lambda=1.0)
        result = summary["episodic_fitness"]
        for component in ("branch", "annualized_trade_rate",
                          "activity_utility", "selection_value"):
            assert component in result


class TestLegacyPathRefusal:
    def test_missing_episodic_config_refuses_no_fallthrough(self):
        summary = _summary(40, -0.05)
        del summary["_episodic_fitness_config"]
        with pytest.raises(ef.EpisodicFitnessError,
                           match="refusing the legacy"):
            _selection_value(
                summary,
                selection_metric="episodic_activity_economic_v1",
                risk_lambda=1.0)

    def test_contract_requiring_episodic_refuses_legacy_metric(self):
        plugin = PipelinePlugin.__new__(
            PipelinePlugin)
        with pytest.raises(ef.EpisodicFitnessError,
                           match="legacy scalar path is refused"):
            plugin._assert_episodic_contract(
                {"require_episodic_fitness": True,
                 "selection_metric": "total_return"})
        # and the episodic metric passes the guard
        plugin._assert_episodic_contract(
            {"require_episodic_fitness": True,
             "selection_metric": "episodic_activity_economic_v1"})

    def test_undeclared_plateau_refuses_in_the_selector(self):
        summary = _summary(40, -0.05)
        summary["_episodic_fitness_config"] = {}
        with pytest.raises(ef.EpisodicFitnessError,
                           match="required and has no default"):
            _selection_value(
                summary,
                selection_metric="episodic_activity_economic_v1",
                risk_lambda=1.0)
