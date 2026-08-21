"""Runtime finding 2026-08-20: summary trades_total, final cumulative
trace value and authority-derived count must be IDENTICAL."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins import _activity_authority as aa  # noqa: E402
from pipeline_plugins import _return_trace as tr  # noqa: E402


def rows_with(counts):
    return [{"closed_trades_cumulative": c, "trades": c}
            for c in counts]


class TestReconciliation:
    @pytest.mark.parametrize("running,total,settlement,open_pos", [
        ([0, 0, 0], 0, 0, 0),   # zero trades
        ([0, 0, 1], 1, 0, 0),   # one, settled in-episode
        ([0, 1, 2], 3, 1, 1),   # the ONE open position closes at end
        ([1, 2, 5], 5, 0, 0),   # several, no terminal settlement
    ])
    def test_final_equals_summary_exactly(self, running, total,
                                          settlement, open_pos):
        rows = rows_with(running)
        result = tr.reconcile_trace_trades(
            rows, total, terminal_open_positions=open_pos)
        assert rows[-1]["closed_trades_cumulative"] == total
        assert result["terminal_settlement_trades"] == settlement
        if settlement:
            event = result["terminal_settlement_event"]
            assert event["settlement_delta"] == settlement
            assert event["open_positions_at_last_bar"] == open_pos

    def test_fabrication_refused_summary_cannot_mint_trades(self):
        # THE audited case: trace ends at 2, summary claims 100
        with pytest.raises(tr.TraceReconciliationError,
                           match="cannot mint"):
            tr.reconcile_trace_trades(rows_with([0, 1, 2]), 100,
                                      terminal_open_positions=1)

    def test_settlement_without_open_position_refused(self):
        with pytest.raises(tr.TraceReconciliationError,
                           match="cannot mint"):
            tr.reconcile_trace_trades(rows_with([0, 1, 2]), 3,
                                      terminal_open_positions=0)

    def test_monotonicity_violation_refuses(self):
        with pytest.raises(tr.TraceReconciliationError,
                           match="decreased"):
            tr.reconcile_trace_trades(rows_with([0, 3, 1]), 3,
                                      terminal_open_positions=0)

    def test_unexplained_plus_two_jump_refuses(self):
        # TR-8: +2 with one open position is one close too many
        with pytest.raises(tr.TraceReconciliationError,
                           match="cannot mint"):
            tr.reconcile_trace_trades(rows_with([0, 1]), 3,
                                      terminal_open_positions=1)

    def test_settlement_appends_row_market_row_untouched(self):
        rows = rows_with([0, 1, 2])
        rows[-1]["price"] = 123.0
        tr.reconcile_trace_trades(rows, 3, terminal_open_positions=1)
        assert len(rows) == 4                     # appended, not mutated
        assert rows[2]["closed_trades_cumulative"] == 2  # market row
        settlement_row = rows[3]
        assert settlement_row["closed_trades_cumulative"] == 3
        assert "before=2;delta=1;after=3" in \
            settlement_row["terminal_settlement_event"]

    def test_four_way_count_equality(self, tmp_path):
        """final cumulative == summary == authority-derived ==
        settlement after_count (TR-4)."""
        rows = rows_with([0, 1, 2])
        result = tr.reconcile_trace_trades(rows, 3,
                                           terminal_open_positions=1)
        event = result["terminal_settlement_event"]
        assert rows[-1]["closed_trades_cumulative"] == 3
        assert event["after_count"] == 3
        assert result["final_cumulative"] == 3

    @pytest.mark.parametrize("bad", [2.0, "3", True, None,
                                     float("nan"), float("inf")])
    def test_non_integral_counts_refuse_no_truncation(self, bad):
        rows = [{"closed_trades_cumulative": bad, "trades": bad}]
        with pytest.raises(tr.TraceReconciliationError,
                           match="not an integer"):
            tr.reconcile_trace_trades(rows, 1,
                                      terminal_open_positions=1)

    def test_negative_count_refuses(self):
        with pytest.raises(tr.TraceReconciliationError,
                           match="negative"):
            tr.reconcile_trace_trades(
                [{"closed_trades_cumulative": -1}], 1,
                terminal_open_positions=1)

    def test_counter_exceeding_summary_refuses(self):
        with pytest.raises(tr.TraceReconciliationError,
                           match="exceeds"):
            tr.reconcile_trace_trades(rows_with([0, 5]), 3,
                                      terminal_open_positions=1)

    def test_authority_derives_the_same_count(self, tmp_path):
        rows = rows_with([0, 1, 2])
        tr.reconcile_trace_trades(rows, 3, terminal_open_positions=1)
        path = tmp_path / "trace.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["closed_trades_cumulative",
                                    "trades",
                                    "terminal_settlement_event"],
                extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        descriptor = {
            "schema": aa.EVIDENCE_DESCRIPTOR_SCHEMA,
            "role": "inner_validation", "source_kind": "return_trace",
            "artifact_locator": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fact_key": "closed_trades_cumulative",
            "producer_contract_id": "test.v1"}
        verdict = aa.verify_evidence(descriptor,
                                     expected_role="inner_validation")
        assert verdict["verified"] is True
        assert verdict["derived_trades"] == 3     # == summary total

    def test_legacy_trace_without_cumulative_field_refuses(
            self, tmp_path):
        path = tmp_path / "legacy.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["trades"])
            writer.writeheader()
            writer.writerows([{"trades": 3}])
        descriptor = {
            "schema": aa.EVIDENCE_DESCRIPTOR_SCHEMA,
            "role": "inner_validation", "source_kind": "return_trace",
            "artifact_locator": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fact_key": "closed_trades_cumulative",
            "producer_contract_id": "test.v1"}
        verdict = aa.verify_evidence(descriptor,
                                     expected_role="inner_validation")
        assert verdict["verified"] is False
        assert "EVIDENCE_FACT_MISMATCH_INNER_VALIDATION" in \
            verdict["reason_codes"]
        # the ambiguous legacy column is never silently reused


class TestExecutingCallPath:
    """TR-L1: the EXECUTING call site (not just the primitive) refuses
    foreign count types."""

    @pytest.mark.parametrize("bad", ["3", 3.7, True, float("nan"),
                                     float("inf"), -1])
    def test_rollout_call_site_refuses_foreign_counts(self, bad):
        from pipeline_plugins.rl_pipeline_with_validation import (
            _reconcile_rollout_trades,
        )
        rows = [{"closed_trades_cumulative": 0, "position": 0.0}]
        with pytest.raises(tr.TraceReconciliationError):
            _reconcile_rollout_trades(rows, {"trades_total": bad})

    def test_rollout_call_site_source_has_no_precoercion(self):
        import inspect
        from pipeline_plugins.rl_pipeline_with_validation import (
            _reconcile_rollout_trades,
        )
        source = inspect.getsource(_reconcile_rollout_trades)
        assert "int(summary" not in source
        assert 'summary.get("trades_total")' in source

    def test_selection_firewall_test_split_cannot_influence_stopping(
            self):
        """TR-L2 second half: the executing stopping path consumes only
        the train-tail and validation summaries."""
        import inspect
        from pipeline_plugins.rl_pipeline_with_validation import (
            _early_stop_composite,
        )
        params = list(inspect.signature(
            _early_stop_composite).parameters)
        assert params[0] == "train_tail_summary"
        assert params[1] == "val_summary"
        assert not any("test" in name for name in params)
