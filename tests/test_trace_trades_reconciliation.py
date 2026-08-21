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
    @pytest.mark.parametrize("running,total,settlement", [
        ([0, 0, 0], 0, 0),          # zero trades
        ([0, 0, 1], 1, 0),          # one, settled in-episode
        ([0, 1, 2], 3, 1),          # close on the FINAL bar settles +1
        ([1, 2, 5], 7, 2),          # several + terminal settlement
    ])
    def test_final_equals_summary_exactly(self, running, total,
                                          settlement):
        rows = rows_with(running)
        result = tr.reconcile_trace_trades(rows, total)
        assert rows[-1]["closed_trades_cumulative"] == total
        assert result["terminal_settlement_trades"] == settlement

    def test_counter_exceeding_summary_refuses(self):
        with pytest.raises(tr.TraceReconciliationError,
                           match="exceeds"):
            tr.reconcile_trace_trades(rows_with([0, 5]), 3)

    def test_authority_derives_the_same_count(self, tmp_path):
        rows = rows_with([0, 1, 2])
        tr.reconcile_trace_trades(rows, 4)
        path = tmp_path / "trace.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["closed_trades_cumulative",
                                    "trades"])
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
        assert verdict["derived_trades"] == 4     # == summary total

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
