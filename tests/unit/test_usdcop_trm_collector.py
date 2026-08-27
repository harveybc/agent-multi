"""Regressions for the USDCOP TRM collector (WP-DATA, reporting only).

No network in tests: the fetcher is injected. The collector must upsert
idempotently, refuse malformed/non-positive rows, refuse empty ranges,
stamp the REPORTING_ONLY authority into store rows AND provenance, and
keep the provenance manifest free of absolute paths.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.collect_usdcop_trm import (TrmCollectorError,  # noqa: E402
                                      collect, normalize_row)

ROWS = [
    {"valor": "4100.55", "vigenciadesde": "2026-08-24T00:00:00.000",
     "vigenciahasta": "2026-08-24T00:00:00.000", "unidad": "COP"},
    {"valor": "4090.10", "vigenciadesde": "2026-08-25T00:00:00.000",
     "vigenciahasta": "2026-08-25T00:00:00.000", "unidad": "COP"},
]


def test_collect_upserts_and_writes_reporting_only_provenance(tmp_path):
    store = tmp_path / "trm.sqlite"
    manifest = collect(store, "2026-08-24", "2026-08-25", False,
                       fetcher=lambda s, e, latest: ROWS)
    assert manifest["rows_upserted"] == 2
    assert manifest["rows_in_store"] == 2
    assert "REPORTING_ONLY" in manifest["authority"]
    connection = sqlite3.connect(store)
    rows = connection.execute(
        "SELECT vigencia_desde, valor_cop_per_usd, authority "
        "FROM trm_observations ORDER BY vigencia_desde").fetchall()
    connection.close()
    assert [(r[0], r[1]) for r in rows] == [("2026-08-24", 4100.55),
                                            ("2026-08-25", 4090.10)]
    assert all("REPORTING_ONLY" in r[2] for r in rows)


def test_collect_is_idempotent_and_updates_in_place(tmp_path):
    store = tmp_path / "trm.sqlite"
    collect(store, None, None, True, fetcher=lambda s, e, latest: ROWS)
    corrected = [dict(ROWS[0], valor="4111.00"), ROWS[1]]
    manifest = collect(store, None, None, True,
                       fetcher=lambda s, e, latest: corrected)
    assert manifest["rows_in_store"] == 2  # upsert, not duplicate
    connection = sqlite3.connect(store)
    value = connection.execute(
        "SELECT valor_cop_per_usd FROM trm_observations "
        "WHERE vigencia_desde='2026-08-24'").fetchone()[0]
    connection.close()
    assert value == 4111.00


@pytest.mark.parametrize("row, fragment", [
    ({"vigenciadesde": "2026-08-24"}, "malformed"),
    ({"valor": "abc", "vigenciadesde": "2026-08-24"}, "malformed"),
    ({"valor": "-1", "vigenciadesde": "2026-08-24"}, "finite positive"),
], ids=["missing-valor", "non-numeric", "negative"])
def test_malformed_rows_refuse(row, fragment):
    with pytest.raises(TrmCollectorError, match=fragment):
        normalize_row(row)


def test_empty_source_range_refuses(tmp_path):
    with pytest.raises(TrmCollectorError, match="no rows"):
        collect(tmp_path / "trm.sqlite", "2026-01-01", "2026-01-02",
                False, fetcher=lambda s, e, latest: [])


def test_provenance_manifest_is_sanitized(tmp_path):
    store = tmp_path / "trm.sqlite"
    collect(store, None, None, True, fetcher=lambda s, e, latest: ROWS)
    manifest_text = (tmp_path / "trm_provenance.json").read_text()
    for banned in ("/home/", "/tmp/", "harveybc"):
        assert banned not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["store"].startswith("store:agent-multi-local/")
    assert manifest["source_dataset"] == "32sa-8pi3"


# --------------------------------- DATA-SOTA-352: temporal contract

from tools.collect_usdcop_trm import (  # noqa: E402
    TrmAmbiguous, TrmUnavailable, collect as _collect, trm_as_of)


class TestDataSota352TemporalContract:
    def test_garbage_and_impossible_dates_refuse(self):
        """The PRE counterexample: 'garbage-da' was sliced and stored."""
        for bad in ("garbage-date", "2026-02-30T00:00:00.000",
                    "2026-13-01", "2026-08-27X00:00"):
            with pytest.raises(TrmCollectorError,
                               match="ISO date|time suffix"):
                normalize_row({"valor": "4100", "vigenciadesde": bad})

    def test_inverted_interval_refuses(self):
        with pytest.raises(TrmCollectorError, match="inverted"):
            normalize_row({"valor": "4100",
                           "vigenciadesde": "2026-08-27T00:00:00.000",
                           "vigenciahasta": "2026-08-20T00:00:00.000"})

    def test_non_finite_and_wrong_unit_refuse(self):
        with pytest.raises(TrmCollectorError, match="finite positive"):
            normalize_row({"valor": "inf",
                           "vigenciadesde": "2026-08-27"})
        with pytest.raises(TrmCollectorError, match="unit"):
            normalize_row({"valor": "4100",
                           "vigenciadesde": "2026-08-27",
                           "unidad": "USD"})

    @staticmethod
    def _seed(tmp_path, rows):
        store = tmp_path / "trm.sqlite"
        _collect(store, None, None, True,
                 fetcher=lambda s, e, latest: rows)
        return store

    def test_as_of_weekend_span_returns_the_covering_row(self, tmp_path):
        # Friday publication valid through the weekend (hasta Monday)
        store = self._seed(tmp_path, [
            {"valor": "4000", "vigenciadesde": "2026-08-21",
             "vigenciahasta": "2026-08-24"},
            {"valor": "4050", "vigenciadesde": "2026-08-25",
             "vigenciahasta": "2026-08-25"}])
        sunday = trm_as_of(store, "2026-08-23T15:00:00Z")
        assert sunday["valor_cop_per_usd"] == 4000
        assert "REPORTING_ONLY" in sunday["authority"]

    def test_as_of_never_uses_future_effective_rows_early(self, tmp_path):
        """The PRE counterexample: the newest publication (vigencia
        2026-08-27) was collected on 08-26 with no as-of guard."""
        store = self._seed(tmp_path, [
            {"valor": "4111", "vigenciadesde": "2026-08-27",
             "vigenciahasta": "2026-08-27"}])
        with pytest.raises(TrmUnavailable, match="never used early"):
            trm_as_of(store, "2026-08-26T20:00:00Z")
        assert trm_as_of(store, "2026-08-27T00:00:00Z")[
            "valor_cop_per_usd"] == 4111

    def test_as_of_gap_is_typed_unavailable(self, tmp_path):
        store = self._seed(tmp_path, [
            {"valor": "4000", "vigenciadesde": "2026-08-21",
             "vigenciahasta": "2026-08-21"}])
        with pytest.raises(TrmUnavailable):
            trm_as_of(store, "2026-08-23T00:00:00Z")

    def test_as_of_overlap_is_typed_ambiguous(self, tmp_path):
        store = self._seed(tmp_path, [
            {"valor": "4000", "vigenciadesde": "2026-08-21",
             "vigenciahasta": "2026-08-25"},
            {"valor": "4050", "vigenciadesde": "2026-08-24",
             "vigenciahasta": "2026-08-26"}])
        with pytest.raises(TrmAmbiguous, match="overlapping"):
            trm_as_of(store, "2026-08-24T12:00:00Z")

    def test_revised_row_is_what_as_of_returns(self, tmp_path):
        store = self._seed(tmp_path, [
            {"valor": "4000", "vigenciadesde": "2026-08-21",
             "vigenciahasta": "2026-08-21"}])
        _collect(store, None, None, True, fetcher=lambda s, e, latest: [
            {"valor": "4009.99", "vigenciadesde": "2026-08-21",
             "vigenciahasta": "2026-08-21"}])
        assert trm_as_of(store, "2026-08-21T12:00:00Z")[
            "valor_cop_per_usd"] == 4009.99

    def test_provenance_manifest_written_atomically(self, tmp_path):
        store = self._seed(tmp_path, [
            {"valor": "4000", "vigenciadesde": "2026-08-21"}])
        assert not list(tmp_path.glob("*.tmp"))
        manifest = json.loads(
            (tmp_path / "trm_provenance.json").read_text())
        assert "as_of_rule" in manifest
