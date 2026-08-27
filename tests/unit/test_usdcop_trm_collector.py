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
    ({"valor": "-1", "vigenciadesde": "2026-08-24"}, "non-positive"),
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
