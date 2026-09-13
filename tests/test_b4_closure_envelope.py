"""C111: one additive, idempotent B4 closure envelope, built only from the
accepted v4 submission, carrying counts and declared costs and no
per-cell scientific result."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("b4_closure_envelope", ROOT / "tools/b4_closure_envelope.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load()


def test_envelope_carries_the_accepted_closure_and_nothing_else():
    doc = E.build()
    assert doc["campaign_key"] == "b4::screen_b_v7_campaign" and doc["result_class"] == "DEVELOPMENT"
    assert doc["terminal"]["adjudication"] == "SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT"
    assert doc["terminal"]["state"] == "CLOSED"
    assert (doc["terminal"]["eligible_slots"], doc["terminal"]["total_slots"]) == (2, 12)
    assert doc["budget"]["units_verified"] == 2 and doc["budget"]["units_failed"] == 0
    metrics = {u["metric_name"]: u["metric_value"] for u in doc["units"]}
    assert metrics == {"cells_completed_verified": 2, "cells_quarantined_partial": 1,
                       "cells_not_started": 9,
                       "declared_wall_seconds_completed_cells": 35682.4,
                       "declared_wall_seconds_lower_bound_quarantined_partial": 124993.6}
    assert {u["cell_key"] for u in doc["units"]} == {"CAMPAIGN"}
    assert doc["identity"]["record_sha256"] == E.ACCEPTED_SELF_SHA256
    assert "/home/" not in json.dumps(doc)


def test_the_envelope_is_deterministic():
    assert E.build() == E.build()


def test_a_mutated_submission_refuses_before_building(tmp_path):
    raw = E.SUBMISSION.read_bytes()
    bad = tmp_path / E.SUBMISSION.name
    bad.write_bytes(raw.replace(b'"NOT_STARTED": 9', b'"NOT_STARTED": 8', 1))
    with pytest.raises(SystemExit, match="not the accepted v4 file"):
        E.build(bad)


def test_self_digest_and_counts_are_rechecked(monkeypatch, tmp_path):
    doc = json.loads(E.SUBMISSION.read_bytes())
    doc["adjudication"]["counts"]["QUARANTINED_PARTIAL"] = 0
    doc["adjudication"]["counts"]["TERMINAL_FAILED"] = 1
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(doc))
    import hashlib
    monkeypatch.setattr(E, "ACCEPTED_FILE_SHA256", hashlib.sha256(forged.read_bytes()).hexdigest())
    with pytest.raises(SystemExit, match="self digest"):
        E.build(forged)


def test_emitting_twice_writes_once(tmp_path):
    _, ob = E._olap()
    doc = E.build()
    first = ob.emit(doc, kind="envelope", root=tmp_path)
    second = ob.emit(doc, kind="envelope", root=tmp_path)
    assert first["written"] is True and second["written"] is False
    assert first["outbox_entry"] == second["outbox_entry"]
    assert len(list((tmp_path / "pending").glob("envelope-*.json"))) == 1


# ------------------------------------------------ THROWAWAY database only
def _dsn(db):
    import os
    return "postgresql+psycopg2://{u}:{p}@{h}:{P}/{d}".format(
        u=os.environ["PGUSER"], p=os.environ["PGPASSWORD"],
        h=os.environ.get("PGHOST", "localhost"), P=os.environ.get("PGPORT", "5432"), d=db)


@pytest.fixture()
def throwaway_engine():
    import os
    import uuid
    if "PGUSER" not in os.environ or "PGPASSWORD" not in os.environ:
        pytest.skip("no PG credentials")
    sa = pytest.importorskip("sqlalchemy")
    name = "c111_throwaway_" + uuid.uuid4().hex[:12]
    admin = sa.create_engine(_dsn("postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(sa.text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    engine = sa.create_engine(_dsn(name))
    try:
        yield engine
    finally:
        engine.dispose()
        admin = sa.create_engine(_dsn("postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def test_throwaway_database_load_is_additive_and_idempotent(throwaway_engine):
    from sqlalchemy import text
    ce, _ = E._olap()
    doc = E.build()
    first = ce.load_envelope(throwaway_engine, doc)
    second = ce.load_envelope(throwaway_engine, doc)
    assert first["units"] == 5 and second["units"] == 0
    with throwaway_engine.connect() as c:
        runs = c.execute(text(
            "SELECT result_class, terminal_state, adjudication FROM public.dim_campaign_run "
            "WHERE campaign_key = :k"), {"k": E.CAMPAIGN_KEY}).mappings().all()
        units = c.execute(text("SELECT * FROM public.fact_campaign_unit WHERE envelope_sha256 = :e"),
                          {"e": doc["envelope_sha256"]}).mappings().all()
    assert [dict(r) for r in runs] == [{"result_class": "DEVELOPMENT", "terminal_state": "CLOSED",
                                        "adjudication": "SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT"}]
    assert len(units) == 5
    assert not any("FAIL" in str(v).upper() for u in units for v in dict(u).values())
