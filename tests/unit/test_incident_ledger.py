"""Adversarial proofs for the fleet incident ledger (Musashi order §6).

Covered here: identical-observation collapse (1), flap suppression (2),
acknowledgement semantics (3, 4), fail-closed inputs (9), restart-loop
collapse (11), schema refusal, redaction, and reopen-after-resolution.
Router-level proofs live in test_incident_router.py.
"""
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "incident_ledger", REPO_ROOT / "tools" / "incident_ledger.py")
ledger = importlib.util.module_from_spec(_SPEC)
sys.modules["incident_ledger"] = ledger
_SPEC.loader.exec_module(ledger)

NOW = datetime(2026, 8, 4, 18, 0, 0, tzinfo=timezone.utc)
CONFIG = {
    "schema": "agent_multi.incident_config.v1",
    "max_future_skew_seconds": 120,
    "max_evidence_age_seconds": 86400,
    "flap_reopen_window_seconds": 3600,
}


def _conn(tmp_path):
    return ledger.connect(tmp_path / "incidents.sqlite")


def _observe(conn, *, severity="P1", event_code="tws_unavailable",
             payload=None, now=NOW, evidence_at=None, machine="omega",
             source="tws_continuity_monitor", affected_object="-"):
    return ledger.observe(
        conn, CONFIG, source=source, front="front2", machine=machine,
        event_code=event_code, severity=severity,
        affected_object=affected_object,
        payload=payload or {"note": "probe"},
        source_evidence_at=ledger.iso(evidence_at or now),
        now=now,
    )


def _recover(conn, *, event_code="tws_unavailable", now=NOW,
             machine="omega", source="tws_continuity_monitor",
             affected_object="-", evidence=None):
    if evidence is None:
        evidence = {"direct": "fresh broker facts"}
    return ledger.recover(
        conn, CONFIG, source=source, front="front2", machine=machine,
        event_code=event_code, affected_object=affected_object,
        evidence=evidence, now=now,
    )


def test_thousand_identical_observations_one_incident(tmp_path):
    conn = _conn(tmp_path)
    for step in range(1000):
        row = _observe(conn, now=NOW + timedelta(seconds=step))
    rows = ledger.open_incidents(conn)
    assert len(rows) == 1
    assert rows[0]["occurrence_count"] == 1000
    assert row["incident_id"] == rows[0]["incident_id"]


def test_restart_loop_collapses_into_one_incident(tmp_path):
    conn = _conn(tmp_path)
    for restart in range(50):
        _observe(conn, payload={"restart_count": restart},
                 now=NOW + timedelta(seconds=20 * restart))
    rows = ledger.open_incidents(conn)
    assert len(rows) == 1
    assert rows[0]["occurrence_count"] == 50
    assert json.loads(rows[0]["payload_json"])["restart_count"] == 49


def test_flap_reopens_same_incident(tmp_path):
    conn = _conn(tmp_path)
    first = _observe(conn)
    for flap in range(10):
        moment = NOW + timedelta(minutes=flap + 1)
        assert _recover(conn, now=moment) is not None
        reopened = _observe(conn, now=moment + timedelta(seconds=30))
        assert reopened["incident_id"] == first["incident_id"]
    rows = ledger.open_incidents(conn)
    assert len(rows) == 1
    assert rows[0]["occurrence_count"] == 11


def test_reopen_after_window_creates_new_incident(tmp_path):
    conn = _conn(tmp_path)
    first = _observe(conn)
    _recover(conn, now=NOW + timedelta(minutes=1))
    later = NOW + timedelta(hours=2)
    fresh = _observe(conn, now=later, evidence_at=later)
    assert fresh["incident_id"] != first["incident_id"]
    assert fresh["occurrence_count"] == 1


def test_ack_suppresses_but_never_resolves(tmp_path):
    conn = _conn(tmp_path)
    row = _observe(conn)
    acked = ledger.acknowledge(conn, row["incident_id"], "satoshi",
                               "investigating", NOW)
    assert acked["state"] == "acknowledged"
    assert acked["resolved_at"] is None
    assert len(ledger.open_incidents(conn)) == 1   # still visible
    resolved = _recover(conn, now=NOW + timedelta(minutes=5))
    assert resolved["state"] == "resolved"
    assert resolved["resolution_evidence_hash"]


def test_ack_requires_actor_and_reason(tmp_path):
    conn = _conn(tmp_path)
    row = _observe(conn)
    with pytest.raises(ledger.Refusal):
        ledger.acknowledge(conn, row["incident_id"], "", "reason", NOW)
    with pytest.raises(ledger.Refusal):
        ledger.acknowledge(conn, row["incident_id"], "satoshi", "  ", NOW)
    with pytest.raises(ledger.Refusal):
        ledger.acknowledge(conn, "INC-unknown", "satoshi", "reason", NOW)


def test_unack_restores_active(tmp_path):
    conn = _conn(tmp_path)
    row = _observe(conn)
    ledger.acknowledge(conn, row["incident_id"], "satoshi", "looking", NOW)
    restored = ledger.unacknowledge(conn, row["incident_id"], NOW)
    assert restored["state"] == "active"
    with pytest.raises(ledger.Refusal):
        ledger.unacknowledge(conn, row["incident_id"], NOW)


def test_recovery_requires_evidence(tmp_path):
    conn = _conn(tmp_path)
    _observe(conn)
    with pytest.raises(ledger.Refusal):
        _recover(conn, evidence={})
    assert _recover(conn, event_code="never_observed") is None


def test_forged_future_timestamp_fails_closed(tmp_path):
    conn = _conn(tmp_path)
    with pytest.raises(ledger.Refusal):
        _observe(conn, evidence_at=NOW + timedelta(hours=1))
    assert ledger.open_incidents(conn) == []


def test_stale_evidence_fails_closed(tmp_path):
    conn = _conn(tmp_path)
    with pytest.raises(ledger.Refusal):
        _observe(conn, evidence_at=NOW - timedelta(days=2))
    assert ledger.open_incidents(conn) == []


def test_naive_timestamp_fails_closed(tmp_path):
    conn = _conn(tmp_path)
    with pytest.raises(ledger.Refusal):
        ledger.observe(
            conn, CONFIG, source="s", front="f", machine="m",
            event_code="e", severity="P1", affected_object="-",
            payload={}, source_evidence_at="2026-08-04T17:00:00", now=NOW)


def test_wrong_severity_and_payload_fail_closed(tmp_path):
    conn = _conn(tmp_path)
    with pytest.raises(ledger.Refusal):
        _observe(conn, severity="SEV1")
    with pytest.raises(ledger.Refusal):
        ledger.observe(
            conn, CONFIG, source="s", front="f", machine="m",
            event_code="e", severity="P1", affected_object="-",
            payload=["not", "an", "object"],
            source_evidence_at=ledger.iso(NOW), now=NOW)


def test_wrong_schema_fails_closed(tmp_path):
    db = tmp_path / "incidents.sqlite"
    conn = ledger.connect(db)
    conn.execute("UPDATE ledger_meta SET value='alien.v9'"
                 " WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(ledger.Refusal):
        ledger.connect(db)


def test_payload_redaction(tmp_path):
    conn = _conn(tmp_path)
    row = _observe(conn, payload={
        "note": "account DU1234567 rejected",
        "leak": "token: abcd1234efgh5678",
    })
    assert "DU1234567" not in row["payload_json"]
    assert "abcd1234efgh5678" not in row["payload_json"]
    assert "[REDACTED]" in row["payload_json"]


def test_severity_escalation_clears_notification_clock(tmp_path):
    conn = _conn(tmp_path)
    row = _observe(conn, severity="P1")
    ledger.mark_notified(conn, row["incident_id"], "telegram", "hash", NOW)
    escalated = _observe(conn, severity="P0",
                         now=NOW + timedelta(minutes=1))
    assert escalated["severity"] == "P0"
    assert escalated["last_notified_at"] is None
    assert escalated["notification_count"] == 1   # history preserved


def test_restart_preserves_counts_ack_and_history(tmp_path):
    db = tmp_path / "incidents.sqlite"
    conn = ledger.connect(db)
    row = ledger.observe(
        conn, CONFIG, source="s", front="f", machine="m", event_code="e",
        severity="P0", affected_object="-", payload={"n": 1},
        source_evidence_at=ledger.iso(NOW), now=NOW)
    ledger.mark_notified(conn, row["incident_id"], "telegram", "h1", NOW)
    ledger.acknowledge(conn, row["incident_id"], "musashi", "on it", NOW)
    conn.close()
    reopened = ledger.connect(db)               # simulated restart
    persisted = dict(reopened.execute(
        "SELECT * FROM incidents WHERE incident_id=?",
        (row["incident_id"],)).fetchone())
    assert persisted["notification_count"] == 1
    assert persisted["state"] == "acknowledged"
    assert persisted["acknowledged_by"] == "musashi"
    events = [r["kind"] for r in reopened.execute(
        "SELECT kind FROM incident_events ORDER BY seq")]
    assert events == ["observed", "notified", "acknowledged"]


def test_structural_sanitization_key_classes_at_any_depth(tmp_path):
    """Finding 095 (auditor counterexample): normal JSON keys like
    'secret', 'api_key', 'password' and nested 'token' must never reach
    SQLite or Telegram text unchanged — including quoted-JSON strings,
    mixed case and encoded values under secret-class keys."""
    conn = _conn(tmp_path)
    canary = "CANARY-9f3e-not-a-real-secret"
    row = _observe(conn, payload={
        "api_key": canary + "-a",
        "Password": canary + "-b",
        "nested": {"TOKEN": canary + "-c",
                   "deeper": [{"private-key": canary + "-d"}]},
        "quoted": json.dumps({"secret": canary + "-e",
                              "passphrase": canary + "-f"}),
        "account_id": canary + "-g",
        "credentials": "base64:" + canary + "-h",
        "kept": "ordinary operational text",
    })
    stored = row["payload_json"]
    assert canary not in stored
    assert "ordinary operational text" in stored
    # Journal details are sanitized too.
    events = conn.execute(
        "SELECT detail_json FROM incident_events").fetchall()
    assert all(canary not in event["detail_json"] for event in events)


def test_auditor_reproducer_json_string_redaction():
    """redact() on a JSON-shaped STRING sanitizes structurally: the
    auditor's exact case must come back changed with no test values."""
    source = json.dumps({
        "api_key": "PKTESTVALUE",
        "nested": {"token": "not-a-real-token"},
        "password": "not-a-real-password",
        "secret": "not-a-real-secret",
    }, sort_keys=True)
    redacted = ledger.redact(source)
    assert redacted != source
    assert "not-a-real-secret" not in redacted
    assert "not-a-real-token" not in redacted
    assert "PKTESTVALUE" not in redacted
    assert "nested" in redacted                    # structure preserved


def test_sanitizer_preserves_operational_keys():
    clean = ledger.sanitize_structure({
        "account_fingerprint": "c0ff137a3cc1a363",
        "incident_id": "INC-1", "payload_hash": "abc",
        "notification_count": 3, "restart_delta": 5,
        "reservation_id": "rsv-1", "summary": "TWS down",
    })
    assert clean["account_fingerprint"] == "c0ff137a3cc1a363"
    assert clean["reservation_id"] == "rsv-1"
    assert clean["summary"] == "TWS down"


def test_recovery_evidence_is_sanitized(tmp_path):
    conn = _conn(tmp_path)
    _observe(conn)
    canary = "CANARY-77aa-not-a-real-secret"
    _recover(conn, evidence={"direct": "port up",
                             "session_token": canary})
    events = conn.execute(
        "SELECT detail_json FROM incident_events WHERE kind='resolved'"
    ).fetchall()
    assert events
    assert all(canary not in event["detail_json"] for event in events)
