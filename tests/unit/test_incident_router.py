"""Adversarial proofs for the notification router (Musashi order §6).

Covered here: one activation per incident (1), no flap storm (2), ack
suppression with visible active status (3), recovery-resolves semantics
(4), P0 preemption and digest exclusion (5), stale-backlog collapse (6),
restart preservation (7), owner failover with bounded duplicates and no
replay after recovery (8).
"""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for name in ("incident_ledger", "incident_router"):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
ledger = sys.modules["incident_ledger"]
router = sys.modules["incident_router"]

NOW = datetime(2026, 8, 4, 18, 0, 0, tzinfo=timezone.utc)
CONFIG = {
    "schema": "agent_multi.incident_config.v1",
    "max_future_skew_seconds": 120,
    "max_evidence_age_seconds": 86400,
    "flap_reopen_window_seconds": 3600,
    "notification_owner": "omega",
    "receipt_deadline_seconds": {"P0": 60, "P1": 120, "P2": 900,
                                 "P3": 86400},
    "severity_policy": {
        "P0": {"first_within_seconds": 60, "reminders_seconds": [900],
               "repeat_seconds": 3600},
        "P1": {"first_within_seconds": 120, "reminders_seconds": [],
               "repeat_seconds": 7200},
        "P2": {"first_within_seconds": 0, "reminders_seconds": [],
               "repeat_seconds": 21600},
        "P3": {"first_within_seconds": None, "reminders_seconds": [],
               "repeat_seconds": None},
    },
    "digest_hour_utc": 12,
    "recovery_message_severities": ["P0", "P1"],
    "message_max_chars": 3500,
}


class Capture:
    def __init__(self, fail=False):
        self.messages = []
        self.fail = fail

    def __call__(self, message, config=None):
        if self.fail:
            raise RuntimeError("transport down")
        self.messages.append(message)


def _conn(tmp_path):
    return ledger.connect(tmp_path / "incidents.sqlite")


def _observe(conn, *, severity="P0", event_code="tws_unavailable",
             now=NOW, machine="omega", payload=None):
    return ledger.observe(
        conn, CONFIG, source="monitor", front="front2", machine=machine,
        event_code=event_code, severity=severity, affected_object="-",
        payload=payload or {"operator_action": "authenticate TWS Paper"},
        source_evidence_at=ledger.iso(now), now=now)


def _recover(conn, *, event_code="tws_unavailable", now=NOW,
             machine="omega"):
    document = ledger.build_recovery_evidence(
        source="monitor", front="front2", machine=machine,
        event_code=event_code, affected_object="-",
        state={"direct": "port and broker reconciled"},
        observed_at=ledger.iso(now))
    return ledger.recover(
        conn, CONFIG, source="monitor", front="front2", machine=machine,
        event_code=event_code, affected_object="-",
        evidence=document, now=now)


def _pass(conn, *, hostname="omega", now=NOW, transport=None,
          forwarder=None, recovery_forwarder=None, receipt=None):
    transport = transport if transport is not None else Capture()
    receipt_query = receipt if callable(receipt) else (
        lambda i, c: receipt)
    actions = router.run_pass(
        conn, CONFIG, hostname=hostname, now=now, transport=transport,
        forwarder=forwarder or (lambda i, c: None),
        recovery_forwarder=recovery_forwarder or (lambda i, c: None),
        receipt_query=receipt_query)
    return actions, transport


def test_thousand_observations_one_activation_message(tmp_path):
    conn = _conn(tmp_path)
    for step in range(1000):
        _observe(conn, now=NOW + timedelta(seconds=step))
    actions, transport = _pass(conn, now=NOW + timedelta(seconds=1001))
    assert len(transport.messages) == 1
    assert [a["action"] for a in actions] == ["notified"]
    # A second immediate pass sends nothing: the reminder is not yet due.
    actions, transport = _pass(conn, now=NOW + timedelta(seconds=1030))
    assert transport.messages == []


def test_reminder_schedule_p0(tmp_path):
    conn = _conn(tmp_path)
    _observe(conn)
    _, first = _pass(conn, now=NOW)
    assert len(first.messages) == 1
    _, quiet = _pass(conn, now=NOW + timedelta(minutes=10))
    assert quiet.messages == []                     # before 15-min reminder
    _, reminder = _pass(conn, now=NOW + timedelta(minutes=16))
    assert len(reminder.messages) == 1              # 15-min reminder
    _, hourly = _pass(conn, now=NOW + timedelta(minutes=16 + 61))
    assert len(hourly.messages) == 1                # hourly thereafter
    _, early = _pass(conn, now=NOW + timedelta(minutes=16 + 61 + 30))
    assert early.messages == []


def test_flapping_ten_times_is_not_a_storm(tmp_path):
    conn = _conn(tmp_path)
    _observe(conn)
    _pass(conn, now=NOW)                            # one activation
    total = Capture()
    for flap in range(10):
        moment = NOW + timedelta(minutes=1 + flap)
        _recover(conn, now=moment)
        _observe(conn, now=moment + timedelta(seconds=20))
        actions = router.run_pass(
            conn, CONFIG, hostname="omega",
            now=moment + timedelta(seconds=40), transport=total,
            forwarder=lambda i, c: None,
            recovery_forwarder=lambda i, c: None)
    # Ten flaps may produce at most the pending recovery notice for the
    # delivered P0 activation — never one message per flap.
    assert len(total.messages) <= 1


def test_ack_suppresses_reminders_status_stays_visible(tmp_path):
    conn = _conn(tmp_path)
    row = _observe(conn)
    _pass(conn, now=NOW)
    ledger.acknowledge(conn, row["incident_id"], "owner", "restarting TWS",
                       NOW + timedelta(minutes=5))
    _, transport = _pass(conn, now=NOW + timedelta(hours=3))
    assert transport.messages == []                 # no reminders
    active = ledger.open_incidents(conn)
    assert len(active) == 1                         # still visible
    assert active[0]["state"] == "acknowledged"


def test_recovery_resolves_and_sends_single_recovery_message(tmp_path):
    conn = _conn(tmp_path)
    _observe(conn)
    _pass(conn, now=NOW)
    _recover(conn, now=NOW + timedelta(minutes=30))
    _, transport = _pass(conn, now=NOW + timedelta(minutes=31))
    assert len(transport.messages) == 1
    assert "RECOVERED" in transport.messages[0]
    _, again = _pass(conn, now=NOW + timedelta(minutes=32))
    assert again.messages == []                     # exactly once


def test_undelivered_or_low_priority_recovery_is_silent(tmp_path):
    conn = _conn(tmp_path)
    _observe(conn, severity="P2", event_code="gpu_hot")
    _recover(conn, event_code="gpu_hot", now=NOW + timedelta(minutes=1))
    _observe(conn, severity="P0", event_code="never_notified")
    _recover(conn, event_code="never_notified",
             now=NOW + timedelta(minutes=2))
    _, transport = _pass(conn, now=NOW + timedelta(minutes=3))
    assert all("RECOVERED" not in m for m in transport.messages)


def test_p0_preempts_and_never_hides_in_digest(tmp_path):
    conn = _conn(tmp_path)
    digest_time = NOW.replace(hour=13)
    for index in range(3):
        _observe(conn, severity="P3", event_code=f"social_degraded_{index}",
                 now=digest_time)
    _observe(conn, severity="P0", now=digest_time)
    actions, transport = _pass(conn, now=digest_time + timedelta(minutes=1))
    assert "[P0]" in transport.messages[0]          # P0 sent first
    digest = [m for m in transport.messages if "digest" in m]
    assert len(digest) == 1
    assert "tws_unavailable" not in digest[0]       # P0 never in digest
    assert "social_degraded_0" in digest[0]


def test_digest_once_per_day_and_only_after_hour(tmp_path):
    conn = _conn(tmp_path)
    _observe(conn, severity="P3", event_code="collector_degraded", now=NOW)
    early = NOW.replace(hour=9)
    _, transport = _pass(conn, now=early)
    assert transport.messages == []
    late = NOW.replace(hour=13)
    _, transport = _pass(conn, now=late)
    assert len(transport.messages) == 1
    _, transport = _pass(conn, now=late + timedelta(hours=2))
    assert transport.messages == []                 # once per day


def test_stale_backlog_collapses_to_current_state(tmp_path):
    conn = _conn(tmp_path)
    # 500 raw observations across 5 identities accumulated while the
    # router was down: one pass emits exactly 5 activation messages.
    for index in range(500):
        _observe(conn, severity="P1", event_code=f"code_{index % 5}",
                 now=NOW + timedelta(seconds=index))
    _, transport = _pass(conn, now=NOW + timedelta(seconds=600))
    assert len(transport.messages) == 5


def test_router_restart_preserves_history_and_sends_nothing_new(tmp_path):
    db = tmp_path / "incidents.sqlite"
    conn = ledger.connect(db)
    row = _observe(conn)
    _pass(conn, now=NOW)
    conn.close()
    conn = ledger.connect(db)                       # restart
    _, transport = _pass(conn, now=NOW + timedelta(minutes=5))
    assert transport.messages == []                 # no replay
    persisted = dict(conn.execute(
        "SELECT * FROM incidents WHERE incident_id=?",
        (row["incident_id"],)).fetchone())
    assert persisted["notification_count"] == 1


def test_activation_message_is_redacted_and_actionable(tmp_path):
    conn = _conn(tmp_path)
    _observe(conn, payload={
        "operator_action": "authenticate TWS Paper on Omega",
        "exposure_state": "unknown after risk-reducing action",
        "leak": "account DU9999999",
    })
    _, transport = _pass(conn, now=NOW)
    message = transport.messages[0]
    assert "DU9999999" not in message
    assert "authenticate TWS Paper" in message
    assert "incident: INC-" in message


def test_nonowner_forwards_recovery_for_every_severity(tmp_path):
    """Resolution sync is independent of the recovery-message policy: a
    P2 resolved on a worker must resolve on the owner too, exactly once
    (found live during fleet acceptance on 2026-08-04)."""
    conn = _conn(tmp_path)
    _observe(conn, severity="P2", event_code="fleet_p2", machine="dragon")
    _pass(conn, hostname="dragon", now=NOW)          # forwarded (handoff)
    _recover(conn, event_code="fleet_p2", machine="dragon",
             now=NOW + timedelta(minutes=5))
    forwards = []
    actions, transport = _pass(
        conn, hostname="dragon", now=NOW + timedelta(minutes=6),
        recovery_forwarder=lambda i, c: forwards.append(i["incident_id"]))
    assert [a["action"] for a in actions] == ["recovery_forwarded"]
    assert len(forwards) == 1
    assert transport.messages == []                  # no Telegram for P2
    actions, _ = _pass(
        conn, hostname="dragon", now=NOW + timedelta(minutes=7),
        recovery_forwarder=lambda i, c: forwards.append(i["incident_id"]))
    assert actions == []                             # exactly once


def test_no_canary_survives_into_telegram_text(tmp_path):
    """Finding 095 end-to-end: a secret-shaped payload observed into the
    ledger never reaches the formatted Telegram message."""
    conn = _conn(tmp_path)
    canary = "CANARY-5b1c-not-a-real-secret"
    ledger.observe(
        conn, CONFIG, source="monitor", front="front2", machine="omega",
        event_code="leaky_event", severity="P0", affected_object="-",
        payload={"api_key": canary, "summary": "incident with secrets",
                 "nested": {"token": canary},
                 "operator_action": "act"},
        source_evidence_at=ledger.iso(NOW), now=NOW)
    _, transport = _pass(conn, now=NOW)
    assert len(transport.messages) == 1
    assert canary not in transport.messages[0]
    assert "incident with secrets" in transport.messages[0]


def _receipt(delivered_at=None, channel="telegram", message_hash="mh"):
    return {"delivered_at": delivered_at, "channel": channel,
            "message_hash": message_hash}


def test_receipt_scenario_owner_healthy(tmp_path):
    """097(a): forward, then confirm from the owner's end-to-end receipt;
    no local Telegram, no failover, notified only via receipt."""
    conn = _conn(tmp_path)
    _observe(conn, machine="dragon")
    actions, transport = _pass(conn, hostname="dragon", now=NOW,
                               receipt=None)
    assert [a["action"] for a in actions] == ["forwarded"]
    assert transport.messages == []
    delivered = ledger.iso(NOW + timedelta(seconds=20))
    actions, transport = _pass(
        conn, hostname="dragon", now=NOW + timedelta(seconds=30),
        receipt=_receipt(delivered_at=delivered))
    assert [a["action"] for a in actions] == ["receipt_confirmed"]
    assert transport.messages == []
    row = ledger.open_incidents(conn)[0]
    assert row["notification_count"] == 1
    assert row["last_notified_at"] == delivered


def test_receipt_scenario_ssh_up_router_down(tmp_path):
    """097(b/c): forwards succeed (or Telegram is down on the owner) but
    no receipt ever arrives — the worker fails over at the P0 deadline
    with the same incident id."""
    conn = _conn(tmp_path)
    _observe(conn, machine="dragon")
    actions, transport = _pass(conn, hostname="dragon", now=NOW,
                               receipt=None)
    assert transport.messages == []                 # not yet overdue... 
    # 61 s after first observation with no receipt: P0 deadline exceeded.
    actions, transport = _pass(
        conn, hostname="dragon", now=NOW + timedelta(seconds=61),
        receipt=_receipt(delivered_at=None))
    assert "failover_notified" in [a["action"] for a in actions]
    assert len(transport.messages) == 1
    incident_id = ledger.open_incidents(conn)[0]["incident_id"]
    assert incident_id in transport.messages[0]


def test_receipt_scenario_owner_unreachable(tmp_path):
    """097(d): forward fails AND the receipt query is unknown — failover
    still fires at the deadline; before it, no local message."""
    conn = _conn(tmp_path)
    _observe(conn, machine="dragon")

    def bad_forward(incident, config):
        raise RuntimeError("owner unreachable")

    actions, transport = _pass(conn, hostname="dragon", now=NOW,
                               forwarder=bad_forward, receipt=None)
    assert [a["action"] for a in actions] == ["forward_failed"]
    assert transport.messages == []
    actions, transport = _pass(
        conn, hostname="dragon", now=NOW + timedelta(seconds=90),
        forwarder=bad_forward, receipt=None)
    assert "failover_notified" in [a["action"] for a in actions]
    assert len(transport.messages) == 1


def test_receipt_scenario_delayed_receipt_prevents_failover(tmp_path):
    """097(e): a receipt arriving before the deadline prevents failover."""
    conn = _conn(tmp_path)
    _observe(conn, machine="dragon")
    _pass(conn, hostname="dragon", now=NOW, receipt=None)
    delivered = ledger.iso(NOW + timedelta(seconds=45))
    actions, transport = _pass(
        conn, hostname="dragon", now=NOW + timedelta(seconds=55),
        receipt=_receipt(delivered_at=delivered))
    assert [a["action"] for a in actions] == ["receipt_confirmed"]
    assert transport.messages == []
    # Long after: due again per policy, but the deadline clock restarts
    # from the confirmed delivery + reminder interval.
    actions, transport = _pass(
        conn, hostname="dragon",
        now=NOW + timedelta(seconds=45 + 900 + 30),
        receipt=_receipt(delivered_at=delivered))
    assert "failover_notified" not in [a["action"] for a in actions]
    assert transport.messages == []


def test_receipt_scenario_recovery_race(tmp_path):
    """097(f): an incident resolved locally while its receipt is pending
    forwards its resolution and never fails over."""
    conn = _conn(tmp_path)
    _observe(conn, machine="dragon")
    _pass(conn, hostname="dragon", now=NOW, receipt=None)
    _recover(conn, machine="dragon", now=NOW + timedelta(seconds=30))
    forwards = []
    actions, transport = _pass(
        conn, hostname="dragon", now=NOW + timedelta(seconds=90),
        receipt=None,
        recovery_forwarder=lambda i, c: forwards.append(i["incident_id"]))
    assert [a["action"] for a in actions] == ["recovery_forwarded"]
    assert transport.messages == []
    assert len(forwards) == 1


def test_worker_never_marks_notified_on_forward(tmp_path):
    """097 core assertion: SSH ingestion is not delivery."""
    conn = _conn(tmp_path)
    _observe(conn, machine="dragon")
    _pass(conn, hostname="dragon", now=NOW, receipt=None)
    row = ledger.open_incidents(conn)[0]
    assert row["notification_count"] == 0
    assert row["last_notified_at"] is None
    assert row["state"] == "pending"
