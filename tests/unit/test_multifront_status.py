"""Queue-state taxonomy and honesty tests for the multi-front status contract."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.multifront_status import (  # noqa: E402
    QUEUE_STATES,
    QueueStateError,
    collect,
    validate_queue,
    validate_queue_item,
)


def _base(**kw):
    item = {
        "id": "job-x",
        "state": "running",
        "hashes": {"plan_sha256": "a" * 64},
    }
    item.update(kw)
    return item


def test_canonical_states_are_exactly_the_contract_set():
    assert QUEUE_STATES == {
        "running",
        "materialized",
        "dependency_blocked",
        "proposed",
        "owner_blocked",
    }


def test_running_plus_owner_blocked_is_rejected():
    with pytest.raises(QueueStateError):
        validate_queue_item(_base(owner_blocked_reason="pending owner"))


def test_materialized_without_hashes_is_rejected():
    with pytest.raises(QueueStateError):
        validate_queue_item({"id": "j", "state": "materialized", "hashes": {}})


def test_dependency_blocked_requires_named_dependency():
    with pytest.raises(QueueStateError):
        validate_queue_item({"id": "j", "state": "dependency_blocked"})


def test_owner_blocked_requires_named_owner_reason():
    with pytest.raises(QueueStateError):
        validate_queue_item({"id": "j", "state": "owner_blocked"})


def test_unknown_state_is_rejected():
    with pytest.raises(QueueStateError):
        validate_queue_item({"id": "j", "state": "queued"})


def test_double_state_claim_is_rejected():
    with pytest.raises(QueueStateError):
        validate_queue_item(_base(also_states=["proposed"]))


def test_duplicate_ids_are_rejected():
    with pytest.raises(QueueStateError):
        validate_queue([_base(), _base()])


# ── Finding 036 regressions: the auditor's five counterexamples ──

def test_dependency_blocked_cannot_carry_owner_reason():
    with pytest.raises(QueueStateError):
        validate_queue_item(
            {"id": "j", "state": "dependency_blocked", "dependency": "x",
             "owner_blocked_reason": "y"}
        )


def test_owner_blocked_cannot_carry_dependency():
    with pytest.raises(QueueStateError):
        validate_queue_item(
            {"id": "j", "state": "owner_blocked", "owner_blocked_reason": "y",
             "dependency": "x"}
        )


def test_running_cannot_carry_dependency():
    with pytest.raises(QueueStateError):
        validate_queue_item(_base(dependency="x"))


def test_sha256_syntax_is_enforced_for_running():
    with pytest.raises(QueueStateError):
        validate_queue_item({"id": "j", "state": "running",
                             "hashes": {"plan_sha256": "x"}})


def test_sha256_syntax_is_enforced_for_materialized():
    with pytest.raises(QueueStateError):
        validate_queue_item({"id": "j", "state": "materialized",
                             "hashes": {"config_sha256": "not-a-sha"}})


def test_failed_supervisor_job_is_excluded_not_materialized(tmp_path, monkeypatch):
    import tools.multifront_status as mfs
    fake_network = {
        "plan_hash": "a" * 64,
        "plan_jobs": [
            {"job_id": "job-ok", "status": "running"},
            {"job_id": "job-bad", "status": "failed"},
            {"job_id": "job-done", "status": "completed"},
        ],
        "participants": {},
    }
    monkeypatch.setattr(
        mfs, "_get_url",
        lambda url, timeout: fake_network if url.endswith("/api/network") else None,
    )
    packet = mfs.collect(
        snapshot_path=tmp_path / "m.json",
        watchdog_path=tmp_path / "m2.json",
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://mock",
        timeout=0.1,
    )
    queue_ids = {q["id"]: q["state"] for q in packet["queue"]}
    excluded = {q["id"]: q["supervisor_status"] for q in packet["queue_excluded"]}
    assert queue_ids.get("job-ok") == "running"
    assert "job-bad" not in queue_ids and excluded["job-bad"] == "failed"
    assert "job-done" not in queue_ids and excluded["job-done"] == "completed"


# ── Finding 035 regressions: direct counts, never zero-by-absence ──

def _watchdog_fixture(orders=(3, 2, 1), drop_venue=None):
    packet = {
        "generated_at": "2026-08-01T00:00:00+00:00",
        "active_event_keys": [],
        "alpaca": {"complete_sessions": 1,
                   "detail": {"open_orders": orders[0], "open_positions": 0}},
        "ibkr": {"complete_sessions": 1,
                 "latest_complete": {"open_orders": orders[1], "open_positions": 0}},
        "mt5": {"read_only": True, "heartbeat": {"age_seconds": 5.0},
                "latest_snapshot": {"orders_total": orders[2], "positions_total": 0}},
    }
    if drop_venue:
        packet[drop_venue] = {"complete_sessions": 1}
    return packet


def test_six_open_orders_are_reported_not_masked(tmp_path):
    import json as jsonlib
    wd = tmp_path / "wd.json"
    wd.write_text(jsonlib.dumps(_watchdog_fixture()))
    packet = collect(
        snapshot_path=tmp_path / "m.json", watchdog_path=wd,
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://127.0.0.1:1", timeout=0.1,
    )
    front = packet["fronts"]["f2_business_reality"]
    assert front["open_orders"]["aggregate"] == 6
    assert front["open_orders"]["per_venue"] == {"alpaca": 3, "ibkr": 2, "mt5": 1}


def test_missing_venue_count_makes_aggregate_unavailable(tmp_path):
    import json as jsonlib
    wd = tmp_path / "wd.json"
    wd.write_text(jsonlib.dumps(_watchdog_fixture(drop_venue="ibkr")))
    packet = collect(
        snapshot_path=tmp_path / "m.json", watchdog_path=wd,
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://127.0.0.1:1", timeout=0.1,
    )
    front = packet["fronts"]["f2_business_reality"]
    assert front["open_orders"]["aggregate"] is None
    reasons = {entry["field"] for entry in packet["unavailable"]}
    assert "f2_business_reality.orders.aggregate" in reasons


# ── Finding 037 regressions: wrong-type payload honesty ──

def test_wrong_type_snapshot_becomes_unavailable_not_crash(tmp_path):
    bad = tmp_path / "snap.json"
    bad.write_text("[{}]")
    packet = collect(
        snapshot_path=bad, watchdog_path=tmp_path / "m.json",
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://127.0.0.1:1", timeout=0.1,
    )
    fields = {entry["field"] for entry in packet["unavailable"]}
    assert "f4_audit_evidence" in fields


def test_truthy_list_supervisor_status_degrades_not_crash(tmp_path, monkeypatch):
    """Musashi reproduction 1: truthy-list /api/status -> AttributeError."""
    import tools.multifront_status as mfs
    monkeypatch.setattr(
        mfs, "_get_url",
        lambda url, timeout: [{"workers": {}}] if url.endswith("/api/status") else None,
    )
    packet = mfs.collect(
        snapshot_path=tmp_path / "m.json", watchdog_path=tmp_path / "m2.json",
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://mock", timeout=0.1,
    )
    fields = {entry["field"] for entry in packet["unavailable"]}
    assert "f1_optimization" in fields


def test_non_numeric_direct_count_degrades_not_crash(tmp_path):
    """Musashi reproduction 2: non-numeric direct count -> ValueError."""
    import json as jsonlib
    fixture = _watchdog_fixture()
    fixture["alpaca"]["detail"]["open_orders"] = "three"
    fixture["mt5"]["latest_snapshot"]["orders_total"] = True
    wd = tmp_path / "wd.json"
    wd.write_text(jsonlib.dumps(fixture))
    packet = collect(
        snapshot_path=tmp_path / "m.json", watchdog_path=wd,
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://127.0.0.1:1", timeout=0.1,
    )
    front = packet["fronts"]["f2_business_reality"]
    assert front["open_orders"]["aggregate"] is None
    assert front["open_orders"]["per_venue"]["alpaca"] is None
    assert front["open_orders"]["per_venue"]["mt5"] is None  # bool is not a count
    reasons = {
        e["field"]: e["reason"] for e in packet["unavailable"]
    }
    assert "non-numeric" in reasons["f2_business_reality.orders.aggregate"]


def test_wrong_type_plan_job_is_excluded_not_crash(tmp_path, monkeypatch):
    """Musashi reproduction 3: string plan-job element -> AttributeError."""
    import tools.multifront_status as mfs
    fake_network = {
        "plan_hash": "a" * 64,
        "plan_jobs": ["not-a-job-object", {"job_id": "job-ok", "status": "running"}],
        "participants": {},
    }
    monkeypatch.setattr(
        mfs, "_get_url",
        lambda url, timeout: fake_network if url.endswith("/api/network") else None,
    )
    packet = mfs.collect(
        snapshot_path=tmp_path / "m.json", watchdog_path=tmp_path / "m2.json",
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://mock", timeout=0.1,
    )
    queue_ids = {q["id"] for q in packet["queue"]}
    assert "job-ok" in queue_ids
    excluded = {q["supervisor_status"] for q in packet["queue_excluded"]}
    assert "wrong_type" in excluded


def test_truthy_wrong_type_venue_section_degrades_not_crash(tmp_path):
    """Musashi-demanded fourth regression: truthy non-dict venue section."""
    import json as jsonlib
    fixture = _watchdog_fixture()
    fixture["alpaca"] = ["truthy", "list"]
    fixture["mt5"]["latest_snapshot"] = "truthy-string"
    wd = tmp_path / "wd.json"
    wd.write_text(jsonlib.dumps(fixture))
    packet = collect(
        snapshot_path=tmp_path / "m.json", watchdog_path=wd,
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://127.0.0.1:1", timeout=0.1,
    )
    front = packet["fronts"]["f2_business_reality"]
    assert front["open_orders"]["aggregate"] is None
    assert front["open_orders"]["per_venue"]["alpaca"] is None
    assert front["open_orders"]["per_venue"]["mt5"] is None
    fields = {e["field"] for e in packet["unavailable"]}
    assert "f2_business_reality.orders.aggregate" in fields
    assert "f2_business_reality.positions.aggregate" in fields


def test_malformed_live_plan_hash_excludes_job_not_raises(tmp_path, monkeypatch):
    """A running supervisor job with a malformed plan hash cannot make our
    own taxonomy validator raise inside collect()."""
    import tools.multifront_status as mfs
    fake_network = {
        "plan_hash": "not-a-sha",
        "plan_jobs": [{"job_id": "job-ok", "status": "running"}],
        "participants": {},
    }
    monkeypatch.setattr(
        mfs, "_get_url",
        lambda url, timeout: fake_network if url.endswith("/api/network") else None,
    )
    packet = mfs.collect(
        snapshot_path=tmp_path / "m.json", watchdog_path=tmp_path / "m2.json",
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://mock", timeout=0.1,
    )
    queue_ids = {q["id"] for q in packet["queue"]}
    assert "job-ok" not in queue_ids
    reasons = {q["reason"] for q in packet["queue_excluded"]}
    assert any("malformed" in r for r in reasons)


def test_per_account_stats_are_exposed_without_balances(tmp_path):
    import json as jsonlib
    fixture = _watchdog_fixture()
    fixture["alpaca"]["detail"].update({
        "account_fingerprint": "fp-alp", "environment": "paper",
        "account_status": "ACTIVE", "account_shorting_enabled": True,
        "protected_execution_eligible": False, "quotes_received": 42,
        "orders_submitted": 0, "balance": 99999.0,  # must NOT surface
    })
    fixture["mt5"].update({
        "heartbeat": {"age_seconds": 5.0, "environment": "demo",
                      "connected": True, "terminal_build": 6090,
                      "trade_allowed": True},
        "counts": {"heartbeats": 5113},
    })
    wd = tmp_path / "wd.json"
    wd.write_text(jsonlib.dumps(fixture))
    execution_dir = tmp_path / "exec-state"
    execution_dir.mkdir()
    (execution_dir / "alpaca-model-runner-heartbeat.json").write_text(
        jsonlib.dumps({
            "observed_at": "2026-08-05T03:00:00+00:00",
            "state": "monitoring", "read_only": False,
            "environment": "paper", "account_fingerprint": "fp-alp-exec",
            "model_id": "spy-daily-linear-live-v1",
        }))
    packet = collect(
        snapshot_path=tmp_path / "m.json", watchdog_path=wd,
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://127.0.0.1:1", timeout=0.1,
        l0_heartbeat_path=tmp_path / "no-hb.json",
        l0_db_path=tmp_path / "no-l0.sqlite",
        execution_state_dir=execution_dir,
    )
    accounts = packet["fronts"]["f2_business_reality"]["accounts"]
    # Finding 098: identity and mode come from the EXECUTION heartbeat,
    # not the old preflight labels (which survive as observer_* context).
    assert accounts["alpaca_paper"]["account_fingerprint"] == "fp-alp-exec"
    assert accounts["alpaca_paper"]["mode"] == "write_enabled"
    assert accounts["alpaca_paper"]["observer_quotes_received"]["value"] == 42
    assert accounts["oanda_mt5_demo"]["environment"] == "demo"
    # No execution heartbeat for IBKR in this fixture: unknown, never a
    # fabricated read-only label.
    assert accounts["ibkr_paper"]["mode"] == "unknown"
    assert "99999" not in json_dumps_all(accounts)  # injected balance stays out


def json_dumps_all(value):
    import json as jsonlib
    return jsonlib.dumps(value)


def test_l0_runner_heartbeat_and_ledger_are_exposed(tmp_path):
    import json as jsonlib
    import sqlite3 as sqlite3lib
    hb = tmp_path / "heartbeat.json"
    hb.write_text(jsonlib.dumps({
        "at": "2026-08-02T21:00:00+00:00", "outcome": "would_be_order",
        "halt_state": "none", "capability_evidence": "synthetic_fixture",
        "network_submissions_session": 0,
    }))
    db = tmp_path / "l0.sqlite"
    con = sqlite3lib.connect(str(db))
    con.execute("CREATE TABLE decisions (idempotency_key TEXT, outcome TEXT)")
    con.execute("INSERT INTO decisions VALUES ('k1', 'would_be_order')")
    con.execute("CREATE TABLE lifecycle_events (seq INTEGER)")
    con.execute("INSERT INTO lifecycle_events VALUES (1)")
    con.commit()
    con.close()
    packet = collect(
        snapshot_path=tmp_path / "m.json", watchdog_path=tmp_path / "m2.json",
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://127.0.0.1:1", timeout=0.1,
        l0_heartbeat_path=hb, l0_db_path=db,
    )
    l0 = packet["fronts"]["f2_business_reality"]["l0_demo_execution"]
    assert l0["last_outcome"] == "would_be_order"
    assert l0["network_submissions"]["value"] == 0
    assert l0["capability_evidence"] == "synthetic_fixture"
    assert l0["ledger"]["value"]["would_be_orders"] == 1
    names = {source["name"] for source in packet["sources"]}
    assert "l0_demo_execution" in names


def test_missing_l0_heartbeat_degrades_explicitly(tmp_path):
    packet = collect(
        snapshot_path=tmp_path / "m.json", watchdog_path=tmp_path / "m2.json",
        social_db_path=tmp_path / "m.sqlite",
        supervisor_url="http://127.0.0.1:1", timeout=0.1,
        l0_heartbeat_path=tmp_path / "absent.json",
        l0_db_path=tmp_path / "absent.sqlite",
    )
    fields = {entry["field"] for entry in packet["unavailable"]}
    assert "f2_business_reality.l0_demo_execution" in fields


def test_missing_sources_become_unavailable_not_invented(tmp_path):
    packet = collect(
        snapshot_path=tmp_path / "missing.json",
        watchdog_path=tmp_path / "missing2.json",
        social_db_path=tmp_path / "missing.sqlite",
        supervisor_url="http://127.0.0.1:1",  # unreachable
        timeout=0.2,
        l0_heartbeat_path=tmp_path / "missing-hb.json",
        l0_db_path=tmp_path / "missing-l0.sqlite",
    )
    missing_fields = {entry["field"] for entry in packet["unavailable"]}
    assert "f1_optimization" in missing_fields
    assert "f2_business_reality" in missing_fields
    assert "f4_audit_evidence" in missing_fields
    # queue still validates with only static gate entries
    states = {item["state"] for item in packet["queue"]}
    assert states <= QUEUE_STATES
    # nothing invented: fronts that failed carry no fabricated values
    assert "f2_business_reality" not in packet["fronts"]


def test_l1_queue_names_the_current_execution_dependency(tmp_path):
    packet = collect(
        snapshot_path=tmp_path / "missing.json",
        watchdog_path=tmp_path / "missing2.json",
        social_db_path=tmp_path / "missing.sqlite",
        supervisor_url="http://127.0.0.1:1",
        timeout=0.1,
        l0_heartbeat_path=tmp_path / "missing-hb.json",
        l0_db_path=tmp_path / "missing-l0.sqlite",
    )
    l1 = next(item for item in packet["queue"]
              if item["id"] == "ibkr-paper-l1-canary")
    assert l1["state"] == "dependency_blocked"
    assert "IBKR write adapter" in l1["dependency"]
    assert "owner single-use activation" in l1["dependency"]
    assert "24-hour" not in l1["dependency"]
