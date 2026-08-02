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


def test_missing_sources_become_unavailable_not_invented(tmp_path):
    packet = collect(
        snapshot_path=tmp_path / "missing.json",
        watchdog_path=tmp_path / "missing2.json",
        social_db_path=tmp_path / "missing.sqlite",
        supervisor_url="http://127.0.0.1:1",  # unreachable
        timeout=0.2,
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
