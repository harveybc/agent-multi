from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

from tools import audit_snapshot_collector as collector


def git(command: list[str], cwd: Path) -> None:
    subprocess.run(["git", *command], cwd=cwd, check=True, capture_output=True)


def test_collect_repository_records_lineage_and_dirty_count(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(["init"], repo)
    git(["config", "user.email", "test@example.com"], repo)
    git(["config", "user.name", "Test"], repo)
    (repo / "tracked.txt").write_text("initial\n", encoding="utf-8")
    git(["add", "tracked.txt"], repo)
    git(["commit", "-m", "initial"], repo)
    (repo / "untracked.txt").write_text("change\n", encoding="utf-8")

    result = collector.collect_repository(tmp_path, "repo")

    assert result["available"] is True
    assert len(result["head"]) == 40
    assert result["dirty_count"] == 1


def test_compact_network_calculates_fleet_eta() -> None:
    worker = {
        "status": "running",
        "candidate_eta": {"candidates_per_hour": 2.0},
        "shared_population": {
            "generation": 1,
            "pop_size": 20,
            "evaluated": 10,
            "claimed": 2,
            "free": 8,
            "population_fingerprint": "pool",
        },
        "optimization": {
            "domains": [{
                "campaign_progress": {
                    "campaign_candidates_remaining": 10,
                }
            }]
        },
    }
    network = {
        "plan_id": "plan",
        "plan_hash": "hash",
        "participants": {
            "omega": {
                "online": True,
                "status": {"workers": {"omega": worker}},
            },
            "dragon": {
                "online": True,
                "status": {"workers": {"dragon": worker}},
            },
        },
    }

    result = collector.compact_network(network)

    assert result["eta"]["fleet_candidates_per_hour"] == 4.0
    assert result["eta"]["full_budget_remaining_seconds"] == 9000


def test_compact_brokers_excludes_raw_detail(tmp_path: Path) -> None:
    database = tmp_path / "ibkr-paper-lab.sqlite"
    connection = collector.sqlite3.connect(database)
    try:
        connection.execute(
            """
            CREATE TABLE lab_sessions (
                status TEXT,
                ended_at TEXT,
                detail_json TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO lab_sessions VALUES (?,?,?)",
            (
                "complete",
                "2026-07-30T00:00:00Z",
                json.dumps({
                    "account_fingerprint": "fingerprint",
                    "raw_account_id": "MUST_NOT_LEAK",
                    "read_only": True,
                }),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    value = {
        "generated_at": "2026-07-30T00:00:00Z",
        "ibkr": {
            "available": True,
            "complete_sessions": 1,
            "latest_complete": {
                "ended_at": "2026-07-30T00:00:00Z",
                "open_positions": 0,
                "open_orders": 0,
            },
            "socket": {"available": True},
        },
    }

    result = collector.compact_brokers(value, state_root=tmp_path)

    assert result["ibkr"]["account_fingerprint"] == "fingerprint"
    assert "MUST_NOT_LEAK" not in json.dumps(result)


def _ibkr_lab_with_observer_detail(tmp_path: Path, detail: dict) -> None:
    connection = collector.sqlite3.connect(tmp_path / "ibkr-paper-lab.sqlite")
    try:
        connection.execute(
            "CREATE TABLE lab_sessions (status TEXT, ended_at TEXT, detail_json TEXT)"
        )
        connection.execute(
            "INSERT INTO lab_sessions VALUES (?,?,?)",
            ("complete", "2026-08-10T07:00:00Z", json.dumps(detail)),
        )
        connection.commit()
    finally:
        connection.close()


def test_observer_label_never_overwrites_execution_authority(
    tmp_path: Path,
) -> None:
    """Finding 205 regression: the exact live contradiction.

    The observer adapter/lab session says read_only=true while the
    execution-runner heartbeat says read_only=false (write enabled).
    The snapshot must render BOTH facts truthfully, each with its own
    source and freshness, and write authority must come only from the
    execution heartbeat.
    """
    _ibkr_lab_with_observer_detail(tmp_path, {
        "adapter_version": "ibkr.paper.readonly.v1",
        "account_fingerprint": "fp",
        "read_only": True,
    })
    (tmp_path / "ibkr-model-runner-heartbeat.json").write_text(
        json.dumps({
            "schema": "lts.ibkr.model_runner.heartbeat.v1",
            "read_only": False,
            "observed_at": "2026-08-10T07:51:38+00:00",
            "state": "decided",
        }),
        encoding="utf-8",
    )
    (tmp_path / "alpaca-model-runner-heartbeat.json").write_text(
        json.dumps({
            "schema": "lts.alpaca.model_runner.heartbeat.v1",
            "read_only": False,
            "observed_at": "2026-08-10T07:51:32+00:00",
            "state": "monitoring",
        }),
        encoding="utf-8",
    )
    value = {
        "generated_at": "2026-08-10T07:52:21Z",
        "alpaca": {
            "available": True,
            "ended_at": "2026-08-10T07:52:21Z",
            "detail": {"adapter_version": "lts.alpaca.paper.readonly.v1"},
        },
        "ibkr": {
            "available": True,
            "latest_complete": {"ended_at": "2026-08-10T07:47:16Z"},
        },
    }

    result = collector.compact_brokers(value, state_root=tmp_path)

    for venue in ("alpaca", "ibkr"):
        authority = result[venue]["authority"]
        # Both facts are present, truthful, and separate.
        assert authority["observer_read_only"] is True
        assert authority["execution_write_enabled"] is True
        # Write authority is answered only by the execution heartbeat.
        assert authority["authoritative_source"].startswith(
            "heartbeat_file:"
        )
        # Each fact carries its own freshness.
        assert authority["observer_as_of"]
        assert authority["execution_observed_at"]
        # The legacy single ambiguous label is gone.
        assert "read_only" not in result[venue]
    assert result["ibkr"]["authority"]["observer_source"] == (
        "observer_lab_session:ibkr-paper-lab.sqlite"
    )

    summary = collector.markdown_summary({"brokers": result})
    assert "Observer read-only" in summary
    assert "Execution write" in summary


def test_missing_execution_heartbeat_is_unavailable_not_observer(
    tmp_path: Path,
) -> None:
    """Without execution evidence, write authority is 'unavailable';
    the observer label must not fill the gap."""
    _ibkr_lab_with_observer_detail(tmp_path, {"read_only": True})
    value = {
        "generated_at": "2026-08-10T07:52:21Z",
        "ibkr": {
            "available": True,
            "latest_complete": {"ended_at": "2026-08-10T07:47:16Z"},
        },
    }

    result = collector.compact_brokers(value, state_root=tmp_path)

    authority = result["ibkr"]["authority"]
    assert authority["observer_read_only"] is True
    assert authority["execution_write_enabled"] == "unavailable"
    assert authority["authoritative_source"] == "unavailable"


def test_execution_authority_falls_back_to_watchdog_embedded_copy(
    tmp_path: Path,
) -> None:
    value = {
        "generated_at": "2026-08-10T07:52:21Z",
        "ibkr": {
            "available": True,
            "latest_complete": {"ended_at": "2026-08-10T07:47:16Z"},
            "execution_runtime": {
                "schema": "lts.ibkr.model_runner.heartbeat.v1",
                "read_only": False,
                "observed_at": "2026-08-10T07:51:38+00:00",
                "state": "decided",
            },
        },
    }

    result = collector.compact_brokers(value, state_root=tmp_path)

    authority = result["ibkr"]["authority"]
    assert authority["execution_write_enabled"] is True
    assert authority["authoritative_source"] == (
        "watchdog_embedded_execution_runtime"
    )


def test_mt5_authority_reports_bridge_execution_facts(tmp_path: Path) -> None:
    value = {
        "generated_at": "2026-08-10T07:52:21Z",
        "mt5": {
            "available": True,
            "read_only": False,
            "heartbeat": {
                "received_at": "2026-08-10T07:52:06+00:00",
                "connected": True,
            },
        },
    }

    result = collector.compact_brokers(value, state_root=tmp_path)

    authority = result["mt5"]["authority"]
    assert authority["observer_read_only"] == "unavailable"
    assert authority["execution_write_enabled"] is True
    assert authority["authoritative_source"] == (
        "watchdog_mt5_operational_status"
    )
    assert authority["execution_observed_at"] == "2026-08-10T07:52:06+00:00"


def test_snapshot_hash_and_delta_are_deterministic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        collector,
        "collect_provenance",
        lambda _root: {"repo": {"head": "abc", "dirty_count": 0}},
    )
    monkeypatch.setattr(
        collector,
        "collect_runtime",
        lambda _url: {"available": True, "plan_hash": "plan"},
    )
    monkeypatch.setattr(
        collector,
        "collect_machines",
        lambda _targets, _previous: {"omega": {"reachable": True}},
    )
    monkeypatch.setattr(
        collector,
        "collect_brokers",
        lambda _path: {"available": True},
    )
    monkeypatch.setattr(
        collector,
        "collect_tests",
        lambda _path: {"available": False},
    )
    monkeypatch.setattr(
        collector,
        "collect_watchdogs",
        lambda _root, _brokers: {"active": []},
    )
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    first = collector.build_snapshot(
        workspace_root=tmp_path,
        network_url="http://unused",
        machine_targets={"omega": "local"},
        paper_watchdog=tmp_path / "paper.json",
        test_evidence=tmp_path / "tests.json",
        state_root=tmp_path,
        now=now,
    )
    second = collector.build_snapshot(
        workspace_root=tmp_path,
        network_url="http://unused",
        machine_targets={"omega": "local"},
        paper_watchdog=tmp_path / "paper.json",
        test_evidence=tmp_path / "tests.json",
        state_root=tmp_path,
        previous=first,
        now=now,
    )

    assert first["meta"]["snapshot_sha256"] == collector.snapshot_digest(first)
    assert second["delta"]["changed_sections"] == []


def test_write_snapshot_prunes_complete_pairs(tmp_path: Path) -> None:
    for index in range(4):
        packet = {
            "schema": collector.SCHEMA,
            "meta": {
                "generated_at": f"2026-07-30T0{index}:00:00Z",
                "snapshot_sha256": str(index),
            },
            "delta": {"changed_sections": []},
            "runtime": {},
            "machines": {},
            "brokers": {},
        }
        collector.write_snapshot(packet, tmp_path, retention=2)

    assert len(list(tmp_path.glob("audit_snapshot_*.json"))) == 2
    assert len(list(tmp_path.glob("audit_snapshot_*.md"))) == 2


def test_parse_machine_targets() -> None:
    assert collector.parse_machine_targets(
        ["omega=local", "dragon=dragon"]
    ) == {"omega": "local", "dragon": "dragon"}


def test_redaction_removes_secrets_and_personal_home_paths() -> None:
    value = collector.redact_value({
        "api_token": "hidden",
        "account_fingerprint": "safe",
        "error": str(Path.home() / "private/file"),
    })

    assert value["api_token"] == "[REDACTED]"
    assert value["account_fingerprint"] == "safe"
    assert value["error"] == "%h/private/file"


def test_remote_probe_uses_explicit_user_ssh_config(monkeypatch) -> None:
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        return SimpleNamespace(stdout='{"hostname":"dragon"}')

    monkeypatch.setattr(collector.subprocess, "run", run)

    result = collector.remote_machine_data("dragon")

    assert result["hostname"] == "dragon"
    assert captured["command"][0:2] == ["ssh", "-F"]
    assert captured["command"][2] == str(Path.home() / ".ssh/config")
    assert captured["input"] == collector.REMOTE_MACHINE_SCRIPT
