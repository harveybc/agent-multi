from __future__ import annotations

import json
import subprocess
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
