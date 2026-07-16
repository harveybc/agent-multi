from __future__ import annotations

from tools.swarm_telegram_watchdog import (
    collect_global_events,
    collect_local_process_events,
    due_events,
    completion_key,
    format_completion,
    normalize_completion_records,
    select_notification_owner,
    split_telegram_text,
)


PLAN = {
    "participants": [
        {"node_id": "omega"},
        {"node_id": "dragon"},
        {"node_id": "gamma"},
    ],
}


def status(node: str, *, updated: str, job: str = "job-a") -> dict:
    return {
        "node_id": node,
        "updated_at": updated,
        "phase": "running",
        "plan_hash": "plan",
        "job_id": job,
        "domain_id": "domain-a",
        "alerts": [],
        "workers": {
            node: {
                "status": "running",
                "api_error": None,
                "last_seen": updated,
                "bootstrap_evidence": {
                    "genesis_hash": "genesis",
                    "population_block_hash": "population",
                },
                "shared_population": {"generation": 1, "pop_size": 20},
            }
        },
    }


def test_selects_first_fresh_online_participant() -> None:
    snapshot = {
        "participants": {
            "omega": {"online": False},
            "dragon": {
                "online": True,
                "status": status(
                    "dragon", updated="2026-07-16T20:00:00+00:00"
                ),
            },
            "gamma": {
                "online": True,
                "status": status(
                    "gamma", updated="2026-07-16T20:00:00+00:00"
                ),
            },
        }
    }
    assert select_notification_owner(
        PLAN,
        snapshot,
        now=1784232300,
        stale_seconds=600,
    ) == "dragon"


def test_detects_offline_machine_and_parallel_jobs() -> None:
    updated = "2026-07-16T20:00:00+00:00"
    snapshot = {
        "participants": {
            "omega": {"online": True, "status": status("omega", updated=updated)},
            "dragon": {
                "online": True,
                "status": status("dragon", updated=updated, job="job-b"),
            },
            "gamma": {"online": False, "error": "timeout"},
        }
    }
    events = collect_global_events(
        PLAN,
        snapshot,
        now=1784232300,
        stale_seconds=600,
    )
    keys = {item["key"] for item in events}
    assert "participant_offline:gamma" in keys
    assert "parallel_swarm:job_id" in keys


def test_detects_local_duplicate_or_extra_process() -> None:
    events = collect_local_process_events(
        node_id="omega",
        phase="running",
        expected_configs={"/repo/expected.json"},
        processes=[
            {
                "pid": 1,
                "config_path": "/repo/expected.json",
                "domain_id": "domain-a",
            },
            {
                "pid": 2,
                "config_path": "/repo/other.json",
                "domain_id": "domain-b",
            },
        ],
    )
    assert events[0]["key"] == "local_parallel_or_missing:omega"
    assert "LOCAL DOIN PROCESS ANOMALY" in events[0]["message"]


def test_event_grace_repeat_and_recovery() -> None:
    state = {"events": {}}
    current = [{
        "key": "problem",
        "message": "problem detail",
        "severity": "error",
        "grace_seconds": 300,
    }]
    messages, sent, recovered = due_events(
        current, state, now=1000, repeat_seconds=3600
    )
    assert messages == []
    assert sent == []
    assert recovered == []
    messages, sent, recovered = due_events(
        current, state, now=1300, repeat_seconds=3600
    )
    assert messages == ["problem detail"]
    assert sent == ["problem"]
    state["events"]["problem"]["last_sent_at"] = 1300
    messages, sent, recovered = due_events(
        [], state, now=1400, repeat_seconds=3600
    )
    assert "RECOVERED" in messages[0]
    assert sent == []
    assert recovered == ["problem"]


def test_completion_message_labels_metric_period() -> None:
    message = format_completion(
        {
            "job_id": "sol-4h",
            "domain_id": "domain",
            "ordinal": 0,
            "completed_at": "now",
            "champion_fitness": 0.067,
            "champion_peer_id": "peer",
            "artifact_sha256": "abc",
            "artifact_format": "zip",
            "metrics": {
                "optimization_metric": "train_validation_l1_score",
                "total_return": 0.28,
                "risk_adjusted_total_return": 0.26,
                "max_drawdown_pct": 2.3,
            },
        },
        [{"ordinal": 1, "job_id": "btc-1h"}],
    )
    assert "fitness (train_validation_l1_score)" in message
    assert "total return: +28.0000%" in message
    assert "risk-adjusted total return: +26.0000%" in message
    assert "next job: btc-1h" in message
    assert "not annualized unless explicitly labeled annual" in message


def test_telegram_messages_are_split_below_limit() -> None:
    chunks = split_telegram_text(("alert detail\n\n" * 500), limit=300)
    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_completion_identity_is_shared_across_local_completion_times() -> None:
    first = {
        "job_id": "job-a",
        "artifact_sha256": "artifact",
        "completed_at": "2026-07-16T20:00:00+00:00",
    }
    second = {**first, "completed_at": "2026-07-16T20:00:02+00:00"}
    assert completion_key(first) == completion_key(second)


def test_migrates_old_completion_keys_without_losing_notification() -> None:
    records = {
        "job-a:artifact:2026-07-16T20:00:00+00:00": {
            "notified_at": 1000,
        },
        "job-a:artifact:2026-07-16T20:00:02+00:00": {
            "observed_with_owner_at": 1002,
        },
    }
    normalize_completion_records(records)
    assert records == {
        "job-a:artifact": {
            "notified_at": 1000,
            "observed_with_owner_at": 1002,
        }
    }
