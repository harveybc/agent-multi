import json
from pathlib import Path

import pytest

from app.campaign_supervisor import PLAN_SCHEMA, PROFILE_SCHEMA, _sha256_json
from examples.scripts.migrate_doin_campaign_plan import migrate


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    participant = {
        "node_id": "omega",
        "supervisor_url": "http://127.0.0.1:8795",
        "workers": ["omega"],
    }
    first_job = {
        "ordinal": 0,
        "job_id": "job-0",
        "domain_id": "domain-0",
        "worker_configs": {"omega": "worker-0.json"},
    }
    second_job = {
        "ordinal": 1,
        "job_id": "job-1",
        "domain_id": "domain-1",
        "worker_configs": {"omega": "worker-1.json"},
    }
    old_plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "old",
        "participants": [participant],
        "jobs": [first_job],
    }
    new_plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "new",
        "participants": [participant],
        "jobs": [first_job, second_job],
    }
    old_plan_path = tmp_path / "old-plan.json"
    new_plan_path = tmp_path / "new-plan.json"
    _write(old_plan_path, old_plan)
    _write(new_plan_path, new_plan)
    state_dir = tmp_path / "state"
    old_profile = {
        "schema_version": PROFILE_SCHEMA,
        "node_id": "omega",
        "plan_file": str(old_plan_path),
        "state_dir": str(state_dir),
        "workers": {"omega": {}},
    }
    new_profile = {
        "schema_version": PROFILE_SCHEMA,
        "node_id": "omega",
        "plan_file": str(new_plan_path),
        "expected_plan_hash": _sha256_json(new_plan),
        "state_dir": str(state_dir),
        "workers": {"omega": {}},
    }
    old_profile_path = tmp_path / "old-profile.json"
    new_profile_path = tmp_path / "new-profile.json"
    _write(old_profile_path, old_profile)
    _write(new_profile_path, new_profile)
    _write(
        state_dir / "state.json",
        {
            "schema_version": "agent_multi.doin_campaign_state.v1",
            "plan_hash": _sha256_json(old_plan),
            "node_id": "omega",
            "job_index": 0,
            "job_id": "job-0",
            "phase": "running",
            "workers": {},
            "archive": {},
            "alerts": [],
        },
    )
    return old_profile_path, new_profile_path, state_dir


def test_append_only_campaign_plan_migration_preserves_running_job(tmp_path: Path) -> None:
    old_profile, new_profile, state_dir = _fixture(tmp_path)

    preview = migrate(old_profile, new_profile, apply=False)
    assert preview["appended_jobs"] == ["job-1"]
    assert preview["applied"] is False

    applied = migrate(old_profile, new_profile, apply=True)
    state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    assert applied["applied"] is True
    assert state["job_index"] == 0
    assert state["job_id"] == "job-0"
    assert state["phase"] == "running"
    assert state["plan_hash"] == applied["new_plan_hash"]
    assert Path(applied["backup_path"]).is_file()


def test_campaign_plan_migration_rejects_changed_prefix(tmp_path: Path) -> None:
    old_profile, new_profile, _ = _fixture(tmp_path)
    plan_path = Path(json.loads(new_profile.read_text())["plan_file"])
    plan = json.loads(plan_path.read_text())
    plan["jobs"][0]["domain_id"] = "changed"
    _write(plan_path, plan)
    profile = json.loads(new_profile.read_text())
    profile["expected_plan_hash"] = _sha256_json(plan)
    _write(new_profile, profile)

    with pytest.raises(ValueError, match="append-only"):
        migrate(old_profile, new_profile, apply=False)
