#!/usr/bin/env python3
"""Safely migrate a stopped campaign supervisor to an append-only plan."""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
from pathlib import Path
from typing import Any

from app.campaign_supervisor import (
    PLAN_SCHEMA,
    PROFILE_SCHEMA,
    HistoryStore,
    _atomic_json,
    _sha256_json,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _resolve(profile_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (profile_path.parent / path).resolve()


def _profile_and_plan(profile_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any], str]:
    profile_path = profile_path.expanduser().resolve()
    profile = _load(profile_path)
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError(f"unsupported profile schema in {profile_path}")
    plan_path = _resolve(profile_path, str(profile["plan_file"]))
    plan = _load(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"unsupported plan schema in {plan_path}")
    plan_hash = _sha256_json(plan)
    expected = profile.get("expected_plan_hash")
    if expected and expected != plan_hash:
        raise ValueError(
            f"profile expected plan hash {expected}, but {plan_path} is {plan_hash}"
        )
    return profile, plan_path, plan, plan_hash


def migrate(old_profile_path: Path, new_profile_path: Path, *, apply: bool) -> dict[str, Any]:
    old_profile, old_plan_path, old_plan, old_hash = _profile_and_plan(old_profile_path)
    new_profile, new_plan_path, new_plan, new_hash = _profile_and_plan(new_profile_path)

    if old_profile["node_id"] != new_profile["node_id"]:
        raise ValueError("old and new profiles belong to different nodes")
    if old_plan.get("participants") != new_plan.get("participants"):
        raise ValueError("successor plan must preserve the participant topology exactly")
    old_jobs = old_plan.get("jobs") or []
    new_jobs = new_plan.get("jobs") or []
    if len(new_jobs) <= len(old_jobs) or new_jobs[:len(old_jobs)] != old_jobs:
        raise ValueError("successor plan must be a strict append-only extension")

    old_state_dir = _resolve(old_profile_path.resolve(), str(old_profile["state_dir"]))
    new_state_dir = _resolve(new_profile_path.resolve(), str(new_profile["state_dir"]))
    if old_state_dir != new_state_dir:
        raise ValueError("append-only migration must preserve the state directory")
    state_path = old_state_dir / "state.json"
    state = _load(state_path)
    if state.get("plan_hash") != old_hash:
        raise ValueError("persisted state does not belong to the declared old plan")
    if state.get("phase") not in {"starting", "running"}:
        raise ValueError(
            f"migration is allowed only during starting/running, got {state.get('phase')!r}"
        )
    job_index = int(state.get("job_index", -1))
    if not 0 <= job_index < len(old_jobs):
        raise ValueError("persisted job index is outside the old plan")
    if state.get("job_id") != old_jobs[job_index].get("job_id"):
        raise ValueError("persisted job id does not match the old plan")

    result = {
        "node_id": old_profile["node_id"],
        "state_path": str(state_path),
        "old_plan": str(old_plan_path),
        "new_plan": str(new_plan_path),
        "old_plan_hash": old_hash,
        "new_plan_hash": new_hash,
        "current_job_index": job_index,
        "current_job_id": state["job_id"],
        "appended_jobs": [job["job_id"] for job in new_jobs[len(old_jobs):]],
        "applied": False,
    }
    if not apply:
        return result

    lock_path = old_state_dir / "supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "campaign supervisor is still running; stop its user service before migration"
            ) from exc

        backup_path = state_path.with_name(
            f"state.before-plan-{old_hash[:12]}.json"
        )
        if not backup_path.exists():
            shutil.copy2(state_path, backup_path)
        state["plan_hash"] = new_hash
        _atomic_json(state_path, state)
        HistoryStore(old_state_dir / "campaign_history.sqlite").event(
            node_id=str(old_profile["node_id"]),
            job_id=str(state["job_id"]),
            event="campaign_plan_extended",
            detail={
                "old_plan_hash": old_hash,
                "new_plan_hash": new_hash,
                "appended_jobs": result["appended_jobs"],
                "backup_path": str(backup_path),
            },
        )
        result["backup_path"] = str(backup_path)
        result["applied"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-profile", type=Path, required=True)
    parser.add_argument("--new-profile", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = migrate(args.old_profile, args.new_profile, apply=args.apply)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
