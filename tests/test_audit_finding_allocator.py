from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
PATH = REPO / "tools/audit_finding_allocator.py"
SPEC = importlib.util.spec_from_file_location("audit_finding_allocator", PATH)
allocator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(allocator)


def git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def fid(prefix: str, date: str, serial: int) -> str:
    # Keep allocator fixtures out of the real source-tree namespace scan.
    return f"AUD-{prefix}-{date}-{serial:03d}"


def repository(path: Path) -> None:
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "audit@example.invalid")
    git(path, "config", "user.name", "Audit Test")
    (path / "finding.md").write_text(fid("F1", "20260101", 7) + "\n")
    git(path, "add", "finding.md")
    git(path, "commit", "-qm", "first")
    git(path, "branch", "other")
    (path / "finding.md").write_text(fid("GEN", "20260102", 8) + "\n")
    git(path, "commit", "-qam", "second")
    git(path, "switch", "-q", "other")
    (path / "other.md").write_text(fid("SEC", "20260103", 7) + "\n")
    git(path, "add", "other.md")
    git(path, "commit", "-qm", "collision")
    (path / "draft.md").write_text(fid("GEN", "20260103", 9) + "\n")


def test_inventory_scans_all_refs_and_untracked_worktrees(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository(workspace / "repo-a")
    report = allocator.build_inventory(workspace, tmp_path / "ledger.jsonl")
    ids = {row["finding_id"] for row in report["findings"]}
    assert ids == {
        fid("F1", "20260101", 7),
        fid("GEN", "20260102", 8),
        fid("SEC", "20260103", 7),
        fid("GEN", "20260103", 9),
    }
    assert report["next_serial"] == 10
    assert report["serial_conflicts"] == [
        {
            "serial": 7,
            "finding_ids": [
                fid("F1", "20260101", 7),
                fid("SEC", "20260103", 7),
            ],
        }
    ]


def test_reservation_is_host_locked_and_advances_ledger(tmp_path):
    inventory = {
        "workspace": str(tmp_path),
        "findings": [{"finding_id": fid("F1", "20260101", 11)}],
    }
    ledger = tmp_path / "allocations.jsonl"
    first = allocator.reserve(
        inventory=inventory,
        ledger=ledger,
        prefix="GEN",
        date="20260104",
        title="first",
        owner="test",
    )
    second = allocator.reserve(
        inventory=inventory,
        ledger=ledger,
        prefix="F2",
        date="20260104",
        title="second",
        owner="test",
    )
    assert first["finding_id"] == fid("GEN", "20260104", 12)
    assert second["finding_id"] == fid("F2", "20260104", 13)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["finding_id"] for row in rows] == [
        fid("GEN", "20260104", 12),
        fid("F2", "20260104", 13),
    ]
