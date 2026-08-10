"""Suite-wide hermeticity guard (order §7/WP6, audit 2026-08-09).

The complete suite must leave the subject checkout AND every sibling
repository fixture byte-clean: no new, modified or deleted files
relative to the state at session start. Production materialization is
an explicit operator command, never a unit-test side effect — this
guard makes any regression of that rule a suite failure.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SIBLING_FIXTURES = (REPO.parent / "doin-node",)


def _tree_state(root: Path) -> str | None:
    if not root.is_dir():
        return None
    proc = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain",
         "--untracked-files=all"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout


@pytest.fixture(scope="session", autouse=True)
def suite_leaves_checkouts_byte_clean():
    roots = [REPO, *SIBLING_FIXTURES]
    before = {str(root): _tree_state(root) for root in roots}
    yield
    dirt = []
    for root in roots:
        after = _tree_state(root)
        if before[str(root)] is None or after is None:
            continue
        if after != before[str(root)]:
            old = set(before[str(root)].splitlines())
            new_lines = [line for line in after.splitlines()
                         if line not in old]
            removed = [line for line in old
                       if line not in set(after.splitlines())]
            dirt.append(f"{root}: +{new_lines!r} -{removed!r}")
    assert not dirt, (
        "the suite dirtied a subject or sibling checkout — tests must "
        "write only below tmp_path (order §7/WP6): " + "; ".join(dirt))
