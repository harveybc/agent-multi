#!/usr/bin/env python3
"""Enumerate audit finding IDs across Git refs and reserve the next serial."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA = "agent_multi.audit_finding_inventory.v1"
FINDING_RE = re.compile(r"AUD-[A-Z0-9][A-Z0-9-]*-[0-9]{8}-([0-9]{3,})")
GIT_GREP_RE = r"AUD-[A-Z0-9][A-Z0-9-]*-[0-9]{8}-[0-9]{3,}"
DEFAULT_LEDGER = (
    Path.home() / ".local/state/agent-multi/audit-finding-allocations.jsonl"
)


def _run(repo: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo}: {completed.stderr.strip()}"
        )
    return completed.stdout


def repositories(workspace: Path) -> list[Path]:
    result = []
    for path in sorted(workspace.iterdir()):
        if path.is_dir() and (path / ".git").exists():
            result.append(path)
    return result


def refs(repo: Path) -> list[str]:
    output = _run(
        repo,
        "for-each-ref",
        "--format=%(refname)",
        "refs/heads",
        "refs/remotes",
    )
    return sorted(
        line.strip()
        for line in output.splitlines()
        if line.strip() and not line.strip().endswith("/HEAD")
    )


def ids_in_text(text: str) -> set[str]:
    return {match.group(0) for match in FINDING_RE.finditer(text)}


def ids_in_ref(repo: Path, ref: str) -> set[str]:
    output = _run(
        repo,
        "grep",
        "-h",
        "-o",
        "-E",
        GIT_GREP_RE,
        ref,
        "--",
        check=False,
    )
    return ids_in_text(output)


def worktrees(repo: Path) -> list[Path]:
    output = _run(repo, "worktree", "list", "--porcelain")
    return [
        Path(line.removeprefix("worktree "))
        for line in output.splitlines()
        if line.startswith("worktree ")
    ]


def ids_in_worktree(path: Path) -> set[str]:
    if not path.exists():
        return set()
    output = _run(path, "grep", "-h", "-o", "-E", GIT_GREP_RE, "--", check=False)
    found = ids_in_text(output)
    untracked = _run(path, "ls-files", "--others", "--exclude-standard", check=False)
    for relative in untracked.splitlines():
        candidate = path / relative
        if candidate.suffix.lower() not in {".md", ".json", ".jsonl", ".py", ".csv"}:
            continue
        try:
            found.update(ids_in_text(candidate.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return found


def ledger_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        finding_id = value.get("finding_id")
        if isinstance(finding_id, str) and FINDING_RE.fullmatch(finding_id):
            found.add(finding_id)
    return found


def build_inventory(workspace: Path, ledger: Path) -> dict:
    sources: dict[str, set[str]] = defaultdict(set)
    repo_rows = []
    for repo in repositories(workspace):
        repo_refs = refs(repo)
        repo_worktrees = worktrees(repo)
        for ref in repo_refs:
            for finding_id in ids_in_ref(repo, ref):
                sources[finding_id].add(f"{repo.name}:{ref}")
        for path in repo_worktrees:
            for finding_id in ids_in_worktree(path):
                sources[finding_id].add(f"{repo.name}:worktree:{path}")
        repo_rows.append(
            {
                "repository": repo.name,
                "refs_enumerated": len(repo_refs),
                "worktrees_enumerated": len(repo_worktrees),
            }
        )
    for finding_id in ledger_ids(ledger):
        sources[finding_id].add(f"reservation-ledger:{ledger}")

    by_serial: dict[int, set[str]] = defaultdict(set)
    for finding_id in sources:
        match = FINDING_RE.fullmatch(finding_id)
        assert match is not None
        by_serial[int(match.group(1))].add(finding_id)
    conflicts = [
        {"serial": serial, "finding_ids": sorted(values)}
        for serial, values in sorted(by_serial.items())
        if len(values) > 1
    ]
    max_serial = max(by_serial, default=0)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace.resolve()),
        "enumeration": {
            "refs": "git for-each-ref refs/heads refs/remotes; git grep per ref",
            "worktrees": "git worktree list; tracked plus selected untracked text",
            "reservation_ledger": str(ledger),
        },
        "repositories": repo_rows,
        "unique_finding_ids": len(sources),
        "max_serial": max_serial,
        "next_serial": max_serial + 1,
        "serial_conflicts": conflicts,
        "findings": [
            {"finding_id": finding_id, "sources": sorted(finding_sources)}
            for finding_id, finding_sources in sorted(sources.items())
        ],
    }


def reserve(
    *,
    inventory: dict,
    ledger: Path,
    prefix: str,
    date: str,
    title: str,
    owner: str,
) -> dict:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        reserved = ids_in_text(stream.read())
        used_serials = {
            int(match.group(1))
            for value in reserved
            if (match := FINDING_RE.fullmatch(value)) is not None
        }
        used_serials.update(
            int(match.group(1))
            for row in inventory["findings"]
            if (match := FINDING_RE.fullmatch(row["finding_id"])) is not None
        )
        serial = max(used_serials, default=0) + 1
        finding_id = f"AUD-{prefix.upper()}-{date}-{serial:03d}"
        row = {
            "finding_id": finding_id,
            "reserved_at": datetime.now(timezone.utc).isoformat(),
            "title": title,
            "owner": owner,
            "workspace": inventory["workspace"],
        }
        stream.seek(0, 2)
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        return row


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--workspace", type=Path, default=Path.home() / "Documents/GitHub"
    )
    value.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    value.add_argument("--output", type=Path)
    value.add_argument("--fail-on-conflict", action="store_true")
    value.add_argument("--reserve-prefix")
    value.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    value.add_argument("--title", default="")
    value.add_argument("--owner", default="unassigned")
    return value


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    inventory = build_inventory(args.workspace, args.ledger)
    result: dict = inventory
    if args.reserve_prefix:
        result = reserve(
            inventory=inventory,
            ledger=args.ledger,
            prefix=args.reserve_prefix,
            date=args.date,
            title=args.title,
            owner=args.owner,
        )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.fail_on_conflict and inventory["serial_conflicts"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
