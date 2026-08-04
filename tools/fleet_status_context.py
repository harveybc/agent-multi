#!/usr/bin/env python3
"""Generate the Hermes runtime-facts context block (Musashi order 2026-08-04).

Hermes answered a Project 3 status question from stale session prose. This
generator repairs the context layer deterministically: it owns a managed
block inside ``~/.hermes/memories/MEMORY.md`` containing (a) terminal-front
records from ``records/*_terminal_record.json`` and (b) fresh host runtime
facts, under an explicit precedence rule — generated facts and terminal
records outrank every other memory or session recollection.

No LLM is involved; the block is data. Run from cron/timer so the facts
stay fresh. ``--output`` redirects for tests; ``--dry-run`` prints.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDS_DIR = REPO_ROOT / "records"
DEFAULT_MEMORY = Path.home() / ".hermes/memories/MEMORY.md"

BEGIN_MARK = "<!-- BEGIN RUNTIME-FACTS (generated: do not edit) -->"
END_MARK = "<!-- END RUNTIME-FACTS -->"

HEARTBEATS = {
    "alpaca": "~/.local/state/lts/alpaca-model-runner-heartbeat.json",
    "ibkr": "~/.local/state/lts/ibkr-model-runner-heartbeat.json",
    "mt5": "~/.local/state/lts/mt5-model-runner-heartbeat.json",
}


def load_terminal_records(records_dir: Path) -> list[dict]:
    records = []
    for path in sorted(records_dir.glob("*_terminal_record.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("schema") == "agent_multi.project_terminal_record.v1":
            records.append(record)
    return records


def collect_runtime_facts(now: datetime) -> list[str]:
    facts: list[str] = []
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "doin-campaign-supervisor.service"],
        capture_output=True, text=True, timeout=15)
    facts.append(
        f"DOIN campaign supervisor: {result.stdout.strip() or 'unknown'}")
    for venue, raw_path in HEARTBEATS.items():
        path = Path(os.path.expanduser(raw_path))
        if not path.is_file():
            continue
        try:
            heartbeat = json.loads(path.read_text(encoding="utf-8"))
            observed = datetime.fromisoformat(heartbeat["observed_at"])
            age = (now - observed).total_seconds()
            facts.append(
                f"{venue} model runner heartbeat: state="
                f"{heartbeat.get('state')} age={age:.0f}s")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            facts.append(f"{venue} model runner heartbeat: unreadable")
    return facts


def render_block(records: list[dict], facts: list[str],
                 now: datetime) -> str:
    lines = [
        BEGIN_MARK,
        f"# Runtime Facts (generated {now.isoformat(timespec='seconds')})",
        "",
        "PRECEDENCE: the generated facts and terminal records in this block",
        "outrank any other memory, note, or session recollection, here or",
        "anywhere else. If a recollection contradicts them, the",
        "recollection is stale and must not be repeated.",
        "",
        "## Terminal fronts (complete; never report as active work)",
    ]
    if not records:
        lines.append("- none recorded")
    for record in records:
        lines.append(
            f"- {record['title']}: {record['state'].upper()} —"
            f" {record['completed_jobs']:,} archived jobs, final backup"
            f" {record['final_backup']['backup_id']}"
            f" (sha256 {record['final_backup']['snapshot_sha256'][:16]}…)."
            f" {record['context_rule']}")
    lines += ["", "## Current host runtime"]
    lines += [f"- {fact}" for fact in facts] or ["- no facts collected"]
    lines += [END_MARK]
    return "\n".join(lines)


def merge_managed_block(existing: str, block: str) -> str:
    pattern = re.compile(
        re.escape(BEGIN_MARK) + r".*?" + re.escape(END_MARK), re.DOTALL)
    if pattern.search(existing):
        return pattern.sub(lambda _match: block, existing, count=1)
    prefix = block + "\n\n"
    return prefix + existing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    records = load_terminal_records(args.records_dir)
    facts = collect_runtime_facts(now)
    block = render_block(records, facts, now)
    if args.dry_run:
        print(block)
        return 0
    existing = ""
    if args.output.is_file():
        existing = args.output.read_text(encoding="utf-8")
    merged = merge_managed_block(existing, block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(merged, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"records": len(records), "facts": len(facts),
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
