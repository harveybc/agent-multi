#!/usr/bin/env python3
"""EC-13 (order 2026-08-18 WP2): systemd unit contract coherence.

The defect: the installed base template carried a v1 contract default
and an old screen gate, so a unit could pass its ``ExecStartPre`` gate
check against ONE contract and then run ``ExecStart`` against ANOTHER.
Musashi deployed a local v2 override; this module is the upstream guard
so a future contract can never inherit that stale default again.

Two independent hazards are checked, because textual equality is not
enough:

1. ``ExecStartPre`` and ``ExecStart`` must name the SAME contract file;
2. they must RESOLVE to the same file. The gate check is invoked with an
   absolute path while the runner is invoked with a path relative to
   ``WorkingDirectory``; those agree only while the working directory is
   the intended runtime, so the resolved paths are compared too.

Read-only: it parses effective unit properties, never writes or
restarts anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

SCHEMA = "agent_multi.systemd_unit_coherence.v1"
CONTRACT_RE = re.compile(r"[\w./-]*p1_difficulty_lr_factorial[\w.-]*\.json")


class CoherenceError(RuntimeError):
    """Typed refusal for a malformed request."""


def contracts_in(command: str) -> list[str]:
    """Every contract path named by one Exec* property line."""
    return CONTRACT_RE.findall(command or "")


def resolve(path: str, working_directory: str | None) -> str:
    candidate = Path(path)
    if not candidate.is_absolute() and working_directory:
        candidate = Path(working_directory) / candidate
    return str(candidate)


def check_unit(*, exec_start_pre: str, exec_start: str,
               working_directory: str | None,
               digest_of=None) -> dict:
    """Return a typed coherence verdict for one unit."""
    pre = contracts_in(exec_start_pre)
    run = contracts_in(exec_start)
    problems: list[str] = []

    if not run:
        problems.append(
            "EXEC_START_NAMES_NO_CONTRACT: the runner does not name a "
            "contract file, so it will silently use a built-in default")
    if not pre:
        problems.append(
            "EXEC_START_PRE_NAMES_NO_CONTRACT: the gate check does not "
            "name a contract, so it cannot have gated this run")

    if pre and run:
        if {Path(p).name for p in pre} != {Path(p).name for p in run}:
            problems.append(
                f"CONTRACT_NAME_MISMATCH: ExecStartPre names {sorted(Path(p).name for p in pre)} "
                f"while ExecStart names {sorted(Path(p).name for p in run)} — "
                "the run was gated against a different contract than it "
                "executes")
        resolved_pre = {resolve(p, working_directory) for p in pre}
        resolved_run = {resolve(p, working_directory) for p in run}
        if resolved_pre != resolved_run:
            problems.append(
                f"CONTRACT_PATH_MISMATCH: resolved {sorted(resolved_pre)} "
                f"vs {sorted(resolved_run)}")
        elif digest_of is not None:
            digests = {digest_of(p) for p in resolved_run}
            if len(digests) > 1:
                problems.append(
                    "CONTRACT_DIGEST_MISMATCH: the resolved paths do not "
                    "hash alike")

    return {
        "schema": SCHEMA,
        "coherent": not problems,
        "problems": problems,
        "exec_start_pre_contracts": pre,
        "exec_start_contracts": run,
        "working_directory": working_directory,
    }


def _show(unit: str, prop: str) -> str:
    out = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", prop],
        capture_output=True, text=True, check=False).stdout.strip()
    return out.split("=", 1)[1] if "=" in out else ""


def _sha(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        return f"MISSING:{path}"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", nargs="+", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results = {}
    for unit in args.unit:
        results[unit] = check_unit(
            exec_start_pre=_show(unit, "ExecStartPre"),
            exec_start=_show(unit, "ExecStart"),
            working_directory=_show(unit, "WorkingDirectory") or None,
            digest_of=_sha)
    coherent = all(r["coherent"] for r in results.values())
    print(json.dumps({"schema": SCHEMA, "all_coherent": coherent,
                      "units": results}, indent=1, sort_keys=True))
    return 0 if coherent else 1


if __name__ == "__main__":
    raise SystemExit(main())
