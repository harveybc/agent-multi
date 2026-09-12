#!/usr/bin/env python3
"""R3 (order 2026-09-11): declare each installed P1LR decision seat
executable or historical, in a durable append-only record.

Making the refusal stable (ExecCondition) stops the restart loop but
says nothing about whether the seat SHOULD exist. A seat whose pinned
screen gate was never produced is historical: it will refuse forever,
correctly, and the operator needs a record saying so — not a surprise
in a journal six weeks later.

This tool is READ-ONLY with respect to the fleet. It enumerates the
enabled instances of ``p1lr-decision@``, asks systemd for the EFFECTIVE
gate check and pinned gate of each, evaluates that gate through the
shipped check, and writes one disposition record per seat. It never
starts, stops, enables or disables a unit, never runs training, and —
above all — never reconstructs, copies or repoints an absent gate. An
absent gate is evidence.

    python tools/p1lr_seat_disposition.py [--state-dir DIR] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
DEFAULT_STATE = HOME / ".local/state/agent-multi/p1lr-gate"
SCHEMA = "agent_multi.p1lr_seat_disposition.v1"

EXECUTABLE = "EXECUTABLE"
HISTORICAL = "HISTORICAL_NOT_EXECUTABLE"
UNDETERMINED = "UNDETERMINED"


class SeatDispositionRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _sc(*args: str) -> str:
    out = subprocess.run(("systemctl", "--user", *args),
                         capture_output=True, text=True, timeout=60)
    return out.stdout


def _prop(unit: str, name: str) -> str:
    line = _sc("show", unit, "-p", name).strip()
    return line.split("=", 1)[1] if "=" in line else ""


def enabled_seats() -> list[str]:
    """Instances the user manager actually knows about."""
    seats: set[str] = set()
    wants = HOME / ".config/systemd/user/default.target.wants"
    if wants.is_dir():
        for link in wants.iterdir():
            m = re.fullmatch(r"p1lr-decision@(.+)\.service", link.name)
            if m:
                seats.add(m.group(1))
    for raw in _sc("list-units", "p1lr-decision@*", "--all",
                   "--no-pager", "--no-legend").splitlines():
        m = re.search(r"p1lr-decision@(\S+)\.service", raw)
        if m:
            seats.add(m.group(1))
    return sorted(seats)


def effective_condition(unit: str) -> tuple[str, list[str]]:
    """The LAST ExecCondition systemd would run, with its argv.

    Drop-in Exec* directives accumulate, so the last one is the one the
    campaign actually pinned; reporting the template's would describe a
    command this seat does not run.
    """
    raw = _prop(unit, "ExecCondition")
    entries = re.findall(r"argv\[\]=([^;]+);", raw)
    if not entries:
        return "", []
    argv = shlex.split(entries[-1].strip())
    return (argv[0] if argv else ""), argv[1:]


def evaluate(unit: str, *, state_dir: Path) -> dict:
    check, argv = effective_condition(unit)
    gate = Path(argv[0]) if argv else None
    record = {
        "schema": SCHEMA,
        "unit": unit,
        "observed_at": datetime.now(timezone.utc).replace(microsecond=0)
                               .isoformat().replace("+00:00", "Z"),
        "active_state": _prop(unit, "ActiveState"),
        "result": _prop(unit, "Result"),
        "condition_result": _prop(unit, "ConditionResult") or "UNAVAILABLE",
        "n_restarts": _prop(unit, "NRestarts"),
        "gate_check": check or "UNAVAILABLE",
        "pinned_gate": str(gate) if gate else "UNAVAILABLE",
        "pinned_gate_exists": bool(gate and gate.is_file()),
        "gate_reconstructed": False,
    }
    if not check:
        record["disposition"] = UNDETERMINED
        record["reason"] = ("systemd reports no ExecCondition for this seat; "
                            "its gate cannot be evaluated without guessing")
        return record
    run = subprocess.run(("bash", check, *argv), capture_output=True,
                         text=True, timeout=120,
                         env={**os.environ,
                              "P1LR_GATE_STATE_DIR": str(state_dir)})
    record["gate_check_exit"] = run.returncode
    tail = [ln for ln in run.stdout.splitlines() if ln.startswith("{")]
    payload = json.loads(tail[-1]) if tail else {}
    record["refusals"] = payload.get("refusals", [])
    record["refusal_classes"] = sorted(
        {r.split(":", 1)[0] for r in record["refusals"]})
    if run.returncode == 0:
        record["disposition"] = EXECUTABLE
        record["reason"] = "the pinned gate verifies against this contract"
    elif run.returncode == 4:
        record["disposition"] = HISTORICAL
        record["reason"] = (
            "the pinned gate does not verify, so this seat refuses before "
            "any training process exists. With ExecCondition the refusal is "
            "stable: systemd skips the unit and schedules no restart. The "
            "gate was NOT reconstructed — an absent or non-viable gate is "
            "the evidence, not a file to be produced.")
        record["owner_disposition"] = (
            f"systemctl --user disable --now '{unit}'  &&  "
            f"systemctl --user reset-failed '{unit}'")
    else:
        record["disposition"] = UNDETERMINED
        record["reason"] = (
            f"the gate check exited {run.returncode}, which is neither a "
            "verification (0) nor a typed configuration refusal (4); this is "
            "a harness failure and must not be read as a seat disposition")
    return record


def write_record(record: dict, state_dir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    log = state_dir / "SEAT_DISPOSITIONS.jsonl"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    latest = state_dir / f"{record['unit']}.disposition.json"
    latest.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return latest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--json", action="store_true",
                    help="print the records instead of a summary")
    args = ap.parse_args(argv)

    seats = enabled_seats()
    if not seats:
        print("no p1lr-decision@ seat is installed on this host")
        return 0
    records = []
    for seat in seats:
        unit = f"p1lr-decision@{seat}.service"
        record = evaluate(unit, state_dir=args.state_dir)
        write_record(record, args.state_dir)
        records.append(record)

    if args.json:
        print(json.dumps(records, indent=1, sort_keys=True))
    else:
        for r in records:
            print(f"{r['unit']}: {r['disposition']}")
            print(f"  active={r['active_state']} result={r['result']} "
                  f"restarts={r['n_restarts']} "
                  f"condition={r['condition_result']}")
            print(f"  gate exists={r['pinned_gate_exists']} "
                  f"classes={r.get('refusal_classes', [])}")
            if r.get("owner_disposition"):
                print(f"  owner disposition: {r['owner_disposition']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
