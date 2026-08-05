#!/usr/bin/env python3
"""Shared emission shim: watchdog collectors -> fleet incident ledger.

Replaces the direct per-watchdog Telegram sends (Musashi order 2026-08-04
P1). Collectors keep their thresholds and hysteresis; this shim only
normalizes their verdicts into ledger observations/recoveries. Delivery
policy is owned entirely by tools/incident_router.py — the one autonomous
alert-sender path.

Never raises into a collector: emission failure is reported on stderr and
returned as False so cron logs retain the evidence.
"""
from __future__ import annotations

import json
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incident_ledger as ledger  # noqa: E402

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "examples/configs/incident_ledger_v1.json"
)


def sanitize_code(text: str) -> str:
    code = re.sub(r"[^a-z0-9_.:\-]", "_", text.lower()).strip("_")
    return code or "unnamed_event"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _open(config_path: Path | None, db_override: Path | None):
    config = ledger.load_config(config_path or DEFAULT_CONFIG)
    db_path = db_override or Path(
        str(Path(config["ledger_db"]).expanduser()))
    return config, ledger.connect(db_path)


def observe_incident(*, source: str, event_code: str, severity: str,
                     summary: str, front: str = "front4",
                     machine: str | None = None,
                     affected_object: str = "-",
                     payload: dict | None = None,
                     config_path: Path | None = None,
                     db_override: Path | None = None) -> bool:
    try:
        config, conn = _open(config_path, db_override)
        body = dict(payload or {})
        body.setdefault("summary", summary)
        ledger.observe(
            conn, config, source=source, front=front,
            machine=machine or socket.gethostname(),
            event_code=sanitize_code(event_code), severity=severity,
            affected_object=affected_object, payload=body,
            source_evidence_at=_now_iso(),
        )
        return True
    except Exception as exc:
        print(f"incident emission failed ({event_code}): {exc}",
              file=sys.stderr)
        return False


def recover_incident(*, source: str, event_code: str,
                     evidence: dict, front: str = "front4",
                     machine: str | None = None,
                     affected_object: str = "-",
                     config_path: Path | None = None,
                     db_override: Path | None = None) -> bool:
    try:
        config, conn = _open(config_path, db_override)
        bound_machine = machine or socket.gethostname()
        code = sanitize_code(event_code)
        document = ledger.build_recovery_evidence(
            source=source, front=front, machine=bound_machine,
            event_code=code, affected_object=affected_object,
            state=evidence or {"summary": "source recovered"},
        )
        ledger.recover(
            conn, config, source=source, front=front,
            machine=bound_machine, event_code=code,
            affected_object=affected_object, evidence=document,
        )
        return True
    except Exception as exc:
        print(f"incident recovery emission failed ({event_code}): {exc}",
              file=sys.stderr)
        return False


def emit_watchdog_messages(*, source: str, pairs, front: str = "front4",
                           machine: str | None = None,
                           severity: str = "P2",
                           code_map=None,
                           config_path: Path | None = None,
                           db_override: Path | None = None) -> int:
    """Emit (key, message) pairs from a legacy watchdog. Messages whose
    first line starts with a recovery marker resolve; others observe.
    Returns the number of failed emissions."""
    failures = 0
    for key, message in pairs:
        event_code = (code_map(key) if code_map else sanitize_code(key))
        first_line = message.splitlines()[0] if message else ""
        recovered = first_line.startswith("✅")
        if recovered:
            ok = recover_incident(
                source=source, event_code=event_code, front=front,
                machine=machine, evidence={"summary": first_line[:200]},
                config_path=config_path, db_override=db_override)
        else:
            ok = observe_incident(
                source=source, event_code=event_code, severity=severity,
                summary=first_line[:200], front=front, machine=machine,
                payload={"detail": message[:1500]},
                config_path=config_path, db_override=db_override)
        failures += 0 if ok else 1
    return failures
