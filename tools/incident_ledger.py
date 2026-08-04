#!/usr/bin/env python3
"""Fleet incident ledger: one durable SQLite record of operational incidents.

Owner doctrine (doc 29 §14, Musashi order 2026-08-04): Telegram is an
exception channel. Producers (watchdogs, continuity monitors, runners) emit
normalized observations into this ledger; the notification router decides
what is due. Acknowledgement suppresses reminders but never rewrites the
source condition as healthy — only fresh direct recovery evidence resolves
an incident.

Design constraints:

- stdlib only: must run under /usr/bin/python3 from cron on every host;
- append-only journal (`incident_events`) plus one materialized row per
  open incident; at most one non-resolved incident per fingerprint;
- fail-closed inputs: future timestamps beyond skew, unknown severities,
  malformed payloads and wrong schemas are refusals, never rows;
- no secrets: payloads are redacted against prohibited patterns before
  storage so a leaked credential can never reach Telegram downstream.

Incident identity (Musashi order §P1):
    fingerprint = sha256(source|front|venue_or_machine|event_code|affected_object)

CLI:
    observe  --source S --front F --machine M --event-code E [--severity Pn]
             [--object OBJ] [--payload-json JSON | --payload-stdin]
             [--evidence-at ISO] [--config PATH]
    recover  --source S --front F --machine M --event-code E [--object OBJ]
             --evidence-json JSON
    status   [--active] [--severity P0,P1,...] [--json]
    show     INCIDENT_ID
    ack      INCIDENT_ID --actor NAME --reason TEXT
    unack    INCIDENT_ID
    history  [--since ISO] [--incident INCIDENT_ID] [--json]

Exit codes: 0 ok, 1 refusal/validation failure, 2 usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_VERSION = "agent_multi.incident_ledger.v1"
DEFAULT_DB = Path.home() / ".local/state/agent-multi/incident-ledger/incidents.sqlite"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "examples/configs/incident_ledger_v1.json"
)

SEVERITIES = ("P0", "P1", "P2", "P3")
OPEN_STATES = ("pending", "active", "acknowledged")
ALL_STATES = OPEN_STATES + ("resolved",)

# Prohibited content is redacted from stored payloads: broker account ids,
# bot tokens, API keys, private keys. The ledger is upstream of Telegram.
REDACTION_PATTERNS = (
    re.compile(r"\bDU[0-9]{5,}\b"),
    re.compile(r"\b\d{8,}:[A-Za-z0-9_\-]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passphrase)\s*[:=]\s*\S+"),
)

_TABLES = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    severity TEXT NOT NULL,
    front TEXT NOT NULL,
    source TEXT NOT NULL,
    event_code TEXT NOT NULL,
    venue_or_machine TEXT NOT NULL,
    affected_object TEXT NOT NULL,
    first_observed_at TEXT NOT NULL,
    last_observed_at TEXT NOT NULL,
    source_evidence_at TEXT NOT NULL,
    state TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL,
    last_notified_at TEXT,
    notification_count INTEGER NOT NULL DEFAULT 0,
    delivered_severity TEXT,
    acknowledged_at TEXT,
    acknowledged_by TEXT,
    acknowledgement_reason TEXT,
    resolved_at TEXT,
    resolution_evidence_hash TEXT,
    payload_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_open_fingerprint
    ON incidents(fingerprint) WHERE state != 'resolved';
CREATE TABLE IF NOT EXISTS incident_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id TEXT NOT NULL,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail_json TEXT NOT NULL
);
"""


class Refusal(Exception):
    """A fail-closed refusal: nothing was written."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str, field: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise Refusal(f"{field} is not an ISO-8601 timestamp: {value!r}")
    if moment.tzinfo is None:
        raise Refusal(f"{field} must carry an explicit timezone: {value!r}")
    return moment


def load_config(path: Path | None) -> dict:
    config_path = path or DEFAULT_CONFIG
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Refusal(f"config not found: {config_path}")
    except json.JSONDecodeError as exc:
        raise Refusal(f"config is not valid JSON: {config_path}: {exc}")
    if config.get("schema") != "agent_multi.incident_config.v1":
        raise Refusal(f"config schema mismatch in {config_path}")
    return config


def redact(text: str) -> str:
    for pattern in REDACTION_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def fingerprint_of(source: str, front: str, machine: str,
                   event_code: str, affected_object: str) -> str:
    material = "|".join((source, front, machine, event_code, affected_object))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_TABLES)
    row = conn.execute(
        "SELECT value FROM ledger_meta WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO ledger_meta(key, value) VALUES('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()
    elif row["value"] != SCHEMA_VERSION:
        conn.close()
        raise Refusal(
            f"ledger schema {row['value']!r} does not match {SCHEMA_VERSION!r};"
            " refusing to operate on an unknown schema"
        )
    return conn


def _journal(conn: sqlite3.Connection, incident_id: str, kind: str,
             detail: dict, at: datetime | None = None) -> None:
    conn.execute(
        "INSERT INTO incident_events(incident_id, at, kind, detail_json)"
        " VALUES(?,?,?,?)",
        (incident_id, iso(at or utcnow()), kind,
         json.dumps(detail, sort_keys=True)),
    )


def _validate_identity(source: str, front: str, machine: str,
                       event_code: str, affected_object: str) -> None:
    for field, value in (("source", source), ("front", front),
                         ("machine", machine), ("event_code", event_code)):
        if not value or not re.match(r"^[a-z0-9][a-z0-9_.:\-]*$", value):
            raise Refusal(
                f"{field} must be a non-empty lowercase token, got {value!r}"
            )
    if affected_object is None:
        raise Refusal("affected_object must not be None (use '-' for global)")


def observe(conn: sqlite3.Connection, config: dict, *, source: str,
            front: str, machine: str, event_code: str, severity: str,
            affected_object: str, payload: dict,
            source_evidence_at: str, observed_at: str | None = None,
            now: datetime | None = None) -> dict:
    """Record one observation. Returns the current incident row as a dict."""
    now = now or utcnow()
    _validate_identity(source, front, machine, event_code, affected_object)
    if severity not in SEVERITIES:
        raise Refusal(f"severity must be one of {SEVERITIES}, got {severity!r}")
    if not isinstance(payload, dict):
        raise Refusal("payload must be a JSON object")
    skew = float(config.get("max_future_skew_seconds", 120))
    evidence_at = parse_iso(source_evidence_at, "source_evidence_at")
    observed = parse_iso(observed_at, "observed_at") if observed_at else now
    horizon = now + timedelta(seconds=skew)
    if evidence_at > horizon:
        raise Refusal(
            f"source_evidence_at {source_evidence_at} is in the future"
            f" beyond {skew:.0f}s skew — forged or misclocked, refusing"
        )
    if observed > horizon:
        raise Refusal(
            f"observed_at is in the future beyond {skew:.0f}s skew, refusing"
        )
    max_age = float(config.get("max_evidence_age_seconds", 86400))
    if (now - evidence_at).total_seconds() > max_age:
        raise Refusal(
            f"source_evidence_at {source_evidence_at} is older than"
            f" {max_age:.0f}s — stale evidence is a refusal, re-observe first"
        )

    payload_text = redact(json.dumps(payload, sort_keys=True))
    payload_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    fingerprint = fingerprint_of(source, front, machine, event_code,
                                 affected_object)
    open_row = conn.execute(
        "SELECT * FROM incidents WHERE fingerprint=? AND state != 'resolved'",
        (fingerprint,),
    ).fetchone()

    if open_row is None:
        # Flap suppression: a source that recovered moments ago and fails
        # again reopens its incident, keeping the notification clock, so a
        # flapping condition can never mint a message storm.
        flap_window = float(config.get("flap_reopen_window_seconds", 3600))
        recent = conn.execute(
            "SELECT * FROM incidents WHERE fingerprint=? AND state='resolved'"
            " ORDER BY resolved_at DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
        if recent is not None and recent["resolved_at"] is not None:
            resolved_at = datetime.fromisoformat(recent["resolved_at"])
            if (now - resolved_at).total_seconds() <= flap_window:
                reopened_state = ("active" if recent["notification_count"]
                                  else "pending")
                conn.execute(
                    "UPDATE incidents SET state=?, resolved_at=NULL,"
                    " resolution_evidence_hash=NULL, occurrence_count=?,"
                    " last_observed_at=?, source_evidence_at=?, severity=?,"
                    " payload_hash=?, payload_json=? WHERE incident_id=?",
                    (reopened_state, recent["occurrence_count"] + 1,
                     iso(observed), iso(evidence_at), severity,
                     payload_hash, payload_text, recent["incident_id"]),
                )
                _journal(conn, recent["incident_id"], "reopened",
                         {"severity": severity,
                          "occurrence": recent["occurrence_count"] + 1},
                         observed)
                conn.commit()
                return dict(conn.execute(
                    "SELECT * FROM incidents WHERE incident_id=?",
                    (recent["incident_id"],),
                ).fetchone())
        incident_id = "INC-{}-{}".format(
            observed.strftime("%Y%m%d%H%M%S"), fingerprint[:8]
        )
        conn.execute(
            "INSERT INTO incidents(incident_id, fingerprint, severity, front,"
            " source, event_code, venue_or_machine, affected_object,"
            " first_observed_at, last_observed_at, source_evidence_at, state,"
            " occurrence_count, payload_hash, payload_json)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (incident_id, fingerprint, severity, front, source, event_code,
             machine, affected_object, iso(observed), iso(observed),
             iso(evidence_at), "pending", 1, payload_hash, payload_text),
        )
        _journal(conn, incident_id, "observed",
                 {"severity": severity, "occurrence": 1,
                  "payload_hash": payload_hash}, observed)
        conn.commit()
        return dict(conn.execute(
            "SELECT * FROM incidents WHERE incident_id=?", (incident_id,)
        ).fetchone())

    incident_id = open_row["incident_id"]
    occurrence = open_row["occurrence_count"] + 1
    escalated = SEVERITIES.index(severity) < SEVERITIES.index(
        open_row["severity"])
    new_severity = severity if escalated else open_row["severity"]
    conn.execute(
        "UPDATE incidents SET occurrence_count=?, last_observed_at=?,"
        " source_evidence_at=?, severity=?, payload_hash=?, payload_json=?"
        " WHERE incident_id=?",
        (occurrence, iso(observed), iso(evidence_at), new_severity,
         payload_hash, payload_text, incident_id),
    )
    _journal(conn, incident_id, "observed",
             {"severity": severity, "occurrence": occurrence,
              "payload_hash": payload_hash}, observed)
    if escalated:
        # Material worsening: clear the notification clock so the router
        # pages immediately at the new severity.
        conn.execute(
            "UPDATE incidents SET last_notified_at=NULL WHERE incident_id=?",
            (incident_id,),
        )
        _journal(conn, incident_id, "severity_changed",
                 {"from": open_row["severity"], "to": severity}, observed)
    conn.commit()
    return dict(conn.execute(
        "SELECT * FROM incidents WHERE incident_id=?", (incident_id,)
    ).fetchone())


def recover(conn: sqlite3.Connection, config: dict, *, source: str,
            front: str, machine: str, event_code: str,
            affected_object: str, evidence: dict,
            now: datetime | None = None) -> dict | None:
    """Resolve the open incident for this identity with direct evidence.

    Returns the resolved row, or None when no open incident exists (a
    recovery for a healthy identity is a no-op, never an error: producers
    report recovery unconditionally on a healthy pass).
    """
    now = now or utcnow()
    _validate_identity(source, front, machine, event_code, affected_object)
    if not isinstance(evidence, dict) or not evidence:
        raise Refusal("recovery requires a non-empty evidence JSON object")
    fingerprint = fingerprint_of(source, front, machine, event_code,
                                 affected_object)
    row = conn.execute(
        "SELECT * FROM incidents WHERE fingerprint=? AND state != 'resolved'",
        (fingerprint,),
    ).fetchone()
    if row is None:
        return None
    evidence_text = redact(json.dumps(evidence, sort_keys=True))
    evidence_hash = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
    conn.execute(
        "UPDATE incidents SET state='resolved', resolved_at=?,"
        " resolution_evidence_hash=? WHERE incident_id=?",
        (iso(now), evidence_hash, row["incident_id"]),
    )
    _journal(conn, row["incident_id"], "resolved",
             {"evidence_hash": evidence_hash, "evidence": evidence_text}, now)
    conn.commit()
    return dict(conn.execute(
        "SELECT * FROM incidents WHERE incident_id=?", (row["incident_id"],)
    ).fetchone())


def acknowledge(conn: sqlite3.Connection, incident_id: str, actor: str,
                reason: str, now: datetime | None = None) -> dict:
    now = now or utcnow()
    if not actor.strip() or not reason.strip():
        raise Refusal("ack requires --actor and a non-empty --reason")
    row = conn.execute(
        "SELECT * FROM incidents WHERE incident_id=?", (incident_id,)
    ).fetchone()
    if row is None:
        raise Refusal(f"unknown incident {incident_id!r}")
    if row["state"] == "resolved":
        raise Refusal(f"{incident_id} is already resolved; nothing to ack")
    conn.execute(
        "UPDATE incidents SET state='acknowledged', acknowledged_at=?,"
        " acknowledged_by=?, acknowledgement_reason=? WHERE incident_id=?",
        (iso(now), actor.strip(), reason.strip(), incident_id),
    )
    _journal(conn, incident_id, "acknowledged",
             {"actor": actor.strip(), "reason": reason.strip()}, now)
    conn.commit()
    return dict(conn.execute(
        "SELECT * FROM incidents WHERE incident_id=?", (incident_id,)
    ).fetchone())


def unacknowledge(conn: sqlite3.Connection, incident_id: str,
                  now: datetime | None = None) -> dict:
    now = now or utcnow()
    row = conn.execute(
        "SELECT * FROM incidents WHERE incident_id=?", (incident_id,)
    ).fetchone()
    if row is None:
        raise Refusal(f"unknown incident {incident_id!r}")
    if row["state"] != "acknowledged":
        raise Refusal(f"{incident_id} is {row['state']}, not acknowledged")
    conn.execute(
        "UPDATE incidents SET state='active', acknowledged_at=NULL,"
        " acknowledged_by=NULL, acknowledgement_reason=NULL"
        " WHERE incident_id=?",
        (incident_id,),
    )
    _journal(conn, incident_id, "unacknowledged", {}, now)
    conn.commit()
    return dict(conn.execute(
        "SELECT * FROM incidents WHERE incident_id=?", (incident_id,)
    ).fetchone())


def mark_notified(conn: sqlite3.Connection, incident_id: str,
                  channel: str, message_hash: str,
                  now: datetime | None = None) -> None:
    """Router callback: record one delivered notification."""
    now = now or utcnow()
    row = conn.execute(
        "SELECT * FROM incidents WHERE incident_id=?", (incident_id,)
    ).fetchone()
    if row is None:
        raise Refusal(f"unknown incident {incident_id!r}")
    new_state = "active" if row["state"] == "pending" else row["state"]
    conn.execute(
        "UPDATE incidents SET last_notified_at=?, notification_count=?,"
        " delivered_severity=?, state=? WHERE incident_id=?",
        (iso(now), row["notification_count"] + 1, row["severity"], new_state,
         incident_id),
    )
    _journal(conn, incident_id, "notified",
             {"channel": channel, "message_hash": message_hash}, now)
    conn.commit()


def open_incidents(conn: sqlite3.Connection,
                   severities: list[str] | None = None) -> list[dict]:
    query = "SELECT * FROM incidents WHERE state != 'resolved'"
    args: list = []
    if severities:
        marks = ",".join("?" for _ in severities)
        query += f" AND severity IN ({marks})"
        args = list(severities)
    query += " ORDER BY severity ASC, first_observed_at ASC"
    return [dict(row) for row in conn.execute(query, args).fetchall()]


def _print_rows(rows: list[dict], as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    if not rows:
        print("no incidents")
        return
    for row in rows:
        print(
            f"{row['severity']}  {row['incident_id']}  {row['state']:<12}"
            f" {row['source']}/{row['event_code']}"
            f" @{row['venue_or_machine']}"
            f" x{row['occurrence_count']}"
            f" first={row['first_observed_at']}"
            f" notif={row['notification_count']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None,
                        help="versioned incident config JSON")
    parser.add_argument("--db", type=Path, default=None,
                        help="override ledger path (tests only)")
    commands = parser.add_subparsers(dest="command", required=True)

    def identity_args(sub):
        sub.add_argument("--source", required=True)
        sub.add_argument("--front", required=True)
        sub.add_argument("--machine", required=True)
        sub.add_argument("--event-code", required=True)
        sub.add_argument("--object", dest="affected_object", default="-")

    sub = commands.add_parser("observe")
    identity_args(sub)
    sub.add_argument("--severity", required=True, choices=SEVERITIES)
    sub.add_argument("--payload-json", default=None)
    sub.add_argument("--payload-stdin", action="store_true")
    sub.add_argument("--evidence-at", required=True,
                     help="ISO timestamp of the direct source evidence")
    sub.add_argument("--observed-at", default=None)

    sub = commands.add_parser("recover")
    identity_args(sub)
    sub.add_argument("--evidence-json", required=True)

    sub = commands.add_parser("status")
    sub.add_argument("--active", action="store_true")
    sub.add_argument("--severity", default=None,
                     help="comma-separated filter, e.g. P0,P1")
    sub.add_argument("--json", action="store_true")

    sub = commands.add_parser("show")
    sub.add_argument("incident_id")

    sub = commands.add_parser("ack")
    sub.add_argument("incident_id")
    sub.add_argument("--actor", required=True)
    sub.add_argument("--reason", required=True)

    sub = commands.add_parser("unack")
    sub.add_argument("incident_id")

    sub = commands.add_parser("history")
    sub.add_argument("--since", default=None)
    sub.add_argument("--incident", default=None)
    sub.add_argument("--json", action="store_true")

    args = parser.parse_args()
    try:
        config = load_config(args.config)
        db_path = args.db or Path(
            os.path.expanduser(config.get("ledger_db", str(DEFAULT_DB)))
        )
        conn = connect(db_path)
        if args.command == "observe":
            if args.payload_stdin:
                raw = sys.stdin.read()
            else:
                raw = args.payload_json or "{}"
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise Refusal(f"payload is not valid JSON: {exc}")
            row = observe(
                conn, config, source=args.source, front=args.front,
                machine=args.machine, event_code=args.event_code,
                severity=args.severity, affected_object=args.affected_object,
                payload=payload, source_evidence_at=args.evidence_at,
                observed_at=args.observed_at,
            )
            print(json.dumps({"incident_id": row["incident_id"],
                              "state": row["state"],
                              "occurrence_count": row["occurrence_count"]}))
        elif args.command == "recover":
            try:
                evidence = json.loads(args.evidence_json)
            except json.JSONDecodeError as exc:
                raise Refusal(f"evidence is not valid JSON: {exc}")
            row = recover(
                conn, config, source=args.source, front=args.front,
                machine=args.machine, event_code=args.event_code,
                affected_object=args.affected_object, evidence=evidence,
            )
            if row is None:
                print(json.dumps({"resolved": None}))
            else:
                print(json.dumps({"resolved": row["incident_id"]}))
        elif args.command == "status":
            severities = (args.severity.split(",") if args.severity else None)
            if severities:
                bad = [s for s in severities if s not in SEVERITIES]
                if bad:
                    raise Refusal(f"unknown severities: {bad}")
            if args.active:
                rows = open_incidents(conn, severities)
            else:
                query = "SELECT * FROM incidents"
                params: list = []
                if severities:
                    marks = ",".join("?" for _ in severities)
                    query += f" WHERE severity IN ({marks})"
                    params = severities
                query += " ORDER BY first_observed_at DESC LIMIT 200"
                rows = [dict(r) for r in conn.execute(query, params)]
            _print_rows(rows, args.json)
        elif args.command == "show":
            row = conn.execute(
                "SELECT * FROM incidents WHERE incident_id=?",
                (args.incident_id,),
            ).fetchone()
            if row is None:
                raise Refusal(f"unknown incident {args.incident_id!r}")
            print(json.dumps(dict(row), indent=2, sort_keys=True))
        elif args.command == "ack":
            row = acknowledge(conn, args.incident_id, args.actor, args.reason)
            print(json.dumps({"incident_id": row["incident_id"],
                              "state": row["state"]}))
        elif args.command == "unack":
            row = unacknowledge(conn, args.incident_id)
            print(json.dumps({"incident_id": row["incident_id"],
                              "state": row["state"]}))
        elif args.command == "history":
            query = "SELECT * FROM incident_events"
            clauses, params = [], []
            if args.incident:
                clauses.append("incident_id=?")
                params.append(args.incident)
            if args.since:
                parse_iso(args.since, "--since")
                clauses.append("at >= ?")
                params.append(args.since)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY seq ASC LIMIT 1000"
            rows = [dict(r) for r in conn.execute(query, params)]
            if args.json:
                print(json.dumps(rows, indent=2, sort_keys=True))
            else:
                for row in rows:
                    print(f"{row['at']}  {row['incident_id']}"
                          f"  {row['kind']}  {row['detail_json']}")
        return 0
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
