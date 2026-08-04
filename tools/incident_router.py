#!/usr/bin/env python3
"""Fleet notification router: the single autonomous Telegram sender.

One pass per invocation (cron/systemd friendly). Reads the local incident
ledger, computes which notifications are due under the owner-approved alert
policy, and delivers them — directly on the notification-owner host, by
SSH forwarding into the owner's ledger elsewhere, with bounded per-severity
failover to direct sending when the owner is unreachable.

Policy (Musashi order 2026-08-04, §Alert Policy):

- P0: first page immediately on the next pass (router cadence <= 60 s),
  one reminder after 15 minutes, then hourly until acknowledged; an
  escalation clears the clock and pages immediately.
- P1: first page on the next pass, repeat every 2 h until acknowledged.
- P2: producers apply their own hysteresis; first page on the next pass
  after emission, repeat every 6 h.
- P3: never paged individually; one daily exception digest while any P3
  incident remains unresolved.
- Acknowledged incidents stop paging entirely; they remain visible in
  `status --active`. Only source recovery resolves.
- Recovery messages are sent once, and only for incidents that actually
  delivered a P0/P1 activation.

There is no message queue: due-ness is recomputed from ledger state, so a
router restart can never replay a backlog and stale duplicates collapse
into current state by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incident_ledger as ledger  # noqa: E402

SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def send_telegram(text: str, config: dict) -> None:
    for entry in config.get("telegram_env_files", []):
        load_env_file(Path(os.path.expanduser(entry)))
    token = os.environ.get(config["telegram_token_env"], "").strip()
    chat_id = os.environ.get(config["telegram_chat_env"], "").strip()
    if not chat_id:
        for fallback in config.get("telegram_chat_env_fallbacks", []):
            chat_id = os.environ.get(fallback, "").strip()
            if chat_id:
                break
    if not token or not chat_id:
        raise RuntimeError("Telegram credentials are not configured")
    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text[: int(config.get("message_max_chars", 3500))],
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.loads(response.read().decode("utf-8", errors="replace"))
    if not result.get("ok"):
        raise RuntimeError("Telegram rejected the notification")


def seconds_since(reference: str | None, now: datetime) -> float | None:
    if not reference:
        return None
    return (now - datetime.fromisoformat(reference)).total_seconds()


def notification_due(incident: dict, config: dict, now: datetime) -> bool:
    """Alert-policy due computation for one open incident."""
    severity = incident["severity"]
    if severity == "P3":
        return False
    if incident["state"] == "acknowledged":
        return False
    policy = config["severity_policy"][severity]
    since_notified = seconds_since(incident["last_notified_at"], now)
    if since_notified is None:
        return True
    reminders = list(policy.get("reminders_seconds", []))
    count = incident["notification_count"]
    if count - 1 < len(reminders):
        return since_notified >= float(reminders[count - 1])
    repeat = policy.get("repeat_seconds")
    return repeat is not None and since_notified >= float(repeat)


def format_activation(incident: dict) -> str:
    payload = {}
    try:
        payload = json.loads(incident["payload_json"])
    except (TypeError, json.JSONDecodeError):
        pass
    lines = [
        f"[{incident['severity']}] {incident['event_code']}",
        f"incident: {incident['incident_id']}",
        f"source: {incident['source']} front: {incident['front']}",
        f"machine/venue: {incident['venue_or_machine']}"
        + (f" object: {incident['affected_object']}"
           if incident["affected_object"] != "-" else ""),
        f"state: {incident['state']} occurrences: "
        f"{incident['occurrence_count']}",
        f"first detected: {incident['first_observed_at']}",
        f"last direct evidence: {incident['source_evidence_at']}",
    ]
    for key in ("exposure_state", "protection_state", "account_fingerprint",
                "summary"):
        if payload.get(key) is not None:
            lines.append(f"{key.replace('_', ' ')}: {payload[key]}")
    action = payload.get("operator_action")
    if action:
        lines.append(f"required action: {action}")
    lines.append(f"ack: incident_ledger.py ack {incident['incident_id']}"
                 " --actor you --reason ...")
    return ledger.redact("\n".join(lines))


def format_recovery(incident: dict) -> str:
    return ledger.redact(
        f"[RECOVERED {incident['delivered_severity']}]"
        f" {incident['event_code']}\n"
        f"incident: {incident['incident_id']}\n"
        f"machine/venue: {incident['venue_or_machine']}\n"
        f"resolved: {incident['resolved_at']}\n"
        f"evidence: {incident['resolution_evidence_hash'][:16]}"
    )


def pending_recoveries(conn, config, now: datetime) -> list[dict]:
    """Resolved incidents that delivered a P0/P1 activation but no recovery
    message yet."""
    marks = ",".join(
        "?" for _ in config.get("recovery_message_severities", []))
    if not marks:
        return []
    rows = conn.execute(
        "SELECT * FROM incidents WHERE state='resolved'"
        f" AND delivered_severity IN ({marks})"
        " AND notification_count > 0",
        list(config.get("recovery_message_severities", [])),
    ).fetchall()
    due = []
    for row in rows:
        sent = conn.execute(
            "SELECT 1 FROM incident_events WHERE incident_id=?"
            " AND kind='recovery_notified' LIMIT 1",
            (row["incident_id"],),
        ).fetchone()
        if sent is None:
            due.append(dict(row))
    return due


def digest_due(conn, config, now: datetime) -> list[dict] | None:
    """Return unresolved P3 incidents once per UTC day after digest hour."""
    if now.hour < int(config.get("digest_hour_utc", 12)):
        return None
    row = conn.execute(
        "SELECT value FROM ledger_meta WHERE key='last_digest_date'"
    ).fetchone()
    today = now.strftime("%Y-%m-%d")
    if row is not None and row["value"] >= today:
        return None
    rows = ledger.open_incidents(conn, ["P3"])
    return rows or None


def mark_digest_sent(conn, now: datetime) -> None:
    conn.execute(
        "INSERT INTO ledger_meta(key, value) VALUES('last_digest_date', ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (now.strftime("%Y-%m-%d"),),
    )
    conn.commit()


def forward_to_owner(incident: dict, config: dict) -> None:
    """Re-emit one incident observation into the owner's ledger over SSH."""
    owner = config["notification_owner"]
    script = str(Path(config["forward_repo_path"]) / "tools"
                 / "incident_ledger.py")
    command = ["ssh", *config.get("forward_ssh_options", []), owner,
               "python3", script, "observe",
               "--source", incident["source"],
               "--front", incident["front"],
               "--machine", incident["venue_or_machine"],
               "--event-code", incident["event_code"],
               "--object", incident["affected_object"],
               "--severity", incident["severity"],
               "--evidence-at", incident["source_evidence_at"],
               "--payload-stdin"]
    result = subprocess.run(
        command, input=incident["payload_json"], capture_output=True,
        text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"forward to {owner} failed: {result.stderr.strip()[:200]}")


def forward_recovery_to_owner(incident: dict, config: dict) -> None:
    owner = config["notification_owner"]
    script = str(Path(config["forward_repo_path"]) / "tools"
                 / "incident_ledger.py")
    evidence = json.dumps({
        "forwarded_resolution_of": incident["incident_id"],
        "resolution_evidence_hash": incident["resolution_evidence_hash"],
    })
    command = ["ssh", *config.get("forward_ssh_options", []), owner,
               "python3", script, "recover",
               "--source", incident["source"],
               "--front", incident["front"],
               "--machine", incident["venue_or_machine"],
               "--event-code", incident["event_code"],
               "--object", incident["affected_object"],
               "--evidence-json", evidence]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"recovery forward to {owner} failed:"
            f" {result.stderr.strip()[:200]}")


def failover_exceeded(conn, incident: dict, config: dict,
                      now: datetime) -> bool:
    """True when forwarding to the owner has failed continuously for the
    severity's failover budget. The streak resets on any successful
    handoff or delivery (journal kind ``notified``)."""
    budget = float(config["failover_after_seconds"][incident["severity"]])
    last_handoff = conn.execute(
        "SELECT MAX(at) AS at FROM incident_events"
        " WHERE incident_id=? AND kind='notified'",
        (incident["incident_id"],),
    ).fetchone()["at"]
    query = ("SELECT MIN(at) AS at FROM incident_events"
             " WHERE incident_id=? AND kind='forward_failed'")
    params: list = [incident["incident_id"]]
    if last_handoff is not None:
        query += " AND at > ?"
        params.append(last_handoff)
    streak_start = conn.execute(query, params).fetchone()["at"]
    if streak_start is None:
        return False
    since = (now - datetime.fromisoformat(streak_start)).total_seconds()
    return since >= budget


def run_pass(conn, config: dict, *, hostname: str, now: datetime,
             transport, forwarder, recovery_forwarder,
             dry_run: bool = False) -> list[dict]:
    """One router pass. Returns the action log (for tests and --dry-run)."""
    actions: list[dict] = []
    is_owner = hostname == config["notification_owner"]

    due = [i for i in ledger.open_incidents(conn)
           if notification_due(i, config, now)]
    due.sort(key=lambda i: (SEVERITY_ORDER[i["severity"]],
                            i["first_observed_at"]))

    for incident in due:
        message = format_activation(incident)
        message_hash = hashlib.sha256(message.encode()).hexdigest()[:16]
        if dry_run:
            actions.append({"action": "would_notify",
                            "incident_id": incident["incident_id"],
                            "message": message})
            continue
        if is_owner:
            transport(message, config)
            ledger.mark_notified(conn, incident["incident_id"], "telegram",
                                 message_hash, now)
            actions.append({"action": "notified",
                            "incident_id": incident["incident_id"]})
        else:
            try:
                forwarder(incident, config)
                ledger.mark_notified(conn, incident["incident_id"],
                                     f"forwarded:{config['notification_owner']}",
                                     message_hash, now)
                actions.append({"action": "forwarded",
                                "incident_id": incident["incident_id"]})
            except Exception as exc:
                with conn:
                    conn.execute(
                        "INSERT INTO incident_events(incident_id, at, kind,"
                        " detail_json) VALUES(?,?,?,?)",
                        (incident["incident_id"], ledger.iso(now),
                         "forward_failed",
                         json.dumps({"error": str(exc)[:200]})),
                    )
                if failover_exceeded(conn, incident, config, now):
                    transport(message, config)
                    ledger.mark_notified(conn, incident["incident_id"],
                                         "telegram-failover", message_hash,
                                         now)
                    actions.append({"action": "failover_notified",
                                    "incident_id": incident["incident_id"]})
                else:
                    actions.append({"action": "forward_failed",
                                    "incident_id": incident["incident_id"],
                                    "error": str(exc)[:200]})

    for incident in pending_recoveries(conn, config, now):
        message = format_recovery(incident)
        if dry_run:
            actions.append({"action": "would_notify_recovery",
                            "incident_id": incident["incident_id"]})
            continue
        try:
            if is_owner:
                transport(message, config)
            else:
                recovery_forwarder(incident, config)
        except Exception as exc:
            actions.append({"action": "recovery_delivery_failed",
                            "incident_id": incident["incident_id"],
                            "error": str(exc)[:200]})
            continue
        with conn:
            conn.execute(
                "INSERT INTO incident_events(incident_id, at, kind,"
                " detail_json) VALUES(?,?,?,?)",
                (incident["incident_id"], ledger.iso(now),
                 "recovery_notified", json.dumps({"host": hostname})),
            )
        actions.append({"action": "recovery_notified",
                        "incident_id": incident["incident_id"]})

    if is_owner:
        digest_rows = digest_due(conn, config, now)
        if digest_rows:
            lines = [f"[P3 digest] {len(digest_rows)} unresolved"
                     " low-priority incident(s):"]
            for row in digest_rows:
                lines.append(
                    f"- {row['incident_id']} {row['source']}/"
                    f"{row['event_code']} @{row['venue_or_machine']}"
                    f" x{row['occurrence_count']} since"
                    f" {row['first_observed_at']}")
            message = ledger.redact("\n".join(lines))
            if dry_run:
                actions.append({"action": "would_send_digest",
                                "count": len(digest_rows)})
            else:
                transport(message, config)
                mark_digest_sent(conn, now)
                for row in digest_rows:
                    ledger.mark_notified(
                        conn, row["incident_id"], "telegram-digest",
                        hashlib.sha256(message.encode()).hexdigest()[:16],
                        now)
                actions.append({"action": "digest_sent",
                                "count": len(digest_rows)})
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--hostname", default=socket.gethostname())
    parser.add_argument("--now", default=None,
                        help="ISO timestamp override for deterministic runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        config = ledger.load_config(args.config)
        db_path = args.db or Path(os.path.expanduser(
            config.get("ledger_db", str(ledger.DEFAULT_DB))))
        conn = ledger.connect(db_path)
        now = (ledger.parse_iso(args.now, "--now") if args.now
               else ledger.utcnow())
        actions = run_pass(
            conn, config, hostname=args.hostname, now=now,
            transport=send_telegram, forwarder=forward_to_owner,
            recovery_forwarder=forward_recovery_to_owner,
            dry_run=args.dry_run,
        )
        for action in actions:
            print(json.dumps(action, sort_keys=True))
        failures = [a for a in actions
                    if a["action"] in ("forward_failed",
                                       "recovery_delivery_failed")]
        return 0 if not failures else 1
    except ledger.Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
