#!/usr/bin/env python3
"""Monitor host and DOIN cgroup memory pressure and notify Telegram."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


STATE_SCHEMA = "agent_multi.memory_pressure_watchdog.v1"
DEFAULT_STATE = Path.home() / ".local/state/agent-multi/memory-pressure-watchdog/state.json"
CGROUP_ROOT = Path("/sys/fs/cgroup")


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


def send_telegram(text: str) -> None:
    load_env_file(Path.home() / ".hermes/.env")
    load_env_file(Path.home() / "Documents/GitHub/financial-data/_metadata/.env")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = (
        os.environ.get("PROJECT3_TELEGRAM_CHAT_ID", "").strip()
        or os.environ.get("TELEGRAM_HOME_CHANNEL", "").strip()
    )
    if not token or not chat_id:
        raise RuntimeError("Hermes Telegram bot or home channel is not configured")
    body = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.loads(response.read().decode("utf-8", errors="replace"))
    if not result.get("ok"):
        raise RuntimeError("Telegram rejected the watchdog notification")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_meminfo(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for raw in text.splitlines():
        if ":" not in raw:
            continue
        key, remainder = raw.split(":", 1)
        parts = remainder.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) * 1024
    return values


def parse_key_values(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) == 2 and parts[1].isdigit():
            values[parts[0]] = int(parts[1])
    return values


def systemd_value(service: str, property_name: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", service, f"--property={property_name}", "--value"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip()


def read_int(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def user_service_cgroup(
    service: str,
    *,
    cgroup_root: Path = CGROUP_ROOT,
    uid: int | None = None,
) -> Path:
    user_id = os.getuid() if uid is None else uid
    return (
        cgroup_root
        / "user.slice"
        / f"user-{user_id}.slice"
        / f"user@{user_id}.service"
        / "app.slice"
        / service
    )


def cgroup_snapshot(cgroup_path: Path) -> dict[str, Any]:
    processes_path = cgroup_path / "cgroup.procs"
    processes = processes_path.read_text(encoding="utf-8").split() if processes_path.exists() else []
    events_path = cgroup_path / "memory.events"
    events = parse_key_values(events_path.read_text(encoding="utf-8")) if events_path.exists() else {}
    return {
        "service_memory_bytes": read_int(cgroup_path / "memory.current"),
        "service_memory_peak_bytes": read_int(cgroup_path / "memory.peak"),
        "service_active_state": "active" if processes else "inactive",
        "memory_events": events,
    }


def read_snapshot(
    service: str,
    *,
    cgroup_root: Path = CGROUP_ROOT,
    uid: int | None = None,
) -> dict[str, Any]:
    meminfo = parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
    total = meminfo.get("MemTotal", 0)
    available = meminfo.get("MemAvailable", 0)
    swap_total = meminfo.get("SwapTotal", 0)
    swap_free = meminfo.get("SwapFree", 0)
    try:
        control_group = systemd_value(service, "ControlGroup")
        cgroup_path = cgroup_root / control_group.lstrip("/")
        service_snapshot = {
            "service_memory_bytes": int(systemd_value(service, "MemoryCurrent") or 0),
            "service_memory_peak_bytes": int(systemd_value(service, "MemoryPeak") or 0),
            "service_active_state": systemd_value(service, "ActiveState"),
            "memory_events": cgroup_snapshot(cgroup_path)["memory_events"],
            "service_snapshot_source": "systemd",
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        # User cron does not inherit the user-systemd D-Bus environment. The
        # cgroup hierarchy remains authoritative and readable without that bus.
        cgroup_path = user_service_cgroup(
            service,
            cgroup_root=cgroup_root,
            uid=uid,
        )
        service_snapshot = cgroup_snapshot(cgroup_path)
        service_snapshot["service_snapshot_source"] = "cgroupfs"
    snapshot = {
        "mem_total_bytes": total,
        "mem_available_bytes": available,
        "mem_available_fraction": available / total if total else 0.0,
        "swap_total_bytes": swap_total,
        "swap_used_bytes": max(0, swap_total - swap_free),
        "swap_used_fraction": (swap_total - swap_free) / swap_total if swap_total else 0.0,
    }
    snapshot.update(service_snapshot)
    return snapshot


def classify_pressure(
    snapshot: dict[str, Any],
    previous_events: dict[str, int],
    *,
    warning_available_gib: float,
    critical_available_gib: float,
    warning_service_gib: float,
    critical_service_gib: float,
) -> tuple[str, list[str]]:
    gib = 1024 ** 3
    available = float(snapshot["mem_available_bytes"]) / gib
    service = float(snapshot["service_memory_bytes"]) / gib
    swap_fraction = float(snapshot["swap_used_fraction"])
    events = snapshot.get("memory_events") or {}
    oom_delta = int(events.get("oom_kill", 0)) - int(previous_events.get("oom_kill", 0))
    reasons: list[str] = []
    severity = "healthy"
    if available <= warning_available_gib:
        severity = "warning"
        reasons.append(f"available RAM {available:.1f} GiB")
    if service >= warning_service_gib:
        severity = "warning"
        reasons.append(f"DOIN cgroup RAM {service:.1f} GiB")
    if swap_fraction >= 0.50:
        severity = "warning"
        reasons.append(f"swap used {swap_fraction:.0%}")
    if available <= critical_available_gib:
        severity = "critical"
    if service >= critical_service_gib:
        severity = "critical"
    if swap_fraction >= 0.80:
        severity = "critical"
    if oom_delta > 0:
        severity = "critical"
        reasons.append(f"new cgroup OOM kills {oom_delta}")
    if snapshot.get("service_active_state") != "active":
        severity = "critical"
        reasons.append(f"service state {snapshot.get('service_active_state')}")
    return severity, reasons


def format_gib(value: Any) -> str:
    return f"{float(value) / (1024 ** 3):.1f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default=socket.gethostname())
    parser.add_argument("--service", default="doin-campaign-supervisor.service")
    parser.add_argument("--warning-available-gib", type=float, default=4.0)
    parser.add_argument("--critical-available-gib", type=float, default=2.0)
    parser.add_argument("--warning-service-gib", type=float, default=18.0)
    parser.add_argument("--critical-service-gib", type=float, default=21.0)
    parser.add_argument("--repeat-minutes", type=float, default=60.0)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-alert", action="store_true")
    args = parser.parse_args()
    if args.critical_available_gib >= args.warning_available_gib:
        raise ValueError("critical available RAM must be below warning RAM")
    if args.critical_service_gib <= args.warning_service_gib:
        raise ValueError("critical service RAM must exceed warning service RAM")

    lock_path = args.state_file.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Memory watchdog skipped: another check is running")
            return 0

        now = time.time()
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        state = read_json(args.state_file)
        if state.get("schema_version") != STATE_SCHEMA:
            state = {"schema_version": STATE_SCHEMA}
        snapshot = read_snapshot(args.service)
        previous_events = state.get("memory_events") or {}
        severity, reasons = classify_pressure(
            snapshot,
            previous_events,
            warning_available_gib=args.warning_available_gib,
            critical_available_gib=args.critical_available_gib,
            warning_service_gib=args.warning_service_gib,
            critical_service_gib=args.critical_service_gib,
        )
        previous_severity = state.get("severity", "healthy")
        repeat_due = now - float(state.get("last_notification_at", 0)) >= args.repeat_minutes * 60
        notify = severity != "healthy" and (severity != previous_severity or repeat_due)
        recovered = severity == "healthy" and previous_severity != "healthy"
        if args.test_alert:
            severity = "critical"
            reasons.append("watchdog test requested")
            notify = True

        if notify or recovered:
            prefix = "MEMORY PRESSURE RECOVERED" if recovered else f"MEMORY PRESSURE {severity.upper()}"
            text = (
                f"{prefix}\n"
                f"machine: {args.machine}\n"
                f"time: {timestamp}\n"
                f"available RAM: {format_gib(snapshot['mem_available_bytes'])}\n"
                f"DOIN cgroup RAM: {format_gib(snapshot['service_memory_bytes'])}\n"
                f"DOIN cgroup peak: {format_gib(snapshot['service_memory_peak_bytes'])}\n"
                f"swap used: {format_gib(snapshot['swap_used_bytes'])} "
                f"({snapshot['swap_used_fraction']:.0%})\n"
                f"service: {snapshot['service_active_state']}"
            )
            if reasons:
                text += "\nreasons: " + "; ".join(reasons)
            if args.dry_run:
                print(text)
            else:
                send_telegram(text)
                state["last_notification_at"] = now
        else:
            print(
                f"Memory watchdog {severity}: {args.machine}, "
                f"available={format_gib(snapshot['mem_available_bytes'])}, "
                f"service={format_gib(snapshot['service_memory_bytes'])}, "
                f"swap={snapshot['swap_used_fraction']:.0%}"
            )

        state.update({
            "schema_version": STATE_SCHEMA,
            "machine": args.machine,
            "service": args.service,
            "severity": severity,
            "last_check_at": now,
            "last_check_iso": timestamp,
            "memory_events": snapshot.get("memory_events") or {},
            "snapshot": snapshot,
        })
        atomic_json(args.state_file, state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Memory watchdog failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
