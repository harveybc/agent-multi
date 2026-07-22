#!/usr/bin/env python3
"""Notify Telegram when all swarm GPUs are idle (0% utilization) for too long."""
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_SCHEMA = "agent_multi.gpu_idle_watchdog.v1"

# ── helpers ──────────────────────────────────────────────────────────

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

def load_notification_environment() -> None:
    load_env_file(Path.home() / ".hermes/.env")
    load_env_file(Path.home() / "Documents/GitHub/financial-data/_metadata/.env")

def send_telegram(text: str) -> None:
    load_notification_environment()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = (
        os.environ.get("PROJECT3_TELEGRAM_CHAT_ID", "").strip()
        or os.environ.get("TELEGRAM_HOME_CHANNEL", "").strip()
    )
    if not token or not chat_id:
        raise RuntimeError("Telegram bot or home channel not configured")
    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.loads(response.read().decode("utf-8", errors="replace"))
    if not result.get("ok"):
        raise RuntimeError("Telegram rejected the notification")

def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)

def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}

# ── GPU querying ─────────────────────────────────────────────────────

MACHINES = {
    "omega": {"host": "localhost", "ssh": False},
    "gamma": {"host": "gamma", "ssh": True},
    "dragon": {"host": "dragon", "ssh": True},
}

def _ssh(host: str, command: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             host, command],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return f"SSH_ERROR:{result.stderr.strip()[:200]}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as exc:
        return f"ERROR:{exc}"

def query_gpus(machine: str, info: dict[str, Any]) -> list[dict[str, Any]]:
    cmd = "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu --format=csv,noheader,nounits"
    if info["ssh"]:
        output = _ssh(info["host"], cmd)
    else:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
            output = result.stdout.strip()
        except Exception as exc:
            output = f"ERROR:{exc}"

    if not output or output.startswith("SSH_ERROR") or output.startswith("TIMEOUT") or output.startswith("ERROR"):
        return [{"machine": machine, "error": output or "no output"}]

    gpus = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append({
                    "machine": machine,
                    "name": parts[0],
                    "temp_c": int(parts[1]),
                    "util_pct": int(parts[2]),
                })
            except ValueError:
                gpus.append({"machine": machine, "raw": line})
    return gpus

def all_gpus_idle(gpus: list[dict[str, Any]]) -> bool:
    """True when every reachable GPU reports 0% utilization."""
    usable = [g for g in gpus if "util_pct" in g]
    if not usable:
        return False  # don't alert if we can't query anything
    return all(g["util_pct"] == 0 for g in usable)

# ── main ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-checks", type=int, default=2,
                        help="Consecutive 0%% checks before alerting (default: 2)")
    parser.add_argument("--repeat-minutes", type=float, default=60.0,
                        help="Repeat alert interval (default: 60)")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-alert", action="store_true")
    args = parser.parse_args(argv)

    state_path = args.state_file or (
        Path.home() / ".local/state/agent-multi/gpu-idle-watchdog/state.json"
    )
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("GPU idle watchdog skipped: another check is running")
            return 0

        now = time.time()
        state = read_json(state_path)
        if state.get("schema_version") != STATE_SCHEMA:
            state = {"schema_version": STATE_SCHEMA}

        # Query all machines
        all_gpus: list[dict[str, Any]] = []
        errors: list[str] = []
        for machine, info in MACHINES.items():
            gpus = query_gpus(machine, info)
            all_gpus.extend(gpus)
            for g in gpus:
                if "error" in g:
                    errors.append(f"{machine}: {g['error']}")

        # Build status line
        lines = []
        for g in all_gpus:
            if "error" in g:
                lines.append(f"  {g['machine']}: ❌ {g['error']}")
            elif "util_pct" in g:
                lines.append(
                    f"  {g['machine']} {g['name']}: {g['temp_c']}°C, {g['util_pct']}%"
                )

        idle = all_gpus_idle(all_gpus)
        streak = state.get("idle_streak", 0)

        if idle:
            streak += 1
        else:
            streak = 0

        state["idle_streak"] = streak
        state["last_check_at"] = now
        state["last_check_iso"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Alert logic
        message: str | None = None
        recovery: str | None = None

        if args.test_alert:
            message = (
                "🧪 GPU IDLE WATCHDOG TEST\n"
                + "\n".join(lines)
                + f"\n\nidle streak: {streak}/{args.confirm_checks}"
            )
        elif idle and streak == args.confirm_checks:
            # Just hit the threshold — alert
            last_alert = float(state.get("last_alert_at", 0))
            if not last_alert or now - last_alert >= args.repeat_minutes * 60:
                message = (
                    "⚠️⚠️⚠️ ALL SWARM GPUs IDLE ⚠️⚠️⚠️\n\n"
                    + "\n".join(lines)
                    + f"\n\nSustained 0% utilization for {streak} consecutive checks."
                    + "\nPossible causes: campaign queue complete, supervisor blocked, or workers crashed."
                )
                state["last_alert_at"] = now
        elif idle and streak > args.confirm_checks:
            # Still idle — repeat alert on interval
            last_alert = float(state.get("last_alert_at", 0))
            if last_alert and now - last_alert >= args.repeat_minutes * 60:
                message = (
                    "⚠️⚠️⚠️ ALL SWARM GPUs STILL IDLE ⚠️⚠️⚠️\n\n"
                    + "\n".join(lines)
                    + f"\n\nIdle for {streak} consecutive checks ({(streak * 90 / 60):.0f} min)."
                )
                state["last_alert_at"] = now
        elif not idle and streak == 0 and state.get("last_alert_at"):
            # Recovered from idle
            recovery = (
                "✅ GPU IDLE RECOVERED — SWARM ACTIVE\n\n"
                + "\n".join(lines)
            )
            state.pop("last_alert_at", None)

        atomic_json(state_path, state)

        if message:
            if args.dry_run:
                print(message)
            else:
                send_telegram(message)
        elif recovery:
            if args.dry_run:
                print(recovery)
            else:
                send_telegram(recovery)
        else:
            status = "IDLE" if idle else "ACTIVE"
            print(
                f"GPU idle watchdog: {status}, streak={streak}/{args.confirm_checks}, "
                f"gpus={len(all_gpus)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
