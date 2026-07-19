#!/usr/bin/env python3
"""Monitor local NVIDIA GPUs and notify the Hermes Telegram channel."""
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


STATE_SCHEMA = "agent_multi.gpu_temperature_watchdog.v1"
DEFAULT_STATE = (
    Path.home() / ".local/state/agent-multi/gpu-temperature-watchdog/state.json"
)


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
    load_env_file(
        Path.home()
        / "Documents/GitHub/financial-data/_metadata/.env"
    )


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


def parse_nvidia_smi(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in output.splitlines():
        parts = [part.strip() for part in raw.split(",", 4)]
        if len(parts) != 5:
            continue
        index, name, temperature, utilization, power = parts
        rows.append({
            "index": int(index),
            "name": name,
            "temperature_c": float(temperature),
            "utilization_pct": float(utilization),
            "power_w": None if power in {"[N/A]", "N/A"} else float(power),
        })
    return rows


def read_gpus() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows = parse_nvidia_smi(result.stdout)
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return rows


def send_telegram(text: str) -> None:
    load_notification_environment()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = (
        os.environ.get("PROJECT3_TELEGRAM_CHAT_ID", "").strip()
        or os.environ.get("TELEGRAM_HOME_CHANNEL", "").strip()
    )
    if not token or not chat_id:
        raise RuntimeError("Hermes Telegram bot or home channel is not configured")
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
        raise RuntimeError("Telegram rejected the watchdog notification")


def format_gpu(row: dict[str, Any]) -> str:
    power = row.get("power_w")
    power_text = "N/A" if power is None else f"{power:.1f} W"
    return (
        f"GPU {row['index']}: {row['name']}\n"
        f"temperature: {row['temperature_c']:.0f} C\n"
        f"utilization: {row['utilization_pct']:.0f}%\n"
        f"power: {power_text}"
    )


def event_due(
    event: dict[str, Any],
    *,
    active: bool,
    now: float,
    repeat_seconds: float,
) -> bool:
    if not active:
        return True
    return now - float(event.get("last_sent_at", 0)) >= repeat_seconds


def active_event_keys(state: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key, event in (state.get("events") or {}).items()
        if isinstance(event, dict) and bool(event.get("active"))
    )


def evaluate(
    *,
    machine: str,
    gpus: list[dict[str, Any]],
    state: dict[str, Any],
    threshold: float,
    recovery_threshold: float,
    expected_gpus: int,
    repeat_seconds: float,
    now: float,
) -> tuple[list[str], list[str]]:
    events = state.setdefault("events", {})
    messages: list[str] = []
    sent_keys: list[str] = []

    count_key = "gpu_count"
    count_event = events.setdefault(count_key, {})
    count_bad = len(gpus) != expected_gpus
    if count_bad and event_due(
        count_event,
        active=bool(count_event.get("active")),
        now=now,
        repeat_seconds=repeat_seconds,
    ):
        messages.append(
            "🚨🚨🚨 GPU COUNT ALERT 🚨🚨🚨\n"
            f"machine: {machine}\n"
            f"expected GPUs: {expected_gpus}\n"
            f"detected GPUs: {len(gpus)}\n"
            "Check eGPU power, cable, enclosure and NVIDIA device state immediately."
        )
        sent_keys.append(count_key)
    elif (
        not count_bad
        and count_event.get("active")
    ):
        messages.append(
            "✅ GPU COUNT RECOVERED\n"
            f"machine: {machine}\n"
            f"detected GPUs: {len(gpus)}/{expected_gpus}"
        )
        sent_keys.append(count_key)
    count_event.update({
        "active": count_bad,
        "expected": expected_gpus,
        "observed": len(gpus),
    })

    observed_keys: set[str] = set()
    for gpu in gpus:
        key = f"temperature:{gpu['index']}"
        observed_keys.add(key)
        event = events.setdefault(key, {})
        temperature = float(gpu["temperature_c"])
        was_active = bool(event.get("active"))
        is_hot = temperature >= threshold
        recovered = was_active and temperature <= recovery_threshold
        if is_hot and event_due(
            event,
            active=was_active,
            now=now,
            repeat_seconds=repeat_seconds,
        ):
            messages.append(
                "🚨🚨🚨 GPU TEMPERATURE ALERT 🚨🚨🚨\n"
                f"machine: {machine}\n"
                f"limit: {threshold:.0f} C\n"
                f"{format_gpu(gpu)}\n"
                "Inspect external fan power and airflow immediately."
            )
            sent_keys.append(key)
        elif recovered:
            messages.append(
                "✅ GPU TEMPERATURE RECOVERED\n"
                f"machine: {machine}\n"
                f"recovery threshold: {recovery_threshold:.0f} C\n"
                f"{format_gpu(gpu)}"
            )
            sent_keys.append(key)
        event.update({
            "active": is_hot or (was_active and not recovered),
            "temperature_c": temperature,
            "gpu_name": gpu["name"],
            "last_observed_at": now,
        })

    for key, event in events.items():
        if key.startswith("temperature:") and key not in observed_keys:
            event["last_observed_at"] = now
    return messages, sent_keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default=socket.gethostname())
    parser.add_argument("--threshold", type=float, default=78.0)
    parser.add_argument("--recovery-threshold", type=float, default=72.0)
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--repeat-minutes", type=float, default=60.0)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-alert", action="store_true")
    args = parser.parse_args()

    if args.recovery_threshold >= args.threshold:
        raise ValueError("recovery threshold must be below alert threshold")
    if args.expected_gpus < 1:
        raise ValueError("expected GPU count must be positive")

    lock_path = args.state_file.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("GPU watchdog skipped: another check is running")
            return 0

        now = time.time()
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        state = read_json(args.state_file)
        if state.get("schema_version") != STATE_SCHEMA:
            state = {"schema_version": STATE_SCHEMA, "events": {}}

        try:
            gpus = read_gpus()
            nvidia_event = state.setdefault("events", {}).setdefault(
                "nvidia_smi", {}
            )
            nvidia_recovered = bool(nvidia_event.get("active"))
            nvidia_event.update({
                "active": False,
                "error": None,
                "last_observed_at": now,
            })
        except Exception as exc:
            gpus = []
            event = state.setdefault("events", {}).setdefault("nvidia_smi", {})
            messages = []
            sent_keys = []
            if event_due(
                event,
                active=bool(event.get("active")),
                now=now,
                repeat_seconds=args.repeat_minutes * 60.0,
            ):
                messages.append(
                    "🚨🚨🚨 NVIDIA MONITORING FAILURE 🚨🚨🚨\n"
                    f"machine: {args.machine}\n"
                    f"time: {timestamp}\n"
                    f"error: {exc}\n"
                    "GPU temperature cannot be verified. Inspect the machine immediately."
                )
                sent_keys.append("nvidia_smi")
            event.update({
                "active": True,
                "error": str(exc),
                "last_observed_at": now,
            })
        else:
            messages, sent_keys = evaluate(
                machine=args.machine,
                gpus=gpus,
                state=state,
                threshold=args.threshold,
                recovery_threshold=args.recovery_threshold,
                expected_gpus=args.expected_gpus,
                repeat_seconds=args.repeat_minutes * 60.0,
                now=now,
            )
            if nvidia_recovered:
                messages.insert(
                    0,
                    "✅ NVIDIA MONITORING RECOVERED\n"
                    f"machine: {args.machine}\n"
                    f"time: {timestamp}\n"
                    f"detected GPUs: {len(gpus)}/{args.expected_gpus}",
                )
                sent_keys.insert(0, "nvidia_smi")

        if args.test_alert:
            messages.append(
                "🧪 GPU WATCHDOG TEST\n"
                f"machine: {args.machine}\n"
                f"time: {timestamp}\n"
                f"threshold: {args.threshold:.0f} C\n"
                f"detected GPUs: {len(gpus)}/{args.expected_gpus}\n\n"
                + "\n\n".join(format_gpu(gpu) for gpu in gpus)
            )

        if messages:
            text = "\n\n".join(messages)
            if args.dry_run:
                print(text)
            else:
                send_telegram(text)
                for key in sent_keys:
                    state["events"][key]["last_sent_at"] = now
                state["last_notification_at"] = now
        else:
            gpu_summary = ", ".join(
                f"GPU {gpu['index']}={gpu['temperature_c']:.0f}C"
                for gpu in gpus
            )
            active_keys = active_event_keys(state)
            if active_keys:
                print(
                    f"GPU watchdog alert remains active: {args.machine}, "
                    f"events={','.join(active_keys)}, {gpu_summary}"
                )
            else:
                print(f"GPU watchdog healthy: {args.machine}, {gpu_summary}")

        state.update({
            "schema_version": STATE_SCHEMA,
            "machine": args.machine,
            "threshold_c": args.threshold,
            "recovery_threshold_c": args.recovery_threshold,
            "expected_gpus": args.expected_gpus,
            "last_check_at": now,
            "last_check_iso": timestamp,
            "gpus": gpus,
        })
        atomic_json(args.state_file, state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"GPU watchdog failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
