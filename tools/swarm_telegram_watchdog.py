#!/usr/bin/env python3
"""Notify Telegram about DOIN campaign completion and swarm health problems."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_SCHEMA = "agent_multi.swarm_telegram_watchdog.v1"


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
        Path.home() / "Documents/GitHub/financial-data/_metadata/.env"
    )


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
    for chunk in split_telegram_text(text):
        body = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": chunk,
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        request = urllib.request.Request(endpoint, data=body, method="POST")
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(
                response.read().decode("utf-8", errors="replace")
            )
        if not result.get("ok"):
            raise RuntimeError("Telegram rejected the swarm notification")


def split_telegram_text(text: str, limit: int = 3800) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


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
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def http_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"non-object response from {url}")
    return value


def parse_time(value: Any) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_notification_owner(
    plan: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    now: float,
    stale_seconds: float,
) -> str | None:
    reports = snapshot.get("participants") or {}
    for participant in plan.get("participants") or []:
        node_id = str(participant.get("node_id") or "")
        report = reports.get(node_id) or {}
        status = report.get("status") or {}
        updated_at = parse_time(status.get("updated_at"))
        if (
            report.get("online")
            and updated_at is not None
            and now - updated_at <= stale_seconds
        ):
            return node_id
    return None


def event(
    key: str,
    message: str,
    *,
    severity: str = "error",
    grace_seconds: float = 300.0,
) -> dict[str, Any]:
    return {
        "key": key,
        "message": message,
        "severity": severity,
        "grace_seconds": grace_seconds,
    }


def collect_global_events(
    plan: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    now: float,
    stale_seconds: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    reports = snapshot.get("participants") or {}
    online_statuses: list[dict[str, Any]] = []

    for participant in plan.get("participants") or []:
        node_id = str(participant.get("node_id") or "")
        report = reports.get(node_id) or {}
        if not report.get("online"):
            result.append(event(
                f"participant_offline:{node_id}",
                "🚨🚨🚨 SWARM MACHINE OFFLINE 🚨🚨🚨\n"
                f"machine: {node_id}\n"
                f"error: {report.get('error') or 'campaign supervisor unreachable'}",
            ))
            continue
        status = report.get("status") or {}
        online_statuses.append(status)
        updated_at = parse_time(status.get("updated_at"))
        if updated_at is None or now - updated_at > stale_seconds:
            age = "unknown" if updated_at is None else f"{(now - updated_at) / 60:.1f} min"
            result.append(event(
                f"participant_stale:{node_id}",
                "🚨🚨🚨 SWARM MACHINE FROZEN OR STALE 🚨🚨🚨\n"
                f"machine: {node_id}\n"
                f"supervisor status age: {age}",
            ))
        for alert in status.get("alerts") or []:
            code = str(alert.get("code") or "unknown")
            result.append(event(
                f"supervisor_alert:{node_id}:{code}",
                "🚨🚨🚨 SWARM SUPERVISOR ALERT 🚨🚨🚨\n"
                f"machine: {node_id}\n"
                f"code: {code}\n"
                f"severity: {alert.get('severity') or 'error'}\n"
                f"message: {alert.get('message') or 'no detail'}",
                severity=str(alert.get("severity") or "error"),
            ))
        if status.get("phase") == "running":
            for worker_id, worker in (status.get("workers") or {}).items():
                if worker.get("status") != "running" or worker.get("api_error"):
                    result.append(event(
                        f"worker_unhealthy:{node_id}:{worker_id}",
                        "🚨🚨🚨 DOIN WORKER UNHEALTHY 🚨🚨🚨\n"
                        f"machine: {node_id}\n"
                        f"worker: {worker_id}\n"
                        f"status: {worker.get('status')}\n"
                        f"API error: {worker.get('api_error') or 'none'}",
                    ))
                last_seen = parse_time(worker.get("last_seen"))
                if last_seen is None or now - last_seen > stale_seconds:
                    age = "unknown" if last_seen is None else f"{(now - last_seen) / 60:.1f} min"
                    result.append(event(
                        f"worker_stale:{node_id}:{worker_id}",
                        "🚨🚨🚨 DOIN WORKER FROZEN OR STALE 🚨🚨🚨\n"
                        f"machine: {node_id}\n"
                        f"worker: {worker_id}\n"
                        f"worker heartbeat age: {age}",
                    ))

    active = [
        status for status in online_statuses
        if status.get("phase") not in {"complete", "stopped"}
    ]
    for field, label in (
        ("plan_hash", "plan hashes"),
        ("job_id", "optimization jobs"),
        ("domain_id", "optimization domains"),
    ):
        values = {str(item.get(field)) for item in active if item.get(field)}
        if len(values) > 1:
            result.append(event(
                f"parallel_swarm:{field}",
                "🚨🚨🚨 PARALLEL SWARMS DETECTED 🚨🚨🚨\n"
                f"inconsistent {label}: {', '.join(sorted(values))}\n"
                "The fleet is not collaborating on one optimization.",
            ))

    lineages: set[tuple[str, str]] = set()
    generations: set[tuple[Any, Any]] = set()
    for status in active:
        for worker in (status.get("workers") or {}).values():
            if worker.get("status") != "running":
                continue
            lineage = worker.get("bootstrap_evidence") or {}
            genesis = lineage.get("genesis_hash")
            population = lineage.get("population_block_hash")
            if genesis and population:
                lineages.add((str(genesis), str(population)))
            pool = worker.get("shared_population") or {}
            if pool.get("generation") is not None:
                generations.add((pool.get("generation"), pool.get("pop_size")))
    if len(lineages) > 1:
        result.append(event(
            "parallel_swarm:blockchain_lineage",
            "🚨🚨🚨 PARALLEL BLOCKCHAINS DETECTED 🚨🚨🚨\n"
            f"distinct genesis/population lineages: {len(lineages)}\n"
            "Workers are not sharing the same optimization swarm.",
        ))
    if len(generations) > 1:
        result.append(event(
            "parallel_swarm:generation",
            "🚨🚨🚨 SWARM GENERATION DIVERGENCE 🚨🚨🚨\n"
            f"reported generation/population pairs: {sorted(generations)}",
        ))
    return result


def scan_local_doin_processes() -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        args = [
            item.decode("utf-8", errors="replace")
            for item in raw.split(b"\0")
            if item
        ]
        if "doin_node.cli" not in " ".join(args):
            continue
        config_value = None
        for index, value in enumerate(args[:-1]):
            if value == "--config":
                config_value = args[index + 1]
                break
        config_path = None
        domain_id = None
        if config_value:
            candidate = Path(config_value).expanduser()
            try:
                cwd = (entry / "cwd").resolve()
                config_path = (
                    candidate if candidate.is_absolute() else cwd / candidate
                ).resolve()
                config = read_json(config_path)
                domains = config.get("domains") or []
                if len(domains) == 1:
                    domain_id = domains[0].get("domain_id")
            except OSError:
                config_path = candidate
        processes.append({
            "pid": int(entry.name),
            "config_path": str(config_path) if config_path else None,
            "domain_id": domain_id,
        })
    return processes


def expected_local_configs(
    profile: dict[str, Any],
    plan: dict[str, Any],
    job_id: str | None,
) -> set[str]:
    if not job_id:
        return set()
    job = next(
        (item for item in plan.get("jobs") or [] if item.get("job_id") == job_id),
        None,
    )
    if not job:
        return set()
    expected: set[str] = set()
    for worker_id, worker_profile in (profile.get("workers") or {}).items():
        relative = (job.get("worker_configs") or {}).get(worker_id)
        if not relative:
            continue
        root = Path(str(worker_profile["doin_node_root"])).expanduser()
        path = Path(str(relative)).expanduser()
        expected.add(str((path if path.is_absolute() else root / path).resolve()))
    return expected


def collect_local_process_events(
    *,
    node_id: str,
    phase: str | None,
    expected_configs: set[str],
    processes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    configs = Counter(
        str(item.get("config_path"))
        for item in processes
        if item.get("config_path")
    )
    domains = {
        str(item.get("domain_id"))
        for item in processes
        if item.get("domain_id")
    }
    duplicates = {path: count for path, count in configs.items() if count > 1}
    actual = set(configs)
    extras = actual - expected_configs
    missing = expected_configs - actual if phase == "running" else set()
    too_many = len(processes) > len(expected_configs)
    if len(domains) > 1 or duplicates or extras or missing or too_many:
        process_text = "\n".join(
            f"pid {item['pid']}: {item.get('domain_id') or '?'} "
            f"[{item.get('config_path') or 'no config'}]"
            for item in processes
        ) or "no DOIN processes found"
        result.append(event(
            f"local_parallel_or_missing:{node_id}",
            "🚨🚨🚨 LOCAL DOIN PROCESS ANOMALY 🚨🚨🚨\n"
            f"machine: {node_id}\n"
            f"expected processes: {len(expected_configs)}\n"
            f"observed processes: {len(processes)}\n"
            f"distinct domains: {len(domains)}\n"
            f"extra configs: {len(extras)}; missing configs: {len(missing)}\n"
            f"{process_text}",
            grace_seconds=300.0,
        ))
    return result


def progress_signature(snapshot: dict[str, Any]) -> str | None:
    rows: list[dict[str, Any]] = []
    for node_id, report in sorted((snapshot.get("participants") or {}).items()):
        if not report.get("online"):
            continue
        status = report.get("status") or {}
        if status.get("phase") != "running":
            continue
        for worker_id, worker in sorted((status.get("workers") or {}).items()):
            pool = worker.get("shared_population") or {}
            rows.append({
                "node": node_id,
                "worker": worker_id,
                "job": status.get("job_id"),
                "generation": pool.get("generation"),
                "evaluated": pool.get("evaluated"),
                "claimed": pool.get("claimed"),
                "free": pool.get("free"),
                "chain_height": worker.get("chain_height"),
                "best": worker.get("best_performance"),
            })
    return canonical_hash(rows) if rows else None


def update_progress_event(
    snapshot: dict[str, Any],
    state: dict[str, Any],
    *,
    now: float,
    stall_seconds: float,
) -> dict[str, Any] | None:
    signature = progress_signature(snapshot)
    progress = state.setdefault("progress", {})
    if not signature:
        progress.clear()
        return None
    if progress.get("signature") != signature:
        progress.update({"signature": signature, "changed_at": now})
        return None
    changed_at = float(progress.get("changed_at", now))
    if now - changed_at < stall_seconds:
        return None
    return event(
        "swarm_progress_stalled",
        "🚨🚨🚨 SWARM PROGRESS STALLED 🚨🚨🚨\n"
        f"no observable generation, candidate, chain or champion progress for "
        f"{(now - changed_at) / 60:.0f} minutes",
        grace_seconds=0,
    )


def completion_key(row: dict[str, Any]) -> str:
    return ":".join([
        str(row.get("job_id") or "unknown"),
        str(row.get("artifact_sha256") or "no-artifact"),
    ])


def normalize_completion_records(records: dict[str, Any]) -> None:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in records.items():
        parts = str(key).split(":")
        target = ":".join(parts[:2]) if len(parts) >= 2 else str(key)
        current = normalized.setdefault(target, {})
        if isinstance(value, dict):
            for field, item in value.items():
                if field.endswith("_at") and current.get(field) is not None:
                    try:
                        current[field] = max(float(current[field]), float(item))
                    except (TypeError, ValueError):
                        current[field] = item
                else:
                    current[field] = item
    records.clear()
    records.update(normalized)


def format_number(value: Any, digits: int = 6) -> str | None:
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return None


def render_metric(value: Any, kind: str) -> str | None:
    try:
        if kind == "percent":
            return f"{float(value) * 100:+.4f}%"
        if kind == "pct_value":
            return f"{float(value):.4f}%"
        if kind == "integer":
            return str(int(value))
        if kind == "money":
            return f"{float(value):,.2f}"
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return None


def format_completion(
    row: dict[str, Any],
    plan_jobs: list[dict[str, Any]],
) -> str:
    metrics = row.get("metrics") or {}
    metric_name = (
        metrics.get("optimization_metric")
        or metrics.get("selection_metric")
        or "configured fitness"
    )
    lines = [
        "🏁🏆 SWARM OPTIMIZATION COMPLETED 🏆🏁",
        f"job: {row.get('job_id')}",
        f"domain: {row.get('domain_id')}",
        f"completed: {row.get('completed_at')}",
        f"fitness ({metric_name}): {format_number(row.get('champion_fitness')) or 'N/A'}",
    ]
    metric_specs = (
        ("mean_weekly_return", "mean weekly return", "percent"),
        ("annual_return", "annual return", "percent"),
        ("mean_weekly_rap", "mean weekly RAP", "percent"),
        ("annual_rap", "annual RAP", "percent"),
        ("total_return", "total return", "percent"),
        ("risk_adjusted_total_return", "risk-adjusted total return", "percent"),
        ("max_drawdown_pct", "max drawdown", "pct_value"),
        ("sharpe_ratio", "Sharpe ratio", "number"),
        ("trades_total", "trades", "integer"),
        ("final_equity", "final equity", "money"),
        ("validation_selection_score", "validation selection score", "number"),
        ("train_validation_l1_score", "train/validation L1 score", "number"),
    )
    for key, label, kind in metric_specs:
        if key not in metrics:
            continue
        value = metrics[key]
        rendered = render_metric(value, kind)
        if rendered is not None:
            lines.append(f"{label}: {rendered}")
    lines.extend([
        f"champion peer: {str(row.get('champion_peer_id') or 'unknown')[:16]}",
        f"artifact: {str(row.get('artifact_sha256') or 'missing')[:16]} "
        f"({row.get('artifact_format') or 'unknown format'})",
    ])
    ordinal = int(row.get("ordinal", -1))
    next_job = next(
        (
            item for item in plan_jobs
            if int(item.get("ordinal", -1)) == ordinal + 1
        ),
        None,
    )
    lines.append(
        f"next job: {next_job.get('job_id') if next_job else 'campaign queue complete'}"
    )
    lines.append(
        "metric period: exactly as stored by the completed optimizer; "
        "not annualized unless explicitly labeled annual"
    )
    return "\n".join(lines)


def due_events(
    current: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    now: float,
    repeat_seconds: float,
) -> tuple[list[str], list[str], list[str]]:
    records = state.setdefault("events", {})
    current_by_key = {item["key"]: item for item in current}
    messages: list[str] = []
    sent_keys: list[str] = []
    recovered_keys: list[str] = []
    for key, item in current_by_key.items():
        record = records.setdefault(key, {})
        if not record.get("active"):
            record["first_seen_at"] = now
        record.update({
            "active": True,
            "message": item["message"],
            "severity": item["severity"],
            "last_seen_at": now,
        })
        age = now - float(record.get("first_seen_at", now))
        last_sent = float(record.get("last_sent_at", 0))
        if (
            age >= float(item.get("grace_seconds", 0))
            and (not last_sent or now - last_sent >= repeat_seconds)
        ):
            messages.append(item["message"])
            sent_keys.append(key)
    for key, record in records.items():
        if not record.get("active") or key in current_by_key:
            continue
        if record.get("last_sent_at"):
            previous = str(record.get("message") or "unknown").splitlines()[0]
            messages.append(
                "✅ SWARM ALERT RECOVERED\n"
                f"event: {key}\n"
                f"previous condition: {previous}"
            )
            recovered_keys.append(key)
        record.update({"active": False, "recovered_at": now})
    return messages, sent_keys, recovered_keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--machine", default=socket.gethostname())
    parser.add_argument("--stale-minutes", type=float, default=10.0)
    parser.add_argument("--stall-minutes", type=float, default=120.0)
    parser.add_argument("--repeat-minutes", type=float, default=60.0)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-alert", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_path = args.profile.expanduser().resolve()
    profile = read_json(profile_path)
    node_id = str(profile.get("node_id") or args.machine)
    plan_path = Path(str(profile["plan_file"])).expanduser()
    if not plan_path.is_absolute():
        plan_path = profile_path.parent / plan_path
    plan = read_json(plan_path.resolve())
    plan_id = str(plan.get("plan_id") or "unknown-plan")
    state_path = args.state_file or (
        Path.home()
        / ".local/state/agent-multi/swarm-telegram-watchdog"
        / plan_id
        / f"{node_id}.json"
    )
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Swarm watchdog skipped: another check is running")
            return 0

        now = time.time()
        state = read_json(state_path)
        if state.get("schema_version") != STATE_SCHEMA:
            state = {"schema_version": STATE_SCHEMA, "events": {}, "completions": {}}

        port = int(profile.get("listen_port", 8795))
        snapshot: dict[str, Any] = {}
        local_events: list[dict[str, Any]] = []
        try:
            snapshot = http_json(f"http://127.0.0.1:{port}/api/network")
        except Exception as exc:
            local_events.append(event(
                f"local_supervisor_unavailable:{node_id}",
                "🚨🚨🚨 LOCAL CAMPAIGN SUPERVISOR UNAVAILABLE 🚨🚨🚨\n"
                f"machine: {node_id}\n"
                f"error: {exc}",
                grace_seconds=300.0,
            ))

        local_status = (
            ((snapshot.get("participants") or {}).get(node_id) or {}).get("status")
            or {}
        )
        processes = scan_local_doin_processes()
        expected = expected_local_configs(
            profile, plan, local_status.get("job_id")
        )
        local_events.extend(collect_local_process_events(
            node_id=node_id,
            phase=local_status.get("phase"),
            expected_configs=expected,
            processes=processes,
        ))

        stale_seconds = args.stale_minutes * 60.0
        owner = (
            select_notification_owner(
                plan, snapshot, now=now, stale_seconds=stale_seconds
            )
            if snapshot else None
        )
        global_events: list[dict[str, Any]] = []
        if owner == node_id:
            global_events.extend(collect_global_events(
                plan, snapshot, now=now, stale_seconds=stale_seconds
            ))
            stalled = update_progress_event(
                snapshot,
                state,
                now=now,
                stall_seconds=args.stall_minutes * 60.0,
            )
            if stalled:
                global_events.append(stalled)

        completion_messages: list[str] = []
        completion_keys: list[str] = []
        completions = state.setdefault("completions", {})
        normalize_completion_records(completions)
        history = snapshot.get("history") or []
        if owner == node_id:
            for row in history:
                if row.get("status") != "completed":
                    continue
                key = completion_key(row)
                record = completions.setdefault(key, {})
                if not (
                    record.get("notified_at")
                    or record.get("acknowledged_by_owner_at")
                ):
                    completion_messages.append(
                        format_completion(row, snapshot.get("plan_jobs") or [])
                    )
                    completion_keys.append(key)
        elif owner:
            for row in history:
                if row.get("status") != "completed":
                    continue
                key = completion_key(row)
                record = completions.setdefault(key, {})
                record.update({
                    "acknowledged_by_owner_at": now,
                    "notification_owner": owner,
                })

        messages, sent_event_keys, recovered_event_keys = due_events(
            local_events + global_events,
            state,
            now=now,
            repeat_seconds=args.repeat_minutes * 60.0,
        )
        messages = completion_messages + messages
        if args.test_alert:
            messages.append(
                "🧪 DOIN SWARM TELEGRAM WATCHDOG TEST\n"
                f"machine: {node_id}\n"
                f"plan: {plan_id}\n"
                f"notification owner: {owner or 'none'}\n"
                f"local DOIN processes: {len(processes)}\n"
                f"completed campaigns visible: "
                f"{sum(row.get('status') == 'completed' for row in history)}"
            )

        if messages:
            text = "\n\n".join(messages)
            if args.dry_run:
                print(text)
            else:
                send_telegram(text)
                for key in completion_keys:
                    completions[key]["notified_at"] = now
                for key in sent_event_keys:
                    state["events"][key]["last_sent_at"] = now
                for key in recovered_event_keys:
                    state["events"][key]["recovery_sent_at"] = now
                state["last_notification_at"] = now
        else:
            print(
                f"Swarm watchdog healthy: node={node_id}, owner={owner}, "
                f"processes={len(processes)}, history={len(history)}"
            )

        state.update({
            "schema_version": STATE_SCHEMA,
            "node_id": node_id,
            "plan_id": plan_id,
            "notification_owner": owner,
            "last_check_at": now,
            "last_check_iso": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "local_processes": processes,
        })
        atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
