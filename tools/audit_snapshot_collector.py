#!/usr/bin/env python3
"""Collect one compact, redacted cross-front audit evidence snapshot."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import socket
import sqlite3
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "agent_multi.audit_snapshot.v1"
COLLECTOR_VERSION = "1.0.0"
DEFAULT_REPOSITORIES = (
    "agent-multi",
    "trading-contracts",
    "gym-fx",
    "heuristic-strategy",
    "doin-core",
    "doin-node",
    "doin-plugins",
    "prediction_provider",
    "lts",
    "financial-data",
    "predictor",
)
DEFAULT_OUTPUT_DIR = (
    Path.home() / ".local/state/agent-multi/audit-snapshots"
)
DEFAULT_PAPER_WATCHDOG = (
    Path.home() / ".local/state/lts/paper-execution-watchdog/latest.json"
)
DEFAULT_TEST_EVIDENCE = (
    Path.home() / ".local/state/agent-multi/audit-test-evidence/latest.json"
)
MAX_ERROR_CHARS = 300
SENSITIVE_KEY_MARKERS = (
    "password",
    "secret",
    "token",
    "credential",
    "raw_account_id",
    "api_key",
)


REMOTE_MACHINE_SCRIPT = r"""
import json, os, shutil, subprocess
from pathlib import Path

def meminfo():
    result = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields = value.strip().split()
        if fields:
            result[key] = int(fields[0]) * 1024
    return result

def gpus():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"[:300]
    rows = []
    for raw in completed.stdout.splitlines():
        fields = [part.strip() for part in raw.split(",", 5)]
        if len(fields) != 6:
            continue
        index, name, util, used, total, temperature = fields
        rows.append({
            "index": int(index),
            "name": name,
            "utilization_pct": float(util),
            "memory_used_mib": float(used),
            "memory_total_mib": float(total),
            "temperature_c": float(temperature),
        })
    return rows, None

def memory_events():
    uid = os.getuid()
    path = Path(
        f"/sys/fs/cgroup/user.slice/user-{uid}.slice/"
        f"user@{uid}.service/app.slice/"
        "doin-campaign-supervisor.service/memory.events"
    )
    result = {}
    try:
        for line in path.read_text().splitlines():
            key, value = line.split(None, 1)
            result[key] = int(value)
    except Exception:
        pass
    return result

memory = meminfo()
disk = shutil.disk_usage(Path.home())
gpu_rows, gpu_error = gpus()
print(json.dumps({
    "hostname": os.uname().nodename,
    "uptime_seconds": float(Path("/proc/uptime").read_text().split()[0]),
    "gpus": gpu_rows,
    "gpu_error": gpu_error,
    "memory": {
        "total_bytes": memory.get("MemTotal"),
        "available_bytes": memory.get("MemAvailable"),
        "swap_total_bytes": memory.get("SwapTotal"),
        "swap_free_bytes": memory.get("SwapFree"),
    },
    "disk": {
        "total_bytes": disk.total,
        "free_bytes": disk.free,
    },
    "campaign_memory_events": memory_events(),
}, sort_keys=True))
"""


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def bounded_error(exc: BaseException) -> str:
    value = f"{type(exc).__name__}: {exc}"
    value = value.replace(str(Path.home()), "%h")
    return value[:MAX_ERROR_CHARS]


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if any(marker in normalized for marker in SENSITIVE_KEY_MARKERS):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = redact_value(item)
        return result
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(Path.home()), "%h")
    return value


def command_text(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 8.0,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def collect_repository(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not (path / ".git").exists():
        return {"available": False, "reason": "repository_missing"}
    try:
        try:
            branch = command_text(
                ["git", "symbolic-ref", "--short", "-q", "HEAD"],
                cwd=path,
            ) or "detached"
        except subprocess.CalledProcessError:
            branch = "detached"
        head = command_text(["git", "rev-parse", "HEAD"], cwd=path)
        dirty = command_text(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=path,
        )
        try:
            upstream = command_text(
                ["git", "rev-parse", "--abbrev-ref", "@{upstream}"],
                cwd=path,
            )
            counts = command_text(
                [
                    "git",
                    "rev-list",
                    "--left-right",
                    "--count",
                    "HEAD...@{upstream}",
                ],
                cwd=path,
            ).split()
            ahead, behind = int(counts[0]), int(counts[1])
        except (subprocess.CalledProcessError, ValueError, IndexError):
            upstream, ahead, behind = None, None, None
        return {
            "available": True,
            "branch": branch,
            "head": head,
            "dirty_count": len(dirty.splitlines()) if dirty else 0,
            "upstream": upstream,
            "ahead": ahead,
            "behind": behind,
        }
    except Exception as exc:
        return {"available": False, "reason": bounded_error(exc)}


def collect_provenance(
    root: Path,
    repositories: Iterable[str] = DEFAULT_REPOSITORIES,
) -> dict[str, Any]:
    return {
        name: collect_repository(root, name)
        for name in repositories
    }


def http_json(url: str, timeout: float = 8.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("HTTP response is not a JSON object")
    return value


def compact_worker(worker: dict[str, Any]) -> dict[str, Any]:
    candidate = worker.get("candidate") or {}
    pool = worker.get("shared_population") or {}
    optimization = worker.get("optimization") or {}
    domains = optimization.get("domains") or []
    domain = domains[0] if domains and isinstance(domains[0], dict) else {}
    return {
        "status": worker.get("status"),
        "last_seen": worker.get("last_seen"),
        "restart_count": worker.get("restart_count"),
        "best_performance": worker.get("best_performance"),
        "chain_height": worker.get("chain_height"),
        "tip_hash": worker.get("tip_hash"),
        "finalized_height": worker.get("finalized_height"),
        "finalized_hash": worker.get("finalized_hash"),
        "component_versions": worker.get("component_versions") or {},
        "peer_count": worker.get("peer_count"),
        "owned_candidates": worker.get("owned_candidate_indices") or [],
        "generation": pool.get("generation"),
        "population": {
            "size": pool.get("pop_size"),
            "evaluated": pool.get("evaluated"),
            "claimed": pool.get("claimed"),
            "free": pool.get("free"),
            "fingerprint": pool.get("population_fingerprint"),
        },
        "candidate": {
            "stage": candidate.get("stage"),
            "total_stages": candidate.get("total_stages"),
            "stage_name": candidate.get("stage_name"),
            "generation": candidate.get("gen"),
            "candidate_num": candidate.get("candidate_num"),
            "total_candidates": candidate.get("total_candidates"),
            "total_evals": candidate.get("total_evals"),
            "no_improve_counter": candidate.get("no_improve_counter"),
        },
        "candidate_eta": {
            key: (worker.get("candidate_eta") or {}).get(key)
            for key in (
                "sample_size",
                "candidates_per_hour",
                "candidate_elapsed_seconds",
                "candidate_eta_seconds",
                "candidate_eta_low_seconds",
                "candidate_eta_high_seconds",
            )
        },
        "campaign_progress": domain.get("campaign_progress") or {},
        "champion_artifact_sha256": worker.get("champion_artifact_sha256"),
        "errors": {
            key: worker.get(key)
            for key in (
                "api_error",
                "monitor_error",
                "optimization_error",
                "shared_population_error",
                "bootstrap_evidence_error",
            )
            if worker.get(key)
        },
    }


def compact_network(network: dict[str, Any]) -> dict[str, Any]:
    participants: dict[str, Any] = {}
    rates: list[float] = []
    campaign_progress: dict[str, Any] = {}
    dataset_hashes: set[str] = set()
    genesis_hashes: set[str] = set()
    for node_id, report in (network.get("participants") or {}).items():
        status = report.get("status") or {}
        coordination = status.get("coordination") or {}
        dataset = coordination.get("dataset_sha256")
        if dataset:
            dataset_hashes.add(str(dataset))
        genesis = (
            (coordination.get("canonical_lineage") or {}).get("genesis_hash")
        )
        if genesis:
            genesis_hashes.add(str(genesis))
        workers = {
            worker_id: compact_worker(worker)
            for worker_id, worker in (status.get("workers") or {}).items()
        }
        for worker in workers.values():
            rate = (worker.get("candidate_eta") or {}).get(
                "candidates_per_hour"
            )
            if isinstance(rate, (int, float)) and rate > 0:
                rates.append(float(rate))
            progress = worker.get("campaign_progress") or {}
            if not campaign_progress and progress:
                campaign_progress = progress
        participants[node_id] = {
            "online": bool(report.get("online")),
            "error": report.get("error"),
            "phase": status.get("phase"),
            "job_id": status.get("job_id"),
            "domain_id": status.get("domain_id"),
            "updated_at": status.get("updated_at"),
            "alerts": status.get("alerts") or [],
            "workers": workers,
        }
    fleet_rate = sum(rates)
    remaining = campaign_progress.get("campaign_candidates_remaining")
    eta_seconds = None
    if fleet_rate > 0 and isinstance(remaining, (int, float)):
        eta_seconds = float(remaining) / fleet_rate * 3600
    return {
        "available": True,
        "plan_id": network.get("plan_id"),
        "plan_hash": network.get("plan_hash"),
        "jobs": network.get("plan_jobs") or [],
        "participants": participants,
        "eta": {
            "fleet_candidates_per_hour": fleet_rate or None,
            "full_budget_remaining_seconds": eta_seconds,
            "basis": (
                "sum_of_local_candidate_rates_without_early_stopping"
                if fleet_rate
                else None
            ),
        },
        "lineage": {
            "active_dataset_sha256": sorted(dataset_hashes),
            "genesis_hash": sorted(genesis_hashes),
        },
    }


def collect_runtime(network_url: str) -> dict[str, Any]:
    try:
        return compact_network(http_json(network_url))
    except Exception as exc:
        return {"available": False, "reason": bounded_error(exc)}


def local_machine_data() -> dict[str, Any]:
    completed = subprocess.run(
        ["python3", "-c", REMOTE_MACHINE_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        timeout=25,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("local machine probe returned a non-object")
    return value


def remote_machine_data(target: str) -> dict[str, Any]:
    command = [
        "ssh",
        "-F",
        str(Path.home() / ".ssh/config"),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=6",
        "-o",
        "StrictHostKeyChecking=yes",
        target,
        "python3",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            input=REMOTE_MACHINE_SCRIPT,
            timeout=35,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "no SSH detail").strip()
        raise RuntimeError(
            f"remote machine probe failed: {detail[:MAX_ERROR_CHARS]}"
        ) from exc
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("remote machine probe returned a non-object")
    return value


def collect_machines(
    targets: dict[str, str],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    previous = previous or {}
    for name, target in targets.items():
        try:
            data = (
                local_machine_data()
                if target == "local"
                else remote_machine_data(target)
            )
            data["reachable"] = True
            prior_events = (
                (previous.get(name) or {}).get("campaign_memory_events") or {}
            )
            events = data.get("campaign_memory_events") or {}
            data["oom_kill_delta"] = max(
                0,
                int(events.get("oom_kill") or 0)
                - int(prior_events.get("oom_kill") or 0),
            )
            result[name] = data
        except Exception as exc:
            result[name] = {
                "reachable": False,
                "reason": bounded_error(exc),
            }
    return result


def latest_session_detail(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            """
            SELECT detail_json
            FROM lab_sessions
            WHERE status='complete'
            ORDER BY ended_at DESC
            LIMIT 1
            """
        ).fetchone()
    except sqlite3.Error:
        return {}
    finally:
        connection.close()
    if row is None:
        return {}
    try:
        value = json.loads(row[0])
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def compact_brokers(
    value: dict[str, Any],
    *,
    state_root: Path,
) -> dict[str, Any]:
    alpaca = value.get("alpaca") or {}
    alpaca_detail = alpaca.get("detail") or {}
    ibkr = value.get("ibkr") or {}
    ibkr_detail = latest_session_detail(
        state_root / "ibkr-paper-lab.sqlite"
    )
    mt5 = value.get("mt5") or {}
    oanda = value.get("oanda") or {}
    return {
        "available": bool(value),
        "generated_at": value.get("generated_at"),
        "active_event_keys": value.get("active_event_keys") or [],
        "alpaca": {
            "available": bool(alpaca.get("available")),
            "complete_sessions": alpaca.get("complete_sessions"),
            "last_complete": alpaca.get("ended_at"),
            "adapter_version": alpaca_detail.get("adapter_version"),
            "environment": alpaca_detail.get("environment"),
            "account_fingerprint": alpaca_detail.get("account_fingerprint"),
            "read_only": True,
            "protected_execution_eligible": bool(
                alpaca_detail.get("protected_execution_eligible")
            ),
            "open_positions": alpaca_detail.get("open_positions"),
            "open_orders": alpaca_detail.get("open_orders"),
            "orders_submitted": alpaca_detail.get("orders_submitted"),
            "quotes_received_count": len(
                alpaca_detail.get("quotes_received") or []
            ),
        },
        "ibkr": {
            "available": bool(ibkr.get("available")),
            "complete_sessions": ibkr.get("complete_sessions"),
            "last_complete": (ibkr.get("latest_complete") or {}).get(
                "ended_at"
            ),
            "adapter_version": ibkr_detail.get("adapter_version"),
            "account_fingerprint": ibkr_detail.get("account_fingerprint"),
            "read_only": bool(ibkr_detail.get("read_only", True)),
            "protected_execution_eligible": bool(
                ibkr_detail.get("protected_execution_eligible")
            ),
            "open_positions": (ibkr.get("latest_complete") or {}).get(
                "open_positions"
            ),
            "open_orders": (ibkr.get("latest_complete") or {}).get(
                "open_orders"
            ),
            "socket_available": bool(
                (ibkr.get("socket") or {}).get("available")
            ),
        },
        "oanda": {
            key: oanda.get(key)
            for key in ("available", "configured", "reason")
        },
        "mt5": {
            key: mt5.get(key)
            for key in (
                "available",
                "reason",
                "last_heartbeat_at",
                "broker_connected",
                "open_positions",
                "open_orders",
            )
            if key in mt5
        },
    }


def collect_brokers(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not value:
        return {"available": False, "reason": "watchdog_snapshot_missing"}
    return compact_brokers(value, state_root=path.parent.parent)


def collect_tests(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not value:
        return {
            "available": False,
            "reason": "test_evidence_not_materialized",
        }
    return {
        "available": True,
        "generated_at": value.get("generated_at"),
        "suites": value.get("suites") or [],
    }


def newest_swarm_state(root: Path) -> dict[str, Any]:
    candidates = sorted(
        root.glob("*/*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return read_json(candidates[0]) if candidates else {}


def active_event_keys(state: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key, event in (state.get("events") or {}).items()
        if isinstance(event, dict) and bool(event.get("active"))
    )


def collect_watchdogs(state_root: Path, brokers: dict[str, Any]) -> dict[str, Any]:
    gpu = read_json(
        state_root / "gpu-temperature-watchdog/state.json"
    )
    memory = read_json(
        state_root / "memory-pressure-watchdog/state.json"
    )
    swarm = newest_swarm_state(
        state_root / "swarm-telegram-watchdog"
    )
    return {
        "gpu": {
            "last_check_iso": gpu.get("last_check_iso"),
            "active_event_keys": active_event_keys(gpu),
        },
        "memory": {
            "last_check_iso": memory.get("last_check_iso"),
            "severity": memory.get("severity"),
            "memory_events": memory.get("memory_events") or {},
        },
        "swarm": {
            "last_check_iso": swarm.get("last_check_iso"),
            "active_event_keys": active_event_keys(swarm),
            "last_notification_at": swarm.get("last_notification_at"),
        },
        "brokers": {
            "generated_at": brokers.get("generated_at"),
            "active_event_keys": brokers.get("active_event_keys") or [],
        },
    }


def collect_hashes(runtime: dict[str, Any]) -> dict[str, Any]:
    artifact_hashes: set[str] = set()
    for participant in (runtime.get("participants") or {}).values():
        for worker in (participant.get("workers") or {}).values():
            artifact = worker.get("champion_artifact_sha256")
            if artifact:
                artifact_hashes.add(str(artifact))
    return {
        "campaign_plan_sha256": runtime.get("plan_hash"),
        "active_dataset_sha256": (
            (runtime.get("lineage") or {}).get("active_dataset_sha256") or []
        ),
        "genesis_hash": (
            (runtime.get("lineage") or {}).get("genesis_hash") or []
        ),
        "champion_artifact_sha256": sorted(artifact_hashes),
    }


def snapshot_digest(packet: dict[str, Any]) -> str:
    value = copy.deepcopy(packet)
    value.setdefault("meta", {})["snapshot_sha256"] = None
    return canonical_hash(value)


def build_snapshot(
    *,
    workspace_root: Path,
    network_url: str,
    machine_targets: dict[str, str],
    paper_watchdog: Path,
    test_evidence: Path,
    state_root: Path,
    previous: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    previous = previous or {}
    now = now or datetime.now(timezone.utc)
    provenance = collect_provenance(workspace_root)
    runtime = collect_runtime(network_url)
    machines = collect_machines(
        machine_targets,
        previous.get("machines") or {},
    )
    brokers = collect_brokers(paper_watchdog)
    tests = collect_tests(test_evidence)
    watchdogs = collect_watchdogs(state_root, brokers)
    hashes = collect_hashes(runtime)
    sections = redact_value({
        "provenance": provenance,
        "runtime": runtime,
        "machines": machines,
        "brokers": brokers,
        "tests": tests,
        "watchdogs": watchdogs,
        "hashes": hashes,
    })
    section_hashes = {
        name: canonical_hash(value)
        for name, value in sections.items()
    }
    prior_hashes = ((previous.get("delta") or {}).get("section_hashes") or {})
    changed = sorted(
        name
        for name, digest in section_hashes.items()
        if prior_hashes.get(name) != digest
    )
    packet = {
        "schema": SCHEMA,
        "meta": {
            "generated_at": now.isoformat().replace("+00:00", "Z"),
            "host": socket.gethostname(),
            "collector_version": COLLECTOR_VERSION,
            "snapshot_sha256": None,
        },
        **sections,
        "delta": {
            "changed_sections": changed,
            "previous_snapshot_sha256": (
                (previous.get("meta") or {}).get("snapshot_sha256")
            ),
            "section_hashes": section_hashes,
        },
    }
    packet["meta"]["snapshot_sha256"] = snapshot_digest(packet)
    return packet


def markdown_summary(packet: dict[str, Any]) -> str:
    runtime = packet.get("runtime") or {}
    eta = runtime.get("eta") or {}
    lines = [
        "# Audit Snapshot",
        "",
        f"- Generated: `{(packet.get('meta') or {}).get('generated_at')}`",
        f"- SHA-256: `{(packet.get('meta') or {}).get('snapshot_sha256')}`",
        f"- Changed: {', '.join((packet.get('delta') or {}).get('changed_sections') or []) or 'none'}",
        f"- Campaign: `{runtime.get('plan_id') or 'unavailable'}`",
        f"- Fleet candidates/hour: `{eta.get('fleet_candidates_per_hour')}`",
        "",
        "## Machines",
        "",
        "| Machine | Reachable | GPUs | Max temp C | RAM available GiB | Swap used GiB | OOM delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, machine in (packet.get("machines") or {}).items():
        memory = machine.get("memory") or {}
        gpus = machine.get("gpus") or []
        temperatures = [
            float(gpu["temperature_c"])
            for gpu in gpus
            if gpu.get("temperature_c") is not None
        ]
        available = memory.get("available_bytes")
        swap_total = memory.get("swap_total_bytes")
        swap_free = memory.get("swap_free_bytes")
        swap_used = (
            int(swap_total) - int(swap_free)
            if swap_total is not None and swap_free is not None
            else None
        )
        lines.append(
            "| "
            + " | ".join([
                name,
                str(bool(machine.get("reachable"))),
                str(len(gpus)),
                f"{max(temperatures):.0f}" if temperatures else "N/A",
                f"{int(available) / 1024 ** 3:.2f}" if available else "N/A",
                f"{swap_used / 1024 ** 3:.2f}" if swap_used is not None else "N/A",
                str(machine.get("oom_kill_delta", "N/A")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Brokers",
        "",
        "| Venue | Available | Read only | Positions | Orders | Last complete |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ])
    for venue in ("alpaca", "ibkr", "oanda", "mt5"):
        broker = (packet.get("brokers") or {}).get(venue) or {}
        lines.append(
            "| "
            + " | ".join([
                venue,
                str(bool(broker.get("available"))),
                str(broker.get("read_only", "N/A")),
                str(broker.get("open_positions", "N/A")),
                str(broker.get("open_orders", "N/A")),
                str(broker.get("last_complete", "N/A")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "This packet is read-only audit input. It does not authorize campaign or "
        "broker actions.",
        "",
    ])
    return "\n".join(lines)


def prune_snapshots(output_dir: Path, retention: int) -> None:
    snapshots = sorted(
        output_dir.glob("audit_snapshot_*.json"),
        key=lambda path: path.name,
        reverse=True,
    )
    for path in snapshots[retention:]:
        path.unlink(missing_ok=True)
        path.with_suffix(".md").unlink(missing_ok=True)


def write_snapshot(
    packet: dict[str, Any],
    output_dir: Path,
    retention: int,
) -> tuple[Path, Path]:
    timestamp = (
        str((packet.get("meta") or {}).get("generated_at"))
        .replace("-", "")
        .replace(":", "")
    )
    json_path = output_dir / f"audit_snapshot_{timestamp}.json"
    md_path = json_path.with_suffix(".md")
    encoded = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    summary = markdown_summary(packet)
    atomic_text(json_path, encoded)
    atomic_text(md_path, summary)
    atomic_text(output_dir / "latest.json", encoded)
    atomic_text(output_dir / "latest.md", summary)
    prune_snapshots(output_dir, retention)
    return json_path, md_path


def parse_machine_targets(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("machine must use NAME=TARGET")
        name, target = (part.strip() for part in value.split("=", 1))
        if not name or not target:
            raise ValueError("machine must use non-empty NAME=TARGET")
        result[name] = target
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.home() / "Documents/GitHub",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--network-url",
        default="http://127.0.0.1:8795/api/network",
    )
    parser.add_argument(
        "--paper-watchdog",
        type=Path,
        default=DEFAULT_PAPER_WATCHDOG,
    )
    parser.add_argument(
        "--test-evidence",
        type=Path,
        default=DEFAULT_TEST_EVIDENCE,
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path.home() / ".local/state/agent-multi",
    )
    parser.add_argument(
        "--machine",
        action="append",
        default=[],
        metavar="NAME=TARGET",
        help="Use TARGET=local or an SSH host alias",
    )
    parser.add_argument("--retention", type=int, default=28)
    args = parser.parse_args()
    if args.retention < 2:
        raise ValueError("retention must be at least 2")
    targets = parse_machine_targets(args.machine) if args.machine else {
        "omega": "local",
        "dragon": "dragon",
        "gamma": "gamma",
    }
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".collector.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Audit snapshot skipped: another collection is running")
            return 0
        previous = read_json(output_dir / "latest.json")
        packet = build_snapshot(
            workspace_root=args.workspace_root.expanduser(),
            network_url=args.network_url,
            machine_targets=targets,
            paper_watchdog=args.paper_watchdog.expanduser(),
            test_evidence=args.test_evidence.expanduser(),
            state_root=args.state_root.expanduser(),
            previous=previous,
        )
        json_path, md_path = write_snapshot(
            packet,
            output_dir,
            args.retention,
        )
    size = json_path.stat().st_size
    print(json.dumps({
        "schema": SCHEMA,
        "snapshot_sha256": packet["meta"]["snapshot_sha256"],
        "changed_sections": packet["delta"]["changed_sections"],
        "json": json_path.name,
        "markdown": md_path.name,
        "size_bytes": size,
        "target_bytes": 32768,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
