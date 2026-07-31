#!/usr/bin/env python3
"""Run bounded regression suites and materialize compact audit evidence."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "agent_multi.audit_test_evidence.v1"
DEFAULT_OUTPUT_DIR = Path.home() / ".local/state/agent-multi/audit-test-evidence"
DEFAULT_STATUS_URL = "http://127.0.0.1:8795/api/status"
PASS_PATTERN = re.compile(r"(\d+)\s+passed")
FAIL_PATTERN = re.compile(r"(\d+)\s+failed")
SKIP_PATTERN = re.compile(r"(\d+)\s+skipped")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def http_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        value = json.loads(response.read().decode("utf-8"))
    return value if isinstance(value, dict) else {}


def gpu_rows() -> list[dict[str, float]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = []
    for raw in completed.stdout.splitlines():
        fields = [item.strip() for item in raw.split(",")]
        if len(fields) != 3:
            continue
        rows.append(
            {
                "index": float(fields[0]),
                "utilization_pct": float(fields[1]),
                "temperature_c": float(fields[2]),
            }
        )
    return rows


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def resource_guard(
    status: Mapping[str, Any],
    gpus: Sequence[Mapping[str, float]],
    *,
    memory_available_bytes: int,
    load_one: float,
    cpu_count: int,
) -> list[str]:
    reasons: list[str] = []
    for worker in (status.get("workers") or {}).values():
        if isinstance(worker, Mapping) and worker.get("owns_candidate"):
            reasons.append("local_doin_candidate_active")
            break
    if any(float(gpu.get("utilization_pct", 0)) >= 70 for gpu in gpus):
        reasons.append("gpu_utilization_guard")
    if memory_available_bytes < 4 * 1024**3:
        reasons.append("available_memory_guard")
    if cpu_count > 0 and load_one >= cpu_count * 0.75:
        reasons.append("cpu_load_guard")
    return reasons


def parse_pytest_summary(output: str) -> dict[str, int]:
    def count(pattern: re.Pattern[str]) -> int:
        values = pattern.findall(output)
        return int(values[-1]) if values else 0

    return {
        "passed": count(PASS_PATTERN),
        "failed": count(FAIL_PATTERN),
        "skipped": count(SKIP_PATTERN),
    }


def git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def suite_specs(workspace_root: Path, python: Path) -> list[dict[str, Any]]:
    return [
        {
            "name": "agent-multi-safety-campaign",
            "cwd": workspace_root / "agent-multi",
            "command": [
                str(python),
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_campaign_supervisor.py",
                "tests/unit/test_swarm_telegram_watchdog.py",
                "tests/unit/test_validation_pipeline_test_firewall.py",
                "tests/unit/test_policy_observation_and_action_contract.py",
                "tests/unit/test_social_intelligence.py",
            ],
        },
        {
            "name": "gym-fx-full",
            "cwd": workspace_root / "gym-fx",
            "command": [str(python), "-m", "pytest", "-q"],
        },
        {
            "name": "doin-node-consensus-focused",
            "cwd": workspace_root / "doin-node",
            "command": [
                str(python),
                "-m",
                "pytest",
                "-q",
                "tests/test_chain.py",
                "tests/test_flooding.py",
                "tests/test_shared_population_contract.py",
                "tests/test_sync.py",
            ],
        },
    ]


def run_suite(spec: Mapping[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    command = [str(item) for item in spec["command"]]
    cwd = Path(spec["cwd"])
    started = time.monotonic()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
        )
        output = f"{completed.stdout}\n{completed.stderr}"
        exit_code = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = f"{exc.stdout or ''}\n{exc.stderr or ''}"
        exit_code = 124
        timed_out = True
    duration = round(time.monotonic() - started, 3)
    summary = parse_pytest_summary(output)
    return {
        "name": spec["name"],
        "repository": cwd.name,
        "commit": git_head(cwd),
        "command_sha256": sha256_json(command),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": duration,
        **summary,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "output_tail": output[-800:].replace(str(Path.home()), "%h"),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.home() / "Documents/GitHub",
    )
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    value.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    value.add_argument(
        "--python",
        type=Path,
        default=Path.home() / "anaconda3/envs/trading-stack/bin/python",
    )
    value.add_argument("--suite-timeout-seconds", type=int, default=900)
    return value


def main() -> int:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / ".runner.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        try:
            status = http_json(args.status_url)
        except Exception:
            status = {}
        campaign_running = bool(
            subprocess.run(
                ["pgrep", "-f", "app.campaign_supervisor"],
                check=False,
                capture_output=True,
            ).stdout.strip()
        )
        reasons = []
        if campaign_running and not status:
            reasons.append("campaign_status_unavailable")
        try:
            gpus = gpu_rows()
        except Exception:
            gpus = []
            if campaign_running:
                reasons.append("gpu_status_unavailable")
        reasons.extend(
            resource_guard(
                status,
                gpus,
                memory_available_bytes=available_memory_bytes(),
                load_one=os.getloadavg()[0],
                cpu_count=os.cpu_count() or 1,
            )
        )
        attempt = {
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "guard": {
                "allowed": not reasons,
                "reasons": sorted(set(reasons)),
                "gpus": gpus,
            },
        }
        if reasons:
            atomic_json(args.output_dir / "last_attempt.json", attempt)
            return 0

        suites = [
            run_suite(spec, timeout_seconds=args.suite_timeout_seconds)
            for spec in suite_specs(args.workspace_root, args.python)
        ]
        packet = {
            **attempt,
            "suites": suites,
            "all_passed": all(item["exit_code"] == 0 for item in suites),
        }
        packet["packet_sha256"] = sha256_json(packet)
        atomic_json(args.output_dir / "latest.json", packet)
        atomic_json(args.output_dir / "last_attempt.json", packet)
        return 0 if packet["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
