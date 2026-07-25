#!/usr/bin/env python3
"""Continuous remote worker for the Project 3 evidence pool."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psutil

from project3_evidence_screen import execute


def _request(
    api_url: str,
    path: str,
    *,
    token: str | None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(api_url.rstrip("/") + path, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"{path} is unreachable: {exc}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"{path} failed: {result}")
    return result


def _gpu_summary() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception:
        return {"available": False, "gpus": []}
    gpus = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            continue
        gpus.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "temperature_c": float(fields[2]),
                "utilization_pct": float(fields[3]),
                "memory_used_mb": float(fields[4]),
                "memory_total_mb": float(fields[5]),
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def _cpu_summary() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "load_average": list(os.getloadavg()),
        "memory_used_pct": memory.percent,
        "memory_available_bytes": memory.available,
    }


def _state_payload(machine_id: str, job_id: str | None, status: str, message: str) -> dict[str, Any]:
    return {
        "machine_id": machine_id,
        "job_id": job_id,
        "status": status,
        "message": message,
        "cpu_summary": _cpu_summary(),
        "gpu_summary": _gpu_summary(),
        "lease_seconds": 300,
    }


def _run_job_with_heartbeats(
    api_url: str,
    token: str | None,
    machine_id: str,
    job: dict[str, Any],
    heartbeat_seconds: int,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    task_type = str(job["task_type"])
    config = dict(job["config"])
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(execute, task_type, config)
        while True:
            _request(
                api_url,
                "/heartbeat",
                token=token,
                payload=_state_payload(
                    machine_id,
                    job_id,
                    "running",
                    f"{task_type} attempt={job['attempt']}",
                ),
            )
            try:
                return future.result(timeout=max(5, heartbeat_seconds))
            except FutureTimeoutError:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--token")
    parser.add_argument("--token-file")
    parser.add_argument("--machine-id", default=socket.gethostname())
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--max-jobs", type=int, default=0)
    args = parser.parse_args()
    token = args.token
    if args.token_file:
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    _request(args.api_url, "/health", token=token)

    completed = 0
    while args.max_jobs <= 0 or completed < args.max_jobs:
        claim = _request(
            args.api_url,
            "/claim",
            token=token,
            payload={"machine_id": args.machine_id, "lease_seconds": 300},
        )
        job = claim.get("job")
        if not job:
            _request(
                args.api_url,
                "/heartbeat",
                token=token,
                payload=_state_payload(args.machine_id, None, "idle", "waiting for eligible jobs"),
            )
            time.sleep(max(1, args.poll_seconds))
            continue
        job_id = str(job["job_id"])
        try:
            result = _run_job_with_heartbeats(
                args.api_url,
                token,
                args.machine_id,
                job,
                args.heartbeat_seconds,
            )
            _request(
                args.api_url,
                "/complete",
                token=token,
                payload={
                    **_state_payload(args.machine_id, job_id, "completed", f"completed {job_id}"),
                    "result": result,
                },
            )
            completed += 1
        except Exception as exc:
            retry = not isinstance(exc, (FileNotFoundError, ValueError))
            _request(
                args.api_url,
                "/fail",
                token=token,
                payload={
                    **_state_payload(args.machine_id, job_id, "failed", str(exc)),
                    "error": f"{type(exc).__name__}: {exc}",
                    "retry": retry,
                },
            )
            time.sleep(2)


if __name__ == "__main__":
    main()
