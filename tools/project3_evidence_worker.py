#!/usr/bin/env python3
"""Continuous remote worker for the Project 3 evidence pool."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
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
        "lease_seconds": 900,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _prune_cache(root: Path, required: set[Path], required_bytes: int, max_bytes: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    current = _cache_size(root)
    if required_bytes > max_bytes:
        raise RuntimeError(
            f"job requires {required_bytes} bytes but cache limit is {max_bytes}"
        )
    target = max_bytes - required_bytes
    candidates = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path not in required and not path.name.endswith(".part")
        ),
        key=lambda path: path.stat().st_mtime,
    )
    for path in candidates:
        if current <= target:
            break
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        current -= size


def _download_file(
    api_url: str,
    token: str | None,
    manifest: dict[str, Any],
    destination: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    expected_size = int(manifest["size_bytes"])
    expected_sha = str(manifest["sha256"])
    if destination.exists() and destination.stat().st_size == expected_size:
        if _file_sha256(destination) == expected_sha:
            destination.touch()
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if str(manifest["path"]).lower().endswith(".csv") else ""
    query = urlencode({"path": str(manifest["path"]), "compression": compression})
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(api_url.rstrip("/") + f"/file?{query}", headers=headers, method="GET")
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.part")
    digest = hashlib.sha256()
    written = 0
    last_progress = time.monotonic()
    try:
        with urlopen(request, timeout=600) as response, temporary.open("wb") as handle:
            source = (
                gzip.GzipFile(fileobj=response, mode="rb")
                if response.headers.get("Content-Encoding") == "gzip"
                else response
            )
            while True:
                chunk = source.read(4 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if progress_callback and time.monotonic() - last_progress >= 20:
                    progress_callback(written, expected_size)
                    last_progress = time.monotonic()
        if written != expected_size:
            raise RuntimeError(
                f"download size mismatch for {manifest['path']}: {written} != {expected_size}"
            )
        if digest.hexdigest() != expected_sha:
            raise RuntimeError(f"download hash mismatch for {manifest['path']}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_job_files(
    *,
    api_url: str,
    token: str | None,
    machine_id: str,
    job: dict[str, Any],
    cache_root: Path,
    max_cache_bytes: int,
) -> dict[str, Any]:
    root = cache_root / machine_id
    manifests = list(job.get("required_files") or [])
    if manifests and all(
        Path(str(manifest["path"])).is_file()
        and Path(str(manifest["path"])).stat().st_size == int(manifest["size_bytes"])
        for manifest in manifests
    ):
        return dict(job["config"])
    destinations = {
        root / str(manifest["relative_path"]): manifest
        for manifest in manifests
    }
    missing_bytes = sum(
        int(manifest["size_bytes"])
        for destination, manifest in destinations.items()
        if not destination.exists() or destination.stat().st_size != int(manifest["size_bytes"])
    )
    _prune_cache(root, set(destinations), missing_bytes, max_cache_bytes)
    job_id = str(job["job_id"])
    for index, (destination, manifest) in enumerate(destinations.items(), start=1):
        _request(
            api_url,
            "/heartbeat",
            token=token,
            payload=_state_payload(
                machine_id,
                job_id,
                "downloading",
                f"file {index}/{len(destinations)} {manifest['relative_path']}",
            ),
        )
        _download_file(
            api_url,
            token,
            manifest,
            destination,
            progress_callback=lambda written, total, relative=manifest["relative_path"]: _request(
                api_url,
                "/heartbeat",
                token=token,
                payload=_state_payload(
                    machine_id,
                    job_id,
                    "downloading",
                    f"{relative} {written}/{total} bytes",
                ),
            ),
        )
    config = dict(job["config"])
    original_root = Path(str(config["data_root"])).resolve()
    input_path = Path(str(config["input_data_file"])).resolve()
    relative_input = input_path.relative_to(original_root)
    config["data_root"] = str(root)
    config["input_data_file"] = str(root / relative_input)
    return config


def _run_job_with_heartbeats(
    api_url: str,
    token: str | None,
    machine_id: str,
    job: dict[str, Any],
    heartbeat_seconds: int,
    cache_root: Path,
    max_cache_bytes: int,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    task_type = str(job["task_type"])
    config = _materialize_job_files(
        api_url=api_url,
        token=token,
        machine_id=machine_id,
        job=job,
        cache_root=cache_root,
        max_cache_bytes=max_cache_bytes,
    )
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
    parser.add_argument("--cache-root", default=str(Path.home() / ".cache/project3-evidence-data"))
    parser.add_argument("--max-cache-gb", type=float, default=12.0)
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
            payload={"machine_id": args.machine_id, "lease_seconds": 900},
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
            Path(args.cache_root),
            int(args.max_cache_gb * 1024**3),
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
