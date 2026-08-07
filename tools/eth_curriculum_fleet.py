#!/usr/bin/env python3
"""Operate the four-GPU ETH N14/EN4_10/E4 decision experiment.

This is a bounded, non-DOIN experiment. Each GPU owns one seed and runs
the three arms sequentially. Remote jobs are transient systemd user
services, so Omega can disconnect without stopping Dragon or Gamma.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/harveybc/anaconda3/envs/trading-stack/bin/python")
RUNNER = REPO / "tools/eth_curriculum_decision_experiment.py"
AGGREGATOR = REPO / "tools/aggregate_curriculum_decision.py"
DEFAULT_ROOT = Path(
    "/home/harveybc/.local/share/agent-multi/"
    "eth_curriculum_decision_20260807_v1"
)
REPOS = ("agent-multi", "gym-fx", "doin-node", "doin-core",
         "doin-plugins", "trading-contracts")


@dataclass(frozen=True)
class Worker:
    name: str
    ssh_target: str | None
    seed: int
    gpu_index: int
    replica_authority: str

    @property
    def unit(self) -> str:
        return f"eth-curriculum-seed{self.seed}-v1.service"


WORKERS = (
    Worker("omega", None, 101, 0, "dragon"),
    Worker("dragon", "dragon", 202, 0, "gamma-replica"),
    Worker("gamma-5070ti", "gamma", 303, 0, "dragon-replica"),
    Worker("gamma-5090", "gamma", 404, 1, "dragon-replica"),
)


def _remote(worker: Worker, argv: list[str], *, check: bool = True,
            timeout: int = 120) -> subprocess.CompletedProcess:
    command = argv if worker.ssh_target is None else [
        "ssh", "-o", "BatchMode=yes", worker.ssh_target,
        shlex.join(argv),
    ]
    return subprocess.run(command, capture_output=True, text=True,
                          check=check, timeout=timeout)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _local_revisions() -> dict[str, str]:
    revisions = {}
    for repo in REPOS:
        path = Path("/home/harveybc/Documents/GitHub") / repo
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True)
        revisions[repo] = result.stdout.strip()
    return revisions


def preflight(output_root: Path) -> dict:
    """Prove exact code/data/GPU/replica readiness before launch."""
    expected = _local_revisions()
    problems: list[str] = []
    facts: dict[str, dict] = {}
    for worker in WORKERS:
        worker_facts: dict = {"revisions": {}, "gpu": None}
        for repo, revision in expected.items():
            repo_path = f"/home/harveybc/Documents/GitHub/{repo}"
            got = _remote(worker, [
                "git", "-C", repo_path, "rev-parse", "HEAD"
            ], check=False)
            actual = got.stdout.strip()
            worker_facts["revisions"][repo] = actual or "unavailable"
            if got.returncode != 0 or actual != revision:
                problems.append(
                    f"{worker.name}: {repo}={actual or 'unavailable'}"
                    f" != {revision}")
            dirty = _remote(worker, [
                "git", "-C", repo_path, "status", "--porcelain",
                "--untracked-files=all"
            ], check=False)
            if dirty.returncode != 0 or dirty.stdout.strip():
                problems.append(f"{worker.name}: {repo} worktree dirty")
        gpu = _remote(worker, [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], check=False)
        rows = [line.strip() for line in gpu.stdout.splitlines()
                if line.strip()]
        worker_facts["gpu"] = rows
        if gpu.returncode != 0 or worker.gpu_index >= len(rows):
            problems.append(
                f"{worker.name}: GPU {worker.gpu_index} unavailable")
        running = _remote(worker, [
            "pgrep", "-af", "eth_curriculum_decision_experiment.py"
        ], check=False)
        if running.returncode == 0 and running.stdout.strip():
            problems.append(
                f"{worker.name}: decision runner already active")
        replica = _remote(worker, [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            worker.replica_authority, "hostname"
        ], check=False)
        worker_facts["replica_host"] = replica.stdout.strip()
        if replica.returncode != 0 or not replica.stdout.strip():
            problems.append(
                f"{worker.name}: replica authority"
                f" {worker.replica_authority} unreachable")
        facts[worker.name] = worker_facts

    payload = {
        "schema": "agent_multi.eth_curriculum_fleet_preflight.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "workers": [asdict(worker) for worker in WORKERS],
        "revisions": expected,
        "facts": facts,
        "problems": problems,
        "ready": not problems,
    }
    _atomic_json(output_root / "fleet_preflight.json", payload)
    if problems:
        raise RuntimeError("fleet preflight failed: " + "; ".join(problems))
    return payload


def start(output_root: Path) -> dict:
    preflight_packet = preflight(output_root)
    launched = []
    errors = []
    for worker in WORKERS:
        seed_dir = output_root / f"seed{worker.seed}"
        _remote(worker, ["mkdir", "-p", str(seed_dir)])
        command = [
            "systemd-run", "--user", f"--unit={worker.unit}",
            "--collect", "--property=Type=exec", "--property=Restart=no",
            f"--working-directory={REPO}",
            f"--setenv=CUDA_VISIBLE_DEVICES={worker.gpu_index}",
            "--setenv=PYTHONUNBUFFERED=1",
            ("--setenv=AGENT_MULTI_REPLICA_AUTHORITY="
             f"{worker.replica_authority}"),
            ("--setenv=AGENT_MULTI_REPLICA_ROOT="
             "~/.local/share/agent-multi/replica"),
            str(PYTHON), str(RUNNER),
            "--output-root", str(output_root),
            "--seed", str(worker.seed),
            "--epoch-timesteps", "20000",
            "--arms", "N14,EN4_10,E4",
        ]
        result = _remote(worker, command, check=False)
        if result.returncode == 0:
            launched.append(worker.name)
        else:
            errors.append({
                "worker": worker.name,
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:],
                "stderr": result.stderr[-1000:],
            })
    packet = {
        "schema": "agent_multi.eth_curriculum_fleet_launch.v1",
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "preflight_sha256": _sha(output_root / "fleet_preflight.json"),
        "launched": launched,
        "errors": errors,
    }
    _atomic_json(output_root / "fleet_launch.json", packet)
    if errors:
        raise RuntimeError(f"partial fleet launch: {errors}")
    return packet


def status(output_root: Path) -> dict:
    rows = []
    for worker in WORKERS:
        service = _remote(worker, [
            "systemctl", "--user", "show", worker.unit,
            "--property=ActiveState,SubState,MainPID,ExecMainStatus",
            "--value"
        ], check=False)
        values = service.stdout.splitlines()
        gpu = _remote(worker, [
            "nvidia-smi",
            "--query-gpu=index,uuid,temperature.gpu,utilization.gpu,"
            "memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], check=False)
        gpu_rows = [line.strip() for line in gpu.stdout.splitlines()
                    if line.strip()]
        logs = _remote(worker, [
            "journalctl", "--user", "-u", worker.unit,
            "-n", "60", "--no-pager", "-o", "cat"
        ], check=False)
        current_arm = "starting"
        completed = 0
        for line in logs.stdout.splitlines():
            if "[decision]" in line and "arm=" in line:
                current_arm = line.strip()
            if "[decision]" in line and " done" in line:
                completed += 1
        rows.append({
            "worker": worker.name,
            "seed": worker.seed,
            "gpu_index": worker.gpu_index,
            "service": {
                "active_state": values[0] if len(values) > 0 else "unknown",
                "sub_state": values[1] if len(values) > 1 else "unknown",
                "main_pid": values[2] if len(values) > 2 else "unknown",
                "exec_main_status": values[3] if len(values) > 3 else "unknown",
            },
            "gpu": (gpu_rows[worker.gpu_index]
                    if worker.gpu_index < len(gpu_rows) else "unavailable"),
            "completed_arms": completed,
            "current_arm": current_arm,
            "latest_log": logs.stdout.splitlines()[-1:][0]
            if logs.stdout.splitlines() else "unavailable",
        })
    payload = {
        "schema": "agent_multi.eth_curriculum_fleet_status.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "workers": rows,
    }
    _atomic_json(output_root / "fleet_status.json", payload)
    return payload


def collect(output_root: Path) -> dict:
    """Collect complete seed packets, re-probe replicas and aggregate."""
    for worker in WORKERS:
        packet = output_root / f"seed{worker.seed}" / "seed_packet.json"
        if worker.ssh_target is not None:
            subprocess.run([
                "rsync", "-a",
                f"{worker.ssh_target}:{output_root}/seed{worker.seed}/",
                f"{output_root}/seed{worker.seed}/",
            ], check=True)
        if not packet.is_file():
            raise RuntimeError(f"missing complete packet for {worker.name}")

    replica_checks = []
    for worker in WORKERS:
        packet_path = output_root / f"seed{worker.seed}" / "seed_packet.json"
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        for arm_name, arm in packet.get("arms", {}).items():
            for label, ref in (arm.get("artifacts") or {}).items():
                observation = ref.get("replica_observation") or {}
                remote_path = observation.get("remote_path")
                result = _remote(worker, [
                    "ssh", "-o", "BatchMode=yes",
                    worker.replica_authority,
                    "sha256sum", str(remote_path)
                ], check=False)
                observed = result.stdout.split()[0] if result.stdout else None
                ok = result.returncode == 0 and observed == ref.get("sha256")
                replica_checks.append({
                    "worker": worker.name, "seed": worker.seed,
                    "arm": arm_name, "artifact": label,
                    "authority": worker.replica_authority,
                    "remote_path": remote_path,
                    "observed_sha256": observed, "ok": ok,
                })
                if not ok:
                    raise RuntimeError(
                        f"replica verification failed: {replica_checks[-1]}")

    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and not path.name.endswith(".tmp"):
            files.append({
                "path": str(path.relative_to(output_root)),
                "sha256": _sha(path), "bytes": path.stat().st_size,
            })
    manifest = {
        "schema": "agent_multi.eth_curriculum_fleet_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workers": [asdict(worker) for worker in WORKERS],
        "replica_checks": replica_checks,
        "files": files,
    }
    _atomic_json(output_root / "fleet_manifest.json", manifest)
    aggregate = subprocess.run([
        str(PYTHON), str(AGGREGATOR),
        "--output-root", str(output_root),
    ], capture_output=True, text=True, check=False)
    if aggregate.returncode != 0:
        raise RuntimeError(
            f"aggregation failed: {aggregate.stdout}\n{aggregate.stderr}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=(
        "preflight", "start", "status", "collect"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    operation = {
        "preflight": preflight,
        "start": start,
        "status": status,
        "collect": collect,
    }[args.command]
    try:
        result = operation(args.output_root.expanduser().resolve())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"ok": True, "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
