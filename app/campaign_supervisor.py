"""Replicated campaign lifecycle supervisor for collaborative DOIN swarms.

Every physical host runs this same state machine against the same immutable
campaign plan.  DOIN continues to own candidate claiming, shared population
reproduction, blockchain consensus, and champion migration.  This supervisor
only handles the boundary between complete optimization campaigns.
"""
from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import html
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


STATE_SCHEMA = "agent_multi.doin_campaign_state.v1"
PLAN_SCHEMA = "agent_multi.doin_campaign_plan.v1"
PROFILE_SCHEMA = "agent_multi.doin_campaign_profile.v1"
HISTORY_SCHEMA = "agent_multi.doin_campaign_history.v1"
TERMINAL_PHASES = {"stopped", "complete", "blocked"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_runtime_path(value: str, roots: dict[str, Any]) -> Path:
    expanded = str(value)
    for key, root in roots.items():
        expanded = expanded.replace(f"${{{key}}}", str(root))
    if "${" in expanded:
        raise ValueError(f"unresolved runtime path variable in {value!r}")
    return Path(expanded).expanduser().resolve()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _http_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"non-object JSON response from {url}")
    return value


def _pid_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def _pid_matches(pid: int | None, start_ticks: int | None) -> bool:
    if not pid or not start_ticks:
        return False
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        state = fields[2]
        observed_ticks = int(fields[21])
    except (OSError, ValueError, IndexError):
        return False
    # A terminated child remains in /proc as a zombie until its parent reaps
    # it.  It no longer owns a worker or port and must not block the stop
    # barrier or be adopted as a live DOIN process.
    return state != "Z" and observed_ticks == start_ticks


def _reap_child(pid: int | None) -> None:
    if not pid:
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, ProcessLookupError):
        pass


def _process_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _cmdline_references_config(
    cmdline: list[str], *, process_cwd: Path, config_path: Path
) -> bool:
    target = config_path.resolve()
    for value in cmdline:
        candidate = Path(value)
        if candidate.name != target.name:
            continue
        try:
            if not candidate.is_absolute():
                candidate = process_cwd / candidate
            if candidate.resolve() == target:
                return True
        except OSError:
            continue
    return False


def _find_doin_process(config_path: Path) -> tuple[int, int] | None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmdline = _process_cmdline(pid)
        if "doin_node.cli" not in " ".join(cmdline):
            continue
        try:
            process_cwd = Path(f"/proc/{pid}/cwd").resolve()
        except OSError:
            continue
        if _cmdline_references_config(
            cmdline, process_cwd=process_cwd, config_path=config_path
        ):
            ticks = _pid_start_ticks(pid)
            if ticks is not None:
                return pid, ticks
    return None


def _domain_semantic_payload(config: dict[str, Any]) -> dict[str, Any]:
    domains = config.get("domains")
    if not isinstance(domains, list) or len(domains) != 1:
        raise ValueError("campaign node config must contain exactly one domain")
    domain = copy.deepcopy(domains[0])
    optimization = domain.get("optimization_config")
    if isinstance(optimization, dict):
        optimization.pop("runtime_overlay", None)
        if not optimization.get("shared_population"):
            optimization.pop("node_seed_offset", None)
    domain.pop("resource_limits", None)
    return domain


def _domain_semantic_hash(config: dict[str, Any]) -> str:
    return _sha256_json(_domain_semantic_payload(config))


def _domain_id(config: dict[str, Any]) -> str:
    return str(_domain_semantic_payload(config).get("domain_id") or "")


def _config_port(config: dict[str, Any]) -> int:
    port = int(config.get("port", 0))
    if not 1 <= port <= 65535:
        raise ValueError("node config port must be between 1 and 65535")
    return port


def _shared_population_seed(config: dict[str, Any]) -> int:
    domains = config.get("domains") or []
    if len(domains) != 1:
        raise ValueError("campaign node config must contain exactly one domain")
    optimization = domains[0].get("optimization_config") or {}
    if not optimization.get("shared_population"):
        raise ValueError("campaign worker must use shared_population")
    if config.get("require_deterministic_seed") is not True:
        raise ValueError("campaign worker must require deterministic seeds")
    offset = optimization.get("node_seed_offset")
    if offset not in (None, 0, "0"):
        raise ValueError("shared-population workers cannot use node_seed_offset")
    configured = optimization.get(
        "shared_population_seed", optimization.get("ga_seed")
    )
    if configured is None:
        raise ValueError("shared-population worker has no explicit seed")
    seed = int(configured)
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("shared-population seed must fit uint32")
    return seed


def _population_fingerprint(population_state: dict[str, Any]) -> str:
    return _sha256_json(population_state)


class HistoryStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS campaigns (
                    job_id TEXT PRIMARY KEY,
                    ordinal INTEGER NOT NULL,
                    domain_id TEXT NOT NULL,
                    plan_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    chain_height INTEGER,
                    tip_hash TEXT,
                    champion_fitness REAL,
                    champion_peer_id TEXT,
                    artifact_sha256 TEXT,
                    artifact_format TEXT,
                    artifact_path TEXT,
                    manifest_path TEXT,
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    evidence_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS worker_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    job_id TEXT,
                    node_id TEXT NOT NULL,
                    worker_id TEXT,
                    event TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS worker_events_job_time
                    ON worker_events(job_id, timestamp);
                """
            )

    def event(
        self,
        *,
        node_id: str,
        event: str,
        job_id: str | None = None,
        worker_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO worker_events
                   (timestamp, job_id, node_id, worker_id, event, detail_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (_utc_now(), job_id, node_id, worker_id, event, _canonical_json(detail or {})),
            )

    def mark_started(self, job: dict[str, Any], plan_hash: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO campaigns
                   (job_id, ordinal, domain_id, plan_hash, status, started_at)
                   VALUES (?, ?, ?, ?, 'running', ?)
                   ON CONFLICT(job_id) DO UPDATE SET
                     ordinal=excluded.ordinal,
                     domain_id=excluded.domain_id,
                     plan_hash=excluded.plan_hash,
                     status='running',
                     started_at=COALESCE(campaigns.started_at, excluded.started_at)""",
                (
                    job["job_id"], int(job["ordinal"]), job["domain_id"], plan_hash,
                    _utc_now(),
                ),
            )

    def mark_completed(self, job: dict[str, Any], plan_hash: str, archive: dict[str, Any]) -> None:
        champion = archive.get("champion") or {}
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO campaigns
                   (job_id, ordinal, domain_id, plan_hash, status, started_at, completed_at,
                    chain_height, tip_hash, champion_fitness, champion_peer_id,
                    artifact_sha256, artifact_format, artifact_path, manifest_path,
                    parameters_json, metrics_json, evidence_json)
                   VALUES (?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job_id) DO UPDATE SET
                    ordinal=excluded.ordinal,
                    domain_id=excluded.domain_id,
                    plan_hash=excluded.plan_hash,
                    status='completed', completed_at=excluded.completed_at,
                    chain_height=excluded.chain_height, tip_hash=excluded.tip_hash,
                    champion_fitness=excluded.champion_fitness,
                    champion_peer_id=excluded.champion_peer_id,
                    artifact_sha256=excluded.artifact_sha256,
                    artifact_format=excluded.artifact_format,
                    artifact_path=excluded.artifact_path,
                    manifest_path=excluded.manifest_path,
                    parameters_json=excluded.parameters_json,
                    metrics_json=excluded.metrics_json,
                    evidence_json=excluded.evidence_json""",
                (
                    job["job_id"], int(job["ordinal"]), job["domain_id"], plan_hash,
                    archive.get("started_at"), _utc_now(), archive.get("chain_height"),
                    archive.get("tip_hash"), champion.get("fitness"),
                    champion.get("peer_id"), champion.get("artifact_sha256"),
                    champion.get("artifact_format"), champion.get("artifact_path"),
                    champion.get("manifest_path"),
                    _canonical_json(champion.get("parameters") or {}),
                    _canonical_json(champion.get("metrics") or {}),
                    _canonical_json(archive),
                ),
            )

    def completed_job_ids(self) -> set[str]:
        with self._connect() as connection:
            return {
                str(row[0])
                for row in connection.execute(
                    "SELECT job_id FROM campaigns WHERE status='completed'"
                )
            }

    def campaigns(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM campaigns ORDER BY ordinal, completed_at"
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("parameters_json", "metrics_json", "evidence_json"):
                item[key[:-5]] = json.loads(item.pop(key) or "{}")
            result.append(item)
        return result


class CampaignSupervisor:
    def __init__(self, profile_path: Path):
        self.profile_path = profile_path.resolve()
        self.profile = _load_json(self.profile_path)
        if self.profile.get("schema_version") != PROFILE_SCHEMA:
            raise ValueError(f"unsupported profile schema: {self.profile.get('schema_version')}")
        self.node_id = str(self.profile["node_id"])
        self.plan_path = self._resolve_path(self.profile["plan_file"])
        self.plan = _load_json(self.plan_path)
        if self.plan.get("schema_version") != PLAN_SCHEMA:
            raise ValueError(f"unsupported plan schema: {self.plan.get('schema_version')}")
        self._validate_plan()
        self.plan_hash = _sha256_json(self.plan)
        expected_plan_hash = self.profile.get("expected_plan_hash")
        if expected_plan_hash and expected_plan_hash != self.plan_hash:
            raise ValueError(
                f"plan hash mismatch: expected {expected_plan_hash}, got {self.plan_hash}"
            )
        self.state_dir = self._resolve_path(self.profile["state_dir"])
        self.state_path = self.state_dir / "state.json"
        self.log_dir = self.state_dir / "logs"
        self.artifact_dir = self.state_dir / "champions"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.history = HistoryStore(self.state_dir / "campaign_history.sqlite")
        self.poll_seconds = float(self.profile.get("poll_seconds", 5.0))
        self.peer_timeout = float(self.profile.get("peer_timeout_seconds", 2.0))
        self.stability_seconds = float(self.profile.get("convergence_stability_seconds", 20.0))
        self.stop_timeout = float(self.profile.get("stop_timeout_seconds", 30.0))
        self.restart_limit = int(self.profile.get("worker_restart_limit", 5))
        self.join_grace_seconds = float(
            self.profile.get("worker_join_grace_seconds", 120.0)
        )
        self.claimless_grace_seconds = float(
            self.profile.get("claimless_worker_grace_seconds", 300.0)
        )
        self._mutex = threading.RLock()
        self._shutdown = threading.Event()
        self._httpd: ThreadingHTTPServer | None = None
        self._lock_handle = None
        self._dataset_validation_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
        self.state = self._load_or_initialize_state()

    def _resolve_path(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.profile_path.parent / path).resolve()

    def _validate_plan(self) -> None:
        participants = self.plan.get("participants")
        jobs = self.plan.get("jobs")
        if not isinstance(participants, list) or not participants:
            raise ValueError("campaign plan needs participants")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("campaign plan needs jobs")
        node_ids = [str(item.get("node_id")) for item in participants]
        if len(node_ids) != len(set(node_ids)) or self.node_id not in node_ids:
            raise ValueError("participant node_ids must be unique and include this node")
        worker_ids: list[str] = []
        for participant in participants:
            workers = participant.get("workers")
            if not isinstance(workers, list) or not workers:
                raise ValueError(f"participant {participant.get('node_id')} has no workers")
            worker_ids.extend(str(worker) for worker in workers)
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("worker ids must be globally unique")
        job_ids: set[str] = set()
        for index, job in enumerate(jobs):
            if int(job.get("ordinal", -1)) != index:
                raise ValueError("job ordinal must match its immutable plan position")
            job_id = str(job.get("job_id") or "")
            if not job_id or job_id in job_ids:
                raise ValueError("job ids must be non-empty and unique")
            job_ids.add(job_id)
            configs = job.get("worker_configs")
            if not isinstance(configs, dict) or set(configs) != set(worker_ids):
                raise ValueError(f"job {job_id} must map every worker exactly once")

    def _participant(self, node_id: str | None = None) -> dict[str, Any]:
        target = node_id or self.node_id
        return next(item for item in self.plan["participants"] if item["node_id"] == target)

    def _local_worker_ids(self) -> list[str]:
        return [str(value) for value in self._participant()["workers"]]

    def _global_worker_ids(self) -> list[str]:
        return [
            str(worker_id)
            for participant in self.plan["participants"]
            for worker_id in participant["workers"]
        ]

    def _bootstrap_worker_id(self) -> str:
        return self._global_worker_ids()[0]

    def _bootstrap_node_id(self) -> str:
        return str(self.plan["participants"][0]["node_id"])

    def _worker_node_id(self, worker_id: str) -> str:
        for participant in self.plan["participants"]:
            if worker_id in participant["workers"]:
                return str(participant["node_id"])
        raise ValueError(f"unknown campaign worker {worker_id}")

    def _worker_profile(self, worker_id: str) -> dict[str, Any]:
        profiles = self.profile.get("workers") or {}
        value = profiles.get(worker_id)
        if not isinstance(value, dict):
            raise ValueError(f"profile is missing local worker {worker_id}")
        return value

    def _job(self, index: int | None = None) -> dict[str, Any] | None:
        target = int(self.state.get("job_index", 0) if index is None else index)
        jobs = self.plan["jobs"]
        return jobs[target] if 0 <= target < len(jobs) else None

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = _load_json(self.state_path)
            if state.get("schema_version") != STATE_SCHEMA:
                raise ValueError("unsupported campaign supervisor state schema")
            if state.get("plan_hash") != self.plan_hash:
                raise ValueError("persisted state belongs to a different campaign plan")
            return state
        completed = self.history.completed_job_ids()
        first_unfinished = next(
            (index for index, job in enumerate(self.plan["jobs"]) if job["job_id"] not in completed),
            len(self.plan["jobs"]),
        )
        state = {
            "schema_version": STATE_SCHEMA,
            "plan_hash": self.plan_hash,
            "node_id": self.node_id,
            "job_index": first_unfinished,
            "job_id": (
                self.plan["jobs"][first_unfinished]["job_id"]
                if first_unfinished < len(self.plan["jobs"]) else None
            ),
            "phase": "starting" if first_unfinished < len(self.plan["jobs"]) else "complete",
            "workers": {},
            "archive": {},
            "alerts": [],
            "completion_candidate_since": None,
            "coordination": {},
            "updated_at": _utc_now(),
        }
        _atomic_json(self.state_path, state)
        return state

    def _save_state(self) -> None:
        self.state["updated_at"] = _utc_now()
        _atomic_json(self.state_path, self.state)

    def _alert(self, code: str, message: str, *, severity: str = "error") -> None:
        alerts = self.state.setdefault("alerts", [])
        current = next((item for item in alerts if item.get("code") == code), None)
        if current:
            current.update({"message": message, "severity": severity, "last_seen": _utc_now()})
        else:
            alerts.append({
                "code": code,
                "message": message,
                "severity": severity,
                "first_seen": _utc_now(),
                "last_seen": _utc_now(),
            })
        self.state["alerts"] = alerts[-50:]

    def _clear_alert(self, code: str) -> None:
        self.state["alerts"] = [
            item for item in self.state.get("alerts", []) if item.get("code") != code
        ]

    def _worker_config_path(self, job: dict[str, Any], worker_id: str) -> Path:
        worker_cfg = job["worker_configs"][worker_id]
        doin_root = self._resolve_path(self._worker_profile(worker_id)["doin_node_root"])
        path = Path(worker_cfg).expanduser()
        return path.resolve() if path.is_absolute() else (doin_root / path).resolve()

    def _validate_dataset_evidence(self, node_config: dict[str, Any]) -> dict[str, Any] | None:
        domains = node_config.get("domains") or []
        if len(domains) != 1:
            return None
        optimization = domains[0].get("optimization_config") or {}
        agent_root_value = optimization.get("agent_multi_root")
        load_config_value = optimization.get("load_config")
        runtime_overlay_value = optimization.get("runtime_overlay")
        if not all((agent_root_value, load_config_value, runtime_overlay_value)):
            return None

        agent_root = Path(str(agent_root_value)).expanduser().resolve()
        canonical_path = Path(str(load_config_value)).expanduser()
        if not canonical_path.is_absolute():
            canonical_path = agent_root / canonical_path
        overlay_path = Path(str(runtime_overlay_value)).expanduser()
        if not overlay_path.is_absolute():
            overlay_path = agent_root / overlay_path
        canonical = _load_json(canonical_path.resolve())
        overlay = _load_json(overlay_path.resolve())
        roots = overlay.get("roots") or {}
        data = canonical.get("data") or {}
        dataset_path = _expand_runtime_path(str(data["input_data_file"]), roots)
        manifest_path = _expand_runtime_path(str(data["dataset_manifest_file"]), roots)
        manifest = _load_json(manifest_path)
        if str(manifest.get("asset")) != str(data.get("asset")):
            raise ValueError(f"dataset manifest asset mismatch for {dataset_path}")
        if str(manifest.get("timeframe")) != str(data.get("timeframe")):
            raise ValueError(f"dataset manifest timeframe mismatch for {dataset_path}")
        expected_hash = str(manifest.get("sha256") or "")
        if not expected_hash:
            raise ValueError(f"dataset manifest has no sha256: {manifest_path}")
        stat = dataset_path.stat()
        cache_key = (
            str(dataset_path),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            expected_hash,
        )
        cached = self._dataset_validation_cache.get(cache_key)
        if cached is not None:
            return cached
        actual_hash = _sha256_file(dataset_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"dataset sha256 {actual_hash} != manifest {expected_hash}: {dataset_path}"
            )
        evidence = {
            "path": str(dataset_path),
            "manifest_path": str(manifest_path),
            "sha256": actual_hash,
            "bytes": int(stat.st_size),
        }
        self._dataset_validation_cache[cache_key] = evidence
        return evidence

    def _validate_local_configs(self, job: dict[str, Any]) -> dict[str, dict[str, Any]]:
        loaded: dict[str, dict[str, Any]] = {}
        semantic_hashes: set[str] = set()
        seeds: set[int] = set()
        population_sizes: set[int] = set()
        for worker_id in self._local_worker_ids():
            path = self._worker_config_path(job, worker_id)
            config = _load_json(path)
            if _domain_id(config) != job["domain_id"]:
                raise ValueError(
                    f"{worker_id} domain {_domain_id(config)!r} != planned {job['domain_id']!r}"
                )
            semantic_hash = _domain_semantic_hash(config)
            semantic_hashes.add(semantic_hash)
            expected = job.get("domain_semantic_hash")
            if expected and semantic_hash != expected:
                raise ValueError(
                    f"{worker_id} semantic hash {semantic_hash} != planned {expected}"
                )
            seed = _shared_population_seed(config)
            domain = (config.get("domains") or [])[0]
            optimization = domain.get("optimization_config") or {}
            population_size = int(
                optimization.get(
                    "shared_population_size",
                    optimization.get("population_size", 0),
                )
            )
            if population_size < 1:
                raise ValueError(f"{worker_id} has no shared population size")
            seeds.add(seed)
            population_sizes.add(population_size)
            dataset_evidence = self._validate_dataset_evidence(config)
            loaded[worker_id] = {
                "path": str(path),
                "port": _config_port(config),
                "semantic_hash": semantic_hash,
                "seed": seed,
                "population_size": population_size,
                "dataset": dataset_evidence,
            }
        if len(semantic_hashes) != 1:
            raise ValueError("local worker configs do not describe the same optimization domain")
        if len(seeds) != 1:
            raise ValueError("local worker configs do not use the same shared seed")
        if len(population_sizes) != 1:
            raise ValueError("local worker configs do not use the same population size")
        return loaded

    def _component_versions(self) -> dict[str, str]:
        configured = self.profile.get("component_versions")
        if isinstance(configured, dict) and configured:
            return {
                str(key): str(value)
                for key, value in sorted(configured.items())
            }
        try:
            from doin_node.versioning import compute_component_versions
        except Exception as exc:
            raise RuntimeError(f"cannot compute DOIN component versions: {exc}") from exc
        versions = compute_component_versions()
        if not isinstance(versions, dict) or not versions:
            raise RuntimeError("DOIN component version map is empty")
        return {str(key): str(value) for key, value in sorted(versions.items())}

    def _coordination_contract(
        self, job: dict[str, Any], configs: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        datasets = {
            (
                value.get("dataset") or {}
            ).get("sha256")
            for value in configs.values()
        }
        datasets.discard(None)
        if len(datasets) > 1:
            raise ValueError("local workers do not use the same dataset bytes")
        seeds = {int(value["seed"]) for value in configs.values()}
        population_sizes = {
            int(value["population_size"]) for value in configs.values()
        }
        contract = {
            "plan_hash": self.plan_hash,
            "job_index": int(job["ordinal"]),
            "job_id": str(job["job_id"]),
            "domain_id": str(job["domain_id"]),
            "domain_semantic_hash": str(job.get("domain_semantic_hash") or ""),
            "shared_population_seed": next(iter(seeds)),
            "shared_population_size": next(iter(population_sizes)),
            "dataset_sha256": next(iter(datasets), None),
            "component_versions": self._component_versions(),
            "bootstrap_node_id": self._bootstrap_node_id(),
            "bootstrap_worker_id": self._bootstrap_worker_id(),
            "worker_join_order": self._global_worker_ids(),
        }
        contract["contract_hash"] = _sha256_json(contract)
        return contract

    def _prepare_coordination(
        self, job: dict[str, Any], configs: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        contract = self._coordination_contract(job, configs)
        existing = self.state.get("coordination") or {}
        if existing.get("job_id") not in (None, job["job_id"]):
            existing = {}
        legacy_adopted = bool(
            existing.get("legacy_adopted")
            or (not existing and self.state.get("phase") == "running")
        )
        coordination = {
            **existing,
            **contract,
            "job_id": job["job_id"],
            "preflight_ready": True,
            "preflight_at": existing.get("preflight_at") or _utc_now(),
            "local_datasets": {
                worker_id: value.get("dataset")
                for worker_id, value in configs.items()
            },
            "role": (
                "bootstrap"
                if self.node_id == contract["bootstrap_node_id"]
                else "follower"
            ),
            "legacy_adopted": legacy_adopted,
        }
        self.state["coordination"] = coordination
        return coordination

    def _worker_state(self, worker_id: str) -> dict[str, Any]:
        workers = self.state.setdefault("workers", {})
        return workers.setdefault(worker_id, {
            "pid": None,
            "pid_start_ticks": None,
            "owns_process_group": False,
            "restart_count": 0,
            "status": "new",
            "last_doin_status": {},
            "last_chain_status": {},
        })

    def _start_or_adopt_worker(self, job: dict[str, Any], worker_id: str, config: dict[str, Any]) -> None:
        worker = self._worker_state(worker_id)
        if _pid_matches(worker.get("pid"), worker.get("pid_start_ticks")):
            return
        discovered = _find_doin_process(Path(config["path"]))
        if discovered:
            worker.update({
                "pid": discovered[0],
                "pid_start_ticks": discovered[1],
                "owns_process_group": False,
                "status": "adopted",
                "adopted_at": _utc_now(),
            })
            self.history.event(
                node_id=self.node_id, job_id=job["job_id"], worker_id=worker_id,
                event="worker_adopted", detail={"pid": discovered[0]},
            )
            return
        if worker.get("restart_count", 0) >= self.restart_limit:
            self._alert(
                f"restart_limit:{worker_id}",
                f"{worker_id} exceeded its restart limit for {job['job_id']}",
            )
            self.state["phase"] = "blocked"
            return
        profile = self._worker_profile(worker_id)
        python = str(profile.get("python") or sys.executable)
        command = [
            python, "-m", "doin_node.cli", "--config", config["path"],
            "--log-level", str(profile.get("log_level", "INFO")),
        ]
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (profile.get("environment") or {}).items()})
        env["PYTHONUNBUFFERED"] = "1"
        log_path = self.log_dir / f"{job['ordinal']:02d}_{job['job_id']}_{worker_id}.log"
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=str(self._resolve_path(profile["doin_node_root"])),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        log_handle.close()
        ticks = _pid_start_ticks(process.pid)
        worker.update({
            "pid": process.pid,
            "pid_start_ticks": ticks,
            "owns_process_group": True,
            "restart_count": int(worker.get("restart_count", 0)) + 1,
            "status": "starting",
            "started_at": _utc_now(),
            "config_path": config["path"],
            "config_semantic_hash": config["semantic_hash"],
            "api_url": f"http://127.0.0.1:{config['port']}",
            "log_path": str(log_path),
            "command_hash": hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(),
            "stopped_verified": False,
            "join_ready": False,
            "join_reason": "worker API has not completed bootstrap verification",
            "join_mismatch_since": None,
        })
        self.history.event(
            node_id=self.node_id, job_id=job["job_id"], worker_id=worker_id,
            event="worker_started", detail={"pid": process.pid, "log_path": str(log_path)},
        )

    def _bootstrap_chain_evidence(
        self, api_url: str, domain_id: str, chain_height: int
    ) -> dict[str, Any]:
        genesis = _http_json(
            api_url + "/chain/block/0", max(self.peer_timeout, 10.0)
        )
        evidence: dict[str, Any] = {
            "genesis_hash": genesis.get("hash"),
            "population_block_index": None,
            "population_block_hash": None,
            "population_transaction_id": None,
            "population_fingerprint": None,
            "shared_population_seed": None,
        }
        for index in range(1, max(1, chain_height)):
            block = _http_json(
                api_url + f"/chain/block/{index}",
                max(self.peer_timeout, 20.0),
            )
            for transaction in block.get("transactions") or []:
                if transaction.get("domain_id") != domain_id:
                    continue
                payload = transaction.get("payload") or {}
                population = payload.get("_shared_population")
                if not isinstance(population, dict):
                    continue
                if int(population.get("generation", -1)) != 0:
                    continue
                evidence.update({
                    "population_block_index": index,
                    "population_block_hash": block.get("hash"),
                    "population_transaction_id": transaction.get("id"),
                    "population_fingerprint": (
                        payload.get("_shared_population_fingerprint")
                        or _population_fingerprint(population)
                    ),
                    "shared_population_seed": (
                        payload.get("_shared_population_seed")
                        if payload.get("_shared_population_seed") is not None
                        else population.get("bootstrap_seed")
                    ),
                })
                return evidence
        return evidence

    def _poll_local_workers(self, job: dict[str, Any], configs: dict[str, dict[str, Any]]) -> None:
        for worker_id, config in configs.items():
            worker = self._worker_state(worker_id)
            api_url = f"http://127.0.0.1:{config['port']}"
            worker["api_url"] = api_url
            try:
                status = _http_json(api_url + "/status", self.peer_timeout)
                chain = _http_json(api_url + "/chain/status", self.peer_timeout)
            except Exception as exc:
                worker["api_error"] = str(exc)
                if _pid_matches(worker.get("pid"), worker.get("pid_start_ticks")):
                    worker["status"] = "starting"
                else:
                    worker["status"] = "exited"
                continue
            worker.update({
                "status": "running",
                "api_error": None,
                "last_seen": _utc_now(),
                "last_doin_status": status,
                "last_chain_status": chain,
            })
            domain = (status.get("domains") or {}).get(job["domain_id"]) or {}
            worker["converged"] = bool(domain.get("converged"))
            worker["best_performance"] = domain.get("best_performance")
            if (
                not worker.get("bootstrap_evidence")
                and int(chain.get("chain_height", 0)) > 1
            ):
                try:
                    worker["bootstrap_evidence"] = self._bootstrap_chain_evidence(
                        api_url, job["domain_id"], int(chain["chain_height"])
                    )
                    worker["bootstrap_evidence_error"] = None
                except Exception as exc:
                    worker["bootstrap_evidence_error"] = str(exc)
            query = urllib.parse.urlencode({"domain_id": job["domain_id"]})
            try:
                shared = _http_json(
                    api_url + f"/api/shared/candidates?{query}",
                    max(self.peer_timeout, 5.0),
                )
            except Exception as exc:
                worker["shared_population_error"] = str(exc)
            else:
                worker["shared_population"] = {
                    key: shared.get(key)
                    for key in (
                        "domain_id",
                        "generation",
                        "pop_size",
                        "evaluated",
                        "claimed",
                        "free",
                        "bootstrap_seed",
                        "population_fingerprint",
                    )
                    if shared.get(key) is not None
                }
                worker["shared_population_error"] = None
                peer_prefix = str(status.get("peer_id") or "")
                owned_claims = [
                    item
                    for item in shared.get("candidates") or []
                    if item.get("state") == "claimed"
                    and str(item.get("owner") or "").startswith(peer_prefix)
                ]
                worker["owns_candidate"] = bool(owned_claims)
                worker["owned_candidate_indices"] = [
                    int(item["index"]) for item in owned_claims
                ]
                progress_signature = _sha256_json({
                    "generation": shared.get("generation"),
                    "evaluated": shared.get("evaluated"),
                    "claimed": shared.get("claimed"),
                    "free": shared.get("free"),
                    "chain_height": chain.get("chain_height"),
                })
                if worker.get("progress_signature") != progress_signature:
                    worker["progress_signature"] = progress_signature
                    worker["last_progress_at"] = time.time()
                if worker["owns_candidate"] or not int(shared.get("free", 0)):
                    worker["claimless_since"] = None
                elif worker.get("claimless_since") is None:
                    worker["claimless_since"] = time.time()
            finalized_height = chain.get("finalized_height")
            worker["finalized_height"] = finalized_height
            if (
                finalized_height is not None
                and int(finalized_height) >= 0
                and worker.get("finalized_hash_height") != int(finalized_height)
            ):
                try:
                    anchor = _http_json(
                        api_url + f"/chain/block/{int(finalized_height)}",
                        max(self.peer_timeout, 20.0),
                    )
                except Exception as exc:
                    worker["finalized_hash_error"] = str(exc)
                else:
                    worker["finalized_hash"] = anchor.get("hash")
                    worker["finalized_hash_height"] = int(finalized_height)
                    worker["finalized_hash_error"] = None

    def _public_worker(self, worker_id: str, worker: dict[str, Any]) -> dict[str, Any]:
        chain = worker.get("last_chain_status") or {}
        doin = worker.get("last_doin_status") or {}
        return {
            "worker_id": worker_id,
            "status": worker.get("status"),
            "pid": worker.get("pid"),
            "restart_count": worker.get("restart_count", 0),
            "converged": bool(worker.get("converged")),
            "best_performance": worker.get("best_performance"),
            "chain_height": chain.get("chain_height"),
            "tip_hash": chain.get("tip_hash"),
            "finalized_height": worker.get("finalized_height"),
            "finalized_hash": worker.get("finalized_hash"),
            "component_versions": chain.get("component_versions") or {},
            "peer_count": doin.get("peers"),
            "owns_candidate": bool(worker.get("owns_candidate")),
            "owned_candidate_indices": worker.get("owned_candidate_indices") or [],
            "shared_population": worker.get("shared_population") or {},
            "shared_population_error": worker.get("shared_population_error"),
            "bootstrap_evidence": worker.get("bootstrap_evidence") or {},
            "bootstrap_evidence_error": worker.get("bootstrap_evidence_error"),
            "join_ready": bool(worker.get("join_ready")),
            "join_reason": worker.get("join_reason"),
            "api_url": worker.get("api_url"),
            "api_error": worker.get("api_error"),
            "last_seen": worker.get("last_seen"),
            "stopped_verified": bool(worker.get("stopped_verified")),
            "log_path": worker.get("log_path"),
        }

    def status_payload(self) -> dict[str, Any]:
        # Peer status must remain readable while the reconciliation thread is
        # itself polling peers. Taking the long-lived tick lock here creates a
        # distributed lock convoy where every supervisor times out on every
        # other supervisor. The main loop is the only writer; this compact
        # snapshot is intentionally lock-free and never mutates state.
        state = copy.deepcopy(self.state)
        job_index = int(state.get("job_index", 0))
        jobs = self.plan["jobs"]
        job = jobs[job_index] if 0 <= job_index < len(jobs) else None
        completed_campaigns = [
            {
                "job_id": item["job_id"],
                "ordinal": item["ordinal"],
                "chain_height": item.get("chain_height"),
                "tip_hash": item.get("tip_hash"),
                "finalized_height": (item.get("evidence") or {}).get("finalized_height"),
                "finalized_hash": (item.get("evidence") or {}).get("finalized_hash"),
                "artifact_sha256": item.get("artifact_sha256"),
                "champion_fitness": item.get("champion_fitness"),
                "completed_at": item.get("completed_at"),
            }
            for item in self.history.campaigns()
            if item.get("status") == "completed"
        ]
        worker_states = state.get("workers") or {}
        return {
            "schema_version": STATE_SCHEMA,
            "node_id": self.node_id,
            "plan_id": self.plan.get("plan_id"),
            "plan_hash": self.plan_hash,
            "phase": state.get("phase"),
            "job_index": job_index,
            "job_id": job.get("job_id") if job else None,
            "domain_id": job.get("domain_id") if job else None,
            "workers": {
                worker_id: self._public_worker(worker_id, worker_states.get(worker_id, {}))
                for worker_id in self._local_worker_ids()
            },
            "coordination": state.get("coordination") or {},
            "archive": state.get("archive") or {},
            "completed_campaigns": completed_campaigns,
            "alerts": state.get("alerts") or [],
            "updated_at": state.get("updated_at"),
        }

    def _network_status(self) -> dict[str, Any]:
        local = self.status_payload()
        history = self.history.campaigns()
        history_by_job = {item["job_id"]: item for item in history}
        current_index = int(local.get("job_index", 0))
        plan_jobs = []
        for job in self.plan["jobs"]:
            ordinal = int(job["ordinal"])
            historical = history_by_job.get(job["job_id"])
            if historical and historical.get("status") == "completed":
                status = "completed"
            elif ordinal == current_index and local.get("phase") != "complete":
                status = str(local.get("phase") or "running")
            elif ordinal < current_index:
                status = "history_missing"
            else:
                status = "queued"
            plan_jobs.append({
                "ordinal": ordinal,
                "job_id": job["job_id"],
                "domain_id": job["domain_id"],
                "purpose": job.get("purpose"),
                "status": status,
                "champion_fitness": historical.get("champion_fitness") if historical else None,
                "artifact_sha256": historical.get("artifact_sha256") if historical else None,
            })
        participants: dict[str, Any] = {self.node_id: {"online": True, "status": local}}
        for participant in self.plan["participants"]:
            node_id = str(participant["node_id"])
            if node_id == self.node_id:
                continue
            url = str(participant["supervisor_url"]).rstrip("/") + "/api/status"
            try:
                remote = _http_json(url, self.peer_timeout)
            except Exception as exc:
                participants[node_id] = {"online": False, "error": str(exc), "url": url}
            else:
                participants[node_id] = {"online": True, "status": remote, "url": url}
        return {
            "plan_id": self.plan.get("plan_id"),
            "plan_hash": self.plan_hash,
            "plan_jobs": plan_jobs,
            "participants": participants,
            "history": history,
            "generated_at": _utc_now(),
        }

    def _network_worker(
        self, network: dict[str, Any], worker_id: str
    ) -> dict[str, Any] | None:
        node_id = self._worker_node_id(worker_id)
        report = (network.get("participants") or {}).get(node_id) or {}
        if not report.get("online"):
            return None
        return ((report.get("status") or {}).get("workers") or {}).get(worker_id)

    def _startup_preflight_ready(
        self, network: dict[str, Any], job: dict[str, Any]
    ) -> tuple[bool, str]:
        contracts: set[str] = set()
        for participant in self.plan["participants"]:
            node_id = str(participant["node_id"])
            report = (network.get("participants") or {}).get(node_id) or {}
            if not report.get("online"):
                return False, f"required startup supervisor {node_id} is offline"
            status = report.get("status") or {}
            if status.get("plan_hash") != self.plan_hash:
                return False, f"{node_id} has a different plan hash"
            if status.get("job_index") != int(job["ordinal"]):
                return False, f"{node_id} has not reached job {job['ordinal']}"
            if status.get("job_id") != job["job_id"]:
                return False, f"{node_id} is on job {status.get('job_id')}"
            coordination = status.get("coordination") or {}
            if not coordination.get("preflight_ready"):
                return False, f"{node_id} has not completed startup preflight"
            contract_hash = str(coordination.get("contract_hash") or "")
            if not contract_hash:
                return False, f"{node_id} has no startup contract"
            contracts.add(contract_hash)
        if len(contracts) != 1:
            return False, "supervisors disagree on seed/data/config/version contract"
        expected = str(
            (self.state.get("coordination") or {}).get("contract_hash") or ""
        )
        if contracts != {expected}:
            return False, "local startup contract differs from the swarm contract"
        return True, "all supervisors passed the identical startup preflight"

    @staticmethod
    def _lineage_key(worker: dict[str, Any] | None) -> tuple[Any, ...] | None:
        if not worker:
            return None
        evidence = worker.get("bootstrap_evidence") or {}
        values = (
            evidence.get("genesis_hash"),
            evidence.get("population_block_hash"),
            evidence.get("population_fingerprint"),
        )
        if any(value in (None, "") for value in values):
            return None
        return values

    def _worker_launch_ready(
        self,
        network: dict[str, Any],
        job: dict[str, Any],
        worker_id: str,
    ) -> tuple[bool, str]:
        ready, reason = self._startup_preflight_ready(network, job)
        if not ready:
            return False, reason
        order = self._global_worker_ids()
        rank = order.index(worker_id)
        for predecessor in order[:rank]:
            worker = self._network_worker(network, predecessor)
            if not worker or not worker.get("join_ready"):
                return False, f"waiting for predecessor {predecessor}"
        if rank:
            bootstrap = self._network_worker(network, order[0])
            if not bootstrap or self._lineage_key(bootstrap) is None:
                return False, "bootstrap worker has no verified chain lineage"
        return True, "ordered startup predecessor barrier passed"

    def _validate_worker_join(
        self,
        network: dict[str, Any],
        job: dict[str, Any],
        worker_id: str,
        config: dict[str, Any],
    ) -> tuple[bool, str]:
        worker = self._worker_state(worker_id)
        if worker.get("status") != "running":
            return False, f"{worker_id} API is not running"
        evidence = worker.get("bootstrap_evidence") or {}
        if not evidence.get("genesis_hash"):
            return False, f"{worker_id} has no genesis evidence"
        if not evidence.get("population_block_hash"):
            return False, f"{worker_id} has no generation-zero population block"
        shared = worker.get("shared_population") or {}
        if shared.get("domain_id") != job["domain_id"]:
            return False, f"{worker_id} shared pool domain mismatch"
        if int(shared.get("pop_size", 0)) != int(config["population_size"]):
            return False, f"{worker_id} shared population size mismatch"
        observed_seed = evidence.get("shared_population_seed")
        if observed_seed is not None and int(observed_seed) != int(config["seed"]):
            return False, f"{worker_id} shared seed mismatch"
        expected_versions = (
            (self.state.get("coordination") or {}).get("component_versions") or {}
        )
        observed_versions = (
            (worker.get("last_chain_status") or {}).get("component_versions") or {}
        )
        if observed_versions != expected_versions:
            return False, f"{worker_id} component versions differ from startup contract"
        if worker_id != self._bootstrap_worker_id():
            bootstrap = self._network_worker(network, self._bootstrap_worker_id())
            expected_lineage = self._lineage_key(bootstrap)
            observed_lineage = self._lineage_key(self._public_worker(worker_id, worker))
            if expected_lineage is None:
                return False, "bootstrap lineage is not yet available"
            if observed_lineage != expected_lineage:
                return False, f"{worker_id} joined a different blockchain lineage"
        return True, "seed, genesis, population and component lineage match"

    def _repair_join_mismatch(
        self,
        job: dict[str, Any],
        worker_id: str,
        config: dict[str, Any],
        reason: str,
    ) -> None:
        worker = self._worker_state(worker_id)
        since = worker.get("join_mismatch_since")
        if since is None:
            worker["join_mismatch_since"] = time.time()
            return
        if time.time() - float(since) < self.join_grace_seconds:
            return
        repairs = int(worker.get("join_repair_count", 0))
        if repairs >= self.restart_limit:
            self.state["phase"] = "blocked"
            self._alert(
                f"join_repair_limit:{worker_id}",
                f"{worker_id} could not join the canonical swarm: {reason}",
            )
            return
        self._stop_worker(job, worker_id, config)
        worker["join_repair_count"] = repairs + 1
        worker["join_mismatch_since"] = None
        worker["bootstrap_evidence"] = {}
        worker["shared_population"] = {}
        worker["join_ready"] = False
        self.history.event(
            node_id=self.node_id,
            job_id=job["job_id"],
            worker_id=worker_id,
            event="worker_join_repair",
            detail={"reason": reason, "repair_count": repairs + 1},
        )

    def _update_join_state(
        self,
        network: dict[str, Any],
        job: dict[str, Any],
        configs: dict[str, dict[str, Any]],
    ) -> None:
        for worker_id, config in configs.items():
            worker = self._worker_state(worker_id)
            if worker.get("status") not in {"running", "starting", "adopted"}:
                continue
            ready, reason = self._validate_worker_join(
                network, job, worker_id, config
            )
            worker["join_ready"] = ready
            worker["join_reason"] = reason
            if ready:
                worker["join_mismatch_since"] = None
                worker["joined_at"] = worker.get("joined_at") or _utc_now()
                self._clear_alert(f"join_mismatch:{worker_id}")
            elif worker.get("status") == "running":
                self._alert(
                    f"join_mismatch:{worker_id}", reason, severity="warning"
                )
                self._repair_join_mismatch(
                    job, worker_id, config, reason
                )

    def _runtime_swarm_health(
        self, network: dict[str, Any], job: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        issues: list[str] = []
        workers: list[dict[str, Any]] = []
        for worker_id in self._global_worker_ids():
            worker = self._network_worker(network, worker_id)
            if not worker or worker.get("status") != "running":
                issues.append(f"{worker_id} is not running")
                continue
            workers.append(worker)
        lineages = {
            lineage for worker in workers
            if (lineage := self._lineage_key(worker)) is not None
        }
        if len(lineages) > 1:
            issues.append("workers report different genesis/population lineage")
        versions = {
            _canonical_json(worker.get("component_versions") or {})
            for worker in workers
        }
        if len(versions) > 1:
            issues.append("workers report different component versions")
        finalized = {
            (worker.get("finalized_height"), worker.get("finalized_hash"))
            for worker in workers
            if worker.get("finalized_hash")
        }
        if len(finalized) > 1:
            issues.append("workers report different finalized blockchain anchors")
        pools = [
            worker.get("shared_population") or {}
            for worker in workers
        ]
        generations = {
            (pool.get("generation"), pool.get("pop_size")) for pool in pools
            if pool.get("generation") is not None
        }
        if len(generations) > 1:
            issues.append("workers report different shared-population generations")
        return not issues, issues

    def _repair_claimless_local_workers(
        self,
        job: dict[str, Any],
        configs: dict[str, dict[str, Any]],
    ) -> None:
        for worker_id, config in configs.items():
            worker = self._worker_state(worker_id)
            since = worker.get("claimless_since")
            shared = worker.get("shared_population") or {}
            if (
                worker.get("status") != "running"
                or worker.get("converged")
                or worker.get("owns_candidate")
                or int(shared.get("free", 0)) < 1
                or since is None
                or time.time() - float(since) < self.claimless_grace_seconds
            ):
                continue
            repairs = int(worker.get("claimless_repair_count", 0))
            if repairs >= self.restart_limit:
                self._alert(
                    f"claimless_repair_limit:{worker_id}",
                    f"{worker_id} remains idle while shared candidates are free",
                )
                continue
            self._stop_worker(job, worker_id, config)
            worker["claimless_repair_count"] = repairs + 1
            worker["claimless_since"] = None
            self.history.event(
                node_id=self.node_id,
                job_id=job["job_id"],
                worker_id=worker_id,
                event="claimless_worker_repair",
                detail={"repair_count": repairs + 1},
            )

    def _completion_evidence(self, network: dict[str, Any], job: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        worker_rows: dict[str, dict[str, Any]] = {}
        for participant in self.plan["participants"]:
            node_id = str(participant["node_id"])
            report = network["participants"].get(node_id) or {}
            if not report.get("online"):
                return False, f"required supervisor {node_id} is offline", {}
            status = report.get("status") or {}
            if status.get("plan_hash") != self.plan_hash:
                return False, f"{node_id} has a different plan hash", {}
            if status.get("job_id") != job["job_id"]:
                return False, f"{node_id} is on job {status.get('job_id')}", {}
            for worker_id in participant["workers"]:
                worker = (status.get("workers") or {}).get(worker_id)
                if not worker:
                    return False, f"{node_id} does not report {worker_id}", {}
                archived = status.get("phase") in {"archiving", "stopping", "stopped"}
                if not worker.get("converged") and not archived:
                    return False, f"{worker_id} has not converged", {}
                if not worker.get("chain_height") or not worker.get("tip_hash"):
                    archive = status.get("archive") or {}
                    if archive.get("chain_height") and archive.get("tip_hash"):
                        worker = dict(worker)
                        worker["chain_height"] = archive["chain_height"]
                        worker["tip_hash"] = archive["tip_hash"]
                    else:
                        return False, f"{worker_id} lacks chain evidence", {}
                worker_rows[str(worker_id)] = worker
        heights = {row["chain_height"] for row in worker_rows.values()}
        if len(heights) != 1:
            return False, "workers have not synchronized to one chain height", worker_rows
        versions = {
            _canonical_json(row.get("component_versions") or {}) for row in worker_rows.values()
        }
        if len(versions) != 1:
            return False, "worker component versions do not match", worker_rows
        tips = {row["tip_hash"] for row in worker_rows.values()}
        if len(tips) == 1:
            return True, "all workers converged on one chain", worker_rows
        anchors = {
            (row.get("finalized_height"), row.get("finalized_hash"))
            for row in worker_rows.values()
        }
        if len(anchors) != 1 or any(value is None for value in next(iter(anchors), ())):
            return False, "terminal forks do not share a verified finalized anchor", worker_rows
        return (
            True,
            "workers converged on equal-height terminal forks after one finalized anchor",
            worker_rows,
        )

    def _stop_barrier_ready(self, network: dict[str, Any], job: dict[str, Any]) -> tuple[bool, str]:
        archives: list[dict[str, Any]] = []
        for participant in self.plan["participants"]:
            node_id = str(participant["node_id"])
            report = network["participants"].get(node_id) or {}
            if not report.get("online"):
                return False, f"required supervisor {node_id} is offline"
            status = report.get("status") or {}
            if status.get("plan_hash") != self.plan_hash:
                return False, f"{node_id} plan hash mismatch"
            if status.get("job_id") == job["job_id"]:
                if status.get("phase") != "stopped":
                    return False, f"{node_id} phase is {status.get('phase')}"
                for worker_id in participant["workers"]:
                    worker = (status.get("workers") or {}).get(worker_id) or {}
                    if not worker.get("stopped_verified"):
                        return False, f"{worker_id} process stop is not verified"
                archive = status.get("archive") or {}
                champion = archive.get("champion") or {}
                if not archive.get("tip_hash") or not champion.get("artifact_sha256"):
                    return False, f"{node_id} archive is incomplete"
                archives.append(archive)
                continue

            # A participant may have crossed the same barrier a few seconds
            # earlier.  Its immutable history acknowledgement must remain
            # usable after its current-job fields move forward.
            completed = next(
                (
                    item for item in status.get("completed_campaigns") or []
                    if item.get("job_id") == job["job_id"]
                ),
                None,
            )
            if not completed:
                return False, f"{node_id} is on another job without a completion acknowledgement"
            if not completed.get("tip_hash") or not completed.get("artifact_sha256"):
                return False, f"{node_id} completion acknowledgement is incomplete"
            archives.append({
                "chain_height": completed.get("chain_height"),
                "tip_hash": completed.get("tip_hash"),
                "finalized_height": completed.get("finalized_height"),
                "finalized_hash": completed.get("finalized_hash"),
                "champion": {"artifact_sha256": completed.get("artifact_sha256")},
            })
        champions = {
            (item.get("champion") or {}).get("artifact_sha256") for item in archives
        }
        if len(champions) != 1 or None in champions:
            return False, "participant archives do not identify the same champion artifact"
        heights = {item.get("chain_height") for item in archives}
        if len(heights) != 1 or None in heights:
            return False, "participant archives do not identify the same chain height"
        tips = {item.get("tip_hash") for item in archives}
        if len(tips) != 1:
            anchors = {
                (item.get("finalized_height"), item.get("finalized_hash"))
                for item in archives
            }
            if len(anchors) != 1 or any(value is None for value in next(iter(anchors), ())):
                return False, "terminal fork archives do not share a finalized anchor"
        return True, "all supervisors archived and stopped"

    def _compact_metrics(self, value: dict[str, Any]) -> dict[str, Any]:
        excluded = {"history", "splits", "summary_table", "best_model_path", "_model_b64"}
        return {key: item for key, item in value.items() if key not in excluded}

    def _archive_champion(self, job: dict[str, Any], api_url: str) -> dict[str, Any]:
        chain = _http_json(api_url + "/chain/status", max(self.peer_timeout, 10.0))
        height = int(chain.get("chain_height", 0))
        finalized_height = int(chain.get("finalized_height", -1))
        finalized_hash = None
        if finalized_height >= 0:
            finalized_block = _http_json(
                api_url + f"/chain/block/{finalized_height}",
                max(self.peer_timeout, 20.0),
            )
            finalized_hash = finalized_block.get("hash")
        higher_is_better = bool(job.get("higher_is_better", True))
        candidates: list[dict[str, Any]] = []
        for index in range(height):
            block = _http_json(api_url + f"/chain/block/{index}", max(self.peer_timeout, 20.0))
            for transaction in block.get("transactions") or []:
                if transaction.get("tx_type") != "optimae_accepted":
                    continue
                if transaction.get("domain_id") != job["domain_id"]:
                    continue
                payload = transaction.get("payload") or {}
                parameters = payload.get("parameters") or {}
                model_b64 = parameters.get("_model_b64") if isinstance(parameters, dict) else None
                if not model_b64:
                    continue
                fitness = payload.get("verified_performance")
                if fitness is None:
                    fitness = payload.get("reported_performance")
                if fitness is None:
                    continue
                candidates.append({
                    "fitness": float(fitness),
                    "block_index": index,
                    "block_hash": block.get("hash"),
                    "transaction_id": transaction.get("id"),
                    "peer_id": transaction.get("peer_id"),
                    "payload": payload,
                    "model_b64": model_b64,
                })
        if not candidates:
            raise RuntimeError(f"no embedded champion artifact found for {job['domain_id']}")
        champion = (max if higher_is_better else min)(candidates, key=lambda item: item["fitness"])
        try:
            model_bytes = base64.b64decode(champion["model_b64"], validate=True)
        except Exception as exc:
            raise RuntimeError("champion model artifact is not valid base64") from exc
        payload = champion["payload"]
        metrics = dict(payload.get("champion_metrics") or {})
        artifact_sha = hashlib.sha256(model_bytes).hexdigest()
        declared_sha = metrics.get("model_artifact_sha256")
        if declared_sha and declared_sha != artifact_sha:
            raise RuntimeError(
                f"champion artifact hash mismatch: declared {declared_sha}, observed {artifact_sha}"
            )
        declared_bytes = metrics.get("model_artifact_bytes")
        if declared_bytes is not None and int(declared_bytes) != len(model_bytes):
            raise RuntimeError("champion artifact size does not match blockchain metadata")
        artifact_format = str(metrics.get("model_artifact_format") or "binary")
        suffix = {
            "stable_baselines3_zip": ".zip",
            "keras": ".keras",
            "pytorch": ".pt",
        }.get(artifact_format, ".bin")
        destination = self.artifact_dir / job["job_id"]
        destination.mkdir(parents=True, exist_ok=True)
        artifact_path = destination / f"{artifact_sha}{suffix}"
        if artifact_path.exists():
            if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != artifact_sha:
                raise RuntimeError(f"existing artifact at {artifact_path} failed hash verification")
        else:
            temporary = artifact_path.with_name(artifact_path.name + ".tmp")
            temporary.write_bytes(model_bytes)
            temporary.replace(artifact_path)
        public_parameters = {
            key: value for key, value in (payload.get("parameters") or {}).items()
            if key != "_model_b64"
        }
        manifest_path = destination / "champion_manifest.json"
        manifest = {
            "schema_version": HISTORY_SCHEMA,
            "plan_id": self.plan.get("plan_id"),
            "plan_hash": self.plan_hash,
            "job_id": job["job_id"],
            "domain_id": job["domain_id"],
            "fitness": champion["fitness"],
            "higher_is_better": higher_is_better,
            "peer_id": champion["peer_id"],
            "block_index": champion["block_index"],
            "block_hash": champion["block_hash"],
            "transaction_id": champion["transaction_id"],
            "chain_height": height,
            "tip_hash": chain.get("tip_hash"),
            "finalized_height": finalized_height,
            "finalized_hash": finalized_hash,
            "component_versions": chain.get("component_versions") or {},
            "parameters": public_parameters,
            "metrics": {key: value for key, value in metrics.items() if key != "_model_b64"},
            "artifact": {
                "sha256": artifact_sha,
                "bytes": len(model_bytes),
                "format": artifact_format,
                "path": str(artifact_path),
            },
            "archived_at": _utc_now(),
        }
        _atomic_json(manifest_path, manifest)
        return {
            "chain_height": height,
            "tip_hash": chain.get("tip_hash"),
            "finalized_height": finalized_height,
            "finalized_hash": finalized_hash,
            "component_versions": chain.get("component_versions") or {},
            "champion": {
                "fitness": champion["fitness"],
                "peer_id": champion["peer_id"],
                "block_index": champion["block_index"],
                "transaction_id": champion["transaction_id"],
                "parameters": public_parameters,
                "metrics": self._compact_metrics(metrics),
                "artifact_sha256": artifact_sha,
                "artifact_bytes": len(model_bytes),
                "artifact_format": artifact_format,
                "artifact_path": str(artifact_path),
                "manifest_path": str(manifest_path),
            },
        }

    def _stop_worker(self, job: dict[str, Any], worker_id: str, config: dict[str, Any]) -> bool:
        worker = self._worker_state(worker_id)
        pid = worker.get("pid")
        ticks = worker.get("pid_start_ticks")
        if _pid_matches(pid, ticks):
            try:
                if worker.get("owns_process_group") and os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGTERM)
                else:
                    os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + self.stop_timeout
            while _pid_matches(pid, ticks) and time.monotonic() < deadline:
                _reap_child(pid)
                time.sleep(0.25)
            _reap_child(pid)
            if _pid_matches(pid, ticks):
                try:
                    if worker.get("owns_process_group") and os.getpgid(pid) == pid:
                        os.killpg(pid, signal.SIGKILL)
                    else:
                        os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 5.0
                while _pid_matches(pid, ticks) and time.monotonic() < deadline:
                    _reap_child(pid)
                    time.sleep(0.1)
                _reap_child(pid)
        process_gone = not _pid_matches(pid, ticks)
        api_down = False
        try:
            _http_json(f"http://127.0.0.1:{config['port']}/status", 0.5)
        except Exception:
            api_down = True
        worker["stopped_verified"] = process_gone and api_down
        worker["status"] = "stopped" if worker["stopped_verified"] else "stop_failed"
        if worker["stopped_verified"]:
            self.history.event(
                node_id=self.node_id, job_id=job["job_id"], worker_id=worker_id,
                event="worker_stopped", detail={"pid": pid, "port": config["port"]},
            )
        return bool(worker["stopped_verified"])

    def _advance_job(self, job: dict[str, Any]) -> None:
        self.history.mark_completed(job, self.plan_hash, self.state["archive"])
        self.history.event(
            node_id=self.node_id, job_id=job["job_id"], event="campaign_completed",
            detail={"archive": self.state["archive"]},
        )
        next_index = int(self.state["job_index"]) + 1
        next_job = self._job(next_index)
        self.state.update({
            "job_index": next_index,
            "job_id": next_job["job_id"] if next_job else None,
            "phase": "starting" if next_job else "complete",
            "workers": {},
            "archive": {},
            "alerts": [],
            "completion_candidate_since": None,
            "coordination": {},
        })

    def tick(self) -> None:
        with self._mutex:
            job = self._job()
            if job is None:
                self.state["phase"] = "complete"
                self._save_state()
                return
            try:
                configs = self._validate_local_configs(job)
                coordination = self._prepare_coordination(job, configs)
            except Exception as exc:
                self.state["phase"] = "blocked"
                self._alert("config_validation", str(exc))
                self._save_state()
                return

            phase = self.state.get("phase", "starting")
            if phase == "starting":
                self._poll_local_workers(job, configs)
                network = self._network_status()
                for worker_id in self._global_worker_ids():
                    if worker_id not in configs:
                        continue
                    worker = self._worker_state(worker_id)
                    if _pid_matches(
                        worker.get("pid"), worker.get("pid_start_ticks")
                    ):
                        continue
                    ready, reason = self._worker_launch_ready(
                        network, job, worker_id
                    )
                    worker["launch_ready"] = ready
                    worker["launch_reason"] = reason
                    if ready:
                        self._start_or_adopt_worker(
                            job, worker_id, configs[worker_id]
                        )
                        break
                self._poll_local_workers(job, configs)
                network = self._network_status()
                self._update_join_state(network, job, configs)
                network = self._network_status()
                all_joined = all(
                    (
                        self._network_worker(network, worker_id) or {}
                    ).get("join_ready")
                    for worker_id in self._global_worker_ids()
                )
                if all_joined:
                    self.history.mark_started(job, self.plan_hash)
                    self.history.event(
                        node_id=self.node_id,
                        job_id=job["job_id"],
                        event="campaign_running",
                        detail={
                            "contract_hash": coordination["contract_hash"],
                            "bootstrap_worker_id": self._bootstrap_worker_id(),
                            "worker_join_order": self._global_worker_ids(),
                        },
                    )
                    self.state["phase"] = "running"
                    self.state["coordination"]["swarm_ready_at"] = _utc_now()
                    self._clear_alert("startup_barrier")
                    self._clear_alert("local_worker_unavailable")
                else:
                    ready, reason = self._startup_preflight_ready(network, job)
                    if ready:
                        reason = "waiting for ordered worker bootstrap/join barrier"
                    self._alert("startup_barrier", reason, severity="warning")

            elif phase == "running":
                network = self._network_status()
                for worker_id, config in configs.items():
                    worker = self._worker_state(worker_id)
                    if _pid_matches(
                        worker.get("pid"), worker.get("pid_start_ticks")
                    ):
                        continue
                    if coordination.get("legacy_adopted"):
                        self._start_or_adopt_worker(job, worker_id, config)
                        continue
                    ready, reason = self._worker_launch_ready(
                        network, job, worker_id
                    )
                    worker["launch_ready"] = ready
                    worker["launch_reason"] = reason
                    if ready:
                        self._start_or_adopt_worker(job, worker_id, config)
                self._poll_local_workers(job, configs)
                if self.state.get("phase") == "blocked":
                    self._save_state()
                    return
                if coordination.get("legacy_adopted"):
                    for worker_id in configs:
                        worker = self._worker_state(worker_id)
                        lineage = self._lineage_key(
                            self._public_worker(worker_id, worker)
                        )
                        worker["join_ready"] = bool(
                            worker.get("status") == "running" and lineage
                        )
                        worker["join_reason"] = (
                            "running swarm adopted; genesis and population lineage verified"
                            if worker["join_ready"]
                            else "waiting for adopted worker lineage evidence"
                        )
                network = self._network_status()
                if not coordination.get("legacy_adopted"):
                    self._update_join_state(network, job, configs)
                    network = self._network_status()
                healthy, issues = self._runtime_swarm_health(network, job)
                if healthy:
                    self._clear_alert("swarm_health")
                else:
                    self._alert(
                        "swarm_health", "; ".join(issues), severity="warning"
                    )
                self._repair_claimless_local_workers(job, configs)
                if all(
                    self._worker_state(worker_id).get("status") == "running"
                    for worker_id in configs
                ):
                    self._clear_alert("local_worker_unavailable")
                else:
                    self._alert(
                        "local_worker_unavailable",
                        "one or more local DOIN workers have not exposed a healthy API",
                        severity="warning",
                    )
                ready, reason, _ = self._completion_evidence(network, job)
                if ready:
                    since = self.state.get("completion_candidate_since")
                    if since is None:
                        self.state["completion_candidate_since"] = time.time()
                    elif time.time() - float(since) >= self.stability_seconds:
                        self.state["phase"] = "archiving"
                        self.history.event(
                            node_id=self.node_id, job_id=job["job_id"],
                            event="global_convergence_barrier_reached",
                        )
                    self._clear_alert("completion_barrier")
                else:
                    self.state["completion_candidate_since"] = None
                    if any(self._worker_state(w).get("converged") for w in configs):
                        self._alert("completion_barrier", reason, severity="warning")

            elif phase == "archiving":
                first_worker = next(iter(configs))
                api_url = f"http://127.0.0.1:{configs[first_worker]['port']}"
                try:
                    archive = self._archive_champion(job, api_url)
                except Exception as exc:
                    self._alert("champion_archive", str(exc))
                else:
                    archive["started_at"] = next(
                        (
                            value.get("started_at")
                            for value in self.state.get("workers", {}).values()
                            if value.get("started_at")
                        ),
                        None,
                    )
                    self.state["archive"] = archive
                    self.state["phase"] = "stopping"
                    self._clear_alert("champion_archive")
                    self.history.event(
                        node_id=self.node_id, job_id=job["job_id"],
                        event="champion_archived",
                        detail={
                            "artifact_sha256": archive["champion"]["artifact_sha256"],
                            "fitness": archive["champion"]["fitness"],
                        },
                    )

            elif phase == "stopping":
                stopped = [
                    self._stop_worker(job, worker_id, config)
                    for worker_id, config in configs.items()
                ]
                if all(stopped):
                    self.state["phase"] = "stopped"
                    self._clear_alert("worker_stop")
                else:
                    self._alert("worker_stop", "one or more DOIN processes failed the stop verification")

            elif phase == "stopped":
                network = self._network_status()
                ready, reason = self._stop_barrier_ready(network, job)
                if ready:
                    self._advance_job(job)
                    self._clear_alert("stop_barrier")
                else:
                    self._alert("stop_barrier", reason, severity="warning")

            elif phase == "blocked":
                pass
            else:
                self.state["phase"] = "blocked"
                self._alert("invalid_phase", f"unknown persisted phase {phase!r}")
            self._save_state()

    def _dashboard_html(self) -> str:
        title = html.escape(str(self.plan.get("plan_id") or "DOIN campaign swarm"))
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--bg:#f5f7f9;--ink:#14202b;--muted:#62707d;--line:#d9e0e6;--ok:#137a51;--warn:#9a6500;--bad:#b42318;--panel:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{background:#17232d;color:#fff;padding:18px 24px}}h1{{font-size:20px;margin:0;letter-spacing:0}}header p{{margin:4px 0 0;color:#c6d0d8}}
main{{max-width:1500px;margin:auto;padding:18px 24px}}section{{margin:0 0 22px}}h2{{font-size:15px;margin:0 0 8px}}
.summary{{display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden}}
.metric{{background:var(--panel);padding:12px;min-height:72px}}.metric b{{display:block;font-size:18px}}.metric span{{color:var(--muted);font-size:12px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{font-size:12px;color:var(--muted);background:#eef2f5}}code{{font-size:12px}}
.ok{{color:var(--ok);font-weight:650}}.warn{{color:var(--warn);font-weight:650}}.bad{{color:var(--bad);font-weight:700}}.alerts{{border-left:4px solid var(--bad);background:#fff3f2;padding:10px 14px;margin-bottom:14px}}a{{color:#075c9c}}
@media(max-width:800px){{main{{padding:12px}}.summary{{grid-template-columns:1fr 1fr}}table{{display:block;overflow:auto}}}}
</style></head><body><header><h1>{title}</h1><p>Replicated campaign lifecycle and champion registry</p></header>
<main><div id="alerts"></div><section><div class="summary" id="summary"></div></section>
<section><h2>Participants</h2><table><thead><tr><th>Node / worker</th><th>Phase</th><th>Process</th><th>Join / candidate</th><th>Chain lineage</th><th>Fitness</th><th>Versions</th></tr></thead><tbody id="workers"></tbody></table></section>
<section><h2>Optimization queue</h2><table><thead><tr><th>#</th><th>Job</th><th>Domain</th><th>Status</th><th>Purpose</th><th>Champion</th></tr></thead><tbody id="queue"></tbody></table></section>
<section><h2>Campaign history</h2><table><thead><tr><th>#</th><th>Job</th><th>Domain</th><th>Status</th><th>Champion</th><th>Artifact</th><th>Completed</th></tr></thead><tbody id="history"></tbody></table></section></main>
<script>
const esc=v=>String(v??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
const short=v=>v?esc(String(v).slice(0,12)):'-'; const pct=v=>v==null?'-':Number(v).toFixed(6);
async function refresh(){{let d;try{{d=await(await fetch('/api/network')).json()}}catch(e){{document.querySelector('#alerts').innerHTML='<div class="alerts">Dashboard API unavailable: '+esc(e)+'</div>';return}}
 const ps=Object.values(d.participants), online=ps.filter(x=>x.online).length, local=ps.find(x=>x.status)?.status||{{}};
 const coordination=local.coordination||{{}}, joined=ps.filter(p=>p.online).flatMap(p=>Object.values((p.status||{{}}).workers||{{}})).filter(w=>w.join_ready).length, workers=ps.filter(p=>p.online).flatMap(p=>Object.values((p.status||{{}}).workers||{{}}));
 document.querySelector('#summary').innerHTML=[['Current job',local.job_id||'complete'],['Phase',local.phase],['Supervisors',online+'/'+ps.length],['Workers joined',joined+'/'+workers.length],['Bootstrap',coordination.bootstrap_worker_id||'-'],['Contract',short(coordination.contract_hash)],['History',d.history.length+' campaigns']].map(x=>`<div class="metric"><b>${{esc(x[1])}}</b><span>${{esc(x[0])}}</span></div>`).join('');
 const alerts=[]; ps.forEach(p=>{{if(!p.online)alerts.push('OFFLINE: '+(p.url||'peer')); else (p.status.alerts||[]).forEach(a=>alerts.push(p.status.node_id+': '+a.message))}});
 document.querySelector('#alerts').innerHTML=alerts.length?`<div class="alerts"><b>SWARM ALERT</b><br>${{alerts.map(esc).join('<br>')}}</div>`:'';
 let rows=''; for(const [node,p] of Object.entries(d.participants)){{if(!p.online){{rows+=`<tr><td>${{esc(node)}}</td><td class="bad">OFFLINE</td><td colspan="5">${{esc(p.error)}}</td></tr>`;continue}} const s=p.status; for(const [wid,w] of Object.entries(s.workers||{{}})){{const cls=w.status==='running'?'ok':w.stopped_verified?'ok':'warn', joinCls=w.join_ready?'ok':'warn', pool=w.shared_population||{{}}, lineage=w.bootstrap_evidence||{{}}; rows+=`<tr><td><b>${{esc(node)}}</b><br>${{esc(wid)}}</td><td class="${{cls}}">${{esc(s.phase)}}</td><td>${{esc(w.status)}}<br>pid ${{esc(w.pid||'-')}}</td><td class="${{joinCls}}">${{w.join_ready?'verified':esc(w.join_reason||'waiting')}}<br>${{w.owns_candidate?'candidate '+esc((w.owned_candidate_indices||[]).join(',')):'no active claim'}}<br>gen ${{esc(pool.generation??'-')}} / free ${{esc(pool.free??'-')}}</td><td>height ${{esc(w.chain_height||'-')}}<br><code>genesis ${{short(lineage.genesis_hash)}}<br>population ${{short(lineage.population_block_hash)}}<br>final ${{short(w.finalized_hash)}}</code></td><td>${{pct(w.best_performance)}}</td><td><code>${{Object.entries(w.component_versions||{{}}).map(([k,v])=>esc(k)+':'+short(v)).join('<br>')}}</code></td></tr>`}}}}
 document.querySelector('#workers').innerHTML=rows||'<tr><td colspan="7">No workers reported</td></tr>';
 document.querySelector('#queue').innerHTML=(d.plan_jobs||[]).map(j=>{{const cls=j.status==='completed'?'ok':j.status==='queued'?'':j.status==='history_missing'?'bad':'warn';return `<tr><td>${{esc(j.ordinal)}}</td><td>${{esc(j.job_id)}}</td><td>${{esc(j.domain_id)}}</td><td class="${{cls}}">${{esc(j.status)}}</td><td>${{esc(j.purpose||'-')}}</td><td>${{pct(j.champion_fitness)}}<br><code>${{short(j.artifact_sha256)}}</code></td></tr>`}}).join('')||'<tr><td colspan="6">No queued jobs</td></tr>';
 document.querySelector('#history').innerHTML=d.history.map(h=>`<tr><td>${{esc(h.ordinal)}}</td><td>${{esc(h.job_id)}}</td><td>${{esc(h.domain_id)}}</td><td class="${{h.status==='completed'?'ok':'warn'}}">${{esc(h.status)}}</td><td>${{pct(h.champion_fitness)}}<br>${{short(h.champion_peer_id)}}</td><td><code>${{short(h.artifact_sha256)}}</code><br>${{esc(h.artifact_format||'')}}</td><td>${{esc(h.completed_at||'-')}}</td></tr>`).join('')||'<tr><td colspan="7">No completed campaigns yet</td></tr>';
}}
refresh();setInterval(refresh,5000);
</script></body></html>"""

    def _make_handler(self):
        supervisor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
                route = self.path.split("?", 1)[0]
                if route == "/api/status":
                    self._json(supervisor.status_payload())
                elif route == "/api/network":
                    self._json(supervisor._network_status())
                elif route == "/api/history":
                    self._json({"history": supervisor.history.campaigns()})
                elif route in {"/", "/dashboard"}:
                    body = supervisor._dashboard_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif route == "/health":
                    self._json({"status": "healthy", "node_id": supervisor.node_id})
                else:
                    self._json({"error": "not found"}, status=404)

            def _json(self, value: Any, status: int = 200) -> None:
                body = json.dumps(value, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def _acquire_lock(self) -> None:
        lock_path = self.state_dir / "supervisor.lock"
        self._lock_handle = lock_path.open("a+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another supervisor already owns {lock_path}") from exc
        self._lock_handle.seek(0)
        self._lock_handle.truncate()
        self._lock_handle.write(str(os.getpid()))
        self._lock_handle.flush()

    def run(self, *, once: bool = False) -> None:
        self._acquire_lock()
        if once:
            self.tick()
            return
        host = str(self.profile.get("listen_host", "0.0.0.0"))
        port = int(self.profile.get("listen_port", 8795))
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        server_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        server_thread.start()

        def request_shutdown(signum, frame):  # noqa: ARG001
            self._shutdown.set()

        signal.signal(signal.SIGTERM, request_shutdown)
        signal.signal(signal.SIGINT, request_shutdown)
        self.history.event(node_id=self.node_id, event="supervisor_started", detail={"pid": os.getpid()})
        try:
            while not self._shutdown.is_set():
                try:
                    self.tick()
                except Exception as exc:
                    with self._mutex:
                        self._alert("tick_failure", str(exc))
                        self._save_state()
                self._shutdown.wait(self.poll_seconds)
        finally:
            if self._httpd:
                self._httpd.shutdown()
                self._httpd.server_close()
            self.history.event(node_id=self.node_id, event="supervisor_stopped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Per-host campaign supervisor profile")
    parser.add_argument("--once", action="store_true", help="Run one reconciliation tick")
    parser.add_argument("--print-plan-hash", action="store_true", help="Validate and print plan hash")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    supervisor = CampaignSupervisor(Path(args.profile))
    if args.print_plan_hash:
        print(supervisor.plan_hash)
        return 0
    supervisor.run(once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
