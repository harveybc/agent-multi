#!/usr/bin/env python3
"""Rootless GPU readiness probe + launch gate (post-outage order §3.2/§3.3).

Root cause being prevented: after the 2026-08-11 power outage dragon and
gamma booted kernel 7.0.0-29-generic whose NVIDIA module was never
installed (only 7.0.0-28 modules existed), so nvidia-smi failed on both
hosts and every GPU worker on them was dead until a human noticed.

This probe binds, in ONE typed heartbeat JSON:

* running kernel + ``modinfo -k <running> nvidia`` outcome,
* boot guard: the NEWEST INSTALLED boot kernel (``/boot/vmlinuz-*``) and
  whether ITS NVIDIA module exists — the kernel-advance-without-module
  fact is known BEFORE a planned reboot, not after;
* driver version + ``nvidia-smi`` outcome,
* the exact observed GPU UUID set vs the host's declared assignment,
* per-GPU temperature (>= threshold alerts once, recovers after cooldown),
* framework compute visibility: a SUBPROCESS torch probe per assigned
  UUID with ``CUDA_VISIBLE_DEVICES`` set to that UUID (missing torch is a
  typed fact, never a crash — and this process NEVER imports torch),
* a disk-budget dispatch gate: available bytes on the output filesystem
  vs expected artifact bytes + reserve.

Host classification is EXACTLY one of::

    GPU_UNAVAILABLE_KERNEL_MODULE_MISSING
    GPU_UNAVAILABLE_DRIVER
    GPU_UUID_MISMATCH
    GPU_FRAMEWORK_MISMATCH
    GPU_READY

never generic healthy/unhealthy text. Disk is an orthogonal dispatch
gate: ``HOST_DISK_INSUFFICIENT`` blocks dispatch on THIS host only.

Alerting REUSES the fleet incident pipeline deployed 2026-08-04
(tools/incident_emit.py -> tools/incident_ledger.py; delivery policy is
owned solely by tools/incident_router.py — no second Telegram stack).
This probe emits one ledger observation per STATE CHANGE, never per
polling cycle (its own dedup markers live under
``~/.local/state/agent-multi/gpu-readiness/``), and one recovery per
incident only after direct recovery evidence; the ledger's fingerprint
dedup + flap suppression is the second layer.

Launch-gate mode (``--gate``) refuses with a typed JSON refusal on
stdout and exit code 4 (the fleet REFUSED_* exit class) BEFORE any
training framework import, when the assigned UUID is absent, the driver
probe fails, CUDA would fall back to CPU, or (when a disk budget is
given) the output filesystem cannot hold the expected artifacts.

For module-missing states the exact remediation package string (e.g.
``linux-modules-nvidia-580-open-7.0.0-29-generic=7.0.0-29.29+1``,
family derived from the installed nvidia package set) is exposed in the
heartbeat and the incident payload. The probe NEVER attempts privileged
installation of anything.

``dispatch_gpu_binding()`` returns the exact binding dict dispatch
records must store (kernel / driver / GPU UUID / framework build /
CUDA_VISIBLE_DEVICES), in the same field style as the mechanism-ladder
heartbeat (``assigned_gpu_uuid`` / ``cuda_visible_devices`` /
``observed_gpu_uuids``).

Rootless by construction: every fact comes from unprivileged reads
(uname, modinfo, nvidia-smi, dpkg-query/apt-cache, statvfs). The only
writes are the probe's own heartbeat/state files and ledger rows.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

HEARTBEAT_SCHEMA = "agent_multi.gpu_readiness_heartbeat.v1"
STATE_SCHEMA = "agent_multi.gpu_readiness_probe_state.v1"
GATE_SCHEMA = "agent_multi.gpu_launch_gate.v1"
BINDING_SCHEMA = "agent_multi.gpu_dispatch_binding.v1"

DEFAULT_STATE_DIR = Path.home() / ".local/state/agent-multi/gpu-readiness"

# Declared per-host GPU UUID contract (order §3.1). Overridable via
# --assignments / --uuid; the embedded values are the fleet source of
# truth as of 2026-08-11.
EXPECTED_GPU_ASSIGNMENTS: dict[str, list[str]] = {
    "omega": ["GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326"],
    "dragon": ["GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9"],
    "gamma": ["GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519",
              "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"],
}

# The ONLY host classifications this tool may report (order §3.2.4).
CLASSIFICATIONS = (
    "GPU_UNAVAILABLE_KERNEL_MODULE_MISSING",
    "GPU_UNAVAILABLE_DRIVER",
    "GPU_UUID_MISMATCH",
    "GPU_FRAMEWORK_MISMATCH",
    "GPU_READY",
)

DISK_INSUFFICIENT = "HOST_DISK_INSUFFICIENT"
DISK_SUFFICIENT = "HOST_DISK_SUFFICIENT"
DISK_NOT_EVALUATED = "HOST_DISK_NOT_EVALUATED"

CLASSIFICATION_EVENT_CODES = {
    "GPU_UNAVAILABLE_KERNEL_MODULE_MISSING":
        "gpu_readiness.kernel_module_missing",
    "GPU_UNAVAILABLE_DRIVER": "gpu_readiness.driver_unavailable",
    "GPU_UUID_MISMATCH": "gpu_readiness.uuid_mismatch",
    "GPU_FRAMEWORK_MISMATCH": "gpu_readiness.framework_mismatch",
}
BOOT_GUARD_EVENT_CODE = "gpu_readiness.boot_kernel_module_missing"
TEMPERATURE_EVENT_CODE = "gpu_readiness.temperature_high"
DISK_EVENT_CODE = "gpu_readiness.disk_insufficient"

DEFAULT_TEMP_ALERT_C = 78.0       # order §10.3: >= 78 C alerts once
DEFAULT_TEMP_RECOVERY_C = 73.0    # cooldown hysteresis before recovery
DEFAULT_RESERVE_BYTES = 10 * 1024 ** 3
DEFAULT_TORCH_TIMEOUT_S = 180.0

# Exit classes follow the fleet launcher/systemd contract
# (tools/l1_fleet_launcher.py): 4 = typed configuration refusal that
# must never be blindly retried.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 4

KERNEL_VERSION_RE = re.compile(r"\d+\.\d+\.\d+-\d+-[A-Za-z0-9]+$")

# Torch probe executed in a SUBPROCESS interpreter so this process (and
# a gated launcher importing this module) never imports the framework.
FRAMEWORK_PROBE_SCRIPT = r"""
import json, os
out = {"torch_present": False, "torch_version": None,
       "torch_cuda_version": None, "cuda_available": None,
       "cuda_device_count": None,
       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
       "error": None}
try:
    import torch
    out["torch_present"] = True
    out["torch_version"] = str(torch.__version__)
    out["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["cuda_device_count"] = int(torch.cuda.device_count())
except ImportError as exc:
    out["error"] = f"torch absent: {exc}"
except Exception as exc:  # typed fact, never a crash
    out["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(cmd: list[str], *, timeout: float = 30.0,
                env: dict | None = None) -> SimpleNamespace:
    """Default shell-out runner. Tests inject a fake with the same
    signature; a missing binary or timeout becomes a typed failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
        return SimpleNamespace(returncode=proc.returncode,
                               stdout=proc.stdout, stderr=proc.stderr)
    except FileNotFoundError as exc:
        return SimpleNamespace(returncode=127, stdout="",
                               stderr=f"{cmd[0]}: not found ({exc})")
    except subprocess.TimeoutExpired:
        return SimpleNamespace(returncode=124, stdout="",
                               stderr=f"{cmd[0]}: timeout after {timeout}s")


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


# ---------------------------------------------------------------------------
# Fact collectors (each takes an injectable runner)
# ---------------------------------------------------------------------------

def running_kernel_release(runner: Callable = run_command) -> dict:
    result = runner(["uname", "-r"])
    release = (result.stdout or "").strip()
    if result.returncode != 0 or not release:
        return {"release": None,
                "error": (result.stderr or "uname failed").strip()}
    return {"release": release, "error": None}


def modinfo_nvidia(kernel: str | None,
                   runner: Callable = run_command) -> dict:
    """``modinfo -k <kernel> nvidia`` for one exact kernel version."""
    if not kernel:
        return {"kernel": None, "ok": False, "module_version": None,
                "error": "kernel release unknown"}
    result = runner(["modinfo", "-k", kernel, "nvidia"])
    if result.returncode != 0:
        return {"kernel": kernel, "ok": False, "module_version": None,
                "error": (result.stderr or result.stdout or
                          "modinfo failed").strip()[:500]}
    version = None
    for line in (result.stdout or "").splitlines():
        if line.startswith("version:"):
            version = line.split(":", 1)[1].strip()
            break
    return {"kernel": kernel, "ok": True, "module_version": version,
            "error": None}


def kernel_sort_key(release: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", release))


def list_boot_kernels(boot_dir: Path = Path("/boot")) -> list[str]:
    """Kernel releases with an installed /boot/vmlinuz-* image."""
    releases = []
    try:
        for entry in boot_dir.glob("vmlinuz-*"):
            release = entry.name[len("vmlinuz-"):]
            if KERNEL_VERSION_RE.match(release):
                releases.append(release)
    except OSError:
        return []
    return sorted(releases, key=kernel_sort_key)


def nvidia_smi_probe(runner: Callable = run_command) -> dict:
    """Driver probe + exact UUID set + per-GPU temperature."""
    result = runner([
        "nvidia-smi",
        "--query-gpu=uuid,name,driver_version,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    if result.returncode != 0:
        return {"ok": False, "driver_version": None, "gpus": [],
                "error": (result.stderr or result.stdout or
                          "nvidia-smi failed").strip()[:500]}
    gpus, driver_version = [], None
    for line in (result.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or not parts[0].startswith("GPU-"):
            continue
        driver_version = driver_version or parts[2]
        try:
            temperature = float(parts[3])
        except ValueError:
            temperature = None
        gpus.append({"uuid": parts[0], "name": parts[1],
                     "temperature_c": temperature})
    if not gpus:
        return {"ok": False, "driver_version": driver_version, "gpus": [],
                "error": "nvidia-smi returned no GPUs"}
    return {"ok": True, "driver_version": driver_version, "gpus": gpus,
            "error": None}


def expected_uuids_for_host(hostname: str,
                            assignments: dict[str, list[str]] | None = None,
                            ) -> list[str]:
    table = assignments if assignments is not None \
        else EXPECTED_GPU_ASSIGNMENTS
    short = hostname.split(".")[0].lower()
    return list(table.get(short, []))


def framework_probe(assigned_uuids: Iterable[str], *,
                    python_exe: str | None = None,
                    runner: Callable = run_command,
                    timeout: float = DEFAULT_TORCH_TIMEOUT_S,
                    base_env: dict | None = None) -> dict:
    """Subprocess torch probe once per assigned UUID with
    CUDA_VISIBLE_DEVICES pinned to that UUID. Missing torch is the typed
    status TORCH_ABSENT, never an exception. This process never imports
    torch."""
    python_exe = python_exe or sys.executable
    per_uuid: dict[str, dict] = {}
    statuses: list[str] = []
    for uuid in assigned_uuids:
        env = dict(base_env if base_env is not None else os.environ)
        env["CUDA_VISIBLE_DEVICES"] = uuid
        result = runner([python_exe, "-c", FRAMEWORK_PROBE_SCRIPT],
                        timeout=timeout, env=env)
        record: dict[str, Any]
        if result.returncode != 0:
            record = {"torch_present": None, "cuda_available": None,
                      "error": (result.stderr or "framework probe "
                                "subprocess failed").strip()[:500]}
            status = "PROBE_ERROR"
        else:
            try:
                record = json.loads((result.stdout or "").strip()
                                    .splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                record = {"torch_present": None, "cuda_available": None,
                          "error": "framework probe emitted no JSON"}
                status = "PROBE_ERROR"
            else:
                if not record.get("torch_present"):
                    status = "TORCH_ABSENT"
                elif record.get("cuda_available") and \
                        (record.get("cuda_device_count") or 0) >= 1:
                    status = "TORCH_CUDA_OK"
                else:
                    status = "TORCH_CPU_FALLBACK"
        record["status"] = status
        per_uuid[uuid] = record
        statuses.append(status)
    if not statuses:
        overall = "NOT_PROBED"
    elif "PROBE_ERROR" in statuses:
        overall = "PROBE_ERROR"
    elif "TORCH_CPU_FALLBACK" in statuses:
        overall = "TORCH_CPU_FALLBACK"
    elif "TORCH_ABSENT" in statuses:
        overall = "TORCH_ABSENT"
    else:
        overall = "TORCH_CUDA_OK"
    return {"status": overall, "per_uuid": per_uuid}


def remediation_package(target_kernel: str | None,
                        runner: Callable = run_command) -> dict:
    """Exact remediation package string for a missing NVIDIA module,
    derived from the installed linux-modules-nvidia-* family. Read-only:
    dpkg-query + apt-cache policy; NEVER installs anything."""
    facts: dict[str, Any] = {"family": None, "package": None,
                             "version": None, "pin": None, "error": None}
    if not target_kernel:
        facts["error"] = "target kernel unknown"
        return facts
    listing = runner(["dpkg-query", "-W", "-f=${Package}\t${Version}\n",
                      "linux-modules-nvidia-*"])
    installed: dict[str, str] = {}
    families: list[str] = []
    if listing.returncode == 0:
        for line in (listing.stdout or "").splitlines():
            if "\t" not in line:
                continue
            package, version = line.split("\t", 1)
            installed[package.strip()] = version.strip()
            match = re.match(
                r"linux-modules-nvidia-(.+?)-(\d+\.\d+\.\d+-\d+-[A-Za-z0-9]+)$",
                package.strip())
            if match:
                families.append(match.group(1))
    if not families:
        facts["error"] = ("no installed linux-modules-nvidia-* package "
                          "found to derive the driver family")
        return facts
    # Highest driver family wins if several are present (e.g. 580-open).
    family = sorted(set(families),
                    key=lambda name: kernel_sort_key(name) or (0,))[-1]
    package = f"linux-modules-nvidia-{family}-{target_kernel}"
    version = installed.get(package)
    if not version:
        policy = runner(["apt-cache", "policy", package])
        if policy.returncode == 0:
            for line in (policy.stdout or "").splitlines():
                line = line.strip()
                if line.startswith("Candidate:"):
                    candidate = line.split(":", 1)[1].strip()
                    if candidate and candidate != "(none)":
                        version = candidate
                    break
    facts.update({
        "family": family,
        "package": package,
        "version": version,
        "pin": f"{package}={version}" if version else package,
    })
    return facts


def disk_budget_gate(output_fs: str | None,
                     expected_artifact_bytes: int | None,
                     reserve_bytes: int = DEFAULT_RESERVE_BYTES, *,
                     statvfs_fn: Callable = os.statvfs) -> dict:
    """Dispatch disk-budget gate (order §3.3): available bytes on the
    output filesystem vs the materialized job's expected artifacts plus
    a reserve. Insufficiency blocks dispatch on THIS host only."""
    gate = {"output_fs": output_fs,
            "available_bytes": None,
            "expected_artifact_bytes": expected_artifact_bytes,
            "reserve_bytes": reserve_bytes,
            "required_bytes": None,
            "classification": DISK_NOT_EVALUATED,
            "dispatch_blocked": False,
            "error": None}
    if output_fs is None or expected_artifact_bytes is None:
        return gate
    try:
        stats = statvfs_fn(output_fs)
    except OSError as exc:
        gate.update({"classification": DISK_INSUFFICIENT,
                     "dispatch_blocked": True,
                     "error": f"statvfs failed: {exc}"})
        return gate
    available = stats.f_bavail * stats.f_frsize
    required = int(expected_artifact_bytes) + int(reserve_bytes)
    sufficient = available >= required
    gate.update({
        "available_bytes": available,
        "required_bytes": required,
        "classification": DISK_SUFFICIENT if sufficient
        else DISK_INSUFFICIENT,
        "dispatch_blocked": not sufficient,
    })
    return gate


# ---------------------------------------------------------------------------
# Classification (order §3.2.4 — exactly five states, in precedence order)
# ---------------------------------------------------------------------------

def classify(*, running_modinfo: dict, smi: dict, expected_uuids: list[str],
             framework_status: str) -> str:
    if not smi["ok"]:
        if not running_modinfo["ok"]:
            return "GPU_UNAVAILABLE_KERNEL_MODULE_MISSING"
        return "GPU_UNAVAILABLE_DRIVER"
    observed = [gpu["uuid"] for gpu in smi["gpus"]]
    if sorted(observed) != sorted(expected_uuids):
        return "GPU_UUID_MISMATCH"
    if framework_status in ("TORCH_CPU_FALLBACK", "PROBE_ERROR"):
        return "GPU_FRAMEWORK_MISMATCH"
    # TORCH_ABSENT is a tolerated typed fact (the probe interpreter may
    # not carry the framework); a worker without torch fails loudly at
    # import, it can never silently select CPU.
    return "GPU_READY"


# ---------------------------------------------------------------------------
# Heartbeat collection
# ---------------------------------------------------------------------------

def collect_heartbeat(*, hostname: str | None = None,
                      expected_uuids: list[str] | None = None,
                      assignments: dict[str, list[str]] | None = None,
                      runner: Callable = run_command,
                      boot_lister: Callable[[], list[str]] | None = None,
                      statvfs_fn: Callable = os.statvfs,
                      python_exe: str | None = None,
                      output_fs: str | None = None,
                      expected_artifact_bytes: int | None = None,
                      reserve_bytes: int = DEFAULT_RESERVE_BYTES,
                      temp_alert_c: float = DEFAULT_TEMP_ALERT_C,
                      temp_recovery_c: float = DEFAULT_TEMP_RECOVERY_C,
                      torch_timeout: float = DEFAULT_TORCH_TIMEOUT_S,
                      probe_framework: bool = True,
                      base_env: dict | None = None) -> dict:
    hostname = hostname or socket.gethostname()
    if expected_uuids is None:
        expected_uuids = expected_uuids_for_host(hostname, assignments)

    kernel = running_kernel_release(runner)
    running_modinfo = modinfo_nvidia(kernel["release"], runner)

    boot_kernels = (boot_lister() if boot_lister is not None
                    else list_boot_kernels())
    newest_kernel = boot_kernels[-1] if boot_kernels else None
    if newest_kernel and newest_kernel == kernel["release"]:
        newest_modinfo = dict(running_modinfo)
    else:
        newest_modinfo = modinfo_nvidia(newest_kernel, runner)
    kernel_advance_without_module = bool(newest_kernel) and \
        not newest_modinfo["ok"]

    smi = nvidia_smi_probe(runner)

    observed_uuids = [gpu["uuid"] for gpu in smi["gpus"]]
    missing = sorted(set(expected_uuids) - set(observed_uuids))
    unexpected = sorted(set(observed_uuids) - set(expected_uuids))
    exact_match = smi["ok"] and not missing and not unexpected

    # Framework probing is only meaningful once driver + UUIDs hold;
    # earlier-stage failures already classify the host.
    if probe_framework and exact_match and expected_uuids:
        framework = framework_probe(
            expected_uuids, python_exe=python_exe, runner=runner,
            timeout=torch_timeout, base_env=base_env)
    else:
        framework = {"status": "NOT_PROBED", "per_uuid": {}}

    classification = classify(
        running_modinfo=running_modinfo, smi=smi,
        expected_uuids=expected_uuids,
        framework_status=framework["status"])

    remediation = None
    if not running_modinfo["ok"] or kernel_advance_without_module:
        target = (kernel["release"] if not running_modinfo["ok"]
                  else newest_kernel)
        remediation = remediation_package(target, runner)

    temperatures = [gpu["temperature_c"] for gpu in smi["gpus"]
                    if gpu["temperature_c"] is not None]
    alerting_uuids = sorted(
        gpu["uuid"] for gpu in smi["gpus"]
        if gpu["temperature_c"] is not None
        and gpu["temperature_c"] >= temp_alert_c)

    disk = disk_budget_gate(output_fs, expected_artifact_bytes,
                            reserve_bytes, statvfs_fn=statvfs_fn)

    return {
        "schema": HEARTBEAT_SCHEMA,
        "hostname": hostname,
        "generated_utc": utcnow_iso(),
        "running_kernel": {
            "release": kernel["release"],
            "error": kernel["error"],
            "modinfo_nvidia": running_modinfo,
        },
        "boot_guard": {
            "installed_boot_kernels": boot_kernels,
            "newest_installed_kernel": newest_kernel,
            "modinfo_nvidia": newest_modinfo,
            "kernel_advance_without_module": kernel_advance_without_module,
        },
        "driver": {
            "nvidia_smi_ok": smi["ok"],
            "driver_version": smi["driver_version"],
            "error": smi["error"],
        },
        "gpus": {
            "expected_uuids": list(expected_uuids),
            "observed_uuids": observed_uuids,
            "missing_uuids": missing,
            "unexpected_uuids": unexpected,
            "exact_match": exact_match,
            "per_gpu": smi["gpus"],
        },
        "framework": framework,
        "temperature": {
            "alert_threshold_c": temp_alert_c,
            "recovery_threshold_c": temp_recovery_c,
            "max_temperature_c": max(temperatures) if temperatures else None,
            "alerting_uuids": alerting_uuids,
        },
        "disk": disk,
        "classification": classification,
        "remediation_package": remediation,
    }


# ---------------------------------------------------------------------------
# Dispatch binding (embedded in dispatch records by launchers)
# ---------------------------------------------------------------------------

def dispatch_gpu_binding(assigned_gpu_uuid: str | None = None, *,
                         hostname: str | None = None,
                         runner: Callable = run_command,
                         python_exe: str | None = None,
                         torch_timeout: float = DEFAULT_TORCH_TIMEOUT_S,
                         probe_framework: bool = True,
                         base_env: dict | None = None) -> dict:
    """Exactly the binding facts every dispatch record must store
    (order §3.2 last paragraph): kernel, driver, GPU UUID, framework
    build, CUDA-visible devices. Field style matches the
    mechanism-ladder heartbeat."""
    env = base_env if base_env is not None else os.environ
    kernel = running_kernel_release(runner)
    smi = nvidia_smi_probe(runner)
    if probe_framework and assigned_gpu_uuid:
        framework = framework_probe(
            [assigned_gpu_uuid], python_exe=python_exe, runner=runner,
            timeout=torch_timeout, base_env=base_env)
        build = framework["per_uuid"].get(assigned_gpu_uuid, {})
        framework_build = {
            "status": build.get("status", "NOT_PROBED"),
            "torch_version": build.get("torch_version"),
            "torch_cuda_version": build.get("torch_cuda_version"),
            "cuda_available": build.get("cuda_available"),
        }
    else:
        framework_build = {"status": "NOT_PROBED", "torch_version": None,
                           "torch_cuda_version": None,
                           "cuda_available": None}
    return {
        "schema": BINDING_SCHEMA,
        "hostname": hostname or socket.gethostname(),
        "running_kernel": kernel["release"],
        "driver_version": smi["driver_version"],
        "assigned_gpu_uuid": assigned_gpu_uuid,
        "observed_gpu_uuids": [gpu["uuid"] for gpu in smi["gpus"]],
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "framework_build": framework_build,
    }


# ---------------------------------------------------------------------------
# Incident emission: one ledger observation per STATE CHANGE, one
# recovery only on direct recovery evidence. The fleet ledger/router
# (deployed 2026-08-04) owns delivery, reminders and flap suppression.
# ---------------------------------------------------------------------------

class LedgerEmitter:
    """Adapter onto tools/incident_emit.py (the fleet shim). Imported
    lazily so tests never touch the real ledger."""

    SOURCE = "gpu_readiness_probe"

    def __init__(self, machine: str, front: str = "front4"):
        self.machine = machine
        self.front = front

    def observe(self, event_code: str, severity: str, summary: str,
                payload: dict, affected_object: str = "-") -> bool:
        import incident_emit
        return incident_emit.observe_incident(
            source=self.SOURCE, event_code=event_code, severity=severity,
            summary=summary, front=self.front, machine=self.machine,
            affected_object=affected_object, payload=payload)

    def recover(self, event_code: str, evidence: dict,
                affected_object: str = "-") -> bool:
        import incident_emit
        return incident_emit.recover_incident(
            source=self.SOURCE, event_code=event_code, evidence=evidence,
            front=self.front, machine=self.machine,
            affected_object=affected_object)


def default_state() -> dict:
    return {
        "schema": STATE_SCHEMA,
        "active_classification_code": None,
        "boot_guard_active": False,
        "temperature_active": {},
        "disk_active": False,
        "last_classification": None,
        "updated_utc": None,
    }


def load_state(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_state()
    if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
        return default_state()
    state = default_state()
    state.update(value)
    return state


def process_incidents(heartbeat: dict, state: dict, emitter) -> dict:
    """Compare this heartbeat against the dedup state and emit at most
    one observation per state CHANGE plus at most one recovery per
    recovered condition. Returns {"emitted": [...], "recovered": [...],
    "state": <new state>}."""
    emitted: list[dict] = []
    recovered: list[dict] = []
    state = dict(state)
    state["temperature_active"] = dict(state.get("temperature_active") or {})
    hostname = heartbeat["hostname"]

    # --- five-state GPU classification -------------------------------
    classification = heartbeat["classification"]
    new_code = CLASSIFICATION_EVENT_CODES.get(classification)
    previous_code = state.get("active_classification_code")
    if new_code != previous_code:
        transition_ok = True
        if previous_code is not None:
            evidence = {
                "summary": f"{hostname}: previous GPU state cleared, "
                           f"now {classification}",
                "classification": classification,
                "nvidia_smi_ok": heartbeat["driver"]["nvidia_smi_ok"],
                "driver_version": heartbeat["driver"]["driver_version"],
                "observed_gpu_uuids": heartbeat["gpus"]["observed_uuids"],
                "framework_status": heartbeat["framework"]["status"],
            }
            if emitter.recover(previous_code, evidence):
                recovered.append({"event_code": previous_code})
            else:
                transition_ok = False
        if new_code is not None:
            remediation = heartbeat.get("remediation_package") or {}
            payload = {
                "classification": classification,
                "running_kernel":
                    heartbeat["running_kernel"]["release"],
                "modinfo_error":
                    heartbeat["running_kernel"]["modinfo_nvidia"]["error"],
                "driver_error": heartbeat["driver"]["error"],
                "expected_gpu_uuids": heartbeat["gpus"]["expected_uuids"],
                "observed_gpu_uuids": heartbeat["gpus"]["observed_uuids"],
                "missing_gpu_uuids": heartbeat["gpus"]["missing_uuids"],
                "unexpected_gpu_uuids":
                    heartbeat["gpus"]["unexpected_uuids"],
                "framework_status": heartbeat["framework"]["status"],
                "remediation_package": remediation.get("pin"),
            }
            summary = f"{classification} on {hostname}"
            if remediation.get("pin"):
                summary += (f" — remediation: sudo apt-get install "
                            f"{remediation['pin']} (owner action; the "
                            f"probe never installs)")
            if emitter.observe(new_code, "P1", summary, payload):
                emitted.append({"event_code": new_code,
                                "classification": classification})
            else:
                transition_ok = False
        # A failed emission keeps the old marker so the next poll
        # retries; the ledger's fingerprint dedup guarantees a retry can
        # never mint a duplicate notification.
        if transition_ok:
            state["active_classification_code"] = new_code

    # --- boot guard: kernel advance without module -------------------
    guard = heartbeat["boot_guard"]
    guard_active = bool(guard["kernel_advance_without_module"])
    if guard_active and not state.get("boot_guard_active"):
        remediation = heartbeat.get("remediation_package") or {}
        summary = (f"boot kernel {guard['newest_installed_kernel']} on "
                   f"{hostname} has NO nvidia module — next reboot loses "
                   f"the GPU")
        if remediation.get("pin"):
            summary += (f"; remediation: sudo apt-get install "
                        f"{remediation['pin']} (owner action)")
        payload = {
            "newest_installed_kernel": guard["newest_installed_kernel"],
            "running_kernel": heartbeat["running_kernel"]["release"],
            "modinfo_error": guard["modinfo_nvidia"]["error"],
            "remediation_package": remediation.get("pin"),
        }
        if emitter.observe(BOOT_GUARD_EVENT_CODE, "P1", summary, payload):
            emitted.append({"event_code": BOOT_GUARD_EVENT_CODE})
            state["boot_guard_active"] = True
    elif not guard_active and state.get("boot_guard_active"):
        evidence = {
            "summary": f"{hostname}: newest boot kernel "
                       f"{guard['newest_installed_kernel']} now has its "
                       f"nvidia module",
            "newest_installed_kernel": guard["newest_installed_kernel"],
            "module_version": guard["modinfo_nvidia"]["module_version"],
        }
        if emitter.recover(BOOT_GUARD_EVENT_CODE, evidence):
            recovered.append({"event_code": BOOT_GUARD_EVENT_CODE})
            state["boot_guard_active"] = False

    # --- per-GPU temperature (order §10.3, hysteresis cooldown) ------
    alert_c = heartbeat["temperature"]["alert_threshold_c"]
    recovery_c = heartbeat["temperature"]["recovery_threshold_c"]
    for gpu in heartbeat["gpus"]["per_gpu"]:
        uuid, temperature = gpu["uuid"], gpu["temperature_c"]
        if temperature is None:
            continue
        active = bool(state["temperature_active"].get(uuid))
        if temperature >= alert_c and not active:
            summary = (f"GPU temperature {temperature:.0f} C >= "
                       f"{alert_c:.0f} C on {hostname} ({uuid})")
            payload = {"gpu_uuid": uuid, "temperature_c": temperature,
                       "alert_threshold_c": alert_c,
                       "recovery_threshold_c": recovery_c}
            if emitter.observe(TEMPERATURE_EVENT_CODE, "P2", summary,
                               payload, affected_object=uuid):
                emitted.append({"event_code": TEMPERATURE_EVENT_CODE,
                                "gpu_uuid": uuid})
                state["temperature_active"][uuid] = True
        elif active and temperature <= recovery_c:
            evidence = {
                "summary": f"GPU cooled to {temperature:.0f} C on "
                           f"{hostname} ({uuid})",
                "gpu_uuid": uuid, "temperature_c": temperature,
                "recovery_threshold_c": recovery_c,
            }
            if emitter.recover(TEMPERATURE_EVENT_CODE, evidence,
                               affected_object=uuid):
                recovered.append({"event_code": TEMPERATURE_EVENT_CODE,
                                  "gpu_uuid": uuid})
                state["temperature_active"][uuid] = False

    # --- disk-budget dispatch gate -----------------------------------
    disk = heartbeat["disk"]
    disk_blocked = disk["classification"] == DISK_INSUFFICIENT
    if disk_blocked and not state.get("disk_active"):
        summary = (f"{DISK_INSUFFICIENT} on {hostname}: "
                   f"{disk['available_bytes']} bytes available on "
                   f"{disk['output_fs']} < required "
                   f"{disk['required_bytes']} — dispatch blocked on this "
                   f"host only")
        if emitter.observe(DISK_EVENT_CODE, "P2", summary, dict(disk),
                           affected_object=str(disk["output_fs"])):
            emitted.append({"event_code": DISK_EVENT_CODE})
            state["disk_active"] = True
    elif not disk_blocked and state.get("disk_active") \
            and disk["classification"] == DISK_SUFFICIENT:
        evidence = {
            "summary": f"{hostname}: {disk['available_bytes']} bytes "
                       f"available on {disk['output_fs']} >= required "
                       f"{disk['required_bytes']}",
            "available_bytes": disk["available_bytes"],
            "required_bytes": disk["required_bytes"],
        }
        if emitter.recover(DISK_EVENT_CODE, evidence,
                           affected_object=str(disk["output_fs"])):
            recovered.append({"event_code": DISK_EVENT_CODE})
            state["disk_active"] = False

    state["last_classification"] = classification
    state["updated_utc"] = utcnow_iso()
    return {"emitted": emitted, "recovered": recovered, "state": state}


# ---------------------------------------------------------------------------
# Launch gate (order §3.2.3): typed refusal BEFORE any framework import
# ---------------------------------------------------------------------------

def launch_gate(heartbeat: dict,
                assigned_gpu_uuid: str | None = None) -> tuple[dict, int]:
    """Refuse worker launch when the assigned UUID is absent, the driver
    probe fails, CUDA would fall back to CPU, or the disk budget cannot
    hold the expected artifacts. Never imports the training framework —
    compute visibility comes from the heartbeat's subprocess probe."""
    classification = heartbeat["classification"]
    blocking: list[str] = []
    reasons: list[str] = []

    if classification != "GPU_READY":
        blocking.append(classification)
        if classification == "GPU_UNAVAILABLE_KERNEL_MODULE_MISSING":
            reasons.append(
                f"running kernel "
                f"{heartbeat['running_kernel']['release']} has no nvidia "
                f"module")
        elif classification == "GPU_UNAVAILABLE_DRIVER":
            reasons.append(f"driver probe failed: "
                           f"{heartbeat['driver']['error']}")
        elif classification == "GPU_UUID_MISMATCH":
            reasons.append(
                f"expected {heartbeat['gpus']['expected_uuids']}, "
                f"observed {heartbeat['gpus']['observed_uuids']}")
        else:
            reasons.append(f"framework compute visibility: "
                           f"{heartbeat['framework']['status']}")
    elif assigned_gpu_uuid is not None and \
            assigned_gpu_uuid not in heartbeat["gpus"]["observed_uuids"]:
        blocking.append("GPU_UUID_MISMATCH")
        reasons.append(f"assigned GPU {assigned_gpu_uuid} is not in the "
                       f"observed UUID set")

    if heartbeat["disk"]["classification"] == DISK_INSUFFICIENT:
        blocking.append(DISK_INSUFFICIENT)
        reasons.append(
            f"{heartbeat['disk']['available_bytes']} bytes available on "
            f"{heartbeat['disk']['output_fs']} < required "
            f"{heartbeat['disk']['required_bytes']}")

    remediation = heartbeat.get("remediation_package") or {}
    payload = {
        "schema": GATE_SCHEMA,
        "gate": "REFUSED" if blocking else "PASS",
        "outcome": ("REFUSED_GPU_UNBOUND" if blocking else "GATE_PASS"),
        "hostname": heartbeat["hostname"],
        "classification": classification,
        "blocking": blocking,
        "reasons": reasons,
        "assigned_gpu_uuid": assigned_gpu_uuid,
        "running_kernel": heartbeat["running_kernel"]["release"],
        "driver_version": heartbeat["driver"]["driver_version"],
        "observed_gpu_uuids": heartbeat["gpus"]["observed_uuids"],
        "framework_status": heartbeat["framework"]["status"],
        "remediation_package": remediation.get("pin"),
        "generated_utc": heartbeat["generated_utc"],
    }
    return payload, (EXIT_REFUSED if blocking else EXIT_OK)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rootless GPU readiness probe, launch gate and "
                    "dispatch-binding reporter (order §3.2/§3.3)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gate", action="store_true",
                      help="launch-gate mode: typed refusal on stdout + "
                           "exit 4 when launch must not proceed")
    mode.add_argument("--binding", action="store_true",
                      help="print the dispatch-record GPU binding dict")
    parser.add_argument("--host", default=None,
                        help="hostname override (default: gethostname)")
    parser.add_argument("--uuid", action="append", default=None,
                        dest="uuids", metavar="GPU-UUID",
                        help="expected/assigned GPU UUID (repeatable; "
                             "overrides the embedded contract)")
    parser.add_argument("--assignments", type=Path, default=None,
                        help="JSON file {host: [gpu-uuid, ...]} overriding "
                             "the embedded per-host contract")
    parser.add_argument("--output-fs", default=None,
                        help="output filesystem path for the disk-budget "
                             "dispatch gate")
    parser.add_argument("--expected-artifact-bytes", type=int, default=None,
                        help="materialized job's expected artifact bytes")
    parser.add_argument("--reserve-bytes", type=int,
                        default=DEFAULT_RESERVE_BYTES,
                        help="disk reserve on top of expected artifacts "
                             "(default 10 GiB)")
    parser.add_argument("--temp-alert-c", type=float,
                        default=DEFAULT_TEMP_ALERT_C)
    parser.add_argument("--temp-recovery-c", type=float,
                        default=DEFAULT_TEMP_RECOVERY_C)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR,
                        help="dedup-state + heartbeat directory")
    parser.add_argument("--python", dest="python_exe", default=None,
                        help="interpreter for the subprocess framework "
                             "probe (default: this interpreter)")
    parser.add_argument("--torch-timeout", type=float,
                        default=DEFAULT_TORCH_TIMEOUT_S)
    parser.add_argument("--no-framework-probe", action="store_true",
                        help="skip the subprocess torch probe")
    parser.add_argument("--no-incidents", action="store_true",
                        help="never touch the incident ledger (gate mode "
                             "never does)")
    parser.add_argument("--front", default="front4",
                        help="incident ledger front (default front4)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hostname = args.host or socket.gethostname()

    assignments = None
    if args.assignments is not None:
        try:
            assignments = json.loads(
                args.assignments.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"error": f"unreadable --assignments: {exc}"}))
            return EXIT_ERROR

    if args.binding:
        assigned = args.uuids[0] if args.uuids else None
        binding = dispatch_gpu_binding(
            assigned, hostname=hostname, python_exe=args.python_exe,
            torch_timeout=args.torch_timeout,
            probe_framework=not args.no_framework_probe)
        print(json.dumps(binding, indent=2, sort_keys=True))
        return EXIT_OK

    # Gate mode: --uuid names the worker's ASSIGNED GPU. The host is
    # still classified against its full declared contract (a gamma
    # worker gating one of two GPUs must not read the second expected
    # GPU as "unexpected"); the assigned UUID's presence is enforced
    # separately by launch_gate(). --uuid only becomes the expected set
    # when the host has no declared contract at all.
    expected_override = args.uuids
    if args.gate and args.uuids and \
            expected_uuids_for_host(hostname, assignments):
        expected_override = None

    heartbeat = collect_heartbeat(
        hostname=hostname,
        expected_uuids=expected_override,
        assignments=assignments,
        python_exe=args.python_exe,
        output_fs=args.output_fs,
        expected_artifact_bytes=args.expected_artifact_bytes,
        reserve_bytes=args.reserve_bytes,
        temp_alert_c=args.temp_alert_c,
        temp_recovery_c=args.temp_recovery_c,
        torch_timeout=args.torch_timeout,
        probe_framework=not args.no_framework_probe,
    )

    if args.gate:
        # The gate is a pure pre-launch check: it must stay fast, quiet
        # and side-effect-free; the timer heartbeat owns alerting.
        assigned = args.uuids[0] if args.uuids else None
        payload, code = launch_gate(heartbeat, assigned)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return code

    state_path = args.state_dir / "state.json"
    heartbeat_path = args.state_dir / "heartbeat.json"
    if args.no_incidents:
        incidents = {"emitted": [], "recovered": [],
                     "state": load_state(state_path)}
    else:
        emitter = LedgerEmitter(hostname, front=args.front)
        incidents = process_incidents(heartbeat, load_state(state_path),
                                      emitter)
        atomic_write_json(state_path, incidents["state"])
    heartbeat["incidents"] = {"emitted": incidents["emitted"],
                              "recovered": incidents["recovered"]}
    atomic_write_json(heartbeat_path, heartbeat)
    print(json.dumps(heartbeat, indent=2, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
