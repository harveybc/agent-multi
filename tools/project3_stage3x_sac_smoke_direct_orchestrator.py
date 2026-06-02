#!/usr/bin/env python3
"""Direct dispatcher for Project 3 Stage 3X SAC smoke configs.

This intentionally does *not* use ``tools/seed_sweep.py`` because seed_sweep
rewrites run directories. The Stage 3X smoke manifest already contains one
locked config per (contract, seed) and exact evidence paths; this dispatcher
preserves those paths and launches ``python -m app.main --load_config ...``.

It can run as a daemon or a single dispatch/status pass. It never touches
Stage C and refuses to launch unless the financial-data acceptance packet says
the manifest was accepted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_ROOT = REPO_ROOT / "experiments" / "stage3x_sac_smoke_plan"
MANIFEST_JSON = PLAN_ROOT / "stage3x_sac_smoke_plan_manifest.json"
STATE_JSON = PLAN_ROOT / "stage3x_sac_smoke_dispatch_state.json"
STATUS_MD = PLAN_ROOT / "stage3x_sac_smoke_dispatch_status.md"
LOCK_FILE = PLAN_ROOT / "stage3x_sac_smoke_dispatch.lock"
ACCEPTANCE_JSON = (
    REPO_ROOT.parent
    / "financial-data"
    / "experiments"
    / "stage3x_sac_smoke_request"
    / "stage3x_sac_smoke_plan_acceptance.json"
)
LAUNCH_DIR = Path.home() / "p4_launch" / "stage3x_sac_smoke"
PYTHON_BIN = "/home/harveybc/anaconda3/envs/tensorflow/bin/python"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", "62024"]
POLL_SEC = 120
DONE_GRACE_SEC = 90
NO_TRADE_ABORT_PROGRESS_PCT = 20.0

HOSTS: dict[str, dict[str, Any]] = {
    "dragon": {
        "addr": "harveybc@100.110.215.85",
        "kind": "remote",
        "max_concurrent": 2,
    },
    "gamma": {
        "addr": "harveybc@100.107.204.49",
        "kind": "remote",
        "max_concurrent": 1,
    },
    "omega": {
        "addr": None,
        "kind": "local",
        "max_concurrent": 1,
    },
}


def configure_paths(*, plan_root: Path | None = None, acceptance_json: Path | None = None,
                    launch_dir: Path | None = None) -> None:
    global PLAN_ROOT, MANIFEST_JSON, STATE_JSON, STATUS_MD, LOCK_FILE, ACCEPTANCE_JSON, LAUNCH_DIR
    if plan_root is not None:
        PLAN_ROOT = plan_root
        MANIFEST_JSON = PLAN_ROOT / "stage3x_sac_smoke_plan_manifest.json"
        STATE_JSON = PLAN_ROOT / "stage3x_sac_smoke_dispatch_state.json"
        STATUS_MD = PLAN_ROOT / "stage3x_sac_smoke_dispatch_status.md"
        LOCK_FILE = PLAN_ROOT / "stage3x_sac_smoke_dispatch.lock"
    if acceptance_json is not None:
        ACCEPTANCE_JSON = acceptance_json
    if launch_dir is not None:
        LAUNCH_DIR = launch_dir


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_cmd(cmd: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def ssh_cmd(host: str, remote_cmd: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    meta = HOSTS[host]
    assert meta["kind"] == "remote"
    # Use OS-level ``timeout`` as well as subprocess timeout; OpenSSH can
    # otherwise survive the Python timeout in some network-stall cases.
    return run_cmd(
        ["timeout", f"{max(1, int(timeout))}s", "ssh", *SSH_OPTS, str(meta["addr"]), remote_cmd],
        timeout=max(2, int(timeout) + 2),
    )


def validate_accepted() -> None:
    acceptance = load_json(ACCEPTANCE_JSON)
    if acceptance.get("accepted") is not True:
        raise SystemExit(f"financial-data acceptance is not true: {ACCEPTANCE_JSON}")
    if acceptance.get("stage_c_access") != "DENIED":
        raise SystemExit("acceptance packet does not deny Stage C")
    manifest = load_json(MANIFEST_JSON)
    if acceptance.get("manifest_file") != str(MANIFEST_JSON):
        raise SystemExit("acceptance packet points at a different manifest")
    if manifest.get("stage_c_access") != "DENIED":
        raise SystemExit("manifest does not deny Stage C")
    if manifest.get("training_launched") is not False:
        raise SystemExit("manifest says training_launched is not false")


def config_to_task(entry: dict[str, Any], idx: int) -> dict[str, Any]:
    cfg = load_json(Path(entry["config_file"]))
    host_order = list(HOSTS.keys())
    preferred_hosts = host_order[(idx - 1) % len(host_order) :] + host_order[: (idx - 1) % len(host_order)]
    return {
        "id": f"stage3x_smoke_{idx:02d}_{entry['contract_id']}__s{entry['seed']}__{entry['cost_scenario']}",
        "status": "pending",
        "contract_id": entry["contract_id"],
        "seed": int(entry["seed"]),
        "cost_scenario": entry["cost_scenario"],
        "asset": entry.get("asset"),
        "timeframe": entry.get("timeframe"),
        "config_file": entry["config_file"],
        "run_dir": entry["run_dir"],
        "progress_file": entry["progress_file"],
        "results_file": cfg.get("results_file"),
        "expected_evidence_file": entry.get("expected_evidence_file"),
        "preferred_hosts": preferred_hosts,
        "assigned_host": None,
        "launched_pid": None,
        "launched_utc": None,
        "finished_utc": None,
        "launch_log": None,
        "notes": "",
    }


def init_state(*, force: bool = False) -> dict[str, Any]:
    validate_accepted()
    if STATE_JSON.exists() and not force:
        return load_json(STATE_JSON)
    manifest = load_json(MANIFEST_JSON)
    tasks = [config_to_task(entry, idx) for idx, entry in enumerate(manifest["configs"], 1)]
    state = {
        "schema_version": "project3_stage3x_sac_smoke_dispatch_state_v1",
        "generated_at": utc_now(),
        "updated_at": utc_now(),
        "stage_c_access": "DENIED",
        "training_launched": False,
        "manifest_file": str(MANIFEST_JSON),
        "acceptance_file": str(ACCEPTANCE_JSON),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    write_json_atomic(STATE_JSON, state)
    write_status_md(state)
    return state


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    state["training_launched"] = any(t.get("status") in {"running", "done"} for t in state["tasks"])
    write_json_atomic(STATE_JSON, state)
    write_status_md(state)


def host_busy_count(host: str) -> int:
    pattern = "python -m app.main --load_config"
    if HOSTS[host]["kind"] == "local":
        proc = run_cmd(["pgrep", "-af", pattern], timeout=10)
        if proc.returncode != 0:
            return 0
        return len([line for line in proc.stdout.splitlines() if is_app_process_line(line)])
    proc = ssh_cmd(host, f"pgrep -af '{pattern}' || true", timeout=15)
    if proc.returncode != 0:
        return HOSTS[host]["max_concurrent"]
    return len([line for line in proc.stdout.splitlines() if is_app_process_line(line)])


def host_gpu_healthy(host: str) -> bool:
    """Return True when the host can see NVIDIA GPUs through NVML.

    A driver/library mismatch after machine updates can leave Python jobs
    spinning on CPU while the dispatcher believes the GPU worker is healthy.
    Fail closed here: an unhealthy host gets zero launch capacity until it is
    rebooted or its driver stack is repaired.
    """
    cmd = "nvidia-smi -L >/dev/null"
    if HOSTS[host]["kind"] == "local":
        return run_cmd(["bash", "-lc", cmd], timeout=10).returncode == 0
    return ssh_cmd(host, cmd, timeout=15).returncode == 0


def is_app_process_line(line: str) -> bool:
    return (
        "python -m app.main --load_config" in line
        and "pgrep" not in line
        and "grep" not in line
        and "ssh " not in line
        and "bash -lc" not in line
    )


def task_running(task: dict[str, Any]) -> bool:
    host = task.get("assigned_host")
    if not host:
        return False
    pattern = str(task["config_file"])
    if HOSTS[host]["kind"] == "local":
        proc = run_cmd(["pgrep", "-af", pattern], timeout=10)
        return proc.returncode == 0 and any(is_app_process_line(line) for line in proc.stdout.splitlines())
    proc = ssh_cmd(host, f"pgrep -af {json.dumps(pattern)} || true", timeout=15)
    return any(is_app_process_line(line) for line in proc.stdout.splitlines())


def terminate_task(task: dict[str, Any]) -> None:
    host = task.get("assigned_host")
    if not host:
        return
    pattern = str(task["config_file"])
    if HOSTS[host]["kind"] == "local":
        run_cmd(["pkill", "-f", pattern], timeout=10)
        return
    ssh_cmd(host, f"pkill -f {json.dumps(pattern)} || true", timeout=15)


def completion_paths(task: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    if task.get("results_file"):
        paths.append(str(task["results_file"]))
    run_dir = str(task.get("run_dir") or "").rstrip("/")
    if run_dir:
        paths.append(f"{run_dir}/results.json")
        paths.append(f"{run_dir}/summary.json")
    if task.get("expected_evidence_file"):
        paths.append(str(task["expected_evidence_file"]))
    seen: set[str] = set()
    return [path for path in paths if path and not (path in seen or seen.add(path))]


def local_completion_artifacts_exist(task: dict[str, Any]) -> bool:
    results = task.get("results_file") or f"{str(task.get('run_dir') or '').rstrip('/')}/results.json"
    evidence = task.get("expected_evidence_file")
    return bool(results and evidence and Path(str(results)).exists() and Path(str(evidence)).exists())


def is_micro_nsga_task(task: dict[str, Any]) -> bool:
    """Micro-NSGA cells are tiny optimizer probes; no-trade is a scored outcome."""
    try:
        cfg = load_json(Path(str(task["config_file"])))
    except Exception:
        return "micro_nsga" in str(task.get("contract_id", ""))
    return (
        cfg.get("micro_nsga_generation") is not None
        or bool(cfg.get("micro_nsga_individual_id"))
        or "micro_nsga" in str(task.get("contract_id", ""))
    )


def path_exists(host: str, path: str) -> bool:
    if HOSTS[host]["kind"] == "local":
        return Path(path).exists()
    proc = ssh_cmd(host, f"test -f {json.dumps(path)}", timeout=15)
    return proc.returncode == 0


def read_json_best_effort(host: str, path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    if HOSTS[host]["kind"] == "local":
        p = Path(path)
        if not p.exists():
            return {}
        try:
            return load_json(p)
        except Exception:
            return {}
    proc = ssh_cmd(host, f"cat {json.dumps(path)} 2>/dev/null || true", timeout=15)
    try:
        return json.loads(proc.stdout) if proc.stdout.strip() else {}
    except Exception:
        return {}


def read_text_best_effort(host: str, path: str | None, *, tail_lines: int = 240) -> str:
    if not path:
        return ""
    if HOSTS[host]["kind"] == "local":
        p = Path(path)
        if not p.exists():
            return ""
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""
        return "\n".join(lines[-tail_lines:])
    proc = ssh_cmd(host, f"tail -n {int(tail_lines)} {json.dumps(path)} 2>/dev/null || true", timeout=15)
    return proc.stdout


def launch_log_path(task: dict[str, Any]) -> str | None:
    value = task.get("launch_log")
    host = task.get("assigned_host")
    if not value:
        return None
    prefix = f"{host}:"
    if host and str(value).startswith(prefix):
        return str(value)[len(prefix) :]
    return str(value)


def parse_latest_log_metrics(text: str) -> dict[str, Any]:
    train_matches = re.findall(r"TRAIN\s+trades=\s*(\d+).*?profit=([+-]?\d+(?:\.\d+)?)%", text)
    val_matches = re.findall(r"VAL\s+trades=\s*(\d+).*?profit=([+-]?\d+(?:\.\d+)?)%", text)
    metrics: dict[str, Any] = {}
    if train_matches:
        trades, profit = train_matches[-1]
        metrics["train_trades"] = int(trades)
        metrics["train_profit_pct"] = float(profit)
    if val_matches:
        trades, profit = val_matches[-1]
        metrics["val_trades"] = int(trades)
        metrics["val_profit_pct"] = float(profit)
    return metrics


def sync_back(task: dict[str, Any]) -> None:
    host = task.get("assigned_host")
    if not host or HOSTS[host]["kind"] == "local":
        return
    run_dir = str(task["run_dir"]).rstrip("/")
    local_dir = Path(run_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{HOSTS[host]['addr']}:{run_dir}/"
    cmd = ["rsync", "-az", "-e", "ssh -p 62024 -o BatchMode=yes", remote, f"{run_dir}/"]
    run_cmd(cmd, timeout=300)


def sync_config_to_remote(host: str, config_file: str) -> bool:
    if HOSTS[host]["kind"] == "local":
        return True
    config_path = Path(config_file)
    ssh_cmd(host, f"mkdir -p {shlex.quote(str(config_path.parent))}", timeout=20)
    remote = f"{HOSTS[host]['addr']}:{config_file}"
    proc = run_cmd(
        [
            "rsync",
            "-az",
            "-e",
            "ssh -p 62024 -o BatchMode=yes",
            str(config_path),
            remote,
        ],
        timeout=120,
    )
    return proc.returncode == 0


def launch_log_name(host: str, task_id: str) -> str:
    """Return a filesystem-safe, bounded launch log name.

    Micro-NSGA descendants keep the full ancestry in task ids for auditability.
    By G07 that can exceed the per-path-component limit, so launch logs use a
    short readable prefix plus a deterministic digest while the full task id
    remains in the dispatch state and manifest.
    """
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:48].strip("._-")
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:16]
    return f"{host}_{safe_prefix}_{digest}.out"


def launch_task(host: str, task: dict[str, Any]) -> int | None:
    LAUNCH_DIR.mkdir(parents=True, exist_ok=True)
    log_name = launch_log_name(host, str(task["id"]))
    if HOSTS[host]["kind"] == "local":
        out_path = LAUNCH_DIR / log_name
        cmd = [
            PYTHON_BIN,
            "-m",
            "app.main",
            "--load_config",
            task["config_file"],
            "--quiet_mode",
        ]
        with out_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        task["launch_log"] = str(out_path)
        return proc.pid

    if not sync_config_to_remote(host, str(task["config_file"])):
        task["notes"] = "remote config sync failed before launch"
        return None

    remote_launch_dir = str(LAUNCH_DIR)
    remote_log = f"{remote_launch_dir}/{log_name}"
    remote_pid = f"{remote_log}.pid"
    inner = (
        f"mkdir -p {shlex.quote(remote_launch_dir)} && "
        "cd /home/harveybc/Documents/GitHub/agent-multi && "
        f"nohup {shlex.quote(PYTHON_BIN)} -m app.main "
        f"--load_config {shlex.quote(str(task['config_file']))} --quiet_mode "
        f"> {shlex.quote(remote_log)} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(remote_pid)}"
    )
    # Launch detached. Do not wait on this SSH process; some remote shells keep
    # the session open while the child starts importing CUDA/TensorFlow.
    subprocess.Popen(
        ["ssh", "-n", "-f", *SSH_OPTS, str(HOSTS[host]["addr"]), f"bash -lc {shlex.quote(inner)}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    task["launch_log"] = f"{host}:{remote_log}"
    time.sleep(1.0)
    pid_proc = ssh_cmd(host, f"cat {shlex.quote(remote_pid)} 2>/dev/null || true", timeout=10)
    try:
        return int(pid_proc.stdout.strip())
    except ValueError:
        return 0


def dispatch_once(state: dict[str, Any]) -> dict[str, int]:
    validate_accepted()
    capacities = {
        host: (
            max(0, int(meta["max_concurrent"]) - host_busy_count(host))
            if host_gpu_healthy(host)
            else 0
        )
        for host, meta in HOSTS.items()
    }
    launched = 0
    for task in state["tasks"]:
        if task.get("status") != "pending":
            continue
        host = next((h for h in task.get("preferred_hosts", HOSTS.keys()) if capacities.get(h, 0) > 0), None)
        if not host:
            continue
        pid = launch_task(host, task)
        if pid is None:
            task["status"] = "failed"
            task["finished_utc"] = utc_now()
            task["notes"] = task.get("notes") or "launch failed"
        else:
            task["status"] = "running"
            task["assigned_host"] = host
            task["launched_pid"] = pid
            task["launched_utc"] = utc_now()
            capacities[host] -= 1
            launched += 1
    save_state(state)
    return {"launched": launched}


def refresh_state(state: dict[str, Any]) -> dict[str, int]:
    changed = 0
    for task in state["tasks"]:
        if task.get("status") == "pending" and local_completion_artifacts_exist(task):
            task["status"] = "done"
            task["assigned_host"] = task.get("assigned_host") or "existing_artifact"
            task["finished_utc"] = task.get("finished_utc") or utc_now()
            task["notes"] = (
                (task.get("notes") + "; ") if task.get("notes") else ""
            ) + "reused_existing_results_and_evidence"
            changed += 1
            continue
        if task.get("status") != "running":
            continue
        launched_at = parse_utc(task.get("launched_utc"))
        age = (dt.datetime.now(dt.timezone.utc) - launched_at).total_seconds() if launched_at else 9999
        host = str(task.get("assigned_host"))
        is_running = task_running(task)
        if not is_running and age >= DONE_GRACE_SEC:
            if any(path_exists(host, path) for path in completion_paths(task)):
                task["status"] = "done"
                task["finished_utc"] = utc_now()
                sync_back(task)
            else:
                task["status"] = "failed"
                task["finished_utc"] = utc_now()
                task["notes"] = "process exited but summary file is missing"
            changed += 1
            continue
        prog = task_progress(task)
        progress_pct = prog.get("progress_pct") or 0
        if (
            is_running
            and not is_micro_nsga_task(task)
            and
            progress_pct >= NO_TRADE_ABORT_PROGRESS_PCT
            and prog.get("no_trade_anomaly") is True
        ):
            terminate_task(task)
            task["status"] = "failed"
            task["finished_utc"] = utc_now()
            task["notes"] = (
                f"aborted_no_trade_at_{progress_pct}%: "
                "zero train/validation trades after smoke threshold"
            )
            changed += 1
            continue
    if changed:
        save_state(state)
    else:
        write_status_md(state)
    return {"changed": changed}


def task_progress(task: dict[str, Any]) -> dict[str, Any]:
    host = str(task.get("assigned_host") or "omega")
    if host not in HOSTS:
        host = "omega"
    progress = read_json_best_effort(host, task.get("progress_file")) if task.get("status") == "running" else {}
    log_metrics = parse_latest_log_metrics(
        read_text_best_effort(host, launch_log_path(task))
    ) if task.get("status") == "running" else {}
    summary: dict[str, Any] = {}
    if task.get("status") in {"done", "failed"}:
        for path in completion_paths(task):
            summary = read_json_best_effort(host, path)
            if summary:
                break
    progress_trades = progress.get("trades_total") or progress.get("trades")
    log_trades = log_metrics.get("val_trades") or log_metrics.get("train_trades")
    has_log_trade_counts = (
        log_metrics.get("train_trades") is not None
        or log_metrics.get("val_trades") is not None
    )
    if has_log_trade_counts:
        trades_total = (
            f"train={log_metrics.get('train_trades', '')},"
            f"val={log_metrics.get('val_trades', '')}"
        )
    else:
        trades_total = progress_trades or summary.get("trades_total") or summary.get("trades")
    no_trade_anomaly = progress.get("no_trade_anomaly")
    if (progress.get("progress_pct") or progress.get("progress_percent") or 0) >= NO_TRADE_ABORT_PROGRESS_PCT:
        if log_trades:
            no_trade_anomaly = False
        elif has_log_trade_counts:
            no_trade_anomaly = True
    return {
        "progress_pct": progress.get("progress_pct") or progress.get("progress_percent"),
        "current_step": progress.get("current_step") or progress.get("num_timesteps"),
        "total_timesteps": progress.get("total_timesteps"),
        "trades_total": trades_total,
        "total_return": log_metrics.get("val_profit_pct", progress.get("total_return") or summary.get("total_return")),
        "no_trade_anomaly": no_trade_anomaly,
    }


def write_status_md(state: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for task in state["tasks"]:
        counts[task.get("status", "unknown")] = counts.get(task.get("status", "unknown"), 0) + 1
    has_active_tasks = any(task.get("status") in {"pending", "running"} for task in state["tasks"])
    next_poll_utc = state.get("next_poll_utc") if has_active_tasks else ""
    lines = [
        "# Stage 3X SAC Smoke Dispatch Status",
        "",
        f"Updated UTC: `{utc_now()}`",
        "",
        f"- Stage C access: `{state['stage_c_access']}`",
        f"- Total tasks: `{len(state['tasks'])}`",
        f"- Pending: `{counts.get('pending', 0)}`",
        f"- Running: `{counts.get('running', 0)}`",
        f"- Done: `{counts.get('done', 0)}`",
        f"- Failed: `{counts.get('failed', 0)}`",
        f"- Next supervisor poll UTC: `{next_poll_utc or ''}`",
        "",
        "| status | host | contract | seed | cost | progress | trades | return | anomaly |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for task in state["tasks"]:
        prog = task_progress(task)
        lines.append(
            f"| `{task.get('status')}` | `{task.get('assigned_host') or ''}` | "
            f"`{task.get('contract_id')}` | {task.get('seed')} | "
            f"`{task.get('cost_scenario')}` | "
            f"{prog.get('progress_pct', '')} | {prog.get('trades_total', '')} | "
            f"{prog.get('total_return', '')} | {prog.get('no_trade_anomaly', '')} |"
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(state: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for task in state["tasks"]:
        counts[task.get("status", "unknown")] = counts.get(task.get("status", "unknown"), 0) + 1
    print(
        json.dumps(
            {
                "stage_c_access": state["stage_c_access"],
                "task_count": len(state["tasks"]),
                "counts": counts,
                "state_file": str(STATE_JSON),
                "status_md": str(STATUS_MD),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true", help="initialize dispatch state from the accepted manifest")
    parser.add_argument("--force-init", action="store_true", help="rebuild dispatch state even if it already exists")
    parser.add_argument("--once", action="store_true", help="refresh, dispatch one pass, then exit")
    parser.add_argument("--status", action="store_true", help="refresh state/status and exit")
    parser.add_argument("--daemon", action="store_true", help="run until queue drains")
    parser.add_argument("--poll", type=int, default=POLL_SEC)
    parser.add_argument("--plan-root", type=Path, default=PLAN_ROOT)
    parser.add_argument("--acceptance-json", type=Path, default=ACCEPTANCE_JSON)
    parser.add_argument("--launch-dir", type=Path, default=LAUNCH_DIR)
    args = parser.parse_args()
    configure_paths(
        plan_root=args.plan_root,
        acceptance_json=args.acceptance_json,
        launch_dir=args.launch_dir,
    )

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)

        if args.init or args.force_init or not STATE_JSON.exists():
            state = init_state(force=args.force_init)
        else:
            state = load_json(STATE_JSON)

        refresh_state(state)
        state = load_json(STATE_JSON)

        if args.status:
            print_summary(state)
            return 0

        if args.once:
            dispatch_once(state)
            print_summary(load_json(STATE_JSON))
            return 0

        if args.daemon:
            while True:
                state = load_json(STATE_JSON)
                refresh_state(state)
                state = load_json(STATE_JSON)
                dispatch_once(state)
                state = load_json(STATE_JSON)
                print_summary(state)
                remaining = [task for task in state["tasks"] if task.get("status") in {"pending", "running"}]
                if not remaining:
                    return 0
                sleep_sec = max(10, int(args.poll))
                next_poll = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=sleep_sec)
                state["next_poll_utc"] = next_poll.strftime("%Y-%m-%dT%H:%M:%SZ")
                save_state(state)
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
                time.sleep(sleep_sec)
                fcntl.flock(lock_fh, fcntl.LOCK_EX)

        print_summary(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
