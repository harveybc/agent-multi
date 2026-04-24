"""auto_orchestrator.py — long-running work dispatcher for the 3-host RL cluster.

Reads tools/work_queue.json, polls Dragon/Gamma/Omega for idle state, and
auto-launches pending tasks on whichever host becomes idle first (respecting
per-host concurrency limits and per-task host preferences). Marks queue state
atomically on each transition. When the queue is drained, runs the P5 eval +
rank pipeline and exits.

USAGE:
    nohup python tools/auto_orchestrator.py \
        > ~/p4_launch/orchestrator.log 2>&1 < /dev/null & disown

To pause (without killing in-flight jobs), create file tools/.orchestrator_pause.
To resume, delete it.
To stop the orchestrator (in-flight jobs keep running): kill the process.

QUEUE FILE FORMAT (tools/work_queue.json):
    {
      "tasks": [
        {
          "id": "unique-string",
          "config": "examples/config/p4_ppo_eth_1h.json",
          "seeds": [0, 1, 2],
          "run_tag": "p4iter7",
          "preferred_hosts": ["dragon","omega"],   # first match with capacity
          "requires_gpu": true,
          "priority": 10,                          # higher = picked first
          "status": "pending",                     # pending|running|done|failed
          "assigned_host": null,
          "launched_utc": null,
          "finished_utc": null,
          "notes": ""
        }
      ],
      "finalize": {"p5_eval": true, "p5_rank": true}
    }

Only fields {id, config, seeds, run_tag} are required for a pending task.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = REPO_ROOT / "tools" / "work_queue.json"
PAUSE_FLAG = REPO_ROOT / "tools" / ".orchestrator_pause"
LAUNCH_DIR = Path.home() / "p4_launch"
PY_LOCAL = "/home/harveybc/anaconda3/envs/tensorflow/bin/python"
SSH_OPTS = ["-o", "BatchMode=yes", "-p", "62024"]
POLL_SEC = 120

HOSTS = {
    "dragon": {
        "addr": "harveybc@100.110.215.85",
        "kind": "remote",
        "has_gpu": True,
        "max_concurrent": 2,
    },
    "gamma": {
        "addr": "harveybc@100.107.204.49",
        "kind": "remote",
        "has_gpu": True,  # has GPU but EURUSD configs target CPU; still busy-detect works
        "max_concurrent": 1,
    },
    "omega": {
        "addr": None,
        "kind": "local",
        "has_gpu": True,
        "max_concurrent": 1,
    },
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{_utc_now()}] {msg}", flush=True)


def _load_queue() -> Dict[str, Any]:
    if not QUEUE_PATH.exists():
        return {"tasks": [], "finalize": {"p5_eval": True, "p5_rank": True}}
    with QUEUE_PATH.open() as fh:
        return json.load(fh)


def _save_queue(q: Dict[str, Any]) -> None:
    tmp = QUEUE_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(q, fh, indent=2)
    tmp.replace(QUEUE_PATH)


def _host_busy_count(host: str) -> int:
    """How many seed_sweep.py processes are running on host."""
    meta = HOSTS[host]
    if meta["kind"] == "local":
        try:
            out = subprocess.check_output(
                ["pgrep", "-af", "tools/seed_sweep.py"], text=True
            )
        except subprocess.CalledProcessError:
            return 0
        lines = [l for l in out.splitlines() if "grep" not in l and "ssh " not in l]
        return len(lines)
    else:
        cmd = ["ssh"] + SSH_OPTS + [meta["addr"], "pgrep -af tools/seed_sweep.py || true"]
        try:
            out = subprocess.check_output(cmd, text=True, timeout=20)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return -1  # treat as unknown/busy
        lines = [l for l in out.splitlines() if l.strip() and "grep" not in l]
        return len(lines)


def _launch_remote(host: str, task: Dict[str, Any]) -> Optional[int]:
    meta = HOSTS[host]
    seeds = " ".join(str(s) for s in task["seeds"])
    out_name = f"auto_{host}_{task['id']}.out"
    inner = (
        f"cd ~/Documents/GitHub/agent-multi && "
        f"python tools/seed_sweep.py "
        f"--config {shlex.quote(task['config'])} "
        f"--seeds {seeds} --run_tag {shlex.quote(task['run_tag'])}"
    )
    remote_cmd = (
        f"mkdir -p ~/p4_launch && "
        f"nohup bash -ic {shlex.quote(inner)} "
        f"> ~/p4_launch/{out_name} 2>&1 < /dev/null & "
        f"echo PID=$!; disown -a"
    )
    cmd = ["ssh"] + SSH_OPTS + [meta["addr"], remote_cmd]
    log(f"  SSH launch on {host}: {inner}")
    try:
        out = subprocess.check_output(cmd, text=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log(f"  launch FAILED: {e}")
        return None
    for line in out.splitlines():
        if line.startswith("PID="):
            try:
                return int(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _launch_local(task: Dict[str, Any]) -> Optional[int]:
    LAUNCH_DIR.mkdir(parents=True, exist_ok=True)
    out_name = LAUNCH_DIR / f"auto_omega_{task['id']}.out"
    seeds = [str(s) for s in task["seeds"]]
    cmd = [
        "nohup", PY_LOCAL,
        "tools/seed_sweep.py",
        "--config", task["config"],
        "--seeds", *seeds,
        "--run_tag", task["run_tag"],
    ]
    log(f"  local launch on omega: {' '.join(cmd)}")
    with out_name.open("w") as fh:
        p = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    return p.pid


def _launch(host: str, task: Dict[str, Any]) -> Optional[int]:
    meta = HOSTS[host]
    if meta["kind"] == "local":
        return _launch_local(task)
    return _launch_remote(host, task)


def _pick_host(task: Dict[str, Any], capacity: Dict[str, int]) -> Optional[str]:
    prefs = task.get("preferred_hosts") or list(HOSTS.keys())
    for h in prefs:
        if h not in HOSTS:
            continue
        if task.get("requires_gpu") and not HOSTS[h]["has_gpu"]:
            continue
        if capacity.get(h, 0) > 0:
            return h
    return None


def _run_finalize() -> None:
    """Run p5_eval_holdout + p5_rank locally after queue drained."""
    LAUNCH_DIR.mkdir(parents=True, exist_ok=True)
    log("Running p5_eval_holdout.py (--skip-if-exists)...")
    with (LAUNCH_DIR / "auto_p5_eval.out").open("w") as fh:
        subprocess.run(
            [PY_LOCAL, "tools/p5_eval_holdout.py", "--skip-if-exists"],
            cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT, check=False,
        )
    log("Running p5_rank.py...")
    with (LAUNCH_DIR / "auto_p5_rank.out").open("w") as fh:
        subprocess.run(
            [PY_LOCAL, "tools/p5_rank.py", "--top", "3", "--min_seeds", "3"],
            cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT, check=False,
        )
    rank_md = REPO_ROOT / "logs" / "partIII" / "p5_rank.md"
    if rank_md.exists():
        log(f"P5 rank written: {rank_md}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=POLL_SEC, help="polling interval seconds")
    ap.add_argument("--once", action="store_true", help="run one dispatch pass then exit")
    ap.add_argument("--no-finalize", action="store_true")
    args = ap.parse_args()

    log(f"auto_orchestrator starting; queue={QUEUE_PATH}; poll={args.poll}s")

    while True:
        if PAUSE_FLAG.exists():
            log("PAUSED (delete tools/.orchestrator_pause to resume)")
            if args.once:
                return 0
            time.sleep(args.poll)
            continue

        q = _load_queue()
        tasks: List[Dict[str, Any]] = q.get("tasks", [])

        # 1. Snapshot host capacity
        capacity: Dict[str, int] = {}
        for h, meta in HOSTS.items():
            busy = _host_busy_count(h)
            if busy < 0:
                log(f"  {h}: busy-check UNKNOWN (treating as full)")
                capacity[h] = 0
            else:
                capacity[h] = max(0, int(meta["max_concurrent"]) - busy)
                log(f"  {h}: busy={busy} free={capacity[h]}")

        # 2. Try to dispatch pending tasks in priority order
        pending = sorted(
            [t for t in tasks if t.get("status", "pending") == "pending"],
            key=lambda t: -int(t.get("priority", 0)),
        )
        dispatched = 0
        for task in pending:
            host = _pick_host(task, capacity)
            if host is None:
                continue
            log(f"  dispatching {task['id']} -> {host}")
            pid = _launch(host, task)
            if pid is None:
                task["status"] = "failed"
                task["notes"] = "launch failed"
            else:
                task["status"] = "running"
                task["assigned_host"] = host
                task["launched_utc"] = _utc_now()
                task["launched_pid"] = pid
                capacity[host] -= 1
                dispatched += 1
            _save_queue(q)

        # 3. Poll running tasks — if the host is now free (no seed_sweep at all)
        #    AND our task was the one we launched, mark it done (best-effort).
        #    Since we can't inspect remote PIDs easily, we use heuristic:
        #    a running task whose host has capacity == max_concurrent is considered done.
        for task in tasks:
            if task.get("status") != "running":
                continue
            h = task.get("assigned_host")
            if h is None:
                continue
            busy = _host_busy_count(h)
            if busy == 0:
                task["status"] = "done"
                task["finished_utc"] = _utc_now()
                log(f"  task {task['id']} on {h} -> done (host idle)")
                _save_queue(q)

        # 4. Exit conditions: queue empty AND all hosts idle.
        remaining = [t for t in tasks if t.get("status") in ("pending", "running")]
        all_idle = all(capacity[h] == HOSTS[h]["max_concurrent"] for h in HOSTS)
        if not remaining and all_idle:
            log(f"queue drained ({len(tasks)} tasks total); all hosts idle")
            if not args.no_finalize and q.get("finalize", {}).get("p5_eval"):
                _run_finalize()
            log("auto_orchestrator exiting cleanly")
            return 0
        if not remaining:
            log("queue empty but hosts still busy with externally-launched jobs; waiting")

        if args.once:
            log(f"--once: dispatched={dispatched}, remaining={len(remaining)}, exiting")
            return 0

        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
