#!/usr/bin/env python3
"""Worker loop for Project 3 weekly walk-forward pool subjobs."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from project3_weekly_materialize import materialize  # noqa: E402
from project3_weekly_pool import (  # noqa: E402
    claim_subjob,
    complete_subjob,
    connect,
    fail_subjob,
    heartbeat,
    init_db,
)


PYTHON_BIN = "/home/harveybc/anaconda3/envs/tensorflow/bin/python"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_result(results_path: Path) -> dict[str, Any]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    splits = payload.get("splits") or {}
    validation = splits.get("validation") or {}
    test = splits.get("test") or {}
    train = splits.get("train") or {}
    train_tail = splits.get("train_tail") or {}
    val_return = _safe_float(validation.get("total_return"))
    test_return = _safe_float(test.get("total_return"))
    val_sharpe = _safe_float(validation.get("sharpe_ratio"))
    test_sharpe = _safe_float(test.get("sharpe_ratio"))
    train_tail_trades = int(_safe_float(train_tail.get("trades_total")))
    validation_trades = int(_safe_float(validation.get("trades_total")))
    test_trades = int(_safe_float(test.get("trades_total")))
    trade_gate_passed = train_tail_trades >= 1 and validation_trades >= 1
    test_trade_gate_passed = test_trades >= 1
    train_validation_composite_score = 0.50 * _safe_float(train_tail.get("total_return")) + 0.50 * val_return
    raw_score = train_validation_composite_score
    score = raw_score if trade_gate_passed else raw_score - 1_000_000.0
    return {
        "score": score,
        "raw_score": raw_score,
        "selection_basis": "train_tail_validation_composite_with_trade_gate",
        "train_validation_composite_score": train_validation_composite_score,
        "trade_gate_passed": trade_gate_passed,
        "test_trade_gate_passed": test_trade_gate_passed,
        "validation_total_return": val_return,
        "test_total_return": test_return,
        "train_total_return": _safe_float(train.get("total_return")),
        "train_tail_total_return": _safe_float(train_tail.get("total_return")),
        "validation_sharpe": val_sharpe,
        "test_sharpe": test_sharpe,
        "train_tail_trades_total": train_tail_trades,
        "validation_trades_total": validation_trades,
        "test_trades_total": test_trades,
        "results_file": str(results_path),
        "return_trace_evidence_file": payload.get("return_trace_evidence_file"),
    }


def read_progress_message(progress_path: Path, stdout_path: Path, pid: int) -> str:
    if not progress_path.exists():
        return f"pid={pid} log={stdout_path}"
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return f"pid={pid} log={stdout_path}"
    pct = progress.get("progress_pct", progress.get("progress_percent"))
    step = progress.get("num_timesteps", progress.get("current_step"))
    total = progress.get("total_timesteps")
    ret = progress.get("total_return")
    trades = progress.get("trades_total_cumulative", progress.get("trades_total"))
    pieces = [f"pid={pid}"]
    if step is not None and total is not None:
        pieces.append(f"steps={step}/{total}")
    if pct is not None:
        pieces.append(f"pct={float(pct):.1f}%")
    if ret is not None:
        pieces.append(f"return={float(ret):+.5f}")
    if trades is not None:
        pieces.append(f"trades={trades}")
    pieces.append(f"log={stdout_path}")
    return " ".join(pieces)


def gpu_summary() -> str | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return "; ".join(line.strip() for line in proc.stdout.splitlines() if line.strip())


def run_one(
    *,
    db_path: str,
    machine_id: str,
    output_root: str,
    python_bin: str,
    cuda_visible_devices: str | None,
    poll_sec: int,
) -> bool:
    conn = connect(db_path)
    init_db(conn)
    task = claim_subjob(conn, machine_id)
    if task is None:
        heartbeat(conn, machine_id, None, "idle", "no pending subjobs", gpu_summary())
        return False

    subjob_id = task["external_id"]
    config_path = materialize(db_path, subjob_id, output_root)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path(cfg["save_model"]).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "subprocess_stdout.log"
    progress_path = Path(cfg["training_progress_file"])

    cmd = [python_bin, "-m", "app.main", "--load_config", str(config_path)]
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    heartbeat(conn, machine_id, subjob_id, "running", f"launching {' '.join(cmd)}", gpu_summary())
    with stdout_path.open("w", encoding="utf-8") as stdout:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while proc.poll() is None:
            heartbeat(
                conn,
                machine_id,
                subjob_id,
                "running",
                read_progress_message(progress_path, stdout_path, proc.pid),
                gpu_summary(),
            )
            time.sleep(max(1, poll_sec))
        rc = proc.returncode

    if rc != 0:
        tail = ""
        try:
            tail = "\n".join(stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        except Exception:
            pass
        fail_subjob(conn, subjob_id, f"process exited rc={rc}; log={stdout_path}\n{tail}")
        heartbeat(conn, machine_id, None, "idle", f"failed {subjob_id}", gpu_summary())
        return True

    results_path = Path(cfg["results_file"])
    if not results_path.exists():
        fallback = Path(cfg["save_model"]).with_name("results.json")
        results_path = fallback if fallback.exists() else results_path
    if not results_path.exists():
        fail_subjob(conn, subjob_id, f"missing results file: {cfg['results_file']}")
        heartbeat(conn, machine_id, None, "idle", f"missing results {subjob_id}", gpu_summary())
        return True

    result = summarize_result(results_path)
    result.update(
        {
            "subjob_id": subjob_id,
            "config_path": str(config_path),
            "run_dir": str(run_dir),
            "stdout_log": str(stdout_path),
        }
    )
    complete_subjob(conn, subjob_id, result)
    heartbeat(conn, machine_id, None, "idle", f"completed {subjob_id} score={result['score']:.6f}", gpu_summary())
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--machine-id", default=socket.gethostname())
    ap.add_argument("--output-root", default=str(REPO_ROOT / "experiments" / "weekly_walkforward_pool"))
    ap.add_argument("--python-bin", default=PYTHON_BIN)
    ap.add_argument("--cuda-visible-devices")
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--max-subjobs", type=int, default=1)
    ap.add_argument(
        "--idle-sleep-sec",
        type=int,
        default=0,
        help="Sleep and keep polling when no subjob is available. Default exits on idle.",
    )
    ap.add_argument(
        "--idle-cycles-before-exit",
        type=int,
        default=1,
        help="Idle polling cycles before exit. Use 0 to run forever.",
    )
    args = ap.parse_args()

    processed = 0
    idle_cycles = 0
    while args.max_subjobs <= 0 or processed < args.max_subjobs:
        did_work = run_one(
            db_path=args.db,
            machine_id=args.machine_id,
            output_root=args.output_root,
            python_bin=args.python_bin,
            cuda_visible_devices=args.cuda_visible_devices,
            poll_sec=args.poll_sec,
        )
        if not did_work:
            idle_cycles += 1
            if args.idle_sleep_sec > 0 and (
                args.idle_cycles_before_exit <= 0
                or idle_cycles < args.idle_cycles_before_exit
            ):
                time.sleep(args.idle_sleep_sec)
                continue
            break
        idle_cycles = 0
        processed += 1
    print(json.dumps({"processed": processed}, indent=2))


if __name__ == "__main__":
    main()
