#!/usr/bin/env python3
"""Remote worker for the Project 3 weekly pool HTTP API."""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from project3_weekly_worker import gpu_summary, read_progress_message, summarize_result  # noqa: E402


PYTHON_BIN = "/home/harveybc/anaconda3/envs/tensorflow/bin/python"
DEFAULT_MAX_ARTIFACT_BYTES = 250 * 1024 * 1024


def _api_post(api_url: str, path: str, payload: dict[str, Any], token: str | None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(api_url.rstrip("/") + path, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API {path} HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"API {path} unreachable: {exc}") from exc
    if not out.get("ok"):
        raise RuntimeError(f"API {path} failed: {out}")
    return out


def _api_get(api_url: str, path: str, token: str | None) -> dict[str, Any]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(api_url.rstrip("/") + path, headers=headers, method="GET")
    with urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    if not out.get("ok"):
        raise RuntimeError(f"API {path} failed: {out}")
    return out


def _write_b64(path: Path, encoded: str) -> None:
    data = base64.b64decode(encoded.encode("ascii"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _read_b64(path: Path, max_bytes: int) -> str | None:
    if not path.exists():
        return None
    size = path.stat().st_size
    if size > max_bytes:
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _tail(path: Path, lines: int = 80) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except Exception:
        return ""


def _ensure_remote_file(api_url: str, token: str | None, path_text: str) -> None:
    path = Path(path_text)
    if path.exists():
        return
    payload = _api_post(api_url, "/file", {"path": path_text}, token)
    file_payload = payload["file"]
    _write_b64(path, file_payload["bytes_b64"])


def _write_claimed_config(claim: dict[str, Any]) -> Path:
    config = claim["config"]
    config_path = Path(claim["config_path"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parent = claim.get("warm_start_model_artifact")
    if parent:
        _write_b64(Path(parent["path"]), parent["bytes_b64"])
    return config_path


def _heartbeat(
    api_url: str,
    token: str | None,
    machine_id: str,
    subjob_id: str | None,
    status: str,
    message: str,
) -> None:
    _api_post(
        api_url,
        "/heartbeat",
        {
            "machine_id": machine_id,
            "subjob_id": subjob_id,
            "status": status,
            "message": message,
            "gpu_summary": gpu_summary(),
        },
        token,
    )


def run_one(
    *,
    api_url: str,
    token: str | None,
    machine_id: str,
    python_bin: str,
    cuda_visible_devices: str | None,
    poll_sec: int,
    max_artifact_bytes: int,
) -> bool:
    claim = _api_post(api_url, "/claim", {"machine_id": machine_id, "gpu_summary": gpu_summary()}, token)
    if not claim.get("task"):
        _heartbeat(api_url, token, machine_id, None, "idle", "no pending subjobs")
        return False

    subjob_id = str(claim["subjob_id"])
    config_path = _write_claimed_config(claim)
    cfg = dict(claim["config"])
    stdout_path = Path(cfg["save_model"]).parent / "subprocess_stdout.log"
    progress_path = Path(cfg["training_progress_file"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _ensure_remote_file(api_url, token, str(cfg["input_data_file"]))
        if cfg.get("oracle_behavior_labels_file"):
            _ensure_remote_file(api_url, token, str(cfg["oracle_behavior_labels_file"]))
        if cfg.get("warm_start_model"):
            warm_start_path = Path(str(cfg["warm_start_model"]))
            if not warm_start_path.exists():
                raise FileNotFoundError(f"warm_start_model missing after claim: {warm_start_path}")

        cmd = [python_bin, "-m", "app.main", "--load_config", str(config_path)]
        env = os.environ.copy()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        _heartbeat(api_url, token, machine_id, subjob_id, "running", f"launching {' '.join(cmd)}")
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
                _heartbeat(
                    api_url,
                    token,
                    machine_id,
                    subjob_id,
                    "running",
                    read_progress_message(progress_path, stdout_path, proc.pid),
                )
                time.sleep(max(1, poll_sec))
            rc = proc.returncode
        if rc != 0:
            raise RuntimeError(f"process exited rc={rc}; log={stdout_path}\n{_tail(stdout_path)}")

        results_path = Path(cfg["results_file"])
        if not results_path.exists():
            fallback = Path(cfg["save_model"]).with_name("results.json")
            results_path = fallback if fallback.exists() else results_path
        if not results_path.exists():
            raise FileNotFoundError(f"missing results file: {cfg['results_file']}")

        result = summarize_result(results_path)
        result.update(
            {
                "subjob_id": subjob_id,
                "config_path": str(config_path),
                "run_dir": str(Path(cfg["save_model"]).parent),
                "stdout_log": str(stdout_path),
                "remote_machine_id": machine_id,
            }
        )
        results_payload = json.loads(results_path.read_text(encoding="utf-8"))
        artifacts = {
            "policy_zip_b64": _read_b64(Path(cfg["save_model"]), max_artifact_bytes),
            "results_json": results_payload,
            "stdout_tail": _tail(stdout_path),
        }
        _api_post(
            api_url,
            "/complete",
            {
                "machine_id": machine_id,
                "subjob_id": subjob_id,
                "result": result,
                "artifacts": artifacts,
                "gpu_summary": gpu_summary(),
            },
            token,
        )
        _heartbeat(api_url, token, machine_id, None, "idle", f"completed {subjob_id} score={result['score']:.6f}")
        return True
    except Exception as exc:
        _api_post(
            api_url,
            "/fail",
            {
                "machine_id": machine_id,
                "subjob_id": subjob_id,
                "error": str(exc),
                "gpu_summary": gpu_summary(),
            },
            token,
        )
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-url", required=True)
    ap.add_argument("--token")
    ap.add_argument("--token-file")
    ap.add_argument("--machine-id", default=socket.gethostname())
    ap.add_argument("--python-bin", default=PYTHON_BIN)
    ap.add_argument("--cuda-visible-devices")
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--max-subjobs", type=int, default=0)
    ap.add_argument("--idle-sleep-sec", type=int, default=60)
    ap.add_argument("--idle-cycles-before-exit", type=int, default=0)
    ap.add_argument("--max-artifact-bytes", type=int, default=DEFAULT_MAX_ARTIFACT_BYTES)
    args = ap.parse_args()
    token = args.token
    if args.token_file:
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    _api_get(args.api_url, "/health", token)

    processed = 0
    idle_cycles = 0
    while args.max_subjobs <= 0 or processed < args.max_subjobs:
        did_work = run_one(
            api_url=args.api_url,
            token=token,
            machine_id=args.machine_id,
            python_bin=args.python_bin,
            cuda_visible_devices=args.cuda_visible_devices,
            poll_sec=args.poll_sec,
            max_artifact_bytes=args.max_artifact_bytes,
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
