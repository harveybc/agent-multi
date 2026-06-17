#!/usr/bin/env python3
"""HTTP API for the Project 3 weekly walk-forward SQLite job pool.

The API keeps the SQLite database centralized on the coordinator machine while
remote workers claim jobs, send heartbeats, and report results over HTTP. Claim
operations reuse the existing SQLite BEGIN IMMEDIATE lease path, so multiple
machines cannot receive the same subjob.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from project3_weekly_materialize import materialize  # noqa: E402
from project3_weekly_pool import (  # noqa: E402
    claim_subjob,
    complete_subjob,
    connect,
    fail_subjob,
    heartbeat,
    init_db,
    status as pool_status,
)


DEFAULT_MAX_FILE_BYTES = 600 * 1024 * 1024
ALLOWED_FILE_ROOTS = (
    Path("/home/harveybc/Documents/GitHub/financial-data/experiments/stage_a_screening/inputs"),
    Path("/home/harveybc/Documents/GitHub/financial-data/experiments/stage3x_event_context"),
    Path("/home/harveybc/Documents/GitHub/financial-data/experiments/oracle_behavior_pretraining"),
    Path("/home/harveybc/Documents/GitHub/agent-multi/experiments/weekly_walkforward_pool"),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_b64(path: Path, *, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> dict[str, Any]:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"refusing to serve {path}: {size} bytes exceeds {max_bytes}")
    data = path.read_bytes()
    return {
        "path": str(path),
        "size": size,
        "sha256": _sha256_bytes(data),
        "bytes_b64": base64.b64encode(data).decode("ascii"),
    }


def _write_b64(path: Path, encoded: str) -> dict[str, Any]:
    data = base64.b64decode(encoded.encode("ascii"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "size": len(data), "sha256": _sha256_bytes(data)}


def _safe_requested_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    for root in ALLOWED_FILE_ROOTS:
        try:
            path.relative_to(root.resolve())
            return path
        except ValueError:
            continue
    raise PermissionError(f"path is outside allowed API roots: {raw_path}")


def _load_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _subjob_claim_owner(conn, subjob_id: str) -> tuple[str | None, str | None]:
    row = conn.execute(
        "SELECT status, claimed_by FROM subjobs WHERE external_id=?",
        (subjob_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown subjob_id: {subjob_id}")
    return row["status"], row["claimed_by"]


def _assert_claimed_by(conn, subjob_id: str, machine_id: str) -> None:
    subjob_status, claimed_by = _subjob_claim_owner(conn, subjob_id)
    if subjob_status != "running" or claimed_by != machine_id:
        raise PermissionError(
            f"subjob {subjob_id} is status={subjob_status!r} claimed_by={claimed_by!r}; "
            f"{machine_id!r} cannot mutate it"
        )


def make_handler(
    *,
    db_path: Path,
    output_root: Path,
    token: str | None,
    max_file_bytes: int,
):
    class Project3WeeklyPoolAPI(BaseHTTPRequestHandler):
        server_version = "Project3WeeklyPoolAPI/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def _authorized(self) -> bool:
            if not token:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {token}"

        def _send(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_error_json(self, exc: Exception, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
            self._send({"ok": False, "error": str(exc), "error_type": exc.__class__.__name__}, status)

        def _conn(self):
            conn = connect(db_path)
            init_db(conn)
            return conn

        def _handle_get(self, parsed_path: str) -> None:
            if parsed_path == "/health":
                self._send({"ok": True, "service": "project3_weekly_pool_api"})
                return
            if parsed_path == "/status":
                conn = self._conn()
                self._send({"ok": True, "status": pool_status(conn)})
                return
            self._send({"ok": False, "error": f"unknown path: {parsed_path}"}, HTTPStatus.NOT_FOUND)

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                self._send({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                return
            try:
                self._handle_get(urlparse(self.path).path)
            except Exception as exc:
                self._send_error_json(exc)

        def _handle_claim(self, body: dict[str, Any]) -> None:
            machine_id = str(body.get("machine_id") or "").strip()
            if not machine_id:
                raise ValueError("machine_id is required")
            conn = self._conn()
            task = claim_subjob(conn, machine_id)
            if task is None:
                self._send({"ok": True, "task": None})
                return
            subjob_id = task["external_id"]
            config_path = materialize(db_path, subjob_id, output_root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            parent_artifact = None
            warm_start_model = config.get("warm_start_model")
            if warm_start_model:
                parent_path = Path(str(warm_start_model))
                if parent_path.exists():
                    parent_artifact = _read_b64(parent_path, max_bytes=max_file_bytes)
            self._send(
                {
                    "ok": True,
                    "task": task,
                    "subjob_id": subjob_id,
                    "config_path": str(config_path),
                    "config": config,
                    "warm_start_model_artifact": parent_artifact,
                }
            )

        def _handle_heartbeat(self, body: dict[str, Any]) -> None:
            machine_id = str(body.get("machine_id") or "").strip()
            subjob_id = body.get("subjob_id")
            if not machine_id:
                raise ValueError("machine_id is required")
            conn = self._conn()
            heartbeat(
                conn,
                machine_id,
                str(subjob_id) if subjob_id else None,
                str(body.get("status") or "running"),
                body.get("message"),
                body.get("gpu_summary"),
            )
            self._send({"ok": True})

        def _handle_file(self, body: dict[str, Any]) -> None:
            raw_path = str(body.get("path") or "")
            if not raw_path:
                raise ValueError("path is required")
            path = _safe_requested_path(raw_path)
            if not path.exists():
                raise FileNotFoundError(str(path))
            self._send({"ok": True, "file": _read_b64(path, max_bytes=max_file_bytes)})

        def _handle_complete(self, body: dict[str, Any]) -> None:
            machine_id = str(body.get("machine_id") or "").strip()
            subjob_id = str(body.get("subjob_id") or "").strip()
            result = dict(body.get("result") or {})
            if not machine_id or not subjob_id:
                raise ValueError("machine_id and subjob_id are required")
            conn = self._conn()
            _assert_claimed_by(conn, subjob_id, machine_id)
            row = conn.execute(
                "SELECT run_dir FROM subjobs WHERE external_id=?",
                (subjob_id,),
            ).fetchone()
            run_dir = Path(row["run_dir"]) if row and row["run_dir"] else output_root / "runs" / subjob_id
            artifacts = body.get("artifacts") or {}
            written: dict[str, Any] = {}
            if artifacts.get("policy_zip_b64"):
                written["policy_zip"] = _write_b64(run_dir / "policy.zip", artifacts["policy_zip_b64"])
            if artifacts.get("results_json"):
                results_path = run_dir / "results.json"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                results_path.write_text(json.dumps(artifacts["results_json"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
                written["results_json"] = {"path": str(results_path), "size": results_path.stat().st_size}
            if artifacts.get("stdout_tail"):
                stdout_path = run_dir / f"subprocess_stdout.{machine_id}.tail.log"
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_path.write_text(str(artifacts["stdout_tail"]), encoding="utf-8")
                written["stdout_tail"] = {"path": str(stdout_path), "size": stdout_path.stat().st_size}
            result.setdefault("remote_machine_id", machine_id)
            result["central_artifacts"] = written
            complete_subjob(conn, subjob_id, result)
            heartbeat(conn, machine_id, None, "idle", f"completed {subjob_id}", body.get("gpu_summary"))
            self._send({"ok": True, "written_artifacts": written})

        def _handle_fail(self, body: dict[str, Any]) -> None:
            machine_id = str(body.get("machine_id") or "").strip()
            subjob_id = str(body.get("subjob_id") or "").strip()
            error = str(body.get("error") or "remote worker failed")
            if not machine_id or not subjob_id:
                raise ValueError("machine_id and subjob_id are required")
            conn = self._conn()
            _assert_claimed_by(conn, subjob_id, machine_id)
            fail_subjob(conn, subjob_id, error)
            heartbeat(conn, machine_id, None, "idle", f"failed {subjob_id}", body.get("gpu_summary"))
            self._send({"ok": True})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                self._send({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                return
            try:
                body = _load_json_body(self)
                path = urlparse(self.path).path
                if path == "/claim":
                    self._handle_claim(body)
                elif path == "/heartbeat":
                    self._handle_heartbeat(body)
                elif path == "/file":
                    self._handle_file(body)
                elif path == "/complete":
                    self._handle_complete(body)
                elif path == "/fail":
                    self._handle_fail(body)
                else:
                    self._send({"ok": False, "error": f"unknown path: {path}"}, HTTPStatus.NOT_FOUND)
            except PermissionError as exc:
                self._send_error_json(exc, HTTPStatus.FORBIDDEN)
            except Exception as exc:
                self._send_error_json(exc)

    return Project3WeeklyPoolAPI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--token")
    ap.add_argument("--token-file")
    ap.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    args = ap.parse_args()

    token = args.token
    if args.token_file:
        token_path = Path(args.token_file)
        token = token_path.read_text(encoding="utf-8").strip()
    handler = make_handler(
        db_path=Path(args.db),
        output_root=Path(args.output_root),
        token=token,
        max_file_bytes=args.max_file_bytes,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        json.dumps(
            {
                "event": "project3_weekly_pool_api_started",
                "host": args.host,
                "port": args.port,
                "db": args.db,
                "output_root": args.output_root,
                "auth_enabled": bool(token),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
