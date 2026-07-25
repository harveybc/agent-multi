#!/usr/bin/env python3
"""HTTP coordinator for the Project 3 evidence-sweep pool."""
from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from project3_evidence_pool import (  # noqa: E402
    claim_job,
    complete_job,
    connect,
    fail_job,
    heartbeat,
    init_db,
    status,
)


def _body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    size = int(handler.headers.get("Content-Length") or 0)
    return json.loads(handler.rfile.read(size).decode("utf-8")) if size else {}


def make_handler(db_path: Path, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Project3EvidencePoolAPI/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(f"{self.address_string()} [{self.log_date_time_string()}] {fmt % args}\n")

        def _authorized(self) -> bool:
            return not token or self.headers.get("Authorization") == f"Bearer {token}"

        def _send(self, payload: Any, code: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _conn(self):
            return connect(db_path)

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                self._send({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                return
            try:
                path = urlparse(self.path).path
                if path == "/health":
                    self._send({"ok": True, "service": "project3_evidence_pool_api"})
                elif path == "/status":
                    conn = self._conn()
                    self._send({"ok": True, "status": status(conn)})
                else:
                    self._send({"ok": False, "error": f"unknown path: {path}"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send(
                    {"ok": False, "error": str(exc), "error_type": type(exc).__name__},
                    HTTPStatus.BAD_REQUEST,
                )

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                self._send({"ok": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                return
            try:
                path = urlparse(self.path).path
                payload = _body(self)
                machine_id = str(payload.get("machine_id") or "").strip()
                if not machine_id:
                    raise ValueError("machine_id is required")
                conn = self._conn()
                if path == "/claim":
                    job = claim_job(
                        conn,
                        machine_id,
                        lease_seconds=int(payload.get("lease_seconds") or 300),
                    )
                    self._send({"ok": True, "job": job})
                elif path == "/heartbeat":
                    heartbeat(
                        conn,
                        machine_id,
                        payload.get("job_id"),
                        status=str(payload.get("status") or "running"),
                        message=str(payload.get("message") or ""),
                        cpu_summary=payload.get("cpu_summary"),
                        gpu_summary=payload.get("gpu_summary"),
                        lease_seconds=int(payload.get("lease_seconds") or 300),
                    )
                    self._send({"ok": True})
                elif path == "/complete":
                    complete_job(conn, machine_id, str(payload["job_id"]), dict(payload["result"]))
                    heartbeat(
                        conn,
                        machine_id,
                        None,
                        status="idle",
                        message=f"completed {payload['job_id']}",
                        cpu_summary=payload.get("cpu_summary"),
                        gpu_summary=payload.get("gpu_summary"),
                    )
                    self._send({"ok": True})
                elif path == "/fail":
                    fail_job(
                        conn,
                        machine_id,
                        str(payload["job_id"]),
                        str(payload.get("error") or "worker failed"),
                        retry=bool(payload.get("retry", True)),
                    )
                    self._send({"ok": True})
                else:
                    self._send({"ok": False, "error": f"unknown path: {path}"}, HTTPStatus.NOT_FOUND)
            except PermissionError as exc:
                self._send({"ok": False, "error": str(exc)}, HTTPStatus.FORBIDDEN)
            except Exception as exc:
                self._send(
                    {"ok": False, "error": str(exc), "error_type": type(exc).__name__},
                    HTTPStatus.BAD_REQUEST,
                )

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8796)
    parser.add_argument("--token")
    parser.add_argument("--token-file")
    args = parser.parse_args()
    token = args.token
    if args.token_file:
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    conn = connect(args.db)
    init_db(conn)
    conn.close()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Path(args.db), token))
    print(
        json.dumps(
            {
                "event": "project3_evidence_pool_api_started",
                "db": args.db,
                "host": args.host,
                "port": args.port,
                "auth_enabled": bool(token),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
