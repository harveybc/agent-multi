"""A terminal stranded by a destination that went away, and what recovery must look like.

R3 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "Demonstrate the new wrapper's real stranded outbox recovery: target unavailable ->
     pending item -> restored target -> exactly one row -> second flush sends nothing. For
     shared wrappers, prove actual delegation and exercise the relevant entry points rather
     than duplicating an entire suite."

So two things are proven here, and nothing is mocked that matters:

* **delegation** — agent-multi's `governed_run` hands its own profile to data-gov's shared
  `consumer_main`; that is asserted on the real call, not on the source text;
* **recovery** — the real `TerminalOutbox` and the real `DataGovClient` talk to a real HTTP
  server that is stopped and restarted. The destination really is unavailable, the envelope
  really stays on disk, and the count of terminals the server received is the evidence.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKOUT = Path(os.environ.get("DATA_GOV_CHECKOUT") or REPO.parent / "data-gov")

if not (CHECKOUT / "app" / "outbox.py").is_file():  # pragma: no cover - environment guard
    pytest.skip("data-gov checkout not available", allow_module_level=True)

sys.path.insert(0, str(CHECKOUT))
from app.client import DataGovClient  # noqa: E402
from app.outbox import TerminalOutbox  # noqa: E402

spec = importlib.util.spec_from_file_location("agent_multi_governed_run",
                                              REPO / "tools" / "governed_run.py")
governed_run = importlib.util.module_from_spec(spec)
sys.modules["agent_multi_governed_run"] = governed_run
spec.loader.exec_module(governed_run)


class Destination:
    """A real HTTP server that can be taken away and brought back."""

    def __init__(self):
        self.terminals = []
        self.server = None
        self.port = None

    def start(self, port=0):
        received = self.terminals

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status, body):
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path.endswith("/terminal"):
                    received.append({"path": self.path, "body": body})
                    return self._json(200, {"accepted": True, "terminal_sha256": "a" * 64})
                return self._json(200, {"ok": True})

            def do_GET(self):
                # the reconciliation shape the shared implementation really checks: a unit is
                # "missing" until its terminal is accepted, and never afterwards
                missing = [] if received else ["agent-multi-outbox-1", "agent-multi-outbox-2"]
                return self._json(200, {"missing_units": missing, "accounting_only": [],
                                        "lake_only": []})

        self.server = HTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.port

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


@pytest.fixture
def destination():
    target = Destination()
    yield target
    target.stop()


def envelope(unit="agent-multi-outbox-1"):
    return {
        "campaign_sha256": "c" * 64,
        "unit_id": unit,
        "terminal": {"schema": "governed_terminal.v1", "generation": 1,
                     "status": "COMPLETED", "reason": None,
                     "started_at": "2026-09-15T00:00:00Z",
                     "finished_at": "2026-09-15T00:00:07Z",
                     "costs": {"wall_seconds": 7.0}, "deliveries": [], "artifacts": {},
                     "metrics": [], "tags": {}},
    }


def flush(client, outbox):
    """The shared implementation's own flush, reached through data-gov's module."""
    spec_exec = importlib.util.spec_from_file_location(
        "data_gov_governed_exec", CHECKOUT / "tools" / "governed_exec.py")
    module = importlib.util.module_from_spec(spec_exec)
    sys.modules["data_gov_governed_exec"] = module
    spec_exec.loader.exec_module(module)
    return module.send_pending(client, outbox)


def test_the_wrapper_delegates_to_the_shared_implementation(monkeypatch):
    """Not asserted on source text: the real call is intercepted and its profile inspected."""
    seen = {}

    class FakeModule:
        @staticmethod
        def consumer_main(profile, argv, repo_root):
            seen["profile"] = profile
            seen["repo_root"] = repo_root
            return 0

    monkeypatch.setattr(governed_run, "_governed_exec", lambda: FakeModule)
    assert governed_run.main(["--load_config", "x.json"]) == 0
    assert seen["profile"]["project"] == "agent-multi"
    assert seen["repo_root"] == REPO
    assert "requested_timesteps" in seen["profile"]["metrics"](
        {"save_log": "replay_log.json"})["keys"]
    assert "observed_timesteps" in seen["profile"]["metrics"](
        {"save_log": "replay_log.json"})["keys"]


def test_a_terminal_survives_a_destination_that_is_not_there(tmp_path, destination):
    """The whole sequence, with a server that is really stopped and really restarted."""
    port = destination.start()
    base = f"http://127.0.0.1:{port}"
    destination.stop()                                   # the destination goes away

    outbox = TerminalOutbox(tmp_path / "outbox")
    item = outbox.put(envelope())
    assert item.path.is_file(), "the envelope is on disk before any send is attempted"

    client = DataGovClient(base_url=base, api_key="k", experiment_key="outbox-recovery")
    first = flush(client, outbox)
    assert first["pending"], "with no destination the envelope must remain pending"
    assert not destination.terminals
    assert item.path.is_file(), "a refused send must not delete the evidence"

    destination.start(port)                              # the destination comes back
    second = flush(client, outbox)
    assert not second["pending"], f"the envelope should have been sent: {second}"
    assert len(destination.terminals) == 1, "exactly one row, no duplicate"
    assert not item.path.is_file(), "the pending file is gone"
    assert list((tmp_path / "outbox" / "sent").glob("*.json")), "and it is recorded as sent"

    third = flush(client, outbox)
    assert len(destination.terminals) == 1, "a second flush must send nothing"
    assert not third["pending"]


def test_the_failure_is_recorded_beside_the_envelope_not_swallowed(tmp_path, destination):
    """A refusal leaves a sidecar saying how many attempts and what kind of failure."""
    port = destination.start()
    destination.stop()
    outbox = TerminalOutbox(tmp_path / "outbox")
    outbox.put(envelope("agent-multi-outbox-2"))
    client = DataGovClient(base_url=f"http://127.0.0.1:{port}", api_key="k",
                           experiment_key="outbox-failure")
    flush(client, outbox)
    status = outbox.status()
    assert status["pending"], status
    record = status["pending"][0]
    assert record.get("attempts", 0) >= 1 or record.get("failure"), record
