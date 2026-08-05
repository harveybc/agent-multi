"""Findings 098/102: multifront_status venue truth derives from current
execution heartbeats plus lifecycle OLAP, never old read-only preflight
labels; cumulative Paper history is preserved."""
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "multifront_status", REPO_ROOT / "tools" / "multifront_status.py")
status = importlib.util.module_from_spec(_SPEC)
sys.modules["multifront_status"] = status
_SPEC.loader.exec_module(status)


def _fixture(tmp_path, *, read_only=False):
    heartbeat = tmp_path / "alpaca-model-runner-heartbeat.json"
    heartbeat.write_text(json.dumps({
        "schema": "lts.alpaca.model_runner.heartbeat.v1",
        "observed_at": "2026-08-05T03:00:00+00:00",
        "state": "replayed_signal", "read_only": read_only,
        "environment": "paper",
        "account_fingerprint": "0123456789abcdef",
        "model_id": "spy-daily-linear-live-v1",
    }))
    olap = tmp_path / "alpaca-model-execution.sqlite"
    con = sqlite3.connect(olap)
    con.executescript("""
        CREATE TABLE decisions (idempotency_key TEXT PRIMARY KEY,
            decided_at TEXT, outcome TEXT, reason TEXT);
        CREATE TABLE l1_effects (effect_id TEXT PRIMARY KEY, state TEXT);
        CREATE TABLE exposures (exposure_id TEXT PRIMARY KEY, state TEXT);
        CREATE TABLE live_model_sessions (session_id TEXT PRIMARY KEY);
        CREATE TABLE service_state (key TEXT PRIMARY KEY, value TEXT);
    """)
    con.executemany(
        "INSERT INTO decisions VALUES (?,?,?,?)",
        [(f"k{i}", f"2026-08-0{4 + (i > 5)}T0{i}:00:00+00:00",
          "would_be_order" if i < 4 else "superseded" if i == 4
          else "rejected", None) for i in range(7)])
    con.executemany("INSERT INTO l1_effects VALUES (?,?)",
                    [(f"e{i}", "terminal_flat") for i in range(4)])
    con.execute("INSERT INTO exposures VALUES ('x1', 'closed')")
    con.execute("INSERT INTO live_model_sessions VALUES ('s1')")
    con.commit()
    con.close()
    return heartbeat, olap


def test_mode_derives_from_execution_heartbeat(tmp_path):
    heartbeat, olap = _fixture(tmp_path, read_only=False)
    truth = status._venue_execution_truth(heartbeat, olap, time.time())
    assert truth["mode"] == "write_enabled"           # never a stale label
    assert truth["account_fingerprint"] == "0123456789abcdef"
    assert truth["model_id"] == "spy-daily-linear-live-v1"
    assert truth["lifecycles_cumulative"] == 4        # Paper history kept
    assert truth["decisions_cumulative"]["would_be_order"] == 4
    assert truth["decisions_cumulative"]["superseded"] == 1
    assert truth["open_exposures"] == 0
    assert truth["sessions_cumulative"] == 1
    assert truth["last_decision"]["outcome"] == "rejected"


def test_read_only_heartbeat_reports_read_only(tmp_path):
    heartbeat, olap = _fixture(tmp_path, read_only=True)
    truth = status._venue_execution_truth(heartbeat, olap, time.time())
    assert truth["mode"] == "read_only"


def test_missing_sources_are_unknown_never_fabricated(tmp_path):
    truth = status._venue_execution_truth(
        tmp_path / "absent.json", tmp_path / "absent.sqlite", time.time())
    assert truth["mode"] == "unknown"
    assert "decisions_cumulative" not in truth
