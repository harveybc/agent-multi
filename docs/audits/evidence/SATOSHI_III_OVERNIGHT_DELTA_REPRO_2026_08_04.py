#!/usr/bin/env python3
"""Socket-free reproducer for overnight-delta finding 106.

Run from agent-multi with the trading-stack interpreter. It imports only the
LTS unit fixture and monitor; no broker, database, SSH or Telegram is touched.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


AGENT_REPO = Path(__file__).resolve().parents[3]
LTS_REPO = AGENT_REPO.parent / "lts"
TEST_FILE = LTS_REPO / "tests" / "unit" / "test_tws_continuity_monitor.py"

spec = importlib.util.spec_from_file_location("tws_monitor_tests", TEST_FILE)
tests = importlib.util.module_from_spec(spec)
sys.modules["tws_monitor_tests"] = tests
spec.loader.exec_module(tests)

probes = tests.healthy_probes()
probes["heartbeat"].update({
    "state": "decided",
    "timeframe": None,
    "last_closed_bar": None,
})
emissions, state = tests.monitor.assess(probes, tests.CONFIG, tests.NOW, {})

observed = tests._observed(emissions, "decision_clock_stale")
recovered = tests._recovered(emissions, "decision_clock_stale")
print(json.dumps({
    "schema": "musashi.audit.tws_clock_state_reproducer.v1",
    "input_state": "decided",
    "clock_observed": bool(observed),
    "clock_recovered": bool(recovered),
    "tws_healthy": bool(state["tws_healthy"]),
    "reproduced": not observed and bool(recovered) and bool(state["tws_healthy"]),
}, indent=2, sort_keys=True))
