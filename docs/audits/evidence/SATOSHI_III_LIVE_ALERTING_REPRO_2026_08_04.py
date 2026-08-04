#!/usr/bin/env python3
"""Socket-free reproducer for the 2026-08-04 Satoshi III delivery audit.

Run from the agent-multi repository with:

    conda run -n trading-stack python \
      docs/audits/evidence/SATOSHI_III_LIVE_ALERTING_REPRO_2026_08_04.py

Every database is temporary. No broker, Telegram, SSH, production ledger or
runtime service is contacted.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


AGENT_REPO = Path(__file__).resolve().parents[3]
LTS_REPO = AGENT_REPO.parent / "lts"


def run_case(name: str, cwd: Path, source: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return {
        "case": name,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


CASES = [
    run_case(
        "resume_clears_racing_kill",
        LTS_REPO,
        r"""
        import importlib.util
        import json
        import tempfile
        from contextlib import contextmanager
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "resume_tests", Path("tests/unit/test_ibkr_l1_resume.py"))
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
        from app.ibkr_l1_journal import L1ExecutionOlap

        olap = L1ExecutionOlap(Path(tempfile.mkdtemp()) / "ledger.sqlite")
        effect = "l1e-f4993c2dda8cdc2a"
        olap.create_effect(effect, "key-1", "bracket_entry", [])
        for state in ("submitted_pending_ack", "acknowledged", "terminal_flat"):
            olap.advance_effect(effect, state)
        olap.set_state("halt", "hold")
        profile = tests.make_profile()
        payload = tests.make_payload(profile)
        record = tests.validate_resume_capability(
            payload, profile=profile, now=tests.NOW)
        original_atomic = olap.atomic_unit

        @contextmanager
        def racing_atomic():
            # Simulates a fresh safety event landing after the pre-check but
            # before resume obtains its write transaction.
            olap.set_state("halt", "kill")
            with original_atomic():
                yield

        olap.atomic_unit = racing_atomic
        result = tests.run(
            olap, profile, payload=payload, record=record)
        print(json.dumps({
            "applied": result["applied"],
            "final_halt": olap.get_state("halt"),
        }, sort_keys=True))
        olap.close()
        """,
    ),
    run_case(
        "json_secret_redaction",
        AGENT_REPO,
        r"""
        import importlib.util
        import json
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "incident_ledger", Path("tools/incident_ledger.py"))
        ledger = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ledger)
        source = json.dumps({
            "api_key": "PKTESTVALUE",
            "nested": {"token": "not-a-real-token"},
            "password": "not-a-real-password",
            "secret": "not-a-real-secret",
        }, sort_keys=True)
        redacted = ledger.redact(source)
        print(json.dumps({
            "unchanged": redacted == source,
            "contains_test_values": "not-a-real-secret" in redacted,
        }, sort_keys=True))
        """,
    ),
    run_case(
        "arbitrary_recovery_evidence",
        AGENT_REPO,
        r"""
        import importlib.util
        import json
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "incident_ledger", Path("tools/incident_ledger.py"))
        ledger = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ledger)
        now = datetime(2026, 8, 4, 19, 30, tzinfo=timezone.utc)
        config = {
            "max_future_skew_seconds": 120,
            "max_evidence_age_seconds": 60,
            "flap_reopen_window_seconds": 3600,
        }
        conn = ledger.connect(Path(tempfile.mkdtemp()) / "incidents.sqlite")
        observed = ledger.observe(
            conn, config, source="watchdog", front="front2",
            machine="omega", event_code="tws_down", severity="P0",
            affected_object="ibkr", payload={"state": "down"},
            source_evidence_at=ledger.iso(now), now=now)
        resolved = ledger.recover(
            conn, config, source="watchdog", front="front2",
            machine="omega", event_code="tws_down",
            affected_object="ibkr", evidence={"ok": True}, now=now)
        print(json.dumps({
            "before": observed["state"],
            "after": resolved["state"],
            "accepted_unbound_evidence": True,
        }, sort_keys=True))
        """,
    ),
]

print(json.dumps({"schema": "musashi.audit.reproducer.v1", "cases": CASES},
                 indent=2, sort_keys=True))
if any(case["returncode"] != 0 for case in CASES):
    raise SystemExit(1)
