"""The watchdog emission shim must translate legacy (key, message) pairs
into ledger observations/recoveries and never raise into a collector."""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for name in ("incident_ledger", "incident_emit"):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
ledger = sys.modules["incident_ledger"]
emit = sys.modules["incident_emit"]

CONFIG_PATH = REPO_ROOT / "examples/configs/incident_ledger_v1.json"


def test_sanitize_code():
    assert emit.sanitize_code("temperature:0") == "temperature:0"
    assert emit.sanitize_code("GPU Count!") == "gpu_count"
    assert emit.sanitize_code("///") == "unnamed_event"


def test_observe_and_recover_roundtrip(tmp_path):
    db = tmp_path / "incidents.sqlite"
    assert emit.observe_incident(
        source="gpu_idle_watchdog", event_code="swarm_gpus_idle",
        severity="P2", summary="all GPUs idle", machine="omega",
        config_path=CONFIG_PATH, db_override=db)
    conn = ledger.connect(db)
    rows = ledger.open_incidents(conn)
    assert len(rows) == 1
    assert rows[0]["event_code"] == "swarm_gpus_idle"
    conn.close()
    assert emit.recover_incident(
        source="gpu_idle_watchdog", event_code="swarm_gpus_idle",
        evidence={"summary": "swarm active"}, machine="omega",
        config_path=CONFIG_PATH, db_override=db)
    conn = ledger.connect(db)
    assert ledger.open_incidents(conn) == []


def test_watchdog_pairs_map_alerts_and_recoveries(tmp_path):
    db = tmp_path / "incidents.sqlite"
    pairs = [
        ("temperature:0", "🚨🚨🚨 GPU TEMPERATURE ALERT 🚨🚨🚨\ndetail"),
        ("gpu_count", "✅ GPU COUNT RECOVERED\ndetail"),
    ]
    failures = emit.emit_watchdog_messages(
        source="gpu_temperature_watchdog", pairs=pairs, machine="omega",
        severity="P2", config_path=CONFIG_PATH, db_override=db)
    assert failures == 0
    conn = ledger.connect(db)
    rows = ledger.open_incidents(conn)
    assert [r["event_code"] for r in rows] == ["temperature:0"]


def test_emission_failure_is_reported_not_raised(tmp_path):
    ok = emit.observe_incident(
        source="s", event_code="e", severity="P2", summary="x",
        config_path=tmp_path / "missing.json",
        db_override=tmp_path / "db.sqlite")
    assert ok is False
