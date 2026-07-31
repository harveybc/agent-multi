from pathlib import Path

from examples.scripts.install_audit_snapshot_collector import (
    service_unit,
    timer_unit,
)


def test_service_is_bounded_and_read_only_except_snapshot_state() -> None:
    value = service_unit(Path("/repo"))

    assert "ExecStart=/usr/bin/python3 /repo/tools/audit_snapshot_collector.py" in value
    assert "CPUQuota=20%" in value
    assert "MemoryMax=256M" in value
    assert "ProtectSystem=strict" in value
    assert "ProtectHome=read-only" in value
    assert "ReadWritePaths=%h/.local/state/agent-multi/audit-snapshots" in value


def test_timer_runs_every_six_hours_with_jitter() -> None:
    value = timer_unit()

    assert "00,06,12,18:15:00" in value
    assert "Persistent=true" in value
    assert "RandomizedDelaySec=10m" in value
