from __future__ import annotations

from pathlib import Path
import subprocess

from tools import memory_pressure_watchdog
from tools.memory_pressure_watchdog import classify_pressure, parse_meminfo


def snapshot(*, available_gib: float, service_gib: float, swap_fraction: float = 0.0) -> dict:
    gib = 1024 ** 3
    return {
        "mem_available_bytes": int(available_gib * gib),
        "service_memory_bytes": int(service_gib * gib),
        "swap_used_fraction": swap_fraction,
        "service_active_state": "active",
        "memory_events": {"oom_kill": 0},
    }


def classify(value: dict, previous: dict | None = None) -> tuple[str, list[str]]:
    return classify_pressure(
        value,
        previous or {"oom_kill": 0},
        warning_available_gib=4,
        critical_available_gib=2,
        warning_service_gib=18,
        critical_service_gib=21,
    )


def test_parse_meminfo_converts_kib_to_bytes() -> None:
    values = parse_meminfo("MemTotal: 1024 kB\nMemAvailable: 512 kB\n")
    assert values == {"MemTotal": 1048576, "MemAvailable": 524288}


def test_pressure_thresholds() -> None:
    assert classify(snapshot(available_gib=10, service_gib=5))[0] == "healthy"
    assert classify(snapshot(available_gib=3, service_gib=5))[0] == "warning"
    assert classify(snapshot(available_gib=1, service_gib=5))[0] == "critical"
    assert classify(snapshot(available_gib=10, service_gib=22))[0] == "critical"


def test_new_cgroup_oom_kill_is_critical() -> None:
    value = snapshot(available_gib=10, service_gib=5)
    value["memory_events"]["oom_kill"] = 2
    severity, reasons = classify(value, {"oom_kill": 1})
    assert severity == "critical"
    assert "new cgroup OOM kills 1" in reasons


def test_read_snapshot_falls_back_to_cgroupfs_without_user_systemd_bus(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service = "doin-campaign-supervisor.service"
    cgroup = memory_pressure_watchdog.user_service_cgroup(
        service,
        cgroup_root=tmp_path,
        uid=1000,
    )
    cgroup.mkdir(parents=True)
    (cgroup / "cgroup.procs").write_text("123\n456\n", encoding="utf-8")
    (cgroup / "memory.current").write_text("1024\n", encoding="utf-8")
    (cgroup / "memory.peak").write_text("2048\n", encoding="utf-8")
    (cgroup / "memory.events").write_text("oom 1\noom_kill 0\n", encoding="utf-8")

    def unavailable_systemd(*_args, **_kwargs) -> str:
        raise subprocess.CalledProcessError(1, "systemctl")

    monkeypatch.setattr(memory_pressure_watchdog, "systemd_value", unavailable_systemd)
    result = memory_pressure_watchdog.read_snapshot(
        service,
        cgroup_root=tmp_path,
        uid=1000,
    )

    assert result["service_memory_bytes"] == 1024
    assert result["service_memory_peak_bytes"] == 2048
    assert result["service_active_state"] == "active"
    assert result["memory_events"] == {"oom": 1, "oom_kill": 0}
    assert result["service_snapshot_source"] == "cgroupfs"
