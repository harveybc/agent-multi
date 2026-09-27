"""RR02 (order 2026-09-26): this repository's fleet launcher reserves before it dispatches.

`tools/eth_curriculum_fleet.py` started each worker's training run with `systemd-run --user` and no
admission call at all, so the load was invisible to every other launcher's capacity reading on that
host.  It now asks the fleet's one memory authority first, through `tools/host_admission.py`.

Nothing here contacts a host, allocates memory or starts a process: the transport and the authority
are both stubbed, and the fleet's own worker table is replaced by one role-named entry.  No host
name, address or account identifier appears in this file.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Reply:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.fixture
def fleet(monkeypatch, tmp_path):
    F = _load("eth_curriculum_fleet")
    one = F.Worker("WORKER_ROLE_A", "worker-role-a", 101, 0, "replica-role")
    monkeypatch.setattr(F, "WORKERS", (one,))
    def preflight(root):
        packet = {"facts": {one.name: {"gpu": ["0, GPU-0000000000000000000000000000000000"]}}}
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "fleet_preflight.json").write_text(json.dumps(packet))
        return packet

    monkeypatch.setattr(F, "preflight", preflight)
    return F, one


def _transport(calls, *, verdict="ADMITTED", launch_rc=0, cgroup="ok"):
    """One stub for every remote command the launcher makes, answering as the host would."""
    def run(worker, argv, check=True, timeout=120):
        calls.append(list(argv))
        joined = " ".join(str(a) for a in argv)
        if "crispdm_admission.py" in joined and "acquire" in joined:
            if verdict == "ADMITTED":
                return Reply(0, json.dumps({"verdict": "ADMITTED", "code": "ADMITTED",
                                            "lease_id": "ethfleet-1-2-abc"}))
            return Reply(75, json.dumps({"verdict": verdict, "code": "HOST_HEADROOM",
                                         "reason": "8.00G requested; 1.00G free"}))
        if "crispdm_admission.py" in joined and "arm" in joined:
            return Reply(0, json.dumps({"ok": True}))
        if "crispdm_admission.py" in joined and "release" in joined:
            return Reply(0, json.dumps({"ok": True, "code": "RELEASED"}))
        if argv[0] == "systemctl":
            return Reply(0, "/user.slice/crispdm-batch.slice/unit.scope" if cgroup == "ok" else "")
        if argv[0] == "systemd-run":
            return Reply(launch_rc, "", "" if launch_rc == 0 else "boom")
        return Reply(0, "")
    return run


def test_RR02_nothing_is_dispatched_without_a_declared_measured_cap(fleet, monkeypatch, tmp_path):
    """A cap chosen to get past a gate is worse than no gate, so the launcher has no default."""
    F, _w = fleet
    calls = []
    monkeypatch.setattr(F, "_remote", _transport(calls))
    packet = F.start(tmp_path)
    assert packet["launched"] == []
    assert packet["memory_admission"]["refusals"][0]["code"] == "NO_MEASURED_CAP_DECLARED"
    assert not any(c[0] == "systemd-run" for c in calls), "no training run was started"


def test_RR02_a_worker_that_is_not_admitted_is_not_launched(fleet, monkeypatch, tmp_path):
    F, _w = fleet
    calls = []
    monkeypatch.setattr(F, "_remote", _transport(calls, verdict="QUEUED"))
    packet = F.start(tmp_path, mem_per_worker_bytes=8 << 30)
    assert packet["launched"] == []
    ref = packet["memory_admission"]["refusals"][0]
    assert ref["verdict"] == "QUEUED" and "free" in ref["reason"]
    assert not any(c[0] == "systemd-run" for c in calls)


def test_RR02_an_admitted_worker_is_launched_and_its_reservation_is_armed_to_its_own_cgroup(
        fleet, monkeypatch, tmp_path):
    F, w = fleet
    calls = []
    monkeypatch.setattr(F, "_remote", _transport(calls))
    packet = F.start(tmp_path, mem_per_worker_bytes=8 << 30)
    assert packet["launched"] == [w.name]
    assert packet["memory_admission"]["reservations"][w.name] == "ethfleet-1-2-abc"
    assert packet["memory_admission"]["cap_bytes_declared"] == 8 << 30
    order = [c for c in calls if c[0] in ("systemd-run",) or "crispdm_admission.py" in " ".join(map(str, c))]
    assert "acquire" in " ".join(map(str, order[0])), "the reservation comes BEFORE the launch"
    assert order[1][0] == "systemd-run"
    armed = [c for c in calls if "arm" in [str(x) for x in c]]
    assert armed and "--cgroup" in armed[0], "a detached unit is witnessed by its own cgroup"


def test_RR02_a_failed_launch_gives_the_reservation_back(fleet, monkeypatch, tmp_path):
    F, _w = fleet
    calls = []
    monkeypatch.setattr(F, "_remote", _transport(calls, launch_rc=1))
    with pytest.raises(RuntimeError):            # the launcher's own pre-existing partial-launch raise
        F.start(tmp_path, mem_per_worker_bytes=4 << 30)
    packet = json.loads((tmp_path / "fleet_launch.json").read_text())
    assert packet["launched"] == []
    assert any("release" in [str(x) for x in c] for c in calls), \
        "a reservation for a load that never started must not be held"


def test_RR02_an_unarmable_reservation_is_reported_not_ignored(fleet, monkeypatch, tmp_path):
    """A lease with no witness is protected only by the arming grace; once that expires the sweep
    reclaims it while the load is still running.  That must never pass silently."""
    F, w = fleet
    calls = []
    monkeypatch.setattr(F, "_remote", _transport(calls, cgroup="missing"))
    with pytest.raises(RuntimeError):            # a launch it cannot account for is not a success
        F.start(tmp_path, mem_per_worker_bytes=4 << 30)
    packet = json.loads((tmp_path / "fleet_launch.json").read_text())
    assert packet["launched"] == [w.name]
    assert any("not accounted for" in (e.get("stderr") or "") for e in packet["errors"])


def test_RR02_the_client_is_a_client_and_never_a_second_authority():
    """It holds no threshold, reads no memory and decides nothing of its own."""
    src = (TOOLS / "host_admission.py").read_text()
    for forbidden in ("MemAvailable", "/proc/meminfo", "memory.pressure", "psutil",
                      "drop_caches", "swapoff", "MemoryMax=", "os.kill", "pkill"):
        assert forbidden not in src, forbidden
    A = _load("host_admission")
    assert A.DEPLOYED_MODULE.endswith("crispdm_admission.py")
    # a missing authority refuses rather than launching
    adm = A.HostAdmission(run=lambda argv: Reply(127, "", "command not found"))
    d = adm.acquire(name="x", cap_bytes=1 << 30, wall_seconds=60)
    assert d["verdict"] == A.REFUSED and d["code"] == A.NO_AUTHORITY
