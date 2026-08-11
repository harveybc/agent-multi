"""GPU readiness probe proofs (post-outage order §3.2/§3.3 + §10.3).

Socket-free: every shell-out (uname, modinfo, nvidia-smi, dpkg-query,
apt-cache, the subprocess torch probe) is injected through a fake
runner, the boot-kernel listing and statvfs are injected callables, and
incident emission goes to a recording fake — the real ledger, Telegram
and any GPU are never touched.

Covered, per the order's §3.2.7 test list plus the disk/temperature
gates:

* kernel advance without module (newest /boot kernel lacks its nvidia
  module BEFORE reboot) is a typed boot-guard fact + one incident;
* running-kernel module missing / driver probe failure / UUID drift /
  framework CPU fallback classify exactly as the five-state contract
  and the launch gate refuses with a typed refusal before any framework
  import;
* one-host loss leaves healthy hosts unaffected;
* repeated failing polls emit ONE incident (dedup) and recovery emits
  exactly one notice, only on direct recovery evidence;
* temperature >= 78 C alerts once and recovers once after cooldown
  hysteresis;
* the disk-budget gate blocks dispatch on the affected host only with
  the typed HOST_DISK_INSUFFICIENT classification;
* the dispatch-binding dict carries exactly kernel/driver/UUID/
  framework-build/CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import gpu_readiness_probe as probe  # noqa: E402

OMEGA_UUID = "GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326"
DRAGON_UUID = "GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9"
GAMMA_UUIDS = ["GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519",
               "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"]

KERNEL_OLD = "7.0.0-28-generic"
KERNEL_NEW = "7.0.0-29-generic"
MODULE_PKG = f"linux-modules-nvidia-580-open-{KERNEL_NEW}"
MODULE_PIN = f"{MODULE_PKG}=7.0.0-29.29+1"

TORCH_OK = {"torch_present": True, "torch_version": "2.4.1",
            "torch_cuda_version": "12.4", "cuda_available": True,
            "cuda_device_count": 1, "error": None}
TORCH_CPU = {"torch_present": True, "torch_version": "2.4.1",
             "torch_cuda_version": "12.4", "cuda_available": False,
             "cuda_device_count": 0, "error": None}
TORCH_ABSENT = {"torch_present": False, "torch_version": None,
                "torch_cuda_version": None, "cuda_available": None,
                "cuda_device_count": None,
                "error": "torch absent: No module named 'torch'"}


def _ns(returncode: int, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout,
                           stderr=stderr)


class FakeRunner:
    """Injectable stand-in for every shell-out the probe performs."""

    def __init__(self, *, kernel: str = KERNEL_NEW,
                 module_kernels=(KERNEL_OLD, KERNEL_NEW),
                 module_version: str = "580.82.07",
                 smi_ok: bool = True,
                 smi_error: str = ("NVIDIA-SMI has failed because it "
                                   "couldn't communicate with the NVIDIA "
                                   "driver."),
                 gpus=None,  # list of (uuid, name, temperature)
                 driver_version: str = "580.82.07",
                 dpkg_rows=None, apt_candidate: str | None = "7.0.0-29.29+1",
                 torch_result=TORCH_OK, torch_rc: int = 0):
        self.kernel = kernel
        self.module_kernels = set(module_kernels)
        self.module_version = module_version
        self.smi_ok = smi_ok
        self.smi_error = smi_error
        self.gpus = gpus if gpus is not None else \
            [(DRAGON_UUID, "NVIDIA GeForce RTX 3090", 45.0)]
        self.driver_version = driver_version
        self.dpkg_rows = dpkg_rows if dpkg_rows is not None else [
            (f"linux-modules-nvidia-580-open-{KERNEL_OLD}", "7.0.0-28.28"),
        ]
        self.apt_candidate = apt_candidate
        self.torch_result = torch_result
        self.torch_rc = torch_rc
        self.calls: list[list[str]] = []
        self.torch_envs: list[str | None] = []

    def __call__(self, cmd, *, timeout: float = 30.0, env=None):
        self.calls.append(list(cmd))
        name = os.path.basename(cmd[0])
        if name == "uname":
            return _ns(0, self.kernel + "\n")
        if name == "modinfo":
            kernel = cmd[2]
            if kernel in self.module_kernels:
                return _ns(0, "filename: /lib/modules/.../nvidia.ko\n"
                              f"version:        {self.module_version}\n")
            return _ns(1, "", "modinfo: ERROR: Module nvidia not found "
                              f"in directory /lib/modules/{kernel}")
        if name == "nvidia-smi":
            if not self.smi_ok:
                return _ns(9, "", self.smi_error)
            rows = "\n".join(
                f"{uuid}, {gpu_name}, {self.driver_version}, "
                f"{temperature:.0f}"
                for uuid, gpu_name, temperature in self.gpus)
            return _ns(0, rows + "\n")
        if name == "dpkg-query":
            rows = "\n".join(f"{pkg}\t{ver}" for pkg, ver in self.dpkg_rows)
            return _ns(0, rows + "\n")
        if name == "apt-cache":
            package = cmd[2]
            candidate = self.apt_candidate or "(none)"
            return _ns(0, f"{package}:\n  Installed: (none)\n"
                          f"  Candidate: {candidate}\n  Version table:\n")
        if len(cmd) >= 3 and cmd[1] == "-c":  # subprocess torch probe
            self.torch_envs.append(
                (env or {}).get("CUDA_VISIBLE_DEVICES"))
            if self.torch_rc != 0:
                return _ns(self.torch_rc, "", "probe interpreter crashed")
            result = dict(self.torch_result)
            result["cuda_visible_devices"] = \
                (env or {}).get("CUDA_VISIBLE_DEVICES")
            return _ns(0, json.dumps(result) + "\n")
        raise AssertionError(f"unexpected command in test: {cmd}")


class FakeEmitter:
    """Recording emitter; True return mimics successful ledger writes."""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.observed: list[dict] = []
        self.recovered: list[dict] = []

    def observe(self, event_code, severity, summary, payload,
                affected_object="-"):
        self.observed.append({
            "event_code": event_code, "severity": severity,
            "summary": summary, "payload": payload,
            "affected_object": affected_object})
        return self.ok

    def recover(self, event_code, evidence, affected_object="-"):
        self.recovered.append({
            "event_code": event_code, "evidence": evidence,
            "affected_object": affected_object})
        return self.ok


def statvfs_with(available_bytes: int):
    def fake_statvfs(path):
        return SimpleNamespace(f_bavail=available_bytes, f_frsize=1)
    return fake_statvfs


def heartbeat(runner: FakeRunner, *, hostname: str = "dragon",
              boot_kernels=(KERNEL_OLD, KERNEL_NEW), **kwargs) -> dict:
    return probe.collect_heartbeat(
        hostname=hostname, runner=runner,
        boot_lister=lambda: list(boot_kernels),
        statvfs_fn=kwargs.pop("statvfs_fn", statvfs_with(10 ** 12)),
        base_env={}, **kwargs)


# ---------------------------------------------------------------------------
# 1. Boot guard: kernel advance without module, BEFORE reboot
# ---------------------------------------------------------------------------

def test_kernel_advance_without_module_is_reported_before_reboot():
    # Running -28 with a module; -29 is installed in /boot with NO
    # module: exactly the 2026-08-11 outage, caught before the reboot.
    runner = FakeRunner(kernel=KERNEL_OLD, module_kernels=(KERNEL_OLD,))
    beat = heartbeat(runner)
    assert beat["classification"] == "GPU_READY"
    guard = beat["boot_guard"]
    assert guard["newest_installed_kernel"] == KERNEL_NEW
    assert guard["kernel_advance_without_module"] is True
    assert guard["modinfo_nvidia"]["ok"] is False
    # remediation package derived from the installed 580-open family,
    # pinned to the exact missing kernel and candidate version
    assert beat["remediation_package"]["package"] == MODULE_PKG
    assert beat["remediation_package"]["pin"] == MODULE_PIN

    emitter = FakeEmitter()
    result = probe.process_incidents(beat, probe.default_state(), emitter)
    codes = [entry["event_code"] for entry in emitter.observed]
    assert codes == [probe.BOOT_GUARD_EVENT_CODE]
    assert MODULE_PIN in emitter.observed[0]["summary"]
    # dedup: an unchanged fact on the next poll emits nothing
    again = probe.process_incidents(beat, result["state"], emitter)
    assert len(emitter.observed) == 1
    assert emitter.recovered == []
    # recovery only once the newest kernel really has its module
    fixed = heartbeat(FakeRunner(kernel=KERNEL_OLD))
    probe.process_incidents(fixed, again["state"], emitter)
    assert [entry["event_code"] for entry in emitter.recovered] == \
        [probe.BOOT_GUARD_EVENT_CODE]


def test_boot_guard_clear_when_newest_kernel_has_module():
    beat = heartbeat(FakeRunner(kernel=KERNEL_NEW))
    assert beat["boot_guard"]["kernel_advance_without_module"] is False


# ---------------------------------------------------------------------------
# 2-4. Five-state classification + launch gate refusals
# ---------------------------------------------------------------------------

def test_running_kernel_module_missing_classifies_and_gates():
    runner = FakeRunner(kernel=KERNEL_NEW, module_kernels=(KERNEL_OLD,),
                        smi_ok=False)
    beat = heartbeat(runner)
    assert beat["classification"] == \
        "GPU_UNAVAILABLE_KERNEL_MODULE_MISSING"
    assert beat["remediation_package"]["pin"] == MODULE_PIN
    payload, code = probe.launch_gate(beat, DRAGON_UUID)
    assert code == probe.EXIT_REFUSED == 4
    assert payload["gate"] == "REFUSED"
    assert payload["outcome"] == "REFUSED_GPU_UNBOUND"
    assert payload["blocking"] == ["GPU_UNAVAILABLE_KERNEL_MODULE_MISSING"]
    assert payload["remediation_package"] == MODULE_PIN


def test_driver_probe_failure_classifies_and_gates():
    runner = FakeRunner(smi_ok=False)  # module exists, driver dead
    beat = heartbeat(runner)
    assert beat["classification"] == "GPU_UNAVAILABLE_DRIVER"
    assert beat["driver"]["nvidia_smi_ok"] is False
    payload, code = probe.launch_gate(beat, DRAGON_UUID)
    assert code == probe.EXIT_REFUSED
    assert payload["blocking"] == ["GPU_UNAVAILABLE_DRIVER"]


def test_uuid_drift_classifies_and_gates():
    stranger = "GPU-00000000-1111-2222-3333-444444444444"
    runner = FakeRunner(gpus=[(stranger, "NVIDIA RTX 3090", 40.0)])
    beat = heartbeat(runner)
    assert beat["classification"] == "GPU_UUID_MISMATCH"
    assert beat["gpus"]["missing_uuids"] == [DRAGON_UUID]
    assert beat["gpus"]["unexpected_uuids"] == [stranger]
    assert beat["gpus"]["exact_match"] is False
    # torch is never probed when the UUID stage already failed
    assert beat["framework"]["status"] == "NOT_PROBED"
    payload, code = probe.launch_gate(beat, DRAGON_UUID)
    assert code == probe.EXIT_REFUSED
    assert payload["blocking"] == ["GPU_UUID_MISMATCH"]


def test_framework_cpu_fallback_classifies_and_gates():
    runner = FakeRunner(torch_result=TORCH_CPU)
    beat = heartbeat(runner)
    assert beat["classification"] == "GPU_FRAMEWORK_MISMATCH"
    assert beat["framework"]["status"] == "TORCH_CPU_FALLBACK"
    # the probe pinned CUDA_VISIBLE_DEVICES to the assigned UUID
    assert runner.torch_envs == [DRAGON_UUID]
    payload, code = probe.launch_gate(beat, DRAGON_UUID)
    assert code == probe.EXIT_REFUSED
    assert payload["blocking"] == ["GPU_FRAMEWORK_MISMATCH"]


def test_missing_torch_is_a_typed_fact_not_a_failure():
    beat = heartbeat(FakeRunner(torch_result=TORCH_ABSENT))
    assert beat["framework"]["status"] == "TORCH_ABSENT"
    assert beat["classification"] == "GPU_READY"
    per_uuid = beat["framework"]["per_uuid"][DRAGON_UUID]
    assert per_uuid["status"] == "TORCH_ABSENT"
    assert "torch absent" in per_uuid["error"]


def test_healthy_host_is_gpu_ready_and_gate_passes():
    beat = heartbeat(FakeRunner())
    assert beat["classification"] == "GPU_READY"
    payload, code = probe.launch_gate(beat, DRAGON_UUID)
    assert code == probe.EXIT_OK == 0
    assert payload["gate"] == "PASS"
    assert payload["outcome"] == "GATE_PASS"
    assert payload["blocking"] == []


def test_classification_vocabulary_is_exactly_the_contract():
    scenarios = [
        FakeRunner(kernel=KERNEL_NEW, module_kernels=(KERNEL_OLD,),
                   smi_ok=False),
        FakeRunner(smi_ok=False),
        FakeRunner(gpus=[("GPU-ffffffff-0000-0000-0000-000000000000",
                          "X", 40.0)]),
        FakeRunner(torch_result=TORCH_CPU),
        FakeRunner(),
    ]
    seen = {heartbeat(runner)["classification"] for runner in scenarios}
    assert seen == set(probe.CLASSIFICATIONS)
    for label in seen:  # never generic healthy/unhealthy text
        assert label not in ("healthy", "unhealthy", "HEALTHY", "OK")


def test_gate_on_multi_gpu_host_with_one_assigned_uuid():
    gpus = [(uuid, "NVIDIA RTX 3090", 50.0) for uuid in GAMMA_UUIDS]
    beat = heartbeat(FakeRunner(gpus=gpus), hostname="gamma")
    assert beat["classification"] == "GPU_READY"
    payload, code = probe.launch_gate(beat, GAMMA_UUIDS[1])
    assert code == probe.EXIT_OK
    # an assigned UUID that is not observed refuses even on a host whose
    # overall set matches
    payload, code = probe.launch_gate(
        beat, "GPU-dead0000-0000-0000-0000-000000000000")
    assert code == probe.EXIT_REFUSED
    assert payload["blocking"] == ["GPU_UUID_MISMATCH"]


def test_gate_never_imports_the_training_framework():
    # the torch probe lives ONLY inside the subprocess script; importing
    # the probe module and running the gate must not import torch here
    assert "import torch" in probe.FRAMEWORK_PROBE_SCRIPT
    source = Path(probe.__file__).read_text(encoding="utf-8")
    body = source.replace(probe.FRAMEWORK_PROBE_SCRIPT, "")
    assert "import torch" not in body
    torch_was_loaded = "torch" in sys.modules
    beat = heartbeat(FakeRunner(torch_result=TORCH_CPU))
    probe.launch_gate(beat, DRAGON_UUID)
    assert ("torch" in sys.modules) == torch_was_loaded


# ---------------------------------------------------------------------------
# 5. Incident dedup, one-host loss, exactly-once recovery
# ---------------------------------------------------------------------------

def test_one_host_loss_leaves_healthy_hosts_unaffected():
    dragon_emitter, omega_emitter = FakeEmitter(), FakeEmitter()
    broken = heartbeat(FakeRunner(smi_ok=False), hostname="dragon")
    healthy = heartbeat(
        FakeRunner(gpus=[(OMEGA_UUID, "NVIDIA RTX 4070", 44.0)]),
        hostname="omega")
    probe.process_incidents(broken, probe.default_state(), dragon_emitter)
    probe.process_incidents(healthy, probe.default_state(), omega_emitter)
    assert [entry["event_code"] for entry in dragon_emitter.observed] == \
        ["gpu_readiness.driver_unavailable"]
    assert omega_emitter.observed == []
    assert omega_emitter.recovered == []
    # the healthy host still dispatches
    _, code = probe.launch_gate(healthy, OMEGA_UUID)
    assert code == probe.EXIT_OK


def test_incident_dedup_across_polls_and_recovery_exactly_once():
    emitter = FakeEmitter()
    state = probe.default_state()
    broken = heartbeat(FakeRunner(smi_ok=False))
    # first failing poll: exactly one observation
    state = probe.process_incidents(broken, state, emitter)["state"]
    assert len(emitter.observed) == 1
    # repeated failing polls: no new emission (state-change dedup)
    for _ in range(5):
        state = probe.process_incidents(broken, state, emitter)["state"]
    assert len(emitter.observed) == 1
    assert emitter.recovered == []
    # recovery evidence: one recovery notice, exactly once
    fixed = heartbeat(FakeRunner())
    state = probe.process_incidents(fixed, state, emitter)["state"]
    assert [entry["event_code"] for entry in emitter.recovered] == \
        ["gpu_readiness.driver_unavailable"]
    evidence = emitter.recovered[0]["evidence"]
    assert evidence["nvidia_smi_ok"] is True  # direct recovery evidence
    assert evidence["observed_gpu_uuids"] == [DRAGON_UUID]
    for _ in range(5):
        state = probe.process_incidents(fixed, state, emitter)["state"]
    assert len(emitter.recovered) == 1
    assert len(emitter.observed) == 1


def test_classification_change_swaps_incident_not_duplicates():
    emitter = FakeEmitter()
    state = probe.default_state()
    module_missing = heartbeat(
        FakeRunner(kernel=KERNEL_NEW, module_kernels=(KERNEL_OLD,),
                   smi_ok=False))
    state = probe.process_incidents(module_missing, state, emitter)["state"]
    drifted = heartbeat(
        FakeRunner(gpus=[("GPU-ffffffff-0000-0000-0000-000000000000",
                          "X", 40.0)]))
    state = probe.process_incidents(drifted, state, emitter)["state"]
    classification_codes = set(probe.CLASSIFICATION_EVENT_CODES.values())
    assert [entry["event_code"] for entry in emitter.observed
            if entry["event_code"] in classification_codes] == [
        "gpu_readiness.kernel_module_missing",
        "gpu_readiness.uuid_mismatch"]
    assert [entry["event_code"] for entry in emitter.recovered
            if entry["event_code"] in classification_codes] == [
        "gpu_readiness.kernel_module_missing"]


def test_failed_emission_retries_on_next_poll_without_state_loss():
    failing = FakeEmitter(ok=False)
    broken = heartbeat(FakeRunner(smi_ok=False))
    state = probe.process_incidents(broken, probe.default_state(),
                                    failing)["state"]
    # the dedup marker must NOT advance on a failed ledger write
    assert state["active_classification_code"] is None
    working = FakeEmitter()
    state = probe.process_incidents(broken, state, working)["state"]
    assert len(working.observed) == 1
    assert state["active_classification_code"] == \
        "gpu_readiness.driver_unavailable"


# ---------------------------------------------------------------------------
# 6. Temperature: >= 78 C alerts once, recovery only after cooldown
# ---------------------------------------------------------------------------

def test_temperature_alert_dedup_and_cooldown_recovery():
    emitter = FakeEmitter()
    state = probe.default_state()

    def beat_at(temperature: float) -> dict:
        return heartbeat(FakeRunner(
            gpus=[(DRAGON_UUID, "NVIDIA RTX 3090", temperature)]))

    hot = beat_at(82.0)
    assert hot["temperature"]["alerting_uuids"] == [DRAGON_UUID]
    assert hot["temperature"]["max_temperature_c"] == 82.0
    state = probe.process_incidents(hot, state, emitter)["state"]
    assert [entry["event_code"] for entry in emitter.observed] == \
        [probe.TEMPERATURE_EVENT_CODE]
    assert emitter.observed[0]["affected_object"] == DRAGON_UUID
    assert emitter.observed[0]["severity"] == "P2"
    # still hot: no second alert
    state = probe.process_incidents(beat_at(83.0), state, emitter)["state"]
    assert len(emitter.observed) == 1
    # inside the hysteresis band (73 < T < 78): neither alert nor recovery
    state = probe.process_incidents(beat_at(75.0), state, emitter)["state"]
    assert len(emitter.observed) == 1
    assert emitter.recovered == []
    # cooled below the recovery threshold: exactly one recovery notice
    state = probe.process_incidents(beat_at(70.0), state, emitter)["state"]
    assert [entry["event_code"] for entry in emitter.recovered] == \
        [probe.TEMPERATURE_EVENT_CODE]
    state = probe.process_incidents(beat_at(69.0), state, emitter)["state"]
    assert len(emitter.recovered) == 1


# ---------------------------------------------------------------------------
# 7. Disk-budget dispatch gate (order §3.3)
# ---------------------------------------------------------------------------

def test_disk_budget_gate_blocks_dispatch_on_this_host_only():
    emitter = FakeEmitter()
    tight = heartbeat(
        FakeRunner(), output_fs="/home/harveybc",
        expected_artifact_bytes=20 * 1024 ** 3,
        reserve_bytes=10 * 1024 ** 3,
        statvfs_fn=statvfs_with(15 * 1024 ** 3))
    assert tight["classification"] == "GPU_READY"  # GPU axis unaffected
    assert tight["disk"]["classification"] == "HOST_DISK_INSUFFICIENT"
    assert tight["disk"]["dispatch_blocked"] is True
    assert tight["disk"]["required_bytes"] == 30 * 1024 ** 3
    payload, code = probe.launch_gate(tight, DRAGON_UUID)
    assert code == probe.EXIT_REFUSED
    assert payload["blocking"] == ["HOST_DISK_INSUFFICIENT"]

    state = probe.process_incidents(tight, probe.default_state(),
                                    emitter)["state"]
    state = probe.process_incidents(tight, state, emitter)["state"]
    disk_alerts = [entry for entry in emitter.observed
                   if entry["event_code"] == probe.DISK_EVENT_CODE]
    assert len(disk_alerts) == 1  # alerts once, not per poll
    assert "dispatch blocked on this host only" in \
        disk_alerts[0]["summary"]

    # space freed: sufficient again, one recovery, dispatch unblocked
    roomy = heartbeat(
        FakeRunner(), output_fs="/home/harveybc",
        expected_artifact_bytes=20 * 1024 ** 3,
        reserve_bytes=10 * 1024 ** 3,
        statvfs_fn=statvfs_with(500 * 1024 ** 3))
    assert roomy["disk"]["classification"] == "HOST_DISK_SUFFICIENT"
    state = probe.process_incidents(roomy, state, emitter)["state"]
    assert [entry["event_code"] for entry in emitter.recovered] == \
        [probe.DISK_EVENT_CODE]
    _, code = probe.launch_gate(roomy, DRAGON_UUID)
    assert code == probe.EXIT_OK
    # a healthy sibling host with room is never touched by this state
    other = heartbeat(
        FakeRunner(gpus=[(OMEGA_UUID, "NVIDIA RTX 4070", 40.0)]),
        hostname="omega")
    assert other["disk"]["classification"] == "HOST_DISK_NOT_EVALUATED"
    assert other["disk"]["dispatch_blocked"] is False


def test_disk_gate_not_evaluated_without_budget_parameters():
    beat = heartbeat(FakeRunner())
    assert beat["disk"]["classification"] == "HOST_DISK_NOT_EVALUATED"
    assert beat["disk"]["dispatch_blocked"] is False


# ---------------------------------------------------------------------------
# 8. Dispatch-record binding dict
# ---------------------------------------------------------------------------

def test_dispatch_binding_carries_exactly_the_required_facts():
    runner = FakeRunner()
    binding = probe.dispatch_gpu_binding(
        DRAGON_UUID, hostname="dragon", runner=runner,
        base_env={"CUDA_VISIBLE_DEVICES": DRAGON_UUID})
    assert binding == {
        "schema": "agent_multi.gpu_dispatch_binding.v1",
        "hostname": "dragon",
        "running_kernel": KERNEL_NEW,
        "driver_version": "580.82.07",
        "assigned_gpu_uuid": DRAGON_UUID,
        "observed_gpu_uuids": [DRAGON_UUID],
        "cuda_visible_devices": DRAGON_UUID,
        "framework_build": {
            "status": "TORCH_CUDA_OK",
            "torch_version": "2.4.1",
            "torch_cuda_version": "12.4",
            "cuda_available": True,
        },
    }
    # the framework probe subprocess was pinned to the assigned UUID
    assert runner.torch_envs == [DRAGON_UUID]


# ---------------------------------------------------------------------------
# 9. Heartbeat plumbing: expected-UUID contract, state files
# ---------------------------------------------------------------------------

def test_embedded_uuid_contract_matches_the_order():
    assert probe.expected_uuids_for_host("omega") == [OMEGA_UUID]
    assert probe.expected_uuids_for_host("dragon") == [DRAGON_UUID]
    assert probe.expected_uuids_for_host("gamma") == GAMMA_UUIDS
    assert probe.expected_uuids_for_host("gamma.local") == GAMMA_UUIDS
    # config override wins over the embedded contract
    assert probe.expected_uuids_for_host(
        "dragon", {"dragon": ["GPU-x"]}) == ["GPU-x"]
    assert probe.expected_uuids_for_host("unknown-host") == []


def test_state_roundtrip_and_corrupt_state_resets(tmp_path):
    path = tmp_path / "state.json"
    assert probe.load_state(path) == probe.default_state()
    state = probe.default_state()
    state["active_classification_code"] = "gpu_readiness.uuid_mismatch"
    probe.atomic_write_json(path, state)
    assert probe.load_state(path)["active_classification_code"] == \
        "gpu_readiness.uuid_mismatch"
    path.write_text("not json", encoding="utf-8")
    assert probe.load_state(path) == probe.default_state()


def test_newest_boot_kernel_ordering():
    kernels = ["7.0.0-9-generic", "7.0.0-28-generic", "7.0.0-29-generic"]
    assert sorted(kernels, key=probe.kernel_sort_key)[-1] == \
        "7.0.0-29-generic"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
