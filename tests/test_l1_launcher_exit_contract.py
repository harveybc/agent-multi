"""WP7 (finding 188): the REAL CLI-to-systemd exit contract.

These tests run the actual launcher CLI in subprocesses and parse the
actual systemd unit — unit tests of SeedLauncher.run alone are
insufficient per the correction order.
"""
from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import l1_fleet_launcher as fl  # noqa: E402
from tools import l1_factorial_screen as runner  # noqa: E402

PY = sys.executable
SERVICE = (REPO / "examples/systemd/l1-factorial@.service").read_text()


def write_fixtures(tmp_path: Path, *, hostname: str,
                   anchor_exists: bool = True) -> tuple[Path, Path, dict,
                                                        dict]:
    contract = runner.load_contract()
    contract = json.loads(json.dumps(
        {k: v for k, v in contract.items()
         if k != "_contract_sha256"}))
    anchor = tmp_path / "anchor101.zip"
    if anchor_exists:
        anchor.write_bytes(b"anchor-bytes")
    contract["anchors"] = {"101": {
        "path": str(anchor),
        "sha256": hashlib.sha256(b"anchor-bytes").hexdigest()}}
    contract["assignments"] = {"101": {"hostname": hostname,
                                       "gpu_uuid": "GPU-test"}}
    contract["output_root"] = str(tmp_path / "out")
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    manifest = {"schema": "agent_multi.system_manifest.v1",
                "plugins": {
                    "agent_plugin": "sac_agent",
                    "pipeline_plugin": "rl_pipeline_with_validation",
                    "curriculum_pipeline_plugin":
                        "rl_pipeline_with_solvency_curriculum",
                }}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    loaded_contract = runner.load_contract(contract_path)
    loaded_manifest = runner.load_system_manifest(manifest_path)
    return contract_path, manifest_path, loaded_contract, loaded_manifest


def run_cli(contract_path: Path, manifest_path: Path,
            timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, "tools/l1_fleet_launcher.py", "--seed", "101",
         "--no-gpu-check", "--contract", str(contract_path),
         "--manifest", str(manifest_path)],
        cwd=REPO, capture_output=True, text=True, timeout=timeout)


class TestCliExitClasses:
    def test_wrong_host_refusal_exits_4(self, tmp_path):
        contract_path, manifest_path, _, _ = write_fixtures(
            tmp_path, hostname="definitely-not-this-host")
        proc = run_cli(contract_path, manifest_path)
        assert proc.returncode == 4, proc.stdout + proc.stderr
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert out["outcome"] == "REFUSED_WRONG_HOST"
        # The refusal heartbeat is visible on disk.
        hb = next((tmp_path / "out").rglob("launcher_heartbeat.json"))
        assert json.loads(hb.read_text())[
            "terminal_state"] == "REFUSED_WRONG_HOST"

    def test_seed_failed_exits_1(self, tmp_path):
        contract_path, manifest_path, _, _ = write_fixtures(
            tmp_path, hostname=socket.gethostname(),
            anchor_exists=False)
        proc = run_cli(contract_path, manifest_path)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert out["outcome"] == "SEED_FAILED"

    def test_already_running_exits_3(self, tmp_path):
        contract_path, manifest_path, contract, manifest = \
            write_fixtures(tmp_path, hostname=socket.gethostname())
        exp_id = runner.experiment_identity(contract, manifest, False)
        lock = (Path(contract["output_root"]).expanduser() / exp_id /
                "locks" / "exclusive_claim.seed101.lock")
        claim = fl.ExclusiveClaim(lock)
        assert claim.acquire()
        try:
            proc = run_cli(contract_path, manifest_path)
        finally:
            claim.release()
        assert proc.returncode == 3, proc.stdout + proc.stderr
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert out["outcome"] == "ALREADY_RUNNING"
        assert out["holder"].get("pid")

    def test_already_complete_exits_0(self, tmp_path):
        contract_path, manifest_path, contract, manifest = \
            write_fixtures(tmp_path, hostname=socket.gethostname())
        exp_id = runner.experiment_identity(contract, manifest, False)
        for cell in contract["cells"]:
            cell_id = runner.cell_identity(exp_id, 101, cell, contract)
            cell_dir = (Path(contract["output_root"]).expanduser() /
                        exp_id / "seed101" / cell)
            cell_dir.mkdir(parents=True)
            terminal = cell_dir / "model.terminal.zip"
            terminal.write_bytes(b"terminal")
            runner.atomic_write_json(
                cell_dir / "l1_cell_record.json",
                {"schema": runner.SCHEMA, "cell_identity": cell_id,
                 "terminal_model_path": str(terminal),
                 "terminal_model_sha256": hashlib.sha256(
                     b"terminal").hexdigest()})
        proc = run_cli(contract_path, manifest_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert out["outcome"] == "ALREADY_COMPLETE"


class TestSystemdUnitContract:
    def test_seed_failed_code_is_never_declared_success(self):
        assert fl.EXIT_CLASS["SEED_FAILED"] == 1
        success = ""
        for line in SERVICE.splitlines():
            if line.startswith("SuccessExitStatus="):
                success = line.split("=", 1)[1]
        assert str(fl.EXIT_CLASS["SEED_FAILED"]) not in success.split()

    def test_already_running_is_clean_no_op(self):
        assert f"SuccessExitStatus={fl.EXIT_CLASS['ALREADY_RUNNING']}" \
            in SERVICE

    def test_config_refusals_are_not_blindly_restarted(self):
        assert ("RestartPreventExitStatus="
                f"{fl.EXIT_CLASS['REFUSED_WRONG_HOST']}") in SERVICE
        assert "Restart=on-failure" in SERVICE

    def test_documented_smoke_invocation_uses_environment_file(self):
        # systemctl does not forward shell variables; the documented
        # smoke invocation must materialize the EnvironmentFile.
        assert "L1_SMOKE=--smoke systemctl" not in SERVICE
        assert "l1-factorial@101.env" in SERVICE


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
