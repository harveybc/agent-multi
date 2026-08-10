"""WP1 launcher tests (order §2): socket-free concurrent
double-dispatch and crash-between-artifact-and-record recovery.

flock claims are per open-file-description, so two SeedLauncher
instances in one process genuinely conflict — no sockets, no GPUs.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import l1_fleet_launcher as fl  # noqa: E402
from tools import l1_factorial_screen as runner  # noqa: E402

CELLS = ("L1_N_M10", "L1_E_M10", "L1_N_M03", "L1_E_M03")


def make_contract(tmp_path: Path, hostname: str = "testhost") -> dict:
    contract = runner.load_contract()
    contract = json.loads(json.dumps(contract))
    contract["_contract_sha256"] = "c" * 64
    anchor = tmp_path / "anchor101.zip"
    anchor.write_bytes(b"anchor-bytes")
    contract["anchors"] = {"101": {
        "path": str(anchor),
        "sha256": hashlib.sha256(b"anchor-bytes").hexdigest()}}
    contract["assignments"] = {"101": {"hostname": hostname,
                                       "gpu_uuid": "GPU-test-uuid"}}
    contract["output_root"] = str(tmp_path / "out")
    return contract


MANIFEST = {"schema": "agent_multi.system_manifest.v1",
            "_manifest_sha256": "ab" * 32}


def make_launcher(contract, *, run_cell_fn,
                  hostname="testhost") -> fl.SeedLauncher:
    return fl.SeedLauncher(
        contract=contract, manifest=MANIFEST, seed=101, smoke=True,
        hostname=hostname, gpu_uuids=["GPU-test-uuid"],
        run_cell_fn=run_cell_fn)


def instant_cell(cell, seed, *, contract, manifest, smoke):
    exp_id = runner.experiment_identity(contract, manifest, smoke)
    cell_dir = (Path(contract["output_root"]).expanduser() / exp_id /
                f"seed{seed}" / cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    record = {"cell": cell, "seed": seed, "experiment_id": exp_id,
              "attempt_dir": str(cell_dir / "attempt-x-01"),
              "terminal_model_path": "/t.zip"}
    (cell_dir / "l1_cell_record.json").write_text(
        json.dumps(record))
    return record


class TestAssignmentEnforcement:
    def test_wrong_host_is_refused(self, tmp_path):
        contract = make_contract(tmp_path, hostname="otherhost")
        result = make_launcher(contract,
                               run_cell_fn=instant_cell).run()
        assert result["outcome"] == "REFUSED_WRONG_HOST"

    def test_missing_gpu_is_refused(self, tmp_path):
        contract = make_contract(tmp_path)
        launcher = fl.SeedLauncher(
            contract=contract, manifest=MANIFEST, seed=101, smoke=True,
            hostname="testhost", gpu_uuids=["GPU-something-else"],
            run_cell_fn=instant_cell)
        assert launcher.run()["outcome"] == "REFUSED_GPU_UNBOUND"

    def test_unassigned_seed_is_refused(self, tmp_path):
        contract = make_contract(tmp_path)
        contract["assignments"] = {}
        result = make_launcher(contract,
                               run_cell_fn=instant_cell).run()
        assert result["outcome"] == "REFUSED_BAD_CONTRACT"


class TestDoubleDispatch:
    def test_concurrent_double_dispatch_single_writer(self, tmp_path):
        contract = make_contract(tmp_path)
        entered = threading.Event()
        release = threading.Event()
        writer_calls = []

        def blocking_cell(cell, seed, *, contract, manifest, smoke):
            writer_calls.append(cell)
            entered.set()
            assert release.wait(30)
            return instant_cell(cell, seed, contract=contract,
                                manifest=manifest, smoke=smoke)

        first = make_launcher(contract, run_cell_fn=blocking_cell)
        results = {}

        def run_first():
            results["first"] = first.run()

        t = threading.Thread(target=run_first)
        t.start()
        assert entered.wait(30)
        # Second dispatch while the first holds the claim.
        second = make_launcher(contract, run_cell_fn=instant_cell)
        results["second"] = second.run()
        release.set()
        t.join(60)

        assert results["second"]["outcome"] == "ALREADY_RUNNING"
        assert results["second"]["holder"].get("pid")
        assert results["first"]["outcome"] == "SEED_COMPLETE"
        # The second invocation never wrote a cell.
        assert writer_calls == list(contract["cells"])

    def test_second_run_after_completion_is_already_complete(self,
                                                             tmp_path):
        contract = make_contract(tmp_path)
        assert make_launcher(
            contract, run_cell_fn=instant_cell).run()[
                "outcome"] == "SEED_COMPLETE"

        calls = []

        def reusing_cell(cell, seed, *, contract, manifest, smoke):
            calls.append(cell)
            record = json.loads(
                (Path(contract["output_root"]).expanduser() /
                 runner.experiment_identity(contract, manifest, smoke) /
                 f"seed{seed}" / cell /
                 "l1_cell_record.json").read_text())
            record["_reuse"] = "ALREADY_COMPLETE"
            return record

        result = make_launcher(contract, run_cell_fn=reusing_cell).run()
        assert result["outcome"] == "ALREADY_COMPLETE"
        assert result["reused_cells"] == list(contract["cells"])


class TestCrashRecovery:
    def test_crash_between_artifact_and_record_recovers_new_attempt(
            self, tmp_path, monkeypatch):
        """Attempt 1 writes artifacts then dies before the record; the
        relaunch must land in a NEW attempt and leave attempt 1 bytes
        untouched."""
        contract = make_contract(tmp_path)
        crashed = {}

        def crashing_cell(cell, seed, *, contract, manifest, smoke):
            exp_id = runner.experiment_identity(contract, manifest, smoke)
            cell_dir = (Path(contract["output_root"]).expanduser() /
                        exp_id / f"seed{seed}" / cell)
            attempt = cell_dir / "attempt-deadbeef-01"
            attempt.mkdir(parents=True)
            (attempt / "model.terminal.zip").write_bytes(b"artifact")
            crashed[cell] = attempt
            raise RuntimeError("simulated death before record publish")

        result = make_launcher(contract, run_cell_fn=crashing_cell).run()
        assert result["outcome"] == "SEED_FAILED"
        first_attempt = crashed["L1_N_M10"]
        before = (first_attempt / "model.terminal.zip").read_bytes()

        def recovering_cell(cell, seed, *, contract, manifest, smoke):
            exp_id = runner.experiment_identity(contract, manifest, smoke)
            cell_dir = (Path(contract["output_root"]).expanduser() /
                        exp_id / f"seed{seed}" / cell)
            attempt = runner._next_attempt_dir(cell_dir, "deadbeef")
            assert attempt != crashed.get(cell)
            (attempt / "model.terminal.zip").write_bytes(b"artifact-2")
            record = {"cell": cell, "seed": seed,
                      "experiment_id": exp_id,
                      "attempt_dir": str(attempt),
                      "terminal_model_path":
                          str(attempt / "model.terminal.zip")}
            runner.atomic_write_json(
                cell_dir / "l1_cell_record.json", record)
            return record

        result2 = make_launcher(contract,
                                run_cell_fn=recovering_cell).run()
        assert result2["outcome"] == "SEED_COMPLETE"
        # First attempt preserved byte-identical; record points at the
        # recovery attempt.
        assert (first_attempt / "model.terminal.zip").read_bytes() == \
            before
        exp_id = runner.experiment_identity(contract, MANIFEST, True)
        record = json.loads(
            (Path(contract["output_root"]).expanduser() / exp_id /
             "seed101" / "L1_N_M10" / "l1_cell_record.json").read_text())
        assert record["attempt_dir"] != str(first_attempt)

    def test_crashed_holder_frees_the_claim(self, tmp_path):
        """The kernel drops a dead holder's flock: after SEED_FAILED the
        claim must be reacquirable immediately."""
        contract = make_contract(tmp_path)

        def failing_cell(cell, seed, **kwargs):
            raise RuntimeError("boom")

        assert make_launcher(
            contract, run_cell_fn=failing_cell).run()[
                "outcome"] == "SEED_FAILED"
        assert make_launcher(
            contract, run_cell_fn=instant_cell).run()[
                "outcome"] == "SEED_COMPLETE"


class TestHeartbeat:
    def test_heartbeat_carries_identity_and_terminal_state(self,
                                                           tmp_path):
        contract = make_contract(tmp_path)
        make_launcher(contract, run_cell_fn=instant_cell).run()
        exp_id = runner.experiment_identity(contract, MANIFEST, True)
        hb = json.loads(
            (Path(contract["output_root"]).expanduser() / exp_id /
             "seed101" / "launcher_heartbeat.json").read_text())
        assert hb["seed"] == 101
        assert hb["pid"] and hb["pid_start_identity"]
        assert hb["terminal_state"] == "SEED_COMPLETE"
        assert hb["progress"].endswith("cells")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))  # noqa: F821
