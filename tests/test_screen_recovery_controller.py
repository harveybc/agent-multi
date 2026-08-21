"""Adversarial tests for the persistent screen-recovery controller.

Post-outage order §B: stale PID, duplicate launch, incomplete report,
power loss between archive and retry, wrong GPU, wrong commit, existing
sidecar, completed-fixed + interrupted-plateau, repeated reboot. All
socket-free on temporary fixtures; nothing touches the live screen.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "screen_recovery_controller",
        REPO / "tools" / "screen_recovery_controller.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load()


CONTRACT = {"l1_patience": 60, "l1_patience_start_epoch": 40}


def _mk(rc, root, *, seed=101, arm="plateau", pid_alive=None, **over):
    base = dict(
        seed=seed, arm=arm, frozen_commit="93880beb",
        config_sha256="c" * 64, gpu_mask="GPU-mask-A",
        output_dir=str(root / f"seed{seed}_{arm}"),
        report_path=str(root / f"seed{seed}_{arm}_report.json"),
        log_path=str(root / f"seed{seed}.log"),
        contract=dict(CONTRACT), clock=lambda: 1000.0)
    base.update(over)
    if pid_alive is not None:
        base["pid_alive"] = pid_alive
    return rc.write_attempt_manifest(root / "attempts", **base)


class TestClassification:
    def test_absence_is_never_completion(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        out = rc.classify_attempt(m)
        assert out["state"] == rc.UNKNOWN
        assert "never completion" in out["detail"]

    def test_completed_requires_parseable_report(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        Path(json.loads(m.read_text())["report_path"]).write_text(
            '{"accepted": true}')
        assert rc.classify_attempt(m)["state"] == rc.COMPLETED

    def test_incomplete_report_is_not_completion(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        Path(doc["report_path"]).write_text('{"accepted": tru')  # cut
        out = rc.classify_attempt(m)
        assert out["state"] == rc.INTERRUPTED_NONRESUMABLE
        assert "not completion" in out["detail"]

    def test_failed_before_training(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        Path(doc["log_path"]).write_text(
            "usage: wp4_cpu_smoke.py\nerror: unrecognized arguments\n")
        assert rc.classify_attempt(m)["state"] == (
            rc.FAILED_BEFORE_TRAINING)

    def test_interrupted_after_epochs(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        Path(doc["log_path"]).write_text("[epoch  84/2000] L1 44/60\n")
        out = rc.classify_attempt(m)
        assert out["state"] == rc.INTERRUPTED_NONRESUMABLE
        assert "never resumed" in out["detail"]

    def test_stale_pid_does_not_count_as_active(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        rc.record_pid(m, 4242)
        # PID exists but cmdline belongs to someone else
        out = rc.classify_attempt(
            m, pid_alive=lambda pid, tok: False)
        assert out["state"] == rc.UNKNOWN
        # matching cmdline counts
        out2 = rc.classify_attempt(
            m, pid_alive=lambda pid, tok: pid == 4242 and "101" in tok)
        assert out2["state"] == rc.ACTIVE


class TestDuplicatesAndRetry:
    def test_duplicate_active_attempt_refuses(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        rc.record_pid(m, 999)
        alive = lambda pid, tok: pid == 999
        with pytest.raises(rc.RecoveryError, match="duplicate active"):
            _mk(rc, tmp_path, pid_alive=alive)

    def test_completed_arm_is_never_rerun(self, rc, tmp_path):
        m = _mk(rc, tmp_path, arm="fixed")
        Path(json.loads(m.read_text())["report_path"]).write_text("{}")
        with pytest.raises(rc.RecoveryError, match="never rerun"):
            _mk(rc, tmp_path, arm="fixed")

    def test_retry_requires_prior_preservation(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        Path(doc["log_path"]).write_text("[epoch  10/2000]\n")
        with pytest.raises(rc.RecoveryError, match="not yet preserved"):
            _mk(rc, tmp_path)

    def test_completed_fixed_plus_interrupted_plateau(self, rc,
                                                      tmp_path):
        """The ordered mixed scenario: fixed completed stays untouched;
        only the plateau arm becomes retryable."""
        f = _mk(rc, tmp_path, arm="fixed")
        Path(json.loads(f.read_text())["report_path"]).write_text("{}")
        p = _mk(rc, tmp_path, arm="plateau",
                log_path=str(tmp_path / "seed101_plateau.log"))
        Path(json.loads(p.read_text())["log_path"]).write_text(
            "[epoch  100/2000]\n")
        assert rc.classify_attempt(f)["state"] == rc.COMPLETED
        assert rc.classify_attempt(p)["state"] == (
            rc.INTERRUPTED_NONRESUMABLE)
        with pytest.raises(rc.RecoveryError, match="never rerun"):
            _mk(rc, tmp_path, arm="fixed")
        rc.preserve_interrupted(p, suffix="interrupted_power_T1451")
        m2 = _mk(rc, tmp_path, arm="plateau",
                 log_path=str(tmp_path / "seed101_plateau.log"))
        assert json.loads(m2.read_text())["attempt_id"] == 2


class TestJournaledPreservation:
    def test_power_loss_between_archive_and_retry(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        out_dir = Path(doc["output_dir"]); out_dir.mkdir()
        (out_dir / "partial.bin").write_bytes(b"x")
        log = Path(doc["log_path"]); log.write_text("[epoch  9/2000]\n")
        # First pass: archive the output dir, then CRASH before the log
        # rename by simulating it — perform a partial manual rename.
        out_dir.rename(out_dir.parent /
                       (out_dir.name + "_interrupted_power_T"))
        # Second pass completes idempotently.
        result = rc.preserve_interrupted(m, suffix="interrupted_power_T")
        notes = {r["dst"]: r["note"] for r in result["preserved"]}
        assert any("already archived" == n for n in notes.values())
        assert not log.exists()
        assert (log.parent / (log.name + "_interrupted_power_T")
                ).exists()
        assert json.loads(m.read_text())["preserved"] is True

    def test_active_and_completed_refuse_preservation(self, rc,
                                                      tmp_path):
        m = _mk(rc, tmp_path)
        Path(json.loads(m.read_text())["report_path"]).write_text("{}")
        with pytest.raises(rc.RecoveryError, match="COMPLETED"):
            rc.preserve_interrupted(m, suffix="s")

    def test_repeated_reboot_is_idempotent(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        Path(doc["log_path"]).write_text("[epoch  5/2000]\n")
        r1 = rc.preserve_interrupted(m, suffix="s1")
        r2 = rc.preserve_interrupted(m, suffix="s1")  # reboot again
        assert all(n["note"] == "already archived"
                   for n in r2["preserved"])
        m2 = _mk(rc, tmp_path)
        assert json.loads(m2.read_text())["attempt_id"] == 2


class TestLaunchPreconditions:
    def _ok_kw(self):
        return dict(git_head=lambda: "93880beb0abc",
                    gpu_masks_present=lambda: ["GPU-mask-A"],
                    expected_config_sha256="c" * 64)

    def test_wrong_commit_refuses(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        kw = self._ok_kw(); kw["git_head"] = lambda: "deadbeef0000"
        with pytest.raises(rc.RecoveryError, match="wrong commit"):
            rc.verify_launch_preconditions(m, **kw)

    def test_wrong_gpu_refuses(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        kw = self._ok_kw(); kw["gpu_masks_present"] = lambda: ["GPU-B"]
        with pytest.raises(rc.RecoveryError, match="wrong-GPU"):
            rc.verify_launch_preconditions(m, **kw)

    def test_config_hash_mismatch_refuses(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        kw = self._ok_kw(); kw["expected_config_sha256"] = "e" * 64
        with pytest.raises(rc.RecoveryError, match="config hash"):
            rc.verify_launch_preconditions(m, **kw)

    def test_existing_sidecar_refuses(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        out = Path(json.loads(m.read_text())["output_dir"])
        out.mkdir()
        (out / "best_model.plateau_lr_state.json").write_text("{}")
        with pytest.raises(rc.RecoveryError,
                           match="REFUSED_PLATEAU_RESUME"):
            rc.verify_launch_preconditions(m, **self._ok_kw())

    def test_model_artifact_refuses(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        out = Path(json.loads(m.read_text())["output_dir"])
        out.mkdir()
        (out / "best_model.zip").write_bytes(b"z")
        with pytest.raises(rc.RecoveryError, match="clean directory"):
            rc.verify_launch_preconditions(m, **self._ok_kw())

    def test_clean_preconditions_pass(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        rc.verify_launch_preconditions(m, **self._ok_kw())


class TestStatusAndUnit:
    def test_status_exposes_heartbeat_fields(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        Path(doc["log_path"]).write_text("[epoch  50/2000] L1 0/60\n")
        rc.record_pid(m, 77)
        rows = rc.status(m.parent, now=lambda: 1000.0 + 50 * 100)
        row = rows[0]
        assert row["state"] == rc.UNKNOWN or row["epoch"] == 50
        rows = rc.status(m.parent, now=lambda: 6000.0)
        assert rows[0]["epoch"] == 50
        assert {"attempt", "seed", "arm", "attempt_id", "state",
                "epoch", "gpu",
                "eta_seconds_to_patience_floor"} <= set(rows[0])

    def test_emitted_unit_is_proposal_only(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        text = rc.emit_persistent_unit(m)
        assert "NOT INSTALLED" in text
        assert "Activation boundary" in text
