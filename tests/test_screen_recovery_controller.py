"""Adversarial tests for the recovery controller (§B + REC-01..04)."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "screen_recovery_controller.py"
FULL = "a" * 40
OTHER = "b" * 40


def _load():
    spec = importlib.util.spec_from_file_location(
        "screen_recovery_controller", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rc():
    return _load()


CONTRACT = {"l1_patience": 60, "l1_patience_start_epoch": 40}


def _mk(rc, root, *, seed=101, arm="plateau", pid_alive=None,
        commit=FULL, **over):
    base = dict(
        seed=seed, arm=arm, frozen_commit=commit,
        config_sha256="c" * 64, gpu_mask="GPU-mask-A",
        output_dir=str(root / f"seed{seed}_{arm}"),
        report_path=str(root / f"seed{seed}_{arm}_report.json"),
        log_path=str(root / f"seed{seed}.log"),
        contract=dict(CONTRACT), argv=["/bin/true"], cwd=str(root),
        clock=lambda: 1000.0)
    base.update(over)
    if pid_alive is not None:
        base["pid_alive"] = pid_alive
    return rc.write_attempt_manifest(root / "attempts", **base)


def _semantic_report(*, seed=101, arm="plateau", accepted=True,
                     commit=FULL, config="c" * 64, **over):
    doc = {"schema": "agent_multi.wp4_smoke.v2", "accepted": accepted,
           "budgets": {"seed": seed},
           "arm_contract": {"scheduler_policy": arm},
           "commit": commit, "config_sha256": config,
           "stop_reason": "l1_early_stop"}
    doc.update(over)
    return doc


class TestSemanticCompletion:
    """AUD-F1-20260821-REC-02."""

    def test_empty_json_is_never_completed(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        Path(json.loads(m.read_text())["report_path"]).write_text("{}")
        out = rc.classify_attempt(m)
        assert out["state"] == rc.INTERRUPTED_NONRESUMABLE
        assert "semantically valid" in out["detail"]

    def test_typed_negative_is_never_completed(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        Path(json.loads(m.read_text())["report_path"]).write_text(
            json.dumps(_semantic_report(accepted=False)))
        assert rc.classify_attempt(m)["state"] == (
            rc.INTERRUPTED_NONRESUMABLE)

    @pytest.mark.parametrize("mutation", [
        {"budgets": {"seed": 202}},
        {"arm_contract": {"scheduler_policy": "fixed"}},
        {"commit": OTHER},
        {"config_sha256": "e" * 64},
        {"stop_reason": ""},
        {"schema": "someone_else.v1"},
    ])
    def test_foreign_or_terminal_less_report_refuses(self, rc,
                                                     tmp_path,
                                                     mutation):
        m = _mk(rc, tmp_path)
        doc = _semantic_report()
        doc.update(mutation)
        Path(json.loads(m.read_text())["report_path"]).write_text(
            json.dumps(doc))
        assert rc.classify_attempt(m)["state"] == (
            rc.INTERRUPTED_NONRESUMABLE)

    def test_semantically_valid_report_completes(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        Path(json.loads(m.read_text())["report_path"]).write_text(
            json.dumps(_semantic_report()))
        assert rc.classify_attempt(m)["state"] == rc.COMPLETED

    def test_absence_is_never_completion(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        out = rc.classify_attempt(m)
        assert out["state"] == rc.UNKNOWN
        assert "never completion" in out["detail"]

    def test_stale_pid_not_active(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        rc.record_pid(m, 4242)
        assert rc.classify_attempt(
            m, pid_alive=lambda p, t: False)["state"] == rc.UNKNOWN


class TestLaunchIdentityBinding:
    """AUD-F1-20260821-REC-03."""

    def _kw(self):
        return dict(git_head=lambda: FULL, git_dirty=lambda: False,
                    gpu_masks_present=lambda: ["GPU-mask-A"],
                    expected_config_sha256="c" * 64)

    def test_short_commit_refused_at_manifest_write(self, rc, tmp_path):
        with pytest.raises(rc.RecoveryError, match="40-hex"):
            _mk(rc, tmp_path, commit="93880beb")

    def test_short_git_head_refused_at_launch(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        kw = self._kw(); kw["git_head"] = lambda: FULL[:12]
        with pytest.raises(rc.RecoveryError, match="collision"):
            rc.verify_launch_preconditions(m, **kw)

    def test_dirty_tree_refused(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        kw = self._kw(); kw["git_dirty"] = lambda: True
        with pytest.raises(rc.RecoveryError, match="dirty"):
            rc.verify_launch_preconditions(m, **kw)

    def test_changed_argument_after_check_refuses(self, rc, tmp_path):
        """Check-to-launch substitution: the artifact is mutated after
        the manifest bound its hash."""
        m = _mk(rc, tmp_path)
        art = Path(json.loads(m.read_text())["launch_artifact"])
        payload = json.loads(art.read_text())
        payload["argv"] = ["/bin/false", "--evil"]
        art.write_text(json.dumps(payload, sort_keys=True))
        with pytest.raises(rc.RecoveryError, match="changed after"):
            rc.verify_launch_preconditions(m, **self._kw())

    def test_symlink_output_dir_refused(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        real = tmp_path / "elsewhere"; real.mkdir()
        out = Path(json.loads(m.read_text())["output_dir"])
        out.symlink_to(real)
        with pytest.raises(rc.RecoveryError, match="symlink"):
            rc.verify_launch_preconditions(m, **self._kw())

    def test_any_stale_file_refuses_not_only_models(self, rc,
                                                    tmp_path):
        m = _mk(rc, tmp_path)
        out = Path(json.loads(m.read_text())["output_dir"])
        out.mkdir()
        (out / "notes.txt").write_text("stale")
        with pytest.raises(rc.RecoveryError, match="ABSENT or EMPTY"):
            rc.verify_launch_preconditions(m, **self._kw())

    def test_clean_preconditions_pass(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        rc.verify_launch_preconditions(m, **self._kw())


class TestDurability:
    """AUD-F1-20260821-REC-04."""

    def test_manifest_write_fsyncs_parent_directory(self, rc,
                                                    tmp_path):
        synced = []
        _mk(rc, tmp_path, fsync_dir=lambda p: synced.append(str(p)))
        assert any(str(tmp_path / "attempts") == s for s in synced)

    def test_failed_directory_fsync_is_loud(self, rc, tmp_path):
        def boom(_p):
            raise OSError("fsync failed")
        with pytest.raises(OSError, match="fsync failed"):
            _mk(rc, tmp_path, fsync_dir=boom)

    def test_preservation_fsyncs_after_renames(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        Path(doc["log_path"]).write_text("[epoch  5/2000]\n")
        synced = []
        rc.preserve_interrupted(
            m, suffix="s1", fsync_dir=lambda p: synced.append(str(p)))
        assert synced


class TestSuperviseLifecycle:
    """AUD-F1-20260821-REC-01: the real, executing lifecycle."""

    def _writer_argv(self, report_path, *, accepted=True, seed=101,
                     arm="plateau"):
        rep = _semantic_report(seed=seed, arm=arm, accepted=accepted)
        script = ("import json,sys; json.dump(%r, open(%r, 'w'))"
                  % (rep, str(report_path)))
        return [sys.executable, "-c", script]

    def _kw(self):
        return dict(git_head=lambda: FULL, git_dirty=lambda: False,
                    gpu_masks_present=lambda: ["GPU-mask-A"],
                    expected_config_sha256="c" * 64,
                    poll_seconds=0.01)

    def test_supervise_launches_real_subprocess_to_completion(
            self, rc, tmp_path):
        report = tmp_path / "seed101_plateau_report.json"
        _mk(rc, tmp_path, argv=self._writer_argv(report),
            report_path=str(report))
        out = rc.supervise(tmp_path / "attempts", 101, "plateau",
                           **self._kw())
        assert out["terminal"] == rc.COMPLETED
        assert out["exit_code"] == 0
        assert out["action"] == "launched"
        # PID was recorded immediately and heartbeat existed
        m = sorted((tmp_path / "attempts").glob(
            "attempt_seed101_plateau_*.json"))[-1]
        assert json.loads(m.read_text())["pid"] is not None

    def test_supervise_failure_is_typed_never_completed(self, rc,
                                                        tmp_path):
        _mk(rc, tmp_path, argv=[sys.executable, "-c",
                                "import sys; sys.exit(7)"])
        out = rc.supervise(tmp_path / "attempts", 101, "plateau",
                           **self._kw())
        assert out["terminal"] in (rc.FAILED_BEFORE_TRAINING,
                                   rc.UNKNOWN)
        assert out["exit_code"] == 7

    def test_supervise_preserves_then_retries_interrupted(self, rc,
                                                          tmp_path):
        report = tmp_path / "seed101_plateau_report.json"
        m = _mk(rc, tmp_path, argv=self._writer_argv(report),
                report_path=str(report))
        doc = json.loads(m.read_text())
        Path(doc["log_path"]).write_text("[epoch  50/2000]\n")
        out = rc.supervise(tmp_path / "attempts", 101, "plateau",
                           **self._kw())
        assert out["terminal"] == rc.COMPLETED
        assert out["attempt"].endswith("_0002.json")
        assert (tmp_path / (
            "seed101.log_interrupted_nonresumable_auto")).exists()

    def test_supervise_never_reruns_completed(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        Path(json.loads(m.read_text())["report_path"]).write_text(
            json.dumps(_semantic_report()))
        out = rc.supervise(tmp_path / "attempts", 101, "plateau",
                           **self._kw())
        assert out["action"] == "none"
        assert out["terminal"] == rc.COMPLETED

    def test_supervise_without_manifest_refuses(self, rc, tmp_path):
        (tmp_path / "attempts").mkdir()
        with pytest.raises(rc.RecoveryError, match="never invents"):
            rc.supervise(tmp_path / "attempts", 101, "plateau",
                         **self._kw())


class TestExecutingGeneratedUnit:
    """REC-01 acceptance: the emitted unit's ExecStart line invokes a
    subcommand that EXISTS and executes end to end via the CLI."""

    @pytest.fixture()
    def git_repo(self, tmp_path):
        if shutil.which("git") is None:
            pytest.skip("git unavailable")
        repo = tmp_path / "repo"; repo.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@t"}
        for cmd in (["git", "init", "-q"],
                    ["git", "commit", "-q", "--allow-empty", "-m", "x"]):
            subprocess.run(cmd, cwd=repo, env=env, check=True)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True
                              ).stdout.strip()
        return repo, head

    def test_execstart_command_executes_end_to_end(self, rc, tmp_path,
                                                   git_repo):
        repo, head = git_repo
        masks = []
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=uuid",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True,
                                 timeout=10).stdout
            masks = [line.strip() for line in out.splitlines()
                     if line.strip()]
        except (OSError, subprocess.TimeoutExpired):
            pass
        if not masks:
            pytest.skip("no GPU enumeration on this host")
        report = tmp_path / "seed101_plateau_report.json"
        rep = _semantic_report(commit=head)
        script = ("import json; json.dump(%r, open(%r, 'w'))"
                  % (rep, str(report)))
        m = _mk(rc, tmp_path, commit=head, gpu_mask=masks[0],
                argv=[sys.executable, "-c", script],
                report_path=str(report), cwd=str(repo))
        unit = rc.emit_persistent_unit(m)
        assert "supervise" in unit
        # execute exactly the ExecStart invocation shape via the CLI
        cli = subprocess.run(
            [sys.executable, str(TOOL), "supervise",
             "--root", str(tmp_path / "attempts"),
             "--seed", "101", "--arm", "plateau",
             "--expected-config-sha256", "c" * 64,
             "--repo-dir", str(repo), "--poll-seconds", "0.01"],
            capture_output=True, text=True, timeout=60)
        assert cli.returncode == 0, cli.stderr[-500:]
        result = json.loads(cli.stdout)
        assert result["terminal"] == "completed"


class TestDuplicatesAndJournal:
    def test_duplicate_active_attempt_refuses(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        rc.record_pid(m, 999)
        alive = lambda pid, tok: pid == 999
        with pytest.raises(rc.RecoveryError, match="duplicate active"):
            _mk(rc, tmp_path, pid_alive=alive)

    def test_completed_arm_is_never_rerun(self, rc, tmp_path):
        m = _mk(rc, tmp_path, arm="fixed")
        Path(json.loads(m.read_text())["report_path"]).write_text(
            json.dumps(_semantic_report(arm="fixed")))
        with pytest.raises(rc.RecoveryError, match="never rerun"):
            _mk(rc, tmp_path, arm="fixed")

    def test_power_loss_between_archive_and_retry(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        out_dir = Path(doc["output_dir"]); out_dir.mkdir()
        (out_dir / "partial.bin").write_bytes(b"x")
        log = Path(doc["log_path"]); log.write_text("[epoch  9/2000]\n")
        out_dir.rename(out_dir.parent / (out_dir.name + "_s"))
        result = rc.preserve_interrupted(m, suffix="s")
        notes = [r["note"] for r in result["preserved"]]
        assert "already archived" in notes
        assert not log.exists()

    def test_repeated_reboot_is_idempotent(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        Path(doc["log_path"]).write_text("[epoch  5/2000]\n")
        rc.preserve_interrupted(m, suffix="s1")
        r2 = rc.preserve_interrupted(m, suffix="s1")
        assert all(n["note"] == "already archived"
                   for n in r2["preserved"])
        m2 = _mk(rc, tmp_path)
        assert json.loads(m2.read_text())["attempt_id"] == 2


class TestStatusAndUnit:
    def test_status_exposes_heartbeat_fields(self, rc, tmp_path):
        m = _mk(rc, tmp_path)
        doc = json.loads(m.read_text())
        Path(doc["log_path"]).write_text("[epoch  50/2000] L1 0/60\n")
        rows = rc.status(m.parent, now=lambda: 6000.0)
        assert {"attempt", "seed", "arm", "attempt_id", "state",
                "epoch", "gpu", "heartbeat_unix",
                "eta_seconds_to_patience_floor"} <= set(rows[0])

    def test_emitted_unit_references_real_subcommand(self, rc,
                                                     tmp_path):
        m = _mk(rc, tmp_path)
        text = rc.emit_persistent_unit(m)
        assert "NOT INSTALLED" in text
        assert " supervise --root" in text
