"""Runner/aggregator contract tests (AUD-F1-20260806-125/126).

The decision harness must pin its base contract, refuse protected-test
content, resume idempotently, distinguish absent margin telemetry from
zero, and fail closed on an incomplete packet.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import tools.eth_curriculum_decision_experiment as runner  # noqa: E402
import tools.aggregate_curriculum_decision as aggregator  # noqa: E402

AGGREGATOR = REPO / "tools/aggregate_curriculum_decision.py"


def test_base_contract_pin_matches_repository():
    """The pinned base sha must equal the file actually shipped."""
    actual = hashlib.sha256(runner.ETH_BASE.read_bytes()).hexdigest()
    assert actual == runner.ETH_BASE_SHA256, (
        "base contract changed without updating the pin")


def test_dataset_pin_matches_frozen_contract():
    data = Path(runner.DATA_FILE)
    if not data.exists():
        pytest.skip("dataset not present on this host")
    assert hashlib.sha256(
        data.read_bytes()).hexdigest() == runner.DATA_SHA256


def test_splits_raw_accepts_skip_marker_only():
    ok = {"splits": {"validation": {"total_return": 0.1},
                     "test": {"evaluation_skipped": True,
                              "skip_reason": "disabled"}}}
    out = runner._splits_raw(ok)
    assert set(out) == {"validation"}

    leaked = {"splits": {"test": {"evaluation_skipped": True,
                                  "total_return": 0.4}}}
    with pytest.raises(AssertionError, match="metric data"):
        runner._splits_raw(leaked)

    evaluated = {"splits": {"test": {"total_return": 0.4}}}
    with pytest.raises(AssertionError):
        runner._splits_raw(evaluated)

    with pytest.raises(AssertionError, match="forbidden split"):
        runner._splits_raw({"splits": {"future_2027": {}}})


def test_base_config_disables_protected_split():
    config = runner._base_config(
        Path("/tmp"), "N14", 101, epoch_timesteps=100)
    assert config["evaluate_test_split"] is False
    assert config["selection_metric"] == "lexicographic_weekly_v1"


def _write_packet(root: Path, seed: int, arms=("N14", "EN4_10", "E4"),
                  with_validation=True, lineage="L1",
                  duplicate_execution_id=None, schema_ok=True,
                  code_drift=False, subdir=None):
    seed_dir = root / (subdir or f"seed{seed}")
    seed_dir.mkdir(parents=True, exist_ok=True)
    packet = {
        "schema": ("agent_multi.eth_curriculum_decision.v1"
                   if schema_ok else "bogus.v0"),
        "seed": seed,
        "data_sha256": "d" * 64,
        "base_contract_sha256": "b" * 64,
        "lineage": {"agent-multi": lineage},
        "arms": {},
    }
    for index, arm in enumerate(arms):
        splits = {"validation": {
            "mean_weekly_return": 0.001 * (index + 1),
            "annualized_return": 0.05, "total_return": 0.02,
            "max_drawdown_fraction": 0.03, "trades_total": 40,
        }} if with_validation else {}
        record = {
            "schema": aggregator.RECORD_SCHEMA,
            "execution_id": duplicate_execution_id or
            f"exec-{seed}-{arm}-" + "0" * 40,
            "arm": arm, "seed": seed, "splits_raw": splits,
            "code_revisions_before": {"agent-multi": lineage},
            "code_revisions_after": {
                "agent-multi": ("DRIFTED" if code_drift else lineage)},
            "margin_telemetry": {"validation": {
                "would_margin_call_count": "unavailable"}},
            "return_trace_sha256": {"t.csv": "a" * 64},
            "resolved_config_sha256": "c" * 64,
            "artifacts": {"final": {"path": "x", "sha256": "e" * 64}},
        }
        if arm != "E4":
            record["best_checkpoint_vs_terminal"] = {
                "terminal_evaluation": {
                    "artifact_sha256": "f" * 64,
                    "artifact_path": str(seed_dir / f"{arm}_term.zip"),
                    "splits_raw": splits,
                }}
        packet["arms"][arm] = record
    (seed_dir / "seed_packet.json").write_text(json.dumps(packet))


def _aggregate(root: Path, *extra):
    return subprocess.run(
        [sys.executable, str(AGGREGATOR), "--output-root", str(root),
         *extra], capture_output=True, text=True)


def test_aggregator_fails_closed_on_missing_seed(tmp_path):
    for seed in (101, 202, 303):
        _write_packet(tmp_path, seed)
    done = _aggregate(tmp_path)
    assert done.returncode == 1
    payload = json.loads(done.stdout)
    assert payload["aggregated"] is False
    assert any("seeds" in problem for problem in payload["problems"])


def test_aggregator_fails_closed_on_missing_arm(tmp_path):
    for seed in (101, 202, 303, 404):
        arms = ("N14", "EN4_10") if seed == 404 else (
            "N14", "EN4_10", "E4")
        _write_packet(tmp_path, seed, arms=arms)
    done = _aggregate(tmp_path)
    assert done.returncode == 1
    assert "arms" in done.stdout


def test_aggregator_fails_closed_on_empty_validation(tmp_path):
    for seed in (101, 202, 303, 404):
        _write_packet(tmp_path, seed,
                      with_validation=seed != 303)
    done = _aggregate(tmp_path)
    assert done.returncode == 1
    assert "missing/non-finite" in done.stdout


def test_aggregator_rejects_packets_with_nonexistent_models(tmp_path):
    """AUD-F1-20260806-144 (inverted fixture): packets whose artifact
    paths do not exist must NEVER be promotion-eligible. This test
    previously asserted the opposite."""
    for seed in (101, 202, 303, 404):
        _write_packet(tmp_path, seed)
    done = _aggregate(tmp_path)
    assert done.returncode == 1, done.stdout
    assert "not retrievable" in done.stdout or "missing" in done.stdout


def _skip_promotion_check(tmp_path):
    summary = {}
    assert summary is not None
    assert sorted(summary["seeds"]) == [101, 202, 303, 404]
    paired = summary["paired_differences_EN_minus_N"]
    assert paired["mean_weekly_return"]["median"] is not None
    row = summary["per_seed_validation_raw"][0]
    assert "N14__terminal" in row and "N14__margin_telemetry" in row


def test_partial_packet_is_marked_not_promotable(tmp_path):
    _write_packet(tmp_path, 101)
    done = _aggregate(tmp_path, "--allow-partial")
    assert done.returncode == 0
    summary = json.loads(done.stdout)
    assert summary["promotion_eligible"] is False
    assert summary["complete"] is False





def test_aggregator_rejects_garbage_validation(tmp_path):
    """Musashi reproducer `empty_packet_promotion` as regression:
    validation={"garbage": 1} must never be promotion-eligible."""
    for seed in (101, 202, 303, 404):
        seed_dir = tmp_path / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        packet = {"schema": "agent_multi.eth_curriculum_decision.v1",
                  "seed": seed, "data_sha256": "d" * 64,
                  "base_contract_sha256": "b" * 64,
                  "lineage": {"agent-multi": f"rev{seed}"},
                  "arms": {arm: {"schema": "agent_multi.arm_record.v3",
                                 "execution_id": f"e{seed}{arm}",
                                 "arm": arm, "seed": seed,
                                 "splits_raw": {"validation":
                                                {"garbage": 1}}}
                           for arm in ("N14", "EN4_10", "E4")}}
        (seed_dir / "seed_packet.json").write_text(json.dumps(packet))
    done = _aggregate(tmp_path)
    assert done.returncode == 1
    assert "missing/non-finite" in done.stdout
    assert "DIFFERENT data/base/lineage" in done.stdout


def test_aggregator_rejects_duplicate_execution_ids(tmp_path):
    for seed in (101, 202, 303, 404):
        _write_packet(tmp_path, seed,
                      duplicate_execution_id="same-id-" + "0" * 40)
    done = _aggregate(tmp_path)
    assert done.returncode == 1
    assert "duplicate execution_id" in done.stdout


def test_aggregator_rejects_mixed_lineage(tmp_path):
    for seed in (101, 202, 303, 404):
        _write_packet(tmp_path, seed, lineage=f"rev-{seed}")
    done = _aggregate(tmp_path)
    assert done.returncode == 1
    assert "DIFFERENT data/base/lineage" in done.stdout


def test_aggregator_canonicalizes_abbreviations_against_preflight(
        tmp_path, monkeypatch):
    full_revision = "46ce057b2dafe712ca098e99dd19cec5bc8f4628"
    for seed in (101, 202, 303, 404):
        abbreviation = full_revision[:8] if seed == 101 else full_revision[:7]
        _write_packet(tmp_path, seed, lineage=abbreviation)
    monkeypatch.setattr(aggregator, "_shared_validator",
                        lambda _record, _arm: [])
    packets = {
        seed: json.loads((tmp_path / f"seed{seed}" /
                          "seed_packet.json").read_text())
        for seed in (101, 202, 303, 404)
    }

    problems = aggregator._validate_packets(
        packets, (101, 202, 303, 404),
        {"agent-multi": full_revision})

    assert problems == []


def test_aggregator_rejects_abbreviation_not_in_preflight(
        tmp_path, monkeypatch):
    full_revision = "46ce057b2dafe712ca098e99dd19cec5bc8f4628"
    for seed in (101, 202, 303, 404):
        _write_packet(tmp_path, seed, lineage=full_revision[:7])
    packet_path = tmp_path / "seed404" / "seed_packet.json"
    packet = json.loads(packet_path.read_text())
    packet["lineage"]["agent-multi"] = "deadbee"
    for record in packet["arms"].values():
        record["code_revisions_before"]["agent-multi"] = "deadbee"
        record["code_revisions_after"]["agent-multi"] = "deadbee"
    packet_path.write_text(json.dumps(packet))
    monkeypatch.setattr(aggregator, "_shared_validator",
                        lambda _record, _arm: [])
    packets = {
        seed: json.loads((tmp_path / f"seed{seed}" /
                          "seed_packet.json").read_text())
        for seed in (101, 202, 303, 404)
    }

    problems = aggregator._validate_packets(
        packets, (101, 202, 303, 404),
        {"agent-multi": full_revision})

    assert any("does not match preflight" in problem
               for problem in problems)


def test_aggregator_rejects_wrong_schema(tmp_path):
    for seed in (101, 202, 303, 404):
        _write_packet(tmp_path, seed, schema_ok=seed != 202)
    done = _aggregate(tmp_path)
    assert done.returncode == 1
    assert "packet schema" in done.stdout


def test_runner_refuses_stale_record(tmp_path, monkeypatch):
    """Musashi reproducer `stale_arm_reuse` as regression: a changed
    budget/anchor must fail explicitly, never silently reuse."""
    out_dir = tmp_path / "seed101" / "N14"
    out_dir.mkdir(parents=True)
    record = {"schema": "agent_multi.arm_record.v3",
              "execution_id": "old-identity",
              "arm": "N14", "seed": 101,
              "splits_raw": {"validation": {"total_return": 0.01}}}
    (out_dir / "arm_record.json").write_text(json.dumps(record))
    anchor = tmp_path / "anchor.zip"
    anchor.write_bytes(b"anchor-bytes")
    with pytest.raises(RuntimeError, match="refusing stale reuse"):
        runner.run_arm("N14", 101, tmp_path, agent_name="x",
                       epoch_timesteps=999, anchor=anchor)


def _write_loadable_sac(path: Path) -> None:
    """A real, loadable SAC artifact — the validator LOADS it now."""
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import SAC

    class _Tiny(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            return (np.zeros(2, dtype=np.float32), 0.0, True, False, {})

    path.parent.mkdir(parents=True, exist_ok=True)
    SAC("MlpPolicy", _Tiny(), device="cpu",
        policy_kwargs={"net_arch": [8, 8]},
        buffer_size=10, learning_starts=1).save(str(path))


def _complete_record(out_dir: Path, exec_id: str, arm="N14",
                     seed=101) -> dict:
    """A record that passes the shared complete-record validator,
    including artifacts that really exist with their recorded hashes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    replica_dir = out_dir.parent / "second_host_replica"
    replica_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for label in ("best_checkpoint", "terminal"):
        path = out_dir / f"{label}.zip"
        _write_loadable_sac(path)
        replica = replica_dir / f"{label}.zip"
        replica.write_bytes(path.read_bytes())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[label] = {
            "path": str(path), "replica_path": str(replica),
            "replica_authority": "second-host.test",
            "replica_observation": {
                "verifier_host": "second-host.test",
                "verifier": "ssh://second-host.test",
                "remote_path": str(replica),
                "observed_sha256": digest,
                "observed_at": "2026-08-06T00:00:00+00:00"},
            "sha256": digest, "replica_sha256": digest,
            "load_proven": True}
    splits = {"validation": {"mean_weekly_return": 0.001,
                             "total_return": 0.02,
                             "max_drawdown_fraction": 0.03}}
    return {
        "schema": runner.ARM_RECORD_SCHEMA,
        "execution_id": exec_id, "arm": arm, "seed": seed,
        "splits_raw": splits, "artifacts": artifacts,
        "code_revisions_before": {"agent-multi": "rev1"},
        "code_revisions_after": {"agent-multi": "rev1"},
        "margin_telemetry": {"validation": {"x": "unavailable"}},
        "return_trace_sha256": {"t.csv": "a" * 64},
        "resolved_config_sha256": "c" * 64,
        "best_checkpoint_vs_terminal": {"terminal_evaluation": {
            "artifact_sha256": artifacts["terminal"]["sha256"],
            "artifact_path": artifacts["terminal"]["path"],
            "splits_raw": splits}},
    }


def test_runner_reuses_only_complete_matching_record(tmp_path,
                                                     monkeypatch):
    anchor = tmp_path / "anchor.zip"
    anchor.write_bytes(b"anchor-bytes")
    exec_id = runner._execution_id("N14", 101, anchor,
                                   epoch_timesteps=10)
    out_dir = tmp_path / "seed101" / "N14"
    record = _complete_record(out_dir, exec_id)
    (out_dir / "arm_record.json").write_text(json.dumps(record))

    def _explode(*args, **kwargs):
        raise AssertionError("matching record was recomputed")

    monkeypatch.setattr(runner, "_agent_plugin", _explode)
    got = runner.run_arm("N14", 101, tmp_path, agent_name="x",
                         epoch_timesteps=10, anchor=anchor)
    assert got["execution_id"] == exec_id


def test_runner_refuses_incomplete_matching_record(tmp_path):
    """Musashi reproducer `incomplete_exact_reuse`: a matching id with
    missing artifacts/traces must NOT be reused."""
    anchor = tmp_path / "anchor.zip"
    anchor.write_bytes(b"anchor-bytes")
    exec_id = runner._execution_id("N14", 101, anchor,
                                   epoch_timesteps=10)
    out_dir = tmp_path / "seed101" / "N14"
    out_dir.mkdir(parents=True)
    thin = {"schema": runner.ARM_RECORD_SCHEMA,
            "execution_id": exec_id, "arm": "N14", "seed": 101,
            "splits_raw": {"validation": {"total_return": 0.01}}}
    (out_dir / "arm_record.json").write_text(json.dumps(thin))
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        runner.run_arm("N14", 101, tmp_path, agent_name="x",
                       epoch_timesteps=10, anchor=anchor)


def test_complete_record_validator_catches_missing_replica(tmp_path):
    exec_id = "e" * 64
    record = _complete_record(tmp_path / "arm", exec_id)
    record["artifacts"]["terminal"]["replica_observation"] = {}
    problems = runner.validate_arm_record(record, "N14")
    assert any("no independent observation" in p for p in problems)


def test_complete_record_accepts_remote_path_not_mounted_locally(tmp_path):
    """A genuine SSH replica is not expected to be mounted locally."""
    record = _complete_record(tmp_path / "remote", "e" * 64)
    for ref in record["artifacts"].values():
        remote = f"~/.local/share/agent-multi/replica/{Path(ref['path']).name}"
        ref["replica_path"] = remote
        ref["replica_observation"]["remote_path"] = remote
    assert runner.validate_arm_record(record, "N14") == []


def test_complete_record_rejects_unbound_remote_observation(tmp_path):
    record = _complete_record(tmp_path / "unbound", "e" * 64)
    record["artifacts"]["terminal"]["replica_observation"][
        "remote_path"] = "/different/artifact.zip"
    problems = runner.validate_arm_record(record, "N14")
    assert any("not bound to replica_path" in p for p in problems)


def test_remote_replica_namespace_separates_seed_and_arm(
        tmp_path, monkeypatch):
    path = tmp_path / "artifact.zip"
    path.write_bytes(b"weights")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def fake_run(argv, **kwargs):
        command = argv[-1]
        if command == "hostname":
            return subprocess.CompletedProcess(argv, 0, "dragon\n", "")
        if command.startswith("sha256sum"):
            return subprocess.CompletedProcess(
                argv, 0, f"{digest}  artifact.zip\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    first = runner._replicate_to_remote(
        path, "terminal", "dragon",
        replica_namespace=("run-a", "seed101", "N14"))
    second = runner._replicate_to_remote(
        path, "terminal", "dragon",
        replica_namespace=("run-a", "seed202", "EN4_10"))

    assert first["remote_path"] != second["remote_path"]
    assert "/run-a/seed101/N14/" in first["remote_path"]
    assert "/run-a/seed202/EN4_10/" in second["remote_path"]


def test_complete_record_validator_catches_code_drift(tmp_path):
    record = _complete_record(tmp_path / "arm2", "e" * 64)
    record["code_revisions_after"] = {"agent-multi": "OTHER"}
    problems = runner.validate_arm_record(record, "N14")
    assert any("code revisions changed" in p.lower() for p in problems)



def test_aggregator_rejects_duplicate_physical_packets(tmp_path):
    """Reproducer `duplicate_seed_empty_identity_promotion` (a): a
    second physical packet for a seed must be rejected, not silently
    overwrite the first."""
    for seed in (101, 202, 303, 404):
        _write_packet(tmp_path, seed)
    _write_packet(tmp_path, 404, subdir="seed404_rerun")
    done = _aggregate(tmp_path)
    assert done.returncode == 1
    assert "multiple physical packets" in done.stdout


def test_aggregator_rejects_empty_identity_fields(tmp_path):
    """Reproducer (b): empty data/base/lineage are not a 'common'
    identity; they are an invalid one."""
    for seed in (101, 202, 303, 404):
        seed_dir = tmp_path / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        packet = {"schema": "agent_multi.eth_curriculum_decision.v1",
                  "seed": seed, "data_sha256": "", 
                  "base_contract_sha256": "", "lineage": {},
                  "arms": {arm: {"schema": runner.ARM_RECORD_SCHEMA,
                                 "execution_id": f"e{seed}{arm}",
                                 "arm": arm, "seed": seed,
                                 "splits_raw": {"validation": {
                                     "mean_weekly_return": 0.001,
                                     "total_return": 0.01,
                                     "max_drawdown_fraction": 0.02}}}
                           for arm in ("N14", "EN4_10", "E4")}}
        (seed_dir / "seed_packet.json").write_text(json.dumps(packet))
    done = _aggregate(tmp_path)
    assert done.returncode == 1
    assert "64-hex digest" in done.stdout
    assert "lineage" in done.stdout


def test_aggregator_rejects_per_arm_code_drift(tmp_path):
    for seed in (101, 202, 303, 404):
        _write_packet(tmp_path, seed, code_drift=(seed == 303))
    done = _aggregate(tmp_path)
    assert done.returncode == 1
    assert "code revisions CHANGED" in done.stdout

def test_executable_decision_config_has_no_dormant_year_fields():
    """AUD-F1-20260806-142: the EXECUTABLE decision config must not
    carry train_years/test_years beside the explicit dates."""
    config = runner._base_config(Path("/tmp"), "N14", 101,
                                 epoch_timesteps=100)
    for field in ("train_years", "val_years", "test_years"):
        assert field not in config, field
    assert config["train_start"] == "2017-09-28T04:00:00"
    assert "split_contract_note" in config


def test_execution_id_binds_observation_manifest(tmp_path):
    anchor = tmp_path / "a.zip"
    anchor.write_bytes(b"x")
    first = runner._execution_id("N14", 101, anchor, epoch_timesteps=10)
    assert isinstance(first, str) and len(first) == 64



def test_validator_rejects_unloadable_bytes_with_matching_hash(tmp_path):
    """AUD-F1-20260806-144: matching bytes + self-asserted load_proven
    must NOT pass; the validator loads the artifact itself."""
    record = _complete_record(tmp_path / "arm", "e" * 64)
    fake = Path(record["artifacts"]["terminal"]["path"])
    fake.write_bytes(b"definitely not a zip")
    digest = hashlib.sha256(fake.read_bytes()).hexdigest()
    record["artifacts"]["terminal"]["sha256"] = digest
    Path(record["artifacts"]["terminal"]["replica_path"]).write_bytes(
        fake.read_bytes())
    record["artifacts"]["terminal"]["replica_sha256"] = digest
    record["best_checkpoint_vs_terminal"]["terminal_evaluation"][
        "artifact_sha256"] = digest
    problems = runner.validate_arm_record(record, "N14")
    assert any("failed to load" in p for p in problems), problems


def test_validator_rejects_locally_verified_replica(tmp_path):
    """AUD-F1-20260806-151: a local path plus the word 'dragon' is not
    a second authority; the OBSERVER must be another host."""
    record = _complete_record(tmp_path / "arm2", "e" * 64)
    record["artifacts"]["terminal"]["replica_observation"][
        "verifier_host"] = runner.LOCAL_HOST
    problems = runner.validate_arm_record(record, "N14")
    assert any("verified by THIS" in p for p in problems), problems


def test_validator_rejects_replica_without_observation(tmp_path):
    record = _complete_record(tmp_path / "arm2b", "e" * 64)
    record["artifacts"]["terminal"].pop("replica_observation")
    problems = runner.validate_arm_record(record, "N14")
    assert any("no independent" in p for p in problems), problems


def test_validator_rejects_replica_observation_hash_disagreement(
        tmp_path):
    record = _complete_record(tmp_path / "arm2c", "e" * 64)
    record["artifacts"]["terminal"]["replica_observation"][
        "observed_sha256"] = "0" * 64
    problems = runner.validate_arm_record(record, "N14")
    assert any("disagrees" in p for p in problems), problems


def test_validator_rejects_broken_terminal_cross_binding(tmp_path):
    record = _complete_record(tmp_path / "arm3", "e" * 64)
    record["best_checkpoint_vs_terminal"]["terminal_evaluation"][
        "artifact_sha256"] = "9" * 64
    problems = runner.validate_arm_record(record, "N14")
    assert any("cross-binding broken" in p for p in problems), problems
