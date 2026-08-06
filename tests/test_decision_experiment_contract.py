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
                  duplicate_execution_id=None, schema_ok=True):
    seed_dir = root / f"seed{seed}"
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
            "schema": "agent_multi.arm_record.v3",
            "execution_id": duplicate_execution_id or
            f"exec-{seed}-{arm}-" + "0" * 40,
            "arm": arm, "seed": seed, "splits_raw": splits,
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


def test_aggregator_complete_packet_is_promotion_eligible(tmp_path):
    for seed in (101, 202, 303, 404):
        _write_packet(tmp_path, seed)
    done = _aggregate(tmp_path)
    assert done.returncode == 0, done.stdout
    summary = json.loads(done.stdout)
    assert summary["promotion_eligible"] is True
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


def test_runner_reuses_only_matching_execution_id(tmp_path, monkeypatch):
    anchor = tmp_path / "anchor.zip"
    anchor.write_bytes(b"anchor-bytes")
    exec_id = runner._execution_id("N14", 101, anchor,
                                   epoch_timesteps=10)
    out_dir = tmp_path / "seed101" / "N14"
    out_dir.mkdir(parents=True)
    record = {"schema": "agent_multi.arm_record.v3",
              "execution_id": exec_id, "arm": "N14", "seed": 101,
              "splits_raw": {"validation": {"total_return": 0.01}}}
    (out_dir / "arm_record.json").write_text(json.dumps(record))

    def _explode(*args, **kwargs):
        raise AssertionError("matching record was recomputed")

    monkeypatch.setattr(runner, "_agent_plugin", _explode)
    got = runner.run_arm("N14", 101, tmp_path, agent_name="x",
                         epoch_timesteps=10, anchor=anchor)
    assert got["execution_id"] == exec_id
