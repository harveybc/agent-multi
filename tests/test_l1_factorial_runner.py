"""Runner contract tests for the L1 factorial screen.

Regression for the 2026-08-09 fleet loss: a never-eligible cell made
the validation pipeline RAISE, killing the seed's remaining cells with
no record. The runner must (a) demand the typed inactive-terminal
result from the pipeline and (b) persist the typed termination facts
into the cell record.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import l1_factorial_screen as runner  # noqa: E402


class _RecorderPipeline:
    """Stands in for the curriculum pipeline; captures the config."""

    captured: dict | None = None
    result: dict = {}

    def __init__(self, config):
        type(self).captured = dict(config)

    def run_pipeline(self, *, config, env_plugin, agent_plugin, mode):
        type(self).captured = dict(config)
        return dict(type(self).result)


def _fixture_contract(tmp_path: Path, seed: int) -> dict:
    contract = runner.load_contract()
    contract = json.loads(json.dumps(contract))  # deep copy
    contract["_contract_sha256"] = "c" * 64
    anchor = tmp_path / f"anchor{seed}.zip"
    anchor.write_bytes(b"anchor-bytes")
    contract["anchors"] = {str(seed): {
        "path": str(anchor),
        "sha256": hashlib.sha256(b"anchor-bytes").hexdigest(),
    }}
    contract["output_root"] = str(tmp_path / "out")
    return contract


def _run(tmp_path, monkeypatch, *, result: dict) -> dict:
    import pipeline_plugins.rl_pipeline_with_solvency_curriculum as curr

    seed = 101
    contract = _fixture_contract(tmp_path, seed)
    _RecorderPipeline.result = result
    monkeypatch.setattr(curr, "PipelinePlugin", _RecorderPipeline)
    monkeypatch.setattr(runner.d1, "_agent_plugin", lambda name: object())
    monkeypatch.setattr(runner.d1, "_git_rev", lambda repo: "deadbeef")
    return runner.run_cell("L1_N_M10", seed, contract=contract, smoke=True)


BASE_RESULT = {
    "curriculum": {"post_easy": {}},
    "best_model_path": None,
    "terminal_model_path": "/tmp/x/model.terminal.zip",
    "history": [{"epoch": 1}],
    "warm_start_transfer_evidence": {"policy_state_transferred": True},
    "nested_split_manifest": None,
}


def test_runner_demands_typed_inactive_terminal_result(tmp_path,
                                                       monkeypatch):
    _run(tmp_path, monkeypatch, result=dict(BASE_RESULT))
    config = _RecorderPipeline.captured
    assert config is not None
    assert config["inactive_terminal_is_typed_result"] is True


def test_record_carries_typed_termination_facts(tmp_path, monkeypatch):
    result = dict(BASE_RESULT)
    result["activity_stopped_without_eligible_checkpoint"] = True
    result["termination_cause"] = "activity-ineligible for 40 epochs"
    record = _run(tmp_path, monkeypatch, result=result)
    assert record["activity_stopped_without_eligible_checkpoint"] is True
    assert "activity-ineligible" in record["termination_cause"]
    on_disk = json.loads(next(
        (tmp_path / "out").rglob("l1_cell_record.json")).read_text())
    assert on_disk["activity_stopped_without_eligible_checkpoint"] is True
    assert on_disk["termination_cause"] == result["termination_cause"]


def test_active_cell_records_no_termination_cause(tmp_path, monkeypatch):
    record = _run(tmp_path, monkeypatch, result=dict(BASE_RESULT))
    assert record["activity_stopped_without_eligible_checkpoint"] is False
    assert record["termination_cause"] is None


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
