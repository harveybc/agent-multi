"""Runner v2 contract tests (order §2/§3, findings 178-187).

The runner must: materialize exclusively through the system manifest,
bind the full execution identity into a schema-v2 record, publish that
record atomically, reuse a complete hash-valid record as
ALREADY_COMPLETE, refuse a corrupt record, and land each attempt in its
own content-addressed directory.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import l1_factorial_screen as runner  # noqa: E402

CLEAN_SOURCES = {
    "agent-multi": {"repo_root": "/repo/agent-multi",
                    "commit": "1" * 40, "dirty": False,
                    "dirty_entries": [], "dirty_untracked_digest": None},
    "gym-fx": {"repo_root": "/repo/gym-fx", "commit": "2" * 40,
               "dirty": False, "dirty_entries": [],
               "dirty_untracked_digest": None},
}


class _RecorderPipeline:
    captured: dict | None = None
    result: dict = {}

    def __init__(self, config):
        pass

    def run_pipeline(self, *, config, env_plugin, agent_plugin, mode):
        type(self).captured = dict(config)
        return dict(type(self).result)


def _contract(tmp_path: Path, seed: int) -> dict:
    contract = runner.load_contract()
    contract = json.loads(json.dumps(contract))
    contract["_contract_sha256"] = "c" * 64
    anchor = tmp_path / f"anchor{seed}.zip"
    anchor.write_bytes(b"anchor-bytes")
    contract["anchors"] = {str(seed): {
        "path": str(anchor),
        "sha256": hashlib.sha256(b"anchor-bytes").hexdigest(),
    }}
    contract["output_root"] = str(tmp_path / "out")
    return contract


def _manifest() -> dict:
    return {"schema": "agent_multi.system_manifest.v1",
            "_manifest_sha256": "ab" * 32,
            "_manifest_path": "/frozen/manifest.json",
            "plugins": {
                "agent_plugin": "sac_agent",
                "pipeline_plugin": "rl_pipeline_with_validation",
                "curriculum_pipeline_plugin":
                    "rl_pipeline_with_solvency_curriculum",
            }}


def _fake_materialize(tmp_path: Path):
    def materialize(contract, manifest, cell, seed, output_dir):
        spec = contract["cells"][cell]
        return {
            "asset": "ethusdt_4h",
            "initial_cash": 10_000.0,
            "save_model": str(Path(output_dir) / "model.zip"),
            "_identity": {
                "system_manifest_sha256": manifest["_manifest_sha256"],
                "system_manifest_path": manifest.get("_manifest_path"),
                "base_config_sha256": "bc" * 32,
                "data_sha256": "da" * 32,
                "data_rows": 18085,
                "data_time_bounds": {"first": "f", "last": "l"},
                "nested_split_contract_sha256": "ns" * 32,
                "observation": {
                    "observation_manifest_sha256": "ob" * 32,
                    "flattened_shape": [2688],
                },
                "env_asset": "ethusdt_4h",
                "asset_label": "ETHUSD",
                "initial_cash": 10_000.0,
                "cost_contract": {"commission": 0.0002},
                "anchor_sha256": contract["anchors"][str(seed)]["sha256"],
                "anchor_policy_tensor_sha256": "a9" * 32,
                "cell": cell,
                "cell_factors": {
                    "phase1_mode": spec["phase1_mode"],
                    "phase2_lr_multiplier": spec["phase2_lr_multiplier"],
                },
                "seed": seed,
                "metric_schema": contract.get("selection_metric"),
            },
        }
    return materialize


def _run(tmp_path, monkeypatch, *, result: dict | None = None,
         seed: int = 101, cell: str = "L1_N_M10") -> dict:
    import pipeline_plugins.rl_pipeline_with_solvency_curriculum as curr

    contract = _contract(tmp_path, seed)
    terminal = tmp_path / "model.terminal.zip"
    terminal.write_bytes(b"terminal-bytes")
    base_result = {
        "curriculum": {"post_easy": {"easy_epochs_run": 2}},
        "best_model_path": None,
        "terminal_model_path": str(terminal),
        "history": [{"epoch": 1}, {"epoch": 2}],
        "stop_reason": "activity_stop_no_eligible_checkpoint",
        "termination_cause": "activity-ineligible for 40 epochs",
        "activity_stopped_without_eligible_checkpoint": True,
        "warm_start_transfer_evidence": {"policy_state_transferred": True},
        "nested_split_manifest": None,
    }
    _RecorderPipeline.result = dict(result or base_result)
    monkeypatch.setattr(curr, "PipelinePlugin", _RecorderPipeline)
    monkeypatch.setattr(runner, "_agent_plugin", lambda name: object())
    monkeypatch.setattr(runner.sysid, "materialize_system_config",
                        _fake_materialize(tmp_path))
    monkeypatch.setattr(runner, "source_identities",
                        lambda: json.loads(json.dumps(CLEAN_SOURCES)))
    monkeypatch.setattr(runner, "_terminal_tensor_sha",
                        lambda path: "t" * 64)
    return runner.run_cell(cell, seed, contract=contract,
                           manifest=_manifest(), smoke=True), contract


def test_record_binds_full_execution_identity(tmp_path, monkeypatch):
    record, _ = _run(tmp_path, monkeypatch)
    assert record["schema"] == "agent_multi.l1_factorial_cell_record.v2"
    for field in ("system_manifest_sha256", "resolved_config_sha256",
                  "observation_manifest_sha256", "terminal_model_sha256",
                  "terminal_policy_tensor_sha256",
                  "phase1_requested_epochs", "phase2_requested_epochs",
                  "subject_code_identity", "initial_cash",
                  "cost_contract", "data_sha256", "cell_identity",
                  "stop_reason", "history_len"):
        assert record.get(field) not in (None, ""), field
    assert record["subject_code_identity"]["agent-multi"][
        "commit"] == "1" * 40
    assert record["terminal_model_sha256"] == hashlib.sha256(
        b"terminal-bytes").hexdigest()


def test_runner_demands_typed_inactive_terminal_result(tmp_path,
                                                       monkeypatch):
    _run(tmp_path, monkeypatch)
    config = _RecorderPipeline.captured
    assert config["inactive_terminal_is_typed_result"] is True


def test_record_publication_is_atomic_and_reused(tmp_path, monkeypatch):
    record, contract = _run(tmp_path, monkeypatch)
    cell_dir = Path(contract["output_root"]).expanduser() / \
        record["experiment_id"] / "seed101" / "L1_N_M10"
    record_path = cell_dir / "l1_cell_record.json"
    assert record_path.is_file()
    assert not record_path.with_name(
        record_path.name + ".tmp").exists()
    # Second invocation: ALREADY_COMPLETE reuse, no second attempt dir.
    attempts_before = sorted(cell_dir.glob("attempt-*"))
    reused = runner.run_cell("L1_N_M10", 101, contract=contract,
                             manifest=_manifest(), smoke=True)
    assert reused["_reuse"] == "ALREADY_COMPLETE"
    assert sorted(cell_dir.glob("attempt-*")) == attempts_before


def test_corrupt_record_is_refused_never_overwritten(tmp_path,
                                                     monkeypatch):
    record, contract = _run(tmp_path, monkeypatch)
    cell_dir = Path(contract["output_root"]).expanduser() / \
        record["experiment_id"] / "seed101" / "L1_N_M10"
    record_path = cell_dir / "l1_cell_record.json"
    original = record_path.read_text()
    record_path.write_text(original.replace(
        record["terminal_model_sha256"], "0" * 64))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner.run_cell("L1_N_M10", 101, contract=contract,
                        manifest=_manifest(), smoke=True)
    assert record_path.read_text() != original  # untouched corrupt state


def test_each_attempt_gets_its_own_directory(tmp_path, monkeypatch):
    record, contract = _run(tmp_path, monkeypatch)
    cell_dir = Path(contract["output_root"]).expanduser() / \
        record["experiment_id"] / "seed101" / "L1_N_M10"
    (cell_dir / "l1_cell_record.json").unlink()  # simulate lost record
    record2, _ = _run(tmp_path, monkeypatch)
    attempts = sorted(cell_dir.glob("attempt-*"))
    assert len(attempts) >= 2  # recovery landed in a NEW attempt dir


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
