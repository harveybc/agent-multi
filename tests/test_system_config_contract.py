"""WP8 mutation tests (findings 191-194) for the system materializer.

A zero-spread or unprotected 'normal_realistic' profile, a plugins
block that does not bind execution, or a manifest generated from a
dirty tree must all refuse before model construction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline_plugins import _system_config as sysid  # noqa: E402
from tools import l1_factorial_screen as runner  # noqa: E402


def real_manifest_fixture() -> tuple[dict, dict]:
    """A synthetic manifest whose file bindings are the REAL repo
    artifacts, so only the deliberately mutated part can refuse."""
    contract = runner.load_contract()
    base_rel = ("examples/results/project3_ethusdt_4h_sac_train_val_"
                "test_v2/config_out.json")
    base_path = REPO / base_rel
    base = json.loads(base_path.read_text())
    nested_rel = contract["nested_split_contract"]
    nested_path = REPO / nested_rel
    nested = json.loads(nested_path.read_text())
    data_path = Path(nested["source_csv"])
    obs = sysid.observation_manifest(base)
    manifest = {
        "schema": sysid.MANIFEST_SCHEMA,
        "_manifest_sha256": "ab" * 32,
        "_manifest_path": "/fixture/manifest.json",
        "asset": contract.get("asset"),
        "env_asset": contract.get("env_asset"),
        "data": {"path": str(data_path),
                 "sha256": nested["source_sha256"],
                 "rows": 18085, "date_column": "DATE_TIME",
                 "time_bounds": {"first": "f", "last": "l"}},
        "base_config": {"path": base_rel,
                        "sha256": sysid.sha_file(base_path)},
        "nested_split_contract": {"path": nested_rel,
                                  "sha256": sysid.sha_file(nested_path)},
        "splits": {"dates": {}, "evaluate_test_split": False},
        "observation": obs,
        "costs": {"config_bindings": {
            "commission": 0.0002,
            "full_spread_rate": 0.0001,
            "slippage": 0.0,
            "require_protected_entries": True,
            "min_equity": 100.0,
            "initial_cash": 10000.0,
            "leverage": 1.0,
        }},
        "plugins": {
            "env_plugin": base.get("env_plugin"),
            "strategy_plugin": base.get("strategy_plugin"),
            "agent_plugin": "sac_agent",
            "pipeline_plugin": "rl_pipeline_with_validation",
            "curriculum_pipeline_plugin":
                "rl_pipeline_with_solvency_curriculum",
        },
        "anchors": {seed: {"path": entry["path"],
                           "sha256": entry["sha256"],
                           "policy_tensor_sha256": "a9" * 32}
                    for seed, entry in contract["anchors"].items()},
        "source_identity_at_manifest": {
            "agent-multi": {"commit": "1" * 40, "dirty": False,
                            "dirty_untracked_digest": None},
        },
    }
    return contract, manifest


def materialize(contract, manifest, tmp_path):
    return sysid.materialize_system_config(
        contract, manifest, "L1_N_M10", 101, tmp_path / "out")


class TestNormalContractCompleteness:
    def test_reviewed_contract_is_applied(self, tmp_path):
        contract, manifest = real_manifest_fixture()
        config = materialize(contract, manifest, tmp_path)
        assert config["require_protected_entries"] is True
        assert config["full_spread_rate"] == 0.0001
        assert config["slippage"] == 0.0
        assert config["min_equity"] == 100.0
        assert config["_identity"]["plugins"][
            "curriculum_pipeline_plugin"] == \
            "rl_pipeline_with_solvency_curriculum"

    def test_zero_spread_normal_profile_is_refused(self, tmp_path):
        contract, manifest = real_manifest_fixture()
        manifest["costs"]["config_bindings"]["full_spread_rate"] = 0.0
        with pytest.raises(RuntimeError, match="zero-spread"):
            materialize(contract, manifest, tmp_path)

    def test_unprotected_normal_profile_is_refused(self, tmp_path):
        contract, manifest = real_manifest_fixture()
        manifest["costs"]["config_bindings"][
            "require_protected_entries"] = False
        with pytest.raises(RuntimeError, match="protected"):
            materialize(contract, manifest, tmp_path)

    def test_undeclared_slippage_is_refused(self, tmp_path):
        contract, manifest = real_manifest_fixture()
        del manifest["costs"]["config_bindings"]["slippage"]
        with pytest.raises(RuntimeError, match="slippage"):
            materialize(contract, manifest, tmp_path)

    def test_implicit_min_equity_is_refused(self, tmp_path):
        contract, manifest = real_manifest_fixture()
        del manifest["costs"]["config_bindings"]["min_equity"]
        with pytest.raises(RuntimeError, match="min_equity"):
            materialize(contract, manifest, tmp_path)


class TestPluginBinding:
    def test_missing_plugins_block_is_refused(self, tmp_path):
        contract, manifest = real_manifest_fixture()
        del manifest["plugins"]
        with pytest.raises(RuntimeError, match="plugins block"):
            materialize(contract, manifest, tmp_path)

    def test_env_plugin_drift_is_refused(self, tmp_path):
        contract, manifest = real_manifest_fixture()
        manifest["plugins"]["env_plugin"] = "some_other_env"
        with pytest.raises(RuntimeError, match="plugin drift"):
            materialize(contract, manifest, tmp_path)

    def test_unbound_curriculum_plugin_is_refused(self, tmp_path):
        contract, manifest = real_manifest_fixture()
        del manifest["plugins"]["curriculum_pipeline_plugin"]
        with pytest.raises(RuntimeError,
                           match="curriculum_pipeline_plugin"):
            materialize(contract, manifest, tmp_path)

    def test_runner_refuses_agent_not_bound_by_manifest(self, tmp_path):
        contract, manifest = real_manifest_fixture()
        with pytest.raises(RuntimeError, match="does not equal the "
                                               "manifest binding"):
            runner.run_cell("L1_N_M10", 101, contract=contract,
                            manifest=manifest, smoke=True,
                            agent_name="some_other_agent")


class TestManifestProvenance:
    def test_dirty_manifest_source_is_refused(self, tmp_path):
        contract, manifest = real_manifest_fixture()
        manifest["source_identity_at_manifest"]["agent-multi"] = {
            "commit": "1" * 40, "dirty": True,
            "dirty_untracked_digest": "f0" * 32}
        with pytest.raises(RuntimeError, match="DIRTY"):
            materialize(contract, manifest, tmp_path)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
