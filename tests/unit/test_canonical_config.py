from __future__ import annotations

import json
import sys

import pytest

from app.canonical_config import (
    ConfigResolutionError,
    load_json_object,
    resolve_config,
    translate_legacy_config,
    write_json_file,
)
from app.config import DEFAULT_VALUES


def test_legacy_config_preserves_unclassified_values() -> None:
    legacy = {
        "agent_plugin": "project3_sac_actor_critic_agent",
        "total_timesteps": 123,
        "custom_plugin_parameter": {"alpha": 0.7},
    }
    resolution = resolve_config(DEFAULT_VALUES, file_config=legacy)
    assert resolution.runtime["agent_plugin"] == "project3_sac_actor_critic_agent"
    assert resolution.runtime["total_timesteps"] == 123
    assert resolution.runtime["custom_plugin_parameter"] == {"alpha": 0.7}
    assert resolution.canonical.experiment["legacy_flat"]["custom_plugin_parameter"] == {
        "alpha": 0.7
    }


def test_precedence_is_defaults_base_file_cli_candidate() -> None:
    base = translate_legacy_config({"total_timesteps": 200}, name="base")
    file_config = {
        "schema_version": "trading_experiment.v1",
        "training": {"total_timesteps": 300},
        "risk": {"rel_volume": 0.05},
    }
    without_candidate = resolve_config(
        DEFAULT_VALUES,
        base_profile=base,
        file_config=file_config,
        cli_overrides={"total_timesteps": 400},
    )
    patch = {
        "schema_version": "candidate_genome_patch.v1",
        "base_config_hash": without_candidate.canonical.canonical_hash,
        "genes": {"/training/total_timesteps": 500},
    }
    resolved = resolve_config(
        DEFAULT_VALUES,
        base_profile=base,
        file_config=file_config,
        cli_overrides={"total_timesteps": 400},
        candidate_patch=patch,
    )
    assert resolved.runtime["total_timesteps"] == 500
    assert resolved.runtime["rel_volume"] == 0.05
    assert resolved.manifest["candidate_patch_hash"].startswith("sha256:")


def test_candidate_patch_rejects_wrong_base_and_unknown_path() -> None:
    config = {
        "schema_version": "trading_experiment.v1",
        "risk": {"rel_volume": 0.05},
    }
    with pytest.raises(ConfigResolutionError, match="base_config_hash mismatch"):
        resolve_config(
            DEFAULT_VALUES,
            file_config=config,
            candidate_patch={
                "base_config_hash": "sha256:" + "0" * 64,
                "genes": {"/risk/rel_volume": 0.1},
            },
        )

    baseline = resolve_config(DEFAULT_VALUES, file_config=config)
    with pytest.raises(ConfigResolutionError, match="path does not exist"):
        resolve_config(
            DEFAULT_VALUES,
            file_config=config,
            candidate_patch={
                "base_config_hash": baseline.canonical.canonical_hash,
                "genes": {"/risk/unknown": 0.1},
            },
        )


def test_unknown_schema_and_embedded_secret_fail_closed() -> None:
    with pytest.raises(ConfigResolutionError, match="unsupported config schema"):
        resolve_config(DEFAULT_VALUES, file_config={"schema_version": "trading_experiment.v2"})

    with pytest.raises(ConfigResolutionError, match="embedded secrets"):
        resolve_config(
            DEFAULT_VALUES,
            file_config={
                "schema_version": "trading_experiment.v1",
                "deployment": {"api_key": "secret-value"},
            },
        )


def test_json_file_io_is_object_only_and_atomic(tmp_path) -> None:
    path = tmp_path / "nested" / "config.json"
    write_json_file(path, {"b": 2, "a": 1})
    assert load_json_object(path) == {"a": 1, "b": 2}
    assert not path.with_name("config.json.tmp").exists()

    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigResolutionError, match="root must be an object"):
        load_json_object(array_path)


def test_resolution_hash_is_stable_across_mapping_order() -> None:
    left = resolve_config(
        DEFAULT_VALUES,
        file_config={
            "schema_version": "trading_experiment.v1",
            "data": {"asset": "fx:EUR/USD", "timeframe": "4h"},
        },
    )
    right = resolve_config(
        DEFAULT_VALUES,
        file_config={
            "data": {"timeframe": "4h", "asset": "fx:EUR/USD"},
            "schema_version": "trading_experiment.v1",
        },
    )
    assert left.canonical.canonical_hash == right.canonical.canonical_hash
    assert json.dumps(left.manifest, sort_keys=True) == json.dumps(right.manifest, sort_keys=True)


def test_generic_section_keys_remain_namespaced() -> None:
    resolution = resolve_config(
        DEFAULT_VALUES,
        file_config={
            "schema_version": "trading_experiment.v1",
            "rush_detector": {"enabled": False},
            "optimization": {"enabled": True},
            "olap": {"enabled": True},
        },
    )
    assert resolution.runtime["use_optimizer"] is True
    assert resolution.runtime["rush_detector_config"]["enabled"] is False
    assert resolution.runtime["olap_config"]["enabled"] is True
    assert "enabled" not in resolution.runtime


def test_doin_asset_profile_resolves_risk_and_optimization_owners() -> None:
    profile = load_json_object(
        "examples/config/doin/trading_asset_solusdt_4h_sac_v1.json"
    )
    resolution = resolve_config(DEFAULT_VALUES, file_config=profile)
    assert resolution.runtime["asset"] == "SOLUSDT"
    assert resolution.runtime["optimizer_plugin"] == "default_optimizer"
    assert resolution.runtime["optimization_metric"] == "risk_adjusted_return"
    assert resolution.runtime["rel_volume"] == 0.1
    assert resolution.runtime["k_sl"] == 1.5
    assert resolution.runtime["k_tp"] == 2.0


def test_main_accepts_config_alias_and_writes_lineage(tmp_path, monkeypatch) -> None:
    from app import main as main_module

    source = tmp_path / "legacy.json"
    source.write_text(
        json.dumps({"mode": "train", "total_timesteps": 12}),
        encoding="utf-8",
    )
    resolved = tmp_path / "resolved.json"
    manifest = tmp_path / "manifest.json"
    results = tmp_path / "results.json"
    flat = tmp_path / "flat.json"

    monkeypatch.setattr(
        main_module,
        "_run",
        lambda config: {"observed_timesteps": config["total_timesteps"]},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-multi",
            "--config",
            str(source),
            "--total_timesteps",
            "34",
            "--resolved_config_file",
            str(resolved),
            "--config_manifest_file",
            str(manifest),
            "--results_file",
            str(results),
            "--save_config",
            str(flat),
            "--quiet_mode",
        ],
    )
    assert main_module.main() == 0

    resolved_payload = json.loads(resolved.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    results_payload = json.loads(results.read_text(encoding="utf-8"))
    assert resolved_payload["schema_version"] == "trading_experiment.v1"
    assert resolved_payload["training"]["total_timesteps"] == 34
    assert manifest_payload["source_kinds"]["file_config"] == "legacy"
    assert manifest_payload["resolved_config_hash"] == results_payload["canonical_config_hash"]
    assert results_payload["observed_timesteps"] == 34
    assert flat.exists()
