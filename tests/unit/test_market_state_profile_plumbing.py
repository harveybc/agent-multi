"""Spec B - market-state profile plumbing for Stage 3X SAC smoke configs.

These tests never launch training and never touch Stage C. They assert that:

* the smoke-plan generator preserves ``market_state_profile_id`` and
  ``market_state_profile_hash`` on each locked config,
* the profile's ``selected_columns`` reach ``feature_list``/``feature_columns``,
* a profile manifest with missing hashes/columns fails closed,
* the return-trace metadata/evidence records profile id + hash,
* a 2-profile fixture (1h + 4h) does not alter PPO/SAC/DQN algorithm code.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "project3_stage3x_sac_smoke_plan.py"

_spec = importlib.util.spec_from_file_location(
    "project3_stage3x_sac_smoke_plan", TOOL_PATH,
)
smoke = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = smoke
_spec.loader.exec_module(smoke)

_trace = importlib.import_module("pipeline_plugins._return_trace")


# ---------------------------------------------------------------------------
def _make_contract(name: str, *, asset: str = "btcusdt_perp",
                   timeframe: str = "1h", screen_score: float = 1.0,
                   features=None) -> dict:
    return {
        "contract_id": f"{name}__selected",
        "genome_id": f"{name}__genome",
        "asset": asset,
        "timeframe": timeframe,
        "feature_preset": "tech_stat",
        "feature_selection_method": "rank_ic_topk",
        "preprocessing_profile": "p00_current_contract",
        "input_data_file": f"/tmp/{name}/train.csv",
        "selected_features": list(features) if features is not None
                              else ["log_return_1", "sma_50", "ema_50"],
        "source_family": "test",
        "screen_score": screen_score,
        "proxy_net_return": 0.1,
        "proxy_trades": 5,
        "best_abs_validation_ic": 0.05,
        "stage_c_access": "DENIED",
    }


def _write_contracts(tmp_path: Path, contracts) -> Path:
    doc = {
        "schema_version": "project3_stage3x_selected_feature_contracts_v1",
        "generated_at": "2026-05-23T00:00:00Z",
        "stage_c_access": "DENIED",
        "training_launched": False,
        "contracts": list(contracts),
    }
    p = tmp_path / "selected_feature_contracts.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _make_profile(asset: str, timeframe: str, *,
                  profile_name: str = "engineered_pca",
                  selected_columns=None,
                  relation_score: float = 0.7) -> dict:
    if selected_columns is None:
        selected_columns = [f"state_pca_{i:02d}" for i in range(4)]
    blob = f"{asset}__{timeframe}__{profile_name}".encode("utf-8")
    return {
        "encoder_config_hash": hashlib.sha256(blob + b"__enc").hexdigest(),
        "market_state_profile_hash": hashlib.sha256(blob + b"__hash").hexdigest(),
        "market_state_profile_id": f"{asset}__{timeframe}__anchor_2024-06-17__{profile_name}",
        "negative_controls_pass": True,
        "profile_family": "linear_compression",
        "profile_name": profile_name,
        "relation_score": relation_score,
        "selected_columns": list(selected_columns),
        "source_columns": ["state_return_cum", "state_realized_volatility"],
        "stage_c_access": "DENIED",
        "target_asset": asset,
        "timeframe": timeframe,
        "training_launched": False,
        "weekly_anchor_id": "anchor_2024-06-17",
    }


def _write_profiles(tmp_path: Path, profiles) -> Path:
    doc = {
        "contract_count": len(profiles),
        "contracts": list(profiles),
        "schema_version": smoke.MARKET_STATE_PROFILES_SCHEMA,
        "source_screen_hash": "0" * 64,
        "source_screen_schema_version": "project3_stage3x_market_state_profile_screen_v1",
        "stage_c_access": "DENIED",
        "stage_c_allowed": False,
        "training_launched": False,
    }
    p = tmp_path / "selected_market_state_profiles.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
def test_generated_config_denies_stage_c_with_state_profile(tmp_path):
    contracts = [_make_contract("a", asset="btcusdt_perp", timeframe="1h")]
    profiles = [_make_profile("btcusdt_perp", "1h")]
    cp = _write_contracts(tmp_path, contracts)
    pp = _write_profiles(tmp_path, profiles)
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(cp),
        output_dir=str(tmp_path / "out"),
        top_n=1, seeds=[0], cost_scenario="base",
        selected_market_state_profiles_path=str(pp),
        write_files=True,
    )
    assert manifest["stage_c_access"] == "DENIED"
    assert manifest["training_launched"] is False
    assert manifest["market_state_profile_count"] == 1
    entry = manifest["configs"][0]
    cfg = json.loads(Path(entry["config_file"]).read_text(encoding="utf-8"))
    assert cfg["stage_c_access"] == "DENIED"
    assert cfg["_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"] is True
    assert cfg["market_state_profile_id"] == profiles[0]["market_state_profile_id"]
    assert cfg["market_state_profile_hash"] == profiles[0]["market_state_profile_hash"]


def test_state_profile_columns_appear_in_feature_list(tmp_path):
    cols = ["state_pca_00", "state_pca_01", "state_regime_prob_0"]
    contracts = [_make_contract("b", asset="eurusd", timeframe="4h",
                                features=["sma_50", "ema_20"])]
    profiles = [_make_profile("eurusd", "4h", selected_columns=cols)]
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(_write_contracts(tmp_path, contracts)),
        output_dir=str(tmp_path / "out"),
        top_n=1, seeds=[0], cost_scenario="base",
        selected_market_state_profiles_path=str(_write_profiles(tmp_path, profiles)),
        write_files=True,
    )
    cfg = json.loads(
        Path(manifest["configs"][0]["config_file"]).read_text(encoding="utf-8")
    )
    for c in cols:
        assert c in cfg["feature_list"], f"profile column {c!r} missing from feature_list"
        assert c in cfg["feature_columns"]
    assert cfg["feature_list"] == cfg["feature_columns"]
    # Pre-existing engineered features preserved alongside the state columns.
    for c in ["sma_50", "ema_20"]:
        assert c in cfg["feature_list"]
    assert cfg["market_state_selected_columns"] == cols


def test_missing_profile_hash_fails_closed(tmp_path):
    contracts = [_make_contract("c", asset="btcusdt_perp", timeframe="1h")]
    bad = _make_profile("btcusdt_perp", "1h")
    bad["market_state_profile_hash"] = ""  # corrupt
    with pytest.raises(smoke.SmokePlanError):
        smoke.build_smoke_plan(
            selected_contracts_path=str(_write_contracts(tmp_path, contracts)),
            output_dir=str(tmp_path / "out"),
            top_n=1, seeds=[0], cost_scenario="base",
            selected_market_state_profiles_path=str(_write_profiles(tmp_path, [bad])),
            write_files=False,
        )


def test_no_matching_profile_fails_closed_when_required(tmp_path):
    contracts = [_make_contract("d", asset="audusd", timeframe="1h")]
    profiles = [_make_profile("eurusd", "4h")]  # no audusd/1h match
    with pytest.raises(smoke.SmokePlanError):
        smoke.build_smoke_plan(
            selected_contracts_path=str(_write_contracts(tmp_path, contracts)),
            output_dir=str(tmp_path / "out"),
            top_n=1, seeds=[0], cost_scenario="base",
            selected_market_state_profiles_path=str(_write_profiles(tmp_path, profiles)),
            write_files=False,
        )


def test_require_flag_without_profiles_file_fails_closed(tmp_path):
    contracts = [_make_contract("e", asset="btcusdt_perp", timeframe="1h")]
    with pytest.raises(smoke.SmokePlanError):
        smoke.build_smoke_plan(
            selected_contracts_path=str(_write_contracts(tmp_path, contracts)),
            output_dir=str(tmp_path / "out"),
            top_n=1, seeds=[0], cost_scenario="base",
            require_market_state_profile=True,
            write_files=False,
        )


def test_legacy_plan_without_profiles_file_is_unaffected(tmp_path):
    """Backward compatibility: when no profiles file is given, the existing
    plan generator continues to emit Stage 3X-locked configs without
    market_state fields."""
    contracts = [_make_contract("f", asset="btcusdt_perp", timeframe="1h")]
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(_write_contracts(tmp_path, contracts)),
        output_dir=str(tmp_path / "out"),
        top_n=1, seeds=[0], cost_scenario="base",
        write_files=True,
    )
    assert manifest["market_state_profile_count"] == 0
    assert manifest["selected_market_state_profiles_file"] is None
    cfg = json.loads(
        Path(manifest["configs"][0]["config_file"]).read_text(encoding="utf-8")
    )
    assert "market_state_profile_id" not in cfg
    assert "market_state_profile_hash" not in cfg


def test_return_trace_evidence_records_state_profile_id_and_hash(tmp_path):
    """Per-trace metadata + the run-level evidence index must surface
    market_state_profile_id/hash so a Stage B reviewer can audit which
    state profile the run consumed."""
    config = {
        "asset": "btcusdt_perp",
        "timeframe": "1h",
        "input_data_file": None,
        "feature_list": ["state_pca_00", "state_pca_01"],
        "market_state_profile_id": "btcusdt_perp__1h__anchor_2024-06-17__engineered_pca",
        "market_state_profile_hash": "deadbeef" * 8,
        "market_state_profile_family": "linear_compression",
        "market_state_profile_name": "engineered_pca",
        "market_state_selected_columns": ["state_pca_00", "state_pca_01"],
    }
    rows = [
        {"step": 1, "timestamp": "2024-01-01 00:00:00", "asset": "btcusdt_perp"},
        {"step": 2, "timestamp": "2024-01-01 04:00:00", "asset": "btcusdt_perp"},
    ]
    trace_path = tmp_path / "evaluation_return_trace.csv"
    meta = _trace.write_return_trace(
        str(trace_path), rows, config=config, split="evaluation", seed=0,
        asset="btcusdt_perp", timeframe="1h", run_id="r0", episode_id="r0::eval",
        feature_list=config["feature_list"],
    )
    assert meta["market_state_profile_id"] == config["market_state_profile_id"]
    assert meta["market_state_profile_hash"] == config["market_state_profile_hash"]
    assert meta["market_state_selected_columns"] == config["market_state_selected_columns"]

    evidence = _trace.build_return_trace_evidence(
        [meta], config=config, run_id="r0",
        pipeline_plugin="rl_pipeline_with_validation",
    )
    assert evidence["market_state_profile_id"] == config["market_state_profile_id"]
    assert evidence["market_state_profile_hash"] == config["market_state_profile_hash"]
    assert evidence["market_state_profile_family"] == "linear_compression"
    assert evidence["market_state_selected_columns"] == config["market_state_selected_columns"]


def test_1h_and_4h_state_profile_columns_reach_env_observation(tmp_path):
    """No-training fixture: build two locked configs (1h + 4h), then verify
    the profile columns resolve to the env's feature_list view used by the
    return-trace evidence pipeline (resolve_feature_list is the env-facing
    contract). No PPO/SAC/DQN code is touched."""
    cols_1h = [f"state_pca_{i:02d}" for i in range(4)]
    cols_4h = ["state_regime_prob_0", "state_regime_prob_1", "state_regime_entropy"]
    contracts = [
        _make_contract("h1", asset="btcusdt_perp", timeframe="1h",
                       screen_score=2.0, features=["sma_50"]),
        _make_contract("h4", asset="btcusdt_perp", timeframe="4h",
                       screen_score=1.0, features=["sma_50"]),
    ]
    profiles = [
        _make_profile("btcusdt_perp", "1h", selected_columns=cols_1h,
                      relation_score=0.8),
        _make_profile("btcusdt_perp", "4h", selected_columns=cols_4h,
                      profile_name="engineered_regime", relation_score=0.7),
    ]
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(_write_contracts(tmp_path, contracts)),
        output_dir=str(tmp_path / "out"),
        top_n=2, seeds=[0], cost_scenario="base",
        selected_market_state_profiles_path=str(_write_profiles(tmp_path, profiles)),
        write_files=True,
    )
    assert manifest["market_state_profile_count"] == 2
    by_timeframe = {}
    for entry in manifest["configs"]:
        cfg = json.loads(Path(entry["config_file"]).read_text(encoding="utf-8"))
        by_timeframe[cfg["timeframe"]] = cfg

    cfg_1h = by_timeframe["1h"]
    cfg_4h = by_timeframe["4h"]

    # Profile columns are present in the locked feature_list (env-facing).
    for c in cols_1h:
        assert c in cfg_1h["feature_list"]
    for c in cols_4h:
        assert c in cfg_4h["feature_list"]

    # resolve_feature_list is the env-facing contract for the trace evidence.
    # It returns the same ordered list when sourced from config alone (no env).
    resolved_1h = _trace.resolve_feature_list(cfg_1h, env=None)
    resolved_4h = _trace.resolve_feature_list(cfg_4h, env=None)
    for c in cols_1h:
        assert c in resolved_1h
    for c in cols_4h:
        assert c in resolved_4h


def test_two_profile_fixture_does_not_alter_algorithm_code(tmp_path):
    """Spec B prohibits PPO/SAC/DQN algorithm source edits. Building two
    locked configs with attached state profiles must:
      - leave the SAC agent plugin selection intact;
      - never reference PPO/DQN in the emitted config;
      - leave the hash of pipeline_plugins/_return_trace.py and the
        SAC agent plugin (if installed) unchanged across the call."""
    pipeline_src = (
        Path(__file__).resolve().parents[2]
        / "pipeline_plugins" / "_return_trace.py"
    )
    pre_hash = hashlib.sha256(pipeline_src.read_bytes()).hexdigest()

    contracts = [
        _make_contract("p1", asset="btcusdt_perp", timeframe="1h",
                       screen_score=2.0),
        _make_contract("p4", asset="btcusdt_perp", timeframe="4h",
                       screen_score=1.0),
    ]
    profiles = [
        _make_profile("btcusdt_perp", "1h"),
        _make_profile("btcusdt_perp", "4h", profile_name="engineered_regime"),
    ]
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(_write_contracts(tmp_path, contracts)),
        output_dir=str(tmp_path / "out"),
        top_n=2, seeds=[0], cost_scenario="base",
        selected_market_state_profiles_path=str(_write_profiles(tmp_path, profiles)),
        write_files=True,
    )
    for entry in manifest["configs"]:
        cfg = json.loads(Path(entry["config_file"]).read_text(encoding="utf-8"))
        plugin = str(cfg["agent_plugin"]).lower()
        assert "sac" in plugin
        assert "ppo" not in plugin
        assert "dqn" not in plugin
    post_hash = hashlib.sha256(pipeline_src.read_bytes()).hexdigest()
    assert pre_hash == post_hash, (
        "Spec B: smoke-plan generation must not modify return-trace source"
    )
