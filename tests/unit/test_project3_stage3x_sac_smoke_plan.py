"""Unit tests for the Project 3 Stage 3X SAC smoke-plan generator.

These tests never launch training and never touch Stage C. They construct
small synthetic ``selected_feature_contracts.json`` documents and assert
that the locked configs the tool emits satisfy the spec.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
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


def _write_contracts(
    tmp_path: Path,
    *,
    contracts=None,
    stage_c_access: str = "DENIED",
    training_launched: bool = False,
) -> Path:
    if contracts is None:
        contracts = [_make_contract("alpha", screen_score=1.0)]
    doc = {
        "schema_version": "project3_stage3x_selected_feature_contracts_v1",
        "generated_at": "2026-05-14T00:00:00Z",
        "stage_c_access": stage_c_access,
        "training_launched": training_launched,
        "contracts": contracts,
    }
    path = tmp_path / "selected_feature_contracts.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _make_contract(name: str, *, screen_score: float = 1.0, features=None,
                   preprocessing: dict | None = None) -> dict:
    base = {
        "contract_id": f"{name}__selected",
        "genome_id": f"{name}__genome",
        "asset": "ethusdt",
        "timeframe": "4h",
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
    if preprocessing:
        base.update(preprocessing)
    return base


def test_generated_config_denies_stage_c(tmp_path):
    contracts_path = _write_contracts(tmp_path)
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(contracts_path),
        output_dir=str(tmp_path / "out"),
        top_n=1,
        seeds=[0, 1, 2],
        cost_scenario="base",
        write_files=True,
    )
    assert manifest["stage_c_access"] == "DENIED"
    assert manifest["final_stage_c_evaluation"] is False
    assert manifest["stage_c_acknowledged"] is False
    for entry in manifest["configs"]:
        cfg = json.loads(Path(entry["config_file"]).read_text(encoding="utf-8"))
        assert cfg["stage_c_access"] == "DENIED"
        assert cfg["final_stage_c_evaluation"] is False
        assert cfg["stage_c_acknowledged"] is False
        assert cfg["_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"] is True
        assert cfg["_project3_stage3x_sac_smoke"] is True


def test_missing_feature_list_fails_closed(tmp_path):
    bad = _make_contract("bad", features=[])
    contracts_path = _write_contracts(tmp_path, contracts=[bad])
    with pytest.raises(smoke.SmokePlanError):
        smoke.build_smoke_plan(
            selected_contracts_path=str(contracts_path),
            output_dir=str(tmp_path / "out"),
            top_n=1,
            seeds=[0],
            cost_scenario="base",
            write_files=False,
        )


def test_selected_preprocessing_fields_preserved(tmp_path):
    preprocessing = {
        "broker_profile": "crypto_exchange_spot",
        "market_type": "crypto_spot",
        "regulatory_profile": "none_or_external",
        "trade_rate_band_id": "crypto_exchange_spot_4h",
        "scaling_mode": "robust",
        "feature_scaling_window": 256,
        "feature_clip": 6.0,
        "window_size": 48,
        "split_anchor": "end",
        "train_days": 14,
        "val_days": 7,
        "test_days": 7,
        "min_split_rows": 30,
        "stage_b_force_close_obs": True,
        "force_close_window_bars": 12,
        "force_close_penalty_coef": 0.5,
        "learning_rate": 0.0003,
        "buffer_size": 20000,
        "learning_starts": 128,
        "batch_size": 384,
        "gamma": 0.985,
        "tau": 0.01,
        "train_freq": 2,
        "gradient_steps": 2,
        "ent_coef": "auto_0.1",
        "target_entropy": "auto",
        "use_sde": False,
        "net_arch": [128, 128],
        "continuous_action_threshold": 0.2,
        "epoch_timesteps": 1000,
        "max_epochs": 8,
        "l1_patience": 7,
        "l1_min_delta": 0.0,
        "total_timesteps": 5000,
    }
    contract = _make_contract("preproc", preprocessing=preprocessing)
    contracts_path = _write_contracts(tmp_path, contracts=[contract])
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(contracts_path),
        output_dir=str(tmp_path / "out"),
        top_n=1,
        seeds=[0],
        cost_scenario="base",
        write_files=True,
    )
    cfg_path = Path(manifest["configs"][0]["config_file"])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key, value in preprocessing.items():
        assert cfg[key] == value, f"preprocessing field {key} not preserved"


def test_force_close_observation_fields_enabled_by_default(tmp_path):
    contracts_path = _write_contracts(tmp_path)
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(contracts_path),
        output_dir=str(tmp_path / "out"),
        top_n=1,
        seeds=[0],
        cost_scenario="base",
        write_files=True,
    )
    cfg_path = Path(manifest["configs"][0]["config_file"])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["stage_b_force_close_obs"] is True
    assert cfg["force_close_dow"] == 4
    assert cfg["force_close_hour"] == 20
    assert cfg["force_close_window_hours"] == 4
    assert cfg["monday_entry_window_hours"] == 4


def test_smoke_plan_does_not_launch_training(tmp_path):
    contracts_path = _write_contracts(tmp_path)
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(contracts_path),
        output_dir=str(tmp_path / "out"),
        top_n=1,
        seeds=[0, 1, 2],
        cost_scenario="base",
        write_files=True,
    )
    assert manifest["training_launched"] is False
    for entry in manifest["configs"]:
        run_dir = Path(entry["run_dir"])
        # run_dir is referenced but must not exist as a launched run
        # (we only created the ``runs/`` parent for path determinism).
        assert not (run_dir / "policy.zip").exists()
        assert not (run_dir / "results.json").exists()
        cfg = json.loads(Path(entry["config_file"]).read_text(encoding="utf-8"))
        assert cfg["_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"] is True
        assert cfg.get("save_model", "").endswith("policy.zip")


def test_evidence_path_contract_present(tmp_path):
    contracts_path = _write_contracts(tmp_path)
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(contracts_path),
        output_dir=str(tmp_path / "out"),
        top_n=1,
        seeds=[0],
        cost_scenario="base",
        write_files=True,
    )
    entry = manifest["configs"][0]
    cfg = json.loads(Path(entry["config_file"]).read_text(encoding="utf-8"))
    assert cfg["return_trace_dir"].endswith("return_traces")
    assert cfg["return_trace_file"].endswith("evaluation_return_trace.csv")
    expected_evidence = entry["expected_evidence_file"]
    assert expected_evidence.endswith("return_traces/evidence.json")
    assert cfg["progress_file"].endswith("training_progress.json")
    assert cfg["training_progress_file"] == cfg["progress_file"]


def test_generated_configs_use_sac_only(tmp_path):
    contracts = [
        _make_contract("alpha", screen_score=2.0),
        _make_contract("beta", screen_score=1.0),
    ]
    contracts_path = _write_contracts(tmp_path, contracts=contracts)
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(contracts_path),
        output_dir=str(tmp_path / "out"),
        top_n=2,
        seeds=[0, 1, 2],
        cost_scenario="base",
        write_files=True,
    )
    # 2 contracts x 3 seeds = 6
    assert manifest["config_count"] == 6
    assert manifest["rules"]["sac_only"] is True
    for entry in manifest["configs"]:
        cfg = json.loads(Path(entry["config_file"]).read_text(encoding="utf-8"))
        plugin = str(cfg["agent_plugin"]).lower()
        assert "sac" in plugin
        assert "ppo" not in plugin
        assert "dqn" not in plugin
        assert cfg["_cost_scenario"] == "base"
        assert cfg["feature_list"], "feature_list must be non-empty"
        assert cfg["feature_columns"] == cfg["feature_list"]


def test_targeted_followup_can_emit_three_cost_scenarios(tmp_path):
    contracts_path = _write_contracts(tmp_path)
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(contracts_path),
        output_dir=str(tmp_path / "out"),
        top_n=1,
        seeds=[0, 1],
        cost_scenarios=["base", "plus_50pct", "plus_100pct"],
        write_files=True,
    )
    assert manifest["config_count"] == 6
    assert manifest["cost_scenarios"] == ["base", "plus_50pct", "plus_100pct"]
    assert manifest["rules"]["smoke_only_base_cost"] is False
    assert manifest["rules"]["targeted_cost_followup"] is True

    by_cost = {}
    for entry in manifest["configs"]:
        cfg = json.loads(Path(entry["config_file"]).read_text(encoding="utf-8"))
        by_cost[entry["cost_scenario"]] = cfg
        assert cfg["_cost_scenario"] == entry["cost_scenario"]
        assert cfg["stage_c_access"] == "DENIED"

    assert by_cost["base"]["commission"] == 0.0002
    assert by_cost["plus_50pct"]["commission"] == pytest.approx(0.0003)
    assert by_cost["plus_100pct"]["commission"] == 0.0004
    assert by_cost["base"]["_cost_multiplier"] == 1.0
    assert by_cost["plus_50pct"]["_cost_multiplier"] == 1.5
    assert by_cost["plus_100pct"]["_cost_multiplier"] == 2.0


def test_short_history_uses_end_anchor_and_two_train_years(tmp_path):
    data = tmp_path / "short_history.csv"
    dates = pd.date_range("2019-09-29 20:00:00", "2023-12-31 20:00:00", freq="4h")
    data.write_text(
        "DATE_TIME,CLOSE,log_return_1,sma_50,ema_50\n"
        + "\n".join(f"{ts},100,0.0,1.0,1.0" for ts in dates)
        + "\n",
        encoding="utf-8",
    )
    contract = _make_contract("short")
    contract["input_data_file"] = str(data)
    contracts_path = _write_contracts(tmp_path, contracts=[contract])
    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(contracts_path),
        output_dir=str(tmp_path / "out"),
        top_n=1,
        seeds=[0],
        cost_scenario="base",
        write_files=True,
    )
    cfg = json.loads(Path(manifest["configs"][0]["config_file"]).read_text(encoding="utf-8"))
    assert cfg["split_anchor"] == "end"
    assert cfg["train_years"] == 2
    assert cfg["val_years"] == 1
    assert cfg["test_years"] == 1


def test_deep_micro_nsga_contract_uses_short_config_and_run_paths(tmp_path):
    deep_id = "btcusdt_perp__4h__sota_low_cost__mutual_info_topk__p03__selected" + (
        "__micro_nsga_g01_i01_00"
        "__micro_nsga_g02_i00_00"
        "__micro_nsga_g03_i00_00"
        "__micro_nsga_g04_i00_00"
        "__micro_nsga_g05_i00_02"
        "__micro_nsga_g06_i00_01"
        "__micro_nsga_g07_i00_01"
        "__micro_nsga_g08_i00_00"
    )
    contract = _make_contract("deep")
    contract["contract_id"] = deep_id
    contracts_path = _write_contracts(tmp_path, contracts=[contract])

    manifest = smoke.build_smoke_plan(
        selected_contracts_path=str(contracts_path),
        output_dir=str(tmp_path / "out"),
        top_n=1,
        seeds=[0],
        cost_scenario="base",
        write_files=True,
    )

    entry = manifest["configs"][0]
    cfg = json.loads(Path(entry["config_file"]).read_text(encoding="utf-8"))
    assert entry["contract_id"] == deep_id
    assert cfg["_stage3x_contract_id"] == deep_id
    assert len(Path(entry["config_file"]).name) < 120
    assert len(Path(entry["run_dir"]).name) < 120
