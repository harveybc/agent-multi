from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from project3_weekly_materialize import materialize  # noqa: E402
from project3_weekly_pool import claim_subjob, complete_subjob, connect, enqueue_plan, init_db, status  # noqa: E402
from project3_weekly_worker import summarize_result  # noqa: E402


def test_weekly_pool_enqueue_claim_and_status(tmp_path):
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": "job_a",
                        "candidate_id": "candidate_a",
                        "asset": "btcusdt_perp",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 4,
                        "training_policy": "scratch_n_years",
                        "input_data_file": "/tmp/input.csv",
                        "feature_columns": ["f1", "f2"],
                        "subjobs": [
                            {
                                "subjob_id": "job_a_20231218",
                                "weekly_anchor_id": "2023-12-18",
                                "train_start": "2019-12-18 00:00:00",
                                "train_end": "2023-12-18 00:00:00",
                                "validation_start": "2023-12-18 00:00:00",
                                "validation_end": "2023-12-25 00:00:00",
                                "test_start": "2023-12-25 00:00:00",
                                "test_end": "2024-01-01 00:00:00",
                                "priority": 1,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    conn = connect(db)
    init_db(conn)
    assert enqueue_plan(conn, plan) == {"inserted_jobs": 1, "inserted_subjobs": 1}
    before = status(conn)
    assert before["subjob_counts"]["pending"] == 1

    claimed = claim_subjob(conn, "local")

    assert claimed is not None
    assert claimed["external_id"] == "job_a_20231218"
    after = status(conn)
    assert after["subjob_counts"]["running"] == 1
    assert after["machines"][0]["machine_id"] == "local"


def test_materializer_preserves_job_early_stop_train_tail_days(tmp_path):
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": "job_configurable_tail",
                        "candidate_id": "job_configurable_tail",
                        "asset": "btcusdt_perp",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 4,
                        "training_policy": "scratch_n_years",
                        "input_data_file": "/tmp/input.csv",
                        "feature_columns": ["f1", "f2"],
                        "validation_days": 14,
                        "test_days": 21,
                        "hyperparameters": {
                            "early_stop_train_tail_days": 28,
                            "l1_patience": 33,
                        },
                        "subjobs": [
                            {
                                "subjob_id": "job_configurable_tail_20231204",
                                "weekly_anchor_id": "2023-12-04",
                                "train_start": "2019-12-04 00:00:00",
                                "train_end": "2023-12-04 00:00:00",
                                "validation_start": "2023-12-04 00:00:00",
                                "validation_end": "2023-12-18 00:00:00",
                                "test_start": "2023-12-18 00:00:00",
                                "test_end": "2024-01-08 00:00:00",
                                "priority": 1,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)

    config_path = materialize(db, "job_configurable_tail_20231204", tmp_path / "out")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["early_stop_train_tail_days"] == 28
    assert config["l1_patience"] == 33
    assert config["validation_days"] == 14
    assert config["test_days"] == 21
    assert config["max_epochs"] == 500
    assert config["strategy_plugin"] == "direct_atr_sltp"
    assert config["atr_period"] == 14


def test_materializer_enables_event_execution_overlay_for_profile(tmp_path):
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": "job_event_overlay",
                        "candidate_id": "job_event_overlay",
                        "asset": "ethusdt",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 1,
                        "training_policy": "scratch_n_years",
                        "execution_profile": "event_no_trade_overlay_v1",
                        "input_data_file": "/tmp/input.csv",
                        "feature_columns": [
                            "feature_a",
                            "event_no_trade_window_active",
                            "event_spread_stress_multiplier",
                            "event_slippage_stress_multiplier",
                        ],
                        "subjobs": [
                            {
                                "subjob_id": "job_event_overlay_20231204",
                                "weekly_anchor_id": "2023-12-04",
                                "train_start": "2022-12-04 00:00:00",
                                "train_end": "2023-12-04 00:00:00",
                                "validation_start": "2023-12-04 00:00:00",
                                "validation_end": "2023-12-11 00:00:00",
                                "test_start": "2023-12-11 00:00:00",
                                "test_end": "2023-12-18 00:00:00",
                                "priority": 1,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)

    config_path = materialize(db, "job_event_overlay_20231204", tmp_path / "out")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["execution_profile"] == "event_no_trade_overlay_v1"
    assert config["event_context_execution_overlay"] is True
    assert config["event_context_force_flat"] is True
    assert config["event_context_block_new_entries"] is True
    assert config["event_context_no_trade_column"] == "event_no_trade_window_active"


def test_materializer_builds_train_only_context_embedding_profile(tmp_path):
    data_path = tmp_path / "event_input.csv"
    rows = [
        "DATE_TIME,OPEN,HIGH,LOW,CLOSE,VOLUME,base_feature,event_surprise,event_importance",
        "2023-01-01 00:00:00,100,101,99,100,10,0.1,0.2,1.0",
        "2023-01-02 00:00:00,101,102,100,101,10,0.2,0.3,2.0",
        "2023-01-03 00:00:00,102,103,101,102,10,0.3,0.1,1.5",
        "2023-01-04 00:00:00,103,104,102,103,10,0.4,-0.2,0.5",
        "2023-01-05 00:00:00,104,105,103,104,10,0.5,-0.1,1.0",
        "2023-01-06 00:00:00,105,106,104,105,10,0.6,0.4,3.0",
        "2023-01-07 00:00:00,106,107,105,106,10,0.7,0.0,1.0",
        "2023-01-08 00:00:00,107,108,106,107,10,0.8,0.5,2.0",
    ]
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": "job_context_embedding",
                        "candidate_id": "job_context_embedding",
                        "asset": "ethusdt",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 1,
                        "training_policy": "scratch_n_years",
                        "input_data_file": str(data_path),
                        "feature_columns": ["base_feature"],
                        "context_embedding_profile": {
                            "enabled": True,
                            "family": "event_token_attention_v1",
                            "source_prefixes": ["event_"],
                            "output_prefix": "ctx_evt",
                            "embedding_dim": 3,
                            "seed": 7,
                            "required": True,
                            "min_fit_rows": 3,
                        },
                        "subjobs": [
                            {
                                "subjob_id": "job_context_embedding_20230105",
                                "weekly_anchor_id": "2023-01-05",
                                "train_start": "2023-01-01 00:00:00",
                                "train_end": "2023-01-05 00:00:00",
                                "validation_start": "2023-01-05 00:00:00",
                                "validation_end": "2023-01-07 00:00:00",
                                "test_start": "2023-01-07 00:00:00",
                                "test_end": "2023-01-09 00:00:00",
                                "priority": 1,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)

    config_path = materialize(db, "job_context_embedding_20230105", tmp_path / "out")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    derived_input = Path(config["input_data_file"])
    manifest_path = Path(config["_context_embedding_manifest_file"])
    assert derived_input.exists()
    assert manifest_path.exists()
    assert "ctx_evt_00" in config["feature_list"]
    assert "ctx_evt_01" in config["feature_list"]
    assert "ctx_evt_02" in config["feature_list"]
    assert "ctx_evt_attn_mass" in config["feature_list"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["fit_scope"] == "train_only"
    assert manifest["fit_window_end"] == "2023-01-05 00:00:00"
    assert manifest["source_columns"] == ["event_surprise", "event_importance"]


def test_weekly_worker_summary_penalizes_no_trade_result(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"total_return": 3.0, "trades_total": 1},
                    "train_tail": {"total_return": 0.10, "trades_total": 1},
                    "validation": {"total_return": 0.00, "trades_total": 0},
                    "test": {"total_return": 0.05, "trades_total": 1},
                }
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_result(results)

    assert summary["raw_score"] == pytest.approx(0.05)
    assert summary["score"] < -999_000
    assert summary["trade_gate_passed"] is False
    assert summary["test_trade_gate_passed"] is True
    assert summary["validation_trades_total"] == 0


def test_weekly_worker_summary_uses_train_tail_validation_composite_not_test(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"total_return": 1.0, "trades_total": 10},
                    "train_tail": {"total_return": -0.20, "trades_total": 1},
                    "validation": {"total_return": 0.12, "trades_total": 1},
                    "test": {"total_return": -0.80, "trades_total": 1},
                }
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_result(results)

    assert summary["raw_score"] == pytest.approx(-0.04)
    assert summary["score"] == pytest.approx(-0.04)
    assert summary["train_validation_composite_score"] == pytest.approx(-0.04)
    assert summary["test_total_return"] == -0.80
    assert summary["trade_gate_passed"] is True
    assert summary["selection_basis"] == "train_tail_validation_composite_with_trade_gate"


def test_warm_start_dependency_is_claimed_only_after_parent_done(tmp_path):
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": "warm_job",
                        "candidate_id": "warm_job",
                        "asset": "btcusdt_perp",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 4,
                        "training_policy": "warm_start_chain_n_years",
                        "input_data_file": "/tmp/input.csv",
                        "feature_columns": ["f1", "f2"],
                        "subjobs": [
                            {
                                "subjob_id": "warm_20231204",
                                "weekly_anchor_id": "2023-12-04",
                                "train_start": "2019-12-04 00:00:00",
                                "train_end": "2023-12-04 00:00:00",
                                "validation_start": "2023-12-04 00:00:00",
                                "validation_end": "2023-12-11 00:00:00",
                                "test_start": "2023-12-11 00:00:00",
                                "test_end": "2023-12-18 00:00:00",
                                "priority": 1,
                            },
                            {
                                "subjob_id": "warm_20231211",
                                "weekly_anchor_id": "2023-12-11",
                                "train_start": "2019-12-11 00:00:00",
                                "train_end": "2023-12-11 00:00:00",
                                "validation_start": "2023-12-11 00:00:00",
                                "validation_end": "2023-12-18 00:00:00",
                                "test_start": "2023-12-18 00:00:00",
                                "test_end": "2023-12-25 00:00:00",
                                "depends_on_subjob_id": "warm_20231204",
                                "warm_start_parent_subjob_id": "warm_20231204",
                                "priority": 0,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)

    first = claim_subjob(conn, "local")

    assert first is not None
    assert first["external_id"] == "warm_20231204"


def test_warm_start_materializer_points_to_parent_policy(tmp_path):
    db = tmp_path / "pool.sqlite"
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("DATE_TIME,CLOSE,f1,f2\n2023-01-01 00:00:00,1,1,2\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": "warm_job",
                        "candidate_id": "warm_job",
                        "asset": "btcusdt_perp",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 4,
                        "training_policy": "warm_start_chain_n_years",
                        "input_data_file": str(input_csv),
                        "feature_columns": ["f1", "f2"],
                        "subjobs": [
                            {
                                "subjob_id": "warm_parent",
                                "weekly_anchor_id": "2023-12-04",
                                "train_start": "2019-12-04 00:00:00",
                                "train_end": "2023-12-04 00:00:00",
                                "validation_start": "2023-12-04 00:00:00",
                                "validation_end": "2023-12-11 00:00:00",
                                "test_start": "2023-12-11 00:00:00",
                                "test_end": "2023-12-18 00:00:00",
                            },
                            {
                                "subjob_id": "warm_child",
                                "weekly_anchor_id": "2023-12-11",
                                "train_start": "2019-12-11 00:00:00",
                                "train_end": "2023-12-11 00:00:00",
                                "validation_start": "2023-12-11 00:00:00",
                                "validation_end": "2023-12-18 00:00:00",
                                "test_start": "2023-12-18 00:00:00",
                                "test_end": "2023-12-25 00:00:00",
                                "depends_on_subjob_id": "warm_parent",
                                "warm_start_parent_subjob_id": "warm_parent",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)
    parent_run = tmp_path / "runs" / "warm_parent"
    parent_run.mkdir(parents=True)
    parent_policy = parent_run / "policy.zip"
    parent_policy.write_text("fake policy", encoding="utf-8")
    complete_subjob(conn, "warm_parent", {"run_dir": str(parent_run)})

    config_path = materialize(db, "warm_child", tmp_path / "out")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert cfg["warm_start_model"] == str(parent_policy)
    assert cfg["_warm_start_parent_subjob_id"] == "warm_parent"
