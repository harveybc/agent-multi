from __future__ import annotations

import json
import sys
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from project3_weekly_materialize import materialize  # noqa: E402
from project3_weekly_pool import claim_subjob, complete_subjob, connect, enqueue_plan, init_db, status  # noqa: E402
from project3_weekly_worker import summarize_result  # noqa: E402
import project3_weekly_phase_orchestrator as orchestrator  # noqa: E402
import project3_weekly_supervisor as supervisor  # noqa: E402
import project3_weekly_adaptive_scheduler as adaptive_scheduler  # noqa: E402
from project3_weekly_artifact_backfill import backfill as backfill_artifacts  # noqa: E402


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


def test_weekly_pool_olap_year_view_groups_seed_rows_by_test_week(tmp_path):
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    subjobs = [
        ("job_a_20231204_seed1", "2023-12-04", "2023-12-11", 1),
        ("job_a_20231204_seed2", "2023-12-04", "2023-12-11", 2),
        ("job_a_20231211_seed1", "2023-12-11", "2023-12-18", 3),
    ]
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": "job_a",
                        "candidate_id": "candidate_a",
                        "asset": "solusdt",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 2,
                        "training_policy": "fine_tune_recent_window_chain",
                        "input_data_file": "/tmp/solusdt.csv",
                        "feature_columns": ["f1", "f2"],
                        "hyperparameters": {"sltp_profile_tag": "aware_rv0p50_base"},
                        "subjobs": [
                            {
                                "subjob_id": subjob_id,
                                "weekly_anchor_id": anchor,
                                "train_start": "2021-12-01 00:00:00",
                                "train_end": f"{anchor} 00:00:00",
                                "validation_start": f"{anchor} 00:00:00",
                                "validation_end": f"{test_start} 00:00:00",
                                "test_start": f"{test_start} 00:00:00",
                                "test_end": f"{test_end} 00:00:00",
                                "priority": priority,
                            }
                            for subjob_id, anchor, test_start, priority in subjobs
                            for test_end in ["2023-12-18" if test_start == "2023-12-11" else "2023-12-25"]
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
    complete_subjob(
        conn,
        "job_a_20231204_seed1",
        {
            "score": 0.10,
            "train_validation_risk_adjusted_composite_score": 0.10,
            "test_total_return": 0.04,
            "test_risk_adjusted_total_return": 0.03,
            "test_max_drawdown_fraction": 0.02,
            "rel_volume": 0.5,
            "sltp_risk_mode": "rel_volume_aware_atr",
        },
    )
    complete_subjob(
        conn,
        "job_a_20231204_seed2",
        {
            "score": 0.20,
            "train_validation_risk_adjusted_composite_score": 0.20,
            "test_total_return": 0.08,
            "test_risk_adjusted_total_return": 0.07,
            "test_max_drawdown_fraction": 0.02,
            "rel_volume": 0.5,
            "sltp_risk_mode": "rel_volume_aware_atr",
        },
    )
    complete_subjob(
        conn,
        "job_a_20231211_seed1",
        {
            "score": 0.30,
            "train_validation_risk_adjusted_composite_score": 0.30,
            "test_total_return": -0.02,
            "test_risk_adjusted_total_return": -0.04,
            "test_max_drawdown_fraction": 0.04,
            "rel_volume": 0.5,
            "sltp_risk_mode": "rel_volume_aware_atr",
        },
    )

    week_rows = conn.execute(
        "SELECT test_week_start, subjob_rows, mean_test_rap FROM weekly_result_test_week_olap ORDER BY test_week_start"
    ).fetchall()
    assert len(week_rows) == 2
    assert week_rows[0]["subjob_rows"] == 2
    assert week_rows[0]["mean_test_rap"] == pytest.approx(0.05)
    assert week_rows[1]["subjob_rows"] == 1
    assert week_rows[1]["mean_test_rap"] == pytest.approx(-0.04)

    year = conn.execute("SELECT * FROM weekly_result_test_year_olap").fetchone()
    assert year["olap_profile_key"] == "aware_rv0p50_base"
    assert year["unique_test_weeks"] == 2
    assert year["subjob_rows"] == 3
    assert year["mean_weekly_test_rap"] == pytest.approx(0.005)
    assert year["observed_test_rap"] == pytest.approx(0.01)
    assert year["projected_annual_test_rap_52w"] == pytest.approx(0.26)
    assert year["worst_weekly_test_rap"] == pytest.approx(-0.04)
    assert year["best_weekly_test_rap"] == pytest.approx(0.05)
    assert year["has_near_full_year_coverage"] == 0


def test_weekly_pool_full_year_protocol_view_requires_configured_coverage(tmp_path):
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"

    def subjob(
        subjob_id: str,
        block: str,
        validation_start: str,
        test_start: str,
        test_end: str,
        priority: int,
    ):
        return {
            "subjob_id": subjob_id,
            "weekly_anchor_id": validation_start,
            "train_start": "2020-01-01 00:00:00",
            "train_end": f"{validation_start} 00:00:00",
            "validation_start": f"{validation_start} 00:00:00",
            "validation_end": f"{test_start} 00:00:00",
            "test_start": f"{test_start} 00:00:00",
            "test_end": f"{test_end} 00:00:00",
            "priority": priority,
            "evaluation_block": block,
        }

    common_job = {
        "candidate_id": "candidate_full_year",
        "asset": "ethusdt",
        "timeframe": "4h",
        "model_family": "sac",
        "train_years": 4,
        "training_policy": "scratch_n_years",
        "input_data_file": "/tmp/ethusdt.csv",
        "feature_columns": ["f1", "f2"],
        "evaluation_protocol": "full_year_validation_test_v1",
        "configured_validation_year": 2022,
        "configured_test_year": 2023,
        "annual_eval_min_weeks": 2,
    }
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        **common_job,
                        "job_id": "candidate_full_year_val",
                        "evaluation_block": "validation_year",
                        "subjobs": [
                            subjob("val_1", "validation_year", "2022-01-03", "2022-01-10", "2022-01-17", 1),
                            subjob("val_2", "validation_year", "2022-01-10", "2022-01-17", "2022-01-24", 2),
                        ],
                    },
                    {
                        **common_job,
                        "job_id": "candidate_full_year_test",
                        "evaluation_block": "test_year",
                        "subjobs": [
                            subjob("test_1", "test_year", "2023-01-02", "2023-01-09", "2023-01-16", 3),
                            subjob("test_2", "test_year", "2023-01-09", "2023-01-16", "2023-01-23", 4),
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)
    for idx, subjob_id in enumerate(["val_1", "val_2", "test_1", "test_2"], start=1):
        complete_subjob(
            conn,
            subjob_id,
            {
                "score": 0.01 * idx,
                "train_validation_risk_adjusted_composite_score": 0.01 * idx,
                "validation_total_return": 0.02 * idx,
                "validation_risk_adjusted_total_return": 0.01 * idx,
                "validation_max_drawdown_fraction": 0.005 * idx,
                "test_total_return": 0.03 * idx,
                "test_risk_adjusted_total_return": 0.015 * idx,
                "test_max_drawdown_fraction": 0.006 * idx,
                "rel_volume": 0.05,
            },
        )

    rows = {
        row["metric_block"]: row
        for row in conn.execute(
            """
            SELECT metric_block, unique_weeks, metric_year, has_near_full_year_coverage,
                   mean_weekly_return, sum_weekly_return, annual_return,
                   mean_weekly_drawdown, sum_weekly_drawdown,
                   mean_weekly_rap, sum_weekly_rap, annual_rap
            FROM weekly_result_full_year_protocol_olap
            ORDER BY metric_block
            """
        )
    }
    assert rows["validation_year"]["metric_year"] == 2022
    assert rows["validation_year"]["unique_weeks"] == 2
    assert rows["validation_year"]["has_near_full_year_coverage"] == 1
    assert rows["validation_year"]["mean_weekly_return"] == pytest.approx(0.03)
    assert rows["validation_year"]["sum_weekly_return"] == pytest.approx(0.06)
    assert rows["validation_year"]["annual_return"] == pytest.approx(0.06)
    assert rows["validation_year"]["mean_weekly_drawdown"] == pytest.approx(0.0075)
    assert rows["validation_year"]["sum_weekly_drawdown"] == pytest.approx(0.015)
    assert rows["validation_year"]["mean_weekly_rap"] == pytest.approx(0.015)
    assert rows["validation_year"]["annual_rap"] == pytest.approx(0.03)
    assert rows["test_year"]["metric_year"] == 2023
    assert rows["test_year"]["unique_weeks"] == 2
    assert rows["test_year"]["has_near_full_year_coverage"] == 1
    assert rows["test_year"]["annual_return"] == pytest.approx(0.21)
    assert rows["test_year"]["sum_weekly_rap"] == pytest.approx(0.105)
    assert rows["test_year"]["annual_rap"] == pytest.approx(0.105)


def test_weekly_pool_artifact_backfill_indexes_files_and_merges_summary(tmp_path):
    db = tmp_path / "pool.sqlite"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "job_a_20231204_seed1"
    run_dir.mkdir(parents=True)
    plan = tmp_path / "plan.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
                {
                    "rel_volume": 0.5,
                    "sltp_risk_mode": "rel_volume_aware_atr",
                    "k_sl": 1.25,
                    "k_tp": 1.50,
                    "hyperparameters": {
                        "rel_volume": 0.5,
                        "sltp_risk_mode": "rel_volume_aware_atr",
                    "k_sl": 1.25,
                    "k_tp": 1.50,
                },
                "strategy_plugin": "direct_atr_sltp",
                "atr_period": 14,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "selection_metric": "risk_adjusted_return",
                "risk_penalty_lambda": 1.0,
                "splits": {
                    "train_tail": {
                        "total_return": 0.08,
                        "max_drawdown": 0.02,
                        "trades_total": 4,
                    },
                    "validation": {
                        "total_return": 0.06,
                        "max_drawdown": 0.03,
                        "trades_total": 5,
                    },
                    "test": {
                        "total_return": 0.05,
                        "max_drawdown": 0.04,
                        "trades_total": 6,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "subprocess_stdout.log").write_text("line 1\ntraining done\n", encoding="utf-8")
    (run_dir / "training_progress.json").write_text(
        json.dumps({"num_timesteps": 100, "total_return": 0.05}),
        encoding="utf-8",
    )
    (run_dir / "policy.zip").write_bytes(b"fake-policy")
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": "job_a",
                        "candidate_id": "candidate_a",
                        "asset": "solusdt",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 3,
                        "training_policy": "fine_tune_recent_window_chain",
                        "input_data_file": "/tmp/solusdt.csv",
                        "feature_columns": ["f1"],
                        "subjobs": [
                            {
                                "subjob_id": "job_a_20231204_seed1",
                                "weekly_anchor_id": "2023-12-04",
                                "train_start": "2020-12-04 00:00:00",
                                "train_end": "2023-12-04 00:00:00",
                                "validation_start": "2023-12-04 00:00:00",
                                "validation_end": "2023-12-11 00:00:00",
                                "test_start": "2023-12-11 00:00:00",
                                "test_end": "2023-12-18 00:00:00",
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
    conn.execute(
        """
        UPDATE subjobs
        SET status='done', config_path=?, run_dir=?, result_json=?
        WHERE external_id='job_a_20231204_seed1'
        """,
        (str(config_path), str(run_dir), json.dumps({"score": 0.01})),
    )
    conn.commit()

    report = backfill_artifacts(
        conn,
        runs_root=runs_root,
        apply=True,
        max_log_tail_chars=2000,
        max_json_bytes=100000,
        refresh_summaries=False,
        mark_done_from_results=False,
    )

    assert report["run_dirs_seen"] == 1
    assert report["artifact_files_seen"] == 4
    assert report["artifact_rows_upserted"] == 4
    artifact_types = {
        row["artifact_type"]
        for row in conn.execute(
            "SELECT artifact_type FROM weekly_result_artifact_olap WHERE subjob_id='job_a_20231204_seed1'"
        )
    }
    assert artifact_types == {
        "results_json",
        "stdout_log",
        "training_progress_json",
        "policy_zip",
    }
    result = json.loads(
        conn.execute(
            "SELECT result_json FROM subjobs WHERE external_id='job_a_20231204_seed1'"
        ).fetchone()["result_json"]
    )
    assert result["score"] == 0.01
    assert result["test_risk_adjusted_total_return"] == pytest.approx(0.01)
    assert result["rel_volume"] == pytest.approx(0.5)
    assert result["artifact_count"] == 4


def test_supervisor_recovers_stale_remote_claim_while_local_worker_active(tmp_path):
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": "job_stale",
                        "candidate_id": "job_stale",
                        "asset": "ethusdt",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 4,
                        "training_policy": "scratch_n_years",
                        "input_data_file": "/tmp/input.csv",
                        "feature_columns": ["f1"],
                        "subjobs": [
                            {
                                "subjob_id": "job_stale_20231204",
                                "weekly_anchor_id": "2023-12-04",
                                "train_start": "2019-12-04 00:00:00",
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
    claimed = claim_subjob(conn, "gamma")
    assert claimed is not None
    stale_time = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat(timespec="seconds")
    with conn:
        conn.execute(
            """
            UPDATE subjobs
            SET heartbeat_at=?, config_path='/tmp/old-config.json', run_dir='/tmp/old-run',
                result_json='{}'
            WHERE external_id='job_stale_20231204'
            """,
            (stale_time,),
        )
        conn.execute(
            """
            UPDATE machine_heartbeats
            SET heartbeat_at=?, gpu_summary='{}', message='old'
            WHERE machine_id='gamma'
            """,
            (stale_time,),
        )

    recovered = supervisor._recover_stale_running(conn, stale_minutes=10, worker_active=True)

    assert recovered == 1
    row = conn.execute(
        "SELECT status, claimed_by, heartbeat_at, config_path, run_dir, result_json, error FROM subjobs"
    ).fetchone()
    assert row["status"] == "pending"
    assert row["claimed_by"] is None
    assert row["heartbeat_at"] is None
    assert row["config_path"] is None
    assert row["run_dir"] is None
    assert row["result_json"] is None
    assert row["error"] is None
    machine = conn.execute("SELECT status, current_subjob_id, message FROM machine_heartbeats").fetchone()
    assert machine["status"] == "stale"
    assert machine["current_subjob_id"] is None
    assert "job_stale_20231204" in machine["message"]
    event = conn.execute(
        "SELECT event_type, payload_json FROM pool_events WHERE event_type='supervisor_requeue_stale'"
    ).fetchone()
    assert event is not None
    payload = json.loads(event["payload_json"])
    assert payload["worker_active"] is True
    assert payload["previous_claimed_by"] == "gamma"


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

    assert summary["train_validation_composite_score"] == pytest.approx(0.05)
    assert summary["train_validation_selection_gap_penalty"] == pytest.approx(0.025)
    assert summary["raw_score"] == pytest.approx(0.025)
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

    assert summary["raw_score"] == pytest.approx(-0.12)
    assert summary["score"] == pytest.approx(-0.12)
    assert summary["train_validation_composite_score"] == pytest.approx(-0.04)
    assert summary["test_total_return"] == -0.80
    assert summary["trade_gate_passed"] is True
    assert summary["selection_basis"] == "train_tail_validation_l1_gap_penalized_composite_with_trade_gate"


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


# ---------------------------------------------------------------------------
# event_token_transformer_v1 context embedding (Part A)
# ---------------------------------------------------------------------------

def _transformer_dataset(path: Path) -> None:
    """Train rows hold small event values; validation/test rows hold huge ones.

    Any leakage of validation/test rows into the fit would visibly distort the
    train-only normalization statistics recorded in the manifest.
    """
    rows = [
        "DATE_TIME,OPEN,HIGH,LOW,CLOSE,VOLUME,base_feature,event_surprise,event_importance",
        "2023-01-01 00:00:00,100,101,99,100,10,0.1,0.1,1.0",
        "2023-01-02 00:00:00,101,102,100,101,10,0.2,0.2,2.0",
        "2023-01-03 00:00:00,102,103,101,102,10,0.3,0.3,1.5",
        "2023-01-04 00:00:00,103,104,102,103,10,0.4,0.4,0.5",
        "2023-01-05 00:00:00,104,105,103,104,10,0.5,0.5,1.0",
        "2023-01-06 00:00:00,105,106,104,105,10,0.6,0.6,3.0",
        "2023-01-07 00:00:00,106,107,105,106,10,9.9,100,100",
        "2023-01-08 00:00:00,107,108,106,107,10,9.9,200,200",
        "2023-01-09 00:00:00,108,109,107,108,10,9.9,300,300",
        "2023-01-10 00:00:00,109,110,108,109,10,9.9,400,400",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _transformer_plan(data_path: Path, *, source_prefixes=None, embedding_dim=4) -> dict:
    profile = {
        "enabled": True,
        "family": "event_token_transformer_v1",
        "source_prefixes": source_prefixes if source_prefixes is not None else ["event_"],
        "output_prefix": "ctx_evt_tr",
        "embedding_dim": embedding_dim,
        "hidden_size": 8,
        "num_heads": 2,
        "num_blocks": 1,
        "seed": 1234,
        "required": True,
        "min_fit_rows": 3,
    }
    return {
        "stage_c_access": "DENIED",
        "training_launched": False,
        "jobs": [
            {
                "job_id": "job_transformer",
                "candidate_id": "job_transformer",
                "asset": "ethusdt",
                "timeframe": "4h",
                "model_family": "sac",
                "train_years": 1,
                "training_policy": "scratch_n_years",
                "input_data_file": str(data_path),
                "feature_columns": ["base_feature"],
                "context_embedding_profile": profile,
                "subjobs": [
                    {
                        "subjob_id": "job_transformer_20230107",
                        "weekly_anchor_id": "2023-01-07",
                        "train_start": "2023-01-01 00:00:00",
                        "train_end": "2023-01-07 00:00:00",
                        "validation_start": "2023-01-07 00:00:00",
                        "validation_end": "2023-01-09 00:00:00",
                        "test_start": "2023-01-09 00:00:00",
                        "test_end": "2023-01-11 00:00:00",
                        "priority": 1,
                    }
                ],
            }
        ],
    }


def test_transformer_profile_builds_train_only_embedding(tmp_path):
    data_path = tmp_path / "event_input.csv"
    _transformer_dataset(data_path)
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_transformer_plan(data_path)), encoding="utf-8")
    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)

    config_path = materialize(db, "job_transformer_20230107", tmp_path / "out")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Materialized config points at the derived CSV, not the original.
    derived_input = Path(config["input_data_file"])
    assert derived_input != data_path
    assert derived_input.exists()
    assert "context_embedding" in str(derived_input)

    # Generated columns are appended to feature_list.
    for col in ("ctx_evt_tr_00", "ctx_evt_tr_01", "ctx_evt_tr_02", "ctx_evt_tr_03"):
        assert col in config["feature_list"]
    assert "ctx_evt_tr_attn_mass" in config["feature_list"]
    assert "ctx_evt_tr_token_count" in config["feature_list"]

    manifest = json.loads(Path(config["_context_embedding_manifest_file"]).read_text(encoding="utf-8"))
    assert manifest["family"] == "event_token_transformer_v1"
    assert manifest["fit_scope"] == "train_only"
    assert manifest["fit_window_start"] == "2023-01-01 00:00:00"
    assert manifest["fit_window_end"] == "2023-01-07 00:00:00"
    assert manifest["source_columns"] == ["event_surprise", "event_importance"]
    assert manifest["training_summary"]["fit_scope"] == "train_only"
    assert manifest["model_config"]["framework"]


def test_transformer_profile_fits_only_on_train_rows(tmp_path):
    data_path = tmp_path / "event_input.csv"
    _transformer_dataset(data_path)
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_transformer_plan(data_path)), encoding="utf-8")
    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)

    config_path = materialize(db, "job_transformer_20230107", tmp_path / "out")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["_context_embedding_manifest_file"]).read_text(encoding="utf-8"))

    # Only the 6 train rows feed the fit; all 10 rows are transformed.
    assert manifest["training_summary"]["n_train_rows"] == 6
    assert manifest["training_summary"]["n_aux_target_rows"] == 5
    assert manifest["training_summary"]["n_total_rows"] == 10

    # Normalization mean must reflect train-only values (0.1..0.6 -> 0.35),
    # not the huge validation/test event values (100..400).
    norm = manifest["model_config"]["normalization"]
    assert norm["event_surprise"]["mean"] == pytest.approx(0.35, abs=1e-9)
    assert norm["event_importance"]["mean"] == pytest.approx(1.5, abs=1e-9)

    # Validation/test rows are still transformed (columns populated).
    import csv as _csv

    with Path(config["input_data_file"]).open("r", encoding="utf-8", newline="") as handle:
        rows = list(_csv.DictReader(handle))
    last = rows[-1]
    assert last["DATE_TIME"] == "2023-01-10 00:00:00"
    assert last["ctx_evt_tr_00"] not in ("", None)
    assert int(last["ctx_evt_tr_token_count"]) == 2


def test_transformer_profile_is_deterministic_by_seed(tmp_path):
    data_path = tmp_path / "event_input.csv"
    _transformer_dataset(data_path)

    def run(out_name: str) -> str:
        db = tmp_path / f"{out_name}.sqlite"
        plan = tmp_path / f"{out_name}.json"
        plan.write_text(json.dumps(_transformer_plan(data_path)), encoding="utf-8")
        conn = connect(db)
        init_db(conn)
        enqueue_plan(conn, plan)
        config_path = materialize(db, "job_transformer_20230107", tmp_path / out_name)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return Path(config["input_data_file"]).read_text(encoding="utf-8")

    assert run("run_a") == run("run_b")


def test_transformer_profile_refuses_missing_source_columns(tmp_path):
    data_path = tmp_path / "no_event.csv"
    data_path.write_text(
        "DATE_TIME,OPEN,HIGH,LOW,CLOSE,VOLUME,base_feature\n"
        "2023-01-01 00:00:00,100,101,99,100,10,0.1\n"
        "2023-01-02 00:00:00,101,102,100,101,10,0.2\n"
        "2023-01-03 00:00:00,102,103,101,102,10,0.3\n"
        "2023-01-04 00:00:00,103,104,102,103,10,0.4\n",
        encoding="utf-8",
    )
    db = tmp_path / "pool.sqlite"
    plan_dict = _transformer_plan(data_path)
    plan_dict["jobs"][0]["subjobs"][0]["validation_start"] = "2023-01-03 00:00:00"
    plan_dict["jobs"][0]["subjobs"][0]["train_end"] = "2023-01-03 00:00:00"
    plan_dict["jobs"][0]["subjobs"][0]["validation_end"] = "2023-01-04 00:00:00"
    plan_dict["jobs"][0]["subjobs"][0]["test_start"] = "2023-01-04 00:00:00"
    plan_dict["jobs"][0]["subjobs"][0]["test_end"] = "2023-01-05 00:00:00"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_dict), encoding="utf-8")
    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)

    with pytest.raises(ValueError, match="no source columns"):
        materialize(db, "job_transformer_20230107", tmp_path / "out")


def test_orchestrator_dry_runs_transformer_phase_without_duplicate_subjobs(tmp_path):
    import argparse

    source_job_id = orchestrator.EVENT_TOKEN_TRANSFORMER_SOURCE_JOBS[0]
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": [
                    {
                        "job_id": source_job_id,
                        "candidate_id": source_job_id,
                        "asset": "ethusdt",
                        "timeframe": "4h",
                        "model_family": "sac",
                        "train_years": 1,
                        "training_policy": "warm_start_chain_n_years",
                        "input_data_file": "/tmp/event_engineered.csv",
                        "feature_columns": ["event_surprise"],
                        "subjobs": [
                            {
                                "subjob_id": f"{source_job_id}_20231204",
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
    with conn:
        conn.execute(
            "UPDATE subjobs SET status='done' WHERE external_id=?",
            (f"{source_job_id}_20231204",),
        )

    args = argparse.Namespace(output_dir=str(tmp_path / "plans"))
    paths = orchestrator._phase_event_token_transformer(conn, args)
    assert paths, "transformer phase produced no plan"

    generated = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    assert generated["jobs"], "transformer phase plan has no jobs"
    for job in generated["jobs"]:
        profile = job["context_embedding_profile"]
        assert profile["family"] == "event_token_transformer_v1"
        assert profile["output_prefix"] == "ctx_evt_tr"
        assert profile["embedding_dim"] == 16

    result = orchestrator._enqueue_paths(conn, paths, dry_run=True)
    assert result.inserted_subjobs >= 1
    # Dry-run must not actually enqueue (no duplicate subjobs created).
    assert not orchestrator._phase_has_subjobs(conn, "event_token_transformer_phase_next_v1")


def test_orchestrator_suffix_generated_plan_prunes_missing_dependency():
    source_plan = {
        "jobs": [
            {
                "job_id": "job_a",
                "candidate_id": "job_a",
                "subjobs": [
                    {
                        "subjob_id": "job_a_ft_20231204",
                        "depends_on_subjob_id": "job_a_scratch_20231204",
                        "warm_start_parent_subjob_id": "job_a_scratch_20231204",
                    }
                ],
            }
        ]
    }

    suffixed = orchestrator._suffix_generated_plan(
        source_plan,
        phase_id="test_phase",
        seed=1,
    )

    assert suffixed["jobs"] == []
    assert suffixed["skipped_subjobs_due_missing_dependencies"] == [
        {
            "subjob_id": "job_a_ft_20231204",
            "missing_dependency_key": "depends_on_subjob_id",
            "missing_dependency": "job_a_scratch_20231204",
        }
    ]


def test_orchestrator_asset_broadening_specs_include_scratch_parent_policy(monkeypatch, tmp_path):
    def fake_existing_input(asset, timeframe, preset):
        if asset == "solusdt" and timeframe == "4h" and preset == "sota_low_cost":
            return tmp_path / "train.csv"
        return None

    monkeypatch.setattr(orchestrator, "_existing_input", fake_existing_input)

    specs = orchestrator._asset_broadening_specs()

    assert len(specs) == 1
    assert specs[0]["policies"] == "scratch,warm_start_chain,fine_tune_recent_window"


# ---------------------------------------------------------------------------
# adaptive scheduler
# ---------------------------------------------------------------------------

def _adaptive_job(job_id: str, *, asset="ethusdt", timeframe="4h", phase="asset_preset_broadening_phase5_v1", n=12, deps=False):
    subjobs = []
    previous = None
    for idx in range(n):
        anchor = datetime(2023, 9, 4, tzinfo=timezone.utc) + timedelta(days=idx * 7)
        train_start = datetime(2020, 9, 4, tzinfo=timezone.utc) + timedelta(days=idx * 7)
        validation_end = anchor + timedelta(days=7)
        test_end = anchor + timedelta(days=14)
        subjob_id = f"{job_id}_{idx:02d}"
        subjob = {
            "subjob_id": subjob_id,
            "weekly_anchor_id": anchor.strftime("%Y-%m-%d"),
            "train_start": train_start.strftime("%Y-%m-%d %H:%M:%S"),
            "train_end": anchor.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_start": anchor.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_end": validation_end.strftime("%Y-%m-%d %H:%M:%S"),
            "test_start": validation_end.strftime("%Y-%m-%d %H:%M:%S"),
            "test_end": test_end.strftime("%Y-%m-%d %H:%M:%S"),
            "priority": idx,
        }
        if deps and previous:
            subjob["depends_on_subjob_id"] = previous
            subjob["warm_start_parent_subjob_id"] = previous
        previous = subjob_id
        subjobs.append(subjob)
    return {
        "job_id": job_id,
        "candidate_id": job_id,
        "asset": asset,
        "timeframe": timeframe,
        "model_family": "sac",
        "train_years": 3,
        "training_policy": "warm_start_chain_n_years" if deps else "scratch_n_years",
        "input_data_file": "/tmp/input.csv",
        "feature_columns": ["f1", "f2"],
        "experiment_phase": phase,
        "subjobs": subjobs,
    }


def _enqueue_adaptive_plan(tmp_path, jobs):
    db = tmp_path / "pool.sqlite"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "stage_c_access": "DENIED",
                "training_launched": False,
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    conn = connect(db)
    init_db(conn)
    enqueue_plan(conn, plan)
    return conn


def _mark_done(conn, subjob_id, score, *, gate=True):
    complete_subjob(
        conn,
        subjob_id,
        {
            "train_validation_composite_score": score,
            "score": score,
            "trade_gate_passed": gate,
            "train_tail_total_return": score,
            "validation_total_return": score,
            "test_total_return": 999.0,
        },
    )


def test_adaptive_scheduler_caps_underprobed_independent_candidate(tmp_path):
    conn = _enqueue_adaptive_plan(tmp_path, [_adaptive_job("probe_job", n=12)])

    plan = adaptive_scheduler.build_defer_plan(
        conn,
        phases=("asset_preset_broadening_phase5_v1",),
        min_probe_weeks=5,
        probe_quota=5,
        keep_top_per_asset_timeframe=2,
        mean_floor=0.0,
        lcb_floor=0.0005,
    )
    applied = adaptive_scheduler.apply_defer_plan(conn, plan)

    assert applied["updated"] == 7
    counts = status(conn)["subjob_counts"]
    assert counts["pending"] == 5
    assert counts["deferred"] == 7


def test_adaptive_scheduler_keeps_warm_start_prefix_when_capping(tmp_path):
    conn = _enqueue_adaptive_plan(tmp_path, [_adaptive_job("warm_probe", n=10, deps=True)])

    plan = adaptive_scheduler.build_defer_plan(
        conn,
        phases=("asset_preset_broadening_phase5_v1",),
        min_probe_weeks=5,
        probe_quota=4,
        keep_top_per_asset_timeframe=2,
        mean_floor=0.0,
        lcb_floor=0.0005,
    )
    adaptive_scheduler.apply_defer_plan(conn, plan)

    rows = conn.execute(
        "SELECT external_id, status FROM subjobs ORDER BY priority"
    ).fetchall()
    assert [row["status"] for row in rows[:4]] == ["pending"] * 4
    assert [row["status"] for row in rows[4:]] == ["deferred"] * 6


def test_adaptive_scheduler_defers_weak_candidate_after_probe(tmp_path):
    conn = _enqueue_adaptive_plan(tmp_path, [_adaptive_job("weak_job", n=10)])
    for idx in range(5):
        _mark_done(conn, f"weak_job_{idx:02d}", -0.01)

    plan = adaptive_scheduler.build_defer_plan(
        conn,
        phases=("asset_preset_broadening_phase5_v1",),
        min_probe_weeks=5,
        probe_quota=8,
        keep_top_per_asset_timeframe=2,
        mean_floor=0.0,
        lcb_floor=0.0005,
    )
    applied = adaptive_scheduler.apply_defer_plan(conn, plan)

    assert applied["updated"] == 5
    counts = status(conn)["subjob_counts"]
    assert counts["done"] == 5
    assert counts["deferred"] == 5


def test_adaptive_scheduler_does_not_touch_running_subjobs(tmp_path):
    conn = _enqueue_adaptive_plan(tmp_path, [_adaptive_job("running_probe", n=10)])
    claimed = claim_subjob(conn, "omega")
    assert claimed is not None

    plan = adaptive_scheduler.build_defer_plan(
        conn,
        phases=("asset_preset_broadening_phase5_v1",),
        min_probe_weeks=5,
        probe_quota=3,
        keep_top_per_asset_timeframe=2,
        mean_floor=0.0,
        lcb_floor=0.0005,
    )
    adaptive_scheduler.apply_defer_plan(conn, plan)

    running = conn.execute("SELECT COUNT(*) AS n FROM subjobs WHERE status='running'").fetchone()["n"]
    assert running == 1
    counts = status(conn)["subjob_counts"]
    assert counts["pending"] == 2
    assert counts["deferred"] == 7


def test_adaptive_scheduler_keeps_top_candidates_per_asset_timeframe(tmp_path):
    conn = _enqueue_adaptive_plan(
        tmp_path,
        [
            _adaptive_job("best_job", asset="solusdt", timeframe="1h", n=8),
            _adaptive_job("mid_job", asset="solusdt", timeframe="1h", n=8),
            _adaptive_job("low_job", asset="solusdt", timeframe="1h", n=8),
        ],
    )
    for prefix, score in (("best_job", 0.02), ("mid_job", 0.01), ("low_job", 0.009)):
        for idx in range(5):
            _mark_done(conn, f"{prefix}_{idx:02d}", score)

    plan = adaptive_scheduler.build_defer_plan(
        conn,
        phases=("asset_preset_broadening_phase5_v1",),
        min_probe_weeks=5,
        probe_quota=8,
        keep_top_per_asset_timeframe=2,
        mean_floor=0.0,
        lcb_floor=0.0005,
    )
    adaptive_scheduler.apply_defer_plan(conn, plan)

    low_pending = conn.execute(
        "SELECT COUNT(*) AS n FROM subjobs WHERE external_id LIKE 'low_job_%' AND status='pending'"
    ).fetchone()["n"]
    low_deferred = conn.execute(
        "SELECT COUNT(*) AS n FROM subjobs WHERE external_id LIKE 'low_job_%' AND status='deferred'"
    ).fetchone()["n"]
    best_deferred = conn.execute(
        "SELECT COUNT(*) AS n FROM subjobs WHERE external_id LIKE 'best_job_%' AND status='deferred'"
    ).fetchone()["n"]
    mid_deferred = conn.execute(
        "SELECT COUNT(*) AS n FROM subjobs WHERE external_id LIKE 'mid_job_%' AND status='deferred'"
    ).fetchone()["n"]
    assert low_pending == 0
    assert low_deferred == 3
    assert best_deferred == 0
    assert mid_deferred == 0


def test_worker_summarize_result_uses_risk_adjusted_selection_metric(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "selection_metric": "risk_adjusted_return",
                "risk_penalty_lambda": 0.5,
                "splits": {
                    "train": {"total_return": 0.30, "max_drawdown_pct": 10.0, "trades_total": 3},
                    "train_tail": {"total_return": 0.10, "max_drawdown_pct": 2.0, "trades_total": 2},
                    "validation": {"total_return": 0.20, "max_drawdown_pct": 4.0, "trades_total": 1},
                    "test": {"total_return": 0.05, "max_drawdown_pct": 1.0, "trades_total": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_result(
        results,
        {
            "strategy_plugin": "direct_atr_sltp",
            "sltp_risk_mode": "rel_volume_aware_atr",
            "atr_period": 14,
            "k_sl": 2.0,
            "k_tp": 3.0,
            "rel_volume": 0.05,
            "max_risk_rel_volume": 0.5,
            "leverage": 1.0,
        },
    )

    assert summary["selection_metric"] == "risk_adjusted_return"
    assert summary["train_validation_composite_score"] == pytest.approx(0.15)
    assert summary["train_tail_risk_adjusted_total_return"] == pytest.approx(0.09)
    assert summary["validation_risk_adjusted_total_return"] == pytest.approx(0.18)
    assert summary["test_risk_adjusted_total_return"] == pytest.approx(0.045)
    assert summary["train_validation_risk_adjusted_composite_score"] == pytest.approx(0.135)
    assert summary["train_validation_selection_mean_score"] == pytest.approx(0.135)
    assert summary["train_validation_selection_gap"] == pytest.approx(0.09)
    assert summary["train_validation_selection_gap_penalty"] == pytest.approx(0.0225)
    assert summary["train_validation_l1_score"] == pytest.approx(0.1125)
    assert summary["raw_score"] == pytest.approx(0.1125)
    assert summary["score"] == pytest.approx(0.1125)
    assert summary["sltp_risk_mode"] == "rel_volume_aware_atr"
    assert summary["reward_risk_ratio"] == pytest.approx(1.5)
    assert summary["business_risk_fraction"] == pytest.approx(0.1)
    assert summary["stop_loss_atr_exposure_multiplier"] == pytest.approx(0.10)


def test_risk_adjusted_followup_phase_clones_winners_with_rap_config(tmp_path):
    conn = _enqueue_adaptive_plan(tmp_path, [_adaptive_job("winner_job", n=48)])
    for idx in range(48):
        _mark_done(conn, f"winner_job_{idx:02d}", 0.02)
    args = Namespace(
        db=str(tmp_path / "pool.sqlite"),
        output_dir=str(tmp_path / "plans"),
        label_dir=str(tmp_path / "labels"),
        python_bin=sys.executable,
    )

    paths = orchestrator._phase_risk_adjusted_followup(conn, args)

    assert len(paths) == 1
    plan = json.loads(paths[0].read_text(encoding="utf-8"))
    assert plan["plan_id"] == "risk_adjusted_reward_phase7_v3"
    assert plan["risk_metric"] == "RAP = total_return - lambda * max_drawdown_fraction"
    assert plan["l1_generalization_gap_penalty_beta"] == pytest.approx(0.25)
    assert len(plan["jobs"]) == 9
    rel_volumes = set()
    lambdas = set()
    for job in plan["jobs"]:
        hparams = job["hyperparameters"]
        assert hparams["reward_plugin"] == "dd_penalized_reward"
        assert hparams["selection_metric"] == "risk_adjusted_return"
        assert hparams["l1_generalization_gap_penalty_beta"] == pytest.approx(0.25)
        assert job["risk_adjusted_followup"] is True
        rel_volumes.add(hparams["rel_volume"])
        lambdas.add(hparams["risk_penalty_lambda"])
        assert len(job["subjobs"]) == 48
        assert all("risk_adjusted_reward_phase7_v3" in s["subjob_id"] for s in job["subjobs"])
    assert rel_volumes == {0.05, 0.075, 0.10}
    assert lambdas == {0.25, 0.5, 1.0}


def test_sltp_risk_geometry_phase_preserves_control_and_adds_aware_profiles(tmp_path):
    conn = _enqueue_adaptive_plan(tmp_path, [_adaptive_job("winner_job", n=48)])
    for idx in range(48):
        _mark_done(conn, f"winner_job_{idx:02d}", 0.02)
    args = Namespace(
        db=str(tmp_path / "pool.sqlite"),
        output_dir=str(tmp_path / "plans"),
        label_dir=str(tmp_path / "labels"),
        python_bin=sys.executable,
    )

    paths = orchestrator._phase_sltp_risk_geometry_followup(conn, args)

    assert len(paths) == 1
    plan = json.loads(paths[0].read_text(encoding="utf-8"))
    assert plan["plan_id"] == "sltp_risk_geometry_phase8_v3"
    assert plan["baseline"] == {"rel_volume": 0.05, "k_sl": 2.0, "k_tp": 3.0}
    assert len(plan["jobs"]) == 12
    modes = set()
    rel_volumes = set()
    control_jobs = []
    for job in plan["jobs"]:
        hparams = job["hyperparameters"]
        assert hparams["reward_plugin"] == "dd_penalized_reward"
        assert hparams["selection_metric"] == "risk_adjusted_return"
        assert hparams["l1_generalization_gap_penalty_beta"] == pytest.approx(0.25)
        assert hparams["k_tp"] >= hparams["k_sl"]
        assert hparams["max_risk_rel_volume"] == 0.5
        assert job["sltp_risk_geometry_followup"] is True
        modes.add(hparams["sltp_risk_mode"])
        rel_volumes.add(hparams["rel_volume"])
        if job["sltp_profile_tag"] == "control_rv0p05_sl2_tp3":
            control_jobs.append(job)
        assert len(job["subjobs"]) == 48
        assert all("sltp_risk_geometry_phase8_v3" in s["subjob_id"] for s in job["subjobs"])
    assert modes == {"fixed_atr", "rel_volume_aware_atr", "margin_aware_atr"}
    assert rel_volumes == {0.05, 0.10, 0.25, 0.50}
    assert len(control_jobs) == 1
    control = control_jobs[0]["hyperparameters"]
    assert control["rel_volume"] == 0.05
    assert control["k_sl"] == 2.0
    assert control["k_tp"] == 3.0
    assert control["sltp_risk_mode"] == "fixed_atr"
