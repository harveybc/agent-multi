import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "project3_phase4_protocol_dry_run.py"


spec = importlib.util.spec_from_file_location("project3_phase4_protocol_dry_run", TOOL_PATH)
dryrun = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = dryrun
spec.loader.exec_module(dryrun)


def _reference_config() -> dict:
    return {
        "asset": "ETHUSDT",
        "input_data_file": "/tmp/real_train.csv",
        "total_timesteps": 25_000,
        "env_plugin": "gym_fx_env",
        "agent_plugin": "sac_agent",
        "strategy_plugin": "rl_direct",
        "reward_plugin": "atr_sltp",
        "metrics_plugin": "default_metrics",
        "broker_plugin": "default_broker",
        "data_feed_plugin": "csv_data_feed",
        "preprocessor_plugin": "default_preprocessor",
        "pipeline_plugin": "rl_pipeline_with_validation",
        "optimizer_plugin": "default_optimizer",
        "features_preset": "tech_stat",
        "action_space_mode": "continuous",
        "continuous_action_threshold": 0.1,
        "window_size": 1,
        "atr_period": 14,
        "k_sl": 1.0,
        "k_tp": 2.0,
        "commission": 0.001,
        "slippage": 0.001,
        "leverage": 1.0,
        "position_size": 0.01,
        "rel_volume": 1.0,
        "size_mode": "fixed",
        "min_order_volume": 0.0,
        "max_order_volume": 1.0,
        "initial_cash": 10_000.0,
        "price_column": "CLOSE",
        "date_column": "DATE_TIME",
        "learning_rate": 0.0003,
        "buffer_size": 100_000,
        "learning_starts": 5_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "use_sde": True,
    }


def _protocol_packet(valid: bool = True) -> dict:
    return {
        "project3_valid_for_training": valid,
        "stage_b_status": "PENDING_APPROVAL",
        "windows": {"heldout_boundary": "2025-01-01T00:00:00"},
        "generator": {
            "family_id": "regime_residual_bootstrap_v1",
            "family_revision": "anti_mem_v1",
        },
        "output_files": {
            "augmented_tech_stat_csv": {
                "path": "/tmp/project3_synthetic_augmented_train.csv"
            },
            "synthetic_only_tech_stat_csv": {
                "path": "/tmp/project3_synthetic_only_train.csv"
            },
        },
    }


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def test_materialize_generates_four_locked_arms_and_manifest(tmp_path):
    ref_path = tmp_path / "reference.json"
    packet_path = tmp_path / "packet.json"
    out_dir = tmp_path / "arms"
    _write_json(ref_path, _reference_config())
    _write_json(packet_path, _protocol_packet())

    result = dryrun.materialize(
        reference_config_path=str(ref_path),
        protocol_packet_path=str(packet_path),
        out_dir=str(out_dir),
        seeds=[0, 1, 2],
        validate_only=False,
    )

    assert result["manifest"]["training_launched"] is False
    assert result["manifest"]["dry_run_only"] is True
    assert set(result["arm_paths"]) == {"arm_a", "arm_b", "arm_c", "arm_d"}
    for arm_path in result["arm_paths"].values():
        cfg = json.loads(Path(arm_path).read_text(encoding="utf-8"))
        assert cfg["_protocol_lock"]["_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"] is True
    arm_c = json.loads(
        Path(result["arm_paths"]["arm_c"]).read_text(encoding="utf-8")
    )
    assert arm_c["_protocol_lock"]["execution_status"] == "DIAGNOSTIC_ONLY_NEVER_PROMOTE"
    assert Path(result["manifest_json"]).exists()
    assert Path(result["manifest_md"]).exists()


def test_validate_only_rejects_invalid_protocol_packet(tmp_path):
    ref_path = tmp_path / "reference.json"
    packet_path = tmp_path / "packet.json"
    _write_json(ref_path, _reference_config())
    _write_json(packet_path, _protocol_packet(valid=False))

    with pytest.raises(RuntimeError, match="project3_valid_for_training"):
        dryrun.materialize(
            reference_config_path=str(ref_path),
            protocol_packet_path=str(packet_path),
            out_dir=str(tmp_path / "arms"),
            seeds=[0],
            validate_only=True,
        )


def test_validator_rejects_arm_d_sac_hyperparameter_drift():
    ref = _reference_config()
    packet = _protocol_packet()
    configs = dryrun.build_arm_configs(
        ref,
        packet,
        seeds=[0],
        protocol_packet_path="/tmp/packet.json",
        protocol_packet_hash="abc",
        reference_config_path="/tmp/reference.json",
    )
    configs["arm_d"]["learning_rate"] = 0.123

    validator = dryrun.DryRunValidator(ref, packet, configs)
    validator.run_all()
    with pytest.raises(RuntimeError, match="learning_rate"):
        validator.raise_if_invalid()


def test_validator_rejects_promotion_eligible_synthetic_only_arm():
    ref = _reference_config()
    packet = _protocol_packet()
    configs = dryrun.build_arm_configs(
        ref,
        packet,
        seeds=[0],
        protocol_packet_path="/tmp/packet.json",
        protocol_packet_hash="abc",
        reference_config_path="/tmp/reference.json",
    )
    configs["arm_c"]["_protocol_lock"]["promotion_eligible"] = True
    configs["arm_c"]["_phase4_arm"]["promotion_eligible"] = True

    validator = dryrun.DryRunValidator(ref, packet, configs)
    validator.run_all()
    with pytest.raises(RuntimeError, match="arm_c"):
        validator.raise_if_invalid()


def test_validator_rejects_synthetic_validation_or_test_paths():
    ref = _reference_config()
    packet = _protocol_packet()
    configs = dryrun.build_arm_configs(
        ref,
        packet,
        seeds=[0],
        protocol_packet_path="/tmp/packet.json",
        protocol_packet_hash="abc",
        reference_config_path="/tmp/reference.json",
    )
    configs["arm_c"]["_phase4_arm"]["validation_input_data_file"] = (
        "/tmp/synthetic_validation.csv"
    )

    validator = dryrun.DryRunValidator(ref, packet, configs)
    validator.run_all()
    with pytest.raises(RuntimeError, match="validation_input_data_file"):
        validator.raise_if_invalid()


def test_locked_synthetic_template_resolves_real_finetune_panel():
    ref = _reference_config()
    ref["input_data_file"] = "/tmp/synthetic-datagen/augmented_regime_residual.csv"
    ref["_protocol_lock"] = {
        "finetune": {
            "input_data_file": "/tmp/project3_real_ethusdt_4h_train.csv",
        },
    }
    packet = _protocol_packet()

    configs = dryrun.build_arm_configs(
        ref,
        packet,
        seeds=[0],
        protocol_packet_path="/tmp/packet.json",
        protocol_packet_hash="abc",
        reference_config_path="/tmp/reference.json",
    )

    assert configs["arm_a"]["input_data_file"] == "/tmp/project3_real_ethusdt_4h_train.csv"
    assert configs["arm_b"]["input_data_file"] == "/tmp/project3_real_ethusdt_4h_train.csv"
    assert configs["arm_c"]["_phase4_arm"]["validation_input_data_file"] == (
        "/tmp/project3_real_ethusdt_4h_train.csv"
    )
    assert configs["arm_d"]["_arm_finetune"]["input_data_file"] == (
        "/tmp/project3_real_ethusdt_4h_train.csv"
    )

    validator = dryrun.DryRunValidator(ref, packet, configs)
    validator.run_all()
    validator.raise_if_invalid()


def test_main_dry_run_validate_protocol_reports_no_training(tmp_path, capsys):
    ref_path = tmp_path / "reference.json"
    packet_path = tmp_path / "packet.json"
    _write_json(ref_path, _reference_config())
    _write_json(packet_path, _protocol_packet())

    rc = dryrun.main(
        [
            "--reference-config",
            str(ref_path),
            "--protocol-packet",
            str(packet_path),
            "--out-dir",
            str(tmp_path / "arms"),
            "--dry-run-validate-protocol",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["training_launched"] is False


def test_generated_arm_configs_are_blocked_by_seed_sweep_lock(tmp_path):
    """Each emitted arm config must trip ``seed_sweep._locked_protocol_reason``."""
    ref_path = tmp_path / "reference.json"
    packet_path = tmp_path / "packet.json"
    _write_json(ref_path, _reference_config())
    _write_json(packet_path, _protocol_packet())
    result = dryrun.materialize(
        reference_config_path=str(ref_path),
        protocol_packet_path=str(packet_path),
        out_dir=str(tmp_path / "arms"),
        seeds=[0, 1, 2],
        validate_only=False,
    )

    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import seed_sweep  # noqa: WPS433
    finally:
        sys.path.pop(0)

    for arm_name, path in result["arm_paths"].items():
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        reason = seed_sweep._locked_protocol_reason(cfg)
        assert reason, f"{arm_name} not blocked by seed_sweep lock"
        assert "Stage B" in reason


def test_arm_b_compute_match_equals_arm_d_pretrain_plus_finetune(tmp_path):
    ref_path = tmp_path / "reference.json"
    packet_path = tmp_path / "packet.json"
    _write_json(ref_path, _reference_config())
    _write_json(packet_path, _protocol_packet())
    result = dryrun.materialize(
        reference_config_path=str(ref_path),
        protocol_packet_path=str(packet_path),
        out_dir=str(tmp_path / "arms"),
        seeds=[0],
        validate_only=False,
    )
    arm_b = json.loads(Path(result["arm_paths"]["arm_b"]).read_text(encoding="utf-8"))
    arm_d = json.loads(Path(result["arm_paths"]["arm_d"]).read_text(encoding="utf-8"))
    pre = arm_d["_arm_pretrain"]["total_timesteps"]
    fine = arm_d["_arm_finetune"]["total_timesteps"]
    assert arm_b["total_timesteps"] == pre + fine
    cm = arm_b["_phase4_arm"]["compute_match"]
    assert cm["matched_to_arm"] == "arm_d"
    assert cm["components"]["pretrain_total_timesteps"] == pre
    assert cm["components"]["finetune_total_timesteps"] == fine


def test_arm_d_execution_status_is_template_only(tmp_path):
    ref_path = tmp_path / "reference.json"
    packet_path = tmp_path / "packet.json"
    _write_json(ref_path, _reference_config())
    _write_json(packet_path, _protocol_packet())
    result = dryrun.materialize(
        reference_config_path=str(ref_path),
        protocol_packet_path=str(packet_path),
        out_dir=str(tmp_path / "arms"),
        seeds=[0],
        validate_only=False,
    )
    arm_d = json.loads(Path(result["arm_paths"]["arm_d"]).read_text(encoding="utf-8"))
    assert arm_d["_protocol_lock"]["execution_status"] == (
        "TEMPLATE_ONLY_MULTI_PHASE_RUNNER_NOT_IMPLEMENTED"
    )
    assert result["manifest"]["multi_phase_runner_implemented"] is False
    assert result["manifest"]["execution_status_arm_d"] == (
        "TEMPLATE_ONLY_MULTI_PHASE_RUNNER_NOT_IMPLEMENTED"
    )
