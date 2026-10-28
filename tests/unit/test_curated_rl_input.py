import hashlib
import json

import pytest

from app.curated_rl_input import validate_curated_rl_input
from env_plugins.gym_fx_env import Plugin
from tools.governed_offline_replay import metrics_of, TRADING_METRIC_KEYS
from tools.governed_offline_replay import merged_config
from tools.governed_run import METRIC_KEYS


def fixture(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("DATE_TIME,OPEN,HIGH,LOW,CLOSE,feature\n"
                    "2024-01-02T12:00:00+00:00,1,2,0.5,1.1,0.1\n"
                    "2024-01-02T13:00:00+00:00,1,2,0.5,1.2,0.2\n")
    stamps = "2024-01-02T12:00:00+00:00\n2024-01-02T13:00:00+00:00\n"
    body = {"schema": "curated_rl_input.v1", "status": "ADMITTED",
            "availability_verified": True, "open_session_coverage_verified": True,
            "split": "validation", "feature_columns": ["feature"],
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selection_sha256": "a"*64, "calendar_sha256": "b"*64,
            "inventory_sha256": "c"*64, "timestamp_semantics": "completed_bar_available_at",
            "period_start": "2024-01-01T00:00:00Z", "period_end": "2025-01-01T00:00:00Z",
            "expected_timestamps_sha256": hashlib.sha256(stamps.encode()).hexdigest(),
            "row_count": 2}
    manifest = tmp_path / "manifest.json"
    cfg = {"require_curated_rl_input": True, "input_data_file": str(path),
           "execution_transition": "next_open_v2", "strategy_plugin": "shared_execution_envelope",
           "curated_input_manifest": str(manifest), "curated_input_split": "validation",
           "feature_columns": ["feature"]}
    seal(cfg, body)
    return cfg, body, path


def seal(config, body):
    from pathlib import Path
    path = Path(config["curated_input_manifest"])
    path.write_text(json.dumps(body))
    config["curated_input_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_admitted_input_reaches_env_config(tmp_path):
    cfg, _, _ = fixture(tmp_path)
    result = Plugin()._build_env_config(cfg)
    assert result["curated_input_identity"]["row_count"] == 2
    assert result["curated_input_identity"]["split"] == "validation"


@pytest.mark.parametrize('change', [{'execution_transition':'legacy_delayed_v1'},
    {'strategy_plugin':'direct_atr_sltp'}, {'simulation_engine':'nautilus'}])
def test_curated_run_cannot_revert_to_an_unverified_execution_path(tmp_path, change):
    cfg, _, _ = fixture(tmp_path)
    cfg.update(change)
    with pytest.raises(ValueError, match='CURATED_NEXT_OPEN_PROTECTED_EXECUTION_REQUIRED'):
        Plugin()._build_env_config(cfg)


@pytest.mark.parametrize("change", ["pin", "bytes", "features", "split", "limit"])
def test_transport_mutations_refused(tmp_path, change):
    cfg, body, path = fixture(tmp_path)
    if change == "pin": cfg["curated_input_manifest_sha256"] = "0"*64
    if change == "bytes": path.write_text(path.read_text().replace("1.2", "1.3"))
    if change == "features": cfg["feature_columns"] = ["wrong"]
    if change == "split": cfg["curated_input_split"] = "train"
    if change == "limit": cfg["max_rows"] = 1
    with pytest.raises(ValueError, match="CURATED"):
        Plugin()._build_env_config(cfg)


@pytest.mark.parametrize("field,value", [("status", "CANDIDATE"),
    ("availability_verified", False), ("open_session_coverage_verified", False),
    ("row_count", 3), ("expected_timestamps_sha256", "d"*64),
    ("selection_sha256", None)])
def test_invalid_producer_receipt_refused(tmp_path, field, value):
    cfg, body, _ = fixture(tmp_path)
    body[field] = value
    seal(cfg, body)
    with pytest.raises(ValueError, match="CURATED"):
        validate_curated_rl_input(cfg)


@pytest.mark.parametrize("change", ["nan", "ohlc", "duplicate", "timezone"])
def test_bad_values_rejected_even_with_updated_content_hash(tmp_path, change):
    cfg, body, path = fixture(tmp_path)
    text = path.read_text()
    if change == "nan": text = text.replace(",0.2\n", ",nan\n")
    if change == "ohlc": text = text.replace(",1.2,", ",3.2,")
    if change == "duplicate": text = text.replace("T13:00:", "T12:00:")
    if change == "timezone": text = text.replace("+00:00", "")
    path.write_text(text)
    body["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    seal(cfg, body)
    with pytest.raises(ValueError, match="CURATED"):
        validate_curated_rl_input(cfg)


def test_governed_metrics_no_fabricated_or_nonfinite_values():
    assert set(TRADING_METRIC_KEYS) <= set(METRIC_KEYS)
    result = metrics_of({"net_profit": 15., "sharpe_ratio": None,
                         "win_rate_pct": float("nan")}, 1., {})
    assert result["net_profit"] == 15.
    assert "sharpe_ratio" not in result and "win_rate_pct" not in result


def test_governed_merge_preserves_admission_and_temporal_config(tmp_path):
    cfg, _, _ = fixture(tmp_path)
    template = tmp_path / "template.json"
    template.write_text("{}")
    cfg.update(observation_encoder="per_feature_conv1d", observation_stride_bars=2,
               window_size=13, require_full_observation_window=True, max_entries_per_day=1)
    merged = merged_config(cfg, template, tmp_path)
    assert merged["curated_input_manifest_sha256"] == cfg["curated_input_manifest_sha256"]
    assert merged["observation_encoder"] == "per_feature_conv1d"
    assert merged["max_rows"] is None and merged["environment"]["max_rows"] is None
    assert validate_curated_rl_input(merged)["row_count"] == 2


def test_daily_quota_never_silently_ignored_by_another_engine():
    with pytest.raises(ValueError, match="DAILY_QUOTA_UNSUPPORTED"):
        Plugin()._build_env_config({"max_entries_per_day": 1,
                                   "simulation_engine": "nautilus"})
