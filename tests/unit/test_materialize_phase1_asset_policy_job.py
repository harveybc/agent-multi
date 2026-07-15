from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.scripts.materialize_phase1_asset_policy_job import materialize


def test_materialize_rewrites_identity_risk_artifacts_and_manifest(tmp_path: Path):
    base = {
        "experiment": {"name": "old"},
        "data": {"date_column": "DATE_TIME"},
        "risk": {},
        "training": {},
        "optimization": {},
        "artifacts": {},
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    dataset = tmp_path / "train.csv"
    with dataset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["DATE_TIME", "CLOSE"])
        writer.writerow(["2020-01-01 00:00:00", 1])
        writer.writerow(["2023-12-31 23:00:00", 2])
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    materialize(
        base_config=base_path,
        output_config=config_path,
        output_manifest=manifest_path,
        dataset=dataset,
        dataset_source_path="inputs/btc/train.csv",
        asset="BTCUSDT",
        timeframe="1h",
        data_profile="kitchen_sink_guarded",
        train_start="2019-01-01T00:00:00",
        train_end="2021-12-31T23:59:59",
        validation_start="2022-01-01T00:00:00",
        validation_end="2022-12-31T23:59:59",
        test_start="2023-01-01T00:00:00",
        test_end="2023-12-31T23:59:59",
        risk_penalty_lambda=1.0,
        k_sl=2.0,
        k_tp=3.0,
        rel_volume=0.05,
    )
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert config["data"]["asset"] == "BTCUSDT"
    assert config["data"]["train_start"].startswith("2019")
    assert config["risk"] == {"rel_volume": 0.05, "k_sl": 2.0, "k_tp": 3.0}
    assert config["training"]["risk_penalty_lambda"] == 1.0
    assert "btcusdt_1h_sac" in config["optimization"]["optimization_champion_model_file"]
    assert manifest["row_count"] == 2
    assert manifest["column_count"] == 2
    assert manifest["selection_uses_test"] is False
