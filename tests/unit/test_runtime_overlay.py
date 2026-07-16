from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from trading_contracts import TradingRuntimeOverlay

from app.canonical_config import ConfigResolutionError
from app.runtime_overlay import collect_git_snapshot, resolve_runtime_overlay


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _make_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-b", "master", str(path)], check=True, capture_output=True)
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.com")
    (path / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-m", "initial")
    return _git(path, "rev-parse", "HEAD")


def test_overlay_resolves_paths_without_mutating_runtime(tmp_path) -> None:
    runtime = {
        "input_data_file": "${DATA_ROOT}/eurusd.csv",
        "save_model": "${ARTIFACT_ROOT}/model.zip",
        "device": "auto",
    }
    payload = {
        "machine_id": "omega",
        "roots": {
            "DATA_ROOT": "data",
            "ARTIFACT_ROOT": "artifacts",
        },
        "devices": {"default": "cuda:0"},
    }
    resolved = resolve_runtime_overlay(
        runtime,
        overlay_payload=payload,
        overlay_base_dir=tmp_path,
    )
    assert runtime["input_data_file"] == "${DATA_ROOT}/eurusd.csv"
    assert resolved.runtime["input_data_file"] == str(tmp_path / "data" / "eurusd.csv")
    assert resolved.runtime["save_model"] == str(tmp_path / "artifacts" / "model.zip")
    assert resolved.runtime["device"] == "cuda:0"
    assert resolved.manifest["runtime_overlay_hash"].startswith("sha256:")


def test_unknown_runtime_root_fails_closed(tmp_path) -> None:
    with pytest.raises(ConfigResolutionError, match="MISSING_ROOT"):
        resolve_runtime_overlay(
            {"input_data_file": "${MISSING_ROOT}/data.csv"},
            overlay_payload=None,
            overlay_base_dir=tmp_path,
        )


def test_git_snapshot_records_commit_and_dirty_state(tmp_path) -> None:
    repo = tmp_path / "repo"
    commit = _make_repo(repo)
    clean = collect_git_snapshot("repo", str(repo), expected_commit=commit)
    assert clean["commit"] == commit
    assert clean["branch"] == "master"
    assert clean["dirty"] is False
    assert clean["matches_expected_commit"] is True

    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    dirty = collect_git_snapshot("repo", str(repo), expected_commit="0" * 40)
    assert dirty["dirty"] is True
    assert dirty["status_entry_count"] == 2
    assert dirty["matches_expected_commit"] is False
    assert dirty["tracked_diff_hash"] != clean["tracked_diff_hash"]
    assert " M tracked.txt" in dirty["status_sample"]
    assert "?? untracked.txt" in dirty["status_sample"]


def test_overlay_collects_repository_snapshots(tmp_path) -> None:
    repo = tmp_path / "repo"
    commit = _make_repo(repo)
    result = resolve_runtime_overlay(
        {"device": "cpu"},
        overlay_payload={
            "machine_id": "test-node",
            "roots": {"DATA_ROOT": "data"},
            "repositories": {"component": str(repo)},
        },
        overlay_base_dir=tmp_path,
        expected_repositories={"component": commit},
    )
    snapshot = result.manifest["repository_snapshots"]["component"]
    assert snapshot["matches_expected_commit"] is True
    assert result.runtime["device"] == "cpu"


def test_main_keeps_canonical_paths_and_writes_resolved_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    from app import main as main_module

    canonical = tmp_path / "experiment.json"
    canonical.write_text(
        json.dumps(
            {
                "schema_version": "trading_experiment.v1",
                "data": {"input_data_file": "${DATA_ROOT}/prices.csv"},
                "training": {"device": "auto"},
            }
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.json"
    overlay.write_text(
        json.dumps(
            {
                "schema_version": "trading_runtime_overlay.v1",
                "machine_id": "omega-test",
                "roots": {"DATA_ROOT": "machine-data"},
                "devices": {"default": "cuda:0"},
            }
        ),
        encoding="utf-8",
    )
    resolved_config = tmp_path / "resolved.json"
    config_manifest = tmp_path / "manifest.json"
    results = tmp_path / "results.json"
    flat = tmp_path / "flat.json"

    observed: dict = {}

    def fake_run(config: dict) -> dict:
        observed.update(config)
        return {"ok": True}

    monkeypatch.setattr(main_module, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-multi",
            "--config",
            str(canonical),
            "--runtime_overlay",
            str(overlay),
            "--resolved_config_file",
            str(resolved_config),
            "--config_manifest_file",
            str(config_manifest),
            "--results_file",
            str(results),
            "--save_config",
            str(flat),
            "--quiet_mode",
        ],
    )
    assert main_module.main() == 0

    canonical_written = json.loads(resolved_config.read_text(encoding="utf-8"))
    manifest_written = json.loads(config_manifest.read_text(encoding="utf-8"))
    flat_written = json.loads(flat.read_text(encoding="utf-8"))
    expected_path = str(tmp_path / "machine-data" / "prices.csv")
    assert canonical_written["data"]["input_data_file"] == "${DATA_ROOT}/prices.csv"
    assert observed["input_data_file"] == expected_path
    assert flat_written["input_data_file"] == expected_path
    assert observed["device"] == "cuda:0"
    assert manifest_written["runtime"]["machine_id"] == "omega-test"
    assert manifest_written["runtime"]["runtime_overlay_hash"].startswith("sha256:")


def test_versioned_machine_overlays_are_valid_and_isolated() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "runtime"
    overlays = {
        path.stem: TradingRuntimeOverlay.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*.json"))
    }
    assert set(overlays) == {"dragon", "gamma_5070ti", "gamma_5090", "omega"}
    # PyTorch enumerates Gamma's external 5090 before the internal 5070 Ti,
    # independently of nvidia-smi's physical-index display order.
    assert overlays["gamma_5070ti"].devices["training"] == "cuda:1"
    assert overlays["gamma_5090"].devices["training"] == "cuda:0"
    assert (
        overlays["gamma_5070ti"].roots["ARTIFACT_ROOT"]
        != overlays["gamma_5090"].roots["ARTIFACT_ROOT"]
    )
    assert len({overlay.machine_id for overlay in overlays.values()}) == 4


def test_canonical_identity_is_independent_of_machine_overlay(tmp_path) -> None:
    from app.canonical_config import resolve_config
    from app.config import DEFAULT_VALUES

    canonical = {
        "schema_version": "trading_experiment.v1",
        "data": {"input_data_file": "${DATA_ROOT}/prices.csv"},
    }
    resolution = resolve_config(DEFAULT_VALUES, file_config=canonical)
    first = resolve_runtime_overlay(
        resolution.runtime,
        overlay_payload={
            "machine_id": "first",
            "roots": {"DATA_ROOT": str(tmp_path / "first")},
        },
        overlay_base_dir=tmp_path,
    )
    second = resolve_runtime_overlay(
        resolution.runtime,
        overlay_payload={
            "machine_id": "second",
            "roots": {"DATA_ROOT": str(tmp_path / "second")},
        },
        overlay_base_dir=tmp_path,
    )
    assert first.runtime["input_data_file"] != second.runtime["input_data_file"]
    assert first.manifest["runtime_overlay_hash"] != second.manifest["runtime_overlay_hash"]
    assert resolution.canonical.canonical_hash == resolution.runtime["canonical_config_hash"]
