from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.import_project3_shortlist import build_manifest, canonical_asset_id


COLUMNS = [
    "selection_role", "asset", "timeframe", "horizon_bucket", "model_family",
    "experiment_phase", "input_family", "unique_test_weeks",
    "has_near_full_year_coverage", "mean_weekly_return_pct", "annual_return_pct",
    "mean_weekly_rap_pct", "annual_rap_pct", "mean_weekly_drawdown_pct",
    "mean_weekly_test_trades", "rel_volume", "sltp_risk_mode", "k_sl", "k_tp",
    "risk_penalty_lambda", "train_years", "training_policy", "candidate_id",
    "olap_profile_key",
]


def _write_evidence(repo: Path) -> tuple[Path, Path, Path]:
    shortlist = repo / "shortlist.csv"
    with shortlist.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "selection_role": "long_full_year_doin_seed",
                "asset": "solusdt",
                "timeframe": "4h",
                "horizon_bucket": "long",
                "model_family": "sac",
                "experiment_phase": "phase",
                "input_family": "technical",
                "unique_test_weeks": "52",
                "has_near_full_year_coverage": "1",
                "mean_weekly_return_pct": "0.1",
                "annual_return_pct": "5.0",
                "mean_weekly_rap_pct": "0.05",
                "annual_rap_pct": "2.5",
                "mean_weekly_drawdown_pct": "0.04",
                "mean_weekly_test_trades": "2",
                "rel_volume": "0.1",
                "sltp_risk_mode": "fixed_atr",
                "k_sl": "2",
                "k_tp": "3",
                "risk_penalty_lambda": "0.5",
                "train_years": "3",
                "training_policy": "scratch",
                "candidate_id": "candidate-1",
                "olap_profile_key": "profile-1",
            }
        )
    report = repo / "report.md"
    proposed = repo / "proposed.csv"
    report.write_text("report\n", encoding="utf-8")
    proposed.write_text("candidate_id\ncandidate-1\n", encoding="utf-8")
    return shortlist, report, proposed


def test_asset_mapping() -> None:
    assert canonical_asset_id("solusdt") == "crypto:SOL/USDT"
    assert canonical_asset_id("eurusd") == "fx:EUR/USD"
    assert canonical_asset_id("SPY") == "research:SPY"


def test_build_manifest_preserves_scope_and_hashes(tmp_path) -> None:
    shortlist, report, proposed = _write_evidence(tmp_path)
    manifest = build_manifest(
        shortlist_path=shortlist,
        selection_report_path=report,
        proposed_selection_path=proposed,
        source_repo=tmp_path,
        generated_at="2026-07-10T20:00:00+00:00",
        commit="a" * 40,
    )
    assert manifest["candidate_count"] == 1
    assert manifest["source_repository"]["commit"] == "a" * 40
    assert all(item["content_hash"].startswith("sha256:") for item in manifest["evidence_files"])
    candidate = manifest["candidates"][0]
    assert candidate["asset_id"] == "crypto:SOL/USDT"
    assert candidate["evidence_scope"] == "near_full_year"
    assert candidate["coverage_weeks"] == 52
    assert candidate["metrics"]["mean_weekly_return_percent"] == 0.1
    assert manifest["selection_policy"]["use"] == "research_seed_only"


def test_generated_at_requires_timezone(tmp_path) -> None:
    shortlist, report, proposed = _write_evidence(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware"):
        build_manifest(
            shortlist_path=shortlist,
            selection_report_path=report,
            proposed_selection_path=proposed,
            source_repo=tmp_path,
            generated_at="2026-07-10T20:00:00",
            commit="a" * 40,
        )
