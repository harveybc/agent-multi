"""Unit tests for the Stage B locked run-plan expander.

These tests never launch training and never touch Stage C data. They
generate small synthetic CSVs with strictly preheldout dates so the
heldout safety check has real data to operate on.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "project3_stageb_run_plan.py"

_spec = importlib.util.spec_from_file_location(
    "project3_stageb_run_plan", TOOL_PATH,
)
runplan = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = runplan
_spec.loader.exec_module(runplan)


def _write_synthetic_data(tmp_path: Path, *, last_year: int = 2024) -> Path:
    """Write a tiny CSV with DATE_TIME spanning years up to ``last_year``."""
    csv_path = tmp_path / "synth_data.csv"
    rows = ["DATE_TIME,CLOSE"]
    # one timestamp per year so first/last are deterministic
    for year in range(2018, last_year + 1):
        rows.append(f"{year}-01-02 00:00:00,1000.0")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path


def _reference_config(tmp_path: Path, csv_path: Path) -> Path:
    cfg = {
        "asset": "ethusdt_4h",
        "agent_plugin": "project3_sac_actor_critic_agent",
        "pipeline_plugin": "rl_pipeline_with_validation",
        "env_plugin": "gym_fx_env",
        "input_data_file": str(csv_path),
        "date_column": "DATE_TIME",
        "price_column": "CLOSE",
        "features_preset": "tech_stat",
        "train_years": 4,
        "val_years": 1,
        "test_years": 1,
        "commission": 0.0002,
        "slippage": 0.0,
        "learning_rate": 0.0001,
        "buffer_size": 200000,
        "batch_size": 256,
        "save_model": "./examples/results/ref/policy.zip",
        "results_file": "./examples/results/ref/summary.json",
    }
    p = tmp_path / "ref_config.json"
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
def test_valid_reference_produces_manifest_and_locked_configs(tmp_path):
    csv = _write_synthetic_data(tmp_path, last_year=2024)
    ref = _reference_config(tmp_path, csv)
    out = tmp_path / "out"

    manifest = runplan.build_run_plan(
        reference_config_path=str(ref),
        candidate_id="ethusdt_4h_sac_tech_stat",
        output_dir=str(out),
        seeds=[0, 1, 2, 3, 4],
        cost_scenarios=["base", "pessimistic"],
        baselines=["no_trade", "buy_and_hold"],
    )
    assert manifest["schema_version"] == "project3_stageb_run_plan_v1"
    assert manifest["training_launched"] is False
    assert manifest["stage_c_access"] == "DENIED"
    # 5 seeds * 2 cost * (1 candidate + 2 baselines) = 30 entries
    assert len(manifest["configs"]) == 5 * 2 * 3
    assert (out / "stageb_run_plan_manifest.json").exists()
    assert (out / "stageb_run_plan_manifest.md").exists()
    # Every emitted config file exists.
    for entry in manifest["configs"]:
        assert Path(entry["config_file"]).exists()


def test_all_generated_configs_are_locked(tmp_path):
    csv = _write_synthetic_data(tmp_path)
    ref = _reference_config(tmp_path, csv)
    out = tmp_path / "out"
    manifest = runplan.build_run_plan(
        reference_config_path=str(ref),
        candidate_id="ethusdt_4h_sac_tech_stat",
        output_dir=str(out),
        seeds=[0, 1, 2, 3, 4],
        cost_scenarios=["base", "pessimistic"],
        baselines=["no_trade"],
    )
    for entry in manifest["configs"]:
        cfg = json.loads(Path(entry["config_file"]).read_text())
        assert cfg["_NOT_TO_RUN_UNTIL_STAGE_B_APPROVED"] is True
        assert cfg["_project3_stage_b_lock"] is True
        assert cfg["heldout_start"] == "2025-01-01"
        assert cfg["stage_c_access"] == "DENIED"
        assert cfg["final_stage_c_evaluation"] is False
        assert cfg["stage_c_acknowledged"] is False
        assert cfg["return_trace_dir"]
        assert cfg["return_trace_file"]


def test_no_generated_config_authorizes_stage_c(tmp_path):
    csv = _write_synthetic_data(tmp_path)
    ref = _reference_config(tmp_path, csv)
    out = tmp_path / "out"
    manifest = runplan.build_run_plan(
        reference_config_path=str(ref),
        candidate_id="ethusdt_4h_sac_tech_stat",
        output_dir=str(out),
        seeds=[0, 1, 2, 3, 4],
        cost_scenarios=["base", "pessimistic"],
        baselines=["no_trade"],
    )
    for entry in manifest["configs"]:
        cfg = json.loads(Path(entry["config_file"]).read_text())
        # The two flags that the trace module checks for Stage C
        # authorization must both be falsy on every emitted config.
        assert not cfg.get("final_stage_c_evaluation")
        assert not cfg.get("stage_c_acknowledged")


def test_default_seeds_at_least_five(tmp_path):
    csv = _write_synthetic_data(tmp_path)
    ref = _reference_config(tmp_path, csv)
    out = tmp_path / "out"
    # Default seeds via CLI parser (5 seeds) — this should succeed.
    manifest = runplan.build_run_plan(
        reference_config_path=str(ref),
        candidate_id="x",
        output_dir=str(out),
        seeds=list(runplan.DEFAULT_SEEDS),
        cost_scenarios=list(runplan.DEFAULT_COST_SCENARIOS),
        baselines=["no_trade"],
    )
    assert len(manifest["seeds"]) >= 5


def test_too_few_seeds_fails_unless_overridden(tmp_path):
    csv = _write_synthetic_data(tmp_path)
    ref = _reference_config(tmp_path, csv)
    out = tmp_path / "out"
    with pytest.raises(runplan.StageBPlanError, match="paired seeds"):
        runplan.build_run_plan(
            reference_config_path=str(ref),
            candidate_id="x", output_dir=str(out),
            seeds=[0], cost_scenarios=["base", "pessimistic"],
            baselines=["no_trade"],
        )
    # With override it succeeds but the plan is non-promotable.
    manifest = runplan.build_run_plan(
        reference_config_path=str(ref),
        candidate_id="x", output_dir=str(out),
        seeds=[0], cost_scenarios=["base", "pessimistic"],
        baselines=["no_trade"], allow_too_few_seeds=True,
    )
    assert "TOO_FEW_PAIRED_SEEDS" in manifest["promotion_blockers"]
    assert manifest["promotion_eligible"] is False


def test_missing_cost_scenarios_fails_unless_overridden(tmp_path):
    csv = _write_synthetic_data(tmp_path)
    ref = _reference_config(tmp_path, csv)
    out = tmp_path / "out"
    with pytest.raises(runplan.StageBPlanError, match="cost scenarios"):
        runplan.build_run_plan(
            reference_config_path=str(ref),
            candidate_id="x", output_dir=str(out),
            seeds=[0, 1, 2, 3, 4], cost_scenarios=["base"],
            baselines=["no_trade"],
        )


def test_deterministic_output_paths(tmp_path):
    csv = _write_synthetic_data(tmp_path)
    ref = _reference_config(tmp_path, csv)
    out = tmp_path / "out"
    manifest = runplan.build_run_plan(
        reference_config_path=str(ref),
        candidate_id="ethusdt_4h_sac_tech_stat",
        output_dir=str(out),
        seeds=[0, 1, 2, 3, 4],
        cost_scenarios=["base", "pessimistic"],
        baselines=["no_trade"],
    )
    for entry in manifest["configs"]:
        # Names must encode (role, baseline, seed, cost) deterministically.
        name = Path(entry["config_file"]).name
        assert f"s{entry['seed']}" in name
        assert entry["cost_scenario"] in name
        # Trace + evidence paths are deterministic per cell.
        assert entry["expected_evidence_file"].endswith("evidence.json")


def test_reference_sha_changes_when_config_changes(tmp_path):
    csv = _write_synthetic_data(tmp_path)
    ref = _reference_config(tmp_path, csv)
    out1 = tmp_path / "out1"
    m1 = runplan.build_run_plan(
        reference_config_path=str(ref),
        candidate_id="x", output_dir=str(out1),
        seeds=[0, 1, 2, 3, 4],
        cost_scenarios=["base", "pessimistic"],
        baselines=["no_trade"],
    )
    # Mutate the reference config
    cfg = json.loads(ref.read_text())
    cfg["learning_rate"] = 0.0002
    ref.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    out2 = tmp_path / "out2"
    m2 = runplan.build_run_plan(
        reference_config_path=str(ref),
        candidate_id="x", output_dir=str(out2),
        seeds=[0, 1, 2, 3, 4],
        cost_scenarios=["base", "pessimistic"],
        baselines=["no_trade"],
    )
    assert m1["reference_config_sha256"] != m2["reference_config_sha256"]


def test_heldout_violating_reference_is_rejected(tmp_path):
    # Data extends well into Stage C territory (last_year=2027) and
    # train+val+test = 6 years from 2018 → would land on 2024, which
    # is fine. But if last_year=2030 AND train_years sum=10, end=2028 → fail.
    csv = _write_synthetic_data(tmp_path, last_year=2030)
    cfg = {
        "asset": "ethusdt_4h",
        "agent_plugin": "project3_sac_actor_critic_agent",
        "pipeline_plugin": "rl_pipeline_with_validation",
        "env_plugin": "gym_fx_env",
        "input_data_file": str(csv),
        "date_column": "DATE_TIME",
        "train_years": 6,
        "val_years": 2,
        "test_years": 2,  # 2018 + 10y = 2028 >= 2025-01-01 → must reject
    }
    ref = tmp_path / "bad_ref.json"
    ref.write_text(json.dumps(cfg), encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(runplan.StageBPlanError, match="Stage C"):
        runplan.build_run_plan(
            reference_config_path=str(ref),
            candidate_id="x", output_dir=str(out),
            seeds=[0, 1, 2, 3, 4],
            cost_scenarios=["base", "pessimistic"],
            baselines=["no_trade"],
        )


def test_reference_with_stage_c_flags_is_rejected(tmp_path):
    csv = _write_synthetic_data(tmp_path)
    cfg = json.loads(_reference_config(tmp_path, csv).read_text())
    cfg["final_stage_c_evaluation"] = True
    cfg["stage_c_acknowledged"] = True
    ref = tmp_path / "bad_stage_c.json"
    ref.write_text(json.dumps(cfg), encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(runplan.StageBPlanError, match="Stage C"):
        runplan.build_run_plan(
            reference_config_path=str(ref),
            candidate_id="x", output_dir=str(out),
            seeds=[0, 1, 2, 3, 4],
            cost_scenarios=["base", "pessimistic"],
            baselines=["no_trade"],
        )


def test_baselines_marked_template_only(tmp_path):
    csv = _write_synthetic_data(tmp_path)
    ref = _reference_config(tmp_path, csv)
    out = tmp_path / "out"
    manifest = runplan.build_run_plan(
        reference_config_path=str(ref),
        candidate_id="x", output_dir=str(out),
        seeds=[0, 1, 2, 3, 4],
        cost_scenarios=["base", "pessimistic"],
        baselines=["no_trade", "buy_and_hold"],
    )
    baseline_entries = [e for e in manifest["configs"] if e["role"] == "baseline"]
    assert baseline_entries
    for entry in baseline_entries:
        assert entry["template_only"] is True
        assert entry["promotion_eligible"] is False
        cfg = json.loads(Path(entry["config_file"]).read_text())
        assert cfg["_baseline_template_only"] is True
        assert cfg["_baseline_promotion_eligible"] is False
    # Manifest must surface the template-only blocker.
    assert "BASELINES_TEMPLATE_ONLY" in manifest["promotion_blockers"]
    assert manifest["promotion_eligible"] is False


def test_dry_run_validate_only_writes_nothing(tmp_path):
    csv = _write_synthetic_data(tmp_path)
    ref = _reference_config(tmp_path, csv)
    out = tmp_path / "out_dryrun"
    manifest = runplan.build_run_plan(
        reference_config_path=str(ref),
        candidate_id="x", output_dir=str(out),
        seeds=[0, 1, 2, 3, 4],
        cost_scenarios=["base", "pessimistic"],
        baselines=["no_trade"],
        write_files=False,
    )
    assert not out.exists() or not any(out.iterdir())
    assert manifest["configs"]
