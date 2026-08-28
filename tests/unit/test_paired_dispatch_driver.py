"""Identity tests for the paired SAC dispatch driver (P4, updated for
the C3 real execution path, order 2026-08-28) — CPU only, model-free.
GPU execution requires Musashi's written dispatch document AND CUDA;
neither exists here, so the GPU path stays a refusal."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.dispatch_paired_pretrain_comparison import (  # noqa: E402
    DRY_RUN_BUDGET, DispatchRefused, assert_no_venue_keys,
    build_cell_config, frozen_o2022_envelope, verify_cell)


@pytest.fixture(scope="module")
def design():
    path = (REPO / "docs/audits/evidence/"
            "PAIRED_PRETRAIN_COMPARISON_DESIGN_2026_08_27.json")
    return json.loads(path.read_text())


def test_unknown_arm_and_seed_refuse(design):
    with pytest.raises(DispatchRefused, match="unknown arm"):
        verify_cell(design, Path("/nonexistent"), 101, "mystery_arm")
    arm = list(design["arms"])[0]
    with pytest.raises(DispatchRefused, match="not in the design"):
        verify_cell(design, Path("/nonexistent"), 999, arm)


def test_two_arm_design_with_frozen_deferred(design):
    assert set(design["arms"]) == {"control_random_init",
                                   "pretrained_finetuned"}
    assert "pretrained_frozen" in design.get("deferred_arms", {})
    assert len(design["trial_ledger"]) == 8  # 2 arms x 4 seeds
    orders = list(design["arm_order_counterbalanced"].values())
    first_positions = [order[0] for order in orders]
    assert first_positions.count("control_random_init") == 2
    assert first_positions.count("pretrained_finetuned") == 2


def test_design_carries_auditor_relabel(design):
    """C2: the treatment is auditor-chosen and exploratory — never
    presented as probe-selected."""
    eligibility = design["shared_bindings"]["pretrain_generation"][
        "eligibility"]
    assert "EXPLORATORY_PAIRED_SAC_TREATMENT_SELECTED_BY_AUDITOR" in \
        eligibility
    assert "NOT selected by probe performance" in eligibility


def test_gpu_execution_stays_gated():
    source = (REPO / "tools/dispatch_paired_pretrain_comparison.py"
              ).read_text()
    assert "gpu-authorized-by-musashi" in source
    assert "NOT_LAUNCHED" in source
    # the full-budget path verifies the TYPED authorization artifact
    # (H1/377) before any CUDA probe or model construction
    assert "verify_authorization" in source
    assert "verify_worktree_identity" in source
    # and the CPU dry run refuses when a GPU is visible
    assert "CUDA is visible" in source


def test_frozen_envelope_is_the_design_binding():
    geometry = frozen_o2022_envelope()
    assert geometry["envelope_mode"] == "atr"
    assert geometry["atr_sl_mult"] == 3.0
    assert geometry["atr_tp_mult"] == 6.0
    assert geometry["leverage_cap"] == 1.0


def test_venue_credential_keys_refuse():
    with pytest.raises(DispatchRefused, match="venue-credential"):
        assert_no_venue_keys({"mt5_password": "x"})
    with pytest.raises(DispatchRefused, match="venue-credential"):
        assert_no_venue_keys({"alpaca_api_key_id": "x"})
    assert_no_venue_keys({"learning_rate": 3e-4})  # clean passes


class TestArmConfigIdentity:
    """C4 adversarial identity: for one seed, the two arms' resolved
    configs differ ONLY in the initialization keys and the per-trial
    output paths — nothing else."""

    INIT_KEYS = {"pretrained_branch_generation_dir",
                 "pretrained_branch_expected_seal"}
    PATH_KEYS = {"save_model", "results_file", "save_config",
                 "nested_split_dir"}

    @staticmethod
    def _cell(design, seed, arm):
        trial = next(t for t in design["trial_ledger"]
                     if t["genesis"]["seed"] == seed
                     and t["genesis"]["arm"] == arm)
        return {
            "trial_id": trial["trial_id"],
            "genesis_sha256": trial["genesis_sha256"],
            "arm": arm, "seed": seed,
            "architecture_digest":
                design["shared_bindings"]["architecture_digest"],
            "pretrain_generation_seal":
                design["shared_bindings"]["pretrain_generation"][
                    "seal_manifest_sha256"],
        }

    def test_same_seed_arms_differ_only_in_init_and_paths(
            self, design, tmp_path):
        control = build_cell_config(
            design, self._cell(design, 101, "control_random_init"),
            Path("/pretrain"), tmp_path, device="cpu")
        treatment = build_cell_config(
            design, self._cell(design, 101, "pretrained_finetuned"),
            Path("/pretrain"), tmp_path, device="cpu")
        assert set(treatment) - set(control) == self.INIT_KEYS
        differing = {k for k in control
                     if control[k] != treatment.get(k)}
        assert differing <= self.PATH_KEYS, differing
        assert treatment["pretrained_branch_expected_seal"] == \
            design["shared_bindings"]["pretrain_generation"][
                "seal_manifest_sha256"]

    def test_shared_facts_are_design_bound(self, design, tmp_path):
        cfg = build_cell_config(
            design, self._cell(design, 202, "control_random_init"),
            Path("/pretrain"), tmp_path, device="cpu")
        sac = design["shared_bindings"]["sac"]
        assert cfg["learning_rate"] == sac["learning_rate"]
        assert cfg["total_timesteps"] == sac["budget_total_timesteps"]
        assert cfg["train_seed"] == 202 and cfg["eval_seed"] == 202
        assert cfg["pipeline_plugin"] == "rl_pipeline_with_validation"
        assert cfg["selection_metric"] == \
            "paired_generalization_weekly_v1"
        assert cfg["strategy_plugin"] == "shared_execution_envelope"
        # ALPACA cost contract (~30.5bp/side), not the raw exchange fee
        assert cfg["commission"] == pytest.approx(0.00295215)
        assert cfg["slippage_perc"] == pytest.approx(0.0001)
        assert cfg["evaluate_test_split"] is False
        assert cfg["require_observation_declaration"] is True
        assert cfg["l1_patience"] == 40

    def test_dry_run_budget_is_disclosed_and_bounded(
            self, design, tmp_path):
        cfg = build_cell_config(
            design, self._cell(design, 303, "control_random_init"),
            Path("/pretrain"), tmp_path, device="cpu",
            dry_run_budget=DRY_RUN_BUDGET)
        assert cfg["dry_run_budget_disclosed"] == DRY_RUN_BUDGET
        assert cfg["total_timesteps"] <= 5000

    def test_nested_contract_seals_2024_and_2025(self):
        contract = json.loads(
            (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
             "eth_nested_split_contract_o2022_paired_v1.json"
             ).read_text())
        roles = contract["roles"]
        # fit ends at the pretraining origin fit_end
        assert roles["fit_train"]["end"] == "2022-01-01T00:00:00"
        # the scored year is inner validation 2022
        assert roles["inner_validation"]["start"] == \
            "2022-01-01T00:00:00"
        # 2024 AND sealed 2025 live inside sealed_test, which mode
        # l1 refuses to materialize — structurally unavailable
        assert roles["sealed_test"]["start"] == "2024-01-01T00:00:00"
        assert roles["sealed_test"]["end"] == "2026-01-01T00:00:00"
