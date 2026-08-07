"""M0 contract tests (SAC inner-curriculum order §8/§12).

Invalid factors must fail BEFORE model construction; compute is equal
across arms; the anchor is the exact hash-verified D1 artifact; 2025
never enters selection; D1 records stay untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.eth_sac_inner_curriculum_screen import (
    CONTRACT_PATH,
    M0_ARM_ORDER,
    load_contract,
    resolve_anchor,
    validate_contract,
)

REPO = Path(__file__).resolve().parent.parent
D1_ROOT = Path.home() / ".local/share/agent-multi/eth_curriculum_decision_20260807_v2"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


class TestFrozenFactors:
    def test_contract_loads_and_validates(self):
        contract = load_contract()
        assert set(contract["arms"]) == set(M0_ARM_ORDER)

    def test_equal_total_compute_across_all_arms(self):
        contract = _contract()
        for name, spec in contract["arms"].items():
            total = (spec["easy_epochs"] + spec["normal_epochs"]) * \
                contract["epoch_timesteps"]
            assert total == contract["total_updates_per_arm"], name

    def test_only_declared_factors_vary(self):
        contract = _contract()
        # the ONLY differences between arms are schedule and normal LR
        rates = {spec["normal_learning_rate"]
                 for spec in contract["arms"].values()}
        assert rates == {1e-4, 3e-5, 1e-5}
        assert contract["easy_learning_rate"] == 1e-4

    @pytest.mark.parametrize("mutation,message", [
        ({"arms": {"N2_LR1": {"easy_epochs": 0, "normal_epochs": 3,
                              "normal_learning_rate": 1e-4}}},
         "unequal compute"),
        ({"arms": {"N2_LR1": {"easy_epochs": 0, "normal_epochs": 2,
                              "normal_learning_rate": -1e-4}}},
         "finite and positive"),
        ({"arms": {"N2_LR1": {"easy_epochs": 0, "normal_epochs": 2,
                              "normal_learning_rate": True}}},
         "must be a number"),
        ({"arms": {"N2_LR1": {"easy_epochs": -1, "normal_epochs": 3,
                              "normal_learning_rate": 1e-4}}},
         "negative"),
        ({"arms": {"BOGUS_ARM": {"easy_epochs": 0, "normal_epochs": 2,
                                 "normal_learning_rate": 1e-4}}},
         "unknown or missing arms"),
        ({"easy_learning_rate": float("nan")}, "positive finite"),
    ])
    def test_invalid_factors_fail_before_model_construction(
        self, mutation, message
    ):
        contract = _contract()
        for key, value in mutation.items():
            if key == "arms":
                contract["arms"] = {**contract["arms"], **value}
                for name in value:
                    if name == "BOGUS_ARM":
                        contract["arms"].pop("E1_N1_LR01")
            else:
                contract[key] = value
        with pytest.raises(ValueError, match=message):
            validate_contract(contract)


class TestAnchorsAndEvidence:
    def test_anchor_hashes_match_the_exact_d1_artifacts(self):
        if not D1_ROOT.exists():
            pytest.skip("D1 evidence root not present on this host")
        contract = load_contract()
        for seed in (101, 202, 303, 404):
            path = resolve_anchor(contract, seed)
            assert path.is_file()

    def test_anchor_hash_mismatch_refuses(self, tmp_path):
        contract = load_contract()
        fake = tmp_path / "anchor_seed101.zip"
        fake.write_bytes(b"not the anchor")
        contract["anchors"]["101"] = {
            "path": str(fake),
            "sha256": contract["anchors"]["101"]["sha256"],
        }
        with pytest.raises(ValueError, match="hash mismatch"):
            resolve_anchor(contract, 101)

    def test_d1_records_remain_loadable_and_unchanged(self):
        if not D1_ROOT.exists():
            pytest.skip("D1 evidence root not present on this host")
        expected = {
            "decision_summary.json":
                "3f3eeb940b04317c3bcc976a7e6bb230b38ce8ab6d23cdd6"
                "212701f9f9f85239",
            "fleet_manifest.json":
                "0f39d7e8e9e7c8d6a9fb007e8ca166950f4d335e2f79bf42"
                "84fcf13f7993c6e2",
        }
        for name, digest in expected.items():
            actual = hashlib.sha256((D1_ROOT / name).read_bytes()).hexdigest()
            assert actual == digest, f"D1 evidence {name} changed"

    def test_2025_cannot_enter_selection(self):
        # the M0 base config comes from the D1 runner, whose splits pin
        # validation to calendar 2024 and disable the 2025 test split
        from tools import eth_curriculum_decision_experiment as d1

        assert d1.SPLITS["validation_end"].startswith("2024-12-31")
        assert "test" not in d1.ALLOWED_SPLITS
        config = d1._base_config(Path("/tmp/m0_probe"), "N2_LR1", 101,
                                 epoch_timesteps=20000)
        assert config["evaluate_test_split"] is False


class TestEasyLearningRateOverride:
    def test_easy_keeps_its_own_rate_while_normal_varies(self):
        from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
            PipelinePlugin,
        )

        pipeline = PipelinePlugin({})
        config = {"learning_rate": 1e-5, "easy_learning_rate": 1e-4}
        easy = pipeline._easy_training_config(config)
        assert easy["learning_rate"] == 1e-4
        assert config["learning_rate"] == 1e-5

    @pytest.mark.parametrize("bad", [True, -1e-4, 0.0, float("nan"), "fast"])
    def test_invalid_easy_rate_refuses(self, bad):
        from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
            PipelinePlugin,
        )

        pipeline = PipelinePlugin({})
        with pytest.raises(ValueError):
            pipeline._easy_training_config(
                {"easy_learning_rate": bad})
