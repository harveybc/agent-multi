"""Frozen counterexamples for DATA-SOTA-341..346 (WP-PRETRAIN M1
correction order, agent-multi@e9af87a3). Reproductions of the original
defects live in docs/audits/evidence/DATA_SOTA_341_346_REPRODUCTIONS
.json; each case here is the PERMANENT regression that keeps the
corrected behavior from regressing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    PretrainContractError, balance_objective_weights,
    build_monotone_quantile_head, load_generation,
    masked_visible_normalize, objective_gradient_diagnostics,
    quantile_crossing_rate, reconstruction_target, resume_identity,
    validate_contract, write_generation)
from tests.unit.test_branch_pretraining import contract_with  # noqa: E402


# ---------------------------------------------------- 341: origin roles

class TestDataSota341CausalPerOriginRoles:
    def test_v1_style_fit_end_through_2023_refuses_for_o2022(self):
        """The original defect: fit_end=2023-12-31 with score-2022
        trains on monitor-2022 and inner-validation-2023."""
        contract = contract_with(
            score_origin={"origin_id": "o2022",
                          "score_start": "2022-01-01"},
            fit_end="2023-12-31T20:00:00")
        with pytest.raises(PretrainContractError,
                           match="DATA-SOTA-341|reserved"):
            validate_contract(contract)

    def test_fit_end_strictly_before_score_start_passes(self):
        contract = contract_with(
            score_origin={"origin_id": "o2022",
                          "score_start": "2022-01-01"},
            fit_end="2021-12-31T20:00:00")
        parsed = validate_contract(contract)
        assert parsed["origin_id"] == "o2022"

    def test_later_origin_requires_typed_earlier_decision(self):
        contract = contract_with(
            score_origin={"origin_id": "o2023",
                          "score_start": "2023-01-01"},
            fit_end="2022-12-31T20:00:00")
        with pytest.raises(PretrainContractError,
                           match="typed earlier_origin_decision"):
            validate_contract(contract)
        # DATA-SOTA-347: a bare string can no longer mint an origin
        contract["earlier_origin_decision_frozen"] = "yes"
        with pytest.raises(PretrainContractError,
                           match="typed earlier_origin_decision"):
            validate_contract(contract)
        contract["earlier_origin_decision"] = {
            "origin_id": "o2022", "decided_at": "2022-12-01T00:00:00Z",
            "artifact": "decision.json", "artifact_sha256": "00"}
        contract["materialized_at"] = "2022-12-15T00:00:00Z"
        assert validate_contract(contract)["origin_id"] == "o2023"

    def test_committed_v1_contract_can_no_longer_run(self):
        """The superseded v1 contract file itself refuses at the
        validator (schema v1 is dead)."""
        v1 = json.loads((REPO / "examples/config/"
                         "pretrain_contract_eth_h4_v1.json").read_text())
        with pytest.raises(PretrainContractError, match="unsupported"):
            validate_contract(v1)

    def test_committed_v2_contract_can_no_longer_run(self):
        v2 = json.loads((REPO / "examples/config/"
                         "pretrain_contract_eth_h4_o2022_v2.json"
                         ).read_text())
        with pytest.raises(PretrainContractError, match="unsupported"):
            validate_contract(v2)

    def test_committed_o2022_v3_contract_validates(self):
        v3 = json.loads((REPO / "examples/config/"
                         "pretrain_contract_eth_h4_o2022_v3.json"
                         ).read_text())
        parsed = validate_contract(v3)
        assert parsed["fit_end"].year == 2021
        assert parsed["origin_id"] == "o2022"


# ------------------------------------------- 342: executing preprocessor

class TestDataSota342ExecutingPipelineParity:
    def test_contract_requires_executing_preprocessor(self):
        contract = contract_with(observation_pipeline={
            "preprocessor_plugin": "my_own_zscore",
            "source_config": "x.json"})
        with pytest.raises(PretrainContractError,
                           match="executing preprocessor"):
            validate_contract(contract)

    def test_collector_emits_the_exact_executing_tensor(self, tmp_path):
        """collect_preprocessed_windows output equals a direct call to
        the executing plugin's make_observation — one shared transform.
        (Env-level bitwise parity is proven in the Tier-A integration
        module against the real GymFxEnv.)"""
        import pandas as pd

        from agent_plugins.branch_pretraining import (
            collect_preprocessed_windows)
        from app.plugin_loader import load_plugin

        rng = np.random.default_rng(5)
        n = 120
        df = pd.DataFrame({
            "DATE_TIME": pd.date_range("2021-01-01", periods=n,
                                       freq="4h"),
            **{f"f{i}": rng.normal(size=n) * 10 ** i
               for i in range(1, 7)},
            "CLOSE": np.exp(np.cumsum(rng.normal(0, 0.01, n))) * 100})
        env_cfg = {"window_size": 16,
                   "feature_columns": [f"f{i}" for i in range(1, 7)],
                   "feature_binary_columns": ["f6"],
                   "feature_scaling": "rolling_zscore",
                   "feature_scaling_window": 32,
                   "include_price_window": False,
                   "include_agent_state": False}
        contract = contract_with()
        steps = [40, 41, 77]
        got = collect_preprocessed_windows(df, contract, env_cfg, steps)
        Pre, _ = load_plugin("preprocessor.plugins",
                             "feature_window_preprocessor")
        pre = Pre()
        for i, t in enumerate(steps):
            expect = pre.make_observation(
                data=df, step=t,
                bridge_state={"initial_cash": 1000.0, "equity": 1000.0,
                              "price": 0.0, "position": 0,
                              "bar_index": t, "total_bars": n},
                config=env_cfg)["features"]
            assert np.array_equal(got[i], expect), f"step {t} diverges"
        # the executing transform is NOT a per-window z-score: binary
        # channel f6 passes through UNSCALED (then the executing clip
        # of ±10 still applies — captured behavior, real binaries are
        # 0/1 and never reach it)
        raw = df[[f"f{i}" for i in range(1, 7)]].to_numpy()[24:40, 5]
        assert np.allclose(got[0][:, 5],
                           np.clip(raw, -10.0, 10.0).astype(np.float32))


# ------------------------------------------------- 343: mask leakage

class TestDataSota343MaskStatisticLeakage:
    def test_masked_values_cannot_change_visible_inputs(self):
        """The frozen adversary: perturb ONLY masked raw values — the
        visible normalized model inputs must remain IDENTICAL."""
        base = torch.randn(3, 16, 4)
        perturbed = base.clone()
        mask = torch.zeros(3, 16, dtype=torch.bool)
        mask[:, 5:9] = True
        perturbed[:, 5:9, :] += 1000.0
        visible = ~mask
        for policy in ({"policy": "identity_preprocessed", "eps": None},
                       {"policy": "window_zscore_visible",
                        "eps": 1e-5}):
            a = reconstruction_target(base, mask, policy)
            b = reconstruction_target(perturbed, mask, policy)
            assert torch.equal(a[visible], b[visible]), (
                f"{policy['policy']}: masked raw values leaked into "
                f"visible target values")
            # the encoder INPUT is policy-independent by construction
            assert torch.equal(
                base.masked_fill(mask.unsqueeze(-1), 0.0)[visible],
                perturbed.masked_fill(mask.unsqueeze(-1),
                                      0.0)[visible])

    def test_visible_only_statistics_are_actually_visible_only(self):
        windows = torch.randn(2, 12, 3)
        mask = torch.zeros(2, 12, dtype=torch.bool)
        mask[:, :4] = True
        normalized = masked_visible_normalize(windows, mask, eps=0.0)
        visible = normalized[:, 4:, :]
        assert torch.allclose(visible.mean(dim=1),
                              torch.zeros(2, 3), atol=1e-5)


# ------------------------------------------------- 344: typed policies

class TestDataSota344TypedNormalizationPolicies:
    def test_declared_eps_reaches_execution(self):
        """The original defect: contract eps changed the digest but not
        the computation."""
        windows = torch.randn(2, 16, 3)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[:, 2:5] = True
        small = reconstruction_target(
            windows, mask, {"policy": "window_zscore_visible",
                            "eps": 1e-5})
        large = reconstruction_target(
            windows, mask, {"policy": "window_zscore_visible",
                            "eps": 1.0})
        assert not torch.allclose(small, large)

    def test_zscore_policy_requires_eps(self):
        contract = contract_with(normalization_policies={
            "alpha": {"policy": "window_zscore_visible"},
            "beta": {"policy": "identity_preprocessed"}})
        with pytest.raises(PretrainContractError, match="eps"):
            validate_contract(contract)

    def test_unknown_policy_refuses(self):
        contract = contract_with(normalization_policies={
            "alpha": {"policy": "raw_passthrough"},
            "beta": {"policy": "identity_preprocessed"}})
        with pytest.raises(PretrainContractError, match="must be one"):
            validate_contract(contract)

    def test_committed_policy_evidence_covers_every_family(self):
        evidence = json.loads(
            (REPO / "docs/audits/evidence/"
             "PRETRAIN_NORMALIZATION_POLICY_EVIDENCE_2026_08_26.json"
             ).read_text())
        v3 = json.loads((REPO / "examples/config/"
                         "pretrain_contract_eth_h4_o2022_v3.json"
                         ).read_text())
        for family in v3["normalization_policies"]:
            stats = evidence["families"][family]
            assert stats["assigned_policy"] == \
                v3["normalization_policies"][family]["policy"]
            # the identity policy is justified by ~unit scale
            assert abs(stats["mean"]) < 0.5 and 0.5 < stats["std"] < 2.0


# ------------------------------------------------ 345: balancing

class TestDataSota345ObjectiveBalancing:
    def test_inverse_initial_loss_rule(self):
        weights = balance_objective_weights(
            {"reconstruction": 2.0, "quantile": 0.02},
            {"reconstruction": 1.0, "quantile": 1.0}, floor=1e-6)
        assert weights["reconstruction"] == pytest.approx(0.5)
        assert weights["quantile"] == pytest.approx(50.0)
        # after balancing, initial contributions are equal — the
        # original 1.0/1.0 weighting let reconstruction dominate 100:1
        assert weights["reconstruction"] * 2.0 == pytest.approx(
            weights["quantile"] * 0.02)

    def test_floor_bounds_the_rule(self):
        weights = balance_objective_weights(
            {"quantile": 0.0}, {"quantile": 1.0}, floor=1e-6)
        assert weights["quantile"] == pytest.approx(1e6)

    def test_monotone_head_is_structurally_non_crossing(self):
        torch.manual_seed(0)
        head = build_monotone_quantile_head(8, n_horizons=3,
                                            n_quantiles=4)
        pred = head(torch.randn(64, 8))
        assert pred.shape == (64, 3, 4)
        assert bool((pred[..., 1:] >= pred[..., :-1]).all())
        assert quantile_crossing_rate(pred) == 0.0

    def test_plain_linear_head_would_cross(self):
        """The frozen v1 counterexample: an unconstrained linear head
        produces quantile crossings."""
        torch.manual_seed(0)
        linear = torch.nn.Linear(8, 12)
        pred = linear(torch.randn(64, 8)).view(64, 3, 4)
        assert quantile_crossing_rate(pred) > 0.0

    def test_gradient_diagnostics_report_norms_and_cosines(self):
        encoder = torch.nn.Linear(6, 4)
        x = torch.randn(8, 6)
        losses = {"a": encoder(x).square().mean(),
                  "b": encoder(x).abs().mean()}
        report = objective_gradient_diagnostics(encoder, losses)
        assert report["norms"]["a"] > 0 and report["norms"]["b"] > 0
        assert -1.0 <= report["cosine:a|b"] <= 1.0


# ------------------------------------------------ 346: durability

class TestDataSota346ResumeDurability:
    def test_resume_identity_binds_every_identity_field(self):
        """The original defect: runner sha, code commit, torch version
        and normalization identity were omitted from resume binding."""
        identity = {"runner_sha256": "r", "code_commit": "c",
                    "torch_version": "t",
                    "normalization_policies_digest": "n",
                    "data_sha256": "d"}
        bound = resume_identity({"identity": identity})
        assert bound == identity  # complete, not a subset

    def test_torn_generation_refuses(self, tmp_path):
        write_generation(tmp_path, {"identity": {"x": 1},
                                    "branch_index": 0,
                                    "epochs_done_in_branch": 1},
                         {"schema": "m"}, generation=1)
        ckpt, manifest, generation = load_generation(tmp_path)
        assert generation == 1 and ckpt["identity"] == {"x": 1}
        manifest_path = tmp_path / "pretrain_manifest.json"
        manifest_path.write_text(manifest_path.read_text() + " ")
        with pytest.raises(PretrainContractError,
                           match="TORN GENERATION"):
            load_generation(tmp_path)

    def test_missing_seal_refuses(self, tmp_path):
        with pytest.raises(PretrainContractError,
                           match="no sealed generation"):
            load_generation(tmp_path)

    def test_generation_files_have_no_tmp_leftovers(self, tmp_path):
        write_generation(tmp_path, {"identity": {}, "branch_index": 0,
                                    "epochs_done_in_branch": 1},
                         {"schema": "m"}, generation=1)
        assert not list(tmp_path.glob("*.tmp"))
