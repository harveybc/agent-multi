"""WP1 regressions (post-transfer objectives order 2026-08-27):
hierarchical contrastive, volatility and barrier-hit — formulas,
causality, calibration-only class weights, collapse/negative
diagnostics and contract refusals.
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
    PretrainContractError, barrier_hit_labels, barrier_loss,
    build_projection_head, causal_scale_view, frozen_class_weights,
    hierarchical_contrastive_loss, realized_volatility_targets,
    validate_contract)
from tests.unit.test_branch_pretraining import contract_with  # noqa: E402

FULL5 = json.loads((REPO / "examples/config/"
                    "pretrain_contract_eth_h4_o2022_full5_v1.json"
                    ).read_text())


# ------------------------------------------------------------ volatility

class TestVolatilityObjective:
    def test_exact_declared_formula(self):
        # constant 1% log return per bar -> vol == 0.01 exactly
        n = 60
        close = 100.0 * np.exp(0.01 * np.arange(n))
        got = realized_volatility_targets(close, steps=[40], horizons=[4],
                                          epsilon=0.0,
                                          periods_per_year=None)
        assert got[0, 0] == pytest.approx(np.log(0.01), abs=1e-6)

    def test_strictly_forward_from_anchor(self):
        rng = np.random.default_rng(3)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
        base = realized_volatility_targets(close, [50], [4], 1e-8, None)
        mutated = close.copy()
        mutated[56:] *= 3.0  # beyond anchor+h: must not matter
        same = realized_volatility_targets(mutated, [50], [4], 1e-8,
                                           None)
        assert base[0, 0] == same[0, 0]
        mutated2 = close.copy()
        mutated2[52] *= 1.5  # inside the forward horizon: must matter
        changed = realized_volatility_targets(mutated2, [50], [4], 1e-8,
                                              None)
        assert changed[0, 0] != base[0, 0]

    def test_annualization_is_declared_multiplication(self):
        close = 100.0 * np.exp(0.01 * np.arange(30))
        plain = realized_volatility_targets(close, [20], [4], 0.0, None)
        annual = realized_volatility_targets(close, [20], [4], 0.0,
                                             periods_per_year=2190)
        assert annual[0, 0] == pytest.approx(
            plain[0, 0] + 0.5 * np.log(2190), abs=1e-6)


# ------------------------------------------------------------ barrier hit

class TestBarrierHitObjective:
    @staticmethod
    def _flat_then(path):
        base = np.full(100, 100.0)
        base[80:80 + len(path)] = path
        return base

    def test_upper_lower_and_censored_labels(self):
        up = self._flat_then([100, 130, 130, 130])  # jump above upper
        down = self._flat_then([100, 70, 70, 70])
        flat = np.full(100, 100.0) * np.exp(
            np.random.default_rng(0).normal(0, 1e-5, 100).cumsum())
        for series, expected in ((up, 0), (down, 1), (flat, 2)):
            labels = barrier_hit_labels(series, steps=[81], horizons=[4],
                                        lookback=16, upper_mult=2.0,
                                        lower_mult=2.0, epsilon=1e-4)
            assert labels[0, 0] == expected, series[78:86]

    def test_first_hit_beyond_horizon_is_censored(self):
        series = self._flat_then([100, 100, 100, 100, 100, 130])
        labels = barrier_hit_labels(series, steps=[81], horizons=[3],
                                    lookback=16, upper_mult=2.0,
                                    lower_mult=2.0, epsilon=1e-4)
        assert labels[0, 0] == 2  # the hit at +5 is outside h=3

    def test_scale_is_past_only(self):
        rng = np.random.default_rng(7)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        a = barrier_hit_labels(close, [150], [4], 32, 2.0, 2.0, 1e-8)
        mutated = close.copy()
        mutated[160:] *= 2.0  # future beyond horizon: scale unchanged,
        b = barrier_hit_labels(mutated, [150], [4], 32, 2.0, 2.0, 1e-8)
        assert a[0, 0] == b[0, 0]

    def test_adverse_first_on_simultaneous_cross(self):
        """With close-only labeling a single close cannot cross both
        barriers; the declared rule is still enforced in code: when the
        first lower hit is at or before the first upper hit, LOWER
        (adverse) wins."""
        # crash through lower on the same bar a later upper hit occurs
        series = self._flat_then([100, 60, 140, 140])
        labels = barrier_hit_labels(series, [81], [4], 16, 2.0, 2.0,
                                    1e-4)
        assert labels[0, 0] == 1

    def test_class_weights_from_calibration_only_and_frozen_form(self):
        labels = np.array([[0], [0], [0], [1], [2], [2]])
        weights = frozen_class_weights(labels)
        assert weights[0][0] == pytest.approx(6 / (3 * 3))
        assert weights[0][1] == pytest.approx(6 / (3 * 1))
        assert weights[0][2] == pytest.approx(6 / (3 * 2))
        absent = frozen_class_weights(np.array([[0], [0]]))
        assert absent[0][1] == 1.0 and absent[0][2] == 1.0

    def test_barrier_loss_finite_and_weighted(self):
        pred = torch.randn(8, 2 * 3)
        labels = torch.randint(0, 3, (8, 2))
        weights = [[1.0, 2.0, 0.5], [1.0, 1.0, 1.0]]
        loss = barrier_loss(pred, labels, weights)
        assert torch.isfinite(loss)


# ------------------------------------------------- hierarchical contrastive

class TestHierarchicalContrastive:
    def test_causal_scale_view_is_in_window_smoothing(self):
        x = torch.arange(8.0).view(1, 8, 1)
        smoothed = causal_scale_view(x, 2)
        # pairs (0,1)(2,3)(4,5)(6,7) -> means repeated
        assert smoothed.view(-1).tolist() == [0.5, 0.5, 2.5, 2.5,
                                              4.5, 4.5, 6.5, 6.5]

    def test_temporal_neighbors_are_excluded_negatives(self):
        enc = torch.nn.Sequential(torch.nn.Flatten(),
                                  torch.nn.Linear(16 * 2, 8))
        proj = build_projection_head(8, 4)
        win = torch.randn(6, 16, 2)
        positions = [100, 101, 102, 500, 600, 700]
        _loss, diag = hierarchical_contrastive_loss(
            enc, proj, win, positions, [2], 0.2, exclusion_steps=5)
        # the three clustered anchors lose two neighbors each
        assert diag["effective_negatives_mean"] == pytest.approx(
            (3 * 3 + 3 * 5) / 6)

    def test_collapse_is_visible_in_diagnostics(self):
        class Collapsed(torch.nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], 8)
        proj = build_projection_head(8, 4)
        _loss, diag = hierarchical_contrastive_loss(
            Collapsed(), proj, torch.randn(6, 16, 2),
            list(range(0, 600, 100)), [2], 0.2, 1)
        assert diag["embedding_std"] == 0.0  # collapse detectable

    def test_gradients_flow_to_encoder(self):
        enc = torch.nn.Sequential(torch.nn.Flatten(),
                                  torch.nn.Linear(16 * 2, 8))
        proj = build_projection_head(8, 4)
        loss, _ = hierarchical_contrastive_loss(
            enc, proj, torch.randn(6, 16, 2),
            list(range(0, 600, 100)), [2, 4], 0.2, 1)
        loss.backward()
        grads = [p.grad.abs().sum().item() for p in enc.parameters()]
        assert sum(grads) > 0


# ------------------------------------------------------- contract refusals

class TestNewObjectiveContractRefusals:
    def _with_objective(self, name, spec):
        contract = contract_with()
        contract["objectives"][name] = spec
        return contract

    @pytest.mark.parametrize("mutation, fragment", [
        ({"estimator": "my_vol"}, "EXPLICITLY"),
        ({"estimator": None}, "EXPLICITLY"),
        ({"units": ""}, "units"),
        ({"annualization": {"periods_per_year": True}}, "annualization"),
        ({"epsilon": 0.0}, "epsilon"),
    ], ids=["wrong-estimator", "no-estimator", "no-units",
            "bool-ppy", "zero-eps"])
    def test_volatility_refusals(self, mutation, fragment):
        spec = {"weight": 1.0, "horizons": [3, 6],
                "estimator": "realized_vol_close_to_close",
                "units": "log(vol+eps)", "annualization": "none",
                "epsilon": 1e-8}
        spec.update(mutation)
        with pytest.raises(PretrainContractError, match=fragment):
            validate_contract(self._with_objective("volatility", spec))

    @pytest.mark.parametrize("mutation, fragment", [
        ({"same_bar_collision": "upper_first"}, "conservative"),
        ({"class_weights_from": "train"}, "calibration_only"),
        ({"barrier_scale": {"estimator": "atr", "lookback": 16,
                            "epsilon": 1e-8}}, "past-only|EXPLICITLY"),
        ({"barrier_scale": {"estimator":
            "trailing_realized_vol_close_to_close",
            "lookback": 999, "epsilon": 1e-8}}, "warmup"),
        ({"upper_mult": 0.0}, "upper_mult"),
    ], ids=["collision-rule", "weights-source", "scale-estimator",
            "lookback-gt-warmup", "zero-mult"])
    def test_barrier_refusals(self, mutation, fragment):
        spec = {"weight": 1.0, "horizons": [4],
                "barrier_scale": {"estimator":
                                  "trailing_realized_vol_close_to_close",
                                  "lookback": 16, "epsilon": 1e-8},
                "upper_mult": 2.0, "lower_mult": 2.0,
                "same_bar_collision": "conservative_adverse_first",
                "class_weights_from": "calibration_only"}
        spec.update(mutation)
        with pytest.raises(PretrainContractError, match=fragment):
            validate_contract(self._with_objective("barrier_hit", spec))

    @pytest.mark.parametrize("mutation, fragment", [
        ({"temperature": 0.0}, "temperature"),
        ({"scales": [1, 2]}, "scales"),
        ({"scales": [2, 2]}, "scales"),
        ({"negatives": {"source": "anywhere", "exclusion_steps": 1,
                        "false_negative_policy": "x"}}, "train_only"),
        ({"negatives": {"source": "train_only", "exclusion_steps": 1,
                        "false_negative_policy": ""}},
         "false_negative_policy"),
    ], ids=["zero-temp", "scale-1", "dup-scale", "neg-source",
            "no-fn-policy"])
    def test_contrastive_refusals(self, mutation, fragment):
        spec = {"weight": 1.0, "scales": [2, 4], "temperature": 0.2,
                "projection_dim": 8,
                "negatives": {"source": "train_only",
                              "exclusion_steps": 3,
                              "false_negative_policy": "declared"}}
        spec.update(mutation)
        with pytest.raises(PretrainContractError, match=fragment):
            validate_contract(self._with_objective(
                "hierarchical_contrastive", spec))

    def test_purge_uses_max_horizon_across_all_objectives(self):
        contract = contract_with()
        contract["objectives"]["volatility"] = {
            "weight": 1.0, "horizons": [30],
            "estimator": "realized_vol_close_to_close",
            "units": "log(vol+eps)", "annualization": "none",
            "epsilon": 1e-8}
        parsed = validate_contract(contract)
        assert parsed["max_horizon_all_objectives"] == 30

    def test_committed_full5_contract_validates(self):
        parsed = validate_contract(FULL5)
        assert parsed["max_horizon_all_objectives"] == 12
        assert set(FULL5["objectives"]) == {
            "masked_patch_reconstruction", "multi_horizon_quantile",
            "hierarchical_contrastive", "volatility", "barrier_hit"}
