"""M1 regressions (multitask gradient mechanism order 2026-08-27):
optimizer mechanism plugins — frozen gradient-norm balancing, PCGrad
projection math/determinism, strict head/encoder isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pretrain_optimizer_plugins.balancing_frozen_gradient_norm import (  # noqa: E402,E501
    Plugin as FrozenGradNorm)
from pretrain_optimizer_plugins.balancing_inverse_initial_loss import (  # noqa: E402,E501
    Plugin as InverseLoss)
from pretrain_optimizer_plugins.combiner_ordinary_sum import (  # noqa: E402
    Plugin as OrdinarySum)
from pretrain_optimizer_plugins.combiner_pcgrad import (  # noqa: E402
    Plugin as PCGrad)


class TestMechanismPluginsResolveThroughEntryPoints:
    def test_real_entry_points(self):
        from importlib.metadata import entry_points
        balancing = {e.name for e in entry_points().select(
            group="pretrain_balancing.plugins")}
        combiner = {e.name for e in entry_points().select(
            group="pretrain_combiner.plugins")}
        assert {"inverse_initial_loss",
                "frozen_gradient_norm"} <= balancing
        assert {"ordinary_sum", "pcgrad"} <= combiner


class TestFrozenGradientNormBalancing:
    def test_formula_and_provenance(self):
        weights, provenance = FrozenGradNorm.compute(
            declared_weights={"a": 1.0, "b": 2.0},
            initial_calibration_losses={"a": 9.0, "b": 9.0},
            calibration_gradient_norms={"a": 0.5, "b": 4.0},
            params={"floor": 1e-8})
        assert weights["a"] == pytest.approx(2.0)   # 1/0.5
        assert weights["b"] == pytest.approx(0.5)   # 2/4
        assert provenance["method"] == "frozen_gradient_norm"
        assert provenance["source"].startswith("calibration")
        # losses (and any monitor value) CANNOT influence the result
        weights2, _ = FrozenGradNorm.compute(
            declared_weights={"a": 1.0, "b": 2.0},
            initial_calibration_losses={"a": 1e9, "b": 1e-9},
            calibration_gradient_norms={"a": 0.5, "b": 4.0},
            params={"floor": 1e-8})
        assert weights == weights2

    def test_floor_bounds_zero_norms(self):
        weights, _ = FrozenGradNorm.compute(
            declared_weights={"a": 1.0},
            initial_calibration_losses={"a": 1.0},
            calibration_gradient_norms={"a": 0.0},
            params={"floor": 1e-8})
        assert weights["a"] == pytest.approx(1e8)

    def test_inverse_loss_control_unchanged(self):
        weights, provenance = InverseLoss.compute(
            declared_weights={"a": 1.0}, 
            initial_calibration_losses={"a": 2.0},
            calibration_gradient_norms={"a": 999.0},
            params={"floor": 1e-6})
        assert weights["a"] == pytest.approx(0.5)
        assert provenance["method"] == "inverse_initial_loss"


class TestPCGradCombiner:
    def test_exact_projection_math(self):
        g = {"a": torch.tensor([1.0, 0.0]),
             "b": torch.tensor([-1.0, 1.0])}
        combined, report = PCGrad.combine(
            g, {"epsilon": 1e-12, "order": "sorted_objective_names"})
        # a' = a - (a.b/|b|^2) b = (0.5, 0.5); b' = b - (b.a/|a|^2) a
        # = (0, 1); combined = (0.5, 1.5)
        assert torch.allclose(combined, torch.tensor([0.5, 1.5]))
        assert report["projections"] == 2
        assert report["pre_negative_pairs"] == 1
        assert report["post_negative_pairs"] == 0

    def test_deterministic_and_order_declared(self):
        g = {"x": torch.randn(64), "y": torch.randn(64),
             "z": torch.randn(64)}
        first, r1 = PCGrad.combine(
            {k: v.clone() for k, v in g.items()},
            {"epsilon": 1e-12, "order": "sorted_objective_names"})
        second, r2 = PCGrad.combine(
            {k: v.clone() for k, v in g.items()},
            {"epsilon": 1e-12, "order": "sorted_objective_names"})
        assert torch.equal(first, second) and r1 == r2

    def test_zero_gradient_is_skipped_not_divided(self):
        g = {"a": torch.tensor([1.0, 1.0]),
             "b": torch.tensor([0.0, 0.0])}
        combined, report = PCGrad.combine(
            g, {"epsilon": 1e-12, "order": "sorted_objective_names"})
        assert torch.allclose(combined, torch.tensor([1.0, 1.0]))
        assert torch.isfinite(combined).all()

    def test_non_conflicting_gradients_untouched(self):
        g = {"a": torch.tensor([1.0, 0.0]),
             "b": torch.tensor([1.0, 1.0])}
        combined, report = PCGrad.combine(
            g, {"epsilon": 1e-12, "order": "sorted_objective_names"})
        assert torch.allclose(combined, torch.tensor([2.0, 1.0]))
        assert report["projections"] == 0

    def test_ordinary_sum_control(self):
        g = {"a": torch.tensor([1.0, 2.0]),
             "b": torch.tensor([3.0, -1.0])}
        combined, report = OrdinarySum.combine(g, {})
        assert torch.allclose(combined, torch.tensor([4.0, 1.0]))
        assert report["projections"] == 0


class TestHeadEncoderIsolation:
    def test_each_head_receives_only_its_own_objective_gradient(self):
        """M1: replicate the runner's isolation loop — scaling the
        OTHER objective's loss by 1000x must leave this head's gradient
        bitwise unchanged (projection can never mix heads)."""
        torch.manual_seed(0)
        encoder = torch.nn.Linear(6, 4)
        head_a = torch.nn.Linear(4, 1)
        head_b = torch.nn.Linear(4, 1)
        x = torch.randn(8, 6)

        def head_grads(scale_b):
            for module in (encoder, head_a, head_b):
                for parameter in module.parameters():
                    parameter.grad = None
            z = encoder(x)
            losses = {"a": head_a(z).square().mean(),
                      "b": scale_b * head_b(z).abs().mean()}
            heads = {"a": head_a, "b": head_b}
            for name, loss in losses.items():
                grads = torch.autograd.grad(
                    loss, list(heads[name].parameters()),
                    retain_graph=True, allow_unused=True)
                for parameter, grad in zip(heads[name].parameters(),
                                           grads):
                    parameter.grad = grad
            return [parameter.grad.clone()
                    for parameter in head_a.parameters()]
        base = head_grads(1.0)
        scaled = head_grads(1000.0)
        for g1, g2 in zip(base, scaled):
            assert torch.equal(g1, g2)

    def test_runner_source_separates_optimizers(self):
        source = (REPO / "tools/pretrain_branches.py").read_text()
        assert "encoder_opt = torch.optim.Adam(encoder_parameters" in \
            source
        assert "head_opt = torch.optim.Adam(heads.parameters()" in \
            source
        assert "combiner_class.combine(" in source
