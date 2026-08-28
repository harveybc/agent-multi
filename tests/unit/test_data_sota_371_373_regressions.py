"""P0 permanent adversarial tests (final probe order 2026-08-28;
DATA-SOTA-371/372/373): five-way ordering/purge/boundaries, frozen
cached encoders (adapter-only gradients), validated adapter fitting,
skill formula refusals, cardinality-invariant ranking, deterministic
replay, probe-score/monitor isolation. Model-free beyond tiny torch
modules.
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
    PretrainContractError, five_way_split)
from agent_plugins.objective_routing import (  # noqa: E402
    ProbeRefusal, fit_adapter_validated, normalized_skill,
    split_adapter_train_val)

PROTOCOL = {"adapter_seeds": [1234, 5678, 9012], "max_steps": 400,
            "min_steps": 40, "validation_cadence_steps": 10,
            "patience_steps": 60, "minimum_improvement_fraction": 0.01,
            "lr": 1e-2, "batch_size": 32, "projection_dim": 8}


class TestFiveWayPartitions:
    FR = {"calibration": 0.1, "probe_fit": 0.12, "probe_score": 0.12,
          "monitor": 0.12}

    def test_ordering_and_purge_between_every_pair(self):
        steps = list(range(1000, 2000))
        blocks, purged = five_way_split(steps, self.FR, 12)
        order = ["encoder_training", "calibration", "probe_fit",
                 "probe_score", "monitor"]
        for left, right in zip(order, order[1:]):
            gap = blocks[right][0] - blocks[left][-1]
            assert gap == 13, f"{left}->{right} gap {gap}"
            # no forward target (h<=12) crosses the boundary
            assert blocks[left][-1] - 1 + 12 < blocks[right][0] - 1
        rebuilt = sorted(sum((blocks[n] for n in order), []) + purged)
        assert rebuilt == steps
        assert len(purged) == 4 * 12

    def test_insufficient_data_refuses(self):
        with pytest.raises(PretrainContractError, match="no encoder"):
            five_way_split(list(range(50)), self.FR, 12)

    def test_probe_blocks_have_separate_digests_in_runner_manifest(self):
        source = (REPO / "tools/pretrain_branches.py").read_text()
        assert "five_way_probe" in source
        assert "PROBE block" in source

    def test_adapter_train_val_split_is_causal_with_purge(self):
        train, val = split_adapter_train_val(np.arange(100), 12)
        assert train.max() < val.min()
        assert val.min() - train.max() == 13
        with pytest.raises(ProbeRefusal):
            split_adapter_train_val(np.arange(10), 12)


class TestFrozenEncoderAndAdapterOnlyGradients:
    def test_cached_embeddings_carry_no_gradient_path(self):
        """The encoder is frozen STRUCTURALLY: probes consume cached
        no_grad embeddings, so adapter training cannot touch it."""
        encoder = torch.nn.Linear(6, 4)
        with torch.no_grad():
            cached = encoder(torch.randn(64, 6))
        adapter = torch.nn.Linear(4, 1)
        loss = adapter(cached).square().mean()
        loss.backward()
        assert all(p.grad is None for p in encoder.parameters())
        assert all(p.grad is not None for p in adapter.parameters())


class TestValidatedAdapterFitting:
    @staticmethod
    def _linear_task(noise=0.05, n=400):
        torch.manual_seed(7)
        x = torch.randn(n, 4)
        w = torch.randn(4, 1)
        y = x @ w + noise * torch.randn(n, 1)
        train, val, score = x[:200], x[200:300], x[300:]
        yt, yv, ys = y[:200], y[200:300], y[300:]

        def fit(a, g):
            idx = torch.randint(0, 200, (32,), generator=g)
            return torch.nn.functional.mse_loss(a(train[idx]), yt[idx])
        return (lambda: torch.nn.Linear(4, 1), fit,
                lambda a: torch.nn.functional.mse_loss(a(val), yv),
                lambda a: torch.nn.functional.mse_loss(a(score), ys))

    def test_happy_path_reports_median_and_dispersion(self):
        build, fit, val, score = self._linear_task()
        result = fit_adapter_validated(build, fit, val, score, PROTOCOL)
        assert len(result["probe_scores_by_seed"]) == 3
        assert result["dispersion"] >= 0
        assert all(c["best_val"] <= c["initial_val"]
                   for c in result["curves"])
        assert all(c["stopped_at"] >= PROTOCOL["min_steps"]
                   for c in result["curves"])

    def test_deterministic_replay(self):
        build, fit, val, score = self._linear_task()
        a = fit_adapter_validated(build, fit, val, score, PROTOCOL)
        b = fit_adapter_validated(build, fit, val, score, PROTOCOL)
        assert a["probe_scores_by_seed"] == b["probe_scores_by_seed"]

    def test_unfittable_task_refuses_not_underfits(self):
        """The 371 counterexample: 'final batch < first batch' called
        convergence. Now a task with NO learnable signal refuses."""
        torch.manual_seed(3)
        x = torch.randn(300, 4)
        y = torch.randn(300, 1) * 5  # pure noise, uncorrelated

        def fit(a, g):
            idx = torch.randint(0, 150, (32,), generator=g)
            return torch.nn.functional.mse_loss(a(x[:150][idx]),
                                                y[:150][idx])
        with pytest.raises(ProbeRefusal,
                           match="ADAPTER_FAILED_TO_FIT"):
            fit_adapter_validated(
                lambda: torch.nn.Linear(4, 1), fit,
                lambda a: torch.nn.functional.mse_loss(
                    a(x[150:220]), y[150:220]),
                lambda a: torch.nn.functional.mse_loss(
                    a(x[220:]), y[220:]), PROTOCOL)

    def test_seed_instability_refuses_never_best_seed(self):
        """A fit that trains fine but scores wildly per seed must
        refuse as MATERIAL_SEED_INSTABILITY — the best seed is never
        selected."""
        build, fit, val, _score = self._linear_task()
        unstable = iter([1.0, 5.0, 0.1])
        with pytest.raises(ProbeRefusal,
                           match="MATERIAL_SEED_INSTABILITY"):
            fit_adapter_validated(
                build, fit, val,
                lambda a: torch.tensor(next(unstable)), PROTOCOL)


class TestSkillFormula:
    def test_random_zero_solo_one(self):
        assert normalized_skill(1.0, 1.0, 0.5)[0] == 0.0
        assert normalized_skill(1.0, 0.5, 0.5)[0] == 1.0

    def test_ill_ordered_and_near_zero_refuse_ranking(self):
        assert normalized_skill(1.0, 0.6, 1.2)[0] is None
        assert normalized_skill(1.0, 0.6, 0.96)[0] is None

    def test_worse_than_random_is_negative(self):
        skill, _ = normalized_skill(1.0, 1.3, 0.5)
        assert skill < -0.05


class TestIsolation:
    def test_probe_score_mutation_cannot_alter_adapter_training(self):
        """Adapter training consumes only adapter-train/val closures;
        the score closure runs AFTER best-state restoration."""
        build, fit, val, score = \
            TestValidatedAdapterFitting._linear_task()
        base = fit_adapter_validated(build, fit, val, score, PROTOCOL)
        mutated = fit_adapter_validated(
            build, fit, val,
            lambda a: score(a) * 1000, PROTOCOL)
        for c_base, c_mut in zip(base["curves"], mutated["curves"]):
            assert c_base["val"] == c_mut["val"]  # training identical
            assert c_base["best_step"] == c_mut["best_step"]

    def test_monitor_and_reserved_data_not_in_probe_tools(self):
        """Route selection can only touch probe_fit/probe_score: the
        tools never index the monitor block, and the runner's fit
        slice already excludes 2022/outer/sealed structurally
        (fit_end < score_start, proven by the standing 341/353
        regressions)."""
        for tool in ("tools/final_probe_screen.py",
                     "tools/objective_routing_screen.py"):
            source = (REPO / tool).read_text()
            assert 'blocks["monitor"]' not in source
            assert 'blocks["probe_fit"]' in source
            assert 'blocks["probe_score"]' in source

    def test_ranking_is_cardinality_invariant(self):
        source = (REPO / "tools/final_probe_screen.py").read_text()
        assert "for task in LONG" in source  # every probe, always
        # C1 (374-376): selection authority moved into the pure
        # select_routes function; the 369 invariant — ranking never
        # keys on trained-objective cardinality — must hold THERE
        assert "select_routes(report" in source
        import inspect

        from agent_plugins.objective_routing import select_routes
        authority = inspect.getsource(select_routes)
        assert "trained_objectives" not in authority
        assert "median" in authority  # skill medians, not counts
