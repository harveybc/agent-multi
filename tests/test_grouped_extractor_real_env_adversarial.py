"""§1 of the grouped-extractor order (2026-08-23): adversarial layout
inspection against the REAL executing gym-fx observation — no
synthetic reconstruction. Proves the time and feature axes are not
transposed, branch boundaries carry the declared shapes, gradients
reach every branch, and feature-order drift refuses."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline_plugins.rl_pipeline_with_validation import (  # noqa: E402
    _load_env_plugin,
)

DATA = Path("/home/harveybc/Documents/GitHub/predictor/examples/data/"
            "project3/ethusdt_4h_tech_stat_full_model_ready.csv")
CONFIG = REPO / ("examples/config/"
                 "project3_ethusdt_4h_sac_grouped_features_v1.json")


@pytest.fixture(scope="module")
def grouped_config(tmp_path_factory):
    if not DATA.is_file():
        pytest.skip("real ETH csv not present")
    cfg = json.loads(CONFIG.read_text())
    sliced = tmp_path_factory.mktemp("eth") / "eth_700.csv"
    with DATA.open() as src, sliced.open("w") as dst:
        for i, line in enumerate(src):
            if i > 700:
                break
            dst.write(line)
    cfg["input_data_file"] = str(sliced)
    cfg["max_steps"] = 460
    return cfg


@pytest.fixture(scope="module")
def real_env(grouped_config):
    plug = _load_env_plugin("gym_fx_env", grouped_config)
    return plug.make_env(grouped_config)


def _arch(grouped_config):
    return grouped_config["feature_extractor_config"]


class TestRealObservationLayout:
    def test_features_block_is_window_by_columns(self, real_env,
                                                 grouped_config):
        obs, _ = real_env.reset(seed=7)
        arch = _arch(grouped_config)
        window = int(grouped_config["window_size"])
        assert obs["features"].shape == (
            window, len(arch["feature_columns"]))
        for key in arch["state_keys"]:
            assert key in obs, f"state key {key} absent from real obs"

    def test_time_axis_is_axis0_oldest_to_newest(self, real_env):
        """One NOP step must SHIFT the window by exactly one bar:
        new[:-1] == old[1:]. If the emission were newest-first the
        opposite alignment would hold; if features were on axis 0 no
        single-row shift alignment would exist at all."""
        obs1, _ = real_env.reset(seed=7)
        obs2, _r, _t, _tr, _i = real_env.step([0.0])
        f1 = np.asarray(obs1["features"], dtype=np.float64)
        f2 = np.asarray(obs2["features"], dtype=np.float64)
        forward = np.allclose(f2[:-1], f1[1:], atol=1e-9)
        backward = np.allclose(f2[1:], f1[:-1], atol=1e-9)
        assert forward, ("consecutive observations do not overlap as a "
                         "one-bar forward shift — time is not axis 0 "
                         "oldest-first")
        assert not backward or np.allclose(f1, f1[0]), (
            "window also matches a backward shift on non-constant "
            "data — ordering ambiguous")

    def test_branch_boundaries_and_gradients_on_real_obs(
            self, real_env, grouped_config):
        import torch
        from agent_plugins.grouped_features_extractor import (
            build_grouped_extractor_class,
        )
        arch = _arch(grouped_config)
        extractor = build_grouped_extractor_class()(
            real_env.observation_space, architecture=arch)
        # Rolling-zscore scaling emits ZEROS until its window fills:
        # a reset-time observation feeds branch 0 all-zero input and
        # zero weight-gradients are then mathematically correct, not
        # extractor evidence. Advance past the warm-up first, and
        # assert the probed inputs are non-degenerate so the gradient
        # claim means something.
        obs1, _ = real_env.reset(seed=7)
        for _step in range(300):  # scaling window is 256 bars
            obs1, _r, term, trunc, _i = real_env.step([0.0])
            if term or trunc:
                pytest.fail("episode ended during scaling warm-up")
        obs2, *_ = real_env.step([0.0])
        cols = arch["feature_columns"]
        for number, branch in enumerate(arch["branches"]):
            idx = [cols.index(n) for n in branch["features"]]
            block = np.asarray(obs1["features"])[:, idx]
            nonzero_fraction = float((block != 0).mean())
            assert nonzero_fraction > 0.5, (
                f"branch {number} input still all-zero after warm-up; "
                "evidence fixture invalid")
        batch = {
            key: torch.tensor(np.stack([
                np.asarray(obs1[key]), np.asarray(obs2[key])]),
                dtype=torch.float32)
            for key in obs1.keys()
        }
        captured = []
        for module in extractor.temporal_branches:
            module.register_forward_hook(
                lambda _m, inp, out, bag=captured: bag.append(
                    (tuple(inp[0].shape), tuple(out.shape))))
        out = extractor(batch)
        window = int(grouped_config["window_size"])
        declared = [len(b["features"]) for b in arch["branches"]]
        assert [shape for shape, _o in captured] == [
            (2, window, n) for n in declared], (
            "branch input shapes do not match the declared families")
        assert out.shape[0] == 2 and out.shape[1] == int(
            extractor.features_dim)
        # gradients must reach EVERY branch (and the state branch).
        # NOTE: out.sum() is a DEGENERATE probe here — the fusion ends
        # in LayerNorm and at init (gamma=1) the feature-sum of a
        # LayerNorm output is constant, so its gradient is identically
        # zero through the whole network. The overfit-fixture loss must
        # break that symmetry: MSE against a fixed random target.
        import torch as _torch
        _torch.manual_seed(3)
        target = _torch.randn_like(out)
        ((out - target) ** 2).mean().backward()
        for number, module in enumerate(extractor.temporal_branches):
            norms = [float(p.grad.abs().sum())
                     for p in module.parameters()
                     if p.grad is not None]
            assert norms and sum(norms) > 0, (
                f"branch {number} received no gradient")
        if extractor.state_branch is not None:
            norms = [float(p.grad.abs().sum())
                     for p in extractor.state_branch.parameters()
                     if p.grad is not None]
            assert norms and sum(norms) > 0, "state branch gradient-dead"

    def test_grouping_is_exhaustive_and_disjoint(self, grouped_config):
        arch = _arch(grouped_config)
        assigned = [f for b in arch["branches"] for f in b["features"]]
        assert len(assigned) == len(set(assigned)), "overlapping branches"
        assert sorted(assigned) == sorted(arch["feature_columns"])

    def test_feature_order_drift_refuses_before_sac(self, real_env,
                                                    grouped_config):
        """§1/§4: a permuted architecture column order must refuse at
        agent build time — the extractor cannot see names, so the
        binding lives in sac_agent and it must be order-sensitive."""
        from agent_plugins.sac_agent import Plugin
        cfg = json.loads(json.dumps(grouped_config))
        arch = cfg["feature_extractor_config"]
        cols = list(arch["feature_columns"])
        arch["feature_columns"] = cols[1:] + cols[:1]  # rotate
        cfg["feature_extractor_plugin"] = "grouped_features_extractor"
        with pytest.raises(ValueError, match="IDENTICAL"):
            Plugin().build(real_env, cfg)
