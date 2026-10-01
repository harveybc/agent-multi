"""RL02: modular time remains present through branches, fusion and core;
task-head reduction is explicit; the baseline receives the same source
information."""
from __future__ import annotations

import numpy as np
import pytest

from ._fixtures import FEATURES, WINDOW, env_config, flat_env, modular_config, write_synthetic_csv
from ._mechanisms import assert_checkout_resolution, require

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("backtrader")


@pytest.fixture
def env(tmp_path):
    csv = write_synthetic_csv(tmp_path / "fixture.csv", rows=200)
    e = flat_env(env_config(csv))
    yield e
    e.close()


def _layout(env):
    Layout = require("RL02", "rl_temporal.observation_layout", "ObservationLayout",
                     "flat observation layout resolved from the Dict space")
    return Layout.from_space(env.unwrapped.observation_space, feature_order=FEATURES)


def test_layout_locates_the_feature_window_inside_the_flat_vector(env):
    assert_checkout_resolution("RL02")
    layout = _layout(env)
    assert layout.window == WINDOW and layout.n_features == len(FEATURES)
    obs, _ = env.reset(seed=0)
    dict_obs = env.unwrapped._make_observation()
    feats, extras = layout.split(torch.as_tensor(obs[None]))
    assert feats.shape == (1, WINDOW, len(FEATURES))
    np.testing.assert_allclose(feats[0].numpy(), dict_obs["features"], atol=1e-6)
    assert extras.shape[1] == obs.shape[0] - WINDOW * len(FEATURES)
    assert layout.feature_order == tuple(FEATURES)


def test_modular_extractor_keeps_time_through_branches_fusion_core(env):
    Extractor = require("RL02", "rl_temporal.modular_torch", "ModularTemporalExtractor",
                        "torch modular temporal features extractor")
    layout = _layout(env)
    ext = Extractor(env.observation_space, layout=layout, modular_config=modular_config())
    obs, _ = env.reset(seed=0)
    x = torch.as_tensor(obs[None])
    stages = ext.stage_outputs(x)
    for name in [f"branch_{i}" for i in range(len(FEATURES))]:
        assert stages["branches"][name].shape == (1, WINDOW, 16), name
    assert stages["fused"].shape == (1, WINDOW, 16 * len(FEATURES))
    assert stages["latent"].shape == (1, 6, 8)
    assert stages["task_head_reduction"] == {"from": [6, 8], "to": 48, "op": "flatten"}
    out = ext(x)
    assert out.shape == (1, ext.features_dim)
    assert ext.features_dim == 48 + layout.extras_dim


def test_branch_output_is_causal_in_time(env):
    Extractor = require("RL02", "rl_temporal.modular_torch", "ModularTemporalExtractor",
                        "causal branch convolution")
    layout = _layout(env)
    ext = Extractor(env.observation_space, layout=layout, modular_config=modular_config())
    obs, _ = env.reset(seed=0)
    x = torch.as_tensor(obs[None]).clone()
    base = ext.stage_outputs(x)["fused"].detach()
    y = x.clone()
    y[0, layout.feature_index(WINDOW - 1, 0)] += 5.0  # last step of feature 0
    moved = ext.stage_outputs(y)["fused"].detach()
    assert torch.allclose(base[0, :-1], moved[0, :-1]), "earlier steps moved: not causal"
    assert not torch.allclose(base[0, -1], moved[0, -1])


def test_baseline_and_modular_consume_the_same_source_information(env):
    Native = require("RL02", "rl_temporal.modular_torch", "NativeFlatExtractor",
                     "declared native baseline extractor")
    Extractor = require("RL02", "rl_temporal.modular_torch", "ModularTemporalExtractor",
                        "torch modular temporal features extractor")
    card = require("RL02", "rl_temporal.modular_torch", "native_baseline_card",
                   "native baseline identification card")
    layout = _layout(env)
    native = Native(env.observation_space)
    modular = Extractor(env.observation_space, layout=layout, modular_config=modular_config())
    obs, _ = env.reset(seed=0)
    x = torch.as_tensor(obs[None])
    assert native(x).shape == (1, obs.shape[0])
    feats, _ = layout.split(x)
    assert torch.equal(modular.source_window(x), feats)
    assert torch.equal(native(x)[0, layout.features_slice], feats.reshape(-1))
    doc = card(layout, net_arch=[64, 64], action_dim=3)
    assert doc["architecture"] == "flatten_mlp"
    assert doc["time_handling"] == "flattened_window_no_temporal_structure"
    assert doc["source_information"]["layout_digest"] == layout.digest
    assert doc["parameter_count"] == (obs.shape[0] * 64 + 64) + (64 * 64 + 64) + (64 * 3 + 3)
