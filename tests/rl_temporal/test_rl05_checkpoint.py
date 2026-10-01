"""RL05: save/reload restores policy, representation, normalization and
action mapping; deterministic evaluation in a fresh process reproduces
decisions within declared device tolerances."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from ._fixtures import FEATURES, env_config, flat_env, modular_config, write_synthetic_csv
from ._mechanisms import ROOT, GYM_FX_ROOT, assert_checkout_resolution, require

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("backtrader")


@pytest.mark.parametrize("algorithm", ["SAC", "DQN"])
def test_bundle_round_trips_in_a_fresh_process(tmp_path, algorithm):
    assert_checkout_resolution("RL05")
    build = require("RL05", "rl_temporal.consumers", "build_model", "model builder")
    save = require("RL05", "rl_temporal.checkpoint", "save_policy_bundle",
                   "policy bundle writer (policy + representation + normalization + action mapping)")
    load = require("RL05", "rl_temporal.checkpoint", "load_policy_bundle", "policy bundle loader")
    decide = require("RL05", "rl_temporal.checkpoint", "decisions", "deterministic decision replay")
    csv = write_synthetic_csv(tmp_path / "fixture.csv", rows=200)
    mode = "continuous" if algorithm == "SAC" else "discrete"
    cfg = env_config(csv, action_space_mode=mode)
    env = flat_env(cfg)
    try:
        model = build(algorithm, env, modular_config=modular_config(), feature_order=FEATURES,
                      share_features_extractor=True, device="cpu", seed=0,
                      learning_starts=8, buffer_size=64, batch_size=8)
        model.learn(total_timesteps=24)
        obs = np.stack([env.reset(seed=s)[0] for s in range(4)]).astype(np.float32)
        expected = decide(model, obs)
        bundle_dir = tmp_path / "bundle"
        doc = save(model, bundle_dir, env_config=cfg, arm="fixture", tolerance_abs=1e-6)
    finally:
        env.close()
    assert doc["action_mapping"]["action_space_mode"] == mode
    assert doc["normalization"]["feature_scaling"] == "rolling_zscore"
    assert doc["normalization"]["feature_scaling_window"] == 64
    assert doc["representation"]["layout_digest"]
    assert doc["representation"]["modular_config_sha256"]
    assert doc["versions"]["stable_baselines3"] and doc["versions"]["torch"]
    assert doc["tolerance"] == {"device": "cpu", "abs": 1e-6}

    # same process
    env2 = flat_env(cfg)
    try:
        reloaded = load(bundle_dir, env2)
        np.testing.assert_allclose(decide(reloaded, obs), expected, atol=1e-6)
    finally:
        env2.close()

    # fresh process
    obs_path = tmp_path / "obs.npy"
    np.save(obs_path, obs)
    out_path = tmp_path / "decisions.json"
    code = (
        "import sys, json, numpy as np\n"
        f"sys.path[:0] = [{str(ROOT)!r}, {str(GYM_FX_ROOT)!r}]\n"
        "from rl_temporal.checkpoint import load_policy_bundle, decisions\n"
        "from tests.rl_temporal._fixtures import flat_env\n"
        f"cfg = json.load(open({str(bundle_dir / 'env_config.json')!r}))\n"
        "env = flat_env(cfg)\n"
        f"model = load_policy_bundle({str(bundle_dir)!r}, env)\n"
        f"out = decisions(model, np.load({str(obs_path)!r}))\n"
        f"json.dump(np.asarray(out).tolist(), open({str(out_path)!r}, 'w'))\n"
        "env.close()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fresh = np.asarray(json.loads(out_path.read_text()), dtype=np.float64)
    np.testing.assert_allclose(fresh, np.asarray(expected, dtype=np.float64), atol=1e-6)


def test_bundle_refuses_a_mismatched_action_mapping(tmp_path):
    build = require("RL05", "rl_temporal.consumers", "build_model", "model builder")
    save = require("RL05", "rl_temporal.checkpoint", "save_policy_bundle", "bundle writer")
    load = require("RL05", "rl_temporal.checkpoint", "load_policy_bundle", "bundle loader")
    csv = write_synthetic_csv(tmp_path / "fixture.csv", rows=200)
    cfg = env_config(csv, action_space_mode="continuous", continuous_action_threshold=0.33)
    env = flat_env(cfg)
    try:
        model = build("SAC", env, modular_config=modular_config(), feature_order=FEATURES,
                      device="cpu", seed=0, learning_starts=8, buffer_size=64, batch_size=8)
        save(model, tmp_path / "bundle", env_config=cfg, arm="fixture")
    finally:
        env.close()
    other = flat_env(env_config(csv, action_space_mode="continuous", continuous_action_threshold=0.1))
    try:
        with pytest.raises(ValueError, match="action_mapping"):
            load(tmp_path / "bundle", other)
    finally:
        other.close()
