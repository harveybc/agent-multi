"""The four arms: paired, frozen, disclosed; R1/R2 blocked without donors; DRAFT refuses pilots."""
from __future__ import annotations

import json

import pytest

from ._fixtures import FEATURES, write_manifest, write_synthetic_csv
from ._mechanisms import assert_checkout_resolution, require

pytest.importorskip("torch")


@pytest.fixture
def binding(tmp_path):
    Binding = require("ARMS", "rl_temporal.lake_binding", "SelectedFeatureBinding", "manifest binding")
    csv = write_synthetic_csv(tmp_path / "fixture.csv")
    return Binding.from_manifest(write_manifest(tmp_path / "m.json", csv, status="DRAFT_NOT_FROZEN"))


def test_matrix_is_paired_and_disclosed(binding, tmp_path):
    assert_checkout_resolution("ARMS")
    arms = require("ARMS", "rl_temporal.arms", "build_matrix", "four-arm matrix builder")
    pairing = require("ARMS", "rl_temporal.arms", "check_pairing", "pairing check")
    cells = arms(binding, out_root=str(tmp_path / "out"), seeds=(1, 2))
    assert sorted({c["arm"] for c in cells}) == ["RL-D0", "RL-D1", "RL-S0", "RL-S1"] and len(cells) == 8
    report = pairing(cells)
    assert all(v["identical_outside_representation"] for v in report.values()), report
    by = {(c["arm"], c["train_seed"]): c for c in cells}
    s1, d1 = by[("RL-S1", 1)], by[("RL-D1", 1)]
    assert s1["representation"]["modular_config_sha256"] == d1["representation"]["modular_config_sha256"]
    assert s1["action_mapping"]["action_space_mode"] == "continuous"
    assert d1["action_mapping"]["action_space_mode"] == "discrete"
    assert by[("RL-S0", 1)]["representation"]["architecture"] == "flatten_mlp"
    assert s1["representation"]["status"] == "RUNNABLE_R0_RANDOM_INIT"
    assert s1["pilot_gate"]["real_data_fit_allowed"] is False
    assert s1["frozen_reward"]["reward_plugin"] == s1["reward_plugin"] == "pnl_reward"
    assert s1["accounting"]["extractor_parameters"] > 0 and by[("RL-S0", 1)]["accounting"]["extractor_parameters"] == 0
    assert s1["hard_limits"]["max_wall_s"] > 0 and s1["heartbeat_interval_s"] <= 60
    json.dumps(cells)  # JSON-able


def test_r1_arm_without_donor_is_blocked_not_relabelled(binding, tmp_path):
    build = require("ARMS", "rl_temporal.arms", "build_arm_config", "arm config builder")
    cfg = build("RL-D1", binding, seed=1, out_dir=str(tmp_path), regimes={"core": "R1"})
    assert cfg["representation"]["status"] == "BLOCKED_NO_COMPATIBLE_DONOR"
    assert cfg["representation"]["regimes"]["core"] == "R1"
    assert cfg["representation"]["regimes_summary"] == "MIXED"


def test_replay_storage_option_reaches_sb3_and_env_never_truncates(tmp_path):
    """Option (c): optimize_memory_usage True + handle_timeout_termination False reach the
    SB3 buffer through the plugins, and gym-fx never truncates (so no bootstrapping change)."""
    import inspect

    from ._fixtures import FEATURES, env_config, flat_env, modular_config, write_synthetic_csv
    build = require("ARMS", "rl_temporal.consumers", "build_model", "model builder")
    import rl_temporal.arms as arms
    kw = arms.ALGO["DQN"]["replay_buffer_kwargs"]
    csv = write_synthetic_csv(tmp_path / "f.csv", rows=200)
    env = flat_env(env_config(csv, action_space_mode="discrete"))
    try:
        model = build("DQN", env, modular_config=modular_config(), feature_order=FEATURES, device="cpu", seed=0,
                      learning_starts=4, buffer_size=64, batch_size=4, replay_buffer_kwargs=kw)
        assert model.replay_buffer.optimize_memory_usage is True
        assert model.replay_buffer.handle_timeout_termination is False
        import gym_fx.env as gym_env
        src = inspect.getsource(gym_env.GymFxEnv.step)
        assert "truncated = False" in src and "truncated = True" not in src
    finally:
        env.close()
