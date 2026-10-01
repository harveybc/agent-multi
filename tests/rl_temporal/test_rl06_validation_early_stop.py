"""RL06: chronological validation episodes drive early stopping, selected
policy restore and bounded evaluation cadence. Training episode reward is
not held-out performance."""
from __future__ import annotations

import time

import numpy as np
import pytest

from ._fixtures import FEATURES, env_config, flat_env, modular_config, write_synthetic_csv
from ._mechanisms import assert_checkout_resolution, require

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("backtrader")


def test_scripted_validation_scores_stop_and_restore_best(tmp_path):
    assert_checkout_resolution("RL06")
    build = require("RL06", "rl_temporal.consumers", "build_model", "model builder")
    Stopper = require("RL06", "rl_temporal.monitoring", "ValidationEpisodeEarlyStopping",
                      "validation-episode early stopping with best-policy restore")
    identity = require("RL06", "rl_temporal.donor_contract", "encoder_identity", "encoder identity")
    csv = write_synthetic_csv(tmp_path / "fixture.csv", rows=200)
    env = flat_env(env_config(csv, action_space_mode="discrete"))
    scores = iter([1.0, 2.0, 1.5, 1.0, 0.5, 0.1])
    seen = []

    def evaluate(model):
        s = next(scores)
        seen.append(identity(model)["q_net"]["core"])
        return {"selection_value": s, "validation_episodes": 1, "net_return": s}

    try:
        model = build("DQN", env, modular_config=modular_config(), feature_order=FEATURES,
                      device="cpu", seed=0, learning_starts=4, buffer_size=64, batch_size=4,
                      train_freq=1)
        stopper = Stopper(evaluate_fn=evaluate, eval_every_steps=8, patience=2,
                          best_path=tmp_path / "best.zip", min_delta=0.0)
        model.learn(total_timesteps=200, callback=stopper)
        assert stopper.stopped_early is True
        assert stopper.evaluations == 4, stopper.history
        assert stopper.best_index == 1 and stopper.best_value == 2.0
        assert [h["selection_value"] for h in stopper.history] == [1.0, 2.0, 1.5, 1.0]
        assert all(h["source"] == "validation_episode" for h in stopper.history)
        stopper.restore_best(model)
        assert identity(model)["q_net"]["core"] == seen[1], "best policy not restored"
        assert stopper.record()["train_episode_reward_is_selection"] is False
    finally:
        env.close()


def test_validation_episodes_are_chronologically_after_training_rows(tmp_path):
    split = require("RL06", "rl_temporal.monitoring", "chronological_validation_factory",
                    "validation env factory bound to rows after the training rows")
    csv = write_synthetic_csv(tmp_path / "fixture.csv", rows=300)
    cfg = env_config(csv, action_space_mode="discrete")
    factory = split(cfg, train_rows=[0, 200], validation_rows=[200, 300])
    venv = factory()
    try:
        frame = venv.unwrapped.dataframe
        assert len(frame) == 100 + 24 + 64, "validation keeps only its scaler/window context before it"
        assert str(frame.index[-1]) > str(frame.index[0])
        assert factory.description["validation_rows"] == [200, 300]
        assert factory.description["context_rows"] == 24 + 64
        assert factory.description["first_decision_row"] == 200
    finally:
        venv.close()


def test_heartbeat_and_hard_limits(tmp_path):
    build = require("RL06", "rl_temporal.consumers", "build_model", "model builder")
    Heartbeat = require("RL06", "rl_temporal.monitoring", "HeartbeatCallback",
                        "time-based heartbeat file (<=60 s)")
    Limits = require("RL06", "rl_temporal.monitoring", "HardLimitCallback",
                     "hard wall/timestep limits that stop learn()")
    csv = write_synthetic_csv(tmp_path / "fixture.csv", rows=200)
    env = flat_env(env_config(csv, action_space_mode="discrete"))
    try:
        model = build("DQN", env, modular_config=modular_config(), feature_order=FEATURES,
                      device="cpu", seed=0, learning_starts=4, buffer_size=64, batch_size=4)
        with pytest.raises(ValueError, match="60"):
            Heartbeat(tmp_path / "hb.json", interval_s=61)
        hb = Heartbeat(tmp_path / "hb.json", interval_s=0.0)
        lim = Limits(max_timesteps=40, max_wall_s=600)
        t0 = time.monotonic()
        model.learn(total_timesteps=1000, callback=[hb, lim])
        assert model.num_timesteps <= 40 + 1
        import json
        doc = json.loads((tmp_path / "hb.json").read_text())
        assert doc["num_timesteps"] >= 1 and doc["interval_s"] == 0.0 and "written_at" in doc
        assert lim.stop_reason == "max_timesteps"
        assert time.monotonic() - t0 < 600
    finally:
        env.close()
