"""WP-D orchestration proofs: the easy->normal inner curriculum runs
easy first (train-only dynamics, budgeted early stop), saves an
immutable post_easy artifact, warm-continues the normal phase from those
weights with a fresh replay buffer, forces normal for every evaluation,
and selection flows only from the parent's normal-validation result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
    PipelinePlugin as CurriculumPipeline,
)
from pipeline_plugins import rl_pipeline_with_validation as parent_mod


class FakeModel:
    def __init__(self, weights="fresh"):
        self.weights = weights
        self.learned_timesteps = 0
        self.replay_buffer_transitions = 0

    def learn(self, total_timesteps, reset_num_timesteps=False):
        self.learned_timesteps += total_timesteps
        self.replay_buffer_transitions += total_timesteps
        self.weights = f"{self.weights}+{total_timesteps}"

    def predict(self, _obs, deterministic=True):
        return 0, None


class FakeAgent:
    def __init__(self):
        self.saved = {}
        self.loads = []

    def build(self, _env, _config):
        return FakeModel()

    def save(self, model, path):
        self.saved[path] = model.weights
        Path(path).write_text(model.weights)

    def load(self, path, _env):
        self.loads.append(str(path))
        # SB3 semantics: weights restored, replay buffer NOT restored.
        model = FakeModel(weights=Path(path).read_text())
        model.replay_buffer_transitions = 0
        return model


class FakeEnv:
    def __init__(self, config):
        self.config = dict(config)
        self.steps = 0

    def reset(self, seed=None):
        self.steps = 0
        return {"obs": 0}, {}

    def step(self, _action):
        self.steps += 1
        done = self.steps >= 3
        info = {"economic_equity": 9000.0 - self.steps,
                "recapitalization_debt": 100.0,
                "recapitalization_count": 1,
                "would_margin_call_count": 1,
                "termination_cause": "data_end" if done else None}
        return {"obs": self.steps}, 0.0, done, False, info

    def close(self):
        pass


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    pipeline = CurriculumPipeline({})
    made_envs = []

    def fake_split_csv(_config):
        return {"train": "train.csv", "train_tail": "tail.csv",
                "val": "val.csv", "test": "test.csv"}

    def fake_make_split_env(_name, config, csv_path, _agent):
        env = FakeEnv({**config, "_csv": csv_path})
        made_envs.append(env)
        plug = type("Plug", (), {"close": staticmethod(lambda: None)})()
        return plug, env

    parent_calls = []

    def fake_parent_run(self=None, *, config, env_plugin, agent_plugin,
                        mode="train"):
        parent_calls.append({"config": dict(config), "mode": mode})
        return {"best_model_path": str(tmp_path / "model.zip"),
                "selection": "normal_validation"}

    monkeypatch.setattr(pipeline, "_split_csv", fake_split_csv)
    monkeypatch.setattr(pipeline, "_make_split_env", fake_make_split_env)
    monkeypatch.setattr(parent_mod.PipelinePlugin, "run_pipeline",
                        fake_parent_run)
    return pipeline, made_envs, parent_calls, tmp_path


def _config(tmp_path):
    return {
        "env_plugin": "gym_fx_env", "env_mode": "training",
        "save_model": str(tmp_path / "candidate" / "model.zip"),
        "epoch_timesteps": 10, "easy_max_epochs": 3, "easy_patience": 1,
        "eval_seed": 7,
    }


def test_easy_phase_runs_first_then_normal_warm_start(harness):
    pipeline, made_envs, parent_calls, tmp_path = harness
    agent = FakeAgent()
    result = pipeline.run_pipeline(
        config=_config(tmp_path), env_plugin=None, agent_plugin=agent,
        mode="train")

    # Easy envs (train + probes) all carried easy train-only dynamics.
    easy_envs = [env for env in made_envs
                 if env.config.get("solvency_mode")
                 == "easy_chronological_continuation"]
    assert easy_envs, "easy phase must construct easy envs"
    assert all(env.config.get("env_mode") == "training"
               for env in easy_envs)

    # The normal phase went through the PARENT pipeline with normal
    # dynamics and a warm start bound to the post_easy artifact hash.
    assert len(parent_calls) == 1
    normal_config = parent_calls[0]["config"]
    assert normal_config["solvency_mode"] == "normal_realistic"
    assert normal_config["warm_start_model"].endswith(".post_easy.zip")
    assert len(normal_config["warm_start_model_sha256"]) == 64
    assert Path(normal_config["warm_start_model"]).exists()

    curriculum = result["curriculum"]
    assert curriculum["phases"] == [
        "easy_chronological_continuation", "normal_realistic"]
    assert curriculum["selection_basis"] == "normal_validation_only"
    meta = json.loads(Path(
        normal_config["warm_start_model"] + ".meta.json").read_text())
    assert meta["history"][0]["would_margin_call_count"] == 1
    assert meta["easy_budget_epochs"] == 3


def test_easy_early_stop_respects_budget(harness):
    pipeline, made_envs, parent_calls, tmp_path = harness
    agent = FakeAgent()
    config = _config(tmp_path)
    config["easy_max_epochs"] = 5
    config["easy_patience"] = 1
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=agent, mode="train")
    meta = json.loads(Path(
        parent_calls[0]["config"]["warm_start_model"]
        + ".meta.json").read_text())
    # FakeEnv economic equity is constant across probes: the first probe
    # sets best, the second exhausts patience=1 — early stop before the
    # declared budget of 5.
    assert meta["easy_epochs_run"] == 2
    assert meta["easy_epochs_run"] < meta["easy_budget_epochs"]


def test_non_train_modes_pass_through_forced_normal(harness):
    pipeline, made_envs, parent_calls, tmp_path = harness
    pipeline.run_pipeline(config=_config(tmp_path), env_plugin=None,
                          agent_plugin=FakeAgent(), mode="inference")
    assert len(parent_calls) == 1
    assert parent_calls[0]["config"]["solvency_mode"] == "normal_realistic"
    assert parent_calls[0]["mode"] == "inference"
    assert not made_envs                       # no easy phase


def test_sb3_reload_semantics_reset_replay_buffer(tmp_path):
    """The dynamics-boundary reload carries weights but starts with an
    empty replay buffer (SB3 load semantics, mirrored by the fake)."""
    agent = FakeAgent()
    model = FakeModel()
    model.learn(50)
    assert model.replay_buffer_transitions == 50
    agent.save(model, str(tmp_path / "post_easy.zip"))
    reloaded = agent.load(str(tmp_path / "post_easy.zip"), None)
    assert reloaded.weights == model.weights           # warm continuation
    assert reloaded.replay_buffer_transitions == 0     # fresh buffer


def test_parent_eval_split_forces_normal(monkeypatch):
    """Defense in depth: the validation pipeline's evaluation path pins
    normal_realistic no matter what the training config says."""
    pipeline = parent_mod.PipelinePlugin({})
    captured = {}

    def fake_make_split_env(_name, config, csv_path, _agent):
        captured["solvency_mode"] = config.get("solvency_mode")
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(pipeline, "_make_split_env", fake_make_split_env)
    with pytest.raises(RuntimeError, match="stop after capture"):
        pipeline._eval_on_split(
            "gym_fx_env",
            {"solvency_mode": "easy_chronological_continuation"},
            "x.csv", None, None, 7, "validation")
    assert captured["solvency_mode"] == "normal_realistic"
