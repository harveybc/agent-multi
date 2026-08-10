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
from pipeline_plugins.rl_pipeline_with_validation import (
    _checkpoint_is_eligible,
)


class _FakePolicy:
    """Real-tensor policy so canonical tensor digests are honest."""

    def __init__(self):
        import torch

        self._tensor = torch.zeros(4)

    def state_dict(self):
        return {"actor.weight": self._tensor.clone()}

    def perturb(self):
        self._tensor += 1.0


class FakeModel:
    def __init__(self, weights="fresh"):
        self.weights = weights
        self.learned_timesteps = 0
        self.replay_buffer_transitions = 0
        self.policy = _FakePolicy()
        self._n_updates = 0

    def learn(self, total_timesteps, reset_num_timesteps=False):
        self.learned_timesteps += total_timesteps
        self.replay_buffer_transitions += total_timesteps
        self._n_updates += total_timesteps
        self.policy.perturb()
        self.weights = f"{self.weights}+{total_timesteps}"

    def predict(self, _obs, deterministic=True):
        return 0, None


class FakeAgent:
    def __init__(self):
        self.saved = {}
        self.loads = []
        self.training_loads = []

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

    def load_for_training(self, path, env, config):
        self.training_loads.append({"path": str(path), "config": dict(config)})
        return self.load(path, env)


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
        trades = int(self.config.get("_fake_trades", 1))
        info = {"economic_equity": 9000.0 - self.steps,
                "recapitalization_debt": 100.0,
                "recapitalization_count": 1,
                "would_margin_call_count": 1,
                "trades": trades,
                "action_diagnostics": {
                    "non_hold_actions": 3 if trades else 0,
                },
                "execution_diagnostics": {
                    "entry_actions_seen": 1 if trades else 0,
                    "entry_orders_submitted": 1 if trades else 0,
                    "protected_entry_rejections": 0,
                },
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

    def fake_eval_on_split(
        _name, config, _csv_path, _agent, _model, _seed, _split_name
    ):
        trades = int(config.get(
            "_fake_normal_trades", config.get("_fake_trades", 1)
        ))
        return {"trades_total": trades, "total_return": 0.0}

    monkeypatch.setattr(pipeline, "_eval_on_split", fake_eval_on_split)
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
    assert all(env.config["continuous_action_threshold"] == 0.0
               for env in easy_envs)
    assert all(env.config["commission"] == pytest.approx(0.00005)
               for env in easy_envs)
    assert all(env.config["full_spread_rate"] == pytest.approx(0.0001)
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
    assert meta["history"][0]["easy_activity_eligible"] is True
    assert meta["activity_contract"]["minimum_trades"] == 1
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
    # Fake summaries carry no common-scale weekly utility, so no paired
    # checkpoint is ever ELIGIBLE: patience cannot fire (ineligible
    # checkpoints never touch improvement patience) and the phase runs
    # its full declared budget, then hands off the TERMINAL trained
    # epoch — never the anchor (invariant 3).
    assert meta["easy_epochs_run"] == 5      # no warm start: 5 trained
    assert meta["selection_basis"] == "terminal_trained_epoch_fallback"
    assert Path(parent_calls[0]["config"]["warm_start_model"]).read_text() == (
        "fresh+10+10+10+10+10"
    )


def test_inactive_phase1_still_hands_off_terminal_never_anchor(harness):
    """Invariant 3 (repair spec §2): a failed/inactive phase-1 result is
    still handed off and MEASURED. The old contract rejected the run and
    saved nothing; that behavior hid the treatment."""
    pipeline, _made_envs, parent_calls, tmp_path = harness
    config = _config(tmp_path)
    config["_fake_trades"] = 0
    pipeline.run_pipeline(
        config=config, env_plugin=None, agent_plugin=FakeAgent(),
        mode="train",
    )
    assert parent_calls, "normal phase must still run"
    post_easy = Path(parent_calls[0]["config"]["warm_start_model"])
    assert post_easy.exists()
    meta = json.loads(Path(str(post_easy) + ".meta.json").read_text())
    assert meta["selection_basis"] == "terminal_trained_epoch_fallback"
    assert meta["epoch0_structurally_ineligible"] is True


def test_normal_probe_collapse_is_telemetry_not_a_gate(harness):
    """AUD-F1-20260808-159: the normal probe never decides whether the
    easy treatment exists. A collapsed normal probe is recorded as
    telemetry and the trained epoch is handed off anyway."""
    pipeline, _made_envs, parent_calls, tmp_path = harness
    config = _config(tmp_path)
    config["_fake_trades"] = 3
    config["_fake_normal_trades"] = 0
    pipeline.run_pipeline(
        config=config,
        env_plugin=None,
        agent_plugin=FakeAgent(),
        mode="train",
    )
    assert parent_calls
    meta = json.loads(Path(
        parent_calls[0]["config"]["warm_start_model"]
        + ".meta.json").read_text())
    assert meta["normal_probe_is_telemetry_only"] is True
    trained = [row for row in meta["history"] if row["epoch"] > 0]
    assert all(row["normal_handoff_eligible_telemetry"] is False
               for row in trained)


def test_epoch_zero_anchor_baseline_is_structurally_ineligible(harness):
    pipeline, _made_envs, parent_calls, tmp_path = harness
    anchor = tmp_path / "anchor.zip"
    anchor.write_text("active-anchor")
    config = _config(tmp_path)
    config.update({
        "warm_start_model": str(anchor),
        "easy_max_epochs": 2,
        "easy_patience": 1,
    })
    agent = FakeAgent()
    pipeline.run_pipeline(
        config=config,
        env_plugin=None,
        agent_plugin=agent,
        mode="train",
    )

    post_easy = Path(parent_calls[0]["config"]["warm_start_model"])
    meta = json.loads(Path(str(post_easy) + ".meta.json").read_text())
    assert agent.loads[0] == str(anchor)
    assert agent.training_loads[0]["path"] == str(anchor)
    # AUD-F1-20260808-159 regression: the ACTIVE anchor baseline may be
    # logged as telemetry, but it can never be the handoff — the
    # artifact must be a TRAINED epoch, tensor-distinct from the anchor.
    assert post_easy.read_text() != "active-anchor"
    assert meta["best_easy_epoch"] > 0
    assert meta["history"][0]["checkpoint_source"] == "warm_start_baseline"
    assert meta["history"][0]["handoff_eligible"] is False
    assert meta["anchor_policy_tensor_sha256"] != \
        meta["phase1_terminal_policy_tensor_sha256"]


def test_checkpoint_eligibility_requires_timesteps_and_trading_activity():
    assert _checkpoint_is_eligible(
        num_timesteps=101,
        minimum_timesteps=101,
        trade_gate_passed=True,
    )
    assert not _checkpoint_is_eligible(
        num_timesteps=100,
        minimum_timesteps=101,
        trade_gate_passed=True,
    )
    assert not _checkpoint_is_eligible(
        num_timesteps=101,
        minimum_timesteps=101,
        trade_gate_passed=False,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("easy_continuous_action_threshold", -0.1),
        ("easy_commission_fraction_per_side", 0.0),
        ("easy_full_spread_rate", 0.0),
        ("easy_slippage_bps_per_side", 0.0),
        ("easy_min_trades", 0),
    ],
)
def test_easy_contract_rejects_invalid_activity_or_costs(
    harness, key, value
):
    pipeline, _made_envs, parent_calls, tmp_path = harness
    config = _config(tmp_path)
    config[key] = value
    with pytest.raises(ValueError):
        pipeline.run_pipeline(
            config=config, env_plugin=None, agent_plugin=FakeAgent(),
            mode="train",
        )
    assert not parent_calls


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


def test_phase1_metadata_is_truthful_for_normal_mode(harness):
    # Finding 195: normal records must say normal — never easy — and
    # the handoff probe facts must describe telemetry, not a gate.
    pipeline, made_envs, parent_calls, tmp_path = harness
    agent = FakeAgent()
    config = _config(tmp_path)
    config["phase1_mode"] = "normal_realistic"
    config["easy_min_trades"] = 1
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=agent, mode="train")
    meta = json.loads(Path(
        parent_calls[0]["config"]["warm_start_model"]
        + ".meta.json").read_text())
    assert meta["solvency_mode"] == "normal_realistic"
    assert meta["phase1_mode"] == "normal_realistic"
    contract = meta["activity_contract"]
    assert contract["normal_handoff_probe_is_telemetry_only"] is True
    assert contract["normal_handoff_activity_gates_selection"] is False
    assert "requires_normal_handoff_train_tail_activity" not in contract
    assert "requires_normal_handoff_validation_activity" not in contract


def test_phase1_metadata_is_truthful_for_easy_mode(harness):
    pipeline, made_envs, parent_calls, tmp_path = harness
    agent = FakeAgent()
    pipeline.run_pipeline(config=_config(tmp_path), env_plugin=None,
                          agent_plugin=agent, mode="train")
    meta = json.loads(Path(
        parent_calls[0]["config"]["warm_start_model"]
        + ".meta.json").read_text())
    assert meta["solvency_mode"] == "easy_chronological_continuation"
    assert meta["activity_contract"][
        "normal_handoff_activity_gates_selection"] is False
