"""WP-D orchestration proofs: the easy->normal inner curriculum runs
easy first (train-only dynamics, budgeted early stop), saves an
immutable post_easy artifact, warm-continues the normal phase from those
weights with a fresh replay buffer, forces normal for every evaluation,
and selection flows only from the parent's normal-validation result."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
    HANDOFF_VIABILITY_VALUES,
    PipelinePlugin as CurriculumPipeline,
    _assert_handoff_evidence_invariants,
    _assert_selected_handoff_invariants,
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


def test_phase1_epoch_facts_exclude_baseline_warm_start(harness):
    # Finding 200: with a warm start the history carries an epoch-0
    # baseline evaluation; realized epochs must count ONLY epoch > 0.
    pipeline, made_envs, parent_calls, tmp_path = harness
    agent = FakeAgent()
    config = _config(tmp_path)
    config["easy_max_epochs"] = 1
    anchor = tmp_path / "anchor.zip"
    anchor.write_bytes(b"anchor")
    config["warm_start_model"] = str(anchor)
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=agent, mode="train")
    meta = json.loads(Path(
        parent_calls[0]["config"]["warm_start_model"]
        + ".meta.json").read_text())
    trained = [h for h in meta["history"] if h["epoch"] > 0]
    baseline = [h for h in meta["history"] if h["epoch"] == 0]
    assert meta["phase1_epochs_run"] == len(trained)
    assert meta["phase1_baseline_evaluations"] == len(baseline)
    # Truthful alias: identical to the trained count, never inflated.
    assert meta["easy_epochs_run"] == meta["phase1_epochs_run"]


def test_phase1_epoch_facts_multi_epoch_no_warm_start(harness):
    pipeline, made_envs, parent_calls, tmp_path = harness
    agent = FakeAgent()
    config = _config(tmp_path)
    config["easy_max_epochs"] = 4
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=agent, mode="train")
    meta = json.loads(Path(
        parent_calls[0]["config"]["warm_start_model"]
        + ".meta.json").read_text())
    assert meta["phase1_epochs_run"] == sum(
        1 for h in meta["history"] if h["epoch"] > 0)
    assert meta["phase1_epochs_run"] + \
        meta["phase1_baseline_evaluations"] == len(meta["history"])


# ---------------------------------------------------------------------------
# Handoff-viability evidence (AUD-F1-20260811-221 / order WP3 §7)
# ---------------------------------------------------------------------------

class _PatternModel(FakeModel):
    """Deterministic policy whose raw action is a fixed function of the
    fake env's observation counter — the same action vector on every
    rollout, so the evidence hashes are reproducible."""

    def __init__(self, pattern, weights="fresh"):
        super().__init__(weights=weights)
        self.pattern = [float(v) for v in pattern]

    def predict(self, obs, deterministic=True):
        idx = int(obs.get("obs", 0)) if isinstance(obs, dict) else 0
        value = np.float32(self.pattern[idx % len(self.pattern)])
        return np.asarray([value], dtype=np.float32), None


class _PatternAgent(FakeAgent):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = list(pattern)

    def build(self, _env, _config):
        return _PatternModel(self.pattern)

    def load(self, path, _env):
        self.loads.append(str(path))
        model = _PatternModel(self.pattern,
                              weights=Path(path).read_text())
        model.replay_buffer_transitions = 0
        return model


def _meta(parent_calls):
    return json.loads(Path(
        parent_calls[0]["config"]["warm_start_model"] + ".meta.json"
    ).read_text())


def _pattern_sha(pattern):
    return hashlib.sha256(
        np.asarray(pattern, dtype=np.float32).tobytes()).hexdigest()


def test_every_checkpoint_carries_typed_viability_evidence(harness):
    pipeline, _made_envs, parent_calls, tmp_path = harness
    pipeline.run_pipeline(config=_config(tmp_path), env_plugin=None,
                          agent_plugin=FakeAgent(), mode="train")
    meta = _meta(parent_calls)
    assert meta["history"], "phase 1 must record checkpoints"
    for row in meta["history"]:
        ev = row["handoff_viability_evidence"]
        assert ev["schema"] == \
            "agent_multi.solvency_curriculum.handoff_viability.v1"
        assert ev["handoff_viability"] in HANDOFF_VIABILITY_VALUES
        assert ev["deterministic"] is True
        assert ev["rollout_solvency_mode"] == "normal_realistic"
        # easy phase-1 threshold comes from the config contract; the
        # candidate config omits a normal deadband, so the evidence
        # declares the gym-fx default rather than guessing.
        assert ev["phase1_threshold"] == 0.0
        assert ev["phase1_threshold_source"] == "config"
        assert ev["phase2_threshold"] == pytest.approx(0.33)
        assert ev["phase2_threshold_source"] == "gym_fx_default"
        for split in ("train_monitor", "inner_validation"):
            block = ev["splits"][split]
            assert block["available"] is True
            assert block["observation_count"] == 3
            assert len(block["action_vector_sha256"]) == 64
            constant = block["constant_policy_classification"]
            assert constant["dtype_eps"] > 0.0
            assert constant["tolerance_derivation"] == \
                "finfo(dtype).eps * max(1.0, observed_max_abs)"
        # the FakeModel emits a constant zero action: the dtype-derived
        # classification must call it CONSTANT_POLICY, never VIABLE.
        assert ev["handoff_viability"] == "CONSTANT_POLICY"
        assert ev["viable_as_trained_treatment"] is False
        assert ev["authority"].startswith("evidence only")


def test_constant_easy_active_policy_is_constant_policy(harness):
    """Finding 221 regression: a policy that is 100% non-hold under the
    easy zero threshold but emits ONE constant raw action passes
    easy_activity_eligible yet must be classified CONSTANT_POLICY."""
    pipeline, _made_envs, parent_calls, tmp_path = harness
    config = _config(tmp_path)
    config["continuous_action_threshold"] = 0.1
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=_PatternAgent([0.010861]),
                          mode="train")
    meta = _meta(parent_calls)
    for row in meta["history"]:
        assert row["easy_activity_eligible"] is True  # the broken label
        ev = row["handoff_viability_evidence"]
        assert ev["phase2_threshold"] == pytest.approx(0.1)
        assert ev["phase2_threshold_source"] == "config"
        assert ev["handoff_viability"] == "CONSTANT_POLICY"
        constant = ev["constant_policy_classification"]
        assert constant["exact_constant"] is True
        assert constant["unique_action_count_exact"] == 1
        assert ev["any_action_crosses_phase2_threshold"] is False
        for split in ("train_monitor", "inner_validation"):
            block = ev["splits"][split]
            assert block["fraction_non_hold_phase1_threshold"] == 1.0
            assert block["fraction_abs_ge_phase2_threshold"] == 0.0
        assert ev["viable_as_trained_treatment"] is False


def test_near_constant_within_dtype_tolerance_is_constant_policy(
        harness):
    """A one-ULP wobble at float32 precision is still a constant policy
    — even when the constant value crosses the phase-2 threshold. The
    tolerance is derived from dtype eps at the observed scale, never
    from an invented floor."""
    pipeline, _made_envs, parent_calls, tmp_path = harness
    base = np.float32(0.25)
    bumped = float(np.nextafter(base, np.float32(1.0)))
    config = _config(tmp_path)
    config["continuous_action_threshold"] = 0.1
    pipeline.run_pipeline(
        config=config, env_plugin=None,
        agent_plugin=_PatternAgent([float(base), bumped, float(base)]),
        mode="train")
    meta = _meta(parent_calls)
    ev = meta["history"][0]["handoff_viability_evidence"]
    constant = ev["constant_policy_classification"]
    assert constant["exact_constant"] is False
    assert constant["unique_action_count_exact"] == 2
    assert constant["near_constant"] is True
    assert 0.0 < constant["peak_to_peak"] <= \
        constant["near_constant_tolerance"]
    assert ev["any_action_crosses_phase2_threshold"] is True
    assert ev["handoff_viability"] == "CONSTANT_POLICY"


def test_varying_below_deadband_is_below_normal_threshold(harness):
    """The observed D2-D4 shape: raw actions vary by ~6e-5 (far above
    float32 eps) but never reach the 0.1 deadband."""
    pipeline, _made_envs, parent_calls, tmp_path = harness
    pattern = [0.014293, 0.014354, 0.014300]
    config = _config(tmp_path)
    config["continuous_action_threshold"] = 0.1
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=_PatternAgent(pattern),
                          mode="train")
    meta = _meta(parent_calls)
    for row in meta["history"]:
        ev = row["handoff_viability_evidence"]
        assert ev["handoff_viability"] == "BELOW_NORMAL_THRESHOLD"
        constant = ev["constant_policy_classification"]
        assert constant["exact_constant"] is False
        assert constant["near_constant"] is False
        assert constant["unique_action_count_exact"] == 3
        assert ev["any_action_crosses_phase2_threshold"] is False
        assert ev["viable_as_trained_treatment"] is False


def test_crossing_policy_with_probe_trades_is_viable(harness):
    pipeline, _made_envs, parent_calls, tmp_path = harness
    pattern = [0.5, -0.4, 0.02]
    config = _config(tmp_path)
    config["continuous_action_threshold"] = 0.1
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=_PatternAgent(pattern),
                          mode="train")
    meta = _meta(parent_calls)
    expected_sha = _pattern_sha(pattern)
    for row in meta["history"]:
        ev = row["handoff_viability_evidence"]
        assert ev["handoff_viability"] == "VIABLE"
        assert ev["any_action_crosses_phase2_threshold"] is True
        assert ev["probe_trades_total"] == 2
        for split in ("train_monitor", "inner_validation"):
            block = ev["splits"][split]
            # the SAME raw vector under both thresholds, provably
            assert block["action_vector_sha256"] == expected_sha
            assert block["fraction_non_hold_phase1_threshold"] == 1.0
            assert block["fraction_non_hold_phase2_threshold"] == \
                pytest.approx(2.0 / 3.0)
            assert block["mapped_counts_phase2_threshold"] == {
                "long": 1, "short": 1, "hold": 1}
        assert ev["viable_as_trained_treatment"] is (row["epoch"] > 0)


def test_crossing_policy_with_zero_probe_trades_is_no_trade(harness):
    pipeline, _made_envs, parent_calls, tmp_path = harness
    config = _config(tmp_path)
    config["continuous_action_threshold"] = 0.1
    config["_fake_normal_trades"] = 0
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=_PatternAgent([0.5, -0.4, 0.02]),
                          mode="train")
    meta = _meta(parent_calls)
    for row in meta["history"]:
        ev = row["handoff_viability_evidence"]
        assert ev["handoff_viability"] == "NO_TRADE"
        assert ev["any_action_crosses_phase2_threshold"] is True
        assert ev["normal_probe"]["train_tail_trades"] == 0
        assert ev["normal_probe"]["validation_trades"] == 0
        assert ev["viable_as_trained_treatment"] is False


def test_epoch_zero_anchor_evidence_is_anchor_passthrough(harness):
    """Even a VIABLE anchor is marked anchor_passthrough and can never
    be viable AS A TRAINED TREATMENT (finding 221)."""
    pipeline, _made_envs, parent_calls, tmp_path = harness
    anchor = tmp_path / "anchor.zip"
    anchor.write_text("anchor-weights")
    config = _config(tmp_path)
    config["continuous_action_threshold"] = 0.1
    config["warm_start_model"] = str(anchor)
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=_PatternAgent([0.5, -0.4, 0.02]),
                          mode="train")
    meta = _meta(parent_calls)
    baseline = meta["history"][0]
    assert baseline["epoch"] == 0
    ev = baseline["handoff_viability_evidence"]
    assert ev["policy_provenance"] == "anchor_passthrough"
    assert ev["trained_treatment"] is False
    assert ev["handoff_viability"] == "VIABLE"          # anchor acts...
    assert ev["viable_as_trained_treatment"] is False   # ...but never
    for row in meta["history"][1:]:
        trained_ev = row["handoff_viability_evidence"]
        assert trained_ev["policy_provenance"] == "trained_epoch"
        assert trained_ev["trained_treatment"] is True


def test_v3_epoch0_selection_is_marked_anchor_never_trained(harness):
    """The v3 as-run path may still SELECT the epoch-0 anchor (that is
    the diagnostic reproduction), but the record must carry the
    explicit anchor_passthrough marker and can never be represented as
    a selected viable trained handoff."""
    pipeline, _made_envs, parent_calls, tmp_path = harness
    anchor = tmp_path / "anchor.zip"
    anchor.write_text("anchor-weights")
    config = _config(tmp_path)
    config["phase1_handoff_semantics"] = "m0_epoch0_eligible_v3"
    config["continuous_action_threshold"] = 0.1
    config["warm_start_model"] = str(anchor)
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=_PatternAgent([0.5, -0.4, 0.02]),
                          mode="train")
    meta = _meta(parent_calls)
    assert meta["best_easy_epoch"] == 0
    assert meta["selection_basis"] == \
        "m0_v3_economic_equity_epoch0_eligible"
    selected = meta["selected_handoff_viability"]
    assert selected["anchor_passthrough"] is True
    assert selected["policy_provenance"] == "anchor_passthrough"
    assert selected["trained_treatment"] is False
    assert selected["selected_as_viable_handoff"] is False
    assert selected["viability_is_selection_authority"] is False


def test_terminal_fallback_is_never_a_selected_viable_handoff(harness):
    """Order WP3 §7: the diagnostic terminal fallback hands off an
    inactive record; even when its action evidence is VIABLE it is
    never represented as a selected viable handoff."""
    pipeline, _made_envs, parent_calls, tmp_path = harness
    config = _config(tmp_path)
    config["continuous_action_threshold"] = 0.1
    pipeline.run_pipeline(config=config, env_plugin=None,
                          agent_plugin=_PatternAgent([0.5, -0.4, 0.02]),
                          mode="train")
    meta = _meta(parent_calls)
    assert meta["selection_basis"] == "terminal_trained_epoch_fallback"
    selected = meta["selected_handoff_viability"]
    assert selected["selection_is_diagnostic_fallback"] is True
    assert selected["handoff_viability"] == "VIABLE"
    assert selected["selected_as_viable_handoff"] is False


def test_unavailable_evidence_is_typed_and_never_gates(harness,
                                                       monkeypatch):
    pipeline, _made_envs, parent_calls, tmp_path = harness

    def broken_rollout(*_args, **_kwargs):
        raise RuntimeError("evidence env exploded")

    monkeypatch.setattr(pipeline, "_handoff_action_rollout",
                        broken_rollout)
    pipeline.run_pipeline(config=_config(tmp_path), env_plugin=None,
                          agent_plugin=FakeAgent(), mode="train")
    assert parent_calls, "evidence failure must never gate the phase"
    meta = _meta(parent_calls)
    for row in meta["history"]:
        ev = row["handoff_viability_evidence"]
        assert ev["handoff_viability"] == "UNAVAILABLE"
        for split in ("train_monitor", "inner_validation"):
            block = ev["splits"][split]
            assert block["available"] is False
            assert "evidence env exploded" in block["error"]
        assert ev["constant_policy_classification"] is None
        assert ev["viable_as_trained_treatment"] is False


def _selected_block(**overrides):
    block = {
        "schema":
            "agent_multi.solvency_curriculum"
            ".selected_handoff_viability.v1",
        "best_easy_epoch": 2,
        "selection_basis": "paired_comparator_best_trained_epoch",
        "phase1_handoff_semantics": "l1_trained_epoch_v4",
        "handoff_viability": "VIABLE",
        "policy_provenance": "trained_epoch",
        "trained_treatment": True,
        "anchor_passthrough": False,
        "selection_is_diagnostic_fallback": False,
        "selected_as_viable_handoff": True,
        "viability_is_selection_authority": False,
    }
    block.update(overrides)
    return block


def test_selected_handoff_source_assertions_fire():
    _assert_selected_handoff_invariants(_selected_block())  # valid
    with pytest.raises(RuntimeError, match="diagnostic terminal"):
        _assert_selected_handoff_invariants(_selected_block(
            selection_is_diagnostic_fallback=True))
    with pytest.raises(RuntimeError, match="anchor passthrough"):
        _assert_selected_handoff_invariants(_selected_block(
            anchor_passthrough=True,
            policy_provenance="anchor_passthrough"))
    with pytest.raises(RuntimeError, match="TRAINED epoch"):
        _assert_selected_handoff_invariants(_selected_block(
            anchor_passthrough=True,
            policy_provenance="anchor_passthrough",
            trained_treatment=False))
    with pytest.raises(RuntimeError, match="VIABLE"):
        _assert_selected_handoff_invariants(_selected_block(
            handoff_viability="CONSTANT_POLICY"))
    with pytest.raises(RuntimeError, match="unknown viability"):
        _assert_selected_handoff_invariants(_selected_block(
            handoff_viability="TOTALLY_FINE"))


def _evidence_block(**overrides):
    block = {
        "schema": "agent_multi.solvency_curriculum.handoff_viability.v1",
        "epoch": 1,
        "policy_provenance": "trained_epoch",
        "trained_treatment": True,
        "handoff_viability": "VIABLE",
        "viable_as_trained_treatment": True,
    }
    block.update(overrides)
    return block


def test_epoch_zero_evidence_source_assertions_fire():
    _assert_handoff_evidence_invariants(_evidence_block())  # valid
    # epoch 0 must carry the anchor_passthrough marker...
    with pytest.raises(RuntimeError, match="anchor_passthrough"):
        _assert_handoff_evidence_invariants(_evidence_block(
            epoch=0, viable_as_trained_treatment=False))
    # ...and can never be viable AS A TRAINED TREATMENT
    with pytest.raises(RuntimeError, match="TRAINED TREATMENT"):
        _assert_handoff_evidence_invariants(_evidence_block(
            epoch=0, policy_provenance="anchor_passthrough",
            trained_treatment=False))
    with pytest.raises(RuntimeError, match="unknown handoff_viability"):
        _assert_handoff_evidence_invariants(_evidence_block(
            handoff_viability="GREAT"))
    with pytest.raises(RuntimeError, match="must equal"):
        _assert_handoff_evidence_invariants(_evidence_block(
            handoff_viability="CONSTANT_POLICY"))
