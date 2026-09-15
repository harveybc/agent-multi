"""What the run actually spent, measured on the real library rather than named after it.

S1 of `predictor/docs/handoffs/MUSASHI_TO_SATOSHI_COUNTERS_ARCHIVE_AND_TERMS_2026_09_15.md`:

    "Distinguish requested training target, hard training-transition limit, rollout steps per
     environment, environment count, collected rollouts, completed optimization epochs, actual
     optimizer calls and evaluation transitions. [...] Do not label PPO _n_updates as optimizer
     calls. [...] A partial rollout must not pretend to have trained."

These run against real Stable-Baselines3 on CartPole, CPU only, one BLAS thread. CartPole is
used deliberately: it is the library's own reference environment, so what these measure is the
LIBRARY's semantics and not this project's environment. No trading environment is constructed,
no market data is read and nothing is trained for any scientific purpose.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from pipeline_plugins._compute_contract import (  # noqa: E402
    ComputeContractRefusal,
    ComputeMeter,
    N_UPDATES_MEANING,
)

torch.set_num_threads(1)


@pytest.fixture
def cartpole():
    envs = []

    def build(n_envs=1):
        env = DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(n_envs)])
        envs.append(env)
        return env

    yield build
    for env in envs:
        env.close()


def ppo(env, **kwargs):
    settings = {"n_steps": 64, "batch_size": 64, "n_epochs": 1}
    settings.update(kwargs)
    return PPO("MlpPolicy", env, device="cpu", seed=0, verbose=0, **settings)


def measured(model, config, learn_kwargs=None):
    meter = ComputeMeter(model, config)
    meter.refuse_if_infeasible()
    with meter:
        model.learn(config["total_timesteps"], **(learn_kwargs or {}))
    return meter


# --- the correction itself -------------------------------------------------------------

def test_ppo_updates_are_epochs_and_are_not_optimizer_calls(cartpole):
    """The finding Musashi measured, now a rule: the two numbers differ, and both are named.

    Four rollouts of 64 transitions, two epochs each, is eight epochs. The optimizer steps
    once per MINIBATCH, and a minibatch is carved out of ONE rollout: 64 transitions at batch
    32 is two minibatches, so sixteen optimizer calls. Anything that reports one of those two
    numbers as the other is wrong by the minibatch count, which grows as the batch shrinks.
    """
    model = ppo(cartpole(), n_steps=64, batch_size=32, n_epochs=2)
    record = measured(model, {"total_timesteps": 256}).record()

    assert record["training_transitions_observed"] == 256
    assert record["collected_rollouts"] == 4
    assert record["sb3_n_updates_meaning"] == "optimization_epochs"
    assert record["optimization_epochs_completed"] == 8, "four rollouts x two epochs"
    assert record["optimizer_step_calls"] == 16, "eight epochs x two minibatches per rollout"
    assert record["optimizer_step_calls"] != record["sb3_n_updates_delta"], (
        "the whole point: naming _n_updates 'gradient updates' was false for PPO")
    assert record["gradient_updates"] is None, "PPO makes no gradient-update claim"


def test_the_meaning_table_matches_the_installed_library(cartpole):
    """The table is only allowed to say what this library was measured to do.

    PPO increments `_n_updates` once per epoch, outside the minibatch loop. The arithmetic is
    checked against the resolved settings rather than a memorised constant.
    """
    assert N_UPDATES_MEANING["PPO"] == "optimization_epochs"
    model = ppo(cartpole(), n_steps=64, batch_size=64, n_epochs=3)
    record = measured(model, {"total_timesteps": 128}).record()
    rollouts = record["collected_rollouts"]
    minibatches = record["training_transitions_observed"] // rollouts // record["batch_size"]
    assert record["sb3_n_updates_delta"] == rollouts * record["optimization_epochs_configured"]
    assert record["optimizer_step_calls"] == (
        rollouts * record["optimization_epochs_configured"] * minibatches)
    assert record["library_version"] == sb3.__version__


def test_an_unsupported_algorithm_reports_unavailable_instead_of_guessing():
    """A counter does not gain a meaning by sharing a name. An unknown algorithm claims nothing."""

    class Mystery:
        num_timesteps = 0
        _n_updates = 0
        policy = None

    meter = ComputeMeter(Mystery(), {"total_timesteps": 10})
    meter.start()
    Mystery.num_timesteps, Mystery._n_updates = 10, 7
    record = meter.stop().record()

    assert record["algorithm"] == "Mystery"
    assert record["sb3_n_updates_meaning"] == "UNKNOWN"
    assert record["sb3_n_updates_delta"] == 7, "the raw counter is still reported"
    assert record["optimization_epochs_completed"] is None
    assert record["gradient_updates"] is None
    assert "optimization_epochs_completed" in record["counters_unavailable"]
    assert "optimizer_step_calls" in record["counters_unavailable"]
    assert record["optimizer_step_calls"] is None


# --- the ceiling -----------------------------------------------------------------------

def test_an_impossible_cap_is_refused_before_a_single_transition(cartpole):
    """The exact 64/256 case: the refusal must come BEFORE any environment step."""
    model = ppo(cartpole(), n_steps=256, batch_size=64, n_epochs=10)
    meter = ComputeMeter(model, {"total_timesteps": 64, "training_transition_cap": 64})
    with pytest.raises(ComputeContractRefusal) as refusal:
        meter.refuse_if_infeasible()
    assert "256" in str(refusal.value) and "64" in str(refusal.value)
    assert model.num_timesteps == 0, "the configuration was refused, not attempted"


def test_a_target_above_the_cap_is_refused_rather_than_the_cap_widened(cartpole):
    model = ppo(cartpole())
    meter = ComputeMeter(model, {"total_timesteps": 128, "training_transition_cap": 64})
    with pytest.raises(ComputeContractRefusal):
        meter.refuse_if_infeasible()
    assert meter.cap == 64, "the declared ceiling is never rewritten to fit the request"


def test_the_declared_mechanical_configuration_fits_its_cap_exactly(cartpole):
    """One environment, n_steps 64, batch 64, one epoch, target 64, cap 64 - and it holds."""
    model = ppo(cartpole(), n_steps=64, batch_size=64, n_epochs=1)
    config = {"total_timesteps": 64, "training_transition_cap": 64,
              "evaluation_transition_cap": 384}
    record = measured(model, config).record(evaluation_transitions=384)

    assert record["minimum_training_transitions"] == 64
    assert record["training_transitions_observed"] == 64
    assert record["collected_rollouts"] == 1
    assert record["partial_rollout"] is False
    assert record["optimization_epochs_completed"] == 1
    assert record["optimizer_step_calls"] == 1
    assert record["cap_respected"] is True
    assert record["evaluation_cap_respected"] is True


# --- counting what actually ran --------------------------------------------------------

def test_vectorized_runs_count_transitions_and_not_calls_to_step(cartpole):
    """Four environments spend four transitions per step: the rollout is n_steps x n_envs."""
    model = ppo(cartpole(n_envs=4), n_steps=64, batch_size=64, n_epochs=1)
    record = measured(model, {"total_timesteps": 64}).record()

    assert record["environment_count"] == 4
    assert record["rollout_steps_per_environment"] == 64
    assert record["minimum_training_transitions"] == 256
    assert record["training_transitions_observed"] == 256, (
        "a target of 64 cannot be honoured by four environments collecting whole rollouts")
    assert record["collected_rollouts"] == 1
    assert record["optimizer_step_calls"] == 4, "256 transitions in minibatches of 64"


def test_a_resumed_model_does_not_report_a_zero_delta_for_real_work(cartpole, tmp_path):
    """The trap: `learn()` RESETS the step counter by default, so 64 -> 64 looks like no work.

    Measured on this library, not assumed. `_n_updates` is not reset, which is how the reset
    is caught: real optimization happened while the naive step delta was zero.
    """
    env = cartpole()
    first = ppo(env)
    first.learn(64)
    saved = tmp_path / "resumed.zip"
    first.save(saved)

    resumed = PPO.load(saved, env=env)
    record = measured(resumed, {"total_timesteps": 64}).record()

    assert record["model_timesteps_before"] == 64
    assert record["model_timesteps_after"] == 64
    assert record["timestep_counter_was_reset"] is True
    assert record["training_transitions_observed"] == 64, (
        "the naive after-minus-before is zero here and would erase a real training call")
    assert record["optimization_epochs_completed"] == 1


def test_a_continued_model_reports_only_this_calls_work(cartpole, tmp_path):
    """With the counter NOT reset, lifetime and this-call must not be confused."""
    env = cartpole()
    first = ppo(env)
    first.learn(64)
    saved = tmp_path / "continued.zip"
    first.save(saved)

    resumed = PPO.load(saved, env=env)
    record = measured(resumed, {"total_timesteps": 64},
                      learn_kwargs={"reset_num_timesteps": False}).record()

    assert record["model_timesteps_before"] == 64
    assert record["model_timesteps_after"] == 128, "lifetime"
    assert record["timestep_counter_was_reset"] is False
    assert record["training_transitions_observed"] == 64, "this call only"


class StopAfter(BaseCallback):
    """Real early termination: the library's own mechanism, stopping inside a rollout."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def _on_step(self) -> bool:
        return self.num_timesteps < self.limit


def test_a_partial_rollout_does_not_pretend_to_have_trained(cartpole):
    """Stopped inside its first rollout: transitions were spent and NOTHING was optimized."""
    model = ppo(cartpole(), n_steps=64, batch_size=64, n_epochs=1)
    meter = ComputeMeter(model, {"total_timesteps": 64, "training_transition_cap": 64})
    meter.refuse_if_infeasible()
    with meter:
        model.learn(64, callback=StopAfter(30))
    record = meter.record()

    assert 0 < record["training_transitions_observed"] < 64
    assert record["collected_rollouts"] == 0, "no whole rollout was collected"
    assert record["partial_rollout"] is True
    assert record["optimization_epochs_completed"] == 0
    assert record["optimizer_step_calls"] == 0, "the optimizer was never called"
    assert record["cap_respected"] is True


def test_the_optimizer_is_instrumented_in_place_and_handed_back(cartpole):
    """The count comes from the real optimizer, and the model is left exactly as found."""
    model = ppo(cartpole())
    optimizer = model.policy.optimizer
    assert "step" not in vars(optimizer), "the method is the class's, before instrumentation"
    meter = ComputeMeter(model, {"total_timesteps": 64})
    meter.start()
    assert "step" in vars(optimizer), "nothing would be counted otherwise"
    model.learn(64)
    meter.stop()
    assert "step" not in vars(optimizer), (
        "the instrumentation must not survive: leaving a wrapper on the instance would count "
        "a later run into this meter and keep the closure alive with the model")
    assert meter.record()["optimizer_step_calls"] == 1


def test_evaluation_transitions_are_never_training_transitions(cartpole):
    """Evaluation is counted, capped and reported apart; it never enters the training numbers."""
    model = ppo(cartpole())
    record = measured(model, {"total_timesteps": 64, "training_transition_cap": 64,
                              "evaluation_transition_cap": 384}).record(
        evaluation_transitions=400)

    assert record["training_transitions_observed"] == 64
    assert record["evaluation_transitions"] == 400
    assert record["evaluation_cap_respected"] is False, "reported, not silently absorbed"
    assert record["cap_respected"] is True, "an evaluation overrun is not a training overrun"


def test_an_unmeasured_evaluation_is_declared_missing_rather_than_zero(cartpole):
    model = ppo(cartpole())
    record = measured(model, {"total_timesteps": 64}).record()
    assert record["evaluation_transitions"] is None
    assert "evaluation_transitions" in record["counters_unavailable"]
    assert record["cap_respected"] is None, "no cap was declared, so none was respected"
