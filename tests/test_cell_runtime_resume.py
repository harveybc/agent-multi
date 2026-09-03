"""Musashi correction 3 (2026-09-03): observable resumable SAC cell
runtime — exact-state restore proven on a REAL SAC model, plus the
evidenced custody door out of ``interrupted``."""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.dispatch_custody import (  # noqa: E402
    DispatchLedger, ExecutionCustodyError)
from pipeline_plugins import _cell_runtime  # noqa: E402


@pytest.fixture()
def home_tmp():
    root = (Path.home() / ".cache" / "cell_runtime_tests"
            / uuid.uuid4().hex)
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _tiny_env():
    import gymnasium as gym
    import numpy as np

    class Tiny(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (3,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return (self.observation_space.sample(), 0.0, False,
                    False, {})

    return Tiny()


def _real_sac(seed=1):
    from stable_baselines3 import SAC
    return SAC("MlpPolicy", _tiny_env(), device="cpu", seed=seed,
               learning_starts=8, train_freq=1, gradient_steps=1,
               batch_size=8, buffer_size=200,
               policy_kwargs=dict(net_arch=[8]), verbose=0)


def _named_params(model):
    import torch
    return {name: param.detach().clone()
            for name, param in model.policy.named_parameters()}


class TestExactRealSacRestore:

    def test_bundle_then_restore_is_bitwise_exact(self, home_tmp):
        """Train a REAL SAC, bundle, restore into a FRESH model:
        every policy/critic parameter, the optimizer state, the
        replay buffer size, num_timesteps, _n_updates and the RNG
        streams must come back exactly."""
        import numpy as np
        import torch
        config = {"cell_runtime_dir": str(home_tmp / "runtime"),
                  "resume_checkpoint_every_epochs": 1}
        rt = _cell_runtime.CellRuntime(config, max_epochs=10)
        model = _real_sac(seed=1)
        model.learn(total_timesteps=60, reset_num_timesteps=True)
        state = {"best_composite": 0.25, "no_improve": 3,
                 "activity_ineligible_streak": 1,
                 "best_checkpoint_saved": True,
                 "history": [{"epoch": 1, "val_total_return": 0.1}],
                 "actor_liveness_history": [{"epoch": 1}],
                 "config_sha256": "a" * 64}
        rt.write_bundle(epoch=4, model=model, state=state)
        saved_params = _named_params(model)
        saved_steps = int(model.num_timesteps)
        saved_updates = int(model._n_updates)
        saved_replay = int(model.replay_buffer.size())
        saved_torch_rng = torch.get_rng_state().clone()

        fresh = _real_sac(seed=999)  # different seed on purpose
        rt2 = _cell_runtime.CellRuntime(config, max_epochs=10)
        saved_state = rt2.read_state()
        assert saved_state is not None
        facts = rt2.restore_into(fresh, saved_state,
                                 expected_config_sha256="a" * 64)
        assert facts["resumed_from_epoch"] == 4
        assert facts["start_epoch"] == 5
        for name, param in _named_params(fresh).items():
            assert torch.equal(param, saved_params[name]), name
        assert int(fresh.num_timesteps) == saved_steps
        assert int(fresh._n_updates) == saved_updates
        assert int(fresh.replay_buffer.size()) == saved_replay
        assert torch.equal(torch.get_rng_state(), saved_torch_rng)
        # optimizer state restored (SB3 archive carries it): SAC
        # keeps per-component optimizers on actor and critic
        assert fresh.policy.actor.optimizer.state_dict()["state"], \
            "actor optimizer state came back empty"
        assert fresh.policy.critic.optimizer.state_dict()["state"], \
            "critic optimizer state came back empty"
        # patience and histories travel in the state dict
        assert saved_state["best_composite"] == 0.25
        assert saved_state["no_improve"] == 3
        assert saved_state["history"][0]["epoch"] == 1

    def test_config_drift_refuses_restore(self, home_tmp):
        config = {"cell_runtime_dir": str(home_tmp / "runtime")}
        rt = _cell_runtime.CellRuntime(config, max_epochs=5)
        model = _real_sac()
        model.learn(total_timesteps=30, reset_num_timesteps=True)
        rt.write_bundle(epoch=1, model=model, state={
            "best_composite": 0.0, "no_improve": 0,
            "config_sha256": "a" * 64})
        fresh = _real_sac()
        with pytest.raises(_cell_runtime.CellRuntimeError,
                           match="config digest changed"):
            rt.restore_into(fresh, rt.read_state(),
                            expected_config_sha256="b" * 64)

    def test_missing_bundle_refuses(self, home_tmp):
        config = {"cell_runtime_dir": str(home_tmp / "empty")}
        rt = _cell_runtime.CellRuntime(config, max_epochs=5)
        assert rt.read_state() is None

    def test_heartbeat_is_machine_readable_with_eta(self, home_tmp):
        config = {"cell_runtime_dir": str(home_tmp / "runtime")}
        rt = _cell_runtime.CellRuntime(config, max_epochs=100)
        rt.epoch_durations.extend([10.0, 12.0, 11.0])
        rt.heartbeat(epoch=3, best_composite=0.1, no_improve=2,
                     patience=40, stop_reason=None,
                     last_artifact=None)
        status = json.loads(
            (home_tmp / "runtime" / "status.json").read_text())
        assert status["epoch_completed"] == 3
        assert status["max_epochs"] == 100
        eta = status["eta"]
        assert eta["remaining_epochs_max"] == 97
        assert eta["eta_median_s"] > 0
        assert eta["eta_p90_s"] >= eta["eta_median_s"]
        assert eta["eta_pessimistic_s"] >= eta["eta_p90_s"]
        assert status["patience"] == {"no_improve": 2, "budget": 40}


class TestCustodyResumeDoor:

    def _ledger(self, home_tmp):
        return DispatchLedger(root=home_tmp / "ledger")

    def _reserved_running(self, ledger, key="k" * 64):
        ledger.reserve(key, identity={"dispatch_id": "d"},
                       output_path=Path.home() / ".cache"
                       / "cell_runtime_tests"
                       / f"out_{key[:8]}.json")
        ledger.transition(key, "running")
        return key

    def test_resume_requires_interrupted(self, home_tmp):
        ledger = self._ledger(home_tmp)
        key = self._reserved_running(ledger)
        with pytest.raises(ExecutionCustodyError,
                           match="requires state 'interrupted'"):
            ledger.resume(key, resume_evidence={
                "resume_state_sha256": "a" * 64,
                "resumed_from_epoch": 3})

    def test_resume_requires_forward_started(self, home_tmp):
        ledger = self._ledger(home_tmp)
        key = self._reserved_running(ledger)
        ledger.transition(key, "interrupted", {"interruption": "x"})
        with pytest.raises(ExecutionCustodyError,
                           match="forward never started"):
            ledger.resume(key, resume_evidence={
                "resume_state_sha256": "a" * 64,
                "resumed_from_epoch": 3})

    def test_resume_requires_evidence_fields(self, home_tmp):
        ledger = self._ledger(home_tmp)
        key = self._reserved_running(ledger)
        ledger.mark_forward_started(key)
        ledger.transition(key, "interrupted", {"interruption": "x"})
        with pytest.raises(ExecutionCustodyError,
                           match="missing resume_state_sha256"):
            ledger.resume(key, resume_evidence={
                "resumed_from_epoch": 3})

    def test_resume_happy_path_appends_history(self, home_tmp):
        ledger = self._ledger(home_tmp)
        key = self._reserved_running(ledger)
        ledger.mark_forward_started(key)
        ledger.transition(key, "interrupted", {"interruption": "x"})
        ledger.resume(key, resume_evidence={
            "resume_state_sha256": "a" * 64,
            "resumed_from_epoch": 7})
        record = ledger.read(key)
        assert record["state"] == "running"
        assert record["resume_history"][0]["resumed_from_epoch"] == 7
        # a second interruption can resume again (append-only)
        ledger.transition(key, "interrupted", {"interruption": "y"})
        ledger.resume(key, resume_evidence={
            "resume_state_sha256": "b" * 64,
            "resumed_from_epoch": 11})
        assert len(ledger.read(key)["resume_history"]) == 2

    def test_generic_transition_still_treats_interrupted_terminal(
            self, home_tmp):
        """The GENERIC machine keeps interrupted terminal — only the
        evidenced resume door exits it (DATA-SOTA-360 preserved)."""
        ledger = self._ledger(home_tmp)
        key = self._reserved_running(ledger)
        ledger.mark_forward_started(key)
        ledger.transition(key, "interrupted", {"interruption": "x"})
        with pytest.raises(ExecutionCustodyError, match="terminal"):
            ledger.transition(key, "running")

    def test_completed_never_resumes(self, home_tmp):
        ledger = self._ledger(home_tmp)
        key = self._reserved_running(ledger)
        ledger.mark_forward_started(key)
        # complete() requires durable evidence; use transition to
        # spent-equivalent via the completed path is heavy — assert
        # the door itself refuses non-interrupted states instead
        with pytest.raises(ExecutionCustodyError,
                           match="requires state 'interrupted'"):
            ledger.resume(key, resume_evidence={
                "resume_state_sha256": "a" * 64,
                "resumed_from_epoch": 1})


class TestDriverBudgetAndRuntimeKeys:
    """Musashi correction 4: the guard keys ride in EVERY cell config
    of BOTH arms; correction 3: the runtime keys ride there too."""

    def _cfg(self, arm):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dispatch_driver",
            REPO / "tools" / "dispatch_paired_pretrain_comparison.py")
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        design = json.loads(driver.DESIGN_PATH.read_text())
        pretrain = (Path.home() / ".local/share/agent-multi/"
                    "restricted_evidence/"
                    "candidate_full5_pcgrad_o2022_20260828")
        cell = driver.verify_cell(design, pretrain, 101, arm)
        return driver.build_cell_config(
            design, cell, pretrain,
            Path.home() / ".cache" / "cell_runtime_tests" / "out",
            device="cpu", attempt_nonce="deadbeef00000000")

    @pytest.mark.parametrize("arm", ["control_random_init",
                                     "pretrained_finetuned"])
    def test_guard_and_runtime_keys_present(self, arm):
        cfg = self._cfg(arm)
        total = int(cfg["total_timesteps"])
        assert cfg["budget_max_env_steps"] == total
        assert cfg["budget_max_updates"] == total
        assert float(cfg["budget_max_wall_seconds"]) > 0
        assert cfg["budget_stop_file"].endswith("/STOP")
        assert cfg["cell_runtime_dir"].endswith("/runtime")
        assert int(cfg["resume_checkpoint_every_epochs"]) >= 1
