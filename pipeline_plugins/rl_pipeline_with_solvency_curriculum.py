"""Stage-integrated ``easy -> normal`` solvency curriculum (order WP-D).

For EVERY candidate, in every outer DOIN stage, this pipeline:

1. decodes/freezes the candidate contract exactly as the validation
   pipeline does (same splits, same observation contract);
2. trains first under ``easy_chronological_continuation`` (train-only
   relaxed solvency dynamics, WP-C) with early stopping and a declared
   maximum budget — the easy score controls only the budget, never
   selection;
3. saves an immutable ``post_easy`` artifact (weights, sha256, easy
   event counters, history);
4. continues under ``normal_realistic`` FROM the learned weights via the
   validation pipeline's warm start; the SB3 artifact reload at the
   dynamics boundary yields a fresh replay buffer, so unrealistic easy
   transitions can never contaminate normal updates;
5. saves the ``post_normal`` artifact and returns the parent pipeline's
   transparent normal-validation selection result unchanged.

Evaluation is structurally normal: the parent's evaluation path forces
``normal_realistic`` for every split env, and gym-fx itself refuses easy
dynamics outside ``env_mode=training``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

from pipeline_plugins.rl_pipeline_with_validation import (
    PipelinePlugin as ValidationPipelinePlugin,
    _verify_artifact_sha256,
)

EASY_MODE = "easy_chronological_continuation"
NORMAL_MODE = "normal_realistic"


class PipelinePlugin(ValidationPipelinePlugin):
    plugin_params = {
        **ValidationPipelinePlugin.plugin_params,
        "easy_epoch_timesteps": None,     # default: epoch_timesteps
        "easy_max_epochs": 4,             # declared maximum budget
        "easy_patience": 2,               # early stop (budget control only)
        "easy_min_delta": 0.0,
    }

    # ------------------------------------------------------------------
    def _easy_probe(
        self,
        env_plugin_name: str,
        easy_config: Dict[str, Any],
        csv_path: str,
        agent_plugin,
        model,
        seed: int,
    ) -> Dict[str, Any]:
        """One deterministic rollout on a FRESH easy env; returns the
        ECONOMIC outcome (operational equity minus recapitalization debt)
        plus solvency event counters. Used exclusively for the easy
        phase's early-stopping budget — never for selection."""
        plug, env = self._make_split_env(
            env_plugin_name, easy_config, csv_path, agent_plugin)
        try:
            obs, _info = env.reset(seed=seed)
            terminated = truncated = False
            last_info: Dict[str, Any] = {}
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, last_info = env.step(
                    action)
            return {
                "economic_equity": float(
                    last_info.get("economic_equity", float("nan"))),
                "recapitalization_debt": float(
                    last_info.get("recapitalization_debt", 0.0)),
                "recapitalization_count": int(
                    last_info.get("recapitalization_count", 0)),
                "would_margin_call_count": int(
                    last_info.get("would_margin_call_count", 0)),
                "termination_cause": last_info.get("termination_cause"),
            }
        finally:
            try:
                plug.close()
            except Exception:
                pass

    def _train_easy_phase(
        self,
        *,
        config: Dict[str, Any],
        env_plugin,
        agent_plugin,
    ) -> Dict[str, Any]:
        """Phase 1: train under easy dynamics, save post_easy immutably."""
        easy_config = dict(config)
        easy_config["solvency_mode"] = EASY_MODE
        easy_config["env_mode"] = "training"
        env_plugin_name = easy_config.get("env_plugin", "gym_fx_env")
        paths = self._split_csv(easy_config)
        _plug, easy_env = self._make_split_env(
            env_plugin_name, easy_config, paths["train"], agent_plugin)
        try:
            model = agent_plugin.build(easy_env, easy_config)
            epoch_ts = int(
                easy_config.get("easy_epoch_timesteps")
                or easy_config.get("epoch_timesteps",
                                   self.params["epoch_timesteps"]))
            max_epochs = int(easy_config.get(
                "easy_max_epochs", self.params["easy_max_epochs"]))
            patience = int(easy_config.get(
                "easy_patience", self.params["easy_patience"]))
            min_delta = float(easy_config.get(
                "easy_min_delta", self.params["easy_min_delta"]))
            seed = int(easy_config.get("eval_seed",
                                       self.params["eval_seed"]))
            best = -math.inf
            waited = 0
            history = []
            for epoch in range(1, max_epochs + 1):
                model.learn(total_timesteps=epoch_ts,
                            reset_num_timesteps=False)
                probe = self._easy_probe(
                    env_plugin_name, easy_config, paths["train"],
                    agent_plugin, model, seed)
                probe["epoch"] = epoch
                history.append(probe)
                score = probe["economic_equity"]
                if math.isfinite(score) and score > best + min_delta:
                    best = score
                    waited = 0
                else:
                    waited += 1
                if waited >= patience:
                    break

            save_model = str(config.get("save_model")
                             or "./agent_model.zip")
            post_easy_path = Path(save_model).with_suffix("")
            post_easy_path = post_easy_path.parent / (
                post_easy_path.name + ".post_easy.zip")
            post_easy_path.parent.mkdir(parents=True, exist_ok=True)
            agent_plugin.save(model, str(post_easy_path))
            post_easy_sha = _verify_artifact_sha256(post_easy_path, None)
            meta = {
                "schema": "agent_multi.solvency_curriculum.post_easy.v1",
                "artifact": str(post_easy_path),
                "artifact_sha256": post_easy_sha,
                "solvency_mode": EASY_MODE,
                "easy_epochs_run": len(history),
                "easy_budget_epochs": max_epochs,
                "easy_epoch_timesteps": epoch_ts,
                "history": history,
            }
            meta_path = Path(str(post_easy_path) + ".meta.json")
            meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True),
                                 encoding="utf-8")
            return {"path": str(post_easy_path), "sha256": post_easy_sha,
                    "meta": meta}
        finally:
            try:
                easy_env.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        *,
        config: Dict[str, Any],
        env_plugin,
        agent_plugin,
        mode: str = "train",
    ) -> Dict[str, Any]:
        if str(mode).lower() != "train" or not bool(
                config.get("solvency_curriculum_enabled", True)):
            passthrough = dict(config)
            passthrough["solvency_mode"] = NORMAL_MODE
            return super().run_pipeline(
                config=passthrough, env_plugin=env_plugin,
                agent_plugin=agent_plugin, mode=mode)

        post_easy = self._train_easy_phase(
            config=config, env_plugin=env_plugin,
            agent_plugin=agent_plugin)

        normal_config = dict(config)
        normal_config["solvency_mode"] = NORMAL_MODE
        # Warm continuation from the learned weights; the artifact reload
        # gives the normal learner a FRESH replay buffer at the dynamics
        # boundary (easy transitions never enter normal updates).
        normal_config["warm_start_model"] = post_easy["path"]
        normal_config["warm_start_model_sha256"] = post_easy["sha256"]
        normal_config["warm_start_expand_observation_space"] = False

        result = super().run_pipeline(
            config=normal_config, env_plugin=env_plugin,
            agent_plugin=agent_plugin, mode="train")
        result["curriculum"] = {
            "schema": "agent_multi.solvency_curriculum.result.v1",
            "phases": ["easy_chronological_continuation",
                       "normal_realistic"],
            "post_easy": post_easy["meta"],
            "post_normal_artifact": result.get("best_model_path"),
            "replay_buffer_boundary": (
                "fresh buffer via artifact reload at the dynamics"
                " boundary; easy transitions excluded from normal"
                " updates"),
            "selection_basis": "normal_validation_only",
        }
        return result
