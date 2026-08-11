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

from agent_plugins.sac_agent import _policy_tensor_hash
from pipeline_plugins import _paired_generalization as _paired
from pipeline_plugins.rl_pipeline_with_validation import (
    PipelinePlugin as ValidationPipelinePlugin,
    _verify_artifact_sha256,
)

EASY_MODE = "easy_chronological_continuation"
NORMAL_MODE = "normal_realistic"

# Phase-1 handoff selection semantics (finding 220 / mechanism-ladder
# order WP3). The M0 screen of 2026-08-07 ran the *v3* boundary: the
# epoch-0 warm-start baseline (the anchor itself) was ELIGIBLE to be
# selected as the post_easy handoff, selection scored the easy-probe
# economic equity, and the normal-handoff probe GATED the save. The
# corrected L1 boundary (v4, findings 159/160/195/200) makes epoch 0
# structurally ineligible, selects on the paired comparator over
# trained epochs only, and demotes the normal probe to telemetry. The
# bounded D0-D4 diagnostic must reproduce the M0 boundary *as the
# screen ran it*, so the semantics are a typed, config-gated choice —
# the default is and stays the corrected v4 behavior.
HANDOFF_SEMANTICS_L1_V4 = "l1_trained_epoch_v4"
HANDOFF_SEMANTICS_M0_V3 = "m0_epoch0_eligible_v3"
_HANDOFF_SEMANTICS = (HANDOFF_SEMANTICS_L1_V4, HANDOFF_SEMANTICS_M0_V3)


class PipelinePlugin(ValidationPipelinePlugin):
    plugin_params = {
        **ValidationPipelinePlugin.plugin_params,
        "easy_epoch_timesteps": None,     # default: epoch_timesteps
        "easy_max_epochs": 4,             # declared maximum budget
        "easy_patience": 2,               # early stop (budget control only)
        "easy_min_delta": 0.0,
        # Easy removes the action deadband and uses the existing strictly
        # positive easy-floor execution costs. Normal evaluation restores
        # the candidate's original threshold and cost contract.
        "easy_continuous_action_threshold": 0.0,
        "easy_commission_fraction_per_side": 0.00005,
        "easy_full_spread_rate": 0.0001,
        "easy_slippage_bps_per_side": 0.25,
        "easy_min_trades": 1,
        "phase1_handoff_semantics": HANDOFF_SEMANTICS_L1_V4,
    }

    plugin_debug_vars = [
        *ValidationPipelinePlugin.plugin_debug_vars,
        "easy_epoch_timesteps", "easy_max_epochs", "easy_patience",
        "easy_min_delta", "easy_continuous_action_threshold",
        "easy_commission_fraction_per_side", "easy_full_spread_rate",
        "easy_slippage_bps_per_side", "easy_min_trades",
        "phase1_handoff_semantics",
    ]

    def _easy_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        easy = dict(config)
        threshold = float(easy.get(
            "easy_continuous_action_threshold",
            self.params["easy_continuous_action_threshold"],
        ))
        commission = float(easy.get(
            "easy_commission_fraction_per_side",
            self.params["easy_commission_fraction_per_side"],
        ))
        full_spread = float(easy.get(
            "easy_full_spread_rate", self.params["easy_full_spread_rate"]
        ))
        slippage_bps = float(easy.get(
            "easy_slippage_bps_per_side",
            self.params["easy_slippage_bps_per_side"],
        ))
        minimum_trades = int(easy.get(
            "easy_min_trades", self.params["easy_min_trades"]
        ))
        if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ValueError(
                "easy_continuous_action_threshold must be finite in [0, 1)"
            )
        if (
            not all(math.isfinite(value) for value in (
                commission, full_spread, slippage_bps
            ))
            or min(commission, full_spread, slippage_bps) <= 0.0
        ):
            raise ValueError(
                "easy execution costs must all be finite and strictly positive"
            )
        if minimum_trades < 1:
            raise ValueError("easy_min_trades must be >= 1")

        easy.update({
            "solvency_mode": EASY_MODE,
            "env_mode": "training",
            "continuous_action_threshold": threshold,
            "commission": commission,
            "full_spread_rate": full_spread,
            "slippage": full_spread / 2.0 + slippage_bps / 10_000.0,
            "easy_min_trades": minimum_trades,
        })
        # M0 order §8.1: the easy phase keeps its own learning rate while
        # the NORMAL fine-tune rate varies per arm. Without this
        # override, a reduced normal LR would silently also slow the
        # easy phase and confound the mechanism screen.
        easy_lr = easy.get("easy_learning_rate")
        if easy_lr is not None:
            if isinstance(easy_lr, bool) or not isinstance(
                easy_lr, (int, float)
            ):
                raise ValueError("easy_learning_rate must be a number")
            easy_lr = float(easy_lr)
            if not math.isfinite(easy_lr) or easy_lr <= 0.0:
                raise ValueError(
                    "easy_learning_rate must be finite and strictly"
                    " positive"
                )
            easy["learning_rate"] = easy_lr
        return easy

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
            action_diagnostics = dict(
                last_info.get("action_diagnostics") or {}
            )
            execution_diagnostics = dict(
                last_info.get("execution_diagnostics") or {}
            )
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
                "trades_total": int(last_info.get("trades", 0) or 0),
                "non_hold_actions": int(
                    action_diagnostics.get("non_hold_actions", 0) or 0
                ),
                "entry_actions_seen": int(
                    execution_diagnostics.get("entry_actions_seen", 0) or 0
                ),
                "entry_orders_submitted": int(
                    execution_diagnostics.get(
                        "entry_orders_submitted", 0
                    ) or 0
                ),
                "protected_entry_rejections": int(
                    execution_diagnostics.get(
                        "protected_entry_rejections", 0
                    ) or 0
                ),
                "action_diagnostics": action_diagnostics,
                "execution_diagnostics": execution_diagnostics,
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
        handoff_semantics = str(config.get(
            "phase1_handoff_semantics",
            self.params["phase1_handoff_semantics"]))
        if handoff_semantics not in _HANDOFF_SEMANTICS:
            raise ValueError(
                f"unknown phase1_handoff_semantics "
                f"{handoff_semantics!r}; expected one of "
                f"{sorted(_HANDOFF_SEMANTICS)}")
        phase1_mode = str(config.get("phase1_mode", EASY_MODE))
        if phase1_mode == NORMAL_MODE:
            # matched-boundary NORMAL phase 1 (doc 38 §5.3): identical
            # phase structure, selection and handoff machinery; only the
            # solvency dynamics differ from the easy arm.
            easy_config = dict(config)
            easy_config["solvency_mode"] = NORMAL_MODE
            easy_config["env_mode"] = "training"
            phase1_lr = easy_config.get("phase1_learning_rate",
                                        easy_config.get("easy_learning_rate"))
            if phase1_lr is not None:
                easy_config["learning_rate"] = float(phase1_lr)
            easy_config["easy_min_trades"] = int(
                easy_config.get("easy_min_trades",
                                self.params["easy_min_trades"]))
        elif phase1_mode == EASY_MODE:
            easy_config = self._easy_training_config(config)
        else:
            raise ValueError(f"unknown phase1_mode {phase1_mode!r}")
        env_plugin_name = easy_config.get("env_plugin", "gym_fx_env")
        paths = self._split_csv(easy_config)
        _plug, easy_env = self._make_split_env(
            env_plugin_name, easy_config, paths["train"], agent_plugin)
        try:
            warm_start_model = easy_config.get("warm_start_model")
            if warm_start_model:
                warm_start_path = Path(str(warm_start_model))
                if not warm_start_path.exists():
                    raise FileNotFoundError(
                        f"warm_start_model not found: {warm_start_path}"
                    )
                _verify_artifact_sha256(
                    warm_start_path,
                    easy_config.get("warm_start_model_sha256"),
                )
                training_loader = getattr(
                    agent_plugin, "load_for_training", None
                )
                if callable(training_loader):
                    model = training_loader(
                        str(warm_start_path), easy_env, easy_config
                    )
                else:
                    model = agent_plugin.load(str(warm_start_path), easy_env)
                try:
                    model.set_env(easy_env)
                except Exception:
                    pass
            else:
                model = agent_plugin.build(easy_env, easy_config)
            # AUD-F1-20260808-160: weight change is proved from the
            # canonical policy tensor digest, never from archive bytes.
            anchor_tensor_sha = _policy_tensor_hash(model.policy)
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
            minimum_trades = int(easy_config["easy_min_trades"])
            best = -math.inf
            best_epoch = None
            waited = 0
            history = []
            save_model = str(config.get("save_model")
                             or "./agent_model.zip")
            post_easy_path = Path(save_model).with_suffix("")
            post_easy_path = post_easy_path.parent / (
                post_easy_path.name + ".post_easy.zip")
            post_easy_path.parent.mkdir(parents=True, exist_ok=True)

            def evaluate_checkpoint(epoch: int, source: str) -> None:
                nonlocal best, best_epoch, waited
                probe = self._easy_probe(
                    env_plugin_name, easy_config, paths["train"],
                    agent_plugin, model, seed)
                probe["epoch"] = epoch
                probe["checkpoint_source"] = source
                probe["easy_activity_eligible"] = bool(
                    probe["trades_total"] >= minimum_trades
                    and probe["non_hold_actions"] > 0
                    and probe["entry_actions_seen"] > 0
                    and probe["entry_orders_submitted"] > 0
                    and probe["protected_entry_rejections"] == 0
                )
                normal_config = dict(config)
                normal_config["solvency_mode"] = NORMAL_MODE
                normal_train_tail = self._eval_on_split(
                    env_plugin_name,
                    normal_config,
                    paths.get("train_tail", paths["train"]),
                    agent_plugin,
                    model,
                    seed,
                    "train_tail_epoch",
                )
                normal_validation = self._eval_on_split(
                    env_plugin_name,
                    normal_config,
                    paths["val"],
                    agent_plugin,
                    model,
                    seed,
                    "validation_epoch",
                )
                normal_default_min = int(normal_config.get(
                    "early_stop_min_trades", 1
                ))
                configured_train_tail_min = normal_config.get(
                    "early_stop_min_train_tail_trades"
                )
                configured_validation_min = normal_config.get(
                    "early_stop_min_validation_trades"
                )
                normal_min_train_tail = int(
                    normal_default_min
                    if configured_train_tail_min is None
                    else configured_train_tail_min
                )
                normal_min_validation = int(
                    normal_default_min
                    if configured_validation_min is None
                    else configured_validation_min
                )
                normal_train_tail_trades = int(
                    normal_train_tail.get("trades_total", 0) or 0
                )
                normal_validation_trades = int(
                    normal_validation.get("trades_total", 0) or 0
                )
                probe["normal_handoff_probe"] = {
                    "train_tail_trades": normal_train_tail_trades,
                    "validation_trades": normal_validation_trades,
                    "minimum_train_tail_trades": normal_min_train_tail,
                    "minimum_validation_trades": normal_min_validation,
                }
                normal_handoff_eligible = bool(
                    normal_train_tail_trades >= normal_min_train_tail
                    and normal_validation_trades >= normal_min_validation
                )
                if handoff_semantics == HANDOFF_SEMANTICS_M0_V3:
                    # Mechanism-ladder D0 (finding 220): the boundary AS
                    # THE M0 SCREEN RAN IT — the normal probe GATES, the
                    # score is the easy-probe economic equity, and the
                    # epoch-0 warm-start baseline is save-eligible. This
                    # reproduces post_easy.v3 selection (observed:
                    # best_easy_epoch=0 on the active M0 arm).
                    probe["normal_handoff_probe_telemetry_only"] = False
                    probe["normal_handoff_eligible"] = (
                        normal_handoff_eligible)
                    probe["activity_eligible"] = bool(
                        probe["easy_activity_eligible"]
                        and normal_handoff_eligible)
                    probe["handoff_eligible"] = True
                    history.append(probe)
                    score = probe["economic_equity"]
                    if (
                        probe["activity_eligible"]
                        and math.isfinite(score)
                        and score > best + min_delta
                    ):
                        best = score
                        best_epoch = epoch
                        waited = 0
                        agent_plugin.save(model, str(post_easy_path))
                    elif best_epoch is not None:
                        waited += 1
                    return
                probe["normal_handoff_probe_telemetry_only"] = True
                # AUD-F1-20260808-159: the normal probe is TELEMETRY.
                # It never gates whether the easy treatment exists.
                probe["normal_handoff_eligible_telemetry"] = (
                    normal_handoff_eligible)
                # checkpoint A: easy utility on the monitor year;
                # checkpoint B: normal-realistic inner validation
                easy_monitor = self._eval_on_split(
                    env_plugin_name, easy_config,
                    paths.get("train_tail", paths["train"]),
                    agent_plugin, model, seed, "easy_monitor_epoch")
                paired = _paired.paired_generalization_weekly_v1(
                    easy_monitor, normal_validation,
                    beta=float(easy_config.get(
                        "l1_gap_penalty_beta", 0.25)),
                    label_a="easy_train_monitor",
                    label_b="normal_inner_validation",
                    candidate_id=f"easy_epoch_{epoch}")
                probe["paired_selection"] = {
                    "eligible": paired["eligible"],
                    "paired_score": paired["paired_score"],
                    "reasons": paired["ineligibility_reasons"],
                }
                probe["handoff_eligible"] = epoch > 0
                history.append(probe)
                # epoch 0 is baseline telemetry and is STRUCTURALLY
                # ineligible as a treatment handoff (order §7.5)
                if (
                    epoch > 0
                    and paired["eligible"]
                    and paired["paired_score"] is not None
                    and paired["paired_score"] > best + min_delta
                ):
                    best = paired["paired_score"]
                    best_epoch = epoch
                    waited = 0
                    agent_plugin.save(model, str(post_easy_path))
                elif best_epoch is not None:
                    waited += 1

            if warm_start_model:
                evaluate_checkpoint(0, "warm_start_baseline")

            for epoch in range(1, max_epochs + 1):
                model.learn(total_timesteps=epoch_ts,
                            reset_num_timesteps=False)
                evaluate_checkpoint(epoch, "easy_training_epoch")
                if best_epoch is not None and waited >= patience:
                    break

            trained_epochs = [row for row in history
                              if row.get("epoch", 0) > 0]
            if not trained_epochs:
                raise RuntimeError(
                    "phase 1 trained zero epochs; a declared phase-1 arm"
                    " cannot hand off (invariant 1)")
            selection_basis = "paired_comparator_best_trained_epoch"
            if best_epoch is None or not post_easy_path.exists():
                if handoff_semantics == HANDOFF_SEMANTICS_M0_V3:
                    # v3 as-run had NO terminal fallback: a phase 1 with
                    # no easy-and-normal activity-eligible checkpoint
                    # refused the handoff outright.
                    raise RuntimeError(
                        "easy curriculum produced no easy-and-normal "
                        "activity-eligible checkpoint under the as-run "
                        "M0 v3 handoff semantics: required trades>="
                        f"{minimum_trades}, non-hold actions, entry "
                        "actions, submitted entries, zero protected-"
                        "entry rejections in easy; train-tail and "
                        "validation trade gates must also pass in "
                        "normal")
                # invariant 3: a failed/inactive phase-1 result is still
                # handed off and measured — the TERMINAL trained epoch,
                # never the anchor.
                best_epoch = trained_epochs[-1]["epoch"]
                selection_basis = "terminal_trained_epoch_fallback"
                agent_plugin.save(model, str(post_easy_path))
            elif handoff_semantics == HANDOFF_SEMANTICS_M0_V3:
                selection_basis = "m0_v3_economic_equity_epoch0_eligible"
            updates_after = int(getattr(model, "_n_updates", 0) or 0)
            if updates_after <= 0:
                raise RuntimeError(
                    "phase 1 applied zero gradient updates; refusing a"
                    " sham handoff")
            terminal_tensor_sha = _policy_tensor_hash(model.policy)
            if terminal_tensor_sha == anchor_tensor_sha:
                raise RuntimeError(
                    "phase-1 final policy is tensor-identical to the"
                    " anchor; archive-byte difference is not weight"
                    " change (AUD-F1-20260808-160)")
            post_easy_sha = _verify_artifact_sha256(post_easy_path, None)
            meta = {
                "schema": "agent_multi.solvency_curriculum.post_easy.v4",
                "anchor_policy_tensor_sha256": anchor_tensor_sha,
                "phase1_terminal_policy_tensor_sha256": terminal_tensor_sha,
                "phase1_gradient_updates": updates_after,
                "selection_basis": selection_basis,
                "phase1_handoff_semantics": handoff_semantics,
                "epoch0_structurally_ineligible": (
                    handoff_semantics != HANDOFF_SEMANTICS_M0_V3),
                "normal_probe_is_telemetry_only": (
                    handoff_semantics != HANDOFF_SEMANTICS_M0_V3),
                "artifact": str(post_easy_path),
                "artifact_sha256": post_easy_sha,
                # Finding 195: the persisted mode is the mode that RAN.
                # Normal phase-1 records say normal; only easy says easy.
                "solvency_mode": (EASY_MODE
                                  if phase1_mode == EASY_MODE
                                  else NORMAL_MODE),
                "best_easy_epoch": best_epoch,
                # Finding 200: trained epochs are epoch > 0 ONLY; the
                # epoch-0 baseline evaluation is telemetry, counted
                # separately. The legacy alias carries the SAME truthful
                # trained count (aliases never lie).
                "phase1_epochs_run": sum(
                    1 for h in history if int(h.get("epoch", 0)) > 0),
                "phase1_baseline_evaluations": sum(
                    1 for h in history if int(h.get("epoch", 0)) == 0),
                "easy_epochs_run": sum(
                    1 for h in history if int(h.get("epoch", 0)) > 0),
                "easy_budget_epochs": max_epochs,
                "easy_epoch_timesteps": epoch_ts,
                "activity_contract": {
                    "minimum_trades": minimum_trades,
                    "requires_non_hold_action": True,
                    "requires_entry_action": True,
                    "requires_submitted_entry": True,
                    "maximum_protected_entry_rejections": 0,
                    # Under the corrected v4 boundary the normal-handoff
                    # probe is telemetry ONLY (finding 160/195); under
                    # the as-run M0 v3 reproduction it gates, and these
                    # facts say which one actually ran.
                    "normal_handoff_probe_is_telemetry_only": (
                        handoff_semantics != HANDOFF_SEMANTICS_M0_V3),
                    "normal_handoff_activity_gates_selection": (
                        handoff_semantics == HANDOFF_SEMANTICS_M0_V3),
                },
                "phase1_mode": phase1_mode,
                "phase1_difficulty": {
                    # easy-relaxation fields; None under normal phase-1
                    # (the candidate's own contract governs there)
                    "continuous_action_threshold": easy_config.get(
                        "continuous_action_threshold"),
                    "commission_fraction_per_side": easy_config.get(
                        "commission"),
                    "full_spread_rate": easy_config.get("full_spread_rate"),
                    "slippage_rate_per_side": easy_config.get("slippage"),
                },
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

        ledger = config.get("total_max_passes")
        if ledger is not None:
            total_max = int(ledger)
            phase1 = int(config.get("easy_max_epochs", 0))
            phase2 = int(config.get("max_epochs", 0))
            fraction = float(config.get("phase1_max_fraction", 0.5))
            normal_min = int(config.get("normal_phase_min_passes", 1))
            if phase1 + phase2 > total_max:
                raise ValueError(
                    f"two-phase budget exceeds total_max_passes:"
                    f" {phase1}+{phase2} > {total_max}")
            if phase1 > int(fraction * total_max):
                raise ValueError(
                    f"phase-1 budget {phase1} exceeds phase1_max_fraction"
                    f" {fraction} of {total_max}")
            if phase2 < normal_min:
                raise ValueError(
                    f"normal phase {phase2} below normal_phase_min_passes"
                    f" {normal_min}")
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
