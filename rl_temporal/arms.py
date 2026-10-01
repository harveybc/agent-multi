"""The four primary arms and their paired configs.

RL-S0  SAC  native baseline (NativeFlatExtractor: flattened window -> MLP)
RL-S1  SAC  modular temporal representation (ModularTemporalExtractor)
RL-D0  DQN  native baseline
RL-D1  DQN  same modular representation contract as RL-S1

Within each contrast (S1-S0, D1-D0) everything but the representation is
identical: selected features (from the bound manifest), window, environment,
action semantics, costs, initial balance, chronological episodes, reward,
evaluation objective and seeds. Across algorithms the ACTION SPACE DIFFERS and
is published: SAC acts in Box(-1, 1) thresholded by the env's tested contract
(three effective actions), DQN acts in Discrete(3). No discrete-SAC exists in
the installed stack (SB3 2.9.0), so no shared-discrete comparison is claimed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from . import ENGINE_PIN
from .checkpoint import action_mapping_from_config
from .lake_binding import SelectedFeatureBinding
from .modular_torch import ModularTemporalEncoder, modular_config_sha256, normalize_modular_config, parameter_accounting

ARMS: Dict[str, Dict[str, str]] = {
    "RL-S0": {"algorithm": "SAC", "representation": "native_flat", "agent_plugin": "sac_agent"},
    "RL-S1": {"algorithm": "SAC", "representation": "modular_temporal", "agent_plugin": "sac_agent"},
    "RL-D0": {"algorithm": "DQN", "representation": "native_flat", "agent_plugin": "dqn_agent"},
    "RL-D1": {"algorithm": "DQN", "representation": "modular_temporal", "agent_plugin": "dqn_agent"},
}
CONTRASTS = {"S1-S0": ("RL-S1", "RL-S0"), "D1-D0": ("RL-D1", "RL-D0")}

#: Frozen before any fit. Changing any of these is a new experiment, not a rerun.
FROZEN_REWARD = {"reward_plugin": "pnl_reward", "metrics_plugin": "trading_metrics"}
FROZEN_EVALUATION = {"selection_metric": "net_return", "selection_source": "validation_episode",
                     "risk_penalty_lambda": 1.0, "report": ["net_return", "max_drawdown_fraction",
                                                            "sharpe(per-bar, ddof=1, not annualized)",
                                                            "turnover_units", "trades_closed", "exposure_fraction"],
                     "baselines": ["no_trade", "heuristic:UNAVAILABLE unless naive_gate passed"]}

#: Shared environment contract for both contrasts (values the ETH 4h task will pin from its manifest).
SHARED_ENV = {
    "env_plugin": "gym_fx_env", "pipeline_plugin": "rl_pipeline_with_validation", "env_mode": "training",
    "data_feed_plugin": "default_data_feed", "broker_plugin": "default_broker",
    "strategy_plugin": "default_strategy", "preprocessor_plugin": "feature_window_preprocessor",
    "feature_scaling": "rolling_zscore", "feature_scaling_window": 256, "feature_clip": 10.0,
    "include_price_window": False, "include_agent_state": True, "agent_state_contract": "legacy_episode_v1",
    "initial_cash": 10000.0, "position_size": 1.0, "commission": 0.001, "slippage": 0.0,
    "min_equity": 100.0, "solvency_mode": "normal_realistic",
    "require_feature_aware_preprocessor": True,
}

#: Equal finite budgets within a comparison. Tuning budget is the same per algorithm family.
SHARED_TRAINING = {
    "epoch_timesteps": 2000, "max_epochs": 200, "l1_patience": 20, "l1_patience_start_epoch": 10,
    "l1_activity_patience": 20, "early_stop_min_trades": 1, "device": "cpu",
    "training_progress_file": "{out_dir}/heartbeat.json", "progress_update_interval_steps": 200,
    "heartbeat_interval_s": 30, "hard_limits": {"max_wall_s": 14400, "max_timesteps": 400000},
    "checkpoint_every_epochs": 5, "best_policy_restore": True,
    # a policy that never becomes activity-eligible is a MEASURED outcome (inactive), not a
    # harness failure: the pipeline saves the terminal weights and returns a typed result
    "inactive_terminal_is_typed_result": True,
}
ALGO = {
    "SAC": {"learning_rate": 3e-4, "buffer_size": 100000, "learning_starts": 2000, "batch_size": 256,
            "tau": 0.005, "gamma": 0.99, "train_freq": 1, "gradient_steps": 1, "ent_coef": "auto",
            "target_update_interval": 1, "use_sde": False, "net_arch": [64, 64],
            "action_space_mode": "continuous", "continuous_action_threshold": 0.33,
            "continuous_action_contract": "legacy_directional_v1"},
    "DQN": {"learning_rate": 1e-4, "buffer_size": 100000, "learning_starts": 2000, "batch_size": 128,
            "tau": 1.0, "gamma": 0.99, "train_freq": 4, "gradient_steps": 1, "target_update_interval": 1000,
            "exploration_fraction": 0.2, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05,
            "net_arch": [64, 64], "action_space_mode": "discrete"},
}
PAIRED_SEEDS: Sequence[int] = (101, 202, 303, 404)

#: Coordinator ruling 2026-10-01 (option c): replay storage change only, same transitions and
#: sampling. SB3 forbids optimize_memory_usage together with handle_timeout_termination; with
#: handle_timeout_termination False a TRUNCATED episode end would be bootstrapped as terminal,
#: but gym-fx never truncates (GymFxEnv.step always returns truncated=False; episodes end only
#: by data_end / min_equity termination), so episode-end bootstrapping is unchanged.
REPLAY_STORAGE = {"replay_buffer_kwargs": {"optimize_memory_usage": True, "handle_timeout_termination": False},
                  "replay_storage_declaration": {"change": "storage_only", "ruling": "coordinator 2026-10-01 option (c)",
                                                 "buffer_size": 100000, "learning_dynamics_change": "none",
                                                 "truncation_in_env": "never (gym-fx truncated=False)",
                                                 "bootstrapping_change": "none"}}
for _algo in ALGO.values():
    _algo.update(REPLAY_STORAGE)


def default_modular_config(feature_order: Sequence[str], *, window: int, sample_hours: float) -> Dict[str, Any]:
    """Hourly-default modular contract: one branch per feature, core [2,2,1] at window 24."""
    return normalize_modular_config({
        "schema": "predictor.modular.v1", "window": int(window), "sample_hours": sample_hours,
        "feature_names": list(feature_order),
        "branches": [{"name": f"branch_{i}", "features": [f]} for i, f in enumerate(feature_order)],
        "output_steps": 6, "output_channels": 8,
    })


def _versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {"engine_pin": ENGINE_PIN}
    for mod in ("stable_baselines3", "torch", "gymnasium", "backtrader"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "NOT_IMPORTABLE"
    return out


def accounting(arm: str, config: Dict[str, Any], observation_dim: int, extras_dim: int) -> Dict[str, Any]:
    """Parameter counts for the extractor and policy heads; compute recorded at run time."""
    algo = ARMS[arm]["algorithm"]
    net = list(config["net_arch"])
    action_dim = 1 if algo == "SAC" else 3
    if ARMS[arm]["representation"] == "modular_temporal":
        enc = ModularTemporalEncoder(config["representation"]["modular_config"])
        acc = parameter_accounting(enc)
        feat_dim = enc.latent_dim + extras_dim
        extractor_params = acc["total"]
    else:
        acc = {"note": "FlattenExtractor has no parameters"}
        feat_dim = observation_dim
        extractor_params = 0
    dims = [feat_dim, *net]
    mlp = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    if algo == "SAC":
        actor = mlp + (net[-1] * action_dim + action_dim) * 2  # mean + log_std
        critic = sum([(feat_dim + action_dim) * net[0] + net[0]] + [net[i] * net[i + 1] + net[i + 1] for i in range(len(net) - 1)]) + net[-1] + 1
        shared = bool(config["representation"].get("share_features_extractor", True))
        heads = {"actor": actor, "critics(2)": 2 * critic, "critic_targets(2)": 2 * critic,
                 "extractor_copies": 1 + (0 if shared else 1) + 1}
    else:
        q = mlp + net[-1] * action_dim + action_dim
        heads = {"q_net": q, "q_net_target": q, "extractor_copies": 2}
    return {"extractor": acc, "extractor_parameters": extractor_params, "policy_heads": heads,
            "features_dim": feat_dim, "observation_dim": observation_dim,
            "compute": {"state": "MEASURED_AT_RUN_TIME", "fields": ["wall_s", "gradient_updates", "transitions", "peak_rss_bytes"]},
            "pretraining_cost": {"state": "NONE" if config["representation"].get("regimes_summary", "R0") == "R0"
                                 else "CHARGED_FROM_DONOR_RECEIPT"}}


def build_arm_config(arm: str, binding: SelectedFeatureBinding, *, seed: int, out_dir: str,
                     window: int = 24, sample_hours: float = 1.0, regimes: Optional[Dict[str, str]] = None,
                     donor: Optional[str] = None, input_data_file: Optional[str] = None,
                     observation_extras_dim: int = 4) -> Dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}; known {sorted(ARMS)}")
    spec = ARMS[arm]
    algo = spec["algorithm"]
    cfg: Dict[str, Any] = {"arm": arm, "algorithm": algo, "agent_plugin": spec["agent_plugin"], "mode": "train"}
    cfg.update(copy.deepcopy(SHARED_ENV))
    cfg.update(copy.deepcopy(ALGO[algo]))
    cfg.update({k: (v.format(out_dir=out_dir) if isinstance(v, str) else copy.deepcopy(v))
                for k, v in SHARED_TRAINING.items()})
    cfg["window_size"] = int(window)
    cfg.update(binding.env_overrides())
    cfg["task"] = {**binding.task, "manifest_path": binding.manifest_path,
                   "manifest_sha256": binding.manifest_sha256, "manifest_status": binding.status,
                   "selected_variant": binding.selected_variant, "split": binding.split}
    cfg["input_data_file"] = input_data_file or (binding.resources.get("model_ready_view") or {}).get("path")
    cfg["train_seed"] = int(seed)
    cfg["eval_seed"] = int(seed)
    cfg["paired_seed_group"] = f"seed{seed}"
    cfg["save_model"] = f"{out_dir}/best_policy.zip"
    cfg["return_trace_dir"] = f"{out_dir}/traces"
    cfg["frozen_reward"] = copy.deepcopy(FROZEN_REWARD)
    cfg.update(FROZEN_REWARD)
    cfg["frozen_evaluation"] = copy.deepcopy(FROZEN_EVALUATION)
    cfg["selection_metric"] = FROZEN_EVALUATION["selection_metric"]
    cfg["action_mapping"] = action_mapping_from_config(cfg)
    cfg["action_space_note"] = ("SAC Box(-1,1) thresholded by the env (3 effective actions) vs DQN Discrete(3): "
                                "not an identical action space; no discrete-SAC is installed (SB3 2.9.0)")
    if spec["representation"] == "modular_temporal":
        mc = default_modular_config(binding.feature_order, window=window, sample_hours=sample_hours)
        resolved = {**{f"branch_{i}": "R0" for i in range(len(binding.feature_order))}, "core": "R0", **(regimes or {})}
        kinds = set(resolved.values())
        cfg["representation"] = {"kind": "modular_temporal", "modular_config": mc,
                                 "modular_config_sha256": modular_config_sha256(mc),
                                 "feature_order": list(binding.feature_order), "regimes": resolved,
                                 "regimes_summary": kinds.pop() if len(kinds) == 1 else "MIXED",
                                 "donor": donor, "share_features_extractor": True,
                                 "engine_pin": ENGINE_PIN}
        if any(r != "R0" for r in resolved.values()) and not donor:
            cfg["representation"]["status"] = "BLOCKED_NO_COMPATIBLE_DONOR"
        elif donor:
            cfg["representation"]["status"] = "DONOR_DECLARED_VERIFY_AT_BUILD"
        else:
            cfg["representation"]["status"] = "RUNNABLE_R0_RANDOM_INIT"
    else:
        cfg["representation"] = {"kind": "native_flat", "regimes_summary": "R0", "share_features_extractor": True,
                                 "architecture": "flatten_mlp", "time_handling": "flattened_window_no_temporal_structure",
                                 "status": "RUNNABLE", "engine_pin": None}
    obs_dim = int(window) * len(binding.feature_order) + int(observation_extras_dim)
    cfg["accounting"] = accounting(arm, cfg, obs_dim, int(observation_extras_dim))
    cfg["versions"] = _versions()
    cfg["pilot_gate"] = {"manifest_status": binding.status, "real_data_fit_allowed": binding.frozen,
                         "availability_class": binding.task.get("availability_class"),
                         "evidence_class": "DEVELOPMENT" if binding.status == "FROZEN_DEVELOPMENT" else
                                           ("GOVERNED" if binding.status == "FROZEN" else None),
                         "reason": None if binding.frozen else f"manifest is {binding.status}"}
    cfg.update(_split_dates(binding.split))
    cfg["evaluate_test_split"] = False            # test 2025 is protected: never selects, never reported here
    cfg["heartbeat_file"] = f"{out_dir}/heartbeat.json"
    cfg["results_file"] = f"{out_dir}/results.json"
    cfg["save_config"] = f"{out_dir}/config_out.json"
    cfg["quiet_mode"] = True
    cfg["observation_contract"] = {k: cfg[k] for k in ("require_feature_aware_preprocessor", "preprocessor_plugin",
                                                       "feature_scaling", "feature_scaling_window", "feature_clip",
                                                       "include_price_window", "include_agent_state", "window_size")}
    cfg["config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    return cfg


def _split_dates(split: Dict[str, Any]) -> Dict[str, Any]:
    """Explicit calendar split for the pipeline (half-open ranges), from the manifest split."""
    def _period(block):
        if isinstance(block, dict):
            return str(block.get("period") or "")
        return str(block or "")
    train, val, test = _period(split.get("train")), _period(split.get("validation")), _period(split.get("test"))
    if not (train and val and test):
        return {}
    t0, t1 = [x.strip() for x in train.split("..")]
    v0, v1 = [x.strip() for x in val.split("..")]
    s0, s1 = [x.strip() for x in test.split("..")]
    import pandas as pd
    end_of = lambda d: str(pd.Timestamp(d) + (pd.Timedelta(days=1) if len(d) == 10 else pd.Timedelta(hours=4)))
    return {"train_start": str(pd.Timestamp(t0)), "train_end": str(pd.Timestamp(v0)),
            "validation_start": str(pd.Timestamp(v0)), "validation_end": str(pd.Timestamp(s0)),
            "test_start": str(pd.Timestamp(s0)), "test_end": end_of(s1),
            "split_source": "manifest split (train end = validation start; validation end = test start)"}


def build_matrix(binding: SelectedFeatureBinding, *, out_root: str, seeds: Sequence[int] = PAIRED_SEEDS,
                 **kwargs: Any) -> List[Dict[str, Any]]:
    cells = []
    for seed in seeds:
        for arm in ARMS:
            cells.append(build_arm_config(arm, binding, seed=seed, out_dir=f"{out_root}/{arm}/seed{seed}", **kwargs))
    return cells


def check_pairing(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Within each contrast and seed, everything but the representation block must be equal."""
    ignore = {"arm", "representation", "accounting", "config_sha256", "save_model", "return_trace_dir",
              "training_progress_file", "heartbeat_file", "results_file", "save_config"}  # per-cell paths
    by = {(c["arm"], c["train_seed"]): c for c in cells}
    report = {}
    for name, (treated, control) in CONTRASTS.items():
        for seed in sorted({c["train_seed"] for c in cells}):
            a, b = by[(treated, seed)], by[(control, seed)]
            diff = sorted(k for k in set(a) | set(b) if k not in ignore and a.get(k) != b.get(k))
            report[f"{name}:seed{seed}"] = {"identical_outside_representation": not diff, "differences": diff}
    return report
