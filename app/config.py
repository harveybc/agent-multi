"""
config.py — agent-multi default values.

Single source of truth for RL training/inference/optimization on gym-fx.
CLI and --load_config JSON may override any of these.
"""
from __future__ import annotations

DEFAULT_VALUES = {
    # execution mode
    "mode": "inference",              # train | inference | optimization
    "use_optimizer": False,
    "quiet_mode": False,

    # plugin selection
    "env_plugin": "gym_fx_env",
    "agent_plugin": "ppo_agent",
    "pipeline_plugin": "rl_pipeline",
    "optimizer_plugin": "default_optimizer",

    # model I/O
    "save_model": "./agent_model.zip",
    "load_model": None,
    "results_file": "./results.json",
    "save_config": "./config_out.json",
    "resolved_config_file": "./resolved_config.json",
    "config_manifest_file": "./config_manifest.json",
    "load_config": None,
    "base_config": None,
    "candidate_patch": None,
    "runtime_overlay": None,
    "save_log": "./debug_out.json",
    "optimizer_output_file": "./optimizer_output.json",

    # gym-fx env forwarded config
    "env_mode": "inference",
    "input_data_file": "examples/data/eurusd_sample.csv",
    "price_column": "CLOSE",
    "date_column": "DATE_TIME",
    "headers": True,
    "max_rows": None,
    "window_size": 32,
    "initial_cash": 10000.0,
    "position_size": 1.0,
    "commission": 0.0,
    "slippage": 0.0,
    "data_feed_plugin": "default_data_feed",
    "broker_plugin": "default_broker",
    "strategy_plugin": "default_strategy",
    "preprocessor_plugin": "default_preprocessor",
    "reward_plugin": "pnl_reward",
    "metrics_plugin": "default_metrics",

    # agent training
    "total_timesteps": 10_000,
    "eval_episodes": 1,
    "eval_seed": 0,
    "train_seed": 0,
    "device": "auto",
    "agent_verbose": 0,

    # optimization (DEAP GA)
    "ga_population": 8,
    "ga_generations": 4,
    "ga_cxpb": 0.5,
    "ga_mutpb": 0.2,
    "ga_eval_timesteps": 2_000,
    "ga_n_jobs": 1,
    "ga_seed": 0,
    "ga_fitness_dd_lambda": 1.0,

    # remote config I/O (legacy, kept for parity)
    "remote_log": None,
    "remote_load_config": None,
    "remote_save_config": None,
    "username": None,
    "password": None,
}
