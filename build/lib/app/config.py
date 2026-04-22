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
    "load_config": None,
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
# config.py

DEFAULT_VALUES = {
    # --- env plugin selection ---
    "env_plugin": "gym_fx_env",

    # --- gym-fx env forwarded config ---
    "env_mode": "inference",
    "input_data_file": "examples/data/eurusd_sample.csv",
    "price_column": "CLOSE",
    "date_column": "DATE_TIME",
    "window_size": 32,
    "initial_cash": 10000.0,
    "position_size": 1.0,
    "commission": 0.0,
    "slippage": 0.0,
    "steps": 500,

    # --- legacy predictor keys (to be phased out) ---
    "use_normalization_json": "examples\\config\\phase_2_normalizer_debug_out.json",


    "x_train_file": "examples\\data\\phase_3\\phase_3_encoder_eval_d2.csv",
    #"x_train_file": "examples\\data\\phase_2\\normalized_d2.csv",
    #"x_train_file": "examples\\data\\phase_3\\extracted_features_transformer_va_d2.csv",
    "y_train_file": "examples\\data\\phase_2\\exp_4\\normalized_d2.csv",
    
    "x_validation_file": "examples\\data\\phase_3\\phase_3_encoder_eval_d3.csv",
    #"x_validation_file": "examples\\data\\phase_2\\normalized_d3.csv",
    #"x_validation_file": "examples\\data\\phase_3\\extracted_features_transformer_va_d2.csv",
    "y_validation_file": "examples\\data\\phase_2\\exp_4\\normalized_d3.csv",
    
    'target_column': 'typical_price',
    'output_file': './prediction.csv',
    'results_file': './results.csv',
    'save_model': './predictor_model.keras',
    'load_model': None,
    'loss_plot_file': './loss_plot.png',
    'model_plot_file': './model_plot.png',	
    'plugin': 'ann',
    'use_daily': False, # isntead of predicting the next time_horizon hours, predict the next time_horizon days.
    'threshold_error': 0.000000001,
    'remote_log': None,
    'remote_load_config': None,
    'remote_save_config': None,
    'username': None,
    'password': None,
    'load_config': None,
    'save_config': './config_out.json',
    'save_log': './debug_out.json',
    'quiet_mode': False,
    'force_date': False,
    'headers': True,
    'input_offset': 0,
    'window_size': 256,  # Number of time steps in each window (e.g., 24 for daily patterns)
    'l2_reg': 1e-4,          # L2 regularization factor
    'early_patience': 30,           # Early stopping patience
    'max_steps_train': 6300,
    'max_steps_val': 6300,
    'max_steps_test': 6300,
    'iterations': 1,
    'epochs': 1000,
    'uncertainty_file': 'prediction_uncertainties.csv',
    'batch_size': 32,
    # Safety: keep inference batched to avoid memory spikes.
    # (Used by BaseBayesianKerasPredictor.predict_with_uncertainty and the MC codepath.)
    'predict_batch_size': 256,
    # When True, BaseKerasPredictor.train skips post-fit MC uncertainty passes.
    # Optimizer forces this on; pipeline keeps it off by default.
    'disable_postfit_uncertainty': False,
    "kl_weight": 1e-6,
    "kl_anneal_epochs": 100,        
    "mmd_lambda": 0.1,
    "overfitting_penalty": 0.1,
    "use_returns": False,
    "use_log1p_targets": False,
    "mc_samples":100,
    "plotted_horizon": 6,
    "min_delta": 1e-4,
    "start_from_epoch": 10,
    "plot_color_predicted": "orange",
    "plot_color_true": "blue",
    "plot_color_uncertainty": "green",
    "uncertainty_color_alpha": 0.01,
    "min_delta": 1e-5,
    "plot_points": 240,
    "use_strategy": False,
    "strategy_plugin_group": "heuristic_strategy.plugins",
    "strategy_plugin_name": "ls_pred_strategy",
    "strategy_1h_prediction": "examples/results/phase_1/phase_1_cnn_25200_1h_prediction.csv",
    "strategy_1h_uncertainty": "examples/results/phase_1/phase_1_cnn_25200_1h_uncertanties.csv",
    "strategy_base_dataset": "examples/data/phase_1/phase_1_base_d3.csv",
    "strategy_load_parameters": "examples/data/phase_1/strategy_parameters.json",
    "target_scaling_factor":1000,
    "optimizer_output_file": "optimizer_output.json",
    "penalty_close_lambda":0.0001, # penalty in thel loss function for the predicted value being 0 (Naive)
    "penalty_far_lambda":0.0001,    # penalty in thel loss function for the predicted value being far from the target value in the opposite dicection of the 0 (Naive)
    "incentive_loss": 10.0,
    "use_log1p_targets": True
}
