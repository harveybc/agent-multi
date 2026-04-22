#!/usr/bin/env python3
"""Generate fixed and optimization config files for all binary ioin plugins."""
import json
import os

SIGNAL_TYPES = ["buy_entry", "sell_entry", "buy_exit", "sell_exit"]

DATA_DIR = "examples/data_downsampled/phase_1_b"
CONFIG_DIR = "examples/config/phase_1b_binary"
OPT_DIR = os.path.join(CONFIG_DIR, "optimization")
RESULTS_DIR = "examples/results/phase_1b_binary"

# ── Common base shared by every binary config ────────────────────────────
COMMON_BASE = {
    "use_normalization_json": f"{DATA_DIR}/normalization_config_b.json",
    "x_train_file": f"{DATA_DIR}/normalized_d4.csv",
    "y_train_file": f"{DATA_DIR}/normalized_d4.csv",
    "x_validation_file": f"{DATA_DIR}/normalized_d5.csv",
    "y_validation_file": f"{DATA_DIR}/normalized_d5.csv",
    "x_test_file": f"{DATA_DIR}/normalized_d6.csv",
    "y_test_file": f"{DATA_DIR}/normalized_d6.csv",
    "target_plugin": "binary_target",
    "pipeline_plugin": "binary_pipeline",
    "preprocessor_plugin": "stl_preprocessor",
    "use_daily": False,
    "max_steps_train": 7300,
    "max_steps_val": 1575,
    "max_steps_test": 1575,
    "mc_samples": 10,
    "batch_size": 32,
    "plot_color_predicted": "red",
    "plot_color_true": "blue",
    "plot_color_uncertainty": "green",
    "plot_color_target": "orange",
    "uncertainty_color_alpha": 0.01,
    "plot_points": 1728,
    "min_delta": 1e-8,
    "epochs": 10,
    "predicted_horizons": [1],
    "plotted_horizon": 1,
    "use_strategy": False,
    "stl_period": 24,
    "use_stl": False,
    "use_wavelets": False,
    "use_multi_tapper": False,
    "use_predicted_decompositions": False,
    "use_real_decompositions": False,
    "use_ideal_predictions": False,
    "quiet_mode": True,
}

# ── Common optimizer fields (NEAT) shared by optimization configs ────────
COMMON_OPTIMIZER = {
    "optimizer_plugin": "neat_optimizer",
    "use_optimizer": True,
    "optimization_resume": False,
    "optimization_pause_on_resume": False,
    "neat_initial_params": None,
    "neat_add_param_prob": 0.35,
    "neat_remove_param_prob": 0.05,
    "neat_compatibility_threshold": 2.0,
    "neat_min_params": 6,
    "neat_survival_rate": 0.5,
    "neat_interspecies_mate_rate": 0.01,
    "neat_elitism": 1,
    "deterministic_training": True,
    "random_seed": 42,
    "population_size": 20,
    "n_generations": 40,
    "optimization_patience": 5,
    "mutpb": 0.2,
}

# ── Per-architecture definitions ─────────────────────────────────────────
ARCH_DEFS = {
    "binary_ann": {
        "fixed_params": {
            "hidden_units": 128,
            "num_hidden_layers": 2,
            "dropout_rate": 0.1,
            "activation": "elu",
            "learning_rate": 1e-3,
            "l2_reg": 1e-5,
            "window_size": 72,
            "early_patience": 80,
            "start_from_epoch": 15,
            "positional_encoding": True,
            "use_log1p_features": ["typical_price"],
        },
        "optimization_stages": [
            {"name": "features", "params": ["window_size", "use_log1p_features", "positional_encoding"], "generations": 10},
            {"name": "architecture", "params": ["hidden_units", "num_hidden_layers", "dropout_rate"], "generations": 10},
            {"name": "training", "params": ["learning_rate", "batch_size", "l2_reg", "min_delta"], "generations": 10},
            {"name": "refinement", "params": "all", "generations": 10},
        ],
        "hyperparameter_bounds": {
            "window_size": [48, 160],
            "hidden_units": [64, 256],
            "num_hidden_layers": [1, 4],
            "dropout_rate": [0.0, 0.3],
            "positional_encoding": [0, 1],
            "learning_rate": [1e-5, 1e-2],
            "batch_size": [16, 64],
            "l2_reg": [1e-7, 1e-3],
            "min_delta": [1e-10, 1e-6],
            "use_log1p_features": [0, 1],
        },
    },
    "binary_cnn": {
        "fixed_params": {
            "intermediate_layers": 4,
            "initial_layer_size": 78,
            "layer_size_divisor": 12,
            "head_layers": 1,
            "activation": "elu",
            "learning_rate": 1e-3,
            "l2_reg": 1e-5,
            "window_size": 72,
            "early_patience": 80,
            "start_from_epoch": 15,
            "positional_encoding": True,
            "use_log1p_features": ["typical_price"],
        },
        "optimization_stages": [
            {"name": "features", "params": ["window_size", "use_log1p_features", "positional_encoding"], "generations": 10},
            {"name": "architecture", "params": ["intermediate_layers", "initial_layer_size", "layer_size_divisor", "head_layers"], "generations": 10},
            {"name": "training", "params": ["learning_rate", "batch_size", "l2_reg", "min_delta"], "generations": 10},
            {"name": "refinement", "params": "all", "generations": 10},
        ],
        "hyperparameter_bounds": {
            "window_size": [48, 160],
            "intermediate_layers": [2, 6],
            "initial_layer_size": [32, 128],
            "layer_size_divisor": [2, 16],
            "head_layers": [1, 3],
            "positional_encoding": [0, 1],
            "learning_rate": [1e-5, 1e-2],
            "batch_size": [16, 64],
            "l2_reg": [1e-7, 1e-3],
            "min_delta": [1e-10, 1e-6],
            "use_log1p_features": [0, 1],
        },
    },
    "binary_lstm": {
        "fixed_params": {
            "intermediate_layers": 3,
            "initial_layer_size": 64,
            "layer_size_divisor": 8,
            "activation": "elu",
            "learning_rate": 1e-3,
            "l2_reg": 1e-5,
            "window_size": 72,
            "early_patience": 80,
            "start_from_epoch": 15,
            "use_log1p_features": ["typical_price"],
        },
        "optimization_stages": [
            {"name": "features", "params": ["window_size", "use_log1p_features"], "generations": 10},
            {"name": "architecture", "params": ["intermediate_layers", "initial_layer_size", "layer_size_divisor"], "generations": 10},
            {"name": "training", "params": ["learning_rate", "batch_size", "l2_reg", "min_delta"], "generations": 10},
            {"name": "refinement", "params": "all", "generations": 10},
        ],
        "hyperparameter_bounds": {
            "window_size": [48, 160],
            "intermediate_layers": [2, 6],
            "initial_layer_size": [32, 128],
            "layer_size_divisor": [2, 16],
            "learning_rate": [1e-5, 1e-2],
            "batch_size": [16, 64],
            "l2_reg": [1e-7, 1e-3],
            "min_delta": [1e-10, 1e-6],
            "use_log1p_features": [0, 1],
        },
    },
    "binary_transformer": {
        "fixed_params": {
            "merged_units": 48,
            "branch_units": 24,
            "num_attention_heads": 2,
            "activation": "elu",
            "learning_rate": 1e-3,
            "l2_reg": 1e-5,
            "window_size": 72,
            "early_patience": 80,
            "start_from_epoch": 15,
            "positional_encoding": True,
            "use_log1p_features": ["typical_price"],
        },
        "optimization_stages": [
            {"name": "features", "params": ["window_size", "use_log1p_features", "positional_encoding"], "generations": 10},
            {"name": "architecture", "params": ["merged_units", "branch_units", "num_attention_heads"], "generations": 10},
            {"name": "training", "params": ["learning_rate", "batch_size", "l2_reg", "min_delta"], "generations": 10},
            {"name": "refinement", "params": "all", "generations": 10},
        ],
        "hyperparameter_bounds": {
            "window_size": [48, 160],
            "merged_units": [32, 128],
            "branch_units": [16, 64],
            "num_attention_heads": [1, 8],
            "positional_encoding": [0, 1],
            "learning_rate": [1e-5, 1e-2],
            "batch_size": [16, 64],
            "l2_reg": [1e-7, 1e-3],
            "min_delta": [1e-10, 1e-6],
            "use_log1p_features": [0, 1],
        },
    },
    "binary_n_beats": {
        "fixed_params": {
            "nbeats_blocks": 3,
            "nbeats_layers": 4,
            "nbeats_units": 128,
            "dropout_rate": 0.1,
            "activation": "elu",
            "learning_rate": 1e-3,
            "l2_reg": 1e-6,
            "window_size": 72,
            "early_patience": 80,
            "start_from_epoch": 15,
            "positional_encoding": True,
            "use_log1p_features": ["typical_price"],
        },
        "optimization_stages": [
            {"name": "features", "params": ["window_size", "use_log1p_features", "positional_encoding"], "generations": 10},
            {"name": "architecture", "params": ["nbeats_blocks", "nbeats_layers", "nbeats_units", "dropout_rate"], "generations": 10},
            {"name": "training", "params": ["learning_rate", "batch_size", "l2_reg", "min_delta"], "generations": 10},
            {"name": "refinement", "params": "all", "generations": 10},
        ],
        "hyperparameter_bounds": {
            "window_size": [48, 160],
            "nbeats_blocks": [1, 5],
            "nbeats_layers": [2, 8],
            "nbeats_units": [32, 256],
            "dropout_rate": [0.0, 0.5],
            "positional_encoding": [0, 1],
            "learning_rate": [1e-5, 1e-2],
            "batch_size": [16, 64],
            "l2_reg": [1e-7, 1e-3],
            "min_delta": [1e-10, 1e-6],
            "use_log1p_features": [0, 1],
        },
    },
    "binary_tft": {
        "fixed_params": {
            "tft_hidden_units": 32,
            "tft_num_heads": 2,
            "tft_lstm_layers": 1,
            "tft_dropout": 0.15,
            "activation": "elu",
            "learning_rate": 1e-3,
            "l2_reg": 1e-6,
            "window_size": 72,
            "early_patience": 80,
            "start_from_epoch": 15,
            "positional_encoding": True,
            "use_log1p_features": ["typical_price"],
        },
        "optimization_stages": [
            {"name": "features", "params": ["window_size", "use_log1p_features", "positional_encoding"], "generations": 10},
            {"name": "architecture", "params": ["tft_hidden_units", "tft_lstm_layers", "tft_num_heads", "tft_dropout"], "generations": 10},
            {"name": "training", "params": ["learning_rate", "batch_size", "l2_reg", "min_delta"], "generations": 10},
            {"name": "refinement", "params": "all", "generations": 10},
        ],
        "hyperparameter_bounds": {
            "window_size": [48, 160],
            "tft_hidden_units": [16, 64],
            "tft_lstm_layers": [1, 3],
            "tft_num_heads": [1, 8],
            "tft_dropout": [0.0, 0.3],
            "positional_encoding": [0, 1],
            "learning_rate": [1e-5, 1e-2],
            "batch_size": [16, 64],
            "l2_reg": [1e-7, 1e-3],
            "min_delta": [1e-10, 1e-6],
            "use_log1p_features": [0, 1],
        },
    },
    "binary_tcn": {
        "fixed_params": {
            "tcn_filters": 32,
            "tcn_kernel_size": 3,
            "tcn_stack_layers": 2,
            "tcn_dilations_per_stack": 3,
            "tcn_dropout": 0.15,
            "tcn_use_batch_norm": False,
            "tcn_use_layer_norm": True,
            "tcn_head_layers": 1,
            "tcn_head_units": 32,
            "activation": "elu",
            "learning_rate": 1e-3,
            "l2_reg": 1e-6,
            "window_size": 72,
            "early_patience": 80,
            "start_from_epoch": 15,
            "positional_encoding": True,
            "use_log1p_features": ["typical_price"],
        },
        "optimization_stages": [
            {"name": "features", "params": ["window_size", "use_log1p_features", "positional_encoding"], "generations": 10},
            {"name": "architecture", "params": ["tcn_filters", "tcn_kernel_size", "tcn_stack_layers",
                                                  "tcn_dilations_per_stack", "tcn_head_layers", "tcn_head_units",
                                                  "tcn_use_batch_norm", "tcn_use_layer_norm"], "generations": 10},
            {"name": "training", "params": ["learning_rate", "batch_size", "tcn_dropout", "l2_reg", "min_delta"], "generations": 10},
            {"name": "refinement", "params": "all", "generations": 10},
        ],
        "hyperparameter_bounds": {
            "window_size": [48, 160],
            "tcn_filters": [16, 128],
            "tcn_kernel_size": [2, 7],
            "tcn_stack_layers": [1, 4],
            "tcn_dilations_per_stack": [2, 6],
            "tcn_head_layers": [1, 3],
            "tcn_head_units": [16, 64],
            "tcn_use_batch_norm": [0, 1],
            "tcn_use_layer_norm": [0, 1],
            "positional_encoding": [0, 1],
            "learning_rate": [1e-5, 1e-2],
            "batch_size": [16, 64],
            "tcn_dropout": [0.0, 0.3],
            "l2_reg": [1e-7, 1e-3],
            "min_delta": [1e-10, 1e-6],
            "use_log1p_features": [0, 1],
        },
    },
    "binary_mimo": {
        "fixed_params": {
            "encoder_conv_layers": 1,
            "encoder_base_filters": 57,
            "encoder_lstm_units": 10,
            "decoder_dropout": 0.03,
            "activation": "elu",
            "learning_rate": 1e-3,
            "l2_reg": 1e-5,
            "window_size": 79,
            "early_patience": 80,
            "start_from_epoch": 15,
            "positional_encoding": True,
            "use_log1p_features": ["typical_price"],
        },
        "optimization_stages": [
            {"name": "features", "params": ["window_size", "use_log1p_features", "positional_encoding"], "generations": 10},
            {"name": "architecture", "params": ["encoder_conv_layers", "encoder_base_filters", "encoder_lstm_units"], "generations": 10},
            {"name": "training", "params": ["learning_rate", "batch_size", "decoder_dropout", "l2_reg", "min_delta"], "generations": 10},
            {"name": "refinement", "params": "all", "generations": 10},
        ],
        "hyperparameter_bounds": {
            "window_size": [48, 160],
            "encoder_conv_layers": [1, 3],
            "encoder_base_filters": [16, 64],
            "encoder_lstm_units": [8, 32],
            "decoder_dropout": [0.0, 0.5],
            "positional_encoding": [0, 1],
            "learning_rate": [1e-5, 1e-2],
            "batch_size": [16, 64],
            "l2_reg": [1e-7, 1e-3],
            "min_delta": [1e-10, 1e-6],
            "use_log1p_features": [0, 1],
        },
    },
}


def make_fixed_config(arch, signal_type):
    """Build a fixed-parameter config dict."""
    tag = f"phase_1b_{arch}_{signal_type}_1d"
    cfg = {}
    # Data paths
    cfg.update(COMMON_BASE)
    # Output paths
    cfg["output_file"] = f"{RESULTS_DIR}/{tag}_prediction.csv"
    cfg["results_file"] = f"{RESULTS_DIR}/{tag}_results.csv"
    cfg["loss_plot_file"] = f"{RESULTS_DIR}/{tag}_loss_plot.png"
    cfg["model_plot_file"] = f"{RESULTS_DIR}/{tag}_model_plot.png"
    cfg["uncertainties_file"] = f"{RESULTS_DIR}/{tag}_uncertainties.csv"
    cfg["predictions_plot_file"] = f"{RESULTS_DIR}/{tag}_predictions_plot.png"
    # Plugin selection
    cfg["predictor_plugin"] = arch
    cfg["optimizer_plugin"] = "default_optimizer"
    cfg["plugin"] = arch
    cfg["signal_type"] = signal_type
    cfg["memory_log_file"] = f"{RESULTS_DIR}/{tag}_rss.csv"
    cfg["use_optimizer"] = False
    # Architecture-specific params
    cfg.update(ARCH_DEFS[arch]["fixed_params"])
    return cfg


def make_optimization_config(arch, signal_type):
    """Build a NEAT optimization config dict."""
    tag = f"phase_1b_{arch}_{signal_type}_1d"
    cfg = {}
    # Data paths
    cfg.update(COMMON_BASE)
    # Output paths
    cfg["output_file"] = f"{RESULTS_DIR}/{tag}_prediction.csv"
    cfg["results_file"] = f"{RESULTS_DIR}/{tag}_results.csv"
    cfg["loss_plot_file"] = f"{RESULTS_DIR}/{tag}_loss_plot.png"
    cfg["model_plot_file"] = f"{RESULTS_DIR}/{tag}_model_plot.png"
    cfg["uncertainties_file"] = f"{RESULTS_DIR}/{tag}_uncertainties.csv"
    cfg["predictions_plot_file"] = f"{RESULTS_DIR}/{tag}_predictions_plot.png"
    cfg["optimization_statistics"] = f"{RESULTS_DIR}/{tag}_optimization_stats.json"
    cfg["optimization_parameters_file"] = f"{RESULTS_DIR}/{tag}_optimization_parameters.json"
    # Plugin selection
    cfg["predictor_plugin"] = arch
    cfg["plugin"] = arch
    cfg["signal_type"] = signal_type
    cfg["memory_log_file"] = f"{RESULTS_DIR}/{tag}_rss.csv"
    cfg["optimization_resume_file"] = f"{RESULTS_DIR}/{tag}_optimization_resume.json"
    # NEAT optimizer params
    cfg.update(COMMON_OPTIMIZER)
    # Architecture-specific stages and bounds
    ad = ARCH_DEFS[arch]
    cfg["optimization_stages"] = ad["optimization_stages"]
    cfg["hyperparameter_bounds"] = ad["hyperparameter_bounds"]
    # Architecture default params
    cfg.update(ad["fixed_params"])
    return cfg


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(OPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fixed_count = 0
    opt_count = 0

    for arch in ARCH_DEFS:
        for st in SIGNAL_TYPES:
            # Fixed config
            fc = make_fixed_config(arch, st)
            fp = os.path.join(CONFIG_DIR, f"phase_1b_{arch}_{st}_1d_config.json")
            write_json(fp, fc)
            fixed_count += 1

            # Optimization config
            oc = make_optimization_config(arch, st)
            op = os.path.join(OPT_DIR, f"phase_1b_{arch}_{st}_1d_optimization_config.json")
            write_json(op, oc)
            opt_count += 1

    print(f"Created {fixed_count} fixed configs in {CONFIG_DIR}/")
    print(f"Created {opt_count} optimization configs in {OPT_DIR}/")
    print(f"Results directory: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
