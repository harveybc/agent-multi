"""
Meta-Training Data Logger for Level 3 Neural Network Training

Logs all candidate evaluations with 27 input parameters and performance metrics
to build a training dataset for the surrogate model (Level 3).
"""

import csv
import os
from pathlib import Path
from .meta_stages import get_all_meta_parameters


# Categorical mappings for Level 3 encoding
ACTIVATION_MAP = {
    "relu": 0,
    "elu": 1,
    "selu": 2,
    "tanh": 3,
    "sigmoid": 4,
    "swish": 5,
    "gelu": 6,
    "leaky_relu": 7
}


def _encode_categorical(param_name, value):
    """
    Encode categorical parameters as integers for Level 3 training.
    
    Args:
        param_name: Name of parameter
        value: Raw value from config
        
    Returns:
        Encoded integer value
    """
    if param_name == "activation":
        return ACTIVATION_MAP.get(value, 0)
    elif param_name == "use_log1p_features":
        # ["typical_price"] -> 1, None or [] -> 0
        if value and isinstance(value, list) and len(value) > 0:
            return 1
        return 0
    elif param_name == "predicted_horizons":
        # Encode as comma-separated string or hash for now
        # Level 3 may need special handling for variable-length lists
        if isinstance(value, list):
            return len(value)  # Simple: use count as feature
        return 1
    elif isinstance(value, bool):
        return 1 if value else 0
    else:
        return value


def initialize_meta_log(log_path, config, overwrite=False):
    """
    Initialize the meta_training_data.csv file with headers.
    
    Args:
        log_path: Path to CSV file
        config: Configuration dict with optimization_stages
        overwrite: If True, creates new file even if exists
        
    Returns:
        bool: True if file was created/exists
    """
    # Ensure directory exists
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and we're not overwriting
    if os.path.exists(log_path) and not overwrite:
        return True
    
    # Get all parameters from config stages (dynamic count)
    input_params = get_all_meta_parameters(config)
    
    # Define output metrics (targets for Level 3)
    output_metrics = [
        # Training metrics
        "train_mae",
        "train_rmse",
        "train_permutation_ok",
        "train_directional_accuracy",
        "train_naive_mae",
        # Validation metrics
        "val_mae",
        "val_rmse", 
        "val_permutation_ok",
        "val_directional_accuracy",
        "val_naive_mae",
        # Test metrics
        "test_mae",
        "test_rmse",
        "test_permutation_ok",
        "test_directional_accuracy",
        "test_naive_mae",
        # Metadata
        "stage",
        "generation",
        "candidate_id",
        "fitness"
    ]
    
    # Create CSV with headers
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(input_params + output_metrics)
    
    print(f"[META-LOG] Initialized meta-training log: {log_path}")
    print(f"[META-LOG] Inputs: {len(input_params)} parameters")
    print(f"[META-LOG] Outputs: {len(output_metrics)} metrics")
    return True


def log_candidate_evaluation(log_path, params_dict, metrics_dict, stage, generation, candidate_id, config):
    """
    Log a single candidate evaluation to the meta-training CSV.
    
    Args:
        log_path: Path to CSV file
        params_dict: Dictionary with all parameter values
        metrics_dict: Dictionary with training/val/test metrics
        stage: Current meta-optimization stage
        generation: Current generation number
        candidate_id: Unique candidate identifier
        config: Configuration dict with optimization_stages
    """
    # Get ordered parameter list from config
    input_params = get_all_meta_parameters(config)
    
    # Extract and encode input values
    input_values = []
    for param_name in input_params:
        raw_value = params_dict.get(param_name, 0)  # Default to 0 if missing
        encoded_value = _encode_categorical(param_name, raw_value)
        input_values.append(encoded_value)
    
    # Extract output metrics with safe defaults
    output_values = [
        # Training
        metrics_dict.get("train_mae", float("inf")),
        metrics_dict.get("train_rmse", float("inf")),
        metrics_dict.get("train_permutation_ok", 0),
        metrics_dict.get("train_directional_accuracy", 0.0),
        metrics_dict.get("train_naive_mae", float("inf")),
        # Validation
        metrics_dict.get("val_mae", float("inf")),
        metrics_dict.get("val_rmse", float("inf")),
        metrics_dict.get("val_permutation_ok", 0),
        metrics_dict.get("val_directional_accuracy", 0.0),
        metrics_dict.get("val_naive_mae", float("inf")),
        # Test
        metrics_dict.get("test_mae", float("inf")),
        metrics_dict.get("test_rmse", float("inf")),
        metrics_dict.get("test_permutation_ok", 0),
        metrics_dict.get("test_directional_accuracy", 0.0),
        metrics_dict.get("test_naive_mae", float("inf")),
        # Metadata
        stage,
        generation,
        candidate_id,
        metrics_dict.get("fitness", float("inf"))
    ]
    
    # Append to CSV
    try:
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(input_values + output_values)
    except Exception as e:
        print(f"[META-LOG] ERROR: Failed to write to log: {e}")


def get_meta_log_path(config):
    """Get meta-training log path from config."""
    return config.get("meta_training_log", "meta_training_data.csv")
