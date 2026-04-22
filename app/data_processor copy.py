import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import json
import sys
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
import logging
from sklearn.metrics import r2_score  # Ensure sklearn is imported at the top
import contextlib
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import json
from plugin_loader import load_plugin

# Updated import: use tensorflow.keras instead of keras.
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import Huber

# --- CHUNK: create_sliding_windows_x (Replace your old version with this one) ---
def create_sliding_windows_x(data, window_size, stride=1, date_times=None):
    """
    Create sliding windows for input data only.

    Args:
        data (np.ndarray or pd.DataFrame): Input data array of shape (n_samples, n_features).
        window_size (int): The number of time steps in each window.
        stride (int): The stride between successive windows.
        date_times (pd.DatetimeIndex, optional): Corresponding date times for each sample.

    Returns:
        If date_times is provided:
            tuple: (windows, date_time_windows) where windows is an array of shape 
                   (n_windows, window_size, n_features) and date_time_windows is a list of 
                   the DATE_TIME value corresponding to the last time step (most current) of each window.
        Otherwise:
            np.ndarray: Array of sliding windows.
    """
    windows = []
    date_windows = []
    for i in range(window_size, len(data), stride):
        windows.append(data[i-window_size: i])
        if date_times is not None:
            # Use the date corresponding to the lastest, most current element in the window
            date_windows.append(date_times[i])
    return np.array(windows), date_windows


def create_multi_step(y_df, horizon, use_returns=False):
    """
    Creates multi-step targets for time-series prediction.
    For each base tick at index i, the target is taken from the next 'horizon' rows 
    (i.e. corresponding to t+1, t+2, ..., t+horizon). Assumes that data is in ascending order.
    
    Args:
        y_df (pd.DataFrame): Target data as a DataFrame.
        horizon (int): Number of future steps to predict.
        use_returns (bool): If True, returns are computed as (future – base).
    
    Returns:
        pd.DataFrame: Multi-step targets with shape (L-horizon, horizon).
        (if use_returns is True) pd.DataFrame: Baseline values corresponding to each target row.
    """
    blocks = []
    baselines = []
    L = len(y_df)
    for i in range(L - horizon -1):
        base = y_df.iloc[i].values.flatten()
        future_values = y_df.iloc[i+1: i+1+horizon].values.flatten()
        if use_returns:
            target = future_values - base  # (t+1 to t+horizon) - base value at t
        else:
            target = future_values
        blocks.append(target)
        if use_returns:
            baselines.append(base)
    df_targets = pd.DataFrame(blocks, index=y_df.index[:L - horizon-1])
    if use_returns:
        df_baselines = pd.DataFrame(baselines, index=y_df.index[:L - horizon-1])
        return df_targets, df_baselines
    else:
        return df_targets


def create_multi_step_daily(y_df, horizon, use_returns=False):
    """
    Creates multi-step targets for time-series prediction using daily data.
    If use_returns is True, targets are computed as the difference between each future value 
    and the current (baseline) value.
    
    Args:
        y_df (pd.DataFrame): Target data as a DataFrame.
        horizon (int): Number of future days to predict.
        use_returns (bool): If True, compute returns instead of absolute values.
    
    Returns:
        pd.DataFrame: Multi-step targets aligned with the input data.
        (if use_returns is True) pd.DataFrame: Baseline values corresponding to each target row.
    """
    blocks = []
    baselines = []
    for i in range(len(y_df) - horizon * 24):
        base = y_df.iloc[i].values.flatten()
        window = []
        for d in range(1, horizon + 1):
            future_values = y_df.iloc[i + d * 24].values.flatten()
            if use_returns:
                #window.extend(list(val - base))
                target = future_values - base  # (t+1 to t+horizon) - base value at t
            else:
                target = future_values
            window.append(target)    
        blocks.append(window)
        if use_returns:
            baselines.append(base)
    df_targets = pd.DataFrame(blocks, index=y_df.index[:-horizon * 24])
    if use_returns:
        df_baselines = pd.DataFrame(baselines, index=y_df.index[:-horizon * 24])
        return df_targets, df_baselines
    else:
        return df_targets


def process_data(config):
    """
    Processes data for different plugins, including ANN, CNN, LSTM, and Transformer.
    Loads and processes training, validation, and test datasets; extracts DATE_TIME information,
    and trims each pair (x and y) to their common date range so that they share the same number of rows.
    
    Returns:
        dict: Processed datasets for training, validation, and test, along with corresponding
              DATE_TIME arrays. Additionally, if config['use_returns'] is True, the corresponding 
              baseline target values (the original CLOSE values) are also returned.
    """
    import pandas as pd
    # 1) LOAD CSVs for train, validation, and test
    x_train = load_csv(
        config["x_train_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_train")
    )
    y_train = load_csv(
        config["y_train_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_train")
    )
    x_val = load_csv(
        config["x_validation_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_val")
    )
    y_val = load_csv(
        config["y_validation_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_val")
    )
    x_test = load_csv(
        config["x_test_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_test")
    )
    y_test = load_csv(
        config["y_test_file"],
        headers=config["headers"],
        max_rows=config.get("max_steps_test")
    )
    
    # 1a) Trim to common date range if possible.
    if isinstance(x_train.index, pd.DatetimeIndex) and isinstance(y_train.index, pd.DatetimeIndex):
        common_train_index = x_train.index.intersection(y_train.index)
        x_train = x_train.loc[common_train_index]
        y_train = y_train.loc[common_train_index]
    if isinstance(x_val.index, pd.DatetimeIndex) and isinstance(y_val.index, pd.DatetimeIndex):
        common_val_index = x_val.index.intersection(y_val.index)
        x_val = x_val.loc[common_val_index]
        y_val = y_val.loc[common_val_index]
    if isinstance(x_test.index, pd.DatetimeIndex) and isinstance(y_test.index, pd.DatetimeIndex):
        common_test_index = x_test.index.intersection(y_test.index)
        x_test = x_test.loc[common_test_index]
        y_test = y_test.loc[common_test_index]
    
    # Save original DATE_TIME indices AFTER trimming.
    train_dates_orig = x_train.index if isinstance(x_train.index, pd.DatetimeIndex) else None
    val_dates_orig = x_val.index if isinstance(x_val.index, pd.DatetimeIndex) else None
    test_dates_orig = x_test.index if isinstance(x_test.index, pd.DatetimeIndex) else None


    # 2) EXTRACT THE TARGET COLUMN
    target_col = config["target_column"]
    def extract_target(df, col):
        if isinstance(col, str):
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found.")
            return df[[col]]
        elif isinstance(col, int):
            return df.iloc[:, [col]]
        else:
            raise ValueError("`target_column` must be str or int.")
    y_train = extract_target(y_train, target_col)
    y_val = extract_target(y_val, target_col)
    y_test = extract_target(y_test, target_col)
    test_close_prices = y_test.copy()  # Save for later use
    if isinstance(test_close_prices, pd.DataFrame):
        test_close_prices = test_close_prices.to_numpy()
        

    # === STEP 3: CONVERT EACH DF TO NUMERIC (keep DataFrames) ===
    # === STEP 3: CONVERT EACH DF TO NUMERIC (keep DataFrames) ===
    x_train = x_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = y_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_val   = x_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_val   = y_val.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_test  = x_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test  = y_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    

    # === DERIVE CLEAN, LEAKAGE-FREE FEATURES (KEEP DATAFRAMES) ===
    # Compute previous CLOSE for overnight gap calculation
    x_train['Prev_CLOSE'] = x_train['CLOSE'].shift(1)
    x_val['Prev_CLOSE']   = x_val['CLOSE'].shift(1)
    x_test['Prev_CLOSE']  = x_test['CLOSE'].shift(1)

    # Calculate Overnight Gap and High-Low Range normalized by OPEN
    x_train['Overnight_Gap'] = (x_train['OPEN'] - x_train['Prev_CLOSE']) / x_train['Prev_CLOSE']
    x_val['Overnight_Gap']   = (x_val['OPEN'] - x_val['Prev_CLOSE']) / x_val['Prev_CLOSE']
    x_test['Overnight_Gap']  = (x_test['OPEN'] - x_test['Prev_CLOSE']) / x_test['Prev_CLOSE']

    x_train['HL_Range'] = (x_train['HIGH'] - x_train['LOW']) / x_train['OPEN']
    x_val['HL_Range']   = (x_val['HIGH'] - x_val['LOW']) / x_val['OPEN']
    x_test['HL_Range']  = (x_test['HIGH'] - x_test['LOW']) / x_test['OPEN']

    # Fill NaNs that may result from shifting operations
    x_train.fillna(0, inplace=True)
    x_val.fillna(0, inplace=True)
    x_test.fillna(0, inplace=True)

    # --- NEW: Add normalized BC-BO feature ---
    norm_json = config.get("use_normalization_json")
    if isinstance(norm_json, str):
        with open(norm_json, 'r') as f:
            norm_json = json.load(f)
    if "BC-BO" in norm_json:
        bcbo_min = norm_json["BC-BO"]["min"]
        bcbo_max = norm_json["BC-BO"]["max"]
        x_train['Norm_BC_BO'] = 2 * (x_train['BC-BO'] - bcbo_min) / (bcbo_max - bcbo_min) - 1
        x_val['Norm_BC_BO'] = 2 * (x_val['BC-BO'] - bcbo_min) / (bcbo_max - bcbo_min) - 1
        x_test['Norm_BC_BO'] = 2 * (x_test['BC-BO'] - bcbo_min) / (bcbo_max - bcbo_min) - 1
        print("DEBUG: Normalized BC-BO feature added as 'Norm_BC_BO'.")
    else:
        print("Warning: 'BC-BO' normalization parameters not found; BC-BO feature will not be normalized.")

    # Drop raw absolute price columns and leakage columns, but keep the new normalized BC-BO
    cols_to_drop = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'Prev_CLOSE', 'VOLUME', 'BC-BO']
    x_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    x_val.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    x_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print("DEBUG: Dropped raw price columns and leakage columns; normalized BC-BO is preserved.")



    # === NOW (AFTER FEATURE ENGINEERING) CONVERT TO NUMPY ===
    x_train = x_train.to_numpy().astype(np.float32)
    x_val = x_val.to_numpy().astype(np.float32)
    x_test = x_test.to_numpy().astype(np.float32)



    # 4) MULTI-STEP TARGETS
    time_horizon = config["time_horizon"]
    if config.get("use_daily", False):
        y_train_ma = y_train.rolling(window=3, center=True, min_periods=1).mean()
        y_val_ma = y_val.rolling(window=3, center=True, min_periods=1).mean()
        y_test_ma = y_test.rolling(window=3, center=True, min_periods=1).mean() 
        if config.get("use_returns", False):
            y_train_multi, baseline_train = create_multi_step_daily(y_train_ma, time_horizon, use_returns=True)
            y_val_multi, baseline_val = create_multi_step_daily(y_val_ma, time_horizon, use_returns=True)
            y_test_multi, baseline_test = create_multi_step_daily(y_test_ma, time_horizon, use_returns=True)
        else:
            y_train_multi = create_multi_step_daily(y_train_ma, time_horizon, use_returns=False)
            y_val_multi = create_multi_step_daily(y_val_ma, time_horizon, use_returns=False)
            y_test_multi = create_multi_step_daily(y_test_ma, time_horizon, use_returns=False)
    else:
        if config.get("use_returns", False):
            y_train_multi, baseline_train = create_multi_step(y_train, time_horizon, use_returns=True)
            y_val_multi, baseline_val = create_multi_step(y_val, time_horizon, use_returns=True)
            y_test_multi, baseline_test = create_multi_step(y_test, time_horizon, use_returns=True)
        else:
            y_train_multi = create_multi_step(y_train, time_horizon, use_returns=False)
            y_val_multi = create_multi_step(y_val, time_horizon, use_returns=False)
            y_test_multi = create_multi_step(y_test, time_horizon, use_returns=False)


    # === STEP 4.1: Scale multi-step target returns (if using returns) ===
    if config.get("use_returns", False):
        scale_factor = config.get("target_scaling_factor", 100.0)  # default scaling factor of 100
        print(f"DEBUG: Scaling multi-step target returns by factor {scale_factor}.")
        # Multiply every element in the target DataFrames by the scaling factor.
        y_train_multi = y_train_multi.applymap(lambda v: v * scale_factor)
        y_val_multi = y_val_multi.applymap(lambda v: v * scale_factor)
        y_test_multi = y_test_multi.applymap(lambda v: v * scale_factor)


        
    # 5) PER-PLUGIN PROCESSING
    # Use sliding windows only if explicitly enabled by config['use_sliding_windows'] or if the plugin is "lstm".
    if config["plugin"] in ["lstm", "cnn", "transformer","ann"]:
        print("Processing data with sliding windows...")

        window_size = config["window_size"]

        # Convert x_* to numpy arrays only if they are not already
        if not isinstance(x_train, np.ndarray):
            x_train_np = x_train.to_numpy().astype(np.float32)
        else:
            x_train_np = x_train.astype(np.float32)

        if not isinstance(x_val, np.ndarray):
            x_val_np = x_val.to_numpy().astype(np.float32)
        else:
            x_val_np = x_val.astype(np.float32)

        if not isinstance(x_test, np.ndarray):
            x_test_np = x_test.to_numpy().astype(np.float32)
        else:
            x_test_np = x_test.astype(np.float32)

        # Create sliding windows and get aligned dates
        x_train, train_dates = create_sliding_windows_x(x_train_np, window_size, stride=1, date_times=train_dates_orig)
        x_val, val_dates = create_sliding_windows_x(x_val_np, window_size, stride=1, date_times=val_dates_orig)
        x_test, test_dates = create_sliding_windows_x(x_test_np, window_size, stride=1, date_times=test_dates_orig)

        # Adjust multi-step targets accordingly
        y_train_multi = y_train_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
        y_val_multi = y_val_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)
        y_test_multi = y_test_multi.iloc[window_size - 1:].to_numpy().astype(np.float32)

        if config.get("use_returns", False):
            print("Processing data with sliding windows with returns...")
            baseline_train = baseline_train.iloc[window_size - 1:].to_numpy().astype(np.float32)
            baseline_val = baseline_val.iloc[window_size - 1:].to_numpy().astype(np.float32)
            baseline_test = baseline_test.iloc[window_size - 1:].to_numpy().astype(np.float32)
        # Trim the original date indices to match sliding windows
        if train_dates_orig is not None:
            train_dates_orig = train_dates_orig[window_size - 1:]
        else:
            train_dates = None
        if val_dates_orig is not None:
            val_dates_orig = val_dates_orig[window_size - 1:]
        else:
            val_dates = None
        if test_dates_orig is not None:
            test_dates_orig = test_dates_orig[window_size - 1:]
        else:
            test_dates = None

        # Ensure test_close_prices array alignment
        min_len_test = min(len(x_test), len(y_test_multi))
        test_close_prices = test_close_prices[window_size - 1 : window_size - 1 + min_len_test, -1]
    else:
        print("Not using sliding windows; converting data to NumPy arrays without windowing.")
        # Convert x_train, x_val, x_test to numpy arrays if they are not already
        if not isinstance(x_train, np.ndarray):
            x_train = x_train.to_numpy().astype(np.float32)
        else:
            x_train = x_train.astype(np.float32)

        if not isinstance(x_val, np.ndarray):
            x_val = x_val.to_numpy().astype(np.float32)
        else:
            x_val = x_val.astype(np.float32)

        if not isinstance(x_test, np.ndarray):
            x_test = x_test.to_numpy().astype(np.float32)
        else:
            x_test = x_test.astype(np.float32)

        y_train_multi = y_train_multi.to_numpy().astype(np.float32)
        y_val_multi = y_val_multi.to_numpy().astype(np.float32)
        y_test_multi = y_test_multi.to_numpy().astype(np.float32)
        if config.get("use_returns", False):
            baseline_train = baseline_train.to_numpy().astype(np.float32)
            baseline_val = baseline_val.to_numpy().astype(np.float32)
            baseline_test = baseline_test.to_numpy().astype(np.float32)
        train_dates = train_dates_orig
        val_dates = val_dates_orig
        test_dates = test_dates_orig
        min_len_test = min(len(x_test), len(y_test_multi))
        test_close_prices = test_close_prices[:min_len_test, -1]





    # 6) TRIM x TO MATCH THE LENGTH OF y (for each dataset)
    min_len_train = min(len(x_train), len(y_train_multi))
    x_train = x_train[:min_len_train]
    y_train_multi = y_train_multi[:min_len_train]
    if config.get("use_returns", False):
        baseline_train = baseline_train[:min_len_train]
    min_len_val = min(len(x_val), len(y_val_multi))
    x_val = x_val[:min_len_val]
    y_val_multi = y_val_multi[:min_len_val]
    if config.get("use_returns", False):
        baseline_val = baseline_val[:min_len_val]
    min_len_test = min(len(x_test), len(y_test_multi))
    x_test = x_test[:min_len_test]
    y_test_multi = y_test_multi[:min_len_test]
    if config.get("use_returns", False):
        baseline_test = baseline_test[:min_len_test]
    # trim also the dates of the datasets
    train_dates_orig = train_dates_orig[:min_len_train] if train_dates_orig is not None else None
    val_dates_orig = val_dates_orig[:min_len_val] if val_dates_orig is not None else None
    test_dates_orig = test_dates_orig[:min_len_test] if test_dates_orig is not None else None

    train_dates = train_dates_orig if train_dates_orig is not None else None
    val_dates = val_dates_orig if val_dates_orig is not None else None
    test_dates = test_dates_orig if test_dates_orig is not None else None
    test_close_prices = test_close_prices[:min_len_test]


    print("Processed datasets:")
    print(" x_train:", x_train.shape, " y_train:", y_train_multi.shape)
    print(" x_val:  ", x_val.shape, " y_val:  ", y_val_multi.shape)
    print(" x_test: ", x_test.shape, " y_test: ", y_test_multi.shape)
    print(" test_close_prices: ", test_close_prices.shape)
    
    

    # --- NEW CODE to convert targets to list of arrays ---
    # Assuming y_train_multi, y_val_multi, y_test_multi have shape (samples, time_horizon)
    y_train_multi_list = [y_train_multi[:, i] for i in range(y_train_multi.shape[1])]
    y_val_multi_list   = [y_val_multi[:, i] for i in range(y_val_multi.shape[1])]
    y_test_multi_list  = [y_test_multi[:, i] for i in range(y_test_multi.shape[1])]


    
    # Update the return dictionary to use the lists instead of the single 2D arrays:
    ret = {
        "x_train": x_train,
        "y_train": y_train_multi_list,
        "x_val": x_val,
        "y_val": y_val_multi_list,
        "x_test": x_test,
        "y_test": y_test_multi_list,
        "dates_train": train_dates,
        "dates_val": val_dates,
        "dates_test": test_dates,
        'test_close_prices': test_close_prices
    }
    # --- END NEW CODE ---


    # --- NEW CODE: Also stack multi-output targets into 2D arrays for evaluation ---
    y_train_array = np.stack(y_train_multi_list, axis=1)  # shape: (n_samples, time_horizon)
    y_val_array   = np.stack(y_val_multi_list, axis=1)
    y_test_array  = np.stack(y_test_multi_list, axis=1)
    ret["y_train_array"] = y_train_array
    ret["y_val_array"] = y_val_array
    ret["y_test_array"] = y_test_array
    print("DEBUG: Stacked y_train shape:", y_train_array.shape)
    print("DEBUG: Stacked y_val shape:", y_val_array.shape)
    print("DEBUG: Stacked y_test shape:", y_test_array.shape)
    # --- END NEW CODE ---


    if config.get("use_returns", False):
        ret["baseline_train"] = baseline_train
        ret["baseline_val"] = baseline_val
        ret["baseline_test"] = baseline_test
    return ret


def run_prediction_pipeline(config, plugin):
    """
    Runs the prediction pipeline using training, validation, and test datasets.
    Trains the model (with 5-fold cross-validation), saves metrics, predictions,
    uncertainty estimates, and plots. Predictions (and uncertainties) are denormalized;
    when use_returns is True, predicted returns are converted to close prices by adding
    the corresponding denormalized baseline close. In the plot, only the prediction at
    the horizon given by config['plotted_horizon'] (default=6) is shown.
    """
    import time, numpy as np, pandas as pd, json, matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from sklearn.model_selection import TimeSeriesSplit

    start_time = time.time()
    iterations = config.get("iterations", 1)
    print(f"Number of iterations: {iterations}")

    # Lists for metrics
    training_mae_list, training_r2_list, training_unc_list, training_snr_list, training_profit_list, training_risk_list = [], [], [], [], [], []
    validation_mae_list, validation_r2_list, validation_unc_list, validation_snr_list, validation_profit_list, validation_risk_list = [], [], [], [], [], []
    test_mae_list, test_r2_list, test_unc_list, test_snr_list, test_profit_list, test_risk_list = [], [], [], [], [], []



    print("Loading and processing datasets...")
    datasets = process_data(config)
    x_train, y_train = datasets["x_train"], datasets["y_train"]
    x_val, y_val = datasets["x_val"], datasets["y_val"]
    x_test, y_test = datasets["x_test"], datasets["y_test"]
    # --- NEW CODE: Retrieve stacked multi-output targets ---
    y_train_array = datasets["y_train_array"]  # shape: (n_samples, time_horizon)
    y_val_array   = datasets["y_val_array"]
    y_test_array  = datasets["y_test_array"]
    print("DEBUG: Retrieved stacked y_train shape:", y_train_array.shape)
    print("DEBUG: Retrieved stacked y_val shape:", y_val_array.shape)
    print("DEBUG: Retrieved stacked y_test shape:", y_test_array.shape)
    # --- END NEW CODE ---

    train_dates = datasets.get("dates_train")
    val_dates = datasets.get("dates_val")
    test_dates = datasets.get("dates_test")
    test_close_prices = datasets.get("test_close_prices")
    # When using returns, process_data returns baseline values
    if config.get("use_returns", False):
        baseline_train = datasets.get("baseline_train")
        baseline_val = datasets.get("baseline_val")
        baseline_test = datasets.get("baseline_test")

    # If sliding windows output is a tuple, extract the data.
    if isinstance(x_train, tuple): x_train = x_train[0]
    if isinstance(x_val, tuple): x_val = x_val[0]
    if isinstance(x_test, tuple): x_test = x_test[0]

    print(f"Training data shapes: x_train: {x_train.shape}, y_train: {[a.shape for a in y_train]}")
    print(f"Validation data shapes: x_val: {x_val.shape}, y_val: {[a.shape for a in y_val]}")
    print(f"Test data shapes: x_test: {x_test.shape}, y_test: {[a.shape for a in y_test]}")
    # --- NEW CODE: Stack multi-output target lists into 2D arrays ---
    y_train_array = np.stack(y_train, axis=1)  # Shape: (n_samples, time_horizon)
    y_val_array   = np.stack(y_val, axis=1)
    y_test_array  = np.stack(y_test, axis=1)
    print("DEBUG: Stacked y_train shape:", y_train_array.shape)
    print("DEBUG: Stacked y_val shape:", y_val_array.shape)
    print("DEBUG: Stacked y_test shape:", y_test_array.shape)
    # --- END NEW CODE ---

    # --- NEW CODE: Stack multi-output target lists into arrays ---
    y_train_stacked = np.stack(y_train, axis=1)  # shape: (samples, time_horizon)
    y_val_stacked   = np.stack(y_val, axis=1)
    y_test_stacked  = np.stack(y_test, axis=1)
    print("DEBUG: Stacked y_train shape:", y_train_stacked.shape)
    print("DEBUG: Stacked y_val shape:", y_val_stacked.shape)
    print("DEBUG: Stacked y_test shape:", y_test_stacked.shape)
    # --- END NEW CODE ---
    # --- CHUNK: Training iterations ---
    time_horizon = config.get("time_horizon")
    window_size = config.get("window_size")
    if time_horizon is None:
        raise ValueError("`time_horizon` is not defined in the configuration.")
    if config["plugin"] in ["lstm", "cnn", "transformer","ann"] and window_size is None:
        raise ValueError("`window_size` must be defined for CNN, Transformer and LSTM plugins.")
    print(f"Time Horizon: {time_horizon}")
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    threshold_error = config["threshold_error"]

    # Ensure variables are NumPy arrays.
    for var in ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]:
        arr = locals()[var]
        if isinstance(arr, pd.DataFrame):
            locals()[var] = arr.to_numpy().astype(np.float32)

    if config["plugin"] in ["lstm", "cnn", "transformer","ann"]:
        if x_train.ndim != 3:
            raise ValueError(f"For CNN and LSTM, x_train must be 3D. Found: {x_train.shape}")
        print("Using pre-processed sliding windows for CNN and LSTM.")
    plugin.set_params(time_horizon=time_horizon)
    
    # Training iterations
    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")
        iter_start = time.time()
        if config["plugin"] in ["lstm", "cnn", "transformer","ann"]:
            plugin.build_model(input_shape=(window_size, x_train.shape[2]), x_train=x_train, config=config)
        elif config["plugin"] in ["transformer", "transformer_mmd"]:
            plugin.build_model(input_shape=x_train.shape[1], x_train=x_train, config=config)
        else:
            if len(x_train.shape) != 2:
                raise ValueError(f"Expected 2D x_train for {config['plugin']}; got {x_train.shape}")
            plugin.build_model(input_shape=x_train.shape[1], x_train=x_train, config=config)

        history,  train_preds, train_unc, val_preds, val_unc = plugin.train(
            x_train, y_train, epochs=epochs, batch_size=batch_size,
            threshold_error=threshold_error, x_val=x_val, y_val=y_val, config=config
        )

        # === STEP 7: Inverse-scale the predictions if using returns ===
        if config.get("use_returns", False):
            inv_scale_factor = 1.0 / config.get("target_scaling_factor", 100.0)
            print(f"DEBUG: Inversely scaling predictions by factor {inv_scale_factor}.")
            # Multiply the prediction arrays by the inverse scaling factor.
            train_preds = train_preds * inv_scale_factor
            val_preds = val_preds * inv_scale_factor
            test_predictions = test_predictions * inv_scale_factor


        # If using returns, recalc r2 based on baseline + predictions.
        # Ensure predictions arrays are correctly squeezed to match the shape of stacked ground truth
        train_preds_squeezed = np.squeeze(train_preds, axis=-1)  # from (samples, horizons, 1) to (samples, horizons)
        val_preds_squeezed = np.squeeze(val_preds, axis=-1)

        if config.get("use_returns", False):
            train_r2 = r2_score(
                (baseline_train[:, -1] + y_train_stacked[:, -1]).flatten(),
                (baseline_train[:, -1] + train_preds_squeezed[:, -1]).flatten()
            )
            val_r2 = r2_score(
                (baseline_val[:, -1] + y_val_stacked[:, -1]).flatten(),
                (baseline_val[:, -1] + val_preds_squeezed[:, -1]).flatten()
            )
        else:
            train_r2 = r2_score(
                y_train_stacked[:, -1].flatten(), 
                train_preds_squeezed[:, -1].flatten()
            )
            val_r2 = r2_score(
                y_val_stacked[:, -1].flatten(), 
                val_preds_squeezed[:, -1].flatten()
            )

        # Debugging statements for verification
        print("DEBUG: train_preds_squeezed shape:", train_preds_squeezed.shape)
        print("DEBUG: val_preds_squeezed shape:", val_preds_squeezed.shape)
        print("DEBUG: baseline_train shape:", baseline_train.shape if config.get("use_returns", False) else "Not using returns")
        print("DEBUG: y_train_stacked shape:", y_train_stacked.shape)




        # Calculate MAE  train_mae = np.mean(np.abs(train_predictions - y_train[:n_test]))
        n_train = train_preds.shape[0]
        n_val = val_preds.shape[0]
        train_mae = np.mean(np.abs(train_preds[:, -1] - y_train_stacked[:n_train, -1]))
        val_mae = np.mean(np.abs(val_preds[:, -1] - y_val_stacked[:n_val, -1]))


        # Save loss plot
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f"Model Loss for {config['plugin'].upper()} - {iteration}")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Val"], loc="upper left")
        plt.savefig(config['loss_plot_file'])
        plt.close()
        print(f"Loss plot saved to {config['loss_plot_file']}")

        print("\nEvaluating on test dataset...")
        #test_predictions = plugin.predict(x_test)
        mc_samples = config.get("mc_samples", 100)
        test_predictions, uncertainty_estimates = plugin.predict_with_uncertainty(x_test, mc_samples=mc_samples)
        n_test = test_predictions.shape[0]
        # Convert y_test (which is a list of arrays) to a single NumPy array
        y_test_array = np.stack(y_test, axis=1)  # shape: (n_samples, time_horizon)

        # Squeeze predictions to match shapes properly
        test_predictions_squeezed = np.squeeze(test_predictions, axis=-1)  # shape: (samples, horizons)

        if config.get("use_returns", False):
            test_r2 = r2_score(
                (baseline_test[:, -1] + y_test_array[:n_test, -1]).flatten(),
                (baseline_test[:, -1] + test_predictions_squeezed[:n_test, -1]).flatten()
            )
        else:
            test_r2 = r2_score(
                y_test_array[:n_test, -1].flatten(),
                test_predictions_squeezed[:n_test, -1].flatten()
            )

        # Debugging shapes for verification
        print("DEBUG: test_predictions_squeezed shape:", test_predictions_squeezed.shape)
        print("DEBUG: baseline_test shape:", baseline_test.shape if config.get("use_returns", False) else "Not using returns")
        print("DEBUG: y_test_array shape:", y_test_array.shape)


        test_mae = np.mean(np.abs(test_predictions[:, -1] - y_test_array[:n_test, -1]))


        # calculate mmd for train, val and test
        #print("\nCalculating MMD for train, val and test datasets...")
        #train_mmd = plugin.compute_mmd(train_preds.astype(np.float64) , y_train.astype(np.float64), sigma=1.0, sample_size=mc_samples)
        #val_mmd = plugin.compute_mmd(val_preds.astype(np.float64), y_val.astype(np.float64), sigma=1.0, sample_size=mc_samples)
        #test_mmd = plugin.compute_mmd(test_predictions.astype(np.float64), y_test.astype(np.float64), sigma=1.0, sample_size=mc_samples)
        
        # calculate the mean of the last prediction (time_horizon column) uncertainty values
        train_unc_last = np.mean(train_unc[ : , -1])
        val_unc_last = np.mean(val_unc[ : , -1])
        test_unc_last = np.mean(uncertainty_estimates[ : , -1])
        
        # calcula los promedios de la señal para calcular SNR (la desviación es el uncertainty)
        train_mean = np.mean(baseline_train[ : , -1] + train_preds[ : , -1])
        val_mean = np.mean(baseline_val[ : , -1] + val_preds[ : , -1])
        test_mean = np.mean(baseline_test[ : , -1] + test_predictions[ : , -1])
        
        # calcula the the SNR as the 1/(uncertainty/mae)^2
        train_snr = 1/(train_unc_last/train_mean)
        val_snr = 1/(val_unc_last/val_mean)
        test_snr = 1/(test_unc_last/test_mean)
        
        # calcula el profit y el risk si se está usando una estrategia de trading
        test_profit = 0.0
        test_risk = 0.0
        if (config.get("use_strategy", False) and config.get("use_daily"), True):
            candidate = None
            # carga el plugin usando strategy_plugin_group y strategy_plugin_name
            strategy_plugin_group = config.get("strategy_plugin_group", None)
            strategy_plugin_name = config.get("strategy_plugin_name", None)
            if strategy_plugin_group is None or strategy_plugin_name is None:
                raise ValueError("strategy_plugin_group and strategy_plugin_name must be defined in the configuration.")
            plugin_class, _ = load_plugin(strategy_plugin_group, strategy_plugin_name)
            strategy_plugin=plugin_class()
            # load simulation parameters (mandatory)
            if config.get("strategy_load_parameters") is not None:
                try:
                    with open(config["strategy_load_parameters"], "r") as f:
                        loaded_params = json.load(f)
                    print(f"Loaded evaluation parameters from {config['strategy_load_parameters']}: {loaded_params}")
                    # load the parameters from the loaded file
                    candidate = [
                        loaded_params.get("profit_threshold"),
                        loaded_params.get("tp_multiplier"),
                        loaded_params.get("sl_multiplier"),
                        loaded_params.get("lower_rr_threshold"),
                        loaded_params.get("upper_rr_threshold"),
                        int(loaded_params.get("time_horizon", 3))
                    ]
                except Exception as e:
                    raise ValueError(f"Failed to load parameters from {config['strategy_load_parameters']}: {e}")
            else:   
                raise ValueError("Parameters json file for strategy are required.")
            def load_csv_d(file_path, headers=True, **kwargs):
                # Read CSV with header if specified
                df = pd.read_csv(file_path, header=0 if headers else None, **kwargs)
                # If headers are enabled and the first column is 'DATE_TIME', use it as the index
                if headers and df.columns[0].strip().upper() == "DATE_TIME":
                    df.index = pd.to_datetime(df.iloc[:, 0], errors='raise')
                    df.drop(df.columns[0], axis=1, inplace=True)
                return df
            # load the denormalized hourly predictions from the strategy_1h_prediction file
            hourly_df = load_csv_d(config["strategy_1h_prediction"], headers=config["headers"])
            # load the denormalized predictions uncertainty from the strategy_1h_uncertainty file
            uncertainty_hourly_df = load_csv_d(config["strategy_1h_uncertainty"], headers=config["headers"])
            # use the current iteration normalized daily predictions
            daily_df = None
            uncertainty_daily_df = None
            # denormalize the hourly predictions 
            if config.get("use_normalization_json") is not None:
                norm_json = config.get("use_normalization_json")
                if isinstance(norm_json, str):
                    with open(norm_json, 'r') as f:
                        norm_json = json.load(f)
                if config.get("use_returns", False):
                    if "CLOSE" in norm_json:
                        close_min = norm_json["CLOSE"]["min"]
                        close_max = norm_json["CLOSE"]["max"]
                        diff = close_max - close_min
                        # Final predicted close = (predicted_return + baseline)*diff + close_min
                        # Ensure compatible shapes by squeezing test_predictions and properly broadcasting baseline_test
                        test_predictions_squeezed = np.squeeze(test_predictions, axis=-1)  # shape: (6293, 6)

                        # Expand baseline_test along the horizon dimension to match predictions shape
                        baseline_test_expanded = np.repeat(baseline_test, test_predictions_squeezed.shape[1], axis=1)  # (6293, 6)

                        # Correct broadcasting
                        daily_df = (test_predictions_squeezed + baseline_test_expanded) * diff + close_min

                        # Debugging shapes for verification
                        print("DEBUG: test_predictions_squeezed shape:", test_predictions_squeezed.shape)
                        print("DEBUG: baseline_test_expanded shape:", baseline_test_expanded.shape)

                    else:
                        print("Warning: 'CLOSE' not found; skipping denormalization for returns.")
                else:
                    if "CLOSE" in norm_json:
                        close_min = norm_json["CLOSE"]["min"]
                        close_max = norm_json["CLOSE"]["max"]
                        daily_df = test_predictions * (close_max - close_min) + close_min
            # Rename columns and add DATE_TIME column if required
            daily_df = pd.DataFrame(
                daily_df, columns=[f"Prediction_{i+1}" for i in range(daily_df.shape[1])]
            )
            if test_dates is not None:
                daily_df['DATE_TIME'] = pd.Series(test_dates[:len(daily_df)])
            else:
                daily_df['DATE_TIME'] = pd.NaT
            cols = ['DATE_TIME'] + [col for col in daily_df.columns if col != 'DATE_TIME']
            daily_df = daily_df[cols]
            
            # Denormalize uncertainties using CLOSE range only 
            if config.get("use_normalization_json") is not None:
                norm_json = config.get("use_normalization_json")
                if isinstance(norm_json, str):
                    with open(norm_json, 'r') as f:
                        norm_json = json.load(f)
                if "CLOSE" in norm_json:
                    diff = norm_json["CLOSE"]["max"] - norm_json["CLOSE"]["min"]
                    uncertainty_daily_df = uncertainty_estimates * diff
                else:
                    print("Warning: 'CLOSE' not found; uncertainties remain normalized.")
                    uncertainty_daily_df = uncertainty_estimates
            else:
                uncertainty_daily_df = uncertainty_estimates
            # Ensure uncertainty_daily_df has the correct 2D shape before DataFrame creation
            uncertainty_daily_df_squeezed = np.squeeze(uncertainty_daily_df, axis=-1)  # shape: (6293, 6)

            # Verify shape after squeeze operation (debugging)
            print("DEBUG: uncertainty_daily_df_squeezed shape:", uncertainty_daily_df_squeezed.shape)

            # Create DataFrame with properly formatted uncertainty data
            uncertainty_daily_df = pd.DataFrame(
                uncertainty_daily_df_squeezed,
                columns=[f"Uncertainty_{i+1}" for i in range(uncertainty_daily_df_squeezed.shape[1])]
            )
    
            # Add DATE_TIME column to uncertainties if available                
            if test_dates is not None:  
                uncertainty_daily_df['DATE_TIME'] = pd.Series(test_dates[:len(uncertainty_daily_df)])
            else:
                uncertainty_daily_df['DATE_TIME'] = pd.NaT
            cols = ['DATE_TIME'] + [col for col in uncertainty_daily_df.columns if col != 'DATE_TIME']
            uncertainty_daily_df = uncertainty_daily_df[cols]
            
            # load the strategy base (unnormalized) hourly data
            base_df = load_csv_d(config["strategy_base_dataset"], headers=config["headers"])

            # Ensure all datasets have a datetime index based on DATE_TIME column.
            def ensure_datetime(df, name):
                if not isinstance(df.index, pd.DatetimeIndex):
                    # Try to find a column named "DATE_TIME" (case-insensitive)
                    dt_col = None
                    for col in df.columns:
                        if col.strip().upper() == "DATE_TIME":
                            dt_col = col
                            break
                    if dt_col is not None:
                        df.index = pd.to_datetime(df[dt_col])
                    elif len(df.columns) > 0:
                        # Fallback: attempt to convert the first column to datetime
                        try:
                            df.index = pd.to_datetime(df.iloc[:, 0])
                        except Exception as e:
                            raise ValueError(f"{name} does not have a valid DATE_TIME column: {e}")
                    else:
                        raise ValueError(f"{name} does not have a DATE_TIME column.")
                return df


            base_df = ensure_datetime(base_df, "base_df")
            hourly_df = ensure_datetime(hourly_df, "hourly_df")
            daily_df = ensure_datetime(daily_df, "daily_df")
            if uncertainty_hourly_df is not None:
                uncertainty_hourly_df = ensure_datetime(uncertainty_hourly_df, "uncertainty_hourly_df")
            if uncertainty_daily_df is not None:
                uncertainty_daily_df = ensure_datetime(uncertainty_daily_df, "uncertainty_daily_df")

            # Compute common index across all datasets (only include uncertainties if available)
            common_index = base_df.index.intersection(hourly_df.index).intersection(daily_df.index)
            if uncertainty_hourly_df is not None:
                common_index = common_index.intersection(uncertainty_hourly_df.index)
            if uncertainty_daily_df is not None:
                common_index = common_index.intersection(uncertainty_daily_df.index)
            
            # Print date ranges for debugging
            print("Base dataset date range:", base_df.index.min(), "to", base_df.index.max())
            print("Hourly predictions date range:", hourly_df.index.min(), "to", hourly_df.index.max())
            print("Daily predictions date range:", daily_df.index.min(), "to", daily_df.index.max())
            if uncertainty_hourly_df is not None:
                print("Hourly uncertainties date range:", uncertainty_hourly_df.index.min(), "to", uncertainty_hourly_df.index.max())
            if uncertainty_daily_df is not None:
                print("Daily uncertainties date range:", uncertainty_daily_df.index.min(), "to", uncertainty_daily_df.index.max())
            
            if common_index.empty:
                raise ValueError("No common date range found among base, predictions, and uncertainties.")


            # Trim all datasets to the common date range
            base_df = base_df.loc[common_index]
            hourly_df = hourly_df.loc[common_index]
            daily_df = daily_df.loc[common_index]
            if uncertainty_hourly_df is not None:
                uncertainty_hourly_df = uncertainty_hourly_df.loc[common_index]
            if uncertainty_daily_df is not None:
                uncertainty_daily_df = uncertainty_daily_df.loc[common_index]

            # Apply max_steps if provided: truncate all datasets to the same number of rows.
            if "max_steps" in config:
                max_steps = config["max_steps"]
                base_df = base_df.iloc[:max_steps]
                hourly_df = hourly_df.iloc[:max_steps]
                daily_df = daily_df.iloc[:max_steps]
                if uncertainty_hourly_df is not None:
                    uncertainty_hourly_df = uncertainty_hourly_df.iloc[:max_steps]
                if uncertainty_daily_df is not None:
                    uncertainty_daily_df = uncertainty_daily_df.iloc[:max_steps]

            # Print aligned date ranges and shapes.
            print(f"Aligned Base dataset range: {base_df.index.min()} to {base_df.index.max()}")
            print(f"Aligned Hourly predictions range: {hourly_df.index.min()} to {hourly_df.index.max()}")
            print(f"Aligned Daily predictions range: {daily_df.index.min()} to {daily_df.index.max()}")
            if uncertainty_hourly_df is not None:
                print(f"Aligned Hourly uncertainties range: {uncertainty_hourly_df.index.min()} to {uncertainty_hourly_df.index.max()}")
            if uncertainty_daily_df is not None:
                print(f"Aligned Daily uncertainties range: {uncertainty_daily_df.index.min()} to {uncertainty_daily_df.index.max()}")

            # Print the candidate.
            individual = candidate
            print(f"[EVALUATE] Evaluating candidate (genome): {individual}")
            
            result = strategy_plugin.evaluate_candidate(individual, base_df, hourly_df, daily_df, config)
            
            # If the result returns both profit and stats, extract and print them.
            test_profit, stats = result
            test_risk = stats.get('risk', 0)
            print(f"[EVALUATE] Strategy result on Test Data => Profit: {test_profit:.2f}, Risk: {test_risk:.2f}",
                f"Trades: {stats.get('num_trades', 0)}, "
                f"Win%: {stats.get('win_pct', 0):.1f}, "
                f"MaxDD: {stats.get('max_dd', 0):.2f}, "
                f"Sharpe: {stats.get('sharpe', 0):.2f}")

        else: #end use strategy
            # trow error on no parameters loaded, and exit execution
            raise ValueError("Both strategy_plugin_group and strategy_plugin_name must be True.")            
            
        
        # Append the calculated train values
        training_mae_list.append(train_mae)
        training_r2_list.append(train_r2)
        training_unc_list.append(train_unc_last)
        training_snr_list.append(train_snr)
        training_profit_list.append(0)
        training_risk_list.append(0)
        # Append the calculated validation values
        validation_mae_list.append(val_mae)
        validation_r2_list.append(val_r2)
        validation_unc_list.append(val_unc_last)
        validation_snr_list.append(val_snr)
        validation_profit_list.append(0)
        validation_risk_list.append(0)
        # Append the calculated test values
        test_mae_list.append(test_mae)
        test_r2_list.append(test_r2)
        test_unc_list.append(test_unc_last)
        test_snr_list.append(test_snr)
        test_profit_list.append(test_profit)
        test_risk_list.append(test_risk)
        # print iteration results
        print("************************************************************************")
        print(f"Iteration {iteration} completed.")
        print(f"Training MAE: {train_mae}, Training R²: {train_r2}, Training Uncertainty: {train_unc_last}, Trainign SNR: {train_snr}")
        print(f"Validation MAE: {val_mae}, Validation R²: {val_r2}, Validation Uncertainty: {val_unc_last}, Validation SNR: {val_snr}")
        print(f"Test MAE: {test_mae}, Test R²: {test_r2}, Test Uncertainty: {test_unc_last}, Test SNR: {test_snr}, Test Profit: {test_profit}, Test Risk: {test_risk}")
        print("************************************************************************")
        print(f"Iteration {iteration} completed in {time.time()-iter_start:.2f} seconds")
    # Save consolidated results
    if config.get("use_strategy", False): 
        results = {
            "Metric": ["Training MAE", "Training R²", "Training Uncertainty", "Training SNR", "Train Profit", "Train Risk", 
                        "Validation MAE", "Validation R²", "Validation Uncertainty", "Validation SNR", "Validation Profit", "Validation Risk",
                        "Test MAE", "Test R²", "Test Uncertainty", "Test SNR", "Test Profit", "Test Risk"],
            "Average": [np.mean(training_mae_list), np.mean(training_r2_list), np.mean(training_unc_list), np.mean(training_snr_list), np.mean(training_profit_list), np.mean(training_risk_list),
                        np.mean(validation_mae_list), np.mean(validation_r2_list), np.mean(validation_unc_list), np.mean(validation_snr_list), np.mean(validation_profit_list), np.mean(validation_risk_list),
                        np.mean(test_mae_list), np.mean(test_r2_list), np.mean(test_unc_list), np.mean(test_snr_list), np.mean(test_profit_list), np.mean(test_risk_list)],
            "Std Dev": [np.std(training_mae_list), np.std(training_r2_list), np.std(training_unc_list), np.std(training_snr_list), np.std(training_profit_list), np.std(training_risk_list),
                        np.std(validation_mae_list), np.std(validation_r2_list), np.std(validation_unc_list), np.std(validation_snr_list), np.std(validation_profit_list), np.std(validation_risk_list),
                        np.std(test_mae_list), np.std(test_r2_list), np.std(test_unc_list), np.std(test_snr_list), np.std(test_profit_list), np.std(test_risk_list)],
            "Max": [np.max(training_mae_list), np.max(training_r2_list), np.max(training_unc_list), np.max(training_snr_list), np.max(training_profit_list), np.max(training_risk_list),
                    np.max(validation_mae_list), np.max(validation_r2_list), np.max(validation_unc_list), np.max(validation_snr_list), np.max(validation_profit_list), np.max(validation_risk_list),
                    np.max(test_mae_list), np.max(test_r2_list), np.max(test_unc_list), np.max(test_snr_list), np.max(test_profit_list), np.max(test_risk_list)],
            "Min": [np.min(training_mae_list), np.min(training_r2_list), np.min(training_unc_list), np.min(training_snr_list), np.min(training_profit_list), np.min(training_risk_list),
                    np.min(validation_mae_list), np.min(validation_r2_list), np.min(validation_unc_list), np.min(validation_snr_list), np.min(validation_profit_list), np.min(validation_risk_list),
                    np.min(test_mae_list), np.min(test_r2_list), np.min(test_unc_list), np.min(test_snr_list), np.min(test_profit_list), np.min(test_risk_list)]
            }
    else:
        results = {
            "Metric": ["Training MAE", "Training R²", "Training Uncertainty", "Training SNR", 
                        "Validation MAE", "Validation R²", "Validation Uncertainty", "Validation SNR",
                        "Test MAE", "Test R²", "Test Uncertainty", "Test SNR"],
            "Average": [np.mean(training_mae_list), np.mean(training_r2_list), np.mean(training_unc_list), np.mean(training_snr_list),
                        np.mean(validation_mae_list), np.mean(validation_r2_list), np.mean(validation_unc_list), np.mean(validation_snr_list),
                        np.mean(test_mae_list), np.mean(test_r2_list), np.mean(test_unc_list), np.mean(test_snr_list)],
            "Std Dev": [np.std(training_mae_list), np.std(training_r2_list), np.std(training_unc_list), np.std(training_snr_list),
                        np.std(validation_mae_list), np.std(validation_r2_list), np.std(validation_unc_list), np.std(validation_snr_list),
                        np.std(test_mae_list), np.std(test_r2_list), np.std(test_unc_list), np.std(test_snr_list)],
            "Max": [np.max(training_mae_list), np.max(training_r2_list), np.max(training_unc_list), np.max(training_snr_list),
                    np.max(validation_mae_list), np.max(validation_r2_list), np.max(validation_unc_list), np.max(validation_snr_list),
                    np.max(test_mae_list), np.max(test_r2_list), np.max(test_unc_list), np.max(test_snr_list)],
            "Min": [np.min(training_mae_list), np.min(training_r2_list), np.min(training_unc_list), np.min(training_snr_list),
                    np.min(validation_mae_list), np.min(validation_r2_list), np.min(validation_unc_list), np.min(validation_snr_list),
                    np.min(test_mae_list), np.min(test_r2_list), np.min(test_unc_list), np.min(test_snr_list)],
            }
    # Save consolidated results to CSV
    results_file = config.get("results_file", "results.csv")
    pd.DataFrame(results).to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    # --- Denormalize final test predictions (if normalization provided) ---
    denorm_test_close_prices = test_close_prices
    if config.get("use_normalization_json") is not None:
        norm_json = config.get("use_normalization_json")
        if isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        if config.get("use_returns", False):
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                diff = close_max - close_min
                # Final predicted close = (predicted_return + baseline)*diff + close_min
                # Squeeze the predictions array to remove extra dimensions (6293, 6, 1) → (6293, 6)
                test_predictions_squeezed = np.squeeze(test_predictions, axis=-1)

                # Expand baseline_test dimensions explicitly to match test_predictions
                baseline_test_expanded = np.expand_dims(baseline_test, axis=-1)  # (6293, 1) → (6293, 1, 1)

                # Perform broadcasting explicitly and safely
                test_predictions = (test_predictions_squeezed + baseline_test_expanded.squeeze(-1)) * diff + close_min

                # --- NEW CODE: Correctly stack y_test into a (n_samples, time_horizon) array ---
                y_test_array = np.stack(y_test, axis=1)  # Ensure y_test is now (n_samples, time_horizon)
                denorm_y_test = (y_test_array + baseline_test) * diff + close_min
                # --- END NEW CODE ---

            else:
                print("Warning: 'CLOSE' not found; skipping denormalization for returns.")
        else:
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                test_predictions = test_predictions * (close_max - close_min) + close_min
                denorm_y_test = y_test * (close_max - close_min) + close_min
        denorm_test_close_prices = test_close_prices * (close_max - close_min) + close_min 
    # Save final predictions CSV
    final_test_file = config.get("output_file", "test_predictions.csv")
    test_predictions_df = pd.DataFrame(
        test_predictions, columns=[f"Prediction_{i+1}" for i in range(test_predictions.shape[1])]
    )
    # Use test_dates (which now hold the base dates for each window)
    if test_dates is not None:
        test_predictions_df['DATE_TIME'] = pd.Series(test_dates[:len(test_predictions_df)])
    else:
        test_predictions_df['DATE_TIME'] = pd.NaT
    cols = ['DATE_TIME'] + [col for col in test_predictions_df.columns if col != 'DATE_TIME']
    test_predictions_df = test_predictions_df[cols]
    # Add the denorm_y_test to the existing test_predictions_df dictionary as new columns, it names columsn as Target_1, Target_2, etc
    denorm_y_test_df = pd.DataFrame(
        denorm_y_test, columns=[f"Target_{i+1}" for i in range(denorm_y_test.shape[1])]
    )
    test_predictions_df = pd.concat([test_predictions_df, denorm_y_test_df], axis=1)
    # Add the denorm_test_close_prices to the existing test_predictions_df dictionary as a new column
    test_predictions_df['CLOSE'] = denorm_test_close_prices
    # Save the final test predictions to a CSV file
    
    write_csv(file_path=final_test_file, data=test_predictions_df, include_date=False, headers=config.get('headers', True))
    print(f"Final validation predictions saved to {final_test_file}")

    # --- Compute and save uncertainty estimates (denormalized) ---
    print("Computing uncertainty estimates using MC sampling...")
    try:
        mc_samples = config.get("mc_samples", 100)
        _, uncertainty_estimates = plugin.predict_with_uncertainty(x_test, mc_samples=mc_samples)
        # Denormalize uncertainties using CLOSE range only (do not add offset)
        if config.get("use_normalization_json") is not None:
            norm_json = config.get("use_normalization_json")
            if isinstance(norm_json, str):
                with open(norm_json, 'r') as f:
                    norm_json = json.load(f)
            if "CLOSE" in norm_json:
                diff = norm_json["CLOSE"]["max"] - norm_json["CLOSE"]["min"]
                denorm_uncertainty = uncertainty_estimates * diff
            else:
                print("Warning: 'CLOSE' not found; uncertainties remain normalized.")
                denorm_uncertainty = uncertainty_estimates
        else:
            denorm_uncertainty = uncertainty_estimates
        uncertainty_df = pd.DataFrame(
            denorm_uncertainty, columns=[f"Uncertainty_{i+1}" for i in range(denorm_uncertainty.shape[1])]
        )
        if test_dates is not None:
            uncertainty_df['DATE_TIME'] = pd.Series(test_dates[:len(uncertainty_df)])
        else:
            uncertainty_df['DATE_TIME'] = pd.NaT
        cols = ['DATE_TIME'] + [col for col in uncertainty_df.columns if col != 'DATE_TIME']
        uncertainty_df = uncertainty_df[cols]
        uncertainties_file = config.get("uncertainties_file", "test_uncertainties.csv")
        uncertainty_df.to_csv(uncertainties_file, index=False)
        print(f"Uncertainty predictions saved to {uncertainties_file}")
    except Exception as e:
        print(f"Failed to compute or save uncertainty predictions: {e}")

    # --- Plot predictions (only the prediction at the selected horizon) ---
    # Define the plotted horizon (zero-indexed)
    plotted_horizon = config.get("plotted_horizon", 6)
    plotted_idx = plotted_horizon - 1  # Zero-based index for the chosen horizon

    # Ensure indices are valid
    if plotted_idx >= test_predictions.shape[1]:
        raise ValueError(f"Plotted horizon index {plotted_idx} is out of bounds for predictions shape {test_predictions.shape}")

    # Extract predictions for the selected horizon
    pred_plot = test_predictions[:, plotted_idx]

    # Define the test dates for plotting
    
    n_plot = config.get("plot_points",1575)  # Number of points to display
    if len(pred_plot) > n_plot:
        pred_plot = pred_plot[-n_plot:]
        test_dates_plot = test_dates[-n_plot:] if test_dates is not None else np.arange(len(pred_plot))
    else:
        test_dates_plot = test_dates if test_dates is not None else np.arange(len(pred_plot))

    # Extract and correctly denormalize the baseline close value (current tick's true value)
    if "baseline_test" in datasets:
        baseline_plot = datasets["baseline_test"][:, 0]  # Use first column if multi-step

        # Keep only last n_plot values if necessary
        if len(baseline_plot) > n_plot:
            baseline_plot = baseline_plot[-n_plot:]
    else:
        raise ValueError("Baseline test values not found; unable to reconstruct actual predictions.")

    # --- Correcting Denormalization ---
    if config.get("use_normalization_json") is not None:
        norm_json = config.get("use_normalization_json")
        if isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        if "CLOSE" in norm_json:
            close_min = norm_json["CLOSE"]["min"]
            close_max = norm_json["CLOSE"]["max"]
            diff = close_max - close_min

            # ✅ Correct Denormalization
            # True values (Baseline Close)
            true_plot = baseline_plot * diff + close_min  # ✅ Correct
            # Predictions (Adding Correctly Denormalized Returns)
            #pred_plot = true_plot + (pred_plot * diff)  # ✅ Fixing double denormalization
        else:
            print("Warning: 'CLOSE' not found; skipping denormalization for predictions.")
            true_plot = baseline_plot
            pred_plot = baseline_plot + pred_plot
    else:
        print("Warning: Normalization JSON not provided; assuming raw values.")
        true_plot = baseline_plot
        pred_plot = baseline_plot + pred_plot

    # Extract uncertainty for the plotted horizon
    uncertainty_plot = denorm_uncertainty[:, plotted_idx]
    if len(uncertainty_plot) > n_plot:
        uncertainty_plot = uncertainty_plot[-n_plot:]

    # Plot results
    plot_color_predicted = config.get("plot_color_predicted", "blue")
    plot_color_true = config.get("plot_color_true", "red")  # Default: red
    plot_color_uncertainty = config.get("plot_color_uncertainty", "green")  # Default: green    
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates_plot, pred_plot, label="Predicted Price", color=plot_color_predicted, linewidth=2)
    plt.plot(test_dates_plot, true_plot, label="True Price", color=plot_color_true, linewidth=2)
    # Ensure pred_plot and uncertainty_plot are explicitly squeezed to be 1-dimensional.
    pred_plot = np.squeeze(pred_plot)
    uncertainty_plot = np.squeeze(uncertainty_plot)

    # Check final dimensions to ensure correctness:
    assert pred_plot.ndim == 1, f"pred_plot must be 1-dimensional, got shape {pred_plot.shape}"
    assert uncertainty_plot.ndim == 1, f"uncertainty_plot must be 1-dimensional, got shape {uncertainty_plot.shape}"

    # Now safely plot uncertainty
    plt.fill_between(
        test_dates_plot,
        pred_plot - uncertainty_plot,
        pred_plot + uncertainty_plot,
        color=plot_color_uncertainty,
        alpha=0.15,
        label="Uncertainty"
    )

    if config.get("use_daily", False):    
        plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} days)")
    else:
        plt.title(f"Predictions vs True Values (Horizon: {plotted_horizon} hours)")
    plt.xlabel("Close Time")
    plt.ylabel("EUR Price [USD]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    try:
        predictions_plot_file = config.get("predictions_plot_file", "predictions_plot.png")
        plt.savefig(predictions_plot_file, dpi=300)
        plt.close()
        print(f"Prediction plot saved to {predictions_plot_file}")
    except Exception as e:
        print(f"Failed to generate prediction plot: {e}")


    # Plot the model
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(
            plugin.model,
            to_file=config['model_plot_file'],
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            expand_nested=True,
            dpi=300,
            show_layer_activations=True
        )
        print(f"Model plot saved to {config['model_plot_file']}")
    except Exception as e:
        print(f"Failed to generate model plot: {e}")
        print("Download Graphviz from https://graphviz.org/download/")

    save_model_file = config.get("save_model", "pretrained_model.keras")
    try:
        plugin.save(save_model_file)
        print(f"Model saved to {save_model_file}")
    except Exception as e:
        print(f"Failed to save model to {save_model_file}: {e}")
    
    if config.get("use_strategy", False):
        print("*************************************************")
        print("Training Statistics:")
        print(f"MAE - Avg: {results['Average'][0]:.4f}, Std: {results['Std Dev'][0]:.4f}, Max: {results['Max'][0]:.4f}, Min: {results['Min'][0]:.4f}")
        print(f"R²  - Avg: {results['Average'][1]:.4f}, Std: {results['Std Dev'][1]:.4f}, Max: {results['Max'][1]:.4f}, Min: {results['Min'][1]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][2]:.4f}, Std: {results['Std Dev'][2]:.4f}, Max: {results['Max'][2]:.4f}, Min: {results['Min'][2]:.4f}")
        print(f"SNR - Avg: {results['Average'][3]:.4f}, Std: {results['Std Dev'][3]:.4f}, Max: {results['Max'][3]:.4f}, Min: {results['Min'][3]:.4f}")
        print(f"Profit - Avg: {results['Average'][4]:.4f}, Std: {results['Std Dev'][4]:.4f}, Max: {results['Max'][4]:.4f}, Min: {results['Min'][4]:.4f}")
        print(f"Risk - Avg: {results['Average'][5]:.4f}, Std: {results['Std Dev'][5]:.4f}, Max: {results['Max'][5]:.4f}, Min: {results['Min'][5]:.4f}")
        print("*************************************************")
        print("Validation Statistics:")
        print(f"MAE - Avg: {results['Average'][6]:.4f}, Std: {results['Std Dev'][6]:.4f}, Max: {results['Max'][6]:.4f}, Min: {results['Min'][6]:.4f}")
        print(f"R²  - Avg: {results['Average'][7]:.4f}, Std: {results['Std Dev'][7]:.4f}, Max: {results['Max'][7]:.4f}, Min: {results['Min'][7]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][8]:.4f}, Std: {results['Std Dev'][8]:.4f}, Max: {results['Max'][8]:.4f}, Min: {results['Min'][8]:.4f}")
        print(f"SNR - Avg: {results['Average'][9]:.4f}, Std: {results['Std Dev'][9]:.4f}, Max: {results['Max'][9]:.4f}, Min: {results['Min'][9]:.4f}")
        print(f"Profit - Avg: {results['Average'][10]:.4f}, Std: {results['Std Dev'][10]:.4f}, Max: {results['Max'][10]:.4f}, Min: {results['Min'][10]:.4f}")
        print(f"Risk - Avg: {results['Average'][11]:.4f}, Std: {results['Std Dev'][11]:.4f}, Max: {results['Max'][11]:.4f}, Min: {results['Min'][11]:.4f}")
        print("*************************************************")
        print("Test Statistics:")
        print(f"MAE - Avg: {results['Average'][12]:.4f}, Std: {results['Std Dev'][12]:.4f}, Max: {results['Max'][12]:.4f}, Min: {results['Min'][12]:.4f}")
        print(f"R²  - Avg: {results['Average'][13]:.4f}, Std: {results['Std Dev'][13]:.4f}, Max: {results['Max'][13]:.4f}, Min: {results['Min'][13]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][14]:.4f}, Std: {results['Std Dev'][14]:.4f}, Max: {results['Max'][14]:.4f}, Min: {results['Min'][14]:.4f}")
        print(f"SNR - Avg: {results['Average'][15]:.4f}, Std: {results['Std Dev'][15]:.4f}, Max: {results['Max'][15]:.4f}, Min: {results['Min'][15]:.4f}")
        print(f"Profit - Avg: {results['Average'][16]:.4f}, Std: {results['Std Dev'][16]:.4f}, Max: {results['Max'][16]:.4f}, Min: {results['Min'][16]:.4f}")
        print(f"Risk - Avg: {results['Average'][17]:.4f}, Std: {results['Std Dev'][17]:.4f}, Max: {results['Max'][17]:.4f}, Min: {results['Min'][17]:.4f}")
        print("*************************************************")
    else:
        print("*************************************************")
        print("Training Statistics:")
        print(f"MAE - Avg: {results['Average'][0]:.4f}, Std: {results['Std Dev'][0]:.4f}, Max: {results['Max'][0]:.4f}, Min: {results['Min'][0]:.4f}")
        print(f"R²  - Avg: {results['Average'][1]:.4f}, Std: {results['Std Dev'][1]:.4f}, Max: {results['Max'][1]:.4f}, Min: {results['Min'][1]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][2]:.4f}, Std: {results['Std Dev'][2]:.4f}, Max: {results['Max'][2]:.4f}, Min: {results['Min'][2]:.4f}")
        print(f"SNR - Avg: {results['Average'][3]:.4f}, Std: {results['Std Dev'][3]:.4f}, Max: {results['Max'][3]:.4f}, Min: {results['Min'][3]:.4f}")
        print("*************************************************")
        print("Validation Statistics:")
        print(f"MAE - Avg: {results['Average'][4]:.4f}, Std: {results['Std Dev'][4]:.4f}, Max: {results['Max'][4]:.4f}, Min: {results['Min'][4]:.4f}")
        print(f"R²  - Avg: {results['Average'][5]:.4f}, Std: {results['Std Dev'][5]:.4f}, Max: {results['Max'][5]:.4f}, Min: {results['Min'][5]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][6]:.4f}, Std: {results['Std Dev'][6]:.4f}, Max: {results['Max'][6]:.4f}, Min: {results['Min'][6]:.4f}")
        print(f"SNR - Avg: {results['Average'][7]:.4f}, Std: {results['Std Dev'][7]:.4f}, Max: {results['Max'][7]:.4f}, Min: {results['Min'][7]:.4f}") 
        print("*************************************************")
        print("Test Statistics:")
        print(f"MAE - Avg: {results['Average'][8]:.4f}, Std: {results['Std Dev'][8]:.4f}, Max: {results['Max'][8]:.4f}, Min: {results['Min'][8]:.4f}")
        print(f"R²  - Avg: {results['Average'][9]:.4f}, Std: {results['Std Dev'][9]:.4f}, Max: {results['Max'][9]:.4f}, Min: {results['Min'][9]:.4f}")
        print(f"Uncertainty - Avg: {results['Average'][10]:.4f}, Std: {results['Std Dev'][10]:.4f}, Max: {results['Max'][10]:.4f}, Min: {results['Min'][10]:.4f}")
        print(f"SNR - Avg: {results['Average'][11]:.4f}, Std: {results['Std Dev'][11]:.4f}, Max: {results['Max'][11]:.4f}, Min: {results['Min'][11]:.4f}")
        print("*************************************************")
    
    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")


def load_and_evaluate_model(config, plugin):
    """
    Loads a pre-trained model and evaluates it on the validation data.
    Predictions are denormalized; if use_returns is True, predicted returns are converted
    to predicted close values by adding the corresponding baseline (using the "CLOSE" parameters).
    The final predictions CSV includes a DATE_TIME column.
    """
    import sys, numpy as np, pandas as pd, json
    from tensorflow.keras.models import load_model

    print(f"Loading pre-trained model from {config['load_model']}...")
    try:
        custom_objects = {"combined_loss": combined_loss, "mmd": mmd_metric, "huber": huber_metric}
        plugin.model = load_model(config['load_model'], custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load the model from {config['load_model']}: {e}")
        sys.exit(1)

    print("Loading and processing validation data for evaluation...")
    try:
        datasets = process_data(config)
        x_val = datasets["x_val"]
        val_dates = datasets.get("dates_val")
        if config["plugin"] in ["lstm", "cnn", "transformer","ann"]:
            print("Creating sliding windows for CNN...")
            x_val, val_date_windows = create_sliding_windows(
                x_val, config['window_size'], stride=1, date_times=val_dates
            )
            val_dates = val_date_windows
            print(f"Sliding windows created: x_val: {x_val.shape}")
            if x_val.ndim != 3:
                raise ValueError(f"For CNN and LSTM, x_val must be 3D. Found: {x_val.shape}.")

        print(f"Processed validation data: X shape: {x_val.shape}")
    except Exception as e:
        print(f"Failed to process validation data: {e}")
        sys.exit(1)

    print("Making predictions on validation data...")
    try:
        x_val_array = x_val if isinstance(x_val, np.ndarray) else x_val.to_numpy()

        #predictions = plugin.predict(x_val_array)
        mc_samples = config.get("mc_samples", 100)
        predictions, uncertainty_estimates = plugin.predict_with_uncertainty(x_val_array, mc_samples=mc_samples)
        
        
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"Failed to make predictions: {e}")
        sys.exit(1)

    if config.get("use_normalization_json") is not None:
        norm_json = config.get("use_normalization_json")
        if isinstance(norm_json, str):
            with open(norm_json, 'r') as f:
                norm_json = json.load(f)
        # Denormalize predictions using CLOSE range.
        if config.get("use_returns", False):
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                diff = close_max - close_min
                if "baseline_val" in datasets:
                    baseline = datasets["baseline_val"]
                    predictions = (predictions + baseline) * diff + close_min
                else:
                    print("Warning: Baseline validation values not found; cannot convert returns to predicted close values.")
            else:
                print("Warning: 'CLOSE' not found; skipping proper denormalization for returns.")
        else:
            if "CLOSE" in norm_json:
                close_min = norm_json["CLOSE"]["min"]
                close_max = norm_json["CLOSE"]["max"]
                predictions = predictions * (close_max - close_min) + close_min

    if predictions.ndim == 1 or predictions.shape[1] == 1:
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
    else:
        num_steps = predictions.shape[1]
        pred_cols = [f'Prediction_{i+1}' for i in range(num_steps)]
        predictions_df = pd.DataFrame(predictions, columns=pred_cols)

    if val_dates is not None:
        predictions_df['DATE_TIME'] = pd.Series(val_dates[:len(predictions_df)])
    else:
        predictions_df['DATE_TIME'] = pd.NaT
        print("Warning: DATE_TIME for validation predictions not captured.")

    cols = ['DATE_TIME'] + [col for col in predictions_df.columns if col != 'DATE_TIME']
    predictions_df = predictions_df[cols]
    evaluate_filename = config['output_file']
    try:
        write_csv(file_path=evaluate_filename, data=predictions_df,
                  include_date=False, headers=config.get('headers', True))
        print(f"Validation predictions with DATE_TIME saved to {evaluate_filename}")
    except Exception as e:
        print(f"Failed to save validation predictions to {evaluate_filename}: {e}")
        sys.exit(1)



def create_sliding_windows(x, window_size, stride=1, date_times=None):
    """
    Creates sliding windows for input features and targets with a specified stride.

    Args:
        x (numpy.ndarray): Input features of shape (N, features).
        y (numpy.ndarray): Targets of shape (N,) or (N, 1).
        window_size (int): Number of past steps to include in each window.
        time_horizon (int): Number of future steps to predict.
        stride (int): Step size between windows.
        date_times (pd.DatetimeIndex, optional): Corresponding date times for each sample.

    Returns:
        tuple:
            - x_windowed (numpy.ndarray): Shaped (samples, window_size, features).
            - y_windowed (numpy.ndarray): Shaped (samples, time_horizon).
            - date_time_windows (list): List of date times for each window (if provided).
    """
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    elif y.ndim > 2:
        raise ValueError("y should be a 1D or 2D array with a single column.")

    x_windowed = []
    y_windowed = []
    date_time_windows = []

    for i in range(0, len(x) - window_size, stride):
        x_window = x[i:i + window_size]
        x_windowed.append(x_window)
        if date_times is not None:
            date_time_windows.append(date_times[i + window_size - 1])

    return np.array(x_windowed), date_time_windows


def generate_positional_encoding(num_features, pos_dim=16):
    """
    Generates positional encoding for a given number of features.

    Args:
        num_features (int): Number of features in the dataset.
        pos_dim (int): Dimension of the positional encoding.

    Returns:
        np.ndarray: Positional encoding of shape (1, num_features * pos_dim).
    """
    position = np.arange(num_features)[:, np.newaxis]
    div_term = np.exp(np.arange(0, pos_dim, 2) * -(np.log(10000.0) / pos_dim))
    pos_encoding = np.zeros((num_features, pos_dim))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    pos_encoding_flat = pos_encoding.flatten().reshape(1, -1)  # Shape: (1, num_features * pos_dim)
    return pos_encoding_flat


def gaussian_kernel_matrix(x, y, sigma):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    x_expanded = tf.reshape(x, [x_size, 1, dim])
    y_expanded = tf.reshape(y, [1, y_size, dim])
    squared_diff = tf.reduce_sum(tf.square(x_expanded - y_expanded), axis=2)
    return tf.exp(-squared_diff / (2.0 * sigma**2))

def combined_loss(y_true, y_pred):
    huber_loss = Huber(delta=1.0)(y_true, y_pred)
    sigma = 1.0
    stat_weight = 1.0
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    K_xx = gaussian_kernel_matrix(y_true_flat, y_true_flat, sigma)
    K_yy = gaussian_kernel_matrix(y_pred_flat, y_pred_flat, sigma)
    K_xy = gaussian_kernel_matrix(y_true_flat, y_pred_flat, sigma)
    m = tf.cast(tf.shape(y_true_flat)[0], tf.float32)
    n = tf.cast(tf.shape(y_pred_flat)[0], tf.float32)
    mmd = tf.reduce_sum(K_xx) / (m * m) + tf.reduce_sum(K_yy) / (n * n) - 2 * tf.reduce_sum(K_xy) / (m * n)
    return huber_loss + stat_weight * mmd

def mmd_metric(y_true, y_pred):
    sigma = 1.0
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    K_xx = gaussian_kernel_matrix(y_true_flat, y_true_flat, sigma)
    K_yy = gaussian_kernel_matrix(y_pred_flat, y_pred_flat, sigma)
    K_xy = gaussian_kernel_matrix(y_true_flat, y_pred_flat, sigma)
    m = tf.cast(tf.shape(y_true_flat)[0], tf.float32)
    n = tf.cast(tf.shape(y_pred_flat)[0], tf.float32)
    return tf.reduce_sum(K_xx) / (m * m) + tf.reduce_sum(K_yy) / (n * n) - 2 * tf.reduce_sum(K_xy) / (m * n)
mmd_metric.__name__ = "mmd"

def huber_metric(y_true, y_pred):
    return Huber(delta=1.0)(y_true, y_pred)

huber_metric.__name__ = "huber"
