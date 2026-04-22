import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

def load_normalized_csv(config):
    """Step 1: Load normalized CSV data as-is, limiting rows to max_steps_ configurations.
               also returns the corresponding dates.
    """

    data = {}
    dates = {}        
    file_mappings = [
        ('x_train_df', 'x_train_file', 'max_steps_train'),
        ('y_train_df', 'y_train_file', 'max_steps_train'),
        ('x_val_df', 'x_validation_file', 'max_steps_val'),
        ('y_val_df', 'y_validation_file', 'max_steps_val'),
        ('x_test_df', 'x_test_file', 'max_steps_test'),
        ('y_test_df', 'y_test_file', 'max_steps_test')
    ]
    
    import os
    allowed_exts = {'.csv', '.tsv', '.txt'}
    for data_key, file_key, max_steps_key in file_mappings:
        path = config.get(file_key)
        if not path:
            continue
        path = str(path).strip()
        lower = path.lower()
        root, ext = os.path.splitext(lower)

        # Auto-recovery: if a dataset key mistakenly points to a CONFIG/JSON file, try to extract the real CSV path from inside.
        if ext == '.json':
            try:
                with open(path, 'r') as f:
                    embedded_cfg = json.load(f)
                embedded = embedded_cfg.get(file_key)
                if embedded and isinstance(embedded, str):
                    emb_ext = os.path.splitext(embedded.lower())[1]
                    if emb_ext in allowed_exts and os.path.exists(embedded):
                        print(f"AUTO-RECOVER: {file_key} incorrectly pointed to config JSON {path}; using embedded CSV {embedded}")
                        path = embedded
                        lower = path.lower()
                        root, ext = os.path.splitext(lower)
                    else:
                        print(f"SKIP: {file_key} points to JSON ({path}); embedded value invalid or missing; skipping")
                        continue
                else:
                    print(f"SKIP: {file_key} points to JSON ({path}); no embedded replacement; skipping")
                    continue
            except Exception as rec_e:
                print(f"SKIP: {file_key} JSON recovery failed ({path}): {rec_e}; skipping")
                continue

        # Enforce extension whitelist
        if ext not in allowed_exts:
            print(f"SKIP: {file_key} has unsupported extension '{ext}' (allowed: {sorted(allowed_exts)})")
            continue

        if not os.path.exists(path):
            print(f"ERROR: File not found for {file_key}: {path}")
            continue
        try:
            df = pd.read_csv(path, parse_dates=True, index_col=0)
            max_steps_val = config.get(max_steps_key)
            if isinstance(max_steps_val, int) and max_steps_val > 0:
                df = df.head(max_steps_val)
            if df.empty:
                print(f"WARNING: Empty dataframe loaded from {path}")
            else:
                print(f"  Loaded {data_key}: {df.shape}")
            data[data_key] = df
            # Extract DATE_TIME column if present; otherwise use index
            if "DATE_TIME" in df.columns:
                dates[data_key] = df["DATE_TIME"].values
            else:
                try:
                    dates[data_key] = df.index.values
                except Exception:
                    dates[data_key] = None
        except Exception as e:
            print(f"ERROR loading {file_key} ({path}): {e}")
            continue
    if not data:
        raise ValueError("No data files could be loaded - check file paths in config")
    return data, dates

def load_normalization_json(config):
    """Loads normalization parameters from JSON file."""
    if config.get("use_normalization_json"):
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            try:
                with open(norm_json, 'r') as f:
                    norm_json = json.load(f)
                return norm_json
            except Exception as e:
                print(f"WARN: Failed to load norm JSON {norm_json}: {e}")
                return {}
        return norm_json
    return {}

def denormalize(data, norm_json, column_name="typical_price"):
    """Denormalizes data using JSON normalization parameters."""
    data = np.asarray(data)
    if isinstance(norm_json, dict) and column_name in norm_json:
        try:
            if "mean" in norm_json[column_name] and "std" in norm_json[column_name]:
                mean = norm_json[column_name]["mean"]
                std = norm_json[column_name]["std"]
                return (data * std) + mean
            else:
                print(f"WARN: Missing 'mean' or 'std' in norm JSON for {column_name}")
                return data
        except Exception as e:
            print(f"WARN: Error during denormalize for {column_name}: {e}")
            return data
    return data

def denormalize_all_datasets(normalized_data, config):
    """Step 2: Denormalize all datasets using JSON parameters."""
    norm_json = load_normalization_json(config)
    denormalized = {}
        
    for data_key, normalized_df in normalized_data.items():
        denorm_df = normalized_df.copy()
        for column in denorm_df.columns:
            if column in norm_json:
                denorm_df[column] = denormalize(normalized_df[column].values, norm_json, column)
        denormalized[data_key] = denorm_df
        
    return denormalized
    
def exclude_columns_from_datasets(datasets, preprocessor_params, config):
    # Perform selective column exclusion from datasets.
    excluded_columns = config.get("excluded_columns", [])
    if excluded_columns:
        print(f"\n--- Removing excluded columns from datasets: {excluded_columns} ---")
        # Get column names from datasets (returned by preprocessor as feature_names)
        column_names = datasets.get("feature_names", None)
        if column_names is None:
            print("WARNING: No feature_names available from preprocessor, cannot exclude columns by name")
        else:
            print(f"Available columns: {column_names}")
            
            # Find indices of columns to exclude
            excluded_indices = []
            for col_name in excluded_columns:
                if col_name in column_names:
                    excluded_indices.append(column_names.index(col_name))
                    print(f"  Excluding column '{col_name}' at index {column_names.index(col_name)}")
                else:
                    print(f"  WARNING: Column '{col_name}' not found in dataset columns")
            
            if excluded_indices:
                # Remove excluded columns from all datasets (last dimension = features)
                remaining_indices = [i for i in range(len(column_names)) if i not in excluded_indices]
                print(f"  Keeping {len(remaining_indices)} columns out of {len(column_names)} original columns")

                # Fetch arrays
                X_train = datasets.get("x_train")
                X_val = datasets.get("x_val")
                X_test = datasets.get("x_test")
                if X_train is None or X_val is None or X_test is None:
                    print("WARNING: Missing X arrays in datasets; cannot exclude columns")
                else:
                    print(f"  Original shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
                    X_train = X_train[:, :, remaining_indices]
                    X_val = X_val[:, :, remaining_indices]
                    X_test = X_test[:, :, remaining_indices]
                    print(f"  New shapes after exclusion: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
                    # Update datasets
                    datasets["x_train"] = X_train
                    datasets["x_val"] = X_val
                    datasets["x_test"] = X_test
                
                # Update column names in datasets and preprocessor_params for reference
                new_column_names = [column_names[i] for i in remaining_indices]
                datasets["feature_names"] = new_column_names
                preprocessor_params["feature_names"] = new_column_names
                print(f"  Updated column names: {new_column_names}")
            else:
                print("  No valid columns to exclude")
    else:
        print("No columns specified for exclusion")
    return datasets, preprocessor_params