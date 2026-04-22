#!/usr/bin/env python
"""
Phase 2.6 Preprocessor Plugin - Row-by-Row Windowing for Pre-processed Data

This preprocessor plugin is designed to work with data that has already been:
1. Feature engineered (all features pre-calculated and included)
2. Pre-processed and split into D1-D6 datasets

Key processing per USER REQUIREMENTS:
- Takes precomputed features row-by-row to compose sliding windows
- No additional feature generation or processing (all done beforehand)
- Strict sliding window: data[t-window_size : t] (EXCLUDES current tick t to prevent data leakage)
- Always calculates targets as returns: CLOSE[t+horizon] - CLOSE[t]
- Maintains strict causality with no data leakage
"""

import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler


def denormalize_close(normalized_data, config):
    """Denormalizes z-score normalized CLOSE data using normalization config."""
    use_normalization_json = config.get("use_normalization_json")
    if use_normalization_json and os.path.exists(use_normalization_json):
        try:
            with open(use_normalization_json, 'r') as f:
                norm_json = json.load(f)
            if isinstance(norm_json, dict) and "CLOSE" in norm_json:
                close_mean = norm_json["CLOSE"]["mean"]
                close_std = norm_json["CLOSE"]["std"]
                return (normalized_data * close_std) + close_mean
            else:
                print(f"WARN: CLOSE normalization data not found in {use_normalization_json}")
                return normalized_data
        except Exception as e:
            print(f"WARN: Error loading normalization config: {e}")
            return normalized_data
    else:
        print(f"WARN: Normalization config file not found: {use_normalization_json}")
        return normalized_data


class PreprocessorPlugin:
    # Default plugin parameters optimized for phase 2.6 preprocessed data
    plugin_params = {
        # --- File Paths ---
        "x_train_file": "examples/data/phase_2_6/normalized_d4.csv",
        "x_validation_file": "examples/data/phase_2_6/normalized_d5.csv", 
        "x_test_file": "examples/data/phase_2_6/normalized_d6.csv",
        # --- Data Loading ---
        "headers": True,
        "max_steps_train": None, "max_steps_val": None, "max_steps_test": None,
        "target_column": "CLOSE", # Target column for prediction
        # --- Windowing & Horizons ---
        "window_size": 288, # Default window size for phase 2.6
        "predicted_horizons": [24, 48, 72, 96, 120, 144], # Multi-horizon support
        # --- Phase 2.6 specific parameters ---
        "date_column": "DATE_TIME", # Date column name
    }
    
    plugin_debug_vars = [
        "window_size", "predicted_horizons", "target_column", "exclude_features", "exclude_from_mtm"
    ]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        """Update plugin parameters with global configuration."""
        for key, value in kwargs.items(): 
            self.params[key] = value
        # No parameter resolution needed for preprocessed data

    def get_debug_info(self): 
        return {var: self.params.get(var) for var in self.plugin_debug_vars}
    
    def add_debug_info(self, debug_info): 
        debug_info.update(self.get_debug_info())



    def _load_data(self, file_path, max_rows, headers):
        """Loads CSV file for preprocessed data."""
        print(f"Loading preprocessed data from {file_path}...", end="")
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            df = pd.read_csv(file_path, nrows=max_rows, header=0 if headers else None)
            
            # Try to parse DATE_TIME column as datetime index
            if self.params.get("date_column", "DATE_TIME") in df.columns:
                try:
                    df[self.params["date_column"]] = pd.to_datetime(df[self.params["date_column"]])
                    df.set_index(self.params["date_column"], inplace=True)
                    print(" OK (with datetime index).")
                except Exception as e:
                    print(f" OK (datetime parsing failed: {e}).")
            else:
                print(" OK (no date column found).")
            
            # Validate that we have the expected target column
            target_col_name = self.params.get("target_column", "CLOSE")
            if target_col_name not in df.columns:
                raise ValueError(f"Target column '{target_col_name}' not found in {file_path}")
                
            print(f" Shape: {df.shape}, Columns: {len(df.columns)}")
            return df
            
        except FileNotFoundError:
            print(f"\nERROR: File not found: {file_path}")
            raise
        except Exception as e:
            print(f"\nERROR loading/processing {file_path}: {e}")
            raise

    def create_sliding_windows(self, data, window_size, date_times=None, max_horizon=1):
        """
        Creates sliding windows for feature data with STRICT CAUSALITY per user requirements.
        
        USER REQUIREMENTS:
        - For each tick t (starting from window_size), take previous window_size ticks as sliding window
        - Window: data[t-window_size : t] (EXCLUDES current tick t to prevent data leakage)
        - Prediction timestamp: t (current tick)
        - The sliding window fed to model MUST NOT include the current tick to maintain causality
        - Must ensure enough data remains for target calculation at max_horizon
        
        Args:
            data: 2D numpy array (n_samples, n_features) or 1D array for single feature
            window_size: Size of the sliding window
            date_times: Optional datetime array
            max_horizon: Maximum prediction horizon to ensure enough future data
        """
        print(f"Creating sliding windows (USER REQUIREMENTS - Size={window_size}, EXCLUDING current tick, max_horizon={max_horizon})...", end="")
        
        # Handle both 1D and 2D data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        windows = []
        date_windows = []
        
        # USER REQUIREMENTS: Start from tick window_size (so we have window_size previous ticks)
        # and create windows that EXCLUDE the current tick to prevent data leakage
        # For tick t, window is data[t-window_size : t] (so it has exactly window_size elements BEFORE tick t)
        start_tick = window_size  # First tick where we can create a full window (0-indexed)
        
        # CRITICAL: Ensure we have enough future data for target calculation
        # Last tick we can use is n_samples - max_horizon - 1 (so target at t+max_horizon is valid)
        end_tick = n_samples - max_horizon
        num_possible_windows = end_tick - start_tick
        
        if num_possible_windows <= 0:
             print(f" WARN: Data short ({n_samples}) for window_size={window_size} and max_horizon={max_horizon}. No windows.")
             return np.array(windows, dtype=np.float32), np.array(date_windows, dtype=object)
             
        for t in range(start_tick, end_tick):
            # USER REQUIREMENTS: Window from [t-window_size : t] (EXCLUDES current tick t)
            # This gives us exactly window_size elements: data[t-window_size], ..., data[t-2], data[t-1]
            window_start = t - window_size
            window_end = t  # Exclusive, so we get [t-window_size : t]
            window = data[window_start:window_end]  # Shape: (window_size, n_features)
            
            # Verify window has correct size
            if window.shape[0] != window_size:
                print(f" ERROR: Window at tick {t} has size {window.shape[0]}, expected {window_size}")
                continue
            
            windows.append(window)
            
            if date_times is not None:
                # PREDICTION TIMESTAMP: Current tick t
                if t < len(date_times): 
                    date_windows.append(date_times[t])
                else: 
                    date_windows.append(None)
                    
        # Convert to arrays
        if windows:
            windows = np.array(windows, dtype=np.float32)  # Shape: (n_windows, window_size, n_features)
        else:
            windows = np.array([], dtype=np.float32).reshape(0, window_size, n_features)
            
        # Convert dates
        if date_times is not None:
             date_windows_arr = np.array(date_windows, dtype=object)
             if all(isinstance(d, pd.Timestamp) for d in date_windows if d is not None):
                  try: date_windows_arr = np.array(date_windows, dtype='datetime64[ns]')
                  except (ValueError, TypeError): pass
        else: 
            date_windows_arr = np.array(date_windows, dtype=object)
            
        print(f" Done ({len(windows)} windows, prediction timestamps from tick {start_tick} to {end_tick-1}).")
        return windows, date_windows_arr

    def _apply_causal_mtm_decomposition(self, data, window_size, n_components, data_name):
        """
        Apply STRICTLY causal MTM decomposition to remove current-value bias.
        
        CRITICAL FIX: This method now uses ONLY PAST data for decomposition,
        ensuring the model learns patterns rather than current values.
        
        Args:
            data: 1D numpy array of feature values
            window_size: Size of MTM analysis window
            n_components: Number of frequency components to extract
            data_name: Name for debugging
            
        Returns:
            2D numpy array (n_samples, n_components) with MTM frequency components
        """
        print(f"    STRICTLY CAUSAL MTM decomposing {data_name} (length={len(data)}, window={window_size}, components={n_components})...")
        
        if len(data) < window_size + 1:  # Need at least window_size + 1 for causal analysis
            print(f"    WARNING: Data too short for strictly causal MTM, using zero components")
            return np.zeros((len(data), n_components), dtype=np.float32)
        
        n_samples = len(data)
        mtm_components = np.zeros((n_samples, n_components), dtype=np.float32)
        
        # CRITICAL FIX: For each time point, use ONLY PREVIOUS data (strictly causal)
        for t in range(n_samples):
            if t < window_size:
                # Not enough past data - use zeros (no predictive information)
                mtm_components[t, :] = 0.0
                continue
            
            # STRICTLY CAUSAL: Use ONLY past data [t-window_size : t] (EXCLUDES current tick t)
            # This ensures we NEVER use current or future information
            window_start = t - window_size
            window_end = t  # Exclusive - does NOT include current tick
            past_data = data[window_start:window_end]
            
            if len(past_data) != window_size:
                print(f"    ERROR: Past data length {len(past_data)} != window_size {window_size}")
                mtm_components[t, :] = 0.0
                continue
            
            # Apply frequency decomposition to PAST data only
            try:
                # Calculate differences (returns) of past data to remove level bias
                past_returns = np.diff(past_data)
                if len(past_returns) == 0:
                    mtm_components[t, :] = 0.0
                    continue
                
                # Pad returns to window_size for consistent FFT
                padded_returns = np.zeros(window_size)
                padded_returns[:len(past_returns)] = past_returns
                
                # Apply FFT to past returns (not levels)
                fft_result = np.fft.fft(padded_returns)
                
                # Extract magnitude of first n_components frequencies
                freq_magnitudes = np.abs(fft_result[:n_components])
                
                # Normalize to prevent scale bias
                if np.sum(freq_magnitudes) > 0:
                    freq_magnitudes = freq_magnitudes / np.sum(freq_magnitudes)
                else:
                    freq_magnitudes = np.ones(n_components) / n_components  # Uniform if no signal
                
                mtm_components[t, :] = freq_magnitudes.astype(np.float32)
                
            except Exception as e:
                print(f"    WARNING: Strictly causal MTM failed at t={t}, using zeros: {e}")
                mtm_components[t, :] = 0.0
        
        # Apply differencing to MTM components to remove any remaining level bias
        mtm_differenced = np.zeros_like(mtm_components)
        for comp in range(n_components):
            comp_data = mtm_components[:, comp]
            # First value is zero (no past information)
            mtm_differenced[0, comp] = 0.0
            # Subsequent values are differences
            mtm_differenced[1:, comp] = np.diff(comp_data)
        
        # Final normalization using StandardScaler for consistency
        scaler = StandardScaler()
        # Handle edge case where all values might be zero
        if np.any(mtm_differenced != 0):
            mtm_normalized = scaler.fit_transform(mtm_differenced).astype(np.float32)
        else:
            mtm_normalized = mtm_differenced.astype(np.float32)
        
        print(f"    STRICTLY CAUSAL MTM complete: {data_name} -> {mtm_normalized.shape} frequency components")
        print(f"    Component stats: mean={mtm_normalized.mean():.6f}, std={mtm_normalized.std():.6f}")
        print(f"    BIAS ELIMINATED: Uses only past returns, not current levels")
        
        return mtm_normalized


    def process_data(self, config):
        """
        Processes preprocessed z-score normalized data, but ensures the exact same feature stacking, ordering, and logic as the STL preprocessor.
        """
        print("\n" + "="*15 + " Starting Phase 2.6 Preprocessing (STL-Compatible) " + "="*15)
        self.set_params(**config)
        config = self.params

        # Get key parameters
        window_size = config['window_size']
        predicted_horizons = config['predicted_horizons']
        if not isinstance(predicted_horizons, list) or not predicted_horizons:
            raise ValueError("'predicted_horizons' must be a non-empty list.")
        max_horizon = max(predicted_horizons)

        # --- 1. Load Preprocessed Data ---
        print("\n--- 1. Loading Preprocessed Data ---")
        x_train_df = self._load_data(config["x_train_file"], config.get("max_steps_train"), config.get("headers"))
        x_val_df = self._load_data(config["x_validation_file"], config.get("max_steps_val"), config.get("headers"))
        x_test_df = self._load_data(config["x_test_file"], config.get("max_steps_test"), config.get("headers"))

        # --- 2. Calculate log_return Feature (Exact STL Method) ---
        print("\n--- 2. Calculate log_return Feature (Exact STL Method) ---")
        target_column = config["target_column"]
        
        # Extract CLOSE values and denormalize them first
        print("Extracting and denormalizing CLOSE values for log return calculation...")
        close_train_normalized = x_train_df[target_column].values.astype(np.float32)
        close_val_normalized = x_val_df[target_column].values.astype(np.float32)
        close_test_normalized = x_test_df[target_column].values.astype(np.float32)
        
        # Denormalize CLOSE values to get actual prices
        close_train = denormalize_close(close_train_normalized, config)
        close_val = denormalize_close(close_val_normalized, config)
        close_test = denormalize_close(close_test_normalized, config)
        
        # EXACT STL METHOD: Apply log transform with safety (matching STL exactly)
        log_train = np.log1p(np.maximum(0, close_train))
        log_val = np.log1p(np.maximum(0, close_val))
        log_test = np.log1p(np.maximum(0, close_test))
        print(f"Log transform applied. Train shape: {log_train.shape}")
        
        # EXACT STL METHOD: Calculate log returns (matching STL exactly)
        log_ret_train = np.diff(log_train, prepend=log_train[0])
        log_ret_val = np.diff(log_val, prepend=log_val[0])
        log_ret_test = np.diff(log_test, prepend=log_test[0])
        
        # EXACT STL METHOD: Normalize using StandardScaler (matching STL exactly)
        # Fit scaler on training data
        scaler = StandardScaler()
        log_ret_train_normalized = scaler.fit_transform(log_ret_train.reshape(-1, 1)).flatten().astype(np.float32)
        log_ret_val_normalized = scaler.transform(log_ret_val.reshape(-1, 1)).flatten().astype(np.float32)
        log_ret_test_normalized = scaler.transform(log_ret_test.reshape(-1, 1)).flatten().astype(np.float32)
        
        # Add normalized log_return to dataframes (using STL naming convention)
        x_train_df['log_return'] = log_ret_train_normalized
        x_val_df['log_return'] = log_ret_val_normalized
        x_test_df['log_return'] = log_ret_test_normalized
        
        print(f"STL METHOD: log_return calculated and normalized using StandardScaler")
        print(f"Train log_return stats: mean={log_ret_train_normalized.mean():.6f}, std={log_ret_train_normalized.std():.6f}")
        print(f"Val log_return stats: mean={log_ret_val_normalized.mean():.6f}, std={log_ret_val_normalized.std():.6f}")
        print(f"Test log_return stats: mean={log_ret_test_normalized.mean():.6f}, std={log_ret_test_normalized.std():.6f}")

        # --- 2.5. OPTIONAL CAUSAL MTM DECOMPOSITION ---
        print("\n--- 2.5. OPTIONAL CAUSAL MTM DECOMPOSITION ---")
        
        # Check if MTM decomposition is enabled
        use_mtm_for_all = config.get('use_mtm_for_all_features', False)
        
        if not use_mtm_for_all:
            print("MTM decomposition DISABLED - using original features")
            print("This avoids potential current-value bias from frequency analysis")
            
            # Get all feature columns (excluding target) and apply exclusions
            all_feature_columns = [col for col in x_train_df.columns if col != target_column]
            
            # Apply feature exclusion from global config
            exclude_features = config.get('exclude_features', [])
            if exclude_features:
                print(f"Excluding features from config: {exclude_features}")
                excluded_features = [col for col in all_feature_columns if col in exclude_features]
                all_feature_columns = [col for col in all_feature_columns if col not in exclude_features]
                if excluded_features:
                    print(f"Features excluded: {excluded_features}")
                    # Remove excluded features from dataframes
                    x_train_df = x_train_df.drop(columns=excluded_features, errors='ignore')
                    x_val_df = x_val_df.drop(columns=excluded_features, errors='ignore')
                    x_test_df = x_test_df.drop(columns=excluded_features, errors='ignore')
            
            print(f"TRADITIONAL PREPROCESSING: Using {len(all_feature_columns)} original features")
            print(f"Features preserved: log_return calculation, normalization, feature ordering")
            
        else:
            print("MTM decomposition ENABLED - applying frequency analysis to features")
            print("This transforms features into frequency components to reduce current-value bias")
            
            # Get all feature columns (excluding target)
            all_feature_columns = [col for col in x_train_df.columns if col != target_column]
            
            # Apply feature exclusion from MTM decomposition ONLY (keep features in sliding windows)
            exclude_from_mtm = config.get('exclude_from_mtm', [])
            if exclude_from_mtm:
                print(f"Excluding features from MTM decomposition: {exclude_from_mtm}")
                excluded_from_mtm = [col for col in all_feature_columns if col in exclude_from_mtm]
                mtm_feature_columns = [col for col in all_feature_columns if col not in exclude_from_mtm]
                if excluded_from_mtm:
                    print(f"Features excluded from MTM: {excluded_from_mtm}")
            else:
                mtm_feature_columns = all_feature_columns
            
            # Determine which features to apply MTM to
            features_for_mtm = mtm_feature_columns
            print(f"Applying MTM to ALL {len(features_for_mtm)} features (excluding decomposition features)")
            
            # MTM parameters
            mtm_window_size = config.get('window_size', 288)
            mtm_components = 5
            
            print(f"MTM Parameters: window_size={mtm_window_size}, components={mtm_components}")
            
            # Apply MTM decomposition to all eligible features
            mtm_train_dict = {}
            mtm_val_dict = {}
            mtm_test_dict = {}
            
            # Apply MTM decomposition to each feature
            for feature_name in features_for_mtm:
                print(f"  Decomposing feature: {feature_name}...")
                
                feature_train = x_train_df[feature_name].values.astype(np.float32)
                feature_val = x_val_df[feature_name].values.astype(np.float32)
                feature_test = x_test_df[feature_name].values.astype(np.float32)
                
                mtm_train_components = self._apply_causal_mtm_decomposition(
                    feature_train, mtm_window_size, mtm_components, f"{feature_name}_train")
                mtm_val_components = self._apply_causal_mtm_decomposition(
                    feature_val, mtm_window_size, mtm_components, f"{feature_name}_val")
                mtm_test_components = self._apply_causal_mtm_decomposition(
                    feature_test, mtm_window_size, mtm_components, f"{feature_name}_test")
                
                for comp_idx in range(mtm_components):
                    comp_name = f"{feature_name}_mtm_{comp_idx+1}"
                    mtm_train_dict[comp_name] = mtm_train_components[:, comp_idx]
                    mtm_val_dict[comp_name] = mtm_val_components[:, comp_idx]
                    mtm_test_dict[comp_name] = mtm_test_components[:, comp_idx]
            
            # Add excluded features back (the ones we preserved without MTM)
            exclude_from_mtm = config.get('exclude_from_mtm', [])
            for feature_name in exclude_from_mtm:
                if feature_name in x_train_df.columns:
                    mtm_train_dict[feature_name] = x_train_df[feature_name].values
                    mtm_val_dict[feature_name] = x_val_df[feature_name].values
                    mtm_test_dict[feature_name] = x_test_df[feature_name].values
            
            # Create new dataframes with MTM components + excluded features
            train_index = x_train_df.index
            val_index = x_val_df.index
            test_index = x_test_df.index
            
            x_train_df = pd.DataFrame(mtm_train_dict, index=train_index)
            x_val_df = pd.DataFrame(mtm_val_dict, index=val_index)
            x_test_df = pd.DataFrame(mtm_test_dict, index=test_index)
            
            # Add target column back
            x_train_df[target_column] = close_train_normalized
            x_val_df[target_column] = close_val_normalized
            x_test_df[target_column] = close_test_normalized
            
            all_feature_columns = list(mtm_train_dict.keys())
            
            print(f"MTM PROCESSING COMPLETE: {len(features_for_mtm)} features decomposed into {len(features_for_mtm)*mtm_components} MTM components")
            print(f"EXCLUDED FEATURES PRESERVED: {len(exclude_from_mtm)} existing decomposition features kept as-is")
            print(f"Total features: {len(all_feature_columns)}")
        print("\n--- 3. Extract Target and Organize Features ---")
        target_column = config["target_column"]
        if target_column not in x_train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Extract dates
        dates_train = x_train_df.index if isinstance(x_train_df.index, pd.DatetimeIndex) else None
        dates_val = x_val_df.index if isinstance(x_val_df.index, pd.DatetimeIndex) else None
        dates_test = x_test_df.index if isinstance(x_test_df.index, pd.DatetimeIndex) else None
        
        # Extract target values (CLOSE) - these are already preprocessed and aligned
        close_train_normalized = x_train_df[target_column].astype(np.float32).values
        close_val_normalized = x_val_df[target_column].astype(np.float32).values
        close_test_normalized = x_test_df[target_column].astype(np.float32).values
        
        # CRITICAL FIX: Use ALREADY denormalized CLOSE values from step 2 (no double denormalization!)
        print(f"CRITICAL FIX: Using already denormalized CLOSE values from step 2 (avoiding double denormalization)...")
        # close_train, close_val, close_test were already denormalized in step 2 - reuse them!
        
        print(f"Already denormalized CLOSE stats (from step 2):")
        print(f"  Train: mean={close_train.mean():.6f}, std={close_train.std():.6f}, range=[{close_train.min():.6f}, {close_train.max():.6f}]")
        print(f"  Val: mean={close_val.mean():.6f}, std={close_val.std():.6f}, range=[{close_val.min():.6f}, {close_val.max():.6f}]")
        print(f"  Test: mean={close_test.mean():.6f}, std={close_test.std():.6f}, range=[{close_test.min():.6f}, {close_test.max():.6f}]")
        
        # Remove target column from features and enforce MTM-enhanced feature ordering
        all_feature_columns = [col for col in x_train_df.columns if col != target_column]
        
        # NOTE: Feature exclusion was already applied before MTM decomposition above
        # all_feature_columns now contains only MTM components (no excluded features)
        
        # CRITICAL FIX: Optimize MTM feature ordering for Conv1D pattern detection
        # Group MTM components by original feature, then by frequency component
        feature_columns = []
        
        # Check if log_return MTM components are available
        log_return_mtm_features = [col for col in all_feature_columns if col.startswith('log_return_mtm_')]
        log_return_available = len(log_return_mtm_features) > 0
        
        if log_return_available:
            # 1. log_return MTM components go first (calculated using exact STL method)
            log_return_mtm_features.sort(key=lambda x: int(x.split('_')[-1]))  # Sort by component number
            feature_columns.extend(log_return_mtm_features)
            remaining_features = [col for col in all_feature_columns if not col.startswith('log_return_mtm_')]
            print(f"Using log_return MTM components as anchor features: {log_return_mtm_features}")
        else:
            # log_return was excluded by user - proceed without it
            remaining_features = [col for col in all_feature_columns]
            print("WARNING: log_return MTM components not available - proceeding without anchor features")
        
        # 2. Group remaining MTM features by original feature name
        print("Organizing features for Conv1D-optimized ordering...")
        
        if remaining_features:
            # Group features by type and original feature
            feature_groups = {}
            for col in remaining_features:
                if '_mtm_' in col:
                    # Extract original feature name (everything before _mtm_)
                    base_feature = col.split('_mtm_')[0]
                    
                    # Handle different MTM formats
                    try:
                        mtm_part = col.split('_mtm_')[1]
                        if mtm_part.isdigit():
                            # Simple format: feature_mtm_1
                            comp_num = int(mtm_part)
                        else:
                            # Complex format: feature_mtm_band_1_0.000_0.010
                            # Extract band number if available
                            if 'band_' in mtm_part:
                                band_part = mtm_part.split('band_')[1]
                                comp_num = int(band_part.split('_')[0])
                            else:
                                comp_num = 0  # Default for unparseable MTM features
                    except (ValueError, IndexError):
                        comp_num = 0  # Default for unparseable MTM features
                    
                    if base_feature not in feature_groups:
                        feature_groups[base_feature] = []
                    feature_groups[base_feature].append((col, comp_num))
                else:
                    # Non-MTM feature
                    if 'other' not in feature_groups:
                        feature_groups['other'] = []
                    feature_groups['other'].append((col, 0))
            
            # Calculate correlation of each feature group with log_return MTM (if available)
            if log_return_available and feature_groups:
                print("Calculating feature group correlations for Conv1D optimization...")
                
                # Get first log_return MTM component as reference
                reference_feature = log_return_mtm_features[0]
                reference_data = x_train_df[reference_feature].astype(np.float32)
                
                # Calculate average correlation for each feature group
                group_correlations = {}
                for base_feature, components in feature_groups.items():
                    correlations = []
                    for comp_name, comp_num in components:
                        comp_data = x_train_df[comp_name].astype(np.float32)
                        corr = np.corrcoef(reference_data, comp_data)[0, 1]
                        if np.isfinite(corr):
                            correlations.append(abs(corr))
                    
                    # Average correlation for this feature group
                    avg_corr = np.mean(correlations) if correlations else 0.0
                    group_correlations[base_feature] = avg_corr
                
                # Sort feature groups by correlation (descending)
                sorted_groups = sorted(group_correlations.items(), key=lambda x: x[1], reverse=True)
                print(f"Feature groups ordered by correlation with {reference_feature}:")
                for base_feature, corr in sorted_groups[:5]:  # Show top 5
                    print(f"  {base_feature}: {corr:.4f}")
                
            else:
                # No log_return available - use alphabetical ordering
                sorted_groups = [(base_feature, 0.0) for base_feature in sorted(feature_groups.keys())]
                print("No log_return reference - using alphabetical feature group ordering")
            
            # Build final feature order: for each group, add components in order
            mtm_ordered_features = []
            for base_feature, _ in sorted_groups:
                # Sort components within group by component number
                group_components = feature_groups[base_feature]
                group_components.sort(key=lambda x: x[1])  # Sort by component number
                mtm_ordered_features.extend([comp_name for comp_name, _ in group_components])
            
            feature_columns.extend(mtm_ordered_features)
            
            print(f"MTM-CONV1D OPTIMIZED: Feature ordering for frequency pattern detection:")
            if log_return_available:
                print(f"  1. log_return MTM components (anchor): {len(log_return_mtm_features)} components")
            print(f"  2. Feature groups ordered by correlation: {len(sorted_groups)} groups")
            print(f"  3. Within groups: MTM components 1-5 in frequency order")
            print(f"  4. Total MTM features: {len(feature_columns)}")
            print(f"  5. Current-value bias eliminated: All features are frequency patterns")
            
        else:
            print("WARNING: No remaining MTM features - only log_return MTM components available")
        
        # CRITICAL: Ensure CLOSE column is definitively excluded from sliding windows
        # Debug: Check if CLOSE is somehow still in feature_columns
        print(f"DEBUG: Before CLOSE removal - feature_columns contains CLOSE: {target_column in feature_columns}")
        print(f"DEBUG: feature_columns length before CLOSE removal: {len(feature_columns)}")
        if target_column in feature_columns:
            feature_columns.remove(target_column)
            print(f"REMOVED: {target_column} column excluded from sliding windows to reduce features from 55 to 54")
        print(f"DEBUG: After CLOSE removal - feature_columns contains CLOSE: {target_column in feature_columns}")
        print(f"DEBUG: feature_columns length after CLOSE removal: {len(feature_columns)}")
         
        # Extract features (already preprocessed and normalized)
        features_train = x_train_df[feature_columns].astype(np.float32).values
        features_val = x_val_df[feature_columns].astype(np.float32).values  
        features_test = x_test_df[feature_columns].astype(np.float32).values
        
        print(f"Loaded preprocessed data with {features_train.shape[1]} features")
        print(f"Feature columns: {feature_columns}")
        print(f"Data shapes - Train: {features_train.shape}, Val: {features_val.shape}, Test: {features_test.shape}")
        print(f"CLOSE shapes - Train: {close_train.shape}, Val: {close_val.shape}, Test: {close_test.shape}")
    
        # Verify features and CLOSE target
        has_log_return_mtm = any(col.startswith('log_return_mtm_') for col in feature_columns)
        has_log_return_original = 'log_return' in feature_columns
        has_close = target_column in feature_columns
        has_old_logreturn = 'logreturn' in feature_columns
        has_old_close_logreturn = 'close_logreturn' in feature_columns
        
        # Check MTM usage
        use_mtm_for_all = config.get('use_mtm_for_all_features', False)
        using_mtm = use_mtm_for_all
        
        # Check if log_return was excluded by user
        log_return_excluded = 'log_return' in config.get('exclude_features', [])
        
        print(f"FEATURE VERIFICATION:")
        print(f"  - CLOSE column removed from features: {not has_close}")
        print(f"  - Using MTM decomposition: {using_mtm}")
        
        if using_mtm:
            if log_return_excluded:
                print(f"  - log_return MTM excluded by user: {log_return_excluded}")
            else:
                print(f"  - log_return MTM components available: {has_log_return_mtm}")
            
            if use_mtm_for_all:
                all_mtm = all('_mtm_' in col for col in feature_columns)
                print(f"  - All features are MTM components: {all_mtm}")
                print(f"  - Current-value bias eliminated via frequency analysis: {all_mtm}")
            else:
                mtm_count = sum(1 for col in feature_columns if '_mtm_' in col)
                original_count = len(feature_columns) - mtm_count
                print(f"  - Hybrid approach: {mtm_count} MTM + {original_count} original features")
                print(f"  - Partial bias reduction via selective MTM: True")
        else:
            if log_return_excluded:
                print(f"  - log_return excluded by user: {log_return_excluded}")
            else:
                print(f"  - log_return original feature available: {has_log_return_original}")
            print(f"  - Traditional preprocessing: Using original normalized features")
            print(f"  - Current-value bias: May be present (no MTM filtering)")
        
        print(f"  - Old 'logreturn' removed: {not has_old_logreturn}")
        print(f"  - Old 'close_logreturn' removed: {not has_old_close_logreturn}")
        
        # Verification based on configuration
        if using_mtm:
            if not log_return_excluded and not has_log_return_mtm:
                raise ValueError("log_return MTM components not found (and not excluded by user)!")
        else:
            # Traditional processing
            if not log_return_excluded and not has_log_return_original:
                raise ValueError("log_return feature not found (and not excluded by user)!")
        
        if has_close:
            raise ValueError(f"Target column '{target_column}' should not be in features!")
        if has_old_logreturn:
            raise ValueError("Old 'logreturn' column should be removed from features!")
        if has_old_close_logreturn:
            raise ValueError("Old 'close_logreturn' column should be removed from features!")
        
        print(f"VERIFIED: Feature configuration matches settings")
        print(f"VERIFIED: {len(feature_columns)} features ready for model training")

        # --- 4. Create Sliding Windows (USER REQUIREMENTS) ---
        print("\n--- 4. Creating Sliding Windows (USER REQUIREMENTS) ---")
        
        # USER REQUIREMENTS: For each tick t, window EXCLUDES current tick to prevent data leakage
        # Window: data[t-window_size : t] (EXCLUDES current tick t)
        
        # Create windows for features
        X_train_windows, train_dates_windows = self.create_sliding_windows(
            features_train, window_size, dates_train, max_horizon)
        X_val_windows, val_dates_windows = self.create_sliding_windows(
            features_val, window_size, dates_val, max_horizon)
        X_test_windows, test_dates_windows = self.create_sliding_windows(
            features_test, window_size, dates_test, max_horizon)
        
        # The windows are already in the correct shape: (samples, window_size, features)
        X_train_combined = X_train_windows
        X_val_combined = X_val_windows
        X_test_combined = X_test_windows
        
        print(f"Final X shapes: Train={X_train_combined.shape}, Val={X_val_combined.shape}, Test={X_test_combined.shape}")
        print(f"Feature columns: {feature_columns}")
        
        # DEBUG: Print feature information
        print(f"\nDEBUG: Complete feature list ({len(feature_columns)} features):")
        
        using_mtm = config.get('use_mtm_for_all_features', False)
        
        for i, feature_name in enumerate(feature_columns):
            feature_info = ""
            if feature_name.startswith('log_return_mtm_'):
                feature_info = " (log_return MTM frequency component)"
            elif '_mtm_' in feature_name:
                base_feature = feature_name.split('_mtm_')[0]
                comp_num = feature_name.split('_mtm_')[1]
                feature_info = f" (MTM freq component {comp_num} of {base_feature})"
            elif feature_name == 'log_return':
                feature_info = " (original log_return STL method)"
            else:
                feature_info = " (original normalized feature)"
            print(f"  Feature {i+1:2d}: {feature_name}{feature_info}")
        
        print(f"DEBUG: Total features in sliding windows: {len(feature_columns)}")
        if X_train_combined.shape[2] != len(feature_columns):
            print(f"ERROR: Mismatch between feature_columns length ({len(feature_columns)}) and sliding window features ({X_train_combined.shape[2]})")
        else:
            print(f"VERIFIED: Sliding window feature count matches feature_columns list")
            
            if using_mtm:
                print(f"VERIFIED: MTM frequency features optimized for pattern detection")
                print(f"VERIFIED: Current-value bias reduced through frequency decomposition")
            else:
                print(f"VERIFIED: Traditional features ordered for Conv1D spatial detection")
                print(f"NOTE: Current-value bias may be present (MTM disabled)")
        
        # --- 5. Baseline and Target Calculation (USER REQUIREMENTS) ---
        print("\n--- 5. Calculating Baselines and Targets (USER REQUIREMENTS) ---")
        
        # Get number of samples from windowing
        num_samples_train = X_train_combined.shape[0]
        num_samples_val = X_val_combined.shape[0]
        num_samples_test = X_test_combined.shape[0]
        
        # USER REQUIREMENTS: Baseline and Target Calculation
        # Windows start at tick (window_size) and go up to tick (n-1)
        # For each window, the prediction timestamp is the current tick t
        # Baseline should be CLOSE[t] (current tick value)
        # Target should be CLOSE[t+horizon] - CLOSE[t] (future value - current value)
        
        baseline_start_idx = window_size  # Updated to match the new windowing logic
        
        print(f"USER REQUIREMENTS: Baseline calculation")
        print(f"  Window size: {window_size}")
        print(f"  Baseline start index: {baseline_start_idx}")
        print(f"  Number of samples: Train={num_samples_train}, Val={num_samples_val}, Test={num_samples_test}")
        
        # USER REQUIREMENTS: Calculate baseline indices
        # For window i (i=0 to num_samples-1), prediction timestamp is at baseline_start_idx + i
        # Baseline should be CLOSE[prediction_timestamp] = CLOSE[baseline_start_idx + i]
        baseline_train_indices = [baseline_start_idx + i for i in range(num_samples_train)]
        baseline_val_indices = [baseline_start_idx + i for i in range(num_samples_val)]  
        baseline_test_indices = [baseline_start_idx + i for i in range(num_samples_test)]
        
        # Verify indices are valid
        if max(baseline_train_indices) >= len(close_train):
            raise ValueError(f"Baseline train indices out of bounds. Max needed: {max(baseline_train_indices)}, Available: {len(close_train)}")
        if max(baseline_val_indices) >= len(close_val):
            raise ValueError(f"Baseline val indices out of bounds. Max needed: {max(baseline_val_indices)}, Available: {len(close_val)}")
        if max(baseline_test_indices) >= len(close_test):
            raise ValueError(f"Baseline test indices out of bounds. Max needed: {max(baseline_test_indices)}, Available: {len(close_test)}")
        
        baseline_train = close_train[baseline_train_indices]
        baseline_val = close_val[baseline_val_indices]
        baseline_test = close_test[baseline_test_indices]
        
        print(f"Baseline shapes (USER REQUIREMENTS): Train={baseline_train.shape}, Val={baseline_val.shape}, Test={baseline_test.shape}")
        
        # --- 6. Target Calculation (USER REQUIREMENTS - Always Returns) ---
        print("\n--- 6. Target Calculation (USER REQUIREMENTS - Always Returns) ---")
        # USER REQUIREMENTS: Always calculate targets as returns: CLOSE[t+horizon] - CLOSE[t]
        y_train_list = []; y_val_list = []; y_test_list = []
        print(f"Processing targets per USER REQUIREMENTS for horizons: {predicted_horizons} (Always using returns)...")
        print(f"Max horizon: {max_horizon}, ensuring enough future data for all targets")
        
        for h_idx, h in enumerate(predicted_horizons):
            print(f"  Horizon {h_idx+1}/{len(predicted_horizons)}: H={h} - Calculating targets per USER REQUIREMENTS...")
            
            # USER REQUIREMENTS: Target indices
            # For window i (i=0 to num_samples-1), prediction timestamp is at baseline_start_idx + i
            # Target at horizon h is at baseline_start_idx + i + h
            target_train_indices = [baseline_start_idx + i + h for i in range(num_samples_train)]
            target_val_indices = [baseline_start_idx + i + h for i in range(num_samples_val)]
            target_test_indices = [baseline_start_idx + i + h for i in range(num_samples_test)]
            
            # Verify target indices are valid
            if max(target_train_indices) >= len(close_train):
                raise ValueError(f"Target train indices out of bounds for H={h}. Max needed: {max(target_train_indices)}, Available: {len(close_train)}")
            if max(target_val_indices) >= len(close_val):
                raise ValueError(f"Target val indices out of bounds for H={h}. Max needed: {max(target_val_indices)}, Available: {len(close_val)}")
            if max(target_test_indices) >= len(close_test):
                raise ValueError(f"Target test indices out of bounds for H={h}. Max needed: {max(target_test_indices)}, Available: {len(close_test)}")
            
            # Extract target values
            target_train_h_raw = close_train[target_train_indices]
            target_val_h_raw = close_val[target_val_indices]
            target_test_h_raw = close_test[target_test_indices]
            
            # USER REQUIREMENTS: Apply target calculation
            # Target = CLOSE[t+horizon] - CLOSE[t] (future value - current value)
            # This respects causality and prevents future data leakage
            target_train_h = target_train_h_raw - baseline_train
            target_val_h = target_val_h_raw - baseline_val
            target_test_h = target_test_h_raw - baseline_test
            print(f"    USER REQUIREMENTS: target = CLOSE[t+{h}] - CLOSE[t]")
            
            # Validation: Check for any obvious issues
            train_finite = np.isfinite(target_train_h).sum()
            val_finite = np.isfinite(target_val_h).sum()
            test_finite = np.isfinite(target_test_h).sum()
            print(f"    Finite values: Train={train_finite}/{len(target_train_h)}, Val={val_finite}/{len(target_val_h)}, Test={test_finite}/{len(target_test_h)}")
            
            # Add target statistics for verification
            print(f"    Target stats - Train: mean={target_train_h.mean():.6f}, std={target_train_h.std():.6f}")
            print(f"    Target stats - Val: mean={target_val_h.mean():.6f}, std={target_val_h.std():.6f}")
            print(f"    Target stats - Test: mean={target_test_h.mean():.6f}, std={target_test_h.std():.6f}")
            
            y_train_list.append(target_train_h.astype(np.float32))
            y_val_list.append(target_val_h.astype(np.float32))
            y_test_list.append(target_test_h.astype(np.float32))
            
            print(f"    Target shapes: Train={target_train_h.shape}, Val={target_val_h.shape}, Test={target_test_h.shape}")

        # Final verification of target lists
        print(f"\nTARGET VERIFICATION:")
        print(f"  Generated {len(y_train_list)} target sets for {len(predicted_horizons)} horizons")
        print(f"  Expected horizons: {predicted_horizons}")
        print(f"  Target list lengths: Train={len(y_train_list)}, Val={len(y_val_list)}, Test={len(y_test_list)}")
        if len(y_train_list) != len(predicted_horizons):
            raise ValueError(f"Mismatch: {len(y_train_list)} target sets != {len(predicted_horizons)} horizons")
        for i, h in enumerate(predicted_horizons):
            print(f"  Horizon {h}: Train={y_train_list[i].shape}, Val={y_val_list[i].shape}, Test={y_test_list[i].shape}")

        # --- 7. Prepare Date Arrays ---
        y_dates_train = train_dates_windows
        y_dates_val = val_dates_windows
        y_dates_test = test_dates_windows

        # --- 8. Prepare Return Dictionary ---
        print("\n--- 8. Preparing Final Output ---")
        ret = {}
        ret["x_train"] = X_train_combined
        ret["x_val"] = X_val_combined
        ret["x_test"] = X_test_combined
        ret["y_train"] = y_train_list
        ret["y_val"] = y_val_list
        ret["y_test"] = y_test_list
        ret["x_train_dates"] = train_dates_windows
        ret["y_train_dates"] = y_dates_train
        ret["x_val_dates"] = val_dates_windows
        ret["y_val_dates"] = y_dates_val
        ret["x_test_dates"] = test_dates_windows
        ret["y_test_dates"] = y_dates_test
        ret["baseline_train"] = baseline_train  # Current tick CLOSE values per USER REQUIREMENTS
        ret["baseline_val"] = baseline_val      # Current tick CLOSE values per USER REQUIREMENTS
        ret["baseline_test"] = baseline_test    # Current tick CLOSE values per USER REQUIREMENTS
        ret["baseline_train_dates"] = y_dates_train
        ret["baseline_val_dates"] = y_dates_val
        ret["baseline_test_dates"] = y_dates_test
        ret["test_close_prices"] = baseline_test  # USER REQUIREMENTS: baseline_test contains current tick CLOSE values
        ret["feature_names"] = feature_columns
        
        print(f"Final shapes:")
        print(f"  X: Train={X_train_combined.shape}, Val={X_val_combined.shape}, Test={X_test_combined.shape}")
        print(f"  Y: {len(y_train_list)} horizons, Train[0]={y_train_list[0].shape}")
        print(f"  Baselines: Train={baseline_train.shape}, Val={baseline_val.shape}, Test={baseline_test.shape}")
        print(f"  Features: {len(ret['feature_names'])}")
        print("\n" + "="*15 + " Phase 2.6 Preprocessing Finished (USER REQUIREMENTS) " + "="*15)
        return ret

    def run_preprocessing(self, config):
        """Convenience method to execute data processing."""
        # Merge instance defaults with passed config
        run_config = self.params.copy()
        run_config.update(config)
        # Call set_params again AFTER merge to resolve defaults
        self.set_params(**run_config)
        # Run with the fully resolved self.params
        return self.process_data(self.params)

# --- NO if __name__ == '__main__': block ---
