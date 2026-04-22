"""
Anti-Naive-Lock Preprocessing Module

This module implements selective feature preprocessing to prevent naive lock
where the model simply learns to copy input features to outputs.

The module applies different preprocessing strategies based on feature types:
1. Cyclic encoding for temporal features
2. Log returns for raw price features  
3. First differences for trend features
4. Preserve already-stationary technical indicators
5. Special handling for constant daily features

Author: GitHub Copilot
Date: 2025-08-03
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging


#TODO: move to anti_naive_lock.py
def apply_anti_naive_lock_to_datasets(denormalized_data, config):
    """Step 6: Apply anti-naive-lock transformations to FULL denormalized datasets to create processed data."""
    if not config.get("anti_naive_lock_enabled", True):
        return denormalized_data
    
    print("  Applying anti-naive-lock transformations to full denormalized datasets...")
    processed = {}
    
    for split in ['train', 'val', 'test']:
        x_key = f'x_{split}_df'
        y_key = f'y_{split}_df'
        
        if x_key in denormalized_data:
            x_df = denormalized_data[x_key]
            feature_names = list(x_df.columns)
            
            # Apply anti-naive-lock transformations directly to the time series data
            processed_df = apply_anti_naive_lock_to_time_series(x_df, feature_names, config)
            processed[x_key] = processed_df
            print(f"    Processed {split} dataset with anti-naive-lock (shape: {processed_df.shape})")
        
        # Y data unchanged
        if y_key in denormalized_data:
            processed[y_key] = denormalized_data[y_key]
    
    return processed

#TODO: move to anti_naive_lock.py
def apply_anti_naive_lock_to_time_series(df, feature_names, config):
    """Apply anti-naive-lock transformations to a full time series DataFrame."""
    processed_df = df.copy()
    
    # Get feature categories from config with comprehensive defaults
    price_features = config.get('price_features', ['OPEN', 'LOW', 'HIGH', 'typical_price', 'open', 'low', 'high', 'close'])
    temporal_features = config.get('temporal_features', [
        'day_of_week', 'hour_of_day', 'day_of_month', 'month_of_year',
        'dayofweek', 'hourofday', 'dayofmonth', 'monthofyear'
    ])
    trend_features = config.get('trend_features', [
        'stl_trend', 'trend', 'STL_trend', 'STL_Trend', 'TREND'
    ])
    stationary_indicators = config.get('stationary_indicators', [
        'RSI', 'MACD', 'MACD_Histogram', 'MACD_Signal', 'EMA', 'SMA', 'BB_upper', 'BB_lower',
        'rsi', 'macd', 'macd_histogram', 'macd_signal', 'ema', 'sma', 'bb_upper', 'bb_lower'
    ])
    target_column = config.get('target_column', 'typical_price')
    excluded_columns = config.get('excluded_columns', [])
    
    print(f"      Applying anti-naive-lock to {len(feature_names)} features...")
    
    for feature_name in feature_names:
        try:
            # Check if feature name matches any category (case-insensitive)
            feature_lower = feature_name.lower()
            
            if any(tf.lower() in feature_lower for tf in temporal_features) and config.get('use_cyclic_encoding', True):
                # Apply cyclic encoding to temporal features
                processed_df[feature_name] = apply_cyclic_encoding_to_series(
                    processed_df[feature_name], feature_name
                )
                print(f"        Applied cyclic encoding to {feature_name}")
                
            elif (any(pf.lower() in feature_lower for pf in price_features) and 
                    config.get('use_log_returns', True) and 
                    feature_name != target_column and 
                    feature_name not in excluded_columns):
                # Apply log returns to price features (except target)
                processed_df[feature_name] = apply_log_returns_to_series(
                    processed_df[feature_name]
                )
                print(f"        Applied log returns to {feature_name}")

            elif (any(si.lower() in feature_lower for si in stationary_indicators) and 
                    config.get('use_log_returns', True) and 
                    feature_name != target_column and 
                    feature_name not in excluded_columns):
                # Apply log returns to stationary indicators (except target)
                processed_df[feature_name] = apply_log_returns_to_series(
                    processed_df[feature_name]
                )
                print(f"        Applied log returns to {feature_name}")
                
            elif any(tf.lower() in feature_lower for tf in trend_features) and config.get('use_first_differences', True):
                # Apply first differences to trend features
                processed_df[feature_name] = apply_first_differences_to_series(
                    processed_df[feature_name]
                )
                print(f"        Applied first differences to {feature_name}")
                
            else:
                # Preserve other features; do NOT transform target or excluded columns
                if feature_name == target_column or feature_name in excluded_columns:
                    print(f"        Preserved {feature_name} (target/excluded)")
                else:
                    # As a safe default, apply log returns to non-target, non-excluded features
                    processed_df[feature_name] = apply_log_returns_to_series(
                        processed_df[feature_name]
                    )
                    print(f"        Applied log returns to {feature_name}")
                
        except Exception as e:
            print(f"        ERROR processing {feature_name}: {e}")
            # In case of error, preserve the original feature
            processed_df[feature_name] = df[feature_name]
    
    return processed_df

def apply_cyclic_encoding_to_series( series, feature_name):
    """Apply cyclic encoding to a pandas Series."""
    # Determine period based on feature name (case-insensitive)
    feature_lower = feature_name.lower()
    
    if 'hour' in feature_lower:
        period = 24
    elif 'day_of_week' in feature_lower or 'dayofweek' in feature_lower:
        period = 7
    elif 'day_of_month' in feature_lower or 'dayofmonth' in feature_lower:
        period = 31
    elif 'month' in feature_lower:
        period = 12
    else:
        # Fallback: use the range of the data
        period = max(series) - min(series) + 1 if len(series) > 0 else 1
    
    # Apply cyclic encoding (sin component only) with safe handling
    try:
        angle = 2 * np.pi * series / period
        return np.sin(angle)
    except Exception as e:
        print(f"        Warning: Cyclic encoding failed for {feature_name}: {e}")
        return series  # Return original if encoding fails

def apply_log_returns_to_series( series):
    """Apply log returns to a pandas Series: ln(x_t / x_{t-1})."""
    log_returns = series.copy()
    
    # Calculate log returns with safe handling
    try:
        for i in range(1, len(series)):
            if pd.notna(series.iloc[i-1]) and pd.notna(series.iloc[i]) and series.iloc[i-1] > 0 and series.iloc[i] > 0:
                log_returns.iloc[i] = np.log(series.iloc[i] / series.iloc[i-1])
            else:
                log_returns.iloc[i] = 0.0
        
        # First value gets zero change
        log_returns.iloc[0] = 0.0
        
        # Replace any inf or nan values with 0
        log_returns = log_returns.replace([np.inf, -np.inf, np.nan], 0.0)
        
    except Exception as e:
        print(f"        Warning: Log returns calculation failed: {e}")
        return series  # Return original if calculation fails
    
    return log_returns

def apply_first_differences_to_series( series):
    """Apply first differences to a pandas Series: x_t - x_{t-1}."""
    differences = series.copy()
    
    try:
        # Calculate first differences with safe handling
        differences.iloc[1:] = series.iloc[1:].values - series.iloc[:-1].values
        differences.iloc[0] = 0.0  # First value gets zero change
        
        # Replace any inf or nan values with 0
        differences = differences.replace([np.inf, -np.inf, np.nan], 0.0)
        
    except Exception as e:
        print(f"        Warning: First differences calculation failed: {e}")
        return series  # Return original if calculation fails
    
    return differences


def apply_feature_normalization( 
                                x_train: np.ndarray,
                                x_val: np.ndarray, 
                                x_test: np.ndarray,
                                feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply feature-wise z-score normalization after preprocessing.
    
    Args:
        x_train, x_val, x_test: Sliding window matrices
        feature_names: Feature names for logging
        
    Returns:
        Normalized matrices and normalization statistics
    """
    norm_stats = {}
    
    print("Applying post-processing normalization...")
    
    # Calculate statistics from training data only
    for i in range(x_train.shape[2]):
        # Flatten across samples and time steps for each feature
        train_feature_flat = x_train[:, :, i].flatten()
        
        # Remove any NaN or infinite values for statistics calculation
        valid_mask = np.isfinite(train_feature_flat)
        if np.any(valid_mask):
            train_feature_clean = train_feature_flat[valid_mask]
            feature_mean = np.mean(train_feature_clean)
            feature_std = np.std(train_feature_clean)
            
            # Avoid division by zero
            if feature_std < 1e-8:
                feature_std = 1.0
                
            # Apply normalization to all datasets
            x_train[:, :, i] = (x_train[:, :, i] - feature_mean) / feature_std
            x_val[:, :, i] = (x_val[:, :, i] - feature_mean) / feature_std
            x_test[:, :, i] = (x_test[:, :, i] - feature_mean) / feature_std
            
            norm_stats[feature_names[i]] = {'mean': feature_mean, 'std': feature_std}
        else:
            print(f"WARNING: Feature {feature_names[i]} has no valid values for normalization")
            norm_stats[feature_names[i]] = {'mean': 0.0, 'std': 1.0}
    
    return x_train, x_val, x_test, norm_stats
