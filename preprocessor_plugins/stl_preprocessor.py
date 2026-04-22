import numpy as np
import pandas as pd
from .helpers import load_normalization_json, denormalize_all_datasets, load_normalized_csv, exclude_columns_from_datasets
from .sliding_windows import create_sliding_windows, extract_baselines_from_sliding_windows
from .anti_naive_lock import apply_log_returns_to_series, apply_feature_normalization
import os as _os
_QUIET = _os.environ.get('PREDICTOR_QUIET', '0') == '1'


class STLPreprocessorZScore:
    """
    1. Load already normalized CSV data ✅
    2. Denormalize all input datasets using JSON parameters
    3. Create sliding windows from denormalized data
    4. Extract baselines (last elements of each window for target column)
    5. Calculate log return targets with those baselines (train, validation, test)
     6. Create SECOND sliding windows matrix from the ORIGINAL normalized datasets transformed with per-column log-returns
         (applies to all numeric features). Dates preserved; no change to target pipeline.
     7. Keep baselines and targets unchanged (they're already calculated correctly)
    """

    # Plugin-specific parameters they get overwritten if declared in the config
    plugin_params = {
        "window_size": 48,
        "predicted_horizons": [1, 2, 3, 4, 5, 6],
        "target_column": "typical_price",
        "use_returns": True,
        # Targets and baselines are derived from *denormalized* baselines (real-world price units).
        # This prevents downstream metric code from accidentally denormalizing twice.
        "targets_are_denormalized": True,
        "anti_naive_lock_enabled": True,
    "feature_preprocessing_strategy": "selective",
    "add_window_stats": False,
    "window_stats_periods": [12, 48],
    "reverse_time_axis": False,
    # New: optional multi-scale returns augmentation (causal, within-window)
    "add_multi_scale_returns": False,
    "multi_scale_return_periods": [6, 24, 72],
    "use_log1p_features": ["typical_price"],
    # Temporal feature encoding (online generation from DATE_TIME)
    "use_temporal_features": True,
    "hod_encoding": "sincos",   # "sincos", "onehot", or "none"
    "dow_encoding": "sincos",   # "sincos", "onehot", or "none"
    "moy_encoding": "sincos",   # "sincos", "onehot", or "none"
    }
    
    plugin_debug_vars = ["window_size", "predicted_horizons", "target_column", "use_log1p_features"]

    # Start of plugin interface methods    
    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
    
    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}
    
    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())
    # End of plugin interface methods

    def process_data(self, target_plugin, config):
        # Main process orchestration
        try:
            self.set_params(**config)
            config = self.params
            
            predicted_horizons = config['predicted_horizons']
            if not isinstance(predicted_horizons, list) or not predicted_horizons:
                raise ValueError("predicted_horizons must be a non-empty list")
            
            # 1. Load already normalized CSV data
            if not _QUIET: print("Step 1: Load normalized CSV data")
            normalized_data, dates = load_normalized_csv(config)
            if not normalized_data:
                raise ValueError("No data loaded - check file paths in config")
            
            # 2. Denormalize all input datasets using JSON parameters
            if not _QUIET: print("Step 2: Denormalize all input datasets")
            denormalized_data = denormalize_all_datasets(normalized_data, config)
            
            # 3. Create FIRST sliding windows from denormalized data used only and only for baseline extraction
            if not _QUIET: print("Step 3: Create first sliding windows from denormalized data")
            denorm_sliding_windows = create_sliding_windows(denormalized_data, config, dates)
            if 'X_train' not in denorm_sliding_windows or denorm_sliding_windows.get('X_train') is None or len(denorm_sliding_windows.get('X_train')) == 0:
                print("CRITICAL: First-pass sliding windows missing or empty for TRAIN split before baseline extraction. This leads to empty baseline_train and missing y_train targets.")

            # 4. Extract baselines from the sliding windows (last elements of each window for target column)
            if not _QUIET: print("Step 4: Extract baselines from sliding windows")
            baselines = extract_baselines_from_sliding_windows(denorm_sliding_windows, config)

            # 4b. Pass label DataFrames (aligned to window count) for binary target plugins
            window_size = config.get('window_size', 72)
            for split, df_key in [("train", "y_train_df"), ("val", "y_val_df"), ("test", "y_test_df")]:
                df = normalized_data.get(df_key)
                n_windows = len(baselines.get(f"baseline_{split}", []))
                if df is not None and not df.empty and n_windows > 0:
                    baselines[f"labels_{split}"] = df.iloc[window_size - 1 : window_size - 1 + n_windows].reset_index(drop=True)

            # 5. Calculate targets directly from baselines
            if not _QUIET: print("Step 5: Calculate targets from baselines")
            #TODO: verify this method is correct
            targets = target_plugin.calculate_targets_from_baselines(baselines, config)

            # 6. Create SECOND sliding windows from normalized data
            if not _QUIET: print("Step 6: Create second sliding windows from normalized data")
            final_sliding_windows = create_sliding_windows(normalized_data, config, dates)

            # 6b. Apply log1p to specified features
            self._apply_log1p_to_features(final_sliding_windows, config)

            # 6c. Add temporal features (hour-of-day, day-of-week, month-of-year)
            self._add_temporal_features(final_sliding_windows, config)

            # 6d. Add window statistics features (rolling std, ema, price-minus-ema)
            self._add_window_stats_features(final_sliding_windows, config)

            # 7. Align final sliding windows with target data length
            if not _QUIET: print("Step 7: Align sliding windows with target data")
            final_sliding_windows = self._align_sliding_windows_with_targets(final_sliding_windows, targets, config)

            # Return final results
            #TODO: verify this method is correct and required
            output, preprocessor_params = self._prepare_final_output(final_sliding_windows, targets, baselines, config)
            # attach naive info if present
            #for k, v in naive_info.items():
            #    output[k] = v
            
            # Store baselines for access in output preparation
            self.extracted_baselines = baselines
            
            self.params.update(preprocessor_params)
            return output

        except Exception as e:
            print(f"ERROR in process_data: {e}")
            raise

    def _apply_log1p_to_features(self, sliding_windows, config):
        """Apply np.log1p to specified features in the sliding windows."""
        features_to_log = config.get("use_log1p_features", [])
        if not features_to_log:
            return

        if not _QUIET: print(f"Step 6b: Applying log1p to features: {features_to_log}")
        
        feature_names = sliding_windows.get('feature_names', [])
        if not feature_names:
            if not _QUIET: print("  WARNING: No feature names found in sliding windows, cannot apply log1p")
            return

        # Find indices for the features
        indices = []
        found_features = []
        for i, name in enumerate(feature_names):
            if name in features_to_log:
                indices.append(i)
                found_features.append(name)
        
        if not indices:
            if not _QUIET: print(f"  WARNING: None of the requested features {features_to_log} found in dataset features")
            return
            
        if not _QUIET: print(f"  Found features at indices {indices}: {found_features}")

        for key in ['X_train', 'X_val', 'X_test']:
            if key in sliding_windows:
                data = sliding_windows[key]
                # Check if data is not empty and has correct dimensions
                if hasattr(data, 'shape') and len(data.shape) == 3:
                    # data shape: (samples, window_size, features)
                    
                    # Use symmetric log1p: sign(x) * log1p(|x|)
                    # This handles negative values (common in normalized data) without losing information
                    # or causing NaNs for values <= -1.
                    
                    features_data = data[..., indices]
                    sliding_windows[key][..., indices] = np.sign(features_data) * np.log1p(np.abs(features_data))
                    
                    if not _QUIET: print(f"  Applied symmetric log1p (sign(x)*log1p(|x|)) to {key} to handle negative normalized values")
                else:
                    if not _QUIET: print(f"  Skipping {key}: Invalid shape or type")

    # ── Encoding name constants (kept in sync with neat_optimizer.py) ──
    ENCODING_NAMES = ["none", "sincos", "onehot"]

    def _add_temporal_features(self, sliding_windows, config):
        """Add temporal features derived from DATE_TIME of each window's baseline tick.

        Encoding per variable is configurable: "sincos", "onehot", or "none".
        Integer values (from NEAT optimizer) are mapped to strings automatically.
        Features are constant across all timesteps within a window.
        """
        use_temporal = config.get("use_temporal_features", True)
        if isinstance(use_temporal, (int, float)):
            use_temporal = bool(int(round(use_temporal)))
        if not use_temporal:
            return

        hod_enc = config.get("hod_encoding", "sincos")
        dow_enc = config.get("dow_encoding", "sincos")
        moy_enc = config.get("moy_encoding", "sincos")

        # Map int values to strings (for NEAT optimizer which passes ints)
        for varname in ("hod_enc", "dow_enc", "moy_enc"):
            val = locals()[varname]
            if isinstance(val, (int, float)):
                idx = max(0, min(int(round(val)), len(self.ENCODING_NAMES) - 1))
                locals()[varname]  # can't reassign locals, use explicit below

        if isinstance(hod_enc, (int, float)):
            hod_enc = self.ENCODING_NAMES[max(0, min(int(round(hod_enc)), 2))]
        if isinstance(dow_enc, (int, float)):
            dow_enc = self.ENCODING_NAMES[max(0, min(int(round(dow_enc)), 2))]
        if isinstance(moy_enc, (int, float)):
            moy_enc = self.ENCODING_NAMES[max(0, min(int(round(moy_enc)), 2))]

        if hod_enc == "none" and dow_enc == "none" and moy_enc == "none":
            return

        # Build feature name list
        new_names = []
        if hod_enc == "sincos":
            new_names += ["hod_sin", "hod_cos"]
        elif hod_enc == "onehot":
            new_names += [f"hod_{h}" for h in [0, 4, 8, 12, 16, 20]]

        if dow_enc == "sincos":
            new_names += ["dow_sin", "dow_cos"]
        elif dow_enc == "onehot":
            new_names += [f"dow_{d}" for d in range(5)]

        if moy_enc == "sincos":
            new_names += ["moy_sin", "moy_cos"]
        elif moy_enc == "onehot":
            new_names += [f"moy_{m}" for m in range(1, 13)]

        if not new_names:
            return

        n_new = len(new_names)
        if not _QUIET:
            print(f"Step 6c: Adding {n_new} temporal features: {new_names}")

        for split in ['train', 'val', 'test']:
            X_key = f'X_{split}'
            dates_key = f'x_dates_{split}'

            if X_key not in sliding_windows or dates_key not in sliding_windows:
                continue

            X = sliding_windows[X_key]
            dates = sliding_windows[dates_key]

            if X is None or not hasattr(X, 'shape') or len(X.shape) != 3:
                continue

            n_windows, window_size, _ = X.shape
            dt_index = pd.to_datetime(dates)

            # Build temporal array (n_windows, n_new)
            temporal = np.zeros((n_windows, n_new), dtype=np.float32)
            col = 0

            if hod_enc == "sincos":
                slots = dt_index.hour / 4.0  # 4h bars → slots 0-5
                temporal[:, col] = np.sin(2 * np.pi * slots / 6.0)
                temporal[:, col + 1] = np.cos(2 * np.pi * slots / 6.0)
                col += 2
            elif hod_enc == "onehot":
                hours = dt_index.hour
                for i, h in enumerate([0, 4, 8, 12, 16, 20]):
                    temporal[:, col + i] = (hours == h).astype(np.float32)
                col += 6

            if dow_enc == "sincos":
                dow = dt_index.dayofweek
                temporal[:, col] = np.sin(2 * np.pi * dow / 5.0)
                temporal[:, col + 1] = np.cos(2 * np.pi * dow / 5.0)
                col += 2
            elif dow_enc == "onehot":
                dow = dt_index.dayofweek
                for i in range(5):
                    temporal[:, col + i] = (dow == i).astype(np.float32)
                col += 5

            if moy_enc == "sincos":
                month = dt_index.month
                temporal[:, col] = np.sin(2 * np.pi * (month - 1) / 12.0)
                temporal[:, col + 1] = np.cos(2 * np.pi * (month - 1) / 12.0)
                col += 2
            elif moy_enc == "onehot":
                month = dt_index.month
                for i in range(1, 13):
                    temporal[:, col + i - 1] = (month == i).astype(np.float32)
                col += 12

            # Broadcast to (n_windows, window_size, n_new) — constant across timesteps
            temporal_3d = np.tile(temporal[:, np.newaxis, :], (1, window_size, 1))
            sliding_windows[X_key] = np.concatenate([X, temporal_3d], axis=2)

            if not _QUIET:
                print(f"  {split}: {X.shape} → {sliding_windows[X_key].shape}")

        # Update feature name lists
        for key in ['feature_names', 'feature_names_train', 'feature_names_val', 'feature_names_test']:
            if key in sliding_windows:
                sliding_windows[key] = list(sliding_windows[key]) + new_names

    def _add_window_stats_features(self, sliding_windows, config):
        """Add rolling window statistics as features (std, ema, price-minus-ema).

        For each window, computes statistics from the target column across the
        window timesteps for each configured period.  Features are broadcast to
        all timesteps (constant within a window).

        Controlled by ``add_window_stats`` (bool / 0|1 from NEAT).
        Periods come from ``window_stats_periods`` (default [12, 48]).
        """
        use_stats = config.get("add_window_stats", True)
        if isinstance(use_stats, (int, float)):
            use_stats = bool(int(round(use_stats)))
        if not use_stats:
            return

        periods = config.get("window_stats_periods", [12, 48])
        if not periods:
            return

        target_col = config.get("target_column", "typical_price")
        feature_names = sliding_windows.get("feature_names", [])
        if target_col not in feature_names:
            if not _QUIET:
                print(f"  WARNING: target '{target_col}' not in feature_names, skipping window stats")
            return
        target_idx = feature_names.index(target_col)

        # Build feature names: rolling_std_{p}, rolling_ema_{p}, price_minus_ema_{p}
        new_names = []
        for p in periods:
            new_names += [f"rolling_std_{p}", f"rolling_ema_{p}", f"price_minus_ema_{p}"]
        n_new = len(new_names)

        if not _QUIET:
            print(f"Step 6d: Adding {n_new} window stats features: {new_names}")

        for split in ["train", "val", "test"]:
            X_key = f"X_{split}"
            if X_key not in sliding_windows:
                continue
            X = sliding_windows[X_key]
            if X is None or not hasattr(X, "shape") or len(X.shape) != 3:
                continue

            n_windows, window_size, _ = X.shape
            stats = np.zeros((n_windows, n_new), dtype=np.float32)

            # Extract target column for all windows: (n_windows, window_size)
            price = X[:, :, target_idx]

            col = 0
            for p in periods:
                # Use last `p` timesteps (or all if window < p)
                span = min(p, window_size)
                tail = price[:, -span:]  # (n_windows, span)

                # Rolling std over the tail
                stats[:, col] = np.std(tail, axis=1, dtype=np.float32)

                # EMA over the tail (decay = 2/(span+1))
                alpha = 2.0 / (span + 1.0)
                ema = np.copy(tail[:, 0])
                for t in range(1, tail.shape[1]):
                    ema = alpha * tail[:, t] + (1.0 - alpha) * ema
                stats[:, col + 1] = ema.astype(np.float32)

                # Price minus EMA (last timestep price − ema)
                stats[:, col + 2] = (price[:, -1] - ema).astype(np.float32)
                col += 3

            # Broadcast to (n_windows, window_size, n_new)
            stats_3d = np.tile(stats[:, np.newaxis, :], (1, window_size, 1))
            sliding_windows[X_key] = np.concatenate([X, stats_3d], axis=2)

            if not _QUIET:
                print(f"  {split}: {X.shape} → {sliding_windows[X_key].shape}")

        # Update feature name lists
        for key in ["feature_names", "feature_names_train", "feature_names_val", "feature_names_test"]:
            if key in sliding_windows:
                sliding_windows[key] = list(sliding_windows[key]) + new_names

    def _align_sliding_windows_with_targets(self, sliding_windows, targets, config):
        """Align sliding windows with target data to ensure same number of samples."""
        if not _QUIET: print("  Aligning sliding windows with target data...")
        
        # Get the first target to determine the target length
        predicted_horizons = config['predicted_horizons']
        first_horizon = predicted_horizons[0]
        
        # Find target lengths for each split
        target_lengths = {}
        for split in ['train', 'val', 'test']:
            target_key = f'y_{split}'
            if target_key in targets and f'output_horizon_{first_horizon}' in targets[target_key]:
                target_length = len(targets[target_key][f'output_horizon_{first_horizon}'])
                target_lengths[split] = target_length
                if not _QUIET: print(f"    {split} target length: {target_length}")
            else:
                target_lengths[split] = 0
        
        # Trim sliding windows to match target lengths
        aligned_windows = {}

        for key, windows in sliding_windows.items():
            if key.startswith('X_'):
                # Extract split name (train, val, test)
                split = key.split('_')[1]
                if split in target_lengths and target_lengths[split] > 0:
                    target_length = target_lengths[split]
                    if hasattr(windows, 'shape') and len(windows) > target_length:
                        aligned_windows[key] = windows[:target_length]
                        if not _QUIET: print(f"    Trimmed {key} from {len(windows)} to {target_length} samples")
                    else:
                        aligned_windows[key] = windows
                        
                else:
                    aligned_windows[key] = windows
                    
            else:
                # Keep non-window data as is
                aligned_windows[key] = windows
                

        return aligned_windows

    
    def _prepare_final_output(self, sliding_windows, targets, baselines, config):
        """Prepare final output structure."""
        # Use the baselines passed as parameter (extracted from denormalized data)
        baseline_data = {}
        if isinstance(baselines, dict):
            # baselines is already in the correct format
            baseline_data = baselines
        else:
            # Handle legacy format
            for split in ['train', 'val', 'test']:
                baseline_key = f'baseline_{split}'
                baseline_data[baseline_key] = np.array([])
        
        # Validate that we have the required data structures
        required_sliding_window_keys = ['X_train', 'X_val', 'X_test']
        required_target_keys = ['y_train', 'y_val', 'y_test']
        
        for key in required_sliding_window_keys:
            if key not in sliding_windows:
                if not _QUIET: print(f"WARNING: Missing sliding window data: {key}")
                sliding_windows[key] = np.array([])
        
        for key in required_target_keys:
            if key not in targets:
                if not _QUIET: print(f"WARNING: Missing target data: {key}")
                targets[key] = {}
        
        output = {
            # Final sliding windows for model (SECOND sliding windows after anti-naive-lock)
            "x_train": sliding_windows['X_train'],
            "x_val": sliding_windows['X_val'],
            "x_test": sliding_windows['X_test'],
            
            # Targets by horizon (calculated from FIRST sliding windows)
            "y_train": targets['y_train'],
            "y_val": targets['y_val'],
            "y_test": targets['y_test'],
            
            # Dates
            "x_train_dates": sliding_windows.get('x_dates_train'),
            "y_train_dates": sliding_windows.get('x_dates_train'),
            "x_val_dates": sliding_windows.get('x_dates_val'),
            "y_val_dates": sliding_windows.get('x_dates_val'),
            "x_test_dates": sliding_windows.get('x_dates_test'),
            "y_test_dates": sliding_windows.get('x_dates_test'),
            
            # Baselines for prediction reconstruction
            "baseline_train": baseline_data.get('baseline_train', np.array([])),
            "baseline_val": baseline_data.get('baseline_val', np.array([])),
            "baseline_test": baseline_data.get('baseline_test', np.array([])),
            
            # Metadata
            "feature_names": sliding_windows.get('feature_names', []),
            "feature_names_train": sliding_windows.get('feature_names_train', []),
            "feature_names_val": sliding_windows.get('feature_names_val', []),
            "feature_names_test": sliding_windows.get('feature_names_test', []),
            "target_returns_means": targets.get('target_returns_means', []),
            "target_returns_stds": targets.get('target_returns_stds', []),
            "predicted_horizons": config['predicted_horizons'],
            "normalization_json": load_normalization_json(config),
        }

        # Inject any additional keys from targets (e.g. decomposed components like y_train_trend)
        for key, value in targets.items():
            if key not in output:
                output[key] = value
        
        # Print summary statistics
        if not _QUIET: print("\nPreprocessing Summary:")
        if not _QUIET: print(f"  X_train shape: {output['x_train'].shape if hasattr(output['x_train'], 'shape') else 'N/A'}")
        if not _QUIET: print(f"  X_val shape: {output['x_val'].shape if hasattr(output['x_val'], 'shape') else 'N/A'}")
        if not _QUIET: print(f"  X_test shape: {output['x_test'].shape if hasattr(output['x_test'], 'shape') else 'N/A'}")
        if not _QUIET: print(f"  Feature names: {len(output['feature_names'])}")
        if not _QUIET: print(f"  Predicted horizons: {output['predicted_horizons']}")
        if not _QUIET: print(f"  Target normalization parameters available: {len(output['target_returns_means'])}")
        if not _QUIET: print(f"  Baseline train length: {len(output['baseline_train'])}")
        if not _QUIET: print(f"  Baseline val length: {len(output['baseline_val'])}")
        if not _QUIET: print(f"  Baseline test length: {len(output['baseline_test'])}")

        output, preprocessor_params = exclude_columns_from_datasets(output, self.params, config)

        return output, preprocessor_params

    def run_preprocessing(self, target_plugin, config):
        """Run preprocessing with configuration."""
        processed_data = self.process_data(target_plugin, config)
        return processed_data


# Plugin interface alias for the system
PreprocessorPlugin = STLPreprocessorZScore
