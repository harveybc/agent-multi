import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

class TargetPlugin:
    """
    STL Target Plugin.
    
    Responsibilities:
    - Decompose the target signal into Trend, Seasonal, and Residual components using STL.
    - Provide targets for each component and the total signal.
    """

    plugin_params = {
        "predicted_horizons": [1],
        "target_column": "typical_price",
        "stl_period": 24,  # Periodicity for STL
        "stl_seasonal": 13, # Seasonal smoother length
        "stl_trend": None,  # Trend smoother length (None = auto)
    }

    plugin_debug_vars = ["predicted_horizons", "target_column", "stl_period"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def _decompose_signal(self, series):
        """Helper to decompose a series using STL."""
        period = self.params.get("stl_period", 24)
        seasonal = self.params.get("stl_seasonal", 13)
        trend = self.params.get("stl_trend", None)
        
        try:
            # STL requires a pandas Series or array.
            stl = STL(series, period=period, seasonal=seasonal, trend=trend)
            res = stl.fit()
            return res.trend, res.seasonal, res.resid
        except Exception as e:
            print(f"STL Decomposition failed: {e}. Returning 0s for components.")
            zeros = np.zeros_like(series)
            return zeros, zeros, zeros

    def calculate_targets_from_baselines(self, baseline_data, config):
        """
        Calculate targets for Total, Trend, Seasonal, and Residual.
        """
        self.set_params(**config)
        cfg = self.params
        horizons = cfg["predicted_horizons"]
        
        results = {
            "y_train": {}, "y_val": {}, "y_test": {},
            "y_train_trend": {}, "y_val_trend": {}, "y_test_trend": {},
            "y_train_seasonal": {}, "y_val_seasonal": {}, "y_test_seasonal": {},
            "y_train_resid": {}, "y_val_resid": {}, "y_test_resid": {},
            "predicted_horizons": horizons
        }

        for split in ("train", "val", "test"):
            base_key = f"baseline_{split}"
            if base_key not in baseline_data or len(baseline_data[base_key]) == 0:
                continue

            baselines = np.array(baseline_data[base_key])
            
            # Decompose the entire baseline first
            trend, seasonal, resid = self._decompose_signal(baselines)
            
            max_h = max(horizons)
            max_samples = len(baselines) - max_h
            
            if max_samples <= 0:
                continue

            for h in horizons:
                # Slice targets for horizon h
                # If we predict at t for t+h, the target is baseline[t+h]
                # The input window ends at t.
                
                # Total
                results[f"y_{split}"][f"output_horizon_{h}"] = baselines[h:][:max_samples].astype(np.float32)
                
                # Components
                results[f"y_{split}_trend"][f"output_horizon_{h}"] = trend[h:][:max_samples].astype(np.float32)
                results[f"y_{split}_seasonal"][f"output_horizon_{h}"] = seasonal[h:][:max_samples].astype(np.float32)
                results[f"y_{split}_resid"][f"output_horizon_{h}"] = resid[h:][:max_samples].astype(np.float32)

        return results
