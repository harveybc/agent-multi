"""Binary target plugin for binary classification predictors.

Maps a single label column to the y_train / y_val / y_test dict format
expected by the pipeline.  The ``signal_type`` parameter selects which
label column to extract.

The preprocessor passes aligned label DataFrames via
``baselines["labels_{split}"]``.  Output uses the key ``output_horizon_1``
so the standard stl_pipeline can handle it transparently.
"""
import numpy as np
import pandas as pd

SIGNAL_MAP = {
    "buy_entry":  ("buy_entry_label",  "buy_entry_binary"),
    "sell_entry": ("sell_entry_label", "sell_entry_binary"),
    "buy_exit":   ("buy_exit_label",   "buy_exit_binary"),
    "sell_exit":  ("sell_exit_label",  "sell_exit_binary"),
}

# Pipeline-compatible output key (single "horizon")
_OUTPUT_KEY = "output_horizon_1"


class TargetPlugin:
    plugin_params = {
        "signal_type": "buy_entry",
    }
    plugin_debug_vars = ["signal_type"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def calculate_targets_from_baselines(self, baseline_data, config):
        """Extract binary labels aligned to sliding-window count.

        Parameters
        ----------
        baseline_data : dict
            Contains ``labels_train``, ``labels_val``, ``labels_test``
            as :class:`pd.DataFrame` (with label columns, already aligned
            to window count by the preprocessor).
        config : dict
            Must contain ``signal_type``.

        Returns
        -------
        dict
            ``{"y_train": {"output_horizon_1": array}, ...}``
        """
        self.set_params(**config)
        signal_type = self.params.get("signal_type", "buy_entry")
        if signal_type not in SIGNAL_MAP:
            raise ValueError(
                f"Invalid signal_type '{signal_type}'. "
                f"Must be one of: {list(SIGNAL_MAP.keys())}"
            )
        label_col, output_name = SIGNAL_MAP[signal_type]

        targets = {"train": {}, "val": {}, "test": {}}
        for split in ("train", "val", "test"):
            key = f"labels_{split}"
            if key not in baseline_data:
                targets[split][_OUTPUT_KEY] = np.array([], dtype=np.float32)
                continue

            data = baseline_data[key]
            if isinstance(data, pd.DataFrame):
                if label_col not in data.columns:
                    raise ValueError(
                        f"Column '{label_col}' not found in {split} data. "
                        f"Available: {list(data.columns)}"
                    )
                labels = data[label_col].values
            else:
                labels = np.asarray(data, dtype=np.float32)

            labels = labels.astype(np.float32).reshape(-1, 1)
            targets[split][_OUTPUT_KEY] = labels

            pos_rate = float(np.mean(labels))
            print(
                f"[binary_target] {split} {output_name}: "
                f"{len(labels)} samples, pos_rate={pos_rate:.3f}"
            )

        return {
            "y_train": targets["train"],
            "y_val": targets["val"],
            "y_test": targets["test"],
            "predicted_horizons": [1],
        }
