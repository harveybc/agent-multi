#!/usr/bin/env python
"""RandomForest multi-horizon baseline refactored to use BasePredictorPlugin.

Provides deterministic ensemble uncertainty (tree stddev). Only non-Keras
training/prediction implemented locally; metrics & param utilities inherited.
"""
from __future__ import annotations
import numpy as np, pickle
from sklearn.ensemble import RandomForestRegressor
from .common.base import BasePredictorPlugin


class Plugin(BasePredictorPlugin):
    plugin_params = {
        "batch_size": 0,  # unused placeholder for interface parity
        "predicted_horizons": [1],
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "random_state": 42,
        "mc_samples": 100,
        "early_patience": 0,  # placeholder (not used)
    }
    plugin_debug_vars = [
        "predicted_horizons","n_estimators","max_depth","min_samples_leaf","mc_samples"
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self.model = {}  # horizon -> fitted RandomForestRegressor
        self.output_names = [f"output_horizon_{h}" for h in self.params["predicted_horizons"]]

    def set_params(self, **kwargs):  # override only to refresh output names
        super().set_params(**kwargs)
        if "predicted_horizons" in kwargs:
            self.output_names = [f"output_horizon_{h}" for h in self.params["predicted_horizons"]]

    def build_model(self, input_shape, x_train, config):  # no-op (lazy build in train)
        if config:
            self.params.update(config)
        return None

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        if config:
            self.params.update(config)
        if not isinstance(y_train, dict):
            raise TypeError("y_train must be dict mapping output_horizon_<H> -> array")
        horizons = self.params["predicted_horizons"]
        flat_x = x_train.reshape(len(x_train), -1)
        for h in horizons:
            key = f"output_horizon_{h}"
            if key not in y_train:
                raise ValueError(f"Missing {key} in y_train")
            y = y_train[key].reshape(-1)
            rf = RandomForestRegressor(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth"),
                min_samples_leaf=self.params.get("min_samples_leaf", 1),
                random_state=self.params.get("random_state", 42),
                n_jobs=-1,
            )
            rf.fit(flat_x, y)
            self.model[h] = rf
        mc = self.params.get("mc_samples", 100)
        train_preds, train_unc = self.predict_with_uncertainty(x_train, mc)
        val_preds, val_unc = ([], [])
        if x_val is not None and y_val is not None:
            val_preds, val_unc = self.predict_with_uncertainty(x_val, mc)
        class MockHistory:  # simple history stub for pipeline compatibility
            def __init__(self):
                self.history = {"loss": [], "val_loss": []}
        history = MockHistory()
        return history, train_preds, train_unc, val_preds, val_unc

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        flat_x = x_test.reshape(len(x_test), -1)
        means, stds = [], []
        for h in self.params["predicted_horizons"]:
            rf = self.model.get(h)
            if rf is None:
                raise ValueError(f"Horizon {h} model missing")
            trees = rf.estimators_
            mc = min(mc_samples, len(trees))
            idx = np.random.choice(len(trees), size=mc, replace=False)
            per_tree = np.stack([trees[i].predict(flat_x) for i in idx], axis=0)
            mean = rf.predict(flat_x)
            std = per_tree.std(axis=0)
            means.append(mean.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))
        return means, stds

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({'params': self.params, 'model': self.model}, f)
        print(f"Model saved to {file_path}")
    def load(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.params = data['params']
        self.model = data['model']
        self.output_names = [f"output_horizon_{h}" for h in self.params['predicted_horizons']]
        print(f"Ioin model loaded from {file_path}")

    # Metrics come from BasePredictorPlugin (calculate_mae / calculate_r2)

if __name__ == "__main__":
    import numpy as np
    pl = Plugin({"predicted_horizons": [1,2,3]})
    X = np.random.randn(50, 24, 2)
    y = {f"output_horizon_{h}": np.random.randn(50,1) for h in [1,2,3]}
    pl.build_model((24,2), None, {})
    pl.train(X, y, 0, 0, 0, None, None, {})
    means, stds = pl.predict_with_uncertainty(X[:5])
    print([m.shape for m in means], [s.shape for s in stds])