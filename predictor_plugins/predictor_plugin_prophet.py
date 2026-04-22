#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predictor_plugin_prophet.py

Ioin plugin using Meta Prophet.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

from .common.base import BasePredictorPlugin

class MockHistory:
    def __init__(self):
        self.history = {'loss': [0.0], 'val_loss': [0.0]}

class Plugin(BasePredictorPlugin):
    """
    Ioin plugin wrapping Meta Prophet.
    Trains one Prophet model per horizon.
    """

    plugin_params: Dict[str, Any] = {
        "predicted_horizons": [1],
        "prophet_params": {},  # Params passed to Prophet constructor
        "interval_width": 0.6827, # Approx 1 sigma
        "add_country_holidays": None, # e.g. "US"
        "daily_seasonality": "auto",
        "weekly_seasonality": "auto",
        "yearly_seasonality": "auto",
        "use_regressors": False,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "regressors_prior_scale": 10.0,
    }

    plugin_debug_vars: List[str] = [
        "predicted_horizons",
        "prophet_params",
        "interval_width",
        "add_country_holidays",
        "daily_seasonality",
        "weekly_seasonality",
        "yearly_seasonality",
        "use_regressors",
        "changepoint_prior_scale",
        "seasonality_prior_scale",
        "holidays_prior_scale",
        "regressors_prior_scale",
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.models: Dict[int, Prophet] = {}
        self.output_names: List[str] = []

    def build_model(
        self,
        input_shape: Tuple[int, ...],
        x_train: Any,
        config: Dict[str, Any],
    ) -> None:
        """
        Setup models dict. Actual creation happens in train because we need data.
        """
        if config:
            self.params.update(config)
        
        horizons = self.params.get("predicted_horizons", [1])
        print(f"Prophet build_model: predicted_horizons = {horizons}")
        self.output_names = [f"output_horizon_{h}" for h in horizons]
        print(f"Prophet build_model: output_names = {self.output_names}")
        # Models will be created in train

    def train(
        self,
        x_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int,
        threshold_error: float,
        x_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[Any, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        
        if config:
            self.params.update(config)

        train_dates = self.params.get("train_dates")
        val_dates = self.params.get("val_dates")

        if train_dates is None:
            raise ValueError("Prophet plugin requires 'train_dates' in params. Ensure the pipeline provides them.")

        horizons = self.params.get("predicted_horizons", [1])
        prophet_params = self.params.get("prophet_params", {})
        interval_width = self.params.get("interval_width", 0.6827)
        country_holidays = self.params.get("add_country_holidays")
        daily_seasonality = self.params.get("daily_seasonality", "auto")
        weekly_seasonality = self.params.get("weekly_seasonality", "auto")
        yearly_seasonality = self.params.get("yearly_seasonality", "auto")
        use_regressors = self.params.get("use_regressors", False)
        feature_names = self.params.get("feature_names", [])
        
        # Prior scales
        changepoint_prior_scale = self.params.get("changepoint_prior_scale", 0.05)
        seasonality_prior_scale = self.params.get("seasonality_prior_scale", 10.0)
        holidays_prior_scale = self.params.get("holidays_prior_scale", 10.0)
        regressors_prior_scale = self.params.get("regressors_prior_scale", 10.0)

        train_preds_list = []
        train_unc_list = []
        val_preds_list = []
        val_unc_list = []

        # Prepare regressors if enabled
        regressor_cols = []
        x_train_reg = None
        x_val_reg = None
        
        if use_regressors and len(feature_names) > 0:
            # x_train is (N, W, F). We take the last step of the window: (N, F)
            x_train_reg = x_train[:, -1, :]
            regressor_cols = feature_names
            
            if x_val is not None:
                x_val_reg = x_val[:, -1, :]

        for h in horizons:
            key = f"output_horizon_{h}"
            y = y_train.get(key)
            if y is None:
                continue
            
            print(f"Prophet training horizon {h}: y shape={y.shape}, train_dates type={type(train_dates)}")
            if train_dates is not None:
                print(f"  train_dates length={len(train_dates) if hasattr(train_dates, '__len__') else 'N/A'}")
            
            # Trim train_dates to match y length (preprocessing may have trimmed samples)
            dates_to_use = train_dates[:len(y)] if train_dates is not None else None
            
            # Prepare DataFrame for Prophet
            data = {
                'ds': pd.to_datetime(dates_to_use),
                'y': y.flatten()
            }
            
            print(f"  ds length={len(data['ds'])}, y length={len(data['y'])}")
            
            if use_regressors and regressor_cols and x_train_reg is not None:
                for idx, col in enumerate(regressor_cols):
                    if idx < x_train_reg.shape[1]:
                        data[col] = x_train_reg[:, idx]
            
            df = pd.DataFrame(data)

            # Initialize and train Prophet
            m = Prophet(
                interval_width=interval_width,
                daily_seasonality=daily_seasonality,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=yearly_seasonality,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                **prophet_params
            )
            
            if country_holidays:
                m.add_country_holidays(country_name=country_holidays)
            
            if use_regressors and regressor_cols:
                for col in regressor_cols:
                    m.add_regressor(col, prior_scale=regressors_prior_scale)
            
            m.fit(df)
            self.models[h] = m

            # Predict on train
            # For training prediction, we use the same df we trained on
            forecast_train = m.predict(df)
            yhat_train = forecast_train['yhat'].values.reshape(-1, 1)
            unc_train = ((forecast_train['yhat_upper'] - forecast_train['yhat_lower']) / 2).values.reshape(-1, 1)
            
            train_preds_list.append(yhat_train)
            train_unc_list.append(unc_train)

            # Predict on val
            if val_dates is not None and key in y_val:
                # Trim val_dates to match y_val length
                val_dates_trimmed = val_dates[:len(y_val[key])]
                
                val_data = {
                    'ds': pd.to_datetime(val_dates_trimmed),
                    'y': y_val[key].flatten()
                }
                if use_regressors and regressor_cols and x_val_reg is not None:
                    for idx, col in enumerate(regressor_cols):
                        if idx < x_val_reg.shape[1]:
                            val_data[col] = x_val_reg[:, idx]
                            
                df_val = pd.DataFrame(val_data)
                forecast_val = m.predict(df_val)
                yhat_val = forecast_val['yhat'].values.reshape(-1, 1)
                unc_val = ((forecast_val['yhat_upper'] - forecast_val['yhat_lower']) / 2).values.reshape(-1, 1)
                
                val_preds_list.append(yhat_val)
                val_unc_list.append(unc_val)
            else:
                val_preds_list.append(np.zeros((len(x_val), 1)))
                val_unc_list.append(np.zeros((len(x_val), 1)))

        return MockHistory(), train_preds_list, train_unc_list, val_preds_list, val_unc_list

    def predict_with_uncertainty(
        self,
        x_test: np.ndarray,
        mc_samples: int = 50
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
        test_dates = self.params.get("test_dates")
        if test_dates is None:
             raise ValueError("Prophet plugin requires 'test_dates' in params.")

        horizons = self.params.get("predicted_horizons", [1])
        use_regressors = self.params.get("use_regressors", False)
        feature_names = self.params.get("feature_names", [])
        
        preds_list = []
        unc_list = []

        # Prepare regressors if enabled
        regressor_cols = []
        x_test_reg = None
        if use_regressors and len(feature_names) > 0:
            x_test_reg = x_test[:, -1, :]
            regressor_cols = feature_names

        for h in horizons:
            m = self.models.get(h)
            if m is None:
                preds_list.append(np.zeros((len(x_test), 1)))
                unc_list.append(np.zeros((len(x_test), 1)))
                continue

            # Trim test_dates to match x_test length
            test_dates_trimmed = test_dates[:len(x_test)]
            test_data = {'ds': pd.to_datetime(test_dates_trimmed)}
            
            if use_regressors and regressor_cols and x_test_reg is not None:
                for idx, col in enumerate(regressor_cols):
                    if idx < x_test_reg.shape[1]:
                        test_data[col] = x_test_reg[:, idx]

            df_test = pd.DataFrame(test_data)
            forecast = m.predict(df_test)
            
            yhat = forecast['yhat'].values.reshape(-1, 1)
            unc = ((forecast['yhat_upper'] - forecast['yhat_lower']) / 2).values.reshape(-1, 1)
            
            preds_list.append(yhat)
            unc_list.append(unc)

        return preds_list, unc_list

    def save(self, file_path: str) -> None:
        # Save all models in a pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"Prophet models saved to {file_path}")

    def load(self, file_path: str) -> None:
        with open(file_path, 'rb') as f:
            self.models = pickle.load(f)
        print(f"Prophet models loaded from {file_path}")
