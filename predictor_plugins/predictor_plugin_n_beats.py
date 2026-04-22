#!/usr/bin/env python
"""Pure N-BEATS Ioin (Deterministic).

Implements the standard N-BEATS architecture (Oreshkin et al., 2020).
- Inherits from BaseKerasPredictor (no Bayesian overhead).
- Deterministic predictions (uncertainty = 0).
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Add, Subtract, Flatten, Activation, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from .common.losses import mae_magnitude
from .common.base import BaseKerasPredictor
from .common.positional_encoding import positional_encoding

class Plugin(BaseKerasPredictor):
    plugin_params = {
        "nbeats_blocks": 3,
        "nbeats_layers": 4,
        "nbeats_units": 128,
        "activation": "swish",
        "l2_reg": 1e-5,
        "dropout_rate": 0.0,
        "learning_rate": 1e-3,
        "early_patience": 20,
        "batch_size": 64,
        "predicted_horizons": [1],
        "positional_encoding": False,
    }
    
    plugin_debug_vars = [
        "nbeats_blocks", "nbeats_layers", "nbeats_units",
        "activation", "l2_reg", "learning_rate", "predicted_horizons"
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        time_steps, channels = input_shape
        input_dim = time_steps * channels
        ph = self.params["predicted_horizons"]
        
        blocks = self.params["nbeats_blocks"]
        layers = self.params["nbeats_layers"]
        units = self.params["nbeats_units"]
        act = self.params.get("activation", "swish")
        l2_reg_v = self.params.get("l2_reg", 1e-5)
        dropout_rate = self.params.get("dropout_rate", 0.0)

        # --- Input ---
        inputs = Input(shape=(time_steps, channels), name="input_layer")
        
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(time_steps, channels)
            seq_in = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            seq_in = inputs

        # Flatten
        flat_in = Flatten(name="flatten_input")(seq_in)
        
        # --- N-BEATS Stack ---
        residual = flat_in
        forecast_accum = None

        for b in range(blocks):
            x = residual
            for l in range(layers):
                x = Dense(
                    units, 
                    activation=act,
                    kernel_regularizer=l2(l2_reg_v),
                    name=f"b{b}_dense{l}"
                )(x)
                if dropout_rate > 0:
                    x = Dropout(dropout_rate, name=f"b{b}_drop{l}")(x)

            # Backcast
            backcast = Dense(
                input_dim, 
                activation="linear", 
                name=f"b{b}_backcast"
            )(x)
            
            # Forecast
            forecast = Dense(
                units, 
                activation="linear", 
                name=f"b{b}_forecast"
            )(x)

            # Update Residual
            residual = Subtract(name=f"b{b}_residual")([residual, backcast])
            
            # Accumulate Forecast
            if forecast_accum is None:
                forecast_accum = forecast
            else:
                forecast_accum = Add(name=f"b{b}_accum")([forecast_accum, forecast])

        # --- Output Heads ---
        outputs = []
        self.output_names = []

        for horizon in ph:
            # Explicitly naming the layer to match output_names
            out = Dense(1, activation="linear", name=f"output_horizon_{horizon}")(forecast_accum)
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")

        self.model = Model(inputs=inputs, outputs=outputs, name=f"NBEATS_Pure_{len(ph)}H")
        
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        loss_dict = {nm: Huber() for nm in self.output_names}
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
        
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)

    def predict_with_uncertainty(self, x_test, mc_samples: int = 1):
        """Deterministic prediction (uncertainty = 0)."""
        preds = self.model.predict(x_test, verbose=0)
        if not isinstance(preds, list):
            preds = [preds]
        
        # Return predictions and zero uncertainty
        unc = [np.zeros_like(p) for p in preds]
        return preds, unc

if __name__ == '__main__':
    plug = Plugin({"predicted_horizons": [1,3], "positional_encoding": True})
    plug.build_model((96, 8), None, {})
    print('Outputs:', plug.output_names)
