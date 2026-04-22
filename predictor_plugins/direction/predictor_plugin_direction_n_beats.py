#!/usr/bin/env python
"""Direction N-BEATS ioin plugin (deterministic).

Thin subclass of the regression N-BEATS plugin for direction classification.
Replicates the N-BEATS block stack (backcast/forecast) and adds a single
Dense(1, sigmoid) direction output head.
"""
from __future__ import annotations

from tensorflow.keras.layers import (
    Input, Dense, Lambda, Add, Subtract, Flatten, Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l2

from ..predictor_plugin_n_beats import Plugin as NBeatsPlugin
from ..common.positional_encoding import positional_encoding
from .direction_base import DirectionMixin, BinaryF1Score, VALID_DIRECTION_TYPES


class Plugin(DirectionMixin, NBeatsPlugin):
    plugin_params = {
        **NBeatsPlugin.plugin_params,
        "signal_type": "direction_long",
    }
    plugin_params.pop("predicted_horizons", None)

    plugin_debug_vars = [
        "nbeats_blocks", "nbeats_layers", "nbeats_units",
        "activation", "l2_reg", "learning_rate", "signal_type",
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        time_steps, channels = input_shape
        input_dim = time_steps * channels
        blocks = self.params["nbeats_blocks"]
        layers = self.params["nbeats_layers"]
        units = self.params["nbeats_units"]
        act = self.params.get("activation", "swish")
        l2_reg_v = self.params.get("l2_reg", 1e-5)
        dropout_rate = self.params.get("dropout_rate", 0.0)

        signal_type = self.params.get("signal_type", "direction_long")
        if signal_type not in VALID_DIRECTION_TYPES:
            raise ValueError(f"Invalid signal_type: {signal_type}")
        out_name = "output_horizon_1"

        inputs = Input(shape=(time_steps, channels), name="input_layer")
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(time_steps, channels)
            seq_in = Lambda(
                lambda t, pe=pe: t + pe, name="add_positional_encoding"
            )(inputs)
        else:
            seq_in = inputs

        flat_in = Flatten(name="flatten_input")(seq_in)

        residual = flat_in
        forecast_accum = None

        for b in range(blocks):
            x = residual
            for l_idx in range(layers):
                x = Dense(
                    units, activation=act,
                    kernel_regularizer=l2(l2_reg_v),
                    name=f"b{b}_dense{l_idx}",
                )(x)
                if dropout_rate > 0:
                    x = Dropout(dropout_rate, name=f"b{b}_drop{l_idx}")(x)

            backcast = Dense(
                input_dim, activation="linear", name=f"b{b}_backcast"
            )(x)
            forecast = Dense(
                units, activation="linear", name=f"b{b}_forecast"
            )(x)

            residual = Subtract(name=f"b{b}_residual")(
                [residual, backcast]
            )
            if forecast_accum is None:
                forecast_accum = forecast
            else:
                forecast_accum = Add(name=f"b{b}_accum")(
                    [forecast_accum, forecast]
                )

        output = Dense(1, activation="sigmoid", name=out_name)(
            forecast_accum
        )

        self.output_names = [out_name]
        self.model = Model(
            inputs=inputs, outputs=[output],
            name=f"DirectionNBEATS_{signal_type}",
        )
        self.model.compile(
            optimizer=AdamW(
                learning_rate=self.params.get("learning_rate", 1e-3)
            ),
            loss={out_name: BinaryCrossentropy()},
            metrics={out_name: ["accuracy", AUC(name="auc"), BinaryF1Score()]},
        )
        self.model.summary(line_length=140)
