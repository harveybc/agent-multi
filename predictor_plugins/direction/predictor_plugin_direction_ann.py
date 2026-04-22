#!/usr/bin/env python
"""Direction ANN ioin plugin.

Thin subclass of the regression ANN plugin for direction classification.
Replicates the shared trunk (Flatten → Dense stack with Dropout) and
replaces the multi-horizon output heads with a single Dense(1, sigmoid)
binary head predicting price direction.
"""
from __future__ import annotations

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

from ..predictor_plugin_ann import Plugin as ANNPlugin
from ..common.positional_encoding import positional_encoding
from .direction_base import DirectionMixin, BinaryF1Score, VALID_DIRECTION_TYPES


class Plugin(DirectionMixin, ANNPlugin):
    plugin_params = {
        **ANNPlugin.plugin_params,
        "signal_type": "direction_long",
    }
    plugin_params.pop("predicted_horizons", None)

    plugin_debug_vars = [
        "batch_size", "hidden_units", "num_hidden_layers", "dropout_rate",
        "learning_rate", "signal_type", "positional_encoding",
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        window, channels = input_shape
        act = self.params.get("activation", "relu")
        hidden = self.params.get("hidden_units", 256)
        n_layers = self.params.get("num_hidden_layers", 2)
        dr = self.params.get("dropout_rate", 0.0)

        signal_type = self.params.get("signal_type", "direction_long")
        if signal_type not in VALID_DIRECTION_TYPES:
            raise ValueError(f"Invalid signal_type: {signal_type}")
        out_name = "output_horizon_1"

        inputs = Input(shape=(window, channels), name="input_layer")
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(window, channels)
            enc = Lambda(
                lambda t, pe=pe: t + pe, name="add_positional_encoding"
            )(inputs)
        else:
            enc = inputs

        x = Flatten(name="flatten_inputs")(enc)
        for i in range(n_layers):
            x = Dense(hidden, activation=act, name=f"shared_dense_{i}")(x)
            if dr > 0:
                x = Dropout(dr, name=f"shared_dropout_{i}")(x)

        output = Dense(1, activation="sigmoid", name=out_name)(x)

        self.output_names = [out_name]
        self.model = Model(
            inputs=inputs, outputs=[output], name=f"DirectionANN_{signal_type}"
        )
        self.model.compile(
            optimizer=AdamW(
                learning_rate=self.params.get("learning_rate", 1e-3)
            ),
            loss={out_name: BinaryCrossentropy()},
            metrics={out_name: ["accuracy", AUC(name="auc"), BinaryF1Score()]},
        )
        self.model.summary(line_length=140)
